import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher

from chemgraph.hpc_configs.loader import resolve_worker_init


def get_polaris_config(
    run_dir=None,
    worker_init: str | None = None,
    max_workers_per_node: int | None = None,
):
    """Generate the Parsl configuration for the Polaris supercomputer.

    Workers are partitioned by CPU cores (not by GPU): each Polaris node
    has 32 physical cores / 64 logical CPUs, and the default is 4 workers
    per node with 16 logical CPUs pinned to each via ``cpu_affinity``.
    No GPU is bound, so this targets CPU calculators (e.g. GFN2-xTB,
    UMA-on-CPU) for high-concurrency throughput.

    Parameters
    ----------
    run_dir : str, optional
        Directory used as Parsl's run directory.
    worker_init : str, optional
        Explicit shell snippet for worker init. When ``None`` (default),
        :func:`resolve_worker_init` picks ``CHEMGRAPH_WORKER_INIT`` /
        ``VIRTUAL_ENV`` / ``CONDA_PREFIX`` over a bare ``export TMPDIR=/tmp``
        fallback.
    max_workers_per_node : int, optional
        Number of workers per node. When ``None`` (default), reads
        ``CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE`` and falls back to 4.
        The total pool size (concurrency limit) is
        ``max_workers_per_node * num_nodes``.

    Returns
    -------
    parsl.config.Config
        Configured Parsl ``Config`` for Polaris.
    """
    if run_dir is None:
        run_dir = os.getcwd()

    if worker_init is None:
        worker_init = resolve_worker_init(run_dir, fallback="true")

    if max_workers_per_node is None:
        max_workers_per_node = int(
            os.getenv("CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE", "4")
        )

    # Polaris node = 64 logical CPUs (0-63). Split into contiguous blocks,
    # one per worker, so each worker is pinned to an exclusive core range.
    # Default 4 workers -> 16 logical CPUs each: 0-15 / 16-31 / 32-47 / 48-63.
    _n_cpus = 64
    _block = _n_cpus // max_workers_per_node
    _ranges = [
        f"{i * _block}-{(i + 1) * _block - 1}"
        for i in range(max_workers_per_node)
    ]
    cpu_affinity = "list:" + ":".join(_ranges)

    # Get the number of nodes from the PBS environment
    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r", encoding="utf-8") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        # Fallback for testing/local runs without PBS
        print("Warning: PBS_NODEFILE not found. Defaulting to 1 node.")
        num_nodes = 1

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=30,
                heartbeat_threshold=360,
                worker_debug=True,
                max_workers_per_node=max_workers_per_node,
                cpu_affinity=cpu_affinity,
                prefetch_capacity=0,
                provider=LocalProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=1 --ppn 1"
                    ),
                    worker_init=worker_init,
                    nodes_per_block=num_nodes,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                ),
            ),
        ],
        run_dir=run_dir,
    )

    return config
