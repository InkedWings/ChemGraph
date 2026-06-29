"""Backend-aware general ASE MCP server.

Same general-purpose chemistry tools as :mod:`chemgraph.mcp.mcp_tools`
(``run_ase`` + name/SMILES/coordinate/IO helpers), but built on
:class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP` so that ``run_ase`` is
submitted to an execution backend (Parsl / EnsembleLauncher / local)
instead of running inline in the server process.

Why this exists: under high concurrency many independent ``run_ase``
calls would otherwise execute in-process and contend for the same
CPUs/GPUs. Routing each call through the backend turns the worker pool
into a global rate limiter -- with the Parsl backend on Polaris, every
call becomes a TaskSpec dispatched to a fixed pool of workers, each
pinned to its own GPU + CPU core range (see ``hpc_configs/polaris_parsl``),
so concurrent calls queue against that pool instead of fighting over
resources.

The lightweight helpers (name->SMILES, SMILES->coords, JSON loading)
stay in-process via ``add_tool`` -- they are pure I/O and do not need
the backend.

Nothing requiring the backend is initialised at import time, so worker
subprocesses can re-import this module safely.
"""

import logging
from pathlib import Path
from typing import Literal

from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import extract_output_json_core, run_ase_core
from chemgraph.tools.cheminformatics_core import (
    molecule_name_to_smiles_core,
    smiles_to_coordinate_file_core,
)

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/ase_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph ASE Tools",
    instructions="""
        You provide general chemistry tools: converting molecule names to
        SMILES, building 3D coordinates, running ASE simulations (geometry
        optimization, thermochemistry, vibrational calculations), and
        reading results.

        Available tools:
        1. run_ase: run a single ASE calculation on the configured
           execution backend.
        2. molecule_name_to_smiles: resolve a molecule name to SMILES.
        3. smiles_to_coordinate_file: build 3D coordinates from SMILES.
        4. extract_output_json: load results from a run_ase output file.
        5. check_job_status / get_job_results / list_jobs / cancel_job:
           job batch management (relevant for async-remote backends).

        Guidelines:
        - Use each tool only when its input schema matches the request.
        - Do not invent data. If a tool raises an error, report it as-is.
        - Keep outputs compact; large results are written to files.
        - Use absolute paths when returning artifacts.
        - Energies are in eV, vibrational frequencies in cm^-1, wall
          times in seconds.
    """,
)


# ── Backend-submitted compute tool ─────────────────────────────────────
#
# ``run_ase_core`` is a top-level importable callable, so the framework
# can pickle it by reference and execute it on a backend worker. The
# ``mcp.tool()`` decorator wraps the call: each invocation becomes a
# TaskSpec submitted to the backend, and concurrent invocations pool
# against the backend's workers. Registered in ``__main__`` so the
# resource hints (ppn / gpus) can come from CLI args, mirroring
# ``mace_mcp_hpc``.


def run_ase(params: ASEInputSchema) -> dict:
    """Run ASE calculations using specified input parameters.

    Parameters
    ----------
    params : ASEInputSchema
        Input parameters for the ASE calculation

    Returns
    -------
    dict
        Output containing calculation status

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails
    """
    import io
    from contextlib import redirect_stdout

    # On a backend worker, params arrives as a dict (the framework's
    # to_picklable converts the pydantic arg before pickling); rebuild it.
    if isinstance(params, dict):
        params = ASEInputSchema(**params)
    f = io.StringIO()
    with redirect_stdout(f):
        return run_ase_core(params)


# ── Lightweight in-process tools (no backend involvement) ──────────────


def molecule_name_to_smiles(name: str) -> str:
    """Resolve a molecule name to its canonical SMILES via PubChem.

    Parameters
    ----------
    name : str
        Molecule name to resolve.

    Returns
    -------
    str
        Canonical SMILES string.
    """
    return molecule_name_to_smiles_core(name)


def smiles_to_coordinate_file(
    smiles: str,
    output_file: str = "molecule.xyz",
    seed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> dict:
    """Convert a SMILES string to a coordinate file on disk.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    output_file : str, optional
        Coordinate file path to write.
    seed : int, optional
        Random seed used for conformer generation.
    fmt : {"xyz"}, optional
        Output coordinate format.

    Returns
    -------
    dict
        Coordinate-generation result metadata.
    """
    return smiles_to_coordinate_file_core(
        smiles, output_file=output_file, seed=seed, fmt=fmt
    )


def extract_output_json(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by run_ase.

    Parameters
    ----------
    json_file : str
        Path to the JSON output file.

    Returns
    -------
    dict
        Parsed simulation results.
    """
    return extract_output_json_core(json_file)


mcp.add_tool(
    molecule_name_to_smiles,
    name="molecule_name_to_smiles",
    description="Convert a molecule name to a canonical SMILES string using PubChem.",
)
mcp.add_tool(
    smiles_to_coordinate_file,
    name="smiles_to_coordinate_file",
    description="Convert a SMILES string to a coordinate file.",
)
mcp.add_tool(
    extract_output_json,
    name="extract_output_json",
    description="Load simulation results from a JSON file produced by run_ase.",
)


if __name__ == "__main__":
    import argparse as _ap
    import sys

    from chemgraph.mcp.server_utils import run_mcp_server

    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument(
        "--ppn", type=int, default=1,
        help="Processes per node for backend tasks",
    )
    _parser.add_argument(
        "--ngpus-per-process", type=int, default=0,
        help="GPUs per process for backend tasks",
    )
    _args, _remaining = _parser.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    # Register run_ase as a backend-submitted tool. Resource hints flow
    # into each TaskSpec the framework builds per invocation.
    mcp.tool(
        name="run_ase",
        description="Run a single ASE calculation on the execution backend.",
        processes_per_node=_args.ppn,
        gpus_per_task=_args.ngpus_per_process,
    )(run_ase)

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

    try:
        run_mcp_server(mcp, default_port=9005)
    finally:
        mcp.shutdown_backend()
