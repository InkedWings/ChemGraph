# ChemGraph compute-bound benchmark sweep (Exp C1)

Drives the established 42-query ChemGraph workload through the **4-node MCP+Parsl
deployment** (see `../README.md`) as a 2-axis grid, to characterize where compute
resources saturate — the compute-bound counterpart in `BENCHMARK_PLAN.md` §7.2.

## The grid

| axis | values | meaning |
|------|--------|---------|
| **N** (concurrency) | 2, 4, 8, 16, 32 | number of multi-tenant workers; each runs the full 42-query suite, so N agent queries are in flight at once |
| **W** (partition) | 2, 4, 8 | Parsl workers per compute node → pool size **P = W×2**; physical cores per worker = **32/W** |
| **OMP** (bound to W) | 16, 8, 4 | `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 32/W` |

15 cells (5 N × 3 W). Each cell = N tenants × 42 queries. OMP is **not** an
independent axis — it tracks the per-worker physical-core count so a worker
threads exactly across the cores it owns (no oversubscription, no waste).

### Why these numbers (hardware)
Polaris CPU node = **AMD EPYC 7543P**: 32 physical cores (logical 0–31), each with
an SMT sibling at +32 (logical 32–63), in 4 NUMA nodes of 8 cores
(NUMA0=0–7, NUMA1=8–15, NUMA2=16–23, NUMA3=24–31). The Parsl `cpu_affinity`
(in `hpc_configs/polaris_parsl.py`) pins each worker to a contiguous block of
physical cores **plus their SMT siblings**, NUMA-aligned. W=4 → each worker = one
NUMA node = `list:0-7,32-39 : 8-15,40-47 : 16-23,48-55 : 24-31,56-63`.

This makes the utilization/timing data clean: a worker never shares a physical
core with another worker, and never straddles a NUMA boundary.

## Why the sweep is structured the way it is

- **Pool size is baked at MCP-server startup** (`CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE`
  read once). ⇒ changing W requires **restarting the MCP server**. The sweep groups
  cells by W and restarts MCP only at W-block boundaries.
- **No MCP-layer rate limit**: all N concurrent calls submit; Parsl queues the
  overflow. ⇒ N and pool size P are independent. N<P leaves slots idle (light tasks
  waste a fat partition); N>P queues (heavy tasks starve a thin partition). That
  crossover is the whole point of the sweep.
- **NWChem** (Exp8) is provided to workers via `CHEMGRAPH_WORKER_INIT` (PATH +
  `NWCHEM_BASIS_LIBRARY` to the conda-forge prefix at `venvs/nwchem`). **Exp14**
  runs as `-w multi_agent`; the CLI passes MCP tools as `executor_tools`, so its
  `run_ase` still executes on the Parsl pool.

## Files

| file | role |
|------|------|
| `bench_queries.txt` | the 42-query workload (`id<TAB>workflow<TAB>recursion_limit<TAB>query`) |
| `bench_worker_pool.sh` | one cell's load generator: N tenants, each runs all 42 in a deterministic per-tenant shuffle (`sha256(seed:tenant)`); true concurrency N |
| `bench_cell.sh` | run ONE (W,N) cell: samplers on B+C+D, run the pool, snapshot vLLM `/metrics`, write per-cell metadata + Parsl task counts |
| `bench_sweep.sh` | the manifest-driven, **resumable** driver over the 15-cell grid |
| `bench_report.py` | per-cell report, and `--sweep` grid aggregation → `grid_summary.csv` |
| `bench_run.sh` | (legacy) the original single-point cc=8 runner; superseded by the sweep |

## Running it

**Must run from the `qsub -I` primary shell on the HEAD node** (cross-node Parsl
needs the real PBS env). vLLM and MCP are started automatically if down.

```bash
cd .../4node_e2e/bench
bash bench_sweep.sh                 # runs the whole grid, cheapest cells first
```

Order: **W=4 block first** (balanced baseline), then W=8, then W=2; ascending N
within each block. So a usable prefix of the grid lands early.

### Resume (survives job death)
Each finished cell drops a `DONE` marker. To continue an interrupted sweep in a
later job, point at the same run root:
```bash
BENCH_RUN_ROOT=.../cg_logs/bench_sweep_<stamp> bash bench_sweep.sh
```
Finished cells are skipped. To cap a single invocation: `BENCH_MAX_CELLS=3`.

### Report
```bash
python3 bench_report.py --sweep .../cg_logs/bench_sweep_<stamp>   # grid CSV + table
python3 bench_report.py .../cells/W4_N8                            # one cell, detailed
```

## What success looks like
- Per cell: lightweight + xTB tasks PASS; heavy UMA tasks may TIMEOUT at high N /
  thin W — that is the saturation being measured, not a deployment break.
- Distributed exec: each cell's `parsl_task_counts.txt` shows tasks on C+D, with the
  number of managers = the pool size for that W.
- Compute-bound: `grid_summary.csv` shows C+D CPU% rising with N while the vLLM
  node's CPU and request queue stay low.

## Teardown
The sweep leaves vLLM+MCP up between cells (restarts MCP only per W). When fully
done: `bash ../e2e_4_check_and_teardown.sh`.
