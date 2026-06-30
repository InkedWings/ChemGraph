#!/bin/bash
# Manifest-driven, resumable ChemGraph compute-bound sweep (Exp C1).
#
# Grid: concurrency N in {2,4,8,16,32} x partition W in {2,4,8}.
#   W = workers/compute-node -> pool P = W*2 ; cores/worker = OMP = 32/W.
# Each cell = N multi-tenant workers, each running the full 42-query suite.
#
# Ordering: cheapest/most-sensible first, grouped by W to minimize MCP restarts
# (pool size is baked at MCP startup, so W changes require a restart):
#   W=4 block (balanced baseline) -> W=8 -> W=2 ; ascending N within each block.
#
# Resumable: each finished cell drops a DONE marker; re-running skips it. Safe to
# kill (job death) and restart in a later job -- it continues where it stopped.
#
# Run from the qsub -I primary shell on the HEAD node.
#
# Tunables (env):
#   BENCH_N_LIST   default "2 4 8 16 32"
#   BENCH_W_LIST   default "4 8 2"   (W-block order)
#   BENCH_RUN_ROOT default $REPO/cg_logs/bench_sweep_<stamp>  (set to resume a prior run)
#   BENCH_MAX_CELLS  stop after this many cells run this invocation (default: all)
#   BENCH_SEED     per-tenant shuffle seed (default 20260630)
#   SAMPLE_INTERVAL  sampler period s (default 1)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../e2e_env.sh"

BENCH_N_LIST="${BENCH_N_LIST:-2 4 8 16 32}"
BENCH_W_LIST="${BENCH_W_LIST:-4 8 2}"
BENCH_SEED="${BENCH_SEED:-20260630}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
BENCH_MAX_CELLS="${BENCH_MAX_CELLS:-0}"   # 0 = no limit
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-$REPO/cg_logs/bench_sweep_$STAMP}"

NWCHEM_PREFIX=/lus/eagle/projects/ChemGraph/zhijing/venvs/nwchem

mkdir -p "$BENCH_RUN_ROOT/cells"
SWEEP_SUMMARY="$BENCH_RUN_ROOT/sweep_summary.csv"
[ -f "$SWEEP_SUMMARY" ] || \
    printf 'cell,W,N,pool_P,omp,makespan_s,pass,fail,timeout,total\n' > "$SWEEP_SUMMARY"

echo "==================== ChemGraph sweep ===================="
echo "  BENCH_RUN_ROOT=$BENCH_RUN_ROOT"
echo "  N list=[$BENCH_N_LIST]  W blocks=[$BENCH_W_LIST]  seed=$BENCH_SEED"

# ---- node roles ------------------------------------------------------------
# Ensure PBS_NODEFILE/PBS_JOBID are exported IN THIS PROCESS (not just a subshell)
# so the per-W MCP restart + Parsl mpiexec inherit them from any ssh session.
e2e_ensure_pbs_env || { echo "  ERROR: no running PBS job / nodefile found." >&2; exit 1; }
if ! e2e_resolve_nodes >/dev/null; then exit 1; fi
HEAD_NODE="$(e2e_head_node)"; VLLM_NODE_NAME="$(e2e_vllm_node)"
THIS_NODE="$(hostname -s)"
[ "$THIS_NODE" = "$HEAD_NODE" ] || { echo "  ERROR: must run on HEAD ($HEAD_NODE)" >&2; exit 1; }
echo "  HEAD=$HEAD_NODE  vLLM=$VLLM_NODE_NAME"

# ---- bring vLLM up once (W-independent) -------------------------------------
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "  vLLM: UP"
else
    echo "  vLLM: down -> starting..."
    bash "$HERE/../e2e_1_start_vllm.sh" || { echo "  ERROR: vLLM start failed" >&2; exit 1; }
fi

# ---- helper: (re)start MCP for a given W -----------------------------------
# Builds the worker_init for this W: cg-mcp venv + NWChem on PATH + basis lib +
# OMP/OPENBLAS/MKL pinned to cores/worker (=32/W). Kills any running MCP first.
restart_mcp_for_W() {
    local w="$1" omp=$((32 / "$1"))
    echo "-------------------- (re)start MCP: W=$w, OMP=$omp --------------------"
    # Invalidate the W sentinel FIRST: until the new MCP confirms startup, no cell
    # may believe a correctly-sized pool is running (H3 guard then fails closed).
    rm -f "$STATE_DIR/mcp_workers_per_node"
    # Kill existing MCP process group if present. Keep the PGID file until the new
    # server overwrites it -- if the restart fails we still have a handle to the
    # (now-killed) old group rather than losing it (M3).
    if [ -f "$MCP_PGID_FILE" ]; then
        local pgid; pgid="$(cat "$MCP_PGID_FILE")"
        [ -n "$pgid" ] && { kill -TERM "-$pgid" 2>/dev/null || true; sleep 3; kill -9 "-$pgid" 2>/dev/null || true; }
    fi
    # Sweep any orphan Parsl workers on the compute nodes before the new pool.
    local cn
    for cn in $(e2e_compute_nodes); do
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$cn" \
            'for p in $(pgrep -u $USER -f process_worker_pool); do kill -9 $p 2>/dev/null; done' 2>/dev/null || true
    done
    # worker_init: venv activate FIRST (so cg-mcp python wins), then NWChem PATH.
    export CHEMGRAPH_WORKER_INIT="module use /soft/modulefiles; module load conda; conda activate base; source $VENV/bin/activate; export PATH=$NWCHEM_PREFIX/bin:\$PATH; export NWCHEM_BASIS_LIBRARY=$NWCHEM_PREFIX/share/nwchem/libraries/; export OMP_NUM_THREADS=$omp; export OPENBLAS_NUM_THREADS=$omp; export MKL_NUM_THREADS=$omp; export TMPDIR=/tmp"
    export CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE="$w"
    # e2e_2_start_mcp.sh overwrites MCP_PGID_FILE and writes the W sentinel only
    # after it confirms the server is listening.
    bash "$HERE/../e2e_2_start_mcp.sh" || { echo "  ERROR: MCP start failed for W=$w" >&2; return 1; }
}

# ---- sweep -----------------------------------------------------------------
CURRENT_W=""
CELLS_RUN=0
for W in $BENCH_W_LIST; do
    for N in $BENCH_N_LIST; do
        CELL="W${W}_N${N}"
        CELL_ROOT="$BENCH_RUN_ROOT/cells/$CELL"
        if [ -f "$CELL_ROOT/DONE" ]; then
            echo "  skip $CELL (DONE)"
            continue
        fi
        # Restart MCP only when entering a new W block.
        if [ "$W" != "$CURRENT_W" ]; then
            restart_mcp_for_W "$W" || exit 1
            CURRENT_W="$W"
        fi

        echo "######## RUN cell $CELL ########"
        if BENCH_RUN_ROOT="$BENCH_RUN_ROOT" CELL_ROOT="$CELL_ROOT" \
           SAMPLE_INTERVAL="$SAMPLE_INTERVAL" BENCH_SEED="$BENCH_SEED" \
           bash "$HERE/bench_cell.sh" "$W" "$N"; then
            # Append a row to the master summary from this cell's outputs.
            python3 - "$CELL_ROOT" "$SWEEP_SUMMARY" "$CELL" "$W" "$N" <<'PY'
import csv, json, os, sys
cell_root, sweep_csv, cell, W, N = sys.argv[1:6]
meta = {}
mp = os.path.join(cell_root, "profile_metadata.json")
if os.path.exists(mp): meta = json.load(open(mp))
P = meta.get("pool_size_P", "")
omp = meta.get("omp_num_threads", "")
mk = ""
try: mk = f'{float(meta["profile_end_epoch"])-float(meta["profile_start_epoch"]):.1f}'
except Exception: pass
pa=fa=to=tot=0
sp = os.path.join(cell_root, "summary.csv")
if os.path.exists(sp):
    for r in csv.DictReader(open(sp)):
        tot += 1
        s = r.get("status","")
        if s=="PASS": pa+=1
        elif s=="FAIL": fa+=1
        elif s=="TIMEOUT": to+=1
with open(sweep_csv,"a",newline="") as f:
    csv.writer(f).writerow([cell,W,N,P,omp,mk,pa,fa,to,tot])
print(f"  appended sweep row: {cell} pass={pa} fail={fa} timeout={to} makespan={mk}s")
PY
            touch "$CELL_ROOT/DONE"
            CELLS_RUN=$((CELLS_RUN + 1))
        else
            echo "  WARN: cell $CELL exited non-zero; NOT marking DONE (will retry on resume)."
        fi

        if [ "$BENCH_MAX_CELLS" -gt 0 ] && [ "$CELLS_RUN" -ge "$BENCH_MAX_CELLS" ]; then
            echo "  reached BENCH_MAX_CELLS=$BENCH_MAX_CELLS; stopping."
            break 2
        fi
    done
done

echo "==================== sweep pass complete ===================="
echo "  cells run this invocation: $CELLS_RUN"
echo "  master summary: $SWEEP_SUMMARY"
column -t -s, "$SWEEP_SUMMARY" 2>/dev/null || cat "$SWEEP_SUMMARY"
echo
echo "  report: python3 $HERE/bench_report.py --sweep $BENCH_RUN_ROOT"
echo "  resume: BENCH_RUN_ROOT=$BENCH_RUN_ROOT bash $HERE/bench_sweep.sh"
echo "  teardown when fully done: bash $HERE/../e2e_4_check_and_teardown.sh"
