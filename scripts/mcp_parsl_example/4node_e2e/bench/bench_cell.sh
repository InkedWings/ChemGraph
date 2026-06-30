#!/bin/bash
# Run ONE (W, N) cell of the ChemGraph compute-bound sweep.
#   W = workers per compute node   -> Parsl pool P = W * (#compute nodes)
#                                     phys cores/worker = 32/W ; OMP = 32/W
#   N = concurrency = number of multi-tenant workers (each runs the full 42)
#
# Assumes vLLM (node B) is already up and the MCP server is already running with
# THIS W (bench_sweep.sh restarts MCP at each W-block boundary; running a cell
# standalone requires the MCP server to have been started with the matching
# CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE). Use bench_sweep.sh for the full grid.
#
# Inputs via env or args:
#   W (arg1 or $BENCH_W)   N (arg2 or $BENCH_N)
#   CELL_ROOT  output dir for this cell (default: derived under BENCH_RUN_ROOT)
#   SAMPLE_INTERVAL (default 1)   BENCH_SEED (default 20260630)
# Must run on the HEAD node.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../e2e_env.sh"

W="${1:-${BENCH_W:?bench_cell.sh requires W (arg1 or BENCH_W)}}"
N="${2:-${BENCH_N:?bench_cell.sh requires N (arg2 or BENCH_N)}}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
BENCH_SEED="${BENCH_SEED:-20260630}"

AGENTIC_DIR="${AGENTIC_DIR:-/eagle/lc-mpi/ZhijingYe/Agentic}"
CPU_SAMPLER="$AGENTIC_DIR/chemgraph_profile_cpu_sampler.py"
GPU_SAMPLER="$AGENTIC_DIR/chemgraph_profile_gpu_sampler.py"

# Derived hardware partition facts (AMD EPYC 7543P: 32 phys cores/node).
PHYS_PER_NODE=32
CORES_PER_WORKER=$((PHYS_PER_NODE / W))
OMP=$CORES_PER_WORKER

CELL="W${W}_N${N}"
CELL_ROOT="${CELL_ROOT:-${BENCH_RUN_ROOT:-$REPO/cg_logs/bench_sweep}/cells/$CELL}"
mkdir -p "$CELL_ROOT"

echo "==================== cell $CELL ===================="
echo "  W=$W  N=$N  cores/worker=$CORES_PER_WORKER  OMP=$OMP"
echo "  CELL_ROOT=$CELL_ROOT"

# ---- node roles ------------------------------------------------------------
e2e_ensure_pbs_env || { echo "  ERROR: no running PBS job / nodefile found." >&2; exit 1; }
if ! e2e_resolve_nodes >/dev/null; then exit 1; fi
HEAD_NODE="$(e2e_head_node)"; VLLM_NODE_NAME="$(e2e_vllm_node)"
THIS_NODE="$(hostname -s)"
mapfile -t COMPUTE_ARR < <(e2e_compute_nodes)
[ "$THIS_NODE" = "$HEAD_NODE" ] || { echo "  ERROR: must run on HEAD ($HEAD_NODE)" >&2; exit 1; }
for s in "$CPU_SAMPLER" "$GPU_SAMPLER"; do
    [ -f "$s" ] || { echo "  ERROR: missing sampler $s" >&2; exit 1; }
done
P=$((W * ${#COMPUTE_ARR[@]}))
echo "  pool P=$P over compute=${COMPUTE_ARR[*]}"

# ---- preconditions: vLLM up, MCP up ----------------------------------------
ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1 \
    || { echo "  ERROR: vLLM not up on $VLLM_NODE_NAME:$VLLM_PORT" >&2; exit 1; }
ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$MCP_PORT$" \
    || { echo "  ERROR: MCP server not listening on $MCP_PORT (start it for W=$W first)" >&2; exit 1; }
# H3: the running MCP must have been started with THIS cell's W (pool size is
# baked at MCP startup). Refuse to run against a mismatched pool -> wrong numbers.
MCP_WPN_FILE="$STATE_DIR/mcp_workers_per_node"
if [ -f "$MCP_WPN_FILE" ]; then
    RUNNING_W="$(cat "$MCP_WPN_FILE" 2>/dev/null)"
    [ "$RUNNING_W" = "$W" ] || {
        echo "  ERROR: MCP is running with W=$RUNNING_W but this cell needs W=$W." >&2
        echo "         Restart MCP for W=$W (bench_sweep.sh does this per W-block)." >&2
        exit 1
    }
else
    echo "  ERROR: no MCP W sentinel ($MCP_WPN_FILE); cannot confirm pool size = W=$W." >&2
    exit 1
fi

# ---- vLLM backend log offset -----------------------------------------------
VLLM_PID="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "pgrep -f 'vllm serve.*--port $VLLM_PORT' | head -1" 2>/dev/null | awk '/^[0-9]+$/{print;exit}')"
VLLM_LOG_PATH=""; VLLM_LOG_START_SIZE=0
if [ -n "$VLLM_PID" ]; then
    VLLM_LOG_PATH="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "readlink /proc/$VLLM_PID/fd/1" 2>/dev/null | awk '/^\//{print;exit}')"
    [ -n "$VLLM_LOG_PATH" ] && VLLM_LOG_START_SIZE="$(ssh -o BatchMode=yes -o ConnectTimeout=8 \
        "$VLLM_NODE_NAME" "stat -c %s '$VLLM_LOG_PATH'" 2>/dev/null | awk '/^[0-9]+$/{print;exit}')"
fi

ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
    > "$CELL_ROOT/vllm_metrics_before.prom" 2>/dev/null || echo "  warn: metrics_before failed"

# ---- start samplers (B + each compute node) --------------------------------
sampler_pids=()
start_sampler() {  # name node command outfile errfile
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$2" "$3" > "$4" 2> "$5" &
    sampler_pids+=("$!")
}
cleanup_samplers() {
    for p in "${sampler_pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
    for p in "${sampler_pids[@]:-}"; do wait "$p" 2>/dev/null || true; done
}
trap cleanup_samplers EXIT
start_sampler "cpu_vllm" "$VLLM_NODE_NAME" \
    "python3 '$CPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE_NAME'" \
    "$CELL_ROOT/cpu_vllm.csv" "$CELL_ROOT/cpu_vllm.err"
start_sampler "gpu_vllm" "$VLLM_NODE_NAME" \
    "python3 '$GPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE_NAME'" \
    "$CELL_ROOT/gpu_vllm.csv" "$CELL_ROOT/gpu_vllm.err"
start_sampler "vllm_metrics" "$VLLM_NODE_NAME" \
    "while true; do printf '# sample_epoch='; date +%s.%N; curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics; printf '# end_sample\n'; sleep '$SAMPLE_INTERVAL'; done" \
    "$CELL_ROOT/vllm_metrics_samples.prom" "$CELL_ROOT/vllm_metrics_samples.err"
for cn in "${COMPUTE_ARR[@]}"; do
    start_sampler "cpu_$cn" "$cn" \
        "python3 '$CPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$cn'" \
        "$CELL_ROOT/cpu_$cn.csv" "$CELL_ROOT/cpu_$cn.err"
    start_sampler "gpu_$cn" "$cn" \
        "python3 '$GPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$cn'" \
        "$CELL_ROOT/gpu_$cn.csv" "$CELL_ROOT/gpu_$cn.err"
done

# ---- run the workload (N tenants x 42) -------------------------------------
echo "-------------------- workload: $N tenants x 42 queries --------------------"
PROFILE_START_EPOCH="$(date +%s.%N)"
PROFILE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
set +e
CELL="$CELL" QUERIES_FILE="$HERE/bench_queries.txt" RUN_ROOT="$CELL_ROOT" \
    CONCURRENCY="$N" VLLM_NODE_NAME="$VLLM_NODE_NAME" BENCH_SEED="$BENCH_SEED" \
    bash "$HERE/bench_worker_pool.sh" 2>&1 | tee "$CELL_ROOT/worker_pool.log"
WORKLOAD_STATUS="${PIPESTATUS[0]}"
set -e
PROFILE_END_EPOCH="$(date +%s.%N)"
PROFILE_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

# ---- stop samplers, snapshot after -----------------------------------------
cleanup_samplers; trap - EXIT
ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
    > "$CELL_ROOT/vllm_metrics_after.prom" 2>/dev/null || echo "  warn: metrics_after failed"
if [ -n "$VLLM_LOG_PATH" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "tail -c +$((VLLM_LOG_START_SIZE + 1)) '$VLLM_LOG_PATH'" \
        > "$CELL_ROOT/vllm_backend_incremental.log" 2>/dev/null || true
fi

# ---- Parsl per-node task counts (this cell's run dir) ----------------------
# CELL_FAIL accumulates any hard problem so we still write metadata, then exit
# non-zero at the end -- the sweep's DONE guard then retries this cell on resume.
CELL_FAIL=0
[ "$WORKLOAD_STATUS" -eq 0 ] || { echo "  WARN: workload exited $WORKLOAD_STATUS"; CELL_FAIL=1; }

LATEST=$(ls -dt "$REPO"/[0-9][0-9][0-9]/htex/ 2>/dev/null | head -1)
PARSL_COUNTS="$CELL_ROOT/parsl_task_counts.txt"
PARSL_TOTAL=0
if [ -z "$LATEST" ]; then
    # No Parsl run dir => the workload produced zero tasks (e.g. agents crashed
    # before the first run_ase). Record it as a FAILED cell, not a silent 0.
    echo "  ERROR: no Parsl run dir under $REPO/NNN/htex -- cell did no compute" | tee "$PARSL_COUNTS"
    CELL_FAIL=1
else
    {
        echo "latest_run_dir=$LATEST"
        _total=0; _nworkers=0
        for mlog in "$LATEST"block-*/*/manager.log; do
            [ -f "$mlog" ] || continue
            hn=$(grep -oE "'hostname': '[^']+'" "$mlog" 2>/dev/null | head -1 | sed "s/.*: '//; s/'//")
            cnt=$(grep -oE "cumulative count of tasks: [0-9]+" "$mlog" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1)
            echo "  ${hn:-<unknown>} : ${cnt:-0} tasks"
            _total=$((_total + ${cnt:-0})); _nworkers=$((_nworkers + 1))
        done
        echo "  total: $_total  managers: $_nworkers"
    } | tee "$PARSL_COUNTS"
    PARSL_TOTAL=$(grep -oE "total: [0-9]+" "$PARSL_COUNTS" | grep -oE "[0-9]+" | head -1)
    [ "${PARSL_TOTAL:-0}" -ge 1 ] || { echo "  ERROR: Parsl ran 0 tasks"; CELL_FAIL=1; }
fi

# ---- per-cell metadata -----------------------------------------------------
python3 - "$CELL_ROOT/profile_metadata.json" <<PY
import json, sys
meta = {
    "benchmark": "chemgraph_compute_bound_expC1_sweep",
    "pipeline": "mcp_parsl_4node",
    "cell": "$CELL",
    "W_workers_per_node": int("$W"),
    "N_concurrency": int("$N"),
    "pool_size_P": int("$P"),
    "cores_per_worker": int("$CORES_PER_WORKER"),
    "omp_num_threads": int("$OMP"),
    "n_queries_per_tenant": 42,
    "total_executions_expected": int("$N") * 42,
    "profile_start_epoch": "$PROFILE_START_EPOCH",
    "profile_end_epoch": "$PROFILE_END_EPOCH",
    "profile_start_iso": "$PROFILE_START_ISO",
    "profile_end_iso": "$PROFILE_END_ISO",
    "workload_status": int("$WORKLOAD_STATUS"),
    "cell_fail": int("$CELL_FAIL"),
    "parsl_total_tasks": int("${PARSL_TOTAL:-0}"),
    "sample_interval_s": float("$SAMPLE_INTERVAL"),
    "seed": "$BENCH_SEED",
    "head_node": "$HEAD_NODE",
    "vllm_node": "$VLLM_NODE_NAME",
    "compute_nodes": "${COMPUTE_ARR[*]}".split(),
    "vllm_pid": "$VLLM_PID",
    "vllm_log_path": "$VLLM_LOG_PATH",
    "cell_root": "$CELL_ROOT",
}
json.dump(meta, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1])
PY

MAKESPAN=$(python3 -c "print(f'{float(\"$PROFILE_END_EPOCH\")-float(\"$PROFILE_START_EPOCH\"):.1f}')")
echo "==================== cell $CELL done: makespan=${MAKESPAN}s workload_status=$WORKLOAD_STATUS cell_fail=$CELL_FAIL ===================="

# Propagate failure so bench_sweep.sh does NOT mark this cell DONE (H1/H2): a
# workload error or a zero-task run => non-zero exit => retried on resume.
[ "$CELL_FAIL" -eq 0 ] || exit 1
exit 0
