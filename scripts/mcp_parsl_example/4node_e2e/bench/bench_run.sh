#!/bin/bash
# ChemGraph compute-bound benchmark orchestrator (Exp C1) for the 4-node
# MCP+Parsl deployment. Run from the qsub -I primary shell on the HEAD node.
#
# It (1) brings vLLM + MCP up if they are down, (2) starts double-ended
# instrumentation (vLLM /metrics + CPU/GPU samplers on node B and compute nodes
# C+D), (3) runs the 36-query workload at concurrency 8 via bench_worker_pool.sh,
# (4) snapshots metrics and writes profile_metadata.json, (5) prints the Parsl
# per-node task counts. It does NOT tear down services (so a sweep can follow).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../e2e_env.sh"

# Reuse the existing standalone samplers from the Agentic benchmark suite.
AGENTIC_DIR="${AGENTIC_DIR:-/eagle/lc-mpi/ZhijingYe/Agentic}"
CPU_SAMPLER="$AGENTIC_DIR/chemgraph_profile_cpu_sampler.py"
GPU_SAMPLER="$AGENTIC_DIR/chemgraph_profile_gpu_sampler.py"

CONCURRENCY="${CONCURRENCY:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${BENCH_RUN_ROOT:-$REPO/cg_logs/bench_cc${CONCURRENCY}_$STAMP}"

echo "==================== ChemGraph benchmark (Exp C1) ===================="
echo "RUN_ROOT=$RUN_ROOT"
mkdir -p "$RUN_ROOT"

# ---- node roles ------------------------------------------------------------
if ! e2e_resolve_nodes >/dev/null; then exit 1; fi
HEAD_NODE="$(e2e_head_node)"
VLLM_NODE_NAME="$(e2e_vllm_node)"
THIS_NODE="$(hostname -s)"
mapfile -t COMPUTE_ARR < <(e2e_compute_nodes)
echo "  HEAD=$HEAD_NODE  vLLM=$VLLM_NODE_NAME  compute=${COMPUTE_ARR[*]}"
if [ "$THIS_NODE" != "$HEAD_NODE" ]; then
    echo "  ERROR: must run on HEAD ($HEAD_NODE); cross-node Parsl needs the qsub -I shell." >&2
    exit 1
fi
[ "${#COMPUTE_ARR[@]}" -ge 1 ] || { echo "  ERROR: no compute nodes resolved." >&2; exit 1; }
for s in "$CPU_SAMPLER" "$GPU_SAMPLER"; do
    [ -f "$s" ] || { echo "  ERROR: missing sampler $s" >&2; exit 1; }
done

# ---- bring services up if needed (idempotent) ------------------------------
echo "-------------------- service check --------------------"
# vLLM on node B: probe /v1/models; if down, start it.
if ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "  vLLM: UP on $VLLM_NODE_NAME:$VLLM_PORT"
else
    echo "  vLLM: down -> starting (e2e_1_start_vllm.sh)..."
    bash "$HERE/../e2e_1_start_vllm.sh" || { echo "  ERROR: vLLM start failed" >&2; exit 1; }
fi

# MCP on HEAD: PGID file present AND port listening; else (re)start.
if [ -f "$MCP_PGID_FILE" ] && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$MCP_PORT$"; then
    echo "  MCP: UP on 127.0.0.1:$MCP_PORT (PGID $(cat "$MCP_PGID_FILE"))"
else
    echo "  MCP: down -> starting (e2e_2_start_mcp.sh)..."
    bash "$HERE/../e2e_2_start_mcp.sh" || { echo "  ERROR: MCP start failed" >&2; exit 1; }
fi

# ---- vLLM backend log offset (for incremental capture) ---------------------
VLLM_PID="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "pgrep -f 'vllm serve.*--port $VLLM_PORT' | head -1" 2>/dev/null | awk '/^[0-9]+$/{print;exit}')"
VLLM_LOG_PATH=""; VLLM_LOG_START_SIZE=0
if [ -n "$VLLM_PID" ]; then
    VLLM_LOG_PATH="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "readlink /proc/$VLLM_PID/fd/1" 2>/dev/null | awk '/^\//{print;exit}')"
    [ -n "$VLLM_LOG_PATH" ] && VLLM_LOG_START_SIZE="$(ssh -o BatchMode=yes -o ConnectTimeout=8 \
        "$VLLM_NODE_NAME" "stat -c %s '$VLLM_LOG_PATH'" 2>/dev/null | awk '/^[0-9]+$/{print;exit}')"
fi
echo "  vLLM PID=$VLLM_PID log=$VLLM_LOG_PATH offset=$VLLM_LOG_START_SIZE"

# ---- metrics snapshot (before) ---------------------------------------------
ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
    > "$RUN_ROOT/vllm_metrics_before.prom" 2>/dev/null || echo "  warn: metrics_before failed"

# ---- start samplers --------------------------------------------------------
echo "-------------------- start samplers (interval ${SAMPLE_INTERVAL}s) --------------------"
sampler_pids=()
start_sampler() {  # name node command outfile errfile
    echo "  sampler $1 on $2"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$2" "$3" > "$4" 2> "$5" &
    sampler_pids+=("$!")
}
cleanup_samplers() {
    for p in "${sampler_pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
    for p in "${sampler_pids[@]:-}"; do wait "$p" 2>/dev/null || true; done
}
trap cleanup_samplers EXIT

# vLLM node (B): cpu + gpu + /metrics poller
start_sampler "cpu_vllm" "$VLLM_NODE_NAME" \
    "python3 '$CPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE_NAME'" \
    "$RUN_ROOT/cpu_vllm.csv" "$RUN_ROOT/cpu_vllm.err"
start_sampler "gpu_vllm" "$VLLM_NODE_NAME" \
    "python3 '$GPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$VLLM_NODE_NAME'" \
    "$RUN_ROOT/gpu_vllm.csv" "$RUN_ROOT/gpu_vllm.err"
start_sampler "vllm_metrics" "$VLLM_NODE_NAME" \
    "while true; do printf '# sample_epoch='; date +%s.%N; curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics; printf '# end_sample\n'; sleep '$SAMPLE_INTERVAL'; done" \
    "$RUN_ROOT/vllm_metrics_samples.prom" "$RUN_ROOT/vllm_metrics_samples.err"

# compute nodes (C, D): cpu + gpu each
for cn in "${COMPUTE_ARR[@]}"; do
    start_sampler "cpu_$cn" "$cn" \
        "python3 '$CPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$cn'" \
        "$RUN_ROOT/cpu_$cn.csv" "$RUN_ROOT/cpu_$cn.err"
    start_sampler "gpu_$cn" "$cn" \
        "python3 '$GPU_SAMPLER' --interval '$SAMPLE_INTERVAL' --host '$cn'" \
        "$RUN_ROOT/gpu_$cn.csv" "$RUN_ROOT/gpu_$cn.err"
done

# ---- run the workload ------------------------------------------------------
echo "-------------------- run workload (cc=$CONCURRENCY) --------------------"
PROFILE_START_EPOCH="$(date +%s.%N)"
PROFILE_START_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
set +e
QUERIES_FILE="$HERE/bench_queries.txt" \
    RUN_ROOT="$RUN_ROOT" \
    CONCURRENCY="$CONCURRENCY" \
    VLLM_NODE_NAME="$VLLM_NODE_NAME" \
    bash "$HERE/bench_worker_pool.sh" 2>&1 | tee "$RUN_ROOT/worker_pool.log"
WORKLOAD_STATUS="${PIPESTATUS[0]}"
set -e
PROFILE_END_EPOCH="$(date +%s.%N)"
PROFILE_END_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"

# ---- stop samplers, snapshot after -----------------------------------------
cleanup_samplers
trap - EXIT
ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
    "curl --noproxy '*' -fsS -m 8 http://127.0.0.1:$VLLM_PORT/metrics" \
    > "$RUN_ROOT/vllm_metrics_after.prom" 2>/dev/null || echo "  warn: metrics_after failed"
if [ -n "$VLLM_LOG_PATH" ]; then
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$VLLM_NODE_NAME" \
        "tail -c +$((VLLM_LOG_START_SIZE + 1)) '$VLLM_LOG_PATH'" \
        > "$RUN_ROOT/vllm_backend_incremental.log" 2>/dev/null || true
fi

# ---- Parsl per-node task counts (Stage-4 authoritative method) -------------
echo "-------------------- Parsl per-node task counts --------------------"
LATEST=$(ls -dt "$REPO"/[0-9][0-9][0-9]/htex/ 2>/dev/null | head -1)
PARSL_COUNTS="$RUN_ROOT/parsl_task_counts.txt"
{
    echo "latest_run_dir=$LATEST"
    _total=0
    for mlog in "$LATEST"block-*/*/manager.log; do
        [ -f "$mlog" ] || continue
        hn=$(grep -oE "'hostname': '[^']+'" "$mlog" 2>/dev/null | head -1 | sed "s/.*: '//; s/'//")
        cnt=$(grep -oE "cumulative count of tasks: [0-9]+" "$mlog" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1)
        echo "  ${hn:-<unknown>} : ${cnt:-0} tasks"
        _total=$((_total + ${cnt:-0}))
    done
    echo "  total: $_total"
} | tee "$PARSL_COUNTS"

# ---- metadata --------------------------------------------------------------
python3 - "$RUN_ROOT/profile_metadata.json" <<PY
import json, sys
meta = {
    "benchmark": "chemgraph_compute_bound_expC1",
    "pipeline": "mcp_parsl_4node",
    "profile_start_epoch": "$PROFILE_START_EPOCH",
    "profile_end_epoch": "$PROFILE_END_EPOCH",
    "profile_start_iso": "$PROFILE_START_ISO",
    "profile_end_iso": "$PROFILE_END_ISO",
    "workload_status": int("$WORKLOAD_STATUS"),
    "concurrency": int("$CONCURRENCY"),
    "sample_interval_s": float("$SAMPLE_INTERVAL"),
    "n_queries": 36,
    "skipped": {"Exp8": "no NWChem in cg-mcp venv", "Exp14": "no multi_agent_mcp workflow"},
    "head_node": "$HEAD_NODE",
    "vllm_node": "$VLLM_NODE_NAME",
    "compute_nodes": "${COMPUTE_ARR[*]}".split(),
    "vllm_pid": "$VLLM_PID",
    "vllm_log_path": "$VLLM_LOG_PATH",
    "run_root": "$RUN_ROOT",
}
json.dump(meta, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1])
PY

# ---- finish ----------------------------------------------------------------
MAKESPAN=$(python3 -c "print(f'{float('$PROFILE_END_EPOCH')-float('$PROFILE_START_EPOCH'):.1f}')")
echo "==================== benchmark done ===================="
echo "  makespan: ${MAKESPAN}s   workload_status=$WORKLOAD_STATUS"
echo "  results : $RUN_ROOT"
echo "  report  : python3 $HERE/bench_report.py $RUN_ROOT"
echo "  NOTE: services left UP. Teardown: bash $HERE/../e2e_4_check_and_teardown.sh"
