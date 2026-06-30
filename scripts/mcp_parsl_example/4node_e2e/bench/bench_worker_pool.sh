#!/bin/bash
# Multi-tenant concurrency-N launcher for the ChemGraph compute-bound benchmark.
#
# Runs on the HEAD node. Models N independent "tenants" sharing one backend
# (one vLLM on node B + one Parsl pool on C+D) -- the multi-tenant macro layer
# from BENCHMARK_PLAN. Each tenant runs the FULL 42-query suite in its OWN
# deterministic-shuffled order, so:
#   - exactly N agent queries are in flight at once  => true concurrency N,
#   - the N tenants' phases are decorrelated (shuffle) => realistic arrival mix,
#   - the order is fixed per (seed, tenant) => reproducible & comparable.
#
# Driven by bench_cell.sh, but runnable standalone. Inputs via env:
#   QUERIES_FILE   path to bench_queries.txt (default: alongside this script)
#   RUN_ROOT       output dir (summary.csv + per-tenant logdirs land here)
#   CONCURRENCY    number of tenants = N concurrent queries (default 8)
#   VLLM_NODE_NAME vLLM host short name (for --base-url + NO_PROXY)
#   CELL           cell label for the summary rows (e.g. W4_N8); default "NA"
#   BENCH_SEED     base seed for the per-tenant shuffle (default 20260630)
#   BENCH_TIMEOUT  per-query wall cap in seconds (default 900)
#   (MODEL / MCP_PORT / MCP_SERVER_NAME / VLLM_PORT come from e2e_env.sh)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../e2e_env.sh"

QUERIES_FILE="${QUERIES_FILE:-$HERE/bench_queries.txt}"
RUN_ROOT="${RUN_ROOT:?bench_worker_pool.sh requires RUN_ROOT}"
CONCURRENCY="${CONCURRENCY:-8}"
VLLM_NODE_NAME="${VLLM_NODE_NAME:?bench_worker_pool.sh requires VLLM_NODE_NAME}"
CELL="${CELL:-NA}"
BENCH_SEED="${BENCH_SEED:-20260630}"
TIMEOUT_SECONDS="${BENCH_TIMEOUT:-900}"

[ -f "$QUERIES_FILE" ] || { echo "Missing queries file: $QUERIES_FILE" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

BASE_URL="http://$VLLM_NODE_NAME:$VLLM_PORT/v1"
MCP_URL="http://127.0.0.1:$MCP_PORT/mcp/"
SUMMARY="$RUN_ROOT/summary.csv"
SLOCK="$RUN_ROOT/.summary.lock"
TLOGS="$RUN_ROOT/tenant_logs"; mkdir -p "$TLOGS"

# Load the active query lines (skip comments/blanks). Each: id<TAB>wf<TAB>rl<TAB>query
mapfile -t QLINES < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$QUERIES_FILE")
TOTAL="${#QLINES[@]}"
[ "$TOTAL" -gt 0 ] || { echo "No active queries in $QUERIES_FILE" >&2; exit 2; }

# ---- env for the CLI workers -----------------------------------------------
e2e_activate
cd "$REPO"
# Bypass the ALCF proxy for localhost (MCP) + node B (vLLM).
_no_proxy="127.0.0.1,localhost,::1,$VLLM_NODE_NAME"
export NO_PROXY="$_no_proxy" no_proxy="$_no_proxy"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_vllm_key}"

echo "RUN_ROOT=$RUN_ROOT"
echo "CELL=$CELL  TENANTS(N)=$CONCURRENCY  QUERIES/tenant=$TOTAL  TIMEOUT=${TIMEOUT_SECONDS}s"
echo "BASE_URL=$BASE_URL  MCP_URL=$MCP_URL  MODEL=$MODEL  SEED=$BENCH_SEED"

# summary header
printf 'cell,tenant,run_index,id,status,seconds,start_epoch,end_epoch,workflow,recursion_limit,log\n' > "$SUMMARY"

# Deterministic per-tenant shuffle of [0..TOTAL): sha256("seed:tenant") -> RNG.
# Same recipe as the existing in-process harness
# (chemgraph_separate_node_vllm_exp_smoke.sh) so orders are reproducible.
tenant_order() {  # $1 = tenant_id ; echoes indices, one per line
    python3 - "$BENCH_SEED" "$1" "$TOTAL" <<'PY'
import hashlib, random, sys
seed, tenant, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed_int = int.from_bytes(hashlib.sha256(f"{seed}:{tenant}".encode()).digest()[:8], "big")
order = list(range(count))
random.Random(seed_int).shuffle(order)
print(" ".join(map(str, order)))
PY
}

# Classify a finished query exactly like the existing in-process harness
# (chemgraph_local_vllm_exp_smoke.sh:classify): exit code first, then log scan.
classify() {
    local code="$1" log="$2"
    [ "$code" -eq 124 ] && { echo TIMEOUT; return; }
    [ "$code" -ne 0 ] && { echo FAIL; return; }
    if grep -Eq 'Traceback|Unsupported workflow type|Error running workflow|Error processing query|Recursion limit|APIConnectionError|BadRequestError|"status": "failure"|CalculationFailed|failed with command|MPI_ABORT|list index out of range|too many values to unpack' "$log"; then
        echo FAIL; return
    fi
    echo PASS
}

run_one() {  # tenant run_index id workflow rl query
    local tid="$1" ridx="$2" id="$3" workflow="$4" rl="$5" query="$6"
    local log="$TLOGS/${tid}_${ridx}_${id}.log"
    printf 'tenant: %s\nrun_index: %s\nquery: %s\nrecursion_limit: %s\n' \
        "$tid" "$ridx" "$query" "$rl" > "$log.query"

    local start end secs code status
    start="$(date +%s)"
    timeout "$TIMEOUT_SECONDS" chemgraph run \
        -q "$query" \
        -m "$MODEL" \
        --base-url "$BASE_URL" \
        --mcp-url "$MCP_URL" \
        --mcp-server-name "$MCP_SERVER_NAME" \
        -w "$workflow" \
        --recursion-limit "$rl" \
        -o last_message \
        > "$log" 2>&1
    code="$?"
    end="$(date +%s)"
    secs="$((end - start))"
    status="$(classify "$code" "$log")"

    ( flock 8
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$CELL" "$tid" "$ridx" "$id" "$status" "$secs" "$start" "$end" \
        "$workflow" "$rl" "$log" >> "$SUMMARY"
    ) 8>"$SLOCK"
    printf '  [%s] #%-2s %-9s %-7s %4ss\n' "$tid" "$ridx" "$id" "$status" "$secs"
}

# One tenant: run all 42 in its own shuffled order, sequentially.
tenant_loop() {
    local tid="$1" order ridx=0 idx line id workflow rl query
    read -ra order < <(tenant_order "$tid")
    for idx in "${order[@]}"; do
        ridx=$((ridx + 1))
        line="${QLINES[$idx]}"
        IFS=$'\t' read -r id workflow rl query <<< "$line"
        run_one "$tid" "$ridx" "$id" "$workflow" "$rl" "$query"
    done
}

# ---- launch N tenants concurrently -----------------------------------------
echo "---- workload start ($CONCURRENCY tenants x $TOTAL queries) ----"
tpids=()
for i in $(seq 1 "$CONCURRENCY"); do
    tenant_loop "t$i" &
    tpids+=("$!")
done
for p in "${tpids[@]}"; do wait "$p"; done
echo "---- workload done ----"

# ---- tally -----------------------------------------------------------------
pass=$(grep -c ',PASS,' "$SUMMARY" || true)
fail=$(grep -c ',FAIL,' "$SUMMARY" || true)
to=$(grep -c ',TIMEOUT,' "$SUMMARY" || true)
exp=$((CONCURRENCY * TOTAL))
echo "summary: PASS=$pass FAIL=$fail TIMEOUT=$to  (of $exp expected)  -> $SUMMARY"
