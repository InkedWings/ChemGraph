#!/bin/bash
# Stage 4/4 -- verify end-to-end success, then tear everything down.
#
# MUST run on the HEAD node (it kills the local MCP process group and ssh-kills
# Parsl workers on C+D). Run AFTER Stage 3.
#
# Verification (the claim of success):
#   1. the agent printed a final NL answer containing an energy (Stage 3 log),
#   2. run_ase actually ran on the C+D Parsl pool (NOT inline): the latest
#      Parsl run dir's manager.log files show hostname = the C+D nodes and a
#      cumulative task count >= 1. (Same authoritative method as the perf verify.)
# Teardown:
#   vLLM (node B) -> MCP server PGID (HEAD) -> orphan Parsl workers (C+D).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/e2e_env.sh"

echo "==================== Stage 4: verify + teardown ===================="

# ---- node context ----------------------------------------------------------
if ! e2e_resolve_nodes >/dev/null; then
    exit 1
fi
HEAD_NODE="$(e2e_head_node)"
VLLM_NODE_NAME="$(e2e_vllm_node)"
THIS_NODE="$(hostname -s)"
mapfile -t COMPUTE_ARR < <(e2e_compute_nodes)

if [ "$THIS_NODE" != "$HEAD_NODE" ]; then
    echo "  WARNING: not on HEAD ($HEAD_NODE); teardown of local MCP PGID may not apply."
fi

AGENT_LOG="$LOG_DIR/agent.log"
overall_ok=1

# ---- Check 1: agent final answer contains an energy ------------------------
echo "-------------------- Check 1: agent final answer --------------------"
if [ ! -f "$AGENT_LOG" ]; then
    echo "  MISS: $AGENT_LOG not found (did Stage 3 run?)."
    overall_ok=0
else
    if grep -qiE "traceback|unsupported model|connection refused|errno" "$AGENT_LOG"; then
        echo "  WARN: agent log contains error-like lines:"
        grep -iE "traceback|unsupported model|connection refused|errno" "$AGENT_LOG" | head -5 | sed 's/^/    /'
    fi
    # Look for a number followed by an energy unit (eV / hartree / kcal).
    if grep -qiE "[-+]?[0-9]+\.[0-9]+[[:space:]]*(ev|hartree|kcal)" "$AGENT_LOG"; then
        echo "  OK: final answer mentions an energy:"
        grep -iE "[-+]?[0-9]+\.[0-9]+[[:space:]]*(ev|hartree|kcal)" "$AGENT_LOG" | tail -3 | sed 's/^/    /'
    else
        echo "  MISS: no energy value found in the agent output. Tail:"
        tail -15 "$AGENT_LOG" | sed 's/^/    /'
        overall_ok=0
    fi
    # Tool-call routing visible (single_agent_mcp called run_ase)?
    if grep -qiE "run_ase" "$AGENT_LOG"; then
        echo "  OK: run_ase tool call present in the agent trace."
    else
        echo "  MISS: no run_ase tool call seen in the agent trace."
        overall_ok=0
    fi
fi

# ---- Check 2: run_ase ran on the C+D Parsl pool ----------------------------
echo "-------------------- Check 2: tasks per node (Parsl logs) --------------------"
LATEST=$(ls -dt "$REPO"/[0-9][0-9][0-9]/htex/ 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "  MISS: no Parsl run dir ($REPO/NNN/htex/) found -- backend may have run inline."
    overall_ok=0
else
    echo "  latest Parsl run dir: $LATEST"
    _total=0
    _nodes_seen=""
    for mlog in "$LATEST"block-*/*/manager.log; do
        [ -f "$mlog" ] || continue
        hn=$(grep -oE "'hostname': '[^']+'" "$mlog" 2>/dev/null | head -1 | sed "s/.*: '//; s/'//")
        cnt=$(grep -oE "cumulative count of tasks: [0-9]+" "$mlog" 2>/dev/null | grep -oE "[0-9]+$" | sort -n | tail -1)
        echo "    ${hn:-<unknown>} : ${cnt:-0} tasks"
        _total=$((_total + ${cnt:-0}))
        [ -n "$hn" ] && _nodes_seen="$_nodes_seen ${hn%%.*}"
    done
    echo "    ---- total: $_total task(s)"
    # Confirm the nodes that ran tasks are the compute nodes C+D.
    on_compute=0
    for cn in "${COMPUTE_ARR[@]}"; do
        echo "$_nodes_seen" | tr ' ' '\n' | grep -qx "$cn" && on_compute=$((on_compute + 1))
    done
    if [ "$_total" -ge 1 ] && [ "$on_compute" -ge 1 ]; then
        echo "  OK: run_ase executed on the compute pool ($on_compute of ${#COMPUTE_ARR[@]} C/D nodes ran tasks)."
    else
        echo "  MISS: tasks did not land on the C+D compute nodes."
        overall_ok=0
    fi
fi

# ---- Verdict ---------------------------------------------------------------
echo "-------------------- Verdict --------------------"
if [ "$overall_ok" -eq 1 ]; then
    echo "  END-TO-END: PASS"
else
    echo "  END-TO-END: NEEDS REVIEW (see misses above). Proceeding to teardown."
fi

# ---- Teardown --------------------------------------------------------------
echo "-------------------- Teardown --------------------"

# 1) vLLM on node B
if [ -n "$VLLM_NODE_NAME" ]; then
    echo "  stopping vLLM on $VLLM_NODE_NAME ..."
    VLLM_PBS_JOBID="${PBS_JOBID:-7228360}" VLLM_NODE="$VLLM_NODE_NAME" \
        bash "$VLLM_MANAGE" stop 2>&1 | sed 's/^/    /' || true
fi

# 2) MCP server process group on HEAD
if [ -f "$MCP_PGID_FILE" ]; then
    PGID="$(cat "$MCP_PGID_FILE")"
    if [ -n "$PGID" ]; then
        echo "  killing MCP server process group (PGID=$PGID) ..."
        kill -TERM "-$PGID" 2>/dev/null || true
        sleep 3
        kill -9 "-$PGID" 2>/dev/null || true
    fi
    rm -f "$MCP_PGID_FILE"
else
    echo "  no MCP PGID file -- skipping (already torn down?)."
fi

# 3) Orphan Parsl workers on C+D (mpiexec-launched; they outlive the server)
for cn in "${COMPUTE_ARR[@]}"; do
    echo "  sweeping Parsl workers on $cn ..."
    ssh -o StrictHostKeyChecking=no "$cn" \
        'for p in $(pgrep -u $USER -f process_worker_pool); do kill -9 $p 2>/dev/null; done' \
        2>/dev/null || true
done

# ---- Confirm clean (ps|grep -v grep, NOT pgrep which self-matches) ---------
echo "-------------------- Post-teardown check --------------------"
echo "  MCP/uvicorn on HEAD:"
ps -u "$USER" -o pid,cmd | grep -E "ase_mcp_hpc|uvicorn" | grep -v grep | sed 's/^/    /' || true
[ -z "$(ps -u "$USER" -o pid,cmd | grep -E 'ase_mcp_hpc|uvicorn' | grep -v grep)" ] && echo "    (none)"
for cn in "${COMPUTE_ARR[@]}"; do
    left=$(ssh -o StrictHostKeyChecking=no "$cn" \
        'ps -u $USER -o pid,cmd | grep process_worker_pool | grep -v grep | wc -l' 2>/dev/null || echo "?")
    echo "  $cn: $left Parsl worker proc(s) left"
done

echo "==================== Stage 4 done ===================="
[ "$overall_ok" -eq 1 ] && exit 0 || exit 1
