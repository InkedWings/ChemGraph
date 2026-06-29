#!/bin/bash
# Stage 1/4 -- start vLLM (Qwen3-32B, tool-calling) on node B.
#
# MUST be run from the qsub -I primary shell on the HEAD node (so PBS_NODEFILE
# is populated and node roles resolve). vllm_manage.sh ssh-hops to node B
# itself; vLLM needs no PBS env, so the ssh hop is fine for it.
#
# Gate: does NOT return success until /v1/models AND a test chat completion
# both succeed. The model is large (~32B, TP=4) -- first load can take minutes.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/e2e_env.sh"

mkdir -p "$STATE_DIR"

echo "==================== Stage 1: start vLLM on node B ===================="

# Resolve + persist node roles for the later stages.
if ! e2e_resolve_nodes > "$NODES_FILE"; then
    exit 1
fi
HEAD_NODE="$(e2e_head_node)"
VLLM_NODE_NAME="$(e2e_vllm_node)"
COMPUTE_NODES="$(e2e_compute_nodes)"
THIS_NODE="$(hostname -s)"

echo "  resolved node roles:"
echo "    HEAD (MCP+CLI) : $HEAD_NODE"
echo "    B    (vLLM)    : $VLLM_NODE_NAME"
echo "    C..D (Parsl)   : $(echo "$COMPUTE_NODES" | tr '\n' ' ')"
echo "  this shell is on : $THIS_NODE"

if [ "$THIS_NODE" != "$HEAD_NODE" ]; then
    echo "  WARNING: this shell ($THIS_NODE) is NOT the HEAD node ($HEAD_NODE)."
    echo "  Stages 2-3 MUST run on HEAD for cross-node Parsl. Continuing anyway."
fi
if [ -z "$VLLM_NODE_NAME" ]; then
    echo "  ERROR: could not resolve a vLLM node (need >=2 nodes in the job)."
    exit 1
fi

# Per the plan: keep apptainer scratch on node-local /local/scratch; HF cache +
# container default to the Agentic dir (already populated), so we don't override
# them. VLLM_NODE pins the target node; VLLM_PBS_JOBID lets vllm_manage discover
# it if needed.
export VLLM_PBS_JOBID="${PBS_JOBID:-7228360}"
export VLLM_NODE="$VLLM_NODE_NAME"
export APPTAINER_TMPDIR="/local/scratch/$USER-vllm-tmp"
export APPTAINER_CACHEDIR="/local/scratch/$USER-vllm-cache"

echo
echo "  starting vLLM (this ssh-hops to $VLLM_NODE_NAME)..."
bash "$VLLM_MANAGE" start
echo

echo "  polling for readiness (up to ~12 min)..."
# vllm_manage.sh test does: GET /v1/models + a chat completion. Both must pass.
deadline=$(( SECONDS + 720 ))
ok=0
while [ "$SECONDS" -lt "$deadline" ]; do
    if bash "$VLLM_MANAGE" test >/tmp/e2e_vllm_test.$$ 2>&1; then
        ok=1
        break
    fi
    sleep 10
done

if [ "$ok" -ne 1 ]; then
    echo "  ERROR: vLLM did not become ready in time. Last test output:"
    sed 's/^/    /' /tmp/e2e_vllm_test.$$ 2>/dev/null | tail -30
    echo "  Check server log:  bash $VLLM_MANAGE logs 80"
    rm -f /tmp/e2e_vllm_test.$$
    exit 1
fi

echo "  vLLM is READY. Test chat completion returned:"
sed 's/^/    /' /tmp/e2e_vllm_test.$$ | tail -8
rm -f /tmp/e2e_vllm_test.$$

echo
echo "  vLLM base URL for the CLI: http://$VLLM_NODE_NAME:$VLLM_PORT/v1"
echo "==================== Stage 1 done ===================="
echo "Next: bash e2e_2_start_mcp.sh"
