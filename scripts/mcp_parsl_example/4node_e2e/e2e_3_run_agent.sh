#!/bin/bash
# Stage 3/4 -- run the agent query on the HEAD node.
#
# Drives the full chain: NL query -> vLLM (node B) -> tool call -> CLI ->
# MCP server (local on HEAD) -> Parsl compute (C+D) -> result -> vLLM ->
# final NL answer.
#
# MUST run on the HEAD node (the CLI talks to the local MCP server on
# 127.0.0.1 and to vLLM on node B). NO_PROXY must include the node B short
# name or the cross-node CLI->vLLM call hangs behind the ALCF proxy.
#
# The first run_ase is slow: Parsl worker pools on C+D lazy-start on the first
# call. That latency is expected, not an error.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/e2e_env.sh"

mkdir -p "$STATE_DIR" "$LOG_DIR"

echo "==================== Stage 3: run the agent query ===================="

# ---- node context ----------------------------------------------------------
if ! e2e_resolve_nodes >/dev/null; then
    exit 1
fi
HEAD_NODE="$(e2e_head_node)"
VLLM_NODE_NAME="$(e2e_vllm_node)"
THIS_NODE="$(hostname -s)"

if [ "$THIS_NODE" != "$HEAD_NODE" ]; then
    echo "  ERROR: not on HEAD node ($HEAD_NODE). Run Stage 3 on HEAD."
    exit 1
fi
if [ -z "$VLLM_NODE_NAME" ]; then
    echo "  ERROR: could not resolve the vLLM node."
    exit 1
fi

# Sanity: MCP server up?
if [ ! -f "$MCP_PGID_FILE" ]; then
    echo "  ERROR: $MCP_PGID_FILE missing -- run Stage 2 first."
    exit 1
fi

BASE_URL="http://$VLLM_NODE_NAME:$VLLM_PORT/v1"
MCP_URL="http://127.0.0.1:$MCP_PORT/mcp/"

echo "  vLLM base URL : $BASE_URL"
echo "  MCP URL       : $MCP_URL"
echo "  model         : $MODEL"
echo "  workflow      : $WORKFLOW"

# ---- env -------------------------------------------------------------------
e2e_activate
cd "$REPO"

# Bypass ALCF proxy for: localhost (MCP), node B (vLLM). The CLI itself does
# NOT need PBS_NODEFILE (only the MCP server's Parsl backend does), so we leave
# the CLI's PBS env untouched.
_no_proxy="127.0.0.1,localhost,::1,$VLLM_NODE_NAME"
export NO_PROXY="$_no_proxy"
export no_proxy="$_no_proxy"
# ChatOpenAI requires an api_key value even though vLLM ignores it.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy_vllm_key}"

# One molecule, GFN2-xTB geometry opt: forces the smiles->coords->run_ase chain
# and yields a final energy to report. Keep it to one molecule for the first
# end-to-end proof.
QUERY="${E2E_QUERY:-Run a geometry optimization of caffeine (SMILES CN1C=NC2=C1C(=O)N(C(=O)N2C)C) using GFN2-xTB and report its final total energy in eV.}"

AGENT_LOG="$LOG_DIR/agent.log"
echo "  query: $QUERY"
echo "  agent output -> $AGENT_LOG (also streamed below)"
echo "----------------------------------------------------------------------"

# -v for INFO-level (shows tool_calls routing). tee so Stage 4 can inspect.
set -o pipefail
chemgraph run \
    -q "$QUERY" \
    -m "$MODEL" \
    --base-url "$BASE_URL" \
    --mcp-url "$MCP_URL" \
    --mcp-server-name "$MCP_SERVER_NAME" \
    -w "$WORKFLOW" \
    -v 2>&1 | tee "$AGENT_LOG"
rc=${PIPESTATUS[0]}

echo "----------------------------------------------------------------------"
echo "  CLI exit code: $rc"
echo "==================== Stage 3 done ===================="
echo "Next: bash e2e_4_check_and_teardown.sh"
exit "$rc"
