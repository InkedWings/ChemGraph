#!/bin/bash
# Stage 2/4 -- start the ASE MCP server (Parsl backend) on the HEAD node.
#
# MUST run in the qsub -I primary shell on the HEAD node. Parsl's
# MpiExecLauncher needs the real PBS/PALS env to place workers cross-node; an
# ssh session has an empty PBS_NODEFILE and would silently fall back to 1 node.
#
# We override PBS_NODEFILE to contain ONLY nodes C+D, so mpiexec places the
# Parsl worker pools on those two compute nodes (HEAD stays free for the
# server+CLI, node B stays free for vLLM).
#
# The server is launched in its OWN process group (setsid) and LEFT RUNNING in
# the background -- teardown happens in Stage 4 via the saved PGID. This script
# therefore does NOT trap-kill on exit.
#
# NOTE: the Parsl backend is lazy-init -- worker pools on C+D only spin up on
# the FIRST run_ase call (Stage 3), not at server start.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/e2e_env.sh"

mkdir -p "$STATE_DIR" "$LOG_DIR"

echo "==================== Stage 2: start MCP server on HEAD ===================="

# ---- node context (resolve from the REAL PBS_NODEFILE before overriding) ----
# Rebuild PBS_NODEFILE in-process if this is a fresh ssh session (not qsub -I).
# Verified: ssh-to-head + rebuilt PBS_NODEFILE drives cross-node PALS mpiexec fine.
e2e_ensure_pbs_env || { echo "  ERROR: no running PBS job / nodefile found." >&2; exit 1; }
if ! e2e_resolve_nodes >/dev/null; then
    exit 1
fi
HEAD_NODE="$(e2e_head_node)"
THIS_NODE="$(hostname -s)"
mapfile -t COMPUTE_ARR < <(e2e_compute_nodes)

echo "  this shell is on : $THIS_NODE"
echo "  HEAD node        : $HEAD_NODE"
echo "  compute (C..D)   : ${COMPUTE_ARR[*]}"

if [ "$THIS_NODE" != "$HEAD_NODE" ]; then
    echo "  ERROR: not on HEAD node ($HEAD_NODE). Must run on the head node so PALS"
    echo "  mpiexec can place Parsl workers cross-node. (An ssh session works as long"
    echo "  as it is on the head node and PBS_NODEFILE was rebuilt -- done above.)"
    exit 1
fi
if [ "${#COMPUTE_ARR[@]}" -lt 1 ]; then
    echo "  ERROR: no compute nodes resolved (need >=3 nodes in the job)."
    exit 1
fi

# ---- build the compute-only nodefile (C+D) ---------------------------------
printf '%s\n' "${COMPUTE_ARR[@]}" > "$COMPUTE_NODEFILE"
echo "  wrote compute nodefile -> $COMPUTE_NODEFILE:"
sed 's/^/    /' "$COMPUTE_NODEFILE"

# ---- env for the Parsl backend ---------------------------------------------
e2e_activate
cd "$REPO"

export CHEMGRAPH_EXECUTION_BACKEND=parsl
export COMPUTE_SYSTEM=polaris
# Explicit worker_init so workers on C+D get a working interpreter (module load
# + venv activate), not just a bare venv activate.
# Honor a pre-set CHEMGRAPH_WORKER_INIT (the benchmark sweep injects NWChem PATH +
# NWCHEM_BASIS_LIBRARY + OMP/OPENBLAS thread pins here); else use the default.
export CHEMGRAPH_WORKER_INIT="${CHEMGRAPH_WORKER_INIT:-module use /soft/modulefiles; module load conda; conda activate base; source $VENV/bin/activate; export TMPDIR=/tmp}"
export CHEMGRAPH_LOG_DIR="$LOG_DIR"
# Override PBS_NODEFILE -> Parsl/mpiexec uses ONLY C+D.
export PBS_NODEFILE="$COMPUTE_NODEFILE"
# Bypass the ALCF proxy for all local + in-job traffic.
_no_proxy="127.0.0.1,localhost,::1,$(echo "${COMPUTE_ARR[*]}" | tr ' ' ',')"
export NO_PROXY="$_no_proxy"
export no_proxy="$_no_proxy"

echo "  backend=$CHEMGRAPH_EXECUTION_BACKEND system=$COMPUTE_SYSTEM"
echo "  PBS_NODEFILE (override)=$PBS_NODEFILE"
echo "  CHEMGRAPH_LOG_DIR=$CHEMGRAPH_LOG_DIR"

# ---- launch server (own process group, background, persistent) -------------
SERVER_LOG="$LOG_DIR/server.log"
: > "$SERVER_LOG"
# -u (unbuffered) so uvicorn's "Application startup complete" / "Uvicorn
# running" lines flush to the log immediately instead of sitting in Python's
# block buffer when stdout is a file (which made the gate below time out
# even though the server was actually up).
setsid python -u -m chemgraph.mcp.ase_mcp_hpc \
    --transport streamable_http --host 0.0.0.0 --port "$MCP_PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$(ps -o pgid= -p "$SERVER_PID" | tr -d ' ')
echo "  server PID=$SERVER_PID PGID=$SERVER_PGID  (log: $SERVER_LOG)"
echo "$SERVER_PGID" > "$MCP_PGID_FILE"

# ---- gate on startup -------------------------------------------------------
echo "  waiting for startup..."
up=0
# Two independent readiness signals (either suffices): the uvicorn startup
# lines in the log, OR the port actually listening (authoritative since the
# server is local to this shell). 240 * 0.5s = up to 2 min.
for _ in $(seq 1 240); do
    if grep -q "Application startup complete\|Uvicorn running" "$SERVER_LOG" 2>/dev/null; then
        up=1
        break
    fi
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$MCP_PORT$"; then
        up=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  ERROR: server died during startup. Log tail:"
        tail -40 "$SERVER_LOG" | sed 's/^/    /'
        exit 1
    fi
    sleep 0.5
done

if [ "$up" -ne 1 ]; then
    echo "  ERROR: server did not report startup in time. Log tail:"
    tail -40 "$SERVER_LOG" | sed 's/^/    /'
    exit 1
fi

echo "  MCP server is UP at http://127.0.0.1:$MCP_PORT/mcp/"
echo "  (Parsl workers on C+D will lazy-start on the first run_ase in Stage 3.)"

# Sentinel: record the pool size this server was launched with, so the benchmark
# sweep can assert a cell runs against an MCP started for the matching W (the
# pool size is baked at startup and cannot change without a restart).
echo "${CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE:-4}" > "$STATE_DIR/mcp_workers_per_node"
echo "==================== Stage 2 done ===================="
echo "Next: bash e2e_3_run_agent.sh"
