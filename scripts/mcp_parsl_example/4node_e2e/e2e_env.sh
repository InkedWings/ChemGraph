# Shared config + helpers for the 4-node end-to-end agent pipeline.
#
# Sourced by e2e_1..e2e_4. NOT executable on its own.
#
# Node roles for PBS job 7228360 (select=4). The qsub -I primary shell lands
# on the FIRST node = HEAD; we run the MCP server + CLI there so Parsl/mpiexec
# inherits the real PBS/PALS env for cross-node worker placement. vLLM goes on
# node B (reached by ssh from vllm_manage.sh; it needs no PBS env). Parsl
# compute workers run on nodes C+D via an overridden PBS_NODEFILE.
#
# Node roles are derived from the live PBS_NODEFILE at run time (not
# hardcoded), so the same scripts work if the job is recreated on other nodes:
#   HEAD = first node in PBS_NODEFILE      -> MCP server + CLI (this shell)
#   B    = second node                     -> vLLM
#   C,D  = remaining nodes                 -> Parsl compute pool

# ---- static config ---------------------------------------------------------
REPO=/lus/eagle/projects/ChemGraph/zhijing/ChemGraph
VENV=/lus/eagle/projects/ChemGraph/zhijing/venvs/cg-mcp
VLLM_MANAGE=/lus/eagle/projects/lc-mpi/ZhijingYe/Agentic/vllm_manage.sh

E2E_DIR=/lus/eagle/projects/ChemGraph/zhijing
STATE_DIR="$E2E_DIR/e2e_state"          # runtime state shared across stages
LOG_DIR="$REPO/cg_logs/e2e"             # MCP server + agent logs

MCP_PORT=9005
VLLM_PORT=8000
MCP_SERVER_NAME="ChemGraph ASE Tools"   # must match CGFastMCP name in ase_mcp_hpc.py
MODEL=chemgraph-qwen3-32b               # served-model-name from vllm_manage.sh
WORKFLOW=single_agent_mcp

# Files written by earlier stages, read by later ones.
COMPUTE_NODEFILE="$STATE_DIR/mcp_compute_nodes.txt"   # nodes C+D for Parsl
MCP_PGID_FILE="$STATE_DIR/mcp.pgid"                   # MCP server process group
NODES_FILE="$STATE_DIR/nodes.txt"                     # resolved HEAD/B/C/D roles

# ---- venv activation (every shell / every node) ----------------------------
e2e_activate() {
    module use /soft/modulefiles
    module load conda
    conda activate base
    source "$VENV/bin/activate"
}

# ---- PBS env reconstruction ------------------------------------------------
# PBS_NODEFILE is a per-process env var: it lives in the qsub -I process tree,
# NOT node-wide. A fresh `ssh <head>` session does NOT inherit it. But PBS keeps
# the job's nodefile on disk at /var/spool/pbs/aux/<jobid>, so we can rebuild
# PBS_NODEFILE/PBS_JOBID from the user's single running job. This lets every
# stage run from any ssh session on the head node, not only the qsub -I shell.
e2e_ensure_pbs_env() {
    [ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE:-}" ] && return 0
    local aux=/var/spool/pbs/aux jobfile=""
    # Prefer the jobid qstat reports as Running for this user; else newest aux file.
    local jid
    jid="$(qstat -u "$USER" 2>/dev/null | awk '$10=="R"{print $1}' | head -1 | cut -d. -f1)"
    if [ -n "$jid" ]; then
        jobfile="$(ls -1 "$aux"/${jid}.* 2>/dev/null | head -1)"
    fi
    [ -z "$jobfile" ] && jobfile="$(ls -1t "$aux"/*.polaris-pbs* 2>/dev/null | head -1)"
    if [ -n "$jobfile" ] && [ -f "$jobfile" ]; then
        export PBS_NODEFILE="$jobfile"
        export PBS_JOBID="${PBS_JOBID:-$(basename "$jobfile")}"
        echo "  (rebuilt PBS_NODEFILE from $jobfile)" >&2
        return 0
    fi
    return 1
}

# ---- node-role resolution --------------------------------------------------
# Echoes "HEAD B C D ..." (short names, unique, in PBS_NODEFILE order).
# HEAD = first, B = second, C..D = the rest.
e2e_resolve_nodes() {
    if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE:-}" ]; then
        e2e_ensure_pbs_env || true
    fi
    if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE:-}" ]; then
        echo "ERROR: PBS_NODEFILE not set and could not be rebuilt from /var/spool/pbs/aux." >&2
        echo "       Run on the head node of a running PBS job." >&2
        return 1
    fi
    # Unique nodes preserving first-seen order; strip domain to short names.
    awk '!seen[$0]++' "$PBS_NODEFILE" | sed 's/\..*//'
}

# Convenience accessors (re-resolve each call; cheap).
e2e_head_node() { e2e_resolve_nodes | sed -n '1p'; }
e2e_vllm_node() { e2e_resolve_nodes | sed -n '2p'; }
e2e_compute_nodes() { e2e_resolve_nodes | sed -n '3,$p'; }  # C, D, ...
