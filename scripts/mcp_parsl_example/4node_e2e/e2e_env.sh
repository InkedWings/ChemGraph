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

# ---- node-role resolution --------------------------------------------------
# Echoes "HEAD B C D ..." (short names, unique, in PBS_NODEFILE order).
# HEAD = first, B = second, C..D = the rest.
e2e_resolve_nodes() {
    if [ -z "${PBS_NODEFILE:-}" ] || [ ! -f "${PBS_NODEFILE:-}" ]; then
        echo "ERROR: PBS_NODEFILE not set/found. Run inside the qsub -I shell." >&2
        return 1
    fi
    # Unique nodes preserving first-seen order; strip domain to short names.
    awk '!seen[$0]++' "$PBS_NODEFILE" | sed 's/\..*//'
}

# Convenience accessors (re-resolve each call; cheap).
e2e_head_node() { e2e_resolve_nodes | sed -n '1p'; }
e2e_vllm_node() { e2e_resolve_nodes | sed -n '2p'; }
e2e_compute_nodes() { e2e_resolve_nodes | sed -n '3,$p'; }  # C, D, ...
