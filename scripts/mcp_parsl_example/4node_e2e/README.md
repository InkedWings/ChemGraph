# 4-node end-to-end agent pipeline on Polaris

Proves the **complete** ChemGraph agent chain across a 4-node PBS job:

```
NL query -> vLLM (LLM) -> tool call -> CLI -> MCP server -> Parsl compute
         -> result -> vLLM -> final natural-language answer
```

Unlike `../run_mcp_parsl.py` (which exercises the MCP+Parsl backend directly),
this puts a real LLM (vLLM-served Qwen3-32B) in the loop and drives everything
through the `chemgraph run` CLI with the `single_agent_mcp` workflow.

## Node layout

The `qsub -I` primary shell lands on the **first** node, which is where the
PBS/PALS environment lives. Parsl's `MpiExecLauncher` (used by the MCP server's
backend) needs that environment to place workers cross-node, so the PBS-sensitive
component (MCP server) runs on the head node, and the PBS-insensitive one (vLLM,
just a GPU server) is reached by `ssh`:

| Node | Role | Reached via |
|------|------|-------------|
| HEAD (1st in `PBS_NODEFILE`) | MCP server **+** CLI | the `qsub -I` shell itself |
| B (2nd) | vLLM (Qwen3-32B, tool-calling) | `ssh` from `vllm_manage.sh` |
| C, D (rest) | Parsl compute workers | overridden `PBS_NODEFILE` (C+D only) |

Node roles are resolved from the live `PBS_NODEFILE` at run time (see
`e2e_env.sh`), so re-creating the job on different nodes needs no edits to the
role logic.

## Run order

All stages run from the **`qsub -I` primary shell on the HEAD node** (not over a
plain `ssh` hop -- an ssh session has an empty `PBS_NODEFILE` and cannot drive
cross-node `mpiexec`):

```bash
bash e2e_1_start_vllm.sh         # start vLLM on node B; gate on a real chat completion
bash e2e_2_start_mcp.sh          # start MCP server on HEAD; PBS_NODEFILE -> C+D
bash e2e_3_run_agent.sh          # run the agent query (caffeine GFN2-xTB opt)
bash e2e_4_check_and_teardown.sh # verify cross-node run_ase, then tear everything down
```

The first `run_ase` is slow: the Parsl worker pools on C+D lazy-start on the
first call. That latency is expected, not a hang.

## What success looks like

- The agent trace shows `run_ase` being called (visible with `-v`).
- `run_ase` executes on the C+D Parsl pool -- the latest `NNN/htex/` run dir's
  `block-*/<mgr>/manager.log` files show `hostname` = the C/D nodes and a
  cumulative task count >= 1 (NOT an inline run on the head node).
- The CLI prints a final natural-language answer containing the computed energy.
- Teardown leaves no orphan processes on any node.

## Machine-specific assumptions (edit before reuse)

These scripts were written for one concrete deployment. The paths in
`e2e_env.sh` are absolute and **not** portable as-is:

- `REPO`, `VENV` -- the ChemGraph checkout and its venv.
- `VLLM_MANAGE` -- an external vLLM launcher
  (`.../Agentic/vllm_manage.sh`) that serves Qwen3-32B as the
  `chemgraph-qwen3-32b` model on port 8000 with `--enable-auto-tool-choice
  --tool-call-parser hermes`, from a pre-cached HF model + apptainer container.
- `MODEL=chemgraph-qwen3-32b` -- matches that launcher's `--served-model-name`.
  The CLI accepts this unregistered name because `ChemGraph.__init__` treats any
  unknown model + `--base-url` as a vLLM/OpenAI-compatible endpoint.

The Parsl pool size is `CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE` (default 4) x the
number of compute nodes -- 8 slots for the 2-node C+D pool here. The e2e proof
itself issues a single query (1 task); concurrency is a separate phase.

## Files

- `e2e_env.sh` -- sourced config + node-role helpers (not executable).
- `e2e_1_start_vllm.sh` .. `e2e_4_check_and_teardown.sh` -- the four stages.
- Runtime state is written to `e2e_state/` next to the scripts' parent dir
  (resolved compute nodefile, MCP server PGID for teardown).
