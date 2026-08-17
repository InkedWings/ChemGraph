"""Lightweight per-query instrumentation for the evaluation harness.

Collects, for each evaluated query:

* **LLM** wall-time and token usage (prompt/completion/total/cached), via a
  callback handler attached to the model instance.  Attaching it as a model
  field (``llm.callbacks = [handler]``) is deliberate: it survives the
  ``llm.bind_tools(...)`` performed inside the graph nodes, whereas callbacks
  passed through the run config do **not** reach the bare ``llm.invoke(...)``
  calls those nodes make.
* **Per-tool** execution wall-time, via LangGraph ``ToolNode`` callbacks
  (these *do* fire through the run config).
* **Calculator/model load** time, via a process-global accumulator written by
  :func:`chemgraph.tools.ase_core.load_calculator`.  ``run_ase`` reloads the
  ASE/MACE calculator on every call (no caching), and that load happens inside
  the tool — possibly on an executor thread — so it cannot be measured from the
  callback alone.

All capture is best-effort and must never raise into the evaluation path.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler

# ---------------------------------------------------------------------------
# Process-global calculator-load accumulator.
#
# The runner resets this immediately before each query (``reset_calc_load``)
# and reads it immediately after (``pop_calc_load``).  Because the eval runs
# queries strictly sequentially, a process-global is safe even though the load
# itself may occur on a tool-executor thread.
# ---------------------------------------------------------------------------
_calc_lock = threading.Lock()
_calc_load_seconds = 0.0
_calc_load_count = 0


def reset_calc_load() -> None:
    """Clear the calculator-load accumulator (call before each query)."""
    global _calc_load_seconds, _calc_load_count
    with _calc_lock:
        _calc_load_seconds = 0.0
        _calc_load_count = 0


def record_calc_load(seconds: float) -> None:
    """Record one calculator/model load.

    Called from :func:`chemgraph.tools.ase_core.load_calculator`.
    """
    global _calc_load_seconds, _calc_load_count
    with _calc_lock:
        _calc_load_seconds += float(seconds)
        _calc_load_count += 1


def pop_calc_load() -> Dict[str, float]:
    """Return and clear the accumulated calculator-load time/count."""
    global _calc_load_seconds, _calc_load_count
    with _calc_lock:
        out = {
            "calc_load_s": round(_calc_load_seconds, 4),
            "calc_load_count": _calc_load_count,
        }
        _calc_load_seconds = 0.0
        _calc_load_count = 0
    return out


class EvalInstrument(BaseCallbackHandler):
    """Capture LLM timing + token usage and per-tool execution time.

    Attach the *same instance* in two places:

    * to the model — ``llm.callbacks = [inst]`` (set by ``ChemGraph`` via the
      ``instrument_callbacks`` argument) — to capture LLM events;
    * to the run config — ``config={"callbacks": [inst]}`` — to capture tool
      events fired by the LangGraph ``ToolNode``.

    Call :meth:`reset` before each query and :meth:`snapshot` after.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero all counters (call before each query)."""
        with self._lock:
            self._llm_starts: Dict[Any, float] = {}
            self._tool_starts: Dict[Any, tuple] = {}
            self.llm_calls = 0
            self.llm_time_s = 0.0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.cached_tokens = 0
            self.tools: Dict[str, Dict[str, float]] = {}

    # ---- LLM events -------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        with self._lock:
            self._llm_starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        # Completion-style models (not used by the eval, but harmless).
        with self._lock:
            self._llm_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        now = time.perf_counter()
        with self._lock:
            t0 = self._llm_starts.pop(run_id, None)
            if t0 is not None:
                self.llm_time_s += now - t0
            self.llm_calls += 1

            p = c = t = cached = 0
            # Preferred: standard usage_metadata on the AIMessage.
            try:
                msg = response.generations[0][0].message
                um = getattr(msg, "usage_metadata", None) or {}
                p = um.get("input_tokens", 0) or 0
                c = um.get("output_tokens", 0) or 0
                t = um.get("total_tokens", 0) or 0
                cached = (um.get("input_token_details") or {}).get("cache_read", 0) or 0
            except Exception:
                pass
            # Fallback: provider llm_output token_usage.
            if not t:
                try:
                    tu = (response.llm_output or {}).get("token_usage", {}) or {}
                    p = tu.get("prompt_tokens", p) or p
                    c = tu.get("completion_tokens", c) or c
                    t = tu.get("total_tokens", t) or t
                except Exception:
                    pass

            self.prompt_tokens += p
            self.completion_tokens += c
            self.total_tokens += t
            self.cached_tokens += cached

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            self._llm_starts.pop(run_id, None)

    # ---- Tool events ------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name", "unknown")
        with self._lock:
            self._tool_starts[run_id] = (time.perf_counter(), name)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._close_tool(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._close_tool(run_id)

    def _close_tool(self, run_id) -> None:
        now = time.perf_counter()
        with self._lock:
            rec = self._tool_starts.pop(run_id, None)
            if rec is None:
                return
            t0, name = rec
            entry = self.tools.setdefault(name, {"calls": 0, "time_s": 0.0})
            entry["calls"] += 1
            entry["time_s"] += now - t0

    # ---- snapshot ---------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of what has been captured."""
        with self._lock:
            tools = {
                k: {"calls": v["calls"], "time_s": round(v["time_s"], 4)}
                for k, v in self.tools.items()
            }
            return {
                "llm": {
                    "calls": self.llm_calls,
                    "time_s": round(self.llm_time_s, 4),
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "cached_tokens": self.cached_tokens,
                },
                "tools": tools,
                "tool_time_s": round(sum(v["time_s"] for v in self.tools.values()), 4),
                "tool_calls": sum(v["calls"] for v in self.tools.values()),
            }


# ---------------------------------------------------------------------------
# Aggregation helpers (sum per-query records into model+workflow totals).
# ---------------------------------------------------------------------------
def aggregate_tokens(per_query: List[dict]) -> Dict[str, Any]:
    """Sum per-query token-usage dicts into a model+workflow total."""
    keys = ["prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "llm_calls"]
    agg = {k: 0 for k in keys}
    for q in per_query:
        for k in keys:
            agg[k] += int(q.get(k, 0) or 0)
    n = len(per_query) or 1
    agg["n_queries"] = len(per_query)
    agg["avg_total_tokens_per_query"] = round(agg["total_tokens"] / n, 1)
    return agg


def aggregate_timing(per_query: List[dict], agent_init_s: float = 0.0) -> Dict[str, Any]:
    """Sum per-query timing dicts into a model+workflow total.

    ``agent_init_s`` is the one-time ChemGraph/graph construction cost for this
    (model, workflow) pair — part of "loading" time, reported once here rather
    than per query.
    """
    time_keys = [
        "agent_wall_s", "llm_s", "tool_s", "tool_compute_s",
        "calc_load_s", "other_s", "judge_s",
    ]
    count_keys = ["llm_calls", "calc_load_count"]
    agg: Dict[str, Any] = {k: 0.0 for k in time_keys}
    for k in count_keys:
        agg[k] = 0
    per_tool: Dict[str, Dict[str, float]] = {}

    for q in per_query:
        for k in time_keys:
            agg[k] += float(q.get(k, 0.0) or 0.0)
        for k in count_keys:
            agg[k] += int(q.get(k, 0) or 0)
        for name, info in (q.get("per_tool") or {}).items():
            entry = per_tool.setdefault(name, {"calls": 0, "time_s": 0.0})
            entry["calls"] += int(info.get("calls", 0) or 0)
            entry["time_s"] += float(info.get("time_s", 0.0) or 0.0)

    for k in time_keys:
        agg[k] = round(agg[k], 3)
    for name in per_tool:
        per_tool[name]["time_s"] = round(per_tool[name]["time_s"], 3)

    agg["per_tool"] = per_tool
    agg["agent_init_s"] = round(agent_init_s, 3)
    agg["n_queries"] = len(per_query)
    return agg
