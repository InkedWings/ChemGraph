#!/usr/bin/env python3
"""Summarize a ChemGraph compute-bound benchmark run (Exp C1).

Reads one RUN_ROOT produced by bench_run.sh and reports:
  - end-to-end throughput (PASS/makespan) and makespan
  - per-query latency table + status breakdown
  - peak/mean CPU+GPU utilization on the compute nodes (C+D) vs the vLLM node (B)
    -> the compute-bound claim: compute nodes high, vLLM node low
  - vLLM /metrics delta + sampled peaks (running/waiting queue, KV-cache, tokens)
    -> the LLM backend stays largely idle

Self-contained: only stdlib. Column names match the Agentic samplers
(chemgraph_profile_{cpu,gpu}_sampler.py) and standard vLLM metric names.
"""
import csv
import glob
import json
import os
import re
import sys


def load_summary(run_root):
    path = os.path.join(run_root, "summary.csv")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    return rows


def col_stats(path, column):
    """Return (peak, mean, n) of a numeric column in a sampler CSV, ignoring blanks."""
    vals = []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(column, "")
            if v not in ("", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    if not vals:
        return None
    return max(vals), sum(vals) / len(vals), len(vals)


def cpu_peak_mean(path):
    return col_stats(path, "cpu_util_pct")


def gpu_peak_mean(path):
    """GPU CSV has one row per GPU per tick; aggregate peak/mean over all rows."""
    return col_stats(path, "gpu_util_pct")


_METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


def parse_prom(path, names):
    """Return {metric_name: float} for the requested names from a prom snapshot.

    For metrics that appear multiple times (labels), keeps the max.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = _METRIC_RE.match(line.strip())
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if name in names:
                try:
                    fv = float(val)
                except ValueError:
                    continue
                out[name] = max(out.get(name, float("-inf")), fv)
    return out


def peak_from_samples(path, names):
    """Peak of each metric across the sampled /metrics stream (# sample_epoch blocks)."""
    peaks = {n: float("-inf") for n in names}
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            m = _METRIC_RE.match(line.strip())
            if not m:
                continue
            name, val = m.group(1), m.group(2)
            if name in peaks:
                try:
                    peaks[name] = max(peaks[name], float(val))
                except ValueError:
                    pass
    return {k: v for k, v in peaks.items() if v != float("-inf")}


def mean_cpu(path):
    """Mean cpu_util_pct over a sampler CSV, or None."""
    s = col_stats(path, "cpu_util_pct")
    return s[1] if s else None


def report_sweep(sweep_root):
    """Aggregate every cell under <sweep_root>/cells/* into a grid CSV + table.

    Emits <sweep_root>/grid_summary.csv with one row per cell:
      cell,N,W,pool_P,omp,makespan_s,throughput_per_min,
      pass,fail,timeout,total,cd_cpu_mean,vllm_cpu_mean,vllm_run_peak,vllm_wait_peak,kv_peak_pct
    """
    cells_dir = os.path.join(sweep_root, "cells")
    if not os.path.isdir(cells_dir):
        print(f"no cells/ under {sweep_root}", file=sys.stderr)
        sys.exit(2)

    qnames = {
        "vllm:num_requests_running", "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc",
    }
    out_rows = []
    for cell in sorted(os.listdir(cells_dir)):
        cdir = os.path.join(cells_dir, cell)
        meta_p = os.path.join(cdir, "profile_metadata.json")
        if not os.path.exists(meta_p):
            continue
        with open(meta_p) as _mf:
            meta = json.load(_mf)
        W = meta.get("W_workers_per_node", "")
        N = meta.get("N_concurrency", "")
        P = meta.get("pool_size_P", "")
        omp = meta.get("omp_num_threads", "")
        try:
            makespan = float(meta["profile_end_epoch"]) - float(meta["profile_start_epoch"])
        except Exception:
            makespan = None

        pa = fa = to = tot = 0
        for r in load_summary(cdir):
            tot += 1
            st = r.get("status", "")
            pa += st == "PASS"; fa += st == "FAIL"; to += st == "TIMEOUT"
        thr = (pa / makespan * 60) if (makespan and makespan > 0) else None

        # mean CPU across compute nodes
        cd_means = []
        for cn in meta.get("compute_nodes", []):
            m = mean_cpu(os.path.join(cdir, f"cpu_{cn}.csv"))
            if m is not None:
                cd_means.append(m)
        cd_cpu = sum(cd_means) / len(cd_means) if cd_means else None
        vllm_cpu = mean_cpu(os.path.join(cdir, "cpu_vllm.csv"))

        peaks = peak_from_samples(os.path.join(cdir, "vllm_metrics_samples.prom"), qnames)
        run_peak = peaks.get("vllm:num_requests_running")
        wait_peak = peaks.get("vllm:num_requests_waiting")
        kv = peaks.get("vllm:kv_cache_usage_perc", peaks.get("vllm:gpu_cache_usage_perc"))
        kv_pct = (kv * 100 if (kv is not None and kv <= 1) else kv)

        out_rows.append(dict(
            cell=cell, N=N, W=W, pool_P=P, omp=omp,
            makespan_s=(f"{makespan:.1f}" if makespan else ""),
            throughput_per_min=(f"{thr:.2f}" if thr else ""),
            pass_=pa, fail=fa, timeout=to, total=tot,
            cd_cpu_mean=(f"{cd_cpu:.0f}" if cd_cpu is not None else ""),
            vllm_cpu_mean=(f"{vllm_cpu:.0f}" if vllm_cpu is not None else ""),
            vllm_run_peak=(f"{run_peak:.0f}" if run_peak is not None else ""),
            vllm_wait_peak=(f"{wait_peak:.0f}" if wait_peak is not None else ""),
            kv_peak_pct=(f"{kv_pct:.1f}" if kv_pct is not None else ""),
        ))

    cols = ["cell", "N", "W", "pool_P", "omp", "makespan_s", "throughput_per_min",
            "pass_", "fail", "timeout", "total", "cd_cpu_mean", "vllm_cpu_mean",
            "vllm_run_peak", "vllm_wait_peak", "kv_peak_pct"]
    grid_csv = os.path.join(sweep_root, "grid_summary.csv")
    with open(grid_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c.rstrip("_") for c in cols])
        for r in sorted(out_rows, key=lambda x: (int(x["W"] or 0), int(x["N"] or 0))):
            w.writerow([r[c] for c in cols])

    print("=" * 78)
    print(f"ChemGraph compute-bound SWEEP grid  ({len(out_rows)} cells)")
    print(f"  {sweep_root}")
    print("=" * 78)
    hdr = ["cell", "N", "W", "P", "omp", "makespan", "thr/min",
           "PASS", "FAIL", "TO", "C+D cpu%", "vllm cpu%", "run", "wait", "kv%"]
    print("  " + " ".join(f"{h:>9}" for h in hdr))
    for r in sorted(out_rows, key=lambda x: (int(x["W"] or 0), int(x["N"] or 0))):
        vals = [r["cell"], r["N"], r["W"], r["pool_P"], r["omp"], r["makespan_s"],
                r["throughput_per_min"], r["pass_"], r["fail"], r["timeout"],
                r["cd_cpu_mean"], r["vllm_cpu_mean"], r["vllm_run_peak"],
                r["vllm_wait_peak"], r["kv_peak_pct"]]
        print("  " + " ".join(f"{str(v):>9}" for v in vals))
    print(f"\n  grid CSV -> {grid_csv}")
    print("  (compute-bound: C+D cpu% high while vllm cpu% + run/wait queue stay low)")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--sweep":
        report_sweep(sys.argv[2])
        return
    if len(sys.argv) < 2:
        print("usage: bench_report.py RUN_ROOT", file=sys.stderr)
        print("       bench_report.py --sweep SWEEP_ROOT", file=sys.stderr)
        sys.exit(2)
    run_root = sys.argv[1]
    meta = {}
    mpath = os.path.join(run_root, "profile_metadata.json")
    if os.path.exists(mpath):
        with open(mpath) as _mf:
            meta = json.load(_mf)

    print("=" * 70)
    print(f"ChemGraph compute-bound benchmark report")
    print(f"  run_root   : {run_root}")
    print(f"  concurrency: {meta.get('concurrency', '?')}   n_queries: {meta.get('n_queries', '?')}")
    print(f"  vllm_node  : {meta.get('vllm_node', '?')}   compute: {meta.get('compute_nodes', '?')}")
    print("=" * 70)

    # ---- end-to-end ----
    rows = load_summary(run_root)
    n = len(rows)
    by_status = {}
    secs = []
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        try:
            secs.append(float(r["seconds"]))
        except (ValueError, KeyError):
            pass
    n_pass = by_status.get("PASS", 0)

    makespan = None
    try:
        makespan = float(meta["profile_end_epoch"]) - float(meta["profile_start_epoch"])
    except (KeyError, ValueError):
        # fall back to summary epochs
        try:
            starts = [float(r["start_epoch"]) for r in rows]
            ends = [float(r["end_epoch"]) for r in rows]
            makespan = max(ends) - min(starts)
        except (ValueError, KeyError):
            pass

    print("\n-- end-to-end --")
    print(f"  status: {by_status}")
    if makespan:
        print(f"  makespan        : {makespan:.1f} s")
        print(f"  throughput(PASS): {n_pass / makespan * 60:.2f} tasks/min  ({n_pass}/{n} PASS)")
    if secs:
        secs_sorted = sorted(secs)
        p50 = secs_sorted[len(secs_sorted) // 2]
        p99 = secs_sorted[min(len(secs_sorted) - 1, int(len(secs_sorted) * 0.99))]
        print(f"  per-task seconds: min={min(secs):.0f} p50={p50:.0f} p99={p99:.0f} max={max(secs):.0f}")

    # ---- per-query table ----
    print("\n-- per-query --")
    print(f"  {'id':<10} {'status':<8} {'sec':>5}  worker")
    for r in sorted(rows, key=lambda x: x["id"]):
        print(f"  {r['id']:<10} {r['status']:<8} {r['seconds']:>5}  {r.get('worker','')}")

    # ---- hardware: compute (C+D) vs vLLM (B) ----
    print("\n-- hardware utilization (the compute-bound claim) --")
    compute = meta.get("compute_nodes", [])
    print("  compute nodes (expect HIGH cpu):")
    for cn in compute:
        cpu = cpu_peak_mean(os.path.join(run_root, f"cpu_{cn}.csv"))
        gpu = gpu_peak_mean(os.path.join(run_root, f"gpu_{cn}.csv"))
        cpu_s = f"cpu peak={cpu[0]:.0f}% mean={cpu[1]:.0f}%" if cpu else "cpu n/a"
        gpu_s = f"gpu peak={gpu[0]:.0f}%" if gpu else "gpu n/a"
        print(f"    {cn}: {cpu_s}  {gpu_s}")
    cpu_v = cpu_peak_mean(os.path.join(run_root, "cpu_vllm.csv"))
    gpu_v = gpu_peak_mean(os.path.join(run_root, "gpu_vllm.csv"))
    print("  vLLM node (expect LOW cpu, gpu mostly idle for compute-bound):")
    print(f"    {meta.get('vllm_node','vllm')}: "
          f"{'cpu peak=%.0f%% mean=%.0f%%' % (cpu_v[0], cpu_v[1]) if cpu_v else 'cpu n/a'}  "
          f"{'gpu peak=%.0f%%' % gpu_v[0] if gpu_v else 'gpu n/a'}")

    # ---- vLLM serving metrics: should show the backend not saturated ----
    print("\n-- vLLM serving load (expect NOT saturated for compute-bound) --")
    names = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
    }
    before = parse_prom(os.path.join(run_root, "vllm_metrics_before.prom"), names)
    after = parse_prom(os.path.join(run_root, "vllm_metrics_after.prom"), names)
    peaks = peak_from_samples(os.path.join(run_root, "vllm_metrics_samples.prom"), names)
    for tok in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
        if tok in before or tok in after:
            d = after.get(tok, 0) - before.get(tok, 0)
            print(f"  {tok.split(':')[1]:<26} delta={d:,.0f}")
    for q in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if q in peaks:
            print(f"  {q.split(':')[1]:<26} peak={peaks[q]:.0f}")
    for kv in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        if kv in peaks:
            print(f"  {kv.split(':')[1]:<26} peak={peaks[kv]*100 if peaks[kv] <= 1 else peaks[kv]:.1f}%")

    # ---- Parsl distribution ----
    pc = os.path.join(run_root, "parsl_task_counts.txt")
    if os.path.exists(pc):
        print("\n-- Parsl per-node task counts (distributed execution proof) --")
        print("  " + open(pc).read().strip().replace("\n", "\n  "))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
