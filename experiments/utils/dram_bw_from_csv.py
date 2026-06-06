#!/usr/bin/env python3
"""Aggregate DRAM-traffic / bandwidth columns from a tt-perf-report CSV.

Round 8 of the Gemma 4 perf chase needs explicit DRAM-traffic accounting:
Round 7 left the diagnosis "DRAM-bandwidth bound at B=1" but the analysis
was indirect (HiFi2 produced 0 gain). This helper makes the BW story
quantitative per op-class:

  - DRAM BW UTIL (%)            — per-row bandwidth utilisation when Tracy
                                  populates it (often empty; matmul + cache
                                  ops usually have it).
  - PM BANDWIDTH [ns]           — performance-model "bandwidth-bound time".
                                  When > PM COMPUTE [ns], the op is BW-bound.
  - PM COMPUTE [ns]             — performance-model "compute-bound time".
  - DEVICE KERNEL DURATION [ns] — actual wall-clock kernel time (sum).

Forks `count_ops_in_csv.py:14-44` (same signpost-region extraction); adds
DRAM/BW columns + a "BW-bound vs compute-bound" classifier per op.

Usage:
  .venv/bin/python experiments/utils/dram_bw_from_csv.py <csv>

Output columns:
  count               | rows in region for that OP CODE
  kernel_ms_sum       | sum of DEVICE KERNEL DURATION [ns] / 1e6
  kernel_ms_pct       | percent of total kernel time (across all ops)
  dram_bw_rows        | rows where DRAM BW UTIL (%) was populated
  dram_bw_avg         | mean DRAM BW UTIL across populated rows
  pm_bw_ms_sum        | sum of PM BANDWIDTH [ns] / 1e6
  pm_compute_ms_sum   | sum of PM COMPUTE [ns] / 1e6
  bound               | "BW" if pm_bw_sum > pm_compute_sum, else "COMP"
"""
import csv
import sys
from collections import Counter, defaultdict


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def main(path: str) -> int:
    with open(path) as f:
        rows = list(csv.DictReader(f))

    start_idx = end_idx = None
    for i, row in enumerate(rows):
        if row.get("OP TYPE", "").strip() == "signpost":
            c = row.get("OP CODE", "").strip().strip('"').lower()
            if "start" in c and start_idx is None:
                start_idx = i + 1
            elif "end" in c and end_idx is None:
                end_idx = i
    if start_idx is not None and end_idx is not None:
        region = rows[start_idx:end_idx]
        print(f"signposted region: rows [{start_idx}, {end_idx}) = {len(region)} ops")
    else:
        region = rows
        print(f"no signposts; using all {len(rows)} rows")

    op_count = Counter()
    op_kernel_ns = defaultdict(float)
    op_dram_rows = Counter()
    op_dram_bw_sum = defaultdict(float)
    op_pm_bw_ns = defaultdict(float)
    op_pm_compute_ns = defaultdict(float)

    for row in region:
        op = row.get("OP CODE", "").strip()
        if not op or row.get("OP TYPE", "").strip() == "signpost":
            continue
        op_count[op] += 1
        op_kernel_ns[op] += _to_float(row.get("DEVICE KERNEL DURATION [ns]", ""))
        dram_bw = (row.get("DRAM BW UTIL (%)", "") or "").strip()
        if dram_bw:
            op_dram_rows[op] += 1
            op_dram_bw_sum[op] += _to_float(dram_bw)
        op_pm_bw_ns[op] += _to_float(row.get("PM BANDWIDTH [ns]", ""))
        op_pm_compute_ns[op] += _to_float(row.get("PM COMPUTE [ns]", ""))

    total_kernel_ms = sum(op_kernel_ns.values()) / 1e6
    print(f"\nTotal kernel time (signposted, summed across rows): {total_kernel_ms:.3f} ms")
    print(
        f"\n{'OP CODE':<32} {'count':>6} {'kern_ms':>10} {'kern_%':>7} "
        f"{'drmRows':>8} {'drmBW%':>8} {'pmBW_ms':>10} {'pmCmp_ms':>10} {'bound':>6}"
    )
    print("-" * 110)
    for op, _ in op_count.most_common():
        n = op_count[op]
        ker_ms = op_kernel_ns[op] / 1e6
        ker_pct = (ker_ms / total_kernel_ms * 100.0) if total_kernel_ms > 0 else 0.0
        d_rows = op_dram_rows[op]
        d_avg = (op_dram_bw_sum[op] / d_rows) if d_rows > 0 else 0.0
        pm_bw_ms = op_pm_bw_ns[op] / 1e6
        pm_comp_ms = op_pm_compute_ns[op] / 1e6
        bound = "BW" if pm_bw_ms > pm_comp_ms else "COMP"
        if pm_bw_ms == 0 and pm_comp_ms == 0:
            bound = "?"
        print(
            f"{op[:32]:<32} {n:>6} {ker_ms:>10.3f} {ker_pct:>7.2f} "
            f"{d_rows:>8} {d_avg:>8.2f} {pm_bw_ms:>10.3f} {pm_comp_ms:>10.3f} {bound:>6}"
        )

    print()
    bw_total = sum(op_pm_bw_ns.values()) / 1e6
    comp_total = sum(op_pm_compute_ns.values()) / 1e6
    print(f"PM BANDWIDTH total (signposted): {bw_total:.3f} ms")
    print(f"PM COMPUTE   total (signposted): {comp_total:.3f} ms")
    print(f"PM ratio bw/comp: {(bw_total/comp_total) if comp_total>0 else float('inf'):.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) or 0)
