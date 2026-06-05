#!/usr/bin/env python3
"""Aggregate per-op-code host timing from a tt-perf-report tracy CSV.

When the Tracy DRAM marker buffer overflows (`Profiler DRAM buffers were full,
markers were dropped`), the `DEVICE KERNEL DURATION` column is zeroed for the
dropped ops but the `HOST START TS` / `HOST END TS` columns stay correct. This
lets us still aggregate per-op-code time even if kernel-marker timing is
broken.

Forks `experiments/utils/analyze_ops_perf_results.py:1-30` (same CSV-reader
pattern + per-op-code aggregator) but consumes host timing columns instead
of device-kernel columns and filters by signposts in the same OP TYPE field.

Usage:
    python experiments/utils/host_time_tracy_csv.py <ops_perf_results.csv> \\
        [--start-signpost "Performance pass start"] \\
        [--end-signpost   "Performance pass end"]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="ops_perf_results CSV from tracy")
    ap.add_argument("--start-signpost", default="Performance pass start")
    ap.add_argument("--end-signpost", default="Performance pass end")
    ap.add_argument("--device-id", default=None, help="Filter to one device id (else all)")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Find signpost boundaries by global call count (signposts have OP TYPE 'signpost').
    start_idx = end_idx = None
    for i, row in enumerate(rows):
        if row.get("OP TYPE", "").strip() == "signpost":
            code = row.get("OP CODE", "").strip().strip('"')
            if args.start_signpost.lower() in code.lower() and start_idx is None:
                start_idx = i + 1
            elif args.end_signpost.lower() in code.lower() and end_idx is None:
                end_idx = i
    if start_idx is None or end_idx is None:
        print(f"signposts not found; using all rows. start_idx={start_idx} end_idx={end_idx}")
        start_idx = 0
        end_idx = len(rows)
    print(f"signposted region: rows [{start_idx}, {end_idx}) = {end_idx - start_idx} ops")

    region = rows[start_idx:end_idx]
    if args.device_id is not None:
        region = [r for r in region if r.get("DEVICE ID", "").strip() == args.device_id]
        print(f"  filtered to DEVICE ID={args.device_id} → {len(region)} rows")

    per_op = defaultdict(lambda: {"n": 0, "host_ns": 0, "ker_ns": 0, "op2op_ns": 0})
    total_host_ns = 0
    total_ker_ns = 0
    total_op2op_ns = 0
    for row in region:
        opc = row.get("OP CODE", "").strip()
        if not opc or opc.startswith("Performance pass"):
            continue
        try:
            host_dur = int(row.get("HOST DURATION [ns]", "0") or 0)
        except ValueError:
            host_dur = 0
        try:
            ker_dur = int(row.get("DEVICE KERNEL DURATION [ns]", "0") or 0)
        except ValueError:
            ker_dur = 0
        try:
            op2op = int(row.get("OP TO OP LATENCY [ns]", "0") or 0)
        except ValueError:
            op2op = 0
        bucket = per_op[opc]
        bucket["n"] += 1
        bucket["host_ns"] += host_dur
        bucket["ker_ns"] += ker_dur
        bucket["op2op_ns"] += op2op
        total_host_ns += host_dur
        total_ker_ns += ker_dur
        total_op2op_ns += op2op

    # Sort by host time (descending).
    rows_out = sorted(per_op.items(), key=lambda kv: -kv[1]["host_ns"])
    print()
    print(f"{'OP CODE':45} {'count':>6}  {'tot_host_ms':>12}  {'tot_ker_ms':>11}  "
          f"{'avg_host_us':>12}  {'avg_ker_us':>11}")
    print("-" * 110)
    for opc, b in rows_out:
        n = b["n"]
        avg_host = b["host_ns"] / max(n, 1) / 1e3
        avg_ker = b["ker_ns"] / max(n, 1) / 1e3
        print(f"{opc[:45]:45} {n:>6}  {b['host_ns']/1e6:>12.2f}  {b['ker_ns']/1e6:>11.2f}  "
              f"{avg_host:>12.1f}  {avg_ker:>11.1f}")
    print("-" * 110)
    print(f"  TOTAL host: {total_host_ns/1e6:.2f} ms")
    print(f"  TOTAL ker:  {total_ker_ns/1e6:.2f} ms")
    print(f"  TOTAL op2op:{total_op2op_ns/1e6:.2f} ms")


if __name__ == "__main__":
    main()
