#!/usr/bin/env python3
"""Analyze tt-perf-report's ops_perf_results CSV between signposts.

Filters to rows after 'Performance pass start' and before 'Performance pass
end' (rows with OP TYPE == 'signpost' mark the boundaries). Reports:
  - per OP CODE: count, total device kernel time, total op2op gap, average
    per-op kernel / gap (the dispatch indicator)
  - aggregate kernel vs gap to compute dispatch fraction
"""
import csv
import sys
import statistics
from collections import defaultdict


def fmt_ms(ns):
    return f"{ns / 1e6:8.2f}"


def fmt_us(ns):
    return f"{ns / 1e3:7.1f}"


def main():
    path = sys.argv[1]
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Find signpost boundaries
    start_idx = end_idx = None
    for i, row in enumerate(rows):
        if row.get("OP TYPE", "").strip() == "signpost":
            code = row.get("OP CODE", "").strip().strip('"')
            if "start" in code.lower() and start_idx is None:
                start_idx = i + 1
            elif "end" in code.lower() and end_idx is None:
                end_idx = i
    if start_idx is None or end_idx is None:
        print(f"signposts not found. start_idx={start_idx} end_idx={end_idx}")
        print("Falling back to last 1000 rows.")
        start_idx = max(0, len(rows) - 1000)
        end_idx = len(rows)
    print(f"signposted region: rows [{start_idx}, {end_idx}) = {end_idx - start_idx} ops")

    region = rows[start_idx:end_idx]
    per_op = defaultdict(lambda: {"n": 0, "ker_ns": 0, "op2op_ns": 0, "host_ns": 0, "ker_per_op": [], "op2op_per_op": []})
    for row in region:
        code = row.get("OP CODE", "").strip()
        try:
            ker = int(row.get("DEVICE KERNEL DURATION [ns]") or 0)
            op2op = int(row.get("OP TO OP LATENCY [ns]") or 0)
            host = int(row.get("HOST DURATION [ns]") or 0)
        except Exception:
            continue
        # Skip rows with bogus (negative or > 1 second per op) data.
        if ker < 0 or ker > 1_000_000_000:
            continue
        s = per_op[code]
        s["n"] += 1
        s["ker_ns"] += ker
        s["op2op_ns"] += max(op2op, 0)
        s["host_ns"] += host
        s["ker_per_op"].append(ker)
        s["op2op_per_op"].append(max(op2op, 0))

    print()
    print(f"{'OP CODE':50s}  {'count':>6}  {'tot_ker[ms]':>12}  {'tot_op2op[ms]':>14}  {'med_ker[us]':>12}  {'med_op2op[us]':>14}")
    for code, s in sorted(per_op.items(), key=lambda x: -x[1]["ker_ns"]):
        n = s["n"]
        med_ker = statistics.median(s["ker_per_op"]) if s["ker_per_op"] else 0
        med_op2op = statistics.median(s["op2op_per_op"]) if s["op2op_per_op"] else 0
        print(f"{code:50s}  {n:>6}  {fmt_ms(s['ker_ns']):>12}  {fmt_ms(s['op2op_ns']):>14}  {fmt_us(med_ker):>12}  {fmt_us(med_op2op):>14}")

    tot_ker = sum(s["ker_ns"] for s in per_op.values())
    tot_op2op = sum(s["op2op_ns"] for s in per_op.values())
    print()
    print(f"Region totals: kernel={tot_ker/1e6:.2f} ms, op2op={tot_op2op/1e6:.2f} ms")
    if tot_ker + tot_op2op > 0:
        print(f"Dispatch fraction (op2op / (kernel + op2op)): {tot_op2op / (tot_ker + tot_op2op):.3f}")

    # Also show stats over ALL rows (not just signposted) for matmuls that
    # have device data — this is our real signal even if signpost filter
    # dropped them.
    print()
    print("=== All matmul rows in entire CSV (regardless of signposts) ===")
    all_matmul_ker = []
    all_matmul_op2op = []
    shape_stats = defaultdict(lambda: {"n": 0, "ker_ns": 0, "op2op_ns": 0})
    for row in rows:
        code = row.get("OP CODE", "").strip()
        if "Matmul" not in code:
            continue
        try:
            ker = int(row.get("DEVICE KERNEL DURATION [ns]") or 0)
            op2op = int(row.get("OP TO OP LATENCY [ns]") or 0)
        except Exception:
            continue
        if ker <= 0 or ker > 100_000_000:  # filter sentinel / aggregated zones
            continue
        all_matmul_ker.append(ker)
        all_matmul_op2op.append(max(op2op, 0))
        s = shape_stats[code]
        s["n"] += 1
        s["ker_ns"] += ker
        s["op2op_ns"] += max(op2op, 0)
    if all_matmul_ker:
        print(f"  n={len(all_matmul_ker)}")
        print(f"  kernel us — min/p25/med/p75/p95/max: "
              f"{min(all_matmul_ker)/1e3:.1f}/{sorted(all_matmul_ker)[len(all_matmul_ker)//4]/1e3:.1f}/"
              f"{statistics.median(all_matmul_ker)/1e3:.1f}/{sorted(all_matmul_ker)[3*len(all_matmul_ker)//4]/1e3:.1f}/"
              f"{sorted(all_matmul_ker)[int(0.95*len(all_matmul_ker))]/1e3:.1f}/{max(all_matmul_ker)/1e3:.1f}")
        print(f"  op2op us — min/p25/med/p75/p95/max: "
              f"{min(all_matmul_op2op)/1e3:.1f}/{sorted(all_matmul_op2op)[len(all_matmul_op2op)//4]/1e3:.1f}/"
              f"{statistics.median(all_matmul_op2op)/1e3:.1f}/{sorted(all_matmul_op2op)[3*len(all_matmul_op2op)//4]/1e3:.1f}/"
              f"{sorted(all_matmul_op2op)[int(0.95*len(all_matmul_op2op))]/1e3:.1f}/{max(all_matmul_op2op)/1e3:.1f}")
        tot_ker_mm = sum(all_matmul_ker)
        tot_op2op_mm = sum(all_matmul_op2op)
        print(f"  total: kernel={tot_ker_mm/1e6:.1f} ms, op2op={tot_op2op_mm/1e6:.1f} ms, "
              f"dispatch_frac={tot_op2op_mm/(tot_ker_mm+tot_op2op_mm):.3f}")
    print()
    print("Per matmul shape (filtered, valid rows only):")
    print(f"  {'shape':50s}  {'count':>6}  {'tot_ker[ms]':>12}  {'tot_op2op[ms]':>14}  {'avg_ker[us]':>12}  {'avg_op2op[us]':>14}")
    for code, s in sorted(shape_stats.items(), key=lambda x: -x[1]["ker_ns"])[:15]:
        if s["n"] == 0:
            continue
        print(f"  {code:50s}  {s['n']:>6}  {fmt_ms(s['ker_ns']):>12}  {fmt_ms(s['op2op_ns']):>14}  "
              f"{fmt_us(s['ker_ns']/s['n']):>12}  {fmt_us(s['op2op_ns']/s['n']):>14}")


if __name__ == "__main__":
    main()
