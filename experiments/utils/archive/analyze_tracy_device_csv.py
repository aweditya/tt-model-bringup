#!/usr/bin/env python3
"""Pragmatic analyzer for Tracy's cpp_device_perf_report.csv when the official
post-processing fails. We hit the "device data missing" assertion on the 35B
traced decode, but the raw CSV is intact and rich enough to answer the BIG
question: dispatch-bound or compute-bound?

Each row in cpp_device_perf_report.csv = ONE on-device op-call instance.
Key columns:
  DEVICE FW DURATION [ns]      — firmware setup + kernel time (the "op cost")
  DEVICE KERNEL DURATION [ns]  — pure kernel compute time
  OP TO OP LATENCY [ns]        — gap between this op's start and the previous op's start
                                  on the same device (= dispatch/sync overhead)
  CORE COUNT                   — how many Tensix cores the op used
  DEVICE BRISC/NCRISC/TRISC*   — per-RISCV time inside the kernel

We compute, per device and overall:
  - kernel time distribution (min/median/p95/max us)
  - op-to-op gap distribution
  - the dispatch fraction = gap / (gap + kernel)

If dispatch_fraction > 0.5 → ops are dispatch-bound → batching helps
If dispatch_fraction < 0.2 → ops are compute/BW-bound → batching won't help much

Usage:
  python3 analyze_tracy_device_csv.py <cpp_device_perf_report.csv> [--top-n 10]
"""
import csv
import sys
from collections import defaultdict
import statistics


def fmt_us(ns):
    return f"{ns / 1e3:7.1f}"


def main():
    path = sys.argv[1]
    rows_per_device = defaultdict(list)
    cols_seen = set()

    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            did = row.get("DEVICE ID", "")
            try:
                fw = int(row["DEVICE FW DURATION [ns]"] or 0)
                ker = int(row["DEVICE KERNEL DURATION [ns]"] or 0)
                op2op = int(row["OP TO OP LATENCY [ns]"] or 0)
                cores = int(row.get("CORE COUNT") or 0)
            except (ValueError, KeyError):
                continue
            rows_per_device[did].append({
                "fw": fw, "ker": ker, "op2op": op2op, "cores": cores,
                "gcc": row.get("GLOBAL CALL COUNT", ""),
            })

    # Per-device distribution
    print(f"{'device':>6}  {'#ops':>6}  {'kernel_us min/med/p95/max':>32}  {'op2op_us min/med/p95/max':>32}  {'sum_ker[ms]':>11}  {'sum_op2op[ms]':>13}  {'dispatch_frac':>14}")
    overall_ker = []
    overall_op2op = []
    overall_cores = []
    for did, rows in sorted(rows_per_device.items()):
        kers = sorted([r["ker"] for r in rows])
        gaps = sorted([r["op2op"] for r in rows])
        cores_seen = [r["cores"] for r in rows if r["cores"]]
        n = len(kers)
        if n == 0:
            continue
        ker_min = kers[0]; ker_med = kers[n // 2]; ker_p95 = kers[int(n * 0.95)]; ker_max = kers[-1]
        gap_min = gaps[0]; gap_med = gaps[n // 2]; gap_p95 = gaps[int(n * 0.95)]; gap_max = gaps[-1]
        sum_ker = sum(kers); sum_gap = sum(gaps)
        frac = sum_gap / max(1, sum_ker + sum_gap)
        ker_str = f"{fmt_us(ker_min)}/{fmt_us(ker_med)}/{fmt_us(ker_p95)}/{fmt_us(ker_max)}"
        gap_str = f"{fmt_us(gap_min)}/{fmt_us(gap_med)}/{fmt_us(gap_p95)}/{fmt_us(gap_max)}"
        print(f"{did:>6}  {n:>6}  {ker_str:>32}  {gap_str:>32}  {sum_ker/1e6:>11.1f}  {sum_gap/1e6:>13.1f}  {frac:>14.3f}")
        overall_ker.extend(kers)
        overall_op2op.extend(gaps)
        overall_cores.extend(cores_seen)

    print()
    if overall_cores:
        cnts = {}
        for c in overall_cores:
            cnts[c] = cnts.get(c, 0) + 1
        print("Core-count distribution (Blackhole has 110 worker cores):")
        for c, n in sorted(cnts.items(), key=lambda x: -x[1])[:8]:
            pct = n / len(overall_cores) * 100
            print(f"  cores={c:3d}  ops={n:6d}  {pct:5.1f}%")

    # Long-tail check: are a few super-long ops dominating?
    print()
    print(f"Top 8 longest single-op kernel durations (across all devices):")
    sorted_ker = sorted(overall_ker, reverse=True)[:8]
    for v in sorted_ker:
        print(f"  {fmt_us(v):>10} us")
    print()
    print(f"Bottom 8 shortest non-zero single-op kernel durations:")
    sorted_low = sorted([k for k in overall_ker if k > 0])[:8]
    for v in sorted_low:
        print(f"  {fmt_us(v):>10} us")


if __name__ == "__main__":
    main()
