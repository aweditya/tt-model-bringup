#!/usr/bin/env python3
"""Top-N per-op-NAME breakdown from Tracy CSVs.

Complements tracy_analyze_ops.py (which buckets by category). This script
shows the raw TT_DNN_DEVICE_OP op names sorted by total host-dispatch time
per step, so we can identify *specific* hot ops, not just categories.

Usage:
    python3 experiments/utils/tracy_top_ops_breakdown.py \
        --log-dir research/probe_logs/tracy_qb1_traced/.logs \
        --eager-forwards 3 \
        --top 30
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict


RX_OP = re.compile(
    r'^`TT_DNN_DEVICE_OP:\s+"([^"]+)",\s+\d+,\s+\d+,\s+(?:true|false),\s+(\d+)'
)


def parse_ops_data(path):
    gcc_to_op = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RX_OP.match(line)
            if m:
                gcc_to_op[int(m.group(2))] = m.group(1)
    return gcc_to_op


def parse_ops_times(path, gcc_to_op):
    op_total = defaultdict(int)
    op_count = defaultdict(int)
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        h = next(r)
        ci_name = h.index("name")
        ci_zt = h.index("zone_text")
        ci_ns = h.index("exec_time_ns")
        for row in r:
            if len(row) <= ci_ns:
                continue
            if row[ci_name] != "TT_DNN_DEVICE_OP":
                continue
            zt = row[ci_zt]
            if not zt.startswith("id:"):
                continue
            try:
                gcc = int(zt[3:])
                ns = int(row[ci_ns])
            except ValueError:
                continue
            op = gcc_to_op.get(gcc)
            if op is not None:
                op_total[op] += ns
                op_count[op] += 1
    return op_total, op_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--eager-forwards", type=int, default=3)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    p_data = os.path.join(args.log_dir, "tracy_ops_data.csv")
    p_times = os.path.join(args.log_dir, "tracy_ops_times.csv")
    if not os.path.exists(p_data) or not os.path.exists(p_times):
        print(f"ERROR: missing CSV(s) in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    gcc_to_op = parse_ops_data(p_data)
    op_total, op_count = parse_ops_times(p_times, gcc_to_op)
    fwd = max(1, args.eager_forwards)

    rows = sorted(op_total.items(), key=lambda x: -x[1])
    print(f"{'OP NAME':<40} {'calls/step':>10} {'ms/step':>10} {'us/call':>10}")
    print("-" * 72)
    total_ms = 0.0
    for op, ns in rows[:args.top]:
        ms = ns / fwd / 1e6
        us = ns / op_count[op] / 1e3
        total_ms += ms
        print(f"{op:<40} {op_count[op]/fwd:>10.1f} {ms:>10.3f} {us:>10.2f}")
    print("-" * 72)
    grand_ms = sum(ns for _, ns in rows) / fwd / 1e6
    print(f"top-{args.top} sum: {total_ms:.2f} ms/step  ({total_ms/grand_ms*100:.1f}% of total)")
    print(f"grand total : {grand_ms:.2f} ms/step  (all {len(rows)} unique op names)")


if __name__ == "__main__":
    main()
