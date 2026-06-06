#!/usr/bin/env python3
"""Count ops by OP CODE in a tt-perf-report ops_perf_results CSV.

If signposts ('Performance pass start'/'end') are present, restricts to the
signposted region; otherwise counts the whole CSV.

Why this exists: when Tracy DRAM-buffer overflow zeroes per-op DEVICE KERNEL
DURATION, op COUNTS are still reliable and let us reason about hot dispatch
sources. Used for the round-4 lever pick (gemma4 perf chase on qb2).

Usage:
  .venv/bin/python experiments/utils/count_ops_in_csv.py <csv>
"""
import csv
import sys
from collections import Counter


def main(path: str) -> int:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    start_idx = end_idx = None
    for i, row in enumerate(rows):
        if row.get("OP TYPE", "").strip() == "signpost":
            code = row.get("OP CODE", "").strip().strip('"')
            if "start" in code.lower() and start_idx is None:
                start_idx = i + 1
            elif "end" in code.lower() and end_idx is None:
                end_idx = i
    if start_idx is not None and end_idx is not None:
        region = rows[start_idx:end_idx]
        print(f"signposted region: rows [{start_idx}, {end_idx}) = {len(region)} ops")
    else:
        region = rows
        print(f"no signposts; counting all {len(rows)} rows")
    ops = Counter()
    for row in region:
        ops[row.get("OP CODE", "").strip()] += 1
    print()
    print(f"{'count':>7}  OP CODE")
    print("-" * 70)
    for code, n in ops.most_common():
        print(f"{n:>7}  {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) or 0)
