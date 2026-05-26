#!/usr/bin/env python3
"""Delete inclusive line range from a file in place.

Usage: python experiments/utils/delete_line_range.py <path> <start> <end> [<start> <end> ...]
Ranges are 1-indexed and inclusive. Multiple ranges are applied in a single pass,
so all <start>/<end> indices refer to the ORIGINAL file. Ranges must not overlap.
"""
import sys


def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("usage: delete_line_range.py <path> <start> <end> [<start> <end> ...]")
        sys.exit(2)
    path = sys.argv[1]
    raw = list(zip(sys.argv[2::2], sys.argv[3::2]))
    ranges = sorted([(int(a), int(b)) for a, b in raw])
    for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
        if b1 >= a2:
            print(f"FAIL: overlapping ranges ({a1},{b1}) and ({a2},{b2})")
            sys.exit(1)
    with open(path) as f:
        lines = f.readlines()
    keep = []
    deleted = 0
    for i, line in enumerate(lines, start=1):
        if any(s <= i <= e for s, e in ranges):
            deleted += 1
        else:
            keep.append(line)
    with open(path, "w") as f:
        f.writelines(keep)
    print(f"OK: {path}  deleted {deleted} lines  ranges={ranges}")


if __name__ == "__main__":
    main()
