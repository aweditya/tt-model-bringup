#!/usr/bin/env python3
"""Print top device ops by time from a tracy capture's CSV outputs.

Reads either:
  - `cpp_device_perf_report.csv` (DEVICE KERNEL DURATION, post-process)
    or
  - `tracy_ops_data.csv`         (host-side per-op total_ns, available
                                   even when the device CSV is unnamed)

`tracy_top_ops.py <logs_dir> [--top N] [--source device|host]` —
`logs_dir` is the `.logs/` subdir under `<run>/reports/<...>/.logs/`
that tracy emits next to the perf report.

When tracy's post-processing crashes mid-pipeline (we hit
`AssertionError: Device data missing: Op ... not present in
cpp_device_perf_report.csv` on the Gemma 4 capture 2026-06-08), the
device CSV has empty `OP NAME` rows — fall back to `--source host`,
which parses op names out of `tracy_ops_data.csv`'s message strings.

Run on qb1/qb2 (the host that owns the capture):
  ssh qb1 'cd ~/tt-xla && \\
    .venv/bin/python experiments/utils/tracy_top_ops.py \\
      .cache/perf_logs/tracy_gemma4_v2_132855/.logs/'
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

DEVICE_CSV = "cpp_device_perf_report.csv"
HOST_CSV = "tracy_ops_data.csv"

# Tracy host messages look like:
#   `TT_DNN_DEVICE_OP: "TilizeWithValPaddingDeviceOperation", 102, 3, false, 1027 -> ...`
HOST_MSG_RE = re.compile(r'TT_DNN_DEVICE_OP:\s*"([^"]+)"')


def _aggregate_device(csv_path: Path):
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    n_blank = 0
    n_total = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row.get("OP NAME") or ""
            try:
                dur_ns = float(row.get("DEVICE KERNEL DURATION [ns]", 0) or 0)
            except ValueError:
                continue
            if dur_ns <= 0:
                continue
            if not op:
                n_blank += 1
                op = "(unnamed)"
            totals[op][0] += dur_ns
            totals[op][1] += 1
            n_total += 1
    return totals, n_total, n_blank


def _aggregate_host(csv_path: Path):
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0])
    n_total = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            msg = (row.get("MessageName") or "")
            try:
                dur_ns = float(row.get("total_ns", 0) or 0)
            except ValueError:
                continue
            if dur_ns <= 0:
                continue
            m = HOST_MSG_RE.search(msg)
            if m:
                op = m.group(1)
            else:
                # Bucket non-op tracing zones together as "other".
                op = "other"
            totals[op][0] += dur_ns
            totals[op][1] += 1
            n_total += 1
    return totals, n_total, 0


def _print_table(label: str, totals: dict[str, list], top_n: int,
                  n_total: int, n_blank: int) -> None:
    total = sum(v[0] for v in totals.values())
    print(f"{label}")
    print(f"  total time: {total / 1e6:.1f} ms across {n_total} op "
          f"invocations" +
          (f" ({n_blank} unnamed)" if n_blank else ""))
    print()
    print(f"{'OP':<46s} {'calls':>8s} {'total_ms':>11s} "
          f"{'pct':>7s} {'avg_us':>10s}")
    print("-" * 86)
    items = sorted(totals.items(), key=lambda x: -x[1][0])
    for op, (dur_ns, n) in items[:top_n]:
        op_short = op[:44]
        print(f"{op_short:<46s} {n:>8d} {dur_ns / 1e6:>11.1f} "
              f"{dur_ns / total * 100:>6.1f}% {dur_ns / n / 1000:>10.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_dir", type=Path,
                         help="path to the .logs/ subdir of a tracy capture")
    parser.add_argument("--top", type=int, default=20,
                         help="rows to print (default 20)")
    parser.add_argument(
        "--source", choices=("device", "host", "auto"), default="auto",
        help="device = cpp_device_perf_report.csv; host = tracy_ops_data.csv; "
             "auto picks device unless its OP NAMEs are all blank "
             "(tracy post-processing failure) → falls back to host"
    )
    args = parser.parse_args()

    if not args.logs_dir.is_dir():
        print(f"ERROR: not a directory: {args.logs_dir}", file=sys.stderr)
        return 2

    device_csv = args.logs_dir / DEVICE_CSV
    host_csv = args.logs_dir / HOST_CSV

    use = args.source
    if use == "auto":
        # Try device first; auto-fall back to host if names are blank.
        if device_csv.is_file():
            totals_d, n_d, n_blank = _aggregate_device(device_csv)
            named = sum(1 for k in totals_d if k != "(unnamed)")
            if named >= 5:
                _print_table(f"DEVICE KERNEL TIME ({device_csv.name})",
                              totals_d, args.top, n_d, n_blank)
                return 0
            print(f"NOTE: device CSV has {named} named ops (all/most "
                  f"unnamed); falling back to host CSV", file=sys.stderr)
        use = "host"

    if use == "device":
        if not device_csv.is_file():
            print(f"ERROR: missing {device_csv}", file=sys.stderr)
            return 2
        totals, n, n_blank = _aggregate_device(device_csv)
        _print_table(f"DEVICE KERNEL TIME ({device_csv.name})",
                      totals, args.top, n, n_blank)
        return 0

    # host
    if not host_csv.is_file():
        print(f"ERROR: missing {host_csv}", file=sys.stderr)
        return 2
    totals, n, _ = _aggregate_host(host_csv)
    _print_table(f"HOST WALL TIME ({host_csv.name})",
                  totals, args.top, n, 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
