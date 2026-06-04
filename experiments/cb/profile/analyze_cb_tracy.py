#!/usr/bin/env python3
"""Analyse a Tracy profile of the CB step (run by tracy_cb_step.py).

Bypasses tt-perf-report's broken post-processing path (which asserts on
op consistency across devices and crashes mid-CB-engine). Joins the raw
CSVs ourselves to answer two questions:

  1. What's the per-step wall time inside the signposted region?
  2. Which device ops dominate it (% of total device FW time)?

Parsing:
  - `tracy_ops_data.csv` is line-prefix-tagged messages (`MessageName;total_ns`).
    Op records:       ``TT_DNN_DEVICE_OP: "<op>", <hash>, <dev>, <bool>, <call>``
    Signpost records: ``TT_SIGNPOST: <name>``
    Both carry their `ns_since_start` in the `total_ns` column, in
    chronological order — so the signposts naturally bracket the op records
    that fall inside the window.
  - `cpp_device_perf_report.csv` has per-op `DEVICE FW DURATION [ns]` keyed by
    GLOBAL CALL COUNT. Join on that to get the op-name attribution for each
    device-side measurement.

Usage:
  python3 experiments/cb/profile/analyze_cb_tracy.py \\
      --logs .cache/perf_logs/tracy_cb_b4/.logs --top 20
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

OP_RE = re.compile(
    r'^`TT_DNN_DEVICE_OP:\s*"([^"]+)"\s*,\s*[^,]+,\s*(\d+),\s*\w+,\s*(\d+)\s*->'
)
SIG_RE = re.compile(r'^`TT_SIGNPOST:\s*(\S+?)`')


def _parse_ops_data(data_csv: Path, start_signpost: str, end_signpost: str):
    """Returns (call_count→op_name dict for region, region_start_ns,
    region_end_ns, all_call_counts_in_region_set)."""
    in_region = False
    region_start_ns = None
    region_end_ns = None
    call_to_op: dict[int, str] = {}
    region_calls: set[int] = set()
    with open(data_csv, encoding="utf-8", errors="replace") as f:
        for line in f:
            # `tracy_ops_data.csv` is mostly multi-line JSON continuations; we
            # only care about the lines that START with a backtick (the
            # MessageName lines).
            if not line.startswith("`"):
                continue
            # split off the `total_ns` (after the trailing backtick + `;`).
            try:
                msg, ns_str = line.rstrip("\n").rsplit(";", 1)
                ns = int(ns_str)
            except ValueError:
                continue
            sm = SIG_RE.match(msg)
            if sm is not None:
                name = sm.group(1).rstrip("`")
                if name == start_signpost:
                    in_region = True
                    region_start_ns = ns
                elif name == end_signpost:
                    region_end_ns = ns
                    in_region = False
                continue
            if not in_region:
                continue
            om = OP_RE.match(msg)
            if om is not None:
                op_name = om.group(1)
                call = int(om.group(3))
                call_to_op[call] = op_name
                region_calls.add(call)
    return call_to_op, region_start_ns, region_end_ns, region_calls


def _per_op_breakdown(device_csv: Path, call_to_op: dict[int, str],
                      region_calls: set[int]):
    per_op_ns = defaultdict(int)
    per_op_calls = defaultdict(int)
    per_device_ns = defaultdict(int)
    o2o_ns_total = 0
    op_rows = 0
    with open(device_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                call = int(row["GLOBAL CALL COUNT"])
            except (KeyError, ValueError):
                continue
            if region_calls and call not in region_calls:
                continue
            op = call_to_op.get(call, "")
            if not op:
                continue
            try:
                fw_ns = int(float(row["DEVICE FW DURATION [ns]"] or 0))
                o2o_ns = int(float(row.get("OP TO OP LATENCY [ns]") or 0))
                dev = int(row["DEVICE ID"])
            except (KeyError, ValueError):
                continue
            per_op_ns[op] += fw_ns
            per_op_calls[op] += 1
            per_device_ns[dev] += fw_ns
            o2o_ns_total += o2o_ns
            op_rows += 1
    return per_op_ns, per_op_calls, per_device_ns, op_rows, o2o_ns_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=".cache/perf_logs/tracy_cb_b4/.logs")
    ap.add_argument("--start-signpost", default="cb_perf_start")
    ap.add_argument("--end-signpost", default="cb_perf_end")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    logs = Path(args.logs)
    data_csv = logs / "tracy_ops_data.csv"
    device_csv = logs / "cpp_device_perf_report.csv"
    for p in (data_csv, device_csv):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    call_to_op, t0_ns, t1_ns, region_calls = _parse_ops_data(
        data_csv, args.start_signpost, args.end_signpost)
    if t0_ns is None or t1_ns is None:
        raise SystemExit(f"signposts not found: start={args.start_signpost!r} end={args.end_signpost!r}")

    region_wall_s = (t1_ns - t0_ns) / 1e9
    print(f"=== Tracy CB step analysis ({logs}) ===")
    print(f"  signpost window: {args.start_signpost!r} → {args.end_signpost!r}")
    print(f"  region wall:     {region_wall_s:.3f} s")
    print(f"  region op records: {len(region_calls)} unique global-call-count "
          f"values in the host trace inside the window")

    per_op_ns, per_op_calls, per_device_ns, n_dev_rows, o2o_ns = _per_op_breakdown(
        device_csv, call_to_op, region_calls)
    if not n_dev_rows:
        raise SystemExit("no joined device rows in region")

    total_ns = sum(per_op_ns.values())
    n_devices = len(per_device_ns)
    capacity_s = region_wall_s * n_devices
    util_pct = (total_ns / 1e9) / capacity_s * 100 if capacity_s else 0.0

    print(f"\n  joined device rows in region: {n_dev_rows}")
    print(f"  sum of device FW time:        {total_ns / 1e9:.3f} s  "
          f"(across {n_devices} chips)")
    print(f"  total op-to-op gap:           {o2o_ns / 1e9:.3f} s  "
          f"({o2o_ns / max(total_ns, 1) * 100:.1f}% of device-busy)")
    print(f"  device utilization estimate:  {util_pct:.1f}% of "
          f"({region_wall_s:.2f} s × {n_devices} chips = {capacity_s:.2f} chip-s)")
    print("  per-device FW totals:")
    for dev in sorted(per_device_ns):
        print(f"    device {dev}: {per_device_ns[dev] / 1e9:.3f} s  "
              f"({per_device_ns[dev] / 1e9 / region_wall_s * 100:.1f}% busy)")

    rows = sorted(per_op_ns.items(), key=lambda kv: kv[1], reverse=True)[:args.top]
    print(f"\n  top {len(rows)} device ops by total FW time:")
    print(f"  {'op':<55} {'calls':>8} {'total ms':>12} {'mean us':>10} {'%':>6}")
    for op, ns in rows:
        calls = per_op_calls[op]
        print(f"  {op:<55} {calls:>8} {ns/1e6:>12.2f} {(ns/calls)/1e3:>10.1f} "
              f"{ns/total_ns*100:>5.1f}%")


if __name__ == "__main__":
    main()
