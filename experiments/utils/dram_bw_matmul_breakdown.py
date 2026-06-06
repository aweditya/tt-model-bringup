#!/usr/bin/env python3
"""Per-shape DRAM-bandwidth breakdown of Matmul ops in a tt-perf-report CSV.

Round 8 needs to know WHICH matmul shapes dominate DRAM bandwidth.
`dram_bw_from_csv.py` shows Matmul = ~99% of PM-bandwidth budget; this
helper groups Matmul rows by (in0_shape, in1_shape) and sums kernel time
and PM bandwidth per shape, so we can spot the lm_head, down_proj, and
sliding/global Q/K/V/O outliers.

Usage:
  .venv/bin/python experiments/utils/dram_bw_matmul_breakdown.py <csv>
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


def _shape(row, idx):
    keys = [
        f"INPUT_{idx}_W_PAD[LOGICAL]",
        f"INPUT_{idx}_Z_PAD[LOGICAL]",
        f"INPUT_{idx}_Y_PAD[LOGICAL]",
        f"INPUT_{idx}_X_PAD[LOGICAL]",
    ]
    vals = [row.get(k, "").strip() for k in keys]
    return "x".join(v if v else "?" for v in vals)


def _shape_out(row):
    keys = [
        "OUTPUT_0_W_PAD[LOGICAL]",
        "OUTPUT_0_Z_PAD[LOGICAL]",
        "OUTPUT_0_Y_PAD[LOGICAL]",
        "OUTPUT_0_X_PAD[LOGICAL]",
    ]
    vals = [row.get(k, "").strip() for k in keys]
    return "x".join(v if v else "?" for v in vals)


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
    region = rows[start_idx:end_idx] if start_idx is not None and end_idx is not None else rows
    print(f"signposted region: {len(region)} rows")

    shape_count = Counter()
    shape_kernel_ns = defaultdict(float)
    shape_pm_bw_ns = defaultdict(float)
    shape_pm_comp_ns = defaultdict(float)

    for row in region:
        if row.get("OP CODE", "").strip() != "MatmulDeviceOperation":
            continue
        sh = (_shape(row, 0), _shape(row, 1), _shape_out(row))
        shape_count[sh] += 1
        shape_kernel_ns[sh] += _to_float(row.get("DEVICE KERNEL DURATION [ns]", ""))
        shape_pm_bw_ns[sh] += _to_float(row.get("PM BANDWIDTH [ns]", ""))
        shape_pm_comp_ns[sh] += _to_float(row.get("PM COMPUTE [ns]", ""))

    total_kernel_ms = sum(shape_kernel_ns.values()) / 1e6
    total_pm_bw_ms = sum(shape_pm_bw_ns.values()) / 1e6
    print(f"\nMatmul total: kernel {total_kernel_ms:.2f} ms, PM-BW {total_pm_bw_ms:.3f} ms")
    print(
        f"\n{'in0':<26} {'in1':<26} {'out':<26} {'cnt':>5} {'kern_ms':>9} "
        f"{'pmBW_ms':>9} {'pmBW_%':>7} {'bnd':>4}"
    )
    print("-" * 125)
    by_pmbw = sorted(shape_count.keys(), key=lambda s: -shape_pm_bw_ns[s])
    for sh in by_pmbw:
        n = shape_count[sh]
        ker_ms = shape_kernel_ns[sh] / 1e6
        bw_ms = shape_pm_bw_ns[sh] / 1e6
        bw_pct = (bw_ms / total_pm_bw_ms * 100.0) if total_pm_bw_ms > 0 else 0.0
        cp_ms = shape_pm_comp_ns[sh] / 1e6
        bnd = "BW" if bw_ms > cp_ms else "COMP"
        a, b, o = sh
        print(
            f"{a[:26]:<26} {b[:26]:<26} {o[:26]:<26} {n:>5} {ker_ms:>9.2f} "
            f"{bw_ms:>9.3f} {bw_pct:>7.2f} {bnd:>4}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) or 0)
