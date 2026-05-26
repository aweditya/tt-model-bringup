#!/usr/bin/env python3
"""Layer-vs-position grid view of an existing cosine_ladder_35b run.

Helps answer: at WHICH LAYER does drift first appear, and at WHICH POSITION?

Reads the per-position cos_per_layer arrays already saved by
cosine_ladder_35b.py. Tabulates a (layer × position) grid at selected
positions. Highlights the first cell below the threshold per layer.

Usage:
  python3 experiments/utils/cosine_ladder_35b_drift_onset.py \\
      .cache/sanity_2026_05_22/cosine_ladder_35b_full85.json \\
      [--threshold 0.99] \\
      [--positions 0,1,2,3,5,10,20,40,84]
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--positions", default="0,1,2,3,5,10,20,40,80,84",
                    help="comma-separated positions to inspect")
    args = ap.parse_args()

    d = json.loads(Path(args.json_path).read_text())
    per_pos = d["per_pos"]
    n_layers = d["n_layers"]
    n_pos = d["n_positions_tested"]

    positions = [int(x) for x in args.positions.split(",") if int(x) < n_pos]
    print(f"file: {args.json_path}")
    print(f"prompt: {d['prompt'][:80]!r}{'...' if len(d['prompt']) > 80 else ''}")
    print(f"layers: {n_layers}, positions: {n_pos}, threshold: {args.threshold}")
    print()

    # Build lookup: per_pos by position
    pp_by_pos = {p["pos"]: p for p in per_pos}

    # Header
    header = f"{'layer':>7}  " + "  ".join(f"pos{p:>3}" for p in positions)
    print(header)
    print("-" * len(header))

    # n_layers + 1 entries in cos_per_layer (embed at 0, decoder L0..L39 at 1..40)
    for li in range(n_layers + 1):
        name = "embed" if li == 0 else f"L{li-1:02d}"
        row = f"{name:>7}  "
        for p in positions:
            v = pp_by_pos[p]["cos_per_layer"][li]
            marker = "*" if v < args.threshold else " "
            row += f"{v:>6.4f}{marker} "
        print(row)
    print()

    # First-divergence (cos < threshold) for each LAYER as we scan positions
    print(f"For each layer, FIRST POSITION where cos < {args.threshold}:")
    print(f"{'layer':>7}  {'first_pos':>9}  {'cos_at_first':>12}")
    print("-" * 35)
    for li in range(n_layers + 1):
        name = "embed" if li == 0 else f"L{li-1:02d}"
        first_pos = None
        first_cos = None
        for p in per_pos:
            v = p["cos_per_layer"][li]
            if v < args.threshold:
                first_pos = p["pos"]
                first_cos = v
                break
        if first_pos is None:
            print(f"{name:>7}  {'never':>9}  {'-':>12}")
        else:
            print(f"{name:>7}  {first_pos:>9}  {first_cos:>12.4f}")
    print()

    # For each POSITION, the SHALLOWEST layer that first goes below threshold
    print(f"For each position, SHALLOWEST LAYER where cos < {args.threshold}:")
    print(f"{'position':>9}  {'first_layer':>11}  {'cos':>8}")
    print("-" * 32)
    for p in per_pos:
        first_li = None
        first_v = None
        for li, v in enumerate(p["cos_per_layer"]):
            if v < args.threshold:
                first_li = li
                first_v = v
                break
        if first_li is None:
            continue  # this position is clean through all layers
        name = "embed" if first_li == 0 else f"L{first_li-1:02d}"
        if p["pos"] in positions or p["pos"] < 5 or p["pos"] == n_pos - 1:
            print(f"{p['pos']:>9}  {name:>11}  {first_v:>8.4f}")


if __name__ == "__main__":
    main()
