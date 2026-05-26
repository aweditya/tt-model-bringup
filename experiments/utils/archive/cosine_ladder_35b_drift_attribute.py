#!/usr/bin/env python3
"""Phase 1 root-cause analyzer for the 35B-A3B cosine ladder.

Given one or more cosine_ladder_35b JSON outputs, compute the per-AT-layer
contribution to drift at each position. This distinguishes:
  - H2 (per-step noise, e.g. RoPE-broadcast 4x bf16 ops): per-AT-layer drop
    is roughly CONSTANT across positions.
  - H5/H6 (accumulating noise from bf16 KV cache / softmax precision over
    growing context): per-AT-layer drop INCREASES with position.

No device time. Pure JSON analysis.

Usage:
  python3 experiments/utils/cosine_ladder_35b_drift_attribute.py \\
    .cache/sanity_2026_05_22/cosine_ladder_35b_full85.json \\
    [.cache/sanity_2026_05_22/cosine_ladder_35b_sdpa_full85.json]

Output:
  - cos_final vs position curve (one row per input JSON for comparison)
  - per-AT-layer Δcos contribution table: rows = AT layers, cols = position buckets
  - linear-fit slope of "AT-layer drop vs position" — flat (~0) = H2; positive = H5/H6
"""
import json
import math
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def at_layer_indices(layer_types):
    # 0-indexed decoder layer numbers where type == "full_attention".
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def cos_to_theta_deg(c):
    # Convert cosine to angle in degrees so deltas are interpretable + additive.
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def buckets(seq_len, n=5):
    # Position buckets for the column headers
    step = max(1, seq_len // n)
    return [(i*step, min((i+1)*step, seq_len)) for i in range(n)]


def at_layer_contribution(per_pos, at_idxs):
    """For each AT layer, compute the cosine DROP it contributes at each position
    relative to the immediately preceding decoder layer.

    Returns: dict[at_layer_idx] -> list of drops per position
    """
    out = {a: [] for a in at_idxs}
    for p in per_pos:
        cpl = p["cos_per_layer"]  # 41 entries: embed + 40 decoder layers
        for a in at_idxs:
            # cos_per_layer[a+1] = layer a (0-indexed decoder); preceding = cos_per_layer[a]
            prev = cpl[a]      # decoder layer a-1 (or embed if a==0)
            here = cpl[a + 1]  # decoder layer a
            drop_cos = prev - here  # positive = degradation
            out[a].append(drop_cos)
    return out


def linear_fit_slope(xs, ys):
    """Least-squares slope (y vs x). Returns slope only (we don't need intercept here)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0


def main():
    if len(sys.argv) < 2:
        print("usage: cosine_ladder_35b_drift_attribute.py <json> [<json> ...]", file=sys.stderr)
        sys.exit(2)
    runs = [(Path(p).stem, load(p)) for p in sys.argv[1:]]

    # Layer types are stored in the HF oracle's meta.json, not the ladder JSON.
    # Derive from the alternating drop pattern if necessary; for 35B-A3B we know
    # the pattern is repeating (DN, DN, DN, AT) → AT at indices 3, 7, 11, ..., 39.
    at_idxs = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    # 1) cos_final vs position summary table
    print("=== cos_final vs position (every 5th position) ===")
    header = "pos  " + "  ".join(f"{name[:24]:>24}" for name, _ in runs)
    print(header)
    sample_runs = [d for _, d in runs]
    if sample_runs:
        n_pos = sample_runs[0]["n_positions_tested"]
        for pos in range(0, n_pos, 5):
            row = f"{pos:>3}  "
            for _, d in runs:
                if pos < len(d["per_pos"]):
                    row += f"{d['per_pos'][pos]['cos_final_norm']:>24.4f}  "
                else:
                    row += f"{'-':>24}  "
            print(row)
    print()

    # 2) For each run, per-AT-layer drop table + slope-vs-position
    for name, d in runs:
        print(f"=== {name}: per-AT-layer Δcos by position bucket ===")
        contrib = at_layer_contribution(d["per_pos"], at_idxs)
        n_pos = d["n_positions_tested"]
        bks = buckets(n_pos)
        # Header: AT layer, bucket means, slope
        bk_hdr = "  ".join(f"[{lo:>2}..{hi:>2})" for lo, hi in bks)
        print(f"  AT   {bk_hdr}   slope/pos    H_verdict")
        for a in at_idxs:
            drops = contrib[a]
            means = []
            for lo, hi in bks:
                bk_vals = drops[lo:hi]
                m = sum(bk_vals) / len(bk_vals) if bk_vals else 0.0
                means.append(m)
            xs = list(range(n_pos))
            ys = drops
            slope = linear_fit_slope(xs, ys)
            # H verdict: slope near 0 -> H2 (per-step); slope > some thresh -> H5/H6
            if abs(slope) < 1e-4:
                verdict = "H2 (flat)"
            elif slope > 0:
                verdict = "H5/H6 (accumulating)"
            else:
                verdict = "decreasing??"
            means_str = "  ".join(f"{m:>10.4f}" for m in means)
            print(f"  L{a:02d}  {means_str}   {slope:>+.6f}   {verdict}")
        print()

    # 3) Aggregate verdict across all AT layers
    print("=== aggregate: mean slope across all 10 AT layers ===")
    for name, d in runs:
        contrib = at_layer_contribution(d["per_pos"], at_idxs)
        xs = list(range(d["n_positions_tested"]))
        slopes = []
        for a in at_idxs:
            slopes.append(linear_fit_slope(xs, contrib[a]))
        mean_slope = sum(slopes) / len(slopes)
        print(f"  {name}: mean per-AT-layer slope = {mean_slope:+.6f} per pos")
        if abs(mean_slope) < 1e-4:
            print(f"    → H2 (per-step noise): drops are flat with position; suspects = RoPE-broadcast, weight matmul precision")
        elif mean_slope > 0:
            print(f"    → H5/H6 (accumulating noise): drops grow with position; suspects = bf16 KV cache or softmax")
        else:
            print(f"    → drops shrink with position (weird — investigate)")
    print()

    # 4) Total drift accumulation across the AT layers per position
    # cos_final ≈ product over layers of (1 - per-layer drop fraction)?
    # Simpler: print cos_final and cumulative AT-drops side by side
    print("=== drift accumulation: cos_final + total AT drop per position ===")
    for name, d in runs:
        print(f"  {name}:")
        contrib = at_layer_contribution(d["per_pos"], at_idxs)
        for pos in range(0, d["n_positions_tested"], 10):
            total_at_drop = sum(contrib[a][pos] for a in at_idxs)
            cos_final = d["per_pos"][pos]["cos_final_norm"]
            print(f"    pos={pos:>3}: cos_final={cos_final:.4f}  sum_of_AT_drops={total_at_drop:.4f}")
        print()


if __name__ == "__main__":
    main()
