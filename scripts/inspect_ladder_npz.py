#!/usr/bin/env python3
"""Inspect a ladder.npz produced by
`experiments/cb/isolate/gemma4_long_decode_vs_hf_ladder.py`.

Prints per-(step, layer) cos heatmap-ish summary so we can locate the
drift onset. Permanent helper (per non-negotiable #3) — analysis goes
in files, not inline python -c.

Usage:
    python3 scripts/inspect_ladder_npz.py [.cache/gm4_long_decode_ladder/<ts>/ladder.npz]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / ".cache" / "gm4_long_decode_ladder" / "ladder.npz"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"FATAL: {path} not found", file=sys.stderr)
        return 2

    d = np.load(path)
    print(f"loaded {path}")
    print(f"  keys: {list(d.keys())}")
    for k in d.keys():
        a = d[k]
        print(f"  {k:20s}  shape={a.shape}  dtype={a.dtype}")

    cos_layer_h = d["cos_layer_h"]   # [N, n_layers]
    cos_logits = d["cos_logits"]     # [N]
    cos_final_norm = d["cos_final_norm"]  # [N]
    argmax_match = d["argmax_match"].astype(bool)  # [N]
    N, n_layers = cos_layer_h.shape

    print()
    print(f"N steps={N}  n_layers={n_layers}")
    print(f"argmax-match rate: {int(argmax_match.sum())}/{N} = "
          f"{argmax_match.sum() / N:.3f}")

    # Step 0 detailed per-layer.
    print()
    print("══ STEP 0 — per-layer cos ═══════════════════════════════════")
    for L in range(n_layers):
        marker = " ✓" if cos_layer_h[0, L] >= 0.99 else (
            " ⚠" if cos_layer_h[0, L] >= 0.5 else " ✗")
        print(f"  layer {L:2d}: cos = {cos_layer_h[0, L]:+.6f}{marker}")
    print(f"  final_norm: {cos_final_norm[0]:+.6f}")
    print(f"  logits:     {cos_logits[0]:+.6f}")

    # First layer to drop below 0.99 across all steps.
    print()
    print("══ FIRST LAYER PER STEP TO DROP BELOW THRESHOLD ══════════════")
    for thr in (0.99, 0.9, 0.5, 0.1):
        first_below_per_step = []
        for k in range(N):
            below = np.where(cos_layer_h[k] < thr)[0]
            first_below_per_step.append(int(below[0]) if len(below) else -1)
        rate = sum(1 for x in first_below_per_step if x >= 0) / N
        first_steps_below = [k for k in range(N)
                             if first_below_per_step[k] >= 0]
        if first_steps_below:
            first_step = first_steps_below[0]
            first_layer = first_below_per_step[first_step]
            print(f"  threshold {thr}: first (step,layer) = "
                  f"({first_step},{first_layer}) "
                  f"cos={cos_layer_h[first_step, first_layer]:+.6f}; "
                  f"{int(rate*N)}/{N} steps had at least one layer below")

    # Per-step summary: which layer is the MIN at each step?
    print()
    print("══ PER-STEP MIN-COS LAYER ════════════════════════════════════")
    min_layer_per_step = cos_layer_h.argmin(axis=1)
    print("  step  min_layer  min_cos  argmax_match")
    for k in range(min(40, N)):
        print(f"  {k:4d}  {min_layer_per_step[k]:9d}  "
              f"{cos_layer_h[k, min_layer_per_step[k]]:+.6f}  "
              f"{'✓' if argmax_match[k] else '✗'}")

    # Worst step per layer (which decode steps are worst across all layers).
    print()
    print("══ WORST 5 STEPS BY MIN-COS ══════════════════════════════════")
    worst_steps = np.argsort(cos_layer_h.min(axis=1))[:5]
    for k in worst_steps:
        L = int(cos_layer_h[k].argmin())
        print(f"  step {k:3d}: min cos = {cos_layer_h[k, L]:+.6f} at layer {L}")

    # Per-layer drift trajectory: how does each layer's cos evolve across steps?
    print()
    print("══ PER-LAYER COS STATS (across steps) ════════════════════════")
    print("  layer    mean      median    min       max")
    for L in range(n_layers):
        cs = cos_layer_h[:, L]
        print(f"  {L:5d}  {cs.mean():+.4f}  {np.median(cs):+.4f}  "
              f"{cs.min():+.4f}  {cs.max():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
