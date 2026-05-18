#!/usr/bin/env python3
"""Aggregate cosine_ladder_tp comparison JSONs across multiple prompts.

Usage:
    python experiments/utils/cosine_ladder_aggregate.py \
        .cache/qb2_tp_deltanet/cosine_ladder_tp_compare_*.json

For each input JSON (produced by `client_tp cosine_ladder_tp` with ≥2 modes),
prints a per-prompt comparison row and an aggregate summary across prompts
covering: min/median/mean cosine, percentiles, sub-threshold counts, total
top-1 disagreement rate, and the resulting token-overlap rate.

Used to evaluate the Tier 1 long-context coherence gate for the owned GDN
kernel vs the manual TTNN recurrence reference; see
research/owned_gdn_diagnosis_2026_05_18.md.
"""
import sys
import json
import glob
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    paths = []
    for p in sys.argv[1:]:
        paths.extend(sorted(glob.glob(p)))
    if not paths:
        print("no matching files", file=sys.stderr)
        sys.exit(2)

    print(f"{'prompt':<60} {'base':<10} {'mode':<14} "
          f"{'min_cos':>10} {'med_cos':>10} {'mean_cos':>10} {'top1_d':>9} {'first_d':>8}")

    all_cosines_by_mode = {}
    total_disagreements_by_mode = {}
    total_steps = 0
    n_prompts = 0

    for path in paths:
        with open(path) as f:
            d = json.load(f)
        prompt = d["prompt"]
        prompt_short = prompt[:55] + ("..." if len(prompt) > 55 else "")
        base = d["base_mode"]
        for mode, c in d["comparisons"].items():
            cos = np.array(c["cosines"])
            disagree_str = f"{c['top1_disagree_count']}/{d['n_steps']}"
            print(f"{prompt_short:<60} {base:<10} {mode:<14} "
                  f"{cos.min():>10.6f} {np.median(cos):>10.6f} {cos.mean():>10.6f} "
                  f"{disagree_str:>9} {c['first_disagree_step']:>8}")
            all_cosines_by_mode.setdefault(mode, []).extend(cos.tolist())
            total_disagreements_by_mode[mode] = (
                total_disagreements_by_mode.get(mode, 0) + c["top1_disagree_count"])
        total_steps += d["n_steps"]
        n_prompts += 1

    print(f"\n=== aggregate across {n_prompts} prompts ({total_steps} total positions) ===")
    for mode, all_cos in all_cosines_by_mode.items():
        arr = np.array(all_cos)
        d = total_disagreements_by_mode[mode]
        print(f"  mode={mode}:")
        print(f"    n            = {len(arr)} positions")
        print(f"    min_cos      = {arr.min():.6f}")
        print(f"    med_cos      = {np.median(arr):.6f}")
        print(f"    mean_cos     = {arr.mean():.6f}")
        print(f"    p1 / p5 /p10 = {np.percentile(arr, 1):.6f} / "
              f"{np.percentile(arr, 5):.6f} / {np.percentile(arr, 10):.6f}")
        print(f"    num <0.99    = {int((arr < 0.99).sum())} / {len(arr)}")
        print(f"    num <0.95    = {int((arr < 0.95).sum())} / {len(arr)}")
        print(f"    num <0.90    = {int((arr < 0.90).sum())} / {len(arr)}")
        print(f"    top1_disagree= {d} / {len(arr)} ({d / len(arr) * 100:.3f}%)")
        print(f"    overlap_rate = {(len(arr) - d) / len(arr):.6f}")


if __name__ == "__main__":
    main()
