#!/usr/bin/env python3
"""Compare two cosine_ladder_tp NPZ dumps (one per recurrence mode) produced
from separate server lifetimes. Emits a comparison JSON in the same schema
as the in-process client_tp cosine_ladder_tp compare path.

Used when the two modes can't be compared in a single client invocation
(e.g. because the qb2 owned_gdn eager slowdown forces one mode per server
lifetime — see research/owned_gdn_diagnosis_2026_05_18.md and commit
2905470).

Usage:
    python experiments/utils/cosine_ladder_compare_two_npzs.py \
        --base   .cache/qb2_tp_deltanet/_cosine_ladder_tp_manual_*.npz \
        --other  .cache/qb2_tp_deltanet/_cosine_ladder_tp_owned_gdn_*.npz \
        --base-mode manual --other-mode owned_gdn \
        --prompt "Implement a JSON parser combinator in Rust" \
        --out .cache/qb2_tp_deltanet/cosine_ladder_tp_compare_<ts>.json

Validates that the two NPZs share prompt_ids and generated_ids (i.e. were
teacher-forced through the same baseline stream); the compare is meaningless
otherwise. Greedy generate_tp on a fixed prompt is deterministic, so two
independent server lifetimes will produce the same baseline stream.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="path to base-mode NPZ (e.g. manual)")
    p.add_argument("--other", required=True, help="path to other-mode NPZ (e.g. owned_gdn)")
    p.add_argument("--base-mode", required=True, help="name of base mode (label)")
    p.add_argument("--other-mode", required=True, help="name of other mode (label)")
    p.add_argument("--prompt", required=True, help="prompt text (recorded in compare JSON)")
    p.add_argument("--out", required=True, help="output JSON path")
    args = p.parse_args()

    base = np.load(args.base)
    other = np.load(args.other)
    if not np.array_equal(base["prompt_ids"], other["prompt_ids"]):
        print(f"ERROR: prompt_ids differ between NPZs — comparison invalid", file=sys.stderr)
        sys.exit(2)
    if not np.array_equal(base["generated_ids"], other["generated_ids"]):
        print(f"ERROR: generated_ids differ between NPZs — different teacher-forced "
              f"streams, comparison invalid", file=sys.stderr)
        sys.exit(2)

    base_logits = base["logits"].astype(np.float32)
    other_logits = other["logits"].astype(np.float32)
    if base_logits.shape != other_logits.shape:
        print(f"ERROR: logits shape mismatch: base {base_logits.shape} vs other "
              f"{other_logits.shape}", file=sys.stderr)
        sys.exit(2)
    M, V = base_logits.shape

    dots = np.sum(base_logits * other_logits, axis=1)
    cosines = dots / (np.linalg.norm(base_logits, axis=1) *
                       np.linalg.norm(other_logits, axis=1) + 1e-30)
    argmax_base = np.argmax(base_logits, axis=1)
    argmax_other = np.argmax(other_logits, axis=1)
    disagree_mask = (argmax_base != argmax_other)
    disagree = int(disagree_mask.sum())
    first_disagree = int(np.argmax(disagree_mask)) if disagree else -1

    comparison = {
        "prompt": args.prompt,
        "base_mode": args.base_mode,
        "n_prompt": int(len(base["prompt_ids"])),
        "n_steps": int(M),
        "vocab": int(V),
        "prompt_ids": base["prompt_ids"].tolist(),
        "generated_ids": base["generated_ids"].tolist(),
        "comparisons": {
            args.other_mode: {
                "min_cos": float(cosines.min()),
                "med_cos": float(np.median(cosines)),
                "mean_cos": float(cosines.mean()),
                "max_cos": float(cosines.max()),
                "top1_disagree_count": disagree,
                "top1_disagree_rate": disagree / M,
                "first_disagree_step": first_disagree,
                "cosines": cosines.tolist(),
            }
        },
        "source": "cosine_ladder_compare_two_npzs.py",
        "source_npzs": {
            args.base_mode: os.path.abspath(args.base),
            args.other_mode: os.path.abspath(args.other),
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"[compare] base={args.base_mode} other={args.other_mode} M={M} V={V}")
    print(f"  min_cos       = {cosines.min():.6f}")
    print(f"  med_cos       = {np.median(cosines):.6f}")
    print(f"  mean_cos      = {cosines.mean():.6f}")
    print(f"  p1 / p5       = {np.percentile(cosines, 1):.6f} / {np.percentile(cosines, 5):.6f}")
    print(f"  num <0.99     = {int((cosines < 0.99).sum())} / {M}")
    print(f"  num <0.95     = {int((cosines < 0.95).sum())} / {M}")
    print(f"  num <0.90     = {int((cosines < 0.90).sum())} / {M}")
    print(f"  top1_disagree = {disagree} / {M} ({disagree / M * 100:.3f}%)")
    print(f"  first_disag   = step {first_disagree}")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
