#!/usr/bin/env python3
"""Print a detailed breakdown of the cosine_ladder_tt_results.json:
  - per-position cos histogram + worst positions
  - bucketed percentiles (early/mid/late context)
  - mismatch positions + nearby token context

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/cosine_ladder_analyze.py
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

RESULTS_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_tt_results.json")
HF_REF_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_hf_ref.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=RESULTS_PATH)
    p.add_argument("--ref", default=HF_REF_PATH)
    args = p.parse_args()

    d = json.load(open(args.results))
    cos = np.array(d["per_pos_cos"])
    match = np.array(d["per_pos_top1_match"])
    hf_ids = d["generated_ids_hf"]
    tt_ids = d["generated_ids_tt"]
    M = len(cos)
    print(f"M = {M} positions, prompt = {d['prompt']!r}, ref_dtype = {d['ref_dtype']}",
          flush=True)
    print(flush=True)

    # Worst 10 positions
    order = np.argsort(cos)
    print(f"Worst 10 cosines:", flush=True)
    print(f"  {'pos':>4s}  {'cos':>10s}  {'match':>5s}  hf_id  tt_id", flush=True)
    for i in order[:10]:
        print(f"  {i+1:4d}  {cos[i]:10.6f}  {match[i]!s:>5s}  "
              f"{hf_ids[i]:6d}  {tt_ids[i]:6d}", flush=True)
    print(flush=True)

    # All mismatches
    mismatches = np.where(~match)[0]
    print(f"Top-1 mismatches: {len(mismatches)}/{M}", flush=True)
    for i in mismatches[:20]:
        print(f"  pos {i+1:3d}: hf={hf_ids[i]:6d} tt={tt_ids[i]:6d}  "
              f"cos={cos[i]:.6f}", flush=True)
    print(flush=True)

    # Bucketed percentiles by position
    buckets = [(1, 10), (10, 25), (25, 50), (50, 75), (75, 100)]
    print(f"Cosine percentiles by position bucket:", flush=True)
    print(f"  {'range':>10s}  {'min':>10s}  {'p25':>10s}  {'p50':>10s}  "
          f"{'p75':>10s}  {'max':>10s}  match%", flush=True)
    for lo, hi in buckets:
        seg = cos[lo-1:hi]
        seg_match = match[lo-1:hi]
        print(f"  [{lo:3d},{hi:3d}]  {seg.min():10.6f}  "
              f"{np.percentile(seg,25):10.6f}  {np.percentile(seg,50):10.6f}  "
              f"{np.percentile(seg,75):10.6f}  {seg.max():10.6f}  "
              f"{100.0*seg_match.mean():.1f}%", flush=True)
    print(flush=True)

    # Threshold positions
    thrs = [0.9999, 0.999, 0.99, 0.95, 0.9, 0.5]
    print(f"First-break thresholds:", flush=True)
    for t in thrs:
        b = np.where(cos < t)[0]
        first = int(b[0] + 1) if len(b) else None
        print(f"  cos < {t}: position = {first}  "
              f"(of {len(b)} positions ever below)", flush=True)


if __name__ == "__main__":
    main()
