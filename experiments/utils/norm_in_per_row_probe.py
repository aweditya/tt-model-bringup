#!/usr/bin/env python3
"""Per-row cosine analysis: does global 0.99996 hide catastrophic per-row mismatches?

For layer 2 norm_in (48 heads × 5 positions = 240 rows), compute cosine
of each row separately between OUR captured and HF captured. Report
distribution + worst rows.

Run on qb2:
    .venv/bin/python experiments/utils/norm_in_per_row_probe.py
"""
import os
import numpy as np

TT = os.path.expanduser("~/tt-xla/.cache/ttnn_layer2_substeps_full.npz")
HF = os.path.expanduser("~/tt-xla/.cache/hf_layer2_substeps.npz")
N_V = 48


def main():
    tt = dict(np.load(TT))
    hf = dict(np.load(HF))

    print("Per-row cosine of norm_in (ttnn vs HF) — 240 rows total")
    print()
    all_cos = []
    all_norms = []
    for pos in range(5):
        ours = tt[f"pos{pos}.norm_in"].reshape(N_V, 128)
        hf_n = hf["linear_attn.norm.in"][pos * N_V:(pos + 1) * N_V]
        for h in range(N_V):
            o = ours[h].astype(np.float64)
            v = hf_n[h].astype(np.float64)
            d = (o @ v) / (np.linalg.norm(o) * np.linalg.norm(v) + 1e-30)
            all_cos.append(float(d))
            all_norms.append((float(np.linalg.norm(o)), float(np.linalg.norm(v))))

    cos = np.array(all_cos)
    norms_o = np.array([n[0] for n in all_norms])
    norms_h = np.array([n[1] for n in all_norms])
    print(f"Per-row cosine stats:")
    print(f"  count:  {len(cos)}")
    print(f"  mean:   {cos.mean():.6f}")
    print(f"  median: {np.median(cos):.6f}")
    print(f"  min:    {cos.min():.6f}")
    print(f"  max:    {cos.max():.6f}")
    print(f"  < 0.99: {(cos < 0.99).sum()}/{len(cos)} rows")
    print(f"  < 0.9:  {(cos < 0.9).sum()}/{len(cos)} rows")
    print(f"  < 0.5:  {(cos < 0.5).sum()}/{len(cos)} rows")
    print(f"  < 0.0:  {(cos < 0.0).sum()}/{len(cos)} rows")

    # Show the 10 worst rows + their magnitudes
    worst_idx = np.argsort(cos)[:10]
    print(f"\n10 worst rows (sorted by cosine):")
    print(f"{'row':>4s}  {'pos':>4s}  {'head':>4s}  {'cosine':>10s}  "
          f"{'‖ours‖':>10s}  {'‖HF‖':>10s}")
    for idx in worst_idx:
        pos = idx // N_V
        head = idx % N_V
        print(f"{idx:4d}  {pos:4d}  {head:4d}  {cos[idx]:10.6f}  "
              f"{norms_o[idx]:10.6f}  {norms_h[idx]:10.6f}")

    # Correlation: do low-cosine rows have small magnitudes?
    small_thresh = np.percentile(norms_h, 10)
    big_thresh = np.percentile(norms_h, 90)
    small_rows = norms_h < small_thresh
    big_rows = norms_h > big_thresh
    print(f"\nCosine vs magnitude:")
    print(f"  bottom 10% magnitude rows: mean cosine = {cos[small_rows].mean():.6f}")
    print(f"  top    10% magnitude rows: mean cosine = {cos[big_rows].mean():.6f}")


if __name__ == "__main__":
    main()
