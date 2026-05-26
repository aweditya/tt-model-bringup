#!/usr/bin/env python3
"""H3: compare TT vs HF MoE router top-k expert decisions per position.

HF (modeling_qwen3_5_moe.py:776) does the router softmax in fp32:
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_indices = torch.topk(router_probs, top_k).indices

TT (server_35b_ttnn.py:moe_forward_ttnn) does it in bf16:
    probs = ttnn.softmax(logits, dim=-1)
    top_vals, top_idxs = ttnn.topk(probs, k=TOP_K, dim=-1)

If two expert logits are within bf16 precision (~0.8% relative), TT's
bf16 softmax can rank them differently than HF's fp32 softmax, flipping
the top-k expert selection. This probe tests that hypothesis.

Inputs:
  - TT npz from cosine_ladder_35b.py --capture-attn-layer N (has
    pos<P>_moe_top_idxs and pos<P>_moe_top_weights per position).
  - HF oracle dir from hf_reference_35b.py --hook-attn-layer N (has
    L0_moe_L<N>_router_top_idxs.npy of shape [seq, top_k]).

Reports per position:
  - intersect = |TT_top_k ∩ HF_top_k| / top_k
  - exact_match = (sorted(TT) == sorted(HF))
  - rank_match = (TT == HF in order)
And aggregate stats.

Usage:
  python3 experiments/utils/moe_router_compare_35b.py \\
      --tt-npz .cache/sanity_2026_05_22/cosine_ladder_35b_needle_sdpa_L39_moe.attn_sub.npz \\
      --hf-dir .cache/hf_oracle_35b_needle100_L39 \\
      --layer 39
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-npz", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--positions", default="",
                    help="comma-separated positions to dump in detail; empty=all")
    args = ap.parse_args()

    tt_data = np.load(args.tt_npz)
    hf_idxs = np.load(Path(args.hf_dir) / f"L0_moe_L{args.layer}_router_top_idxs.npy")  # [seq, top_k]
    n_pos, top_k = hf_idxs.shape

    print(f"comparing TT={args.tt_npz}")
    print(f"      vs HF={args.hf_dir} (layer {args.layer})")
    print(f"positions={n_pos}, top_k={top_k}")
    print()

    # Collect positions that have TT data
    tt_positions = []
    for p in range(n_pos):
        if f"pos{p:03d}_moe_top_idxs" in tt_data:
            tt_positions.append(p)
    print(f"TT has data for {len(tt_positions)}/{n_pos} positions")
    if not tt_positions:
        print("no TT data found — exiting")
        return

    # Aggregate stats
    intersect_sizes = []
    exact_matches = 0
    rank_matches = 0
    mismatch_positions = []
    for p in tt_positions:
        tt = np.array(tt_data[f"pos{p:03d}_moe_top_idxs"]).flatten()
        hf = hf_idxs[p].flatten()
        tt_set = set(int(x) for x in tt)
        hf_set = set(int(x) for x in hf)
        overlap = len(tt_set & hf_set)
        intersect_sizes.append(overlap)
        if tt_set == hf_set:
            exact_matches += 1
        if tt.tolist() == hf.tolist():
            rank_matches += 1
        if overlap < top_k:
            mismatch_positions.append((p, overlap, sorted(tt_set), sorted(hf_set)))

    print(f"=== AGGREGATE ===")
    print(f"  rank match (TT == HF in order):     {rank_matches}/{len(tt_positions)} = {100*rank_matches/len(tt_positions):.1f}%")
    print(f"  set match (same {top_k} experts):     {exact_matches}/{len(tt_positions)} = {100*exact_matches/len(tt_positions):.1f}%")
    print(f"  mean intersect size:                {np.mean(intersect_sizes):.2f}/{top_k}")
    print(f"  positions with any mismatch:        {len(mismatch_positions)}/{len(tt_positions)}")
    print()

    # Show positions with most disagreement
    if mismatch_positions:
        print("=== TOP DISAGREEMENT POSITIONS (worst first) ===")
        mismatch_positions.sort(key=lambda x: x[1])  # smallest overlap first
        for p, overlap, tt_set, hf_set in mismatch_positions[:20]:
            only_tt = sorted(set(tt_set) - set(hf_set))
            only_hf = sorted(set(hf_set) - set(tt_set))
            print(f"  pos {p:>3}: intersect {overlap}/{top_k}  TT-only={only_tt[:5]}  HF-only={only_hf[:5]}")
        print()

    # Detail for specific positions
    detail_positions = []
    if args.positions:
        detail_positions = [int(x) for x in args.positions.split(",")]
    else:
        # Auto-pick: 0, 1, 5, then the worst mismatches, then last
        detail_positions = sorted(set([0, 1, 5] + [p for p, _, _, _ in mismatch_positions[:5]] + [n_pos - 1]))
    print("=== PER-POSITION DETAIL ===")
    print(f"{'pos':>4}  TT top-8 idxs                              HF top-8 idxs                              intersect")
    for p in detail_positions:
        if p not in tt_positions:
            continue
        tt = np.array(tt_data[f"pos{p:03d}_moe_top_idxs"]).flatten()
        hf = hf_idxs[p].flatten()
        overlap = len(set(int(x) for x in tt) & set(int(x) for x in hf))
        tt_str = ",".join(f"{int(x):3d}" for x in tt)
        hf_str = ",".join(f"{int(x):3d}" for x in hf)
        print(f"  {p:>3}  [{tt_str}]  [{hf_str}]  {overlap}/{top_k}")


if __name__ == "__main__":
    main()
