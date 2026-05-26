#!/usr/bin/env python3
"""Compare TT attn sub-step captures to HF attn submodule hook outputs at a
specified position. Identifies which sub-op first diverges between TT and HF
inside a target attention layer.

Inputs:
  - TT npz from cosine_ladder_35b.py --capture-attn-layer N:
      keys "pos{P:03d}_<subname>"  e.g. pos091_attn_q_proj_full
  - HF dir from hf_reference_35b.py --hook-attn-layer N:
      .npy files named L0_attn_L<N>_<subname>.npy  (the "L0_" prefix is a
      historical artifact of the save loop; the layer they reference is N)

Usage:
  python3 experiments/utils/cosine_ladder_35b_attn_sub_compare.py \\
      --tt-npz .cache/sanity_2026_05_22/cosine_ladder_35b_needle_sdpa_L31capture.attn_sub.npz \\
      --hf-dir .cache/hf_oracle_35b_needle100 \\
      --layer 31 \\
      --positions 0,1,91,92,96
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Map between TT sub_capture keys and HF .npy file basenames (without
# the leading "L0_" prefix and without the .npy extension). The HF files
# are saved by hf_reference_35b.py under OUT_DIR/L0_attn_L{N}_{sub}.npy.
TT_TO_HF = {
    # TT key                  HF basename suffix (we'll prepend L0_attn_L{N}_)
    "attn_q_proj_full":       "q_proj",          # both: [seq, NQ * HD * 2]
    "attn_sdpa_out":          "o_proj_input",    # both: [seq, NQ * HD] (pre-gate vs gate-applied — these differ!)
    "attn_post_gate":         "o_proj_input",    # both: [seq, NQ * HD] (post-gate, = HF o_proj input)
    "attn_o_proj_summed":     "o_proj",          # both: [seq, HIDDEN] — TT sum-of-partials vs HF o_proj output
    "attn_out_post_ar":       "o_proj",          # both: [seq, HIDDEN] — TT post-AR vs HF o_proj output
}

# MoE-side mapping. HF saves as L0_moe_L{N}_out.npy (the mlp output is the
# full MoE-block output). For TT, "moe_final" is the corresponding value.
TT_TO_HF_MOE = {
    "moe_final":     ("moe", "out"),   # both: [seq, HIDDEN]
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-npz", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--layer", type=int, required=True,
                    help="decoder layer index that was hooked / captured")
    ap.add_argument("--positions", default="0,1,91,92,96",
                    help="comma-separated positions to compare")
    args = ap.parse_args()

    tt_data = np.load(args.tt_npz)
    hf_dir = Path(args.hf_dir)

    positions = [int(x) for x in args.positions.split(",")]
    N = args.layer

    print(f"comparing TT={args.tt_npz} ↔ HF={args.hf_dir}, layer={N}")
    print(f"positions: {positions}")
    print()

    header = f"{'sub_op':>22}  " + "  ".join(f"pos{p:>3}" for p in positions)
    print(header)
    print("-" * len(header))
    for tt_key, hf_suffix in TT_TO_HF.items():
        hf_path = hf_dir / f"L0_attn_L{N}_{hf_suffix}.npy"
        if not hf_path.exists():
            print(f"{tt_key:>22}  HF file missing: {hf_path}")
            continue
        hf_arr = np.load(hf_path)  # [seq, ...] or similar
        row = f"{tt_key:>22}  "
        for p in positions:
            tt_npz_key = f"pos{p:03d}_{tt_key}"
            if tt_npz_key not in tt_data:
                row += "    miss  "
                continue
            tt_vec = tt_data[tt_npz_key]
            # HF at this position: [seq, D] -> hf_arr[p].reshape(-1)
            hf_vec = hf_arr[p]
            c = cos(tt_vec, hf_vec)
            row += f"  {c:>6.4f}"
        print(row)
    # MoE side
    print()
    print("--- MoE block ---")
    for tt_key, (hf_prefix, hf_suffix) in TT_TO_HF_MOE.items():
        hf_path = hf_dir / f"L0_{hf_prefix}_L{N}_{hf_suffix}.npy"
        if not hf_path.exists():
            print(f"{tt_key:>22}  HF file missing: {hf_path}")
            continue
        hf_arr = np.load(hf_path)
        row = f"{tt_key:>22}  "
        for p in positions:
            tt_npz_key = f"pos{p:03d}_{tt_key}"
            if tt_npz_key not in tt_data:
                row += "    miss  "
                continue
            tt_vec = tt_data[tt_npz_key]
            hf_vec = hf_arr[p]
            c = cos(tt_vec, hf_vec)
            row += f"  {c:>6.4f}"
        print(row)
    # Layer-level intermediates (in_norm, mixer_out, after_mixer, post_attn_norm,
    # moe_out). Compute HF equivalents from oracle + report cos.
    print()
    print("--- Layer-level intermediates (computed HF reference) ---")
    hf_hidden = hf_dir / "hidden_states.npy"
    hf_attn_o = hf_dir / f"L0_attn_L{N}_o_proj.npy"
    hf_moe_out = hf_dir / f"L0_moe_L{N}_out.npy"
    if hf_hidden.exists() and hf_attn_o.exists() and hf_moe_out.exists():
        hs = np.load(hf_hidden)  # [n_layers+1, seq, HIDDEN]
        ao = np.load(hf_attn_o)  # [seq, HIDDEN]
        mo = np.load(hf_moe_out)  # [seq, HIDDEN]
        # HF after_mixer at layer N = hidden_states[N, pos] + ao[pos]
        # HF final at layer N = hidden_states[N+1, pos] (= after_mixer + moe_out)
        def hf_after_mixer(p): return hs[N, p] + ao[p]
        def hf_layer_out(p):   return hs[N + 1, p]
        def hf_moe_block(p):   return mo[p]

        for tt_key, hf_fn in [
            ("layer_after_mixer", hf_after_mixer),
            ("layer_moe_out",     hf_moe_block),
        ]:
            row = f"{tt_key:>22}  "
            for p in positions:
                k = f"pos{p:03d}_{tt_key}"
                if k not in tt_data:
                    row += "    miss  "
                    continue
                c = cos(tt_data[k], hf_fn(p))
                row += f"  {c:>6.4f}"
            print(row)
        # Reconstruct TT layer output = after_mixer + moe_out and compare
        row = f"{'layer_out (TT recon)':>22}  "
        for p in positions:
            ka = f"pos{p:03d}_layer_after_mixer"
            km = f"pos{p:03d}_layer_moe_out"
            if ka in tt_data and km in tt_data:
                tt_out = tt_data[ka] + tt_data[km]
                c = cos(tt_out, hf_layer_out(p))
                row += f"  {c:>6.4f}"
            else:
                row += "    miss  "
        print(row)
    else:
        print(f"  layer-level HF references missing (hidden_states.npy + attn_L{N}_o_proj.npy + moe_L{N}_out.npy)")
    print()
    # Also report top_idxs match per position (not a cosine comparison —
    # an exact set match, since router decisions are discrete).
    print()
    print(f"{'sub_op':>22}  " + "  ".join(f"pos{p:>3}" for p in positions))
    print("--- MoE router decisions (TT only; HF top-k not currently saved) ---")
    row = f"{'moe_top_idxs (TT)':>22}  "
    for p in positions:
        k = f"pos{p:03d}_moe_top_idxs"
        if k in tt_data:
            idxs = tt_data[k].tolist()
            row += "  " + ",".join(str(x) for x in idxs[:4]) + (".." if len(idxs) > 4 else "") + " "
        else:
            row += "    miss "
    print(row)
    print()
    print("Interpretation:")
    print("  - High cos (>=0.99) at all positions → this sub-op is precision-clean")
    print("  - Cos drops at the same positions where final cos drops → noise source localized")
    print("  - Cos stays high while later sub-ops drop → noise originates downstream")


if __name__ == "__main__":
    main()
