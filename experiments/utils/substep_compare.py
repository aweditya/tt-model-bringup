#!/usr/bin/env python3
"""
Permanent utility — compare HF substep dump vs ttnn substep dump.

Loads two npz files:
  ~/tt-xla/.cache/hf_layer0_substeps.npz   (from utils/hf_layer0_substep_dump.py)
  ~/tt-xla/.cache/ttnn_layer0_substeps.npz (from experiments/91q_*)

The two use DIFFERENT key conventions because HF uses module-path keys
on a [batch=1, seq, ...] tensor while ttnn captures per-position keys
on [1, ...]. This utility holds the mapping.

For each known substep, computes per-position cosine and max|Δ|. Prints
a ranked list of the most-divergent substeps so the next bug hunt knows
where to look.

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/utils/substep_compare.py
"""
import os, sys
import numpy as np


HF_PATH = os.path.expanduser("~/tt-xla/.cache/hf_layer0_substeps.npz")
TTNN_PATH = os.path.expanduser("~/tt-xla/.cache/ttnn_layer0_substeps.npz")

# Mapping from a "logical" substep name → (hf_key, ttnn_key_pattern)
# ttnn_key_pattern uses {pos} as a placeholder for the position index.
# Some HF keys index into the seq dim; some are flat. We handle both.
#
# For each HF key, the slicing operation:
#   - "seq2": tensor.shape == (1, seq, *), slice [0, pos, :]
#   - "seq3": tensor.shape == (seq*n_heads, head_dim), per-pos = stride view (not used here)
#   - "flat": tensor.shape == (1, *), already per-position (e.g., our ttnn)
#
# The list below is what we actually capture in both sides.
LAYOUT = [
    # (logical name, hf_key, hf_slice_kind, ttnn_key)
    ("embed",                    "__embed__.out",                          "seq2",  "pos{pos}.embed"),
    ("input_layernorm.in",       "input_layernorm.in",                     "seq2",  "pos{pos}.input_layernorm.in"),
    ("input_layernorm.out",      "input_layernorm.out",                    "seq2",  "pos{pos}.input_layernorm.out"),
    ("in_proj_qkv.out",          "linear_attn.in_proj_qkv.out",            "seq2",  "pos{pos}.in_proj_qkv.out"),
    ("in_proj_z.out",            "linear_attn.in_proj_z.out",              "seq2",  "pos{pos}.in_proj_z.out"),
    ("in_proj_a.out",            "linear_attn.in_proj_a.out",              "seq2",  "pos{pos}.in_proj_a.out"),
    ("in_proj_b.out",            "linear_attn.in_proj_b.out",              "seq2",  "pos{pos}.in_proj_b.out"),
    # linear_attn.norm is reshaped to [seq*n_v, head_v_dim] = (240, 128) in HF
    # while our ttnn captures per-pos as (48, 128). Compare by per-pos slice.
    ("linear_attn.norm.in",      "linear_attn.norm.in",                    "240_48", "pos{pos}.norm.in"),
    ("linear_attn.norm.out",     "linear_attn.norm.out",                   "240_48", "pos{pos}.norm.out_pre_gate"),
    ("out_proj.out",             "linear_attn.out_proj.out",               "seq2",  "pos{pos}.out_proj.out"),
    ("linear_attn.full.out",     "linear_attn.out",                        "seq2",  "pos{pos}.linear_attn.out"),
    ("post_attn_layernorm.out",  "post_attention_layernorm.out",           "seq2",  "pos{pos}.post_attention_layernorm.out"),
    ("mlp.gate_proj_silu",       "mlp.act_fn.out",                         "seq2",  "pos{pos}.gate_proj_silu"),
    ("mlp.up_proj.out",          "mlp.up_proj.out",                        "seq2",  "pos{pos}.up_proj.out"),
    ("mlp.down_proj.out",        "mlp.down_proj.out",                      "seq2",  "pos{pos}.down_proj.out"),
    ("layer.out",                "__layer__.out",                          "seq2",  "pos{pos}.layer.out"),
]


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def hf_slice(arr, pos, kind):
    if kind == "seq2":
        # (1, seq, ...) → (..)
        return arr[0, pos]
    elif kind == "240_48":
        # (seq * n_v, head_v_dim) → (n_v, head_v_dim) for position pos
        n_v = 48
        return arr[pos * n_v : (pos + 1) * n_v]
    else:
        raise ValueError(f"unknown slice kind {kind}")


def ttnn_get(arr, pos):
    # ttnn captures are per-position, possibly with leading 1
    if arr.shape[0] == 1 and arr.ndim >= 2:
        return arr[0]
    return arr


def main():
    if not os.path.exists(HF_PATH):
        print(f"HF dump missing at {HF_PATH}")
        print("Run experiments/utils/hf_layer0_substep_dump.py first")
        sys.exit(1)
    if not os.path.exists(TTNN_PATH):
        print(f"ttnn dump missing at {TTNN_PATH}")
        print("Run experiments/91q_ttnn_layer0_substep_dump.py first")
        sys.exit(1)

    hf = dict(np.load(HF_PATH))
    tt = dict(np.load(TTNN_PATH))
    print(f"loaded HF: {len(hf)} tensors, ttnn: {len(tt)} tensors")

    n_positions = 5  # prompt length
    rows = []
    for logical, hf_key, kind, ttnn_pat in LAYOUT:
        if hf_key not in hf:
            print(f"  WARN: HF missing key {hf_key!r}")
            continue
        cos_per_pos = []
        norm_per_pos = []
        for pos in range(n_positions):
            ttnn_key = ttnn_pat.format(pos=pos)
            if ttnn_key not in tt:
                print(f"  WARN: ttnn missing key {ttnn_key!r}")
                continue
            hf_v = hf_slice(hf[hf_key], pos, kind).flatten()
            tt_v = ttnn_get(tt[ttnn_key], pos).flatten()
            if hf_v.shape != tt_v.shape:
                print(f"  SHAPE MISMATCH {logical} pos={pos}: hf {hf_v.shape} vs ttnn {tt_v.shape}")
                continue
            c = cosine(hf_v, tt_v)
            cos_per_pos.append(c)
            norm_per_pos.append((float(np.linalg.norm(hf_v)),
                                 float(np.linalg.norm(tt_v))))
        if not cos_per_pos:
            continue
        rows.append((logical, cos_per_pos, norm_per_pos))

    # Sort by worst-position cosine (ascending — worst first)
    rows.sort(key=lambda r: min(r[1]))

    print("\nPer-substep cosine summary (worst position first):")
    print(f"{'substep':>40s} | {'pos 0':>8s} {'pos 1':>8s} {'pos 2':>8s} {'pos 3':>8s} {'pos 4':>8s} | {'worst':>8s}")
    print("-" * 110)
    for logical, cos_per_pos, norm_per_pos in rows:
        line = f"{logical:>40s} |"
        for c in cos_per_pos:
            line += f" {c:8.5f}"
        for _ in range(5 - len(cos_per_pos)):
            line += f" {'—':>8s}"
        line += f" | {min(cos_per_pos):8.5f}"
        print(line)

    print("\nThe top entries above are likely where the bug lives.")


if __name__ == "__main__":
    main()
