#!/usr/bin/env python3
"""
Experiment 91t — Compare ttnn layer 2 substeps to HF layer 2 substeps.

Loads:
  ~/tt-xla/.cache/hf_layer2_substeps.npz       (from hf_layer0_substep_dump.py --layer 2)
  ~/tt-xla/.cache/ttnn_layer2_substeps_full.npz (from 91s)

For each substep boundary, computes per-position cosine. Reports a
ranked table — worst (least HF-aligned) substep first. That's where
the bug lives.

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/91t_layer2_substep_compare.py
"""
import os, sys
import numpy as np

HF_PATH = os.path.expanduser("~/tt-xla/.cache/hf_layer2_substeps.npz")
TT_PATH = os.path.expanduser("~/tt-xla/.cache/ttnn_layer2_substeps_full.npz")
N_V_HEADS = 48
N_POSITIONS = 5


# (logical name, hf_key, hf_slice_kind, ttnn_key_pattern)
#
# slice kinds:
#   "seq2":   hf shape (1, seq, ...) → pick [0, pos]
#   "seq3_c": hf shape (1, channels, seq) → pick [0, :, pos]
#   "240_48": hf shape (seq*n_v, head_v) → pick [pos*n_v:(pos+1)*n_v]
#   "scalar_g": (a + dt_bias is shape (1, seq, n_v_heads)) → pick [0, pos, :]
LAYOUT = [
    # Layer entry
    ("layer.in",                    "__layer__.in",                       "seq2",   "pos{pos}.x_in"),
    # First norm
    ("input_layernorm.out",         "input_layernorm.out",                "seq2",   "pos{pos}.h_after_input_norm"),
    # Projections
    ("in_proj_qkv.out",             "linear_attn.in_proj_qkv.out",        "seq2",   "pos{pos}.in_proj_qkv"),
    ("in_proj_z.out",               "linear_attn.in_proj_z.out",          "seq2",   "pos{pos}.in_proj_z"),
    ("in_proj_a.out",               "linear_attn.in_proj_a.out",          "seq2",   "pos{pos}.in_proj_a"),
    ("in_proj_b.out",               "linear_attn.in_proj_b.out",          "seq2",   "pos{pos}.in_proj_b"),
    # Conv1d (HF is [1, channels, seq+padding])
    ("conv1d.out",                  "linear_attn.conv1d.out",             "seq3_c", "pos{pos}.conv_out"),
    # RMSNormGated
    ("linear_attn.norm.in",         "linear_attn.norm.in",                "240_48", "pos{pos}.norm_in"),
    ("linear_attn.norm.out",        "linear_attn.norm.out",               "240_48", "pos{pos}.norm_out_pre_gate"),
    # Out proj + post-residual DeltaNet output
    ("out_proj.out",                "linear_attn.out_proj.out",           "seq2",   "pos{pos}.out_proj"),
    ("linear_attn.full.out",        "linear_attn.out",                    "seq2",   "pos{pos}.post_deltanet"),
    # Post-attn norm
    ("post_attn_layernorm.out",     "post_attention_layernorm.out",       "seq2",   "pos{pos}.post_attn_norm"),
    # MLP
    ("mlp.gate_proj_silu",          "mlp.act_fn.out",                     "seq2",   "pos{pos}.mlp_gate_silu"),
    ("mlp.up_proj.out",             "mlp.up_proj.out",                    "seq2",   "pos{pos}.mlp_up"),
    ("mlp.down_proj.out",           "mlp.down_proj.out",                  "seq2",   "pos{pos}.mlp_down"),
    # Final layer output
    ("layer.out",                   "__layer__.out",                      "seq2",   "pos{pos}.post_mlp"),
]


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def hf_slice(arr, pos, kind):
    if kind == "seq2":
        return arr[0, pos]
    elif kind == "seq3_c":
        # (1, channels, seq) — but conv1d output has trailing padding
        # take only valid output positions; assume HF dumped the convolution
        # output without the trailing pad in our slice
        if arr.shape[-1] > N_POSITIONS:
            return arr[0, :, pos]
        return arr[0, :, pos]
    elif kind == "240_48":
        return arr[pos * N_V_HEADS:(pos + 1) * N_V_HEADS]
    raise ValueError(f"unknown slice kind {kind}")


def ttnn_view(arr, pos=None):
    # Strip leading singletons
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main():
    if not os.path.exists(HF_PATH):
        print(f"ERROR: HF dump missing: {HF_PATH}")
        sys.exit(1)
    if not os.path.exists(TT_PATH):
        print(f"ERROR: ttnn dump missing: {TT_PATH}")
        sys.exit(1)
    hf = dict(np.load(HF_PATH))
    tt = dict(np.load(TT_PATH))
    print(f"loaded HF: {len(hf)} tensors, ttnn: {len(tt)} tensors")

    rows = []
    for logical, hf_key, kind, tt_pat in LAYOUT:
        if hf_key not in hf:
            print(f"  WARN: HF missing {hf_key!r}")
            continue
        cos_per_pos = []
        norms = []
        shapes = (None, None)
        for pos in range(N_POSITIONS):
            tt_key = tt_pat.format(pos=pos)
            if tt_key not in tt:
                print(f"  WARN: ttnn missing {tt_key!r}")
                continue
            hf_v = hf_slice(hf[hf_key], pos, kind).flatten()
            tt_v = ttnn_view(tt[tt_key]).flatten()
            if hf_v.shape != tt_v.shape:
                # Try slicing ttnn vector
                if hf_v.size <= tt_v.size:
                    tt_v = tt_v[:hf_v.size]
                else:
                    print(f"  SHAPE MISMATCH {logical} pos={pos}: hf {hf_v.shape} vs tt {tt_v.shape}")
                    continue
            c = cosine(hf_v, tt_v)
            cos_per_pos.append(c)
            norms.append((float(np.linalg.norm(hf_v)), float(np.linalg.norm(tt_v))))
            shapes = (hf_v.shape, tt_v.shape)
        if cos_per_pos:
            rows.append((logical, cos_per_pos, norms, shapes))

    rows.sort(key=lambda r: min(r[1]))

    print("\nPer-substep cosine (worst position first):")
    print(f"{'substep':>34s}  {'pos0':>9s} {'pos1':>9s} {'pos2':>9s} {'pos3':>9s} {'pos4':>9s}  {'worst':>9s}")
    print("-" * 110)
    for logical, cs, norms, shapes in rows:
        line = f"{logical:>34s} "
        for c in cs:
            line += f" {c:9.5f}"
        for _ in range(5 - len(cs)):
            line += f" {'—':>9s}"
        line += f"  {min(cs):9.5f}"
        print(line)


if __name__ == "__main__":
    main()
