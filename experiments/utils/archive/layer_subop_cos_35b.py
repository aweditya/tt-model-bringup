#!/usr/bin/env python3
"""Isolate where INSIDE a single decoder layer TT drifts from HF.

Works for both full_attention and linear_attention (DN) layers.

For each sub-point we report:
    cos(TT, HF)   |TT|   |HF|   |TT-HF|   max|TT-HF|

A clean (cos=1.0) prefix followed by the first sub-point where cos drops
tells us which TT sub-op introduces the drift.

Layer-level points (always compared if present):
  layer_input(HF only) → in_norm → mixer_out → after_mixer → post_attn_norm
                       → moe_out → layer_output (raw, reconstructed)

DN sub-points (compared when layer is linear_attention):
  dn_in_proj_qkv, dn_in_proj_z, dn_in_proj_a, dn_in_proj_b, dn_conv1d,
  dn_core_attn_out, dn_norm_gate_z, dn_norm, dn_out_proj

after_mixer = layer_input + mixer_out  (residual ADD 1)
layer_output (raw) = layer_input + mixer_out + moe_out (NOT hidden_states[N+1]
  for the last layer — Qwen3.6 stores post-final-norm there)

Inputs:
  --tt-npz   from cosine_ladder_35b.py --capture-layer N (contains
             pos<P>_<key> for layer + dn/attn/moe sub_capture keys)
  --hf-dir   from hf_reference_35b.py --hook-attn-layer N or --hook-dn-layer N
  --layer N  the decoder layer index to analyze
  --pos      comma-separated list of positions (default: 0)
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def stats(tt_v, hf_v):
    return {
        "cos": cos(tt_v, hf_v),
        "tt_l2": float(np.linalg.norm(tt_v.astype(np.float64))),
        "hf_l2": float(np.linalg.norm(hf_v.astype(np.float64))),
        "diff_l2": float(np.linalg.norm((tt_v - hf_v).astype(np.float64))),
        "max_abs_diff": float(np.max(np.abs((tt_v - hf_v).astype(np.float64)))),
    }


def print_row(label, s):
    print(f"{label:>24}  {s['cos']:10.6f}  {s['tt_l2']:10.4f}  {s['hf_l2']:10.4f}  "
          f"{s['diff_l2']:10.4f}  {s['max_abs_diff']:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-npz", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", default="0")
    args = ap.parse_args()

    N = args.layer
    tt = np.load(args.tt_npz)
    hf_dir = Path(args.hf_dir)
    hf_hidden = np.load(hf_dir / "hidden_states.npy")
    print(f"hidden_states shape={list(hf_hidden.shape)}")

    # Layer type detection by which sub-op files are present
    is_dn = (hf_dir / f"L0_dn_in_proj_qkv_L{N}.npy").exists()
    layer_kind = "DN" if is_dn else "AT"
    print(f"layer L{N} kind={layer_kind} (per HF captures)")

    # Load HF mid-layer captures (full sequence)
    hf_caps = {}
    layer_keys = [(f"in_norm_L{N}", "in_norm"),
                  (f"mixer_out_L{N}", "mixer_out"),
                  (f"post_attn_norm_L{N}", "post_attn_norm"),
                  (f"moe_L{N}_out", "moe_out")]
    if is_dn:
        for sub in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                    "conv1d", "core_attn_out", "norm_gate_z", "norm", "out_proj"):
            layer_keys.append((f"dn_{sub}_L{N}", f"dn_{sub}"))
    for k_hf, k_tt in layer_keys:
        path = hf_dir / f"L0_{k_hf}.npy"
        if not path.exists():
            print(f"WARN: {path} missing", file=sys.stderr)
            continue
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[0]  # drop batch
        hf_caps[k_tt] = arr

    layer_input_full = hf_hidden[N]

    positions = [int(p) for p in args.pos.split(",")]

    for p in positions:
        print()
        print(f"=== layer L{N} ({layer_kind}), position {p} ===")
        print(f"{'sub-op':>24}  {'cos':>10}  {'|TT|':>10}  {'|HF|':>10}  {'|TT-HF|':>10}  {'max|Δ|':>10}")
        hf_in = layer_input_full[p]
        print(f"{'layer_input (HF ref)':>24}  {'-':>10}  {'-':>10}  "
              f"{float(np.linalg.norm(hf_in)):10.4f}  {'-':>10}  {'-':>10}")

        # DN-specific sub-ops (in chronological order through dn_forward_ttnn)
        if is_dn:
            for op in ("dn_in_proj_qkv", "dn_in_proj_z", "dn_in_proj_a", "dn_in_proj_b",
                       "dn_conv1d", "dn_core_attn_out", "dn_norm_gate_z", "dn_norm", "dn_out_proj"):
                tt_key = f"pos{p:03d}_{op}"
                if tt_key not in tt or op not in hf_caps:
                    continue
                tt_v = tt[tt_key].reshape(-1).astype(np.float32)
                # HF DN sub-ops are per-token in [batch, seq, ...] or [seq*heads, head_dim].
                # For conv1d output the shape is [B, channels, seq+pad]; need per-position slice.
                hf_arr = hf_caps[op]
                if op == "dn_conv1d":
                    # [B, channels, seq+pad] → position p column
                    hf_v = hf_arr[..., p].reshape(-1).astype(np.float32) if hf_arr.ndim >= 2 else hf_arr.reshape(-1).astype(np.float32)
                elif op in ("dn_core_attn_out", "dn_norm_gate_z", "dn_norm"):
                    # [seq*NV_heads, head_dim] flat; need rows for position p
                    # NV_heads from layer config; assume rows-per-pos = arr.shape[0] // seq
                    seq_len = hf_hidden.shape[1]
                    rpp = hf_arr.shape[0] // seq_len if hf_arr.ndim == 2 else None
                    if rpp:
                        hf_v = hf_arr[p * rpp:(p + 1) * rpp].reshape(-1).astype(np.float32)
                    else:
                        hf_v = hf_arr[p].reshape(-1).astype(np.float32)
                else:
                    # [batch, seq, hidden_or_per_head] dropped batch → [seq, hidden]
                    hf_v = hf_arr[p].reshape(-1).astype(np.float32)
                if tt_v.size != hf_v.size:
                    # Shape mismatch — note and continue (different layouts).
                    print(f"{op:>24}  shape-mismatch  TT={tt_v.size}  HF={hf_v.size}")
                    continue
                print_row(op, stats(tt_v, hf_v))

        # Layer-level points (always)
        for op in ("in_norm", "mixer_out", "post_attn_norm", "moe_out"):
            tt_key = f"pos{p:03d}_layer_{op}"
            if tt_key not in tt or op not in hf_caps:
                continue
            tt_v = tt[tt_key].reshape(-1).astype(np.float32)
            hf_v = hf_caps[op][p].reshape(-1).astype(np.float32)
            print_row(op, stats(tt_v, hf_v))

            if op == "mixer_out":
                tt_am_key = f"pos{p:03d}_layer_after_mixer"
                if tt_am_key in tt:
                    tt_am = tt[tt_am_key].reshape(-1).astype(np.float32)
                    hf_am = (layer_input_full[p] + hf_caps["mixer_out"][p]).reshape(-1).astype(np.float32)
                    print_row("after_mixer", stats(tt_am, hf_am))

        # layer_output (raw)
        tt_am_key = f"pos{p:03d}_layer_after_mixer"
        tt_moe_key = f"pos{p:03d}_layer_moe_out"
        if tt_am_key in tt and tt_moe_key in tt and "mixer_out" in hf_caps and "moe_out" in hf_caps:
            tt_lo = (tt[tt_am_key] + tt[tt_moe_key]).reshape(-1).astype(np.float32)
            hf_lo = (layer_input_full[p] + hf_caps["mixer_out"][p] + hf_caps["moe_out"][p]).reshape(-1).astype(np.float32)
            print_row("layer_output (raw)", stats(tt_lo, hf_lo))


if __name__ == "__main__":
    main()
