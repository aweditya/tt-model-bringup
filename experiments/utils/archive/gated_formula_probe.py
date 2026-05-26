#!/usr/bin/env python3
"""
Permanent utility — verify the gated output formula using captured tensors.

Question: our captured `gated` (output of RMSNormGated step) has cosine
0.81 vs HF's `linear_attn.norm.out`. The probe showed ttnn.rms_norm is
correct (cosine 0.999991). So where's the divergence?

This script loads our captures + HF captures, then for each position:
1. Compute "ideal numpy gated" = (norm_in * rsqrt(mean²+eps)) * weight * silu(z)
   using OUR captured norm_in and z
2. Compare to OUR captured gated (= what ttnn produced)
3. Compare to HF's linear_attn.norm.out (= what HF produced)

Three diagnostic comparisons:
  - numpy_gated vs our_captured_gated:
      if HIGH → ttnn is doing the right math
      if LOW  → ttnn op produces different output than the formula
  - numpy_gated vs hf_gated:
      if HIGH → formula matches HF's, our impl just has a ttnn op divergence
      if LOW  → our formula doesn't match HF's
  - our_captured_gated vs hf_gated (already known: 0.81)

If numpy_gated matches BOTH our_captured AND hf_gated, then our captured
gated must also match hf_gated... contradicting the 0.81 cosine. So at
least one of the three CAN'T match.

Run on qb2:
    .venv/bin/python experiments/utils/gated_formula_probe.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
TTNN_DUMP = os.path.expanduser("~/tt-xla/.cache/ttnn_layer2_substeps_full.npz")
HF_DUMP = os.path.expanduser("~/tt-xla/.cache/hf_layer2_substeps.npz")
LAYER_IDX = 2
N_V_HEADS = 48
V_DIM = 128


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float64)


def main():
    print("=" * 64)
    print("Probe: gated = RMSNorm(x) * weight * silu(z) — formula vs ttnn vs HF")
    print("=" * 64)

    tt = dict(np.load(TTNN_DUMP))
    hf = dict(np.load(HF_DUMP))
    print(f"loaded {len(tt)} ttnn captures, {len(hf)} HF captures")

    # Load weight from safetensors
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    wk = f"model.language_model.layers.{LAYER_IDX}.linear_attn.norm.weight"
    wpath = hf_hub_download(MODEL_ID, weight_map[wk])
    with safe_open(wpath, framework="pt") as f:
        weight = f.get_tensor(wk).float().numpy()
    print(f"weight shape: {weight.shape}, mean={weight.mean():+.4f}")

    # Mix-and-match test: compute numpy formula with combinations of
    # {our, hf} norm_in × {our, hf} silu_z. If one combination matches
    # hf_gated, we know which input is wrong.
    print(f"\nMix-and-match cosine matrix (formula output vs HF norm.out):")
    print(f"{'pos':>4s}  {'OUR/OUR':>10s}  {'OUR/HF':>10s}  {'HF/OUR':>10s}  {'HF/HF':>10s}  {'in_cos':>10s}")
    print("-" * 80)

    for pos in range(5):
        # Our captures (per-position)
        our_norm_in = tt[f"pos{pos}.norm_in"]      # (48, 128) — recurrence output
        our_gated = tt[f"pos{pos}.gated"]            # (1, 6144) — final norm.out per HF naming
        our_silu_z = tt[f"pos{pos}.silu_z"]          # (1, 6144)
        # HF captures (batched, slice per-position)
        hf_norm_in = hf["linear_attn.norm.in"][pos * N_V_HEADS:(pos + 1) * N_V_HEADS]  # (48, 128)
        hf_gated_per_head = hf["linear_attn.norm.out"][pos * N_V_HEADS:(pos + 1) * N_V_HEADS]  # (48, 128)

        # Reshape ours to (48, 128) for direct comparison
        our_norm_in_2d = our_norm_in.reshape(N_V_HEADS, V_DIM)
        our_gated_2d = our_gated.reshape(N_V_HEADS, V_DIM)
        our_silu_z_2d = our_silu_z.reshape(N_V_HEADS, V_DIM)

        # Compute numpy_gated using OUR norm_in (so we measure "given our
        # recurrence output, what should gated be?")
        x = our_norm_in_2d.astype(np.float64)
        var = (x ** 2).mean(axis=-1, keepdims=True)
        x_normed = x / np.sqrt(var + EPS)
        # Use HF's z directly; we don't capture raw z, but silu_z we do — derive z back
        # Actually easier: just use our captured silu_z directly (this is what HF would compute too if z matches)
        # But silu_z from us has cosine ~0.9999 vs HF presumably — check input z cosine to verify
        numpy_gated_with_our_siluz = (x_normed * weight.astype(np.float64) * our_silu_z_2d.astype(np.float64)).astype(np.float32)

        # Compute numpy_gated using HF's z. Need to derive z. Hmm we don't have HF's silu_z directly.
        # The HF substep dump didn't hook the silu output. Skip this for now.

        # Get HF's z for this position. HF stores in_proj_z.out shape [1, 5, 6144].
        hf_z = hf["linear_attn.in_proj_z.out"][0, pos]   # [6144]
        hf_z_2d = hf_z.reshape(N_V_HEADS, V_DIM)
        hf_silu_z = (hf_z_2d / (1.0 + np.exp(-hf_z_2d.astype(np.float64)))).astype(np.float32)

        def numpy_formula(norm_in, silu_z):
            """RMSNorm + weight + silu-gate, fp32 throughout (no bf16 cast)."""
            x = norm_in.astype(np.float64)
            var = (x ** 2).mean(axis=-1, keepdims=True)
            x_normed = x / np.sqrt(var + EPS)
            return (x_normed * weight.astype(np.float64) * silu_z.astype(np.float64)).astype(np.float32)

        out_our_our = numpy_formula(our_norm_in_2d, our_silu_z_2d)
        out_our_hf  = numpy_formula(our_norm_in_2d, hf_silu_z)
        out_hf_our  = numpy_formula(hf_norm_in,     our_silu_z_2d)
        out_hf_hf   = numpy_formula(hf_norm_in,     hf_silu_z)

        c_our_our = cosine(out_our_our, hf_gated_per_head)
        c_our_hf  = cosine(out_our_hf,  hf_gated_per_head)
        c_hf_our  = cosine(out_hf_our,  hf_gated_per_head)
        c_hf_hf   = cosine(out_hf_hf,   hf_gated_per_head)
        c_norm_in = cosine(our_norm_in_2d, hf_norm_in)
        print(f"{pos:4d}  {c_our_our:10.6f}  {c_our_hf:10.6f}  "
              f"{c_hf_our:10.6f}  {c_hf_hf:10.6f}  {c_norm_in:10.6f}")

    print()
    print("Interpretation:")
    print("  numpy_vs_ours: if HIGH → ttnn produced the math we wrote")
    print("                 if LOW  → ttnn op bug somewhere")
    print("  numpy_vs_hf  : if HIGH → formula matches HF (our recurrence output direction is right)")
    print("                 if LOW  → formula differs from HF (input direction differs more than cosine suggests)")
    print("  ours_vs_hf   : already known low (≈ 0.81). Confirmed if numpy_vs_ours HIGH and numpy_vs_hf LOW")
    print()
    print("If numpy_vs_ours HIGH and numpy_vs_hf LOW, the recurrence input direction")
    print("matches HF only by cosine, but small magnitude differences ALSO affect the gate output")
    print("(because rms_norm normalizes by magnitude, so per-row scaling matters in HF's vs ours).")


if __name__ == "__main__":
    main()
