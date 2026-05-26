#!/usr/bin/env python3
"""
Permanent utility — probe ttnn.rms_norm behavior at [48, 128] shape.

The DeltaNet bug localized to ttnn.rms_norm on per-head shape [N_V=48,
head_v_dim=128]. Per-row RMSNorm is mathematically scale-invariant —
cosine of input → cosine of output. So if cosine drops, the op isn't
doing what we think.

This script applies three RMSNorm-equivalents to the SAME input and
compares them:
  A: ttnn.rms_norm(out_per_head, weight=W, eps=1e-6)   ← what we currently use
  B: ttnn primitives by hand: (x * rsqrt(mean(x²)+eps)) * W
  C: numpy reference: same formula

Input data is loaded from ~/tt-xla/.cache/ttnn_layer2_substeps_full.npz
(our pos2 norm_in capture, which has cosine 0.99996 vs HF — so reliably
matches HF's tensor).

Verdict logic:
  A == B == C:     bug is elsewhere (silu-gate, mul, etc.)
  A != C but B == C: ttnn.rms_norm is broken at this shape → use hand-rolled
  Else:            unexpected, investigate

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/ttnn_rms_norm_probe.py
"""
import os, sys, json
import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
TTNN_DUMP = os.path.expanduser("~/tt-xla/.cache/ttnn_layer2_substeps_full.npz")
HF_DUMP = os.path.expanduser("~/tt-xla/.cache/hf_layer2_substeps.npz")
LAYER_IDX = 2


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def numpy_rms_norm(x, weight, eps):
    """The standard per-row RMSNorm formula in fp64 for reference."""
    x64 = x.astype(np.float64)
    w64 = weight.astype(np.float64)
    var = (x64 ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x64 * inv * w64).astype(np.float32)


def main():
    print("=" * 64)
    print("Probe: ttnn.rms_norm vs hand-rolled vs numpy on layer 2 norm_in")
    print("=" * 64)

    if not os.path.exists(TTNN_DUMP):
        print(f"missing {TTNN_DUMP} — run 91s first")
        sys.exit(1)

    # ----- Load our captured `out_per_head` for each of 5 positions -----
    tt_caps = dict(np.load(TTNN_DUMP))
    # We captured `norm_in` (= out_per_head before rms_norm) per position
    # AND `norm_out_pre_gate` (= ttnn.rms_norm output) per position
    captured_inputs = []
    captured_outputs = []
    for pos in range(5):
        in_arr = tt_caps[f"pos{pos}.norm_in"]
        out_arr = tt_caps[f"pos{pos}.norm_out_pre_gate"]
        # Strip leading singletons
        while in_arr.ndim > 2 and in_arr.shape[0] == 1:
            in_arr = in_arr[0]
        while out_arr.ndim > 2 and out_arr.shape[0] == 1:
            out_arr = out_arr[0]
        captured_inputs.append(in_arr)
        captured_outputs.append(out_arr)
    print(f"loaded {len(captured_inputs)} positions, "
          f"shape {captured_inputs[0].shape} → {captured_outputs[0].shape}")

    # ----- Load weight from safetensors -----
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    wk = f"model.language_model.layers.{LAYER_IDX}.linear_attn.norm.weight"
    wpath = hf_hub_download(MODEL_ID, weight_map[wk])
    with safe_open(wpath, framework="pt") as f:
        weight_np = f.get_tensor(wk).float().numpy()
    print(f"weight shape: {weight_np.shape}  "
          f"mean={weight_np.mean():+.4f} std={weight_np.std():.4f}")

    # ----- Open device for ttnn paths -----
    device = ttnn.open_device(device_id=0)

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )

    def to_device(arr, dtype=ttnn.float32):
        t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
        while t.dim() < 2:
            t = t.unsqueeze(0)
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)

    def to_host(t):
        a = ttnn.to_torch(t).float().cpu().numpy()
        while a.ndim > 2 and a.shape[0] == 1:
            a = a[0]
        return a

    print(f"\nPer-position comparison:")
    print(f"{'pos':>4s}  {'A cos':>10s}  {'B cos':>10s}  {'A vs B':>10s}  {'A vs ours':>12s}  {'C ‖·‖':>10s}")
    print("-" * 80)
    for pos in range(5):
        x_np = captured_inputs[pos].astype(np.float32)
        if x_np.shape != (48, 128):
            x_np = x_np.reshape(48, 128)
        # Upload to device
        x_tt = to_device(x_np, ttnn.float32)
        w_tt = to_device(weight_np, ttnn.bfloat16)

        # === Path A: ttnn.rms_norm (what we currently use) ===
        a_tt = ttnn.rms_norm(x_tt, weight=w_tt, epsilon=EPS)
        a_np = to_host(a_tt)
        if a_np.shape != (48, 128):
            a_np = a_np.reshape(48, 128)

        # === Path B: hand-rolled ttnn primitives ===
        # x * rsqrt(mean(x²) + eps) * weight
        # Mean = sum / D, but we can fold: rsqrt(sum/D + eps) = sqrt(D) * rsqrt(sum + D*eps)
        # For exactness, compute mean explicitly via sum/D
        D = x_np.shape[-1]   # 128
        sq = ttnn.mul(x_tt, x_tt)
        sum_sq = ttnn.sum(sq, dim=-1, keepdim=True)
        mean_sq = ttnn.mul(sum_sq, 1.0 / D)
        inv_rms = ttnn.rsqrt(ttnn.add(mean_sq, EPS))
        x_normed = ttnn.mul(x_tt, inv_rms)
        b_tt = ttnn.mul(x_normed, w_tt)
        b_np = to_host(b_tt)
        if b_np.shape != (48, 128):
            b_np = b_np.reshape(48, 128)

        # === Path C: numpy reference ===
        c_np = numpy_rms_norm(x_np, weight_np, EPS)

        # Compare A, B, C (A is what we currently get from ttnn.rms_norm)
        cos_a_c = cosine(a_np, c_np)        # ttnn op vs numpy truth
        cos_b_c = cosine(b_np, c_np)        # hand-rolled vs numpy truth
        cos_a_b = cosine(a_np, b_np)        # the two ttnn paths
        # Also compare to the actual captured "norm_out_pre_gate" from 91s
        # (this should equal A if our 91s is reading captures correctly)
        cap_out = captured_outputs[pos].astype(np.float32)
        if cap_out.shape != (48, 128):
            cap_out = cap_out.reshape(48, 128)
        cos_a_cap = cosine(a_np, cap_out)

        # Norms (sanity)
        print(f"{pos:4d}  {cos_a_c:10.6f}  {cos_b_c:10.6f}  {cos_a_b:10.6f}  {cos_a_cap:12.6f}  {np.linalg.norm(c_np):10.4f}")

    print()
    print("Interpretation:")
    print("  'A cos' = ttnn.rms_norm(x, w) vs numpy(x, w) — should be ≥ 0.9999")
    print("  'B cos' = hand-rolled ttnn (sum, rsqrt, mul) vs numpy — should be ≥ 0.9999")
    print("  'A vs B' = the two ttnn implementations vs each other")
    print("  'A vs ours' = sanity check: ttnn.rms_norm here should match 91s capture")
    print()
    print("Verdict:")
    print("  if A cos high & B cos high     → ttnn.rms_norm is fine; bug is elsewhere")
    print("  if A cos low & B cos high      → ttnn.rms_norm is broken at this shape;")
    print("                                   fix: swap to hand-rolled in 91f")
    print("  if both low                    → numpy reference doesn't match (unexpected)")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
