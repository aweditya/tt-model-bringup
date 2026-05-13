#!/usr/bin/env python3
"""
Probe: DN-fusion (4-linear concatenation) isolation test.

DeltaNet's deltanet_step_ondevice runs FOUR separate linears on the SAME
input h_tt:
  mixed_qkv = ttnn.linear(h, w_qkv)   # [1, CONV_DIM=8192]
  z         = ttnn.linear(h, w_z)     # [1, VAL_DIM=4096]
  a         = ttnn.linear(h, w_a)     # [1, N_V_HEADS=32]
  b         = ttnn.linear(h, w_b)     # [1, N_V_HEADS=32]

DN-fusion: concatenate the four weight matrices along output dim, do ONE
linear, then slice the result. Same total compute + bandwidth, but 4×
fewer ttnn dispatch calls + better matmul shape.

This probe validates at SMALL shape (avoids full-model 11min weight load):
  HIDDEN=512, CONV_DIM=1024, VAL_DIM=512, N_V_HEADS=32, n_a=32, n_b=32.

Hypothesis: `linear(h, concat_weight)` then slicing equals doing 4 separate
linears. Math identical; ttnn implementation should match within bf16 noise.

Run on qb1 (qb2 busy with C'4 benchmark):
    cd ~/tt-xla && .venv/bin/python experiments/utils/dn_fusion_isolation_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

# Small shape — same proportions as Qwen3.6 DeltaNet but smaller
HIDDEN = 512
CONV_DIM = 1024
VAL_DIM = 512
N_A = 32
N_B = 32


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe: DN-fusion (4 linears → 1 linear + 4 slices)")
    print(f"  HIDDEN={HIDDEN}, CONV_DIM={CONV_DIM}, VAL_DIM={VAL_DIM}, n_a={N_A}, n_b={N_B}")
    total_out = CONV_DIM + VAL_DIM + N_A + N_B
    print(f"  fused weight output dim: {total_out}")
    print("=" * 64)

    rng = np.random.default_rng(42)
    h_np = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
    w_qkv_np = rng.standard_normal((HIDDEN, CONV_DIM)).astype(np.float32) * 0.05
    w_z_np = rng.standard_normal((HIDDEN, VAL_DIM)).astype(np.float32) * 0.05
    w_a_np = rng.standard_normal((HIDDEN, N_A)).astype(np.float32) * 0.05
    w_b_np = rng.standard_normal((HIDDEN, N_B)).astype(np.float32) * 0.05

    # Numpy reference: 4 separate matmuls
    mixed_qkv_np = h_np @ w_qkv_np
    z_np = h_np @ w_z_np
    a_np = h_np @ w_a_np
    b_np = h_np @ w_b_np

    # Numpy fused: concat weights along output dim, ONE matmul, slice
    w_fused_np = np.concatenate([w_qkv_np, w_z_np, w_a_np, w_b_np], axis=1)
    fused_np = h_np @ w_fused_np
    np_mixed_qkv = fused_np[:, :CONV_DIM]
    np_z = fused_np[:, CONV_DIM:CONV_DIM + VAL_DIM]
    np_a = fused_np[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + N_A]
    np_b = fused_np[:, CONV_DIM + VAL_DIM + N_A:CONV_DIM + VAL_DIM + N_A + N_B]

    # Sanity in numpy first
    print("\nNumpy sanity (fused == separate, in fp32):")
    print(f"  mixed_qkv cos: {_cosine(np_mixed_qkv, mixed_qkv_np):.8f}")
    print(f"  z cos:         {_cosine(np_z, z_np):.8f}")
    print(f"  a cos:         {_cosine(np_a, a_np):.8f}")
    print(f"  b cos:         {_cosine(np_b, b_np):.8f}")

    # ttnn version
    device = ttnn.open_device(device_id=0)
    try:
        hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False,
        )

        h_tt = ttnn.from_torch(torch.from_numpy(h_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)

        # 4 separate weights
        w_qkv_tt = ttnn.from_torch(torch.from_numpy(w_qkv_np), dtype=ttnn.bfloat8_b,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        w_z_tt = ttnn.from_torch(torch.from_numpy(w_z_np), dtype=ttnn.bfloat8_b,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        w_a_tt = ttnn.from_torch(torch.from_numpy(w_a_np), dtype=ttnn.bfloat8_b,
                                  device=device, layout=ttnn.TILE_LAYOUT)
        w_b_tt = ttnn.from_torch(torch.from_numpy(w_b_np), dtype=ttnn.bfloat8_b,
                                  device=device, layout=ttnn.TILE_LAYOUT)

        # Fused weight
        w_fused_tt = ttnn.from_torch(torch.from_numpy(w_fused_np), dtype=ttnn.bfloat8_b,
                                      device=device, layout=ttnn.TILE_LAYOUT)

        # Run 4 separate
        mixed_qkv_tt = ttnn.linear(h_tt, w_qkv_tt, compute_kernel_config=hifi4)
        z_tt = ttnn.linear(h_tt, w_z_tt, compute_kernel_config=hifi4)
        a_tt = ttnn.linear(h_tt, w_a_tt, compute_kernel_config=hifi4)
        b_tt = ttnn.linear(h_tt, w_b_tt, compute_kernel_config=hifi4)
        ttnn_mixed_qkv = ttnn.to_torch(mixed_qkv_tt).float().cpu().numpy()
        ttnn_z = ttnn.to_torch(z_tt).float().cpu().numpy()
        ttnn_a = ttnn.to_torch(a_tt).float().cpu().numpy()
        ttnn_b = ttnn.to_torch(b_tt).float().cpu().numpy()

        # Run fused + slice
        all_tt = ttnn.linear(h_tt, w_fused_tt, compute_kernel_config=hifi4)
        sliced_mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM])
        sliced_z = ttnn.slice(all_tt, [0, CONV_DIM], [1, CONV_DIM + VAL_DIM])
        sliced_a = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM], [1, CONV_DIM + VAL_DIM + N_A])
        sliced_b = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM + N_A],
                              [1, CONV_DIM + VAL_DIM + N_A + N_B])
        ttnn_fused_mixed = ttnn.to_torch(sliced_mixed_qkv).float().cpu().numpy()
        ttnn_fused_z = ttnn.to_torch(sliced_z).float().cpu().numpy()
        ttnn_fused_a = ttnn.to_torch(sliced_a).float().cpu().numpy()
        ttnn_fused_b = ttnn.to_torch(sliced_b).float().cpu().numpy()

        print("\nTTNN: separate-linears vs fused-linear-with-slices:")
        print(f"  mixed_qkv cos: {_cosine(ttnn_fused_mixed, ttnn_mixed_qkv):.8f}, max|Δ|={float(np.abs(ttnn_fused_mixed - ttnn_mixed_qkv).max()):.4e}")
        print(f"  z cos:         {_cosine(ttnn_fused_z, ttnn_z):.8f}, max|Δ|={float(np.abs(ttnn_fused_z - ttnn_z).max()):.4e}")
        print(f"  a cos:         {_cosine(ttnn_fused_a, ttnn_a):.8f}, max|Δ|={float(np.abs(ttnn_fused_a - ttnn_a).max()):.4e}")
        print(f"  b cos:         {_cosine(ttnn_fused_b, ttnn_b):.8f}, max|Δ|={float(np.abs(ttnn_fused_b - ttnn_b).max()):.4e}")

        # Also compare to fp32 numpy reference
        print("\nFused-ttnn vs fp32 numpy (sanity):")
        print(f"  mixed_qkv cos: {_cosine(ttnn_fused_mixed, mixed_qkv_np):.6f}")
        print(f"  z cos:         {_cosine(ttnn_fused_z, z_np):.6f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
