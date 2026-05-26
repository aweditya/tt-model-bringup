#!/usr/bin/env python3
"""
Probe: ttnn implementation of Level 1 partial-rotary trick.

The numpy probe (partial_rope_level1_probe.py) already validated the math:
  q' = q * cos_extended + rotate_half_partial(q) * sin_extended
produces bit-exact identical output to the manual rotate-half implementation.

This probe validates the TTNN version:
  - Are slices on the rotary boundary tile-aligned (rotary_dim=64 is 2 tiles ✓)
  - Does ttnn.concat with 3 inputs of (n_heads, 32), (n_heads, 32), (n_heads, 192) work
  - Does bf16 quantization preserve correctness vs fp32 numpy reference

Two implementations run on device with identical inputs:
  - "baseline": manual rotate-half (current 91f apply_partial_rope, 12 ops)
  - "level1":   extended cos/sin + rotate_half_partial (~7 ops)

Compare to numpy reference; both should agree at cosine ≥ 0.99 (bf16 tolerance).

Run on qb1 (qb2 busy with C'4 benchmark):
    cd ~/tt-xla && .venv/bin/python experiments/utils/partial_rope_level1_ttnn_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

HEAD_DIM = 256
ROTARY_DIM = 64
HALF = ROTARY_DIM // 2
N_HEADS = 32          # n_q_heads — the larger case


def numpy_baseline(q, cos, sin):
    """Reference: manual rotate-half partial-rotary, our current logic."""
    rot = q[..., :ROTARY_DIM]
    passthru = q[..., ROTARY_DIM:]
    x1 = rot[..., :HALF]
    x2 = rot[..., HALF:]
    rotated_half = np.concatenate([-x2, x1], axis=-1)
    rotated = rot * cos + rotated_half * sin
    return np.concatenate([rotated, passthru], axis=-1)


def ttnn_baseline(q_tt, cos_tt, sin_tt, device):
    """ttnn version of the manual rotate-half (matches 91f.apply_partial_rope)."""
    rot = ttnn.slice(q_tt, [0, 0], [N_HEADS, ROTARY_DIM])
    passthru = ttnn.slice(q_tt, [0, ROTARY_DIM], [N_HEADS, HEAD_DIM])
    x1 = ttnn.slice(rot, [0, 0], [N_HEADS, HALF])
    x2 = ttnn.slice(rot, [0, HALF], [N_HEADS, ROTARY_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
    cos_b = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
    sin_b = ttnn.reshape(sin_tt, [1, ROTARY_DIM])
    rotated = ttnn.add(ttnn.mul(rot, cos_b), ttnn.mul(rotated_half, sin_b))
    return ttnn.concat([rotated, passthru], dim=-1)


def ttnn_level1(q_tt, cos_ext_tt, sin_ext_tt, device):
    """Level 1: extended cos/sin tables, rotate_half_partial including passthrough."""
    x1 = ttnn.slice(q_tt, [0, 0], [N_HEADS, HALF])
    x2 = ttnn.slice(q_tt, [0, HALF], [N_HEADS, ROTARY_DIM])
    passthru = ttnn.slice(q_tt, [0, ROTARY_DIM], [N_HEADS, HEAD_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
    cos_b = ttnn.reshape(cos_ext_tt, [1, HEAD_DIM])
    sin_b = ttnn.reshape(sin_ext_tt, [1, HEAD_DIM])
    return ttnn.add(ttnn.mul(q_tt, cos_b), ttnn.mul(rotated_full, sin_b))


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Probe: Level 1 partial-rotary ttnn implementation")
    print(f"  HEAD_DIM={HEAD_DIM}, ROTARY_DIM={ROTARY_DIM}, N_HEADS={N_HEADS}")
    print("=" * 64)

    rng = np.random.default_rng(42)
    device = ttnn.open_device(device_id=0)
    try:
        print(f"\n{'pos':>5} {'base vs np':>12} {'lvl1 vs np':>12} {'lvl1 vs base':>14}")
        print("-" * 60)

        ref_count = 0
        for pos in [0, 1, 5, 31, 32, 64, 127]:
            q_np = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float32) * 0.1
            freqs = 1.0 / (10_000_000.0 ** (np.arange(HALF).astype(np.float32) / HALF))
            angles = pos * freqs
            cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
            sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
            cos_ext_np = np.concatenate([cos_np, np.ones(HEAD_DIM - ROTARY_DIM)]).astype(np.float32)
            sin_ext_np = np.concatenate([sin_np, np.zeros(HEAD_DIM - ROTARY_DIM)]).astype(np.float32)

            # Numpy reference
            np_ref = numpy_baseline(q_np, cos_np, sin_np)

            # ttnn upload
            q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            cos_tt = ttnn.from_torch(torch.from_numpy(cos_np.reshape(1, ROTARY_DIM)),
                                      dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            sin_tt = ttnn.from_torch(torch.from_numpy(sin_np.reshape(1, ROTARY_DIM)),
                                      dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            cos_ext_tt = ttnn.from_torch(torch.from_numpy(cos_ext_np.reshape(1, HEAD_DIM)),
                                          dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            sin_ext_tt = ttnn.from_torch(torch.from_numpy(sin_ext_np.reshape(1, HEAD_DIM)),
                                          dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

            out_base = ttnn.to_torch(ttnn_baseline(q_tt, cos_tt, sin_tt, device)).float().cpu().numpy()
            out_lvl1 = ttnn.to_torch(ttnn_level1(q_tt, cos_ext_tt, sin_ext_tt, device)).float().cpu().numpy()

            cos_base = _cosine(out_base, np_ref)
            cos_lvl1 = _cosine(out_lvl1, np_ref)
            cos_inter = _cosine(out_base, out_lvl1)

            tag = "✓" if (cos_lvl1 > 0.99 and cos_inter > 0.99) else "✗"
            print(f"{pos:>5} {cos_base:>12.6f} {cos_lvl1:>12.6f} {cos_inter:>14.6f}  {tag}")
            if cos_lvl1 > 0.99 and cos_inter > 0.99:
                ref_count += 1

        print()
        if ref_count == 7:
            print("✓ Level 1 ttnn implementation produces matching output (bf16 tolerance).")
            print("  Safe to apply to 91f after C'4 lands.")
        else:
            print(f"⚠ Only {ref_count}/7 positions agree. Investigate.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
