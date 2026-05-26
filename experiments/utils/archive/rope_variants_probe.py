#!/usr/bin/env python3
"""
Partial RoPE variants probe on qb1 — find a faster path than Level 1.

Background: attn_perop_probe found Level 1 partial RoPE costs 1.0 ms/layer
(44.3% of Gated Attention). At 16 layers that's 16 ms/tok — the single
biggest attention lever.

Variants tested:
  V1) Level 1 (current production): extended cos/sin with identity in
      passthrough region. 8 ops: 3× slice + neg + concat + 2× mul + add.
  V2) Rotate-only (no Level 1 trick): slice rotary region → rotate-half →
      concat with passthrough. Avoids working over the full HEAD_DIM but
      adds a concat. 7 ops.
  V3) Native ttnn.experimental.rotary_embedding wrapper on rotary slice +
      concat. Need to figure out the input layout the op expects.
  V4) Rotation matrix (full HEAD_DIM identity-extended). Single matmul
      with [HEAD_DIM, HEAD_DIM] rotation matrix. 1 op but BIG matmul
      vs many small ops — memory note says 2.6× slower than native.

Reports per-variant latency. Best variant becomes new production path.

Shapes (Qwen3.6-27B Gated Attention):
  Q: [N_Q=24, HEAD_DIM=256], rotary_dim=64, passthrough=192
  K: [N_KV=4, HEAD_DIM=256], same rotary structure

Run:
    ssh qb1 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/rope_variants_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


HEAD_DIM = 256
ROTARY_DIM = 64
N_Q = 24
N_KV = 4
DTYPE = ttnn.bfloat16


def sync_time(device, fn, N=100, warmup=10):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / N


def build_state(device):
    rng = np.random.default_rng(42)
    q = rng.standard_normal((N_Q, HEAD_DIM)).astype(np.float32)
    k = rng.standard_normal((N_KV, HEAD_DIM)).astype(np.float32)

    # Rotary cos/sin for rotary slice only [1, ROTARY_DIM]
    angles = np.linspace(0, np.pi / 4, ROTARY_DIM // 2, dtype=np.float32)
    cos_rot = np.zeros(ROTARY_DIM, dtype=np.float32)
    sin_rot = np.zeros(ROTARY_DIM, dtype=np.float32)
    cos_rot[:ROTARY_DIM // 2] = np.cos(angles)
    cos_rot[ROTARY_DIM // 2:ROTARY_DIM] = np.cos(angles)
    sin_rot[:ROTARY_DIM // 2] = np.sin(angles)
    sin_rot[ROTARY_DIM // 2:ROTARY_DIM] = np.sin(angles)

    # Extended cos/sin for Level 1 [1, HEAD_DIM] (identity in passthrough)
    cos_ext = np.ones(HEAD_DIM, dtype=np.float32)
    sin_ext = np.zeros(HEAD_DIM, dtype=np.float32)
    cos_ext[:ROTARY_DIM] = cos_rot
    sin_ext[:ROTARY_DIM] = sin_rot

    def up(arr, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                               device=device, layout=layout)

    return {
        'q': up(q),
        'k': up(k),
        'cos_rot': up(cos_rot.reshape(1, ROTARY_DIM)),
        'sin_rot': up(sin_rot.reshape(1, ROTARY_DIM)),
        'cos_ext': up(cos_ext.reshape(1, HEAD_DIM)),
        'sin_ext': up(sin_ext.reshape(1, HEAD_DIM)),
    }


def v1_level1(t, st, n_heads):
    """Current production: Level 1 with extended cos/sin."""
    half = ROTARY_DIM // 2
    x1 = ttnn.slice(t, [0, 0], [n_heads, half])
    x2 = ttnn.slice(t, [0, half], [n_heads, ROTARY_DIM])
    passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
    return ttnn.add(ttnn.mul(t, st['cos_ext']), ttnn.mul(rotated_full, st['sin_ext']))


def v2_rotate_only(t, st, n_heads):
    """Rotate only the rotary region, concat with passthrough."""
    half = ROTARY_DIM // 2
    rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
    passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
    x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
    x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
    rotated = ttnn.add(ttnn.mul(rot, st['cos_rot']),
                       ttnn.mul(rotated_half, st['sin_rot']))
    return ttnn.concat([rotated, passthru], dim=-1)


def v3_native_wrap(t, st, n_heads, device):
    """Try native ttnn.experimental.rotary_embedding on the rotary slice.

    The C'3 attempt found cos_cache padded_shape constraints conflict with
    partial-rotary slicing. Let's verify by isolating the rotary region first.
    """
    # ttnn.experimental.rotary_embedding expects input in some specific layout
    # — typically [seq_len, 1, B, head_dim] with seq_len=1 for decode.
    rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
    passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])

    # Try reshape to [1, 1, n_heads, ROTARY_DIM] for the native op
    rot_4d = ttnn.reshape(rot, [1, 1, n_heads, ROTARY_DIM])

    # cos/sin layout per memory note: (1, 1, head_dim)
    cos_3d = ttnn.reshape(st['cos_rot'], [1, 1, ROTARY_DIM])
    sin_3d = ttnn.reshape(st['sin_rot'], [1, 1, ROTARY_DIM])

    try:
        rotated_4d = ttnn.experimental.rotary_embedding(rot_4d, cos_3d, sin_3d)
        rotated = ttnn.reshape(rotated_4d, [n_heads, ROTARY_DIM])
    except Exception as e:
        # Fallback path indicator: return original input so probe still runs
        # (we'll catch the exception at the outer level and report)
        raise

    return ttnn.concat([rotated, passthru], dim=-1)


def main():
    print("=" * 78)
    print("Partial RoPE variants probe (qb1, single P150)")
    print("=" * 78)
    print(f"Shapes: HEAD_DIM={HEAD_DIM}  ROTARY_DIM={ROTARY_DIM}  N_Q={N_Q}  N_KV={N_KV}")

    print("\n[1] Open device 0...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print("\n[2] Build state...")
        st = build_state(device)
        print("  ✓ state ready")

        # Sanity: ensure all 3 variants produce the same result for Q
        # (RoPE math should be bit-equivalent up to bf16 quantization)
        print("\n[3] Correctness sanity (Q output cosine across variants)...")
        out_v1 = ttnn.to_torch(v1_level1(st['q'], st, N_Q)).float().cpu().numpy()
        out_v2 = ttnn.to_torch(v2_rotate_only(st['q'], st, N_Q)).float().cpu().numpy()
        cos_12 = float(out_v1.flatten() @ out_v2.flatten() /
                       (np.linalg.norm(out_v1) * np.linalg.norm(out_v2) + 1e-12))
        print(f"  cos(V1, V2) = {cos_12:.6f}  (should be ~1.000)")

        try:
            out_v3 = ttnn.to_torch(v3_native_wrap(st['q'], st, N_Q, device))
            out_v3 = out_v3.float().cpu().numpy()
            cos_13 = float(out_v1.flatten() @ out_v3.flatten() /
                           (np.linalg.norm(out_v1) * np.linalg.norm(out_v3) + 1e-12))
            print(f"  cos(V1, V3 native) = {cos_13:.6f}")
            v3_available = True
        except Exception as e:
            v3_available = False
            print(f"  V3 native unavailable: {type(e).__name__}: {str(e)[:200]}")

        print("\n[4] Latency benchmark (N=100, warmup=10) — Q [24, 256]...")
        ms_v1 = sync_time(device, lambda: v1_level1(st['q'], st, N_Q))
        ms_v2 = sync_time(device, lambda: v2_rotate_only(st['q'], st, N_Q))
        print(f"  V1 Level 1 (current):        {ms_v1:.4f} ms")
        print(f"  V2 rotate-only:              {ms_v2:.4f} ms")
        if v3_available:
            ms_v3 = sync_time(device, lambda: v3_native_wrap(st['q'], st, N_Q, device))
            print(f"  V3 native rotary_embedding:  {ms_v3:.4f} ms")

        print("\n[5] Latency benchmark — K [4, 256]...")
        ms_v1_k = sync_time(device, lambda: v1_level1(st['k'], st, N_KV))
        ms_v2_k = sync_time(device, lambda: v2_rotate_only(st['k'], st, N_KV))
        print(f"  V1 Level 1:                  {ms_v1_k:.4f} ms")
        print(f"  V2 rotate-only:              {ms_v2_k:.4f} ms")
        if v3_available:
            ms_v3_k = sync_time(device, lambda: v3_native_wrap(st['k'], st, N_KV, device))
            print(f"  V3 native rotary_embedding:  {ms_v3_k:.4f} ms")

        print("\n" + "=" * 78)
        print("RESULTS")
        print("=" * 78)
        ms_v1_total = ms_v1 + ms_v1_k
        ms_v2_total = ms_v2 + ms_v2_k
        print(f"  V1 Level 1 (Q + K):                 {ms_v1_total:.4f} ms")
        print(f"  V2 rotate-only (Q + K):             {ms_v2_total:.4f} ms  "
              f"({(1 - ms_v2_total / ms_v1_total) * 100:+.1f}%)")
        if v3_available:
            ms_v3_total = ms_v3 + ms_v3_k
            print(f"  V3 native (Q + K):                  {ms_v3_total:.4f} ms  "
                  f"({(1 - ms_v3_total / ms_v1_total) * 100:+.1f}%)")

        # Project to per-token at 16 layers
        print(f"\n  Per-token at 16 attention layers (Q+K RoPE):")
        print(f"    V1: {ms_v1_total * 16:.2f} ms")
        print(f"    V2: {ms_v2_total * 16:.2f} ms")
        if v3_available:
            print(f"    V3: {ms_v3_total * 16:.2f} ms")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
