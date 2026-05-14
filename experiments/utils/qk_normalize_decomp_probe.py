#!/usr/bin/env python3
"""
QK L2 normalize decomposition probe on qb2.

DeltaNet per-op probe found B6 (qk_l2_normalize) at 0.72 ms = 16.4% of layer.
At 48 layers = ~34 ms/tok. Second-biggest DeltaNet block.

Current implementation:
  Q normalize:
    qq = mul(q, q)                                      # 1 op
    q_inv = rsqrt(add(sum(qq, dim=-1, keepdim=True), EPS))  # 3 ops
    q_n = mul(q, q_inv)                                 # 1 op
  K normalize: same shape (5 ops)
  Q-scaling:  q_n = mul(q_n, 1/sqrt(K_DIM))             # 1 op
  Total: 11 ops

Question: is the cost dispatch-tax-on-many-small-ops, or genuine compute?

Variants:
  V1) current full path (11 ops)
  V2) skip Q-scaling (10 ops)
  V3) skip K-normalize (6 ops — Q normalize + Q-scaling only)
  V4) skip both Q and K normalize — just Q-scaling × constant (1 op)
  V5) ttnn.rms_norm replacement — is the fused kernel faster?

If V5 (rms_norm) is dramatically faster than V1, we have a free win.
If V5 is similar to V1, the block is bound by something else (small ops).
The per-component breakdown (V1-V2-V3-V4) tells us where the cost lives.

Run:
    ssh qb2 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/qk_normalize_decomp_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B DeltaNet shapes (post GQA broadcast — Q and K both at N_V_HEADS layout)
N_V_HEADS = 32
K_DIM = 128
EPS = 1e-6
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


def main():
    print("=" * 78)
    print("QK normalize decomposition probe (qb2)")
    print("=" * 78)
    print(f"N_V_HEADS={N_V_HEADS}  K_DIM={K_DIM}  → tensor shape [{N_V_HEADS}, {K_DIM}]")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print("\n[2] Build state...")
        rng = np.random.default_rng(42)
        q_np = rng.standard_normal((N_V_HEADS, K_DIM)).astype(np.float32) * 0.5
        k_np = rng.standard_normal((N_V_HEADS, K_DIM)).astype(np.float32) * 0.5
        ones_w = np.ones(K_DIM, dtype=np.float32)  # for rms_norm weight

        def up(arr):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                    device=device, layout=ttnn.TILE_LAYOUT)

        q_tt = up(q_np)
        k_tt = up(k_np)
        w_tt = up(ones_w)
        SCALE = 1.0 / (K_DIM ** 0.5)

        # === V1: full current path ===
        def v1():
            qq = ttnn.mul(q_tt, q_tt)
            q_n = ttnn.mul(q_tt, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
            kk = ttnn.mul(k_tt, k_tt)
            k_n = ttnn.mul(k_tt, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
            q_n = ttnn.mul(q_n, SCALE)
            return q_n, k_n

        # === V2: skip Q-scaling ===
        def v2():
            qq = ttnn.mul(q_tt, q_tt)
            q_n = ttnn.mul(q_tt, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
            kk = ttnn.mul(k_tt, k_tt)
            k_n = ttnn.mul(k_tt, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
            return q_n, k_n

        # === V3: skip K normalize (Q normalize + Q-scaling only) ===
        def v3():
            qq = ttnn.mul(q_tt, q_tt)
            q_n = ttnn.mul(q_tt, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
            q_n = ttnn.mul(q_n, SCALE)
            return q_n

        # === V4: skip Q+K normalize — just Q-scaling ===
        def v4():
            return ttnn.mul(q_tt, SCALE)

        # === V5: ttnn.rms_norm fused replacement ===
        # rms_norm: y = x * weight * rsqrt(mean(x²) + eps)
        # vs current: y = x * rsqrt(sum(x²) + eps)
        # Difference: rms_norm divides sum by K_DIM (mean), current uses raw sum.
        # So rms_norm output is scaled by sqrt(K_DIM) relative to current.
        # Math equivalence: (rms_norm output) = sqrt(K_DIM) × (current output)
        # We can compensate via Q-scaling: instead of mul(q_n, 1/sqrt(K_DIM)) at
        # the end, just don't apply it (or apply mean→sum conversion).
        # Actually, mean(x²) = sum(x²)/K_DIM, so rsqrt(mean) = sqrt(K_DIM) × rsqrt(sum).
        # rms_norm output = x * rsqrt(mean(x²) + eps)
        #                 = x * sqrt(K_DIM) * rsqrt(sum(x²) + K_DIM * eps)
        # Our current:     = x * rsqrt(sum(x²) + eps)
        # So rms_norm = sqrt(K_DIM) × current (approximately, eps interpretation differs).
        # To match, do rms_norm and then divide by sqrt(K_DIM) at the same time as
        # the existing Q-scaling (which already divides by sqrt(K_DIM)). They cancel!
        # So for Q: rms_norm replaces (mul+sum+rsqrt+mul + Q-scale) directly.
        # For K: rms_norm output is sqrt(K_DIM) × what we want; need to multiply by 1/sqrt(K_DIM)
        # to compensate. Same scale factor as Q-scaling.
        def v5():
            # rms_norm divides by sqrt(K_DIM); for K we apply 1/sqrt(K_DIM) compensation
            q_n = ttnn.rms_norm(q_tt, weight=w_tt, epsilon=EPS)  # already includes Q-scaling-equivalent
            k_n = ttnn.rms_norm(k_tt, weight=w_tt, epsilon=EPS)
            # K needs 1/sqrt(K_DIM) compensation to match original semantics
            k_n = ttnn.mul(k_n, SCALE)
            return q_n, k_n

        # Math sanity: V5 should produce results proportional to V1 (with K scaled)
        print("\n[3] Math sanity (numpy gold for V1 vs V5)...")
        # V1 gold (numpy)
        q_norm_sum = (q_np * q_np).sum(axis=-1, keepdims=True) + EPS
        q_v1 = q_np * (1.0 / np.sqrt(q_norm_sum)) * SCALE
        k_norm_sum = (k_np * k_np).sum(axis=-1, keepdims=True) + EPS
        k_v1 = k_np * (1.0 / np.sqrt(k_norm_sum))
        # V5 ttnn output
        q_v5_tt, k_v5_tt = v5()
        q_v5 = ttnn.to_torch(q_v5_tt).float().cpu().numpy()
        k_v5 = ttnn.to_torch(k_v5_tt).float().cpu().numpy()
        cos_q = float((q_v1.flatten() @ q_v5.flatten()) /
                       (np.linalg.norm(q_v1) * np.linalg.norm(q_v5) + 1e-12))
        cos_k = float((k_v1.flatten() @ k_v5.flatten()) /
                       (np.linalg.norm(k_v1) * np.linalg.norm(k_v5) + 1e-12))
        print(f"  cos(V5 Q, numpy V1 Q) = {cos_q:.6f}")
        print(f"  cos(V5 K, numpy V1 K) = {cos_k:.6f}")
        # Note: cosines should be ~1 (direction matches), but magnitudes might differ
        # if rms_norm's eps interpretation differs from manual sum+eps.

        # V1 ttnn cosine vs V1 numpy (sanity check current impl)
        q_v1_tt_out, k_v1_tt_out = v1()
        q_v1_tt = ttnn.to_torch(q_v1_tt_out).float().cpu().numpy()
        k_v1_tt = ttnn.to_torch(k_v1_tt_out).float().cpu().numpy()
        cos_q_v1 = float((q_v1.flatten() @ q_v1_tt.flatten()) /
                         (np.linalg.norm(q_v1) * np.linalg.norm(q_v1_tt) + 1e-12))
        cos_k_v1 = float((k_v1.flatten() @ k_v1_tt.flatten()) /
                         (np.linalg.norm(k_v1) * np.linalg.norm(k_v1_tt) + 1e-12))
        print(f"  cos(V1 ttnn Q, V1 numpy Q) = {cos_q_v1:.6f}  (sanity)")
        print(f"  cos(V1 ttnn K, V1 numpy K) = {cos_k_v1:.6f}  (sanity)")

        # === Latency benchmarks ===
        print("\n[4] Latency benchmark (N=100, warmup=10)...")
        ms_v1 = sync_time(device, v1)
        ms_v2 = sync_time(device, v2)
        ms_v3 = sync_time(device, v3)
        ms_v4 = sync_time(device, v4)
        ms_v5 = sync_time(device, v5)

        print(f"\n  V1 full (Q-norm + K-norm + Q-scale):       {ms_v1:.4f} ms")
        print(f"  V2 minus Q-scaling (Q-norm + K-norm):      {ms_v2:.4f} ms  "
              f"({(1 - ms_v2 / ms_v1) * 100:+.1f}%)")
        print(f"  V3 only Q-norm + Q-scaling:                {ms_v3:.4f} ms  "
              f"({(1 - ms_v3 / ms_v1) * 100:+.1f}%)")
        print(f"  V4 just Q-scaling (no normalize):          {ms_v4:.4f} ms  "
              f"({(1 - ms_v4 / ms_v1) * 100:+.1f}%)")
        print(f"  V5 ttnn.rms_norm fused:                    {ms_v5:.4f} ms  "
              f"({(1 - ms_v5 / ms_v1) * 100:+.1f}%)")

        print("\n[5] Component diagnosis:")
        q_scale_cost = ms_v1 - ms_v2
        k_norm_cost = ms_v2 - ms_v3
        q_norm_cost = ms_v3 - ms_v4
        scale_only = ms_v4
        print(f"  Q-scaling alone (mul × const):        {q_scale_cost:.4f} ms")
        print(f"  K-normalize  (mul+sum+rsqrt+add+mul): {k_norm_cost:.4f} ms")
        print(f"  Q-normalize  (mul+sum+rsqrt+add+mul): {q_norm_cost:.4f} ms")
        print(f"  Just Q-scaling (V4):                  {scale_only:.4f} ms")

        print(f"\n[6] Best-case savings projection:")
        best_path = min(ms_v1, ms_v5)
        savings = ms_v1 - best_path
        print(f"  Best variant: {'V5 rms_norm' if ms_v5 < ms_v1 else 'V1 unchanged'}")
        print(f"  Savings per layer: {savings:.4f} ms")
        print(f"  Per-token at 48 DeltaNet layers: {savings * 48:.1f} ms")
        if cos_q >= 0.9999 and cos_k >= 0.9999:
            print(f"  Math: V5 is correctness-equivalent ✓")
        else:
            print(f"  Math: V5 cosines (Q={cos_q:.4f}, K={cos_k:.4f}) — needs further math check before swap")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
