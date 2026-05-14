#!/usr/bin/env python3
"""
DeltaNet recurrence body (B8) decomposition probe on qb2.

DeltaNet per-op probe found B8 at 0.687 ms = 15.6% of DeltaNet layer.
At 48 layers = ~33 ms/tok. Third-biggest after conv1d and QK normalize.

The block:
  H_decayed = H * decay
  kv_mem = sum(H_decayed * k_col, dim=-2)
  delta = (v - kv_mem) * beta
  H_new = H_decayed + k_col * delta
  out = sum(H_new * q_col, dim=-2)

~12-15 dispatched ops in total (mul, reshape, mul, sum, reshape, sub, mul,
reshape, add, mul, reshape, mul, sum, reshape).

Question: like QK normalize, can a fused kernel (or a different op
arrangement) reduce dispatch count? Specifically:
  - `sum(mul(A, B), dim=-2)` appears 2× — tensor contraction. Is there a
    ttnn.matmul or batched-contract path that fuses these?
  - Can we use matmul for H @ k_col / H_new @ q_col since they're inner
    products over a single dim?

Variants:
  V1) Current full sequence (15 ops)
  V2) Use ttnn.matmul for the kv_mem and out reductions instead of mul+sum
  V3) Skip the second sum (out computation) — measure just first half

Run:
    ssh qb2 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/recurrence_body_decomp_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# DeltaNet recurrence shapes (post GQA)
N_V_HEADS = 32
K_DIM = 128
V_DIM = 128
DTYPE = ttnn.bfloat16


def sync_time(device, fn, N=50, warmup=5):
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
    print("DeltaNet recurrence body (B8) decomposition probe (qb2)")
    print("=" * 78)
    print(f"N_V_HEADS={N_V_HEADS}  K_DIM={K_DIM}  V_DIM={V_DIM}")
    print(f"H shape: [1, {N_V_HEADS}, {K_DIM}, {V_DIM}] = {N_V_HEADS*K_DIM*V_DIM*2/1024:.0f} KB")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print("\n[2] Build state (random, deterministic seed)...")
        rng = np.random.default_rng(42)
        # SSM state H: [1, N_V, K_DIM, V_DIM]
        H_np = rng.standard_normal((1, N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.3
        # Q, K (post-normalize, post-GQA): [N_V, K_DIM]
        q_np = rng.standard_normal((N_V_HEADS, K_DIM)).astype(np.float32) * 0.1
        k_np = rng.standard_normal((N_V_HEADS, K_DIM)).astype(np.float32) * 0.1
        # V: [N_V, V_DIM]
        v_np = rng.standard_normal((N_V_HEADS, V_DIM)).astype(np.float32) * 0.5
        # decay: [1, N_V, 1, 1]
        decay_np = (1.0 - rng.uniform(0, 0.1, (1, N_V_HEADS, 1, 1))).astype(np.float32)
        # beta: [N_V] scalar gate per head
        beta_np = rng.uniform(0.4, 0.6, (N_V_HEADS,)).astype(np.float32)

        def up(arr):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                    device=device, layout=ttnn.TILE_LAYOUT)

        H_tt = up(H_np)
        q_tt = up(q_np)
        k_tt = up(k_np)
        v_tt = up(v_np)
        decay_tt = up(decay_np)
        beta_tt = up(beta_np)

        # === V1: current path ===
        def v1():
            H_decayed = ttnn.mul(H_tt, decay_tt)
            k_col = ttnn.reshape(k_tt, [1, N_V_HEADS, K_DIM, 1])
            kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                                   [1, N_V_HEADS, V_DIM])
            v_3d = ttnn.reshape(v_tt, [1, N_V_HEADS, V_DIM])
            delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                              ttnn.reshape(beta_tt, [1, N_V_HEADS, 1]))
            H_new = ttnn.add(H_decayed,
                              ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
            q_col = ttnn.reshape(q_tt, [1, N_V_HEADS, K_DIM, 1])
            out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, V_DIM * N_V_HEADS])
            return out, H_new

        # === V2: skip out (just compute H_new) ===
        # Diff from V1 tells us how expensive the final out = sum(H_new * q_col) is.
        def v2_no_out():
            H_decayed = ttnn.mul(H_tt, decay_tt)
            k_col = ttnn.reshape(k_tt, [1, N_V_HEADS, K_DIM, 1])
            kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                                   [1, N_V_HEADS, V_DIM])
            v_3d = ttnn.reshape(v_tt, [1, N_V_HEADS, V_DIM])
            delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                              ttnn.reshape(beta_tt, [1, N_V_HEADS, 1]))
            H_new = ttnn.add(H_decayed,
                              ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
            return H_new

        # === V3: skip kv_mem too (just decay + identity-pass) ===
        def v3_decay_only():
            H_decayed = ttnn.mul(H_tt, decay_tt)
            return H_decayed

        # === V4: just the H_new update (assuming kv_mem, delta precomputed) ===
        # Subset of V2: only the H_new = H_decayed + k_col * delta step.
        H_decayed_pre = ttnn.mul(H_tt, decay_tt)
        k_col_pre = ttnn.reshape(k_tt, [1, N_V_HEADS, K_DIM, 1])
        # Compute kv_mem once outside the timed loop
        kv_mem_pre = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed_pre, k_col_pre), dim=-2),
                                   [1, N_V_HEADS, V_DIM])
        v_3d_pre = ttnn.reshape(v_tt, [1, N_V_HEADS, V_DIM])
        delta_pre = ttnn.mul(ttnn.sub(v_3d_pre, kv_mem_pre),
                              ttnn.reshape(beta_tt, [1, N_V_HEADS, 1]))

        def v4_just_h_new():
            return ttnn.add(H_decayed_pre,
                             ttnn.mul(k_col_pre, ttnn.reshape(delta_pre, [1, N_V_HEADS, 1, V_DIM])))

        # === V5: just q_col @ H_new (assume H_new precomputed) ===
        H_new_pre = v2_no_out()
        q_col_pre = ttnn.reshape(q_tt, [1, N_V_HEADS, K_DIM, 1])

        def v5_just_out():
            return ttnn.reshape(ttnn.sum(ttnn.mul(H_new_pre, q_col_pre), dim=-2),
                                 [1, V_DIM * N_V_HEADS])

        print("\n[3] Math sanity (numpy gold for V1)...")
        # Numpy reference
        H_dec = H_np * decay_np
        k_col_np = k_np.reshape(N_V_HEADS, K_DIM, 1)
        kv_mem_np = (H_dec[0] * k_col_np).sum(axis=-2)  # [N_V, V_DIM]
        v_3d_np = v_np  # already [N_V, V_DIM]
        delta_np = (v_3d_np - kv_mem_np) * beta_np[:, None]
        H_new_np = H_dec[0] + k_col_np * delta_np[:, None, :]  # [N_V, K_DIM, V_DIM]
        q_col_np = q_np.reshape(N_V_HEADS, K_DIM, 1)
        out_np = (H_new_np * q_col_np).sum(axis=-2).flatten()  # [V_DIM * N_V_HEADS]

        out_v1_tt, H_new_v1_tt = v1()
        out_v1 = ttnn.to_torch(out_v1_tt).float().cpu().numpy().flatten()
        cos = float(out_v1 @ out_np / (np.linalg.norm(out_v1) * np.linalg.norm(out_np) + 1e-12))
        print(f"  cos(V1 ttnn, numpy gold) = {cos:.6f}")

        print("\n[4] Latency benchmark (N=50, warmup=5)...")
        ms_v1 = sync_time(device, v1)
        ms_v2 = sync_time(device, v2_no_out)
        ms_v3 = sync_time(device, v3_decay_only)
        ms_v4 = sync_time(device, v4_just_h_new)
        ms_v5 = sync_time(device, v5_just_out)

        print(f"\n  V1 full path:                            {ms_v1:.4f} ms")
        print(f"  V2 no final out (compute H_new only):    {ms_v2:.4f} ms  "
              f"({(1 - ms_v2 / ms_v1) * 100:+.1f}%)")
        print(f"  V3 decay only (H*decay):                 {ms_v3:.4f} ms  "
              f"({(1 - ms_v3 / ms_v1) * 100:+.1f}%)")
        print(f"  V4 just H_new update (rest pre-computed):{ms_v4:.4f} ms")
        print(f"  V5 just out (H_new pre-computed):        {ms_v5:.4f} ms")

        out_cost = ms_v1 - ms_v2
        h_new_cost = ms_v2 - ms_v3
        decay_cost = ms_v3
        print(f"\n  Component diagnosis:")
        print(f"    final out (q @ H_new sum):    {out_cost:.4f} ms ({out_cost/ms_v1*100:.1f}% of V1)")
        print(f"    H_new build (decay → H_new):  {h_new_cost:.4f} ms ({h_new_cost/ms_v1*100:.1f}% of V1)")
        print(f"    decay alone:                  {decay_cost:.4f} ms ({decay_cost/ms_v1*100:.1f}% of V1)")
        print(f"    V4 isolated H_new update:     {ms_v4:.4f} ms")
        print(f"    V5 isolated final out:        {ms_v5:.4f} ms")

        print(f"\n  Per-layer V1 cost: {ms_v1:.4f} ms")
        print(f"  Per-token at 48 DeltaNet layers: {ms_v1 * 48:.1f} ms")

        # Now test ttnn.matmul as a fused alternative for the sum-mul pattern
        # ttnn.sum(ttnn.mul(A, B), dim=-2) can sometimes be expressed as
        # transposed matmul: A.transpose(-1,-2) @ B (if shapes allow).
        # For our case: H_dec [1, N_V, K_DIM, V_DIM] × k_col [1, N_V, K_DIM, 1]
        # → sum over K_DIM → [1, N_V, V_DIM, 1]
        # This is a batched matmul: H_dec.transpose(K, V) @ k_col → [1, N_V, V_DIM, 1]
        # Let's see if ttnn.matmul handles it.
        print("\n[5] ttnn.matmul fusion test for sum(mul, dim=-2)...")
        try:
            def v6_matmul_replace():
                H_decayed = ttnn.mul(H_tt, decay_tt)  # [1, N_V, K_DIM, V_DIM]
                # Transpose K_DIM and V_DIM dims to enable matmul
                H_dec_T = ttnn.transpose(H_decayed, -2, -1)  # [1, N_V, V_DIM, K_DIM]
                k_col = ttnn.reshape(k_tt, [1, N_V_HEADS, K_DIM, 1])
                # batched matmul → [1, N_V, V_DIM, 1]
                kv_mem_mm = ttnn.matmul(H_dec_T, k_col)
                # Continue the recurrence with the rest of V1
                kv_mem = ttnn.reshape(kv_mem_mm, [1, N_V_HEADS, V_DIM])
                v_3d = ttnn.reshape(v_tt, [1, N_V_HEADS, V_DIM])
                delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                                  ttnn.reshape(beta_tt, [1, N_V_HEADS, 1]))
                H_new = ttnn.add(H_decayed,
                                  ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
                q_col = ttnn.reshape(q_tt, [1, N_V_HEADS, K_DIM, 1])
                # Second matmul for out
                H_new_T = ttnn.transpose(H_new, -2, -1)  # [1, N_V, V_DIM, K_DIM]
                out_mm = ttnn.matmul(H_new_T, q_col)
                out = ttnn.reshape(out_mm, [1, V_DIM * N_V_HEADS])
                return out
            # Math sanity
            out_v6 = ttnn.to_torch(v6_matmul_replace()).float().cpu().numpy().flatten()
            cos_v6 = float(out_v6 @ out_np / (np.linalg.norm(out_v6) * np.linalg.norm(out_np) + 1e-12))
            print(f"  cos(V6 matmul, numpy gold) = {cos_v6:.6f}")
            ms_v6 = sync_time(device, v6_matmul_replace)
            print(f"  V6 matmul-replace path: {ms_v6:.4f} ms  "
                  f"({(1 - ms_v6 / ms_v1) * 100:+.1f}% vs V1)")
        except Exception as e:
            print(f"  ✗ V6 matmul-replace failed: {type(e).__name__}: {str(e)[:200]}")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
