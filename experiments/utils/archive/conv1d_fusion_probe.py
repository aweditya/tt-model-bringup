#!/usr/bin/env python3
"""
Conv1d fusion probe on qb1.

Background: DeltaNet per-op probe found B4 (conv1d) costs 0.95 ms/layer
(21.5% of DeltaNet). At 48 layers = ~46 ms/tok. Single biggest remaining
DeltaNet target after V2 RoPE.

The current implementation is 6 ttnn ops:
  1) reshape: mixed_qkv [1, CONV_DIM] → [CONV_DIM, 1]
  2) concat:  [conv_state, mixed_col] → [CONV_DIM, KERNEL]
  3) mul:     conv_input * conv_weight → [CONV_DIM, KERNEL]
  4) sum:     reduce dim=-1 → [CONV_DIM]
  5) silu:    activation → [CONV_DIM]
  6) slice:   conv_input[..., 1:KERNEL] → new conv state

Question: where does the 0.95 ms live? Is it (a) the actual mul+sum compute
(memory-bound on a [8192, 3] tensor = 96 KB) or (b) the dispatch overhead of
6 separate ops?

Variants tested:
  V1) Current full path (6 ops)
  V2) Skip concat+slice (assume state buffer is updated externally): 4 ops
  V3) Skip everything except mul+sum+silu: 3 ops
  V4) Just the mul (1 op) — bandwidth probe

If V2 ≈ V1 → dispatch is cheap, optimization must reduce compute (custom kernel)
If V3 << V2 → concat+slice is heavy (state buffer pre-allocation wins)
If V4 ≈ V1 → all the cost is in mul; sum+silu+slice are noise

Run:
    ssh qb1 'pkill -9 -f serve.server; cd ~/tt-xla && .venv/bin/python experiments/utils/conv1d_fusion_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


CONV_DIM = 8192
KERNEL = 3
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
    print("Conv1d fusion probe (qb1, single P150)")
    print("=" * 78)
    print(f"Shapes: CONV_DIM={CONV_DIM}  KERNEL={KERNEL}")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print("\n[2] Build state (random, deterministic seed)...")
        rng = np.random.default_rng(42)
        mixed_qkv = rng.standard_normal((1, CONV_DIM)).astype(np.float32) * 0.3
        conv_state = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.5
        conv_weight = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
        # Pre-built [CONV_DIM, KERNEL] input (no concat needed for V2)
        prebuilt_input = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3

        def up(arr):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        mq = up(mixed_qkv)
        cs = up(conv_state)
        cw = up(conv_weight)
        pi = up(prebuilt_input)

        # === V1: current full path ===
        def v1():
            mixed_col = ttnn.reshape(mq, [CONV_DIM, 1])
            conv_input = ttnn.concat([cs, mixed_col], dim=-1)
            conv_prod = ttnn.mul(conv_input, cw)
            conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
            conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])
            return conv_out, conv_state_new

        # === V2: skip concat + slice (assume external state mgmt) ===
        def v2():
            conv_prod = ttnn.mul(pi, cw)
            conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
            return conv_out

        # === V3: skip silu too ===
        def v3():
            conv_prod = ttnn.mul(pi, cw)
            return ttnn.sum(conv_prod, dim=-1)

        # === V4: just mul (bandwidth check) ===
        def v4():
            return ttnn.mul(pi, cw)

        print("\n[3] Math sanity (V1 should give correct conv1d output)...")
        out_v1, state_v1 = v1()
        out_v1_np = ttnn.to_torch(out_v1).float().cpu().numpy().flatten()[:CONV_DIM]
        # Numpy gold
        mixed_col_np = mixed_qkv.reshape(CONV_DIM, 1)
        conv_input_np = np.concatenate([conv_state, mixed_col_np], axis=-1)
        gold = conv_input_np * conv_weight
        gold = gold.sum(axis=-1)
        gold = gold * (1.0 / (1.0 + np.exp(-gold)))
        cos_v1 = float(out_v1_np @ gold / (np.linalg.norm(out_v1_np) * np.linalg.norm(gold) + 1e-12))
        print(f"  cos(V1 out, numpy gold) = {cos_v1:.6f}")

        print("\n[4] Latency benchmark (N=100, warmup=10)...")
        ms_v1 = sync_time(device, v1)
        ms_v2 = sync_time(device, v2)
        ms_v3 = sync_time(device, v3)
        ms_v4 = sync_time(device, v4)

        print(f"\n  V1 (concat + mul + sum + silu + slice):  {ms_v1:.4f} ms")
        print(f"  V2 (mul + sum + silu — no state mgmt):   {ms_v2:.4f} ms  "
              f"({(1 - ms_v2 / ms_v1) * 100:+.1f}% vs V1)")
        print(f"  V3 (mul + sum):                          {ms_v3:.4f} ms  "
              f"({(1 - ms_v3 / ms_v1) * 100:+.1f}% vs V1)")
        print(f"  V4 (mul only — bandwidth floor):          {ms_v4:.4f} ms  "
              f"({(1 - ms_v4 / ms_v1) * 100:+.1f}% vs V1)")

        print("\n[5] Diagnosis:")
        # State mgmt cost = V1 - V2
        state_cost = ms_v1 - ms_v2
        silu_cost = ms_v2 - ms_v3
        sum_cost = ms_v3 - ms_v4
        mul_cost = ms_v4
        print(f"  state mgmt (concat + slice): {state_cost:.4f} ms ({state_cost / ms_v1 * 100:.1f}% of V1)")
        print(f"  silu:                        {silu_cost:.4f} ms ({silu_cost / ms_v1 * 100:.1f}% of V1)")
        print(f"  sum (reduce dim=-1):         {sum_cost:.4f} ms ({sum_cost / ms_v1 * 100:.1f}% of V1)")
        print(f"  mul (elementwise):           {mul_cost:.4f} ms ({mul_cost / ms_v1 * 100:.1f}% of V1)")

        print(f"\n  Per-token at 48 DeltaNet layers:")
        print(f"    Current V1: {ms_v1 * 48:.1f} ms")
        print(f"    Without state mgmt (V2): {ms_v2 * 48:.1f} ms  "
              f"(savings: {(ms_v1 - ms_v2) * 48:.1f} ms/tok)")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
