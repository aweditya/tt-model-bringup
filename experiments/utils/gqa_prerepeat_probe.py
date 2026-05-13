#!/usr/bin/env python3
"""
GQA pre-repeat probe on qb2.

DeltaNet per-op probe found B5 (gqa_repeat-interleave) costs 0.66 ms/layer
(14.9% of DeltaNet). At 48 layers = ~32 ms/tok. Second-biggest fusion target
after conv1d.

The interleave is pure dispatch tax — reshape + repeat + reshape doesn't
do useful compute, just memory rearrangement. If we pre-arrange Q/K weights
so in_proj output is ALREADY at [N_V_HEADS, K_DIM] layout, we skip the
interleave entirely.

Trade-off: Q+K weight output dim grows 2× (each head duplicated N_REP=2
times in the weight). in_proj matmul becomes ~30% heavier. Expected:
  current  = in_proj (0.31) + gqa_interleave (0.66) = 0.97 ms
  proposed = in_proj * 1.30 (0.40) + 0           = 0.40 ms
  savings  = 0.57 ms/layer × 48 = 27 ms/tok

The probe measures:
  V1) current: small in_proj + gqa_interleave
  V2) prerepeat: bigger in_proj, no interleave
  Cosine should be 1.000 (math identical).
  Latency delta is the headline number.

Run:
    ssh qb2 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/gqa_prerepeat_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B DeltaNet shapes
HIDDEN = 5120
N_K_HEADS = 16
N_V_HEADS = 32
N_REP = N_V_HEADS // N_K_HEADS  # 2
K_DIM = 128
V_DIM = 128
VAL_DIM = N_V_HEADS * V_DIM           # 4096
KEY_DIM_CURRENT = N_K_HEADS * K_DIM   # 2048 (Q at N_K layout)
KEY_DIM_REPEATED = N_V_HEADS * K_DIM  # 4096 (Q at N_V layout, pre-repeated)
EPS = 1e-6
DTYPE = ttnn.bfloat16
hifi4 = None


def sync_time(device, fn, N=50, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / N


def build_state(device):
    """Two parallel weight sets: V1 (current N_K) and V2 (prerepeated N_V)."""
    rng = np.random.default_rng(42)
    h = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.3

    # V1 weights: Q at [HIDDEN, N_K_HEADS * K_DIM]; downstream interleave repeats it.
    w_q_v1 = rng.standard_normal((HIDDEN, KEY_DIM_CURRENT)).astype(np.float32) / np.sqrt(HIDDEN)

    # V2 weights: pre-repeated. For each Q head h ∈ {0..15}, columns h*K_DIM:(h+1)*K_DIM
    # appear N_REP=2 times in the new weight, at positions:
    #   v_head 2h:   columns h*K_DIM:(h+1)*K_DIM
    #   v_head 2h+1: columns h*K_DIM:(h+1)*K_DIM
    # i.e. the new weight is [HIDDEN, N_V_HEADS * K_DIM] where v_head's K_DIM cols
    # are copied from k_head = v_head // N_REP.
    w_q_v1_reshape = w_q_v1.reshape(HIDDEN, N_K_HEADS, K_DIM)
    w_q_v2 = np.zeros((HIDDEN, N_V_HEADS, K_DIM), dtype=np.float32)
    for vh in range(N_V_HEADS):
        kh = vh // N_REP
        w_q_v2[:, vh, :] = w_q_v1_reshape[:, kh, :]
    w_q_v2 = w_q_v2.reshape(HIDDEN, KEY_DIM_REPEATED)

    def up(arr):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                device=device, layout=ttnn.TILE_LAYOUT)
    return {
        'h':    up(h),
        'w_v1': up(w_q_v1),
        'w_v2': up(w_q_v2),
    }, (h, w_q_v1, w_q_v2)


def gqa_interleave(q_flat_tt, n_kh, d):
    """V1: current in-code interleave. q_flat_tt is [n_kh * d]; output [n_kh*N_REP, d]."""
    t = ttnn.reshape(q_flat_tt, [n_kh, 1, d])
    t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
    return ttnn.reshape(t, [n_kh * N_REP, d])


def main():
    global hifi4
    print("=" * 78)
    print("GQA pre-repeat probe (qb2)")
    print("=" * 78)
    print(f"HIDDEN={HIDDEN}  N_K_HEADS={N_K_HEADS}  N_V_HEADS={N_V_HEADS}  N_REP={N_REP}")
    print(f"KEY_DIM_CURRENT={KEY_DIM_CURRENT}  KEY_DIM_REPEATED={KEY_DIM_REPEATED}")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    try:
        state, np_state = build_state(device)
        h_np, w_q_v1_np, w_q_v2_np = np_state

        # Math sanity (numpy): both paths should give same Q at [N_V_HEADS, K_DIM]
        print("\n[2] Math sanity (numpy)...")
        # V1: in_proj small → reshape → repeat-interleave
        q_v1_small = h_np @ w_q_v1_np  # [1, N_K_HEADS * K_DIM]
        q_v1_per_kh = q_v1_small.reshape(N_K_HEADS, K_DIM)
        q_v1_per_vh = np.broadcast_to(q_v1_per_kh[:, None, :],
                                       (N_K_HEADS, N_REP, K_DIM)).reshape(N_V_HEADS, K_DIM)
        # V2: in_proj big → reshape (no interleave)
        q_v2_big = h_np @ w_q_v2_np  # [1, N_V_HEADS * K_DIM]
        q_v2_per_vh = q_v2_big.reshape(N_V_HEADS, K_DIM)

        cos_np = float((q_v1_per_vh.flatten() @ q_v2_per_vh.flatten())
                        / (np.linalg.norm(q_v1_per_vh) * np.linalg.norm(q_v2_per_vh) + 1e-12))
        max_diff_np = float(np.abs(q_v1_per_vh - q_v2_per_vh).max())
        print(f"  cos(V1 numpy, V2 numpy) = {cos_np:.6f}  max|Δ| = {max_diff_np:.4e}")

        # Math sanity (ttnn)
        print("\n[3] Math sanity (ttnn, bf16)...")
        def v1_full():
            q_small_tt = ttnn.linear(state['h'], state['w_v1'], compute_kernel_config=hifi4)
            q_small_flat = ttnn.reshape(q_small_tt, [N_K_HEADS * K_DIM])
            return gqa_interleave(q_small_flat, N_K_HEADS, K_DIM)

        def v2_full():
            q_big_tt = ttnn.linear(state['h'], state['w_v2'], compute_kernel_config=hifi4)
            return ttnn.reshape(q_big_tt, [N_V_HEADS, K_DIM])

        out_v1 = ttnn.to_torch(v1_full()).float().cpu().numpy()
        out_v2 = ttnn.to_torch(v2_full()).float().cpu().numpy()
        cos_tt = float((out_v1.flatten() @ out_v2.flatten())
                        / (np.linalg.norm(out_v1) * np.linalg.norm(out_v2) + 1e-12))
        max_diff_tt = float(np.abs(out_v1 - out_v2).max())
        print(f"  cos(V1 ttnn, V2 ttnn) = {cos_tt:.6f}  max|Δ| = {max_diff_tt:.4e}")

        # Latency benchmarks
        print("\n[4] Latency benchmark (N=50, warmup=5)...")
        ms_v1 = sync_time(device, v1_full, N=50, warmup=5)
        ms_v2 = sync_time(device, v2_full, N=50, warmup=5)
        # Also isolate in_proj and interleave separately
        def in_proj_v1():
            return ttnn.linear(state['h'], state['w_v1'], compute_kernel_config=hifi4)
        def in_proj_v2():
            return ttnn.linear(state['h'], state['w_v2'], compute_kernel_config=hifi4)
        def interleave_only():
            q = in_proj_v1()
            qf = ttnn.reshape(q, [N_K_HEADS * K_DIM])
            return gqa_interleave(qf, N_K_HEADS, K_DIM)

        ms_ip_v1 = sync_time(device, in_proj_v1, N=50, warmup=5)
        ms_ip_v2 = sync_time(device, in_proj_v2, N=50, warmup=5)
        ms_il = sync_time(device, interleave_only, N=50, warmup=5)
        # Approximate interleave-alone = ms_il - ms_ip_v1 (subtract in_proj)
        ms_il_alone = ms_il - ms_ip_v1

        print(f"\n  V1 full (in_proj_small + interleave):  {ms_v1:.4f} ms")
        print(f"  V2 full (in_proj_big, no interleave):  {ms_v2:.4f} ms  "
              f"({(1 - ms_v2 / ms_v1) * 100:+.1f}%)")
        print(f"\n  Component breakdown:")
        print(f"    in_proj V1 (HIDDEN×{KEY_DIM_CURRENT}):    {ms_ip_v1:.4f} ms")
        print(f"    in_proj V2 (HIDDEN×{KEY_DIM_REPEATED}):    {ms_ip_v2:.4f} ms  "
              f"({(ms_ip_v2 / ms_ip_v1 - 1) * 100:+.1f}% bigger)")
        print(f"    interleave alone:                  {ms_il_alone:.4f} ms")

        savings = ms_v1 - ms_v2
        print(f"\n  Per-layer savings: {savings:.4f} ms")
        print(f"  Per-token at 48 DeltaNet layers: {savings * 48:.1f} ms")
        if savings > 0:
            print(f"\n  ✓ Pre-repeat path is FASTER. Recommend integration.")
        else:
            print(f"\n  ✗ Pre-repeat is SLOWER (in_proj matmul overhead exceeds interleave savings).")

        # Same for K (V doesn't need repeat — already N_V_HEADS)
        print(f"\n  Note: applies to BOTH Q and K projections. K savings = same.")
        print(f"  Total potential at Q+K: {savings * 2 * 48:.1f} ms/tok")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
