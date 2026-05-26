#!/usr/bin/env python3
"""
Per-op latency profile for DeltaNet step on qb1 (single P150).

Hypothesis (from feedback_qb1_mlp_at_78pct_peak.md): MLP is already at 78% of
peak BW, so the ~73 ms/token of "other" decode time must live in DeltaNet
recurrence + dispatch overhead. This probe finds WHERE inside one DeltaNet
step the cost lives, so we know what's worth optimizing.

Method: import deltanet_step_ondevice from 91f, then re-implement the same
op sequence here with explicit `ttnn.synchronize_device` around named blocks.
Run N=50 iterations per block; report mean ms.

Blocks (in order):
  B1) input rms_norm
  B2) in_proj_all (big fused linear, HIDDEN → CONV_DIM + VAL_DIM + 2*N_V_HEADS)
  B3) 4× slice (qkv | z | a | b)
  B4) conv1d (reshape + concat + mul + sum + silu)
  B5) GQA repeat-interleave for q, k
  B6) Q/K L2 normalize (mul, sum, rsqrt, mul) + Q-scaling
  B7) gate/decay/beta (softplus, exp, neg, sigmoid)
  B8) recurrence body (H_decayed, kv_mem, delta, H_new, q@H_new sum)
  B9) per-head rms_norm + silu(z) gate
  B10) out_proj (linear)
  B11) residual add

Sync-overhead caveat: each `synchronize_device` adds ~50 µs, so sub-0.1ms
blocks have noise floor. Aggregated block times should sum close to the
unsegmented `deltanet_step_ondevice` baseline.

Run:
    ssh qb1 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/deltanet_perop_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B DeltaNet config
HIDDEN = 5120
N_K_HEADS = 16
N_V_HEADS = 32
K_DIM = 128
V_DIM = 128
KERNEL = 3
KEY_DIM = N_K_HEADS * K_DIM    # 2048
VAL_DIM = N_V_HEADS * V_DIM    # 4096
CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 8192
N_REP = N_V_HEADS // N_K_HEADS  # 2
EPS = 1e-6
DTYPE = ttnn.bfloat16

hifi4 = None  # set after device open


class T:
    """Tracks per-block timing."""
    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, name, ms):
        self.totals[name] = self.totals.get(name, 0.0) + ms
        self.counts[name] = self.counts.get(name, 0) + 1

    def mean(self, name):
        return self.totals[name] / self.counts[name]

    def report(self, label):
        print(f"\n  {label}")
        items = sorted(self.totals.keys(), key=lambda k: -self.mean(k))
        total_ms = sum(self.mean(k) for k in items)
        print(f"  {'block':<32s} {'mean (ms)':>12s} {'% step':>10s}")
        print(f"  {'-' * 32} {'-' * 12} {'-' * 10}")
        for k in items:
            m = self.mean(k)
            pct = m / total_ms * 100 if total_ms > 0 else 0
            print(f"  {k:<32s} {m:>12.4f} {pct:>9.1f}%")
        print(f"  {'TOTAL (sum of blocks)':<32s} {total_ms:>12.4f}")


def sync_time(device, fn, N=50, warmup=5):
    """Run fn N times with sync, return mean ms."""
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / N


def build_state(device):
    """Build random fp32 weights + state in correct shapes/dtypes, upload."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
    # input_layernorm
    in_ln = np.ones(HIDDEN, dtype=np.float32)
    # in_proj_all: [HIDDEN, CONV_DIM + VAL_DIM + 2*N_V_HEADS]
    out_concat = CONV_DIM + VAL_DIM + 2 * N_V_HEADS
    in_proj_all = rng.standard_normal((HIDDEN, out_concat)).astype(np.float32) / np.sqrt(HIDDEN)
    # conv1d_weight: [CONV_DIM, KERNEL]
    conv_w = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3
    # dt_bias: [N_V_HEADS]
    dt_bias = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.1
    # A_log: [N_V_HEADS]
    A_log = rng.standard_normal((N_V_HEADS,)).astype(np.float32) * 0.5
    # linear_attn_norm: [V_DIM]
    norm_w = np.ones(V_DIM, dtype=np.float32)
    # out_proj: [VAL_DIM, HIDDEN]
    out_proj_w = rng.standard_normal((VAL_DIM, HIDDEN)).astype(np.float32) / np.sqrt(VAL_DIM)

    # SSM state: [N_V_HEADS, K_DIM, V_DIM]
    ssm_state = rng.standard_normal((N_V_HEADS, K_DIM, V_DIM)).astype(np.float32) * 0.01
    # Conv state: [CONV_DIM, KERNEL-1] (we have a 3-tap conv, state is past 2 tokens)
    conv_state = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.01

    def up(arr, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                               device=device, layout=layout)

    state = {
        'x':         up(x),
        'in_ln':     up(in_ln),
        'in_proj':   up(in_proj_all),
        'conv_w':    up(conv_w),
        'dt_bias':   up(dt_bias),
        'A_log':     up(A_log),
        'norm_w':    up(norm_w),
        'out_proj':  up(out_proj_w),
        'ssm':       up(ssm_state),
        'conv_st':   up(conv_state),
    }
    return state


def make_blocks(device, st):
    """Build per-block closures that run one DeltaNet block each."""
    timer = T()

    # B1: input rms_norm
    def b1():
        return ttnn.rms_norm(st['x'], weight=st['in_ln'], epsilon=EPS)

    h_tt = b1()
    ttnn.synchronize_device(device)

    # B2: big fused in_proj linear
    def b2():
        return ttnn.linear(h_tt, st['in_proj'], compute_kernel_config=hifi4)

    all_tt = b2()
    ttnn.synchronize_device(device)

    # B3: 4× slice (qkv | z | a | b)
    def b3():
        mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM])
        z_tt = ttnn.slice(all_tt, [0, CONV_DIM], [1, CONV_DIM + VAL_DIM])
        a_tt = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM],
                          [1, CONV_DIM + VAL_DIM + N_V_HEADS])
        b_tt = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM + N_V_HEADS],
                          [1, CONV_DIM + VAL_DIM + 2 * N_V_HEADS])
        return mixed_qkv, z_tt, a_tt, b_tt

    mixed_qkv, z_tt, a_tt, b_tt = b3()
    ttnn.synchronize_device(device)

    # B4: conv1d (reshape, concat, mul, sum, silu)
    def b4():
        mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM, 1])
        conv_input = ttnn.concat([st['conv_st'], mixed_col], dim=-1)
        conv_prod = ttnn.mul(conv_input, st['conv_w'])
        conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
        return conv_out

    conv_out = b4()
    ttnn.synchronize_device(device)

    # B5: GQA repeat-interleave for q, k (skip v which is already n_v shape)
    def b5():
        q_flat = ttnn.slice(conv_out, [0], [KEY_DIM])
        k_flat = ttnn.slice(conv_out, [KEY_DIM], [2 * KEY_DIM])

        def gqa_interleave(t_flat, n_kh, d):
            t = ttnn.reshape(t_flat, [n_kh, 1, d])
            t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
            return ttnn.reshape(t, [n_kh * N_REP, d])

        q = gqa_interleave(q_flat, N_K_HEADS, K_DIM)
        k = gqa_interleave(k_flat, N_K_HEADS, K_DIM)
        v_flat = ttnn.slice(conv_out, [2 * KEY_DIM], [CONV_DIM])
        v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])
        return q, k, v

    q, k, v = b5()
    ttnn.synchronize_device(device)

    # B6: Q/K L2 normalize + Q-scaling
    def b6():
        qq = ttnn.mul(q, q)
        q_n = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
        kk = ttnn.mul(k, k)
        k_n = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
        q_n = ttnn.mul(q_n, 1.0 / (K_DIM ** 0.5))
        return q_n, k_n

    q_n, k_n = b6()
    ttnn.synchronize_device(device)

    # B7: gate/decay/beta
    def b7():
        softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, st['dt_bias'])), 1.0))
        g = ttnn.mul(ttnn.neg(ttnn.exp(st['A_log'])), softplus_a)
        beta = ttnn.sigmoid(b_tt)
        decay = ttnn.reshape(ttnn.exp(g), [1, N_V_HEADS, 1, 1])
        return decay, beta

    decay, beta = b7()
    ttnn.synchronize_device(device)

    # B8: recurrence body
    def b8():
        H_4d = ttnn.reshape(st['ssm'], [1, N_V_HEADS, K_DIM, V_DIM])
        H_decayed = ttnn.mul(H_4d, decay)
        k_col = ttnn.reshape(k_n, [1, N_V_HEADS, K_DIM, 1])
        kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                              [1, N_V_HEADS, V_DIM])
        v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
        delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, N_V_HEADS, 1]))
        H_new = ttnn.add(H_decayed,
                         ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
        q_col = ttnn.reshape(q_n, [1, N_V_HEADS, K_DIM, 1])
        out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM])
        return H_new, out

    H_new, out = b8()
    ttnn.synchronize_device(device)

    # B9: per-head rms_norm + silu(z) gate
    def b9():
        out_per_head = ttnn.reshape(out, [N_V_HEADS, V_DIM])
        out_normed = ttnn.rms_norm(out_per_head, weight=st['norm_w'], epsilon=EPS)
        z_per_head = ttnn.reshape(z_tt, [N_V_HEADS, V_DIM])
        silu_z_per_head = ttnn.silu(z_per_head)
        out_gated_per_head = ttnn.mul(out_normed, silu_z_per_head)
        out_gated = ttnn.reshape(out_gated_per_head, [1, VAL_DIM])
        return out_gated

    out_gated = b9()
    ttnn.synchronize_device(device)

    # B10: out_proj linear
    def b10():
        return ttnn.linear(out_gated, st['out_proj'], compute_kernel_config=hifi4)

    out_proj_res = b10()
    ttnn.synchronize_device(device)

    # B11: residual add
    def b11():
        return ttnn.add(st['x'], out_proj_res)

    return {
        'B1_input_rms_norm':       b1,
        'B2_in_proj_fused_linear': b2,
        'B3_4x_slice':             b3,
        'B4_conv1d':               b4,
        'B5_gqa_repeat':           b5,
        'B6_qk_l2_normalize':      b6,
        'B7_gate_decay_beta':      b7,
        'B8_recurrence_body':      b8,
        'B9_perhead_rms_silu':     b9,
        'B10_out_proj_linear':     b10,
        'B11_residual_add':        b11,
    }, timer


def main():
    global hifi4
    print("=" * 78)
    print("DeltaNet per-op latency profile (qb1, single P150)")
    print("=" * 78)
    print(f"Shapes: HIDDEN={HIDDEN}  N_K_HEADS={N_K_HEADS}  N_V_HEADS={N_V_HEADS}")
    print(f"        K_DIM={K_DIM}  V_DIM={V_DIM}  CONV_DIM={CONV_DIM}")

    print("\n[1] Open device 0...")
    device = ttnn.open_device(device_id=0)
    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    print("  ✓ device open")

    try:
        print("\n[2] Build random weights + state...")
        st = build_state(device)
        print("  ✓ state uploaded")

        print("\n[3] Wire up blocks (eager forward to establish state)...")
        blocks, timer = make_blocks(device, st)
        print(f"  ✓ {len(blocks)} blocks built")

        print("\n[4] Time each block (N=50 iters, warmup=5)...")
        for name, fn in blocks.items():
            ms = sync_time(device, fn, N=50, warmup=5)
            timer.add(name, ms)
            print(f"    {name}: {ms:.4f} ms")

        timer.report("DeltaNet per-block latency breakdown")

        # Sanity: what would 48 of these layers cost?
        total = sum(timer.mean(k) for k in timer.totals.keys())
        print(f"\n  Per-token at 48 DeltaNet layers: {total * 48:.1f} ms")
        print(f"  Reference: full-decode 198 ms/tok (traced) - this DeltaNet budget")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
