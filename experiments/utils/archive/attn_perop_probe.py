#!/usr/bin/env python3
"""
Per-op latency profile for Gated Attention step on qb1 (single P150).

Counterpart to deltanet_perop_probe.py. Goal: find where the 16 attention
layers spend their time, so we know what to fuse / replace.

Blocks (in order through gated_attn_step_ondevice):
  B1) input rms_norm
  B2) attn_qkv fused linear (HIDDEN → QG_DIM + 2*KV_DIM = 14336)
  B3) slice + reshape for Q, gate, K, V
  B4) per-head q_norm + k_norm (rms_norm on [N_Q, HEAD_DIM] / [N_KV, HEAD_DIM])
  B5) partial RoPE Level 1 (slice + neg + concat + mul + add)
  B6) update_cache_for_token_ writes for K and V (the new, 7.2× faster path)
  B7) SDPA-decode (non-paged variant — fits at MAX_POS<=256)
  B8) sigmoid gate + multiply
  B9) out_proj linear (N_Q*HEAD_DIM → HIDDEN)
  B10) residual add

Sync caveat: each block timed with explicit synchronize. ~50µs noise floor.

Run:
    ssh qb1 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/attn_perop_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B Gated Attention config
HIDDEN = 5120
N_Q = 24
N_KV = 4
HEAD_DIM = 256
ROTARY_DIM = 64  # partial_rotary_factor = 0.25 × 256 = 64
QG_DIM = 2 * N_Q * HEAD_DIM    # 12288 (Q + gate fused per head)
KV_DIM = N_KV * HEAD_DIM       # 1024
ATTN_QKV_DIM = QG_DIM + 2 * KV_DIM  # 14336
O_DIM = N_Q * HEAD_DIM         # 6144
MAX_POS = 256
EPS = 1e-6
DTYPE = ttnn.bfloat16

hifi4 = None  # set after device open


class T:
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
    x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
    in_ln = np.ones(HIDDEN, dtype=np.float32)
    attn_qkv = rng.standard_normal((HIDDEN, ATTN_QKV_DIM)).astype(np.float32) / np.sqrt(HIDDEN)
    q_norm = np.ones(HEAD_DIM, dtype=np.float32)
    k_norm = np.ones(HEAD_DIM, dtype=np.float32)
    o_proj = rng.standard_normal((O_DIM, HIDDEN)).astype(np.float32) / np.sqrt(O_DIM)

    # Extended cos/sin for Level 1 partial RoPE (identity in passthrough region)
    cos = np.ones(HEAD_DIM, dtype=np.float32)
    sin = np.zeros(HEAD_DIM, dtype=np.float32)
    angles = np.linspace(0, np.pi / 4, ROTARY_DIM // 2, dtype=np.float32)
    cos[:ROTARY_DIM // 2] = np.cos(angles)
    cos[ROTARY_DIM // 2:ROTARY_DIM] = np.cos(angles)
    sin[:ROTARY_DIM // 2] = np.sin(angles)
    sin[ROTARY_DIM // 2:ROTARY_DIM] = np.sin(angles)
    cos = cos.reshape(1, HEAD_DIM)
    sin = sin.reshape(1, HEAD_DIM)

    # KV cache (N_KV=4, MAX_POS=256, HEAD_DIM=256) — standard shape
    # ttnn expects [1, N_KV, MAX_POS, HEAD_DIM] for SDPA-decode
    k_cache = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05
    v_cache = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05

    def up(arr, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                               device=device, layout=layout)

    state = {
        'x':        up(x),
        'in_ln':    up(in_ln),
        'qkv':      up(attn_qkv),
        'q_norm':   up(q_norm),
        'k_norm':   up(k_norm),
        'o_proj':   up(o_proj),
        'cos':      up(cos),
        'sin':      up(sin),
        'kc':       up(k_cache),
        'vc':       up(v_cache),
    }
    return state


def make_blocks(device, st):
    timer = T()

    # B1: input rms_norm
    def b1():
        return ttnn.rms_norm(st['x'], weight=st['in_ln'], epsilon=EPS)

    h_tt = b1()
    ttnn.synchronize_device(device)

    # B2: attn_qkv fused linear
    def b2():
        return ttnn.linear(h_tt, st['qkv'], compute_kernel_config=hifi4)

    all_tt = b2()
    ttnn.synchronize_device(device)

    # B3: slice + reshape for q, gate, k, v
    def b3():
        qg_flat = ttnn.slice(all_tt, [0, 0], [1, QG_DIM])
        k_flat = ttnn.slice(all_tt, [0, QG_DIM], [1, QG_DIM + KV_DIM])
        v_flat = ttnn.slice(all_tt, [0, QG_DIM + KV_DIM], [1, QG_DIM + 2 * KV_DIM])
        qg_tt = ttnn.reshape(qg_flat, [N_Q, HEAD_DIM * 2])
        q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
        gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])
        k_tt = ttnn.reshape(k_flat, [N_KV, HEAD_DIM])
        v_tt = ttnn.reshape(v_flat, [N_KV, HEAD_DIM])
        return q_tt, gate_tt, k_tt, v_tt

    q_tt, gate_tt, k_tt, v_tt = b3()
    ttnn.synchronize_device(device)

    # B4: per-head q_norm + k_norm
    def b4():
        q_n = ttnn.rms_norm(q_tt, weight=st['q_norm'], epsilon=EPS)
        k_n = ttnn.rms_norm(k_tt, weight=st['k_norm'], epsilon=EPS)
        return q_n, k_n

    q_n, k_n = b4()
    ttnn.synchronize_device(device)

    # B5: partial RoPE Level 1 (slice + neg + concat + mul + add)
    def b5_apply(t, n_heads):
        half = ROTARY_DIM // 2
        x1 = ttnn.slice(t, [0, 0], [n_heads, half])
        x2 = ttnn.slice(t, [0, half], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
        return ttnn.add(ttnn.mul(t, st['cos']), ttnn.mul(rotated_full, st['sin']))

    def b5():
        q_r = b5_apply(q_n, N_Q)
        k_r = b5_apply(k_n, N_KV)
        return q_r, k_r

    q_r, k_r = b5()
    ttnn.synchronize_device(device)

    # B6: update_cache_for_token_ writes (K + V)
    # Cache expects [1, N_KV, MAX_POS, HEAD_DIM]; src for write is [1, N_KV, 1, HEAD_DIM]
    def b6():
        k_for_cache = ttnn.typecast(ttnn.reshape(k_r, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
        v_for_cache = ttnn.typecast(ttnn.reshape(v_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
        ttnn.kv_cache.update_cache_for_token_(st['kc'], k_for_cache, 0)
        ttnn.kv_cache.update_cache_for_token_(st['vc'], v_for_cache, 0)
        return None

    b6()
    ttnn.synchronize_device(device)

    # B7: SDPA-decode (non-paged) — fits at MAX_POS<=256 per memory note
    # Q shape for SDPA: [1, 1, N_Q, HEAD_DIM]
    def b7():
        q_for_sdpa = ttnn.reshape(q_r, [1, 1, N_Q, HEAD_DIM])
        # cur_pos = MAX_POS - 1 means we have a full cache
        try:
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_for_sdpa, st['kc'], st['vc'],
                cur_pos=[MAX_POS - 1],
                scale=1.0 / (HEAD_DIM ** 0.5),
            )
        except Exception as e:
            # Some installs prefer cur_pos_tensor; fall back
            cur_pos_tt = ttnn.from_torch(
                torch.tensor([MAX_POS - 1], dtype=torch.int32),
                device=device,
            )
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                q_for_sdpa, st['kc'], st['vc'],
                cur_pos_tensor=cur_pos_tt,
                scale=1.0 / (HEAD_DIM ** 0.5),
            )
        return out

    sdpa_out = b7()
    ttnn.synchronize_device(device)

    # B8: sigmoid gate + multiply (gate is [N_Q, HEAD_DIM], attn is [1, 1, N_Q, HEAD_DIM])
    def b8():
        attn = ttnn.reshape(sdpa_out, [N_Q, HEAD_DIM])
        sig = ttnn.sigmoid(gate_tt)
        return ttnn.mul(attn, sig)

    gated = b8()
    ttnn.synchronize_device(device)

    # B9: out_proj linear
    def b9():
        gated_flat = ttnn.reshape(gated, [1, O_DIM])
        return ttnn.linear(gated_flat, st['o_proj'], compute_kernel_config=hifi4)

    proj_out = b9()
    ttnn.synchronize_device(device)

    # B10: residual add
    def b10():
        return ttnn.add(st['x'], proj_out)

    return {
        'B1_input_rms_norm':       b1,
        'B2_attn_qkv_linear':      b2,
        'B3_slice_reshape':        b3,
        'B4_qk_per_head_rmsnorm':  b4,
        'B5_partial_rope_level1':  b5,
        'B6_update_cache_k_v':     b6,
        'B7_sdpa_decode':          b7,
        'B8_sigmoid_gate_mul':     b8,
        'B9_out_proj_linear':      b9,
        'B10_residual_add':        b10,
    }, timer


def main():
    global hifi4
    print("=" * 78)
    print("Gated Attention per-op latency profile (qb1, single P150)")
    print("=" * 78)
    print(f"Shapes: HIDDEN={HIDDEN}  N_Q={N_Q}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}")
    print(f"        QG_DIM={QG_DIM}  KV_DIM={KV_DIM}  ATTN_QKV_DIM={ATTN_QKV_DIM}")
    print(f"        MAX_POS={MAX_POS}  ROTARY_DIM={ROTARY_DIM}")

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
            try:
                ms = sync_time(device, fn, N=50, warmup=5)
                timer.add(name, ms)
                print(f"    {name}: {ms:.4f} ms")
            except Exception as e:
                print(f"    {name}: ✗ FAILED: {type(e).__name__}: {str(e)[:160]}")

        timer.report("Gated Attention per-block latency breakdown")

        total = sum(timer.mean(k) for k in timer.totals.keys())
        print(f"\n  Per-token at 16 Gated Attention layers: {total * 16:.1f} ms")
        print(f"  Reference: full-decode 198 ms/tok (traced)")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
