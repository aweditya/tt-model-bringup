#!/usr/bin/env python3
"""
Integration probe: V2 rotate-only RoPE inside full gated_attn_step on qb1.

Goal: verify that swapping V1 Level 1 → V2 rotate-only in production
attention produces (a) bit-equivalent math and (b) measurable latency
savings at the gated_attn_step granularity (not just isolated RoPE).

This is the integration test before we modify 91f.py's apply_partial_rope.

Method:
  - Build random weights + state matching production gated_attn_step shape
  - Run full attn step with V1 RoPE → time
  - Run full attn step with V2 RoPE → time
  - Verify cos(V1_out, V2_out) ≈ 1.0 (math identical)
  - Report delta at gated_attn_step level × 16 layers

Run:
    ssh qb1 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/attn_step_rope_swap_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


HIDDEN = 5120
N_Q = 24
N_KV = 4
HEAD_DIM = 256
ROTARY_DIM = 64
QG_DIM = 2 * N_Q * HEAD_DIM
KV_DIM = N_KV * HEAD_DIM
ATTN_QKV_DIM = QG_DIM + 2 * KV_DIM
O_DIM = N_Q * HEAD_DIM
MAX_POS = 256
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


def apply_rope_v1_level1(t, cos_ext, sin_ext, n_heads):
    """Current production: extended cos/sin spanning HEAD_DIM, identity in passthru."""
    half = ROTARY_DIM // 2
    x1 = ttnn.slice(t, [0, 0], [n_heads, half])
    x2 = ttnn.slice(t, [0, half], [n_heads, ROTARY_DIM])
    passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
    return ttnn.add(ttnn.mul(t, cos_ext), ttnn.mul(rotated_full, sin_ext))


def apply_rope_v2_rotate_only(t, cos_rot, sin_rot, n_heads):
    """Rotate only the ROTARY_DIM region, concat with passthrough."""
    half = ROTARY_DIM // 2
    rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
    passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
    x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
    x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
    neg_x2 = ttnn.neg(x2)
    rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
    rotated = ttnn.add(ttnn.mul(rot, cos_rot), ttnn.mul(rotated_half, sin_rot))
    return ttnn.concat([rotated, passthru], dim=-1)


def gated_attn_step(x_tt, w_tt, kc_tt, vc_tt, cos_tt, sin_tt, rope_version):
    """Full gated attention step. rope_version='v1' or 'v2'."""
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['in_ln'], epsilon=EPS)
    all_tt = ttnn.linear(h_tt, w_tt['qkv'], compute_kernel_config=hifi4)
    qg_flat = ttnn.slice(all_tt, [0, 0], [1, QG_DIM])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM], [1, QG_DIM + KV_DIM])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM + KV_DIM], [1, QG_DIM + 2 * KV_DIM])
    qg_tt = ttnn.reshape(qg_flat, [N_Q, HEAD_DIM * 2])
    q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
    gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_flat, [N_KV, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [N_KV, HEAD_DIM])

    q_tt = ttnn.rms_norm(q_tt, weight=w_tt['q_norm'], epsilon=EPS)
    k_tt = ttnn.rms_norm(k_tt, weight=w_tt['k_norm'], epsilon=EPS)

    if rope_version == 'v1':
        q_tt = apply_rope_v1_level1(q_tt, cos_tt['ext'], sin_tt['ext'], N_Q)
        k_tt = apply_rope_v1_level1(k_tt, cos_tt['ext'], sin_tt['ext'], N_KV)
    else:
        q_tt = apply_rope_v2_rotate_only(q_tt, cos_tt['rot'], sin_tt['rot'], N_Q)
        k_tt = apply_rope_v2_rotate_only(k_tt, cos_tt['rot'], sin_tt['rot'], N_KV)

    # KV cache write
    k_for_cache = ttnn.typecast(ttnn.reshape(k_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    v_for_cache = ttnn.typecast(ttnn.reshape(v_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    ttnn.kv_cache.update_cache_for_token_(kc_tt, k_for_cache, 0)
    ttnn.kv_cache.update_cache_for_token_(vc_tt, v_for_cache, 0)

    # SDPA-decode
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, N_Q, HEAD_DIM])
    sdpa_out = ttnn.transformer.scaled_dot_product_attention_decode(
        q_for_sdpa, kc_tt, vc_tt,
        cur_pos=[MAX_POS - 1],
        scale=1.0 / (HEAD_DIM ** 0.5),
    )
    attn = ttnn.reshape(sdpa_out, [N_Q, HEAD_DIM])

    # Sigmoid gate + multiply
    sig = ttnn.sigmoid(gate_tt)
    attn = ttnn.mul(attn, sig)

    # out_proj
    attn_flat = ttnn.reshape(attn, [1, O_DIM])
    proj_out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, proj_out)


def main():
    global hifi4
    print("=" * 78)
    print("V2 RoPE swap integration probe (qb1, full gated_attn_step)")
    print("=" * 78)
    print(f"HIDDEN={HIDDEN}  N_Q={N_Q}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}  ROTARY_DIM={ROTARY_DIM}")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    try:
        print("[2] Build random state...")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
        in_ln = np.ones(HIDDEN, dtype=np.float32)
        qkv = rng.standard_normal((HIDDEN, ATTN_QKV_DIM)).astype(np.float32) / np.sqrt(HIDDEN)
        q_norm = np.ones(HEAD_DIM, dtype=np.float32)
        k_norm = np.ones(HEAD_DIM, dtype=np.float32)
        o_proj = rng.standard_normal((O_DIM, HIDDEN)).astype(np.float32) / np.sqrt(O_DIM)
        kc = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05
        vc = rng.standard_normal((1, N_KV, MAX_POS, HEAD_DIM)).astype(np.float32) * 0.05

        angles = np.linspace(0, np.pi / 4, ROTARY_DIM // 2, dtype=np.float32)
        cos_rot = np.zeros(ROTARY_DIM, dtype=np.float32)
        sin_rot = np.zeros(ROTARY_DIM, dtype=np.float32)
        cos_rot[:ROTARY_DIM // 2] = np.cos(angles)
        cos_rot[ROTARY_DIM // 2:] = np.cos(angles)
        sin_rot[:ROTARY_DIM // 2] = np.sin(angles)
        sin_rot[ROTARY_DIM // 2:] = np.sin(angles)
        cos_ext = np.ones(HEAD_DIM, dtype=np.float32)
        sin_ext = np.zeros(HEAD_DIM, dtype=np.float32)
        cos_ext[:ROTARY_DIM] = cos_rot
        sin_ext[:ROTARY_DIM] = sin_rot

        def up(arr):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                    device=device, layout=ttnn.TILE_LAYOUT)
        w_tt = {
            'in_ln': up(in_ln), 'qkv': up(qkv),
            'q_norm': up(q_norm), 'k_norm': up(k_norm),
            'o_proj': up(o_proj),
        }
        x_tt = up(x)
        kc_tt = up(kc)
        vc_tt = up(vc)
        cos_tt = {'ext': up(cos_ext.reshape(1, HEAD_DIM)),
                  'rot': up(cos_rot.reshape(1, ROTARY_DIM))}
        sin_tt = {'ext': up(sin_ext.reshape(1, HEAD_DIM)),
                  'rot': up(sin_rot.reshape(1, ROTARY_DIM))}

        # Two separate caches for V1 and V2 runs (cache state changes with update)
        kc_v1 = up(kc.copy())
        vc_v1 = up(vc.copy())
        kc_v2 = up(kc.copy())
        vc_v2 = up(vc.copy())

        print("[3] Math sanity (cosine between V1 and V2 outputs)...")
        out_v1 = ttnn.to_torch(gated_attn_step(x_tt, w_tt, kc_v1, vc_v1, cos_tt, sin_tt, 'v1'))
        out_v2 = ttnn.to_torch(gated_attn_step(x_tt, w_tt, kc_v2, vc_v2, cos_tt, sin_tt, 'v2'))
        v1_flat = out_v1.float().cpu().numpy().flatten()
        v2_flat = out_v2.float().cpu().numpy().flatten()
        cos_12 = float(v1_flat @ v2_flat / (np.linalg.norm(v1_flat) * np.linalg.norm(v2_flat) + 1e-12))
        max_diff = float(np.abs(v1_flat - v2_flat).max())
        print(f"  cos(V1 out, V2 out) = {cos_12:.6f}  max|Δ| = {max_diff:.4e}")

        print("\n[4] Latency benchmark — full gated_attn_step (N=50, warmup=5)...")
        # Fresh caches each rep so update_cache writes don't accumulate
        ms_v1 = sync_time(device,
            lambda: gated_attn_step(x_tt, w_tt, kc_tt, vc_tt, cos_tt, sin_tt, 'v1'),
            N=50, warmup=5)
        ms_v2 = sync_time(device,
            lambda: gated_attn_step(x_tt, w_tt, kc_tt, vc_tt, cos_tt, sin_tt, 'v2'),
            N=50, warmup=5)

        print(f"  V1 Level 1 step:    {ms_v1:.4f} ms")
        print(f"  V2 rotate-only step:{ms_v2:.4f} ms")
        savings = ms_v1 - ms_v2
        pct = savings / ms_v1 * 100
        print(f"  Savings: {savings:+.4f} ms/layer ({pct:+.1f}%)")
        print(f"  Per-token at 16 attn layers: V1 = {ms_v1 * 16:.1f} ms, V2 = {ms_v2 * 16:.1f} ms")
        print(f"  Per-token delta: {savings * 16:+.2f} ms/tok")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        math_ok = cos_12 >= 0.9999
        perf_win = ms_v2 < ms_v1
        ok = math_ok and perf_win
        print(f"  math identical: cos={cos_12:.6f} {'✓' if math_ok else '✗'}")
        print(f"  V2 faster:      {ms_v2:.4f} < {ms_v1:.4f} {'✓' if perf_win else '✗'}")
        print(f"  Result: {'✓ READY for production swap' if ok else '✗ HOLD — re-investigate'}")
    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
