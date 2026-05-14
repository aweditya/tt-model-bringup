#!/usr/bin/env python3
"""
Attention TP traced latency probe — the missing piece for honest multi-chip projection.

So far measured:
  - DN+MLP block traced @ real shapes: 1.20 ms/block (linear up to K=8)
  - Attn TP correctness (C'7.3): cos 0.999937, no latency measurement in trace

Question: what's the per-block traced latency of a GATED ATTENTION TP block?
Need this to fairly project full Qwen3.6-27B multi-chip ms/tok (16 attn + 48 DN).

This probe:
  - Builds 4 random attention-block weights (real Qwen3.6 attn shapes)
  - Each "block" = full Gated Attention forward (in_proj sharded + SDPA per-chip + out_proj all_reduce)
  - Chains them inside a single trace
  - Measures execute_trace latency
  - Per-block traced ms = the missing number

Real Qwen3.6-27B gated attn shapes:
  N_Q=24, N_KV=4, HEAD_DIM=256, HIDDEN=5120
  N_REP=6, MAX_POS=32 (probe-friendly)
  Per chip: 6 Q heads, 1 KV head

Simplifies for probe scope:
  - Skips RMSNorm (input + per-head q_norm/k_norm)
  - Skips RoPE (V2 rotate-only path is fast at our shapes)
  - Uses simple Q@K^T softmax V attention (not paged_SDPA) since this is plumbing/perf
  - Uses NEW KV state each iteration (fresh; no cache mutation between iters)

Output: real ms per attn TP block, traced. Combined with DN+MLP 1.20 ms,
gives honest full-model projection (still missing per-tok overhead).
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Qwen3.6-27B Gated Attention config (real)
HIDDEN = 5120
N_Q = 24
N_KV = 4
HEAD_DIM = 256
N_REP = N_Q // N_KV         # 6
QG_DIM = 2 * N_Q * HEAD_DIM # 12288
KV_DIM = N_KV * HEAD_DIM    # 1024
ATTN_QKV_DIM = QG_DIM + 2 * KV_DIM  # 14336
O_DIM = N_Q * HEAD_DIM      # 6144
MAX_POS = 32
NCHIPS = 4
NQ_PER_CHIP = N_Q // NCHIPS
NKV_PER_CHIP = N_KV // NCHIPS
QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM
KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
ATTN_QKV_DIM_CHIP = QG_DIM_CHIP + 2 * KV_DIM_CHIP
O_DIM_CHIP = NQ_PER_CHIP * HEAD_DIM


def upload_attn_layer(mesh, w_qkv_sharded_np, w_o_sharded_np, k_cache_chip, v_cache_chip):
    """Upload sharded attn weights + per-chip KV cache."""
    return {
        'w_qkv':   ttnn.from_torch(torch.from_numpy(w_qkv_sharded_np),
                                    dtype=ttnn.bfloat16, device=mesh,
                                    layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1)),
        'w_o':     ttnn.from_torch(torch.from_numpy(w_o_sharded_np),
                                    dtype=ttnn.bfloat16, device=mesh,
                                    layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)),
        'kc':      ttnn.from_torch(torch.from_numpy(k_cache_chip),
                                    dtype=ttnn.bfloat16, device=mesh,
                                    layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)),
        'vc':      ttnn.from_torch(torch.from_numpy(v_cache_chip),
                                    dtype=ttnn.bfloat16, device=mesh,
                                    layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)),
    }


def relayout_attn_qkv(w_attn_qkv, n_q_per_chip, nkv_per_chip):
    """Re-arrange [HIDDEN, QG | K | V] into per-chip [Q+gate (heads i) | K (head i) | V (head i)]."""
    qg_full = w_attn_qkv[:, :QG_DIM].reshape(HIDDEN, N_Q, 2 * HEAD_DIM)
    k_full  = w_attn_qkv[:, QG_DIM:QG_DIM + KV_DIM].reshape(HIDDEN, N_KV, HEAD_DIM)
    v_full  = w_attn_qkv[:, QG_DIM + KV_DIM:].reshape(HIDDEN, N_KV, HEAD_DIM)
    chip_slabs = []
    for chip_i in range(NCHIPS):
        qg_slab = qg_full[:, chip_i * n_q_per_chip:(chip_i + 1) * n_q_per_chip, :]
        qg_slab = qg_slab.reshape(HIDDEN, n_q_per_chip * 2 * HEAD_DIM)
        k_slab  = k_full[:, chip_i * nkv_per_chip:(chip_i + 1) * nkv_per_chip, :]
        k_slab  = k_slab.reshape(HIDDEN, nkv_per_chip * HEAD_DIM)
        v_slab  = v_full[:, chip_i * nkv_per_chip:(chip_i + 1) * nkv_per_chip, :]
        v_slab  = v_slab.reshape(HIDDEN, nkv_per_chip * HEAD_DIM)
        chip_slabs.append(np.concatenate([qg_slab, k_slab, v_slab], axis=1))
    return np.concatenate(chip_slabs, axis=1)


def relayout_o(w_o):
    """Re-arrange [N_Q*HEAD_DIM, HIDDEN] for row-parallel sharding by head group."""
    w_o_reshape = w_o.reshape(N_Q, HEAD_DIM, HIDDEN)
    chunks = []
    for chip_i in range(NCHIPS):
        wo_chip = w_o_reshape[chip_i * NQ_PER_CHIP:(chip_i + 1) * NQ_PER_CHIP]
        chunks.append(wo_chip.reshape(O_DIM_CHIP, HIDDEN))
    return np.concatenate(chunks, axis=0)


def build_attn_block_random(rng):
    return {
        'x':      rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.5,
        'w_qkv':  rng.standard_normal((HIDDEN, ATTN_QKV_DIM)).astype(np.float32) / np.sqrt(HIDDEN),
        'w_o':    rng.standard_normal((O_DIM, HIDDEN)).astype(np.float32) / np.sqrt(O_DIM),
        'k_cache': rng.standard_normal((MAX_POS, N_KV, HEAD_DIM)).astype(np.float32) * 0.05,
        'v_cache': rng.standard_normal((MAX_POS, N_KV, HEAD_DIM)).astype(np.float32) * 0.05,
    }


def attn_tp_forward(mesh, x_tt, sharded):
    """One Gated Attention TP forward (head-sharded). Returns [1, HIDDEN] residual (replicated)."""
    all_tt = ttnn.linear(x_tt, sharded['w_qkv'])
    qg_tt = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
    k_new = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
    v_new = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                       [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])

    qg = ttnn.reshape(qg_tt, [NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])

    # Per-chip cache has NKV_PER_CHIP=1 KV head; treat as [MAX_POS, HEAD_DIM]
    assert NKV_PER_CHIP == 1
    k_full = ttnn.reshape(sharded['kc'], [MAX_POS, HEAD_DIM])
    v_full = ttnn.reshape(sharded['vc'], [MAX_POS, HEAD_DIM])

    # Simple SDPA: Q @ K^T -> softmax -> @ V
    scale = 1.0 / np.sqrt(HEAD_DIM)
    kT = ttnn.transpose(k_full, 0, 1)
    scores = ttnn.mul(ttnn.matmul(q_tt, kT), scale)
    attn_w = ttnn.softmax(scores, dim=-1)
    attn = ttnn.matmul(attn_w, v_full)

    # Sigmoid gate + mul
    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))

    # out_proj row-parallel
    attn_flat = ttnn.reshape(attn, [1, O_DIM_CHIP])
    partial = ttnn.linear(attn_flat, sharded['w_o'])

    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    return ttnn.add(x_tt, reduced)


def main():
    print("=" * 78)
    print("Gated Attention TP traced latency probe (qb2)")
    print("=" * 78)
    print(f"N_Q={N_Q}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}  MAX_POS={MAX_POS}")
    print(f"Per chip: {NQ_PER_CHIP} Q heads, {NKV_PER_CHIP} KV head")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {mesh.get_num_devices()} chips")

    try:
        # Build K=4 different attn blocks
        K_MAX = 4
        rng = np.random.default_rng(42)
        blocks = [build_attn_block_random(rng) for _ in range(K_MAX)]
        sharded_blocks = []
        for b in blocks:
            w_qkv_sh = relayout_attn_qkv(b['w_qkv'], NQ_PER_CHIP, NKV_PER_CHIP)
            w_o_sh = relayout_o(b['w_o'])
            k_cache_chip = b['k_cache'].transpose(1, 0, 2)  # [N_KV, MAX_POS, HEAD_DIM]
            v_cache_chip = b['v_cache'].transpose(1, 0, 2)
            sharded_blocks.append(upload_attn_layer(mesh, w_qkv_sh, w_o_sh, k_cache_chip, v_cache_chip))

        x = blocks[0]['x']
        x_tt = ttnn.from_torch(torch.from_numpy(x), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def forward_K(x_in, K):
            cur = x_in
            for j in range(K):
                cur = attn_tp_forward(mesh, cur, sharded_blocks[j])
            return cur

        # Warmup
        print("\nWarmup (3 iters, K=1)...")
        for _ in range(3):
            _ = forward_K(x_tt, 1)
        ttnn.synchronize_device(mesh)
        for _ in range(2):
            _ = forward_K(x_tt, 4)
        ttnn.synchronize_device(mesh)
        print("  ✓ warmup")

        results = []
        for K in [1, 2, 4]:
            print(f"\n[K={K}] measure eager + traced")
            N = max(5, 20 // K)
            t0 = time.perf_counter()
            for _ in range(N):
                _ = forward_K(x_tt, K)
            ttnn.synchronize_device(mesh)
            eager_ms = (time.perf_counter() - t0) * 1000.0 / N

            try:
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                _ = forward_K(x_tt, K)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            except Exception as e:
                print(f"  ✗ trace K={K}: {type(e).__name__}: {str(e)[:300]}")
                results.append((K, eager_ms, None))
                continue

            for _ in range(3):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            traced_ms = (time.perf_counter() - t0) * 1000.0 / 20

            try:
                ttnn.release_trace(mesh, trace_id)
            except Exception as e:
                pass

            speedup = eager_ms / traced_ms if traced_ms > 0 else float('nan')
            per_block_traced = traced_ms / K
            print(f"  K={K}: eager={eager_ms:.3f} ms, traced={traced_ms:.3f} ms, "
                  f"speedup={speedup:.2f}×, per-block={per_block_traced:.3f}")
            results.append((K, eager_ms, traced_ms))

        print("\n" + "=" * 78)
        print("ATTN TP TRACED RESULTS")
        print("=" * 78)
        print(f"{'K':>4s} {'eager (ms)':>12s} {'traced (ms)':>12s} {'speedup':>10s} {'traced/K':>10s}")
        for K, eager_ms, traced_ms in results:
            if traced_ms is None:
                print(f"{K:>4d} {eager_ms:>12.3f} {'FAILED':>12s} {'-':>10s} {'-':>10s}")
            else:
                sp = eager_ms / traced_ms if traced_ms > 0 else float('nan')
                print(f"{K:>4d} {eager_ms:>12.3f} {traced_ms:>12.3f} {sp:>10.2f} {traced_ms/K:>10.3f}")

        valid = [(K, t) for K, _, t in results if t is not None]
        if valid:
            per_block_attn = valid[-1][1] / valid[-1][0]
            print(f"\nPer-block attn TP traced (real measurement): {per_block_attn:.3f} ms")
            print(f"\nCombined model forward extrapolation (still NOT a tok/s):")
            print(f"  48 DN+MLP × 1.20 ms = {48 * 1.20:.1f} ms")
            print(f"  16 attn × {per_block_attn:.2f} ms     = {16 * per_block_attn:.1f} ms")
            total_forward = 48 * 1.20 + 16 * per_block_attn
            print(f"  Forward total: {total_forward:.1f} ms")
            print(f"\nMISSING for real tok/s claim: embedding lookup, lm_head matmul,")
            print(f"  argmax sampling, KV cache writes during decode. Real number needs C'7.8.")
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception as e:
            print(f"close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as e:
            print(f"fabric reset error: {e}")


if __name__ == "__main__":
    main()
