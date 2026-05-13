#!/usr/bin/env python3
"""
C'7.3: Gated Attention tensor-parallel correctness probe on qb2 (4× P150).

Tests the head-sharded TP pattern for attention:
  - in_proj: column-parallel (each chip computes its 6 Q + 1 KV head slice)
  - SDPA: purely local per-chip (heads sharded → no cross-chip comm in attention)
  - out_proj: row-parallel + all_reduce (each chip emits partial; sum reconstructs)

Pattern (canonical for all 16 Gated Attention layers in Qwen3.6-27B):
  N_Q = 24, N_KV = 4, HEAD_DIM = 256, N_REP = 6
  4 chips: N_Q/4 = 6 Q heads/chip; N_KV/4 = 1 KV head/chip
  GQA mapping: q heads i*6..(i+1)*6 all attend to KV head i → each chip self-contained

Math:
  Single-chip reference: numpy fp32 attention(Q, K, V) → output, then o_proj
  TP version:           per-chip attention(Q_i, K_i, V_i) → partial o_proj
                        → all_reduce → output

We use simple QK^T-softmax-V attention (not paged_SDPA_decode) for this probe
because:
  - Goal is to validate TP plumbing, not re-validate SDPA math
  - paged_SDPA + cache state adds complexity orthogonal to head sharding
  - In production the SDPA stays purely local on each chip — the TP pattern
    is identical regardless of which SDPA variant we use

This probe validates the sharding + collective. Production wiring uses
paged_scaled_dot_product_attention_decode with per-chip KV caches.
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
N_REP = N_Q // N_KV          # 6
QG_DIM = 2 * N_Q * HEAD_DIM  # 12288 (Q + gate per head)
KV_DIM = N_KV * HEAD_DIM     # 1024
ATTN_QKV_DIM = QG_DIM + 2 * KV_DIM  # 14336
O_DIM_FULL = N_Q * HEAD_DIM  # 6144
MAX_POS = 32  # short history for probe — keeps numbers tractable
NCHIPS = 4
NQ_PER_CHIP = N_Q // NCHIPS   # 6
NKV_PER_CHIP = N_KV // NCHIPS  # 1
QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM  # 3072
KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM     # 256
ATTN_QKV_DIM_CHIP = QG_DIM_CHIP + 2 * KV_DIM_CHIP  # 3584
O_DIM_CHIP = NQ_PER_CHIP * HEAD_DIM        # 1536

assert N_Q % NCHIPS == 0
assert N_KV % NCHIPS == 0
assert ATTN_QKV_DIM_CHIP * NCHIPS == ATTN_QKV_DIM, (
    f"attn_qkv shard layout mismatch: {ATTN_QKV_DIM_CHIP} * {NCHIPS} != {ATTN_QKV_DIM}")


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def softmax_np(x, axis=-1):
    x_shift = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x_shift)
    return e / e.sum(axis=axis, keepdims=True)


def attention_np(q, k, v):
    """
    Pure numpy fp32 attention.
      q: [N_Q, HEAD_DIM]
      k: [MAX_POS, N_KV, HEAD_DIM]
      v: [MAX_POS, N_KV, HEAD_DIM]
    Returns: [N_Q, HEAD_DIM]
    """
    # GQA: each Q head attends to its corresponding KV head (head_idx // N_REP)
    out = np.zeros((N_Q, HEAD_DIM), dtype=np.float32)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for qi in range(N_Q):
        kv_idx = qi // N_REP
        k_head = k[:, kv_idx, :]  # [MAX_POS, HEAD_DIM]
        v_head = v[:, kv_idx, :]  # [MAX_POS, HEAD_DIM]
        scores = (q[qi] @ k_head.T) * scale  # [MAX_POS]
        attn = softmax_np(scores)
        out[qi] = attn @ v_head
    return out


def single_chip_np_forward(x, w_attn_qkv, w_o, k_cache, v_cache, w_gate=None):
    """
    Reference forward, all in numpy fp32.
    Layout matches production attn_qkv: [Q+gate (12288) | K (1024) | V (1024)].

    Returns: residual output [1, HIDDEN]
    """
    h = x  # skip RMSNorm for probe — we're testing TP plumbing, not norm
    all_p = h @ w_attn_qkv  # [1, ATTN_QKV_DIM]
    qg = all_p[:, :QG_DIM].reshape(N_Q, 2 * HEAD_DIM)
    q = qg[:, :HEAD_DIM]              # [N_Q, HEAD_DIM]
    gate = qg[:, HEAD_DIM:]           # [N_Q, HEAD_DIM] (per-head sigmoid gate)
    k = all_p[:, QG_DIM:QG_DIM + KV_DIM].reshape(N_KV, HEAD_DIM)
    v = all_p[:, QG_DIM + KV_DIM:].reshape(N_KV, HEAD_DIM)

    # Append k, v as the latest token in cache (position 0 for this probe)
    k_cache = k_cache.copy()
    v_cache = v_cache.copy()
    k_cache[0] = k
    v_cache[0] = v

    attn = attention_np(q, k_cache, v_cache)  # [N_Q, HEAD_DIM]
    # Sigmoid gate
    sig = 1.0 / (1.0 + np.exp(-gate))
    attn_gated = attn * sig

    # o_proj: [N_Q * HEAD_DIM, HIDDEN]
    attn_flat = attn_gated.reshape(1, O_DIM_FULL)
    o = attn_flat @ w_o  # [1, HIDDEN]
    return o


def tp_forward(mesh, x, w_attn_qkv, w_o, k_cache, v_cache):
    """
    4-chip TP attention. Returns stacked output [4, 1, HIDDEN]; each row should
    be identical post all-reduce.

    Sharding:
      w_attn_qkv: column-parallel split by (Q_HEAD, K_HEAD, V_HEAD) groups
                  — but a contiguous slice of attn_qkv is mixed, so we must
                  re-layout into per-chip QKV layout BEFORE sharding.
      w_o:        row-parallel split along input dim (head dim)
    """
    # Re-layout w_attn_qkv to per-chip [HIDDEN, ATTN_QKV_DIM_CHIP * NCHIPS] in chip-grouped order.
    # Production layout: [Q+gate (all 24 heads) | K (all 4 heads) | V (all 4 heads)]
    # Per-chip layout we want:
    #   chip0: [Q+gate heads 0..5 (6 heads, 3072 cols) | K head 0 (256) | V head 0 (256)] = 3584
    #   chip1: [Q+gate heads 6..11               | K head 1            | V head 1] = 3584
    #   ...
    qg_full = w_attn_qkv[:, :QG_DIM].reshape(HIDDEN, N_Q, 2 * HEAD_DIM)
    k_full  = w_attn_qkv[:, QG_DIM:QG_DIM + KV_DIM].reshape(HIDDEN, N_KV, HEAD_DIM)
    v_full  = w_attn_qkv[:, QG_DIM + KV_DIM:].reshape(HIDDEN, N_KV, HEAD_DIM)

    chip_slabs = []
    for chip_i in range(NCHIPS):
        qg_slab = qg_full[:, chip_i * NQ_PER_CHIP:(chip_i + 1) * NQ_PER_CHIP, :]
        qg_slab = qg_slab.reshape(HIDDEN, NQ_PER_CHIP * 2 * HEAD_DIM)
        k_slab  = k_full[:, chip_i * NKV_PER_CHIP:(chip_i + 1) * NKV_PER_CHIP, :]
        k_slab  = k_slab.reshape(HIDDEN, NKV_PER_CHIP * HEAD_DIM)
        v_slab  = v_full[:, chip_i * NKV_PER_CHIP:(chip_i + 1) * NKV_PER_CHIP, :]
        v_slab  = v_slab.reshape(HIDDEN, NKV_PER_CHIP * HEAD_DIM)
        chip_slabs.append(np.concatenate([qg_slab, k_slab, v_slab], axis=1))
    w_attn_qkv_sharded_np = np.concatenate(chip_slabs, axis=1)  # [HIDDEN, ATTN_QKV_DIM]
    # Now ShardTensorToMesh on dim=1 will split into 4 equal chunks of 3584 cols,
    # giving each chip its slab.

    # w_o re-layout: per-chip slice [O_DIM_CHIP, HIDDEN]
    # Production: [N_Q * HEAD_DIM, HIDDEN]. Re-arranged so chip i gets q heads i*6..(i+1)*6.
    w_o_reshape = w_o.reshape(N_Q, HEAD_DIM, HIDDEN)
    w_o_chunks = []
    for chip_i in range(NCHIPS):
        wo_chip = w_o_reshape[chip_i * NQ_PER_CHIP:(chip_i + 1) * NQ_PER_CHIP]
        w_o_chunks.append(wo_chip.reshape(O_DIM_CHIP, HIDDEN))
    w_o_sharded_np = np.concatenate(w_o_chunks, axis=0)  # [O_DIM_FULL, HIDDEN]
    # ShardTensorToMesh dim=0 gives each chip its [1536, 5120] slab.

    # Per-chip K/V cache (chip i holds KV head i)
    k_cache_full = k_cache.copy()  # [MAX_POS, N_KV, HEAD_DIM]
    v_cache_full = v_cache.copy()
    # Reshape to [MAX_POS, ATTN_QKV_DIM_CHIP * NCHIPS] is awkward; just split per chip
    # and stack along leading axis for ShardTensorToMesh dim=0 with custom layout.
    # Simpler: build a [NCHIPS, MAX_POS, HEAD_DIM] tensor for K and V, shard dim=0
    # giving each chip [1, MAX_POS, HEAD_DIM] = its single KV head.
    k_cache_chip = k_cache_full.transpose(1, 0, 2)  # [N_KV, MAX_POS, HEAD_DIM]
    v_cache_chip = v_cache_full.transpose(1, 0, 2)

    # Upload to mesh
    x_tt = ttnn.from_torch(
        torch.from_numpy(x), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w_qkv_tt = ttnn.from_torch(
        torch.from_numpy(w_attn_qkv_sharded_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    w_o_tt = ttnn.from_torch(
        torch.from_numpy(w_o_sharded_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    k_cache_tt = ttnn.from_torch(
        torch.from_numpy(k_cache_chip), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    v_cache_tt = ttnn.from_torch(
        torch.from_numpy(v_cache_chip), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )

    # Per-chip forward:
    #   in_proj: x [1, HIDDEN] @ w_qkv_chip [HIDDEN, 3584] → [1, 3584]
    #   slice  : qg [3072], k [256], v [256]
    #   qg.reshape([NQ_PER_CHIP, 2*HEAD_DIM]) → q [6, 256], gate [6, 256]
    #   write new k, v into cache position 0  (skipped here; we already injected
    #     k, v into k_cache[0]/v_cache[0] in numpy reference; for TP we'd need
    #     update_cache_for_token_ but for plumbing test we just use a fresh
    #     replacement: scatter the freshly-computed k, v at pos 0.)
    all_tt = ttnn.linear(x_tt, w_qkv_tt)  # [1, ATTN_QKV_DIM_CHIP] per chip
    qg_tt = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
    k_new = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
    v_new = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                       [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])

    # Reshape to per-head
    qg = ttnn.reshape(qg_tt, [NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_new, [NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_new, [NKV_PER_CHIP, HEAD_DIM])

    # Build per-chip K, V context: replace position 0 of cache with current k, v
    # The simplest probe-friendly approach: read cache to host shape on-device.
    # But for plumbing we can just use the cache as-is (it already has k, v at pos 0
    # in the numpy reference). We approximate by NOT updating cache here — both
    # numpy and TP take the SAME pre-loaded cache as input. So cache[0] already
    # contains the right values from numpy gold setup.
    # For attention math: K [MAX_POS, NKV_PER_CHIP, HEAD_DIM]. We have k_cache_tt
    # [NKV_PER_CHIP, MAX_POS, HEAD_DIM] (per-chip). Transpose for the math.
    # Actually: per-chip cache holds 1 KV head so its [1, MAX_POS, HEAD_DIM].
    # Reshape to [MAX_POS, HEAD_DIM] for attention math (since NKV_PER_CHIP=1).
    assert NKV_PER_CHIP == 1
    k_full_chip = ttnn.reshape(k_cache_tt, [MAX_POS, HEAD_DIM])
    v_full_chip = ttnn.reshape(v_cache_tt, [MAX_POS, HEAD_DIM])

    # Manual attention: Q @ K^T → softmax → @ V
    # q_tt: [NQ_PER_CHIP, HEAD_DIM] = [6, 256]
    # k_full_chip: [MAX_POS, HEAD_DIM] = [32, 256]
    # scores: [NQ_PER_CHIP, MAX_POS] = [6, 32]
    scale = 1.0 / np.sqrt(HEAD_DIM)
    kT = ttnn.transpose(k_full_chip, 0, 1)  # [HEAD_DIM, MAX_POS]
    scores = ttnn.matmul(q_tt, kT)
    scores = ttnn.mul(scores, scale)
    attn_w = ttnn.softmax(scores, dim=-1)
    attn = ttnn.matmul(attn_w, v_full_chip)  # [NQ_PER_CHIP, HEAD_DIM]

    # Sigmoid gate
    sig = ttnn.sigmoid(gate_tt)
    attn_gated = ttnn.mul(attn, sig)

    # o_proj: per-chip [1, NQ_PER_CHIP * HEAD_DIM] → [1, HIDDEN] partial
    attn_flat = ttnn.reshape(attn_gated, [1, O_DIM_CHIP])
    partial = ttnn.linear(attn_flat, w_o_tt)

    # All-reduce
    try:
        out = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        out = ttnn.all_gather(scattered, dim=1)

    stacked = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    ).float().cpu().numpy()
    return stacked


def main():
    print("=" * 78)
    print("C'7.3: Gated Attention 4-chip tensor-parallel correctness probe")
    print("=" * 78)
    print(f"N_Q={N_Q}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}  N_REP={N_REP}")
    print(f"Per chip: {NQ_PER_CHIP} Q heads, {NKV_PER_CHIP} KV head, ATTN_QKV_DIM_CHIP={ATTN_QKV_DIM_CHIP}")

    print("\n[0] FABRIC_1D init...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    print("[1] Open (1, 4) mesh...")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips")

    try:
        print(f"\n[2] Build random weights + KV cache (MAX_POS={MAX_POS}, seed=42)...")
        rng = np.random.default_rng(42)
        x = rng.standard_normal((1, HIDDEN)).astype(np.float32) * 0.1
        w_attn_qkv = rng.standard_normal((HIDDEN, ATTN_QKV_DIM)).astype(np.float32) / np.sqrt(HIDDEN)
        w_o = rng.standard_normal((O_DIM_FULL, HIDDEN)).astype(np.float32) / np.sqrt(O_DIM_FULL)
        k_cache = rng.standard_normal((MAX_POS, N_KV, HEAD_DIM)).astype(np.float32) * 0.05
        v_cache = rng.standard_normal((MAX_POS, N_KV, HEAD_DIM)).astype(np.float32) * 0.05

        print(f"  x={x.shape}  W_qkv={w_attn_qkv.shape}  W_o={w_o.shape}")
        print(f"  K_cache={k_cache.shape}  V_cache={v_cache.shape}")

        print("\n[3] Numpy fp32 gold forward...")
        # Note: we PRE-COMPUTE k, v from in_proj and inject into cache[0] in
        # single_chip_np_forward; this matches the TP setup where each chip
        # uses cache[0] = newly computed K, V for its KV head.
        # TP probe takes the SAME pre-injected cache (no in-trace update).
        all_p = x @ w_attn_qkv
        k_new = all_p[0, QG_DIM:QG_DIM + KV_DIM].reshape(N_KV, HEAD_DIM)
        v_new = all_p[0, QG_DIM + KV_DIM:].reshape(N_KV, HEAD_DIM)
        k_cache_with_new = k_cache.copy()
        v_cache_with_new = v_cache.copy()
        k_cache_with_new[0] = k_new
        v_cache_with_new[0] = v_new
        y_gold = single_chip_np_forward(x, w_attn_qkv, w_o, k_cache, v_cache).flatten()
        print(f"  y_gold: shape=(1,{HIDDEN}), max|y|={np.abs(y_gold).max():.4f}, std={y_gold.std():.4f}")

        print("\n[4] TP forward (4-chip)...")
        # Inject k_new, v_new into the per-chip cache BEFORE upload (probe
        # simplification — production uses paged_update_cache in-chip).
        y_tp_stacked = tp_forward(mesh, x, w_attn_qkv, w_o,
                                   k_cache_with_new, v_cache_with_new)
        print(f"  stacked shape: {y_tp_stacked.shape}")

        # Verify chips converged (post all-reduce)
        y_chip0 = y_tp_stacked[0].flatten()
        y_chip3 = y_tp_stacked[-1].flatten()
        cos_inter = _cosine(y_chip0, y_chip3)
        print(f"  cos(chip0, chip3) = {cos_inter:.6f} (should be 1.0)")

        cos_tp = _cosine(y_chip0, y_gold)
        max_diff = float(np.abs(y_chip0 - y_gold).max())
        print(f"  cos(TP chip0, numpy gold) = {cos_tp:.6f}  max|Δ| = {max_diff:.4e}")

        print("\n[5] Latency benchmark (warm)")
        # Capture the closure for re-runs
        def one_step():
            return tp_forward(mesh, x, w_attn_qkv, w_o,
                              k_cache_with_new, v_cache_with_new)
        # Warmup (rebuilds tensors each time — this measures including upload,
        # which inflates the timing. Production reuses uploaded weights.)
        for _ in range(2):
            _ = one_step()
        N = 5
        t0 = time.perf_counter()
        for _ in range(N):
            _ = one_step()
        ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  4-chip TP attn step (incl. re-upload): {ms:.2f} ms")
        print(f"  NOTE: includes weight upload; production reuses, ~10× faster.")

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        ok = cos_tp >= 0.998 and cos_inter >= 0.999 and max_diff < 0.1
        print(f"  TP cos: {cos_tp:.6f}  inter-chip cos: {cos_inter:.6f}  max|Δ|: {max_diff:.4e}")
        print(f"  Result: {'✓ PASS' if ok else '✗ FAIL'}")
    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed")
        except Exception as e:
            print(f"  ✗ close error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset")
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}")


if __name__ == "__main__":
    main()
