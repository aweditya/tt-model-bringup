#!/usr/bin/env python3
"""
P19 — Validate paged SDPA decode swap in CHAINED multi-layer gated_attn context.

P18 (commit 0593245, `feedback_mesh_paged_sdpa_works.md`) proved paged SDPA
decode works on (1,4) mesh at real shapes, ISOLATED — 3.59× faster than manual
SDPA, cos 0.999962 vs host gold. But `feedback_v2_rope_perf_wash.md` warns
isolation-to-production losses are real: dispatch overlap and pipelining can
erase isolated wins entirely.

This probe validates the paged SDPA win in a CHAINED context that mirrors
production server_tp.py:gated_attn_step_tp:

  1. Build K=4 random gated_attn-style layers at real Qwen3.6-27B shapes
     (HIDDEN=5120, N_Q=24, N_KV=4, HEAD_DIM=256, partial_rotary 0.25).
  2. Define TWO variants of `gated_attn_step_tp_local`:
        V1 = current production: manual Q@K^T softmax V (matches lines 580-600)
        V2 = paged_scaled_dot_product_attention_decode w/ explicit progcfg
             (CoreCoord(4,4), q/k_chunk_size=0, HiFi2)
  3. Pre-populate paged KV caches with random K/V at cur_pos∈[0..7]
     so SDPA has real attention data to consume.
  4. Chain K=4 invocations with residual-stream threading.
  5. EAGER correctness: cos(V1_chain_out, V2_chain_out) ≥ 0.999
  6. EAGER latency: 30-iter median, V1 vs V2 chain.
  7. TRACE capture: both variants. Validate trace works for V2.
  8. TRACED latency: K=4 chain, V1 vs V2.

Pass criteria:
  - V1 vs V2 chained cosine ≥ 0.999
  - V2 < V1 latency by ≥ 0.25 ms/layer (≥ 1 ms over K=4)
  - V2 trace capture succeeds

Failure modes documented in task brief.

Run:
    ssh qb2 'cd /home/aditya/tt-xla && .venv/bin/python experiments/utils/p19_gated_attn_paged_sdpa_chained.py'
"""
import os
import sys
import time
import json
import traceback

sys.stdout.reconfigure(line_buffering=True)


# Production shapes (Qwen3.6-27B config.json — verified)
HIDDEN = 5120
HEAD_DIM = 256
N_Q = 24
N_KV = 4
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
NCHIPS = 4
NQ_PER_CHIP = N_Q // NCHIPS         # 6
NKV_PER_CHIP = N_KV // NCHIPS       # 1
QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM   # gate concat
KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
QKV_CHIP_OUT = QG_DIM_CHIP + 2 * KV_DIM_CHIP

MAX_POS = 256
BLOCK_SIZE = 32
NUM_BLOCKS = MAX_POS // BLOCK_SIZE
TILE_HEIGHT = 32
EPS = 1e-6

# Chain depth
K_CHAIN = 4
# How many paged slots to pre-populate
NUM_POPULATED = 8


def _cosine(a, b):
    import numpy as np
    a = a.astype("float64").flatten()
    b = b.astype("float64").flatten()
    n = float((a * a).sum() ** 0.5 * (b * b).sum() ** 0.5)
    if n < 1e-30:
        return 0.0
    return float((a * b).sum() / n)


def _rms_norm_manual(ttnn, x_tt, weight_tt, eps, hidden):
    """Match server_tp.py:_rms_norm_manual signature for chip-local RMS norm.
    x_tt: [1, hidden] sharded along dim=1 across chips → per-chip [1, hidden/N].
    We do replicated RMS for simplicity in this probe (correct math when
    weight + var-source are replicated). For multi-chip TP RMS the production
    uses distributed rms; here our `x_tt` is replicated so this is fine."""
    # Square, mean, rsqrt — chip-local on replicated x.
    x_sq = ttnn.mul(x_tt, x_tt)
    var = ttnn.mean(x_sq, dim=-1, keepdim=True)
    var_eps = ttnn.add(var, eps)
    rstd = ttnn.rsqrt(var_eps)
    x_norm = ttnn.mul(x_tt, rstd)
    return ttnn.mul(x_norm, weight_tt)


def _build_weights_one_layer(rng):
    """Return numpy weights for one gated_attn-style layer at production shapes."""
    import numpy as np
    return {
        # input_norm (replicated per chip)
        'input_norm': rng.standard_normal((HIDDEN,)).astype(np.float32) * 0.1 + 1.0,
        # qkv: [HIDDEN, N_Q*HEAD_DIM*2 + N_KV*HEAD_DIM*2]
        # per-chip slab: [HIDDEN, QKV_CHIP_OUT] — sharded along dim=1 (4 chips)
        'w_qkv': rng.standard_normal((HIDDEN, NCHIPS * QKV_CHIP_OUT)).astype(np.float32) * 0.02,
        # q_norm: [HEAD_DIM] — replicated
        'q_norm': rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.05 + 1.0,
        'k_norm': rng.standard_normal((HEAD_DIM,)).astype(np.float32) * 0.05 + 1.0,
        # out_proj row-parallel: [NQ_PER_CHIP*HEAD_DIM, HIDDEN] per chip — sharded dim=0
        'w_o': rng.standard_normal((N_Q * HEAD_DIM, HIDDEN)).astype(np.float32) * 0.02,
    }


def _upload_layer(ttnn, torch, np, mesh, w_np):
    """Upload one layer to the mesh. QKV sharded dim=1, w_o sharded dim=0,
    norms replicated."""
    def replicated(arr, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def sharded(arr, dim, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim))

    # input_norm: [HIDDEN] → [1, HIDDEN] (TILE_LAYOUT needs 2D)
    in_norm_2d = w_np['input_norm'][None, :]
    # q_norm/k_norm: [HEAD_DIM] → [1, HEAD_DIM]
    q_norm_2d = w_np['q_norm'][None, :]
    k_norm_2d = w_np['k_norm'][None, :]
    return {
        'input_norm': replicated(in_norm_2d),
        'w_qkv': sharded(w_np['w_qkv'], dim=1),    # [HIDDEN, 4*QKV_CHIP_OUT] → per chip [HIDDEN, QKV_CHIP_OUT]
        'q_norm': replicated(q_norm_2d),
        'k_norm': replicated(k_norm_2d),
        'w_o': sharded(w_np['w_o'], dim=0),         # [N_Q*HEAD_DIM, HIDDEN] → per chip [NQ_PER_CHIP*HEAD_DIM, HIDDEN]
    }


def _build_paged_caches(ttnn, torch, np, mesh, rng):
    """Build paged K/V caches with NUM_POPULATED slots populated.
    Returns kc, vc, kv_per_pos_np (for host gold)."""
    cache_k_np = np.zeros((NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    cache_v_np = np.zeros((NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM), dtype=np.float32)
    cache_k = ttnn.from_torch(
        torch.from_numpy(cache_k_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    cache_v = ttnn.from_torch(
        torch.from_numpy(cache_v_np), dtype=ttnn.bfloat16,
        device=mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    return cache_k, cache_v


def _build_page_table_and_mem_cfg(ttnn, torch, np, mesh):
    page_table_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, NUM_BLOCKS)
    page_table = ttnn.from_torch(
        torch.from_numpy(page_table_np),
        device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    compute_grid = mesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(1, compute_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                 ttnn.ShardOrientation.ROW_MAJOR)
    paged_write_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return page_table, paged_write_mem_cfg


def _populate_paged_cache(ttnn, torch, np, mesh, cache_k, cache_v,
                           page_table, paged_write_mem_cfg, rng):
    """Populate cur_pos∈[0..NUM_POPULATED-1] with random K/V across all chips."""
    for cp in range(NUM_POPULATED):
        k_np = (rng.standard_normal((1, 1, N_KV, HEAD_DIM)) * 0.3).astype(np.float32)
        v_np = (rng.standard_normal((1, 1, N_KV, HEAD_DIM)) * 0.3).astype(np.float32)
        k_t = ttnn.from_torch(torch.from_numpy(k_np), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        v_t = ttnn.from_torch(torch.from_numpy(v_np), dtype=ttnn.bfloat16,
                                device=mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2))
        k_t = ttnn.pad(k_t, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]], value=0.0)
        v_t = ttnn.pad(v_t, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]], value=0.0)
        k_sharded = ttnn.to_memory_config(k_t, paged_write_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_t, paged_write_mem_cfg)
        idxs = ttnn.from_torch(torch.tensor([cp], dtype=torch.int32),
                                 device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 dtype=ttnn.int32,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        ttnn.experimental.paged_update_cache(cache_k, k_sharded,
                                               update_idxs_tensor=idxs,
                                               page_table=page_table)
        ttnn.experimental.paged_update_cache(cache_v, v_sharded,
                                               update_idxs_tensor=idxs,
                                               page_table=page_table)


def gated_attn_step_V1_manual(ttnn, np, x_tt, attn, cur_pos_tt, cos_tt, sin_tt,
                                page_table_tt, paged_write_mem_cfg):
    """V1 — current production manual SDPA path. Matches server_tp.py:580-600 exactly."""
    h_tt = _rms_norm_manual(ttnn, x_tt, attn['input_norm'], EPS, HIDDEN)
    all_tt = ttnn.linear(h_tt, attn['w_qkv'])
    qg = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                        [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
    qg = ttnn.reshape(qg, [NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_flat, [NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [NKV_PER_CHIP, HEAD_DIM])
    q_tt = _rms_norm_manual(ttnn, q_tt, attn['q_norm'], EPS, HEAD_DIM)
    k_tt = _rms_norm_manual(ttnn, k_tt, attn['k_norm'], EPS, HEAD_DIM)
    # Partial RoPE V2 rotate-only
    half = ROTARY_DIM // 2
    def apply_rope(t, n_heads):
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.add(ttnn.mul(rot, cos_tt),
                            ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_tt))
        return ttnn.concat([rotated, passthru], dim=-1)
    q_tt = apply_rope(q_tt, NQ_PER_CHIP)
    k_tt = apply_rope(k_tt, NKV_PER_CHIP)
    # paged_update_cache
    def _shard_for_paged_write(t_per_head):
        t4d = ttnn.reshape(t_per_head, [1, 1, NKV_PER_CHIP, HEAD_DIM])
        t_padded = ttnn.pad(t4d, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]],
                              value=0.0)
        return ttnn.to_memory_config(t_padded, paged_write_mem_cfg)
    k_sharded = _shard_for_paged_write(k_tt)
    v_sharded = _shard_for_paged_write(v_tt)
    ttnn.experimental.paged_update_cache(attn['kc'], k_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)
    ttnn.experimental.paged_update_cache(attn['vc'], v_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)
    ttnn.deallocate(k_sharded)
    ttnn.deallocate(v_sharded)
    # === MANUAL SDPA (V1) ===
    kc_3d = ttnn.reshape(attn['kc'], [NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM])
    vc_3d = ttnn.reshape(attn['vc'], [NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM])
    kc_flat = ttnn.reshape(kc_3d, [MAX_POS, HEAD_DIM])
    vc_flat = ttnn.reshape(vc_3d, [MAX_POS, HEAD_DIM])
    scale = 1.0 / np.sqrt(HEAD_DIM)
    kT = ttnn.transpose(kc_flat, 0, 1)
    scores = ttnn.mul(ttnn.matmul(q_tt, kT), scale)
    # Optional causal mask for apples-to-apples cosine comparison vs V2 (paged
    # SDPA masks internally). When attn['mask'] is None (production path), this
    # exactly matches server_tp.py:580-600. When set, both paths agree.
    if attn.get('mask') is not None:
        scores = ttnn.add(scores, attn['mask'])
    attn_w = ttnn.softmax(scores, dim=-1)
    attn_per_head = ttnn.matmul(attn_w, vc_flat)
    # gate
    attn_gated = ttnn.mul(attn_per_head, ttnn.sigmoid(gate_tt))
    attn_flat = ttnn.reshape(attn_gated, [1, NQ_PER_CHIP * HEAD_DIM])
    partial = ttnn.linear(attn_flat, attn['w_o'])
    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    return ttnn.add(x_tt, reduced)


def gated_attn_step_V2_paged(ttnn, np, x_tt, attn, cur_pos_tt, cos_tt, sin_tt,
                                page_table_tt, paged_write_mem_cfg,
                                paged_sdpa_progcfg, sdpa_compute_kernel_config):
    """V2 — paged_scaled_dot_product_attention_decode swap (P18 recipe)."""
    h_tt = _rms_norm_manual(ttnn, x_tt, attn['input_norm'], EPS, HIDDEN)
    all_tt = ttnn.linear(h_tt, attn['w_qkv'])
    qg = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                        [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
    qg = ttnn.reshape(qg, [NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_flat, [NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [NKV_PER_CHIP, HEAD_DIM])
    q_tt = _rms_norm_manual(ttnn, q_tt, attn['q_norm'], EPS, HEAD_DIM)
    k_tt = _rms_norm_manual(ttnn, k_tt, attn['k_norm'], EPS, HEAD_DIM)
    half = ROTARY_DIM // 2
    def apply_rope(t, n_heads):
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.add(ttnn.mul(rot, cos_tt),
                            ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_tt))
        return ttnn.concat([rotated, passthru], dim=-1)
    q_tt = apply_rope(q_tt, NQ_PER_CHIP)
    k_tt = apply_rope(k_tt, NKV_PER_CHIP)
    def _shard_for_paged_write(t_per_head):
        t4d = ttnn.reshape(t_per_head, [1, 1, NKV_PER_CHIP, HEAD_DIM])
        t_padded = ttnn.pad(t4d, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]],
                              value=0.0)
        return ttnn.to_memory_config(t_padded, paged_write_mem_cfg)
    k_sharded = _shard_for_paged_write(k_tt)
    v_sharded = _shard_for_paged_write(v_tt)
    ttnn.experimental.paged_update_cache(attn['kc'], k_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)
    ttnn.experimental.paged_update_cache(attn['vc'], v_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)
    ttnn.deallocate(k_sharded)
    ttnn.deallocate(v_sharded)
    # === PAGED SDPA (V2 — P18 recipe) ===
    # Q must be [1, 1, NQ_PER_CHIP, HEAD_DIM]; current q_tt is [NQ_PER_CHIP, HEAD_DIM]
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, NQ_PER_CHIP, HEAD_DIM])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, attn['kc'], attn['vc'],
        cur_pos_tensor=cur_pos_tt,
        page_table_tensor=page_table_tt,
        program_config=paged_sdpa_progcfg,
        compute_kernel_config=sdpa_compute_kernel_config,
    )
    # Output: [1, 1, NQ_PER_CHIP, HEAD_DIM] → [NQ_PER_CHIP, HEAD_DIM]
    attn_per_head = ttnn.reshape(attn_out, [NQ_PER_CHIP, HEAD_DIM])
    # gate
    attn_gated = ttnn.mul(attn_per_head, ttnn.sigmoid(gate_tt))
    attn_flat = ttnn.reshape(attn_gated, [1, NQ_PER_CHIP * HEAD_DIM])
    partial = ttnn.linear(attn_flat, attn['w_o'])
    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    return ttnn.add(x_tt, reduced)


def _bench_latency(ttnn, mesh, fn, iters=30, warmup=5):
    import numpy as np
    for _ in range(warmup):
        fn()
        ttnn.synchronize_device(mesh)
    times = []
    for _ in range(iters):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.min(times))


def main():
    print("=" * 78, flush=True)
    print("P19: CHAINED gated_attn paged SDPA validation on (1,4) mesh", flush=True)
    print("=" * 78, flush=True)
    print(f"  K_CHAIN={K_CHAIN}  MAX_POS={MAX_POS}  NUM_POPULATED={NUM_POPULATED}", flush=True)
    print(f"  HIDDEN={HIDDEN} N_Q={N_Q} N_KV={N_KV} HEAD_DIM={HEAD_DIM}", flush=True)
    print(f"  NQ_PER_CHIP={NQ_PER_CHIP} NKV_PER_CHIP={NKV_PER_CHIP}", flush=True)
    print(f"  ROTARY_DIM={ROTARY_DIM}", flush=True)

    import ttnn
    import torch
    import numpy as np

    results = {
        "shapes": {
            "HIDDEN": HIDDEN, "HEAD_DIM": HEAD_DIM, "N_Q": N_Q, "N_KV": N_KV,
            "NQ_PER_CHIP": NQ_PER_CHIP, "NKV_PER_CHIP": NKV_PER_CHIP,
            "ROTARY_DIM": ROTARY_DIM,
            "MAX_POS": MAX_POS, "BLOCK_SIZE": BLOCK_SIZE,
            "K_CHAIN": K_CHAIN, "NUM_POPULATED": NUM_POPULATED,
        },
    }

    print("\n[1] open mesh + fabric", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ {mesh.get_num_devices()} chips", flush=True)
    print(f"  compute_grid={mesh.compute_with_storage_grid_size()}", flush=True)

    try:
        rng = np.random.default_rng(42)

        # --- Build K layers (random weights) ---
        print(f"\n[2] Build {K_CHAIN} layers of random weights", flush=True)
        layer_np = [_build_weights_one_layer(rng) for _ in range(K_CHAIN)]

        # --- Two copies: one set of caches for V1 chain, one for V2 chain ---
        print(f"\n[3] Upload weights (replicated norms + sharded qkv/out)", flush=True)
        layers_V1 = [_upload_layer(ttnn, torch, np, mesh, lw) for lw in layer_np]
        layers_V2 = [_upload_layer(ttnn, torch, np, mesh, lw) for lw in layer_np]

        # --- Build paged caches per layer (V1 and V2 use separate caches but
        # populated with IDENTICAL data so the chained output is comparable)
        print(f"\n[4] Build + populate paged KV caches (both V1 + V2 sets)", flush=True)
        page_table_tt, paged_write_mem_cfg = _build_page_table_and_mem_cfg(ttnn, torch, np, mesh)

        for i in range(K_CHAIN):
            cache_k_v1, cache_v_v1 = _build_paged_caches(ttnn, torch, np, mesh, rng)
            cache_k_v2, cache_v_v2 = _build_paged_caches(ttnn, torch, np, mesh, rng)
            layers_V1[i]['kc'] = cache_k_v1
            layers_V1[i]['vc'] = cache_v_v1
            layers_V2[i]['kc'] = cache_k_v2
            layers_V2[i]['vc'] = cache_v_v2

        # Populate IDENTICAL data into both V1 + V2 cache pairs.
        # Use a fixed seeded RNG and replay the same sequence into both pairs.
        for i in range(K_CHAIN):
            sub_seed = 100 + i  # distinct per layer, same for V1 + V2 copies
            rng_v1 = np.random.default_rng(sub_seed)
            rng_v2 = np.random.default_rng(sub_seed)
            _populate_paged_cache(ttnn, torch, np, mesh,
                                    layers_V1[i]['kc'], layers_V1[i]['vc'],
                                    page_table_tt, paged_write_mem_cfg, rng_v1)
            _populate_paged_cache(ttnn, torch, np, mesh,
                                    layers_V2[i]['kc'], layers_V2[i]['vc'],
                                    page_table_tt, paged_write_mem_cfg, rng_v2)
        ttnn.synchronize_device(mesh)
        print(f"  ✓ {K_CHAIN} caches × 2 variants × {NUM_POPULATED} slots populated", flush=True)

        # --- Build SDPA configs (P18 winner) ---
        paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
            q_chunk_size=0, k_chunk_size=0,
            exp_approx_mode=False,
        )
        sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        # --- Input + RoPE tables (replicated) ---
        x_np = (rng.standard_normal((1, HIDDEN)) * 0.5).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                 device=mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        cos_np = np.cos(rng.standard_normal((1, ROTARY_DIM)) * 0.1).astype(np.float32)
        sin_np = np.sin(rng.standard_normal((1, ROTARY_DIM)) * 0.1).astype(np.float32)
        cos_tt = ttnn.from_torch(torch.from_numpy(cos_np), dtype=ttnn.bfloat16,
                                   device=mesh, layout=ttnn.TILE_LAYOUT,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        sin_tt = ttnn.from_torch(torch.from_numpy(sin_np), dtype=ttnn.bfloat16,
                                   device=mesh, layout=ttnn.TILE_LAYOUT,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # cur_pos = NUM_POPULATED (will WRITE slot 8, read slots 0..8)
        cur_pos_val = NUM_POPULATED
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos_val], dtype=torch.int32),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # Build attention mask for V1 correctness comparison (matches V2's
        # internal masking). Production V1 (server_tp.py:580-600) does NOT
        # apply this mask, but for an apples-to-apples cosine check we need it.
        # mask: [NQ_PER_CHIP, MAX_POS] — 0 at pos<=cur_pos, -1e9 beyond
        mask_np = np.zeros((NQ_PER_CHIP, MAX_POS), dtype=np.float32)
        mask_np[:, cur_pos_val + 1:] = -1e9
        mask_tt = ttnn.from_torch(torch.from_numpy(mask_np), dtype=ttnn.bfloat16,
                                    device=mesh, layout=ttnn.TILE_LAYOUT,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # === EAGER chained forward ===
        def chain_V1(x_in, with_mask=False):
            cur = x_in
            for i in range(K_CHAIN):
                attn_d = dict(layers_V1[i])
                attn_d['mask'] = mask_tt if with_mask else None
                cur = gated_attn_step_V1_manual(
                    ttnn, np, cur, attn_d, cur_pos_tt, cos_tt, sin_tt,
                    page_table_tt, paged_write_mem_cfg,
                )
            return cur

        def chain_V2(x_in):
            cur = x_in
            for i in range(K_CHAIN):
                cur = gated_attn_step_V2_paged(
                    ttnn, np, cur, layers_V2[i], cur_pos_tt, cos_tt, sin_tt,
                    page_table_tt, paged_write_mem_cfg,
                    paged_sdpa_progcfg, sdpa_compute_kernel_config,
                )
            return cur

        # NOTE: chain mutates KV cache. To get a fair cosine comparison,
        # we must REPOPULATE caches between V1 and V2 runs.
        def repopulate_all():
            for i in range(K_CHAIN):
                sub_seed = 100 + i
                rng_v1 = np.random.default_rng(sub_seed)
                rng_v2 = np.random.default_rng(sub_seed)
                # Zero out via fresh cache rebuild — costlier than reset but safe.
                # Actually simpler: just rewrite the populated slots with same data.
                # The write at cur_pos=NUM_POPULATED is the additional one
                # from gated_attn_step itself. After the chain, slot 8 contains
                # whatever the chain wrote. Resetting requires zeroing the cache.
                _populate_paged_cache(ttnn, torch, np, mesh,
                                        layers_V1[i]['kc'], layers_V1[i]['vc'],
                                        page_table_tt, paged_write_mem_cfg, rng_v1)
                _populate_paged_cache(ttnn, torch, np, mesh,
                                        layers_V2[i]['kc'], layers_V2[i]['vc'],
                                        page_table_tt, paged_write_mem_cfg, rng_v2)

        print(f"\n[5] EAGER chained forward — V1 (manual SDPA, WITH MASK for fair cosine)", flush=True)
        out_v1_tt = chain_V1(x_tt, with_mask=True)
        ttnn.synchronize_device(mesh)
        # all_reduce → replicated; verify replication via per-chip view
        out_v1_chips = ttnn.to_torch(out_v1_tt,
                                 mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
                                ).float().numpy()
        out_v1 = out_v1_chips[0] if out_v1_chips.ndim > 1 else out_v1_chips
        print(f"  V1 chain out: shape={out_v1_chips.shape} max|.|={np.abs(out_v1).max():.4f}", flush=True)
        if out_v1_chips.ndim > 1 and out_v1_chips.shape[0] >= 2:
            ic = _cosine(out_v1_chips[0], out_v1_chips[1])
            print(f"  inter-chip cos V1(chip0, chip1) = {ic:.6f}", flush=True)
            results["inter_chip_cos_V1"] = ic

        print(f"\n[6] Re-populate caches before V2 run (deterministic state)", flush=True)
        # The V1 chain just modified V1 caches AND wrote slot 8.
        # Reset V2 caches to the pre-V1 state by re-populating + clearing slot 8.
        # Simpler: zero the caches entirely and re-populate (rebuild needed because
        # paged_update_cache writes but doesn't clear).
        # Recreate V2 caches from scratch:
        for i in range(K_CHAIN):
            ttnn.deallocate(layers_V2[i]['kc'])
            ttnn.deallocate(layers_V2[i]['vc'])
            cache_k_v2, cache_v_v2 = _build_paged_caches(ttnn, torch, np, mesh, rng)
            layers_V2[i]['kc'] = cache_k_v2
            layers_V2[i]['vc'] = cache_v_v2
            rng_v2 = np.random.default_rng(100 + i)
            _populate_paged_cache(ttnn, torch, np, mesh,
                                    layers_V2[i]['kc'], layers_V2[i]['vc'],
                                    page_table_tt, paged_write_mem_cfg, rng_v2)
        ttnn.synchronize_device(mesh)
        print(f"  ✓ V2 caches reset", flush=True)

        print(f"\n[7] EAGER chained forward — V2 (paged SDPA)", flush=True)
        out_v2_tt = chain_V2(x_tt)
        ttnn.synchronize_device(mesh)
        out_v2 = ttnn.to_torch(out_v2_tt,
                                 mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
                                ).float().numpy()
        out_v2_arr = out_v2[0] if out_v2.ndim > 1 else out_v2
        print(f"  V2 chain out: shape={out_v2.shape} max|.|={np.abs(out_v2_arr).max():.4f}", flush=True)
        if out_v2.ndim > 1 and out_v2.shape[0] >= 2:
            print(f"  inter-chip cos V2(chip0, chip1) = {_cosine(out_v2[0], out_v2[1]):.6f}", flush=True)

        # Eager cosine V1 vs V2 (compare per-chip view: chip 0 to chip 0)
        cos_chain = _cosine(out_v1, out_v2_arr)
        max_abs_delta = float(np.abs(out_v1.astype(np.float64) - out_v2_arr.astype(np.float64)).max())
        # Also: aggregate cosine across all chips' views
        cos_chain_all = _cosine(out_v1_chips, out_v2)
        results["chain_cosine_chip0"] = cos_chain
        results["chain_cosine_all_chips"] = cos_chain_all
        print(f"[CHAIN COSINE all chips] cos = {cos_chain_all:.6f}", flush=True)
        print(f"\n[CHAIN COSINE] cos(V1, V2) = {cos_chain:.6f}", flush=True)
        print(f"[CHAIN |Δ|max] = {max_abs_delta:.4f}", flush=True)
        results["chain_cosine"] = cos_chain
        results["chain_max_abs_delta"] = max_abs_delta
        results["eager_pass_correctness"] = bool(cos_chain >= 0.999)

        # === EAGER latency benchmark ===
        print(f"\n[8] EAGER latency benchmark (K={K_CHAIN})", flush=True)

        # Re-populate before each timed run because chain mutates state at cur_pos.
        # To avoid the repopulate cost confounding latency, set cur_pos_val very small
        # → each call writes slot 8 which is innocuous after warmup.
        # Actually, fixed cur_pos means each call overwrites slot 8 with new data
        # (still writable, still in range). That's fine for latency timing.

        def chain_V1_bench():
            # No mask — matches current production server_tp.py:580-600 exactly
            _ = chain_V1(x_tt, with_mask=False)

        def chain_V2_bench():
            _ = chain_V2(x_tt)

        v1_med, v1_min = _bench_latency(ttnn, mesh, chain_V1_bench, iters=30, warmup=5)
        v2_med, v2_min = _bench_latency(ttnn, mesh, chain_V2_bench, iters=30, warmup=5)
        print(f"  V1 eager chain: median {v1_med:.3f} ms  min {v1_min:.3f}", flush=True)
        print(f"  V2 eager chain: median {v2_med:.3f} ms  min {v2_min:.3f}", flush=True)
        delta = v1_med - v2_med
        per_layer = delta / K_CHAIN
        print(f"  Δ = V1 − V2 = {delta:.3f} ms ({delta/v1_med*100:+.1f}%)", flush=True)
        print(f"  per-layer Δ = {per_layer:.3f} ms", flush=True)
        results["eager_latency_ms"] = {
            "V1_median": v1_med, "V1_min": v1_min,
            "V2_median": v2_med, "V2_min": v2_min,
            "delta_median": delta, "per_layer_delta": per_layer,
            "K": K_CHAIN,
        }
        results["eager_pass_latency"] = bool(delta >= 1.0)  # ≥1ms over K=4 = ≥0.25/layer

        # === TRACE capture ===
        print(f"\n[9] TRACE capture — V1", flush=True)
        # Trace V1 (production-faithful, no mask)
        trace_v1_ok = False
        trace_v1_err = None
        try:
            # Warm up to ensure all kernels are compiled
            _ = chain_V1(x_tt, with_mask=False)
            ttnn.synchronize_device(mesh)
            tid_v1 = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_out_v1 = chain_V1(x_tt, with_mask=False)
            ttnn.end_trace_capture(mesh, tid_v1, cq_id=0)
            ttnn.synchronize_device(mesh)
            trace_v1_ok = True
            print(f"  ✓ V1 trace captured (tid={tid_v1})", flush=True)
        except Exception as e:
            trace_v1_err = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"  ✗ V1 trace failed: {trace_v1_err}", flush=True)

        print(f"\n[10] TRACE capture — V2", flush=True)
        trace_v2_ok = False
        trace_v2_err = None
        try:
            _ = chain_V2(x_tt)
            ttnn.synchronize_device(mesh)
            tid_v2 = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_out_v2 = chain_V2(x_tt)
            ttnn.end_trace_capture(mesh, tid_v2, cq_id=0)
            ttnn.synchronize_device(mesh)
            trace_v2_ok = True
            print(f"  ✓ V2 trace captured (tid={tid_v2})", flush=True)
        except Exception as e:
            trace_v2_err = f"{type(e).__name__}: {str(e)[:500]}"
            print(f"  ✗ V2 trace failed: {trace_v2_err}", flush=True)

        results["trace_v1_ok"] = trace_v1_ok
        results["trace_v1_err"] = trace_v1_err
        results["trace_v2_ok"] = trace_v2_ok
        results["trace_v2_err"] = trace_v2_err

        # === TRACED latency ===
        if trace_v1_ok and trace_v2_ok:
            print(f"\n[11] TRACED latency benchmark (K={K_CHAIN})", flush=True)

            def exec_v1():
                ttnn.execute_trace(mesh, tid_v1, cq_id=0, blocking=False)

            def exec_v2():
                ttnn.execute_trace(mesh, tid_v2, cq_id=0, blocking=False)

            t_v1_med, t_v1_min = _bench_latency(ttnn, mesh, exec_v1, iters=30, warmup=5)
            t_v2_med, t_v2_min = _bench_latency(ttnn, mesh, exec_v2, iters=30, warmup=5)
            print(f"  V1 traced: median {t_v1_med:.3f} ms  min {t_v1_min:.3f}", flush=True)
            print(f"  V2 traced: median {t_v2_med:.3f} ms  min {t_v2_min:.3f}", flush=True)
            t_delta = t_v1_med - t_v2_med
            t_per_layer = t_delta / K_CHAIN
            print(f"  Δ traced = {t_delta:.3f} ms ({t_delta/t_v1_med*100:+.1f}%)", flush=True)
            print(f"  per-layer Δ traced = {t_per_layer:.3f} ms", flush=True)
            results["traced_latency_ms"] = {
                "V1_median": t_v1_med, "V1_min": t_v1_min,
                "V2_median": t_v2_med, "V2_min": t_v2_min,
                "delta_median": t_delta, "per_layer_delta": t_per_layer,
                "K": K_CHAIN,
            }
            results["traced_pass_latency"] = bool(t_delta >= 1.0)
            try:
                ttnn.release_trace(mesh, tid_v1)
                ttnn.release_trace(mesh, tid_v2)
            except Exception:
                pass

        # === Final verdict ===
        ship = bool(
            results.get("eager_pass_correctness", False) and
            results.get("eager_pass_latency", False) and
            results.get("trace_v2_ok", False)
        )
        results["SHIP"] = ship
        print(f"\n{'=' * 78}", flush=True)
        print(f"FINAL: SHIP={'YES' if ship else 'NO'}", flush=True)
        print(f"  correctness pass: {results.get('eager_pass_correctness')}", flush=True)
        print(f"  eager latency pass: {results.get('eager_pass_latency')}", flush=True)
        print(f"  trace V2 OK: {results.get('trace_v2_ok')}", flush=True)
        print(f"{'=' * 78}", flush=True)

    finally:
        out_dir = os.path.join(os.path.expanduser("~/tt-xla"), ".cache",
                                "p19_chained_paged_sdpa")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "results.json")
        try:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n[results] wrote {out_path}", flush=True)
        except Exception as e:
            print(f"[results] save error: {e}", flush=True)

        try:
            ttnn.close_mesh_device(mesh)
            print("  ✓ mesh closed", flush=True)
        except Exception as e:
            print(f"  ✗ close error: {e}", flush=True)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric disabled", flush=True)
        except Exception as e:
            print(f"  ✗ fabric reset error: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
