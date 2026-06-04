#!/usr/bin/env python3
"""Continuous-batching server for Gemma 4 12B — v1.

Forks `experiments/serve/server_tp_cb.py` (27B CB; closest reuse — Gemma 4
is dense like 27B, no DN/MoE) and adapts shapes for Gemma 4's:
  - Hybrid sliding (40 layers, head_dim=256, NKV_PER_CHIP=2) +
    global (8 layers, head_dim=512, NKV_PER_CHIP=1) attention.
  - Sliding uses TWO paged-SDPA calls per layer (NKV=1 each, matches
    35B's clean kernel contract — see [[paged-update-cache-nkv-per-chip]]).
  - Global has attention_k_eq_v=True (V is K_raw clone post-projection).
  - Per-layer learned `layer_scalar` multiply at end of each decoder
    block ([[gemma4-layer-scalar]]).
  - SDPA `scale=1.0` (HF `self.scaling=1.0`, [[feedback-gemma4-sdpa-scale-1]]).
  - p-RoPE on global (partial 0.25 over head_dim=512).
  - Embedding scaled by sqrt(HIDDEN) + 30·tanh(x/30) logit softcap.

Imports the v0 base server (`server_gemma4_unified_ttnn`) for State,
bootstrap, weight upload, and the `_apply_full_rope` / `all_reduce_tt`
helpers. Base is untouched (zero regression on v0.4 trace).

Validation ladder (cb_validate_gm4.py):
  3a. B=1 batched forward bit-identical to v0.4 single-slot forward.
  3b. Identical inputs at two slots produce identical outputs.
  3c. Distinct inputs at two slots stay isolated (no cross-talk).

v1.x sub-staging (filled in via the dev harness):
  v1.0  setup_cb_state allocator + smoke (this commit)
  v1.1  batched embed + RoPE prelude
  v1.2  batched sliding-attention layer
  v1.3  batched global-attention layer
  v1.4  end-to-end batched forward (3a)
  v1.5  3b + 3c gates at B=2, then B=4
  v1.6  trace capture at fixed B
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import ttnn  # noqa: E402

# Shape constants (re-exported from base for one-line CB reads).
NUM_LAYERS         = base.NUM_LAYERS          # 48
NQ_PER_CHIP        = base.NQ_PER_CHIP         # 4
NKV_PER_CHIP_SLID  = base.NKV_PER_CHIP_SLIDING  # 2
NKV_GLOBAL         = base.NUM_KV_HEADS_GLOBAL   # 1 (replicated across mesh)
HEAD_DIM_SLID      = base.HEAD_DIM_SLIDING    # 256
HEAD_DIM_GLOB      = base.HEAD_DIM_GLOBAL     # 512
SLIDING_WINDOW     = base.SLIDING_WINDOW      # 1024
EMBED_SCALE        = base.EMBED_SCALE
FINAL_LOGIT_SOFTCAP = base.FINAL_LOGIT_SOFTCAP
EPS                = base.EPS
HIFI4              = base.HIFI4
BLOCK_SIZE         = 32
MAX_KV             = base.MAX_KV


# ── Per-slot CB state ──────────────────────────────────────────────────
def setup_cb_state(state, B, blocks_per_seq=None):
    """Allocate batched buffers + per-layer KV caches (sliding + global)
    + per-slot page table. Call AFTER base.bootstrap. Mirrors
    server_tp_cb.setup_cb_state with Gemma 4 shape changes.

    B: number of concurrent slots (fixed batch width, e.g. 4).
    blocks_per_seq: KV blocks per slot. Default = MAX_KV/BLOCK_SIZE = 128
      (each slot gets the same context budget as the single-slot path).
    """
    mesh = state.mesh
    state.cb_B = B
    if blocks_per_seq is None:
        blocks_per_seq = MAX_KV // BLOCK_SIZE
    state.cb_blocks_per_seq = blocks_per_seq
    total_blocks = B * blocks_per_seq
    state.cb_total_blocks = total_blocks

    # Per-slot page table [B, blocks_per_seq]: contiguous physical block
    # range per slot. Slot b owns blocks [b*BPS, (b+1)*BPS).
    page_table_np = np.stack([
        np.arange(b * blocks_per_seq, (b + 1) * blocks_per_seq, dtype=np.int32)
        for b in range(B)
    ], axis=0)
    state.cb_page_table_tt = ttnn.from_torch(
        torch.from_numpy(page_table_np), dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # KV caches: sized for total_blocks. Sliding layers have TWO caches
    # (per-KV-head, NKV=1 each, matches 35B's clean contract). Global
    # layers have ONE cache (NKV=1 replicated). Each cache holds:
    #   sliding: (total_blocks, 1, BLOCK_SIZE, HEAD_DIM_SLID) sharded on dim 1
    #     across the mesh (note dim 1 is NCHIPS axis since we have 2 caches
    #     per sliding layer, one per KV head per chip).
    #   global:  (total_blocks, NKV_GLOBAL, BLOCK_SIZE, HEAD_DIM_GLOB)
    #     replicated.
    NCHIPS = 4
    state.cb_kv_caches_tt = []
    for L in range(NUM_LAYERS):
        if state.layer_types[L] == "sliding_attention":
            layer_caches = []
            for _ in range(NKV_PER_CHIP_SLID):
                shp = (total_blocks, NCHIPS, BLOCK_SIZE, HEAD_DIM_SLID)
                init = torch.zeros(shp, dtype=torch.float32)
                kc = ttnn.from_torch(init, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=mesh,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
                vc = ttnn.from_torch(init, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, device=mesh,
                                     mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
                layer_caches.append((kc, vc))
            state.cb_kv_caches_tt.append(layer_caches)
        else:  # full_attention (global), NKV=1 replicated
            shp = (total_blocks, NKV_GLOBAL, BLOCK_SIZE, HEAD_DIM_GLOB)
            init = torch.zeros(shp, dtype=torch.float32)
            kc = ttnn.from_torch(init, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            vc = ttnn.from_torch(init, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            state.cb_kv_caches_tt.append([(kc, vc)])  # list-wrap for uniform indexing

    # Batched input buffers — B-leading.
    state.cb_tok_buf = ttnn.from_torch(
        torch.zeros(B, 1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    # cur_pos -1 = empty slot (paged SDPA skips it). Initialize empty.
    state.cb_cur_pos_buf = ttnn.from_torch(
        torch.full((B,), -1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state.cb_rot_idxs_buf = ttnn.from_torch(
        torch.zeros(B, 1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # B-core HEIGHT_SHARDED L1 mem cfg for paged KV writes (one shard per
    # slot). Sliding writes shape [1, B*NKV=1, BLOCK_SIZE, HEAD_DIM_SLID].
    grid = mesh.compute_with_storage_grid_size()
    if B > grid.x * grid.y:
        raise RuntimeError(f"CB B={B} > available cores {grid.x*grid.y}")
    shard_grid = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
    state.cb_write_mem_cfg_sliding = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [BLOCK_SIZE, HEAD_DIM_SLID],
                       ttnn.ShardOrientation.ROW_MAJOR))
    state.cb_write_mem_cfg_global = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [BLOCK_SIZE, HEAD_DIM_GLOB],
                       ttnn.ShardOrientation.ROW_MAJOR))

    # Batched SDPA program configs. SDPA decode parallelizes over batch,
    # so num_cores >= B (sdpa_decode_program_factory). Use the full grid
    # to leave headroom; SDPA only uses B cores, rest idle. Auto chunking
    # keeps per-slot numerics identical to single-slot.
    state.cb_sdpa_progcfg_sliding = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)
    # Global head_dim=512 needs the canonical Tenstorrent config (8,4) +
    # smaller k_chunk so the per-core CB fits the 1.5 MB L1 budget.
    state.cb_sdpa_progcfg_global = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False)
    return state


def cb_reset_states(state):
    """Zero ALL slots — fresh start. cur_pos → -1 (empty); KV caches will
    be overwritten on next write at each slot's cur_pos."""
    host = ttnn.from_torch(
        torch.full((state.cb_B,), -1, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(host, state.cb_cur_pos_buf)


def cb_reset_slots(state, slot_ids):
    """Reset ONLY the given slot indices for mid-batch admission of a new
    sequence. Gemma 4 has NO recurrent state to clear (no DeltaNet); the KV
    cache needs no reset because the SDPA cur_pos bound means a sequence
    restarting at pos=0 overwrites its own slots as it prefills and never
    reads stale data (same logic as the 27B comment, minus the DN reset).
    Implemented as a partial cur_pos write — set only `slot_ids` to -1 so
    other live slots are untouched.
    """
    if not slot_ids:
        return
    # Readback current pos, mutate only the targeted slots, write back.
    cur = ttnn.to_torch(state.cb_cur_pos_buf,
                        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    # to_torch over a [B] tensor replicated across mesh → [B*NCHIPS]; take
    # the first B as the single replica copy (all chips agree).
    cur_np = cur.reshape(-1)[:state.cb_B].clone()
    for s in slot_ids:
        cur_np[s] = -1
    host = ttnn.from_torch(
        cur_np.to(torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(host, state.cb_cur_pos_buf)


def update_input_buffers_batched(state, token_ids, cur_positions):
    """Host→device write to cb_tok_buf [B,1], cb_cur_pos_buf [B],
    cb_rot_idxs_buf [B,1]. Outside any captured trace.

    token_ids:     length-B list/array of int token ids.
    cur_positions: length-B list/array of int positions (use -1 for empty slot).
    """
    B = state.cb_B
    assert len(token_ids) == B and len(cur_positions) == B

    tok_host = ttnn.from_torch(
        torch.tensor(token_ids, dtype=torch.int32).reshape(B, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(tok_host, state.cb_tok_buf)

    pos_host = ttnn.from_torch(
        torch.tensor(cur_positions, dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(pos_host, state.cb_cur_pos_buf)

    # rot_idxs uses ABSOLUTE position (clamped to 0 for empty slots).
    rot_pos = [max(0, p) for p in cur_positions]
    rot_host = ttnn.from_torch(
        torch.tensor(rot_pos, dtype=torch.int32).reshape(B, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(rot_host, state.cb_rot_idxs_buf)


# ── Batched forward ──────────────────────────────────────────────────
def _apply_full_rope_b(x_bnD, cos_1D, sin_1D, n_heads, head_dim):
    """Batched RoPE. x_bnD [B, n_heads, head_dim]; cos/sin [B, 1, head_dim]
    (broadcast over n_heads). Same math as base._apply_full_rope, B-leading.
    Returns [B, n_heads, head_dim]. View-decay: x1/x2 are SLICE views of x;
    do NOT deallocate them (the rule that bit v0.3.1).
    """
    half = head_dim // 2
    x1 = ttnn.slice(x_bnD, [0, 0, 0], [x_bnD.shape[0], n_heads, half])
    x2 = ttnn.slice(x_bnD, [0, 0, half], [x_bnD.shape[0], n_heads, head_dim])
    neg_x2 = ttnn.neg(x2)
    rotated = ttnn.concat([neg_x2, x1], dim=-1)
    ttnn.deallocate(neg_x2)
    x_cos = ttnn.mul(x_bnD, cos_1D)
    rotated_sin = ttnn.mul(rotated, sin_1D)
    ttnn.deallocate(rotated)
    x_rope = ttnn.add(x_cos, rotated_sin)
    ttnn.deallocate(x_cos); ttnn.deallocate(rotated_sin)
    return x_rope


def _shard_for_paged_write_b(t_bND, n_kv_heads, head_dim, mem_cfg):
    """Reshape [B, n_kv_heads, head_dim] → HEIGHT_SHARDED [1, B*n_kv_heads,
    BLOCK_SIZE, head_dim] for paged_update_cache. With n_kv_heads=1
    (Gemma 4's split-per-KV-head pattern), dim(1)=B and the per-slot
    page_table [B, ...] aligns naturally.
    """
    B = t_bND.shape[0]
    t_rm = ttnn.to_layout(t_bND, ttnn.ROW_MAJOR_LAYOUT)
    t4 = ttnn.reshape(t_rm, [1, B * n_kv_heads, 1, head_dim])
    ttnn.deallocate(t_rm)
    t_pad = ttnn.pad(t4, [[0, 0], [0, 0], [0, BLOCK_SIZE - 1], [0, 0]], value=0.0)
    ttnn.deallocate(t4)
    t_tile = ttnn.to_layout(t_pad, ttnn.TILE_LAYOUT)
    ttnn.deallocate(t_pad)
    return ttnn.to_memory_config(t_tile, mem_cfg)


def _layer_sliding_batched(state, h_norm, w, layer_idx):
    """Batched sliding-attention layer. h_norm [B, HIDDEN].
    Two paged-SDPA calls (per KV head, NKV=1 each); same GQA mapping as
    single-slot (Q heads 0,1 attend KV head 0; Q heads 2,3 attend KV head 1).
    Returns [B, HIDDEN]."""
    B = state.cb_B
    layer_caches = state.cb_kv_caches_tt[layer_idx]
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    v = ttnn.matmul(h_norm, w["v_proj"], compute_kernel_config=HIFI4)

    q_h = ttnn.reshape(q, [B, NQ_PER_CHIP, HEAD_DIM_SLID])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [B, NKV_PER_CHIP_SLID, HEAD_DIM_SLID])
    ttnn.deallocate(k)
    v_h = ttnn.reshape(v, [B, NKV_PER_CHIP_SLID, HEAD_DIM_SLID])
    ttnn.deallocate(v)

    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)
    v_n = ttnn.rms_norm(v_h, weight=state.ones_head_dim_sliding, epsilon=EPS)
    ttnn.deallocate(v_h)

    # RoPE — per-slot cos/sin via embedding(cb_rot_idxs_buf, ...).
    cos_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.cos_sliding_tt)
    sin_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.sin_sliding_tt)
    cos_b = ttnn.to_layout(cos_raw, ttnn.TILE_LAYOUT)
    sin_b = ttnn.to_layout(sin_raw, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_raw); ttnn.deallocate(sin_raw)
    # cos_b/sin_b are [B, 1, head_dim] — broadcast over n_heads.
    q_n = _apply_full_rope_b(q_n_pre, cos_b, sin_b, NQ_PER_CHIP, HEAD_DIM_SLID)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope_b(k_n_pre, cos_b, sin_b, NKV_PER_CHIP_SLID, HEAD_DIM_SLID)
    ttnn.deallocate(k_n_pre)
    ttnn.deallocate(cos_b); ttnn.deallocate(sin_b)

    # Two SDPA passes (per (cache, KV-head, Q-half) trio).
    attn_outs = []
    Q_HALF = NQ_PER_CHIP // NKV_PER_CHIP_SLID  # 2
    for kv_idx in range(NKV_PER_CHIP_SLID):
        kc, vc = layer_caches[kv_idx]
        # Slice k/v to row kv_idx — [B, 1, head_dim].
        k_i = ttnn.slice(k_n, [0, kv_idx, 0], [B, kv_idx + 1, HEAD_DIM_SLID])
        v_i = ttnn.slice(v_n, [0, kv_idx, 0], [B, kv_idx + 1, HEAD_DIM_SLID])
        k_sh = _shard_for_paged_write_b(k_i, 1, HEAD_DIM_SLID, state.cb_write_mem_cfg_sliding)
        v_sh = _shard_for_paged_write_b(v_i, 1, HEAD_DIM_SLID, state.cb_write_mem_cfg_sliding)
        ttnn.experimental.paged_update_cache(
            kc, k_sh, update_idxs_tensor=state.cb_cur_pos_buf,
            page_table=state.cb_page_table_tt)
        ttnn.experimental.paged_update_cache(
            vc, v_sh, update_idxs_tensor=state.cb_cur_pos_buf,
            page_table=state.cb_page_table_tt)
        ttnn.deallocate(k_sh); ttnn.deallocate(v_sh)

        # Q for this half: slice [B, Q_HALF, head_dim] then reshape [1, B, Q_HALF, head_dim].
        q_half = ttnn.slice(q_n, [0, kv_idx * Q_HALF, 0],
                            [B, (kv_idx + 1) * Q_HALF, HEAD_DIM_SLID])
        q_for_sdpa = ttnn.reshape(q_half, [1, B, Q_HALF, HEAD_DIM_SLID])
        attn_i = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_for_sdpa, kc, vc,
            cur_pos_tensor=state.cb_cur_pos_buf,
            page_table_tensor=state.cb_page_table_tt,
            scale=1.0,  # Gemma 4: self.scaling=1.0 ([[feedback-gemma4-sdpa-scale-1]])
            program_config=state.cb_sdpa_progcfg_sliding,
            compute_kernel_config=state.sdpa_compute_kernel_config,
            sliding_window_size=SLIDING_WINDOW,
        )
        attn_outs.append(attn_i)
    ttnn.deallocate(q_n); ttnn.deallocate(k_n); ttnn.deallocate(v_n)

    # Concat the two halves along Q-head axis. attn_i [1, B, Q_HALF, HEAD_DIM]
    # → concat dim=2 → [1, B, NQ_PER_CHIP, HEAD_DIM] → flatten [B, NQ*HEAD_DIM].
    attn_concat = ttnn.concat(attn_outs, dim=2)
    for a in attn_outs:
        ttnn.deallocate(a)
    attn_flat = ttnn.reshape(attn_concat, [B, NQ_PER_CHIP * HEAD_DIM_SLID])
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(attn_concat)
    out = base.all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_global_batched(state, h_norm, w, layer_idx):
    """Batched global-attention layer. h_norm [B, HIDDEN]. NKV=1, head_dim=512,
    attention_k_eq_v=True (V is K_raw clone post-projection)."""
    B = state.cb_B
    kc, vc = state.cb_kv_caches_tt[layer_idx][0]
    q = ttnn.matmul(h_norm, w["q_proj"], compute_kernel_config=HIFI4)
    k = ttnn.matmul(h_norm, w["k_proj"], compute_kernel_config=HIFI4)
    q_h = ttnn.reshape(q, [B, NQ_PER_CHIP, HEAD_DIM_GLOB])
    ttnn.deallocate(q)
    k_h = ttnn.reshape(k, [B, NKV_GLOBAL, HEAD_DIM_GLOB])
    ttnn.deallocate(k)
    v_raw = ttnn.clone(k_h)

    q_n_pre = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=EPS)
    k_n_pre = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=EPS)
    v_n = ttnn.rms_norm(v_raw, weight=state.ones_head_dim_global, epsilon=EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h); ttnn.deallocate(v_raw)

    cos_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.cos_global_tt)
    sin_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.sin_global_tt)
    cos_b = ttnn.to_layout(cos_raw, ttnn.TILE_LAYOUT)
    sin_b = ttnn.to_layout(sin_raw, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_raw); ttnn.deallocate(sin_raw)
    q_n = _apply_full_rope_b(q_n_pre, cos_b, sin_b, NQ_PER_CHIP, HEAD_DIM_GLOB)
    ttnn.deallocate(q_n_pre)
    k_n = _apply_full_rope_b(k_n_pre, cos_b, sin_b, NKV_GLOBAL, HEAD_DIM_GLOB)
    ttnn.deallocate(k_n_pre)
    ttnn.deallocate(cos_b); ttnn.deallocate(sin_b)

    k_sh = _shard_for_paged_write_b(k_n, NKV_GLOBAL, HEAD_DIM_GLOB,
                                     state.cb_write_mem_cfg_global)
    v_sh = _shard_for_paged_write_b(v_n, NKV_GLOBAL, HEAD_DIM_GLOB,
                                     state.cb_write_mem_cfg_global)
    ttnn.deallocate(k_n); ttnn.deallocate(v_n)
    ttnn.experimental.paged_update_cache(
        kc, k_sh, update_idxs_tensor=state.cb_cur_pos_buf,
        page_table=state.cb_page_table_tt)
    ttnn.experimental.paged_update_cache(
        vc, v_sh, update_idxs_tensor=state.cb_cur_pos_buf,
        page_table=state.cb_page_table_tt)
    ttnn.deallocate(k_sh); ttnn.deallocate(v_sh)

    q_for_sdpa = ttnn.reshape(q_n, [1, B, NQ_PER_CHIP, HEAD_DIM_GLOB])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=state.cb_cur_pos_buf,
        page_table_tensor=state.cb_page_table_tt,
        scale=1.0,
        program_config=state.cb_sdpa_progcfg_global,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    ttnn.deallocate(q_n)
    attn_flat = ttnn.reshape(attn_out, [B, NQ_PER_CHIP * HEAD_DIM_GLOB])
    partial = ttnn.matmul(attn_flat, w["o_proj"], compute_kernel_config=HIFI4)
    out = base.all_reduce_tt(partial, state.mesh)
    ttnn.deallocate(partial)
    return out


def _layer_forward_batched(state, h_in, layer_idx):
    """One full Gemma 4 decoder layer at B>1. Matches the single-slot
    `base._layer_forward_pos0_paged` math, leading B."""
    w = state.per_layer_tt[layer_idx]
    lt = state.layer_types[layer_idx]
    residual_1 = ttnn.clone(h_in)
    h_norm = ttnn.rms_norm(h_in, weight=w["input_layernorm"], epsilon=EPS)
    if lt == "sliding_attention":
        mixer = _layer_sliding_batched(state, h_norm, w, layer_idx)
    else:
        mixer = _layer_global_batched(state, h_norm, w, layer_idx)
    ttnn.deallocate(h_norm)
    post_attn = ttnn.rms_norm(mixer, weight=w["post_attention_layernorm"], epsilon=EPS)
    ttnn.deallocate(mixer)
    h_after_attn = ttnn.add(residual_1, post_attn)
    ttnn.deallocate(residual_1); ttnn.deallocate(post_attn)
    pre_ff = ttnn.rms_norm(h_after_attn, weight=w["pre_feedforward_layernorm"], epsilon=EPS)
    gate = ttnn.matmul(pre_ff, w["gate_proj"], compute_kernel_config=HIFI4)
    up = ttnn.matmul(pre_ff, w["up_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(pre_ff)
    gelu_gate = ttnn.gelu(gate, fast_and_approximate_mode=False)
    ttnn.deallocate(gate)
    mid = ttnn.mul(gelu_gate, up)
    ttnn.deallocate(gelu_gate); ttnn.deallocate(up)
    mlp_partial = ttnn.matmul(mid, w["down_proj"], compute_kernel_config=HIFI4)
    ttnn.deallocate(mid)
    mlp_out = base.all_reduce_tt(mlp_partial, state.mesh)
    ttnn.deallocate(mlp_partial)
    post_ff = ttnn.rms_norm(mlp_out, weight=w["post_feedforward_layernorm"], epsilon=EPS)
    ttnn.deallocate(mlp_out)
    h_residual_2 = ttnn.add(h_after_attn, post_ff)
    ttnn.deallocate(h_after_attn); ttnn.deallocate(post_ff)
    h_out = ttnn.multiply(h_residual_2, w["layer_scalar"])
    ttnn.deallocate(h_residual_2)
    return h_out


def forward_batch_gm4_inner(state, return_logits=False, return_topk=None):
    """Batched B-leading decode forward. Reads cb_tok_buf, cb_cur_pos_buf,
    cb_rot_idxs_buf. Returns:
      - default (greedy): per-slot argmax tensor [B, 1] (UINT32).
      - return_logits=True: post-softcap logits [B, vocab].
      - return_topk=K: (top_vals, top_idxs) tensors each [B, K].
    """
    B = state.cb_B
    embed = ttnn.embedding(state.cb_tok_buf, state.embed_tt)
    h = ttnn.multiply(ttnn.to_layout(embed, ttnn.TILE_LAYOUT), EMBED_SCALE)
    ttnn.deallocate(embed)
    h = ttnn.reshape(h, [B, base.HIDDEN])

    for L in range(NUM_LAYERS):
        h_new = _layer_forward_batched(state, h, L)
        ttnn.deallocate(h)
        h = h_new

    final = ttnn.rms_norm(h, weight=state.final_norm_tt, epsilon=EPS)
    ttnn.deallocate(h)
    # Vocab-sharded lm_head + softcap + on-device argmax. Forks 27B P22
    # via the base server's _lm_head_argmax helper (lm_head is sharded
    # on dim=1, per-chip matmul, all_gather, slice, untilize, argmax).
    argmax_tt, full_logits = base._lm_head_argmax(
        state, final, capture_logits=(return_logits or return_topk is not None))
    if return_logits:
        # Full logits already gathered, sliced to [B, vocab].
        ttnn.deallocate(argmax_tt)
        return full_logits
    if return_topk is not None:
        # Top-k over the already-gathered logits — small readback per slot.
        top_vals, top_idxs = ttnn.topk(full_logits, k=int(return_topk), dim=-1,
                                       largest=True, sorted=True)
        ttnn.deallocate(argmax_tt)
        ttnn.deallocate(full_logits)
        return (top_vals, top_idxs)
    return argmax_tt


# Scheduler contract: cb_api/cb_scheduler call `cb.forward_batch_tp_inner`.
# Alias the Gemma 4 name so the same scheduler binding works without per-
# backend branching.
forward_batch_tp_inner = forward_batch_gm4_inner


def step_forward_cb(state, token_ids, cur_positions):
    """High-level CB step. Returns a length-B list of int argmax tokens."""
    update_input_buffers_batched(state, token_ids, cur_positions)
    argmax_tt = forward_batch_gm4_inner(state)
    arr = ttnn.to_torch(argmax_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    ttnn.deallocate(argmax_tt)
    # All chips agree on argmax (replicated). Take chip 0 → [B, 1] → [B].
    return [int(x) for x in arr.reshape(-1)[:state.cb_B].tolist()]


if __name__ == "__main__":
    # v1.0 smoke: bootstrap + setup_cb_state at B=2; ensure allocator
    # doesn't OOM and the cb_* buffers are valid.
    import time
    state = base.State()
    base.bootstrap(state)
    t0 = time.time()
    setup_cb_state(state, B=2)
    print(f"[cb v1.0 smoke] setup_cb_state(B=2) in {time.time()-t0:.1f}s")
    print(f"  cb_B={state.cb_B}, blocks/seq={state.cb_blocks_per_seq}, "
          f"total_blocks={state.cb_total_blocks}")
    print(f"  cb_kv_caches_tt: {len(state.cb_kv_caches_tt)} layers")
    print(f"  cb_tok_buf shape={list(state.cb_tok_buf.shape)}")
    print(f"  cb_cur_pos_buf shape={list(state.cb_cur_pos_buf.shape)}")
    update_input_buffers_batched(state, token_ids=[2, 818], cur_positions=[0, 0])
    print("  ✓ update_input_buffers_batched(B=2) OK")
    ttnn.close_device(state.mesh)
