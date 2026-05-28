#!/usr/bin/env python3
"""Continuous-batching server for Qwen3.6-27B — experimental, isolated.

Imports the production server_tp.py machinery (State, bootstrap, weight
upload, _rms_norm_manual, _tp_all_reduce, shape-agnostic mlp_step_tp) and
redefines ONLY the forward as a BATCHED path that runs B concurrent decode
slots. server_tp.py is untouched (zero regression risk).

Design (vLLM, adopted — not reinvented):
  - PagedAttention: per-slot page tables into a shared physical block pool
    (CB2). Batched paged SDPA decode (validated:
    cb_paged_sdpa_batch_isolation.py).
  - DeltaNet recurrence: per-slot recurrent + conv state, manual recurrence
    (validated to batch bit-exact: cb_dn_recurrence_batch_isolation.py).
    Owned_gdn kernel is B=1-only, so CB uses the manual path.
  - Dense MLP: shape-agnostic in the leading dim — reuse base mlp_step_tp.
  - Empty slots: cur_pos=-1 → paged SDPA skips them; we ignore their output.

Validation ladder (run via cb_validate_27b.py):
  1. B=1 batched forward bit-identical to server_tp B=1 forward.
  2. B>1 all-identical-slots match B=1.
  3. B>1 different slots each match its own B=1 reference.

This file is CB1+CB2. The Orca scheduler is CB3 (separate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base  # noqa: E402  — production machinery, untouched
import ttnn  # noqa: E402

_rms_norm_manual = base._rms_norm_manual
_tp_all_reduce = base._tp_all_reduce


# ── Per-slot CB state ──────────────────────────────────────────────────
def setup_cb_state(state, B, blocks_per_seq=None):
    """Allocate batched buffers + per-layer per-slot DN/conv state + a shared
    paged KV block pool with per-slot page tables. Call AFTER base.bootstrap.

    B: number of concurrent slots (fixed batch width, e.g. 32).
    blocks_per_seq: KV blocks reserved per slot (default = current NUM_BLOCKS,
      i.e. each slot gets MAX_POS/BLOCK_SIZE blocks).
    """
    cfg = state.cfg
    mesh = state.mesh
    state.cb_B = B
    BLOCK_SIZE = base.BLOCK_SIZE
    if blocks_per_seq is None:
        blocks_per_seq = base.NUM_BLOCKS  # per slot, same context budget as B=1
    state.cb_blocks_per_seq = blocks_per_seq
    total_blocks = B * blocks_per_seq
    state.cb_total_blocks = total_blocks

    # Per-slot page table [B, blocks_per_seq]: contiguous physical block range.
    page_table_np = np.stack([
        np.arange(b * blocks_per_seq, (b + 1) * blocks_per_seq, dtype=np.int32)
        for b in range(B)
    ], axis=0)
    state.cb_page_table_tt = ttnn.from_torch(
        torch.from_numpy(page_table_np), dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    # Re-allocate per-attn-layer KV caches sized for the B-slot block pool.
    # Cache [total_blocks, NKV_per_chip, BLOCK_SIZE, HEAD_DIM] sharded on dim 1.
    HEAD_DIM = cfg['head_dim']
    n_kv = cfg['n_kv_heads']  # total across mesh; sharded → 1 per chip
    state.cb_kv = {}
    for li, layer in enumerate(state.layers):
        if layer['type'] != 'full_attention':
            continue
        shp = (total_blocks, n_kv, BLOCK_SIZE, HEAD_DIM)
        kc = ttnn.from_torch(torch.zeros(shp), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=mesh,
                             mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        vc = ttnn.from_torch(torch.zeros(shp), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=mesh,
                             mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
        state.cb_kv[li] = {'kc': kc, 'vc': vc}

    # Per-DN-layer per-slot recurrent (ssm) + conv state, batched on dim 0 (B).
    # Mirrors dn['ssm'] [1,NV,K,V] and dn['conv_st'] [CONV_DIM_CHIP,KERNEL-?]
    # but with a leading B. Sharded like the originals.
    from full_layer_tp_probe import (
        NV_PER_CHIP, K_DIM, V_DIM, CONV_DIM_CHIP,
    )
    CONV_K = cfg['conv_kernel']
    state.cb_dn = {}
    for li, layer in enumerate(state.layers):
        if layer['type'] != 'linear_attention':
            continue
        # ssm: [B, NV_PER_CHIP, K_DIM, V_DIM] per chip (replicate over mesh —
        # NV_PER_CHIP already the per-chip head count from the base sharding).
        ssm = ttnn.from_torch(
            torch.zeros(B, NV_PER_CHIP, K_DIM, V_DIM), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        # conv_st: [B, CONV_DIM_CHIP, CONV_K-1] (the running window, base stores
        # CONV_K-1 columns and concats the new col each step).
        conv_st = ttnn.from_torch(
            torch.zeros(B, CONV_DIM_CHIP, CONV_K - 1), dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        state.cb_dn[li] = {'ssm': ssm, 'conv_st': conv_st}

    # Batched input buffers.
    state.cb_tok_buf = ttnn.from_torch(
        torch.zeros(B, 1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state.cb_cur_pos_buf = ttnn.from_torch(
        torch.full((B,), -1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    state.cb_rot_idxs_buf = ttnn.from_torch(
        torch.zeros(B, 1, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    return state


def cb_reset_states(state):
    """Zero all per-slot DN/conv/KV state (fresh start)."""
    for li, d in state.cb_dn.items():
        for key in ('ssm', 'conv_st'):
            t = d[key]
            z = ttnn.mul(t, 0.0)
            ttnn.copy(z, t)
            ttnn.deallocate(z)
    # KV caches: leave as-is; per-slot cur_pos + page table govern reads. A
    # full reset would re-zero; for validation we reset by re-running setup.


def update_input_buffers_batched(state, token_ids, cur_positions):
    """token_ids: list[int] len B; cur_positions: list[int] len B (-1 = empty)."""
    B = state.cb_B
    mesh = state.mesh
    tok = ttnn.from_torch(torch.tensor(token_ids, dtype=torch.int32).reshape(B, 1),
                          layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(tok, state.cb_tok_buf)
    cp = ttnn.from_torch(torch.tensor(cur_positions, dtype=torch.int32).reshape(B),
                         layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(cp, state.cb_cur_pos_buf)
    # rotary index = max(cur_pos, 0) so empty slots read a valid (unused) row.
    rot = [max(p, 0) for p in cur_positions]
    rt = ttnn.from_torch(torch.tensor(rot, dtype=torch.int32).reshape(B, 1),
                         layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
                         mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(rt, state.cb_rot_idxs_buf)


# ── Batched step functions ─────────────────────────────────────────────
def deltanet_step_batched(state, x_tt, dn, li, cfg):
    """Batched DeltaNet step. x_tt [B,HIDDEN]. Per-slot conv/ssm state from
    state.cb_dn[li]. Manual recurrence (owned_gdn is B=1-only). Mirrors
    server_tp.deltanet_step_tp + _deltanet_step_tp_from_inproj with a leading
    B dim. Returns [B,HIDDEN] residual-added output; mutates cb_dn[li] state.
    """
    from full_layer_tp_probe import (
        K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )
    B = state.cb_B
    HIDDEN = cfg['hidden']
    CONV_K = cfg['conv_kernel']
    slot = state.cb_dn[li]

    # 1. pre-norm + in_proj (matmul batches over leading B)
    h_tt = _rms_norm_manual(x_tt, dn['input_norm'], EPS, HIDDEN)
    all_tt = ttnn.linear(h_tt, dn['w_in'])            # [B, IN_PROJ_OUT_CHIP]
    ttnn.deallocate(h_tt)

    # 3. slices on the last dim (dim 1 of [B, N])
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [B, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [B, CONV_DIM_CHIP + VAL_DIM_CHIP])
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                      [B, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                      [B, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    ttnn.deallocate(all_tt)

    # 4. conv1d: window = [conv_st (B,CONV,K-1) | cur (B,CONV,1)] → mul w_conv → sum → silu
    cur_col = ttnn.reshape(mixed_qkv, [B, CONV_DIM_CHIP, 1])
    ttnn.deallocate(mixed_qkv)
    conv_input = ttnn.concat([slot['conv_st'], cur_col], dim=-1)   # [B,CONV,K]
    ttnn.deallocate(cur_col)
    # w_conv is [CONV_DIM_CHIP, K]; reshape to [1,CONV,K] to broadcast over B.
    w_conv_b = ttnn.reshape(dn['w_conv'], [1, CONV_DIM_CHIP, CONV_K])
    conv_prod = ttnn.mul(conv_input, w_conv_b)
    ttnn.deallocate(w_conv_b)
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))              # [B,CONV]
    ttnn.deallocate(conv_prod)
    # new conv state = last K-1 columns
    conv_state_new = ttnn.slice(conv_input, [0, 0, 1], [B, CONV_DIM_CHIP, CONV_K])
    ttnn.deallocate(conv_input)
    ttnn.copy(conv_state_new, slot['conv_st'])  # commit per-slot conv state
    ttnn.deallocate(conv_state_new)

    # 5. q/k/v head slices on last dim
    q_flat = ttnn.slice(conv_out, [0, 0], [B, KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [0, KEY_DIM_CHIP], [B, 2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [0, 2 * KEY_DIM_CHIP], [B, CONV_DIM_CHIP])
    ttnn.deallocate(conv_out)

    def gqa_b(t, n_kh, d):
        # [B, n_kh*d] → [B, n_kh, 1, d] → repeat N_REP → [B, n_kh*N_REP, d]
        t2 = ttnn.reshape(t, [B, n_kh, 1, d])
        t3 = ttnn.repeat(t2, ttnn.Shape([1, 1, N_REP, 1]))
        return ttnn.reshape(t3, [B, n_kh * N_REP, d])
    q = gqa_b(q_flat, NK_PER_CHIP, K_DIM)    # [B, NV_PER_CHIP, K_DIM]
    k = gqa_b(k_flat, NK_PER_CHIP, K_DIM)
    v = ttnn.reshape(v_flat, [B, NV_PER_CHIP, V_DIM])
    ttnn.deallocate(q_flat); ttnn.deallocate(k_flat); ttnn.deallocate(v_flat)

    # 6. QK L2-norm (rms_norm over last dim; rank-agnostic over leading B,NV)
    EPS_RMS = EPS / K_DIM
    q = _rms_norm_manual(q, dn['q_l2_scale'], EPS_RMS, K_DIM)
    k = _rms_norm_manual(k, dn['k_l2_scale'], EPS_RMS, K_DIM)

    # 7. decay/gate (manual path). a_tt/b_tt [B,NV]; dt_bias/A_log [NV] → [1,NV]
    dt_bias_b = ttnn.reshape(dn['dt_bias'], [1, NV_PER_CHIP])
    A_log_b = ttnn.reshape(dn['A_log'], [1, NV_PER_CHIP])
    a_biased = ttnn.add(a_tt, dt_bias_b)
    softplus_a = ttnn.softplus(a_biased)
    ttnn.deallocate(a_biased)
    neg_exp_alog = ttnn.neg(ttnn.exp(A_log_b))
    g = ttnn.mul(softplus_a, neg_exp_alog)
    ttnn.deallocate(softplus_a)
    beta = ttnn.sigmoid(b_tt)                              # [B,NV]
    decay = ttnn.reshape(ttnn.exp(g), [B, NV_PER_CHIP, 1, 1])
    ttnn.deallocate(g)
    ttnn.deallocate(a_tt); ttnn.deallocate(b_tt)

    # 8. recurrence (manual — validated to batch bit-exact)
    H_4d = slot['ssm']                                    # [B,NV,K,V]
    H_decayed = ttnn.mul(H_4d, decay)
    ttnn.deallocate(decay)
    k_col = ttnn.reshape(k, [B, NV_PER_CHIP, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [B, NV_PER_CHIP, V_DIM])
    v_3d = ttnn.reshape(v, [B, NV_PER_CHIP, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [B, NV_PER_CHIP, 1]))
    ttnn.deallocate(kv_mem); ttnn.deallocate(beta)
    H_new = ttnn.add(H_decayed,
                     ttnn.mul(k_col, ttnn.reshape(delta, [B, NV_PER_CHIP, 1, V_DIM])))
    ttnn.deallocate(H_decayed); ttnn.deallocate(delta)
    q_col = ttnn.reshape(q, [B, NV_PER_CHIP, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [B, VAL_DIM_CHIP])
    ttnn.copy(H_new, slot['ssm'])  # commit per-slot recurrent state
    ttnn.deallocate(H_new); ttnn.deallocate(q_col); ttnn.deallocate(k_col)
    ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v); ttnn.deallocate(v_3d)

    # 9. Per-head RMSNormGated: norm over V_DIM (head dim, NOT full VAL_DIM_CHIP)
    # then * silu(z), matching production deltanet_step_tp lines 820-826.
    out_ph = ttnn.reshape(out, [B, NV_PER_CHIP, V_DIM])
    ttnn.deallocate(out)
    out_normed = _rms_norm_manual(out_ph, dn['linear_attn_norm'], EPS, V_DIM)
    ttnn.deallocate(out_ph)
    z_ph = ttnn.reshape(z_tt, [B, NV_PER_CHIP, V_DIM])
    ttnn.deallocate(z_tt)
    silu_z = ttnn.silu(z_ph)
    ttnn.deallocate(z_ph)
    gated = ttnn.reshape(ttnn.mul(out_normed, silu_z), [B, VAL_DIM_CHIP])
    ttnn.deallocate(out_normed); ttnn.deallocate(silu_z)
    partial = ttnn.linear(gated, dn['w_out'])             # [B,HIDDEN] partial
    ttnn.deallocate(gated)
    reduced = _tp_all_reduce(state, partial)
    ttnn.deallocate(partial)
    res = ttnn.add(x_tt, reduced)
    ttnn.deallocate(reduced)
    return res


if __name__ == "__main__":
    print("server_tp_cb is a library module. Run cb_validate_27b.py to test.",
          flush=True)
