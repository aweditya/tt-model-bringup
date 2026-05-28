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


if __name__ == "__main__":
    print("server_tp_cb is a library module. Run cb_validate_27b.py to test.",
          flush=True)
