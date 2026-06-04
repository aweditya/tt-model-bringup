#!/usr/bin/env python3
"""Continuous-batching server for Qwen3.6-27B — experimental, isolated.

Imports the production server_tp.py machinery (State, bootstrap, weight
upload, _rms_norm_manual, _tp_all_reduce, shape-agnostic mlp_step_tp) and
redefines ONLY the forward as a BATCHED path that runs B concurrent decode
slots. server_tp.py is untouched (zero regression risk).

Design (vLLM, adopted — not reinvented):
  - PagedAttention: per-slot page tables into a shared physical block pool
    (CB2). Batched paged SDPA decode (validated:
    cb/isolate/paged_sdpa.py).
  - DeltaNet recurrence: per-slot recurrent + conv state, manual recurrence
    (validated to batch bit-exact: cb/isolate/dn_recurrence.py).
    Owned_gdn kernel is B=1-only, so CB uses the manual path.
  - Dense MLP: shape-agnostic in the leading dim — reuse base mlp_step_tp.
  - Empty slots: cur_pos=-1 → paged SDPA skips them; we ignore their output.

Validation ladder (run via cb/validate/forward.py):
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

    Default fast-path flags (the CB forward consults these via getattr):
      - cb_dn_recurrence_mode = "owned_gdn" → routes the DN step through the
        custom kernel (qwen36_gdn_decode_owned). 376 / 593 tok/s at B=32/64
        in `research/27b_cb_scope.md:687-688` requires this. The base
        single-stream defaults (server_tp.MeshServerState.__init__) are
        already "owned_gdn"; the CB analogue must match.
      - cb_conv_mode = "shiftacc" → 3-column shift-accumulate conv1d
        (DNK-G4 default; produces the 593 number).

    Both are explicitly set here so the CB engine never silently falls
    through to the slow `manual` defaults. Caller can override AFTER
    setup_cb_state if needed for an A/B test, but should not leave them
    unset.
    """
    cfg = state.cfg
    mesh = state.mesh
    state.cb_B = B
    if not hasattr(state, 'cb_dn_recurrence_mode'):
        state.cb_dn_recurrence_mode = "owned_gdn"
    if not hasattr(state, 'cb_conv_mode'):
        state.cb_conv_mode = "shiftacc"
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
        # conv state as CONV_K-1 SEPARATE [B, CONV_DIM_CHIP] columns (padding-free
        # shift register; DNK-G4). A single [B,C,K-1] tensor would pad the K-1
        # dim to a full 32-row tile (8× waste) — the very cost we're removing.
        # Shift-accumulate reads these columns directly (no padded slices).
        conv_cols = [
            ttnn.from_torch(torch.zeros(B, CONV_DIM_CHIP), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            for _ in range(CONV_K - 1)]
        state.cb_dn[li] = {'ssm': ssm, 'conv_cols': conv_cols}

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
    setup_cb_write_mem_cfg(state)  # B-core HEIGHT_SHARDED L1 for paged KV writes

    # paged SDPA decode parallelizes over batch → requires num_cores >= B
    # (sdpa_decode_program_factory: "num_cores_available >= B"). Production's
    # progcfg is CoreCoord(4,4)=16 cores (fine for B<=16). Size a CB grid that
    # covers B. SDPA only uses B of the cores; the rest idle. q/k chunking is
    # auto (0) and per-batch-element, so a larger grid keeps per-slot numerics
    # identical to production (verified: B=1 logit_cos stays 1.0).
    grid = mesh.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    if B > ncores:
        raise RuntimeError(f"CB B={B} exceeds {ncores} available cores for SDPA decode")
    state.cb_sdpa_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False)
    return state


def cb_reset_states(state):
    """Zero all per-slot DN/conv/KV state (fresh start)."""
    for li, d in state.cb_dn.items():
        for t in [d['ssm'], *d['conv_cols']]:
            z = ttnn.mul(t, 0.0)
            ttnn.copy(z, t)
            ttnn.deallocate(z)
    # KV caches: leave as-is; per-slot cur_pos + page table govern reads. A
    # full reset would re-zero; for validation we reset by re-running setup.


def cb_reset_slots(state, slot_ids):
    """Reset ONLY the given slots' DN recurrent + conv state (mid-batch
    admission of a new sequence — Mamba-style per-slot state slot reuse).

    Masked multiply: keep=1 for live slots, 0 for admitted slots, broadcast
    over the head/dim axes. The other slots' state is untouched (validated by
    the ragged test — slot isolation holds across a reset).

    KV needs NO reset: per-slot cur_pos bounds the SDPA read, so a sequence
    restarting at pos 0 overwrites its own blocks as it prefills and never reads
    stale data. Only the position-unbounded DN accumulator must be cleared.
    """
    B = state.cb_B
    mesh = state.mesh
    keep = torch.ones(B, dtype=torch.float32)
    for s in slot_ids:
        keep[s] = 0.0
    mask4 = ttnn.from_torch(keep.reshape(B, 1, 1, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    mask2 = ttnn.from_torch(keep.reshape(B, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    for li, d in state.cb_dn.items():
        for t, mask in [(d['ssm'], mask4), *[(c, mask2) for c in d['conv_cols']]]:
            masked = ttnn.mul(t, mask)
            ttnn.copy(masked, t)
            ttnn.deallocate(masked)
    ttnn.deallocate(mask4); ttnn.deallocate(mask2)


def cb_prefill_transplant(state, slot_s, L):
    """S2.3+S2.6 — copy POST-PREFILL production state into CB slot `slot_s`,
    entirely on-device (no host round-trip).

    Precondition: caller has just run S1a (`forward_prefill_chunked_tp`) +
    S1b chunked DN on a fresh `_reset_state_buffers(state)`, so the production
    per-layer state (attn['kc']/['vc'] paged blocks, dn['ssm'], dn['conv_st'])
    holds the prompt's full forward result.

    Postcondition: CB slot `slot_s` of `cb_kv[li]` / `cb_dn[li]['ssm']` /
    `cb_dn[li]['conv_cols']` matches the production state.

    Mechanism (on-device):
      KV:  slice prod kc[0:n_used_blocks] → transpose to seq-major →
           paged_fill_cache(cb_kv, src, cb_page_table_tt, batch_idx=slot_s).
      DN ssm:  cb_ssm * keep4_mask + prod_ssm * slot4_mask, where keep4 is 1
               on rows != slot_s and slot4 = 1 - keep4. Broadcast does the row
               splice without a per-element scatter.
      DN conv_cols[k]: same mask pattern with [B, 1] mask; prod_conv col k is
                      sliced and reshaped to [1, CONV_DIM_CHIP] for broadcast.
      cur_pos[slot_s] = L: still a small host hop (one int32 [B] tensor); the
                          cost is negligible (<5 ms) and the read of other
                          slots' positions must come from somewhere.
    """
    import torch  # masks are constructed host-side
    from full_layer_tp_probe import CONV_DIM_CHIP
    cfg = state.cfg
    mesh = state.mesh
    B = state.cb_B
    blocks_per_seq = state.cb_blocks_per_seq
    BLOCK_SIZE = base.BLOCK_SIZE
    n_used_blocks = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    if n_used_blocks > blocks_per_seq:
        raise ValueError(f"prefill L={L} needs {n_used_blocks} blocks, "
                         f"slot has {blocks_per_seq}")
    HEAD_DIM = cfg['head_dim']
    NKV_PER_CHIP = cfg['n_kv_heads'] // 4
    if NKV_PER_CHIP != 1:
        raise NotImplementedError("on-device transplant assumes NKV_PER_CHIP=1 "
                                  "(Qwen3.6-27B); >1 needs a different permute")
    CONV_K = cfg['conv_kernel']

    # Splice masks: keep = 1 on rows != slot_s, slot = 1 only on row slot_s.
    keep_h = torch.ones(B, dtype=torch.float32); keep_h[slot_s] = 0.0
    slot_h = 1.0 - keep_h
    keep4 = ttnn.from_torch(keep_h.reshape(B, 1, 1, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    slot4 = ttnn.from_torch(slot_h.reshape(B, 1, 1, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    keep2 = ttnn.from_torch(keep_h.reshape(B, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    slot2 = ttnn.from_torch(slot_h.reshape(B, 1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    for li, layer in enumerate(state.layers):
        if layer['type'] == 'full_attention':
            attn = layer['attn']
            kv = state.cb_kv[li]
            for key in ('kc', 'vc'):
                # Production layout per chip: [NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM].
                # Take first n_used_blocks (slot 0's populated range — prod page
                # table is contiguous 0..NUM_BLOCKS-1 by construction).
                src = ttnn.slice(attn[key], [0, 0, 0, 0],
                                 [n_used_blocks, 1, BLOCK_SIZE, HEAD_DIM])
                # paged_fill_cache wants [1, NKV_PER_CHIP, seq_len, HEAD_DIM].
                # With NKV_PER_CHIP=1: swap blocks-dim and NKV-dim via transpose.
                src_t = ttnn.transpose(src, 0, 1)        # [1, n_used, BLK, D]
                src_seq = ttnn.reshape(src_t,
                    [1, 1, n_used_blocks * BLOCK_SIZE, HEAD_DIM])
                src_L = ttnn.slice(src_seq, [0, 0, 0, 0], [1, 1, L, HEAD_DIM])
                ttnn.experimental.paged_fill_cache(kv[key], src_L,
                                                   state.cb_page_table_tt,
                                                   batch_idx=slot_s)
                # Views: src_t/src_seq/src_L alias src — let scope free. src is
                # independent of attn[key].
                ttnn.deallocate(src_L)
                ttnn.deallocate(src_seq)
                ttnn.deallocate(src_t)
                ttnn.deallocate(src)
        else:
            dn = layer['dn']
            slot = state.cb_dn[li]
            # ssm splice: cb[!s] preserved, cb[s] := prod (broadcast over B).
            masked_cb = ttnn.mul(slot['ssm'], keep4)            # [B,NV,K,V]
            prod_at_slot = ttnn.mul(dn['ssm'], slot4)           # broadcasts [1,NV,K,V]×[B,1,1,1]
            new_ssm = ttnn.add(masked_cb, prod_at_slot)
            ttnn.copy(new_ssm, slot['ssm'])
            ttnn.deallocate(new_ssm)
            ttnn.deallocate(prod_at_slot)
            ttnn.deallocate(masked_cb)
            # conv splice: per col k, take prod_conv[:, k] → broadcast row.
            for k in range(CONV_K - 1):
                prod_col = ttnn.slice(dn['conv_st'], [0, k],
                                      [CONV_DIM_CHIP, k + 1])  # [C, 1]
                prod_col_2d = ttnn.reshape(prod_col, [1, CONV_DIM_CHIP])
                masked_c = ttnn.mul(slot['conv_cols'][k], keep2)
                prod_at_c = ttnn.mul(prod_col_2d, slot2)        # [B, C] via broadcast
                new_c = ttnn.add(masked_c, prod_at_c)
                ttnn.copy(new_c, slot['conv_cols'][k])
                ttnn.deallocate(new_c)
                ttnn.deallocate(prod_at_c)
                ttnn.deallocate(masked_c)
                ttnn.deallocate(prod_col_2d)
                ttnn.deallocate(prod_col)

    ttnn.deallocate(slot2); ttnn.deallocate(keep2)
    ttnn.deallocate(slot4); ttnn.deallocate(keep4)

    # cur_pos[slot_s] = L. Tiny tensor; read+write via the existing pattern.
    cur_full = ttnn.to_torch(state.cb_cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).int()[:B].clone()
    cur_full[slot_s] = L
    host = ttnn.from_torch(cur_full, dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(host, state.cb_cur_pos_buf)


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
    # cb_dn_skip: profiling-only set ({'conv','qknorm','recur','outgate'}); a
    # skipped sub-op is replaced by a shape-valid passthrough (TIMING ablation —
    # corrupts values, used only to rank the DN vector ops). Default empty → no
    # change. Used by cb/profile/dn.py.
    dn_skip = getattr(state, 'cb_dn_skip', None) or set()

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

    # 4. conv1d — SHIFT-ACCUMULATE (DNK-G4): out = silu(Σ_k w[:,k]·window_k) as
    # muls+adds on [B,C] tiles, NO K-dim tile padding (28.76× vs the [B,C,K]
    # form, which padded K=4→32). State = 3 separate padding-free [B,C] columns
    # (slot['conv_cols'], oldest→newest = c0,c1,c2); window = [c0,c1,c2,cur].
    # Weight taps w_conv[:,k] as [1,C] (sliced from a one-time transpose; tiny,
    # batch-independent). State shifts in place: c0←c1, c1←c2, c2←cur — safe on
    # the in-order command queue (all acc reads complete before the shift writes).
    # DEFAULT "kdim" is BIT-IDENTICAL to production (reconstruct the [B,C,K]
    # window from the 3 columns + sum-reduce) — the correct path for long context.
    # "shiftacc" is the fast (28.76× isolated) path but its bf16 op-order differs
    # from the sum-reduce and DRIFTS over a sequence (DNK-G4: argmax flip by
    # pos 31, worst logit_cos 0.963) — opt-in, gated on the needle test.
    cols = slot['conv_cols']                 # [c0, c1, c2], each [B, C], oldest→newest
    cur = mixed_qkv                          # [B, C]
    if 'conv' in dn_skip:
        conv_out = mixed_qkv  # profiling passthrough
    elif getattr(state, 'cb_conv_mode', 'shiftacc') == 'shiftacc':
        if 'w_conv_T' not in dn:
            dn['w_conv_T'] = ttnn.transpose(dn['w_conv'], -2, -1)  # [K, C], one-time
        wT = dn['w_conv_T']
        def _tap(kk):  # w_conv[:,kk] as [1,C] — view of cached wT, batch-indep
            return ttnn.reshape(ttnn.slice(wT, [kk, 0], [kk + 1, CONV_DIM_CHIP]),
                                [1, CONV_DIM_CHIP])
        acc = ttnn.mul(cols[0], _tap(0))
        for j in range(1, CONV_K - 1):
            term = ttnn.mul(cols[j], _tap(j)); nacc = ttnn.add(acc, term)
            ttnn.deallocate(acc); ttnn.deallocate(term); acc = nacc
        term = ttnn.mul(cur, _tap(CONV_K - 1)); nacc = ttnn.add(acc, term)
        ttnn.deallocate(acc); ttnn.deallocate(term); acc = nacc
        conv_out = ttnn.silu(acc); ttnn.deallocate(acc)
    else:  # kdim — bit-identical to production
        wnd = ttnn.concat([ttnn.reshape(cols[0], [B, CONV_DIM_CHIP, 1]),
                           ttnn.reshape(cols[1], [B, CONV_DIM_CHIP, 1]),
                           ttnn.reshape(cols[2], [B, CONV_DIM_CHIP, 1]),
                           ttnn.reshape(cur, [B, CONV_DIM_CHIP, 1])], dim=-1)  # [B,C,K]
        w_conv_b = ttnn.reshape(dn['w_conv'], [1, CONV_DIM_CHIP, CONV_K])
        prod = ttnn.mul(wnd, w_conv_b)
        conv_out = ttnn.silu(ttnn.sum(prod, dim=-1))
        ttnn.deallocate(prod); ttnn.deallocate(wnd)
    # shift state in place (both paths; in-order CQ: acc reads done before writes)
    if 'conv' not in dn_skip:
        ttnn.copy(cols[1], cols[0])          # c0 ← c1
        ttnn.copy(cols[2], cols[1])          # c1 ← c2
        ttnn.copy(cur, cols[2])              # c2 ← cur
        ttnn.deallocate(mixed_qkv)

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
    if 'qknorm' not in dn_skip:
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

    # 8. recurrence — manual (validated bit-exact) OR batched owned_gdn kernel.
    if 'recur' in dn_skip:
        # passthrough: materialize out from v (independent copy) and skip the
        # recurrence. mul(v,1.0) so the downstream dealloc(out) is safe.
        out = ttnn.reshape(ttnn.mul(v, 1.0), [B, VAL_DIM_CHIP])
        ttnn.deallocate(decay); ttnn.deallocate(beta)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
    elif getattr(state, 'cb_dn_recurrence_mode', 'manual') == 'owned_gdn':
        # FOLD batch into slots (each slot's recurrence is independent) and call
        # the owned-GDN fused kernel with debug_mode=10 (batched-safe two-CB
        # output; see experiments/kernel_patches/qwen36_gdn_decode_owned/). The
        # op updates state in place — passing the folded VIEW of slot['ssm']
        # commits the next state directly (create_output_tensors returns the
        # state tensor). out is flat [1, S*V_DIM] → [B, VAL_DIM_CHIP].
        S = B * NV_PER_CHIP
        H_in = ttnn.reshape(slot['ssm'], [1, S, K_DIM, V_DIM])
        q_f = ttnn.reshape(q, [1, S, 1, K_DIM])
        k_f = ttnn.reshape(k, [1, S, 1, K_DIM])
        v_f = ttnn.reshape(v, [1, S, 1, V_DIM])
        alpha_f = ttnn.reshape(decay, [1, S, 1, 1])
        beta_f = ttnn.reshape(beta, [1, S, 1, 1])
        H_new, out_f = ttnn.experimental.qwen36_gdn_decode_owned(
            H_in, q_f, k_f, v_f, alpha_f, beta_f,
            native_io=True, debug_mode=10, output_memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out_f, [B, VAL_DIM_CHIP])
        # H_in/H_new alias slot['ssm'] (in-place commit) — do NOT deallocate.
        # out aliases out_f — keep. decay/beta/q/k/v consumed by the op.
        ttnn.deallocate(decay); ttnn.deallocate(beta)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
    else:
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
    if 'outgate' in dn_skip:
        out_normed = out_ph  # passthrough (skip output RMSNorm)
    else:
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


def gated_attn_step_batched(state, x_tt, attn, li, cos_tt, sin_tt, cfg):
    """Batched gated attention. x_tt [B,HIDDEN]; cos/sin [B,ROTARY_DIM].
    Per-slot KV in state.cb_kv[li], per-slot pos in state.cb_cur_pos_buf,
    page table state.cb_page_table_tt. Mirrors server_tp.gated_attn_step_tp
    with leading B. All primitives validated (RoPE broadcast + paged
    write/read isolations). Returns [B,HIDDEN] residual-added output."""
    B = state.cb_B
    HIDDEN = cfg['hidden']
    HEAD_DIM = cfg['head_dim']
    N_Q = cfg['n_q_heads']; N_KV = cfg['n_kv_heads']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
    NQ_PER_CHIP = N_Q // 4
    NKV_PER_CHIP = N_KV // 4
    QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM
    KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
    EPS = 1e-6
    TILE_H = 32
    assert NKV_PER_CHIP == 1, "paged SDPA per-chip assumes 1 KV head"
    half = ROTARY_DIM // 2
    kv = state.cb_kv[li]

    h_tt = _rms_norm_manual(x_tt, attn['input_norm'], EPS, HIDDEN)
    all_tt = ttnn.linear(h_tt, attn['w_qkv'])           # [B, QG+2KV]
    ttnn.deallocate(h_tt)
    # CRITICAL (view-decay): ttnn.slice/reshape return VIEWS. q_tt/gate_tt/k_tt/
    # v_tt alias all_tt's buffer until materialized (q_tt/k_tt by rms_norm below;
    # gate_tt/v_tt only when consumed). Production (gated_attn_step_tp) NEVER
    # deallocates all_tt/qg/k_flat/v_flat for exactly this reason — freeing them
    # here corrupts the views (was the CB attn-divergence bug). Let scope free them.
    qg = ttnn.slice(all_tt, [0, 0], [B, QG_DIM_CHIP])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [B, QG_DIM_CHIP + KV_DIM_CHIP])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                        [B, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
    qg = ttnn.reshape(qg, [B, NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0, 0], [B, NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, 0, HEAD_DIM], [B, NQ_PER_CHIP, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_flat, [B, NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [B, NKV_PER_CHIP, HEAD_DIM])

    q_tt = _rms_norm_manual(q_tt, attn['q_norm'], EPS, HEAD_DIM)
    k_tt = _rms_norm_manual(k_tt, attn['k_norm'], EPS, HEAD_DIM)

    cos_b = ttnn.reshape(cos_tt, [B, 1, ROTARY_DIM])
    sin_b = ttnn.reshape(sin_tt, [B, 1, ROTARY_DIM])
    def rope_b(t, n_heads):
        rot = ttnn.slice(t, [0, 0, 0], [B, n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, 0, ROTARY_DIM], [B, n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0, 0], [B, n_heads, half])
        x2 = ttnn.slice(rot, [0, 0, half], [B, n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.add(ttnn.mul(rot, cos_b),
                           ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_b))
        out = ttnn.concat([rotated, passthru], dim=-1)
        # rot/passthru/x1/x2 are VIEWS of t — freeing them frees t's buffer (and
        # the caller frees t below). Only free the independent temporaries here.
        ttnn.deallocate(neg_x2); ttnn.deallocate(rotated)
        return out
    q_r = rope_b(q_tt, NQ_PER_CHIP); ttnn.deallocate(q_tt)
    k_r = rope_b(k_tt, NKV_PER_CHIP); ttnn.deallocate(k_tt)
    # cos_b/sin_b are VIEWS of the per-forward cos_tt/sin_tt shared across all
    # 16 attn layers — do NOT deallocate (would free cos_tt, crash next layer).

    # Paged KV write: input [1, B, TILE_H, HEAD_DIM] height-sharded on B cores.
    def shard_write(t):  # t [B, NKV=1, HEAD_DIM]
        t4 = ttnn.reshape(t, [1, B, 1, HEAD_DIM])
        t4 = ttnn.pad(t4, [[0, 0], [0, 0], [0, TILE_H - 1], [0, 0]], value=0.0)
        return ttnn.to_memory_config(t4, state.cb_write_mem_cfg)
    # V is NOT RoPE-rotated — write the original v_tt.
    k_sh = shard_write(k_r); v_sh = shard_write(v_tt)
    ttnn.experimental.paged_update_cache(kv['kc'], k_sh,
                                         update_idxs_tensor=state.cb_cur_pos_buf,
                                         page_table=state.cb_page_table_tt)
    ttnn.experimental.paged_update_cache(kv['vc'], v_sh,
                                         update_idxs_tensor=state.cb_cur_pos_buf,
                                         page_table=state.cb_page_table_tt)
    ttnn.deallocate(k_sh); ttnn.deallocate(v_sh)
    ttnn.deallocate(k_r)  # k_r is an independent RoPE output — safe to free.
    # Do NOT deallocate v_tt: it's a VIEW of all_tt, which gate_tt also aliases
    # and _attn_finish still needs. Freeing it would crash the gate (view-decay).

    # Paged SDPA decode + gate + out_proj + residual (q_r is the query).
    return _attn_finish(state, x_tt, q_r, gate_tt, kv, attn, B, NQ_PER_CHIP, HEAD_DIM)


def _attn_finish(state, x_tt, q_r, gate_tt, kv, attn, B, NQ_PER_CHIP, HEAD_DIM):
    # view-decay: q_sdpa/attn_ph/flat are reshape VIEWS — do not free their
    # sources while the view is live (frees the buffer the view reads). gate_tt
    # is a VIEW of all_tt; let scope free all_tt. Only independent tensors freed.
    q_sdpa = ttnn.reshape(q_r, [1, B, NQ_PER_CHIP, HEAD_DIM])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_sdpa, kv['kc'], kv['vc'],
        cur_pos_tensor=state.cb_cur_pos_buf,
        page_table_tensor=state.cb_page_table_tt,
        program_config=state.cb_sdpa_progcfg,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    attn_ph = ttnn.reshape(attn_out, [B, NQ_PER_CHIP, HEAD_DIM])
    gated = ttnn.mul(attn_ph, ttnn.sigmoid(gate_tt))
    flat = ttnn.reshape(gated, [B, NQ_PER_CHIP * HEAD_DIM])
    partial = ttnn.linear(flat, attn['w_o'])
    reduced = _tp_all_reduce(state, partial)
    ttnn.deallocate(partial)
    res = ttnn.add(x_tt, reduced)
    ttnn.deallocate(reduced)
    return res


def setup_cb_write_mem_cfg(state):
    """Build the B-core HEIGHT_SHARDED L1 mem config for paged KV writes
    (validated in cb/isolate/paged_update_cache.py)."""
    B = state.cb_B
    HEAD_DIM = state.cfg['head_dim']
    grid = state.mesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [32, HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR)
    state.cb_write_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def forward_batch_tp_inner(state, return_logits=False, return_topk=None):
    """Batched decode forward over B slots. Reads cb_tok_buf [B,1],
    cb_cur_pos_buf [B], cb_rot_idxs_buf [B,1]. Returns per-slot argmax [B]."""
    cfg = state.cfg
    HIDDEN = cfg['hidden']
    B = state.cb_B
    embed_out = ttnn.embedding(state.cb_tok_buf, state.embed_tt,
                               layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_tt = ttnn.reshape(embed_out, [B, HIDDEN])
    cos_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.cos_table_tt,
                             layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_raw = ttnn.embedding(state.cb_rot_idxs_buf, state.sin_table_tt,
                             layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cos_tt = ttnn.reshape(cos_raw, [B, state.rotary_dim])
    sin_tt = ttnn.reshape(sin_raw, [B, state.rotary_dim])
    # cb_skip_blocks: profiling-only set ({'dn','attn','mlp'}); a skipped block
    # is a no-op (x passes through). Default empty → zero behaviour change. Used
    # by cb/profile/blocks.py to attribute per-token compute via trace timing.
    skip = getattr(state, 'cb_skip_blocks', None) or set()
    for li, layer in enumerate(state.layers):
        if layer['type'] == 'linear_attention':
            if 'dn' not in skip:
                x_tt = deltanet_step_batched(state, x_tt, layer['dn'], li, cfg)
        else:
            if 'attn' not in skip:
                x_tt = gated_attn_step_batched(state, x_tt, layer['attn'], li, cos_tt, sin_tt, cfg)
        if 'mlp' not in skip:
            x_tt = base.mlp_step_tp(state, x_tt, layer['mlp'])  # shape-agnostic
    x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
    # P22 vocab-sharded LM head + on-device argmax, batched over B. Mirrors
    # server_tp.forward_token_tp_inner:1742-1748 with leading B.
    # Match production lm_head (forward_token_tp_inner): NO intermediate
    # deallocates — slice returns a VIEW of gathered; freeing gathered before
    # untilize hits the view-decay rule ("Tensor is not allocated"). lm_head
    # runs once per forward so the few intermediates are fine to leave.
    sharded = ttnn.linear(x_tt, state.lm_head_tt)        # [B, VOCAB_PAD/NCHIPS]/chip
    ttnn.deallocate(x_tt)
    gathered = ttnn.all_gather(sharded, dim=-1)           # [B, VOCAB_PAD] replicated
    sliced = ttnn.slice(gathered, [0, 0], [B, state.vocab_size])  # TILE layout
    if return_topk is not None:
        # W2: on-device top-k for sampling. Read back [B, K] instead of
        # [B, vocab] (~2000× less data at K=128), so the per-slot host
        # sample math runs over K elements not 248k. Sorted=True so index 0
        # is the argmax (free greedy path). topk requires TILE layout — keep
        # sliced (pre-untilize) for it.
        top_vals, top_idxs = ttnn.topk(sliced, k=int(return_topk), dim=-1,
                                        largest=True, sorted=True)
        return (top_vals, top_idxs)
    rm = ttnn.untilize(sliced, use_multicore=True)
    if return_logits:
        return rm
    argmax_tt = ttnn.argmax(rm, dim=-1, keepdim=True, use_multicore=True)
    return argmax_tt


if __name__ == "__main__":
    print("server_tp_cb is a library module. Run cb/validate/forward.py to test.",
          flush=True)
