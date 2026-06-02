"""server_35b_cb.py — CB layer for Qwen3.6-35B-A3B.

Analogue of `server_tp_cb.py` for the 35B-A3B MoE hybrid.

v0 strategy (CB35-1): B=1 ONLY. Each function wraps the existing
single-stream 35B paths from `server_35b_ttnn` (step_forward_inner,
update_input_buffers, dn_caches_tt, kv_caches_tt). The point is to
validate end-to-end CB integration (cb_engine + cb_scheduler driving
35B forwards through /v1/*) before tackling true B>1 batching.

v1 (CB35-2) generalizes to B>1 with batched DN / attn / MoE primitives.
Plan: research/35b_cb_bringup_plan.md.

Imported by cb_api when TT_BACKEND=35b (MM1).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import server_35b_ttnn as base  # noqa: E402
import ttnn  # noqa: E402

# Constants re-exported from base.
NV_PER_CHIP = base.NV_PER_CHIP        # 8
NK_PER_CHIP = base.NK_PER_CHIP        # 4
HEAD_K_DIM = base.HEAD_K_DIM          # 128
HEAD_V_DIM = base.HEAD_V_DIM          # 128
CONV_DIM_CHIP = base.CONV_DIM_CHIP
CONV_KERNEL = base.CONV_KERNEL        # 4
HIDDEN = base.HIDDEN                  # 2048
NQ_PER_CHIP = base.NQ_PER_CHIP        # 4
NUM_KV_HEADS = base.NUM_KV_HEADS      # 2
HEAD_DIM_ATTN = base.HEAD_DIM_ATTN    # 256
ROTARY_DIM = base.ROTARY_DIM          # 64
E_LOCAL = base.E_LOCAL                # 64
TOP_K = base.TOP_K                    # 8
MOE_INTER_CHIP = base.MOE_INTER_CHIP
NCHIPS = base.NCHIPS                  # 4
VOCAB = base.VOCAB                    # 248320


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def setup_cb_state(state, B, blocks_per_seq=None):
    """Allocate per-slot DN + KV caches and per-iter input buffers for B slots.

    v0 fast path (B=1): aliases the existing single-stream
    state.dn_caches_tt[li] / state.kv_caches_tt[li] as cb_dn[li] / cb_kv[li].
    No extra alloc; forward delegates to base.step_forward_inner.

    v1+ path (B>1): allocates separate B-leading per-slot caches plus a
    per-slot page table for paged KV. Forward will read/write through
    cb_dn / cb_kv (B=1 single-stream caches still exist but unused).

    Layouts (per chip):
      cb_dn[li]["cs"]:   [B, CONV_DIM_CHIP, KERNEL]   (each chip a column shard, dim=0)
      cb_dn[li]["rs"]:   [B, NV_PER_CHIP, K, V]       (each chip a head shard, dim=0)
      cb_kv[li]["kc"]:   [B*sdpa_num_blocks, 1, BLOCK_SIZE, HEAD_DIM]  (dim=1 shard)
      cb_kv[li]["vc"]:   same shape as kc
      cb_page_table_tt:  [B, sdpa_num_blocks] int32 — slot s owns blocks
                         [s*nb, (s+1)*nb)
    """
    cfg = state.text_cfg
    mesh = state.mesh
    state.cb_B = B

    # base.bootstrap doesn't call reset_caches_ttnn (the single-stream
    # main() does). Call it here so per-layer caches exist for v0 alias path.
    if state.dn_caches_tt is None or state.kv_caches_tt is None:
        state.reset_caches_ttnn()

    state.cb_dn = {}
    state.cb_kv = {}

    if B == 1:
        # v0 fast path: alias existing single-stream caches. No extra alloc.
        for li in range(cfg.num_hidden_layers):
            if state.layer_types[li] == "linear_attention":
                cs, rs = state.dn_caches_tt[li]
                state.cb_dn[li] = {"cs": cs, "rs": rs}
            else:
                if state.kv_caches_tt[li] is not None:
                    kc, vc = state.kv_caches_tt[li]
                    state.cb_kv[li] = {"kc": kc, "vc": vc}
        state.cb_page_table_tt = None  # B=1 path uses state.page_table_tt
    else:
        # v1+ path: per-slot caches with leading B dim.
        # DN state shapes mirror base.reset_caches_ttnn but with B in the
        # slot axis (replacing the "1"). Mesh-sharded on dim=0 (the NCHIPS
        # axis) so per-chip view is [B, NV_PER_CHIP, K, V] / [B, CONV_DIM_CHIP, KERNEL].
        _rs_dtype = getattr(state, "dn_state_dtype", ttnn.bfloat16)
        for li in range(cfg.num_hidden_layers):
            if state.layer_types[li] == "linear_attention":
                cs_init = np.zeros(
                    (NCHIPS, B, CONV_DIM_CHIP, CONV_KERNEL), dtype=np.float32)
                rs_init = np.zeros(
                    (NCHIPS, B, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float32)
                cs_tt = ttnn.from_torch(
                    torch.from_numpy(cs_init), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                rs_tt = ttnn.from_torch(
                    torch.from_numpy(rs_init), dtype=_rs_dtype,
                    layout=ttnn.TILE_LAYOUT, device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                )
                state.cb_dn[li] = {"cs": cs_tt, "rs": rs_tt}
            else:
                if state.attn_mode == "sdpa":
                    # KV pool sized B * sdpa_num_blocks total, partitioned via
                    # cb_page_table_tt. Per-chip layout mirrors base.reset_caches_ttnn
                    # but with B * sdpa_num_blocks in dim 0.
                    total_blocks = B * state.sdpa_num_blocks
                    cache_shape = (total_blocks, NCHIPS,
                                   state.sdpa_block_size, HEAD_DIM_ATTN)
                    kc_init = np.zeros(cache_shape, dtype=np.float32)
                    vc_init = np.zeros(cache_shape, dtype=np.float32)
                    kc = ttnn.from_torch(
                        torch.from_numpy(kc_init), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                    )
                    vc = ttnn.from_torch(
                        torch.from_numpy(vc_init), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=mesh,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                    )
                    state.cb_kv[li] = {"kc": kc, "vc": vc}

        # Per-slot page table: slot s owns blocks [s*nb, (s+1)*nb).
        nb = state.sdpa_num_blocks
        page_table_np = np.stack([
            np.arange(s * nb, (s + 1) * nb, dtype=np.int32) for s in range(B)
        ], axis=0)
        state.cb_page_table_tt = ttnn.from_torch(
            torch.from_numpy(page_table_np), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    # Batched input buffers — same shape pattern for B=1 or B>1.
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


# ----------------------------------------------------------------------------
# Slot lifecycle (v0: trivial at B=1)
# ----------------------------------------------------------------------------
def cb_reset_states(state):
    """Reset ALL slot's DN + KV state to zero.

    At B=1: cb_dn/cb_kv ARE the single-stream caches. Deallocate them
    explicitly (base.reset_caches_ttnn just rebinds the list, leaking
    the old tensors — [[ttnn-list-rebinding-leaks]]), then re-alias.

    At B>1: cb_dn/cb_kv are separately-allocated B-leading tensors.
    In-place zero them via masked-zero multiply (same pattern as
    27B cb_reset_slots, but for all slots).
    """
    if state.cb_B == 1:
        if state.dn_caches_tt is not None:
            for entry in state.dn_caches_tt:
                if entry is not None:
                    cs, rs = entry
                    ttnn.deallocate(cs)
                    ttnn.deallocate(rs)
        if state.kv_caches_tt is not None:
            for entry in state.kv_caches_tt:
                if entry is not None:
                    kc, vc = entry
                    ttnn.deallocate(kc)
                    ttnn.deallocate(vc)

        state.reset_caches_ttnn()
        cfg = state.text_cfg
        for li in range(cfg.num_hidden_layers):
            if state.layer_types[li] == "linear_attention":
                cs, rs = state.dn_caches_tt[li]
                state.cb_dn[li] = {"cs": cs, "rs": rs}
            else:
                if state.kv_caches_tt[li] is not None:
                    kc, vc = state.kv_caches_tt[li]
                    state.cb_kv[li] = {"kc": kc, "vc": vc}
    else:
        # In-place zero via ttnn.mul(t, 0.0) → ttnn.copy.
        for d in state.cb_dn.values():
            for t in (d["cs"], d["rs"]):
                z = ttnn.mul(t, 0.0)
                ttnn.copy(z, t)
                ttnn.deallocate(z)
        for d in state.cb_kv.values():
            for t in (d["kc"], d["vc"]):
                z = ttnn.mul(t, 0.0)
                ttnn.copy(z, t)
                ttnn.deallocate(z)


def cb_reset_slots(state, slot_ids):
    """Reset specific slots' DN state. At B=1 with slot 0 in slot_ids:
    same effect as cb_reset_states. v1 will use masked multiply."""
    if state.cb_B == 1 and 0 in slot_ids:
        cb_reset_states(state)
    # else: no-op (slot 0 wasn't reset, nothing to do)


# ----------------------------------------------------------------------------
# Input buffer update
# ----------------------------------------------------------------------------
def update_input_buffers_batched(state, token_ids, cur_positions):
    """Host→device copy of [B] token IDs + [B] cur_pos. At B=1: delegates
    to the existing update_input_buffers."""
    if state.cb_B != 1:
        raise NotImplementedError("v0 supports B=1 only")
    base.update_input_buffers(state, int(token_ids[0]), int(cur_positions[0]))
    # Also update cb_*_buf so scheduler's batched view stays consistent.
    # (Single-source-of-truth would be nicer, but base.update_input_buffers
    # writes its own state.tok_buf etc., not the cb_ aliases.)
    mesh = state.mesh
    tok_host = ttnn.from_torch(
        torch.tensor([[int(token_ids[0])]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(tok_host, state.cb_tok_buf)
    cp_host = ttnn.from_torch(
        torch.tensor([int(cur_positions[0])], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(cp_host, state.cb_cur_pos_buf)


# ----------------------------------------------------------------------------
# Forward (v0: delegates to single-stream step_forward_inner)
# ----------------------------------------------------------------------------
def forward_batch_tp_inner(state, return_logits=False, return_topk=None):
    """One batched forward step. v0: B=1.

    Delegates to base.step_forward_inner — at B=1 we ARE the single-stream
    path. base reads state.sampler_topk to choose between argmax-mode and
    topk-mode internally; we set it transiently around the call.

    return_logits is NOT supported by base; v0 routes it to topk K=64 and
    raises (cb_engine's logits-mode shouldn't be used for 35B — see
    TT_CB_TOPK_K=64 default in cb_api).

    Returns:
      - return_topk=K:   (top_vals [1, K], top_idxs [1, K])
      - default:         argmax_tt UINT32 [1, 1]
    """
    if state.cb_B != 1:
        raise NotImplementedError("v0 supports B=1 only")

    if return_logits:
        raise NotImplementedError(
            "v0 doesn't support return_logits — 35B's [1, VOCAB] bulk readback "
            "is broken (issue #149). Use topk-mode (TT_CB_TOPK_K>0).")

    saved_topk = getattr(state, "sampler_topk", 0)
    try:
        state.sampler_topk = int(return_topk) if return_topk is not None else 0
        return base.step_forward_inner(state)
    finally:
        state.sampler_topk = saved_topk


# ----------------------------------------------------------------------------
# cb_prefill_transplant — not needed for v0 (no chunked prefill on 35B)
# ----------------------------------------------------------------------------
def cb_prefill_transplant(state, slot_s, L):
    """No-op stub for v0. 35B uses 1-tok/iter prefill via the decode path;
    no chunked prefill → no transplant needed."""
    pass


# ----------------------------------------------------------------------------
# setup_cb_write_mem_cfg — not needed for v0 (no paged_update_cache batching)
# ----------------------------------------------------------------------------
def setup_cb_write_mem_cfg(state):
    """No-op for v0; v1 paged_update_cache batching will need this."""
    pass
