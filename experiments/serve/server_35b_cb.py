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
    """v0: B=1 only. Sets cb_B and aliases cb_kv/cb_dn to the existing
    single-stream caches. v1 will allocate true per-slot pools.

    The 35B single-stream State already has dn_caches_tt[li] = (cs, rs)
    for DN layers and kv_caches_tt[li] = (kc, vc) for attention layers
    (built by base.reset_caches_ttnn). For B=1 these ARE the per-slot
    state — slot 0 IS the only slot.

    Pre-allocates batched input buffers (cb_tok_buf, cb_cur_pos_buf,
    cb_rot_idxs_buf) so scheduler can feed them via copy_host_to_device.
    """
    if B != 1:
        raise NotImplementedError(
            f"v0 (CB35-1) supports B=1 only; got B={B}. CB35-2 will lift "
            f"this to true batched forward.")
    cfg = state.text_cfg
    mesh = state.mesh
    state.cb_B = B

    # base.bootstrap doesn't call reset_caches_ttnn (the single-stream
    # main() does). Call it here so the per-layer caches exist before we
    # alias them.
    if state.dn_caches_tt is None or state.kv_caches_tt is None:
        state.reset_caches_ttnn()

    # Alias existing single-stream caches as the per-slot cache. At B=1
    # the layout is identical to what dn_forward_ttnn / attn_forward_ttnn
    # already produce/consume.
    state.cb_dn = {}
    state.cb_kv = {}
    for li in range(cfg.num_hidden_layers):
        if state.layer_types[li] == "linear_attention":
            cs, rs = state.dn_caches_tt[li]
            state.cb_dn[li] = {"cs": cs, "rs": rs}
        else:
            if state.kv_caches_tt[li] is not None:
                kc, vc = state.kv_caches_tt[li]
                state.cb_kv[li] = {"kc": kc, "vc": vc}

    # Batched input buffers. At B=1 these are 1-element, but the cb_scheduler
    # always indexes by [B] / [B,1] so shape it that way.
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
    """Reset ALL slot's DN + KV state to zero. At B=1: re-runs the
    single-stream allocator, which zeros everything.

    base.reset_caches_ttnn() just REPLACES the list, it doesn't deallocate
    the old per-layer tensors — those leak. Over many resets in a long-lived
    harness, the allocator fragments and subsequent matmul output lands in
    stale memory → forward returns garbage. Explicitly deallocate first.
    """
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
    # Re-alias after reset since reset_caches_ttnn re-allocates.
    cfg = state.text_cfg
    for li in range(cfg.num_hidden_layers):
        if state.layer_types[li] == "linear_attention":
            cs, rs = state.dn_caches_tt[li]
            state.cb_dn[li] = {"cs": cs, "rs": rs}
        else:
            if state.kv_caches_tt[li] is not None:
                kc, vc = state.kv_caches_tt[li]
                state.cb_kv[li] = {"kc": kc, "vc": vc}


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
