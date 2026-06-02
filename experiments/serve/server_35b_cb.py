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
    single-stream allocator, which zeros everything."""
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

    Mirrors base.step_forward_inner's body (server_35b_ttnn.py:1602) but
    routes the final op based on the requested return mode (instead of
    base's hardcoded sampler_topk check). At B=1 this is functionally
    identical to step_forward_inner.

    Returns shape per cb_scheduler convention:
      - return_logits=False, return_topk=None: argmax_tt UINT32 [1, 1]
      - return_logits=True:                    logits tensor [1, VOCAB]
      - return_topk=K:                         (values [1, K], indices [1, K])
    """
    if state.cb_B != 1:
        raise NotImplementedError("v0 supports B=1 only")

    # ── Same prelude as step_forward_inner (embed + RoPE rows). ────────
    embed_out = ttnn.embedding(state.tok_buf, state.embed_tt)
    h_tt = ttnn.to_layout(embed_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_out)

    cos_row = ttnn.embedding(state.rot_idxs_buf, state.cos_table_tt)
    sin_row = ttnn.embedding(state.rot_idxs_buf, state.sin_table_tt)
    cos_tt = ttnn.to_layout(cos_row, ttnn.TILE_LAYOUT)
    sin_tt = ttnn.to_layout(sin_row, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_row); ttnn.deallocate(sin_row)
    cos_tt = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
    sin_tt = ttnn.reshape(sin_tt, [1, ROTARY_DIM])

    # ── Layer chain (same as base.step_forward_inner lines 1627-1656). ─
    n = state.text_cfg.num_hidden_layers
    for L in range(n):
        lt = state.layer_types[L]
        h_new, new_dn, new_kv = base.layer_forward_ttnn(
            h_tt, state.per_layer_tt[L], lt, state.mesh,
            cos_tt, sin_tt, state.dn_caches_tt[L], state.kv_caches_tt[L],
            sub_capture=None, state=state,
        )
        ttnn.deallocate(h_tt)
        h_tt = h_new
        if new_dn is not None:
            old_conv, old_rec = state.dn_caches_tt[L]
            new_conv, new_rec = new_dn
            if new_conv is not old_conv:
                ttnn.deallocate(old_conv)
            if new_rec is not old_rec:
                ttnn.deallocate(old_rec)
            state.dn_caches_tt[L] = new_dn
        if new_kv is not None:
            state.kv_caches_tt[L] = new_kv

    ttnn.deallocate(cos_tt); ttnn.deallocate(sin_tt)

    # ── Final norm + LM head. ──────────────────────────────────────────
    h_norm = ttnn.rms_norm(h_tt, weight=state.final_norm_tt, epsilon=base.EPS)
    ttnn.deallocate(h_tt)
    logits = ttnn.matmul(h_norm, state.lm_head_tt, compute_kernel_config=base.HIFI4)
    ttnn.deallocate(h_norm)

    # ── Final op: branch on requested mode. ────────────────────────────
    if return_topk is not None:
        top_vals, top_idxs = ttnn.topk(logits, k=int(return_topk), dim=-1)
        ttnn.deallocate(logits)
        return (top_vals, top_idxs)

    if return_logits:
        # Convert to ROW_MAJOR for the cb_scheduler's per-slot host sample.
        # NOTE: do NOT deallocate `logits` here — to_layout returns a view;
        # deallocating the source while caller still holds `rm` causes the
        # readback to read garbage (view-decay, feedback_ttnn_slice_view_decay).
        # Caller's `ttnn.deallocate(rm)` cleans up via the underlying buffer.
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        return rm

    # Default: on-device argmax (greedy). Same as base.step_forward_inner.
    # Here the argmax kernel consumes rm immediately, so deallocating
    # logits is safe (the kernel materializes its input).
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(logits)
    argmax_tt = ttnn.argmax(rm, dim=-1, keepdim=True, use_multicore=True)
    ttnn.deallocate(rm)
    return argmax_tt


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
