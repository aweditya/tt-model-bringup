"""server_35b_cb.py — CB layer for Qwen3.6-35B-A3B.

Analogue of `server_tp_cb.py` for the 35B-A3B MoE hybrid. Imports
`server_35b_ttnn` as `base` for single-stream primitives + constants,
adds per-slot state + batched forwards on top.

Reuse map (mirror of server_tp_cb.py):
    setup_cb_state(state, B)           — per-slot DN + KV + cur_pos
    cb_reset_states(state)             — reset all DN slots
    cb_reset_slots(state, slot_ids)    — reset specific DN slots (masked mul)
    cb_prefill_transplant(state,s,L)   — copy single-stream state into slot
    update_input_buffers_batched       — batched host→device copy
    deltanet_step_batched(state, ...)  — batched DN forward (manual recurrence)
    gated_attn_step_batched(state, ...) — batched paged SDPA
    moe_step_batched(state, ...)       — calls existing Pattern A MoE
    forward_batch_35b_inner(state, ...) — full layered forward at B=N

Plan: research/35b_cb_bringup_plan.md (v0 = B=1, v1 = B>1, v2 = trace).
v0 status: SKELETON ONLY — fill in each function from the 27B template
when device validation is unblocked.

Constants imported from server_35b_ttnn (NOT full_layer_tp_probe, which
is 27B-specific):
    NV_PER_CHIP = 8  (vs 27B's 12)
    NK_PER_CHIP = 4  (vs 27B's 6)
    HEAD_K_DIM = HEAD_V_DIM = 128
    CONV_DIM_CHIP, CONV_KERNEL = 4
    HIDDEN = 2048 (vs 27B's 5120)
    NQ_PER_CHIP = 4, NUM_KV_HEADS = 2, HEAD_DIM_ATTN = 256
    ROTARY_DIM = 64 (partial)
    E_LOCAL = 64, TOP_K = 8, MOE_INTER = 512
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import server_35b_ttnn as base  # noqa: E402  — single-stream machinery + constants
import ttnn  # noqa: E402

# ----------------------------------------------------------------------------
# Constants — re-exported from base for clarity. Mirrors what 27B's
# server_tp_cb.py gets from full_layer_tp_probe.
# ----------------------------------------------------------------------------
NV_PER_CHIP = base.NV_PER_CHIP        # 8 — DN value heads per chip
NK_PER_CHIP = base.NK_PER_CHIP        # 4
HEAD_K_DIM = base.HEAD_K_DIM          # 128
HEAD_V_DIM = base.HEAD_V_DIM          # 128
CONV_DIM_CHIP = base.CONV_DIM_CHIP
CONV_KERNEL = base.CONV_KERNEL        # 4
HIDDEN = base.HIDDEN                  # 2048
NQ_PER_CHIP = base.NQ_PER_CHIP        # 4 — attn Q heads per chip
NUM_KV_HEADS = base.NUM_KV_HEADS      # 2
HEAD_DIM_ATTN = base.HEAD_DIM_ATTN    # 256
ROTARY_DIM = base.ROTARY_DIM          # 64 (partial)
E_LOCAL = base.E_LOCAL                # 64 — MoE experts per chip
TOP_K = base.TOP_K                    # 8
MOE_INTER_CHIP = base.MOE_INTER_CHIP
NCHIPS = base.NCHIPS                  # 4
VOCAB = base.VOCAB                    # 248320


# ----------------------------------------------------------------------------
# v0 / CB35-1 — Per-slot state allocation
# ----------------------------------------------------------------------------
def setup_cb_state(state, B, blocks_per_seq=None):
    """Allocate per-slot DN + KV state on top of single-stream 35B state.

    Mirror of server_tp_cb.py:45 (setup_cb_state). For each of the 40
    layers in the 35B model:
      - DN layers (3 per block, 30 total): allocate per-slot conv_state
        + recurrent (H_t) state. State shape matches base.reset_caches_ttnn
        but with leading B dim. H_t rank-5 sharded dim=2 (NCHIPS axis).
      - GatedAttention layers (1 per block, 10 total): paged KV cache,
        shape (NUM_BLOCKS, NCHIPS, BLOCK_SIZE, HEAD_DIM_ATTN) sharded
        along dim=1 — same as base.reset_caches_ttnn but with per-slot
        page_table indirection (paged_update_cache).
      - MoE expert weights: NO per-slot state (experts are shared across
        slots; intermediate activations are per-token, allocated
        transiently inside forward).

    Sets:
        state.cb_B
        state.cb_dn[li]  per-DN-layer dict {'rs': [B,...], 'cs': [B,...]}
        state.cb_kv[li]  per-attn-layer dict {'kc': [...paged...], 'vc': [...]}
        state.cb_cur_pos_buf  [B] int32 device
        state.cb_rot_idxs_buf [B, 1] int32 device
        state.cb_page_table_tt [B, blocks_per_seq] int32 device — per-slot
            paged KV pages, identity for B=1 to start
        state.cb_blocks_per_seq

    TODO: copy from server_tp_cb.py:45-144 and:
      - swap NV_PER_CHIP / K_DIM / V_DIM / CONV_DIM_CHIP constants
      - use base.HEAD_K_DIM / HEAD_V_DIM for DN H_t shape
      - 35B has 10 attn layers vs 27B's 14 — count from base.text_cfg.layer_types
      - 35B H_t state is rank-5 (NCHIPS, 1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM)
        sharded dim=0 (see server_35b_ttnn.py:1409); per-slot adds leading B
        → (B, NCHIPS, 1, NV_PER_CHIP, HEAD_K_DIM, HEAD_V_DIM) sharded dim=1
      - conv state: rank-4 (NCHIPS, 1, CONV_DIM_CHIP, CONV_KERNEL); per-slot
        adds leading B → (B, NCHIPS, 1, CONV_DIM_CHIP, CONV_KERNEL) sharded dim=1
    """
    raise NotImplementedError("CB35-1 v0 stub — port from server_tp_cb.py:45-144")


def cb_reset_states(state):
    """Reset ALL slot's DN state to zero. Mirror of server_tp_cb.py:146.

    Pure math: ttnn.mul(t, 0.0) → ttnn.copy(z, t) → deallocate. Same for
    both 27B and 35B since shape only adds a B leading dim that gets
    multiplied by 0 anyway.
    """
    raise NotImplementedError("CB35-1 v0 stub — port from server_tp_cb.py:146-155")


def cb_reset_slots(state, slot_ids):
    """Masked-multiply reset for specific slots. Mirror of server_tp_cb.py:157.

    Build mask[B] = 1 for slots NOT being reset, 0 for slots being reset.
    Apply to each layer's DN ssm + conv_cols. Other slots' state preserved.
    """
    raise NotImplementedError("CB35-1 v0 stub — port from server_tp_cb.py:157-185")


def cb_prefill_transplant(state, slot_s, L):
    """Copy single-stream prefill state into slot s. Mirror of server_tp_cb.py:188.

    Optional for v0 — only needed if we use chunked_prefill mode. v0 uses
    1-tok/iter prefill via the decode trace, so this can be a no-op stub
    initially. Wire in if/when 35B grows a forward_prefill_chunked_tp.
    """
    raise NotImplementedError("CB35-1 v0 OPTIONAL stub — port from server_tp_cb.py:188-307")


# ----------------------------------------------------------------------------
# v0 / CB35-1 — Batched input buffer update
# ----------------------------------------------------------------------------
def update_input_buffers_batched(state, token_ids, cur_positions):
    """Host→device copy of [B] token IDs + [B] cur_pos into pre-allocated
    state.cb_*_buf tensors. Called OUTSIDE the trace, before
    ttnn.execute_trace.

    Mirror of server_tp_cb.py:310. Identical between 27B and 35B.
    """
    raise NotImplementedError("CB35-1 v0 stub — port from server_tp_cb.py:310-329")


# ----------------------------------------------------------------------------
# v1 / CB35-2 — Batched per-primitive forward steps
# ----------------------------------------------------------------------------
def deltanet_step_batched(state, x_tt, dn_li, cfg):
    """Batched DN forward step at B=cb_B for layer index dn_li.

    Manual recurrence path (owned_gdn kernel is B=1-only). Updates
    state.cb_dn[dn_li]['rs'] and ['cs'] in-place.

    Mirror of server_tp_cb.py:331-518. 35B-specific adaptations:
      - State shape: H_t rank-6 with leading B (vs 27B's rank-5)
      - NV_PER_CHIP=8 (vs 12), HEAD_K_DIM=HEAD_V_DIM=128 (same)
      - Conv state same layout as 27B with B dim
      - Q/K L2-norm: use base.dn_fused_qk_norm path if state flag set

    Reference for the math: server_35b_ttnn.py:dn_forward_ttnn (line 372)
    is the existing B=1 forward; lift its body with a leading B axis.
    """
    raise NotImplementedError("CB35-2 v1 stub — port from server_tp_cb.py:331-517")


def gated_attn_step_batched(state, x_tt, attn_li, cos_tt, sin_tt, cfg):
    """Batched paged SDPA forward step at B=cb_B for attn layer.

    Mirror of server_tp_cb.py:519-625. 35B-specific:
      - NQ_PER_CHIP=4 (vs 27B's 6), NUM_KV_HEADS=2 (vs 27B's 8)
      - HEAD_DIM_ATTN=256 (vs 27B's 128) — partial RoPE rotary_dim=64
      - attn_output_gate=True doubles Q proj — per-head chunk split
        (see feedback_qwen36_attn_qgate_chunk_per_head.md)
      - Lift attention math from server_35b_ttnn.py:attn_forward_ttnn_sdpa
        (line 743) with a leading B axis.

    Uses ttnn.experimental.paged_update_cache (with update_idxs_tensor =
    state.cb_cur_pos_buf) for KV write and paged_scaled_dot_product_attention_decode
    for read — both already correct in 27B's CB, just need 35B shapes.
    """
    raise NotImplementedError("CB35-2 v1 stub — port from server_tp_cb.py:519-625")


def moe_step_batched(state, x_tt, moe_li, cfg):
    """Batched MoE FFN forward step.

    HOT TAKE FROM RESEARCH: this is essentially FREE because the existing
    moe_forward_ttnn_pattern_a_batched (server_35b_ttnn.py:1225) already
    supports B>1 unchanged — ttnn.matmul broadcasts over leading dim.

    Implementation: reshape x_tt from [B, HIDDEN] to [B, 1, HIDDEN] (the
    shape Pattern A expects), call base.moe_forward_ttnn_pattern_a_batched,
    reshape back to [B, HIDDEN].

    NO per-slot state mutation — MoE experts are stateless across slots.
    """
    # NOTE: this is the easy one. Implement first to confirm the pattern.
    # Then deltanet_step_batched + gated_attn_step_batched are the bulk.
    raise NotImplementedError("CB35-2 v1 stub — wrap base.moe_forward_ttnn_pattern_a_batched")


# ----------------------------------------------------------------------------
# v0 + v1 / CB35-1, CB35-2 — Full batched forward
# ----------------------------------------------------------------------------
def forward_batch_35b_inner(state, return_logits=False, return_topk=None):
    """One batched forward step at B=cb_B. Returns per-slot argmax/topk/logits.

    Layer dispatch loop:
        for L in range(num_layers):
            if state.layer_types[L] == 'linear_attention':
                x = deltanet_step_batched(state, x, dn_li=L, ...)
            else:
                x = gated_attn_step_batched(state, x, attn_li=L, cos, sin, ...)
            x = moe_step_batched(state, x, moe_li=L, ...)
        Then: final_norm, lm_head (vocab-sharded), per-slot argmax/topk.

    Mirror of server_tp_cb.py:638-end. The only model-specific changes are
    layer-type dispatch (35B has GDN layers interleaved every block of 4
    vs 27B's different pattern) and MoE call (FREE per above).

    Returns: same shape as 27B's forward_batch_tp_inner.
    """
    raise NotImplementedError("CB35-1+2 stub — port from server_tp_cb.py:638-end")


# ----------------------------------------------------------------------------
# Helpers (mirror server_tp_cb.py setup_cb_write_mem_cfg + _attn_finish)
# ----------------------------------------------------------------------------
def setup_cb_write_mem_cfg(state):
    """Pre-build the HEIGHT_SHARDED L1 mem config used by paged_update_cache.
    Mirror of server_tp_cb.py:626-636.
    """
    raise NotImplementedError("CB35-1 v0 stub — port from server_tp_cb.py:626-636")
