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
    prev_B = getattr(state, "cb_B", None)
    state.cb_B = B

    # base.bootstrap doesn't call reset_caches_ttnn (the single-stream
    # main() does). Call it here so per-layer caches exist for v0 alias path.
    if state.dn_caches_tt is None or state.kv_caches_tt is None:
        state.reset_caches_ttnn()

    # Repeat setup_cb_state calls (e.g., changing B between test runs in the
    # dev harness): free anything we previously allocated.
    if prev_B is not None and prev_B > 1:
        # B>1 cb_dn/cb_kv are NEW allocations (not aliases). Free them.
        # B=1 cb_dn/cb_kv share storage with state.dn_caches_tt and must NOT
        # be freed here (base path still uses them).
        for d in getattr(state, "cb_dn", {}).values():
            ttnn.deallocate(d["cs"])
            ttnn.deallocate(d["rs"])
        for d in getattr(state, "cb_kv", {}).values():
            ttnn.deallocate(d["kc"])
            ttnn.deallocate(d["vc"])
        if getattr(state, "cb_page_table_tt", None) is not None:
            ttnn.deallocate(state.cb_page_table_tt)
    # Input buffers are allocated unconditionally below — always free old ones.
    if prev_B is not None:
        for name in ("cb_tok_buf", "cb_cur_pos_buf", "cb_rot_idxs_buf"):
            old = getattr(state, name, None)
            if old is not None:
                ttnn.deallocate(old)

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

    # B-sized paged config (write mem cfg + SDPA program config).
    setup_cb_paged_cfgs(state)

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
    """Host→device copy of [B] token IDs + [B] cur_pos + clamped rot_idxs.

    At B=1: ALSO writes the single-stream state.tok_buf / state.cur_pos_buf
    / state.rot_idxs_buf via base.update_input_buffers — the v0 forward
    path delegates to base.step_forward_inner which reads those buffers.

    For empty slots (`cur_pos == -1`): rot_idxs is clamped to 0 (the embed
    is read but never used downstream). Matches 27B's convention.
    """
    B = state.cb_B
    assert len(token_ids) == B, f"expected {B} tokens, got {len(token_ids)}"
    assert len(cur_positions) == B, f"expected {B} positions, got {len(cur_positions)}"
    mesh = state.mesh

    if B == 1:
        # Keep single-stream buffers consistent for the v0 delegate-to-base path.
        base.update_input_buffers(state, int(token_ids[0]), int(cur_positions[0]))

    # cb_* batched buffers (used by v1+ forward path).
    tok_host = ttnn.from_torch(
        torch.tensor([int(t) for t in token_ids], dtype=torch.int32).reshape(B, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(tok_host, state.cb_tok_buf)

    cp_host = ttnn.from_torch(
        torch.tensor([int(p) for p in cur_positions], dtype=torch.int32).reshape(B),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(cp_host, state.cb_cur_pos_buf)

    rot_idxs = [max(int(p), 0) for p in cur_positions]
    rt_host = ttnn.from_torch(
        torch.tensor(rot_idxs, dtype=torch.int32).reshape(B, 1),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(rt_host, state.cb_rot_idxs_buf)


def dn_step_batched_35b(state, h_tt, w, cb_dn_layer, *, qk_l2_weight_tt=None, qk_l2_eps=None):
    """Batched 35B DN block. h_tt [B, HIDDEN] or [B, 1, HIDDEN] per chip.

    Per-slot state in cb_dn_layer = {"cs":[B,CONV_DIM_CHIP,KERNEL],
                                    "rs":[B,NV_PER_CHIP,K,V]} (per chip).
    Manual recurrence (no owned-GDN kernel — that's B=1 only).
    Returns out [B, HIDDEN] replicated. Mutates cb_dn_layer state in place.

    Mirrors base.dn_forward_ttnn but with B in the slot axis. All compute
    ops broadcast over B; reshapes are the only B-aware step.
    """
    B = state.cb_B
    mesh = state.mesh
    cs_in = cb_dn_layer["cs"]
    rs_in = cb_dn_layer["rs"]

    # ── In-proj fused matmul + slice ────────────────────────────────────
    fused = ttnn.matmul(h_tt, w["in_proj_combined"], compute_kernel_config=base.HIFI4)
    fr = len(list(fused.shape))
    OFF_QKV_END = CONV_DIM_CHIP
    OFF_Z_END   = OFF_QKV_END + base.VALUE_DIM_CHIP
    OFF_A_END   = OFF_Z_END + NV_PER_CHIP
    OFF_B_END   = OFF_A_END + NV_PER_CHIP
    if fr == 3:
        mixed_qkv = ttnn.slice(fused, [0, 0, 0],         [B, 1, OFF_QKV_END])
        z         = ttnn.slice(fused, [0, 0, OFF_QKV_END],[B, 1, OFF_Z_END])
        a         = ttnn.slice(fused, [0, 0, OFF_Z_END], [B, 1, OFF_A_END])
        b         = ttnn.slice(fused, [0, 0, OFF_A_END], [B, 1, OFF_B_END])
    else:
        mixed_qkv = ttnn.slice(fused, [0, 0],             [B, OFF_QKV_END])
        z         = ttnn.slice(fused, [0, OFF_QKV_END],   [B, OFF_Z_END])
        a         = ttnn.slice(fused, [0, OFF_Z_END],     [B, OFF_A_END])
        b         = ttnn.slice(fused, [0, OFF_A_END],     [B, OFF_B_END])
    ttnn.deallocate(fused)

    # ── Conv1d shift+accumulate (batched) ───────────────────────────────
    cur = ttnn.reshape(mixed_qkv, [B, CONV_DIM_CHIP, 1])
    ttnn.deallocate(mixed_qkv)
    cs_rank = len(list(cs_in.shape))
    if cs_rank == 4:
        prior = ttnn.slice(cs_in, [0, 0, 0, 1], [1, B, CONV_DIM_CHIP, CONV_KERNEL])
        prior = ttnn.reshape(prior, [B, CONV_DIM_CHIP, CONV_KERNEL - 1])
    else:
        prior = ttnn.slice(cs_in, [0, 0, 1], [B, CONV_DIM_CHIP, CONV_KERNEL])
    cs_new = ttnn.concat([prior, cur], dim=-1)
    ttnn.deallocate(prior); ttnn.deallocate(cur)

    cw_rank = len(list(w["conv1d_weight"].shape))
    if cw_rank == 4:
        w_conv = ttnn.reshape(w["conv1d_weight"], [1, CONV_DIM_CHIP, CONV_KERNEL])
    else:
        w_conv = w["conv1d_weight"]
    state_w = ttnn.mul(cs_new, w_conv)  # broadcast w_conv over B
    if cw_rank == 4:
        ttnn.deallocate(w_conv)
    conv_out_3d = ttnn.sum(state_w, dim=-1, keepdim=True)
    ttnn.deallocate(state_w)
    conv_out = ttnn.reshape(conv_out_3d, [B, CONV_DIM_CHIP])
    ttnn.deallocate(conv_out_3d)
    silu_out = ttnn.silu(conv_out)
    ttnn.deallocate(conv_out)

    # ── Split Q/K/V ──────────────────────────────────────────────────────
    sr = len(list(silu_out.shape))
    if sr == 3:
        q_flat = ttnn.slice(silu_out, [0, 0, 0], [B, 1, base.KEY_DIM_CHIP])
        k_flat = ttnn.slice(silu_out, [0, 0, base.KEY_DIM_CHIP], [B, 1, 2 * base.KEY_DIM_CHIP])
        v_flat = ttnn.slice(silu_out, [0, 0, 2 * base.KEY_DIM_CHIP], [B, 1, CONV_DIM_CHIP])
    else:
        q_flat = ttnn.slice(silu_out, [0, 0], [B, base.KEY_DIM_CHIP])
        k_flat = ttnn.slice(silu_out, [0, base.KEY_DIM_CHIP], [B, 2 * base.KEY_DIM_CHIP])
        v_flat = ttnn.slice(silu_out, [0, 2 * base.KEY_DIM_CHIP], [B, CONV_DIM_CHIP])
    ttnn.deallocate(silu_out)

    q_h = ttnn.reshape(q_flat, [B, NK_PER_CHIP, HEAD_K_DIM])
    k_h = ttnn.reshape(k_flat, [B, NK_PER_CHIP, HEAD_K_DIM])
    v_h = ttnn.reshape(v_flat, [B, NV_PER_CHIP, HEAD_V_DIM])

    # ── L2-norm Q/K (fused rms_norm path; same as base) ────────────────
    if qk_l2_weight_tt is not None:
        q_n = ttnn.rms_norm(q_h, weight=qk_l2_weight_tt, epsilon=qk_l2_eps)
        k_n = ttnn.rms_norm(k_h, weight=qk_l2_weight_tt, epsilon=qk_l2_eps)
        ttnn.deallocate(q_h); ttnn.deallocate(k_h)
    else:
        # Manual chain (matches base when qk_l2_weight_tt is None).
        q_sq = ttnn.mul(q_h, q_h)
        q_sumsq = ttnn.sum(q_sq, dim=-1, keepdim=True)
        ttnn.deallocate(q_sq)
        q_inv = ttnn.rsqrt(ttnn.add(q_sumsq, base.EPS))
        ttnn.deallocate(q_sumsq)
        q_n = ttnn.mul(q_h, q_inv)
        ttnn.deallocate(q_h); ttnn.deallocate(q_inv)
        k_sq = ttnn.mul(k_h, k_h)
        k_sumsq = ttnn.sum(k_sq, dim=-1, keepdim=True)
        ttnn.deallocate(k_sq)
        k_inv = ttnn.rsqrt(ttnn.add(k_sumsq, base.EPS))
        ttnn.deallocate(k_sumsq)
        k_n = ttnn.mul(k_h, k_inv)
        ttnn.deallocate(k_h); ttnn.deallocate(k_inv)

    # Q scale: 1/sqrt(HEAD_K_DIM)
    q_scale = 1.0 / (HEAD_K_DIM ** 0.5)
    q_n_scaled = ttnn.multiply(q_n, q_scale)
    ttnn.deallocate(q_n)
    q_n = q_n_scaled
    q_h = q_n; k_h = k_n

    # ── beta + g (manual chain; owned-decay-gate kernel is B=1 only) ────
    beta = ttnn.sigmoid(b)
    ttnn.deallocate(b)
    a_plus_dt = ttnn.add(a, w["dt_bias"])
    ttnn.deallocate(a)
    softplus_v = ttnn.softplus(a_plus_dt)
    ttnn.deallocate(a_plus_dt)
    neg_exp_alog = ttnn.neg(ttnn.exp(w["A_log"]))
    g_decay = ttnn.exp(ttnn.mul(softplus_v, neg_exp_alog))
    ttnn.deallocate(softplus_v); ttnn.deallocate(neg_exp_alog)

    # ── GQA broadcast: NK heads → NV heads via reshape→repeat→reshape ──
    GQA_REPEAT = NV_PER_CHIP // NK_PER_CHIP
    q_4d = ttnn.reshape(q_h, [B, NK_PER_CHIP, 1, HEAD_K_DIM])
    k_4d = ttnn.reshape(k_h, [B, NK_PER_CHIP, 1, HEAD_K_DIM])
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)
    q_rep_4d = ttnn.repeat(q_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
    k_rep_4d = ttnn.repeat(k_4d, ttnn.Shape([1, 1, GQA_REPEAT, 1]))
    ttnn.deallocate(q_4d); ttnn.deallocate(k_4d)
    q_rep = ttnn.reshape(q_rep_4d, [B, NV_PER_CHIP, HEAD_K_DIM])
    k_rep = ttnn.reshape(k_rep_4d, [B, NV_PER_CHIP, HEAD_K_DIM])
    ttnn.deallocate(q_rep_4d); ttnn.deallocate(k_rep_4d)

    # ── Recurrence (manual; owned-GDN kernel is B=1 only) ──────────────
    g_b = ttnn.reshape(g_decay, [B, NV_PER_CHIP, 1, 1])
    ttnn.deallocate(g_decay)
    rs_decayed = ttnn.mul(rs_in, g_b)
    ttnn.deallocate(g_b)

    k_col = ttnn.reshape(k_rep, [B, NV_PER_CHIP, HEAD_K_DIM, 1])
    state_k = ttnn.mul(rs_decayed, k_col)
    kv_mem = ttnn.sum(state_k, dim=-2)
    ttnn.deallocate(state_k)
    kv_mem_3d = ttnn.reshape(kv_mem, [B, NV_PER_CHIP, HEAD_V_DIM])
    ttnn.deallocate(kv_mem)

    v_minus_kv = ttnn.sub(v_h, kv_mem_3d)
    ttnn.deallocate(kv_mem_3d); ttnn.deallocate(v_h)
    beta_b = ttnn.reshape(beta, [B, NV_PER_CHIP, 1])
    ttnn.deallocate(beta)
    delta = ttnn.mul(v_minus_kv, beta_b)
    ttnn.deallocate(v_minus_kv); ttnn.deallocate(beta_b)

    delta_row = ttnn.reshape(delta, [B, NV_PER_CHIP, 1, HEAD_V_DIM])
    ttnn.deallocate(delta)
    k_delta = ttnn.mul(k_col, delta_row)
    ttnn.deallocate(k_col); ttnn.deallocate(delta_row)
    rs_new = ttnn.add(rs_decayed, k_delta)
    ttnn.deallocate(rs_decayed); ttnn.deallocate(k_delta)

    q_col = ttnn.reshape(q_rep, [B, NV_PER_CHIP, HEAD_K_DIM, 1])
    ttnn.deallocate(q_rep)
    state_q = ttnn.mul(rs_new, q_col)
    ttnn.deallocate(q_col)
    core_attn_out_4d = ttnn.sum(state_q, dim=-2)
    ttnn.deallocate(state_q)
    core_attn_out = ttnn.reshape(core_attn_out_4d, [B, NV_PER_CHIP, HEAD_V_DIM])
    ttnn.deallocate(core_attn_out_4d)

    # ── RMSNormGated: keep [B, NV, HEAD_V] shape so rms_norm sees per-head
    # rows EXACTLY the same as at B=1 (folding to [B*NV, HEAD_V] introduced
    # ~0.009 mad drift, suspected kernel-shape sensitivity in bf16).
    core_3d = core_attn_out  # already [B, NV, HEAD_V]
    z_3d = ttnn.reshape(z, [B, NV_PER_CHIP, HEAD_V_DIM])
    ttnn.deallocate(z)
    normed_raw = ttnn.rms_norm(core_3d, epsilon=base.EPS)
    ttnn.deallocate(core_3d)
    normed = ttnn.mul(normed_raw, w["norm_weight"])
    ttnn.deallocate(normed_raw)
    gated = ttnn.mul(z_3d, normed, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(normed); ttnn.deallocate(z_3d)
    gated_2d = ttnn.reshape(gated, [B, base.VALUE_DIM_CHIP])
    ttnn.deallocate(gated)

    # ── out_proj + all_reduce ────────────────────────────────────────────
    partial = ttnn.matmul(gated_2d, w["out_proj"], compute_kernel_config=base.HIFI4)
    ttnn.deallocate(gated_2d)
    out = base.all_reduce_tt(partial, mesh)
    ttnn.deallocate(partial)

    # ── In-place state commit ───────────────────────────────────────────
    cs_new = ttnn.reshape(cs_new, list(cs_in.shape))
    ttnn.copy(cs_new, cs_in)
    ttnn.deallocate(cs_new)
    rs_new = ttnn.reshape(rs_new, list(rs_in.shape))
    ttnn.copy(rs_new, rs_in)
    ttnn.deallocate(rs_new)
    return out


def _batched_prelude(state):
    """Embed + RoPE rows for the B-slot batch. Returns:
      h_tt:    TILE [B, 1, HIDDEN] per chip (same shape pattern as base, B in slot axis)
      cos_tt:  TILE [B, ROTARY_DIM] per chip
      sin_tt:  TILE [B, ROTARY_DIM] per chip
    """
    B = state.cb_B
    embed_out = ttnn.embedding(state.cb_tok_buf, state.embed_tt)
    h_tt = ttnn.to_layout(embed_out, ttnn.TILE_LAYOUT)
    ttnn.deallocate(embed_out)

    cos_row = ttnn.embedding(state.cb_rot_idxs_buf, state.cos_table_tt)
    sin_row = ttnn.embedding(state.cb_rot_idxs_buf, state.sin_table_tt)
    cos_tt = ttnn.to_layout(cos_row, ttnn.TILE_LAYOUT)
    sin_tt = ttnn.to_layout(sin_row, ttnn.TILE_LAYOUT)
    ttnn.deallocate(cos_row); ttnn.deallocate(sin_row)
    cos_tt = ttnn.reshape(cos_tt, [B, ROTARY_DIM])
    sin_tt = ttnn.reshape(sin_tt, [B, ROTARY_DIM])
    return h_tt, cos_tt, sin_tt


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
# setup_cb_paged_cfgs — B-sized HEIGHT_SHARDED L1 mem cfg for paged KV writes
# ----------------------------------------------------------------------------
def setup_cb_paged_cfgs(state):
    """Build the B-core HEIGHT_SHARDED L1 mem config used as input to
    paged_update_cache, plus the B-tile SDPA program config.

    At B=1: reuses state.paged_write_mem_cfg / state.paged_sdpa_progcfg
    (already configured for single-stream by base.bootstrap).
    """
    B = state.cb_B
    if B == 1:
        state.cb_write_mem_cfg = state.paged_write_mem_cfg
        state.cb_sdpa_progcfg = state.paged_sdpa_progcfg
        return
    SDPA_TILE_HEIGHT = 32
    grid = state.mesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid, [SDPA_TILE_HEIGHT, HEAD_DIM_ATTN],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    state.cb_write_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec,
    )
    state.cb_sdpa_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        q_chunk_size=0, k_chunk_size=0,
        exp_approx_mode=False,
    )


def _apply_partial_rope_b(x_3d, cos_b, sin_b, n_heads, B):
    """Rank-3 partial RoPE. x_3d: [B, n_heads, HEAD_DIM_ATTN];
    cos_b / sin_b: [B, 1, ROTARY_DIM]. Same math as base._apply_partial_rope
    but with leading B dim throughout (avoids the rank-2 slice/concat bug
    on single rows — [[qwen36-attn-rope-single-row-ttnn-bug]]).
    """
    x_rot = ttnn.slice(x_3d, [0, 0, 0], [B, n_heads, ROTARY_DIM])
    x_pass = ttnn.slice(x_3d, [0, 0, ROTARY_DIM], [B, n_heads, HEAD_DIM_ATTN])
    half = ROTARY_DIM // 2
    x1 = ttnn.slice(x_rot, [0, 0, 0], [B, n_heads, half])
    x2 = ttnn.slice(x_rot, [0, 0, half], [B, n_heads, ROTARY_DIM])
    neg_x2 = ttnn.neg(x2)
    ttnn.deallocate(x2)
    rotated = ttnn.concat([neg_x2, x1], dim=-1)
    ttnn.deallocate(neg_x2); ttnn.deallocate(x1)
    x_rot_cos = ttnn.mul(x_rot, cos_b)
    rotated_sin = ttnn.mul(rotated, sin_b)
    ttnn.deallocate(x_rot); ttnn.deallocate(rotated)
    x_rot_embed = ttnn.add(x_rot_cos, rotated_sin)
    ttnn.deallocate(x_rot_cos); ttnn.deallocate(rotated_sin)
    x_embed = ttnn.concat([x_rot_embed, x_pass], dim=-1)
    ttnn.deallocate(x_rot_embed); ttnn.deallocate(x_pass)
    return x_embed


def attn_step_batched_35b(state, h_tt, w, cb_kv_layer, cos_tt, sin_tt):
    """Batched 35B GatedAttention block via paged SDPA decode.

    h_tt: [B, 1, HIDDEN] or [B, HIDDEN] per chip.
    cos_tt / sin_tt: [B, ROTARY_DIM] per chip (from _batched_prelude).
    cb_kv_layer = {"kc":[B*NUM_BLOCKS, NCHIPS, BLOCK_SIZE, HEAD_DIM_ATTN],
                   "vc":[...]} (sharded on dim=1, per-chip view [B*NB, 1, BLOCK, HEAD]).
    Reads per-slot cur_pos from state.cb_cur_pos_buf, page table from
    state.cb_page_table_tt (B=1: state.page_table_tt).

    Returns out [B, HIDDEN] replicated. Mutates kc/vc in place.
    """
    B = state.cb_B
    mesh = state.mesh
    kc = cb_kv_layer["kc"]
    vc = cb_kv_layer["vc"]

    # Choose per-slot or single-stream paged config buffers.
    cur_pos_buf = state.cb_cur_pos_buf if B > 1 else state.cur_pos_buf
    page_table_tt = state.cb_page_table_tt if B > 1 else state.page_table_tt
    write_mem_cfg = state.cb_write_mem_cfg
    sdpa_progcfg = state.cb_sdpa_progcfg

    # ── Q/K/V projections (per-chip head shards) ───────────────────────
    q_full = ttnn.matmul(h_tt, w["q_proj"], compute_kernel_config=base.HIFI4)
    k = ttnn.matmul(h_tt, w["k_proj"], compute_kernel_config=base.HIFI4)
    v = ttnn.matmul(h_tt, w["v_proj"], compute_kernel_config=base.HIFI4)

    # ── Split Q + gate per head ────────────────────────────────────────
    # q_full per chip: [B, 1, NQ_PER_CHIP * HEAD_DIM_ATTN * 2] (gate-doubled).
    # Reshape to [B, NQ_PER_CHIP, HEAD_DIM_ATTN * 2] then slice Q and gate halves.
    q_full_3d = ttnn.reshape(q_full, [B, NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full)
    q_h = ttnn.slice(q_full_3d, [0, 0, 0], [B, NQ_PER_CHIP, HEAD_DIM_ATTN])
    gate_per_head = ttnn.slice(q_full_3d, [0, 0, HEAD_DIM_ATTN],
                                [B, NQ_PER_CHIP, HEAD_DIM_ATTN * 2])
    ttnn.deallocate(q_full_3d)
    # Flatten gate to [B, NQ_PER_CHIP * HEAD_DIM_ATTN] for the post-SDPA mul.
    gate_part = ttnn.reshape(gate_per_head, [B, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(gate_per_head)

    # ── Q/K rms_norm — keep rank-3 (lesson from v1.2) ───────────────────
    k_h = ttnn.reshape(k, [B, 1, HEAD_DIM_ATTN])  # 1 KV head per chip
    ttnn.deallocate(k)
    q_n = ttnn.rms_norm(q_h, weight=w["q_norm"], epsilon=base.EPS)
    k_n = ttnn.rms_norm(k_h, weight=w["k_norm"], epsilon=base.EPS)
    ttnn.deallocate(q_h); ttnn.deallocate(k_h)

    # ── RoPE: rank-3 helper sidesteps the rank-2 single-row slice bug ──
    cos_b = ttnn.reshape(cos_tt, [B, 1, ROTARY_DIM])
    sin_b = ttnn.reshape(sin_tt, [B, 1, ROTARY_DIM])
    q_n_rope = _apply_partial_rope_b(q_n, cos_b, sin_b, NQ_PER_CHIP, B)
    ttnn.deallocate(q_n)
    k_n_rope = _apply_partial_rope_b(k_n, cos_b, sin_b, 1, B)
    ttnn.deallocate(k_n)
    ttnn.deallocate(cos_b); ttnn.deallocate(sin_b)

    # ── V reshape (per-chip 1 KV head) ──────────────────────────────────
    v_h = ttnn.reshape(v, [B, 1, HEAD_DIM_ATTN])
    ttnn.deallocate(v)

    # ── Paged KV write at slot's own cur_pos ───────────────────────────
    # Per-chip write tensor: [1, B, 1, HEAD_DIM_ATTN] padded on dim -2 to
    # SDPA_BLOCK_SIZE then HEIGHT_SHARDED across B cores.
    SDPA_TILE_H = state.sdpa_block_size
    def _shard_for_paged_write(t_3d):
        # t_3d: [B, 1, HEAD_DIM_ATTN]
        t4 = ttnn.reshape(t_3d, [1, B, 1, HEAD_DIM_ATTN])
        t_pad = ttnn.pad(t4, [[0, 0], [0, 0], [0, SDPA_TILE_H - 1], [0, 0]],
                         value=0.0)
        return ttnn.to_memory_config(t_pad, write_mem_cfg)
    k_sharded = _shard_for_paged_write(k_n_rope)
    v_sharded = _shard_for_paged_write(v_h)
    ttnn.deallocate(k_n_rope); ttnn.deallocate(v_h)
    ttnn.experimental.paged_update_cache(
        kc, k_sharded,
        update_idxs_tensor=cur_pos_buf,
        page_table=page_table_tt,
    )
    ttnn.experimental.paged_update_cache(
        vc, v_sharded,
        update_idxs_tensor=cur_pos_buf,
        page_table=page_table_tt,
    )
    ttnn.deallocate(k_sharded); ttnn.deallocate(v_sharded)

    # ── Paged SDPA decode ──────────────────────────────────────────────
    # Q shape [1, B, NQ_PER_CHIP, HEAD_DIM] per chip per 27B convention.
    q_for_sdpa = ttnn.reshape(q_n_rope, [1, B, NQ_PER_CHIP, HEAD_DIM_ATTN])
    ttnn.deallocate(q_n_rope)
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kc, vc,
        cur_pos_tensor=cur_pos_buf,
        page_table_tensor=page_table_tt,
        scale=1.0 / (HEAD_DIM_ATTN ** 0.5),
        program_config=sdpa_progcfg,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )  # [1, B, NQ_PER_CHIP, HEAD_DIM] per chip
    ttnn.deallocate(q_for_sdpa)
    attn_flat = ttnn.reshape(attn_out, [B, NQ_PER_CHIP * HEAD_DIM_ATTN])
    ttnn.deallocate(attn_out)

    # ── attn_output_gate: attn_out * sigmoid(gate) ─────────────────────
    gate_sig = ttnn.sigmoid(gate_part)
    ttnn.deallocate(gate_part)
    gated = ttnn.mul(attn_flat, gate_sig)
    ttnn.deallocate(attn_flat); ttnn.deallocate(gate_sig)

    # ── o_proj column-sharded + all_reduce ─────────────────────────────
    partial = ttnn.matmul(gated, w["o_proj"], compute_kernel_config=base.HIFI4)
    ttnn.deallocate(gated)
    out = base.all_reduce_tt(partial, mesh)
    ttnn.deallocate(partial)
    return out
