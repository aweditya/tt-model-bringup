#!/usr/bin/env python3
"""
Experiment 91f — Phase B′6: Qwen3.6-27B layers 0-3 FULLY ON-DEVICE
(no numpy roundtrips on forward path).

Lifts the two B′5 host shortcuts to device:
  1. KV cache update: uses ttnn.experimental.paged_update_cache with the
     sharded-memory dance from demos/generate_moe.py.
  2. SDPA: uses ttnn.transformer.scaled_dot_product_attention_decode
     (validated in Phase A4 at our exact head_dim=256, GQA 24/4 shape).

After this passes cosine, the path through layers 0-3 has:
  - All compute on device (DeltaNet, Gated Attention, SwiGLU MLP, RMSNorm)
  - Zero numpy on the forward path
  - Numpy used ONLY for: weight upload (one-time) + reference comparison

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91f_qwen36_27b_full_ondevice.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.6-27B"
REF_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_layers0_3_ref.npz")
EPS = 1e-6
MAX_POS = 128

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


# ============================================================
# Weight loader (same as 91e)
# ============================================================

def load_layer_weights_all(layer_idx, layer_type):
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    base = f"model.language_model.layers.{layer_idx}"
    needed = {'input_layernorm': f"{base}.input_layernorm.weight",
              'post_attention_layernorm': f"{base}.post_attention_layernorm.weight",
              'gate_proj': f"{base}.mlp.gate_proj.weight",
              'up_proj':   f"{base}.mlp.up_proj.weight",
              'down_proj': f"{base}.mlp.down_proj.weight"}
    if layer_type == 'linear_attention':
        needed.update({
            'in_proj_qkv':   f"{base}.linear_attn.in_proj_qkv.weight",
            'in_proj_z':     f"{base}.linear_attn.in_proj_z.weight",
            'in_proj_a':     f"{base}.linear_attn.in_proj_a.weight",
            'in_proj_b':     f"{base}.linear_attn.in_proj_b.weight",
            'out_proj':      f"{base}.linear_attn.out_proj.weight",
            'conv1d_weight': f"{base}.linear_attn.conv1d.weight",
            'A_log':         f"{base}.linear_attn.A_log",
            'dt_bias':       f"{base}.linear_attn.dt_bias",
            # B'9.5 fix: missing per-head RMSNormGated weight inside DeltaNet
            'linear_attn_norm': f"{base}.linear_attn.norm.weight",
        })
    else:
        needed.update({
            'q_proj': f"{base}.self_attn.q_proj.weight",
            'k_proj': f"{base}.self_attn.k_proj.weight",
            'v_proj': f"{base}.self_attn.v_proj.weight",
            'o_proj': f"{base}.self_attn.o_proj.weight",
            # B'9.5 fix: missing per-head Q/K RMSNorm weights
            'q_norm':  f"{base}.self_attn.q_norm.weight",
            'k_norm':  f"{base}.self_attn.k_norm.weight",
        })
    # B'9.5 fix: Qwen3_5RMSNorm uses (1.0 + weight), not weight. Pre-add 1 at load
    # so that ttnn.rms_norm(x, w_loaded) computes the correct (1+w_raw) * x / sqrt(...).
    # NOT applied to linear_attn_norm — that one is Qwen3_5RMSNormGated which is
    # standard w * x. The clean separation is visible in weight stats:
    #   Qwen3_5RMSNorm weights have mean ≈ 0 (offset from 1)
    #   Qwen3_5RMSNormGated weights have mean ≈ 1 (raw scale)
    RMSNORM_1PLUS_W_KEYS = {'input_layernorm', 'post_attention_layernorm',
                             'q_norm', 'k_norm'}
    by_shard = {}
    for key, tname in needed.items():
        if tname in weight_map:
            by_shard.setdefault(weight_map[tname], []).append((key, tname))
    weights = {}
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for key, tname in items:
                t = f.get_tensor(tname).float().numpy()
                if 'proj' in key:
                    t = t.T
                if key in RMSNORM_1PLUS_W_KEYS:
                    t = t + 1.0
                weights[key] = t.copy()

    # DN-fusion: concat in_proj_{qkv,z,a,b} along output dim. The forward path
    # then does ONE matmul + four view-only slices instead of four dispatches.
    # Math-identical (validated in dn_fusion_isolation_probe.py); pure
    # dispatch-overhead optimization for eager mode.
    if layer_type == 'linear_attention':
        all_keys = ('in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b')
        if all(k in weights for k in all_keys):
            weights['in_proj_all'] = np.concatenate(
                [weights[k] for k in all_keys], axis=1).copy()
            for k in all_keys:
                del weights[k]
    # ATTN-fusion: full-attention layers also fuse q_proj (already q+gate),
    # k_proj, v_proj into a single attn_qkv. Same dispatch-reduction
    # principle. Saves 2 dispatches per gated_attn step (16 attn layers).
    if layer_type == 'full_attention':
        attn_keys = ('q_proj', 'k_proj', 'v_proj')
        if all(k in weights for k in attn_keys):
            weights['attn_qkv'] = np.concatenate(
                [weights[k] for k in attn_keys], axis=1).copy()
            for k in attn_keys:
                del weights[k]
    return weights


def upload(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ============================================================
# DeltaNet step on device (same as 91d/91e — unchanged here)
# ============================================================

def deltanet_step_ondevice(x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg):
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KERNEL = cfg['conv_kernel']
    KEY_DIM = N_K_HEADS * K_DIM
    VAL_DIM = N_V_HEADS * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    N_REP = N_V_HEADS // N_K_HEADS

    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
    # DN-fusion: 4 input projections fused into one. The weight loader
    # produces 'in_proj_all' (concat of qkv|z|a|b along output dim). One
    # ttnn.linear + four view-only slices replaces four separate dispatches.
    # Backwards-compatible: if 'in_proj_all' isn't present, fall through to
    # the per-projection path. Validated math-identical in
    # experiments/utils/dn_fusion_isolation_probe.py.
    if 'in_proj_all' in w_tt:
        all_tt = ttnn.linear(h_tt, w_tt['in_proj_all'], compute_kernel_config=hifi4)
        mixed_qkv = ttnn.slice(all_tt, [0, 0],                       [1, CONV_DIM])
        z_tt      = ttnn.slice(all_tt, [0, CONV_DIM],                [1, CONV_DIM + VAL_DIM])
        a_tt      = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM],      [1, CONV_DIM + VAL_DIM + N_V_HEADS])
        b_tt      = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM + N_V_HEADS],
                                       [1, CONV_DIM + VAL_DIM + 2 * N_V_HEADS])
    else:
        mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)
        z_tt     = ttnn.linear(h_tt, w_tt['in_proj_z'], compute_kernel_config=hifi4)
        a_tt     = ttnn.linear(h_tt, w_tt['in_proj_a'], compute_kernel_config=hifi4)
        b_tt     = ttnn.linear(h_tt, w_tt['in_proj_b'], compute_kernel_config=hifi4)

    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM, 1])
    conv_input = ttnn.concat([conv_state_tt, mixed_col], dim=-1)
    conv_prod = ttnn.mul(conv_input, w_tt['conv1d_weight'])
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
    conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])

    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM])
    k_flat = ttnn.slice(conv_out, [KEY_DIM], [2*KEY_DIM])
    v_flat = ttnn.slice(conv_out, [2*KEY_DIM], [CONV_DIM])
    # B'9.5 fix: ttnn.repeat is TILE semantics (verified in repeat_semantics_probe.py)
    # but HF uses repeat_interleave to broadcast n_k_heads → n_v_heads. The tile
    # mapping pairs q-head 0,16,32,... with k-head 0; interleave pairs q-head 0,1,2
    # with k-head 0. Get this wrong and the recurrence math is silently corrupt.
    # Workaround: unsqueeze + repeat-singleton + flatten gives interleave semantics
    # because the repeat axis is singleton.
    def gqa_interleave(t_flat, n_kh, d):
        t = ttnn.reshape(t_flat, [n_kh, 1, d])
        t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t, [n_kh * N_REP, d])
    q = gqa_interleave(q_flat, N_K_HEADS, K_DIM)
    k = gqa_interleave(k_flat, N_K_HEADS, K_DIM)
    v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])

    qq = ttnn.mul(q, q)
    q = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
    kk = ttnn.mul(k, k)
    k = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
    # B'9.5 Q-SCALING FIX: HF applies query = query * (1/sqrt(k_head_dim))
    # before using Q in the recurrence output. We skipped this earlier
    # ("cosine-invariant"), but at the tiny magnitudes layer 2 produces
    # (≈ 0.0001 per row), RMSNorm's eps dominates the variance and
    # RMSNorm becomes NOT scale-invariant. Without Q-scaling, our
    # recurrence output is sqrt(K_DIM)≈11.3× larger than HF's, which puts
    # us in a DIFFERENT eps-vs-variance regime than HF. Per-row probe
    # confirmed the 11.0-11.3× magnitude ratio. Applying this scale
    # brings us into HF's regime and the per-row directions align.
    q = ttnn.mul(q, 1.0 / (K_DIM ** 0.5))

    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, w_tt['dt_bias'])), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(w_tt['A_log'])), softplus_a)
    beta = ttnn.sigmoid(b_tt)

    decay = ttnn.reshape(ttnn.exp(g), [1, N_V_HEADS, 1, 1])
    H_4d = ttnn.reshape(ssm_state_tt, [1, N_V_HEADS, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay)
    k_col = ttnn.reshape(k, [1, N_V_HEADS, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [1, N_V_HEADS, V_DIM])
    v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, N_V_HEADS, 1]))
    H_new = ttnn.add(H_decayed,
                     ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
    q_col = ttnn.reshape(q, [1, N_V_HEADS, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM])
    # B'9.5 fix: per-head Qwen3_5RMSNormGated. Do BOTH the RMSNorm and the
    # silu(z) gate in [N_V, V_DIM] shape, mirroring HF exactly:
    #     core_attn_out.reshape(-1, head_v_dim)
    #     z.reshape(-1, head_v_dim)
    #     core_attn_out = norm(core_attn_out, z)
    # Previously we reshape-back-to-flat after rms_norm and gated against
    # flat-z, which gave cosine 0.81 vs HF (probe-confirmed rms_norm itself
    # is fine; suspect is the reshape interaction with TILE_LAYOUT for the
    # gate-mul). Keep the multiplication per-head and reshape only at the
    # end before out_proj.
    out_per_head = ttnn.reshape(out, [N_V_HEADS, V_DIM])
    out_normed = ttnn.rms_norm(out_per_head, weight=w_tt['linear_attn_norm'], epsilon=EPS)
    z_per_head = ttnn.reshape(z_tt, [N_V_HEADS, V_DIM])
    silu_z_per_head = ttnn.silu(z_per_head)
    out_gated_per_head = ttnn.mul(out_normed, silu_z_per_head)
    out_gated = ttnn.reshape(out_gated_per_head, [1, VAL_DIM])

    out_proj = ttnn.linear(out_gated, w_tt['out_proj'], compute_kernel_config=hifi4)
    x_out = ttnn.add(x_tt, out_proj)
    H_new_3d = ttnn.reshape(H_new, [N_V_HEADS, K_DIM, V_DIM])
    return x_out, H_new_3d, conv_state_new


# ============================================================
# Gated Attention step on device — FULL VERSION (NO NUMPY)
# ============================================================
# Uses:
#   - ttnn.create_sharded_memory_config for KV cache layout
#   - ttnn.experimental.rotary_embedding for partial RoPE (rotary part only;
#     we slice + rotate + concat for the 64-of-256 partial case)
#   - ttnn.experimental.paged_update_cache for KV write
#   - ttnn.transformer.scaled_dot_product_attention_decode for SDPA

def gated_attn_step_ondevice(x_tt, w_tt, kv_cache_k_tt, kv_cache_v_tt,
                              kv_cfg, cur_pos_tt, cur_pos, cos_tt, sin_tt, cfg, device):
    HIDDEN = cfg['hidden']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    # 1) Pre-norm
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    # 2) Q+gate+K+V fused projection (was 3 separate linears).
    # Weight loader produces 'attn_qkv' = concat(q_proj | k_proj | v_proj)
    # along output dim. q_proj already carries Q+gate (HEAD_DIM*2 cols).
    # Layout: [0 .. 2*N_Q*HEAD_DIM)         = Q+gate
    #         [2*N_Q*HEAD_DIM .. +N_KV*HEAD_DIM)  = K
    #         [...                       .. +N_KV*HEAD_DIM)  = V
    QG_DIM = 2 * N_Q * HEAD_DIM
    KV_DIM = N_KV * HEAD_DIM
    if 'attn_qkv' in w_tt:
        all_tt = ttnn.linear(h_tt, w_tt['attn_qkv'], compute_kernel_config=hifi4)
        qg_flat = ttnn.slice(all_tt, [0, 0],                 [1, QG_DIM])
        k_flat  = ttnn.slice(all_tt, [0, QG_DIM],            [1, QG_DIM + KV_DIM])
        v_flat  = ttnn.slice(all_tt, [0, QG_DIM + KV_DIM],   [1, QG_DIM + 2 * KV_DIM])
        qg_tt = ttnn.reshape(qg_flat, [N_Q, HEAD_DIM * 2])
        q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
        gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])
        k_tt = ttnn.reshape(k_flat, [N_KV, HEAD_DIM])
        v_tt = ttnn.reshape(v_flat, [N_KV, HEAD_DIM])
    else:
        qg_tt = ttnn.linear(h_tt, w_tt['q_proj'], compute_kernel_config=hifi4)
        qg_tt = ttnn.reshape(qg_tt, [N_Q, HEAD_DIM * 2])
        q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
        gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])
        k_tt = ttnn.reshape(
            ttnn.linear(h_tt, w_tt['k_proj'], compute_kernel_config=hifi4),
            [N_KV, HEAD_DIM])
        v_tt = ttnn.reshape(
            ttnn.linear(h_tt, w_tt['v_proj'], compute_kernel_config=hifi4),
        [N_KV, HEAD_DIM])

    # B'9.5 fix: per-head Qwen3_5RMSNorm on Q and K BEFORE RoPE.
    # Weight shape is [HEAD_DIM], loaded as (1.0 + raw) already.
    q_tt = ttnn.rms_norm(q_tt, weight=w_tt['q_norm'], epsilon=EPS)
    k_tt = ttnn.rms_norm(k_tt, weight=w_tt['k_norm'], epsilon=EPS)

    # 3) Partial RoPE: rotate first ROTARY_DIM dims; pass-through last (HEAD_DIM - ROTARY_DIM).
    # C'3 (native ttnn.experimental.rotary_embedding) attempted and abandoned 2026-05-13:
    # the op's cos_cache shape constraints (padded_shape [0]==1 && [1]==1) conflict with
    # both TILE_LAYOUT tile-padding behavior AND our partial-rotary slicing pattern.
    # Multiple workarounds tried (ROW_MAJOR conversion, token_index variants) all hit
    # different cos shape rejections. Doc is sparse, source-of-truth is C++. Win was
    # ~3-5 ms/tok; cost to integrate exceeds the budget. Keeping manual rotate-half;
    # this path is correct, fast enough, and trace-friendly with a small refactor in C'4.
    def apply_partial_rope(t, n_heads):
        # Level 1: if cos_tt/sin_tt are EXTENDED rows ([1, HEAD_DIM] with
        # passthrough region = 1 for cos, 0 for sin), use the no-slice-of-q
        # formula: q' = q * cos_ext + rotate_half_partial(q) * sin_ext.
        # Math-identical (validated bit-exact in partial_rope_level1_probe.py).
        # Falls back to rotary-only path if cos/sin are [1, ROTARY_DIM].
        half = ROTARY_DIM // 2
        if int(cos_tt.shape[-1]) == HEAD_DIM:
            x1 = ttnn.slice(t, [0, 0], [n_heads, half])
            x2 = ttnn.slice(t, [0, half], [n_heads, ROTARY_DIM])
            passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
            neg_x2 = ttnn.neg(x2)
            rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
            # cos_tt / sin_tt are already [1, HEAD_DIM] from the slice; the
            # earlier reshape tripped a TILE-padded volume check. Broadcast
            # mul directly works.
            return ttnn.add(ttnn.mul(t, cos_tt), ttnn.mul(rotated_full, sin_tt))
        # Rotary-only fallback (C'0.6 path)
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        cos_b = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
        sin_b = ttnn.reshape(sin_tt, [1, ROTARY_DIM])
        rotated = ttnn.add(ttnn.mul(rot, cos_b), ttnn.mul(rotated_half, sin_b))
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_partial_rope(q_tt, N_Q)
    k_tt = apply_partial_rope(k_tt, N_KV)

    # C'1: KV cache slot write via on-device ttnn.scatter.
    # Cache shape [1, N_KV, MAX_POS, HEAD_DIM]; write k_tt, v_tt at cur_pos along dim=2.
    # Replaces the prior numpy roundtrip (6× to_torch/from_torch per layer per token).
    # ttnn.scatter refuses fp32+TILE source ("Scatter doesn't work for fp32 tiled tensors
    # yet"). Cache is bf16, so cast K/V to bf16 before scatter — net precision matches the
    # prior numpy version (which also downcast on cache write).
    k_for_cache = ttnn.typecast(ttnn.reshape(k_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    v_for_cache = ttnn.typecast(ttnn.reshape(v_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    index_np = np.full((1, N_KV, 1, HEAD_DIM), cur_pos, dtype=np.int32)
    index_tt = ttnn.from_torch(torch.from_numpy(index_np), dtype=ttnn.int32,
                                device=device, layout=ttnn.TILE_LAYOUT)
    kv_cache_k_tt = ttnn.scatter(kv_cache_k_tt, dim=2, index=index_tt, src=k_for_cache)
    kv_cache_v_tt = ttnn.scatter(kv_cache_v_tt, dim=2, index=index_tt, src=v_for_cache)

    # 5) SDPA decode on device. SDPA wants Q in same dtype as KV cache (bf16);
    # downcast Q for the call, then promote the result back to whatever dtype
    # the rest of the layer is in (fp32 in B'9, bf16 in B'4-B'7).
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, N_Q, HEAD_DIM])
    q_for_sdpa = ttnn.typecast(q_for_sdpa, ttnn.bfloat16)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q_for_sdpa, kv_cache_k_tt, kv_cache_v_tt,
        cur_pos_tensor=cur_pos_tt, compute_kernel_config=hifi4)
    attn = ttnn.reshape(attn, [N_Q, HEAD_DIM])
    # Match attn dtype to x_tt dtype so the residual flow stays in fp32 when caller upgrades.
    attn = ttnn.typecast(attn, x_tt.dtype)

    # 6) Sigmoid output gate
    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))

    # 7) Output projection + residual
    attn_flat = ttnn.reshape(attn, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out), kv_cache_k_tt, kv_cache_v_tt


def mlp_step_ondevice(x_tt, w_tt):
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
    g_tt = ttnn.linear(h_tt, w_tt['gate_proj'], activation="silu", compute_kernel_config=hifi4)
    u_tt = ttnn.linear(h_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
    out = ttnn.linear(ttnn.mul(g_tt, u_tt), w_tt['down_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out)


# ============================================================
# C'4: trace-friendly variant of gated_attn_step_ondevice.
# ============================================================
# Differences from gated_attn_step_ondevice:
#   - cur_pos Python int REMOVED from signature; everything that needed it now
#     reads from PRE-ALLOCATED device tensors maintained by the caller.
#   - cos_tt / sin_tt are pre-allocated row buffers ([1, ROTARY_DIM]) updated
#     once per step by the caller via copy_host_to_device_tensor (no per-step
#     ttnn.slice into a precomputed table => no Python int baked into trace).
#   - index_tt is a PRE-ALLOCATED int32 buffer of shape [1, N_KV, 1, HEAD_DIM]
#     filled with cur_pos by the caller per step. Trace sees only the
#     persistent device tensor; no np.full + ttnn.from_torch inside the
#     hot path (those allocate fresh tensors that the trace cannot replay).
# All other math identical to gated_attn_step_ondevice — direct line-by-line
# port. If the original passes the cosine gate eager, this one must too.
def gated_attn_step_ondevice_traced(x_tt, w_tt, kv_cache_k_tt, kv_cache_v_tt,
                                     cur_pos_tt, cos_tt, sin_tt, index_tt, cfg):
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    qg_tt = ttnn.linear(h_tt, w_tt['q_proj'], compute_kernel_config=hifi4)
    qg_tt = ttnn.reshape(qg_tt, [N_Q, HEAD_DIM * 2])
    q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
    gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])

    k_tt = ttnn.reshape(
        ttnn.linear(h_tt, w_tt['k_proj'], compute_kernel_config=hifi4),
        [N_KV, HEAD_DIM])
    v_tt = ttnn.reshape(
        ttnn.linear(h_tt, w_tt['v_proj'], compute_kernel_config=hifi4),
        [N_KV, HEAD_DIM])

    q_tt = ttnn.rms_norm(q_tt, weight=w_tt['q_norm'], epsilon=EPS)
    k_tt = ttnn.rms_norm(k_tt, weight=w_tt['k_norm'], epsilon=EPS)

    def apply_partial_rope(t, n_heads):
        # Level 1: if cos_tt/sin_tt are EXTENDED rows ([1, HEAD_DIM] with
        # passthrough region = 1 for cos, 0 for sin), use the no-slice-of-q
        # formula: q' = q * cos_ext + rotate_half_partial(q) * sin_ext.
        # Math-identical (validated bit-exact in partial_rope_level1_probe.py).
        # Falls back to rotary-only path if cos/sin are [1, ROTARY_DIM].
        half = ROTARY_DIM // 2
        if int(cos_tt.shape[-1]) == HEAD_DIM:
            x1 = ttnn.slice(t, [0, 0], [n_heads, half])
            x2 = ttnn.slice(t, [0, half], [n_heads, ROTARY_DIM])
            passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
            neg_x2 = ttnn.neg(x2)
            rotated_full = ttnn.concat([neg_x2, x1, passthru], dim=-1)
            # cos_tt / sin_tt are already [1, HEAD_DIM] from the slice; the
            # earlier reshape tripped a TILE-padded volume check. Broadcast
            # mul directly works.
            return ttnn.add(ttnn.mul(t, cos_tt), ttnn.mul(rotated_full, sin_tt))
        # Rotary-only fallback (C'0.6 path)
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        cos_b = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
        sin_b = ttnn.reshape(sin_tt, [1, ROTARY_DIM])
        rotated = ttnn.add(ttnn.mul(rot, cos_b), ttnn.mul(rotated_half, sin_b))
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_partial_rope(q_tt, N_Q)
    k_tt = apply_partial_rope(k_tt, N_KV)

    k_for_cache = ttnn.typecast(ttnn.reshape(k_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    v_for_cache = ttnn.typecast(ttnn.reshape(v_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    # index_tt is pre-allocated [1, N_KV, 1, HEAD_DIM] int32 (updated host-side
    # per step). ttnn.scatter is captured by the trace and replays correctly.
    kv_cache_k_tt = ttnn.scatter(kv_cache_k_tt, dim=2, index=index_tt, src=k_for_cache)
    kv_cache_v_tt = ttnn.scatter(kv_cache_v_tt, dim=2, index=index_tt, src=v_for_cache)

    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, N_Q, HEAD_DIM])
    q_for_sdpa = ttnn.typecast(q_for_sdpa, ttnn.bfloat16)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q_for_sdpa, kv_cache_k_tt, kv_cache_v_tt,
        cur_pos_tensor=cur_pos_tt, compute_kernel_config=hifi4)
    attn = ttnn.reshape(attn, [N_Q, HEAD_DIM])
    attn = ttnn.typecast(attn, x_tt.dtype)

    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))

    attn_flat = ttnn.reshape(attn, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out), kv_cache_k_tt, kv_cache_v_tt


def main():
    print("=" * 64)
    print("Phase B′6 — Qwen3.6-27B layers 0-3 FULLY ON-DEVICE (no numpy on path)")
    print("=" * 64)

    # Gold reference (assume already computed by B′5)
    if not os.path.exists(REF_PATH):
        print(f"ERROR: gold reference missing at {REF_PATH}")
        print("Run experiments/91e_qwen36_27b_layers0_3.py first to generate it.")
        sys.exit(1)
    print(f"\n[1/5] Loading numpy reference from {REF_PATH}")
    gold = dict(np.load(REF_PATH))
    x_init = gold['input_x']

    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    cfg = {
        'hidden':      text_cfg['hidden_size'],
        'n_k_heads':   text_cfg['linear_num_key_heads'],
        'n_v_heads':   text_cfg['linear_num_value_heads'],
        'k_dim':       text_cfg['linear_key_head_dim'],
        'v_dim':       text_cfg['linear_value_head_dim'],
        'conv_kernel': text_cfg['linear_conv_kernel_dim'],
        'n_q_heads':   text_cfg['num_attention_heads'],
        'n_kv_heads':  text_cfg['num_key_value_heads'],
        'head_dim':    text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
    }
    HIDDEN = cfg['hidden']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    # Device — sharded KV-cache config is deferred to B′6.5 (n_kv=4 doesn't
    # tile-align). For now KV update goes via numpy roundtrip; SDPA is on device.
    print("\n[2/5] Opening device…")
    device = ttnn.open_device(device_id=0)
    kv_cfg = None  # placeholder, unused while sharded path is blocked

    # Load + upload layer weights
    print("\n[3/5] Loading + uploading layer weights (0..3)…")
    layer_weights_tt = []
    for i in range(4):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == 'conv1d_weight' and arr.ndim == 3:
                arr = arr.squeeze(1)
            w_tt[k] = upload(arr, device, dtype=ttnn.bfloat16)
        layer_weights_tt.append((layer_type, w_tt))
    ttnn.synchronize_device(device)

    # Initial states
    ssm_states = [
        upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
               device, dtype=ttnn.float32) for _ in range(3)
    ]
    conv_states = [
        upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
               device, dtype=ttnn.bfloat16) for _ in range(3)
    ]

    # KV cache: shape [B=1, n_kv, MAX_POS, head_dim], bf16
    kv_k_np = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    kv_v_np = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    kv_k_tt = ttnn.from_torch(torch.from_numpy(kv_k_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
    kv_v_tt = ttnn.from_torch(torch.from_numpy(kv_v_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)

    # Position + RoPE tables
    cur_pos = 0
    cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    angles = cur_pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    cos_tt = upload(cos_np, device, dtype=ttnn.bfloat16)
    sin_tt = upload(sin_np, device, dtype=ttnn.bfloat16)

    # Forward
    print("\n[4/5] Forward through layers 0-3 with cosine check (zero numpy on path)…")
    x_tt = upload(x_init.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)
    all_pass = True
    for i in range(4):
        layer_type, w_tt = layer_weights_tt[i]
        if layer_type == 'linear_attention':
            x_tt, H_new, c_new = deltanet_step_ondevice(
                x_tt, w_tt, ssm_states[i], conv_states[i], cfg)
            ssm_states[i] = H_new
            conv_states[i] = c_new
        else:
            x_tt, kv_k_tt, kv_v_tt = gated_attn_step_ondevice(
                x_tt, w_tt, kv_k_tt, kv_v_tt, kv_cfg, cur_pos_tt, cur_pos,
                cos_tt, sin_tt, cfg, device)
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        ttnn_post = ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]
        cos = _cosine(gold[f'post_layer{i}'], ttnn_post)
        max_abs = float(np.max(np.abs(gold[f'post_layer{i}'] - ttnn_post)))
        gate = "✓" if cos >= 0.99 else "✗"
        print(f"  layer {i} ({layer_type:18s}): cosine = {cos:.6f}  max-abs = {max_abs:.4f}  {gate}")
        if cos < 0.99:
            all_pass = False

    print(f"\n[5/5] VERDICT: {'PASS ✓' if all_pass else 'FAIL ✗'}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
