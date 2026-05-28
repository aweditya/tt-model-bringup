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
# P21 — runtime-selectable SDPA compute_kernel_config.
# ------------------------------------------------------------
# Reads a sentinel file ~/tt-xla/.cache/p21_sdpa_variant.txt at module
# import time. If absent or unrecognized, falls back to `hifi4` (production
# default). This lets cliff probes flip SDPA precision flags without a
# server restart — write the sentinel, then call the `reload_kernels` RPC.
#
# Variants (case-insensitive name in sentinel file):
#   A    : hifi4 (production default; HiFi4 + fp32_dest_acc_en=True)
#   B    : hifi4 + packer_l1_acc=True (matches 70b galaxy's compute_kernel_config_hifi4)
#   B2   : "compute_kernel_config_sdpa" from 70b galaxy
#          (HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False)
#          — same as A but explicit; included for completeness.
#   B3   : SDPA_DECODE_COMPUTE_PROGCFG from 70b galaxy
#          (HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False) — counterpoint;
#          tests whether SDPA decode really wants LESS precision.
#   B4   : Maxed-out (HiFi4 + fp32_dest_acc_en + packer_l1_acc + dst_full_sync_en)
# Any other token → fall back to A (with a stderr warning).
# ============================================================

_P21_SDPA_VARIANT_PATH = os.path.expanduser("~/tt-xla/.cache/p21_sdpa_variant.txt")

def _p21_make_sdpa_cfg(variant_name: str):
    v = (variant_name or "A").strip().upper()
    if v == "A":
        return hifi4, "A: HiFi4+fp32_dest_acc_en (production default)"
    if v == "B":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ), "B: HiFi4+fp32_dest_acc+packer_l1_acc (70b galaxy general-matmul style)"
    if v == "B2":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ), "B2: 70b galaxy compute_kernel_config_sdpa (HiFi4+fp32_dest_acc, no packer_l1_acc)"
    if v == "B3":
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ), "B3: 70b galaxy SDPA_DECODE_COMPUTE_PROGCFG (HiFi2, no fp32_dest_acc) — counter-test"
    if v == "B4":
        # dst_full_sync_en is supported on Wormhole; safe to pass on Blackhole P150
        # since it's a Wormhole compute config (P150 follows WH ABI for compute).
        try:
            return ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            ), "B4: HiFi4+fp32_dest_acc+packer_l1_acc+dst_full_sync_en (maxed)"
        except TypeError:
            # dst_full_sync_en kwarg absent in this ttnn build; degrade to B
            cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            return cfg, "B4-degraded: dst_full_sync_en kwarg absent → identical to B"
    sys.stderr.write(f"[91f] WARNING: unknown P21 SDPA variant {v!r}, falling back to A\n")
    return hifi4, f"A (fallback for unknown variant {v!r})"


def _p21_resolve_sdpa_cfg():
    variant = "A"
    try:
        if os.path.exists(_P21_SDPA_VARIANT_PATH):
            with open(_P21_SDPA_VARIANT_PATH) as f:
                variant = f.read().strip()
    except Exception:
        pass
    cfg, label = _p21_make_sdpa_cfg(variant)
    sys.stderr.write(f"[91f] SDPA kernel config: {label}\n")
    return cfg, variant


sdpa_kcfg, sdpa_kcfg_variant = _p21_resolve_sdpa_cfg()


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
        # QK normalize fusion constants. ttnn.rms_norm replaces the 11-op manual
        # sequence (mul+sum+rsqrt+mul ×2 + Q-scale) at 88% lower latency.
        # Math: rms_norm(x, weight=w, eps=EPS/K_DIM) = x * rsqrt(mean(x²) + EPS/K_DIM) * w
        #                                            = x * rsqrt((sum(x²) + EPS)/K_DIM) * w
        #                                            = x * sqrt(K_DIM) * rsqrt(sum(x²)+EPS) * w
        # For Q (current = q * rsqrt(sum+EPS) / sqrt(K_DIM)):  w = 1/K_DIM
        # For K (current = k * rsqrt(sum+EPS)):                w = 1/sqrt(K_DIM)
        # Validation: feedback_qk_normalize_fusion.md derivation + qk_normalize_decomp_probe.py
        K_DIM_LOCAL = 128  # Qwen3.6 linear_attn K_DIM (k_head_dim)
        weights['q_l2_scale'] = np.full(K_DIM_LOCAL, 1.0 / K_DIM_LOCAL, dtype=np.float32)
        weights['k_l2_scale'] = np.full(K_DIM_LOCAL, 1.0 / np.sqrt(K_DIM_LOCAL), dtype=np.float32)
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

    # QK normalize via ttnn.rms_norm fused kernel (replaces 11-op manual
    # sequence: mul+sum+rsqrt+mul ×2 + Q-scaling). 88.6% lower latency at
    # block level (0.797 → 0.091 ms/layer) ≈ 33 ms/tok at 48 layers.
    # Math is bit-identical to the prior path including B'9.5 fix:
    #   rms_norm(x, weight=w, epsilon=EPS/K_DIM) = x * rsqrt(mean(x²) + EPS/K_DIM) * w
    #                                            = x * sqrt(K_DIM) * rsqrt(sum(x²)+EPS) * w
    # Q weight = 1/K_DIM bakes in the B'9.5 Q-scaling. K weight = 1/sqrt(K_DIM)
    # leaves K unscaled (matches current K = k * rsqrt(sum(kk)+EPS)).
    # Validated 2026-05-13: qk_normalize_decomp_probe.py + feedback_qk_normalize_fusion.md
    EPS_RMS = EPS / K_DIM
    q = ttnn.rms_norm(q, weight=w_tt['q_l2_scale'], epsilon=EPS_RMS)
    k = ttnn.rms_norm(k, weight=w_tt['k_l2_scale'], epsilon=EPS_RMS)

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
    # V2 rotate-only RoPE (supersedes V1 Level 1). Math bit-identical
    # (max|Δ|=0 vs V1 in attn_step_rope_swap_probe.py) but 8.7% faster
    # at full gated_attn_step level — mul/add ops touch ROTARY_DIM=64
    # instead of full HEAD_DIM=256. Saves ~2.2 ms/tok at 16 attn layers.
    if int(cos_tt.shape[-1]) == HEAD_DIM:
        cos_rot_lifted = ttnn.slice(cos_tt, [0, 0], [1, ROTARY_DIM])
        sin_rot_lifted = ttnn.slice(sin_tt, [0, 0], [1, ROTARY_DIM])
    else:
        cos_rot_lifted = cos_tt
        sin_rot_lifted = sin_tt

    def apply_partial_rope(t, n_heads):
        half = ROTARY_DIM // 2
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        rotated = ttnn.add(ttnn.mul(rot, cos_rot_lifted),
                           ttnn.mul(rotated_half, sin_rot_lifted))
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_partial_rope(q_tt, N_Q)
    k_tt = apply_partial_rope(k_tt, N_KV)

    # C'1 → C'1+: KV cache slot write via ttnn.kv_cache.update_cache_for_token_.
    # Validated 7.2× faster than ttnn.scatter at production shape (0.019 vs 0.137 ms)
    # per experiments/utils/update_cache_probe.py on qb2. In-place mutation; no need
    # to reassign the cache reference. Per-token saving: ~3.78 ms (32 writes/tok).
    # INTERLEAVED memory config (the default) sidesteps Blackhole hang #16674.
    k_for_cache = ttnn.typecast(ttnn.reshape(k_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    v_for_cache = ttnn.typecast(ttnn.reshape(v_tt, [1, N_KV, 1, HEAD_DIM]), ttnn.bfloat16)
    ttnn.kv_cache.update_cache_for_token_(kv_cache_k_tt, k_for_cache, cur_pos)
    ttnn.kv_cache.update_cache_for_token_(kv_cache_v_tt, v_for_cache, cur_pos)

    # 5) SDPA decode on device. SDPA wants Q in same dtype as KV cache (bf16);
    # downcast Q for the call, then promote the result back to whatever dtype
    # the rest of the layer is in (fp32 in B'9, bf16 in B'4-B'7).
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, N_Q, HEAD_DIM])
    q_for_sdpa = ttnn.typecast(q_for_sdpa, ttnn.bfloat16)
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q_for_sdpa, kv_cache_k_tt, kv_cache_v_tt,
        cur_pos_tensor=cur_pos_tt, compute_kernel_config=sdpa_kcfg)
    attn = ttnn.reshape(attn, [N_Q, HEAD_DIM])
    # Match attn dtype to x_tt dtype so the residual flow stays in fp32 when caller upgrades.
    attn = ttnn.typecast(attn, x_tt.dtype)

    # 6) Sigmoid output gate
    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))

    # 7) Output projection + residual
    attn_flat = ttnn.reshape(attn, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out), kv_cache_k_tt, kv_cache_v_tt


def gated_attn_step_ondevice_paged(x_tt, w_tt, kv_cache_k_tt, kv_cache_v_tt,
                                     page_table_tt, cur_pos_tt, cos_tt, sin_tt, cfg):
    """Paged variant of gated_attn_step_ondevice. Same math + ATTN-QKV fusion +
    Level 1 partial RoPE as the original, but:
      - kv_cache_{k,v}_tt are in paged layout [max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM]
      - page_table_tt is [1, max_num_blocks_per_seq] int32 ROW_MAJOR
      - Cache writes use ttnn.experimental.paged_update_cache (sharded input)
      - SDPA uses ttnn.transformer.paged_scaled_dot_product_attention_decode

    Validated 2026-05-13:
      - paged SDPA bit-identical to non-paged at MAX_POS=256 (cos 0.999972 vs numpy)
      - paged SDPA latency at our shape: 0.115 ms (vs 0.113 ms non-paged — 1.8%)
      - paged_update_cache works on Blackhole with HEIGHT_SHARDED input
        (shard shape [32, head_dim] across NUM_USERS cores)

    Per-token decode cost vs non-paged path:
      <0.05 ms/tok at MAX_POS=256, ~1.25 ms/tok at MAX_POS=8192. Long-context
      unlock for ~0.6% perf cost.
    """
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    # ATTN-QKV fusion (mirrors gated_attn_step_ondevice)
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

    q_tt = ttnn.rms_norm(q_tt, weight=w_tt['q_norm'], epsilon=EPS)
    k_tt = ttnn.rms_norm(k_tt, weight=w_tt['k_norm'], epsilon=EPS)

    # Partial RoPE V2 rotate-only (supersedes V1 Level 1, math bit-identical)
    if int(cos_tt.shape[-1]) == HEAD_DIM:
        cos_rot_lifted = ttnn.slice(cos_tt, [0, 0], [1, ROTARY_DIM])
        sin_rot_lifted = ttnn.slice(sin_tt, [0, 0], [1, ROTARY_DIM])
    else:
        cos_rot_lifted = cos_tt
        sin_rot_lifted = sin_tt

    def apply_partial_rope(t, n_heads):
        half = ROTARY_DIM // 2
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        rotated = ttnn.add(ttnn.mul(rot, cos_rot_lifted),
                           ttnn.mul(rotated_half, sin_rot_lifted))
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_partial_rope(q_tt, N_Q)
    k_tt = apply_partial_rope(k_tt, N_KV)

    # Paged KV cache write: paged_update_cache wants input shape
    # [1, num_users=1, num_heads, head_dim] padded to TILE_HEIGHT, HEIGHT_SHARDED in L1
    # across NUM_USERS cores. For our shape: pad N_KV=4 -> 32 in dim 2.
    TILE_HEIGHT = 32
    NUM_USERS = 1
    device = x_tt.device()

    def shard_for_paged_write(tt_per_head):
        """tt_per_head: [N_KV, HEAD_DIM] bf16 TILE -> sharded [1, 1, 32, HEAD_DIM]."""
        # Reshape to [1, 1, N_KV, HEAD_DIM]
        t = ttnn.reshape(tt_per_head, [1, 1, N_KV, HEAD_DIM])
        # Pad dim -2 from N_KV to TILE_HEIGHT with zeros
        t = ttnn.pad(t, [[0, 0], [0, 0], [0, TILE_HEIGHT - N_KV], [0, 0]], value=0.0)
        # Cast to bf16 (paged_update_cache wants bf16)
        t = ttnn.typecast(t, ttnn.bfloat16)
        # Shard
        compute_grid = device.compute_with_storage_grid_size()
        shard_grid = ttnn.num_cores_to_corerangeset(NUM_USERS, compute_grid, row_wise=True)
        shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, HEAD_DIM],
                                       ttnn.ShardOrientation.ROW_MAJOR)
        mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                                       ttnn.BufferType.L1, shard_spec)
        return ttnn.to_memory_config(t, mem_cfg)

    k_sharded = shard_for_paged_write(k_tt)
    v_sharded = shard_for_paged_write(v_tt)
    ttnn.experimental.paged_update_cache(kv_cache_k_tt, k_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)
    ttnn.experimental.paged_update_cache(kv_cache_v_tt, v_sharded,
                                           update_idxs_tensor=cur_pos_tt,
                                           page_table=page_table_tt)

    # Paged SDPA decode
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, N_Q, HEAD_DIM])
    q_for_sdpa = ttnn.typecast(q_for_sdpa, ttnn.bfloat16)
    attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, kv_cache_k_tt, kv_cache_v_tt, page_table_tt,
        cur_pos_tensor=cur_pos_tt, compute_kernel_config=sdpa_kcfg)
    attn = ttnn.reshape(attn, [N_Q, HEAD_DIM])
    attn = ttnn.typecast(attn, x_tt.dtype)

    # Sigmoid gate + output projection + residual
    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))
    attn_flat = ttnn.reshape(attn, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out)


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
    # C'4 v4: save input cache references so we can commit new state back at the
    # end via in-trace ttnn.copy. Validated in trace_state_thread_probe.py —
    # ttnn.copy(new, original_buf) inside a trace correctly persists state
    # across execute_trace calls. Caller pre-allocates kv_cache_{k,v}_tt and
    # NEVER rebinds; this function mutates contents in place via the copy.
    k_cache_in = kv_cache_k_tt
    v_cache_in = kv_cache_v_tt

    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    # ATTN-QKV fusion: weight loader concats q_proj | k_proj | v_proj into
    # attn_qkv and deletes the individual keys. Mirror the eager kernel's
    # attn_qkv path so the traced variant works against the same weight dict.
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

    q_tt = ttnn.rms_norm(q_tt, weight=w_tt['q_norm'], epsilon=EPS)
    k_tt = ttnn.rms_norm(k_tt, weight=w_tt['k_norm'], epsilon=EPS)

    # V2 rotate-only RoPE (traced variant — same as eager). Math bit-identical.
    if int(cos_tt.shape[-1]) == HEAD_DIM:
        cos_rot_lifted = ttnn.slice(cos_tt, [0, 0], [1, ROTARY_DIM])
        sin_rot_lifted = ttnn.slice(sin_tt, [0, 0], [1, ROTARY_DIM])
    else:
        cos_rot_lifted = cos_tt
        sin_rot_lifted = sin_tt

    def apply_partial_rope(t, n_heads):
        half = ROTARY_DIM // 2
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        rotated = ttnn.add(ttnn.mul(rot, cos_rot_lifted),
                           ttnn.mul(rotated_half, sin_rot_lifted))
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
        cur_pos_tensor=cur_pos_tt, compute_kernel_config=sdpa_kcfg)
    attn = ttnn.reshape(attn, [N_Q, HEAD_DIM])
    attn = ttnn.typecast(attn, x_tt.dtype)

    attn = ttnn.mul(attn, ttnn.sigmoid(gate_tt))

    attn_flat = ttnn.reshape(attn, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    # C'4 v4 state commit: write the post-scatter cache back to the input buffer.
    # The scatter on kv_cache_{k,v}_tt above produced new tensors (functional);
    # these copies are what makes the new state visible to the NEXT execute_trace
    # call (which reads from k_cache_in / v_cache_in addresses).
    ttnn.copy(kv_cache_k_tt, k_cache_in)
    ttnn.copy(kv_cache_v_tt, v_cache_in)
    return ttnn.add(x_tt, out)


def deltanet_step_ondevice_traced(x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg):
    """C'4 v4 trace-friendly wrapper around deltanet_step_ondevice.

    Same compute as the eager version, but the new SSM + conv state are
    committed back to the input buffers via in-trace ttnn.copy at the end.
    Caller pre-allocates ssm_state_tt and conv_state_tt and NEVER rebinds them;
    each execute_trace updates contents in place via the copies.

    Validated mechanism: trace_state_thread_probe.py.
    """
    x_out, H_new_3d, conv_state_new = deltanet_step_ondevice(
        x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg)
    ttnn.copy(H_new_3d, ssm_state_tt)
    ttnn.copy(conv_state_new, conv_state_tt)
    return x_out


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
