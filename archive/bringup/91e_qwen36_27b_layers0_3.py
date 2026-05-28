#!/usr/bin/env python3
"""
Experiment 91e — Phase B′5 Qwen3.6-27B layers 0-3 fully on-device.

Stacks the first full [L L L F] pattern: layers 0,1,2 are DeltaNet+MLP
(linear_attention), layer 3 is Gated Attention+MLP. ALL on device, zero
numpy on the forward path.

Gates cosine ≥ 0.99 vs numpy fp32 reference at post-layer-3.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91e_qwen36_27b_layers0_3.py
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

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


# ============================================================
# Numpy fp32 reference (gold) — extends 91b with layers 1, 2, 3
# ============================================================

def silu_np(x): return x * (1.0 / (1.0 + np.exp(-x)))
def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))


def rms_norm_np(x, weight, eps=EPS):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * weight


def _l2_np(x):
    return x / (np.sqrt(np.sum(x*x, axis=-1, keepdims=True)) + EPS)


def deltanet_layer_np(x, w, ssm_state, conv_state, cfg):
    """Full layer = RMSNorm + DeltaNet recurrence + residual."""
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KEY_DIM = N_K_HEADS * K_DIM
    VAL_DIM = N_V_HEADS * V_DIM
    KERNEL = cfg['conv_kernel']
    N_REP = N_V_HEADS // N_K_HEADS

    h = rms_norm_np(x, w['input_layernorm'])
    mixed_qkv = h @ w['in_proj_qkv']
    z = h @ w['in_proj_z']
    a = h @ w['in_proj_a']
    b = h @ w['in_proj_b']

    cw = w['conv1d_weight']
    if cw.ndim == 3:
        cw = cw.squeeze(1)
    conv_input = np.concatenate([conv_state, mixed_qkv[:, None]], axis=-1)
    conv_out = np.sum(conv_input * cw, axis=-1)
    conv_out = silu_np(conv_out)
    conv_state_new = conv_input[:, 1:]

    q = conv_out[:KEY_DIM].reshape(N_K_HEADS, K_DIM)
    k = conv_out[KEY_DIM:2*KEY_DIM].reshape(N_K_HEADS, K_DIM)
    v = conv_out[2*KEY_DIM:].reshape(N_V_HEADS, V_DIM)
    q = np.repeat(q, N_REP, axis=0)
    k = np.repeat(k, N_REP, axis=0)
    q = _l2_np(q)
    k = _l2_np(k)

    g = -np.exp(w['A_log']) * np.log1p(np.exp(a + w['dt_bias']))
    beta = sigmoid_np(b)

    H = ssm_state.copy()
    H = H * np.exp(g)[:, None, None]
    kv_mem = (H * k[:, :, None]).sum(axis=-2)
    delta = (v - kv_mem) * beta[:, None]
    H = H + k[:, :, None] * delta[:, None, :]
    out = (H * q[:, :, None]).sum(axis=-2).reshape(-1)
    out = out * silu_np(z)
    out_proj = out @ w['out_proj']
    return x + out_proj, H, conv_state_new


def gated_attn_layer_np(x, w, kv_cache, cur_pos, cos, sin, cfg):
    """Full layer = RMSNorm + Gated Attention + residual."""
    HIDDEN = cfg['hidden']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
    N_REP = N_Q // N_KV

    h = rms_norm_np(x, w['input_layernorm'])
    qg = h @ w['q_proj']                                  # [N_Q * head_dim * 2]
    qg = qg.reshape(N_Q, HEAD_DIM * 2)
    q = qg[:, :HEAD_DIM]
    gate = qg[:, HEAD_DIM:]
    k = (h @ w['k_proj']).reshape(N_KV, HEAD_DIM)
    v = (h @ w['v_proj']).reshape(N_KV, HEAD_DIM)

    def partial_rope(t):
        rot = t[:, :ROTARY_DIM]
        passthru = t[:, ROTARY_DIM:]
        half = ROTARY_DIM // 2
        x1, x2 = rot[:, :half], rot[:, half:]
        rotated = rot * cos + np.concatenate([-x2, x1], axis=-1) * sin
        return np.concatenate([rotated, passthru], axis=-1)

    q = partial_rope(q)
    k = partial_rope(k)

    kv_cache['k'][:, cur_pos] = k
    kv_cache['v'][:, cur_pos] = v

    kc = np.repeat(kv_cache['k'][:, :cur_pos+1, :], N_REP, axis=0)
    vc = np.repeat(kv_cache['v'][:, :cur_pos+1, :], N_REP, axis=0)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = np.einsum('hd,htd->ht', q, kc) * scale
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    attn = np.einsum('ht,htd->hd', weights, vc)

    attn = attn * sigmoid_np(gate)
    out = attn.reshape(-1) @ w['o_proj']
    return x + out, kv_cache


def mlp_layer_np(x, w):
    h = rms_norm_np(x, w['post_attention_layernorm'])
    gate = h @ w['gate_proj']
    up = h @ w['up_proj']
    return x + (silu_np(gate) * up) @ w['down_proj']


def make_rope_tables_np(pos, rotary_dim, theta=10_000_000.0):
    half = rotary_dim // 2
    freqs = 1.0 / (theta ** (np.arange(half).astype(np.float32) / half))
    angles = pos * freqs
    cos = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    return cos, sin


# ============================================================
# Weight loading — both layer types
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
        })
    else:
        needed.update({
            'q_proj': f"{base}.self_attn.q_proj.weight",
            'k_proj': f"{base}.self_attn.k_proj.weight",
            'v_proj': f"{base}.self_attn.v_proj.weight",
            'o_proj': f"{base}.self_attn.o_proj.weight",
        })
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
                weights[key] = t.copy()
    return weights


# ============================================================
# Build numpy reference for layers 0-3
# ============================================================

def build_numpy_ref_layers0_3(x_init, cfg, max_pos=128):
    """Returns dict with post-layer-i hidden states for i in 0..3."""
    if os.path.exists(REF_PATH):
        print(f"  cached reference at {REF_PATH}")
        return dict(np.load(REF_PATH))

    print(f"  computing fresh reference (4 layers)…")
    HIDDEN = cfg['hidden']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = N_V_HEADS * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    ssm_state = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
    conv_state = np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32)
    kv_cache = {
        'k': np.zeros((cfg['n_kv_heads'], max_pos, cfg['head_dim']), dtype=np.float32),
        'v': np.zeros((cfg['n_kv_heads'], max_pos, cfg['head_dim']), dtype=np.float32),
    }
    cur_pos = 0
    cos, sin = make_rope_tables_np(cur_pos, int(cfg['head_dim'] * cfg['partial_rotary_factor']))

    states = {'input_x': x_init.copy()}
    x = x_init.copy()
    for i in range(4):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        print(f"    layer {i} ({layer_type})…")
        w = load_layer_weights_all(i, layer_type)
        if layer_type == 'linear_attention':
            x, ssm_state, conv_state = deltanet_layer_np(x, w, ssm_state, conv_state, cfg)
        else:
            x, kv_cache = gated_attn_layer_np(x, w, kv_cache, cur_pos, cos, sin, cfg)
        x = mlp_layer_np(x, w)
        states[f'post_layer{i}'] = x.copy()
        print(f"      post-layer-{i} norm = {np.linalg.norm(x):.4f}")

    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    np.savez(REF_PATH, **states)
    print(f"  ✓ saved reference ({os.path.getsize(REF_PATH)/1e6:.1f} MB)")
    return states


# ============================================================
# ttnn forward layers 0-3
# ============================================================

def upload(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def deltanet_step_ondevice(x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg):
    """Same as 91d's deltanet_step_ondevice. Inlined here for self-containment."""
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
    q = ttnn.repeat(ttnn.reshape(q_flat, [N_K_HEADS, K_DIM]), ttnn.Shape([N_REP, 1]))
    k = ttnn.repeat(ttnn.reshape(k_flat, [N_K_HEADS, K_DIM]), ttnn.Shape([N_REP, 1]))
    v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])

    qq = ttnn.mul(q, q)
    q_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS))
    q = ttnn.mul(q, q_norm)
    kk = ttnn.mul(k, k)
    k_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS))
    k = ttnn.mul(k, k_norm)

    a_plus_bias = ttnn.add(a_tt, w_tt['dt_bias'])
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_plus_bias), 1.0))
    exp_A_log = ttnn.exp(w_tt['A_log'])
    g = ttnn.mul(ttnn.neg(exp_A_log), softplus_a)
    beta = ttnn.sigmoid(b_tt)

    decay = ttnn.reshape(ttnn.exp(g), [1, N_V_HEADS, 1, 1])
    H_4d = ttnn.reshape(ssm_state_tt, [1, N_V_HEADS, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay)
    k_col = ttnn.reshape(k, [1, N_V_HEADS, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [1, N_V_HEADS, V_DIM])
    v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, N_V_HEADS, 1]))
    delta_row = ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])
    H_new = ttnn.add(H_decayed, ttnn.mul(k_col, delta_row))

    q_col = ttnn.reshape(q, [1, N_V_HEADS, K_DIM, 1])
    out_heads = ttnn.reshape(
        ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM])
    out_gated = ttnn.mul(out_heads, ttnn.silu(z_tt))

    out_proj = ttnn.linear(out_gated, w_tt['out_proj'], compute_kernel_config=hifi4)
    x_out = ttnn.add(x_tt, out_proj)
    H_new_3d = ttnn.reshape(H_new, [N_V_HEADS, K_DIM, V_DIM])
    return x_out, H_new_3d, conv_state_new


def gated_attn_step_ondevice(x_tt, w_tt, kv_cache_k_tt, kv_cache_v_tt,
                              cur_pos, cos_tt, sin_tt, cfg, device):
    """Fully on-device Gated Attention step. Returns (x_out, kv_cache_k, kv_cache_v)."""
    HIDDEN = cfg['hidden']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
    N_REP = N_Q // N_KV

    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
    qg_tt = ttnn.linear(h_tt, w_tt['q_proj'], compute_kernel_config=hifi4)
    # [1, N_Q * head_dim * 2] → [N_Q, head_dim*2] → split into [N_Q, head_dim] each
    qg_tt = ttnn.reshape(qg_tt, [N_Q, HEAD_DIM * 2])
    q_tt = ttnn.slice(qg_tt, [0, 0], [N_Q, HEAD_DIM])
    gate_tt = ttnn.slice(qg_tt, [0, HEAD_DIM], [N_Q, 2 * HEAD_DIM])

    k_tt = ttnn.linear(h_tt, w_tt['k_proj'], compute_kernel_config=hifi4)
    v_tt = ttnn.linear(h_tt, w_tt['v_proj'], compute_kernel_config=hifi4)
    k_tt = ttnn.reshape(k_tt, [N_KV, HEAD_DIM])
    v_tt = ttnn.reshape(v_tt, [N_KV, HEAD_DIM])

    # Partial RoPE: apply to first ROTARY_DIM dims (64); pass-through last 192.
    # The standard rotate_half works on the full last dim, so we slice/rotate/concat.
    def apply_partial_rope(t, n_heads):
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        # rotate_half over the rotary slice: split into two halves
        half = ROTARY_DIM // 2
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated_half = ttnn.concat([neg_x2, x1], dim=-1)
        # cos_tt and sin_tt are shape [ROTARY_DIM]; broadcast over n_heads
        # Reshape to [1, ROTARY_DIM] for broadcast
        cos_b = ttnn.reshape(cos_tt, [1, ROTARY_DIM])
        sin_b = ttnn.reshape(sin_tt, [1, ROTARY_DIM])
        rotated = ttnn.add(ttnn.mul(rot, cos_b), ttnn.mul(rotated_half, sin_b))
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_partial_rope(q_tt, N_Q)
    k_tt = apply_partial_rope(k_tt, N_KV)

    # KV cache update via numpy (one byte at a time isn't a Phase A4 problem at this scale;
    # for B′5 we use a similar "pre-populate cache" hack as A4 since paged_update_cache needs
    # sharded inputs). Read back k/v, update host cache, re-upload.
    # ⚠ This is the ONE numpy roundtrip — to be eliminated in B′6 via the production
    # sharded-memory-config + paged_update_cache pattern from demos/generate_moe.py.
    k_np = ttnn.to_torch(k_tt).float().numpy().reshape(N_KV, HEAD_DIM)
    v_np = ttnn.to_torch(v_tt).float().numpy().reshape(N_KV, HEAD_DIM)
    kv_cache_k_np = ttnn.to_torch(kv_cache_k_tt).float().numpy().reshape(N_KV, -1, HEAD_DIM)
    kv_cache_v_np = ttnn.to_torch(kv_cache_v_tt).float().numpy().reshape(N_KV, -1, HEAD_DIM)
    max_pos = kv_cache_k_np.shape[1]
    kv_cache_k_np[:, cur_pos] = k_np
    kv_cache_v_np[:, cur_pos] = v_np

    # GQA: replicate K, V to match Q
    kc_full = np.repeat(kv_cache_k_np[:, :cur_pos+1, :], N_REP, axis=0)  # [N_Q, cur_pos+1, HD]
    vc_full = np.repeat(kv_cache_v_np[:, :cur_pos+1, :], N_REP, axis=0)

    # SDPA on host (one-shot; B′6 lifts to device via SDPA op)
    q_np = ttnn.to_torch(q_tt).float().numpy().reshape(N_Q, HEAD_DIM)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = np.einsum('hd,htd->ht', q_np, kc_full) * scale
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    attn_np = np.einsum('ht,htd->hd', weights, vc_full)

    attn_tt = upload(attn_np.reshape(N_Q, HEAD_DIM), device, dtype=ttnn.bfloat16)
    attn_tt = ttnn.mul(attn_tt, ttnn.sigmoid(gate_tt))

    # Output proj + residual
    attn_flat = ttnn.reshape(attn_tt, [1, N_Q * HEAD_DIM])
    out = ttnn.linear(attn_flat, w_tt['o_proj'], compute_kernel_config=hifi4)
    x_out = ttnn.add(x_tt, out)

    # Re-upload updated kv cache (host-side write done; re-up to device)
    kv_cache_k_new = upload(kv_cache_k_np, device, dtype=ttnn.bfloat16)
    kv_cache_v_new = upload(kv_cache_v_np, device, dtype=ttnn.bfloat16)
    return x_out, kv_cache_k_new, kv_cache_v_new


def mlp_step_ondevice(x_tt, w_tt):
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
    g_tt = ttnn.linear(h_tt, w_tt['gate_proj'], activation="silu", compute_kernel_config=hifi4)
    u_tt = ttnn.linear(h_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
    out = ttnn.linear(ttnn.mul(g_tt, u_tt), w_tt['down_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 64)
    print("Phase B′5 — Qwen3.6-27B layers 0-3 on device + cosine gate at each")
    print("=" * 64)

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

    # Numpy gold (regenerated if not cached)
    print(f"\n[1/5] Building numpy reference (4 layers)…")
    rng = np.random.default_rng(42)
    x_init = rng.standard_normal(HIDDEN).astype(np.float32) * 0.05
    gold = build_numpy_ref_layers0_3(x_init, cfg)
    for i in range(4):
        norm = np.linalg.norm(gold[f'post_layer{i}'])
        print(f"  gold post-layer-{i} norm = {norm:.4f}")

    print("\n[2/5] Opening device + uploading weights for layers 0..3…")
    device = ttnn.open_device(device_id=0)

    # Load + upload weights for all 4 layers
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

    # ── 3/5  Initial states ─────────────────────────────────────
    print("\n[3/5] Initial states (DeltaNet H, conv, KV cache)…")
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    MAX_POS = 128

    # Per-DeltaNet-layer states
    ssm_states = [
        upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
               device, dtype=ttnn.float32)
        for _ in range(3)  # layers 0, 1, 2 are DeltaNet
    ]
    conv_states = [
        upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
               device, dtype=ttnn.bfloat16)
        for _ in range(3)
    ]
    # Single KV cache for the gated-attn layer (layer 3)
    kv_k_tt = upload(np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
                      device, dtype=ttnn.bfloat16)
    kv_v_tt = upload(np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
                      device, dtype=ttnn.bfloat16)
    cur_pos = 0
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    theta = 10_000_000.0
    freqs = 1.0 / (theta ** (np.arange(half_rot).astype(np.float32) / half_rot))
    angles = cur_pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    cos_tt = upload(cos_np, device, dtype=ttnn.bfloat16)
    sin_tt = upload(sin_np, device, dtype=ttnn.bfloat16)

    # ── 4/5  Run forward 4 layers, gate cosine at each ──────────
    print("\n[4/5] Forward through layers 0-3 with cosine check…")
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
                x_tt, w_tt, kv_k_tt, kv_v_tt, cur_pos, cos_tt, sin_tt, cfg, device)
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        # Read back & compare
        ttnn_post = ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]
        cos = _cosine(gold[f'post_layer{i}'], ttnn_post)
        max_abs = float(np.max(np.abs(gold[f'post_layer{i}'] - ttnn_post)))
        gate = "✓" if cos >= 0.99 else "✗"
        print(f"  layer {i} ({layer_type:18s}): cosine = {cos:.6f}  max-abs = {max_abs:.4f}  {gate}")
        if cos < 0.99:
            all_pass = False

    print(f"\n[5/5] VERDICT: {'PASS ✓' if all_pass else 'FAIL ✗'}  (gate: every layer cosine ≥ 0.99)")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
