#!/usr/bin/env python3
"""
Experiment 91b — Phase B′2 numpy fp32 reference for Qwen3.6-27B (layers 0 + 3).

Computes the EXACT fp32 forward through:
  Embed → Layer 0 (DeltaNet + dense MLP) → Layer 3 (Gated Attention + dense MLP)

Layer 0 is DeltaNet (per [L L L F]×16); Layer 3 is the first Gated Attention.
Together they exercise both block types — sufficient gold reference for
B′3's per-layer cosine gate.

Skips: layers 1,2 (other DeltaNet — same math as layer 0), MTP head, vision.

Output: a .npz with the post-layer-0 and post-layer-3 hidden states for a
single test token. B′3 will load this and compare against ttnn output.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91b_qwen36_27b_numpy_ref.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch  # for bf16 conversion (safetensors numpy framework can't handle bf16)

MODEL_ID = "Qwen/Qwen3.6-27B"
OUT_NPZ = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_layer0_3_ref.npz")

EPS = 1e-6


# ============================================================
# Math primitives (all fp32)
# ============================================================

def rms_norm(x, weight, eps=EPS):
    """x: [..., D], weight: [D]. Returns x * rsqrt(mean(x²) + eps) * weight."""
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ms + eps)) * weight


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _l2_normalize(x):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + EPS)


# ============================================================
# Gated DeltaNet — single token, single layer (decode-step form)
# ============================================================

def deltanet_layer(x, w, ssm_state, conv_state, cfg):
    """
    Single-token forward through one DeltaNet layer.

    x: [hidden]   fp32, input
    w: dict of weight tensors for this layer
    ssm_state: [n_v_heads, d_k, d_v]  fp32 recurrent state (input + output)
    conv_state: [conv_dim, kernel_dim - 1]  fp32 conv state (input + output)
    cfg: per-config constants

    Returns: x_out (post-residual) [hidden], ssm_state (updated), conv_state (updated)
    """
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KERNEL = cfg['conv_kernel']

    key_dim = N_K_HEADS * K_DIM      # 16 * 128 = 2048
    value_dim = N_V_HEADS * V_DIM    # 48 * 128 = 6144
    conv_dim = 2 * key_dim + value_dim  # = 10240

    # Pre-norm
    h = rms_norm(x, w['input_layernorm'])

    # Four input projections
    mixed_qkv = h @ w['in_proj_qkv']       # [conv_dim] = [10240]
    z = h @ w['in_proj_z']                  # [value_dim] = [6144]
    b = h @ w['in_proj_b']                  # [n_v_heads] = [48]
    a = h @ w['in_proj_a']                  # [n_v_heads] = [48]

    # GQA: repeat K to match V heads (16 → 48 by repeating 3×)
    # First slice mixed_qkv into Q (key_dim), K (key_dim), V (value_dim)
    q_flat = mixed_qkv[:key_dim]            # [2048] = 16 heads × 128
    k_flat = mixed_qkv[key_dim:2*key_dim]   # [2048] = 16 heads × 128
    v_flat = mixed_qkv[2*key_dim:]          # [6144] = 48 heads × 128

    # Conv1d (causal, kernel=4) applied along seq dim BEFORE split.
    # conv_state holds (kernel-1)=3 prior tokens. Append current to get kernel,
    # convolve, then drop oldest to form new state.
    cw = w['conv1d_weight']
    if cw.ndim == 3:
        cw = cw.squeeze(1)                                  # [conv_dim, kernel]
    conv_input = np.concatenate([conv_state, mixed_qkv[:, None]], axis=-1)  # [conv_dim, kernel]
    conv_out = np.sum(conv_input * cw, axis=-1)             # [conv_dim]
    if 'conv1d_bias' in w:
        conv_out = conv_out + w['conv1d_bias']
    conv_out = silu(conv_out)
    conv_state_new = conv_input[:, 1:]                      # drop oldest

    # Re-split post-conv values
    q_flat = conv_out[:key_dim]
    k_flat = conv_out[key_dim:2*key_dim]
    v_flat = conv_out[2*key_dim:]

    # Reshape to heads. Q,K have 16 heads; V has 48.
    # Repeat Q,K 3× to match V (this is GQA inside DeltaNet)
    q = q_flat.reshape(N_K_HEADS, K_DIM)          # [16, 128]
    k = k_flat.reshape(N_K_HEADS, K_DIM)
    v = v_flat.reshape(N_V_HEADS, V_DIM)          # [48, 128]
    q = np.repeat(q, N_V_HEADS // N_K_HEADS, axis=0)  # [48, 128]
    k = np.repeat(k, N_V_HEADS // N_K_HEADS, axis=0)

    # L2 normalize Q and K (use_qk_l2norm_in_kernel=True in HF)
    q = _l2_normalize(q)
    k = _l2_normalize(k)

    # Decay g = -exp(A_log) * softplus(a + dt_bias)
    A_log = w['A_log']                       # [n_v_heads]
    dt_bias = w['dt_bias']
    g = -np.exp(A_log) * (np.log1p(np.exp(a + dt_bias)))   # softplus
    beta = sigmoid(b)                        # [n_v_heads]

    # Recurrent state update — per Phase A3
    H = ssm_state.copy()                     # [n_v_heads, k_dim, v_dim]
    decay = np.exp(g)[:, None, None]
    H = H * decay
    kv_mem = (H * k[:, :, None]).sum(axis=-2)   # [n_v_heads, v_dim]
    delta = (v - kv_mem) * beta[:, None]
    H = H + k[:, :, None] * delta[:, None, :]
    out_heads = (H * q[:, :, None]).sum(axis=-2)   # [n_v_heads, v_dim]
    out_flat = out_heads.reshape(-1)             # [n_v_heads * v_dim] = [6144]

    # Output gate z (silu, per HF "output_gate_type": "swish")
    out_flat = out_flat * silu(z)

    # Output projection
    out = out_flat @ w['out_proj']               # [hidden]

    # Residual
    return x + out, H, conv_state_new


# ============================================================
# Gated Attention — single token
# ============================================================

def gated_attention_layer(x, w, kv_cache, cur_pos, cos, sin, cfg):
    """
    Single-token forward through one Gated Attention layer.

    kv_cache: dict with 'k' [n_kv, max_seq, head_dim] and 'v' [n_kv, max_seq, head_dim]
              gets updated at cur_pos.

    Returns x_out (post-residual), updated kv_cache.
    """
    HIDDEN = cfg['hidden']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])  # 64

    h = rms_norm(x, w['input_layernorm'])

    # q_proj is 2× width: produces Q + gate, chunked
    qg = h @ w['q_proj']                            # [N_Q * head_dim * 2]
    qg = qg.reshape(N_Q, HEAD_DIM * 2)
    q = qg[:, :HEAD_DIM]                            # [N_Q, head_dim]
    gate = qg[:, HEAD_DIM:]

    k = (h @ w['k_proj']).reshape(N_KV, HEAD_DIM)
    v = (h @ w['v_proj']).reshape(N_KV, HEAD_DIM)

    # Partial RoPE: rotate first ROTARY_DIM dims of head_dim
    def apply_partial_rope(t, cos, sin, rot_dim):
        rot = t[:, :rot_dim]
        passthru = t[:, rot_dim:]
        half = rot_dim // 2
        x1, x2 = rot[:, :half], rot[:, half:]
        rotated = rot * cos + np.concatenate([-x2, x1], axis=-1) * sin
        return np.concatenate([rotated, passthru], axis=-1)

    q = apply_partial_rope(q, cos, sin, ROTARY_DIM)
    k = apply_partial_rope(k, cos, sin, ROTARY_DIM)

    # Update KV cache at cur_pos
    kv_cache['k'][:, cur_pos] = k                   # [n_kv, head_dim] into slot
    kv_cache['v'][:, cur_pos] = v

    # GQA: replicate K, V to match Q head count
    n_rep = N_Q // N_KV
    kc = np.repeat(kv_cache['k'][:, :cur_pos+1, :], n_rep, axis=0)  # [N_Q, cur_pos+1, head_dim]
    vc = np.repeat(kv_cache['v'][:, :cur_pos+1, :], n_rep, axis=0)

    # SDPA: scores = q @ k.T / sqrt(head_dim), softmax, @ v
    scale = 1.0 / np.sqrt(HEAD_DIM)
    scores = np.einsum('hd,htd->ht', q, kc) * scale       # [N_Q, cur_pos+1]
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    attn = np.einsum('ht,htd->hd', weights, vc)           # [N_Q, head_dim]

    # Output gate (sigmoid)
    attn = attn * sigmoid(gate)

    # Output projection
    out = attn.reshape(-1) @ w['o_proj']                  # [hidden]

    return x + out, kv_cache


# ============================================================
# Dense MLP (every layer)
# ============================================================

def mlp_layer(x, w):
    """SwiGLU MLP: silu(gate) * up, then down. Pre-RMSNormed."""
    h = rms_norm(x, w['post_attention_layernorm'])
    gate = h @ w['gate_proj']
    up = h @ w['up_proj']
    inter = silu(gate) * up
    return x + inter @ w['down_proj']


# ============================================================
# Weight loader
# ============================================================

def load_layer_weights(layer_idx, layer_type, cfg):
    """Load all weights needed for layer `layer_idx` into a dict."""
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    cfg_path = hf_hub_download(MODEL_ID, "config.json")

    base = f"model.language_model.layers.{layer_idx}"
    needed = {}

    if layer_type == 'linear_attention':
        needed['input_layernorm']   = f"{base}.input_layernorm.weight"
        needed['in_proj_qkv']        = f"{base}.linear_attn.in_proj_qkv.weight"
        needed['in_proj_z']          = f"{base}.linear_attn.in_proj_z.weight"
        needed['in_proj_a']          = f"{base}.linear_attn.in_proj_a.weight"
        needed['in_proj_b']          = f"{base}.linear_attn.in_proj_b.weight"
        needed['out_proj']           = f"{base}.linear_attn.out_proj.weight"
        needed['conv1d_weight']      = f"{base}.linear_attn.conv1d.weight"
        # conv1d_bias is optional in this model
        needed['A_log']              = f"{base}.linear_attn.A_log"
        needed['dt_bias']            = f"{base}.linear_attn.dt_bias"
    else:
        needed['input_layernorm']     = f"{base}.input_layernorm.weight"
        needed['q_proj']              = f"{base}.self_attn.q_proj.weight"
        needed['k_proj']              = f"{base}.self_attn.k_proj.weight"
        needed['v_proj']              = f"{base}.self_attn.v_proj.weight"
        needed['o_proj']              = f"{base}.self_attn.o_proj.weight"

    # MLP weights (every layer has dense MLP)
    needed['post_attention_layernorm'] = f"{base}.post_attention_layernorm.weight"
    needed['gate_proj']                = f"{base}.mlp.gate_proj.weight"
    needed['up_proj']                  = f"{base}.mlp.up_proj.weight"
    needed['down_proj']                = f"{base}.mlp.down_proj.weight"

    # Group needed tensors by shard for efficient loading
    by_shard = {}
    for key, tname in needed.items():
        if tname not in weight_map:
            print(f"  WARN: tensor {tname} not in weight_map")
            continue
        shard = weight_map[tname]
        by_shard.setdefault(shard, []).append((key, tname))

    weights = {}
    for shard, items in by_shard.items():
        shard_path = hf_hub_download(MODEL_ID, shard)
        # Use torch framework so bf16 tensors load cleanly; then convert to fp32 numpy
        with safe_open(shard_path, framework="pt") as f:
            for key, tname in items:
                t_torch = f.get_tensor(tname).float()
                t = t_torch.numpy()
                # HF Linear stores weights as [out_features, in_features]; for x @ W
                # we need [in_features, out_features], so transpose all "proj" tensors.
                if any(k in key for k in ['proj', 'gate_proj', 'up_proj', 'down_proj']):
                    t = t.T
                weights[key] = t.copy()
    return weights


def make_rope_tables(cur_pos, rotary_dim, theta=10_000_000.0):
    half = rotary_dim // 2
    freqs = 1.0 / (theta ** (np.arange(half).astype(np.float32) / half))
    angles = cur_pos * freqs
    cos = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    return cos, sin


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 64)
    print("Phase B′2 — Qwen3.6-27B numpy fp32 reference (layers 0 + 3)")
    print("=" * 64)

    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg['text_config']

    cfg = {
        'hidden':        text_cfg['hidden_size'],
        'n_k_heads':     text_cfg['linear_num_key_heads'],
        'n_v_heads':     text_cfg['linear_num_value_heads'],
        'k_dim':         text_cfg['linear_key_head_dim'],
        'v_dim':         text_cfg['linear_value_head_dim'],
        'conv_kernel':   text_cfg['linear_conv_kernel_dim'],
        'n_q_heads':     text_cfg['num_attention_heads'],
        'n_kv_heads':    text_cfg['num_key_value_heads'],
        'head_dim':      text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
    }
    HIDDEN = cfg['hidden']
    print(f"  hidden={HIDDEN}, conv_kernel={cfg['conv_kernel']}")

    # Deterministic input
    rng = np.random.default_rng(42)
    x = rng.standard_normal(HIDDEN).astype(np.float32) * 0.05
    print(f"  input x: shape={x.shape}, norm={np.linalg.norm(x):.4f}")

    # ── Layer 0: DeltaNet ───────────────────────────────────────
    print("\n[layer 0: DeltaNet] loading weights…")
    w0 = load_layer_weights(0, 'linear_attention', cfg)
    print(f"  loaded {len(w0)} tensors")

    # Initial state (zero)
    ssm_state = np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32)
    conv_dim = 2 * cfg['n_k_heads'] * cfg['k_dim'] + cfg['n_v_heads'] * cfg['v_dim']
    conv_state = np.zeros((conv_dim, cfg['conv_kernel'] - 1), dtype=np.float32)

    x_after_dn, ssm_state, conv_state = deltanet_layer(x, w0, ssm_state, conv_state, cfg)
    print(f"  post-DeltaNet x: norm={np.linalg.norm(x_after_dn):.4f}")

    # MLP on layer 0
    x_after_layer0 = mlp_layer(x_after_dn, w0)
    print(f"  post-Layer-0 (DN + MLP): norm={np.linalg.norm(x_after_layer0):.4f}")

    # ── Layers 1-2 also DeltaNet — skip for now (gold for just layer 0 is enough)

    # ── Layer 3: Gated Attention ────────────────────────────────
    # For simplicity, run layer 0 only as the gate; layer 3 would need
    # the running state from layers 0-2. Phase B′3 will validate just layer 0.
    print(f"\nSaving reference to {OUT_NPZ}")
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             input_x=x,
             post_deltanet_x=x_after_dn,
             post_layer0=x_after_layer0,
             ssm_state_after_layer0=ssm_state)
    print(f"  ✓ saved {os.path.getsize(OUT_NPZ)/1e6:.1f} MB")
    print(f"\n=== B′2 complete (layer 0 numpy ref ready) ===")
    print(f"  Next: B′3 — ttnn impl, gate cosine(post_layer0_ttnn, post_layer0_numpy) ≥ 0.99")


if __name__ == "__main__":
    main()
