#!/usr/bin/env python3
"""
Experiment 91w — Phase B.1 numpy fp32 prefill reference for Qwen3.6-27B.

Extends 91b (decode-mode, single token) to **prefill** (seq_len > 1).

This is the GOLD REFERENCE that Phase B.2 ttnn prefill must match (cos ≥
0.999) and that Phase B.3 chunked-parallel DeltaNet must match (cos ≥ 0.99
within bf16 noise of the same math).

Three layers exercised:
  - Layer 0: DeltaNet — sequential loop over seq_len positions (per
    research/deltanet_parallel_prefill_research.md, this matches both our
    MVP and the gold target for the future Neumann chunked-parallel form)
  - Layer 3: Gated Attention — parallel SDPA across seq_len with causal mask
  - MLP (every layer): trivially batched on leading dim

Output: .npz with per-position hidden states post-layer-0, post-layer-3,
plus the final SSM and KV cache state (so downstream decode can resume
correctly).

Run on qb1 (math-only; no device required):
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91w_qwen36_27b_prefill_numpy_ref.py

Or locally (downloads weights via HF):
    python experiments/91w_qwen36_27b_prefill_numpy_ref.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.expanduser("~"))

from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torch

# Reuse the math primitives + weight loader from 91b
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_91b_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "91b_qwen36_27b_numpy_ref.py")
_spec = importlib.util.spec_from_file_location("_91b", _91b_path)
_91b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91b)

rms_norm = _91b.rms_norm
silu = _91b.silu
sigmoid = _91b.sigmoid
_l2_normalize = _91b._l2_normalize
load_layer_weights = _91b.load_layer_weights
make_rope_tables = _91b.make_rope_tables
MODEL_ID = _91b.MODEL_ID
EPS = _91b.EPS

OUT_NPZ = os.path.expanduser(
    "~/tt-xla/.cache/qwen36_27b_prefill_numpy_ref_seq128.npz")
SEQ_LEN = 128
SEED = 42


# ============================================================
# DeltaNet — sequential prefill loop
# ============================================================
#
# DeltaNet's recurrence makes each position depend on the previous position's
# SSM state and conv state. The simplest correct implementation is to loop
# over seq_len positions, calling the single-token decode function from 91b.
#
# This is what our Phase B.2 ttnn MVP will also do (sequential loop). The
# Phase B.3 upgrade replaces this loop with the chunked-parallel Neumann form
# but must match THIS output bit-for-bit (within bf16 noise).
#
# Cost: O(seq_len) sequential ops per layer. At seq_len=128, this is the
# correct reference — fast enough for validation, slow in production.
# ============================================================

def deltanet_prefill_sequential(x_seq, w, init_ssm_state, init_conv_state, cfg):
    """
    x_seq: [seq_len, hidden]
    w: layer weight dict (from 91b.load_layer_weights)
    init_ssm_state: [n_v_heads, k_dim, v_dim] — input recurrence state
    init_conv_state: [conv_dim, kernel - 1] — input conv state

    Returns:
      x_out_seq: [seq_len, hidden] — per-position output (post-residual)
      ssm_state: updated, same shape as init
      conv_state: updated, same shape as init
    """
    seq_len = x_seq.shape[0]
    hidden = x_seq.shape[1]
    out = np.empty((seq_len, hidden), dtype=np.float32)

    ssm_state = init_ssm_state.copy()
    conv_state = init_conv_state.copy()

    for t in range(seq_len):
        x_t = x_seq[t]
        out_t, ssm_state, conv_state = _91b.deltanet_layer(
            x_t, w, ssm_state, conv_state, cfg)
        out[t] = out_t

    return out, ssm_state, conv_state


# ============================================================
# Gated Attention — parallel prefill
# ============================================================
#
# Unlike DeltaNet, gated attention has no recurrence on the residual stream:
# each position's Q is independent. We can process all seq_len positions in
# one pass, build the full K/V tensors, then do causal SDPA.
#
# This matches the prefill SDPA path (ttnn.transformer.scaled_dot_product_attention
# with is_causal=True) we plan to use in Phase B.2.
# ============================================================

def gated_attention_prefill(x_seq, w, init_kv_cache, start_pos, cfg):
    """
    x_seq: [seq_len, hidden]
    w: layer weight dict
    init_kv_cache: dict with 'k', 'v' shape [n_kv, max_seq, head_dim]
                   (assumed pre-allocated to max sequence length)
    start_pos: int — KV cache offset where this prefill writes (0 for first prefill)

    Returns:
      x_out_seq: [seq_len, hidden]
      kv_cache: updated in-place; positions [start_pos : start_pos+seq_len) populated
    """
    seq_len = x_seq.shape[0]
    hidden = x_seq.shape[1]
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HEAD_DIM = cfg['head_dim']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

    # Pre-norm: element-wise across positions
    h_seq = np.stack([rms_norm(x_seq[t], w['input_layernorm'])
                      for t in range(seq_len)], axis=0)  # [seq_len, hidden]

    # Q proj outputs Q + gate concatenated; reshape per HF convention
    qg_seq = h_seq @ w['q_proj']  # [seq_len, N_Q * head_dim * 2]
    qg_seq = qg_seq.reshape(seq_len, N_Q, HEAD_DIM * 2)
    q_seq = qg_seq[..., :HEAD_DIM]            # [seq_len, N_Q, head_dim]
    gate_seq = qg_seq[..., HEAD_DIM:]          # [seq_len, N_Q, head_dim]

    k_seq = (h_seq @ w['k_proj']).reshape(seq_len, N_KV, HEAD_DIM)
    v_seq = (h_seq @ w['v_proj']).reshape(seq_len, N_KV, HEAD_DIM)

    # Partial RoPE — apply per-position cos/sin. cos_seq/sin_seq are
    # [seq_len, ROTARY_DIM]; broadcast across head dim.
    def apply_partial_rope_batched(t_seq, cos_seq, sin_seq, rot_dim):
        # t_seq: [seq_len, n_heads, head_dim]
        rot = t_seq[..., :rot_dim]                        # [seq_len, n_heads, rot_dim]
        passthru = t_seq[..., rot_dim:]
        half = rot_dim // 2
        x1, x2 = rot[..., :half], rot[..., half:]
        # cos_seq/sin_seq are [seq_len, rot_dim]; add head dim for broadcast
        cos_b = cos_seq[:, None, :]                       # [seq_len, 1, rot_dim]
        sin_b = sin_seq[:, None, :]
        rotated = rot * cos_b + np.concatenate([-x2, x1], axis=-1) * sin_b
        return np.concatenate([rotated, passthru], axis=-1)

    cos_seq = np.empty((seq_len, ROTARY_DIM), dtype=np.float32)
    sin_seq = np.empty((seq_len, ROTARY_DIM), dtype=np.float32)
    for t in range(seq_len):
        cos_seq[t], sin_seq[t] = make_rope_tables(start_pos + t, ROTARY_DIM)

    q_seq = apply_partial_rope_batched(q_seq, cos_seq, sin_seq, ROTARY_DIM)
    k_seq = apply_partial_rope_batched(k_seq, cos_seq, sin_seq, ROTARY_DIM)

    # Write K/V to cache at positions [start_pos, start_pos + seq_len)
    init_kv_cache['k'][:, start_pos:start_pos + seq_len, :] = k_seq.transpose(1, 0, 2)
    init_kv_cache['v'][:, start_pos:start_pos + seq_len, :] = v_seq.transpose(1, 0, 2)

    # Causal SDPA across all seq_len positions
    # GQA: repeat K, V from N_KV → N_Q
    n_rep = N_Q // N_KV
    k_full = np.repeat(k_seq, n_rep, axis=1)  # [seq_len, N_Q, head_dim]
    v_full = np.repeat(v_seq, n_rep, axis=1)

    scale = 1.0 / np.sqrt(HEAD_DIM)
    # scores[t, h, s] = q_seq[t, h, :] · k_full[s, h, :]
    scores = np.einsum('thd,shd->ths', q_seq, k_full) * scale  # [seq_len, N_Q, seq_len]

    # Causal mask: position t can only attend to positions [0, t]
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(mask[:, None, :], -np.inf, scores)

    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)

    # attn[t, h, d] = sum_s weights[t, h, s] * v_full[s, h, d]
    attn = np.einsum('ths,shd->thd', weights, v_full)  # [seq_len, N_Q, head_dim]

    # Output gate
    attn = attn * sigmoid(gate_seq)

    # Output projection
    attn_flat = attn.reshape(seq_len, N_Q * HEAD_DIM)
    out = attn_flat @ w['o_proj']  # [seq_len, hidden]

    return x_seq + out, init_kv_cache


# ============================================================
# MLP prefill — trivially batched
# ============================================================

def mlp_prefill(x_seq, w):
    """SwiGLU MLP on [seq_len, hidden]. Per-position rms_norm + matmuls."""
    seq_len = x_seq.shape[0]
    # rms_norm is element-wise across feature dim; loop or vectorize
    h_seq = np.stack([rms_norm(x_seq[t], w['post_attention_layernorm'])
                      for t in range(seq_len)], axis=0)
    gate = h_seq @ w['gate_proj']
    up = h_seq @ w['up_proj']
    inter = silu(gate) * up
    return x_seq + inter @ w['down_proj']


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 64)
    print(f"Phase B.1 — Qwen3.6-27B numpy prefill reference (seq_len={SEQ_LEN})")
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
    print(f"  hidden={HIDDEN}, conv_kernel={cfg['conv_kernel']}, "
          f"seq_len={SEQ_LEN}")

    # Deterministic input
    rng = np.random.default_rng(SEED)
    x_seq = (rng.standard_normal((SEQ_LEN, HIDDEN)).astype(np.float32) * 0.05)
    print(f"  input x_seq: shape={x_seq.shape}, "
          f"per-position norm mean={np.linalg.norm(x_seq, axis=-1).mean():.4f}")

    # ── Layer 0: DeltaNet prefill (sequential gold ref) ─────────
    print(f"\n[layer 0: DeltaNet prefill, seq_len={SEQ_LEN}] loading weights...")
    w0 = load_layer_weights(0, 'linear_attention', cfg)
    print(f"  loaded {len(w0)} tensors")

    n_v = cfg['n_v_heads']; k_d = cfg['k_dim']; v_d = cfg['v_dim']
    ssm_state = np.zeros((n_v, k_d, v_d), dtype=np.float32)
    conv_dim = 2 * cfg['n_k_heads'] * cfg['k_dim'] + cfg['n_v_heads'] * cfg['v_dim']
    conv_state = np.zeros((conv_dim, cfg['conv_kernel'] - 1), dtype=np.float32)

    print(f"  running sequential prefill loop ({SEQ_LEN} positions)...")
    x_after_dn, ssm_state, conv_state = deltanet_prefill_sequential(
        x_seq, w0, ssm_state, conv_state, cfg)
    print(f"  post-DeltaNet x_seq: shape={x_after_dn.shape}, "
          f"per-position norm mean={np.linalg.norm(x_after_dn, axis=-1).mean():.4f}")
    print(f"  SSM state Frobenius norm: {np.linalg.norm(ssm_state):.4f}")
    print(f"  Conv state Frobenius norm: {np.linalg.norm(conv_state):.4f}")

    # ── MLP on layer 0 output ────────────────────────────────────
    x_after_layer0 = mlp_prefill(x_after_dn, w0)
    print(f"  post-Layer-0 (DN + MLP): per-position norm mean="
          f"{np.linalg.norm(x_after_layer0, axis=-1).mean():.4f}")

    # ── Layer 3: Gated Attention prefill (parallel) ─────────────
    # We feed layer 0's MLP output directly to layer 3 (skipping layers 1, 2
    # which are also DeltaNet — for validation purposes we want both block
    # types exercised, not the full chain).
    print(f"\n[layer 3: Gated Attention prefill, seq_len={SEQ_LEN}] loading weights...")
    w3 = load_layer_weights(3, 'full_attention', cfg)
    print(f"  loaded {len(w3)} tensors")

    MAX_POS = max(SEQ_LEN, 512)  # match our server's MAX_POS
    kv_cache = {
        'k': np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
        'v': np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
    }

    print(f"  running parallel attention prefill (causal SDPA)...")
    x_after_attn, kv_cache = gated_attention_prefill(
        x_after_layer0, w3, kv_cache, start_pos=0, cfg=cfg)
    print(f"  post-Attention x_seq: per-position norm mean="
          f"{np.linalg.norm(x_after_attn, axis=-1).mean():.4f}")

    x_after_layer3 = mlp_prefill(x_after_attn, w3)
    print(f"  post-Layer-3 (Attn + MLP): per-position norm mean="
          f"{np.linalg.norm(x_after_layer3, axis=-1).mean():.4f}")

    # ── Save reference ────────────────────────────────────────────
    print(f"\nSaving reference to {OUT_NPZ}")
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ,
             input_x_seq=x_seq,
             post_deltanet_seq=x_after_dn,
             post_layer0_seq=x_after_layer0,
             post_attention_seq=x_after_attn,
             post_layer3_seq=x_after_layer3,
             ssm_state_after_layer0=ssm_state,
             conv_state_after_layer0=conv_state,
             kv_cache_k_after_layer3=kv_cache['k'],
             kv_cache_v_after_layer3=kv_cache['v'])
    print(f"  ✓ saved {os.path.getsize(OUT_NPZ)/1e6:.1f} MB")

    print(f"\n=== B.1 numpy prefill ref complete ===")
    print(f"  Per-position cosine gate for B.2 ttnn impl: cosine ≥ 0.999")
    print(f"  Next: B.2 — single-chip qb1 ttnn prefill, compare per-position.")


if __name__ == "__main__":
    main()
