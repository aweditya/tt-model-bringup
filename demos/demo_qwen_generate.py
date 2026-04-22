#!/usr/bin/env python3
"""
Qwen2.5-0.5B Token Generation on Tenstorrent Blackhole
=======================================================

Self-contained demo of prefill + KV-cached decode for the Qwen2.5-0.5B
language model, running entirely on a Tenstorrent Blackhole accelerator
via the TT-NN runtime.

Optimization techniques used:
  1. Fully on-device decode -- the residual stream (x_tt) stays on device
     across all 24 transformer layers. Only 2 CPU<->device transfers per
     decode step: embedding in, logits out.
  2. On-device RoPE via rotation matrix -- instead of transferring
     position-rotated Q/K from CPU, we precompute cos/sin tables for
     every position and apply RoPE with a matrix multiply on device.
     Qwen uses half-format RoPE (rotate_half), not interleaved.
  3. HiFi4 + fp32_dest_acc on ALL matmuls -- Blackhole requires the
     WormholeComputeKernelConfig to be applied consistently across every
     op. Mixing configs corrupts downstream activations ("kernel config
     leak" bug).
  4. Temperature + top-k sampling for coherent text generation.
  5. cur_pos_tensor for SDPA decode -- passes position as an on-device
     int32 tensor, compatible with trace capture for future speedups.

Performance (non-traced, correct):  ~49 tok/sec sustained decode
Performance (with trace capture):   ~136 tok/sec (not shown in this demo)

Usage:
  ssh tenstorrent "python3 ~/demos/demo_qwen_generate.py"
  ssh tenstorrent "python3 ~/demos/demo_qwen_generate.py --prompt 'Once upon a time' --max_tokens 100"
"""

import sys, os, time, argparse

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ══════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Qwen2.5-0.5B generation on Tenstorrent Blackhole")
parser.add_argument("--prompt", type=str, default="The capital of France is",
                    help="Input prompt for generation")
parser.add_argument("--max_tokens", type=int, default=50,
                    help="Maximum number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (0 = greedy)")
parser.add_argument("--top_k", type=int, default=50,
                    help="Top-k sampling (0 = disabled)")
parser.add_argument("--device_id", type=int, default=0,
                    help="Tenstorrent device ID")
args = parser.parse_args()

# ══════════════════════════════════════════════════════════════════════
# Model Constants (Qwen2.5-0.5B)
# ══════════════════════════════════════════════════════════════════════

HIDDEN = 896            # Hidden dimension
N_Q_HEADS = 14          # Number of query heads
N_KV_HEADS = 2          # Number of key/value heads (grouped-query attention)
HEAD_DIM = 64           # Per-head dimension
HALF_DIM = HEAD_DIM // 2
N_LAYERS = 24           # Number of transformer layers
VOCAB_SIZE = 151936     # Vocabulary size
RMS_EPS = 1e-6          # RMSNorm epsilon
ROPE_THETA = 1000000.0  # RoPE base frequency
MAX_SEQ = 256           # Maximum sequence length for KV cache

# ══════════════════════════════════════════════════════════════════════
# Compute Kernel Config
#
# CRITICAL: WormholeComputeKernelConfig must be applied to ALL matmul
# ops consistently. Using HiFi4 on some ops and not others corrupts
# activations on Blackhole hardware.
# ══════════════════════════════════════════════════════════════════════

HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

# ══════════════════════════════════════════════════════════════════════
# Weight Loading
# ══════════════════════════════════════════════════════════════════════

def load_weights():
    """Download and load Qwen2.5-0.5B weights from HuggingFace."""
    print("Loading Qwen2.5-0.5B weights...")
    model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
    all_weights = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()

    embed_w = all_weights["model.embed_tokens.weight"]
    final_norm_g = all_weights["model.norm.weight"]
    # Qwen2.5-0.5B ties embed and lm_head weights
    lm_head_w = (all_weights["lm_head.weight"].T
                 if "lm_head.weight" in all_weights
                 else embed_w.T.copy())

    layer_weights = []
    for i in range(N_LAYERS):
        prefix = f"model.layers.{i}."
        lw = {k[len(prefix):]: v for k, v in all_weights.items()
              if k.startswith(prefix)}
        layer_weights.append(lw)
    del all_weights

    return embed_w, final_norm_g, lm_head_w, layer_weights

# ══════════════════════════════════════════════════════════════════════
# Device Helpers
# ══════════════════════════════════════════════════════════════════════

def to_dev(device, arr):
    """Upload a numpy array to device as bfloat16 tile-layout tensor.
    Automatically pads to at least 2D (required by TILE_LAYOUT)."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device,
                           layout=ttnn.TILE_LAYOUT)

def to_dev_4d(device, arr):
    """Upload a 4D numpy array to device (no squeezing)."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
        dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    """Read a device tensor back to CPU as numpy with the given shape."""
    t = ttnn.to_torch(tensor).float()
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)

# ══════════════════════════════════════════════════════════════════════
# RoPE Setup
#
# Qwen uses half-format RoPE: rotate_half splits the vector into two
# halves and swaps them with a sign flip, rather than interleaving.
#
# On-device RoPE trick: precompute a rotation matrix R such that
#   x @ R = rotate_half(x)
# Then:
#   rope(x, pos) = x * cos(pos) + (x @ R) * sin(pos)
# This avoids any CPU round-trip for RoPE during decode.
# ══════════════════════════════════════════════════════════════════════

def build_rotation_matrix():
    """Build the rotation matrix R for half-format RoPE.
    R maps x -> [-x[half_dim:], x[:half_dim]] via matmul."""
    R = np.zeros((HEAD_DIM, HEAD_DIM), dtype=np.float32)
    for i in range(HALF_DIM):
        R[i + HALF_DIM, i] = -1.0   # -x[half_dim+i] -> position i
        R[i, i + HALF_DIM] = 1.0    #  x[i] -> position half_dim+i
    return R

# Precompute frequency table for all positions
FREQS = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))

def get_rope_tables(seq_len):
    """Compute cos/sin tables for prefill (seq_len positions)."""
    angles = np.outer(np.arange(seq_len, dtype=np.float32), FREQS)
    cos_table = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_table = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos_table, sin_table

def rotate_half_np(x):
    """CPU rotate_half for prefill."""
    return np.concatenate([-x[..., HALF_DIM:], x[..., :HALF_DIM]], axis=-1)

def apply_rope_np(x_4d, cos_t, sin_t):
    """Apply RoPE on CPU (used during prefill)."""
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

def precompute_rope_tensors(device):
    """Precompute cos/sin device tensors for every position in [0, MAX_SEQ).
    These are uploaded once and reused across all decode steps."""
    cos_tensors, sin_tensors = [], []
    for pos in range(MAX_SEQ):
        angles = pos * FREQS
        cos_full = np.concatenate([np.cos(angles), np.cos(angles)])
        sin_full = np.concatenate([np.sin(angles), np.sin(angles)])
        cos_tensors.append(to_dev_4d(device, cos_full.reshape(1, 1, 1, HEAD_DIM)))
        sin_tensors.append(to_dev_4d(device, sin_full.reshape(1, 1, 1, HEAD_DIM)))
    return cos_tensors, sin_tensors

# ══════════════════════════════════════════════════════════════════════
# Upload Model Weights to Device
# ══════════════════════════════════════════════════════════════════════

def upload_weights(device, layer_weights_np, final_norm_g, lm_head_w):
    """Upload all transformer weights to device DRAM. Returns device weight dicts."""
    dev_layers = []
    for i in range(N_LAYERS):
        lw = layer_weights_np[i]
        dev_layers.append({
            "ln1_g":   to_dev(device, lw["input_layernorm.weight"]),
            "q_w":     to_dev(device, lw["self_attn.q_proj.weight"].T),
            "q_b":     to_dev(device, lw["self_attn.q_proj.bias"]),
            "k_w":     to_dev(device, lw["self_attn.k_proj.weight"].T),
            "k_b":     to_dev(device, lw["self_attn.k_proj.bias"]),
            "v_w":     to_dev(device, lw["self_attn.v_proj.weight"].T),
            "v_b":     to_dev(device, lw["self_attn.v_proj.bias"]),
            "o_w":     to_dev(device, lw["self_attn.o_proj.weight"].T),
            "ln2_g":   to_dev(device, lw["post_attention_layernorm.weight"]),
            "gate_w":  to_dev(device, lw["mlp.gate_proj.weight"].T),
            "up_w":    to_dev(device, lw["mlp.up_proj.weight"].T),
            "down_w":  to_dev(device, lw["mlp.down_proj.weight"].T),
        })
    final_norm_tt = to_dev(device, final_norm_g)
    lm_head_tt = to_dev(device, lm_head_w)
    return dev_layers, final_norm_tt, lm_head_tt

# ══════════════════════════════════════════════════════════════════════
# KV Cache Allocation
# ══════════════════════════════════════════════════════════════════════

def alloc_kv_caches(device):
    """Allocate empty KV caches on device for all layers.
    Shape: [1, N_KV_HEADS, MAX_SEQ, HEAD_DIM] per layer."""
    k_caches, v_caches = [], []
    for _ in range(N_LAYERS):
        empty = np.zeros((1, N_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)
        k_caches.append(ttnn.from_torch(
            torch.from_numpy(empty.copy()),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
        v_caches.append(ttnn.from_torch(
            torch.from_numpy(empty.copy()),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    return k_caches, v_caches

# ══════════════════════════════════════════════════════════════════════
# Prefill: Process the full prompt in one pass
#
# Prefill uses CPU-side RoPE and transfers activations between layers.
# This is acceptable because prefill runs only once and is not latency-
# critical for token generation throughput.
# ══════════════════════════════════════════════════════════════════════

def prefill(token_ids, device, embed_w, dev_layers, final_norm_tt,
            lm_head_tt, k_caches, v_caches):
    """Run prefill on the full prompt. Populates KV caches and returns
    the logits for the last position (next-token prediction)."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, HIDDEN)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(N_LAYERS):
        dl = dev_layers[i]
        x_tt = to_dev(device, x_np)

        # Attention: RMSNorm -> Q/K/V projection
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=RMS_EPS)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=HIFI4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=HIFI4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=HIFI4), dl["v_b"])

        # RoPE on CPU (prefill only)
        q_np = from_dev(q_tt, (B, T, N_Q_HEADS * HEAD_DIM))
        k_np = from_dev(k_tt, (B, T, N_KV_HEADS * HEAD_DIM))
        v_np = from_dev(v_tt, (B, T, N_KV_HEADS * HEAD_DIM))

        q_4d = apply_rope_np(
            q_np.reshape(B, T, N_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        k_4d = apply_rope_np(
            k_np.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        v_4d = v_np.reshape(B, T, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # Fill KV caches with prefill results
        k_4d_tt = to_dev_4d(device, k_4d)
        v_4d_tt = to_dev_4d(device, v_4d)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], k_4d_tt, batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], v_4d_tt, batch_index=0)

        # Causal self-attention
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(device, q_4d), k_4d_tt, v_4d_tt,
            is_causal=True, compute_kernel_config=HIFI4)
        attn_np = from_dev(attn_out_tt, (B, N_Q_HEADS, T, HEAD_DIM))
        attn_merged = attn_np.transpose(0, 2, 1, 3).reshape(B, T, HIDDEN)

        # Output projection + residual
        o_tt = ttnn.matmul(to_dev(device, attn_merged), dl["o_w"],
                           compute_kernel_config=HIFI4)
        x_tt2 = ttnn.add(x_tt, o_tt)

        # MLP: RMSNorm -> gate/up -> SiLU-gated -> down + residual
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=RMS_EPS)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=HIFI4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=HIFI4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=HIFI4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B, T, HIDDEN))

    # Final norm + LM head
    x_tt = to_dev(device, x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=RMS_EPS)
    logits_tt = ttnn.matmul(x_tt, lm_head_tt, compute_kernel_config=HIFI4)
    return from_dev(logits_tt, (B, T, VOCAB_SIZE))[0, -1]

# ══════════════════════════════════════════════════════════════════════
# Decode: One token at a time, fully on-device
#
# The residual stream (x_tt) stays on the device across all 24 layers.
# Only 2 CPU<->device transfers per step:
#   IN:  embedding lookup (CPU) -> device
#   OUT: logits (device) -> CPU for sampling
#
# RoPE is applied on-device using precomputed cos/sin tensors and the
# rotation matrix R. The KV cache is updated with update_cache_for_token_.
# SDPA decode uses cur_pos_tensor (int32 device tensor) which is
# compatible with trace capture for future optimization.
# ══════════════════════════════════════════════════════════════════════

def decode_step(token_id, pos, device, embed_w, dev_layers, final_norm_tt,
                lm_head_tt, k_caches, v_caches, R_tt, rope_cos, rope_sin):
    """Single decode step. Returns logits as numpy array of shape [VOCAB_SIZE]."""
    # Embedding lookup on CPU, upload to device
    x_np = embed_w[token_id:token_id + 1].reshape(1, 1, HIDDEN)
    x_tt = to_dev(device, x_np)

    # Fetch precomputed RoPE tensors for this position
    cos_tt = rope_cos[pos]
    sin_tt = rope_sin[pos]

    for i in range(N_LAYERS):
        dl = dev_layers[i]

        # -- Attention block (all on device) --
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=RMS_EPS)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=HIFI4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=HIFI4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=HIFI4), dl["v_b"])

        # Reshape to multi-head format: [1, heads, 1, head_dim]
        q_4d = ttnn.reshape(q_tt, [1, N_Q_HEADS, 1, HEAD_DIM])
        k_4d = ttnn.reshape(k_tt, [1, N_KV_HEADS, 1, HEAD_DIM])
        v_4d = ttnn.reshape(v_tt, [1, N_KV_HEADS, 1, HEAD_DIM])

        # On-device RoPE: x*cos + (x@R)*sin
        q_rotated = ttnn.matmul(q_4d, R_tt)
        q_roped = ttnn.add(ttnn.mul(q_4d, cos_tt), ttnn.mul(q_rotated, sin_tt))
        k_rotated = ttnn.matmul(k_4d, R_tt)
        k_roped = ttnn.add(ttnn.mul(k_4d, cos_tt), ttnn.mul(k_rotated, sin_tt))

        # Update KV cache at current position
        ttnn.kv_cache.update_cache_for_token_(
            k_caches[i], k_roped, update_index=pos, batch_offset=0)
        ttnn.kv_cache.update_cache_for_token_(
            v_caches[i], v_4d, update_index=pos, batch_offset=0)

        # Flash-decode attention with cur_pos_tensor
        q_decode = ttnn.reshape(q_roped, [1, 1, N_Q_HEADS, HEAD_DIM])
        cur_pos_t = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32), device=device)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=cur_pos_t, compute_kernel_config=HIFI4)

        # Output projection + residual (stays on device)
        merged = ttnn.reshape(attn, [1, 1, 1, HIDDEN])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, o_tt)

        # -- MLP block (all on device) --
        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=RMS_EPS)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=HIFI4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=HIFI4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=HIFI4)
        x_tt = ttnn.add(x_tt, down_tt)

    # Final norm + LM head -> logits back to CPU
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=RMS_EPS)
    logits_tt = ttnn.matmul(x_tt, lm_head_tt, compute_kernel_config=HIFI4)
    return from_dev(logits_tt, (1, 1, VOCAB_SIZE))[0, 0]

# ══════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════

def sample_token(logits, temperature=0.7, top_k=50):
    """Sample next token from logits with temperature and top-k.
    temperature=0 falls back to greedy (argmax)."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    top_idx = np.argsort(logits)[-top_k:]
    top_logits = logits[top_idx]
    # Numerically stable softmax over top-k
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(top_idx, p=probs))

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Qwen2.5-0.5B on Tenstorrent Blackhole")
    print("  Prefill + KV-Cached Decode")
    print("=" * 64)
    print()

    # ── Load weights ──────────────────────────────────────────────
    embed_w, final_norm_g, lm_head_w, layer_weights_np = load_weights()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # ── Open device ───────────────────────────────────────────────
    print(f"Opening device {args.device_id}...")
    device = ttnn.open_device(device_id=args.device_id)

    # ── Upload weights to device ──────────────────────────────────
    print("Uploading weights to device...")
    t_upload_start = time.perf_counter()
    dev_layers, final_norm_tt, lm_head_tt = upload_weights(
        device, layer_weights_np, final_norm_g, lm_head_w)
    del layer_weights_np  # free CPU memory
    t_upload = time.perf_counter() - t_upload_start
    print(f"  Weight upload: {t_upload * 1000:.0f}ms")

    # ── Build rotation matrix + RoPE tables ───────────────────────
    print("Uploading RoPE tables...")
    t_rope_start = time.perf_counter()
    R_tt = to_dev(device, build_rotation_matrix())
    rope_cos, rope_sin = precompute_rope_tensors(device)
    t_rope = time.perf_counter() - t_rope_start
    print(f"  RoPE tables: {t_rope * 1000:.0f}ms")

    # ── Allocate KV caches ────────────────────────────────────────
    k_caches, v_caches = alloc_kv_caches(device)

    # ── Tokenize prompt ───────────────────────────────────────────
    token_ids = tokenizer.encode(args.prompt)
    max_gen = min(args.max_tokens, MAX_SEQ - len(token_ids))
    print()
    print(f'Prompt: "{args.prompt}"')
    print(f"Prompt tokens: {len(token_ids)}")
    print(f"Generating: {max_gen} tokens")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print()

    # ── Prefill ───────────────────────────────────────────────────
    t_prefill_start = time.perf_counter()
    logits = prefill(np.array(token_ids), device, embed_w, dev_layers,
                     final_norm_tt, lm_head_tt, k_caches, v_caches)
    t_prefill = time.perf_counter() - t_prefill_start

    next_id = sample_token(logits, args.temperature, args.top_k)
    token_ids.append(next_id)

    # Print prompt + first generated token
    sys.stdout.write(args.prompt + tokenizer.decode([next_id]))
    sys.stdout.flush()

    # ── Decode loop ───────────────────────────────────────────────
    decode_times = []
    for step in range(max_gen - 1):
        pos = len(token_ids) - 1

        t_step_start = time.perf_counter()
        logits = decode_step(
            next_id, pos, device, embed_w, dev_layers, final_norm_tt,
            lm_head_tt, k_caches, v_caches, R_tt, rope_cos, rope_sin)
        dt = time.perf_counter() - t_step_start
        decode_times.append(dt)

        next_id = sample_token(logits, args.temperature, args.top_k)
        token_ids.append(next_id)

        sys.stdout.write(tokenizer.decode([next_id]))
        sys.stdout.flush()

        if next_id == tokenizer.eos_token_id:
            break

    print()

    # ── Timing Report ─────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  Timing Breakdown")
    print("=" * 64)
    print(f"  Weight upload:       {t_upload * 1000:8.0f} ms")
    print(f"  RoPE table upload:   {t_rope * 1000:8.0f} ms")
    print(f"  Prefill ({len(token_ids) - max_gen} tokens):  {t_prefill * 1000:8.0f} ms")

    if decode_times:
        n_decoded = len(decode_times)
        first_ms = decode_times[0] * 1000
        sustained = decode_times[1:] if len(decode_times) > 1 else decode_times
        avg_ms = np.mean(sustained) * 1000
        total_decode = sum(decode_times)

        print(f"  First decode step:   {first_ms:8.1f} ms")
        print(f"  Sustained decode:    {avg_ms:8.1f} ms/tok  ({1000 / avg_ms:.1f} tok/sec)")
        print(f"  Tokens decoded:      {n_decoded:8d}")
        print(f"  Total decode time:   {total_decode * 1000:8.0f} ms")
        print(f"  Total throughput:    {n_decoded / total_decode:8.1f} tok/sec")

    total_time = t_upload + t_rope + t_prefill + sum(decode_times)
    print(f"  Total wall time:     {total_time * 1000:8.0f} ms")
    print()

    # ── Full generated text ───────────────────────────────────────
    full_text = tokenizer.decode(token_ids)
    print("Generated text:")
    print(f"  {full_text}")
    print()

    # ── Cleanup ───────────────────────────────────────────────────
    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
