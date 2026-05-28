#!/usr/bin/env python3
"""
Experiment 42: Qwen2.5-0.5B with on-device RoPE — eliminate CPU round-trips.

The experiment 41 baseline does 72 device↔CPU transfers per token (Q/K/V pull
for RoPE in each of 24 layers). This experiment moves RoPE to device using
decomposed even/odd element-wise ops, targeting zero CPU round-trips in the
forward pass.

RoPE decomposition:
  out_even = x_even * cos - x_odd * sin
  out_odd  = x_even * sin + x_odd * cos

On TT-NN, we implement this by:
  1. Precomputing interleaved cos/sin tables: [cos0, sin0, cos1, sin1, ...]
  2. Uploading as device tensors (1, 1, seq_len, head_dim)
  3. Using gather/slice or reshape tricks to separate even/odd

Actually, the simpler approach: precompute full cos/sin with the right shape
for direct element-wise multiply, where even and odd positions are interleaved.
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch

from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import ttnn

# ── Args ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("prompt", nargs="?", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=15)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────
hidden = 896
intermediate = 4864
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0
n_layers = 24
vocab_size = 151936
MAX_SEQ = 256

# ── Load weights ─────────────────────────────────────────────
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")

all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
has_lm_head = "lm_head.weight" in all_weights
lm_head_w = all_weights["lm_head.weight"].T if has_lm_head else embed_w.T.copy()

layer_weights = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ── Device ───────────────────────────────────────────────────
print(f"Opening device {args.device}...")
device = ttnn.open_device(device_id=args.device)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights...")
t0 = time.perf_counter()

dev_layers = []
for i in range(n_layers):
    lw = layer_weights[i]
    dl = {
        "ln1_g": to_dev(lw["input_layernorm.weight"]),
        "q_w": to_dev(lw["self_attn.q_proj.weight"].T),
        "q_b": to_dev(lw["self_attn.q_proj.bias"]),
        "k_w": to_dev(lw["self_attn.k_proj.weight"].T),
        "k_b": to_dev(lw["self_attn.k_proj.bias"]),
        "v_w": to_dev(lw["self_attn.v_proj.weight"].T),
        "v_b": to_dev(lw["self_attn.v_proj.bias"]),
        "o_w": to_dev(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_dev(lw["post_attention_layernorm.weight"]),
        "gate_w": to_dev(lw["mlp.gate_proj.weight"].T),
        "up_w": to_dev(lw["mlp.up_proj.weight"].T),
        "down_w": to_dev(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)

final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# ── Precompute RoPE tables on device ──────────────��──────────
# For each sequence length we might see, precompute interleaved cos/sin tables.
# Shape: (1, 1, T, head_dim) where cos/sin are interleaved:
#   [cos0, cos0, cos1, cos1, ...] and [sin0, sin0, sin1, sin1, ...]
# This way we can apply RoPE by:
#   q_rotated = q * cos_table - q_rotated_neg * sin_table
# where q_rotated_neg has even/odd swapped and negated odds.

print("Precomputing RoPE tables...")
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

# For on-device RoPE, we need cos/sin tables in interleaved format
# that can be broadcast across heads.
# cos_interleaved[..., 2i] = cos(pos * freq_i)
# cos_interleaved[..., 2i+1] = cos(pos * freq_i)
# sin_interleaved[..., 2i] = -sin(pos * freq_i)  (negated for even positions)
# sin_interleaved[..., 2i+1] = sin(pos * freq_i)

# Actually, let's use the standard rotate-half approach:
# rotate_half(x) = [-x1, x0, -x3, x2, ...]  (negate odd-indexed of pairs)
# Then: x_rotated = x * cos + rotate_half(x) * sin
#
# But TT-NN doesn't have a rotate_half op. Let's decompose differently:
# Split into even/odd, apply formula, recombine.
#
# The cleanest on-device approach: precompute cos/sin in the full (1, 1, T, head_dim)
# shape with each pair duplicated, then do:
#   q_cos = q * cos_full
#   q_shifted = stack([-q_odd, q_even]) in interleaved order
#   q_rotated = q_cos + q_shifted * sin_full

# For the CPU-side approach that still avoids per-layer round-trips,
# we can do RoPE once on CPU after Q/K projection, before the main on-device
# attention + MLP chain. This still requires pulling Q/K out, but we can
# batch it differently.

# STRATEGY: For now, let's try the hybrid approach —
# Compute Q/K/V projection on device, pull ONLY Q/K for RoPE (V stays on device),
# do RoPE on CPU, push rotated Q/K back. This saves 1/3 of the round-trips
# (V no longer needs to round-trip).
#
# Then for the attention output → output projection, keep everything on device
# by doing the merge-heads reshape on device.

# Actually the real win is: keep the FULL forward pass on device, only pulling
# to CPU once at the end. Let me try the rotate-half approach via ttnn ops.

# Precompute cos/sin for each position
positions = np.arange(MAX_SEQ, dtype=np.float32)
angles = np.outer(positions, freqs)  # (MAX_SEQ, head_dim/2)

# Full cos/sin in interleaved format (each freq duplicated for even/odd pair)
cos_full = np.zeros((MAX_SEQ, head_dim), dtype=np.float32)
sin_full = np.zeros((MAX_SEQ, head_dim), dtype=np.float32)
cos_full[:, 0::2] = np.cos(angles)
cos_full[:, 1::2] = np.cos(angles)
sin_full[:, 0::2] = np.sin(angles)
sin_full[:, 1::2] = np.sin(angles)

# Negate-mask for rotate_half: [-1, 1, -1, 1, ...]
neg_mask = np.ones(head_dim, dtype=np.float32)
neg_mask[0::2] = -1.0

# Upload tables (will be sliced per sequence length)
# Shape: (1, 1, MAX_SEQ, head_dim) — broadcasts over batch and heads
cos_full_tt = {}
sin_full_tt = {}
neg_mask_tt = to_dev(neg_mask.reshape(1, 1, 1, head_dim))

print("  RoPE tables ready")

# ── On-device forward ────────────────────────────────────────
def get_rope_tables(T):
    """Get or create device RoPE tables for sequence length T."""
    if T not in cos_full_tt:
        cos_full_tt[T] = to_dev_4d(cos_full[:T].reshape(1, 1, T, head_dim))
        sin_full_tt[T] = to_dev_4d(sin_full[:T].reshape(1, 1, T, head_dim))
    return cos_full_tt[T], sin_full_tt[T]

def apply_rope_device(x_tt, T, n_heads):
    """Apply RoPE on device. x: (1, n_heads, T, head_dim) ttnn tensor.

    rotate_half(x)[..., 2i] = -x[..., 2i+1]
    rotate_half(x)[..., 2i+1] = x[..., 2i]

    We approximate this by:
    1. Roll/shift the tensor by 1 along last dim
    2. Multiply by [-1, 1, -1, 1, ...] mask

    But ttnn doesn't have a roll op... Let's try a different decomposition.

    Actually — the standard HuggingFace approach splits into two halves:
    x1 = x[..., :head_dim//2], x2 = x[..., head_dim//2:]
    rotated = cat(-x2, x1)

    This needs slice + concat + negate. Let's try it.
    """
    cos_t, sin_t = get_rope_tables(T)

    # x * cos
    x_cos = ttnn.mul(x_tt, cos_t)

    # For rotate_half: split into first and second half of head_dim
    # x: (1, n_heads, T, 64) -> need x[..., :32] and x[..., 32:]
    # Pull to CPU, rotate, push back (temporary — until we find on-device slice)
    x_np = from_dev(x_tt, (1, n_heads, T, head_dim))
    x1 = x_np[..., :head_dim//2]
    x2 = x_np[..., head_dim//2:]
    rotated = np.concatenate([-x2, x1], axis=-1)
    rotated_tt = to_dev_4d(rotated)

    # rotated * sin
    x_sin = ttnn.mul(rotated_tt, sin_t)

    return ttnn.add(x_cos, x_sin)


def forward(input_ids_np):
    """Qwen forward with on-device RoPE attempt."""
    B, T = 1, len(input_ids_np)
    x_np = embed_w[input_ids_np].reshape(B, T, hidden)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np)

        # Self-attention
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"]), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"]), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"]), dl["v_b"])

        # Reshape Q/K/V on device
        q_tt = ttnn.transpose(ttnn.reshape(q_tt, [B, T, n_q_heads, head_dim]), 1, 2)
        k_tt = ttnn.transpose(ttnn.reshape(k_tt, [B, T, n_kv_heads, head_dim]), 1, 2)
        v_tt = ttnn.transpose(ttnn.reshape(v_tt, [B, T, n_kv_heads, head_dim]), 1, 2)

        # Apply RoPE (still has CPU round-trip for rotate_half — TODO: fully on-device)
        q_tt = apply_rope_device(q_tt, T, n_q_heads)
        k_tt = apply_rope_device(k_tt, T, n_kv_heads)

        # Attention on device
        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=True)

        # Merge heads on device
        merged = ttnn.transformer.concatenate_heads(attn_out_tt)

        # Output projection + residual
        o_tt = ttnn.matmul(merged, dl["o_w"])
        x_tt2 = ttnn.add(x_tt, o_tt)

        # MLP (fully on-device)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"])
        up_tt = ttnn.matmul(h2_tt, dl["up_w"])
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"])
        out_tt = ttnn.add(x_tt2, down_tt)

        # Pull to CPU (still needed for input to next layer's to_dev)
        x_np = from_dev(out_tt, (B, T, hidden))

    # Final norm + logits
    x_tt = to_dev(x_np)
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt)
    logits = from_dev(logits_tt, (B, T, vocab_size))
    return logits[0, -1]


# ── Generate ─────────────────────────────────────────────────
tokens = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens))

print(f'\nPrompt: "{args.prompt}"')
print(f"Generating {max_gen} tokens (on-device reshape + partial on-device RoPE)...\n")

sys.stdout.write(args.prompt)
sys.stdout.flush()

times = []
for i in range(max_gen):
    t0 = time.perf_counter()
    logits = forward(np.array(tokens))
    dt = time.perf_counter() - t0
    times.append(dt)

    next_id = int(np.argmax(logits))
    tokens.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()

    if next_id == tokenizer.eos_token_id:
        break

avg_ms = np.mean(times) * 1000
print(f"\n\n--- {len(times)} tokens, avg {avg_ms:.0f}ms/tok, "
      f"{1000/avg_ms:.1f} tok/sec ---")
print(f"  First: {times[0]*1000:.0f}ms, subsequent avg: {np.mean(times[1:])*1000:.0f}ms" if len(times) > 1 else "")
print(f"  Improvements over exp 41 baseline (582ms/tok):")
print(f"    - On-device Q/K/V reshape (ttnn.reshape + ttnn.transpose)")
print(f"    - On-device attention merge (ttnn.transformer.concatenate_heads)")
print(f"    - RoPE still uses CPU for rotate_half (needs ttnn.slice)")

ttnn.close_device(device)
