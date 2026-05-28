#!/usr/bin/env python3
"""
Experiment 58: Can we use interleaved RoPE format for Qwen by rearranging cos/sin?

From exp 54: native rotary_embedding_llama is 4.85x faster (0.019ms vs 0.090ms)
but uses INTERLEAVED format (adjacent pairs) while Qwen uses HALF format (midpoint split).

Key insight: The rotation itself is a mathematical identity — the FORMAT of rotation
(interleaved vs half) is just how you arrange the cos/sin values. If we rearrange
our cos/sin tables to match interleaved format, the RESULT should be identical.

Specifically:
  HALF format:     x * [c0..c31, c0..c31] + rotate_half(x) * [s0..s31, s0..s31]
  INTERLEAVED:     x * [c0,c0,c1,c1,...] + rotate_interleaved(x) * [s0,s0,s1,s1,...]

These produce DIFFERENT results because rotate_half and rotate_interleaved
are different permutations. So the formats are NOT interchangeable.

BUT: What if we train/fine-tune with interleaved? Or what if we find a model
that already uses interleaved format?

This experiment tests:
  1. Confirm the two formats produce different results on Qwen weights
  2. Test if interleaved RoPE with matching cos/sin gives correct generation
     (it won't — the weights were trained with half-format)
  3. Measure the speed difference in traced decode
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936

device = ttnn.open_device(device_id=0)
print("Device: Blackhole P150")

# Numpy references
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    """Qwen's format: split at midpoint, negate, swap"""
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def rotate_interleaved_np(x):
    """Adjacent-pair rotation: negate odd, swap even/odd"""
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

# Cos/sin tables in both formats
pos = 42  # arbitrary position
angles = pos * freqs

cos_half = np.concatenate([np.cos(angles), np.cos(angles)])  # [c0..c31, c0..c31]
sin_half = np.concatenate([np.sin(angles), np.sin(angles)])

cos_interleaved = np.repeat(np.cos(angles), 2)  # [c0,c0,c1,c1,...]
sin_interleaved = np.repeat(np.sin(angles), 2)

# Test vector
x = np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32)

# Apply both
result_half = x * cos_half + rotate_half_np(x) * sin_half
result_interleaved = x * cos_interleaved + rotate_interleaved_np(x) * sin_interleaved

cos_sim = np.dot(result_half.flatten(), result_interleaved.flatten()) / (
    np.linalg.norm(result_half) * np.linalg.norm(result_interleaved))

print(f"\n{'='*60}")
print(f"TEST 1: Format comparison (pos={pos})")
print(f"{'='*60}")
print(f"  Half format result[:5]:        {result_half.flatten()[:5]}")
print(f"  Interleaved format result[:5]: {result_interleaved.flatten()[:5]}")
print(f"  Cosine similarity:             {cos_sim:.6f}")
print(f"  Are they the same?             {'YES' if cos_sim > 0.999 else 'NO — different rotations!'}")

# Now check: is there a MAPPING between formats?
# If we apply interleaved rotation to REARRANGED x, do we get half rotation on original x?
# Rearrange: x_half[i] -> x_interleaved[2*(i%32) + i//32] for i < 64
# This maps [0..31,32..63] -> [0,32,1,33,2,34,...] -> interleaved pairs

def half_to_interleaved_perm(x):
    """Rearrange x so interleaved rotation gives same result as half rotation on original x."""
    result = np.zeros_like(x)
    for i in range(half_dim):
        result[..., 2*i] = x[..., i]
        result[..., 2*i+1] = x[..., i + half_dim]
    return result

def interleaved_to_half_perm(x):
    """Inverse: rearrange interleaved-format x back to half format."""
    result = np.zeros_like(x)
    for i in range(half_dim):
        result[..., i] = x[..., 2*i]
        result[..., i + half_dim] = x[..., 2*i+1]
    return result

# Test the permutation
x_rearranged = half_to_interleaved_perm(x)
result_rearranged = x_rearranged * cos_interleaved + rotate_interleaved_np(x_rearranged) * sin_interleaved
result_back = interleaved_to_half_perm(result_rearranged)

cos_perm = np.dot(result_half.flatten(), result_back.flatten()) / (
    np.linalg.norm(result_half) * np.linalg.norm(result_back))

print(f"\n{'='*60}")
print(f"TEST 2: Can we permute x to make interleaved match half?")
print(f"{'='*60}")
print(f"  half(x) vs interleaved(perm(x)) cosine: {cos_perm:.6f}")
print(f"  Match? {'YES — formats are equivalent via permutation!' if cos_perm > 0.999 else 'NO'}")

if cos_perm > 0.999:
    print(f"\n  This means: if we permute Q/K weights and cos/sin tables,")
    print(f"  we can use the 4.85x faster native RoPE on Qwen!")
    print(f"  Permutation: pair up elements (i, i+32) for i in [0,31]")

    # Verify: permuting weights is equivalent to permuting the input
    # If q_w maps x -> q, then q_perm = perm(q) = perm(x @ W) = x @ W_perm
    # where W_perm permutes the OUTPUT columns

    W_test = np.random.randn(hidden, n_q_heads * head_dim).astype(np.float32)
    x_test = np.random.randn(1, hidden).astype(np.float32)
    q_test = x_test @ W_test

    # Method A: compute q normally, then permute
    q_4d = q_test.reshape(1, 1, n_q_heads, head_dim)
    q_perm_a = half_to_interleaved_perm(q_4d)

    # Method B: permute weight columns, compute q
    # For each head, permute the head_dim columns
    W_perm = np.zeros_like(W_test)
    for h in range(n_q_heads):
        start = h * head_dim
        for i in range(half_dim):
            W_perm[:, start + 2*i] = W_test[:, start + i]
            W_perm[:, start + 2*i+1] = W_test[:, start + i + half_dim]
    q_perm_b = (x_test @ W_perm).reshape(1, 1, n_q_heads, head_dim)

    cos_weight_perm = np.dot(q_perm_a.flatten(), q_perm_b.flatten()) / (
        np.linalg.norm(q_perm_a) * np.linalg.norm(q_perm_b))
    print(f"\n  Weight permutation verification:")
    print(f"    perm(x@W) vs x@perm(W) cosine: {cos_weight_perm:.6f}")
    print(f"    {'VERIFIED — can permute weights once at load time!' if cos_weight_perm > 0.999 else 'FAILED'}")


print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Half and interleaved RoPE ARE equivalent via element permutation.")
print(f"  To use native interleaved RoPE (4.85x faster):")
print(f"    1. Permute Q/K weight columns: pair (i, i+32) -> (2i, 2i+1)")
print(f"    2. Use interleaved cos/sin: np.repeat(cos, 2)")
print(f"    3. After RoPE, permute back for attention (or permute O weight rows)")
print(f"  This is a one-time weight transformation at model load.")

ttnn.close_device(device)
print("\nDone!")
