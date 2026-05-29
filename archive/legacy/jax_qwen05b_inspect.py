#!/usr/bin/env python3
"""Inspect StableHLO lowering for Qwen2.5-0.5B decode-step subroutines.

Used to verify the JAX-side RoPE / attention / MLP all lower to ops our
PJRT engine supports AND avoid host-transfer ops during trace capture.

CPU-only inspection (does not touch the TT device).
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp


def show(name, fn, *args):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    lowered = jax.jit(fn).lower(*args)
    text = lowered.as_text(dialect="stablehlo")
    # Collect unique op names
    ops = set()
    for line in text.split('\n'):
        s = line.strip()
        if 'stablehlo.' in s:
            for part in s.split():
                if 'stablehlo.' in part:
                    op = part.strip('"').split('(')[0].split('%')[0].rstrip(',').rstrip(')')
                    if op.startswith('stablehlo.'):
                        ops.add(op)
                    break
    print(f"  Unique ops: {sorted(ops)}")
    return ops, text


# ── Variant A: RoPE via slice + concatenate (jnp.split semantics) ──
def rope_slice(x, cos, sin):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin

# ── Variant B: RoPE via rotation matrix (matmul-only) ──
import numpy as np
def make_rotation(head_dim):
    half = head_dim // 2
    R = np.zeros((head_dim, head_dim), dtype=np.float32)
    for i in range(half):
        R[i + half, i] = -1.0
        R[i, i + half] = 1.0
    return R

def rope_matmul(x, R, cos, sin):
    # x: [n_h, head_dim], R: [head_dim, head_dim]
    return x * cos + (x @ R) * sin


x = jnp.ones((14, 64), dtype=jnp.float32)
cos = jnp.ones((64,), dtype=jnp.float32)
sin = jnp.zeros((64,), dtype=jnp.float32)
R = jnp.asarray(make_rotation(64))

a_ops, _ = show("RoPE via slice+concat", rope_slice, x, cos, sin)
b_ops, _ = show("RoPE via rotation matrix", rope_matmul, x, R, cos, sin)


# ── Attention with precomputed mask (no compare) ──
def gqa_attn(q, k_cache, v_cache, mask):
    """q: [B=1, n_q=14, 1, d=64], k_cache: [B, n_kv=2, T, d], same for v.
    mask: [T] precomputed (-inf past pos, 0 at/before pos)."""
    B, n_q, _, d = q.shape
    _, n_kv, T, _ = k_cache.shape
    groups = n_q // n_kv  # 7
    # broadcast k,v across groups
    k = jnp.repeat(k_cache, groups, axis=1)  # [B, n_q, T, d]
    v = jnp.repeat(v_cache, groups, axis=1)
    scale = 1.0 / jnp.sqrt(jnp.float32(d))
    scores = (q @ jnp.swapaxes(k, -1, -2)) * scale  # [B, n_q, 1, T]
    scores = scores + mask[None, None, None, :]
    probs = jax.nn.softmax(scores, axis=-1)
    return probs @ v  # [B, n_q, 1, d]

q = jnp.ones((1, 14, 1, 64))
kc = jnp.ones((1, 2, 128, 64))
vc = jnp.ones((1, 2, 128, 64))
m = jnp.zeros((128,))
c_ops, _ = show("GQA attention with precomputed mask", gqa_attn, q, kc, vc, m)


# ── RMS norm (jnp pattern) ──
def rms(x, g, eps=1e-6):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * g

d_ops, _ = show("RMS norm", rms, jnp.ones((1, 1, 896)), jnp.ones((896,)))


# ── MLP (gate/up/down SiLU) ──
def mlp(x, gate, up, down):
    g = jax.nn.silu(x @ gate)
    u = x @ up
    return (g * u) @ down

e_ops, _ = show("MLP SiLU", mlp, jnp.ones((1, 1, 896)),
                jnp.ones((896, 4864)), jnp.ones((896, 4864)),
                jnp.ones((4864, 896)))


# ── Union ──
all_ops = a_ops | b_ops | c_ops | d_ops | e_ops
print("\n" + "=" * 70)
print("  ALL OPS UNION")
print("=" * 70)
for o in sorted(all_ops):
    print(f"  {o}")

# Engine ops (from engine.py)
SUPPORTED = {
    'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply',
    'stablehlo.divide', 'stablehlo.maximum', 'stablehlo.minimum',
    'stablehlo.negate', 'stablehlo.abs', 'stablehlo.exponential',
    'stablehlo.log', 'stablehlo.tanh', 'stablehlo.rsqrt',
    'stablehlo.sqrt', 'stablehlo.convert', 'stablehlo.broadcast_in_dim',
    'stablehlo.reshape', 'stablehlo.transpose', 'stablehlo.dot_general',
    'stablehlo.constant', 'stablehlo.reduce', 'stablehlo.slice',
    'stablehlo.compare', 'stablehlo.select', 'stablehlo.iota',
    'stablehlo.concatenate', 'stablehlo.scatter', 'stablehlo.gather',
}
HOST_TRANSFER_DURING_TRACE = {
    'stablehlo.slice', 'stablehlo.gather', 'stablehlo.scatter',
    'stablehlo.and', 'stablehlo.or', 'stablehlo.compare',
    # reduce_argmax is a sub-pattern of reduce; flagged via name
}
DATA_INDEP = {'stablehlo.constant', 'stablehlo.iota'}
unsupported = all_ops - SUPPORTED
host_transfer = (all_ops & HOST_TRANSFER_DURING_TRACE) - DATA_INDEP
print(f"\n  Unsupported: {sorted(unsupported)}")
print(f"  Host-transfer during trace (data-dependent breaks trace): {sorted(host_transfer)}")
