"""Inspect StableHLO IR for transformer DECODE operations.

Goes beyond the core compute blocks (softmax, layernorm, matmul) to cover
the ops needed for a full transformer decode step:
- Multi-head attention with reshape/transpose
- KV cache read + update (dynamic_update_slice)
- Causal masking (compare, select, iota)
- Embedding lookup (gather or take)
- Argmax for greedy decoding

Run on remote: python3 inspect_transformer_decode.py
"""

import os
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import jax._src.interpreters.mlir as jax_mlir


def show_ir(name, fn, *args):
    """Lower a function and print its StableHLO text IR."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    lowered = jax.jit(fn).lower(*args)
    text = lowered.as_text(dialect="stablehlo")
    print(text)

    ops = set()
    for line in text.split('\n'):
        line = line.strip()
        if 'stablehlo.' in line:
            for part in line.split():
                if part.startswith('stablehlo.'):
                    ops.add(part.split('(')[0].split('%')[0].rstrip(','))
                    break
    print(f"\n  Unique ops: {sorted(ops)}")
    return ops


def show_bytecode_ir(name, fn, *args):
    """Lower and show BYTECODE text format (what engine actually parses)."""
    from jax_plugins.tt.engine import bytecode_to_text
    print(f"\n{'=' * 70}")
    print(f"  {name} [bytecode format]")
    print(f"{'=' * 70}")
    lowered = jax.jit(fn).lower(*args)
    module = lowered.compiler_ir(dialect="stablehlo")
    bc = jax_mlir.module_to_bytecode(module)
    text = bytecode_to_text(bc)
    print(text)

    ops = set()
    for line in text.split('\n'):
        line = line.strip()
        if 'stablehlo.' in line:
            for part in line.split():
                if part.startswith('"stablehlo.') or part.startswith('stablehlo.'):
                    op = part.strip('"').split('(')[0].split('%')[0].rstrip(',')
                    ops.add(op)
                    break
    print(f"\n  Unique ops: {sorted(ops)}")
    return ops


all_ops = set()

# ============================================================
# 1. Reshape for multi-head attention (split heads)
# ============================================================

def split_heads(x):
    """[1, 8, 64] -> [1, 4, 8, 16] (4 heads, head_dim=16)"""
    return x.reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)

ops = show_ir(
    "split_heads: reshape + transpose for MHA",
    split_heads,
    jnp.ones((1, 8, 64)),
)
all_ops |= ops

# ============================================================
# 2. Multi-head attention (full, no mask)
# ============================================================

def mha(x, wq, wk, wv, wo):
    """MHA with 4 heads, dim=64, head_dim=16."""
    q = (x @ wq).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
    k = (x @ wk).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
    v = (x @ wv).reshape(1, 8, 4, 16).transpose(0, 2, 1, 3)
    scores = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(16.0), axis=-1)
    attn = (scores @ v).transpose(0, 2, 1, 3).reshape(1, 8, 64)
    return attn @ wo

ops = show_ir(
    "MHA: multi-head attention (no mask)",
    mha,
    jnp.ones((1, 8, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
)
all_ops |= ops

# ============================================================
# 3. Causal mask generation
# ============================================================

def causal_mask(x):
    """Generate 8x8 causal attention mask. x is dummy input for shape."""
    mask = jnp.tril(jnp.ones((8, 8)))
    return jnp.where(mask == 0, -1e9, 0.0)

ops = show_ir(
    "causal_mask: tril + where",
    causal_mask,
    jnp.ones(1),  # dummy
)
all_ops |= ops

# ============================================================
# 4. KV cache update (dynamic_update_slice pattern)
# ============================================================

def kv_cache_update(cache, new_kv):
    """Update KV cache at position 5 (static)."""
    return cache.at[:, :, 5:6, :].set(new_kv)

ops = show_ir(
    "kv_cache_update: cache.at[5].set(new_kv)",
    kv_cache_update,
    jnp.ones((1, 4, 32, 16)),  # [B, heads, max_seq, head_dim]
    jnp.ones((1, 4, 1, 16)),   # new KV
)
all_ops |= ops

# ============================================================
# 5. Embedding lookup
# ============================================================

def embedding_lookup(table, token_ids):
    """Standard embedding lookup."""
    return table[token_ids]

ops = show_ir(
    "embedding_lookup: table[ids]",
    embedding_lookup,
    jnp.ones((1000, 64)),  # vocab_size x dim
    jnp.array([1, 2, 3]),  # token ids
)
all_ops |= ops

# ============================================================
# 6. Argmax (greedy decode)
# ============================================================

def greedy_decode(logits):
    return jnp.argmax(logits, axis=-1)

ops = show_ir(
    "argmax: greedy decoding",
    greedy_decode,
    jnp.ones((1, 1000)),  # [batch, vocab]
)
all_ops |= ops

# ============================================================
# 7. Concatenate (for KV cache or head reassembly)
# ============================================================

def concat_kv(old_k, new_k):
    return jnp.concatenate([old_k, new_k], axis=-2)

ops = show_ir(
    "concatenate: KV cache append",
    concat_kv,
    jnp.ones((1, 4, 7, 16)),  # existing cache
    jnp.ones((1, 4, 1, 16)),  # new entry
)
all_ops |= ops

# ============================================================
# 8. Slice (extracting from cache or splitting)
# ============================================================

def slice_kv(cache):
    """Slice first 8 entries from KV cache."""
    return cache[:, :, :8, :]

ops = show_ir(
    "slice: extract KV from cache",
    slice_kv,
    jnp.ones((1, 4, 32, 16)),
)
all_ops |= ops

# ============================================================
# 9. Full decode step (tiny model)
# ============================================================

def tiny_decode_step(token_emb, wq, wk, wv, wo, w1, w2, k_cache, v_cache):
    """One decode step: attention + FFN with KV cache.

    Fixed shapes: B=1, D=64, n_heads=4, head_dim=16, pos=5.
    Uses static pos to avoid dynamic shape issues during tracing.
    """
    pos = 5

    # RMS norm
    ms = jnp.mean(token_emb ** 2, axis=-1, keepdims=True)
    x = token_emb / jnp.sqrt(ms + 1e-6)

    # Single-token QKV: [1, 64] -> [1, 1, 4, 16] -> [1, 4, 1, 16]
    q = (x @ wq).reshape(1, 1, 4, 16).transpose(0, 2, 1, 3)
    k = (x @ wk).reshape(1, 1, 4, 16).transpose(0, 2, 1, 3)
    v = (x @ wv).reshape(1, 1, 4, 16).transpose(0, 2, 1, 3)

    # Update KV cache at pos
    k_cache = k_cache.at[:, :, pos:pos+1, :].set(k)
    v_cache = v_cache.at[:, :, pos:pos+1, :].set(v)

    # Attention with cached KV (slice to pos+1)
    k_use = k_cache[:, :, :pos+1, :]
    v_use = v_cache[:, :, :pos+1, :]

    scores = jax.nn.softmax(
        q @ k_use.transpose(0, 1, 3, 2) / jnp.sqrt(16.0),
        axis=-1,
    )
    attn = (scores @ v_use).transpose(0, 2, 1, 3).reshape(1, 64)
    h = token_emb + (attn @ wo)

    # FFN: relu MLP
    ms2 = jnp.mean(h ** 2, axis=-1, keepdims=True)
    h_norm = h / jnp.sqrt(ms2 + 1e-6)
    out = h + jax.nn.relu(h_norm @ w1) @ w2

    return out, k_cache, v_cache

ops = show_ir(
    "tiny_decode_step: full decode with KV cache",
    tiny_decode_step,
    jnp.ones((1, 64)),        # token_emb
    jnp.ones((64, 64)),       # wq
    jnp.ones((64, 64)),       # wk
    jnp.ones((64, 64)),       # wv
    jnp.ones((64, 64)),       # wo
    jnp.ones((64, 128)),      # w1
    jnp.ones((128, 64)),      # w2
    jnp.ones((1, 4, 32, 16)), # k_cache
    jnp.ones((1, 4, 32, 16)), # v_cache
)
all_ops |= ops

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 70}")
print(f"  SUMMARY: All unique StableHLO ops for transformer decode")
print(f"{'=' * 70}")
for op in sorted(all_ops):
    print(f"  {op}")
print(f"\n  Total: {len(all_ops)} unique ops")

# Check which ones our engine supports
SUPPORTED = {
    'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply',
    'stablehlo.divide', 'stablehlo.maximum', 'stablehlo.minimum',
    'stablehlo.negate', 'stablehlo.abs', 'stablehlo.exponential',
    'stablehlo.log', 'stablehlo.tanh', 'stablehlo.rsqrt',
    'stablehlo.sqrt', 'stablehlo.convert', 'stablehlo.broadcast_in_dim',
    'stablehlo.reshape', 'stablehlo.transpose', 'stablehlo.dot_general',
    'stablehlo.constant', 'stablehlo.reduce',
}

missing = all_ops - SUPPORTED
supported = all_ops & SUPPORTED
print(f"\n  Already supported: {len(supported)}/{len(all_ops)}")
print(f"  Missing: {len(missing)}")
for op in sorted(missing):
    print(f"    - {op}")
