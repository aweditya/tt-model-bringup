"""Inspect StableHLO IR for composite JAX operations.

Lowers various JAX functions to StableHLO text and prints the IR.
This tells us exactly which ops we need to implement in the engine.

Run: JAX_PLATFORMS=cpu python3 inspect_stablehlo.py
"""

import os
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

    # Also collect unique op names
    ops = set()
    for line in text.split('\n'):
        line = line.strip()
        if '=' in line and 'stablehlo.' in line:
            # Extract op name
            for part in line.split():
                if part.startswith('stablehlo.'):
                    ops.add(part.split('(')[0].split('%')[0].rstrip(','))
                    break
        elif line.startswith('stablehlo.'):
            for part in line.split():
                if part.startswith('stablehlo.'):
                    ops.add(part.split('(')[0].split('%')[0].rstrip(','))
                    break
    print(f"\n  Unique ops: {sorted(ops)}")
    return ops


# ============================================================
# 1. Softmax — the first composite op to support
# ============================================================

all_ops = set()

ops = show_ir(
    "softmax: jax.nn.softmax(x, axis=-1)",
    lambda x: jax.nn.softmax(x, axis=-1),
    jnp.ones((2, 64)),
)
all_ops |= ops

# ============================================================
# 2. Layer norm (manual, since JAX doesn't have a builtin)
# ============================================================

def layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(var + 1e-5) + b

ops = show_ir(
    "layer_norm: manual LN",
    layer_norm,
    jnp.ones((2, 64)),
    jnp.ones(64),
    jnp.zeros(64),
)
all_ops |= ops

# ============================================================
# 3. RMS norm (used by Llama/Qwen)
# ============================================================

def rms_norm(x, g):
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return g * x / jnp.sqrt(ms + 1e-6)

ops = show_ir(
    "rms_norm: RMSNorm(x, g)",
    rms_norm,
    jnp.ones((2, 64)),
    jnp.ones(64),
)
all_ops |= ops

# ============================================================
# 4. MLP: linear + relu + linear
# ============================================================

def mlp(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2

ops = show_ir(
    "MLP: linear+relu+linear",
    mlp,
    jnp.ones((2, 64)),
    jnp.ones((64, 128)),
    jnp.ones(128),
    jnp.ones((128, 64)),
    jnp.ones(64),
)
all_ops |= ops

# ============================================================
# 5. SiLU gated MLP (used by Llama/Qwen)
# ============================================================

def silu_mlp(x, w_gate, w_up, w_down):
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down

ops = show_ir(
    "SiLU MLP: gate*up pattern",
    silu_mlp,
    jnp.ones((2, 64)),
    jnp.ones((64, 128)),
    jnp.ones((64, 128)),
    jnp.ones((128, 64)),
)
all_ops |= ops

# ============================================================
# 6. Simple attention (single head, no mask)
# ============================================================

def attention(x, wq, wk, wv, wo):
    q = x @ wq
    k = x @ wk
    v = x @ wv
    d = jnp.float32(q.shape[-1])
    scores = jax.nn.softmax(q @ k.T / jnp.sqrt(d), axis=-1)
    return (scores @ v) @ wo

ops = show_ir(
    "attention: single-head self-attention",
    attention,
    jnp.ones((8, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
    jnp.ones((64, 64)),
)
all_ops |= ops

# ============================================================
# Summary
# ============================================================

print(f"\n{'=' * 70}")
print(f"  SUMMARY: All unique StableHLO ops across all tests")
print(f"{'=' * 70}")
for op in sorted(all_ops):
    print(f"  {op}")
print(f"\n  Total: {len(all_ops)} unique ops")

# Check which ones our engine already supports
SUPPORTED = {
    'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply',
    'stablehlo.divide', 'stablehlo.maximum', 'stablehlo.minimum',
    'stablehlo.negate', 'stablehlo.abs', 'stablehlo.exponential',
    'stablehlo.log', 'stablehlo.tanh', 'stablehlo.rsqrt',
    'stablehlo.sqrt', 'stablehlo.convert', 'stablehlo.broadcast_in_dim',
    'stablehlo.reshape', 'stablehlo.transpose', 'stablehlo.dot_general',
    'stablehlo.constant',
}

missing = all_ops - SUPPORTED
print(f"\n  Already supported: {len(all_ops & SUPPORTED)}")
print(f"  Missing: {len(missing)}")
for op in sorted(missing):
    print(f"    - {op}")
