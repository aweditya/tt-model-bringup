"""Check what bytecode_to_text produces for reduce ops.

The engine parses VHLO portable artifacts, not as_text() output.
This script verifies whether reduce uses `applies stablehlo.XXX`
shorthand or full body regions in the bytecode→text path.

Also checks call @fn format for private functions.

Run: JAX_PLATFORMS=cpu python3 inspect_bytecode_format.py
"""

import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax._src.interpreters.mlir as jax_mlir
from jax_plugins.tt.engine import bytecode_to_text


def show_bytecode_text(name, fn, *args):
    """Lower → bytecode → text via our engine's actual path."""
    print(f"\n{'=' * 70}")
    print(f"  {name} (bytecode_to_text output)")
    print(f"{'=' * 70}")
    lowered = jax.jit(fn).lower(*args)
    module = lowered.compiler_ir(dialect="stablehlo")
    bc = jax_mlir.module_to_bytecode(module)
    text = bytecode_to_text(bc)
    print(text)
    return text


# 1. Softmax (has reduce)
show_bytecode_text(
    "softmax",
    lambda x: jax.nn.softmax(x, axis=-1),
    jnp.ones((2, 64)),
)

# 2. MLP with ReLU (has call @relu)
def mlp(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2

show_bytecode_text(
    "MLP with relu",
    mlp,
    jnp.ones((2, 64)),
    jnp.ones((64, 128)),
    jnp.ones(128),
    jnp.ones((128, 64)),
    jnp.ones(64),
)

# 3. SiLU MLP (has call @silu)
def silu_mlp(x, w_gate, w_up, w_down):
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return (gate * up) @ w_down

show_bytecode_text(
    "SiLU MLP",
    silu_mlp,
    jnp.ones((2, 64)),
    jnp.ones((64, 128)),
    jnp.ones((64, 128)),
    jnp.ones((128, 64)),
)
