"""Check bytecode_to_text format for the new ops we need to implement.

The engine parses bytecode_to_text output, NOT as_text output.
These formats can differ (e.g. transpose uses dims vs permutation).
"""

import os
import sys
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import jax._src.interpreters.mlir as jax_mlir
from jax_plugins.tt.engine import bytecode_to_text


def show(name, fn, *args):
    from jaxlib.mlir._mlir_libs._stablehlo import (
        serialize_portable_artifact, get_current_version
    )
    lowered = jax.jit(fn).lower(*args)
    module = lowered.compiler_ir(dialect="stablehlo")
    # Use portable artifact format (what PJRT actually receives)
    version = get_current_version()
    bc = serialize_portable_artifact(module, version)
    text = bytecode_to_text(bc)
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(text)


# 1. slice
show("slice", lambda x: x[:, :, :8, :], jnp.ones((1, 4, 32, 16)))

tests = [
    ("compare+select (where)", lambda x: jnp.where(x > 0, x, 0.0), [jnp.ones(8)]),
    ("iota+compare+select (tril)", lambda x: jnp.tril(x), [jnp.ones((4, 4))]),
    ("concatenate", lambda x, y: jnp.concatenate([x, y], axis=-1), [jnp.ones((2, 3)), jnp.ones((2, 4))]),
    ("and/or", lambda x, y: (x > 0) & (y > 0), [jnp.ones(4), jnp.ones(4)]),
    ("compare only", lambda x: x > 0.5, [jnp.ones(4)]),
    ("select only", lambda x: jnp.maximum(x, 0.0), [jnp.ones(4)]),
]

for name, fn, args in tests:
    try:
        show(name, fn, *args)
    except Exception as e:
        # Show as_text format instead if bytecode fails
        print(f"\n{'='*60}\n  {name} [BYTECODE FAILED: {e.__class__.__name__}]")
        print(f"  Showing as_text format instead:")
        print(f"{'='*60}")
        lowered = jax.jit(fn).lower(*args)
        print(lowered.as_text(dialect="stablehlo"))
