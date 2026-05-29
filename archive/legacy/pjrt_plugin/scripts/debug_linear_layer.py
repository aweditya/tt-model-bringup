"""Reproduce test_linear_layer failure with full traceback."""
import os, sys, traceback
os.environ['TT_PJRT_USE_DEVICE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from jax_plugins.tt import initialize
try:
    initialize()
except Exception as e:
    print(f"plugin already loaded: {e}")

import jax
dev = jax.devices('tt')[0]

# Same input as test_linear_layer
f = jax.jit(lambda x, w, b: x @ w + b)
x = np.random.randn(2, 3).astype(np.float32)
w = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4).astype(np.float32)

try:
    print(f"x.shape={x.shape}, w.shape={w.shape}, b.shape={b.shape}")
    result = jax.device_get(f(
        jax.device_put(x, dev),
        jax.device_put(w, dev),
        jax.device_put(b, dev),
    ))
    print(f"OK: result shape={result.shape}")
    print(f"ref: {x @ w + b}")
    print(f"got: {result}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Also dump the StableHLO IR so we can see what JAX sends
print()
print("=" * 60)
print("StableHLO IR:")
print("=" * 60)
ir = jax.jit(f).lower(x, w, b).compiler_ir(dialect="stablehlo")
print(ir)
