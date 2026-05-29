"""Debug embedding_lookup failure."""
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

@jax.jit
def embed(table, ids):
    return table[ids]

table = np.random.randn(100, 64).astype(np.float32)
ids = np.array([0, 5, 99], dtype=np.int32)

print("StableHLO IR:")
print(jax.jit(embed).lower(table, ids).compiler_ir(dialect="stablehlo"))
print()

try:
    result = jax.device_get(embed(
        jax.device_put(table, jax.devices('tt')[0]),
        jax.device_put(ids, jax.devices('tt')[0]),
    ))
    print(f"OK: {result.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
