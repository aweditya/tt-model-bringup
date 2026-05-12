"""Minimal repro of full PJRT pipeline in device mode.

Run on qb1:
    cd ~/tt-xla && TT_PJRT_USE_DEVICE=1 .venv/bin/python pjrt_plugin/scripts/debug_pjrt_device.py
"""

import os
import sys
import traceback

# Ensure device mode + plugin path
os.environ['TT_PJRT_USE_DEVICE'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Trigger plugin registration
from jax_plugins.tt import initialize
try:
    initialize()
    print("[debug] plugin registered")
except Exception as e:
    print(f"[debug] plugin already registered: {e}")

import jax
print(f"[debug] jax.devices() = {jax.devices()}")
print(f"[debug] jax.devices('tt') = {jax.devices('tt')}")

dev = jax.devices('tt')[0]
print(f"[debug] tt device: {dev}")

# Minimal computation
x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
x_dev = jax.device_put(x, dev)
print(f"[debug] put x on device: {x_dev}")

f = jax.jit(lambda a: a + 1.0)
print(f"[debug] running jit...")

try:
    result = f(x_dev)
    print(f"[debug] result: {result}")
    print(f"[debug] result.shape: {result.shape}, dtype: {result.dtype}")
    h = jax.device_get(result)
    print(f"[debug] host result: {h}")
except Exception as e:
    print(f"[debug] FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
