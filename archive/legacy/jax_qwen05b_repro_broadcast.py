#!/usr/bin/env python3
"""Repro the broadcast verifier error we see in jax_qwen05b_pjrt.py.

Targets the TT PJRT plugin. Tests increasingly-complex broadcasts to
isolate which pattern triggers the MLIR verification failure.
"""
import os
import sys

# Plugin discovery
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pjrt_plugin"))

import jax
import jax.numpy as jnp
import jax_plugins.tt as tt
try:
    tt.initialize()
except Exception as e:
    if "ALREADY_EXISTS" not in str(e):
        raise

tt_dev = jax.devices("tt")[0]


def trial(name, f, *args):
    args_dev = [jax.device_put(a, tt_dev) for a in args]
    try:
        r = jax.jit(f)(*args_dev)
        out = jax.device_get(r)
        print(f"OK   {name}: out shape={out.shape}")
    except Exception as e:
        msg = str(e).splitlines()
        for ln in msg:
            if 'verif' in ln.lower() or 'error' in ln.lower():
                print(f"FAIL {name}: {ln[:200]}")
                break
        else:
            print(f"FAIL {name}: {msg[0][:200]}")


# Plain broadcasts
trial("rank-1 broadcast: [1,1,8]*[8]",
      lambda x, g: x * g, jnp.ones((1, 1, 8)), jnp.ones((8,)))

trial("rank-2 broadcast: [1,1,8]*[1,8]",
      lambda x, g: x * g, jnp.ones((1, 1, 8)), jnp.ones((1, 8)))

trial("rank-3 broadcast: [1,1,8]*[1,1,8]",
      lambda x, g: x * g, jnp.ones((1, 1, 8)), jnp.ones((1, 1, 8)))

# Add bias rank-1
trial("bias add: [1,1,8]+[8]",
      lambda x, b: x + b, jnp.ones((1, 1, 8)), jnp.ones((8,)))

# RMS norm style
trial("rms-style: x * rsqrt(ms+eps) * g",
      lambda x, g: x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * g,
      jnp.ones((1, 1, 8)), jnp.ones((8,)))

# Bigger
trial("rms-style HIDDEN=896",
      lambda x, g: x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * g,
      jnp.ones((1, 1, 896)), jnp.ones((896,)))

# Linear with bias
trial("linear w bias: [1,1,8]@[8,16]+[16]",
      lambda x, w, b: x @ w + b,
      jnp.ones((1, 1, 8)), jnp.ones((8, 16)), jnp.ones((16,)))

# cos/sin like
trial("q * cos: [1,14,1,64]*[1,1,1,64]",
      lambda q, c: q * c,
      jnp.ones((1, 14, 1, 64)), jnp.ones((1, 1, 1, 64)))

# Mask add
trial("mask add: [1,14,1,128]+[128]",
      lambda s, m: s + m,
      jnp.ones((1, 14, 1, 128)), jnp.zeros((128,)))

trial("mask add explicit broadcast: [1,14,1,128]+[1,1,1,128]",
      lambda s, m: s + m,
      jnp.ones((1, 14, 1, 128)), jnp.zeros((1, 1, 1, 128)))

# Stack
def stack_kvs(a, b, c):
    return jnp.stack([a, b, c], axis=0)

trial("stack rank-4 -> rank-5",
      stack_kvs,
      jnp.ones((1, 2, 1, 64)), jnp.ones((1, 2, 1, 64)), jnp.ones((1, 2, 1, 64)))
