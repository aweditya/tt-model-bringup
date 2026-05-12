#!/usr/bin/env python3
"""Test if registering the func dialect makes func.call print with callee names."""
import os
import sys
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pjrt_plugin"))
import jax
import jax.numpy as jnp
import jax_plugins.tt as tt
try: tt.initialize()
except Exception as e:
    if "ALREADY_EXISTS" not in str(e): raise
tt_dev = jax.devices("tt")[0]


def silu_layers(x):
    for _ in range(3):
        x = jax.nn.silu(x)
    return x

x = jnp.ones((1, 1, 64))
x_d = jax.device_put(x, tt_dev)

from jax_plugins.tt import engine as eng
_orig = eng.execute_stablehlo
_captured = []
def _spy(bc, inp):
    _captured.append(bc)
    raise RuntimeError("captured")
eng.execute_stablehlo = _spy

try:
    jax.jit(silu_layers)(x_d)
except RuntimeError:
    pass

bc = _captured[0]
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo as shlo
from jaxlib.mlir._mlir_libs._stablehlo import deserialize_portable_artifact_str

# Try: with func dialect loaded
native_bc = deserialize_portable_artifact_str(bc)
print("=== Without func dialect (current default) ===")
with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    shlo.register_dialect(ctx)
    m = ir.Module.parse(native_bc, ctx)
    txt = str(m)
for ln in txt.splitlines():
    if 'func.call' in ln or 'func.func' in ln:
        print(f"  {ln.strip()[:200]}")

print("\n=== With func dialect loaded ===")
with ir.Context() as ctx:
    shlo.register_dialect(ctx)
    # Try multiple ways to load func dialect
    try:
        from jaxlib.mlir.dialects import func as func_dialect
        func_dialect.register_dialect(ctx)
        print("  registered func via mlir.dialects.func")
    except Exception as e:
        print(f"  no func dialect: {e}")
    ctx.allow_unregistered_dialects = True
    m = ir.Module.parse(native_bc, ctx)
    txt = str(m)
for ln in txt.splitlines():
    if 'func.call' in ln or 'func.func' in ln:
        print(f"  {ln.strip()[:200]}")
