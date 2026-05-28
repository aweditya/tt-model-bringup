#!/usr/bin/env python3
"""Try different MLIR print options to expose func.call callee names."""
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pjrt_plugin"))
import jax, jax.numpy as jnp
import jax_plugins.tt as tt
try: tt.initialize()
except Exception as e:
    if "ALREADY_EXISTS" not in str(e): raise
tt_dev = jax.devices("tt")[0]

def silu_layers(x):
    for _ in range(3):
        x = jax.nn.silu(x)
    return x

x_d = jax.device_put(jnp.ones((1, 1, 64)), tt_dev)
from jax_plugins.tt import engine as eng
_captured = []
def _spy(bc, inp):
    _captured.append(bc); raise RuntimeError()
eng.execute_stablehlo = _spy
try: jax.jit(silu_layers)(x_d)
except RuntimeError: pass

bc = _captured[0]
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo as shlo
from jaxlib.mlir._mlir_libs._stablehlo import deserialize_portable_artifact_str
native_bc = deserialize_portable_artifact_str(bc)

with ir.Context() as ctx:
    ctx.allow_unregistered_dialects = True
    shlo.register_dialect(ctx)
    m = ir.Module.parse(native_bc, ctx)

    # Try generic op printing — exposes attributes more fully
    import io
    buf = io.StringIO()
    m.operation.print(file=buf, use_local_scope=False, print_generic_op_form=True)
    txt = buf.getvalue()
    print("=== print_generic_op_form=True ===")
    for ln in txt.splitlines():
        if 'func.call' in ln or 'func.func' in ln or 'callee' in ln:
            print(f"  {ln[:280]}")
    print()

    buf2 = io.StringIO()
    m.operation.print(file=buf2)
    txt2 = buf2.getvalue()
    print("=== default print ===")
    for ln in txt2.splitlines():
        if 'func.call' in ln or 'func.func' in ln or 'callee' in ln:
            print(f"  {ln[:280]}")

    # Try walking the module structure programmatically
    print("\n=== walk operations ===")
    def walk(op, depth=0):
        name = op.name
        if name == "func.call" or name == "func.func":
            # access attributes
            attrs = {}
            try:
                for a in op.attributes:
                    attrs[a.name] = str(a.attr)
            except Exception:
                pass
            print(f"  {'  '*depth}{name} attrs={attrs}")
        for r in op.regions:
            for blk in r:
                for child_op in blk:
                    walk(child_op, depth + 1)
    walk(m.operation)
