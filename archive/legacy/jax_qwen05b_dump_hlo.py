#!/usr/bin/env python3
"""Dump StableHLO IR for the 2-layer pattern that breaks the PJRT plugin.

Compares against 1-layer (which works) to spot the structural difference.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp
import jax._src.interpreters.mlir as jax_mlir

HIDDEN = 896; N_Q = 14; N_KV = 2; D = 64; MAX_SEQ = 128; EPS = 1e-6

def one_layer(x, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b, o_w, ln2_g, gate_w, up_w, down_w, R, cos, sin, kc, vc, mask):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    h = x * jax.lax.rsqrt(ms + EPS) * ln1_g
    q = (h @ q_w + q_b).reshape(1, N_Q, 1, D)
    k = (h @ k_w + k_b).reshape(1, N_KV, 1, D)
    v = (h @ v_w + v_b).reshape(1, N_KV, 1, D)
    qr = q * cos + (q @ R) * sin
    kr = k * cos + (k @ R) * sin
    k_b_ = jnp.repeat(kc, N_Q // N_KV, axis=1)
    v_b_ = jnp.repeat(vc, N_Q // N_KV, axis=1)
    scores = (qr @ jnp.swapaxes(k_b_, -1, -2)) * (1.0 / jnp.sqrt(jnp.float32(D)))
    scores = scores + mask[None, None, None, :]
    probs = jax.nn.softmax(scores, axis=-1)
    attn = probs @ v_b_
    attn = attn.transpose(0, 2, 1, 3).reshape(1, 1, HIDDEN)
    o = attn @ o_w
    x = x + o
    ms2 = jnp.mean(x * x, axis=-1, keepdims=True)
    h2 = x * jax.lax.rsqrt(ms2 + EPS) * ln2_g
    g_ = jax.nn.silu(h2 @ gate_w)
    u_ = h2 @ up_w
    d_ = (g_ * u_) @ down_w
    return x + d_, kr, v


def two_layers(x, *args):
    for _ in range(2):
        x, kr, v = one_layer(x, *args)
    return x

x = jnp.ones((1, 1, HIDDEN))
ln1 = jnp.ones((HIDDEN,))
qw = jnp.ones((HIDDEN, N_Q * D)); qb = jnp.ones((N_Q * D,))
kw = jnp.ones((HIDDEN, N_KV * D)); kbs = jnp.ones((N_KV * D,))
vw = jnp.ones((HIDDEN, N_KV * D)); vbs = jnp.ones((N_KV * D,))
ow = jnp.ones((HIDDEN, HIDDEN)); ln2 = jnp.ones((HIDDEN,))
gw = jnp.ones((HIDDEN, 4864)); uw = jnp.ones((HIDDEN, 4864))
dw = jnp.ones((4864, HIDDEN))
R = jnp.ones((D, D))
cos = jnp.ones((1, 1, 1, D))
sin = jnp.ones((1, 1, 1, D))
kc = jnp.ones((1, N_KV, MAX_SEQ, D))
vc = jnp.ones((1, N_KV, MAX_SEQ, D))
mask = jnp.zeros((MAX_SEQ,))

# Dump both
print("=" * 70 + "\n  1-layer\n" + "=" * 70)
lowered = jax.jit(one_layer).lower(x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask)
text = lowered.as_text(dialect="stablehlo")
# Count func.func and func.call
print(f"  text size: {len(text)} chars")
print(f"  num func.func: {text.count('func.func')}")
print(f"  num func.call: {text.count('func.call')}")
print(f"  num stablehlo.: {text.count('stablehlo.')}")

print("\n" + "=" * 70 + "\n  2-layer (Python loop)\n" + "=" * 70)
lowered2 = jax.jit(two_layers).lower(x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask)
text2 = lowered2.as_text(dialect="stablehlo")
print(f"  text size: {len(text2)} chars")
print(f"  num func.func: {text2.count('func.func')}")
print(f"  num func.call: {text2.count('func.call')}")
print(f"  num stablehlo.: {text2.count('stablehlo.')}")

# Show function signatures
import re
for name, t in (("1-layer", text), ("2-layer", text2)):
    print(f"\n  {name} function signatures:")
    for m in re.finditer(r'(public |private )?func\.func\s+@(\w+)\s*\([^)]*\)', t):
        print(f"    {m.group(0)[:120]}")

# Dump 2-layer bytecode and check the bytecode_to_text path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pjrt_plugin"))
from jax_plugins.tt.engine import bytecode_to_text, parse_stablehlo

print("\n" + "=" * 70 + "\n  2-layer bytecode round-trip\n" + "=" * 70)
module = lowered2.compiler_ir(dialect="stablehlo")
bc = jax_mlir.module_to_bytecode(module)
print(f"  bytecode size: {len(bc)} bytes")
try:
    txt = bytecode_to_text(bc)
    print(f"  bytecode_to_text: {len(txt)} chars (OK)")
    print(f"  func.func count after round-trip: {txt.count('func.func')}")
    print(f"  func.call count after round-trip: {txt.count('func.call')}")
    args, ops, returns, private_fns = parse_stablehlo(txt)
    print(f"  parse_stablehlo: main args={len(args)}, ops={len(ops)}, returns={len(returns)}, private_fns={len(private_fns)}")
    # Count func.call ops
    n_calls = sum(1 for o in ops if o['op'] == 'func_call')
    print(f"  parsed main func.call ops: {n_calls}")
    if n_calls > 0:
        print(f"  -> need >= {n_calls} private functions, have {len(private_fns)}")
except Exception as e:
    print(f"  bytecode_to_text FAILED: {e}")
