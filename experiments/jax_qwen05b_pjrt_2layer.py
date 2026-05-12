#!/usr/bin/env python3
"""Run just the 2-layer composition through the actual TT PJRT plugin.

Captures the raw bytecode being passed to the plugin and inspects what
the engine sees.
"""
import os
import sys

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

args_dev = [jax.device_put(a, tt_dev) for a in [x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask]]

# Also dump what bytecode_to_text sees from the PJRT VHLO bytes
# (we intercept by monkey-patching execute_stablehlo)
from jax_plugins.tt import engine as eng
_orig_exec = eng.execute_stablehlo
_captured = []
def _spy(bc, inputs):
    _captured.append(bc)
    return _orig_exec(bc, inputs)
eng.execute_stablehlo = _spy

try:
    r = jax.jit(two_layers)(*args_dev)
    out = jax.device_get(r)
    print(f"OK out shape={out.shape}")
except Exception as e:
    print(f"FAIL: {str(e)[:400]}")

print(f"\nCaptured {len(_captured)} bytecode blobs from execute_stablehlo")
for i, bc in enumerate(_captured):
    print(f"  blob {i}: {len(bc)} bytes, has StableHLO marker: {b'StableHLO' in bc[:30]}")
    try:
        txt = eng.bytecode_to_text(bc)
        n_calls = txt.count('func.call')
        n_funcs = txt.count('func.func')
        n_publics = txt.count('public')
        n_privates = txt.count('private')
        print(f"    text size: {len(txt)} chars, func.call={n_calls}, "
              f"func.func={n_funcs}, public={n_publics}, private={n_privates}")
        args, ops, returns, private_fns = eng.parse_stablehlo(txt)
        n_call_ops = sum(1 for o in ops if o['op'] == 'func_call')
        print(f"    parsed: main_args={len(args)}, ops={len(ops)}, returns={len(returns)}, "
              f"private_fns={len(private_fns)}, main_func_call_ops={n_call_ops}")
        if n_call_ops > 0 and n_call_ops > len(private_fns):
            print(f"    !!! MISMATCH: {n_call_ops} call ops but {len(private_fns)} private fns")
            # Print the first ~30 lines
            print("    First 40 lines of bytecode_to_text:")
            for ln in txt.splitlines()[:40]:
                print(f"      {ln[:200]}")
    except Exception as e:
        print(f"    bytecode_to_text FAILED: {str(e)[:200]}")
