#!/usr/bin/env python3
"""Bisect the MLIR verifier error: build the decode step bit by bit.

Runs each variant through the TT PJRT plugin to find which composition
trips broadcast_dimensions verification.
"""
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

HIDDEN = 896; N_Q = 14; N_KV = 2; D = 64; MAX_SEQ = 128; EPS = 1e-6


def trial(name, f, *args):
    args_dev = [jax.device_put(a, tt_dev) for a in args]
    try:
        r = jax.jit(f)(*args_dev)
        if isinstance(r, (list, tuple)):
            shapes = [x.shape for x in r]
        else:
            shapes = r.shape
        print(f"OK   {name}: {shapes}")
    except Exception as e:
        s = str(e).splitlines()
        first = next((l for l in s if "broadcast_dim" in l or "error:" in l), s[0])
        print(f"FAIL {name}: {first[:300]}")


def attn_no_mask(x, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b, R, cos, sin, kc, vc):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    h = x * jax.lax.rsqrt(ms + EPS) * ln1_g
    q = (h @ q_w + q_b).reshape(1, N_Q, 1, D)
    k = (h @ k_w + k_b).reshape(1, N_KV, 1, D)
    v = (h @ v_w + v_b).reshape(1, N_KV, 1, D)
    qr = q * cos + (q @ R) * sin
    kr = k * cos + (k @ R) * sin
    k_b_ = jnp.repeat(kc, N_Q // N_KV, axis=1)
    v_b_ = jnp.repeat(vc, N_Q // N_KV, axis=1)
    scores = qr @ jnp.swapaxes(k_b_, -1, -2)
    return scores

x = jnp.ones((1, 1, HIDDEN))
ln1 = jnp.ones((HIDDEN,))
qw = jnp.ones((HIDDEN, N_Q * D)); qb = jnp.ones((N_Q * D,))
kw = jnp.ones((HIDDEN, N_KV * D)); kbs = jnp.ones((N_KV * D,))
vw = jnp.ones((HIDDEN, N_KV * D)); vbs = jnp.ones((N_KV * D,))
R = jnp.ones((D, D))
cos = jnp.ones((1, 1, 1, D))
sin = jnp.ones((1, 1, 1, D))
kc = jnp.ones((1, N_KV, MAX_SEQ, D))
vc = jnp.ones((1, N_KV, MAX_SEQ, D))

trial("attn no mask -> scores", attn_no_mask, x, ln1, qw, qb, kw, kbs, vw, vbs, R, cos, sin, kc, vc)


def attn_w_mask(x, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b, R, cos, sin, kc, vc, mask):
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
    return attn

mask = jnp.zeros((MAX_SEQ,))
trial("attn w mask -> attn", attn_w_mask, x, ln1, qw, qb, kw, kbs, vw, vbs, R, cos, sin, kc, vc, mask)


# Full single layer
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

ow = jnp.ones((HIDDEN, HIDDEN)); ln2 = jnp.ones((HIDDEN,))
gw = jnp.ones((HIDDEN, 4864)); uw = jnp.ones((HIDDEN, 4864))
dw = jnp.ones((4864, HIDDEN))

trial("one full layer", one_layer, x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask)


# 2 layers as a loop
def two_layers(x, *args):
    for _ in range(2):
        x, kr, v = one_layer(x, *args)
    return x

trial("two layers loop", two_layers, x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask)


# Stack of new_ks
def step_with_stack(x, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b, o_w, ln2_g, gate_w, up_w, down_w, R, cos, sin, kc, vc, mask):
    new_ks = []
    new_vs = []
    for _ in range(3):
        x, kr, v = one_layer(x, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b, o_w, ln2_g, gate_w, up_w, down_w, R, cos, sin, kc, vc, mask)
        new_ks.append(kr); new_vs.append(v)
    return x, jnp.stack(new_ks, axis=0), jnp.stack(new_vs, axis=0)

trial("3 layers + stack KVs", step_with_stack, x, ln1, qw, qb, kw, kbs, vw, vbs, ow, ln2, gw, uw, dw, R, cos, sin, kc, vc, mask)
