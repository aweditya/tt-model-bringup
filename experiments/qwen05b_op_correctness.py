#!/usr/bin/env python3
"""Investigation B — independent bf16 op correctness via the PJRT engine.

Tests THE specific ops that Qwen2.5-0.5B uses, at THE shapes it uses, and
compares the PJRT-engine result (bf16 on Blackhole) to a pure-numpy fp32
reference. Reports cosine + max-abs-error.

Run modes:
  python3 experiments/qwen05b_op_correctness.py            # CPU (numpy ref vs CPU PJRT)
  TT_PJRT_USE_DEVICE=1 python3 experiments/qwen05b_op_correctness.py

The CPU mode is a sanity check that the engine itself isn't broken
arithmetically. The TT mode is the real test — that's where bf16 lives.

We deliberately go through jax.jit + the PJRT plugin (not through ttnn
directly) so the answer reflects what the full Qwen2.5-0.5B run sees.
"""
import os
import sys
import argparse

import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))
PJRT_DIR = os.path.join(PROJ, "pjrt_plugin")
sys.path.insert(0, PJRT_DIR)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cpu", action="store_true",
                    help="Also force CPU mode regardless of TT_PJRT_USE_DEVICE")
args = parser.parse_args()

DEVICE_MODE = os.environ.get("TT_PJRT_USE_DEVICE", "0") == "1" and not args.cpu


import jax
import jax.numpy as jnp


if DEVICE_MODE:
    import jax_plugins.tt as tt_plugin
    try:
        tt_plugin.initialize()
    except Exception as e:
        if "ALREADY_EXISTS" not in str(e):
            raise
    devs = jax.devices("tt")
    target = devs[0] if devs else jax.devices("cpu")[0]
else:
    target = jax.devices("cpu")[0]

print(f"target device: {target} ({'TT' if DEVICE_MODE else 'JAX CPU'})")


def put(a):
    arr = jnp.asarray(a, dtype=jnp.float32)
    return jax.device_put(arr, target)


def _cosine(a, b):
    a = np.asarray(a).astype(np.float64).flatten()
    b = np.asarray(b).astype(np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a).astype(np.float64) -
                                np.asarray(b).astype(np.float64))))


def _report(name, ref, got):
    cos = _cosine(ref, got)
    mae = _max_abs(ref, got)
    print(f"  {name:<48} cos={cos:.6f}  max_abs={mae:.4f}")
    return cos, mae


def _run(name, fn, args_np):
    """Compile, run, return numpy result."""
    fn_jit = jax.jit(fn)
    args_d = [put(a) for a in args_np]
    out_d = fn_jit(*args_d)
    if isinstance(out_d, (list, tuple)):
        return [np.asarray(jax.device_get(x)) for x in out_d]
    return np.asarray(jax.device_get(out_d))


rng = np.random.default_rng(args.seed)
results = []

# ---------------------------------------------------------------
# Op 1: matmul (Q-projection-sized) — [1, 896] @ [896, 1024] → [1, 1024]
# ---------------------------------------------------------------
print("\n[1] matmul (Q-proj shape): [1, 896] @ [896, 1024]")
A = rng.standard_normal((1, 896), dtype=np.float32) * 0.02
B = rng.standard_normal((896, 1024), dtype=np.float32) * 0.02
ref = A @ B
got = _run("matmul", lambda a, b: a @ b, [A, B])
results.append(("matmul [1,896]@[896,1024]", *_report("matmul", ref, got)))

# ---------------------------------------------------------------
# Op 2: softmax over attention scores — [1, 14, 100]
# ---------------------------------------------------------------
print("\n[2] softmax (attn scores): [1, 14, 100]")
X = rng.standard_normal((1, 14, 100), dtype=np.float32) * 2.0
# numpy reference
m = np.max(X, axis=-1, keepdims=True)
e = np.exp(X - m)
ref = e / np.sum(e, axis=-1, keepdims=True)
got = _run("softmax", lambda x: jax.nn.softmax(x, axis=-1), [X])
results.append(("softmax [1,14,100]", *_report("softmax", ref, got)))

# ---------------------------------------------------------------
# Op 3: RMS norm — [1, 1, 896]
# ---------------------------------------------------------------
print("\n[3] rms_norm: [1, 1, 896]")
X = rng.standard_normal((1, 1, 896), dtype=np.float32)
G = rng.standard_normal((896,), dtype=np.float32) * 0.5 + 1.0
eps = 1e-6
ms = np.mean(X * X, axis=-1, keepdims=True)
ref = X * (1.0 / np.sqrt(ms + eps)) * G

def _rms_norm(x, g):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * g

got = _run("rms_norm", _rms_norm, [X, G])
results.append(("rms_norm [1,1,896]", *_report("rms_norm", ref, got)))

# ---------------------------------------------------------------
# Op 4: SwiGLU step — silu(gate) * up over [1, 1, 4864]
# ---------------------------------------------------------------
print("\n[4] swiglu (silu(gate) * up): [1, 1, 4864]")
GATE = rng.standard_normal((1, 1, 4864), dtype=np.float32)
UP = rng.standard_normal((1, 1, 4864), dtype=np.float32)
ref = (GATE / (1.0 + np.exp(-GATE))) * UP

def _swiglu(gate, up):
    return jax.nn.silu(gate) * up

got = _run("swiglu", _swiglu, [GATE, UP])
results.append(("swiglu [1,1,4864]", *_report("swiglu", ref, got)))

# ---------------------------------------------------------------
# Op 5: GQA-style batched matmul Q·Kᵀ — [1, 14, 1, 64] @ [1, 14, 64, 128]
# This is what attention does after K-cache broadcast.
# ---------------------------------------------------------------
print("\n[5] attn Q·Kᵀ: [1, 14, 1, 64] @ [1, 14, 64, 128]")
Q = rng.standard_normal((1, 14, 1, 64), dtype=np.float32) * 0.1
K = rng.standard_normal((1, 14, 64, 128), dtype=np.float32) * 0.1
ref = Q @ K  # → [1, 14, 1, 128]

def _attn_qk(q, k):
    return q @ k

got = _run("attn_qk", _attn_qk, [Q, K])
results.append(("attn_qk [1,14,1,64]@[1,14,64,128]", *_report("attn_qk", ref, got)))

# ---------------------------------------------------------------
# Op 6: full SDPA-style chain (Q·Kᵀ → scale → mask → softmax → ·V)
# ---------------------------------------------------------------
print("\n[6] full sdpa chain (no head split): [1, 14, 1, 64] vs [1, 14, 128, 64]")
Q = rng.standard_normal((1, 14, 1, 64), dtype=np.float32) * 0.1
K = rng.standard_normal((1, 14, 128, 64), dtype=np.float32) * 0.1
V = rng.standard_normal((1, 14, 128, 64), dtype=np.float32) * 0.1
mask = np.zeros((128,), dtype=np.float32)
mask[10:] = -1e9
scale = 1.0 / np.sqrt(64.0)
scores = (Q @ K.transpose(0, 1, 3, 2)) * scale  # [1,14,1,128]
scores = scores + mask[None, None, None, :]
m = np.max(scores, axis=-1, keepdims=True)
e = np.exp(scores - m)
probs = e / np.sum(e, axis=-1, keepdims=True)
ref = probs @ V  # [1,14,1,64]

def _sdpa(q, k, v, mask):
    scale = 1.0 / jnp.sqrt(jnp.float32(64))
    s = (q @ jnp.swapaxes(k, -1, -2)) * scale
    s = s + mask[None, None, None, :]
    p = jax.nn.softmax(s, axis=-1)
    return p @ v

got = _run("sdpa", _sdpa, [Q, K, V, mask])
results.append(("sdpa chain [1,14,1,64] over 128", *_report("sdpa", ref, got)))

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\nSUMMARY")
print(f"{'op':<48} {'cos':>10} {'max_abs':>10}")
print("-" * 70)
for name, cos, mae in results:
    flag = "" if cos > 0.99 else "  ← BAD"
    print(f"{name:<48} {cos:>10.6f} {mae:>10.4f}{flag}")

bad = [r for r in results if r[1] < 0.99]
print(f"\n{'PASS' if not bad else 'FAIL'} — {len(bad)} op(s) below cos=0.99 threshold")


# ---------------------------------------------------------------
# Op 7: layer-0 of Qwen2.5-0.5B — composite of all the above ops
# Tests whether the per-op correctness composes correctly.
# Same code as decode-step layer 0, with random weights.
# ---------------------------------------------------------------
print("\n[7] composite layer 0: attn (Q/K/V→RoPE→SDPA→O) + MLP (gate/up→silu→down)")
HIDDEN_ = 896; N_Q_ = 14; N_KV_ = 2; D_ = 64; INTER_ = 4864; T_ = 100
rng2 = np.random.default_rng(7)
# Inputs
x = rng2.standard_normal((1, 1, HIDDEN_), dtype=np.float32) * 0.5
cos_ = rng2.standard_normal((1, 1, 1, D_), dtype=np.float32) * 0.1 + 1.0
sin_ = rng2.standard_normal((1, 1, 1, D_), dtype=np.float32) * 0.1
mask_full = np.full((128,), -1e9, dtype=np.float32); mask_full[:T_] = 0.0
# Build R (head_dim rotation matrix)
R_ = np.zeros((D_, D_), dtype=np.float32)
for ii in range(D_ // 2):
    R_[ii + D_ // 2, ii] = -1.0
    R_[ii, ii + D_ // 2] = 1.0
# Weights
ln1_g = rng2.standard_normal((HIDDEN_,), dtype=np.float32) * 0.1 + 1.0
q_w = rng2.standard_normal((HIDDEN_, N_Q_ * D_), dtype=np.float32) * 0.02
q_b = rng2.standard_normal((N_Q_ * D_,), dtype=np.float32) * 0.01
k_w = rng2.standard_normal((HIDDEN_, N_KV_ * D_), dtype=np.float32) * 0.02
k_b = rng2.standard_normal((N_KV_ * D_,), dtype=np.float32) * 0.01
v_w = rng2.standard_normal((HIDDEN_, N_KV_ * D_), dtype=np.float32) * 0.02
v_b = rng2.standard_normal((N_KV_ * D_,), dtype=np.float32) * 0.01
o_w = rng2.standard_normal((HIDDEN_, HIDDEN_), dtype=np.float32) * 0.02
ln2_g = rng2.standard_normal((HIDDEN_,), dtype=np.float32) * 0.1 + 1.0
gate_w = rng2.standard_normal((HIDDEN_, INTER_), dtype=np.float32) * 0.02
up_w = rng2.standard_normal((HIDDEN_, INTER_), dtype=np.float32) * 0.02
down_w = rng2.standard_normal((INTER_, HIDDEN_), dtype=np.float32) * 0.02
kc = rng2.standard_normal((1, N_KV_, 128, D_), dtype=np.float32) * 0.05
vc = rng2.standard_normal((1, N_KV_, 128, D_), dtype=np.float32) * 0.05


def _layer(x, cos, sin, mask, kc, vc, ln1_g, q_w, q_b, k_w, k_b, v_w, v_b,
           o_w, ln2_g, gate_w, up_w, down_w, R):
    eps = 1e-6
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    h = x * jax.lax.rsqrt(ms + eps) * ln1_g
    q = (h @ q_w + q_b).reshape(1, N_Q_, 1, D_)
    k = (h @ k_w + k_b).reshape(1, N_KV_, 1, D_)
    v = (h @ v_w + v_b).reshape(1, N_KV_, 1, D_)
    q_r = q * cos + (q @ R) * sin
    k_r = k * cos + (k @ R) * sin
    groups = N_Q_ // N_KV_
    k_b_ = jnp.repeat(kc, groups, axis=1)
    v_b_ = jnp.repeat(vc, groups, axis=1)
    scale = float(1.0 / np.sqrt(D_))
    scores = (q_r @ jnp.swapaxes(k_b_, -1, -2)) * scale
    scores = scores + mask[None, None, None, :]
    p = jax.nn.softmax(scores, axis=-1)
    attn = (p @ v_b_).transpose(0, 2, 1, 3).reshape(1, 1, HIDDEN_)
    x = x + attn @ o_w
    post_attn = x
    ms2 = jnp.mean(x * x, axis=-1, keepdims=True)
    h2 = x * jax.lax.rsqrt(ms2 + eps) * ln2_g
    g_ = jax.nn.silu(h2 @ gate_w)
    u_ = h2 @ up_w
    x = x + (g_ * u_) @ down_w
    return post_attn, x


# numpy fp32 ref
def _layer_np(x, cos, sin, mask, kc, vc):
    eps = 1e-6
    ms = np.mean(x * x, axis=-1, keepdims=True)
    h = x * (1.0 / np.sqrt(ms + eps)) * ln1_g
    q = (h @ q_w + q_b).reshape(1, N_Q_, 1, D_)
    k = (h @ k_w + k_b).reshape(1, N_KV_, 1, D_)
    v = (h @ v_w + v_b).reshape(1, N_KV_, 1, D_)
    q_r = q * cos + (q @ R_) * sin
    k_r = k * cos + (k @ R_) * sin
    groups = N_Q_ // N_KV_
    k_b_ = np.repeat(kc, groups, axis=1)
    v_b_ = np.repeat(vc, groups, axis=1)
    scale = 1.0 / np.sqrt(D_)
    scores = (q_r @ k_b_.transpose(0, 1, 3, 2)) * scale
    scores = scores + mask[None, None, None, :]
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    p = e / np.sum(e, axis=-1, keepdims=True)
    attn = (p @ v_b_).transpose(0, 2, 1, 3).reshape(1, 1, HIDDEN_)
    x = x + attn @ o_w
    post_attn = x
    ms2 = np.mean(x * x, axis=-1, keepdims=True)
    h2 = x * (1.0 / np.sqrt(ms2 + eps)) * ln2_g
    g_ = h2 @ gate_w; g_ = g_ / (1.0 + np.exp(-g_))
    u_ = h2 @ up_w
    x = x + (g_ * u_) @ down_w
    return post_attn, x

post_attn_ref, post_mlp_ref = _layer_np(x, cos_, sin_, mask_full, kc, vc)
got = _run("layer0", _layer, [x, cos_, sin_, mask_full, kc, vc,
                                ln1_g, q_w, q_b, k_w, k_b, v_w, v_b,
                                o_w, ln2_g, gate_w, up_w, down_w, R_])
post_attn_got, post_mlp_got = got

print(f"  layer0 post_attn   cos={_cosine(post_attn_ref, post_attn_got):.6f} "
      f"max_abs={_max_abs(post_attn_ref, post_attn_got):.4f}")
print(f"  layer0 post_mlp    cos={_cosine(post_mlp_ref, post_mlp_got):.6f} "
      f"max_abs={_max_abs(post_mlp_ref, post_mlp_got):.4f}")
