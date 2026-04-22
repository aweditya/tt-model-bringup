"""
Experiment 26: jit_tt API demo — jax.jit-like interface for Blackhole.

Demonstrates the high-level API: define JAX functions, compile once,
run repeatedly with different inputs at trace speed.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
import ttnn
import time

from tt_jax import Executor

# ── Open device ───────────────────────────────────────────────
device = ttnn.open_device(device_id=0)
executor = Executor(device)

# ── Example 1: Simple MLP ────────────────────────────────────
print("=== Example 1: MLP ===")

@executor.jit
def mlp(x, w1, b1, w2, b2):
    h = jax.nn.relu(jnp.dot(x, w1) + b1)
    return jnp.dot(h, w2) + b2

rng = np.random.RandomState(42)
x = rng.randn(32, 64).astype(np.float32) * 0.1
w1 = rng.randn(64, 128).astype(np.float32) * 0.1
b1 = rng.randn(128).astype(np.float32) * 0.1
w2 = rng.randn(128, 32).astype(np.float32) * 0.1
b2 = rng.randn(32).astype(np.float32) * 0.1

# First call: compiles (warmup + trace capture)
t0 = time.perf_counter()
result = mlp(x, w1, b1, w2, b2)
t_compile = time.perf_counter() - t0

# Verify
ref = np.array(jax.nn.relu(jnp.dot(x, w1) + b1))
ref = np.array(jnp.dot(ref, w2) + b2)
print(f"  Compile time: {t_compile*1000:.1f} ms")
print(f"  Max error: {np.abs(result - ref).max():.6f}")

# Subsequent calls: fast path (copy + trace)
for _ in range(3):
    mlp(x, w1, b1, w2, b2)
N = 200
t0 = time.perf_counter()
for _ in range(N):
    x_new = rng.randn(32, 64).astype(np.float32) * 0.1
    mlp(x_new, w1, b1, w2, b2)
t_avg = (time.perf_counter() - t0) / N
print(f"  Avg call: {t_avg*1000:.3f} ms ({1/t_avg:.0f}/sec)")

# ── Example 2: Transformer ───────────────────────────────────
print("\n=== Example 2: Transformer ===")

@executor.jit
def transformer(x, w_q, w_k, w_v, w_o, w1, w2, g1, b1, g2, b2):
    q = jnp.dot(x, w_q); k = jnp.dot(x, w_k); v = jnp.dot(x, w_v)
    scores = jnp.dot(q, k.T) / jnp.sqrt(jnp.array(128.0))
    attn = jax.nn.softmax(scores, axis=-1)
    context = jnp.dot(attn, v)
    h = x + jnp.dot(context, w_o)
    m = jnp.mean(h, axis=-1, keepdims=True)
    v_ = jnp.mean((h - m) ** 2, axis=-1, keepdims=True)
    h = g1 * (h - m) / jnp.sqrt(v_ + 1e-5) + b1
    ff = jax.nn.relu(jnp.dot(h, w1))
    ff = jnp.dot(ff, w2)
    h2 = h + ff
    m2 = jnp.mean(h2, axis=-1, keepdims=True)
    v2 = jnp.mean((h2 - m2) ** 2, axis=-1, keepdims=True)
    return g2 * (h2 - m2) / jnp.sqrt(v2 + 1e-5) + b2

d = 128
args = [
    rng.randn(64, d).astype(np.float32) * 0.01,    # x
    rng.randn(d, d).astype(np.float32) * 0.01,     # w_q
    rng.randn(d, d).astype(np.float32) * 0.01,     # w_k
    rng.randn(d, d).astype(np.float32) * 0.01,     # w_v
    rng.randn(d, d).astype(np.float32) * 0.01,     # w_o
    rng.randn(d, d*4).astype(np.float32) * 0.01,   # w1
    rng.randn(d*4, d).astype(np.float32) * 0.01,   # w2
    np.ones(d, dtype=np.float32),
    np.zeros(d, dtype=np.float32),
    np.ones(d, dtype=np.float32),
    np.zeros(d, dtype=np.float32),
]

t0 = time.perf_counter()
result = transformer(*args)
t_compile = time.perf_counter() - t0
print(f"  Compile time: {t_compile*1000:.1f} ms")
print(f"  Output shape: {result.shape}")

for _ in range(5):
    transformer(*args)
N = 200
t0 = time.perf_counter()
for _ in range(N):
    transformer(*args)
t_avg = (time.perf_counter() - t0) / N
print(f"  Avg traced call: {t_avg*1000:.3f} ms ({1/t_avg:.0f}/sec)")

# With new inputs each time
N2 = 100
t0 = time.perf_counter()
for _ in range(N2):
    new_x = rng.randn(64, d).astype(np.float32) * 0.01
    transformer(new_x, *args[1:])
t_avg2 = (time.perf_counter() - t0) / N2
print(f"  Avg with new input: {t_avg2*1000:.3f} ms ({1/t_avg2:.0f}/sec)")

# ── Example 3: Softmax (simple) ──────────────────────────────
print("\n=== Example 3: Softmax ===")

@executor.jit
def softmax(x):
    return jax.nn.softmax(x, axis=-1)

x_sm = rng.randn(64, 128).astype(np.float32)
result = softmax(x_sm)
ref = np.array(jax.nn.softmax(jnp.array(x_sm), axis=-1))
print(f"  Max error: {np.abs(result - ref).max():.6f}")

for _ in range(5):
    softmax(x_sm)
N = 500
t0 = time.perf_counter()
for _ in range(N):
    softmax(x_sm)
t_avg = (time.perf_counter() - t0) / N
print(f"  Avg call: {t_avg*1000:.3f} ms ({1/t_avg:.0f}/sec)")

# ── Cleanup ──────────────────────────────────────────────────
executor.release_all()
ttnn.close_device(device)
print("\nDone!")
