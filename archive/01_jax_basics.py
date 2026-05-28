"""
Experiment 01: JAX Basics
=========================
Goal: Understand what JAX does and WHY it's interesting.

Key questions we're answering:
  Q1: What does jax.jit actually do? How fast is it vs non-jit?
  Q2: What does the compiled XLA graph look like?
  Q3: How does jax.grad work?
  Q4: How does jax.vmap work?
  Q5: What happens when you jit a realistic computation (small MLP forward pass)?

Run on: ssh tenstorrent (CPU only for now)
"""

import jax
import jax.numpy as jnp
import time

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print()

# ============================================================
# Q1: What does jax.jit do? How fast is it vs non-jit?
# ============================================================
print("=" * 60)
print("Q1: jax.jit performance")
print("=" * 60)

def matmul_chain(x, w1, w2, w3):
    """Three matrix multiplies in sequence — a simplified MLP."""
    x = x @ w1
    x = jnp.maximum(x, 0)  # ReLU
    x = x @ w2
    x = jnp.maximum(x, 0)
    x = x @ w3
    return x

# Create some data
key = jax.random.PRNGKey(42)
k1, k2, k3, k4 = jax.random.split(key, 4)
x = jax.random.normal(k1, (128, 512))
w1 = jax.random.normal(k2, (512, 256))
w2 = jax.random.normal(k3, (256, 256))
w3 = jax.random.normal(k4, (256, 128))

# Warmup (first call traces + compiles)
matmul_chain_jit = jax.jit(matmul_chain)
_ = matmul_chain_jit(x, w1, w2, w3).block_until_ready()

# Benchmark non-jit
N = 100
start = time.perf_counter()
for _ in range(N):
    result = matmul_chain(x, w1, w2, w3).block_until_ready()
no_jit_time = (time.perf_counter() - start) / N

# Benchmark jit
start = time.perf_counter()
for _ in range(N):
    result = matmul_chain_jit(x, w1, w2, w3).block_until_ready()
jit_time = (time.perf_counter() - start) / N

print(f"  No JIT:  {no_jit_time*1000:.3f} ms per call")
print(f"  JIT:     {jit_time*1000:.3f} ms per call")
print(f"  Speedup: {no_jit_time/jit_time:.1f}x")
print()

# ============================================================
# Q2: What does the compiled XLA graph look like?
# ============================================================
print("=" * 60)
print("Q2: Inspecting the XLA HLO")
print("=" * 60)

# jax.jit returns a compiled function — we can inspect its HLO
lowered = jax.jit(matmul_chain).lower(x, w1, w2, w3)
print("StableHLO text (first 1500 chars):")
print("-" * 40)
hlo_text = lowered.as_text()
print(hlo_text[:1500])
print(f"... ({len(hlo_text)} total chars)")
print()

# We can also see the optimized HLO that XLA produces
compiled = lowered.compile()
print(f"Compiled cost analysis (first 800 chars):")
print("-" * 40)
cost = compiled.cost_analysis()
print(str(cost)[:800])
print()

# ============================================================
# Q3: How does jax.grad work?
# ============================================================
print("=" * 60)
print("Q3: Automatic differentiation with jax.grad")
print("=" * 60)

def loss_fn(w, x, y_true):
    """Simple MSE loss for a single linear layer."""
    y_pred = x @ w
    return jnp.mean((y_pred - y_true) ** 2)

# Create data
x_small = jax.random.normal(k1, (32, 8))
w_small = jax.random.normal(k2, (8, 4))
y_true = jax.random.normal(k3, (32, 4))

# Compute gradient of loss w.r.t. first argument (w)
grad_fn = jax.grad(loss_fn)
grads = grad_fn(w_small, x_small, y_true)

print(f"  w shape:    {w_small.shape}")
print(f"  grad shape: {grads.shape}  (same as w — one gradient per parameter)")
print(f"  grad norm:  {jnp.linalg.norm(grads):.4f}")

# You can also get value AND gradient together
val_and_grad_fn = jax.value_and_grad(loss_fn)
loss_val, grads2 = val_and_grad_fn(w_small, x_small, y_true)
print(f"  loss value: {loss_val:.4f}")
print(f"  grads match: {jnp.allclose(grads, grads2)}")
print()

# ============================================================
# Q4: How does jax.vmap work?
# ============================================================
print("=" * 60)
print("Q4: Automatic vectorization with jax.vmap")
print("=" * 60)

def single_dot(a, b):
    """Dot product of two vectors."""
    return jnp.dot(a, b)

# Without vmap: loop over batch
batch_a = jax.random.normal(k1, (1000, 64))
batch_b = jax.random.normal(k2, (1000, 64))

# vmap transforms single-example fn into batched fn
batched_dot = jax.vmap(single_dot)
result_vmap = batched_dot(batch_a, batch_b)

# Manual loop for comparison
result_loop = jnp.array([single_dot(batch_a[i], batch_b[i]) for i in range(1000)])

print(f"  vmap result shape: {result_vmap.shape}")
print(f"  Results match: {jnp.allclose(result_vmap, result_loop)}")

# Benchmark
batched_dot_jit = jax.jit(batched_dot)
_ = batched_dot_jit(batch_a, batch_b).block_until_ready()

start = time.perf_counter()
for _ in range(100):
    batched_dot_jit(batch_a, batch_b).block_until_ready()
vmap_time = (time.perf_counter() - start) / 100

start = time.perf_counter()
for _ in range(10):
    jnp.array([single_dot(batch_a[i], batch_b[i]) for i in range(1000)])
loop_time = (time.perf_counter() - start) / 10

print(f"  vmap+jit: {vmap_time*1000:.3f} ms")
print(f"  loop:     {loop_time*1000:.3f} ms")
print(f"  Speedup:  {loop_time/vmap_time:.0f}x")
print()

# ============================================================
# Q5: Putting it together — small MLP forward + backward
# ============================================================
print("=" * 60)
print("Q5: MLP forward + backward pass")
print("=" * 60)

def mlp_loss(params, x, y_true):
    """2-layer MLP with ReLU, MSE loss."""
    h = jnp.maximum(x @ params['w1'] + params['b1'], 0)
    y_pred = h @ params['w2'] + params['b2']
    return jnp.mean((y_pred - y_true) ** 2)

# Initialize params
params = {
    'w1': jax.random.normal(k1, (512, 256)) * 0.01,
    'b1': jnp.zeros(256),
    'w2': jax.random.normal(k2, (256, 10)) * 0.01,
    'b2': jnp.zeros(10),
}
x_train = jax.random.normal(k3, (256, 512))
y_train = jax.random.normal(k4, (256, 10))

# JIT compile the value_and_grad
@jax.jit
def train_step(params, x, y):
    loss, grads = jax.value_and_grad(mlp_loss)(params, x, y)
    # Simple SGD update
    new_params = jax.tree.map(lambda p, g: p - 0.001 * g, params, grads)
    return new_params, loss

# Warmup
params_new, loss0 = train_step(params, x_train, y_train)

# Train for a few steps and time it
start = time.perf_counter()
p = params
for i in range(100):
    p, loss = train_step(p, x_train, y_train)
_ = jax.tree.map(lambda x: x.block_until_ready(), p)
train_time = time.perf_counter() - start

print(f"  Initial loss: {loss0:.4f}")
print(f"  Final loss:   {loss:.4f}")
print(f"  100 steps in: {train_time*1000:.1f} ms ({train_time*10:.2f} ms/step)")
print()

# Show the HLO for the full train step
lowered_train = jax.jit(train_step).lower(params, x_train, y_train)
hlo = lowered_train.as_text()
n_ops = hlo.count('\n')
print(f"  Train step HLO: {len(hlo)} chars, ~{n_ops} lines")
print(f"  (This single compiled function includes forward, backward, AND parameter update)")
print()
print("KEY INSIGHT: jax.jit compiles forward + backward + update into ONE fused program.")
print("No Python overhead between steps. XLA can optimize across all of it.")
