"""
Experiment 04: Interpretation vs Compilation
=============================================
Goal: Make "interpretation" concrete. Show exactly what happens when
PyTorch executes a graph vs when JAX compiles one.

Key insight: "interpretation" means walking a graph node-by-node and
dispatching each operation. "Compilation" means transforming the whole
graph into a single optimized program.
"""

import time
import torch
import jax
import jax.numpy as jnp

# ============================================================
# PART 1: Building a computation graph and "interpreting" it
# ============================================================
print("=" * 60)
print("PART 1: What 'interpretation' actually means")
print("=" * 60)

# Let's define a graph EXPLICITLY and interpret it by hand.
# This is essentially what PyTorch's eager mode does under the hood.

class Op:
    """A node in a computation graph."""
    def __init__(self, name, fn, inputs):
        self.name = name
        self.fn = fn
        self.inputs = inputs  # list of Op or tensor

def interpret(graph, env):
    """Walk the graph node by node, executing each operation."""
    for op in graph:
        args = [env[inp] if isinstance(inp, str) else inp for inp in op.inputs]
        env[op.name] = op.fn(*args)
    return env

# Define: y = relu(x @ w + b)
# As an explicit graph:
graph = [
    Op("matmul",    lambda x, w: x @ w,          ["x", "w"]),
    Op("add",       lambda a, b: a + b,           ["matmul", "b"]),
    Op("relu",      lambda a: torch.relu(a),      ["add"]),
]

x_t = torch.randn(64, 256)
w_t = torch.randn(256, 128)
b_t = torch.randn(128)

# "Interpret" the graph: walk nodes, dispatch each op
env = {"x": x_t, "w": w_t, "b": b_t}
interpret(graph, env)
print(f"\n  Result shape: {env['relu'].shape}")

print("""
  That's interpretation:
    for each node in graph:
        look up inputs → call the operation → store the result

  Each operation is dispatched INDEPENDENTLY:
    1. matmul: calls BLAS, writes result to memory
    2. add: reads matmul result from memory, writes sum to memory
    3. relu: reads sum from memory, writes relu'd result to memory

  3 operations = 3 kernel dispatches + 3 memory write + 2 memory reads
  of intermediate results.
""")

# ============================================================
# PART 2: PyTorch's three execution modes
# ============================================================
print("=" * 60)
print("PART 2: PyTorch execution modes")
print("=" * 60)

def mlp_layer(x, w, b):
    return torch.relu(x @ w + b)

# Mode 1: Eager (interpret each op as Python executes it)
print("\nMode 1: EAGER (default PyTorch)")
for _ in range(10):
    mlp_layer(x_t, w_t, b_t)
times = []
for _ in range(1000):
    start = time.perf_counter()
    _ = mlp_layer(x_t, w_t, b_t)
    times.append(time.perf_counter() - start)
print(f"  {sum(times)/len(times)*1e6:.1f} µs per call")

# Mode 2: torch.compile (trace → optimize → compile)
print("\nMode 2: torch.compile (trace + compile)")
try:
    compiled_mlp = torch.compile(mlp_layer)
    start = time.perf_counter()
    _ = compiled_mlp(x_t, w_t, b_t)
    print(f"  First call (compilation): {(time.perf_counter()-start)*1000:.0f} ms")
    for _ in range(10):
        compiled_mlp(x_t, w_t, b_t)
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = compiled_mlp(x_t, w_t, b_t)
        times.append(time.perf_counter() - start)
    print(f"  Subsequent: {sum(times)/len(times)*1e6:.1f} µs per call")
except Exception as e:
    print(f"  Failed: {e}")

# Mode 3: torch.jit.trace (older TorchScript approach)
print("\nMode 3: torch.jit.trace (TorchScript)")
try:
    traced_mlp = torch.jit.trace(mlp_layer, (x_t, w_t, b_t))
    for _ in range(10):
        traced_mlp(x_t, w_t, b_t)
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = traced_mlp(x_t, w_t, b_t)
        times.append(time.perf_counter() - start)
    print(f"  {sum(times)/len(times)*1e6:.1f} µs per call")
    print(f"\n  TorchScript graph:")
    print(f"  {traced_mlp.graph}")
except Exception as e:
    print(f"  Failed: {e}")

# ============================================================
# PART 3: JAX — show what fusion actually eliminates
# ============================================================
print("\n" + "=" * 60)
print("PART 3: What XLA fusion eliminates")
print("=" * 60)

def mlp_layer_jax(x, w, b):
    return jnp.maximum(x @ w + b, 0)

x_j = jnp.array(x_t.numpy())
w_j = jnp.array(w_t.numpy())
b_j = jnp.array(b_t.numpy())

# Show the UNFUSED operations
print("\nUNFUSED (what you'd execute without a compiler):")
print("  1. matmul: read x(64KB) + w(128KB), write tmp1(32KB)")
print("  2. add:    read tmp1(32KB) + b(0.5KB), write tmp2(32KB)")
print("  3. relu:   read tmp2(32KB), write out(32KB)")
print("  Total memory traffic: ~352 KB")

# Show the FUSED version
print("\nFUSED (what XLA compiles to):")
lowered = jax.jit(mlp_layer_jax).lower(x_j, w_j, b_j)
compiled = lowered.compile()
hlo = compiled.as_text()

# Count fusions
fusion_count = hlo.count("fusion(")
print(f"  Number of fusion nodes: {fusion_count}")
print(f"  XLA optimized HLO:")
print()
# Print just the ENTRY function
for line in hlo.split('\n'):
    stripped = line.strip()
    if stripped.startswith('ENTRY') or stripped.startswith('ROOT') or \
       stripped.startswith('%') and ('parameter' in stripped or 'dot' in stripped or 'fusion' in stripped):
        print(f"    {stripped}")

print("""
  With fusion, add+broadcast+relu become ONE kernel:
    1. matmul: read x + w, write tmp (can't fuse matmul — it's a library call)
    2. fused(add+relu): read tmp + b, write out (ONE pass, not two!)

  Eliminated: one full read+write of the intermediate tensor.
  On GPU/TPU where HBM bandwidth is precious, this is huge.
""")

# ============================================================
# PART 4: Concrete memory traffic comparison
# ============================================================
print("=" * 60)
print("PART 4: Measuring the difference (larger tensors)")
print("=" * 60)

# Make it big enough that memory traffic matters
def big_chain(x, w1, w2, w3):
    """Chain of operations with many intermediates."""
    h1 = x @ w1
    h1 = jnp.maximum(h1, 0)       # relu
    h1 = h1 * 0.5 + 0.1           # scale + shift
    h2 = h1 @ w2
    h2 = jnp.maximum(h2, 0)
    h2 = h2 * 0.5 + 0.1
    h3 = h2 @ w3
    return jnp.maximum(h3, 0)

key = jax.random.PRNGKey(0)
x_big = jax.random.normal(key, (512, 1024))
w1_big = jax.random.normal(key, (1024, 512))
w2_big = jax.random.normal(key, (512, 512))
w3_big = jax.random.normal(key, (512, 256))

# Unfused: call each op separately
def big_chain_unfused(x, w1, w2, w3):
    h1 = jnp.dot(x, w1)
    h1 = jnp.maximum(h1, 0)
    h1 = h1 * 0.5
    h1 = h1 + 0.1
    h2 = jnp.dot(h1, w2)
    h2 = jnp.maximum(h2, 0)
    h2 = h2 * 0.5
    h2 = h2 + 0.1
    h3 = jnp.dot(h2, w3)
    return jnp.maximum(h3, 0)

# Warmup
big_jit = jax.jit(big_chain)
_ = big_jit(x_big, w1_big, w2_big, w3_big).block_until_ready()
_ = big_chain_unfused(x_big, w1_big, w2_big, w3_big).block_until_ready()

N = 500
start = time.perf_counter()
for _ in range(N):
    big_chain_unfused(x_big, w1_big, w2_big, w3_big).block_until_ready()
unfused_time = (time.perf_counter() - start) / N

start = time.perf_counter()
for _ in range(N):
    big_jit(x_big, w1_big, w2_big, w3_big).block_until_ready()
fused_time = (time.perf_counter() - start) / N

print(f"  Unfused (9 separate ops): {unfused_time*1000:.3f} ms")
print(f"  Fused (jit compiled):     {fused_time*1000:.3f} ms")
print(f"  Speedup: {unfused_time/fused_time:.1f}x")

# Show the fused HLO
lowered_big = jax.jit(big_chain).lower(x_big, w1_big, w2_big, w3_big)
compiled_big = lowered_big.compile()
hlo_big = compiled_big.as_text()
fusions = hlo_big.count("fusion(")
dots = hlo_big.count("dot(")
print(f"\n  Optimized HLO: {dots} dot ops + {fusions} fusion ops")
print(f"  (relu+scale+shift fused into single kernels around each matmul)")
print(f"  Cost analysis: {compiled_big.cost_analysis()}")
