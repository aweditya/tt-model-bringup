"""
Experiment 02: Understanding JAX Internals
==========================================
Goal: Demystify jit, grad, vmap, StableHLO, and the lowering pipeline.

We trace each transformation step-by-step to see EXACTLY what happens.
"""

import jax
import jax.numpy as jnp

# ============================================================
# PART 1: What does jax.jit ACTUALLY do? (Step by step)
# ============================================================
print("=" * 60)
print("PART 1: What jax.jit does, step by step")
print("=" * 60)

def add_relu(x, y):
    z = x + y
    return jnp.maximum(z, 0)

x = jnp.array([1.0, -2.0, 3.0, -4.0])
y = jnp.array([0.5,  3.0, -5.0, 1.0])

# Step 1: jax.make_jaxpr shows the TRACE (jaxpr = JAX expression)
print("\nStep 1: JAX traces your function into a 'jaxpr'")
print("  (This is JAX's internal IR — before XLA sees anything)")
print()
jaxpr = jax.make_jaxpr(add_relu)(x, y)
print(f"  {jaxpr}")

# Step 2: Lower to StableHLO
print("\nStep 2: The jaxpr gets lowered to StableHLO (an MLIR dialect)")
lowered = jax.jit(add_relu).lower(x, y)
print()
print(lowered.as_text())

# Step 3: Compile to machine code
print("Step 3: XLA compiles StableHLO into machine code")
compiled = lowered.compile()
print(f"  Cost analysis: {compiled.cost_analysis()}")
print(f"  Result: {compiled(x, y)}")
print(f"  Expected: {add_relu(x, y)}")

# Step 4: Show what the compiler optimized into
print("\nStep 4: The OPTIMIZED HLO (what XLA actually runs)")
print("  (Use .as_text() on compiled to see post-optimization IR)")
hlo_text = compiled.as_text()
print(hlo_text[:2000])
print()

# ============================================================
# PART 2: What does jax.grad ACTUALLY do?
# ============================================================
print("=" * 60)
print("PART 2: What jax.grad does, step by step")
print("=" * 60)

def simple_loss(w, x):
    """f(w, x) = sum((w * x)^2). Gradient w.r.t. w = 2 * w * x^2."""
    return jnp.sum((w * x) ** 2)

w = jnp.array([1.0, 2.0, 3.0])
x_val = jnp.array([2.0, 1.0, 0.5])

# What grad returns
grad_fn = jax.grad(simple_loss)  # differentiates w.r.t. first arg (w)
grads = grad_fn(w, x_val)

# Manual gradient: d/dw sum((w*x)^2) = 2*w*x^2
manual_grads = 2 * w * x_val ** 2

print(f"\n  f(w, x) = sum((w * x)^2)")
print(f"  w = {w}")
print(f"  x = {x_val}")
print(f"  f(w, x) = {simple_loss(w, x_val)}")
print(f"\n  jax.grad result:  {grads}")
print(f"  Manual gradient:  {manual_grads}")
print(f"  Match: {jnp.allclose(grads, manual_grads)}")

# Show the jaxpr of the gradient function
print(f"\n  Jaxpr of grad(f):")
grad_jaxpr = jax.make_jaxpr(grad_fn)(w, x_val)
print(f"  {grad_jaxpr}")

# Show the StableHLO of the gradient
print(f"\n  StableHLO of jit(grad(f)):")
grad_lowered = jax.jit(grad_fn).lower(w, x_val)
print(grad_lowered.as_text())

# ============================================================
# PART 3: What does jax.vmap ACTUALLY do?
# ============================================================
print("=" * 60)
print("PART 3: What jax.vmap does, step by step")
print("=" * 60)

def dot_product(a, b):
    """Dot product of two vectors."""
    return jnp.sum(a * b)

# Without vmap: operates on single vectors
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
print(f"\n  Single: dot({a}, {b}) = {dot_product(a, b)}")

# With vmap: operates on batches
batched_dot = jax.vmap(dot_product)
batch_a = jnp.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
batch_b = jnp.array([[7.0, 8.0, 9.0],
                      [1.0, 0.0, 1.0]])
print(f"  Batch:  vmap(dot)({batch_a.tolist()}, {batch_b.tolist()}) = {batched_dot(batch_a, batch_b)}")

# Show jaxpr: notice how vmap adds a batch dimension
print(f"\n  Jaxpr of dot_product (single):")
print(f"  {jax.make_jaxpr(dot_product)(a, b)}")

print(f"\n  Jaxpr of vmap(dot_product) (batched):")
print(f"  {jax.make_jaxpr(batched_dot)(batch_a, batch_b)}")

# The StableHLO reveals what vmap actually compiles to
print(f"\n  StableHLO of jit(vmap(dot_product)):")
lowered_vmap = jax.jit(batched_dot).lower(batch_a, batch_b)
print(lowered_vmap.as_text())

# ============================================================
# PART 4: The full lowering pipeline
# ============================================================
print("=" * 60)
print("PART 4: The full lowering pipeline for a matmul+relu")
print("=" * 60)

def matmul_relu(x, w):
    return jnp.maximum(x @ w, 0)

x_mat = jnp.ones((4, 3))
w_mat = jnp.ones((3, 2))

print("\n--- Level 0: Python function ---")
print("  def matmul_relu(x, w): return jnp.maximum(x @ w, 0)")

print("\n--- Level 1: Jaxpr (JAX's internal trace) ---")
print(f"  {jax.make_jaxpr(matmul_relu)(x_mat, w_mat)}")

print("\n--- Level 2: StableHLO (MLIR dialect, portable) ---")
lowered_mr = jax.jit(matmul_relu).lower(x_mat, w_mat)
print(lowered_mr.as_text())

print("--- Level 3: Optimized HLO (XLA's internal, device-specific) ---")
compiled_mr = lowered_mr.compile()
print(compiled_mr.as_text()[:2000])

print(f"\n--- Level 4: Machine code (not directly printable) ---")
print(f"  The compiled object contains native machine code.")
print(f"  On CPU: this is x86 instructions generated via LLVM.")
print(f"  On GPU: this would be PTX -> SASS.")
print(f"  On Tenstorrent via tt-xla: StableHLO -> TTIR -> TTNN -> Metalium kernels.")
print(f"\n  Result: {compiled_mr(x_mat, w_mat)}")
