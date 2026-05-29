"""
Experiment 03: JIT Overhead and Eager vs Compiled Execution
===========================================================
Goal: Measure the ACTUAL cost of tracing+compilation, show why JIT is
slow the first time, and compare JAX compiled vs PyTorch eager execution.

Key questions:
  Q1: How much time does tracing + compilation take?
  Q2: How does recompilation behave with different input shapes?
  Q3: What does PyTorch eager execution actually do differently?
  Q4: How does torch.compile compare?
"""

import time
import jax
import jax.numpy as jnp

# ============================================================
# Q1: Measuring tracing + compilation overhead
# ============================================================
print("=" * 60)
print("Q1: The cost of tracing + compilation")
print("=" * 60)

def mlp_forward(x, w1, w2):
    h = jnp.maximum(x @ w1, 0)
    return h @ w2

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (64, 256))
w1 = jax.random.normal(key, (256, 128))
w2 = jax.random.normal(key, (128, 64))

# Measure each phase separately
print("\nPhase 1: Tracing (Python → Jaxpr)")
start = time.perf_counter()
jaxpr = jax.make_jaxpr(mlp_forward)(x, w1, w2)
trace_time = time.perf_counter() - start
print(f"  Time: {trace_time*1000:.3f} ms")

print("\nPhase 2: Lowering (Jaxpr → StableHLO)")
start = time.perf_counter()
lowered = jax.jit(mlp_forward).lower(x, w1, w2)
lower_time = time.perf_counter() - start
print(f"  Time: {lower_time*1000:.3f} ms")

print("\nPhase 3: Compilation (StableHLO → machine code)")
start = time.perf_counter()
compiled = lowered.compile()
compile_time = time.perf_counter() - start
print(f"  Time: {compile_time*1000:.3f} ms")

print("\nPhase 4: First execution (run compiled code)")
start = time.perf_counter()
result = compiled(x, w1, w2).block_until_ready()
first_exec = time.perf_counter() - start
print(f"  Time: {first_exec*1000:.3f} ms")

print("\nPhase 5: Subsequent execution (cached)")
times = []
for _ in range(1000):
    start = time.perf_counter()
    result = compiled(x, w1, w2).block_until_ready()
    times.append(time.perf_counter() - start)
avg_exec = sum(times) / len(times)
print(f"  Time: {avg_exec*1000:.3f} ms (avg over 1000 runs)")

total_first = trace_time + lower_time + compile_time + first_exec
print(f"\n  TOTAL first call:      {total_first*1000:.1f} ms")
print(f"  Each subsequent call:  {avg_exec*1000:.3f} ms")
print(f"  Break-even after:      ~{int(total_first / avg_exec)} calls")

# ============================================================
# Q2: What triggers recompilation?
# ============================================================
print("\n" + "=" * 60)
print("Q2: What triggers recompilation?")
print("=" * 60)

mlp_jit = jax.jit(mlp_forward)

# Same shapes → cache hit
print("\nCall 1 (shape 64x256, first compile):")
start = time.perf_counter()
_ = mlp_jit(x, w1, w2).block_until_ready()
print(f"  Time: {(time.perf_counter()-start)*1000:.1f} ms")

print("Call 2 (same shape, cache hit):")
start = time.perf_counter()
_ = mlp_jit(x, w1, w2).block_until_ready()
print(f"  Time: {(time.perf_counter()-start)*1000:.1f} ms")

# Different shapes → recompilation!
x2 = jax.random.normal(key, (32, 256))  # batch size changed
print("Call 3 (shape 32x256, RECOMPILE):")
start = time.perf_counter()
_ = mlp_jit(x2, w1, w2).block_until_ready()
print(f"  Time: {(time.perf_counter()-start)*1000:.1f} ms")

print("Call 4 (shape 32x256, cache hit):")
start = time.perf_counter()
_ = mlp_jit(x2, w1, w2).block_until_ready()
print(f"  Time: {(time.perf_counter()-start)*1000:.1f} ms")

# Back to original shape → cache hit (both are cached now)
print("Call 5 (shape 64x256 again, cache hit):")
start = time.perf_counter()
_ = mlp_jit(x, w1, w2).block_until_ready()
print(f"  Time: {(time.perf_counter()-start)*1000:.1f} ms")

print("\n  KEY: JAX recompiles when input SHAPES change, not values.")
print("  Each unique shape combo gets its own cached binary.")

# ============================================================
# Q3: JAX compiled vs PyTorch eager — apples to apples
# ============================================================
print("\n" + "=" * 60)
print("Q3: JAX compiled vs PyTorch eager")
print("=" * 60)

import torch

# Same computation, same sizes
def mlp_torch(x, w1, w2):
    h = torch.relu(x @ w1)
    return h @ w2

x_t = torch.randn(64, 256)
w1_t = torch.randn(256, 128)
w2_t = torch.randn(128, 64)

# PyTorch eager: no compilation, just execute ops one by one
print("\nPyTorch EAGER (no compilation):")
# Warmup
for _ in range(10):
    _ = mlp_torch(x_t, w1_t, w2_t)

times_torch = []
for _ in range(1000):
    start = time.perf_counter()
    result_t = mlp_torch(x_t, w1_t, w2_t)
    times_torch.append(time.perf_counter() - start)
avg_torch = sum(times_torch) / len(times_torch)
print(f"  Avg: {avg_torch*1000:.3f} ms per call")

# JAX compiled
print("\nJAX COMPILED (after compilation):")
print(f"  Avg: {avg_exec*1000:.3f} ms per call")

print(f"\n  Ratio: PyTorch eager is {avg_torch/avg_exec:.1f}x vs JAX compiled")

# Now try torch.compile
print("\nPyTorch COMPILED (torch.compile):")
try:
    mlp_compiled = torch.compile(mlp_torch)
    # First call triggers compilation
    start = time.perf_counter()
    _ = mlp_compiled(x_t, w1_t, w2_t)
    torch_compile_time = time.perf_counter() - start
    print(f"  First call (compile): {torch_compile_time*1000:.1f} ms")

    # Warmup
    for _ in range(10):
        _ = mlp_compiled(x_t, w1_t, w2_t)

    times_torch_compiled = []
    for _ in range(1000):
        start = time.perf_counter()
        result_tc = mlp_compiled(x_t, w1_t, w2_t)
        times_torch_compiled.append(time.perf_counter() - start)
    avg_tc = sum(times_torch_compiled) / len(times_torch_compiled)
    print(f"  Subsequent: {avg_tc*1000:.3f} ms per call")
except Exception as e:
    print(f"  torch.compile not available or failed: {e}")

# ============================================================
# Q4: What "eager execution" actually means — step by step
# ============================================================
print("\n" + "=" * 60)
print("Q4: What 'eager execution' means, concretely")
print("=" * 60)

print("""
When PyTorch runs `h = torch.relu(x @ w1)`, here's what ACTUALLY happens:

  1. Python evaluates `x @ w1`:
     → Python calls x.__matmul__(w1)
     → PyTorch's C++ dispatcher receives the call
     → Dispatcher selects the CPU matmul kernel
     → Kernel runs, allocates output tensor, returns to Python
     → Python now holds a reference to the result tensor

  2. Python evaluates `torch.relu(...)`:
     → Python calls into torch.relu C++ code
     → Dispatcher selects the CPU relu kernel
     → Kernel runs, allocates ANOTHER output tensor, returns
     → The matmul result tensor is now "done" — sitting in memory

  Each operation is independent. PyTorch doesn't know relu comes after
  matmul until it actually happens. That's "eager" — execute immediately,
  one op at a time.

  Cost of eager:
  - Python interpreter overhead between every op (~1-5µs each)
  - Dispatcher overhead (type checking, kernel selection)
  - Memory allocation for every intermediate result
  - No fusion: matmul writes to memory, relu reads it back
  - But: zero compilation cost, immediate execution, easy to debug

When JAX runs `jax.jit(f)(x, w1)`, here's what happens:

  First call:
  1. JAX traces f with abstract values (records ops, doesn't run them)
  2. Produces jaxpr: [matmul, relu] as a graph
  3. Lowers to StableHLO (MLIR)
  4. XLA optimizes: fuses matmul+relu into one kernel
  5. XLA compiles to machine code via LLVM
  6. Runs the compiled binary
  → Slow! All of steps 1-5 are overhead.

  Subsequent calls (same shapes):
  1. Look up cached compiled binary
  2. Run it
  → Fast! No Python, no dispatch, fused kernels.

  Cost of compiled:
  - First call is MUCH slower (compilation)
  - But: no Python overhead per op
  - But: fused kernels (matmul+relu = one memory pass)
  - But: compiler can reorder, simplify, vectorize
""")

# Let's actually measure the per-operation overhead
print("Measuring per-operation overhead:")
print()

# PyTorch: cost of dispatch + execution for a tiny op
tiny = torch.randn(4)
times_dispatch = []
for _ in range(10000):
    start = time.perf_counter()
    _ = torch.relu(tiny)
    times_dispatch.append(time.perf_counter() - start)
avg_dispatch = sum(times_dispatch) / len(times_dispatch)
print(f"  PyTorch relu(float[4]): {avg_dispatch*1e6:.1f} µs")
print(f"    (most of this is Python→C++ dispatch, not actual computation)")

# JAX eager (no jit): similar dispatch overhead
tiny_jax = jnp.array([1.0, -2.0, 3.0, -4.0])
times_jax_eager = []
for _ in range(10000):
    start = time.perf_counter()
    _ = jnp.maximum(tiny_jax, 0).block_until_ready()
    times_jax_eager.append(time.perf_counter() - start)
avg_jax_eager = sum(times_jax_eager) / len(times_jax_eager)
print(f"  JAX relu(float[4]) no jit: {avg_jax_eager*1e6:.1f} µs")

# JAX compiled
relu_jit = jax.jit(lambda x: jnp.maximum(x, 0))
_ = relu_jit(tiny_jax).block_until_ready()
times_jax_jit = []
for _ in range(10000):
    start = time.perf_counter()
    _ = relu_jit(tiny_jax).block_until_ready()
    times_jax_jit.append(time.perf_counter() - start)
avg_jax_jit = sum(times_jax_jit) / len(times_jax_jit)
print(f"  JAX relu(float[4]) jit:    {avg_jax_jit*1e6:.1f} µs")

print(f"""
  On tiny tensors, dispatch overhead dominates — JIT may not help.
  The win comes with larger computations where fusion matters.
""")
