# How JAX Works Under the Hood

## Q: What does `jax.jit` actually do?

**A:** It does three things in sequence:

### Step 1: Tracing → Jaxpr

When you call `jax.jit(f)(x, y)` for the first time, JAX **traces** your Python function. It runs your function with abstract "tracer" values (not real data) and records every operation. The output is a **jaxpr** (JAX expression) — a simple functional IR.

For `add_relu(x, y) = max(x + y, 0)`:

```
{ lambda ; a:f32[4] b:f32[4]. let
    c:f32[4] = add a b
    d:f32[4] = max c 0.0
  in (d,) }
```

Read it like this: "given inputs `a` and `b` (both float32 vectors of length 4), compute `c = a + b`, then `d = max(c, 0)`, return `d`." That's it — it's just a list of operations with typed inputs and outputs. No Python control flow, no side effects.

### Step 2: Lowering → StableHLO

The jaxpr gets converted to **StableHLO**, which is an MLIR dialect (more on this below):

```mlir
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
  %2 = stablehlo.maximum %0, %1 : tensor<4xf32>
  return %2 : tensor<4xf32>
}
```

This is almost 1:1 with the jaxpr but in MLIR syntax. Notice the `broadcast_in_dim` — the scalar `0.0` needs to be broadcast to match the vector shape. The jaxpr hid this; StableHLO makes it explicit.

### Step 3: Compilation → Machine Code

The XLA compiler (or a PJRT plugin) takes the StableHLO and:
1. Converts it to HLO (XLA's internal IR)
2. Runs optimization passes (fusion, layout assignment, etc.)
3. Generates device code (LLVM IR → x86 for CPU, PTX for GPU, etc.)

The **optimized HLO** for our function shows XLA fused everything into one operation:

```
ENTRY %main.7 {
  %Arg_0.1 = f32[4]{0} parameter(0)
  %Arg_1.2 = f32[4]{0} parameter(1)
  ROOT %broadcast_maximum_fusion = f32[4]{0} fusion(%Arg_0.1, %Arg_1.2),
       kind=kLoop, calls=%fused_computation
}
```

The `add` + `broadcast` + `maximum` became a single **fused computation**. One kernel, one pass over the data. This is XLA's main trick — fusing operations to minimize memory traffic.

**After the first call**, the compiled machine code is cached. Subsequent calls with the same input shapes skip all of this and run the cached binary directly.

---

## Q: What does `jax.grad` actually do?

**A:** `jax.grad(f)` returns a **new function** that computes the gradient of `f` with respect to its first argument. It works by **transforming the jaxpr** using automatic differentiation rules.

For `f(w, x) = sum((w * x)²)`, the mathematical gradient is `∂f/∂w = 2 * w * x²`.

JAX computes this automatically:

```python
>>> jax.grad(simple_loss)(w, x)
[8.  4.  1.5]   # matches manual: 2 * [1,2,3] * [2,1,0.5]² = [8, 4, 1.5]
```

The **jaxpr of the gradient function** reveals the chain rule at work:

```
{ lambda ; a:f32[3] b:f32[3]. let
    c:f32[3] = mul a b         # w * x
    e:f32[3] = integer_pow c   # (w * x)^1
    f:f32[3] = mul 2.0 e       # 2 * (w * x)
    h:f32[3] = mul f b         # 2 * (w * x) * x = 2 * w * x²
  in (h,) }
```

Key insight: `jax.grad` doesn't compute gradients numerically (finite differences) or symbolically (like Mathematica). It **transforms the computation graph** by applying the chain rule to each operation. The result is another jaxpr that, when compiled and run, produces exact gradients.

When you combine `jax.jit(jax.grad(f))`, the gradient computation itself gets compiled into optimized machine code — just as fast as the forward pass.

---

## Q: What does `jax.vmap` actually do?

**A:** `jax.vmap(f)` transforms a function that works on single examples into one that works on **batches**, by adding a batch dimension to every operation in the trace.

For a dot product function:

```
# Original jaxpr (single vectors):
{ lambda ; a:f32[3] b:f32[3]. let
    c:f32[3] = mul a b
    d:f32[] = reduce_sum[axes=(0,)] c
  in (d,) }

# vmap'd jaxpr (batch of vectors):
{ lambda ; a:f32[2,3] b:f32[2,3]. let
    c:f32[2,3] = mul a b
    d:f32[2] = reduce_sum[axes=(1,)] c    ← axis shifted!
  in (d,) }
```

Notice what changed:
- Input shapes: `f32[3]` → `f32[2,3]` (batch dimension prepended)
- Output shape: `f32[]` → `f32[2]` (one result per batch element)
- Reduce axis: `(0,)` → `(1,)` (sum over the vector dimension, not the batch dimension)

**vmap didn't write a loop.** It rewrote the operations to work on higher-rank tensors. When compiled, this becomes a single vectorized kernel — which is why it was 2,364x faster than a Python loop in Experiment 01.

---

## Q: What exactly is StableHLO?

**A:** StableHLO is a **vocabulary of ~100 tensor operations** defined as an MLIR dialect. Think of it as a standardized language for expressing ML computations.

### What "MLIR dialect" means

**MLIR** (Multi-Level Intermediate Representation) is a compiler framework. It doesn't define one IR — it lets you define many **dialects**, each with their own operations. A dialect is just a namespace of operations with defined semantics.

StableHLO is one such dialect. Its operations include:
- `stablehlo.add`, `stablehlo.multiply` — elementwise arithmetic
- `stablehlo.dot_general` — matrix multiplication (generalized)
- `stablehlo.reduce` — reductions (sum, max, etc.)
- `stablehlo.broadcast_in_dim` — shape broadcasting
- `stablehlo.maximum`, `stablehlo.minimum` — elementwise comparisons
- ~95 more (convolution, gather, scatter, sort, etc.)

### How to read StableHLO / MLIR syntax

```mlir
%0 = stablehlo.dot_general %arg0, %arg1,
     contracting_dims = [1] x [0],
     precision = [DEFAULT, DEFAULT]
     : (tensor<4x3xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
```

Parsing this:
- `%0` = name of the result (SSA variable — each value assigned once)
- `stablehlo.dot_general` = the operation (dialect.operation_name)
- `%arg0, %arg1` = inputs
- `contracting_dims = [1] x [0]` = contract dim 1 of arg0 with dim 0 of arg1 (standard matmul)
- `: (tensor<4x3xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>` = type signature

General pattern: `%result = dialect.op %inputs, attributes : (input_types) -> output_type`

### Why StableHLO exists

It's the **portability layer**. Any framework that can output StableHLO (JAX, PyTorch, TensorFlow) can target any compiler that consumes it (XLA, IREE, TT-MLIR). Guaranteed backward compatible for 5 years.

---

## Q: How does StableHLO/MLIR get lowered to something the CPU understands?

**A:** Through a pipeline of progressive transformations. Each step lowers to a more concrete representation.

### The Full Pipeline (CPU path)

```
Python function
  ↓ JAX tracing
Jaxpr (abstract operations on typed arrays)
  ↓ lowering
StableHLO (MLIR dialect — ~100 tensor ops)
  ↓ conversion
HLO (XLA's internal IR — same ops, different format)
  ↓ optimization passes (fusion, layout assignment, simplification)
Optimized HLO (fused operations, concrete memory layouts)
  ↓ code generation
LLVM IR (low-level, close to assembly)
  ↓ LLVM backend
x86 machine code (actual CPU instructions)
```

We can see the first three levels in our experiment:

**Level 1 — Jaxpr** (abstract):
```
dot_general a b → max result 0.0
```

**Level 2 — StableHLO** (portable MLIR):
```mlir
%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
%2 = stablehlo.maximum %0, %1
```

**Level 3 — Optimized HLO** (device-specific, after fusion):
```
%dot.5 = dot(%Arg_0.1, %Arg_1.2)
ROOT %broadcast_maximum_fusion = fusion(%dot.5), kind=kLoop, calls=%fused_computation
```

XLA decided: keep the `dot` as its own op (because it maps to an optimized BLAS call), but fuse `broadcast + maximum` into a single loop kernel.

**Level 4 — LLVM IR → Machine Code**: not human-readable from JAX, but this is where LLVM takes the abstract operations and emits actual `vmulps`, `vaddps`, `vmaxps` x86 SIMD instructions (or calls into MKL/Eigen for the matmul).

### The Tenstorrent Path (via tt-xla)

```
StableHLO (same as above — the portability layer)
  ↓ tt-mlir compiler
TTIR (TT Intermediate Representation — hardware-agnostic TT dialect)
  ↓ lowering passes
TTNN dialect (maps directly to TT-NN API calls)
  ↓ code generation
C++ / Python calling TT-NN → TT-Metalium → Blackhole hardware
```

The key: **StableHLO is where frameworks and hardware diverge.** Everything above StableHLO is framework-specific (JAX vs PyTorch). Everything below is hardware-specific (CPU vs GPU vs Tenstorrent). StableHLO is the meeting point.

---

## Experiment

Run `experiments/02_jax_internals.py` on the remote host to see all of this live:
```bash
ssh tenstorrent
source ~/tt-xla-env/bin/activate
python3 ~/02_jax_internals.py
```

## Sources
- Experiment 02 results (run 2026-04-21 on CPU)
- StableHLO spec: https://openxla.org/stablehlo
- MLIR overview: https://mlir.llvm.org/
- XLA architecture: https://openxla.org/xla
