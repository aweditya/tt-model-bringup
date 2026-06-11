# 20: The tt_jax Module — A Modular Jaxpr-to-TT-NN Backend

## What is tt_jax?

**Q: What is the tt_jax module and how does it differ from the earlier interpreter experiments?**

A: tt_jax is the cleaned-up, modular, unit-tested version of everything we learned in experiments 14-18. Instead of a single monolithic script, it's decomposed into four focused modules:

| Module | Responsibility | Lines |
|--------|---------------|-------|
| `tensors.py` | Host ↔ device transfer with tile padding | ~100 |
| `ops.py` | Op registry: 20 Jaxpr primitives → TT-NN | ~260 |
| `interpret.py` | Jaxpr walker + sub-jaxpr handler | ~115 |
| `trace.py` | Trace capture for dispatch elimination | ~110 |

The key design principle: **the registry is a plain dict**. Adding a new op is one function + one dict entry. No class hierarchies, no metaclasses.

## Architecture

**Q: How does a JAX function end up running on Blackhole?**

```
JAX function → jax.make_jaxpr() → Jaxpr (flat IR) → Interpreter.run() → TT-NN ops on Blackhole
```

Step by step:
1. JAX traces the Python function into a Jaxpr — a flat list of equations like `c = add a b`
2. Our `Interpreter` walks each equation, looks up the primitive name in `REGISTRY`
3. The registry handler converts Jaxpr semantics to TT-NN API calls
4. Tensors are moved to/from device with tile padding (multiples of 32)

**Q: What's the role of the trace executor?**

`TracedExecutor` wraps the interpreter with TT-NN's trace capture API:
1. `compile()`: runs the Jaxpr once (warm-up), then captures a second run as a trace
2. `run()`: overwrites input buffers and replays the trace — no Python dispatch overhead
3. `release()`: frees device trace resources

This gives 2-3x speedup by eliminating the ~21µs per-op Python dispatch cost.

## Op Coverage

**Q: What Jaxpr primitives does tt_jax support?**

20 primitives across 5 categories:

| Category | Primitives |
|----------|-----------|
| Elementwise | add, sub, mul, div, neg, exp, log, sqrt, rsqrt, reciprocal, max, integer_pow |
| Matmul | dot_general |
| Reductions | reduce_max, reduce_sum |
| Shape | broadcast_in_dim, reshape, transpose, squeeze |
| Pass-through | convert_element_type, stop_gradient |

Notable implementation details:
- **max(x, 0) → relu**: detected and optimized automatically
- **integer_pow**: unrolled as repeated multiplication (x^2 = mul(x,x))
- **broadcast_in_dim**: CPU round-trip because TT-NN TILE_LAYOUT can't broadcast mismatched shapes
- **Scalar ops**: detected via `jax_core.Literal` and handled with `ttnn.add(tensor, scalar)` instead of tensor-tensor ops

## Test Results

**Q: How do you verify correctness?**

19 unit tests compare TT-NN output against JAX CPU reference within bf16 tolerance:

```
  [PASS] add(x, y):                max_err=0.023139  mean_err=0.002464
  [PASS] add(x, 3.0):              max_err=0.023118  mean_err=0.004369
  [PASS] sub(x, y):                max_err=0.019441  mean_err=0.002428
  [PASS] mul(x, y):                max_err=0.032045  mean_err=0.001492
  [PASS] mul(x, 2.5):              max_err=0.037000  mean_err=0.003885
  [PASS] neg(x):                   max_err=0.007776  mean_err=0.001116
  [PASS] exp(x):                   max_err=0.035530  mean_err=0.002094
  [PASS] log(x):                   max_err=0.010050  mean_err=0.001792
  [PASS] sqrt(x):                  max_err=0.006094  mean_err=0.001444
  [PASS] relu(x):                  max_err=0.007637  mean_err=0.000567
  [PASS] x^2:                      max_err=0.068866  mean_err=0.003313
  [PASS] matmul(x, w):             max_err=0.001909  mean_err=0.000345
  [PASS] matmul+bias:              max_err=0.002458  mean_err=0.000358
  [PASS] sum(x, axis=-1):          max_err=0.035554  mean_err=0.014090
  [PASS] max(x, axis=-1):          max_err=0.007595  mean_err=0.004394
  [PASS] softmax(x):               max_err=0.001164  mean_err=0.000066
  [PASS] layer_norm(x):            max_err=0.026537  mean_err=0.002689
  [PASS] MLP(x):                   max_err=0.002936  mean_err=0.000431
  [PASS] Linear+ReLU+LayerNorm:    max_err=0.044173  mean_err=0.004885
```

All errors are within bf16 precision expectations. Matmul and softmax are particularly accurate.

## What's Missing for a Full Transformer?

**Q: What additional ops would we need?**

To run a complete transformer (GPT-2 style):
- **gather/dynamic_slice** — for token embeddings
- **concatenate** — for multi-head attention head merging  
- **slice** — for splitting Q/K/V
- **tanh/sigmoid** — for GELU activation (or use the approximation via existing ops)
- **reduce_mean** — can be built from reduce_sum + div, but a direct handler would be cleaner

The current 20 ops already handle the core compute: matmul, softmax, layer norm, and all the elementwise ops in the residual stream.

## Key Lessons

1. **Plain dict registries beat class hierarchies** — easy to test, easy to extend, easy to read
2. **Broadcast is the hardest part** — TT-NN's tile layout can't implicitly broadcast, so we do CPU round-trips. This is the #1 performance bottleneck to fix.
3. **bf16 is plenty accurate** — worst case mean error is 0.015 (reduce_sum), which is expected for bf16 accumulation over 64 elements
4. **Sub-jaxpr handling is essential** — JAX wraps relu in `custom_jvp_call` and some ops in `pjit`. We recursively interpret these.
