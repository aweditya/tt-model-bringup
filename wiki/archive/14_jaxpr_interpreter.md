# Automated Jaxpr → TT-NN Interpreter

## Q: Can we automatically execute any JAX function on Blackhole?

**A: Mostly yes.** We built a ~200-line Python interpreter (`TTNNInterpreter`) that walks a Jaxpr and maps each equation to TT-NN ops. It handles MLP inference and simple math correctly, failing only on unsupported primitives like `reduce_max`.

## What Works

| Test | Equations | Result | Max Error |
|------|-----------|--------|-----------|
| 2-layer MLP (128→256→10) | 7 (matmul, add, relu, broadcast) | **Correct** | 0.098 |
| Quadratic (x²+2x+1) | 4 (mul, add with scalars) | **Correct** | 0.171 |
| Softmax | 7 (reduce, exp, div) | **Failed** | `reduce_max` unsupported |

## Supported Primitives

| Jaxpr Primitive | TT-NN Mapping | Notes |
|----------------|---------------|-------|
| `dot_general` | `ttnn.matmul` | Matrix multiply |
| `add` | `ttnn.add` / scalar overload | Handles literal scalars |
| `mul` | `ttnn.mul` / `ttnn.multiply(t, scalar)` | Scalar mul needs special path |
| `sub` | `ttnn.sub` | |
| `neg` | `ttnn.neg` | |
| `exp` | `ttnn.exp` | |
| `log` | `ttnn.log` | |
| `max` (relu) | `ttnn.relu` | When `max(x, 0)` |
| `broadcast_in_dim` | No-op | TT-NN broadcasts implicitly |
| `custom_jvp_call` | Recurse into sub-jaxpr | JAX wraps relu in this |
| `pjit` | Recurse into sub-jaxpr | JAX wraps some ops in this |

## Missing Primitives (needed for real models)

| Primitive | Used In | Difficulty |
|-----------|---------|------------|
| `reduce_sum`, `reduce_max` | Softmax, loss functions | Medium — TT-NN has `ttnn.sum`, `ttnn.max` |
| `div` | Softmax, normalization | Easy — `ttnn.div` exists |
| `gather`, `scatter` | Embeddings, indexing | Hard — requires careful indexing |
| `slice`, `dynamic_slice` | Attention masks, padding | Medium |
| `conv_general_dilated` | CNNs | Medium — `ttnn.conv2d` exists |
| `concatenate` | Multi-head attention | Medium |
| `while_loop`, `cond` | RNNs, dynamic control flow | Hard — traces can't handle these |

## Key Insights

### 1. Scalar Operations Need Special Handling
JAX's Jaxpr uses literals for scalar values (e.g., `mul(2.0, x)`). TT-NN's `ttnn.mul` expects two tensors. We handle this by detecting `jax_core.Literal` inputs and using scalar overloads like `ttnn.multiply(tensor, 2.0)`.

### 2. Sub-Jaxprs Are Common
JAX wraps operations like `relu` in `custom_jvp_call` (for automatic differentiation) and some ops in `pjit`. The interpreter must recursively execute these sub-jaxprs. Both `Jaxpr` and `ClosedJaxpr` objects need handling.

### 3. Tile Padding Is Invisible But Critical
TT-NN requires tensors to be padded to 32×32 tile boundaries. Our interpreter handles this in `to_ttnn()` and un-pads in `from_ttnn()`. A real backend would need to be more careful about padding semantics.

### 4. The Performance Tax Is Interpretation, Not Compute
The interpreter runs at 0.66ms per MLP forward pass (vs 0.05ms with trace capture). The 13x overhead is pure Python interpretation — creating tensors, looking up handlers, calling TT-NN from Python. Wrapping the interpreter's ops in trace capture would recover all of this.

## Architecture of the Interpreter

```python
class TTNNInterpreter:
    def interpret(self, jaxpr, args):
        # 1. Bind inputs to environment
        # 2. Walk equations sequentially
        # 3. For each equation:
        #    a. Look up handler by primitive name
        #    b. Evaluate input variables
        #    c. Call TT-NN op
        #    d. Bind output to environment
        # 4. Return output variables
```

This is exactly what a "Level 1" PJRT plugin would do in C++ — just faster because it avoids Python overhead and uses trace capture.

## Experiment

`experiments/14_jaxpr_interpreter.py` — run on Blackhole p150a device 0, 2026-04-21.
