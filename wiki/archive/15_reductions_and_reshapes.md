# Reductions and Reshapes on Blackhole

## Q: Which reduction/reshape ops does TT-NN support on Blackhole?

**A: All the ones we need.** Every reduction (max, sum, mean), reshape, transpose, and even built-in softmax work correctly. This unblocks the Jaxpr interpreter for softmax and other real workloads.

## Results

### Reductions

| Op | Variant | Works? | Max Error vs PyTorch |
|----|---------|--------|---------------------|
| `ttnn.max` | global | Yes | 0.0066 |
| `ttnn.max` | dim=-1 | Yes | 0.0077 |
| `ttnn.max` | dim=0 | Yes | 0.0077 |
| `ttnn.max` | dim=-1, keepdim=True | Yes | — |
| `ttnn.sum` | global | Yes | 0.1182 |
| `ttnn.sum` | dim=-1 | Yes | 0.0683 |
| `ttnn.sum` | dim=0 | Yes | 0.0293 |
| `ttnn.mean` | global | Yes | 0.0001 |
| `ttnn.mean` | dim=-1 | Yes | 0.0011 |

**All reductions work with axis arguments and keepdim.** Errors are small and consistent with bf16 precision (sum accumulates more error over 64 elements, as expected).

### Reshapes and Transpositions

| Op | Variant | Works? | Max Error |
|----|---------|--------|-----------|
| `ttnn.reshape` | 32x64 -> 64x32 | Yes | 0.0077 |
| `ttnn.reshape` | 32x64 -> 1x32x64 | Yes | — |
| `ttnn.reshape` | 32x64 -> 1x2048 (flatten) | Yes | 0.0077 |
| `ttnn.permute` | (1, 0) — 2D transpose | Yes | 0.0077 |
| `ttnn.transpose` | swap dims 0, 1 | Yes | 0.0077 |
| `ttnn.permute` | (0, 2, 1) — 3D | Yes | 0.0077 |

**All reshape/transpose ops work.** The 0.0077 errors are just bf16 rounding, not computational errors.

### Built-in Softmax

| Op | Dim | Max Error | Mean Error |
|----|-----|-----------|------------|
| `ttnn.softmax` | dim=-1 | 0.009966 | 0.000295 |
| `ttnn.softmax` | dim=0 | 0.004220 | 0.000200 |

**TT-NN has a built-in softmax** that works along any axis. Errors are well within bf16 tolerance.

### Softmax Benchmark (128x512 tensor)

| Method | Time (ms) | Speedup vs manual |
|--------|-----------|-------------------|
| Built-in `ttnn.softmax` | 0.061 | 1.00x (baseline) |
| Manual (max + sub + exp + sum + recip + mul) | 0.169 | 0.36x |
| Simple manual (exp + sum + recip + mul) | 0.131 | 0.47x |

**Built-in softmax is 2.8x faster than manual.** The fused kernel avoids intermediate memory traffic. Manual correctness is also good (max err 0.000608).

## Q: Why did experiment 14's interpreter fail on softmax?

**A: It wasn't missing ops — it was missing op *mappings*.** The Jaxpr `reduce_max` and `reduce_sum` primitives have direct TT-NN equivalents (`ttnn.max(t, dim=...)` and `ttnn.sum(t, dim=...)`). The interpreter just needs handlers for them.

## Q: What does this mean for the Jaxpr interpreter?

We can now add these handlers to the interpreter:

```python
def _op_reduce_max(self, invars, params, eqn):
    a = self.eval_var(invars[0])
    axes = params['axes']
    for ax in sorted(axes, reverse=True):
        a = ttnn.max(a, dim=ax, keepdim=True)
    return a

def _op_reduce_sum(self, invars, params, eqn):
    a = self.eval_var(invars[0])
    axes = params['axes']
    for ax in sorted(axes, reverse=True):
        a = ttnn.sum(a, dim=ax, keepdim=True)
    return a
```

Or, for softmax specifically, detect the pattern and use `ttnn.softmax` directly (2.8x faster).

## Q: Should we use built-in composite ops or decompose into primitives?

**Use built-in ops when available.** Built-in softmax is 2.8x faster because the fused kernel avoids materializing intermediates to memory. A smart compiler would pattern-match Jaxpr subgraphs to TT-NN composite ops — this is exactly what XLA does with its fusion passes.

**Hierarchy of dispatch strategies:**
1. Pattern-match to TT-NN composite ops (softmax, layer_norm, etc.) — fastest
2. Map individual Jaxpr primitives to TT-NN ops — correct but slower
3. Fall back to host computation — last resort
