# What is JAX and Why Should You Care?

## Q: What is JAX in one sentence?

**A:** JAX is a Python library that lets you write NumPy-like code, then compiles it with XLA into optimized machine code for CPUs, GPUs, TPUs, or custom accelerators.

## Q: What are JAX's core transformations?

**A:** JAX has four key function transformations:

| Transform | What it does | Why it matters |
|-----------|-------------|----------------|
| `jax.jit` | JIT-compiles a function with XLA | Eliminates Python overhead, enables fusion |
| `jax.grad` | Returns a function computing gradients | Automatic differentiation for free |
| `jax.vmap` | Vectorizes a single-example function over a batch | Write for one sample, run on batches |
| `jax.pmap` | Parallelizes across multiple devices | Multi-GPU/TPU with minimal code |

The key design principle: you write a **pure function**, then **transform** it.

## Q: How much faster is `jax.jit` vs regular JAX?

**A (Experiment 01, CPU):** On a 3-layer matmul chain (128×512 → 256 → 256 → 128):

```
No JIT:  0.620 ms per call
JIT:     0.316 ms per call
Speedup: 2.0x
```

Only 2x on CPU because the CPU XLA backend is already fairly optimized and the bottleneck is the matmuls themselves, not Python overhead. **On GPU/TPU/accelerators, the speedup would be much larger** because JIT eliminates kernel launch overhead between operations.

## Q: How fast is `jax.vmap` vs a Python loop?

**A (Experiment 01, CPU):** Computing 1000 dot products:

```
vmap+jit: 0.070 ms
loop:     164.853 ms
Speedup:  2364x
```

`vmap` transforms a single-vector function into a batched operation that XLA compiles into a single vectorized kernel. The loop version dispatches 1000 separate operations through Python.

## Q: What does JAX actually compile? Can I see it?

**A:** Yes! `jax.jit(fn).lower(args)` gives you the **StableHLO** IR. Here's what our 3-matmul chain looks like:

```mlir
func.func public @main(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>,
                        %arg2: tensor<256x256xf32>, %arg3: tensor<256x128xf32>)
    -> tensor<128x128xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = []
  %2 = stablehlo.maximum %0, %1                    // ReLU
  %3 = stablehlo.dot_general %2, %arg2, ...        // Second matmul
  %5 = stablehlo.maximum %3, %4                    // ReLU
  %6 = stablehlo.dot_general %5, %arg3, ...        // Third matmul
  return %6
}
```

Key observations:
- Each operation is a **tensor-level op** (dot_general, maximum, broadcast)
- No loops, no Python, no overhead — just a clean computation graph
- This is the IR that gets sent to the XLA compiler (or a PJRT plugin like tt-xla)

## Q: Can JAX compile an entire training step into one program?

**A (Experiment 01):** Yes. A 2-layer MLP forward + backward + SGD update compiles into a single HLO module of ~72 lines. Performance:

```
100 training steps in 53.0 ms (0.53 ms/step)
```

The compiled HLO includes forward pass, loss computation, backpropagation, AND parameter updates — all in one compiled function. No Python interpreter between any of these steps.

**This is the key insight**: JAX gives the compiler **maximum visibility** over the entire computation, enabling optimizations that are impossible when operations are dispatched one-at-a-time.

## Q: Why is this especially relevant for Tenstorrent?

**A:** Tenstorrent hardware has:
- **No cache hierarchy** — data movement is explicit and expensive
- **3-kernel structure** (reader/compute/writer) — maps naturally to a dataflow graph
- **32×32 tile-based compute** — the compiler needs to plan data layout

When XLA can see the whole computation graph, it can:
1. **Fuse operations** to keep data in L1 SRAM instead of going back to DRAM
2. **Plan data layout** to match the 32×32 tile structure
3. **Schedule data movement** to overlap with computation

This is exactly what the rvLLM benchmark demonstrated on TPU: compiling the entire forward pass into a single fused while loop, achieving 16,794 tok/s with ~500 lines of JAX and zero custom kernels.

## Experiment

Run `experiments/01_jax_basics.py` on the remote host:
```bash
ssh tenstorrent
source ~/tt-xla-env/bin/activate
python3 ~/01_jax_basics.py
```

## Sources
- Experiment 01 results (run 2026-04-21 on CPU)
- rvLLM benchmarks: https://docs.solidsf.com/docs/bench
- JAX docs: https://jax.readthedocs.io/
