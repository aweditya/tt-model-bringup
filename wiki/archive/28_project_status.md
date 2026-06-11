# Wiki 28: Project Status — tt-xla as of Experiment 31

## Q: What have we built?

**A:** A working JAX-to-TT-NN execution pipeline that runs real pretrained models on Tenstorrent Blackhole. The system has three layers:

1. **Op Registry** (`tt_jax/ops.py`) — Maps 26 Jaxpr primitives to TT-NN operations
2. **Jaxpr Interpreter** (`tt_jax/interpret.py`) — Walks JAX's IR and executes ops on device
3. **Trace Capture** (`tt_jax/jit.py`) — `@executor.jit` API for compiled execution

## Q: What does it actually do?

**A:** GPT-2 small (124M params) generates coherent English text on Blackhole:

```
Prompt: "The meaning of life is"
Output: "not to be confused with the meaning of life. It is to be
         understood as a life of the"
```

Per-token latency: ~95ms. Top-5 predictions match JAX CPU reference exactly.

## Q: What ops are supported?

**A:** 26 Jaxpr primitives, covering 100% of GPT-2's computation graph:

| Category | Ops |
|----------|-----|
| Elementwise | add, sub, mul, div, neg, exp, log, sqrt, rsqrt, reciprocal, max, tanh |
| Comparison | ge, select_n |
| Power | integer_pow |
| Matmul | dot_general |
| Reduction | reduce_sum, reduce_max |
| Shape | broadcast_in_dim, reshape, transpose, squeeze |
| Slicing | split, slice, dynamic_slice, concatenate |
| Index | iota |
| Pass-through | convert_element_type, stop_gradient, pjit |

## Q: What TT-NN native ops do we use?

**A:**

| Op | TT-NN API | Status |
|----|-----------|--------|
| LayerNorm | `ttnn.layer_norm` | Working, cosine 0.999996 |
| GELU | `ttnn.gelu(fast_and_approximate_mode=False)` | Working, matches gelu_new |
| FlashAttention-2 | `ttnn.transformer.scaled_dot_product_attention` | Working, cosine 0.999909 |
| Head concat | `ttnn.transformer.concatenate_heads` | Working |
| Matmul | `ttnn.matmul` | Working, up to 95 TFLOPS at 2048×2048 |
| Trace capture | `ttnn.begin/end/execute_trace` | Working for 2D ops |

## Q: What's the performance progression?

**A:**

| Experiment | What | Speed |
|------------|------|-------|
| 09 | First elementwise on device | ~1ms/op |
| 11 | MLP inference | 17ms |
| 20 | Transformer (random weights) | 5.6ms (179 fwd/sec) |
| 22 | Traced transformer | 0.39ms (2,564 fwd/sec) |
| 24 | Raw matmul peak | 0.07ms at 2048² (95 TFLOPS) |
| 25 | 12-layer transformer (traced) | 4.6ms (217 fwd/sec) |
| 27 | GPT-2 first run | 519ms / 12 layers |
| 29 | GPT-2 + native attention | 430ms |
| 30 | GPT-2 + device LN/GELU | 383ms |
| 31 | GPT-2 text generation | **95ms/token** |

## Q: What are the remaining CPU round-trips?

**A:** Per GPT-2 layer, only 2:

| Operation | Where | Why CPU? |
|-----------|-------|----------|
| QKV split + reshape | CPU | TT-NN lacks native split for (1,T,3C)→3×(1,H,T,D) |
| Head concat + reshape | CPU | Reshape 4D→3D sometimes fails on device |

Everything else (LayerNorm, GELU, matmul, add, FlashAttention) runs on Blackhole.

## Q: What did we learn about Blackhole's architecture?

**A:**

- **110 usable Tensix cores** (11×10 grid, 2 columns harvested + 1 dispatch)
- **8 DRAM banks**, interleaved by default
- **1.5 MB L1 SRAM per core**
- **95 TFLOPS peak matmul** at 2048×2048 bfloat16
- **Sub-linear scaling**: 8x more compute only increases latency 1.75x
- **Trace capture eliminates Python dispatch**: 3x speedup for traced ops
- **DRAM round-trips are the bottleneck**: 16+ per layer with current INTERLEAVED layout

## Q: What's the path to production-grade performance?

**A:** Three remaining optimizations, in priority order:

1. **Eliminate last 2 CPU round-trips** — `ttnn.transformer.split_query_key_value_and_split_heads` + native `concatenate_heads` → enables trace capture for full GPT-2
2. **Trace capture for full model** — Record 12-layer forward pass, replay with `execute_trace` → expected ~30ms for 12 layers (based on small-model benchmarks)
3. **Sharded L1 memory** — Eliminate DRAM round-trips between ops → 2-4x speedup for elementwise chains

Target: **~10ms/token** (100 tok/sec), competitive with GPU inference for GPT-2 small.

## Q: What's the PJRT plugin story?

**A:** To integrate as a real JAX backend (`jax.jit(f, backend='tt')`), we need a C++ PJRT plugin. Our Python interpreter proves the concept works but can't be a JAX backend directly. The architecture would be:

```
JAX → StableHLO → C++ PJRT plugin → TT-NN → Blackhole
```

This is structurally identical to Apple's jax-mps/applejax approach (StableHLO → PJRT → MLX → Apple GPU). The Python prototype validates that the TT-NN op mapping is correct and performant.

---

*Updated after Experiment 31. 11 experiments, 8 wiki pages, 26 ops, 1 real model generating text.*
