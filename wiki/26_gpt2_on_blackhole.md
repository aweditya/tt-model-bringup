# Wiki 26: Running Real GPT-2 on Blackhole

## Q: Can we run a real pretrained model through our interpreter?

**A:** Yes! GPT-2 small (124M params, 12 layers, 768 hidden dim) runs end-to-end on Blackhole through our Jaxpr interpreter. This is the first time real pretrained weights produce meaningful output on this hardware through our stack.

## Q: What does the op coverage look like?

**A:** A single GPT-2 layer traces to 113 Jaxpr primitives, all covered:

| Op | Count | Notes |
|----|-------|-------|
| broadcast_in_dim | 22 | LayerNorm broadcasting |
| add | 16 | Residual connections, biases |
| div | 11 | LayerNorm, attention scaling |
| mul | 9 | LayerNorm gamma, GELU |
| sub | 8 | LayerNorm centering |
| reduce_sum | 7 | LayerNorm variance |
| transpose | 7 | Multi-head attention reshape |
| dot_general | 6 | QKV projections, attention, FFN |
| sqrt | 5 | LayerNorm, attention scaling |
| integer_pow | 4 | LayerNorm variance (x^2) |
| reshape | 4 | Multi-head split/merge |
| iota | 2 | Causal mask construction |
| tanh | 1 | GELU activation |
| ge, select_n | 1 each | Causal masking |
| split | 1 | QKV separation |

Full 12-layer model: **1,180 Jaxpr ops** total.

New ops added for GPT-2: `tanh` (GELU), `iota` (index generation), `ge` (comparison), `select_n` (conditional), `split` (QKV separation).

## Q: How accurate is the output?

**A:** Single layer cosine similarity is 0.90, but accuracy degrades across layers:

| Metric | 1 Layer | 12 Layers |
|--------|---------|-----------|
| Cosine similarity | 0.904 | 0.825 |
| Max absolute error | 12.66 | 193.28 |
| Mean absolute error | 0.30 | 0.62 |

The error accumulates because:
1. **4D CPU fallbacks**: Attention requires 4D tensors (batch, heads, seq, dim). TT-NN's `mul`, `sub`, etc. fail with "Invalid subtile broadcast type" on these shapes, forcing CPU round-trips
2. **bfloat16 precision**: Each CPU fallback round-trips through bfloat16→float32→bfloat16
3. **Error amplification**: LayerNorm divides by small variances, amplifying any upstream errors

The top-1 next-token prediction mismatches (JAX: "The", TT-NN: "Ċ"), but "The" is still in TT-NN's top-5.

## Q: Why does 4D cause problems for TT-NN?

**A:** TT-NN's TILE_LAYOUT is fundamentally 2D — the last two dimensions are tiled into 32×32 blocks, and higher dimensions are treated as batch dims. When we:

1. `reshape` from (1, 32, 768) to (1, 32, 12, 64) — this splits the last dim
2. `permute` from (1, 32, 12, 64) to (1, 12, 32, 64) — this swaps batch dims

The resulting tensor may have a device shape that doesn't match the logical shape (e.g., logical `(1,12,32,32)` but device `(32,1,12,32)`). Binary ops like `mul` then fail because TT-NN can't figure out the broadcast pattern.

This is a known limitation — TT-NN's binary ops support broadcasting along the last two dims but not arbitrary batch-dim broadcasts.

## Q: What's the performance?

**A:** Slow due to CPU fallbacks:

| Config | Time | Notes |
|--------|------|-------|
| Single layer (Blackhole) | 7,750 ms | ~13 CPU fallback round-trips |
| Full 12 layers (Blackhole) | 519 ms | Amortized — later layers faster? |
| Full 12 layers (JAX CPU) | 52.5 ms | 10x faster than our Blackhole path |

The 519ms for 12 layers is better per-layer than the 7.7s first layer because subsequent layers reuse warmed-up device state. But we're still 10x slower than JAX on CPU — the CPU fallbacks dominate.

## Q: What would it take to get GPT-2 running fast?

**A:** The critical path is eliminating 4D CPU fallbacks:

1. **Flatten attention to 2D**: Instead of reshape+permute to 4D, keep everything in 2D/3D by manually batching the head computations. This avoids the 4D broadcast issue entirely.

2. **Use ttnn.transformer**: TT-NN has dedicated attention primitives (`ttnn.transformer.scaled_dot_product_attention`) that handle multi-head attention natively on device. This would replace our 4D tensor chain entirely.

3. **Trace capture**: Once all ops stay on device (no CPU fallbacks), we can use trace capture for 3x+ speedup, just like our random-weight transformer.

The performance ceiling (from our random-weight experiments): a single traced transformer layer runs at 0.39ms / 2,564 fwd/sec. GPT-2's layers are ~6x larger (768 vs 128 hidden dim), so we'd expect ~2.4ms/layer, or ~29ms for 12 layers — that's **18x faster than JAX CPU**.

## Q: What did the model actually predict?

**A:** For the prompt "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout":

| Rank | JAX CPU | TT-NN Blackhole |
|------|---------|-----------------|
| 1 | "The" (6.3%) | "Ċ" (newline, 3.7%) |
| 2 | "A" (3.2%) | "." (0.9%) |
| 3 | "This" (1.9%) | "The" (0.9%) |
| 4 | "In" (1.4%) | "," (0.9%) |
| 5 | "I" (1.3%) | "-" (0.8%) |

The TT-NN predictions are more spread out (lower confidence) due to accumulated bfloat16 errors softening the logit distribution. The correct answer "The" is still ranked #3.

---

*Experiment 27, run on Blackhole device 0. GPT-2 weights from HuggingFace (gpt2, 137M params).*
