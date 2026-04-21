# Q&A: Correctness, Data Flow, and Architecture

*Captured from discussion, 2026-04-21*

## Q: How are we checking correctness, especially for bf16?

**A:** Every experiment follows the same pattern:

1. **Compute a reference** in fp32 using PyTorch or JAX on CPU
2. **Run the same operation** on Blackhole in bf16
3. **Compare** using absolute and relative error metrics

```python
ref = torch_matmul(x_fp32, w_fp32)         # fp32 reference
result_tt = ttnn.to_torch(ttnn_result)       # bf16 result from Blackhole
abs_err = (result_tt.float() - ref).abs()    # compare in fp32
```

The key question is: **what error is acceptable?** BFloat16 has only 7 bits of mantissa (vs 23 for fp32), giving ~2 decimal digits of precision. For a single bf16 operation, the relative error is bounded by ~0.8% (2^-7). But errors **accumulate** through a computation chain:

| Operation | Expected Error | What We Measured |
|-----------|---------------|-----------------|
| Single matmul (256×256) | O(K × 2^-7) ≈ 0.5-1.0 | max 0.87, mean 0.08 |
| 2-layer MLP | Higher (chained matmuls) | max 0.098, mean 0.016 |
| Attention (Q@K^T → softmax → @V) | Higher still | max 0.034, mean ~0.01 |

The rule of thumb: **if class predictions agree 90%+ and errors are under 1.0 for typical values, bf16 is working correctly.** The errors aren't bugs — they're the expected cost of 16-bit floating point.

For our experiments we also check:
- **Softmax rows sum to ~1.0** (verifies the normalization is correct)
- **Class agreement** between bf16 and fp32 (e.g., 94% in experiment 11)
- **Shape correctness** (output dimensions match expected)

A production system would use **PCC (Pearson Correlation Coefficient)** — TT-NN's own test suite uses PCC > 0.9999 as the pass criterion. This is more robust than absolute error for large tensors.

## Q: For matmul, how do input tiles get to the cores that need them?

**A:** You're right — each output tile C[i,j] requires a dot product across the K dimension: `sum_k A[i,k] × B[k,j]`. That's K/32 tile-tile multiplications, and those input tiles could be anywhere.

With **DRAM interleaving** (the fast path):

```
Core #37 needs to compute C[3,7]
  → Needs A[3,0..K/32] and B[0..K/32,7]

Step 1: Core #37 issues read requests to DRAM for tiles A[3,0] and B[0,7]
Step 2: DRAM controller sends tiles over NoC to core #37's L1
Step 3: Core #37's matrix engine multiplies them, accumulates partial sum
Step 4: Repeat for k=1,2,...K/32-1
Step 5: Write output tile C[3,7] back to DRAM
```

The critical insight: **DRAM reads go through a different NoC path than core-to-core L1 reads.** DRAM has dedicated read/write channels and the DRAM controller handles scheduling. TT-NN's matmul kernels pipeline this — while core #37 is computing tile k, it's already prefetching tile k+1 from DRAM. The DRAM bandwidth (8 channels × ~50 GB/s each ≈ 400 GB/s) is high enough to feed all 110 cores simultaneously for large matmuls.

With **L1 interleaving** (the slow path):

```
Core #37 needs tile A[3,0] — but it lives in core #12's L1
  → Core #37 sends a NoC read request to core #12
  → Core #12 reads from its L1 and sends data back over NoC
  → This is a two-hop, point-to-point transfer
  → 110 cores all doing this simultaneously creates NoC congestion
```

The NoC is a 2D mesh — traffic from core #37 to core #12 may pass through multiple intermediate cores. Under heavy load, this creates **congestion and stalls** that DRAM interleaving avoids entirely.

**Sharded L1** avoids this by ensuring core #37's input tiles are in core #37's L1 — zero NoC traffic for local data.

## Q: What would it take to support remaining JAX ops?

The Jaxpr interpreter (experiment 14) handles: dot_general, add, mul, sub, neg, exp, log, relu, broadcast_in_dim, custom_jvp_call, pjit.

Experiment 15 confirmed TT-NN has: max, sum, mean (all with axis), reshape, permute, transpose, softmax.

**Remaining ops for real models, estimated difficulty:**

| Op | Used In | TT-NN Equivalent | Difficulty |
|----|---------|-------------------|------------|
| reduce_sum, reduce_max | Softmax, loss | ttnn.sum, ttnn.max | Easy — just wire up |
| div | Normalization | ttnn.div or ttnn.reciprocal + mul | Easy |
| sqrt, rsqrt | LayerNorm, attention scaling | ttnn.sqrt, ttnn.rsqrt | Easy |
| gather | Embeddings, indexing | ttnn.embedding? | Medium |
| scatter | Gradient accumulation | Unclear | Hard |
| concatenate | Multi-head attention | ttnn.concat | Medium |
| slice | Attention masks, head splitting | ttnn.slice | Medium |
| conv_general_dilated | CNNs | ttnn.conv2d | Medium |
| iota | Index generation | No direct equivalent | Hard — compute on host |
| while_loop, cond | RNNs, dynamic models | Cannot trace-capture | Very hard |
| sort | Top-k sampling | ttnn.topk? | Hard |

For an MLP-only backend: we're basically done (add reduce_sum for loss).
For transformers: need gather (embeddings), concat, slice, plus what we have.
For full StableHLO coverage: ~100 primitives total, maybe 60 are tractable.

## Q: Is the attention we ran scaled dot-product attention?

**A:** Yes. Experiment 16 implements exactly:

```
scores = (Q @ K^T) / sqrt(d_k)
weights = softmax(scores, dim=-1)
output = weights @ V
```

This is the standard scaled dot-product attention from "Attention Is All You Need." The scaling by 1/sqrt(d_k) prevents the dot products from growing too large (which would push softmax into its saturated regime where gradients vanish).

Multi-head attention was implemented as a loop over heads — not the most efficient approach, but it works correctly. A proper implementation would reshape Q, K, V into (batch, heads, seq, d_k) and use batched matmul.
