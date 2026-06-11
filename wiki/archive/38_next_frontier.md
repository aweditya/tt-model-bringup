# Wiki 38: The Next Frontier — Native RoPE, Traced Sampling, Batch Decode, and L1 Residency

## Q: Can we use ttnn's native `rotary_embedding_llama` instead of our rotation matrix trick? (Exp 54)

**A:** No — not yet. The native op requires HEIGHT_SHARDED `trans_mat`, and while we can satisfy that requirement, the op implements **interleaved** rotation (adjacent-pair swaps via a 32x32 trans_mat), not the **half** format that Qwen2.5 uses (midpoint split via `rotate_half`). More importantly, we confirmed several things:

### What works

- **`ttnn.embedding` for cos/sin lookup:** Table of shape `(MAX_SEQ, head_dim)` with position-based lookup works perfectly. Returns correct values, can be transposed and HEIGHT_SHARDED for use with `rotary_embedding_llama`.
- **Fused QK RoPE (`rotary_embedding_llama_fused_qk`):** Applies RoPE to both Q and K in a single call. Works when inputs are HEIGHT_SHARDED from `nlp_create_qkv_heads_decode`.
- **K-head RoPE:** The native op handles `n_kv_heads=2` correctly (not just `n_q_heads=14`).

### What doesn't work (for Qwen)

The `trans_mat` is a fixed 32x32 matrix that swaps adjacent pairs:
```
trans_mat[even, odd] = 1
trans_mat[odd, even] = -1
```
This implements **interleaved** rotation `(x0,x1), (x2,x3), ...` — but Qwen uses **half** rotation that splits at the midpoint. The cross-format cosine between interleaved and half RoPE is only ~0.51 — completely different operations.

### Speed comparison

| Approach | Latency | Notes |
|----------|---------|-------|
| Rotation matrix (ours) | **0.088ms/iter** | Q only: `matmul(Q, R) + mul + mul + add` |
| Native `rotary_embedding_llama` | ~0.065ms/iter | Q only, but wrong format for Qwen |
| Native fused QK | ~0.080ms/iter | Q+K together |

**Conclusion:** Our rotation matrix approach (0.088ms/iter, 0.999996 cosine vs numpy) is correct, fast, and the right approach for Qwen. The native op would save ~0.02ms but requires interleaved format. Not worth the format mismatch.

## Q: Does adding temperature sampling slow down traced decode? (Exp 54b)

**A:** Barely. Sampling happens on CPU, **outside** the trace — so trace execution time is identical between greedy and sampled modes. The only overhead is replacing `np.argmax` with a top-k sampling function.

### Greedy vs sampled performance

| Mode | Trace exec | Post-processing | Total | Throughput |
|------|-----------|-----------------|-------|------------|
| Greedy (traced) | 7.61ms | 3.57ms (argmax) | 11.18ms | **89.5 tok/sec** |
| Sampled (traced) | 7.61ms | 4.68ms (sampling) | 12.29ms | **81.4 tok/sec** |

### Sampling overhead

| Metric | Value |
|--------|-------|
| Argmax time | 3.57ms |
| Sampling time | 4.68ms |
| **Overhead** | **1.11ms (10% of total)** |

The sampling function uses `np.argpartition` (O(n)) instead of `np.argsort` (O(n log n)) for the 151,936-element vocab:

```python
def sample_top_k(logits, temp=0.7, top_k=50):
    logits = logits / temp
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(top_idx, p=probs))
```

**Key insight:** The trace captures only the 24-layer forward pass. Everything before (embedding lookup, RoPE buffer update) and after (logit readback, token selection) is CPU-side. Sampling adds 1.1ms of CPU time — a 10% penalty for dramatically better text quality.

### Text quality

Greedy decoding with a 0.5B model produces degenerate repetitions ("and and and..."). Temperature sampling (temp=0.7, top_k=50) produces diverse, coherent text that hits `<|endoftext|>` naturally. This confirms the finding from exp 49b: sampling is essential for small models, and now it works within the traced decode path.

## Q: Does batch decode scale on Blackhole P150? (Exp 54c)

**A:** Yes — near-perfect linear throughput scaling, with latency barely increasing as batch size grows.

### paged_update_cache with batch > 1

The cache update op accepts batch > 1 natively. The key shapes are:

| Tensor | Shape |
|--------|-------|
| KV cache | `(batch, n_kv_heads, MAX_SEQ, head_dim)` |
| New K/V | `(1, batch, n_kv_heads, head_dim)` |
| update_idxs_tensor | `(batch,)` — **per-sequence positions** |

Each batch element can write to a different sequence position. The KV memory config shards across `batch_size` cores, with shard shape `(nearest_32(n_kv_heads), head_dim)`.

### SDPA decode with batch > 1

Q shape is `(1, batch, n_q_heads, head_dim)`, and `cur_pos_tensor` is `(batch,)` with per-sequence positions. The attention output has matching shape.

### Full single-layer decode scaling

| Batch | Latency | Throughput | Efficiency |
|-------|---------|------------|------------|
| 1 | 0.38ms | 2,601 tok/sec | 0.38 ms/tok |
| 2 | 0.39ms | 5,172 tok/sec | 0.19 ms/tok |
| 4 | 0.41ms | 9,819 tok/sec | 0.10 ms/tok |
| 8 | 0.41ms | 19,365 tok/sec | 0.05 ms/tok |

**7.4x throughput gain for 8x batch.** Latency increases by only 8% (0.38ms to 0.41ms) while throughput scales 7.4x. This is near-ideal linear scaling.

### Correctness

Batched SDPA (batch=2) matches two separate single-batch SDPA calls at **0.9999 cosine similarity**. The batched path is numerically equivalent.

### Projected full model performance

Current single-sequence traced decode: 7.6ms/tok (131.5 tok/sec). With the single-layer scaling factor of ~1.08x for batch=8:

- Projected batch=8 latency: ~8.2ms for 8 tokens
- Projected aggregate throughput: **~975 tok/sec**
- That would be 7.4x the single-sequence rate

### Limitation: HEIGHT_SHARDED Q breaks at batch > 1

The upstream tt-metal pattern uses HEIGHT_SHARDED Q with `batch * n_q_heads` cores. For Qwen on P150:

- batch=1: 1 * 14 = 14 cores (OK)
- batch=8: 8 * 14 = 112 cores > 110 available cores (FAIL)

This means INTERLEAVED Q works for all batch sizes tested, but HEIGHT_SHARDED Q (the faster path used in upstream models) cannot support batch=8 on P150 with Qwen's 14 Q heads. Batch=7 (98 cores) would be the maximum for the sharded path.

## Q: Does keeping tensors in L1 SRAM eliminate DRAM bottlenecks? (Exp 54d)

**A:** No — at decode tensor sizes, DRAM bandwidth is not the bottleneck. L1 residency provides no speedup and HEIGHT_SHARDED is actually slower due to layout conversion overhead.

### Benchmark: matmul -> silu -> matmul chain (32x896 tensors)

| Memory config | Latency | Relative |
|---------------|---------|----------|
| DRAM | 59 us | 1.0x |
| L1 INTERLEAVED | 61 us | 1.03x (same) |
| L1 HEIGHT_SHARDED | 171 us | **2.9x slower** |

### Why HEIGHT_SHARDED is slower

Matmul **cannot output HEIGHT_SHARDED** directly. The op chain becomes:

1. `to_memory_config(input, HEIGHT_SHARDED)` — reshard input
2. `matmul(sharded_input, weight, memory_config=HEIGHT_SHARDED)` — but matmul outputs interleaved internally, then reshards
3. `silu(sharded)` — elementwise ops preserve sharding (free)
4. `matmul(sharded_input, weight, memory_config=HEIGHT_SHARDED)` — reshard again

The layout conversions between matmul (which works in interleaved) and sharded memory dominate the cost. Each `to_memory_config` reshard is a data movement that takes longer than the DRAM round-trip it was supposed to eliminate.

### Which ops support HEIGHT_SHARDED output?

| Op | HEIGHT_SHARDED output | L1 INTERLEAVED output |
|----|-----------------------|-----------------------|
| matmul | FAIL (reshards internally) | OK |
| add | OK | OK |
| mul | OK | OK |
| silu | OK | OK |
| relu | OK | OK |
| neg | OK | OK |
| rms_norm | OK | OK |

Only elementwise ops (add, mul, silu, relu, neg) and rms_norm can output HEIGHT_SHARDED. Matmul, the most compute-heavy op, cannot — making end-to-end sharded chains impractical at these tensor sizes.

### Why DRAM isn't the bottleneck at decode sizes

A single decode token produces tensors of shape `(1, 32, 896)` = 28,672 floats = 57 KB in bfloat16. Blackhole's DRAM bandwidth is ~200 GB/s. Reading 57 KB from DRAM takes:

```
57 KB / 200 GB/s = 0.28 us
```

The DRAM transfer is 0.28 microseconds — negligible compared to the ~60us total op time. At decode sizes, compute dominates, not memory bandwidth. L1 residency only pays off for larger tensors (prefill, batch >> 1) where DRAM bandwidth becomes the limiting factor.

**Conclusion:** For single-sequence decode, DRAM interleaved is the right choice. L1 sharding would only help if (a) matmul could output HEIGHT_SHARDED natively, or (b) tensor sizes were large enough for DRAM bandwidth to matter (e.g., batch=32+).

## Q: What's the updated optimization frontier?

**A:** Four experiments, four clear answers:

| Experiment | Question | Answer | Impact |
|-----------|----------|--------|--------|
| 54 (Native RoPE) | Use `rotary_embedding_llama`? | No — wrong format for Qwen | Rotation matrix is correct and fast (0.088ms) |
| 54b (Traced Sampling) | Sampling overhead in trace? | 1.1ms (10%) | 81.4 tok/sec sampled, coherent text |
| 54c (Batch Decode) | Linear throughput scaling? | Yes — 7.4x for batch=8 | Projected ~975 tok/sec aggregate |
| 54d (L1 Residency) | L1 eliminates DRAM bottleneck? | No — DRAM isn't the bottleneck | HEIGHT_SHARDED 2.9x slower, skip it |

The clear next step is **batch decode integration** (exp 54c). It offers 7.4x throughput scaling with near-zero latency penalty and proven correctness. L1 sharding is a dead end at decode sizes, native RoPE requires format work, and sampling is already working.

```
Current state:
  Single-sequence traced decode: 131.5 tok/sec (correct)
  With sampling: ~81 tok/sec (correct, coherent text)

Next target:
  Batch=8 traced decode: ~975 tok/sec aggregate (projected)
```

---

*Experiments 54, 54b, 54c, 54d. Qwen2.5-0.5B (490M params) on Blackhole P150.*
