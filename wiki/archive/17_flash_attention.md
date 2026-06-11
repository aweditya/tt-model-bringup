# Flash Attention on Blackhole

## Q: Does Blackhole/TT-NN have flash attention, and can we build a tiled version?

**A: Yes, TT-NN has a built-in FlashAttention-2 implementation** via `ttnn.transformer.scaled_dot_product_attention`. It requires 4D tensors `[b, nqh, s, dh]`, runs at 0.057 ms for seq_len=128 (faster than our manual 8-op attention at ~0.2ms), and is explicitly documented as "FlashAttention-2" in the docstring. A manual tiled (flash-style) attention works as a hybrid CPU/device approach but is slow due to round-trips; pure device tiled attention is blocked by broadcasting limitations in `ttnn.subtract`.

## Results

### Test 1: Standard Attention Memory Scaling

For single-head attention with d_k=64, bf16:

| Seq Len | Attn Matrix Size | Fits L1/core (1.5 MB)? | Fits L1 total (165 MB)? | Latency (ms) |
|---------|-----------------|------------------------|------------------------|--------------|
| 128 | 32 KB | YES | YES | 0.229 |
| 256 | 128 KB | YES | YES | 0.205 |
| 512 | 512 KB | YES | YES | 0.230 |
| 1024 | 2.0 MB | NO | YES | 0.229 |
| 2048 | 8.0 MB | NO | YES | 0.340 |

The attention matrix alone overflows per-core L1 at seq_len > 886. Latency is flat (~0.2ms) up to seq_len=1024 (dispatch-dominated), then jumps at 2048 as compute becomes significant.

### Test 2: Built-in Flash Attention in TT-NN

**TT-NN has a rich set of attention ops in `ttnn.transformer`:**

| Op | Description |
|----|-------------|
| `scaled_dot_product_attention` | FlashAttention-2 implementation, causal SDPA |
| `scaled_dot_product_attention_decode` | Flash-Decode for single-token MQA decoding |
| `chunked_scaled_dot_product_attention` | Chunked variant |
| `flash_mla_prefill` | Multi-Latent Attention (MLA) for prefill |
| `flash_multi_latent_attention_decode` | MLA for decode |
| `paged_scaled_dot_product_attention_decode` | Paged KV-cache variant |
| `ring_distributed_scaled_dot_product_attention` | Distributed/ring attention |
| `windowed_scaled_dot_product_attention` | Sliding window attention |
| `joint_scaled_dot_product_attention` | Joint attention variant |
| `attention_softmax` / `attention_softmax_` | Fused attention softmax |
| `concatenate_heads` | Multi-head output concatenation |
| `split_query_key_value_and_split_heads` | QKV split + head reshape |

**Key finding:** `ttnn.transformer.scaled_dot_product_attention` requires 4D tensors `[batch, num_heads, seq_len, d_head]`. 2D tensors fail with a shape index error.

**Benchmark at seq_len=128, d_k=64:**

| Method | Latency |
|--------|---------|
| Built-in SDPA (FlashAttention-2) | **0.057 ms** |
| Manual 8-op attention (eager) | 0.229 ms |

The built-in SDPA is **4x faster** than our manual attention pipeline -- it fuses all operations into a single kernel, eliminating dispatch overhead entirely.

### Test 3: Manual Tiled Attention

**PyTorch validation:** The online softmax tiled algorithm (flash attention) produces exact results (max error = 0.000000) against standard attention. The algorithm is correct.

**TT-NN hybrid implementation** (matmul on device, online softmax bookkeeping on CPU):
- Max error vs standard: 0.0099
- Mean error: 0.0007
- Total time: 625 ms (156 ms per block) -- extremely slow due to CPU round-trips

**Pure device tiled attention: BLOCKED.** The issue is `ttnn.subtract(S_block, m_block)` where `S_block` is `(256, 64)` and `m_block` from `ttnn.max(dim=-1)` returns `Shape([256])` -- a 1D tensor that cannot broadcast against the 2D scores tensor. TT-NN's binary ops require explicit shape matching; the broadcasting rules are stricter than NumPy/PyTorch.

### Test 4: Memory Comparison (seq_len=1024, d_model=256)

| Component | Standard | Flash (block=64) |
|-----------|----------|-------------------|
| Q | 512 KB | 512 KB |
| K (full vs block) | 512 KB | 32 KB |
| V (full vs block) | 512 KB | 32 KB |
| Scores (N^2 vs N*B) | **2.00 MB** | **128 KB** |
| Output / accum | 512 KB | 512 KB |
| Running stats | -- | 4 KB |
| **TOTAL** | **4.00 MB** | **1.19 MB** |

Flash attention uses **70% less memory**. Standard uses 3.4x more.

Critical threshold: the attention matrix alone overflows per-core L1 (1.5 MB) at **seq_len > 886**. Flash attention's block scores (N * block_size) stay within L1 for much longer sequences.

For total L1 (165 MB across 110 cores), the overflow point is seq_len=9300 -- well beyond typical use.

## Key Takeaways

1. **Don't build flash attention from scratch.** TT-NN already ships FlashAttention-2 as `ttnn.transformer.scaled_dot_product_attention`. It's 4x faster than manual attention and handles the tiling/online softmax internally.

2. **The `ttnn.transformer` module is a goldmine** for transformer inference. It has variants for decode (Flash-Decode), paged KV-cache, sliding window, ring/distributed attention, and multi-latent attention. This suggests Tenstorrent is heavily optimized for LLM serving.

3. **SDPA requires 4D tensors** `[batch, heads, seq, d_head]`. This means a JAX backend must reshape attention inputs into this format before calling the fused kernel.

4. **Pure device tiled attention is hard** due to broadcasting limitations in TT-NN binary ops. `ttnn.max(dim=-1)` returns a 1D tensor that doesn't broadcast against 2D tensors in `ttnn.subtract`. You'd need explicit reshaping/tiling of the max values.

5. **Memory matters at scale.** Per-core L1 overflows at seq_len ~886 for the full attention matrix. Flash attention keeps working by only materializing O(N*B) scores at a time. This is exactly what the built-in SDPA handles automatically.

## What This Means for JAX Backend

The JAX backend should map `jax.nn.dot_product_attention` (or the equivalent attention pattern in JaxPR) directly to `ttnn.transformer.scaled_dot_product_attention`. The built-in op:
- Is already FlashAttention-2 (memory-efficient, O(N) per block)
- Handles causal masking
- Supports configurable grid sizes via `SDPAProgramConfig`
- Has decode-specific variants for autoregressive generation
- Is 4x faster than composing individual ops

The main challenge is detecting the attention pattern in JaxPR (matmul + scale + softmax + matmul) and fusing it into a single SDPA call with the right 4D tensor layout.
