# TT-NN Advanced Features for LLM Inference Optimization

Research date: 2026-04-22
Target: Llama-3.1-8B-Instruct on Blackhole P150 (11x10 grid, 450 GB/s DRAM BW)
Current: 18 tok/s (52ms/tok), 64% of theoretical ceiling (28 tok/s)

## 1. On-Device Argmax / Token Selection

### ttnn.argmax
- **Available and functional.** Supports dim=-1 reduction on BFLOAT16/FLOAT32 TILE tensors.
- Returns UINT32 ROW_MAJOR output.
- **Limitation:** "Sharding is not supported" — must be interleaved DRAM/L1.
- We tried this before and got 90ms in trace. The issue is likely that argmax over 128K vocab (vocab_size=128256) on a single core is slow.

### ttnn.topk (better path)
- `ttnn.topk(input, k=1, dim=-1)` returns (values, indices).
- **Multicore execution requires width >= 8192** — our vocab is 128256, so this qualifies.
- k must be <= 64 for multicore. k=1 is perfect.
- Input must be BFLOAT16 TILE layout. Width must be < 65536... wait, 128256 > 65536.
  - **Problem:** "To enable multicore execution, width must be >= 8192 and < 65536."
  - Our vocab is 128256, which exceeds 65536. Topk might fall back to single-core.
  - **Workaround:** Could we do topk(k=1) on each half of the vocab, then compare? Two topk(64K) + one comparison. Need to benchmark.

### ttnn.max + comparison
- `ttnn.max(input, dim=-1)` to get max value, then compare to find index.
- Supports sharded L1 (Width, Height, ND sharding).
- Might be faster than argmax since max has sharded support.

### Recommendation
- **Experiment A:** Benchmark `ttnn.topk(logits, k=1)` on [1,1,1,128256] tensor inside trace. If it's < 5ms, it eliminates the PCIe readback bottleneck.
- **Experiment B:** Split logits into 4x32K chunks, topk(k=1) each, compare winners. All on-device.
- **Experiment C:** The embedding lookup for next token can also be on-device (see section below), so the entire token selection + embedding pipeline could stay on-device.

## 2. On-Device Embedding Lookup

### ttnn.embedding
- **This is a major find.** Currently we do `embed_w[token_id]` on CPU with numpy, then upload.
- `ttnn.embedding(input_indices, weight_table)` does this on-device.
- If we combine on-device argmax/topk with on-device embedding, we eliminate both:
  1. PCIe readback of logits for argmax
  2. PCIe upload of embedding vector
- **This could save 2-4ms per token** (PCIe round-trip overhead).

### Full on-device token pipeline
```
logits = decode_forward()
_, idx = ttnn.topk(logits, k=1)   # on-device argmax
embedding = ttnn.embedding(idx, embed_weight_table)  # on-device lookup
# Feed embedding back into next decode step — zero PCIe round-trips
```

## 3. Fused QKV Projection: minimal_matmul_split

### ttnn.experimental.minimal_matmul_split
- **Fused matmul + split operation.** Does `A @ B` and splits output into `chunks` tensors along last dim.
- Perfect for fused QKV: concatenate q_w, k_w, v_w into one [4096, 4096+1024+1024=6144] weight matrix, then:
  ```python
  # Instead of 3 separate matmuls:
  # q = matmul(h, q_w); k = matmul(h, k_w); v = matmul(h, v_w)
  
  # Fused version — one matmul, split output:
  # But chunks=3 requires N/3 to be tile-aligned (32-divisible)
  # q_dim=4096, k_dim=1024, v_dim=1024 — not equal chunks!
  ```
- **Problem for GQA models:** chunks must be equal-sized. Llama-8B has 32 Q heads + 8 KV heads, so Q projection is 4096 and K/V are 1024 each. We can't split [6144] into 3 equal chunks of 2048 because Q is 4096.
- **Workaround:** Use chunks=2 to fuse K+V into one matmul:
  ```python
  kv = ttnn.experimental.minimal_matmul_split(h, kv_weight, chunks=2)  # [1024, 1024] -> k, v
  q = ttnn.matmul(h, q_w)  # separate
  ```
  This reduces 3 matmuls to 2. Saves ~15% attention projection time.
- **Alternative:** Use `chunks=6` with equal 1024-dim chunks (4 for Q, 1 for K, 1 for V), then concat Q chunks. But the concat overhead might negate the benefit.

### ttnn.experimental.minimal_matmul
- High-performance matmul with built-in fused activation and bias.
- **Has fused_activation parameter** — can fuse SiLU into the gate projection:
  ```python
  # Instead of: g = silu(matmul(h, gate_w))
  # Fused: g = minimal_matmul(h, gate_w, fused_activation=ttnn.UnaryOpType.SILU)
  ```
- Also supports fused bias addition.
- Default config uses HiFi2 (not HiFi4) — might be slightly less accurate but faster.

## 4. Fused RMSNorm + Activation

### ttnn.experimental.dit_rms_norm_unary_fused
- Fuses RMSNorm + unary activation (silu, gelu) in one kernel pass.
- Also supports **fused residual add**: `RMSNorm(input + residual)`.
- **Direct application:** In our MLP block, we do:
  ```python
  h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
  g = ttnn.matmul(h2, dl["gate_w"])
  ```
  The RMSNorm and matmul are separate, but we could use the fused residual variant to combine the residual add + rms_norm:
  ```python
  # Instead of: x2 = add(x, o); h2 = rms_norm(x2)
  # Fused: h2 = dit_rms_norm_unary_fused(o, residual_input_tensor=x)
  ```
  This saves one memory write/read of the intermediate tensor.

### ttnn.fused_rms_minimal
- Fuses "pre RMS, all gather, post rms, residual add, gamma" into one op.
- Designed for multi-device, but the residual fusing is interesting.
- Constraint: input must be shape (1,1,32,M) — probably too restrictive for single-device decode.

## 5. Fused KV Cache Update

### ttnn.experimental.paged_fused_update_cache
- Updates TWO cache tensors (K and V) in parallel in a single kernel call.
- We currently do 4 separate paged_update_cache calls per layer (K_lo, V_lo, K_hi, V_hi).
- With fused update: 4 calls -> 2 calls (one for lo pair, one for hi pair).
- **Direct savings:** Halves cache update dispatch overhead per layer.

## 6. Fused RoPE: rotary_embedding_llama_fused_qk

### ttnn.experimental.rotary_embedding_llama_fused_qk
- Applies RoPE to **both Q and K in parallel** in one kernel call.
- Currently we apply RoPE to Q and K separately (2 ops each = multiply by cos, multiply rotated by sin, add).
- This could replace 6+ ops with 1 fused op.
- Input shape: `[1, batch, num_heads, head_dim]` — matches our decode layout after reshape.
- Requires precomputed cos/sin caches of shape `[1, 2*batch, 32, head_dim]` and trans_mat `[1, 2*batch, 32, 32]`.

### ttnn.experimental.rotary_embedding_llama
- Single-tensor RoPE specifically for Llama architecture.
- We already use `ttnn.experimental.rotary_embedding` — this llama variant might be optimized differently.

## 7. Decode-Optimized QKV Head Splitting

### ttnn.experimental.nlp_create_qkv_heads_decode
- Shuffles fused QKV tensor `[1, S=1, B, head_dim * (num_heads + 2*num_kv_heads)]` into separate Q, K, V heads.
- **Designed specifically for decode** (S=1, B=32 padded).
- Has `overlap_qk_coregrid` flag for parallel Q/K processing.
- **Requires sharded input** and B=32.
- Combined with `minimal_matmul_split`, this could be the full fused attention projection pipeline.

### ttnn.experimental.nlp_concat_heads_decode
- Inverse operation — concatenates attention output heads back for output projection.
- `[S=1, B=32, 32(num_heads), head_dim]` -> `[S=1, 1, B=32, num_heads * head_dim]`
- Default width-sharded output by num_heads.

## 8. Memory Layout Optimizations

### Sharding strategies
Blackhole has 11x10 = 110 compute cores with ~1.5MB L1 each.

- **HEIGHT_SHARDED:** Distribute rows across cores. Good for batch dimension parallelism.
- **WIDTH_SHARDED:** Distribute columns across cores. Good for hidden dimension parallelism.
- **BLOCK_SHARDED:** 2D distribution across a grid. Best for large matmuls.

### Key insight: L1 residency between ops
If consecutive ops use compatible sharding, data stays in L1 (no DRAM round-trip).
- `ttnn.create_sharded_memory_config(shape, core_grid, strategy)` creates the config.
- Pass as `memory_config=` to matmul/rms_norm/etc.
- **Constraint:** Shard sizes must be tile-aligned (multiple of 32).

### DRAM-sharded weights
- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` — weights sharded across DRAM banks.
- For decode (M=1, K=4096, N=4096), each DRAM bank reads its weight shard independently.
- Could improve DRAM bandwidth utilization for our weight-bound matmuls.

### Practical sharding for decode
For decode with batch=1, hidden=4096:
- Activation tensor is tiny (1x4096 = 8KB in bf16). Fits easily in one core's L1.
- Weight matrices are large (4096x4096 = 32MB). Must stream from DRAM.
- **Best strategy:** WIDTH_SHARD the activation across cores, use DRAM-sharded matmul config.
- The matmul program configs (1D multicast, DRAM sharded) are designed for exactly this pattern.

## 9. Matmul Configurations

### Fused activation in matmul
Standard `ttnn.matmul` supports `activation=` parameter:
```python
g = ttnn.matmul(h, gate_w, activation="silu")  # fused gate + silu
```
This eliminates the separate `ttnn.silu(g)` call. **Easy win.**

### MatmulMultiCoreReuseMultiCast1DProgramConfig
For our decode case (M=1 tile, large K and N):
- `mcast_in0=True`: Broadcast activation, each core has its weight shard. Good for large-N ops.
- `mcast_in0=False`: Broadcast weight blocks, each core has activation shard. Good for large-M ops.
- `fuse_batch=True`: Fuse batch dims into M.
- `fused_activation`: Can fuse SiLU/GELU into the matmul.
- `untilize_out=True`: Directly produce row-major output (saves an untilize pass if needed).

### Core grid control
`core_grid=ttnn.CoreGrid(x, y)` parameter on matmul lets you control parallelism.
- Default uses full grid (11x10 = 110 cores on Blackhole).
- For small matmuls (K_proj, V_proj with N=1024), using fewer cores might reduce dispatch overhead.

## 10. Trace Optimizations

### trace_region_size parameter
`ttnn.open_device(device_id=0, trace_region_size=N)` — pre-allocates trace region.
- Default is 0 (auto-sized). Setting explicitly might avoid reallocation.
- **Experiment:** Try large trace_region_size (e.g., 256MB) to ensure trace fits without fragmentation.

### Trace captures entire decode loop
Our trace captures one full `decode_forward()`. Key optimization:
- **Ensure all intermediate tensors are pre-allocated.** Trace replays the exact same memory allocation pattern.
- **Output tensor reuse:** Use `optional_output_tensor=` parameter to write results into pre-allocated buffers.
- This prevents trace from allocating new tensors each replay.

### Multiple traces
Could capture separate traces for different sequence length ranges (e.g., short vs long KV cache).
But the KV cache update position is dynamic (via pos_buf tensor), so this isn't necessary.

## 11. Async / Multi-Queue Execution

### num_command_queues parameter
`ttnn.open_device(device_id=0, num_command_queues=2)` — opens device with 2 command queues.
- Queue 0 (cq_id=0): primary compute queue.
- Queue 1 (cq_id=1): could be used for async data transfer while queue 0 computes.
- **Use case:** While trace executes on queue 0, upload next token's embedding on queue 1.
- **But:** With on-device embedding lookup, we might not need async transfer at all.

### enable_asynchronous_slow_dispatch
- Exists but is for "slow dispatch" (non-fast dispatch path). Likely not useful for traced execution.

## 12. Paged Attention (Already Used, Could Optimize)

### ttnn.transformer.paged_scaled_dot_product_attention_decode
- We use non-paged `scaled_dot_product_attention_decode`. The paged variant uses a page table.
- **Paged attention benefits:** Better memory utilization for variable-length sequences.
- Accepts `page_table` tensor and `sliding_window_size` parameter.
- `sliding_window_size` could limit KV reads for models that support it (Llama-3.1 uses 128K context, no sliding window by default).

## 13. Blackhole-Specific Notes

### Grid size: 11x10 = 110 cores
- Wormhole has 8x8 = 64 cores. Blackhole has 72% more cores.
- Programs designed for 8x8 will underutilize Blackhole.
- Always specify `compute_with_storage_grid_size=(11,10)` in matmul configs.

### fp32_acc_to_dest bug FIXED on Blackhole
- Wormhole had a rare fp32 accumulation bug. Blackhole fixed it.
- We can safely use `fp32_dest_acc_en=True` at HiFi4 without the rare error.

### WormholeComputeKernelConfig naming
- Despite the name, `WormholeComputeKernelConfig` works on Blackhole.
- This is the correct config class to use.

## 14. Gather Op (On-Device Index Selection)

### ttnn.gather
- Extracts values from a tensor based on an index tensor. On-device.
- Could be used for embedding lookup alternative or for selecting specific logit positions.
- Supports BFLOAT16/FLOAT32 TILE inputs, UINT16/UINT32 TILE indices.

---

## Priority-Ranked Action Items

### P0 — Likely Significant Impact (try first)

1. **On-device token pipeline (topk + embedding)**
   - Eliminate PCIe round-trip for argmax + embedding upload.
   - Benchmark `ttnn.topk(logits_128K, k=1)` in trace. If vocab > 65536, try splitting.
   - Combine with `ttnn.embedding` for fully on-device token selection.
   - Expected savings: 3-5ms/token (6-10% improvement).

2. **Fused SiLU in gate matmul**
   - Change `g = silu(matmul(h, gate_w))` to `g = matmul(h, gate_w, activation="silu")`.
   - Zero effort, guaranteed speedup (eliminates one kernel launch + memory pass).
   - Expected savings: 0.5-1ms/token per layer, 16-32ms total across 32 layers.
   - **Wait — this could be huge if the silu is actually a full memory round-trip.**

3. **Fused KV cache update (paged_fused_update_cache)**
   - Replace 4 paged_update_cache calls with 2 paged_fused_update_cache calls per layer.
   - 32 layers * 2 fewer kernel launches = 64 fewer dispatches.
   - Expected savings: 1-3ms total.

### P1 — Medium Impact (try second)

4. **Fused RoPE (rotary_embedding_llama_fused_qk)**
   - Replace separate Q and K rope applications with single fused call.
   - Reduces ~6 ops to 1 per layer. 32 layers = ~160 fewer ops.
   - Need to validate input shape requirements match our layout.

5. **Fused residual + RMSNorm (dit_rms_norm_unary_fused)**
   - Fuse `x2 = add(x, o); h2 = rms_norm(x2)` into one call.
   - Two instances per layer (attention and MLP residuals) = 64 fusions.
   - Expected savings: 1-2ms total.

6. **Fused K+V matmul (minimal_matmul_split with chunks=2)**
   - Reduce 3 QKV matmuls to 2 (Q separate, K+V fused).
   - Expected savings: ~5% of attention projection time.

### P2 — Requires More Investigation

7. **DRAM-sharded matmul configs** for weight matrices.
   - Could improve DRAM bandwidth utilization.
   - Complex to set up correctly. Need to benchmark vs default.

8. **L1 sharding between consecutive ops** to avoid DRAM round-trips.
   - RMSNorm output -> matmul input could stay in L1.
   - Requires careful shard shape planning across the entire forward pass.

9. **minimal_matmul as replacement for ttnn.matmul**
   - Newer, potentially faster matmul implementation.
   - Default HiFi2 vs our HiFi4 — need to validate accuracy tradeoff.

10. **Multi-queue async execution**
    - Less useful if on-device token pipeline eliminates PCIe transfers.
    - Could still help for prefill overlap.
