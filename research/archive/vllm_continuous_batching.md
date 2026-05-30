# vLLM PagedAttention, Continuous Batching, and Blackhole Implementation

Deep technical dive into LLM serving infrastructure and what it would take to build a vLLM-style system on Tenstorrent Blackhole.

---

## 1. PagedAttention

### The Memory Fragmentation Problem

In naive LLM serving, each request pre-allocates a contiguous KV cache for its maximum possible sequence length. A request that might generate up to 2048 tokens must reserve `2048 * n_layers * 2 * n_kv_heads * head_dim * sizeof(dtype)` bytes at the start, even if it only ends up using 50 tokens. This leads to three types of waste:

1. **Internal fragmentation**: The allocated buffer is larger than what the sequence actually uses. A 50-token response in a 2048-token buffer wastes 97.5% of the allocation.
2. **External fragmentation**: After many allocations and deallocations, free memory becomes scattered in non-contiguous chunks. Even if total free memory is sufficient, no single contiguous block is large enough for a new request.
3. **Reservation waste**: Memory reserved for future tokens in in-flight requests cannot be used by other requests, even though those tokens have not been generated yet.

vLLM (Kwon et al., 2023) reports that existing systems waste 60-80% of KV cache memory to fragmentation. This directly limits the number of concurrent requests the system can serve.

### Virtual Memory Analogy

PagedAttention borrows the core insight of OS virtual memory: decouple logical addresses from physical storage.

In an OS:
- Each process sees a contiguous virtual address space.
- The OS maps virtual pages to physical frames scattered anywhere in RAM.
- A page table stores the mapping.
- The process never knows or cares where its data physically lives.

In PagedAttention:
- Each sequence sees a contiguous logical KV cache (positions 0, 1, 2, ...).
- The system maps logical blocks to physical blocks scattered anywhere in GPU DRAM.
- A **block table** stores the mapping (one per sequence).
- The attention kernel reads the block table to gather the correct KV data.

### Blocks, Block Tables, Block Manager

**Physical block**: A fixed-size contiguous chunk of GPU memory that holds KV pairs for a fixed number of tokens (`block_size`, typically 16). Each physical block stores:
```
K: (block_size, n_kv_heads, head_dim)
V: (block_size, n_kv_heads, head_dim)
```

**Logical block**: An abstract reference to a position range within a sequence. Logical block 0 = tokens 0..15, logical block 1 = tokens 16..31, etc.

**Block table**: A per-sequence mapping from logical block index to physical block index. For a sequence with 50 tokens (block_size=16):
```
block_table[seq_id] = [phys_block_7, phys_block_23, phys_block_4, phys_block_11]
                       tokens 0-15    tokens 16-31   tokens 32-47  tokens 48-50 (partially filled)
```

**Block manager**: The central allocator that:
- Maintains a free list of physical blocks.
- Allocates a new physical block when a sequence needs more space (i.e., its last block is full and a new token is generated).
- Frees physical blocks when a sequence completes.
- Handles preemption by swapping blocks to CPU memory.

Key property: Memory is allocated **on demand**, one block at a time. A sequence that has generated 50 tokens uses exactly `ceil(50/16) = 4` blocks, not `ceil(max_seq_len/16)` blocks. The last block may be partially filled, but the waste is bounded to at most `block_size - 1` tokens per sequence.

### The Custom Attention Kernel

Standard attention implementations assume contiguous KV tensors. PagedAttention requires a custom kernel that:

1. Receives the block table as an additional input.
2. For each query token, iterates over the sequence's logical blocks.
3. For each logical block, looks up the physical block index from the block table.
4. Loads KV data from the physical block's memory address.
5. Computes attention scores and applies softmax as usual.

This is essentially a **gather operation** before standard attention. The extra indirection has minimal overhead because:
- Block tables are small (one int per block per sequence) and fit in registers/L1.
- KV data access is still coalesced within each block.
- The kernel is memory-bandwidth-bound anyway; the block table lookup is compute, not bandwidth.

### Copy-on-Write for Parallel Sampling

When multiple outputs are sampled from the same prompt (e.g., beam search, parallel sampling), the prompt's KV cache can be shared via copy-on-write:
- All beams/samples share the same physical blocks for the prompt portion.
- Their block tables point to the same physical blocks.
- When a sequence diverges (generates a new token that fills a shared block differently), only then is the block copied and the table updated.

This can reduce prompt KV memory by `n_beams`x.

### What We Have vs Full PagedAttention

**What we already have:**
- `paged_update_cache` with `update_idxs_tensor`: can write KV pairs to arbitrary positions in the cache, per-batch-element. This is the write path.
- `scaled_dot_product_attention_decode` with `cur_pos_tensor`: can attend to different sequence lengths per batch element. This is the read path with length masking.
- Per-sequence position tracking in batch decode.

**What we are missing for full PagedAttention:**
1. **Block-level indirection**: Our cache is `(batch, n_kv_heads, MAX_SEQ, head_dim)` -- contiguous per sequence. True PagedAttention would have a flat pool of physical blocks and per-sequence block tables. The attention kernel would need to gather KV data from non-contiguous blocks.
2. **Dynamic block allocation**: We pre-allocate the full `MAX_SEQ` per batch element. PagedAttention allocates blocks incrementally.
3. **Block table input to SDPA**: The TT-NN `scaled_dot_product_attention_decode` does not accept a block table argument. It assumes contiguous KV layout.
4. **Copy-on-write**: No shared block references.

**The gap is smaller than it looks.** For a single-device system with moderate batch sizes (8-32), the contiguous-per-sequence approach works well. PagedAttention's benefits are most critical at scale:
- Serving hundreds of concurrent requests where fragmentation dominates.
- Very long sequences (4K-128K) where reservation waste is enormous.
- Beam search / parallel sampling where CoW saves memory.

For our current Blackhole P150 setup with batch=8 and MAX_SEQ=256, we waste `8 * 256 * 24 * 2 * 2 * 64 * 2 bytes = ~12MB` per model on KV cache. The device likely has 8-32GB DRAM. Fragmentation is not our bottleneck today.

**If we wanted full PagedAttention on TT-NN**, we would need:
- A custom TT-NN kernel (or modification to the existing SDPA decode kernel) that accepts a block table tensor.
- A host-side block manager.
- The block table passed as a device tensor input alongside Q, K-cache, V-cache.
- The SDPA kernel would index into a flat KV pool using the block table rather than assuming `cache[batch][head][pos][dim]` layout.

This is a significant kernel-level change but architecturally clean. TT-Metalium exposes enough kernel programming capability (RISC-V cores, NOC data movement) to implement this.

---

## 2. Continuous Batching

### Static Batching vs Continuous Batching

**Static batching** (traditional):
- Collect N requests into a batch.
- Run all N requests to completion (all finish generating).
- Return results.
- Start the next batch.

Problem: sequences finish at different times. If one sequence in the batch generates 200 tokens and another generates 10, the 10-token sequence sits idle for 190 steps, wasting compute. Throughput is bounded by the slowest sequence.

**Continuous batching** (also called "iteration-level scheduling" or "in-flight batching"):
- At each decode step, the scheduler can:
  - **Insert** a new request (after prefilling its KV cache).
  - **Evict** a completed request (freeing its KV cache).
  - **Preempt** a running request (swapping its KV cache to CPU).
- The batch composition changes from step to step.
- GPU utilization stays high because finished slots are immediately filled.

### Iteration-Level vs Request-Level Scheduling

**Request-level scheduling** (static): Decisions happen at batch boundaries. All requests in a batch run together from start to finish. If a new request arrives mid-batch, it waits for the next batch.

**Iteration-level scheduling** (continuous): Decisions happen at every decode step. After each forward pass:
1. Check which sequences emitted EOS or hit max length -- remove them.
2. Check if there are waiting requests and free KV cache slots -- add them (prefill first, then join decode).
3. Execute the next decode step with the updated batch.

The scheduler runs between every decode iteration. The overhead is minimal because scheduling decisions are simple (check EOS tokens, check free memory) compared to the ~8ms decode step.

### Handling Different Sequence Lengths

In a continuous batch, sequences are at different positions in their generation. Sequence A might be at position 50, sequence B at position 200, sequence C at position 5. This requires:

1. **Per-sequence position tracking**: Each batch element has its own position counter. We already do this with `cur_pos_tensor = [pos_0, pos_1, ..., pos_B]`.

2. **Per-sequence attention masking**: SDPA must only attend to positions `0..cur_pos` for each sequence. The `cur_pos_tensor` argument handles this -- it tells the kernel how far to read in each sequence's KV cache.

3. **Per-sequence RoPE**: Each sequence needs RoPE embeddings for its current position. If all sequences have the same position (as in our current batch decode with identical prompts), a single cos/sin buffer suffices. With different positions, we need per-sequence cos/sin values.

**This is the main thing our batch decode is missing.** In experiment 56, we use the same position for all sequences (`pos = positions[0]`) and compute a single cos/sin buffer. For continuous batching, we need:
```python
# Per-sequence cos/sin: (1, 1, batch_size, head_dim)
cos_per_seq = np.stack([compute_cos(pos) for pos in positions])  # (batch, head_dim)
sin_per_seq = np.stack([compute_sin(pos) for pos in positions])  # (batch, head_dim)
```

The cos/sin buffers would be `(1, 1, batch_size, head_dim)` instead of `(1, 1, 1, head_dim)`, with each row containing the RoPE values for that sequence's position. The elementwise multiply with Q/K (which are also `(1, batch, heads, head_dim)`) would broadcast correctly along the heads dimension.

### Padding Overhead

When sequences have different lengths, two kinds of padding arise:

1. **KV cache padding**: Each sequence's cache is `MAX_SEQ` long, but only `cur_pos` entries are valid. The SDPA kernel masks out positions beyond `cur_pos`. This is already handled by `cur_pos_tensor`.

2. **Compute padding**: All sequences in the batch go through the same matmuls (QKV projection, MLP). The compute cost per step is the same regardless of sequence length because we are in decode mode -- each sequence contributes exactly one token per step. There is no sequence-length-dependent padding in decode.

Padding overhead is primarily a prefill concern. During prefill, sequences of different lengths are padded to the same length for batched matrix multiplication. Solutions:
- **Ragged batching** (FlashInfer): Concatenate all sequences without padding, use a `seq_starts` index tensor. Requires custom kernels.
- **Bucketed prefill**: Group sequences by similar length, pad within buckets.
- **Chunked prefill** (Sarathi, DeepSpeed): Split long prefills into chunks that can be interleaved with decode steps.

---

## 3. Scheduling Algorithms

### First-Come-First-Served (FCFS)

The simplest policy. Requests are served in arrival order. New requests join the batch when a slot opens. Properties:
- Fair: no starvation.
- Simple to implement.
- Can lead to head-of-line blocking: a long-running request occupies a slot that many short requests could have used.

### Shortest-Remaining-First

Prioritize requests expected to finish soonest. Problem: we don't know how many tokens a request will generate. Heuristics:
- Use max_tokens from the request config.
- Predict output length from prompt characteristics.
- Not commonly used in practice because prediction is unreliable.

### Preemption

When KV cache memory is exhausted and a new high-priority request arrives, a running request can be **preempted**:
1. **Swap**: Copy the request's KV cache blocks from GPU DRAM to CPU RAM. The request is paused.
2. **Recompute**: Discard the request's KV cache entirely. When resumed, re-run prefill to reconstruct it.

vLLM implements swapping. The scheduler maintains priority levels:
- **Running**: Currently in the decode batch.
- **Swapped**: KV cache moved to CPU. Waiting to be resumed.
- **Waiting**: Not yet started (no KV cache allocated).

When memory is tight:
1. Evict the lowest-priority running request (swap its blocks to CPU).
2. Use the freed blocks for the new request.
3. When memory frees up later, swap the evicted request back and resume.

Swapping cost: For a Llama-7B sequence at position 1024 with block_size=16:
```
KV per layer: 2 * 1024 * n_kv_heads * head_dim * sizeof(bf16)
Total: 32 layers * 2 * 1024 * 32 * 128 * 2 bytes = ~512MB
```
At PCIe Gen4 x16 bandwidth (~25 GB/s), swapping takes ~20ms. This is 2-3 decode steps, so not free but tolerable if it happens infrequently.

### Blackhole DRAM Constraints

The Blackhole P150 reportedly has 8GB of GDDR6 DRAM (unconfirmed for our exact SKU; could be up to 32GB on higher-end configs). Let's calculate KV cache capacity for Qwen2.5-0.5B:

```
Per token per layer: 2 (K+V) * n_kv_heads(2) * head_dim(64) * 2 bytes(bf16) = 512 bytes
Per token all layers: 512 * 24 = 12,288 bytes = 12KB
Per sequence at MAX_SEQ=256: 12KB * 256 = 3.07MB
Per sequence at MAX_SEQ=2048: 12KB * 2048 = 24.6MB
```

Model weights for Qwen2.5-0.5B in bf16: ~1GB.

With 8GB DRAM, after model weights, we have ~7GB for KV cache:
- At MAX_SEQ=256: 7GB / 3.07MB = ~2,340 concurrent sequences.
- At MAX_SEQ=2048: 7GB / 24.6MB = ~292 concurrent sequences.

For a bigger model like Qwen2.5-7B (n_kv_heads=4, head_dim=128, 28 layers, ~14GB weights in bf16 -- would not fit on 8GB), KV cache per token per layer = 2 * 4 * 128 * 2 = 2048 bytes = 2KB. This model would require a larger Blackhole variant or model parallelism.

For our 0.5B model, memory is abundant. The constraint is more likely to be compute throughput (how many sequences we can process per step without latency degradation) rather than memory capacity.

### Priority Scheduling

For interactive vs batch workloads:
- **Interactive** (chat): Optimize for low latency (TPOT). Small batch, fast response.
- **Batch** (offline processing): Optimize for throughput. Large batch, high GPU utilization.
- **Mixed**: Priority queues. Interactive requests preempt batch requests. Batch requests fill in unused capacity.

A practical approach: reserve a portion of the batch for interactive requests (guaranteed low latency) and fill the rest with batch requests.

---

## 4. Implementation on TT-NN

### What Changes for Continuous Batching

Our current batch decode (experiment 56) has these simplifications:

1. **All sequences start from the same prompt.** In continuous batching, sequences have different prompts and different lengths.
2. **All sequences have the same position.** The `positions` array is `[same, same, ..., same]`. In continuous batching, positions differ.
3. **RoPE cos/sin is shared.** One set of cos/sin values broadcast to all sequences. Need per-sequence values.
4. **No sequence insertion/removal.** The batch is static for the entire generation. Need to add/remove sequences mid-generation.

### Per-Sequence RoPE (The Concrete Fix)

Currently in experiment 56:
```python
pos = positions[0]  # All same
cos_full = compute_cos(pos).reshape(1, 1, 1, head_dim)
sin_full = compute_sin(pos).reshape(1, 1, 1, head_dim)
```

For continuous batching:
```python
# Each sequence at different position
cos_batch = np.stack([compute_cos(p) for p in positions]).reshape(1, 1, batch_size, head_dim)
sin_batch = np.stack([compute_sin(p) for p in positions]).reshape(1, 1, batch_size, head_dim)
```

The RoPE application `q_roped = q * cos + rotate(q) * sin` would then apply different rotations per sequence. The shapes work: Q is `(1, batch, n_q_heads, head_dim)` and cos/sin would be `(1, 1, batch, head_dim)` -- hmm, the broadcast semantics need care.

Actually, our RoPE uses the rotation matrix approach: `q_roped = q * cos + (q @ R) * sin`. Both Q and Q@R are `(1, batch, n_q_heads, head_dim)`. If cos/sin are `(1, batch, 1, head_dim)`, they broadcast across the heads dimension. But we need cos/sin to vary along the batch dimension, which means we cannot use a single `(1,1,1,head_dim)` buffer.

The fix:
```python
rope_cos_buf = to_dev_4d(np.ones((1, batch_size, 1, head_dim)))   # was (1,1,1,head_dim)
rope_sin_buf = to_dev_4d(np.zeros((1, batch_size, 1, head_dim)))  # was (1,1,1,head_dim)
```

Then Q `(1, batch, n_q_heads, head_dim)` * cos `(1, batch, 1, head_dim)` broadcasts correctly: each sequence gets its own rotation. This is a one-line shape change in the buffer allocation and the `update_buffers_batch` function.

### Masking Out Finished Sequences

Instead of removing finished sequences (which would require re-capturing the trace), we can **mask** them:

1. When a sequence emits EOS, mark it as finished in a host-side data structure.
2. On the next step, set its embedding to zeros and its position to its last position (don't advance).
3. The sequence still "runs" through the model but produces garbage -- we ignore its output.
4. When a new request arrives, overwrite the finished sequence's KV cache slot:
   - Zero out the KV cache for that batch index.
   - Run prefill for the new sequence, filling that batch index.
   - Set the position to the prefill length.
   - Mark the slot as active.

This approach keeps the trace valid because the computation graph shape is identical. The trace captures the structure of the computation (which ops, which tensor shapes), not the data. By writing new data into the same buffers, the trace replays the same ops on different data.

**The "max batch graph" approach**: Capture a trace for `batch_size = MAX_BATCH`. When fewer sequences are active, set unused slots to zero embeddings. The model runs at full batch cost regardless of occupancy, but the trace remains valid. This is what TensorRT-LLM calls "inflight batching with padding."

The trade-off: wasted compute on empty slots vs the cost of re-capturing traces. Given that our trace capture takes ~300ms and decode steps take ~8ms, it is much better to pad with zeros than to re-capture.

### Trace Capture Constraints

TT-NN trace capture records the exact sequence of operations and their memory layouts. Key constraints:

1. **Fixed tensor shapes**: All tensor dimensions must be the same during replay as during capture. Cannot change `batch_size` after trace capture.
2. **Fixed memory layout**: Tensors must live at the same addresses. Input buffers (`embed_buf`, `rope_cos_buf`, etc.) are pinned during capture.
3. **Data can change**: The contents of input buffers can be overwritten via `ttnn.copy` between trace replays. This is how we feed new tokens.

For continuous batching, this means:
- Capture the trace once at `MAX_BATCH` size.
- Use `ttnn.copy` to update embeddings, positions, and RoPE values per step.
- Mask empty slots by zeroing their inputs.
- Never need to re-capture as long as `MAX_BATCH` does not change.

### Prefill Integration

Continuous batching requires interleaving prefill and decode. When a new request arrives:

1. **Option A: Separate prefill pass.** Pause decode, run prefill for the new sequence, then resume decode. Simple but adds latency for all in-flight sequences.

2. **Option B: Chunked prefill.** Split the new sequence's prefill into chunks of `chunk_size` tokens. Process one chunk per step alongside the decode batch. The new sequence's KV cache is built up over multiple steps. This amortizes prefill cost but delays time-to-first-token.

3. **Option C: Dedicated prefill batch.** If the device has enough compute headroom, run prefill and decode on separate command queues or alternate between them. TT-NN supports multiple command queues (`cq_id=0`, `cq_id=1`).

For our setup, Option A is simplest. Prefill for Qwen2.5-0.5B takes ~50ms for a typical prompt. During that time, 6-7 decode steps are delayed. With batch=8, that is 6-7 * 8 = 48-56 tokens delayed. Acceptable for many use cases.

---

## 5. Other Serving Frameworks

### TensorRT-LLM

NVIDIA's serving framework for TensorRT-optimized models.

**KV cache management**: Uses a block-based allocator similar to PagedAttention, but tightly integrated with TensorRT's execution engine. Blocks are called "KV cache pools." The allocator is in C++ for performance.

**Inflight batching**: Their term for continuous batching. Supports both "V1" (request-level) and "inflight" (iteration-level) modes. The inflight scheduler runs a C++ event loop:
1. After each step, check for completed sequences.
2. Check for new requests in the queue.
3. Update the batch composition.
4. Execute the next step.

**Paged KV cache with block reuse**: Supports KV cache reuse for prompts that share a common prefix (system prompt reuse). Physical blocks for the shared prefix are reference-counted.

**Key difference from vLLM**: Tighter integration with CUDA graph capture (similar to our trace capture). TensorRT-LLM captures CUDA graphs for the decode path and replays them, using the same "max batch padding" approach for continuous batching within a traced/graphed context.

### SGLang

**RadixAttention**: SGLang's key innovation. Instead of per-request block tables, SGLang maintains a **radix tree** (trie) of all cached KV sequences. When a new request arrives, the system finds the longest prefix match in the tree and reuses those KV cache entries.

This is a superset of system prompt caching:
- If 100 requests all start with the same system prompt, all share the same KV cache for that prefix.
- If a multi-turn conversation has turns A -> B -> C, and a new request has turns A -> B -> D, the KV cache for A -> B is reused.

**Scheduling**: SGLang uses a simpler FCFS scheduler but achieves high throughput through aggressive prefix caching and efficient memory management.

**Chunked prefill**: SGLang implements chunked prefill to avoid large prefill stalls. Long prompts are split into chunks, and each chunk is processed alongside decode tokens.

### FlashInfer

Not a serving framework but a **kernel library** that provides the building blocks.

**Ragged tensors**: Instead of padding sequences to the same length, FlashInfer concatenates all sequences into a single 1D tensor with a `seq_starts` index array:
```
tokens:     [t0_0, t0_1, t0_2, t1_0, t1_1, t2_0, t2_1, t2_2, t2_3]
seq_starts: [0, 3, 5, 9]
```

No padding waste. Each sequence's data is contiguous within the ragged tensor, but sequences are different lengths.

**PagedAttention kernel variants**: FlashInfer provides optimized CUDA kernels for:
- Standard paged attention (block table + flat KV pool).
- Ragged attention (concatenated sequences).
- Cascade attention (multi-level KV cache for prefix sharing).

**Plan/Execute API**: FlashInfer separates attention computation into a "plan" phase (compute metadata like block indices) and an "execute" phase (run the kernel). The plan can be done on CPU while the GPU runs the previous step.

**Relevance to TT-NN**: If we wanted custom attention kernels on Blackhole, FlashInfer's design patterns (ragged tensors, plan/execute separation, block table indirection) are the right reference implementations. The plan/execute pattern maps well to TT-NN's programming model where host-side setup (tensor creation, memory config) is separated from device execution.

### Comparison Table

| Feature | vLLM | TensorRT-LLM | SGLang | Our TT-NN Setup |
|---|---|---|---|---|
| KV cache mgmt | PagedAttention (block tables) | Paged pool (C++) | RadixAttention (trie) | Contiguous per-seq |
| Continuous batching | Yes (iteration-level) | Yes ("inflight") | Yes | Not yet (static batch) |
| Prefix caching | Copy-on-write | Block reuse | Radix tree | No |
| Chunked prefill | Yes | Yes | Yes | No (full prefill) |
| Graph capture | CUDA graphs | CUDA graphs | CUDA graphs | TT-NN trace capture |
| Max batch padding | Yes (within graph) | Yes | Yes | Feasible (same approach) |

---

## 6. Metrics and Performance Comparison

### Key Metrics

**Time-to-First-Token (TTFT)**: Latency from request arrival to first generated token. Dominated by prefill time. For interactive use, target <500ms.

**Time-Per-Output-Token (TPOT)**: Average latency per token during decode. Also called "inter-token latency" or "decode latency." For interactive chat, target <100ms (10 tok/sec perceived by user).

**Throughput (tok/sec)**: Total tokens generated per second across all sequences. This is the key metric for batch/offline workloads. Equal to `batch_size / TPOT`.

**Time-to-Last-Token (TTLT)**: Total request latency = TTFT + (output_length * TPOT). End-to-end user experience.

**Goodput**: Throughput of tokens that actually get used. In speculative decoding or beam search, some generated tokens are discarded. Goodput = throughput * acceptance rate.

### Our Performance Numbers

From experiments 53e and 56, on Qwen2.5-0.5B on Blackhole P150:

| Config | Latency/step | Throughput | Notes |
|---|---|---|---|
| Batch=1 traced | 7.6ms | 132 tok/sec | Experiment 53e |
| Batch=8 non-traced | ~10ms/step | ~800 tok/sec | Experiment 56 |
| Batch=8 traced | ~8.2ms/step | ~975 tok/sec | Projected from 54c scaling |
| Prefill | ~50ms | N/A | For typical 10-20 token prompt |

### Comparison to GPU Serving

For similarly-sized models (~0.5B parameters) on GPUs:

**NVIDIA A100 (80GB)**:
- A 0.5B model is very small for an A100. Typical TPOT at batch=1: ~2-5ms.
- At batch=128+: TPOT ~10-20ms, throughput ~6,000-12,000 tok/sec.
- The A100 costs $10,000-15,000 and draws 300W.

**NVIDIA RTX 4090**:
- TPOT at batch=1 for 0.5B model: ~3-8ms.
- Smaller memory (24GB) limits batch size.
- Cost: ~$1,600, power: 450W.

**Apple M2 Ultra (MLX)**:
- TPOT for 0.5B at batch=1: ~10-20ms.
- Unified memory enables large models but bandwidth-limited.

**Tenstorrent Blackhole P150**:
- Our measured TPOT at batch=1: 7.6ms (132 tok/sec). Competitive with consumer GPUs.
- At batch=8: ~8.2ms/step for 8 tokens = 1.0ms/tok effective = 975 tok/sec projected.
- Cost: reportedly ~$300-500. Power: ~75W.
- **Price-performance could be very competitive** if batch throughput projections hold.

The key advantage of Blackhole is not raw per-token latency (GPUs win there) but **tokens per dollar per watt**. At ~$400 and 75W, getting 975 tok/sec on a 0.5B model would be remarkable efficiency.

### Scaling Considerations

For larger models (7B, 70B), the picture changes:
- 7B models may fit on a single Blackhole with quantization (INT8/INT4).
- 70B models would require multiple devices (Galaxy, multi-chip).
- The matmul-heavy compute in larger models plays to Tensix cores' strengths.
- Memory bandwidth becomes the bottleneck for decode; Blackhole's GDDR6 bandwidth (reportedly ~200-400 GB/s) is lower than A100's HBM2e (2 TB/s).

---

## 7. How Close Are We to a vLLM-Style System?

### What We Have (working)

1. Paged KV cache with per-sequence position updates (`paged_update_cache` + `update_idxs_tensor`).
2. Batched SDPA decode with per-sequence position masking (`cur_pos_tensor`).
3. Trace capture for fast decode replay (~7.6ms/step at batch=1).
4. Near-perfect batch scaling (7.4x throughput for 8x batch in single-layer test).
5. Correct text generation with greedy sampling.

### What We Need

**Tier 1 -- Minimal continuous batching (days of work):**

1. **Per-sequence RoPE.** Change cos/sin buffers from `(1,1,1,head_dim)` to `(1,batch,1,head_dim)`. Update `update_buffers_batch` to compute per-sequence positions. Test correctness with sequences at different positions. This is a small code change.

2. **Sequence masking.** When a sequence finishes (EOS or max_len), zero its embedding input and freeze its position. The slot remains "occupied" in the batch but produces ignored output. When a new request arrives, overwrite the slot.

3. **Host-side scheduler.** A simple Python loop that:
   - Checks for EOS in each batch position after each step.
   - Maintains a request queue.
   - Assigns new requests to empty slots (prefill, then join decode).
   - Tracks per-sequence state (position, output tokens, finished flag).

This gives us a functional continuous batching system without PagedAttention. Memory efficiency is lower (pre-allocated contiguous KV per slot) but sufficient for our current scale.

**Tier 2 -- Production-grade (weeks of work):**

4. **Dynamic KV cache sizing.** Instead of `MAX_SEQ=256` for all slots, allocate KV cache based on expected sequence length. Or use a smaller MAX_SEQ and handle overflow by rejecting/queuing requests.

5. **Chunked prefill.** Split long prompt prefills into chunks to avoid stalling decode for all in-flight sequences.

6. **HTTP/gRPC API server.** Accept requests over network, return streaming responses. Standard OpenAI-compatible API.

7. **Token streaming.** Return tokens as they are generated, not after full completion.

**Tier 3 -- Advanced (months of work, requires kernel development):**

8. **True PagedAttention.** Block-based KV cache with block tables. Requires modifying or writing a custom SDPA kernel for TT-Metalium.

9. **Prefix caching.** Reuse KV cache across requests with shared prefixes (system prompts).

10. **Speculative decoding.** Use a small draft model to predict multiple tokens, verify with the main model. Could leverage Blackhole's multi-core architecture.

### The Bottom Line

We are closer than it might seem. The hardest kernel-level primitives (paged cache update, batched SDPA with per-sequence positions) already work. The gap is primarily host-side orchestration:
- A scheduler that manages sequence lifecycle.
- Per-sequence RoPE (a reshape change).
- Masking for empty slots.

A basic continuous batching demo could be built in a few days on top of experiment 56. The trace capture approach (max-batch graph with masked empty slots) is exactly how TensorRT-LLM and other production systems handle this within captured CUDA graphs.
