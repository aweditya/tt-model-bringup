# Building a vLLM-like Continuous Batching Server for Tenstorrent Blackhole

Research for Stanford CS440LX. Covers the architecture of production LLM serving systems and a concrete plan to build one on Tenstorrent Blackhole.

---

## 1. vLLM Architecture Overview

vLLM (Kwon et al., 2023) is the dominant open-source LLM serving engine. Its architecture has three core innovations that work together: PagedAttention for memory management, continuous batching for scheduling, and a block-based KV cache manager that ties the two together.

### 1.1 System Architecture

vLLM's V1 engine (default since mid-2025) uses a multiprocessing architecture:

```
                    +--------------------+
                    |   API Server       |  (FastAPI, OpenAI-compatible)
                    |   - tokenization   |
                    |   - detokenization |
                    |   - streaming      |
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |   AsyncLLM         |  (orchestration layer)
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |   EngineCore       |  (isolated process)
                    |   - Scheduler      |
                    |   - KV Cache Mgr   |
                    |   - Model Executor |
                    +--------------------+
```

The V1 refactor eliminated the distinction between "prefill" and "decode" phases at the scheduler level. Instead, scheduling decisions are represented as a simple dictionary: `{request_id: num_tokens_to_process}`. This unification enables chunked prefill naturally -- a prefill request simply processes N tokens per step rather than all tokens at once.

The EngineCore runs in an isolated process so CPU-intensive tasks (tokenization, detokenization, multimodal input processing) can overlap with GPU execution. This is critical at high concurrency where CPU overhead becomes the bottleneck.

### 1.2 PagedAttention

PagedAttention borrows the core insight of OS virtual memory: decouple logical addresses from physical storage.

**The problem it solves**: Naive KV cache allocation pre-reserves `max_seq_len` slots per request. A 50-token response in a 2048-token buffer wastes 97.5% of memory. vLLM reports that existing systems waste 60-80% of KV cache memory to fragmentation.

**How it works**:
- KV cache is divided into fixed-size **physical blocks** (typically 16 tokens each)
- Each request has a **block table** mapping logical block indices to physical block indices
- Blocks are allocated on demand as sequences grow
- A **block manager** maintains a free list, handles allocation/deallocation
- The attention kernel receives the block table as input and gathers KV data from non-contiguous physical blocks

**Key properties**:
- Memory waste bounded to at most `block_size - 1` tokens per sequence (last block padding)
- Copy-on-write for parallel sampling (beam search, multiple completions)
- Enables memory sharing across requests with common prefixes

### 1.3 KV Cache Manager

The KV cache manager is the central resource allocator:

```
Block Manager
  |-- free_block_queue: pool of available physical blocks
  |-- block_tables: per-request mapping (logical -> physical)
  |-- ref_counts: per-block reference count (for CoW)
  |
  |-- allocate(request) -> assigns blocks as sequence grows
  |-- free(request) -> returns blocks to free queue
  |-- swap_out(request) -> copies blocks GPU->CPU
  |-- swap_in(request) -> copies blocks CPU->GPU
```

At each decode step:
1. Check if the current token fills the last block -- if so, allocate a new physical block
2. If no free blocks available, trigger preemption (swap lowest-priority request to CPU)
3. When a request completes, free all its blocks immediately

### 1.4 Scheduler

The scheduler runs between every decode iteration. vLLM V1's scheduler is simpler than V0:

```python
# Simplified V1 scheduling logic
def schedule():
    budget = max_num_batched_tokens
    scheduled = {}
    
    # 1. Schedule all running (decode) requests first
    for req in running_requests:
        scheduled[req.id] = 1  # one new token each
        budget -= 1
    
    # 2. Fill remaining budget with waiting (prefill) requests
    for req in waiting_queue:
        tokens_to_process = min(req.remaining_prefill, budget)
        if tokens_to_process > 0:
            scheduled[req.id] = tokens_to_process
            budget -= tokens_to_process
    
    return scheduled
```

Decode requests are always prioritized over new prefills. This keeps inter-token latency (ITL) low for in-flight requests. New prefills fill remaining compute budget, and long prefills are chunked across multiple steps.

---

## 2. Continuous Batching Fundamentals

### 2.1 The Orca Paper (OSDI 2022)

The Orca paper (Yu et al., 2022) introduced **iteration-level scheduling** for transformer inference. The key insight:

**Static batching**: Collect N requests, run all to completion, return results, start next batch. If one sequence generates 200 tokens and another generates 10, the 10-token sequence sits idle for 190 steps. Throughput is bounded by the slowest request.

**Iteration-level scheduling**: At each decode step, the scheduler can insert new requests or evict completed ones. The batch composition changes from step to step. GPU utilization stays high because finished slots are immediately filled.

Orca demonstrated 36.9x throughput improvement over NVIDIA FasterTransformer on GPT-3 175B at the same latency level.

### 2.2 Selective Batching

Orca also introduced **selective batching**: not all operations in a transformer can be batched the same way.

- **Attention**: Sequence-length-dependent. Each sequence attends to its own KV cache up to its current position. Cannot batch sequences of different lengths without masking or ragged tensors.
- **MLP/FFN**: Position-independent. Each token is processed identically regardless of sequence length. Can batch all tokens together trivially.

In decode mode (one token per sequence), this distinction is less critical because all sequences contribute exactly one token. But during prefill, selective batching matters: sequences with different prompt lengths need different attention masks but can share MLP computation.

### 2.3 Chunked Prefill

Chunked prefill (from Sarathi-Serve, DeepSpeed-FastGen, and adopted by vLLM/SGLang) solves the problem of long prefills stalling decode:

**Without chunked prefill**: A 4096-token prompt blocks all decode requests for the entire prefill duration (~200ms for a 7B model). All in-flight sequences experience a latency spike.

**With chunked prefill**: The long prompt is split into chunks (e.g., 512 tokens each). Each step processes one chunk alongside the decode batch. The new request's KV cache is built incrementally over 8 steps instead of 1. Time-to-first-token increases slightly, but decode latency for existing requests stays stable.

vLLM V1 implements this natively because the scheduler simply specifies "process N tokens for each request" -- a prefill request with 4096 tokens might be scheduled as 512 tokens per step for 8 steps.

### 2.4 Preemption and Swapping

When KV cache memory is exhausted:

1. **Swap**: Copy a running request's KV blocks from GPU to CPU. Pause the request. When memory frees up, swap back and resume.
2. **Recompute**: Discard the KV cache entirely. When resumed, re-run prefill from scratch. Cheaper if the sequence is short.

vLLM uses swapping by default. The scheduler maintains three priority queues: **running** (in decode batch), **swapped** (KV on CPU), **waiting** (not yet started).

Swapping cost for a Llama-7B sequence at position 1024:
```
32 layers * 2 (K+V) * 1024 tokens * 32 heads * 128 dim * 2 bytes = ~512 MB
PCIe Gen4 x16 bandwidth: ~25 GB/s
Swap time: ~20ms (2-3 decode steps)
```

Tolerable if infrequent. The scheduler avoids swapping by limiting admission -- better to queue a new request than to swap an in-flight one.

---

## 3. Our Current Batch Demo vs. a Real Server

### 3.1 What We Have (Experiment 65 + demos/batch_serving.py)

Our continuous batching prototype on Qwen2.5-0.5B achieves:

| Metric | Value |
|--------|-------|
| Decode throughput | 1,042 tok/sec (batch=8) |
| Step latency | ~7.7ms per batch step |
| Slot management | Position=-1 masks inactive slots |
| Request cycling | 24 requests through 8 slots |
| Trace capture | Single trace replayed for all steps |
| KV cache | Contiguous per-sequence, paged_update_cache |
| RoPE | Per-sequence cos/sin buffers |
| Sampling | Greedy (argmax) |

Key mechanisms already working:
- **Per-sequence position tracking**: `cur_pos_tensor` with different positions per batch element
- **Per-sequence RoPE**: cos/sin buffers shaped `(1, batch, 1, head_dim)` with per-sequence rotations
- **Slot masking**: position=-1 skips SDPA compute for empty slots
- **Dynamic slot reuse**: when a sequence hits EOS or max_tokens, its slot is immediately reassigned
- **Prefill into arbitrary slot**: `fill_cache_for_user_` writes KV cache for a single batch index

### 3.2 What a Real Server Needs (The Gap)

| Feature | Our Demo | Production Server |
|---------|----------|-------------------|
| Request intake | Hardcoded prompt list | HTTP/gRPC endpoint |
| Response delivery | Print at end | Token streaming (SSE/WebSocket) |
| Concurrency | Sequential script | Async event loop |
| Tokenization | In main loop | Separate thread/process |
| Prefill strategy | Blocks all decode | Chunked or async prefill |
| Error handling | None | Timeouts, retries, backpressure |
| Memory management | Fixed MAX_SEQ=256 | Dynamic, with preemption |
| API compatibility | None | OpenAI-compatible endpoints |
| Monitoring | Print statements | Metrics (Prometheus, etc.) |
| Multi-model | Single model | Model routing |

### 3.3 What We Do NOT Need to Change

The core decode engine is already production-capable:
- Trace capture + replay is the right architecture (equivalent to CUDA graphs)
- Max-batch padding with slot masking is exactly what TensorRT-LLM does
- Paged KV cache updates work correctly
- Per-sequence position and RoPE handling is correct
- The decode kernel runs at near-optimal throughput

The gap is entirely **host-side orchestration**, not device-side compute.

---

## 4. Tenstorrent-Specific Considerations

### 4.1 Metal Traces and Static Shapes

**The constraint**: TT-NN trace capture records the exact sequence of operations, tensor shapes, and memory layouts. During replay, all tensor dimensions must match capture time. You cannot change batch_size after capture.

**The solution (already implemented)**: Capture the trace once at `MAX_BATCH` size. Use `ttnn.copy` to update input buffer contents between replays. Mask empty slots with position=-1 (zeroes in SDPA) and zero embeddings. The model runs at full batch cost regardless of occupancy.

This is identical to how CUDA graph capture works in TensorRT-LLM and vLLM. Both use "max batch padding" -- capture a graph for the maximum batch size and pad smaller batches with dummy tokens.

**Multiple trace variants**: For significant batch size ranges, we could capture traces at multiple batch sizes (e.g., 1, 4, 8, 16, 32) and select the smallest trace that fits the current active count. This reduces wasted compute when occupancy is low. TensorRT-LLM does exactly this with CUDA graph "buckets."

**The cost of padding**: At batch=8 with 3 active sequences, we waste 5/8 = 62.5% of compute. At batch=32 with 3 active, we waste 90.6%. The trade-off: re-capturing a trace takes ~300ms (equivalent to ~40 decode steps). If low occupancy persists for more than 40 steps, trace switching saves compute. In practice, a well-utilized server rarely drops below 50% occupancy.

### 4.2 PCIe Bottleneck for Token Readback

Every decode step requires reading the output logits from device to host for sampling:

```
Logits tensor: batch_size * vocab_size * sizeof(bf16)
  batch=8:   8 * 151,936 * 2 = 2.43 MB
  batch=32:  32 * 151,936 * 2 = 9.73 MB
  batch=128: 128 * 151,936 * 2 = 38.9 MB

PCIe Gen4 x16: ~25 GB/s theoretical, ~15-20 GB/s practical
Transfer time:
  batch=8:   ~0.15 ms (negligible vs 7.7ms decode)
  batch=32:  ~0.6 ms (~8% of decode step)
  batch=128: ~2.4 ms (~25% of decode step)
```

The Blackhole P150 connects via PCIe Gen5 x16 (reportedly ~50 GB/s theoretical), which halves these numbers. At batch=8, PCIe readback is not a concern. At batch=128+, it becomes significant.

**Mitigation strategies**:
1. **On-device sampling**: Run argmax/top-k on device, return only token IDs (batch * 4 bytes instead of batch * vocab * 2 bytes). TT-NN supports `ttnn.argmax`. This reduces readback by 75,000x.
2. **Async readback**: Use a separate command queue (`cq_id=1`) for D2H transfer while the next decode step runs on `cq_id=0`. Overlaps compute and transfer.
3. **Partial logit readback**: Only read the top-K logits if doing top-K sampling. Requires on-device top-K kernel.

We already measured in experiment 81 that `from_dev` (the readback call) adds ~3.9ms constant overhead. Moving sampling on-device is the single biggest optimization for server throughput.

### 4.3 Paged KV Cache (Already Supported)

Our paged KV cache support via `ttnn.experimental.paged_update_cache` with `update_idxs_tensor` is the write path. Combined with `scaled_dot_product_attention_decode` and `cur_pos_tensor` for the read path, we have the essential primitives.

What we have is "contiguous paged" -- each sequence has a contiguous KV buffer, but we can write to arbitrary positions within it. This is sufficient for:
- Per-sequence position tracking (different sequences at different positions)
- Slot reuse (zero out a slot's cache, prefill new sequence)
- Correct attention masking (SDPA reads only up to cur_pos)

What we lack is "block-level paged" -- a flat pool of physical blocks with block tables for indirection. The gap matters at scale (hundreds of concurrent sequences, very long contexts) but not for our current setup with batch=8-32 and MAX_SEQ=256.

**Memory math for Qwen2.5-0.5B on Blackhole P150 (32 GB DRAM)**:
```
Model weights (bf16):     ~1 GB
KV cache per token:       24 layers * 2 (K+V) * 2 heads * 64 dim * 2 bytes = 12 KB
KV cache per seq (256):   12 KB * 256 = 3.07 MB
Max concurrent sequences: (32 GB - 1 GB) / 3.07 MB = ~10,000 sequences

At batch=32, MAX_SEQ=256:  32 * 3.07 MB = 98 MB (trivial)
At batch=128, MAX_SEQ=2048: 128 * 24.6 MB = 3.15 GB (fits easily)
```

Memory is not our bottleneck for the 0.5B model. For larger models (Llama-3.2-3B at ~6.6 GB weights, Llama-3.1-8B at ~16 GB weights), it becomes tighter, but the 0.5B model leaves ample room.

### 4.4 Multi-User Serving via HTTP

For a real server, we need to accept requests over HTTP and stream responses back. The standard approach:

```
                 HTTP clients
                     |
            +--------v--------+
            |  FastAPI Server  |  (async Python)
            |  /v1/completions |
            |  /v1/chat        |
            +--------+--------+
                     |
            +--------v--------+
            |  Request Queue   |  (asyncio.Queue)
            +--------+--------+
                     |
            +--------v--------+
            |  Scheduler       |  (Python, runs between decode steps)
            |  - slot assignment
            |  - prefill trigger
            |  - completion check
            +--------+--------+
                     |
            +--------v--------+
            |  Decode Engine   |  (trace replay loop)
            |  - ttnn.execute_trace
            |  - buffer updates
            +--------+--------+
```

The decode engine runs in a tight loop:
```python
while True:
    # 1. Check for completed sequences, notify waiting clients
    for slot in slots:
        if slot.finished:
            slot.response_future.set_result(slot.output)
            scheduler.free_slot(slot)
    
    # 2. Admit new requests from queue
    while scheduler.has_free_slot() and not request_queue.empty():
        req = request_queue.get_nowait()
        slot = scheduler.assign_slot(req)
        prefill(req.tokens, slot.idx)
    
    # 3. Update buffers and run decode step
    update_buffers(slots)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    
    # 4. Read logits, sample, advance positions
    logits = read_logits()
    for slot in active_slots:
        token = sample(logits[slot.idx])
        slot.advance(token)
        if streaming:
            slot.stream_queue.put(token)
```

**Key design decision**: The decode loop must run synchronously on the device -- we cannot have multiple threads calling `ttnn.execute_trace` concurrently. The HTTP server runs asynchronously in a separate thread, feeding requests into a queue. The decode loop drains the queue between steps.

This is exactly how vLLM's EngineCore works: an isolated event loop that runs the scheduler and model executor, communicating with the API server via IPC.

---

## 5. Implementation Plan

### Phase 1: Async Server Shell (1-2 days)

Build the HTTP layer around our existing decode engine.

**Deliverables**:
- FastAPI server with OpenAI-compatible `/v1/completions` endpoint
- `asyncio.Queue` for request intake
- Background thread running the decode loop
- Server-Sent Events (SSE) for token streaming
- Basic health check and metrics endpoint

**Architecture**:
```python
# server.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio, threading

app = FastAPI()
request_queue = asyncio.Queue()

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    future = asyncio.Future()
    await request_queue.put((req, future))
    if req.stream:
        return StreamingResponse(stream_tokens(future), media_type="text/event-stream")
    return await future

def decode_loop():
    """Runs in background thread. Owns all TT-NN state."""
    # ... existing experiment 65 logic, modified to pull from queue
    pass

threading.Thread(target=decode_loop, daemon=True).start()
```

**What does NOT change**: The entire decode engine (trace capture, buffer updates, KV cache, RoPE) stays identical to experiment 65.

### Phase 2: On-Device Sampling (1 day)

Eliminate the PCIe bottleneck by sampling on device.

**Current**: Read full logits tensor (2.43 MB at batch=8), sample on CPU.
**Target**: Run `ttnn.argmax` on device, read only token IDs (32 bytes at batch=8).

```python
# Before (3.9ms readback)
logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
tokens = np.argmax(logits[0, 0], axis=-1)

# After (~0.01ms readback)
token_ids_tt = ttnn.argmax(logits_ref, dim=-1)
tokens = ttnn.to_torch(token_ids_tt).numpy().flatten()[:batch_size]
```

This alone could improve effective throughput by 30-50% at batch=8 (removing 3.9ms from a 7.7ms step).

**For non-greedy sampling** (temperature, top-k, top-p): These are harder to do on device. Options:
1. Read full logits and sample on CPU (current approach, acceptable for batch<=32)
2. Implement on-device top-k, then read only k candidates per sequence
3. Use ttnn.softmax + ttnn.multinomial if available

### Phase 3: Chunked Prefill (2-3 days)

Current prefill blocks all decode for ~50ms per new request. With batch=8 and request cycling, this means ~50ms stall every time a slot opens.

**Approach**: Split prefill into chunks of 64-128 tokens. Process one chunk per decode step.

```python
class PrefillState:
    def __init__(self, tokens, slot_idx):
        self.tokens = tokens
        self.slot_idx = slot_idx
        self.offset = 0
        self.chunk_size = 128
    
    def next_chunk(self):
        end = min(self.offset + self.chunk_size, len(self.tokens))
        chunk = self.tokens[self.offset:end]
        self.offset = end
        return chunk
    
    @property
    def done(self):
        return self.offset >= len(self.tokens)
```

**Complication**: Our current prefill runs on a CPU path (reads tensors back to CPU for attention). To interleave with traced decode, we need either:
1. A separate prefill trace (captured for the chunk size)
2. CPU-side prefill in a separate thread while decode runs (requires careful synchronization)
3. Accept the stall for now and optimize later (pragmatic for the 0.5B model where prefill is fast)

For the 0.5B model with typical 10-20 token prompts, prefill takes ~50ms. At batch=8, a slot opens roughly every 7-8 seconds (60 tokens * 7.7ms/step / 8 slots). A 50ms stall every 7 seconds is <1% overhead. **Chunked prefill is a nice-to-have, not a blocker.**

### Phase 4: Multi-Trace Batch Buckets (1 day)

Capture traces at multiple batch sizes to reduce waste at low occupancy.

```python
traces = {}
for bs in [1, 4, 8, 16, 32]:
    # Reconfigure KV caches and buffers for batch size bs
    setup_for_batch(bs)
    trace_id = capture_trace(bs)
    traces[bs] = trace_id

def select_trace(active_count):
    """Pick smallest trace that fits."""
    for bs in sorted(traces.keys()):
        if bs >= active_count:
            return traces[bs], bs
    return traces[max(traces.keys())], max(traces.keys())
```

**Trade-off**: Each trace uses device memory (~50-100 MB for the 0.5B model). With 5 batch sizes, that is 250-500 MB. Worthwhile if the server has variable load.

### Phase 5: Production Hardening (2-3 days)

- **Timeouts**: Kill requests that exceed max_tokens or wall-clock limit
- **Backpressure**: Return 429 when request queue exceeds threshold
- **Graceful shutdown**: Drain in-flight requests before stopping
- **Logging**: Structured logging with request IDs, latencies, token counts
- **Metrics**: Prometheus endpoint with throughput, latency percentiles, queue depth, slot utilization
- **Stop sequences**: Support custom stop strings, not just EOS
- **Chat template**: Apply Qwen2.5 chat template for /v1/chat/completions

### Stretch Goals (not in initial scope)

- **Speculative decoding**: Use a smaller draft model (distilled 0.1B?) to predict multiple tokens, verify with the 0.5B model. Blackhole's architecture could potentially run both models on different Tensix core groups.
- **Prefix caching**: Cache KV for common system prompts. Our contiguous KV layout makes this straightforward -- copy the cached prefix into the new slot's KV buffer.
- **Multi-model**: Load multiple models (0.5B, 1B, 3B) and route requests based on complexity or user preference. Each model has its own traces and KV caches.
- **Disaggregated prefill**: Run prefill on CPU (or a second Blackhole device) while the primary device handles decode exclusively. The two Blackhole devices on our host could serve this purpose.

---

## 6. Performance Targets

### 6.1 Throughput Targets

Based on our measured decode performance:

| Config | Step Latency | Throughput | Status |
|--------|-------------|------------|--------|
| batch=8, current | 7.7ms | 1,042 tok/s | Achieved (exp 65) |
| batch=8, on-device sampling | ~4.0ms | ~2,000 tok/s | Projected (remove 3.9ms readback) |
| batch=16, current | ~8.5ms | ~1,900 tok/s | Achieved (demos/batch_serving.py) |
| batch=32, current | ~9.7ms | ~3,300 tok/s | Achieved (demos/batch_serving.py) |
| batch=32, on-device sampling | ~5.8ms | ~5,500 tok/s | Projected |
| batch=64, current | ~13.2ms | ~4,850 tok/s | Achieved (exp 59) |

**Server throughput target**: 2,000-3,000 tok/s sustained at batch=8-16 with on-device sampling. This is the "minimum viable" for a demo server.

**Stretch target**: 5,000+ tok/s at batch=32+ with on-device sampling and multi-trace buckets.

### 6.2 Latency Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| TTFT (time to first token) | <100ms | Prefill ~50ms + 1 decode step |
| ITL (inter-token latency) | <15ms | Single decode step at batch=8 |
| TTLT for 60 tokens | <1 second | 50ms prefill + 60 * 7.7ms decode |
| P99 ITL | <25ms | Allow for occasional prefill stall |

For interactive chat, the key metric is perceived speed: users notice delays above ~100ms between tokens. At 7.7ms/step, even batch=8 delivers tokens faster than humans can read.

### 6.3 Comparison to GPU Serving

| System | Model | Throughput | Cost | tok/$/watt |
|--------|-------|-----------|------|------------|
| vLLM on A100 (80GB) | 0.5B, batch=128 | ~10,000 tok/s | $15,000, 300W | 0.0022 |
| vLLM on RTX 4090 | 0.5B, batch=32 | ~5,000 tok/s | $1,600, 450W | 0.0069 |
| TensorRT-LLM on A100 | 0.5B, batch=128 | ~15,000 tok/s | $15,000, 300W | 0.0033 |
| **Our Blackhole P150** | 0.5B, batch=8 | 1,042 tok/s | ~$400, 75W | 0.035 |
| **Projected (on-dev sample)** | 0.5B, batch=32 | ~5,500 tok/s | ~$400, 75W | 0.183 |

The raw throughput gap is real -- an A100 has 2 TB/s HBM bandwidth vs Blackhole's 512 GB/s GDDR6. But the price-performance story is compelling:
- Blackhole P150 at ~$400, 75W vs A100 at $15,000, 300W
- Even at 1,042 tok/s, our tok/$/watt is **10-16x better** than an A100
- With on-device sampling at batch=32, projected tok/$/watt is **50-80x better**

The comparison is even more favorable for the 0.5B model because the A100's massive compute is underutilized -- small models are memory-bandwidth-bound, and the A100 is paying for compute it cannot use.

### 6.4 Scaling to Larger Models

| Model | Weights (bf16) | KV/token | Batch=8 projected | Fits on P150? |
|-------|---------------|----------|-------------------|---------------|
| Qwen2.5-0.5B | 1 GB | 12 KB | 1,042 tok/s | Yes (32 GB) |
| Llama-3.2-1B | 2.5 GB | 16 KB | ~500 tok/s | Yes |
| Llama-3.2-3B | 6.6 GB | 24 KB | ~270 tok/s | Yes |
| Llama-3.1-8B | 16 GB | 64 KB | ~100 tok/s | Tight (int8 needed) |
| Qwen2.5-7B | 14 GB | 16 KB | ~120 tok/s | Tight |

For models that fit, the server architecture is identical -- only the model weights and layer count change. The 0.5B model is the best target for a server demo because memory is abundant and throughput is highest.

---

## 7. Comparison of Serving Frameworks

### 7.1 Framework Architecture Summary

| Feature | vLLM | TensorRT-LLM | SGLang | Our TT-NN System |
|---------|------|-------------|--------|-------------------|
| **KV cache** | PagedAttention (block tables) | Paged pool (C++) | RadixAttention (trie) | Contiguous per-seq |
| **Continuous batching** | Yes (iteration-level) | Yes ("inflight") | Yes | Yes (exp 65) |
| **Graph capture** | CUDA graphs | CUDA graphs | CUDA graphs | TT-NN trace capture |
| **Max batch padding** | Yes | Yes (with buckets) | Yes | Yes (position=-1) |
| **Chunked prefill** | Yes (V1 native) | Yes | Yes | Not yet |
| **Prefix caching** | Copy-on-write | Block reuse | Radix tree (best) | Not yet |
| **On-device sampling** | Yes (CUDA) | Yes | Yes | Planned |
| **API server** | FastAPI (built-in) | Triton Inference Server | FastAPI | Planned |
| **Streaming** | SSE | gRPC stream | SSE | Planned |
| **Multi-GPU** | Pipeline + tensor parallel | Pipeline + tensor parallel | Data + tensor parallel | Single device |

### 7.2 What We Can Learn from Each

**From vLLM**: The V1 scheduler simplification (uniform token counting instead of prefill/decode distinction) is elegant and applicable. Their multiprocessing architecture (EngineCore in isolated process) is the right pattern for separating CPU work from device work.

**From TensorRT-LLM**: CUDA graph bucketing (capture at multiple batch sizes, select the smallest that fits) directly applies to our trace capture. Their approach to continuous batching within captured graphs matches what we already do.

**From SGLang**: RadixAttention for prefix caching is the most advanced KV reuse system. If we build a chat server where many users share the same system prompt, prefix caching could save significant prefill compute. SGLang's cache-aware scheduling (prioritize requests that hit the cache) is a smart optimization.

**From FlashInfer**: The plan/execute API pattern (CPU plans the attention layout while GPU executes the previous step) maps well to TT-NN's host/device split. We could compute buffer updates on CPU while the trace replays on device.

---

## 8. Open Questions

1. **ttnn.argmax reliability**: Does `ttnn.argmax` on a `(1, 1, batch, vocab_size)` tensor return correct per-sequence results? Need to validate before relying on it for on-device sampling.

2. **Trace memory overhead**: How much device memory does each captured trace consume? This determines how many batch-size buckets we can afford.

3. **Concurrent command queues**: Can we run prefill on `cq_id=1` while decode trace replays on `cq_id=0`? TT-NN supports multiple command queues, but we have not tested concurrent execution.

4. **Position=-1 correctness**: We use position=-1 to mask empty slots in SDPA. Does this truly zero the contribution, or does it produce NaN/garbage that leaks into the batch? Need a rigorous test with mixed active/inactive slots.

5. **FastAPI + TT-NN threading**: TT-NN device operations are not thread-safe. The decode loop must run in a single thread. Can FastAPI's async handlers safely enqueue requests to a threading.Queue consumed by the decode thread?

6. **Warmup latency**: The first request after server start requires model loading (~2s), trace capture (~300ms), and warmup (~100ms). Can we pre-warm on startup so the first real request is fast?

---

## Sources

- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [vLLM V1 Engine Architecture (GitHub Issue #8779)](https://github.com/vllm-project/vllm/issues/8779)
- [vLLM V1 Alpha: A Major Upgrade (Red Hat)](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture)
- [vLLM Roadmap Q1 2026](https://github.com/vllm-project/vllm/issues/32455)
- [PagedAttention & vLLM (Woosuk Kwon lecture slides)](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-22-vLLM_woosuk_kwon-1f34697dbb1a1fb5b798daf6eff14b67.pdf)
- [Paged Attention — vLLM Docs](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Orca Paper PDF](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [Iteration Batching (Friendli AI Blog)](https://friendli.ai/blog/llm-iteration-batching)
- [LLM Inference: Continuous Batching and PagedAttention](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/)
- [Chunked Prefills and Decode Maximal Batching](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)
- [vLLM Chunked Prefill RFC (GitHub Issue #3130)](https://github.com/vllm-project/vllm/issues/3130)
- [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)
- [SGLang: Fast and Expressive LLM Inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [Efficient LLM Inference with SGLang (Ying Sheng, 2025 lecture)](https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-25-sglang-72edc5043338f59db34d47e5b96ac870.pdf)
- [Tenstorrent Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Blackhole Product Page](https://tenstorrent.com/hardware/blackhole)
- [Dissecting Tenstorrent Blackhole via Microbenchmarking (ASPLOS 2025)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Mind the Memory Gap: GPU Bottlenecks in Large-Batch LLM Inference](https://arxiv.org/html/2503.08311v2)
- [MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks](https://arxiv.org/html/2512.16056)
- [Achieve 23x LLM Inference Throughput (Anyscale Blog)](https://www.anyscale.com/blog/continuous-batching-llm-inference)
