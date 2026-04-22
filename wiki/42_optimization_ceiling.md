# Wiki 42: The Optimization Ceiling — Where Does Time Go?

## Question
We've hit 7.1ms/tok at batch=1 and it won't go lower. What's the actual bottleneck?

## Experiments 58-63b Summary

### What We Tried (and What Didn't Help)

| Experiment | Change | Result | Why No Help |
|-----------|--------|--------|-------------|
| 58 | Interleaved ↔ half RoPE equivalence | cosine=1.0 (math proof) | Foundation for native RoPE |
| 59 | Batch + bf8 MLP combined | 4,867 tok/sec (b=64) | 2% over bf16 — compute-bound |
| 59c | Native `rotary_embedding` | 2.6x faster per-op | **Actual speedup** |
| 60 | **Native RoPE in traced decode** | **7.1ms (140 tok/sec)** | **5% win — new record** |
| 61 | Per-layer bf8 ablation | All 24 layers safe | No sensitive layers found |
| 62 | Full bf8 + native RoPE | 7.1ms (same) | NOT bandwidth-bound |
| 62b | Full bf8 batch=32 | 9.4ms (same) | NOT bandwidth-bound even at b=32 |
| 63 | HiFi2 MLP (full recompute) | 29% faster | Compute savings in eager mode |
| 63b | HiFi2 MLP (traced) | 7.1ms (same) | Compute not bottleneck in trace |

### The 7.1ms Breakdown (estimated)

```
SDPA decode (24 layers × reading KV cache):     ~3.0ms
Matmuls (168 per forward):                      ~2.5ms
RoPE + cache update + reshape:                   ~0.8ms
Trace execution overhead:                        ~0.5ms
PCIe readback (151K logits):                     ~0.3ms
                                                -------
Total:                                           ~7.1ms
```

### Key Insight: The Bottleneck Shifts with Batch Size

At **batch=1**: SDPA is the bottleneck (reading all MAX_SEQ positions per head per layer).
At **batch=64**: Everything scales ~linearly, so 13.2ms/64 = 0.21ms/tok amortized.

The per-sequence overhead doesn't change, but you process more sequences per step.

## What WOULD Help Single-Sequence Latency

1. **Reduce MAX_SEQ** — smaller KV cache = faster SDPA (but shorter context)
2. **Multi-query attention** — fewer KV heads = less cache to read (model architecture change)
3. **Paged attention with block-sparse** — skip empty/distant blocks
4. **Custom SDPA kernel** — fused attention with early termination

## What DOES Help Throughput

1. **Batch scaling** — already proven: 1,050 → 4,867 tok/sec (b=1→64)
2. **Continuous batching** — next experiment: keep all slots full
3. **Speculative decoding** — generate 3-4 tokens per verification step

## Updated Performance Timeline

```
exp 41:     582ms/tok    1.7 tok/s    Full recompute, default config
exp 47:      54ms/tok   18.4 tok/s    HiFi4 generation
exp 49:      35ms/tok   28.6 tok/s    KV-cached decode
exp 51c:     21ms/tok   46.6 tok/s    Fully on-device decode
exp 53e:    7.6ms/tok  131.5 tok/s    Traced + paged KV
exp 57c:    7.4ms/tok  134.3 tok/s    + bf8 MLP weights
exp 60:     7.1ms/tok  140.4 tok/s    + native RoPE ← CEILING
exp 56 b=8: 7.6ms/step 1,050 tok/s    Batch scaling begins
exp 59 b=64:13.2ms/step 4,867 tok/s   ← PEAK AGGREGATE
```

**82x single-sequence speedup. 3,477x aggregate (582ms×1 → 13.2ms×64).**
