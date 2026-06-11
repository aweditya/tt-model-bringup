# Wiki 29: KV-Cached Decode on Blackhole

## Q: What is KV-cached decode?

**A:** Instead of recomputing attention over the full sequence for every new token (O(n²) per token), cache K/V from previous tokens and only compute the new token's Q. Each decode step is O(n) — just the new Q attending to the full cache.

## Q: What TT-NN APIs does it use?

**A:**

| API | Purpose | Phase |
|-----|---------|-------|
| `ttnn.kv_cache.fill_cache_for_user_(cache, input, batch_index)` | Write all K/V from prefill into cache | Prefill |
| `ttnn.kv_cache.update_cache_for_token_(cache, new_kv, update_index, batch_offset)` | Append single token's K/V to cache | Decode |
| `ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, cur_pos=[N])` | Flash-Decode: single Q against full cache | Decode |

## Q: What's the tensor layout for Flash-Decode?

**A:** Different from regular SDPA:

| Tensor | Regular SDPA (prefill) | Flash-Decode |
|--------|----------------------|--------------|
| Q | `[B, NH, Sq, HD]` | `[1, B, NH, HD]` |
| K | `[B, NH, Sk, HD]` | `[B, NH, S_max, HD]` |
| V | `[B, NH, Sv, HD]` | `[B, NH, S_max, HD]` |

The "MQA only" note in the docs is outdated — MHA works when `n_kv_heads == n_q_heads`. The kernel parallelizes over `batch × n_kv_heads`.

## Q: What's the performance?

**A:**

| Phase | Latency | Notes |
|-------|---------|-------|
| Prefill (5 tokens) | 10ms | Regular SDPA, fills caches |
| Decode (per token) | **11.5ms** | Constant regardless of position |
| Total (30 tokens) | ~355ms | 10ms prefill + 30 × 11.5ms |

Comparison with previous approaches:

| Approach | Token 1 | Token 50 | Token 200 |
|----------|---------|----------|-----------|
| Full recompute (no trace) | 95ms | 200ms+ | stuck |
| Traced (4 buckets) | 2.7ms | 3.4ms | 6.6ms (max 256) |
| KV-cached decode | 11.5ms | 11.5ms | 11.5ms (max 1024) |

## Q: Why is traced faster than KV-cached?

**A:** Traced execution (2.7-6.6ms) eliminates Python dispatch overhead — the entire forward pass replays as a single device command. KV-cached decode still dispatches ~150 TT-NN ops per token from Python.

The optimal approach combines both: KV-cached decode with trace capture for the decode step. Since decode is fixed-shape (always 1 token), it's theoretically traceable. Expected: ~1-3ms/token with full 1024 context.

## Q: What are the remaining CPU round-trips in decode?

**A:** One per layer: the Flash-Decode output `[1, 1, 12, 64]` needs reshaping to `[1, 1, 768]` before the output projection matmul. This goes through CPU (numpy reshape + re-upload). Eliminating this would make decode fully on-device and traceable.

## Q: How much memory do the KV caches use?

**A:** 36 MB total — trivial on Blackhole's 12GB DRAM.

```
Per cache:  1 × 12 × 1024 × 64 × 2 bytes = 1.5 MB
Per layer:  2 caches (K + V) = 3 MB
12 layers:  36 MB total
```

---

*Experiment 33/33b/35. KV cache APIs confirmed, Flash-Decode working with MHA, 87 tok/sec constant-time decode.*
