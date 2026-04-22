# Wiki 34: Qwen2.5-0.5B KV-Cached Decode on Blackhole

## Q: Does KV-cached decode work for Qwen on Blackhole?

**A:** Yes! Experiment 49 demonstrates prefill + Flash-Decode for the full 24-layer model.

### Performance
| Metric | Value |
|--------|-------|
| Prefill (5 tokens) | 264ms |
| First decode | 1338ms (JIT compilation) |
| Sustained decode | **35ms/tok (28.6 tok/sec)** |
| KV cache size | 3.0 MB total |
| Speedup vs baseline | **16.8x** (vs 582ms/tok full recompute) |

## Q: How does GQA affect the KV cache?

**A:** Qwen uses GQA with 14 Q heads but only 2 KV heads. This means:
- KV cache shape: `(1, 2, max_seq=256, 64)` per layer — tiny!
- Total: 24 layers × 2 caches × 2 × 256 × 64 × 2 bytes = **3.0 MB**
- Compare GPT-2: 12 layers × 2 × 12 × 1024 × 64 × 2 = **37.7 MB**

The 7:1 head ratio means we get the attention quality of 14 heads with the memory cost of 2.

## Q: How does prefill + decode work?

```
PREFILL (runs once per prompt):
  Full prompt → all 24 layers
  At each layer: compute Q/K/V, apply RoPE, run full-sequence SDPA
  Store K/V in persistent caches via ttnn.kv_cache.fill_cache_for_user_()

DECODE (runs per token):
  Single token embedding → all 24 layers
  At each layer:
    1. Compute Q/K/V for single position (matmul, HiFi4)
    2. Apply RoPE for current position only
    3. Update cache: ttnn.kv_cache.update_cache_for_token_(cache, new_kv, pos)
    4. Flash-Decode: ttnn.transformer.scaled_dot_product_attention_decode(
         q, k_cache, v_cache, cur_pos=[pos])
    5. Output projection + MLP (same as full forward)
  Final norm → logit projection → argmax
```

## Q: What's the first-decode JIT overhead?

**A:** 1338ms for the first decode step. This is because the single-token decode path has different tensor shapes than prefill, triggering JIT kernel compilation. Solutions:
1. **Trace capture** (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture`) — record the decode step once, replay it
2. **Warmup** — run a dummy decode step before timing

With trace capture, first-decode overhead drops to near zero (proven with GPT-2 in earlier experiments).

## Q: Why is sustained decode (35ms) faster than full recompute (54ms)?

**A:** With KV cache, each decode step processes only 1 token through all layers (constant work). Full recompute processes the entire growing sequence (linear in generated tokens). For the first few tokens the difference is small, but it compounds:
- Token 5: full recompute processes 5 tokens, decode processes 1
- Token 20: full recompute processes 20 tokens, decode processes 1
- Token 100: full recompute is ~20x slower; decode stays at 35ms

## Q: What are the remaining bottlenecks?

1. **CPU round-trips for RoPE**: 6 transfers per layer per decode step (Q/K out + rotated Q/K in + V out)
2. **JIT compilation**: First decode is 38x slower than sustained. Trace capture would fix this.
3. **No sampling**: Greedy decoding produces repetitive text with 0.5B models. Need temperature/top-k.
4. **Attention merge CPU round-trip**: Flash-Decode output needs reshaping via CPU.

## Q: What's the path to <10ms/tok?

1. **Trace capture**: Eliminate JIT overhead, reduce dispatch cost
2. **On-device RoPE**: Requires HEIGHT_SHARDED tensors (significant refactor)
3. **Batch decode**: Process multiple sequences to increase hardware utilization
4. **Program cache**: TT-NN caches compiled kernels across runs

---

*Experiment 49. Qwen2.5-0.5B KV-cached decode at 35ms/tok (28.6 tok/sec) on Blackhole P150.*
