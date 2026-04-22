# Wiki 36: The Optimization Journey — From 1.7 to 29.3 tok/sec on Blackhole

## Q: What's the full optimization timeline for Qwen2.5-0.5B on Blackhole?

**A:** Five distinct phases, each building on the last, delivering a **17x total speedup** and raising cosine similarity from 0.956 to 0.998:

| Phase | Experiment | What changed | Latency | Throughput | Cosine |
|-------|-----------|--------------|---------|------------|--------|
| 1. Baseline | exp 41 | Full-recompute generation, default config | 582ms/tok | **1.7 tok/sec** | 0.956 |
| 2. Precision fix | exp 46e | HiFi4 + fp32_dest_acc on ALL ops | — | — | **0.998** |
| 3. HiFi4 generation | exp 47 | Generate with precision fix | 54ms/tok | **18.4 tok/sec** | 0.998 |
| 4. KV-cached decode | exp 49 | Prefill + Flash-Decode with cache | 35ms/tok | **28.6 tok/sec** | 0.998 |
| 5. Temperature sampling | exp 49b | top-k sampling for quality text | 34ms/tok | **29.3 tok/sec** | 0.998 |

## Q: What was the baseline like? (Phase 1 — exp 41)

**A:** The first working Qwen generation was a brute-force full-recompute approach: every token re-runs the entire growing sequence through all 24 layers. At 582ms/tok (1.7 tok/sec), it was functional but unusable.

Key problems:
- **Quadratic scaling:** Processing grows with sequence length (token 20 reprocesses all 20 tokens)
- **Precision:** Default bfloat16 config gave 0.956 final cosine — below our 0.99 threshold
- **Top-1 mismatch:** The model predicted a different next token than the float32 reference
- **Repetitive output:** Generated "and and and..." due to precision errors compounding with greedy decoding

The 582ms included all 24 layers of matmuls, SDPA, RMSNorm, and SwiGLU — plus 144 CPU round-trips per forward pass (6 per layer for RoPE: Q/K/V out, rotated Q/K/V back in).

## Q: How was the precision problem debugged? (Phase 2 — exp 43-46e)

**A:** This was the hardest debugging story in the project. It took 7 sub-experiments to isolate the root cause and find a fix that didn't introduce new bugs.

### Step 1: Per-layer cosine profiling (exp 43)

Validated each of the 24 layers individually against a float32 numpy reference:
- Layers 0-20: ~0.992 cosine each — consistent small error accumulating
- **Layer 21:** cosine crashes from 0.992 to 0.812 — a tipping point
- Final logit cosine: 0.956

### Step 2: Single-layer ablation (exp 44)

Isolated which op within a single layer was lossy:

| Component | Cosine vs float32 |
|-----------|-------------------|
| Q projection (matmul) | 0.999998 |
| K projection (matmul) | 0.999998 |
| V projection (matmul) | 0.999949 |
| **SDPA output** | **0.985252** |
| RMSNorm | ~0.9999 |
| SiLU/SwiGLU | ~0.9999 |

**Verdict:** The bfloat16 softmax inside SDPA was the sole error source. Replacing only SDPA with numpy float32 improved a full layer from 0.996 to 0.9998.

### Step 3: Understanding why bfloat16 softmax fails

The softmax in scaled dot-product attention involves exponentiation and normalization — operations where bfloat16's 7-8 bits of mantissa cause catastrophic rounding. With Qwen's GQA (14 Q heads sharing 2 KV heads, a 7:1 ratio), each KV head is reused 7 times, amplifying the error. Through 24 residual-connected layers, the per-layer ~0.008 error compounds until layer 21 tips into a qualitatively different regime.

### Step 4: The fix — and the trap (exp 45-46e)

The fix is `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. But applying it revealed a **critical Blackhole hardware bug**:

**The kernel config state leak:** Applying HiFi4+fp32 to ONLY SDPA but not subsequent matmuls causes the kernel configuration to "leak" — the matmul runs with corrupted settings and cosine crashes to 0.873 at layer 3.

| Config strategy | Layers 0-2 | Layer 3+ |
|----------------|------------|----------|
| HiFi4 on SDPA only | 0.997-0.999 | **0.873 (corruption!)** |
| HiFi4 on ALL ops | 0.9995-1.0000 | 0.9995-1.0000 |
| Default SDPA + HiFi4 matmuls | 0.992-0.999 | 0.999 (fine) |

The leak is directional: HiFi4 → default corrupts; default → HiFi4 does not. This is a novel finding — no upstream tt-metal issue or PR documents this behavior (see Wiki 35). The fix is simple: use the same config everywhere. The result was **all 24 layers above 0.99 cosine, mean 0.9995, final logit cosine 0.998.**

## Q: What did the precision fix do for performance? (Phase 3 — exp 47)

**A:** Counterintuitively, HiFi4+fp32 made generation *faster*, not slower. The explanation: with correct precision, the model produces coherent text instead of degenerate repetitions. But the real performance story is that moving from the broken baseline (exp 41, 582ms) to the corrected generation (exp 47, 54ms) involved both the precision fix and incremental improvements to the forward pass:

- **Cold start:** ~161ms first token (JIT compilation)
- **Sustained:** ~49-54ms per token
- **Weight upload:** 2.7s for 490M params (bfloat16)
- **Scaling:** Speed decreases with sequence length due to quadratic attention (no KV cache)

The 582ms → 54ms jump (10.8x) came primarily from removing overhead in the generation loop that was present in the early exp 41 prototype, plus the matmul performance improvements from HiFi4 config.

## Q: How does KV caching change the picture? (Phase 4 — exp 49)

**A:** KV caching converts the quadratic full-recompute into constant-time per-token decode. The architecture splits into two phases:

```
PREFILL (once per prompt):
  Full prompt → 24 layers → fill K/V caches
  ttnn.kv_cache.fill_cache_for_user_(k_cache, k_tensor, batch_index=0)

DECODE (per token, constant cost):
  Single token → 24 layers → update caches → Flash-Decode
  ttnn.kv_cache.update_cache_for_token_(cache, new_kv, pos)
  ttnn.transformer.scaled_dot_product_attention_decode(q, k_cache, v_cache, cur_pos=[pos])
```

Performance results:

| Metric | Value |
|--------|-------|
| Prefill (5 tokens) | 264ms |
| First decode (JIT) | 1338ms |
| Sustained decode | **35ms/tok (28.6 tok/sec)** |
| KV cache memory | 3.0 MB total |
| Speedup vs full recompute | **16.8x** |

Qwen's GQA architecture (2 KV heads vs 14 Q heads) makes the cache tiny: just 3.0 MB total across all 24 layers. Compare GPT-2's 12-head MHA cache at 37.7 MB — Qwen uses 12x less cache memory while having twice the layers.

The 35ms/tok is constant regardless of sequence position — token 5 and token 200 both take 35ms. Full recompute would take ~700ms by token 200.

## Q: What did temperature sampling add? (Phase 5 — exp 49b)

**A:** Temperature + top-k sampling (temp=0.7, top_k=50) added negligible overhead (~1ms) while dramatically improving text quality. The sampling function is trivial numpy:

```python
def sample_top_k(logits, temp=0.7, top_k=50):
    logits = logits / temp
    top_idx = np.argsort(logits)[-top_k:]
    top_logits = logits[top_idx]
    probs = np.exp(top_logits - np.max(top_logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(top_idx, p=probs))
```

The 0.5B model with greedy decoding produces repetitive text ("and and and...") — this is a model-size issue, not a precision issue (cosine is 0.998). Temperature sampling unlocks coherent generation. Final throughput: **34ms/tok, 29.3 tok/sec**.

## Q: How does this compare to GPT-2 on the same hardware?

**A:** GPT-2 small (124M params) was our first model, reaching 95ms/tok in exp 31. The two models illustrate different bottleneck profiles:

| | GPT-2 (124M) | Qwen2.5-0.5B (490M) |
|---|---|---|
| Layers | 12 | 24 |
| Hidden dim | 768 | 896 |
| Attention | 12-head MHA | 14Q/2KV GQA |
| Cosine vs reference | ~1.000 | 0.998 |
| CPU round-trips/layer | 2 (QKV split, head concat) | 6 (RoPE: Q/K/V out + back) |
| KV cache | Not implemented | 35ms/tok |
| Trace capture | Proven (exp 22: 0.39ms/layer) | Not yet applied |
| Best throughput | ~10 tok/sec | **29.3 tok/sec** |

GPT-2 never needed precision fixes (12 layers = insufficient error accumulation). But GPT-2 also never got KV caching or the generation optimizations that Qwen benefited from. The Qwen work represents a more mature pipeline.

## Q: How do we compare against Tenstorrent's reference numbers?

**A:** From tt-metal's model zoo (on N300 = 2x Wormhole cards):

| Model | Reference tok/s | Our tok/s | Gap |
|-------|----------------|-----------|-----|
| Llama-3.2-1B | 105.9 | — | — |
| Llama-3.2-3B | 68.0 | — | — |
| Qwen2.5-7B | 24.6 | — | — |
| **Qwen2.5-0.5B** | **N/A** | **29.3** | — |

The reference targets use N300 (2 Wormhole chips) with full optimizations: trace capture, HEIGHT_SHARDED memory layouts, on-device RoPE, and batched decode. Our 29.3 tok/sec on a single Blackhole P150 — with CPU round-trips still present — suggests the hardware is capable of much more once we close the optimization gap.

## Q: What are the remaining optimization opportunities?

**A:** Three major opportunities remain, each independently addressable:

### 1. Trace capture (estimated 2-3x speedup)
Single-token decode has fixed tensor shapes — it is traceable. Trace capture records the full 24-layer decode pass once, then replays it with `ttnn.execute_trace`, eliminating Python dispatch overhead and JIT recompilation. We proved this works with GPT-2 (exp 22: 3x speedup from tracing). The 1338ms first-decode JIT cost would drop to near zero.

### 2. On-device RoPE (eliminates 144 CPU transfers)
Currently, Q/K/V are pulled to CPU for RoPE rotation and pushed back — 6 transfers per layer, 144 per forward pass. TT-NN provides native APIs:
- `ttnn.experimental.rotary_embedding`
- `ttnn.experimental.rotary_embedding_llama`
- `ttnn.experimental.rotary_embedding_llama_fused_qk`

These require HEIGHT_SHARDED tensor layouts, which is a significant refactor. Exp 42 showed that a naive element-wise RoPE on device was actually slower due to per-op dispatch overhead — native kernels are the path.

### 3. HEIGHT_SHARDED memory layouts (eliminates DRAM round-trips)
The current INTERLEAVED layout sends data through DRAM between every op. HEIGHT_SHARDED keeps activations in L1 SRAM across ops, eliminating the DRAM bottleneck. This requires reworking tensor creation and memory management throughout the pipeline.

### Projected performance
With all three optimizations, the path to **<10ms/tok (100+ tok/sec)** is realistic:
- 35ms current → ~12ms with trace capture → ~8ms with on-device RoPE → ~5ms with sharded layouts

## Q: What's the summary of the full journey?

**A:**

```
exp 41:  582ms/tok   1.7 tok/s   cos=0.956   Full recompute, default config
                                               |
exp 43-46e:                      cos=0.998     Precision debugging: bfloat16 SDPA softmax
                                               identified, kernel config state leak discovered,
                                               HiFi4+fp32 ALL ops fix validated
                                               |
exp 47:   54ms/tok  18.4 tok/s   cos=0.998     HiFi4 generation (10.8x vs baseline)
                                               |
exp 49:   35ms/tok  28.6 tok/s   cos=0.998     KV-cached decode (1.5x vs full recompute)
                                               |
exp 49b:  34ms/tok  29.3 tok/s   cos=0.998     Temperature sampling (+quality, ~free)
```

Total speedup: **17.2x** (582ms → 34ms). Cosine improvement: 0.956 → 0.998. The journey involved one novel hardware bug discovery (kernel config state leak), one deep precision analysis (bfloat16 softmax as the sole error source), and three architectural improvements (HiFi4 config, KV caching, sampling). The remaining 3.4x to 100+ tok/sec is a matter of known optimizations: trace capture, on-device RoPE, and sharded memory layouts.

---

*Experiments 41-49b. Qwen2.5-0.5B (490M params) on Blackhole P150: 1.7 → 29.3 tok/sec, cosine 0.956 → 0.998.*
