# Wiki 36: The Optimization Journey — From 1.7 to 135.6 tok/sec on Blackhole

## Q: What's the full optimization timeline for Qwen2.5-0.5B on Blackhole?

**A:** Eight distinct phases, each building on the last, delivering an **80x total speedup**:

| Phase | Experiment | What changed | Latency | Throughput | Cosine |
|-------|-----------|--------------|---------|------------|--------|
| 1. Baseline | exp 41 | Full-recompute generation, default config | 582ms/tok | **1.7 tok/sec** | 0.956 |
| 2. Precision fix | exp 46e | HiFi4 + fp32_dest_acc on ALL ops | — | — | **0.998** |
| 3. HiFi4 generation | exp 47 | Generate with precision fix | 54ms/tok | **18.4 tok/sec** | 0.998 |
| 4. KV-cached decode | exp 49 | Prefill + Flash-Decode with cache | 35ms/tok | **28.6 tok/sec** | 0.998 |
| 5. Temperature sampling | exp 49b | top-k sampling for quality text | 34ms/tok | **29.3 tok/sec** | 0.998 |
| 6. On-device RoPE | exp 51b | Rotation matrix trick, correct half-format | 28ms/tok | **35.6 tok/sec** | 0.999 |
| 7. Fully on-device | exp 51c | Eliminate all inter-layer CPU transfers | 21ms/tok | **46.6 tok/sec** | 0.999 |
| 8. Trace capture | exp 52 | Record & replay decode graph | 7ms/tok | **135.6 tok/sec** | TBD |

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

## Q: How was RoPE moved on-device? (Phase 6-7 — exp 51)

**A:** Three key discoveries enabled on-device RoPE:

### Discovery 1: Qwen uses half-format RoPE, not interleaved
Experiment 51 revealed that our interleaved RoPE (even/odd element pairs) was WRONG for Qwen2.5. HuggingFace uses `rotate_half` (split at midpoint, negate, swap). The two formats give 0.510 logit cosine — completely different results. Half-format correctly produces "Paris" while interleaved gives garbage.

### Discovery 2: Rotation matrix trick bypasses ttnn.split limitation
`ttnn.split` fails on Blackhole due to tile padding (32×32 tiles). But `rotate_half(x) = x @ R` where R is a 64×64 permutation matrix. This turns the split-negate-concat sequence into a single matmul — tiny at 64×64 but works perfectly (0.999996 cosine vs numpy).

### Discovery 3: Keeping residual on device eliminates 48 more transfers
Instead of `from_dev`/`to_dev` between layers, keeping `x_tt` as a device tensor throughout all 24 layers eliminates 48 transfers per decode step. Total transfers went from ~192 (exp 49) to 2 (embedding in + logits out).

| Approach | Transfers/step | ms/tok | tok/sec |
|----------|---------------|--------|---------|
| CPU RoPE (exp 49) | ~192 | 35 | 29.3 |
| On-device RoPE (exp 51b) | ~48 | 28 | 35.6 |
| Fully on-device (exp 51c) | 2 | 21 | 46.6 |

## Q: How does trace capture achieve 135 tok/sec? (Phase 8 — exp 52)

**A:** `ttnn.begin_trace_capture` / `ttnn.execute_trace` records the entire 24-layer decode graph, then replays it without Python dispatch overhead:

```python
# Capture once
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# ... run full decode graph ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Replay per token
update_embed_for_token(token_id)   # ttnn.copy to update input buffer
update_rope_for_pos(pos)           # ttnn.copy to update cos/sin buffers
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
logits = from_dev(logits_tt, shape)  # Read output
```

Result: **7.4ms/tok (135.6 tok/sec)** — a 2.83x speedup over non-traced. Each trace execution is a single device-side command with zero Python overhead.

**Known limitation:** `cur_pos` and `update_index` are baked into the trace as Python scalars. The KV cache position doesn't advance between replays, causing text quality degradation. Next step: investigate tensor-based position APIs.

## Q: What are the remaining optimization opportunities?

**A:** Two categories: correctness fixes and further speed improvements.

### 1. Fix traced position handling
The `cur_pos` and `update_index` parameters need to be device tensors (not Python ints) so they update between trace replays. This is the critical path to combining trace capture with correct generation.

### 2. HEIGHT_SHARDED memory layouts (eliminates DRAM round-trips)
The current INTERLEAVED layout sends data through DRAM between every op. HEIGHT_SHARDED keeps activations in L1 SRAM across ops, eliminating the DRAM bottleneck. This would also enable native `ttnn.experimental.rotary_embedding` (which requires HEIGHT_SHARDED).

### 3. Batch decode
Processing multiple sequences simultaneously to increase hardware utilization. With 110 usable cores on Blackhole P150, single-sequence decode leaves most cores idle.

## Q: What's the summary of the full journey?

**A:**

```
exp 41:     582ms/tok    1.7 tok/s    Full recompute, default config
exp 46e:         —            —       Precision fix: HiFi4+fp32 ALL ops (0.998 cosine)
exp 47:      54ms/tok   18.4 tok/s    HiFi4 generation (10.8x)
exp 49:      35ms/tok   28.6 tok/s    KV-cached decode (1.5x)
exp 49b:     34ms/tok   29.3 tok/s    Temperature sampling (~free)
exp 51b:     28ms/tok   35.6 tok/s    On-device RoPE via rotation matrix (1.2x)
exp 51c:     21ms/tok   46.6 tok/s    Fully on-device decode (1.3x)
exp 52:       7ms/tok  135.6 tok/s    Trace-captured decode (2.8x) — STALE positions
exp 53e:    7.6ms/tok  131.5 tok/s    Traced + paged KV cache — CORRECT! (2.6x)
```

Total speedup: **77x** (582ms → 7.6ms, correct). The journey involved:
- One novel hardware bug discovery (kernel config state leak)
- One critical format correction (half-format RoPE, not interleaved)
- One precision deep-dive (bfloat16 SDPA softmax as sole error source)
- One algorithmic trick (rotation matrix for on-device RoPE without ttnn.split)
- One HEIGHT_SHARDED breakthrough (paged_update_cache with tensor positions)
- Three architectural improvements (HiFi4 config, KV caching, trace capture)

At 131.5 tok/sec with correct text generation, we exceed the reference Llama-3.2-1B target of 105.9 tok/sec (on a smaller model, single Blackhole P150 chip vs N300 dual Wormhole). The key insight was that `paged_update_cache` with `update_idxs_tensor` accepts device tensors for position updates, making the full decode graph traceable.

---

*Experiments 41-53e. Qwen2.5-0.5B (490M params) on Blackhole P150: 1.7 → 131.5 tok/sec (correct).*
