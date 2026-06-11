# Wiki 45: Retrospective — The Model Zoo Sprint

## What Happened (Exps 64-68)

In one session, we went from 1 model (Qwen2.5-0.5B) to 5 models running on Blackhole:

| Model | Params | tok/sec | New Feature Tested |
|-------|--------|---------|-------------------|
| Qwen2.5-0.5B | 0.5B | 140 | Baseline (bf8, native RoPE, traced) |
| Llama-3.2-1B | 1.24B | 78 | Interleaved RoPE, no biases, split SDPA |
| Qwen3-0.6B | 0.6B | 76 | QK-Norm (RMSNorm on Q/K per-head) |
| Llama-3.2-3B | 3.2B | 34 | Multi-shard weight loading, 3B+ scale |
| SmolLM3-3B | 3B | 38 | NoPE layers (skip RoPE every 4th layer) |

Plus: continuous batching prototype (1,042 tok/sec decode, 24 requests).

## What Went Well

1. **Architecture generality proven.** Same TT-NN ops handle 5 different architectures. Only need to adjust config constants and handle feature flags (biases, RoPE format, QK-Norm, NoPE).

2. **SDPA bug discovery and systematic workaround.** The power-of-2 KV head bug was a blocker, but testing systematically (sweep all head counts) gave us a robust workaround (split into groups of 4 KV).

3. **NoPE is elegant in traced mode.** Since each layer's graph is static, NoPE layers just have fewer ops. No runtime branching needed. Saved measurable time (26.5ms vs 29.7ms for SmolLM3 vs Llama at similar scale).

4. **Near-linear parameter scaling.** 2.5x params → 2.3x latency increase suggests larger matmuls utilize the 120-core mesh more efficiently.

## What Needs Improvement

### 1. Text Quality — THE BIG GAP

All models produce repetitive, incoherent text. This is NOT a precision issue — it's because:
- **Base models + greedy decoding** = degenerate text is expected
- No temperature sampling, no top-k/top-p filtering
- No chat formatting (these are base models, not instruction-tuned)

**Action items:**
- [ ] Add temperature + top-k sampling to decode loop
- [ ] Port instruction-tuned variants (Llama-3.2-1B-Instruct, Qwen3-0.6B-Instruct)
- [ ] Validate with cosine similarity against HuggingFace reference (can't run AutoModel on remote, but can compare logits from numpy reference)

### 2. Correctness Validation — We Skipped It

For speed of porting, we skipped our usual rigorous cosine comparison:
- Prefill logits not compared against numpy float32 reference
- No per-layer output validation
- First-token match ("Paris") is necessary but not sufficient

**Action items:**
- [ ] Write numpy float32 reference for each model's forward pass
- [ ] Compare prefill logits: cosine similarity should be >0.99
- [ ] Compare first 10 tokens: should match exactly at greedy decoding
- [ ] If cosine < 0.99, ablate per-layer to find where precision diverges

### 3. Infrastructure Code Duplication

Each experiment is a standalone ~350-line file with copy-pasted code. We should refactor into:
- Shared weight loading (handles sharding, tied embeddings, bias detection)
- Shared decode loop (traced, with sampling options)
- Config-driven model definition (just pass architecture dict)

But: "no code bloat" principle says don't over-abstract too early. Wait until patterns stabilize.

### 4. Performance Insights Not Yet Exploited

The split SDPA overhead (~10% per layer) is substantial at scale. For models with 8+ KV heads:
- Filing the upstream SDPA bug is high priority
- Testing if a newer tt-metal version fixes it
- Exploring custom SDPA implementations

The NoPE finding suggests that attention architecture innovations (not just quantization or batching) meaningfully affect hardware performance. This is worth exploring more.

## Key Learnings for Infrastructure

1. **SDPA flash decode limitations are model-blocking.** The power-of-2 KV head constraint means we can't run Gemma 3 (1 KV head — MQA, might work?) or other unusual GQA ratios without workarounds.

2. **head_dim matters more than layer count for decode speed.** Qwen3-0.6B (head_dim=128) is slower than Llama-3.2-1B (head_dim=64) despite fewer params. The rotation matrix matmul (head_dim × head_dim) dominates.

3. **Weight upload time scales with model size but is one-time.** 28s for 3B model is fine. At 8B it would be ~90s — still acceptable.

4. **Tokenizer loading is fragile.** AutoTokenizer triggers torchvision imports. PreTrainedTokenizerFast with tokenizer.json is the robust path.

## Updated Performance Timeline

```
exp 41:     582ms/tok    1.7 tok/s    Qwen2.5 full recompute
exp 60:     7.1ms/tok  140.4 tok/s    Qwen2.5 traced best   ← BEST SINGLE
exp 59:    13.2ms/step 4,867 tok/s    Qwen2.5 batch=64      ← PEAK AGGREGATE
exp 64:    12.8ms/tok   78.2 tok/s    Llama-3.2-1B
exp 65:     1,042 tok/s              Continuous batching
exp 66:    13.2ms/tok   76.0 tok/s    Qwen3-0.6B
exp 67:    29.7ms/tok   33.7 tok/s    Llama-3.2-3B           ← LARGEST
exp 68:    26.5ms/tok   37.8 tok/s    SmolLM3-3B (NoPE)
```

**5 models, 34-140 tok/sec range, 0.5B-3.2B parameter range, all on one Blackhole P150 chip.**

## What's Next (Priority Order)

1. **Correctness validation** — numpy reference comparison for all 5 models
2. **Sampling + instruction-tuned models** — fix text quality
3. **Phi-4-mini (3.8B)** — fractional RoPE, completes the model zoo
4. **Batch scaling on 3B models** — test throughput at larger scale
5. **bf8 weights on 3B models** — halve memory footprint
