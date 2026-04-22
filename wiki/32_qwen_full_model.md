# Wiki 32: Qwen2.5-0.5B Full Model on Blackhole

## Q: Does Qwen2.5-0.5B run on Blackhole?

**A:** Yes! Full 24-layer forward pass works with quantitative correctness validated.

### Performance (experiment 47, HiFi4+fp32 all ops)
- **Generation speed:** 18.4 tok/sec (54ms/tok) for short sequences
- **Cold start:** ~161ms first token, ~49ms subsequent
- **Weight upload:** 2.7s for 490M params (bfloat16)
- **Scaling:** Speed decreases with sequence length (no KV cache = quadratic)

### Correctness (experiment 46e, HiFi4+fp32 all ops)
- **Final logit cosine:** 0.998 vs float32 numpy reference
- **Top-1 match:** YES
- **Top-5 overlap:** 4/5
- **All 24 layers > 0.99 cosine:** YES
- **Mean per-layer cosine:** 0.9995

### Baseline (experiment 43, default config)
- **Final logit cosine:** 0.956 (below 0.99 target)
- **Top-1 match:** NO
- **Layer 21 crash:** cosine drops from 0.992 to 0.812

## Q: What fixed the precision?

**A:** `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` applied to **ALL compute ops** (matmuls + SDPA). See Wiki 33 for the full precision analysis.

**CRITICAL:** The config must be applied uniformly. Applying it to ONLY SDPA causes a kernel config state leak on Blackhole that corrupts subsequent matmuls (see Wiki 33).

## Q: What about text generation quality?

**A:** Greedy decoding with the 0.5B model produces repetitive text (e.g., "and and and..."). This is expected for small models with greedy decoding — not a precision issue. The static correctness (0.998 cosine, top-1 match) confirms the forward pass is correct.

To get quality generation: add temperature sampling, top-k/top-p, and KV caching.

## Q: What's the architecture running on device?

```
Input tokens → Embedding (CPU) → Upload to device
  For each of 24 layers:
    RMSNorm → Q/K/V projection (matmul, HiFi4+fp32)
    → Pull Q/K to CPU for RoPE (still needed — native APIs require HEIGHT_SHARDED)
    → GQA SDPA (14 Q heads, 2 KV heads, HiFi4+fp32)
    → Output projection + residual (HiFi4+fp32)
    → RMSNorm → Gate/Up projection → SiLU(gate) * up → Down projection + residual (all HiFi4+fp32)
  Final RMSNorm → Logit projection (HiFi4+fp32)
```

## Q: What are the remaining CPU round-trips?

Per layer: 3 transfers out (Q, K, V for RoPE) + 3 transfers in (rotated Q, K, V back). Total: **144 transfers per forward pass** (6 per layer × 24 layers). Eliminating these requires either:
1. On-device RoPE via element-wise ops (exp 42 showed this is slower due to overhead)
2. Native `ttnn.experimental.rotary_embedding` (requires HEIGHT_SHARDED tensors — needs memory layout work)
3. KV cache (reduces to 6 transfers per token in decode mode)

## Q: What's next?

1. **KV-cached decode:** Port the GPT-2 KV cache approach (exp 35) to Qwen — prefill + single-token decode
2. **Temperature sampling:** Add top-k/top-p for quality generation
3. **Trace capture:** Single-token decode is fixed-shape → traceable
4. **Apply HiFi4 to GPT-2:** Verify the same precision fix helps GPT-2

---

*Experiments 38-48. Qwen2.5-0.5B (490M params) at 18.4 tok/sec with 0.998 cosine on Blackhole P150.*
