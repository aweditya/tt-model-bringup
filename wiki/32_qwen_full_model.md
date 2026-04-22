# Wiki 32: Qwen2.5-0.5B Full Model on Blackhole

## Q: Does Qwen2.5-0.5B run on Blackhole?

**A:** Yes! Full 24-layer forward pass works. Key results:
- **Latency:** 36ms/forward (1.5ms/layer, 28 fwd/sec)
- **Cold start:** 1314ms (first run, JIT compilation)
- **Weight upload:** 2.8s for 490M params (bfloat16)
- **Top prediction:** "The capital of France is" → " the" (sensible continuation)

## Q: What's the prediction quality like?

**A:** Without HF reference comparison (transformers import fails on remote due to torchvision conflict), we can only judge qualitatively. For "The capital of France is":

| Rank | Token | Logit |
|------|-------|-------|
| 1 | " the" | 16.375 |
| 2 | " a" | 16.125 |
| 3 | " " | 15.625 |
| 4 | " located" | 15.188 |
| 5 | " an" | 14.188 |

These are reasonable next-token predictions. The logit distribution looks healthy (top tokens are close, then clear drop-off). "Paris" would come after "the capital of France is the" → "capital of" → likely "Paris".

## Q: What's the architecture running on device?

```
Input tokens → Embedding (CPU) → Upload to device
  For each of 24 layers:
    RMSNorm → Q/K/V projection (separate matmuls)
    → RoPE (decomposed even/odd on device)
    → GQA SDPA (14 Q heads, 2 KV heads)
    → Output projection + residual
    → RMSNorm → Gate/Up projection → SiLU(gate) * up → Down projection + residual
  Final RMSNorm → Logit projection (on CPU via tied embeddings)
```

## Q: What about the attention accuracy issue from experiment 37?

**A:** Single-layer cosine was 0.961 in experiment 37 (below 0.99 threshold), but the full 24-layer model produces sensible predictions. Possible explanations:
1. Errors cancel out across layers (unlikely but possible)
2. The accuracy is "good enough" for next-token prediction even if individual layer cosine is <0.99
3. The bfloat16 precision through softmax matters less than expected for generation quality

Still need to get HF reference working for quantitative comparison.

## Q: What's the x norm pattern telling us?

**A:** The norm progression across layers is:
- Layer 0: 15.26 (small — just embedding + first transform)
- Layer 5: 1756.94 (norm grows as residual stream accumulates)
- Layer 11: 1757.07 (stable through middle layers)
- Layer 17: 1758.32 (still stable)
- Layer 23: 217.10 (drops at final layer — RMSNorm + projection)

The plateau at ~1757 across layers 5-17 is healthy. The drop at layer 23 is expected from final normalization.

## Q: What's next?

1. **Text generation:** Add autoregressive loop with tokenizer — the forward pass works, so this is wiring
2. **KV-cached decode:** Port the GPT-2 KV cache approach (Wiki 29) to Qwen — only 2 KV heads per layer means tiny cache (~25MB)
3. **Trace capture:** Single-token decode is fixed-shape → traceable → should get <5ms/token
4. **HF comparison:** Fix transformers import or compute reference logits locally and upload

---

*Experiment 38. Qwen2.5-0.5B (490M params) running at 36ms/forward on Blackhole P150.*
