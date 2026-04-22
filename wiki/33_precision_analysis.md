# Wiki 33: bfloat16 Precision Analysis for Qwen2.5-0.5B

## Q: What's the precision situation with Qwen on Blackhole?

**A:** Per-layer validation (Experiment 43) shows:
- Layers 0-20: ~0.992 cosine vs float32 reference — consistent small error
- Layer 21: drops to 0.812 — accumulated error tips past a threshold
- Final logit cosine: 0.956 — below our 0.99 target

## Q: Where does the error come from?

**A:** Experiment 44's ablation study on a single layer is definitive:

| Component | Cosine vs Reference | Verdict |
|-----------|-------------------|---------|
| Q projection (matmul) | 0.999998 | Perfect |
| K projection (matmul) | 0.999998 | Perfect |
| V projection (matmul) | 0.999949 | Perfect |
| SDPA output | **0.985252** | **THE PROBLEM** |
| RMSNorm | ~0.9999 | Fine |
| SiLU/SwiGLU | ~0.9999 | Fine |

**The SDPA softmax in bfloat16 is the error source.** Replacing only SDPA with numpy float32 improves the full layer cosine from 0.996 → 0.9998. All other ops are essentially lossless.

## Q: Why does bfloat16 softmax lose precision?

**A:** Softmax involves:
1. Exponentiation (exp) — bfloat16 has only 7-8 bits of mantissa
2. Summation for normalization — accumulated rounding errors
3. With GQA (14 Q heads, 2 KV heads), the 7:1 head ratio means each KV head is used 7 times

The attention score matrix is `(n_heads, T, T)` where T=5. With causal masking, many elements are -inf, and the remaining softmax distribution is sharply peaked — bfloat16 can't represent the tail accurately.

## Q: How does the error accumulate?

**A:** Each layer adds ~0.008 error (0.992 cosine per layer). Through residual connections, this compounds:
- Layer 5: x norm plateaus at ~1757 (residual stream is large)
- The relative error is small per layer, but 24 layers of it accumulate
- Layer 21 appears to be a tipping point where the error crosses into a qualitatively different regime

## Q: Does this affect generation quality?

**A:** Partially. The model still generates correct text ("The capital of France is Paris") but:
- Top-1 prediction differs from reference (" the" vs " ")
- Top-5 overlap is 4/5
- Repetition in generation (exp 41) may be partly due to precision, not just greedy decoding

## Q: What are the mitigation options?

**Option 1: Higher precision compute (CONFIRMED WORKING!)**
```python
config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)
attn = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, is_causal=True,
    compute_kernel_config=config
)
```
**Result:** Cosine improves from 0.980 → 0.996 (+0.016). This nearly eliminates the softmax precision loss. Over 24 layers, this should push final cosine from 0.956 to well above 0.99.

**Option 2: Mixed precision**
- Keep weights in bfloat16, do softmax in float32 on CPU
- This is what our current implementation does (Q/K/V pull to CPU) and gets ~0.999
- Trade: CPU round-trips hurt throughput

**Option 3: Attention rescaling**
- Pre-scale Q by `1/sqrt(head_dim)` before the matmul to keep attention scores in a better range for bfloat16
- This changes the magnitude going into softmax

**Option 4: Accept it**
- 0.956 final cosine still produces correct next tokens most of the time
- GPT-2 (12 layers) never had this problem because fewer layers = less accumulation
- Many production LLM deployments use bfloat16 SDPA and handle it with temperature sampling

## Q: What did GPT-2 look like for comparison?

**A:** GPT-2 (12 layers) never showed this issue because:
1. Half the layers = half the accumulated error
2. MHA (all heads same size) vs GQA may matter
3. We used the same bfloat16 SDPA but with fewer layers the error doesn't compound as much

---

## Q: Are there native RoPE APIs on device?

**A:** Yes! Experiment 45 discovered three:
- `ttnn.experimental.rotary_embedding` — basic RoPE
- `ttnn.experimental.rotary_embedding_llama` — Llama-style RoPE
- `ttnn.experimental.rotary_embedding_llama_fused_qk` — fuses Q and K RoPE together

These eliminate the CPU round-trip for RoPE entirely. Combined with HiFi4 SDPA, the full Qwen forward could be zero-CPU-roundtrip (like our GPT-2 demo).

---

*Experiments 43-45. Root cause: bfloat16 softmax in SDPA. Fix: HiFi4 + fp32_dest_acc. Matmuls are essentially lossless (0.999998).*
