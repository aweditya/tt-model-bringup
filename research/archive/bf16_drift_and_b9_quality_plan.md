# bf16 Drift Findings + Phase B′9 Quality Plan

## What the PJRT agent measured (Qwen2.5-0.5B, qb2)

Per-op cosine vs fp32 numpy reference, ALL at decode shapes:

| op | cos | max_abs_diff |
|---|---:|---:|
| matmul [1,896] @ [896,1024] | 0.999961 | 0.0005 |
| softmax [1,14,100] | 0.999982 | 0.0028 |
| rms_norm [1,1,896] | 0.999995 | 0.0268 |
| swiglu [1,1,4864] | 0.999991 | 0.0391 |
| attn_qk | 0.999996 | 0.0009 |
| sdpa chain | 0.999991 | 0.0005 |
| composite layer-0 (post-attn) | 0.999997 | 0.0065 |
| composite layer-0 (post-mlp) | 0.999961 | 0.0194 |

Every per-op cosine ≥ 0.9999, no kernel is broken. The issue is **cumulative bf16 drift across 24 layers** (Qwen2.5-0.5B) — small per-layer error compounds; logit margins for tokens after a strongly-cued prompt become too small to survive.

## Why this matters for Qwen3.6-27B

Our model has **64 layers** vs Qwen2.5-0.5B's 24 — drift compounds ~2.7× more. Without fp32 hot paths we should expect:
- First token: likely correct (logits still have margin)
- Tokens 2-10: degrading
- Tokens 10+: gibberish risk

This is what the prior PJRT real-model test showed: "Paris" correct, then garbage.

## Recommended fixes (Phase B′9 — quality)

Three changes, all activation/accumulator-side. Weights stay bf8/bf16 — we don't have memory to do fp32 weights.

1. **fp32 RMSNorm internals**: keep mean/rsqrt in fp32, bf16 in/out.
   `rms_norm(x, weight, eps)` → cast x to fp32, compute, cast back.
   Cost: ~2× the rms_norm time. Cheap globally (norm is small).

2. **fp32 residual stream**: keep `x` in fp32 across `add` operations, downcast to bf16 only as input to matmul/linear.
   Cost: doubles the memory bandwidth on residual adds. Still small.

3. **fp32 lm_head accumulator**: the final hidden → vocab matmul. fp32 accum on a 5120×248K matmul.
   Cost: marginal — it's a single matmul per token.

## Where to apply for B′9

Phase B′9 is a follow-up to B′8 (first-generation working). Implementation order:
1. Run B′8 first. **If text is coherent past ~10 tokens, defer B′9** — we don't need the fp32 work.
2. If text degrades fast (matches Qwen2.5-0.5B pattern), implement #1 + #2 above. Re-run, measure coherence-length.
3. If still bad, implement #3.
4. Compare side-by-side: bf16-everywhere vs fp32-hot-paths. Document tokens-of-coherence improvement.

## What this is NOT

- Not a fix for the bf16-routing-divergence problem in MoE (different issue, A5 already characterized).
- Not relevant to the architecture being correct — the *math* is right, we just lose precision.
- Not a fix for performance — fp32 paths are slower; quality vs speed tradeoff to be measured.

## Side observation from the agent

> The TT layered-decode run emitted ONE `TT_FATAL: Reads are not supported during trace capture` warning from ttnn (the engine still completed). A host-read is sneaking into the JAX-stack-of-24-layers path that doesn't appear in simpler programs.

Worth investigating in a future engine cleanup pass — could be the same kind of latent host-read we keep finding. Not blocking; flagged.
