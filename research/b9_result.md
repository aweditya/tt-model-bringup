# Phase B′9 — fp32 Residual: Implementation Succeeded, Symptom Persists

**Date**: 2026-05-12
**Status**: fp32 residual stream confirmed end-to-end; output still `'FR'` fixed-point.

## What we built

- `91l_fp32_residual_generate.py` — full 27B forward with fp32 residual stream
- `91f` modified to add typecast(Q→bf16) before SDPA, typecast(attn→x.dtype) after
- 91k API probe confirmed `rms_norm(fp32)`, `linear(fp32)`, `add(fp32, fp32)` all propagate fp32

## What B'9 changed vs B'8

| Operand | B'8 | B'9 |
|---|---|---|
| Residual stream `x` | bf16 | **fp32 (confirmed via `x_final_dtype`)** |
| Matmul outputs | bf16 | fp32 (propagated naturally) |
| RoPE cos/sin tables | bf16 | fp32 |
| conv_state (DeltaNet) | bf16 | fp32 |
| A_log, dt_bias weights | bf16 | fp32 |
| KV cache | bf16 | bf16 (unchanged — storage) |
| Projection weights (bf8/bf16) | unchanged | unchanged |
| HiFi4 + fp32 DEST | unchanged | unchanged |

## What didn't change in the output

| Metric | B'8 step 0 | B'9 step 0 |
|---|---:|---:|
| top-1 token | `'FR'` (10191) | `'FR'` (10191) |
| top-1 logit | 4.750 | 4.775 |
| top-1→top-2 margin | 0.062 | 0.070 |
| top-2 | `'jadi'` | `'jadi'` |
| top-5 contents | all junk subwords | all junk subwords |
| ‖x‖ at layer 63 | 144.72 | 142.65 (-1.4%) |
| ‖x‖ at final_norm | 61.70 | 62.15 (+0.7%) |

Logits drifted by ~0.5%. Margin grew from 0.062 to 0.097 across 5 steps — better but nowhere near the 0.5+ that would indicate the model is actually decisive. `'Paris'` still does not appear in the top-5 at ANY decode step.

## What this rules out

- **bf16 residual stream was NOT the dominant bottleneck.** Promoting to fp32 produced a 0.5% logit change. The drift hypothesis was at best a minor contributor; some larger error source exists.
- **fp32 propagation itself works.** The implementation is correct; the *fix* is what was wrong.

## New hypothesis space (ranked)

1. **lm_head weight is broken** — wrong load, wrong transpose, wrong tensor. All 64 layers may compute fine but the vocab projection is garbage.
2. **Tied embedding / lm_head mishandling** — Qwen3.6 may use tied weights and we're loading lm_head from a stale duplicate that's actually something else.
3. **Our numpy reference itself has a bug.** All B′4-B′7 cosines pass because we're comparing ttnn-to-numpy, both with the same bug. The 0.9996 cosine "validation" is then meaningless against the true model.
4. **Tokenizer/prompt format mismatch.** Qwen3.6 may require a chat template; bare prompt produces degenerate output. But the failure mode (top-5 = junk subwords) is too specific for chat-template miss.
5. **Layer dispatch order is wrong** — DeltaNet vs Gated Attention placement (we use `i % 4 != 3`).

## Performance footnote (parking lot)

- Prefill: 1.8 s → 30.8 s (one-time JIT recompile of 173 kernels for the fp32 dtype change; cache stats: 50/223 hits vs B'8's 241/241).
- Decode: 257 ms/tok → 298 ms/tok (+16%). Acceptable.
- These will recover once kernels are JIT-cached.

## Next step

`91n_lm_head_inspection.py` — pure numpy on qb2, no device. Inspect lm_head + embed, check tie_word_embeddings, sanity-decode with various inputs. Should reveal the smoking gun if it lives in the projection layer, or rule that out cleanly.

If `91n` finds nothing, the suspect is the layer math itself. Next would be a layer-by-layer correctness check, ideally against a reference that's NOT our own numpy (which may be co-buggy). Possible reference sources:
- HF AutoModel on CPU (slow but possible; memory `feedback_numpy_reference.md` says it crashed remotely, but worth re-trying)
- A trusted external implementation (e.g., load Qwen3.6 in vLLM or transformers and dump intermediate hidden states once)
