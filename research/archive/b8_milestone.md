# Phase B′8 — End-to-end Qwen3.6-27B Generation: Milestone Reached, Drift Suspected

**Date**: 2026-05-12
**Host**: qb2 (single P150)
**Status**: Pipeline runs end-to-end; output is a fixed-point, not coherent generation.

## What works

- All 64 layers of bf8 weights load on a single P150 (645s total, ~10 GB/s upload effective)
- Prefill: 5 prompt tokens in 1.8s (361 ms/tok)
- Decode: 60 tokens in 15.1s = **3.97 tok/s** (~250 ms/tok)
- No crashes, no NaNs, no shape errors
- Pre-flight harness (`91i_shape_preflight.py`) caught one shape bug in 76s before the 10-min weight load was paid for the second time

## What's broken

Greedy decode locks to a single token after the first step:

```
Prompt:  "The capital of France is"
Output:  "FRFRFRFRFRFRFRFRFRFR...×60"
Token:   10191 = '▁FR' (Qwen3.6 BPE)
```

Token `'▁FR'` is *plausible* as the first continuation of "France is" — Qwen's BPE produces capital-prefix subword tokens, and "France is FRench" is one valid split. The bug is that **every subsequent step also picks `'▁FR'`**, regardless of accumulating context.

## Two hypotheses

1. **bf16 drift collapse** (most likely). 64 layers of bf16 accumulation distorts the final hidden state until it becomes effectively input-independent. argmax then locks to whichever token has the largest residual logit, deterministically. This matches the bf16-drift pattern PJRT measured on Qwen2.5-0.5B (24 layers) — and 27B has 64 layers, so drift compounds ~2.7× more.

2. **State propagation bug**. DeltaNet H state, conv state, or KV cache reference not flowing across decode steps. If hidden states are *identical* across calls, you'd get this exact symptom.

Diagnostic to distinguish: instrument `forward_one_token` to print `‖x‖` at layers [0, 16, 32, 48, final] and top-5 logits at decode steps 0, 1, 5, 30. If norms differ across steps → drift, do B′9 (fp32 hot paths). If norms identical → loop bug, fix it.

## What we paid

- 645s weight load (one-time per process)
- 1.8s prefill
- 15.1s for 60 tokens of decode
- Per-token cost: 250 ms dominated by dispatch + KV cache numpy roundtrip (~16 MB/token PCIe)

## What we did NOT do

- Test multi-prompt
- Run any sampling (greedy only — sampling would mask the fixed-point under noise)
- Trace capture (defer to post-correctness)
- B'6.5 (paged_update_cache, defer)
- B'9 (fp32 hot paths — that's the obvious next step pending diagnostic)
