# Branch III COMPLETE — Qwen3.6-27B generates coherent Paris response

**Date**: 2026-05-12
**Host**: qb2, single Blackhole P150 chip
**Status**: Correctness achieved. Performance work parked in `research/post_correctness_roadmap.md`.

## Final output

```
Prompt:    "The capital of France is"

Decode:
  step 0:  ' Paris'   (token 11751, HF rank 1 at 57.0% prob)
  step 1:  '.'
  step 2:  '\n\n'
  step 20: ' city'
  step 30: '-central'
  step 50: '<think>'
  ...

Generated text:
  The capital of France is Paris.
  <think>
  </think>
  That is correct. **Paris** is the capital and most populous city of France.
  It is located in the north-central part of the country, along the Seine River.<|endoftext|>...
```

The model is producing coherent, factually correct, instruction-tuned-style output. End-to-end on a single P150 at **3.26 tok/s, 307 ms/tok**.

## Per-layer correctness vs HF transformers ground truth

After all 7 bug fixes:
- DeltaNet layers (`linear_attention` × 48): cosine ≥ **0.99970** at last position
- Full attention layers (`full_attention` × 16): cosine ≥ **0.99989** at all positions
- Layer 63 (final): cosine 0.987 (deep-layer drift, not catastrophic)

vs the pre-fix state where layer 2 collapsed to 0.508.

## The seven bugs

1. Missing `linear_attn.norm.weight` (DeltaNet per-head Qwen3_5RMSNormGated)
2. Missing `self_attn.q_norm.weight` (per-head RMSNorm on Q before RoPE)
3. Missing `self_attn.k_norm.weight` (per-head RMSNorm on K before RoPE)
4. `Qwen3_5RMSNorm` weight needs `(1.0 + raw)` parameterization (not raw)
5. `ttnn.repeat` is tile-style; GQA broadcast needs interleave (workaround via unsqueeze-singleton-dim)
6. Upstream ttnn LLK `int → sfpi::RoundMode` bug; patched 182 sites in 73 files
7. Missing Q-scaling `1.0 / sqrt(k_head_dim)` in DeltaNet recurrence (initially dismissed as "cosine-invariant" — bug surfaced via per-row magnitude probe showing 11.3× = `sqrt(128)` ratio)

Detailed case studies in `wiki/seven_bugs_case_studies.md`.

## What's parked for performance work

See `research/post_correctness_roadmap.md`:
- bf16 residual ablation (B'9 used fp32 for a wrong hypothesis; bf16 likely equivalent quality, faster)
- Trace capture (4-5× decode speedup based on prior model experiments)
- `paged_update_cache` to eliminate per-layer numpy roundtrip for KV writes
- Native RoPE op (2.6× faster than our rotation matrix per prior experiments)
- Chunked prefill (`chunk_gated_delta_rule` instead of sequential single-token prefill)
- Multi-chip TP across all 4 P150s on qb2 (fabric works there)

## Methodology in one line

**Use HF transformers as the authoritative oracle. Never trust your own numpy reference.** The 7 bugs were invisible to our 5-phase ttnn-vs-numpy validation. They surfaced immediately once we compared per-layer ttnn output against HF's hidden states. See `wiki/debugging_methodology.md` for the full workflow.

## What we shipped this session

Beyond the code fixes:
- 10 reusable diagnostic utilities in `experiments/utils/`
- 3 wiki playbooks (bringup checklist, debugging methodology, 7 bugs case studies)
- 5 auto-memory entries with the meta-lessons
- Navigation READMEs for `experiments/` and `experiments/utils/`
- ttnn LLK header patcher (with backup + restore) for the upstream bug

The next model port should take a fraction of the time, because the institutional memory is now captured.
