# B'9.5 — Generation Attempt After 5 Bug Fixes

**Date**: 2026-05-12
**Status**: Pipeline runs; output is now structured but still incorrect; drifts to '0' fixed-point at step 4.

## Output

```
Prompt:  "The capital of France is"
Decode:
  step 0: ' is'  (369)
  step 1: ' '    (220)
  step 2: '1'    (16)
  step 3+: '0'   (15) — fixed-point for 56 more steps

Generated: "The capital of France is is 1000000000000000..."
```

## Improvement vs prior state

| State | First decoded tokens | Pattern |
|---|---|---|
| B'8 / B'9 (5 bugs present) | `'FR' × 60` | Random subword fixed-point |
| **B'9.5 (5 fixes applied)** | `' is', ' ', '1', '0'×57` | English-like prefix, then '0' fixed-point |

Real progress: model attempts meaningful English continuation for first 3 tokens, locks at '0' afterwards. Layer-0 cosine vs HF: **0.997** (up from cosine ≈ 0 before fixes).

## Performance

- Prefill: 3.1 s (611 ms/tok)
- Decode: 18.0 s for 60 tokens = **3.33 tok/s, 300 ms/tok**
- 13 min total including weight load
- Decode latency unchanged vs B'8 baseline (modulo the 16% fp32 overhead)

## Diagnosis

Remaining ~0.3% per-layer drift compounds across 64 layers. With cosine ≈ 0.997 per layer, after 64 layers angular error grows to ~35° (independent error assumption). This is large enough that:
- First few tokens find SOME plausible neighbor in lm_head space
- The `'0'` row of lm_head becomes a global attractor in the distorted space
- Once `'0'` is fed back, the model can't escape (each step produces a hidden state close to the prior, so argmax stays `'0'`)

This is qualitatively different from a precision-only issue. The model has lost so much directional accuracy that it can't navigate even slightly off-axis from the attractor.

## Hypotheses for the remaining 0.3%

Ranked by likelihood:

1. **bf8 weight quantization noise**: in_proj_qkv, in_proj_z, out_proj, gate_proj, up_proj, down_proj, conv1d_weight all stored as bf8. Across 6 large matmuls + 1 conv1d per layer, accumulated bf8 noise could account for ~0.3% per layer.
2. **Conv1d state propagation**: HF uses `causal_conv1d_fn` for multi-token prefill; we apply single-token convolution sequentially. Different numerical paths for the same math; possible accumulated difference.
3. **Softplus stability**: `log(exp(x)+1)` vs torch `F.softplus` for large `x`. Could matter for outlier values in `a + dt_bias`.
4. **Recurrent state init order**: subtle differences in how we initialize/use `ssm_state` vs how HF's `chunk_gated_delta_rule` handles state-zero positions.

## Next step

bf16 weight ablation via `91p --weight-dtype bf16`. If cosine jumps to 0.999+, bf8 is the bottleneck and we have options (keep bf8 with drift, or move some weights to bf16 selectively). If cosine stays at 0.997, bf8 is fine and we need to look at conv1d / softplus / state-init.
