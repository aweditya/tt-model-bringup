# B'9.5 — Where We Are at End of Session

## Wins

5 bugs found and fixed in `91f`:

1. Missing `linear_attn.norm.weight` (DeltaNet per-head RMSNormGated)
2. Missing `self_attn.q_norm.weight` (per-head Q RMSNorm before RoPE)
3. Missing `self_attn.k_norm.weight` (per-head K RMSNorm before RoPE)
4. `Qwen3_5RMSNorm` weights must be loaded as `(1.0 + raw)` (offset-from-1 parameterization)
5. GQA broadcast must be `repeat_interleave`, not `repeat` (ttnn.repeat is tile-style)

## Measurable improvement

| Metric | Before bug fixes | After bug fixes |
|---|---|---|
| Layer-0 cosine vs HF (last token) | essentially 0 | **0.997** |
| Layer-0 cosine vs HF (best token) | essentially 0 | **0.9995** |
| 60-token generation top-1 | `'FR' × 60` (random) | `'is 1 0 0 0 0 ...'` (English-ish, locks at '0' at step 4) |

## Permanent infrastructure built

- `experiments/utils/weight_audit.py` — list/diff safetensors weight keys, stats
- `experiments/utils/repeat_semantics_probe.py` — verify ttnn op semantics (tile vs interleave)
- `experiments/utils/hf_layer0_substep_dump.py` — capture HF layer-0 intermediates via PyTorch hooks
- `experiments/utils/substep_compare.py` — diff HF vs ttnn substeps
- `experiments/91p_ttnn_layer0_vs_hf.py` — single-layer ttnn validation against HF (with `--weight-dtype` CLI)

## What's blocked

Two diagnostic paths to investigate the residual 0.3% drift are blocked by
the same ttnn LLK compilation bug:

### Block #1 — bf16/fp32 weight ablation (`research/ttnn_jit_bug_bf16_weights.md`)
Running `91p --weight-dtype bf16` (or fp32) triggers fresh JIT compilation of
the `layernorm_large_tensor` kernel for the new dtype combination, which
fails with `cannot convert 'int' to 'sfpi::RoundMode'` in ttnn's LLK
header `ckernel_sfpu_*.h`. Affects rsqrt, exp, log, sqrt, trigonometry.

bf8 path works because those kernels are pre-compiled in the ttnn wheel.

### Block #2 — substep dump (91q)
Holding device tensor references across deltanet→mlp boundaries (to defer
host reads to the end) also triggers the same JIT bug, even though
JIT cache stats report 100% hits. Cache is reporting a stale "failed
compile" without re-attempting.

## Hypotheses for the remaining 0.3% (un-tested)

Ranked by likelihood, can't directly test:
1. bf8 weight quantization noise on 7 large matmuls per layer
2. Conv1d state propagation difference (HF uses chunk_gated_delta_rule for prefill; we do single-token sequential)
3. Softplus numerical stability (`log(exp(x)+1)` vs `F.softplus(x)`)

## What we shipped despite the blockers

Generation pipeline ran end-to-end with all 5 bugs fixed. Output is now
producing recognizable English structure ("The capital of France is is 1")
before locking at a `'0'` attractor. The model is computing something
meaningful; it just doesn't have enough directional accuracy in lm_head
space to break free of attractors.

## Options for the next session

1. **Accept layer-0 cosine 0.997 and look elsewhere for the '0' attractor cause**:
   - Try sampling decode with temperature 0.7 instead of greedy — might escape the attractor
   - Look at lm_head top-5 in detail — is `'Paris'` even in the top-100?
2. **Fix ttnn's LLK header bug** — small upstream patch to add a `sfpi::RoundMode` cast
3. **Wait for a fresh ttnn release** with all dtype variants pre-compiled
4. **Switch to a chunked-prefill code path** — implement `chunk_gated_delta_rule` ourselves, avoiding the single-token sequential drift

## Process notes for the wiki

The bringup methodology that emerged:
1. Enumerate safetensors weight keys per layer FIRST; cross-check loader
2. Use HF transformers as authoritative oracle (not your own numpy)
3. Empirically probe library semantics (tile vs interleave) before assuming
4. Convert every inline `python -c` audit into a permanent utility immediately
5. Bug-hunt with substep dumps when final cosine is close but below gate
