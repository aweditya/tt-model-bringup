# Phase B′9.5 — HF Reference Oracle for Layer 0

**Date**: 2026-05-12
**Goal**: Validate (or invalidate) our numpy layer-0 reference against HuggingFace's official Qwen3-Next implementation.

## Why

B'9's fp32 residual fix produced no meaningful change in output. The 'FR' fixed-point survived. lm_head and embed are stats-healthy (91n). The bug must be upstream of lm_head — either in the layer math, the layer wiring, or the architectural details our numpy reference encodes.

Our B'4-B'7 validation chain compares ttnn-to-numpy with cosine ≥ 0.997. But that's comparison against ourselves. If our numpy reference has a bug, ttnn faithfully reproduces it. The bug would never surface as a cosine drop.

HF's `transformers` library has the official Python implementation of Qwen3-Next, used to train and serve the model. It is by definition the correct reference. We just need to use it on layer 0 only (small enough to run on CPU in <1 minute).

## What we DO

1. Probe `transformers` imports — confirm Qwen3-Next classes exist and which class to use
2. Load `AutoConfig` (just JSON — no model weights)
3. Build a minimal pipeline:
   - `torch.nn.Embedding` populated with embed weights from safetensors
   - One `Qwen3NextDecoderLayer` for `layer_idx=0` populated with layer 0 weights
4. Forward `prompt_ids → embed → layer 0` on CPU
5. Save `hidden_after_layer_0[5_tokens, 5120]` to `~/tt-xla/.cache/qwen36_27b_hf_layer0_ref.npz`
6. Load our existing `qwen36_27b_layer0_3_ref.npz` (the B'2 numpy reference)
7. Compute cosine and max|Δ| between HF and numpy at layer 0 output

## What we DON'T DO

- Don't call `AutoModel.from_pretrained` (it crashed in prior auto-memory)
- Don't load all 64 layers' weights (~30 GB)
- Don't run the full forward through 64 layers (slow, unnecessary)
- Don't try to use HF lm_head (we've already validated lm_head structurally)
- Don't fix anything yet — diagnose first

## Outcomes and branches

| Cosine(HF, numpy) | Diagnosis | Next step |
|---:|---|---|
| ≥ 0.999 | numpy ref is correct | bug is in ttnn implementation; do ttnn-vs-numpy layer-by-layer |
| 0.9 ≤ cos < 0.999 | minor numpy bug (RoPE order, norm placement, etc.) | localize via per-substep comparison |
| < 0.9 | major numpy bug (wrong equation, wrong wiring) | re-derive layer 0 from HF source line-by-line |
| (HF import fails) | transformers too old or class names differ | fall back: read modeling_qwen3_next.py from GitHub, audit numpy by hand |

## Risk register

- **HF transformers version on qb2 might not include Qwen3-Next** — the probe step prints the version and available classes; we fall back to source audit
- **Layer.forward needs unfamiliar args** — HF will give a clear TypeError; we add args iteratively
- **CPU fp32 memory** — embed (5 GB) + layer (~2-3 GB) ≈ 8 GB. qb2 has 64+ GB system RAM. No risk.
- **HF defaults to bfloat16 or fp16** — we explicitly cast everything to fp32 to match our reference

## Files we touch

- `experiments/91o_hf_reference_layer0.py` (new): the oracle script
- `research/b9_layer0_oracle_plan.md` (this doc): the plan
- `research/b9_layer0_oracle_result.md` (will write after): the verdict and next steps
