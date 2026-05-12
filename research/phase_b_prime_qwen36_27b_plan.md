# Phase B′ — Qwen3.6-27B on Single Chip

## Context

Branch III pivoted from the original target Qwen3.6-**35B-A3B** (won't fit one chip; needs fabric we don't have) to **Qwen3.6-27B**. Same family, released the same week (April 24, 2026), same hybrid architecture — but **dense MLP instead of 256-expert MoE**, and small enough to fit one P150 with bf8 weights. The friend in charge of qb may yet cable up inter-chip fabric, in which case the original 35B-A3B plan unblocks; Phase B′ doesn't preclude that.

## The architecture (from real config.json, verified)

```
Total params:           ~27B (dense)
hidden_size:            5120
num_hidden_layers:      64
layer_types:            [L L L F] × 16   (48 DeltaNet + 16 Gated Attention)
full_attention_interval: 4

Gated DeltaNet (linear-attn, 48 layers):
  linear_num_key_heads:   16
  linear_num_value_heads: 48
  linear_key_head_dim:    128
  linear_value_head_dim:  128
  linear_conv_kernel_dim: 4
  mamba_ssm_dtype:        float32  (state must stay fp32)

Gated Attention (16 layers, every 4th):
  num_attention_heads:    24 (Q heads)
  num_key_value_heads:    4  (KV heads — GQA 6:1)
  head_dim:               256
  partial_rotary_factor:  0.25 → 64 rotary dims, 192 pass-through
  attn_output_gate:       true
  output_gate_type:       swish
  rope_theta:             10_000_000

Dense MLP (every layer):
  intermediate_size:      17408
  hidden_act:             silu
  (so: gate_proj + up_proj are both 5120 → 17408,
   down_proj is 17408 → 5120, SwiGLU pattern)

Context:                 262K native (1M+ with YaRN)
Vocab:                   248320
tie_word_embeddings:     false
mtp_num_hidden_layers:   1 (multi-token-prediction head — we ignore)
```

## What every Phase A artifact gives us

| Phase A artifact | Reuse in B′ |
|---|---|
| A3 — DeltaNet isolated, cosine 0.999995 | Direct port; just bigger shapes (`hidden=5120`, `n_v_heads=48`) |
| A4 — Gated Attention isolated, cosine 0.999943 | Direct port; GQA 24/4 vs 16/2, head_dim 256 same |
| A6 v1 — chunked-serial scan, 800 tok/s prefill | Direct port; bigger state but same recurrence |
| A0 — MoE regression diag | not applicable (27B has no MoE) — bonus simpler |
| A5 — MoE isolated | not applicable |
| A7 — Multi-chip primitives | parked until friend wires fabric |
| A1, A2 — arch docs + equations | foundation — DeltaNet math unchanged |

**This is why 27B is the *better* target on our hardware**: same architectural learning (DeltaNet stays the new physics) without the MoE+multi-chip complexity that was buying us into.

## Memory math (per Blackhole P150 ~30 GB usable)

Per-shape estimate at bf8 weights:

| Component | Params | bf8 size |
|---|---:|---:|
| DeltaNet × 48 layers | ~6.7B | ~6.7 GB |
| Gated Attention × 16 layers | ~1.5B | ~1.5 GB |
| Dense MLP × 64 layers (5120 × 17408 × 3 each) | ~17.1B | ~17.1 GB |
| Embeddings (in + out, vocab 248K × 5120) | ~2.5B | ~2.5 GB |
| RMSNorms etc | ~0.01B | ~0.01 GB |
| **TOTAL weights** | **~27.8B** | **~27.8 GB** |

### Runtime memory budget by KV-cache length

DeltaNet state H = 48 layers × 48 v-heads × 128 × 128 × 4 bytes (fp32) = ~150 MB (negligible).

KV cache for the 16 full-attention layers = 16 × 2 (K+V) × 4 KV heads × 256 head_dim × N tokens × bytes:

| Context | KV bf16 | KV bf8 | Weights + KV + ~2GB scratch |
|---|---:|---:|---:|
| 1K | 67 MB | 33 MB | ~30 GB ✓ |
| 4K | 268 MB | 134 MB | ~30 GB ✓ marginal |
| 8K | 536 MB | 268 MB | ~30.5 GB tight |
| 16K | 1.1 GB | 536 MB | ~31 GB doesn't fit at bf16 |
| 32K | 2.1 GB | 1.1 GB | bf8 KV: ~31 GB, still tight |

Realistic targets:
- **bf16 KV, 4K context** → fits with ~1 GB headroom. Phase B′ initial target.
- **bf8 KV, 8K context** → fits with ~1 GB headroom. Phase B′ stretch.
- **bf16 KV, 32K context** → doesn't fit one chip. Would need fabric or further quant.

For daily-driver coding use case 4K-8K context is OK. Whole codebase ingestion needs more — that's a Phase C concern.

## Phase B′ substeps

### B′1 — Weight skeleton + memory plan (~1 hr)

Adapt `experiments/90_qwen36_weight_skeleton.py` for the 27B model:
- Fetch its `config.json` + `model.safetensors.index.json`
- Verify layer-types match `[L L L F] × 16`
- Confirm parameter count (~27B)
- Cross-check our per-component memory math against the actual safetensors

### B′2 — Numpy fp32 reference (first 2 layers) (~2 hrs)

Per `feedback_numpy_reference.md` we always write our own reference (HF AutoModel crashes on remote). Reuse Phase A's DeltaNet + Gated Attention math, just at the 27B shapes. Two layers is enough to gate correctness without 70 GB host RAM for the full reference.

### B′3 — ttnn implementation, single layer (~3-4 hrs)

`experiments/91_qwen36_27b_port.py`:
- Load real weights (bf8 quant on upload, layer 0 only first)
- Wire forward: RMSNorm → DeltaNet → residual → RMSNorm → MLP → residual
- Cosine ≥ 0.99 vs numpy ref for layer 0 single token
- Then enable layer 1 (which is full-attention) — cosine ≥ 0.99
- Then layers 2-4 (a full [L L L F] pattern)

### B′4 — Full 64-layer wire-up (~2 hrs)

Loop the layer pattern. Track cosine at layers 0, 4, 8, 16, 32, 48, 60 against numpy reference (just those layers in fp32, won't fit all).

### B′5 — Correctness gate (~2 hrs)

Per `feedback_correctness_first.md`:
- First-token cosine vs fp32 reference ≥ 0.99 (gate)
- 8/8 greedy match with `experiments/76_8b_numpy_reference.py` adapted

Per `feedback_generation_limits.md`: greedy 60-100 tokens, look for quality not just match.

### B′6 — Trace capture (~3 hrs)

Apply to DeltaNet + Gated Attention layers (both fully static). Dense MLP is also static, no MoE routing data-dependence — should be 100% traceable across all 64 layers per token.

### B′7 — Performance measurement (~1 hr)

From the A measurements:
- DeltaNet 27B-scale (proportional): ~400-500 µs traced per layer × 48 = 24 ms
- Attention 27B-scale: ~600-800 µs traced × 16 = 13 ms
- MLP (5120 × 17408 × 3): ~150 µs × 64 = 10 ms
- Other (norms, residuals): ~10 ms
- **Per-token target: ~50-60 ms = 17-20 tok/s** (single chip, traced)

Realistic for daily-driver coding. Reach: 30-40 tok/s with op fusion.

### B′8 — First real generation (~1 hr)

```
"Write a Python function that takes a list of integers and returns the most frequent element."
```

Expect: coherent function, with explanation. Per `feedback_generation_limits.md` generate ≥ 60 tokens.

## Total estimate

**Phase B′: ~12-15 hours of focused work.** Substantially less than the original Phase B because we drop the multi-chip TP step (~10 hrs saved) and the MoE complexity.

## What we keep amenable to "fabric returns"

Even though we go single-chip, the port code should:
1. Use `ttnn.MeshDevice` opening (1×1 mesh) so adding 2×1 later is shape-only
2. Avoid hardcoded `device_id=0` — accept a `device` parameter throughout
3. Keep weight upload functions parameterizable for replicated vs sharded placement

When the friend gets the cables in, we add `mesh_shape=(2,1)` and the dense MLP gets TP-sharded across hidden dim. ~4-8 hours additional work.

## Correctness gates (the non-negotiable list)

Per `CLAUDE.md` and persistent memory:
- ✓ Plan first (this file) — done before any 27B code
- ✓ Numpy fp32 reference (NOT HF AutoModel — see `feedback_numpy_reference.md`)
- ✓ Cosine ≥ 0.99 BEFORE any perf work (`feedback_correctness_first.md`)
- ✓ Test with 60-100+ tokens not 8 (`feedback_generation_limits.md`)
- ✓ bf8 re-validate at 27B — prior memory says safe through 8B, this is 3× bigger
- ✓ Native ttnn.experimental.rotary_embedding for RoPE (2.6× faster than rotation matrix)
- ✓ Compute kernel config all-or-nothing on Blackhole (HiFi4 everywhere)
- ✓ Paged KV cache for traceable decode (the partial-RoPE workaround from A4 carries forward)
- ✓ All scripts permanent files, no /tmp, ssh qb1 only, frequent commits

## Files Phase B′ produces

```
experiments/91_qwen36_27b_port.py             — main port
experiments/91b_qwen36_27b_numpy_ref.py       — 2-layer numpy reference
experiments/91c_qwen36_27b_correctness.py     — cosine + token-match harness
research/phase_b_prime_results.md             — final numbers, lessons
research/qwen36_27b_first_generations.md      — actual model outputs
```

## Status

Plan ready. Implementation starts B′1 next session (or as soon as you say go).
