# Qwen3-Coder-30B-A3B-Instruct Port Plan

## Context

User wants a 32B-class Qwen running locally on qb1 to replace cloud LLMs for coding, research, and learning. After surveying the landscape (`research/candidate_models_2026.md`), the right target is **Qwen3-Coder-30B-A3B-Instruct** — MoE with 30.5B total / 3.3B active per token, coder-tuned, 256K context. It is the goal-state model for the user's daily use; we're not making intermediate stepping stones.

Phase strategy: single-chip first (bf8 quant) → multi-chip TP later for bf16 and longer context.

## Architecture summary

```
Total parameters:   30.5B
Active per token:   3.3B
Number of layers:   48
Q heads / KV heads: 32 / 4   (GQA ratio 8:1)
head_dim:           128 (implied: hidden 4096 / 32 heads)
Hidden size:        4096
MoE experts:        128 total, 8 active per token
Context (native):   262,144 tokens
RoPE theta:         (look up from config.json — likely 1M)
Vocab:              151,936 (Qwen family standard)
License:            Apache 2.0
HF model:           Qwen/Qwen3-Coder-30B-A3B-Instruct
```

Compared to our existing **Qwen1.5-MoE-A2.7B** (which works on qb1):

| | Qwen1.5-MoE | Qwen3-Coder-30B-A3B |
|---|---|---|
| Total | 14.3B | 30.5B |
| Active | 2.7B | 3.3B |
| Layers | 24 | 48 |
| Experts | 60 | 128 |
| Active experts | 4 | 8 |
| Hidden | 2048 | 4096 |
| Q heads | 16 | 32 |
| KV heads | 16 | 4 (proper GQA) |
| Head dim | 128 | 128 |
| Context | 8K | 256K |

So this is **2× more layers, 2× bigger hidden, 2× more experts, 2× active experts per token, more aggressive GQA, much longer context**. Architecturally the same shape — should be a parameter-config change on top of our existing MoE port code, not a new infrastructure layer.

## Memory budget

One Blackhole chip has ~32 GB usable DRAM.

| Storage | Per-param bytes | 30B total |
|---|---|---|
| bf16 weights | 2 | 60 GB — **does not fit** |
| bf8 weights | 1 | 30 GB — fits, tight |
| Mixed (bf8 experts + bf16 attention) | ~1.1 | ~33 GB — marginal |

Plus we need room for:
- KV cache: bf16 × 48 layers × 4 KV heads × 128 head_dim × N tokens × 2 (K+V) = ~50 KB/token. For 4096-token context (modest): 200 MB. For 32K: 1.6 GB. For 256K: 13 GB.
- Activations during forward: ~MBs at batch=1.
- Workspace buffers, prefill scratch.

**Path A (chosen for first cut): all bf8, 4K-context KV cache.** Fits with ~1.8 GB headroom. We've previously validated bf8 has cosine > 0.999 vs fp32 (per `feedback_bf8_weights.md`). This is the bf8-safe regime.

**Path B (later): mixed precision, longer context.** Once Path A works, can experiment.

## Risks

1. **MoE perf regression on ttnn 0.69.** Our Qwen1.5-MoE port dropped from 22.7 to 15.7 tok/s. We must investigate before porting, otherwise we're inheriting a 45% performance hit and won't know where to look.
2. **bf8 quality at 30B.** Prior bf8 validation was on smaller models (≤8B). Deeper models accumulate quantization error layer-by-layer. Cosine check at every 8th layer would be wise.
3. **Top-8 routing.** Our Qwen1.5-MoE uses top-4. Top-8 is 2× more expert dispatches per token. Even if routing logic ports cleanly, perf math changes.
4. **GQA 8:1 ratio.** Llama-style ports we have use 4:1. Need to verify SDPA / KV cache layouts handle 8:1 cleanly.
5. **Long-context.** 256K context is well beyond anything we've tested. For Phase 1, cap context at 4K to stay sane.
6. **Tokenizer.** Qwen3-Coder uses a different tokenizer than Qwen1.5. Need to verify it loads and matches HF behavior.

## Plan in phases

### Phase 0 — MoE regression diagnosis (1-2 hours, blocking)

Why first: if Qwen1.5-MoE is 45% slower, Qwen3-Coder will inherit it and we'll waste time benchmarking.

Concrete:
- Write `experiments/81_moe_regression_microbench.py` that imports our existing Qwen1.5-MoE port code and times the inner ops in isolation: per-layer routing (softmax/topk/sigmoid), expert matmul, expert weight read, attention.
- Identify which op(s) regressed.
- If it's a config change (e.g., new ttnn defaults), revert. If it's a kernel regression, file a note and move on.

Output: a section in `research/qb1_model_validation.md` explaining the regression cause OR documenting that we accept the slower MoE for this port.

### Phase 1 — Single-chip Qwen3-Coder port (5-8 hours)

1. **Spec script** `experiments/81_qwen3_coder_30b_port.py` (or 82 if 81 taken). Same structure as `demos/generate_moe.py`:
   - Load config from `Qwen/Qwen3-Coder-30B-A3B-Instruct` via HF
   - Load weights via `safetensors`, bf8 quant on upload
   - Forward = embed → 48 × decoder_layer → norm → output projection
   - decoder_layer = RMSNorm → attention (GQA 32/4, RoPE, KV cache) → residual → RMSNorm → MoE (top-8 routing over 128 experts) → residual
2. **Correctness gate** (CRITICAL):
   - Write a pure-numpy fp32 reference for the first 2 layers (full reference would OOM, but first 2 layers tells us if attention + MoE are correct)
   - Cosine ≥ 0.99 prefill vs reference
   - Greedy decode 8 tokens, match pure-numpy reference exactly
3. **First generation**: `prompt="Write a Python function that..."`, `max_tokens=200`, greedy. Verify coherent code.

Stopping conditions:
- If cosine < 0.99 at first checkpoint: ablate (skip MoE, just dense MLP) to isolate whether attention or routing is broken.
- If decode crashes: capture log to `~/tt-xla/.cache/qwen3coder_crash.log`, investigate.

### Phase 2 — Performance optimization (3-5 hours, optional)

Trace capture for attention + MoE shared layers (routing is dynamic, can't fully trace per our exp 95 finding).

Target: ≥ 8 tok/s on one chip. Reach: ≥ 15 tok/s once routing dispatch is optimized.

### Phase 3 — Multi-chip TP (separate effort, after Phase 1+2)

Shard across all 4 qb1 chips:
- Hidden dim sharded across chips for tensor parallel
- Experts sharded across chips (each chip owns 32 experts)
- Use bf16 weights since fitting is no longer the constraint
- Verify ttnn's mesh device API supports this on Blackhole

Target: ≥ 20 tok/s with bf16 quality + 32K context.

## Non-negotiables for this port

- All execution via `ssh qb1`
- Device 0 only for Phase 1 (4 chips for Phase 3)
- No /tmp — HF cache at `~/tt-xla/.cache/hf`
- No inline scripts — port goes in `experiments/`
- Plan first (this file), test correctness before perf
- Commit after each phase passes its gate

## Verification gates

| Phase | Gate |
|---|---|
| 0 | MoE regression root cause documented |
| 1 | First-2-layer cosine ≥ 0.99 AND 8/8 greedy token match AND coherent 200-token code generation |
| 2 | ≥ 8 tok/s steady state |
| 3 | bf16 + 32K context running across 4 chips |

## What I'm NOT doing

- No stepping stone via Qwen3-14B. We already have Qwen3-0.6B; the arch is proven.
- No prefill optimization beyond what's already in the prior MoE demo.
- No batch > 1 — single-user inference is the goal.
- No tool-calling / agent loop integration — just text generation.
- No distillation / LoRA — vanilla weights from HF.

## Open question

Should we proceed Phase 0 → 1 immediately, or wait for user sign-off on the plan?

Per the non-negotiables ("plan first, then act"), I'll wait for explicit "go" before starting Phase 0 microbenchmark. The plan file is the artifact of "plan first."
