# Qwen Target — Revised After Latest-Model Survey

Scraper (`pjrt_plugin/scripts/qwen_latest.py`) ran on qb1, scanned 200 most-recently-modified Qwen models, filtered out quantizations/SAEs/quantization mirrors. Three serious candidates emerged for "local coding/research/learning replacement for cloud LLMs":

## Candidates

### A. Qwen3-Coder-30B-A3B-Instruct (Dec 2025)

```
Total / active:   30.5B / 3.3B  (MoE)
Architecture:     STANDARD transformer (GQA + RoPE + SwiGLU + RMSNorm)
Layers / experts: 48 layers, 128 experts, 8 active
Hidden:           4096 (assumed; not explicit in card)
Context:          262K native
Release:          2025-12-03 — 5 months ago
Single-chip fit:  bf8 → ~30 GB → tight but fits one Blackhole
SWE-bench:        not benchmarked in card
HF:               Qwen/Qwen3-Coder-30B-A3B-Instruct
```

### B. Qwen3.6-35B-A3B (April 2026) — latest stable, HYBRID arch

```
Total / active:   35B / 3B  (MoE)
Architecture:     HYBRID  — Gated DeltaNet (linear attn) + Gated Attention
Layout:           10 × (3 × DeltaNet→MoE  +  1 × Attention→MoE)
Layers:           40 total (30 DeltaNet, 10 Attention)
Hidden:           2048
DeltaNet heads:   32 V / 16 QK,  head_dim=128
Attention heads:  16 Q / 2 KV,   head_dim=256
Experts:          256 total, 8 routed + 1 shared = 9 active
Multimodal:       YES (vision encoder; we'd skip vision)
Context:          262K native, 1M+ with YaRN
Release:          2026-04-24 — 2 weeks ago
Single-chip fit:  bf8 → ~35 GB → tight, may need careful KV-cache budgeting
SWE-bench:        73.4% (state of the art for open weights)
License:          Apache 2.0
HF:               Qwen/Qwen3.6-35B-A3B
```

### C. Qwen3-Coder-Next (Feb 2026) — coder-tuned hybrid, REQUIRES multi-chip

```
Total / active:   80B / 3B  (MoE)
Architecture:     HYBRID — same DeltaNet + Attention + MoE recipe as B
Layout:           12 × (3 × DeltaNet→MoE  +  1 × Attention→MoE)
Layers:           48 total
Hidden:           2048
DeltaNet / Attn:  same per-block shape as B
Experts:          512 total, 10 routed + 1 shared = 11 active
Context:          262K native
Release:          2026-02-03 — 3 months ago
Single-chip fit:  ❌ — bf8 is 80 GB, won't fit ~32 GB chip
Multi-chip fit:   bf8 across 4 chips = 20 GB/chip ✓  (perfect for "use whole quietbox")
SWE-bench:        70.6% Verified, 44.3% Pro
License:          Apache 2.0
HF:               Qwen/Qwen3-Coder-Next
```

## What "the latest" actually means here

- A is the **most-downloaded coder-specific** model (2.6M downloads, but 5 months old).
- B is the **most-popular newest model** (3.86M downloads, 1.7K likes — top of charts), general-purpose with strong coding.
- C is the **newest dedicated coder model** with the **hybrid arch that requires the whole quietbox**.

The user's goal — "local 32B-class for coding + scale to whole quietbox" — points directly at C, with B as the natural prerequisite (port the hybrid arch on one chip first, then scale).

## Architecture impact: Gated DeltaNet is NEW for us

We've never ported a linear-attention / state-space layer. Gated DeltaNet is:
- Linear-attention variant in the SSM family (think Mamba but with delta-rule gating)
- O(N) compute vs O(N²) for full attention — that's why 262K context is feasible
- Has a recurrent / state-update structure that's fundamentally different from softmax attention
- ttnn does not have a native `gated_deltanet` op — we'd need to compose from primitives

This is a **significant lift** that doesn't show up in option A. Honest estimate:
- DeltaNet implementation: 6-10 hours (decompose to primitives, validate cosine)
- Multi-chip TP: 8-15 hours
- Full B port end-to-end on single chip: 15-25 hours
- C port building on B: 5-10 additional hours (mainly more experts to wire)

## Decision matrix

| | Time on 1 chip | Time to multi-chip | Risk | Match to "latest" |
|---|---|---|---|---|
| A only | 8-12 hrs | n/a | low | mediocre (5 mo old) |
| A → TP scaling | 8-12 hrs | + 8-15 hrs | low-med | mediocre |
| **B on one chip** | **15-25 hrs** | (later) | **med-high** (DeltaNet new) | **excellent** |
| B → C on quietbox | + 5-10 hrs | + 8-15 hrs for B TP | high | **best** |

## My recommendation (revised)

**Go for B (Qwen3.6-35B-A3B) on one chip first, then C (Qwen3-Coder-Next) on the full quietbox.**

Why:
- It IS "the latest" — released 2 weeks ago, top of HF charts
- The hybrid DeltaNet architecture is the right thing to learn; C uses the same arch shape
- B fits one chip, so we can develop the DeltaNet implementation without simultaneously needing multi-chip TP
- Once B works single-chip + we have multi-chip TP, C is mostly more experts and more layers

The user's instinct — "go for the latest, build toward the whole quietbox" — is exactly this path.

**What I'd NOT recommend:** A as a quick-win-first-then-upgrade. A is a different architecture family entirely. Effort spent on A doesn't carry into B/C. Better to invest the time once on the architecture that scales.

## Cost-honesty section

The full B + C path on the full quietbox is ~3-4 weeks of focused work. The PJRT backend is "complete enough for Phase 5" but Phase 6 (PJRT ABI changes for device tensor persistence) would be another multi-week effort. **Don't take all three on simultaneously.** Phase 6 PJRT work and Qwen 3.6/Coder-Next bringup are roughly equal-effort tracks; I'd hold Phase 6 PJRT in reserve.

## Open question for the user

Three real branches now, picking up from my earlier A/B/C:

**Branch I:** A only (Qwen3-Coder-30B-A3B). Conservative, ~1-2 days, no DeltaNet, no architectural learning. Use as daily driver.

**Branch II:** B on one chip (Qwen3.6-35B-A3B). Latest model, learn DeltaNet, prove single-chip viability of hybrid arch. ~3-5 days. Stretch: add multi-chip TP after.

**Branch III:** B → C on quietbox (Qwen3.6 → Coder-Next, full TP). Build the dream system. ~3-4 weeks. End state: 80B model, 3B active per token, code/research/learning at home, fully utilizing all 4 chips.

I'd vote III if you've got the runway, II if not, I if you need it working tomorrow.
