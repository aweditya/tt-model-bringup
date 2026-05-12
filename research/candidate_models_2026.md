# Candidate Models for qb1 Bringup (May 2026)

## Context

We have ports for Qwen2.5-0.5B → Llama-3.1-8B (dense) and Qwen1.5-MoE (MoE). All validated on qb1. Phase 3 goal: bring up at least one new, recent, architecturally-interesting model.

## Hardware budget

One Blackhole P150: ~32 GB DRAM. Practical ceilings:
- **Dense bf16**: ~14B params (28 GB weights, leaves room for KV cache)
- **Dense bf8**: ~30B params (we've validated full bf8 with cosine > 0.999)
- **MoE bf8**: ~30B total params (must fit ALL weights, not just active)

So Llama-4-Scout (109B total, 17B active) is **out** — total weights are 109 GB, won't fit.

## Architecture landscape (mid-2026)

Surprising finding: most major open models in 2026 use the same recipe.
- **GQA** (Q heads >> KV heads, typically 4-8× ratio)
- **RoPE** with large theta (500k-1M) for long context
- **SwiGLU** MLPs
- **RMSNorm** (not LayerNorm)
- **Head dims** 64 or 128
- **Context** typically 32K → 128K-1M with YaRN scaling

Differentiation is in **training data quality**, **scale**, and **MoE structure**. The "architecturally new" angle for 2026 is mostly **MLA attention** (DeepSeek) or **Mamba/SSM hybrids** (some labs experimenting), neither of which has a fits-one-Blackhole open model.

## Candidates (all open weights, fit on one chip)

### A. Granite-4.1-8B — IBM, May 2026

```
Architecture: dense transformer, GQA 32/8, head_dim=128, 40 layers
Hidden:       4096, intermediate (MLP): 12800
Context:      131K tokens (32× more than Llama-8B's 32K)
License:      Apache 2.0
Size bf16:    ~16 GB; bf8: ~8 GB — both fit easily
HF:           ibm-granite/granite-4.1-8b
Released:     7 days ago (most recent stable)
```

**Why interesting:**
- Same parameter scale and arch family as our Llama-3.1-8B port — direct comparison
- 40 layers (vs Llama-8B's 32) — small perf comparison
- Trained on different data mix (IBM enterprise-leaning)
- Newest release of a maintained model family

**Effort estimate: ~3 hours.** Port = copy of Llama-8B code with layer count + theta change.

### B. Phi-4 — Microsoft, Dec 2024 (still current)

```
Architecture: dense transformer, GQA (specifics not on model card), reasoning-focused
Total:        14B parameters
Context:      16K tokens
License:      MIT
Size bf16:    ~28 GB; bf8: ~14 GB
HF:           microsoft/phi-4
Released:     2024-12-12
```

**Why interesting:**
- Largest dense model we'd have running (1.75× our current 8B)
- Trained on heavy synthetic-reasoning data — different quality profile
- Smaller context (16K) is honest about its strength
- Useful baseline for "what does 14B dense actually buy?"

**Effort estimate: ~5 hours.** Need to inspect config.json on HF for GQA ratios, head_dim, vocab. Likely similar to Llama-8B architecture.

### C. Qwen3-8B — Alibaba, mid-2025

```
Architecture: dense, GQA 32/8, head_dim=128, 36 layers
Hidden:       4096
Context:      32K native, 131K with YaRN
License:      Apache 2.0
Size bf16:    ~16 GB
HF:           Qwen/Qwen3-8B
Released:     2025
```

**Why interesting:**
- "Thinking mode" toggle is uncommon in open weights
- Same architecture family as Qwen3-0.6B (which we already have!) — easy port
- Apache 2.0
- Modern training cutoff

**Effort estimate: ~2 hours.** Port = Qwen3-0.6B with size bump.

### D. Granite-4.1-30B (or 3B for completeness)

```
Granite-4.1-30B: 29B dense, bf8 fits (~15 GB)
Granite-4.1-3b: 3B dense, 40 layers, head_dim=64
```

3B isn't interesting — we have Llama-3B already. 30B is interesting but requires bf8 + careful memory layout. Not the easiest first new bringup.

## Models we explicitly skip

- **Llama-4 family** (Scout 109B, Maverick 400B): too big, MoE total weights exceed Blackhole DRAM
- **DeepSeek-V3 / R1**: 671B, way out of budget
- **DeepSeek-R1-Distill-Llama-8B**: it's a Llama-3.1-8B with different weights — we already have the architecture
- **Gemma 3**: similar to Llama arch, less interesting given what we have
- **Mistral Small 3.1**: nice model but standard arch
- **SAE-Res-Qwen3.5-27B**: interpretability research, base model is fine but SAE wrappers don't add value for us

## Recommendation

**Pick one of A or B.**

- **A (Granite-4.1-8B)** is the safe play: same size as our 8B, ~3 hours port time, direct quality comparison vs Llama, validates we can do "another 8B" cleanly.
- **B (Phi-4)** is the stretch: 14B dense pushes our memory budget, validates that the workflow scales beyond what we've done, gives a different quality story (reasoning-focused training).

If we have time after one, the other is a natural follow-up.

My personal lean: **B (Phi-4)** first, because it actually tests something new (size, training paradigm) vs just doing another 8B. But A is fine if we want to stay conservative.

## Open question for the user

Which one (A or B, or another)? Once chosen, I'll write a port spec and start.
