# Model Bringup Plan (qb1 host)

## Context

We have a working JAX/PJRT backend on qb1 (94 tests green). Separately, we have an existing model-bringup track in `experiments/` covering Qwen2.5-0.5B → Llama-3.2-1B → Llama-3.2-3B → Llama-3.1-8B plus an MoE port. All previously validated on the now-disconnected `tenstorrent` host. We need to:

1. **Validate existing ports on qb1** — establish a baseline on the new device
2. **Research candidate new models** — what's worth bringing up in 2026
3. **Bring up at least one new larger/architecturally-interesting model**

The PJRT/JAX work continues in a separate agent. This plan covers only the direct-ttnn model bringup track.

## Inventory of existing ports

Per `REPRODUCE.md` and `experiments/`:

| Model | Exp | Prior tok/s | Size | Status |
|---|---|---|---|---|
| Qwen2.5-0.5B | 60 | 140 | 0.5B | port + trace + native RoPE |
| Llama-3.2-1B | 64 | 78 | 1B | port |
| Qwen3-0.6B | 66 | ? | 0.6B | port (recent) |
| Llama-3.2-3B | 67 | 34 | 3B | port |
| SmolLM3-3B | 68 | ? | 3B | port |
| Llama-3.1-8B | 73 | 19 | 8B | port + correctness |
| Qwen1.5-MoE-A2.7B | demos/generate_moe.py | 22.7 | 14B (2.7B active) | port |

Total: 7 models across 0.5B–14B sizes, dense + MoE.

## Phase 1: Smoke validation on qb1 (cheap)

Sync `experiments/` to qb1, run smallest first. We don't need full reproductions — just enough to confirm each port still works and hits within 10–20% of prior tok/s on qb1's Blackhole.

Order (smallest → largest):

1. **Qwen2.5-0.5B** (`experiments/60_native_rope_decode.py`)
   - Cheapest to download (~1 GB), fastest to run
   - Establishes the qb1 baseline
   - Expected: ~140 tok/s (will accept 120+)
2. **Qwen3-0.6B** (`experiments/66_qwen3_06b_port.py`)
   - Similar size, different arch — sanity check
3. **Llama-3.2-1B** (`experiments/64_llama32_1b_port.py`) — ~78 tok/s
4. **SmolLM3-3B** (`experiments/68_smollm3_3b_port.py`)
5. **Llama-3.2-3B** (`experiments/67_llama32_3b_port.py`) — ~34 tok/s
6. **Llama-3.1-8B** (`experiments/73_llama8b_instruct.py`) — ~19 tok/s
7. **Llama-3.1-8B correctness** (`experiments/76b_8b_correctness_check.py`) — cosine, top-1 match
8. **MoE** (`demos/generate_moe.py`) — Qwen1.5-MoE — ~22.7 tok/s

For each: record tok/s, first-token latency, correctness if reference exists. Drop into `research/qb1_model_validation.md`.

Stop conditions:
- If any model fails to load (e.g., ttnn API changed in 0.69 vs the 0.68 we used): fix the port, document the API delta.
- If tok/s degrades > 20% from prior numbers: dig in. Possibilities: kernel cache cold (run twice), different memory config, ttnn 0.69 perf regression.

## Phase 2: Research new candidate models

Target: identify 2-3 recent (2025-2026) models worth bringing up. Criteria:

- **Architecturally interesting** — not just another Llama clone. MoE, attention variants (GQA different ratios, SWA, hybrid SSM), long context, multi-modal.
- **Fits one Blackhole** — ≤ ~30 GB weights (8B bf16 = 16 GB; we have room for ~14B dense or larger MoE).
- **Open weights, English-capable** — so we can validate without translation pain.
- **Recent** — released or majorly updated in late 2025 / 2026 if possible.

Candidates I'm aware of (verify dates):
- **Qwen3 family** (Qwen3-1.7B, Qwen3-4B, Qwen3-14B) — newer than what we have
- **Llama-4 / Llama-4-Scout** — Meta's newer release (MoE variant?)
- **Phi-4** — Microsoft's newer reasoning model
- **Gemma 3** — Google's latest
- **Mistral Small 3.1** — recent Mistral
- **DeepSeek-V3 small variants** — MLA attention is unusual
- **Cohere Command R-7B / Aya 32B** — multilingual focus

Output: `research/candidate_models_2026.md` with arch summaries and effort estimates.

## Phase 3: Bring up at least one new model

Pick from Phase 2 research. Pattern from prior ports:

1. Write a port script in `experiments/8X_<name>_port.py`
2. Load weights via `huggingface_hub` + `safetensors`
3. Map weights to ttnn (per-layer, lazy load if > 8B)
4. Implement decode loop: embed → N×(attn + MLP) → norm → output
5. Test prefill cosine against a pure-numpy fp32 reference (per the `numpy_reference` rule)
6. Greedy decode for 60–100 tokens, verify coherent text
7. Add to REPRODUCE.md and a wiki entry

Decision points before coding:
- **Use existing primitives?** Most of our ports share rope/sdpa/mlp patterns. Reuse what's in `experiments/40_*.py` and friends, don't rewrite from scratch.
- **Quantization?** We've validated full bf8 weights — no harm in starting there.
- **Tracing?** Start eager for correctness, then trace once it's right.

## Non-negotiables reminders

- All execution via `ssh qb1` — never local
- Device 0 only — qb1 has 4 chips, we use 1
- No `/tmp` — caches go under `~/tt-xla/.cache/`
- No inline scripts — write files in `experiments/`, `pjrt_plugin/scripts/`, or a new dir
- Plan first, act later — this file is part of that habit
- Frequent commits — after each model validates, after each research doc

## Verification

Phase 1 done when:
- Each of the 8 model runs has a recorded tok/s and a verdict (PASS/INVESTIGATE) in `qb1_model_validation.md`
- At least Qwen2.5-0.5B, one Llama, and the MoE pass

Phase 2 done when:
- `candidate_models_2026.md` exists with 2-3 candidates ranked
- User has signed off on which one to bring up

Phase 3 done when:
- New model generates coherent text on qb1
- Correctness check (cosine > 0.997 prefill, OR token-level match for greedy) passes
- Port is committed and documented in REPRODUCE.md

## Risks

1. **HuggingFace downloads.** Large weights (8B = 16 GB). qb1 has 3.6 TB free — fine. Set `HF_HOME=~/tt-xla/.cache/hf` to keep them in-project.
2. **ttnn 0.69 vs 0.68 API drift.** The old host ran 0.68. We're on 0.69 now. Possible breaking changes in compute kernel config, trace APIs, op signatures.
3. **bf16 precision.** Already known issue from PJRT work. Cosine validation is the gate.
4. **Cold JIT cache.** First run of each model pays ~10s for kernel compile. Don't confuse this with model latency.
