# Branch III — Kickoff & Non-Negotiables Review

Goal: bring up **Qwen3.6-35B-A3B** on a single Blackhole, then scale to **Qwen3-Coder-Next** across all 4 chips of qb1. End state: a local coding/research/learning model running on hardware the user owns.

Estimated effort: 3-4 weeks of focused work, broken into 4 phases.

---

## Non-negotiables — explicit review for this branch

Re-reading the CLAUDE.md + persistent memory; these are the rules I will follow on this branch. If I ever appear to break one, point at this section.

| Rule | What it means here |
|---|---|
| **Plan first, act later** | Each phase below gets its own plan file *before* any implementation. No "I'll figure it out as I go." |
| **Research-driven** | Phase A is mostly research (read papers, read reference implementations, decompose DeltaNet on paper) before writing ttnn ops. |
| **No code bloat** | Reuse Qwen1.5-MoE port primitives where possible. New code only for the genuinely new parts (DeltaNet, multi-chip TP). |
| **Remote execution only — `ssh qb1`** | All code runs there. Local Mac is for editing + research notes. |
| **Single device first, then scale** | Phase B is single-chip. Phase C scales to 4. Don't try to do TP and DeltaNet at the same time. |
| **No inline scripts** | Every script is a file in `experiments/`, `pjrt_plugin/scripts/`, or `pjrt_plugin/tests/`. Every script is committed. |
| **No `/tmp`** | All caches under `~/tt-xla/.cache/` (HF, ttnn-tmp, ttnn-cache). Already set up. |
| **Frequent commits** | Every passing milestone gets a commit. Co-Authored-By line included. |
| **Numpy fp32 reference, not HF AutoModel** | Per `feedback_numpy_reference.md` — HuggingFace AutoModel crashes on remote. Write our own fp32 numpy forward for correctness checks. |
| **Cosine ≥ 0.99 before perf** | Per `feedback_correctness_first.md`. If first-2-layer cosine drops below 0.99, stop and ablate. |
| **Test with 60-100+ tokens** | Per `feedback_generation_limits.md`. Short generations hide quality issues. |
| **Compute kernel config: all-or-nothing** | Per `feedback_compute_kernel_config.md`. Don't mix HiFi4/HiFi2 across ops on Blackhole. |
| **Paged KV cache for traceable decode** | Per `feedback_paged_kv_cache.md`. Don't try to trace with Python-scalar position indices. |
| **Benchmark full decode loop** | Per `feedback_benchmark_methodology.md`. Don't time just the trace; from_dev adds 3.9ms. |
| **EOS sampling care** | Per `feedback_eos_sampling.md`. Test greedy first, then sampling separately. |
| **bf8 not yet validated past 8B** | Per `feedback_bf8_weights.md`, full bf8 was safe through Llama-8B. At 35B we re-validate with cosine checks at every 4th layer. |

---

## Master plan — 4 phases

### Phase A — Foundations (week 1, ~10-15 hrs)

**A1.** Read the Qwen3.6-35B-A3B technical report + config.json. Document the *exact* architecture in `research/qwen36_arch_notes.md`: every shape, every layer ordering, RoPE settings, normalization positions. No guessing. (~2 hrs)

**A2.** Investigate the MoE perf regression on ttnn 0.69 (Qwen1.5-MoE dropped from 22.7 to 15.7 tok/s). This is BLOCKING because Branch III is MoE-heavy. Write `experiments/81_moe_regression_microbench.py`. Either find the regression and document it, or accept the slower MoE baseline and move on. (~2-3 hrs)

**A3.** Implement & validate Gated DeltaNet in isolation. Write `experiments/82_gated_deltanet.py`:
  - Decompose to ttnn primitives (no native op exists)
  - Validate against a pure-numpy fp32 reference of the DeltaNet equations
  - Cosine ≥ 0.99 for a single layer with random weights & input
  - Benchmark dispatch cost  (~5-8 hrs)

**A4.** Implement & validate the Gated Attention layer in isolation. This is closer to what we've ported before (GQA + RoPE + sliding-window-like). Probably 2-3 hrs.

**A5.** Isolated MoE block (8 experts first to debug, then 256). Validate routing + shared-expert-gate against numpy ref. (~3 hrs)

**A6.** Parallel scan kernel for DeltaNet prefill (Blelloch / Heinsen). Decode uses 1-step recurrence; only prefill needs the scan. Start with chunked-serial (chunks of 64 tokens, parallel within chunk), upgrade to full Blelloch tree if needed. (~5-10 hrs)

**A7.** Multi-chip primitives. Open ttnn mesh device (2 chips first), validate all-gather + all-to-all primitives, replicated vs sharded weight load. Isolated tests, not yet wired to model. (~5-8 hrs)

Gate: A1-A7 all green, written up. THEN Phase B (which is multi-chip from day 1).

## Memory math drove a plan change

Re-did memory accounting after user push: bf8 Qwen3.6-35B is ~37 GB at 4K context — does NOT fit one Blackhole's ~30 GB usable DRAM. Choices:

- B-α: bf4 weights — fits 1 chip, but quality risk (not validated)
- B-β: 2-chip TP, bf8 — fits comfortably, validates multi-chip earlier
- B-γ: 4-chip TP, bf16 — best quality, most infra at once

**Choosing B-β.** Phase B is therefore multi-chip from day 1. Pre-builds the infrastructure for Phase C/D which use 4 chips for Qwen3-Coder-Next.

### Phase B — Single-chip Qwen3.6-35B-A3B (week 2, ~15-25 hrs)

**B1.** Write `experiments/83_qwen36_35b_a3b_port.py` shell with weight-loading skeleton. bf8 quantization on upload. (~3 hrs)

**B2.** Wire up the 10-pattern hidden layout: 30 DeltaNet-MoE layers + 10 Attention-MoE layers in the right interleave. (~2 hrs)

**B3.** MoE routing for 256 experts / 8 routed + 1 shared. Adapt from `demos/generate_moe.py` (60 experts / 4 routed in Qwen1.5). (~3 hrs)

**B4.** End-to-end forward + correctness gate. Numpy fp32 reference of first 2 layers (full 40-layer reference would OOM the host). First-2-layer cosine ≥ 0.99 vs reference. (~5 hrs)

**B5.** Greedy decode 8 tokens, match numpy reference exactly. (~2 hrs)

**B6.** First real generation: prompt = "Write a Python function that ...", 200 tokens, verify coherent code. (~1 hr)

**B7.** Trace capture for the static parts (attention + DeltaNet + shared layers). Routing remains dynamic per our prior MoE finding. (~3-5 hrs)

**B8.** Target perf: ≥ 8 tok/s steady-state on one chip. Stretch ≥ 15 tok/s. (~3 hrs)

Gate: B6 passes (real code generation works) + B8 hits ≥ 8 tok/s.

### Phase C — Multi-chip TP (week 3, ~10-15 hrs)

**C1.** Survey ttnn mesh device API on Blackhole. Read `~/tenstorrent/tt-metal/ttnn` (already cloned) for the actual API surface available in 0.69. Document sharding strategy in `research/qb1_multichip_tp.md`. (~3 hrs)

**C2.** Implement TP for attention + DeltaNet across 4 chips (hidden-dim sharded). (~5-8 hrs)

**C3.** Implement expert parallelism for MoE (each chip owns 64 of 256 experts; tokens routed across chips). (~3-5 hrs)

**C4.** Re-validate correctness across 4 chips. Cosine ≥ 0.99 must still hold. (~2 hrs)

**C5.** Move to bf16 (now that we have 4× the memory budget). Compare bf8 vs bf16 quality. (~1 hr)

**C6.** Performance target: ≥ 15 tok/s across 4 chips. Stretch ≥ 25 tok/s. (~2 hrs)

Gate: same prompt produces same-quality output as Phase B, ≥ 15 tok/s.

### Phase D — Scale to Qwen3-Coder-Next (week 4, ~5-10 hrs)

**D1.** Update config: 48 layers (12 patterns), 512 experts, 10+1 active. (~1 hr)

**D2.** Re-load weights (80B → 20 GB per chip in bf8). (~1 hr)

**D3.** Re-validate cosine on first 2 layers. (~2 hrs)

**D4.** Daily-driver test: real coding task ("rewrite this Python function to use asyncio"), evaluate against Claude's answer subjectively. (~1 hr)

**D5.** Document the final system in `research/quietbox_local_llm.md`. (~1 hr)

Gate: passes a real coding task. Subjective quality at least as good as Qwen-Plus / similar cloud-served Qwen.

---

## Decisions I want from you before starting Phase A

1. **Time budget.** This is a 3-4 week commitment. Are you good with that, or do you want me to stop at a particular checkpoint (e.g., end of Phase B = working single-chip latest model = good enough)?
2. **Vision encoder.** Qwen3.6-35B-A3B is multimodal. I plan to **skip the vision tower** entirely (text-only port). Confirm?
3. **Daily-driver expectations.** The end state is "local coding LLM I'd actually use." What's the bar? Just text completion? Tool-calling? Multi-turn chat? I'll default to "good single-turn coding completions" unless you say otherwise.

---

## Parallel track — PJRT background agent

User wants the PJRT plugin work to keep going in a background agent. Phase 5 is complete (Steps 1-7 all done). Next priorities for the PJRT agent:

1. **Vanilla tt-nn comparison benchmark** (NEW — per today's user feedback): build the exact same program in three forms and time them on qb1:
   - Hand-written native tt-nn (the baseline our PJRT is trying to beat)
   - PJRT eager-mode (our engine, no trace cache)
   - PJRT traced (our engine, with trace cache)
   - Programs: softmax, attention, single transformer layer
   - **This is the real validation: are we actually as fast or faster than vanilla tt-nn?**
2. **Phase 6 — PJRT ABI tensor lifetime** (longer term): ~150µs trace replay floor is currently dominated by numpy↔ttnn host transfer. Removing it requires keeping device tensors across PJRT calls.

I'll spawn the agent for #1 and let #2 wait until #1 lands.

---

## Open questions on the PJRT plugin's role

The user's frame: PJRT is useful → keep working on it. My honest current understanding: PJRT-traced is **at parity with vanilla tt-nn for simple programs** because they ultimately call the same ttnn ops. The unique PJRT value props are:

1. **Automatic dispatch elimination** for free (no manual trace setup by the user)
2. **Op fusion from StableHLO patterns** (Step 7 partially landed; full fusion not done)
3. **JAX ecosystem access** (Flax, Orbax, etc. "just work")

The comparison benchmark will tell us where PJRT actually wins and where it's a wash. That data will guide whether to invest in Phase 6.

---

## Status of this file

This file is the contract for Branch III. Edits should be **explicit edits** by the user (not "rewrite this from scratch"). Phase plans (A, B, C, D) get their own files as we approach them.
