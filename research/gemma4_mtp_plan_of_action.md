# Gemma 4 12B IT spec-dec — plan of action (review before building)

Status: **PLAN, awaiting greenlight**. Author: 2026-06-07 session.
Companion: `research/gemma4_mtp_design.md` (feasibility scoping, eb014f5).

This doc converts the design into a phased build plan with discrete
gates. **Read § "Reconvene checkpoint" before greenlighting** — there
are 3 open decisions the user should confirm.

---

## TL;DR

- **Target**: Gemma 4 12B IT (production at 47 ms/tok traced, qb2).
- **Drafter**: `google/gemma-4-12b-it-assistant` (0.4B, Apache 2.0, ungated).
  Vanilla Leviathan small-draft spec-dec — NOT a DeepSeek-V3 MTP-head pattern.
- **Pattern**: spec-dec scheduler wrapper around `cb_scheduler` (Path A,
  Path B rejected — no MTP-head weights ship for 12B IT).
- **Precedent**: DeepSeek-V3 demo at `~/tenstorrent/tt-metal/models/demos/
  deepseek_v3/tt/mtp.py` (419 lines) + `tt/generator.py` (2977 lines) —
  two captured traces + aliased page-table + accept-walk. Pattern is
  reusable; "predict trace = MTP head forward" → "predict trace = drafter
  forward".
- **Estimate**: 7 days build + 1 day buffer = **8 days** of focused work.
- **Realistic gain**: 1.7–2.2× decode tok/s at α≈0.7 (Google quotes 3× = α≈0.85).
- **Host execution**: qb2 (Gemma 4 12B prod path lives there). qb1 stays
  on Nemotron-3 work.

---

## Reconvene checkpoint — confirm before building

### Q1. Drafter feasibility — read transformers code first?

The drafter has `model_type=gemma4_assistant` with FOUR unusual fields:
`use_ordered_embeddings`, `num_centroids=2048`,
`centroid_intermediate_top_k=32`, `backbone_hidden_size=1536` (distinct
from `hidden_size=256`). This is a **centroid-based vector-quantized
embedding** — not a vanilla transformer.

Three outcomes possible after reading `Gemma4AssistantForCausalLM` in
the transformers source:
- (a) Centroid is a lookup-table + projection → reproducible in ttnn
  cleanly. Greenlight v0.1 bringup.
- (b) Centroid is a learned VQ with custom kernels → bring up offline
  with dequantized weights (loses some draft speed but preserves
  correctness). Still doable, 1 day longer.
- (c) `Gemma4AssistantForCausalLM` not in transformers main → drafter
  is uninstantiable until upstream releases. **HARD STOP** until then.

**Recommendation**: Phase 0 starts with a 4-hour code-read gate. If (c),
park the whole effort and pivot back to Nemotron-3 perf. If (a)/(b),
proceed to Phase 1.

**Decision needed**: confirm we run the v0.0 feasibility gate before
committing to days 2-8 of work, OR ship without the gate (risk wasted
days if it's (c)).

### Q2. Determinism prerequisites — port A+B+D from 35B first?

`research/35b_determinism_2026-06-04.md` documents three fixes that
shipped on 35B and gave deterministic argmax:
- A: host-side deterministic argmax tie-break
- B: `argmax(use_multicore=False)`
- D: `fp32_dest_acc` on lm_head matmul

Spec-dec acceptance needs `argmax(target_logits[i]) == draft[i]` to
match deterministically. Without these fixes, bf16 ULP drift across the
48-layer chain can flip near-tie argmaxes → ~5% lower α (lost speedup).
Wins are free regardless of spec-dec (better Gemma 4 reproducibility).

**Decision needed**: port A+B+D to Gemma 4 12B as Phase 0 step 1 (~2 hours),
OR proceed without (accept ~5% α floor, ship faster).

### Q3. qb2 device ownership

qb2 currently has the **Gemma 4 perf agent's `gm4` tmux session active**
(holding all 4 P150 PCIe locks per the RULER agent's report). We can't
spec-dec on qb2 while perf work is running there.

**Decision needed**: when do we want to start? Three options:
- (a) Pause Gemma 4 Round 10+ perf work, free qb2 for spec-dec —
  prioritize spec-dec speedup over the next eager-perf rounds
- (b) Wait for perf work to wind down naturally (uncertain timing)
- (c) Build the drafter on qb1 (currently on Nemotron-3 work). qb1 free
  for Gemma 4 work after Nemotron-3 v0.5.bench finishes. Drafter is small
  enough to share the mesh with Nemotron-3 if needed (0.4B vs 23GB
  Nemotron-3 weights — fits).

**Recommendation**: (a) for fastest path; (c) if Gemma 4 perf is
strategically still live. Need user call.

---

## Phase 0 — feasibility + prereqs (~0.5 day)

| Step | Time | Output | Gate |
|---|---|---|---|
| **0.A** Read `Gemma4AssistantForCausalLM` in transformers source | 4 h | feasibility verdict (a/b/c) | **GO/NO-GO**: if (c), STOP |
| **0.B** Port determinism A+B+D to Gemma 4 12B (optional per Q2) | 2 h | gemma4 produces deterministic argmax across runs | trial-flip rate ≤ 0.5% across 100 prompts |
| **0.C** Re-baseline Gemma 4 12B decode tok/s (post-P1 vocab-shard etc.) | 30 min | current ms/tok number | replaces stale 47 ms/tok claim |

**Phase 0 exit criteria**: feasibility verdict known, determinism rate
floor measured, current target perf baseline recorded.

## Phase 1 — drafter bringup (~3 days)

Forks `[[reference-model-bringup-recipe]]` — the v0.1→v2 ladder that took
Gemma 4 12B from oracle to HTTP chat in ~36 hours. Drafter is smaller
(0.4B vs 12B) → bootstrap is fast; main cost is the unfamiliar centroid
forward.

| Stage | Adds | Gate | Time |
|---|---|---|---|
| v0.0 | HF oracle (5-prompt artifacts), tokenizer probe, weights introspect | argmax @ pos 0 matches HF | 4 h |
| v0.1 | Bootstrap on (1,4) qb2 mesh; embed + final_norm + lm_head | embed cos ≥ 0.999 vs HF | 4 h |
| v0.2 | Full 4-layer forward + argmax matches HF | argmax_last matches HF on 5 prompts | 6 h |
| v0.3 | Multi-step decode + trace capture | 100 traced steps == 100 eager token-for-token | 6 h |
| v0.4 | Drafter v0.5 perf — measure ms/tok at B=1 | ms/tok floor measured | 2 h |

**Phase 1 exit criteria**: drafter runs end-to-end at B=1, produces same
tokens as HF reference greedy on 100-token prompts. Trace captured + 100
traced replays match eager. Drafter forward time measured.

## Phase 2 — target B=K+1 verify trace (~1 day)

Currently `server_gemma4_unified_cb.py` captures only B=1. Spec-dec verify
needs a B=K+1 forward where K+1 logical batch rows alias onto one
physical KV slot.

| Step | Adds | Gate | Time |
|---|---|---|---|
| 2.A | Probe `experiments/cb/isolate/gemma4_verify_kp1_smoke.py` — B=K+1 forward in isolation | K+1 logits ≈ K+1 independent B=1 forwards (cos ≥ 0.999 per row) | 3 h |
| 2.B | Fork `_build_verify_alias_page_table_host` from DeepSeek-V3 `tt/generator.py:43-101` | aliased page-table reads same physical slot K+1 times | 2 h |
| 2.C | Capture B=K+1 trace in `server_gemma4_unified_ttnn.py` alongside existing B=1 (two-phase warmup per `[[ttnn-multi-trace-two-phase-warmup]]`) | both traces capture without TT_FATAL; replays produce matching logits | 3 h |

**Phase 2 exit criteria**: target server has TWO captured traces (B=1
decode + B=K+1 verify); K+1 verify produces logits equivalent to K+1
independent B=1 forwards.

## Phase 3 — spec-dec scheduler (~1.5 days)

Forks Leviathan Algorithm 1 + DeepSeek-V3's accept-walk machinery.

| Step | Adds | Gate | Time |
|---|---|---|---|
| 3.A | `spec_dec_scheduler.py` skeleton — wraps cb_scheduler step interface | drives one model's forward through existing path | 2 h |
| 3.B | Accept-walk core — `draft_step(slot, K) → K tokens`, `target_verify(slot, K) → K logits`, host-side argmax compare, emit accepted prefix + 1 correction | greedy-equivalent: spec-dec output token sequence == B=1 target output on 10 prompts | 5 h |
| 3.C | KV cache rewind on rejected drafts — fork DeepSeek-V3 `_MtpDecodeLoopResult` walk | accept_rate + accept count match Algorithm 1 reference | 3 h |
| 3.D | Dev-harness bench — measure α (acceptance rate) + tok/s gain at K∈{3,5,7} | α ≥ 0.6 + measured gain ≥ 1.4× at one K | 2 h |

**Phase 3 exit criteria**: spec_dec_scheduler runs through dev-harness,
produces greedy-equivalent output to plain B=1, α ≥ 0.6, gain ≥ 1.4×.

## Phase 4 — HTTP wire-up + first prod measurement (~0.5 day)

| Step | Adds | Gate | Time |
|---|---|---|---|
| 4.A | `cb_api.py` BACKENDS dict: `gemma4_spec → (target, drafter)`; `DRAFT_BACKEND` env wiring | server starts with both models loaded | 1 h |
| 4.B | `cb_scheduler.py`: route `TT_BACKEND=gemma4_spec` to `spec_dec_scheduler` | scheduler dispatches to spec-dec path | 1 h |
| 4.C | Bench: `curl /v1/chat/completions` at K∈{3,5,7}, log α + ms/tok | end-to-end HTTP works; chat output coherent | 2 h |

**Phase 4 exit criteria**: spec-dec served via `/v1/chat/completions`,
measured α + tok/s gain logged.

---

## Cumulative gates (what success looks like at the end)

1. **Correctness**: 100-token greedy output identical to plain B=1 target
   (Leviathan Algorithm 1 guarantee).
2. **α**: ≥ 0.7 (matched-family draft + target precedent).
3. **Speedup**: ≥ 1.5× decode tok/s vs current Gemma 4 12B IT baseline (re-baselined post-P1).
4. **No-regressions**: existing Gemma 4 12B IT path (without DRAFT_BACKEND env)
   produces identical output + same perf.
5. **HTTP**: `curl /v1/chat/completions` at temperature=0 produces stable text.

---

## Open architectural decisions (locked at v0 only after build starts)

1. **Drafter mesh placement**: shared (1,4) with target, OR single-chip
   on chip 0. Single-chip is structurally simpler (no TP on drafter); shared
   may save trace memory. Decide at Phase 1 v0.1 after measuring drafter
   weight footprint.
2. **K (lookahead depth)**: defer to Phase 3 bench. Higher K = better
   amortized α per round but more rejected work. K=3 is conservative
   default; bench K∈{3,5,7}.
3. **Sampling**: HOST ONLY (DeepSeek-V3 precedent enforces this; matches
   `[[feedback-ttnn-topk-tie-break-drift]]`). Greedy temperature=0 only in
   v0; layer sampling-temperature later.

---

## Risks (priority-ordered, copied + condensed from design doc)

1. **Centroid embedding architecture** in drafter — feasibility gated at Phase 0.A.
2. **Tokenizer alignment** — `cmp tokenizer.json` between drafter + 12B IT at v0.0.
3. **B=K+1 verify trace** on Gemma 4 sliding+global attention — likely fine,
   verify at Phase 2.A.
4. **bf16 α floor** — port determinism patches (Phase 0.B) for ~5% α uplift.
5. **Prefix-cache interaction** — disable prefix-cache in v0 spec-dec; layer
   back on later (`[[feedback-prefix-cache-multiturn-miss-2026-06-04]]`).
6. **KV cache duplication** — drafter cache ~64 MB at 8K B=1; negligible.
7. **Trace region size** — bump to 150 MB (3× headroom for two large traces).

---

## What I'm asking the user to confirm before kickoff

1. **Q1 — feasibility gate**: run Phase 0.A 4-hour read-before-build, or skip?
2. **Q2 — determinism prereqs**: port A+B+D from 35B (free wins, ~2 h), or skip?
3. **Q3 — host**: pause Gemma 4 perf on qb2 to free the device, or build on qb1?

After confirmation, I'd start with **Phase 0.A** (the feasibility code-read)
on the chosen host. If it returns (a) or (b), I'd proceed through Phase 1.
If (c), I'd stop and reconvene with the user.

I'd ship per-phase commits with named gates (e.g.,
`drafter(v0.2): full 4-layer forward + argmax matches HF`) for easy review.
