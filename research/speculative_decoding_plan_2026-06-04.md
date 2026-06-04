# Speculative decoding feasibility — Qwen 3B draft + Qwen3.6-35B-A3B target

Author: agent research draft. Status: feasibility / go-no-go. Not a build doc.

## 0. The tweet (NOT FETCHED)

The user pointed at `https://x.com/i/status/2061867450413773043`. WebFetch was
denied for that URL in this session, so the tweet content is **not directly
read**. Based on the user's framing ("small-model speculative-decoding scheme",
"Qwen 3B finetune", "35B-A3B target"), the tweet is almost certainly riffing on
the standard Leviathan-style **speculative decoding** pattern: pair a fast small
"draft" model with a slow large "target" so the target only does one forward
per K accepted tokens. Variants in the air (Medusa heads, EAGLE-2/3, lookahead
decoding) are minor refinements of the same idea — the core economics are the
same. **This doc assumes vanilla speculative decoding; if the tweet is actually
EAGLE or Medusa the conclusions about determinism and memory still hold, the
draft-model bringup item changes.** Re-run WebFetch with permission before
committing to a specific variant.

## 1. Speculative decoding 101

Given a target model `T` and a faster draft model `D` (same tokenizer):

- `D` autoregressively proposes K candidate tokens given the running context.
  Cost: `K · t_D` (K cheap forwards).
- `T` runs ONE forward at batch `B = K+1` over the K candidate tokens (and the
  preceding accepted prefix), producing K+1 logit distributions in parallel.
  Cost: ~`t_T` (one expensive forward, batched).
- Compare distributions. With **greedy** sampling, accept the longest prefix
  where `argmax(T_logits[i]) == draft[i]`; emit the first disagreement as `T`'s
  argmax and discard the rest. With **temperature sampling**, use the
  Leviathan rejection-sampling rule (accept with prob `min(1, p_T/p_D)`) which
  is provably distribution-preserving.

Net: instead of one accepted token per `t_T`, you get `N_acc + 1` tokens per
`t_T + K · t_D`. Speedup `≈ (N_acc + 1) / (1 + K · t_D / t_T)`. At
`t_D / t_T ≈ 0.1` and acceptance rate `α ≈ 0.7` with K=5, expected accept
length `(1 - α^{K+1}) / (1 - α) ≈ 2.7`, so speedup `≈ 2.7 / 1.5 ≈ 1.8×`.
Empirically published numbers for matched draft/target families land 2–3×.

## 2. Why Qwen-3B + Qwen3.6-35B-A3B is the natural pair

- **Same tokenizer family** (Qwen2Tokenizer / Qwen3 tokenizer for the 3.6
  generation): vocab is byte-identical or near-identical, so draft proposals
  index the target's vocab directly with no re-tokenization step.
- **Same chat template structure** (`<|im_start|>...<|im_end|>`,
  thinking-block convention): the prefix-cache-friendly preserve-thinking
  rendering we already ship for Qwen3.6 ([[feedback-qwen36-preserve-thinking]])
  carries over.
- **A3B = 3B active params per forward**: a 3B dense draft has a *natural*
  prior on what the MoE's per-token computation will produce — the active-param
  budget matches almost exactly, so the draft is closer to T than a generic
  3B model would be. Acceptance rate should sit toward the high end (0.6–0.8)
  rather than the low end (0.3–0.5).
- **Concrete candidate**: `Qwen/Qwen3-3B` if it exists, else
  `Qwen/Qwen2.5-3B-Instruct` (released, ~6 GB bf16). Need to confirm tokenizer
  exact-match before relying on shared vocab; if a special-token offset differs,
  add a 1-line ID remap.

## 3. The bf16 non-determinism problem — the actual blocker

Per [[35b-needle-haystack-2026-06-04]] and the catalog in
`research/35b_determinism_2026-06-04.md`: the **35B free-run greedy decode is
not bit-reproducible**. Identical prompt, identical seed, identical code →
~50% retrieval flip rate across runs. Root cause: bf16's 7-bit mantissa +
non-associative parallel reductions (40-layer chain, all_reduce across 4
chips, multicore argmax) produce 1-ULP logit perturbations that flip near-tie
argmaxes. This is the same mechanism Liu et al. (2026, arXiv 2506.09501v2)
measure as 99.6–100% example-level divergence under bf16 greedy.

**Why this kills naive spec-dec:**

- Greedy spec-dec accepts a draft token iff `draft[i] == argmax(T_logits[i])`.
- If `argmax(T_logits[i])` itself flips between equivalent runs, then on the
  K=5 candidates the target verification is **stochastic at the argmax level**.
  Acceptance rate becomes a function of the target's own noise floor, not the
  draft's quality.
- Worse: temperature-sampling spec-dec (Leviathan rejection) depends on the
  *probability* of the proposed token under T, which is also a 1-ULP-noisy
  quantity. The math still works (it's distribution-preserving), but you give
  up the "guaranteed identical to standalone T" property — which is the entire
  reason teams ship spec-dec instead of just sampling from D.

**Mitigations (from `research/35b_determinism_2026-06-04.md`, ranked by ROI):**

| Fix | Cost | Effect on spec-dec |
|---|---|---|
| A. Deterministic argmax tie-break by lowest token-id (host side) | ~20 LOC | Removes argmax-flip from 1-ULP logit drift. Required. |
| B. `ttnn.argmax(use_multicore=False)` on lm_head | 1 char | Removes one parallel-reduction source. Required. |
| D. `fp32_dest_acc=True` on lm_head matmul only | half a line | Reduces ULP noise at the surface where it matters most. Required. |
| C. Stabilize `all_reduce` reduce-order | unknown, needs Tenstorrent | Would remove the dominant interior source. Nice-to-have. |

A+B+D are essentially free, are recommended in the determinism plan
**anyway** for the chat product, and are a hard prerequisite for spec-dec. The
all_reduce question (C) is the only open one; if the per-step E1/E2 probes in
the determinism plan show that A+B+D collapse the trial-flip rate to ≤5%, we
can ship spec-dec. If they don't, we need C before spec-dec is worth building.

**Order of work**: ship A+B+D from the determinism plan FIRST. Measure the new
trial-flip rate. Only then begin spec-dec build.

## 4. Memory budget on the (1,4) P150 mesh

Per `[[feedback-p150-memory-bandwidth-measured]]`: each P150 has 31.83 GB DRAM
(roughly 30 GB usable after the trace/L1/runtime overhead) and 1.39 MiB L1/core.

**35B-A3B current footprint** (from `research/35b_cb_bringup_plan.md` and the
27B/35B memory layout in `27b_cb_scope.md`):
- weights bf16-ish ≈ 70 GB total → **~17.5 GB/chip**
- KV + DN state + activations at B=1..2 ≈ 1–2 GB/chip
- Working footprint **~19–20 GB/chip**, leaving ~10 GB/chip headroom.

**Qwen-3B dense draft** at bf16:
- 6 GB total weights → 1.5 GB/chip if sharded the same way, OR 6 GB on one
  chip if replicated.
- KV at MAX_POS=8192 B=1: ~100 MB/chip — negligible.

**Option α — co-resident on the mesh (1,4)**:
  ~19 GB (target) + ~1.5 GB (draft sharded) + ~10 GB free for trace + KV +
  longer-context = **fits comfortably**. Both models behind one HTTP server.
  The mesh bandwidth is shared, but draft forwards are tiny; the draft adds
  maybe 8–10 ms/step on top of the target's 80 ms/step.

**Option β — draft on host CPU**:
  3B bf16 needs ~6 GB host RAM. Modern x86 (qb1 has ample RAM) decodes a 3B
  model at ~5–15 tok/s eager, ~15–30 with llama.cpp Q8/Q4. Latency overhead
  per draft token includes a host↔device sync after each draft batch. Adds
  PCIe round-trip cost (~1 ms) per K-token draft batch. Plausibly workable;
  more complex than α and worse on raw draft throughput.

**Option γ — draft on a dedicated chip of the mesh**:
  Drop TP from (1,4) to (1,3) for the target, keep one chip for the draft
  unsharded. Wastes one chip's compute for the target; bad trade unless we
  hit memory pressure that α can't resolve. Avoid.

**Recommendation**: option α (co-resident, draft sharded). Confirm at bringup
time that the draft sharded across 4 chips runs at acceptable latency — even
at 4× slower than single-chip draft it's still 5–10× faster than target.

## 5. What it takes to bring up Qwen-3B on TT

Best case: Qwen3-3B is a dense (non-MoE) Qwen3-family model. Architecture is a
**strict subset** of Qwen3.6-27B (which is already in production as
`server_tp.py`): same RMSNorm flavor (with the `+1.0` zero-centered fix per
[[feedback-qwen36-qnorm-knorm-zero-centered]]), same q_norm/k_norm pattern,
same GQA structure, same RoPE. Differences vs 27B are size-only:
`n_layers`, `hidden_size`, `intermediate_size`, `n_heads`.

**Build path**: fork `server_tp.py` → `server_qwen3b.py`, swap the config
constants, follow `research/model_bringup_recipe.md` for the v0.1 → v2 ladder
(per-layer cosine ladder against a numpy oracle, then teacher-forced, then
free-run). Reuse `experiments/utils/hf_reference_*` patterns. Eligible for
the existing dev-harness pattern ([[reference-gm4-dev-harness]]).

**Effort**: 1–2 days following the recipe, assuming Qwen3-3B is dense (no DN,
no MoE). If it's a *hybrid* (DN layers like 27B), add a day for re-validating
the DN code paths at the smaller shape. If MoE, drop the draft idea and pick
a pure-dense alternative — MoE draft buys us little since the target's MoE is
the whole point of "A3B" being a meaningful match.

Skip MM1 (`TT_BACKEND` selector) and MM5 from `multi_model_serving_plan.md`?
**No** — MM1 is the right place to wire the draft. Add a `DRAFT_BACKEND=3b`
env that loads a second `base` module alongside the target.

## 6. Architecture sketch

```
┌──────────────────────────────────────────────────────────────┐
│  cb_api.py        (one FastAPI process; chat endpoint)        │
│         │                                                      │
│         ▼                                                      │
│  spec_dec_scheduler  — extends cb_scheduler                    │
│         │                                                      │
│         ├── draft_step(slot, K)  → K tokens from D            │
│         ├── target_verify(slot, K) → 1 fwd at B=K+1 on T      │
│         └── accept_or_correct → emit accepted + correction    │
├──────────────────────────────────────────────────────────────┤
│  T = server_35b_ttnn  (loaded sharded on (1,4))                │
│  D = server_qwen3b    (loaded sharded on (1,4))                │
│         both share one mesh_device handle                      │
└──────────────────────────────────────────────────────────────┘
```

Per-iteration spec-dec step for one slot:

1. **Draft K=4 or 5 tokens** through D's normal autoregressive decode (re-using
   D's traced B=1 forward).
2. **Target verify**: run T at `B = K+1`, **feeding** the prefix-end token +
   the K draft tokens, gather K+1 argmaxes.
3. **Accept-or-correct**: walk the K positions, accept while
   `argmax(T_logits[i]) == draft[i]`. Emit accepted tokens + T's argmax at the
   first mismatch (or T's K+1-th argmax if all K accepted).
4. **Rewind D's KV** to the accepted length. **Update T's KV** to the accepted
   length (paged SDPA + DN state). The DN-state rewind is the genuinely tricky
   bit — DN state is non-checkpointable cleanly ([[project-prefix-caching-design]]),
   so D and T both need the slot's DN to be at the right step. Mitigation: only
   support spec-dec on slots whose target is the dense-only model (Qwen-3B
   draft + an eventual dense target), OR run the K draft tokens with a
   throwaway D forward that doesn't commit state, then commit only the accepted
   prefix.

## 7. Expected gain

Current numbers (from `HANDOFF.md`-style references in MEMORY):
- 35B B=1 traced step: **~81 ms/tok** ([[feedback-35b-perf-2026-05-27]]).
- 35B HTTP B=1 through the CB engine: **~3 tok/s** (different ceiling — CB
  engine + sampling overhead — see [[feedback-dev-harness-vs-cb-engine-gap]]).

Spec-dec multiplies the **dev-harness target rate**, not the HTTP rate
(which has its own unrelated overhead). So the model is:

```
spec_step_ms ≈ t_T + K · t_D
            ≈ 81 ms + K · 10 ms       (D = Qwen-3B sharded, traced; estimate)
expected_tokens_per_step ≈ (1 - α^{K+1}) / (1 - α)
```

| K | α    | step_ms | accept_len | tok/s    |
|---|------|---------|-----------|----------|
| 3 | 0.7  | 111     | 2.17      | 19.5     |
| 5 | 0.7  | 131     | 2.74      | 20.9     |
| 5 | 0.5  | 131     | 1.94      | 14.8     |
| 5 | 0.8  | 131     | 3.36      | 25.7     |
| 7 | 0.7  | 151     | 3.10      | 20.5     |

Sweet spot K≈5 at α≈0.7 → **~21 tok/s** dev-harness target rate, vs ~12 tok/s
baseline (81 ms = 12.3 tok/s). **~1.7×** at α=0.7; **~2.1×** at α=0.8. A 3B
draft on a 3B-active-MoE *should* land toward 0.8 if the draft is fine-tuned
on data resembling the target's; if it's a vanilla off-the-shelf 3B then 0.6
is more honest. **Realistic claim: 1.5–2.0× decode throughput.**

Numbers >3× in some published spec-dec results come from much larger
`t_T/t_D` ratios (70B/7B or 175B/7B), not 35B/3B. With A3B-active our ratio is
weaker than it looks on paper.

## 8. Risks

1. **B>1 empty-slot poisoning on 35B (task #162,
   [[feedback-35b-batched-forward-empty-slot-poison]]).** Spec-dec verify
   requires `B = K+1 ≥ 2`. The current 35B CB path crashes / produces garbage
   at B>1 because empty slots contaminate populated slots' outputs. The
   default `TT_CB_SLOTS=1` workaround is incompatible with spec-dec by
   construction. **This is a hard prerequisite.** Estimated fix: half a day
   (per the task note; mirror the 27B masked-multiply reset pattern).
2. **bf16 non-determinism** (§3). A+B+D collapse the surface symptom; if
   trial-flip rate doesn't drop, spec-dec is borderline.
3. **DN state rewind on rejection**: the GatedDeltaNet recurrent state H_t
   is updated in place by the recurrence. After a K-token draft is partially
   accepted (say 2/5), T's DN state has *already* been advanced by 5 steps in
   the verify forward, but we only want to commit 2 of those steps. Need to
   either (a) verify-without-commit (write to a scratch DN-state, copy back
   only the accepted prefix), or (b) re-run T for exactly the accepted-length
   on its real DN state. (b) is simpler but loses ~half the speedup. (a)
   requires a scratch H_t allocation per slot — ~18 MB/chip per slot at 35B
   (per `27b_cb_scope.md` math), trivial.
4. **Mesh-bandwidth contention between T and D forwards**. D is sharded the
   same way as T, so D's matmuls hit the same DRAM. Unlikely to be the
   bottleneck (D weight footprint ≪ T), but measure rather than assume.
5. **Trace capture compatibility**. Both T and D need their own captured
   decode traces, plus an extra `B=K+1` capture for T's verify forward.
   Three traces × ~6 GB trace_region_size each = manageable but watch the
   mesh memory budget.

## 9. Order of operations (if greenlit)

1. **Ship determinism fixes A+B+D** from `35b_determinism_2026-06-04.md`
   (~2 hours, reversible behind env flag). Measure trial-flip rate. **Go/no-go
   gate**: if rate ≤5%, proceed. If not, fix C first or skip the spec-dec
   investment.
2. **Fix task #162** (35B B>1 empty-slot poisoning). ~half day. Without
   this, spec-dec cannot run.
3. **Bring up Qwen-3B as `server_qwen3b.py`** following the bringup recipe.
   1–2 days. Gate: free-run greedy matches HF for 100 tokens at 3 different
   prompts.
4. **Add `DRAFT_BACKEND` env** to cb_api / cb_engine. Load both T and D on
   the same mesh. Smoke: independent chat through T or D works as today.
   Half a day.
5. **Build `spec_dec_scheduler`** as a thin wrapper over cb_scheduler.
   Reference: vLLM's `spec_decode/` package, or the Leviathan paper Algorithm 1.
   ~2 days for correctness; the verify-without-commit DN-state path is the
   most fiddly piece.
6. **Measure** end-to-end: dev-harness tok/s, expected 1.5–2.0× over baseline.
   If the number disappoints, profile draft acceptance rate at varying K and
   acceptance-mode (greedy vs Leviathan).

## 10. Decision frame

This is **3–5 days of build** with realistic 1.5–2.0× decode throughput on
35B, contingent on two prerequisite tasks (determinism A+B+D, task #162)
that have independent value (chat reproducibility, B>1 unlock). The
determinism fix is essentially free and ships regardless. Task #162 is on
the roadmap anyway.

The honest read: spec-dec is **a reasonable next throughput lever** but not
the highest-ROI item on the board if we still have CB-engine-overhead recovery
sitting at the table (HTTP B=1 is 3 tok/s vs dev-harness 12 tok/s — a 4×
HTTP-path bug that spec-dec does NOT fix). Order-of-operations question for
the user: should we close the HTTP overhead gap first (cheaper, larger
absolute win on the user-visible path), then do spec-dec on top? That
sequencing is probably correct.

## Related memory / files

- `research/35b_determinism_2026-06-04.md` (§3 mitigations A/B/C/D)
- `research/27b_cb_scope.md` (B>1 batched forward, paged SDPA, DN-state slots)
- `research/35b_cb_bringup_plan.md` (35B CB bringup, task #162)
- `research/model_bringup_recipe.md` (Qwen-3B bringup ladder)
- `research/multi_model_serving_plan.md` (MM1 `TT_BACKEND` selector)
- [[35b-needle-haystack-2026-06-04]] — trial-flip data
- [[35b-batched-forward-empty-slot-poison]] — task #162
- [[feedback-bf16-chain-drift-at-B-gt-1]] — bf16 precision floor
- [[feedback-qwen36-preserve-thinking]] — chat-template parity for shared
  draft/target rendering
- Leviathan et al. 2023, *Fast Inference from Transformers via Speculative
  Decoding*, arXiv 2211.17192 — the Algorithm-1 spec we'd fork.
- Liu et al. 2026, *Give Me FP32 or Give Me Death?*, arXiv 2506.09501v2 — the
  determinism literature underlying §3.

## Honest limits

- The tweet was not read (WebFetch denied). If it's actually EAGLE-2 or
  Medusa-style self-speculation (draft head bolted onto the target), the
  bringup item in §5 changes from "stand up a separate 3B" to "train and
  attach draft heads to 35B" — much larger effort.
- Acceptance-rate estimates (0.7–0.8) are folklore for matched-family
  draft/target pairs. Our specific 35B-A3B / Qwen-3B pairing has no public
  measurement; the realistic α is whatever a 4-prompt benchmark shows.
- The 81 ms/tok dev-harness step time is from 2026-05-27; current numbers
  may have shifted. Re-baseline before claiming a spec-dec speedup.
- We have not modeled batched spec-dec across multiple slots concurrently.
  The single-slot model in §7 is conservative; multi-slot would compound the
  CB win and the spec-dec win, but the scheduler complexity also compounds.
