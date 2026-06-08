# Gemma 4 12B IT spec-dec — plan of action (review before building)

Status: **Phase 1 SHIPPED 2026-06-07. Phase 2 next.**
Companions:
- `research/gemma4_mtp_design.md` (initial scoping, `eb014f5`)
- `research/gemma4_assistant_feasibility.md` (Phase 0.A DONE — outcome a, **scope dropped ~2d**)
- `research/gemma4_determinism_audit.md` (Phase 0.B DONE — B+D already shipped on Gemma 4)
- `research/deepseek_v3_alias_page_table_reference.md` (Phase 2.B fork source)
- `research/gemma4_verify_kp1_audit.md` (Phase 2.B.1 foreground audit, `e685c6f`)

## Architectural clarifications from 2026-06-07 spec-dec discussion

Captured in conversation; codifying here for durability:

1. **Drafter placement on (1,4) mesh**: weights REPLICATED across all 4 chips
   (760 MB/chip × 4 = 3 GB total, negligible vs target's 24 GB). Drafter's
   matmuls run independently per chip — no TP on the drafter trunk. lm_head
   is vocab-sharded (P22 pattern, 65536/chip). v0.2 shipped exactly this.
2. **"Parallel" in spec-dec is the verify step, not concurrent execution.**
   Drafter waits for target's hidden + KV (sequential dependency). The win
   is one B=K+1 target forward replaces K sequential B=1 forwards —
   amortizes fixed overheads + reads the same 24 GB of weights once
   instead of K times.
3. **Per-round wall budget**: target B=1 (~47 ms) + drafter B=1 (~5-10 ms) +
   target verify B=K+1 (~70-90 ms) + host accept walk (<1 ms) ≈ 125 ms for
   ~4.5 tokens at α=0.7 → **~28 ms/tok effective, 1.7× speedup over 47 ms baseline**.
4. **Three traces total** on the same mesh; trace_region_size bump 50 MB → 150 MB:
   - target single-step decode (exists)
   - drafter forward (Phase 1 v0.4)
   - target verify B=K+1 (Phase 2.B)
5. **KV layout risk** for Phase 2.A: target's KV is TP-sharded on NKV heads;
   drafter's cross-attention needs all-reduce. Path (a) — drafter does its
   cross-attention TP-sharded — is the right choice; path (b) — all-gather
   target's full KV onto every chip — is too expensive. Verify at Phase 2.A.0
   before committing to the refactor.
6. **Memory budget on (1,4) qb1**: ~9-10 GB/chip total (target ~6 + drafter ~0.76
   + KV ~1.6 + traces 0.15 + activations 1) out of 32 GB. 22+ GB headroom.
7. **Multi-CQ overlap is NOT useful** for single-stream spec-dec (every step
   depends on previous). Could become useful in CB if we want to overlap
   one request's draft with another request's verify; deferred past v1.
8. **Single-client v0, multi-client deferred to v1** (user-decided 2026-06-07):
   ship `num_prompts=1` + `prompt_indices=[0]` through Phase 4 (HTTP + α
   measurement). All architecturally hard pieces (alias page-table, accept
   walk, KV semantics, two-phase warmup, three-trace orchestration) land
   in v0. Multi-client v1 generalizes parameters in `_build_verify_alias_
   page_table_host`, adds per-slot accept-walk + admission/eviction; not
   an architectural rewrite. **Reasons**: (a) Stanford CS440LX demo +
   personal usage is single-stream; (b) CB+spec-dec together = vLLM-scale
   complexity, not our scope; (c) the alias page-table mechanism is
   needed for spec-dec REGARDLESS (it's "the parallel-verify thing", not
   "the multi-client thing" — K+1 alias rows always read the same prompt's
   KV history to amortize K verifies into 1 forward).

**Revised total: ~5 days build + 1 day buffer = ~6 days** (down from 8).

Major scope reductions from Phase 0.A:
1. Centroid masked-embedding DISABLED for 12B (use_ordered_embeddings=False) — standard lm_head, no topk-softmax dance
2. Drafter is PARALLEL not autoregressive — no draft loop, no rejected-draft KV rewind
3. Drafter forks ~80% from existing `server_gemma4_unified_ttnn.py` (4 Gemma 4 layers + pre/post projection + lm_head)
4. transformers ≥5.10.0 required for the HF oracle (current is 5.9.0)

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
| v0.4 | Drafter trace capture (#255) — **SHIPPED 2026-06-08 ✓** | 5/5 PASS: eager 63.6→**traced 6.4 ms (9.99×)**, argmax bit-equiv eager+HF (=597) | DONE |
| v0.5 | Drafter v0.5 perf — measure ms/tok at B=1 | absorbed by v0.4 trace measurement | DONE |

**Phase 1 exit criteria**: drafter runs end-to-end at B=1, produces same
tokens as HF reference greedy on 100-token prompts. Trace captured + 100
traced replays match eager. Drafter forward time measured.

## Phase 2 — target B=K+1 verify trace (~1 day) — **SHIPPED 2026-06-08 ✓**

**ALL 4 GATES PASS** (`2dea4cb` + foreground iterations on `df3e00e`):
- Trace captures cleanly (414 ms wall)
- Replay shape `(Bv,)` non-NaN
- Traced argmaxes bit-equivalent to eager kp1
- Per-row argmaxes match independent B=1 step
- **Eager 114.9 ms / Traced 59.8 ms** (3/3 warm replays)
- Projected spec-dec wall ~112ms per round; α=0.7 → ~28 ms/tok (1.7× over baseline)



Currently `server_gemma4_unified_cb.py` captures only B=1. Spec-dec verify
needs a B=K+1 forward where K+1 logical batch rows alias onto one
physical KV slot.

### Phase 2.A — KV exposure + alias-page-table SHIPPED 2026-06-07
- 2.A.0 layout probe: per-chip reassembly works as documented (`de604fd`)
- 2.A target server change: `read_shared_kv_for_drafter` + `reset_shared_kv_for_drafter` (`952f31e`)
- 2.A.smoke: target + drafter co-resident on (1,4) mesh; argmax MATCH (`486d3e9`)
- 2.B.0 alias helper `build_verify_alias_page_table_host` (`25e3fb3`, 5/5 host probe PASS)
- 2.B.0.5 kernel gate: `paged_update_cache` + `paged_sdpa_decode` accept B=K+1 (`c3124d2`)

### Phase 2.B.1 — B=K+1 verify trace in target server (foreground, ~2-3 h)

Audit at `research/gemma4_verify_kp1_audit.md` (commit `e685c6f`) revised
scope from agent's "1.5-2 day" to **~310 LOC mechanical fork**, since:
- `_lm_head_argmax` already B-generic (no fork)
- All kernels accept B=K+1 (gate `c3124d2`)
- Only 4 functions need fork → `*_kp1` variants

**Decision**: fork (`*_kp1` variants) NOT thread B as parameter — zero
risk to existing 47 ms/tok B=1 path. Read-only verify (skip
`paged_update_cache` for verify rows; feed K+1 Q only, read row 0's KV).

| Step | Adds | Gate | Time |
|---|---|---|---|
| 2.B.1.1 | State buffer alloc: `tok_buf_kp1`, `cur_pos_buf_kp1`, `rot_idxs_buf_kp1`, `page_table_kp1_tt`, `verify_K`, `verify_trace_id` | bootstrap survives + buffers allocated | 20 min |
| 2.B.1.2 | `_set_pos_kp1` + `update_verify_inputs` host writes | unit-call sanity (no device) | 15 min |
| 2.B.1.3 | `_layer_pos0_sliding_paged_kp1` fork + isolation probe | one sliding layer K+1 forward cos ≥ 0.999 per row vs K+1 independent B=1 | 45 min |
| 2.B.1.4 | `_layer_pos0_global_paged_kp1` fork + isolation probe | one global layer K+1 forward cos ≥ 0.999 per row | 40 min |
| 2.B.1.5 | `_layer_forward_pos0_paged_kp1` orchestrator | per-layer dispatch covers full layer types | 20 min |
| 2.B.1.6 | `forward_token_gm4_inner_kp1` full 48-layer | full forward at B=K+1 returns argmax `[K+1, 1]` | 30 min |
| 2.B.1.7 | `_capture_verify_trace_kp1` two-phase warmup | trace captured, no TT_FATAL | 30 min |
| 2.B.1.8 | End-to-end smoke: K+1 trace replay vs K+1 independent B=1 forwards | cos ≥ 0.999 per row | 30 min |

Open risks the audit flagged (verify with small probes before broad fork):
- `_apply_full_rope` at 3D input (rotate-half may not broadcast cleanly)
- TileLayout alignment at K+1=6 < TILE=32 (probably auto-pads fine)

**Phase 2 exit criteria**: target server has TWO captured traces (B=1
decode + B=K+1 verify); K+1 verify produces logits equivalent to K+1
independent B=1 forwards.

## Phase 3 — spec-dec scheduler

**Architectural finding 2026-06-08**: Phase 2.B.1 shipped read-only
verify (skip `paged_fused_update_cache`). This means cache must advance
via target B=1 × N per round → **NO tok/s speedup vs baseline** at v0.
The projected 1.65× requires verify to ALSO write K/V (write-then-rewind
or non-aliased page-table). **User decision (correctness first)**: ship
Phase 3 v0.0 with read-only verify, measure α correctness, accept slow
tok/s. v1.0 perf via non-aliased page table is a follow-up.

### Phase 3 v0.0 — correctness gate (~1 day)

| Step | Adds | Gate | Time |
|---|---|---|---|
| 3.A | `spec_dec_scheduler.py` — flesh out 3 NotImplementedError seams: draft_step, target_verify, accept_walk | scheduler dispatches forward through existing trace paths | 1-2 h |
| 3.B | Accept-walk core — host-side argmax compare drafter[0..K-1] vs verify[0..K-1]; emit accepted prefix + 1 correction; advance cache via target B=1 × N (read-only verify constraint) | greedy-equivalent: spec-dec output token sequence == B=1 target output on 5 prompts | 3-4 h |
| 3.C | Dev-harness bench — measure α (acceptance rate) at K∈{3,5,7} | α ≥ 0.6 measured + reported (tok/s expected SLOWER than baseline due to read-only) | 2 h |

**Phase 3 v0.0 exit criteria**: spec_dec_scheduler runs through dev-harness,
produces **greedy-equivalent output to plain B=1**, α ≥ 0.6 measured at
each K. tok/s deliberately not gated — known-slower with read-only verify.

### Phase 3 v1.0 — perf path (follow-up, ~3-4 h)

Refactor verify trace to non-aliased page table (each K+1 row writes K/V
to its own slot at `cur_pos+1+k`); abandon unused slots after accept
walk (no rewind needed). Fork `_layer_pos0_*_paged_kp1` to UN-skip
`paged_fused_update_cache`; per-row update_idxs + per-row page table.
Expected: spec-dec round 6.4 (drafter) + 60 (verify+write) + <1 (host)
≈ 67 ms / ~4.5 tokens ≈ **15 ms/tok ≈ 3× over 47 ms baseline**.

**Phase 3 v1.0 exit criteria**: tok/s gain ≥ 1.4× at one K (revised
estimate after v0.0 α measurement).

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
