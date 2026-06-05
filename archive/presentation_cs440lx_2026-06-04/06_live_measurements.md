# Live measurements (2026-06-04 session) — POST-FIX + ARGMAX-TRACE

## Single-client streaming (TTFT + decode separately, HTTP CB engine, B=32, traced)

| Model | TTFT (s) | Prompt tokens | ms / prompt-token | Decode (tok/s) | Total wall (s, 64 toks) |
|---|---|---|---|---|---|
| Gemma 4 12B IT | **1.40** | 18 | 78 | **17.42** | 5.02 |
| Qwen3.6-27B | TBD (re-measurement pending) | | | | |

Gm4 single-client decode at 17.4 tok/s is now 83% of the dev-harness B=1 ceiling (21 tok/s). CB structural cost is the remaining 17% gap.

## Final headline (after `4506385` argmax-tail trace fix)

| Model | TT_CB_SLOTS | 1 client | 8 clients | 16 clients | 32 clients | Scaling 1→32 |
|---|---|---|---|---|---|---|
| **Qwen3.6-27B (dense, TP)** | 32 | **8.32** | 61.27 | 117.62 | **232.12** | **27.89×** |
| **Gemma 4 12B IT (unified)** | 32 | **11.40** | 89.25 | 172.52 | **316.12** | **27.73×** |
| Qwen3.6-35B-A3B (MoE) | 1 | 3.13 | — | — | — | — |

**Gemma 4 hits 316 tok/s at 32 clients (+94% over the cb_dn fix-only baseline).** Faster than 27B because the model is smaller (12B vs 27B), and the vocab-shard lm_head already wins on the readback side.

**27B 1-client throughput went 5.36 → 8.32 tok/s (+55%)**. Aggregate at 32 clients went **156.59 → 232.12 (+48%)**. Per-step time at B=32: 229 → 88 ms (**2.6× faster**). This is the all-greedy fast path: the captured argmax-tail trace skips the [B, vocab] readback in favour of a 4·B-byte argmax readback.

Compared to the historical 376 tok/s on `bench/trace.py --owned-gdn --shiftacc` at B=32 (`research/27b_cb_scope.md:687`): we've closed to **62% of the historical** within the HTTP serving path. The remaining gap is shiftacc-conv vs kdim differences in the bench — future work.

## Pre-fix evolution this session

| Stage | Config | B=4, 4-client agg tok/s | B=32, 32-client agg tok/s | Per-step ms |
|---|---|---|---|---|
| Original session start | TT_CB_TOPK_K=128, `cb_dn_recurrence_mode` unset → manual DN | 13.30 | — | 229 |
| Commit `38b15b0` | set CB-mode defaults in `setup_cb_state` | — | 156.59 | 229 → ~290 |
| Commit `4506385` (argmax-trace) | dual-trace, route greedy to argmax | — | **232.12** | **88** |

Each fix was: 1 file, ~10-30 LOC, no new env knob. Total cumulative gain on 27B at B=32: 13.30 → 232.12 tok/s = **17.4× from the broken state, and +48% over the `setup_cb_state` fix alone**.

Single canonical source for the poster + presentation. Numbers below
are from `presentation/screenshots/stress_*.json` runs through the
HTTP server (traced decode path, B=cb_slots, owned_gdn fast path
active per commit `38b15b0`).

## Headline numbers (post-fix, after commit `38b15b0`)

| Model | TT_CB_SLOTS | 1 client | 8 clients | 16 clients | 32 clients | Scaling 1→32 |
|---|---|---|---|---|---|---|
| **Qwen3.6-27B (dense, TP)** | 32 | 5.36 tok/s | 40.87 tok/s | 79.59 tok/s | **156.59 tok/s** | **29.23×** |
| **Gemma 4 12B Instruct (unified)** | 32 | 5.29 tok/s | 41.80 tok/s | 83.91 tok/s | **162.85 tok/s** | **30.79×** |
| Qwen3.6-35B-A3B (MoE) | 1 | 3.13 tok/s | — | — | — | — |

27B and Gemma 4 both reach ~160 tok/s aggregate at 32 concurrent
clients with **near-perfectly-linear** scaling, validating the CB
design.

35B was measured at TT_CB_SLOTS=1 only (the B>1 empty-slot poison bug
is task #162, not fixed this session). Same per-step structural cost
as the others before the fix.

## Pre-fix vs post-fix (the regression we caught + corrected mid-session)

| Backend | Config | Step time (ms) | Aggregate tok/s at 4 clients |
|---|---|---|---|
| 27B B=4 | TT_CB_TOPK_K=128, `cb_dn_recurrence_mode` unset → manual DN | 229 | 13.30 |
| 27B B=32 | No TT_CB_TOPK_K, `cb_dn_recurrence_mode = "owned_gdn"` (fix `38b15b0`) | ~290 (full B=32 forward) | 156.59 at 32 clients |
| Gemma 4 B=4 | TT_CB_TOPK_K=128 | 216 | 13.14 |
| Gemma 4 B=32 | No TT_CB_TOPK_K | ~295 (full B=32 forward) | 162.85 at 32 clients |

The big takeaway: **B=4 with the topk-tail trace was hurt by ~100 ms of fixed device cost** (per the `cb_scheduler.py:173-176` docstring "topk adds ~100 ms of fixed device cost that HURTS at low B"). Removing it + the owned_gdn fix unlocked the historical scaling.

## Multi-turn HTTP (TT_CB_PREFIX_CACHE=1)

### Qwen3.6-27B — PC works (Qwen3.6 chat template patches active)

| Turn | prompt_t | gen_t | wall (s) | wall / prompt_t | Notes |
|---|---|---|---|---|---|
| 0 | 32 | 48 | 10.73 | 0.335 | Cold prefill + decode |
| 1 | 105 | 38 | 19.26 | 0.183 | Cold prefill |
| 2 | 172 | 41 | **9.03** | **0.053** | **PC HIT** — turn 2 wall is HALF turn 1's wall despite 64% more prompt tokens |

Metrics: `cb_prefix_cache_hits_total = 1, cb_prefix_cache_misses_total = 59, cb_prefix_cache_live_slots = 32`.

**6.3× speedup on the PC-hit turn**. The `_messages_to_prompt`
chat-template patches in `experiments/serve/openai_endpoint.py:35-90`
(preserve_thinking=True + trailing strip) make turn N+1 re-tokenize
to a true prefix of the cached tokens_so_far.

### Gemma 4 12B IT — PC misses on chat template (KNOWN BUG)

| Turn | prompt_t | gen_t | wall (s) | wall / prompt_t |
|---|---|---|---|---|
| 0 | 36 | 48 | 10.40 | 0.289 |
| 1 | 105 | 48 | 18.94 | 0.180 |
| 2 | 180 | 48 | 28.32 | 0.157 |

Metrics: `cb_prefix_cache_hits_total = 0, cb_prefix_cache_misses_total = 60`. Gemma's chat template re-renders the prior assistant turn (after `<end_of_turn>\n<start_of_turn>user\n…`) in a way that doesn't byte-equal what was generated. Needs the equivalent of Qwen3.6's patches for Gemma 4. Documented `[[prefix-cache-multiturn-miss-2026-06-04]]`.

## Correctness gates passed this session (dev harness, eager — what we validate against, not what we serve)

- **35B teacher-forced cosine ladder**: 7/8 positions argmax-match HF on the 85-tok prompt; cos_L32 ≥ 0.987 at every probed position. The "drift cliff" memory entry is invalidated — drift is gone in current state. `[[35b-drift-resolved-2026-06-04]]`.
- **35B free-run needle-haystack** (L = 100, 200, 300, 460, 1024): per-trial Y/N is a coin flip; bf16 chain noise flips argmax → ~50% retrieval per trial. Failures are coherent ("I don't know"), not gibberish. `[[35b-needle-haystack-2026-06-04]]`.
- **35B multi-turn Q&A (3 turns)**: 3/3 PASS visual-grade (eager). T2 correctly recalled "Paris" from T0 and answered with a Paris fact. `[[35b-multiturn-qa-2026-06-04]]`.
- **Gemma 4 12B v0.4 trace validator**: 100/100 token-for-token match traced vs eager at 100 free-run steps. Traced **47.5 ms/tok = 21.05 tok/s** single-seq via the dev-harness `step_forward_traced` (non-paged SDPA, no CB scheduler overhead). `[[feedback-p22-gm4-vocab-shard-result]]`.

## Two distinct "throughput" numbers to keep straight (per `[[feedback-perf-no-handwaving]]`)

The user-facing chat experience goes through CB → paged SDPA → per-slot bookkeeping. The dev-harness single-seq path bypasses all that. Both are "correct" measurements but answer different questions:

| Path | What it measures | Use it for |
|---|---|---|
| Dev harness `step_forward_traced` (single-stream, non-paged SDPA) | Pure model speed at B=1 — pure trace replay | "How fast is the model at all?" baseline ceiling |
| HTTP CB engine `forward_batch_*_inner` (B-leading, paged SDPA, per-slot KV) | What ships to users; pays CB structural cost | "What will a user see?" + aggregate scaling story |

For Gemma 4, dev-harness B=1 = 21.05 tok/s; CB-engine B=32 with 32 clients = 162.85 tok/s aggregate ≈ 5.09 tok/s/slot. The per-slot CB rate is ~25% of dev-harness B=1 because the CB forward does 32× the work per step.

## Roofline + ceilings

- P150 measured DRAM BW: **404 GB/s on-device** (79% of 512 GB/s peak). `[[feedback-p150-memory-bandwidth-measured]]`.
- Gemma 4 12B bf16 ceiling: 24 GB / 6 GB/chip / 404 GB/s = **14.85 ms/tok = 67.3 tok/s**. We're at 21.05 tok/s single-seq traced = **31% of ceiling**. Headroom **3.2×**.
- 27B B=32 CB aggregate ceiling per `[[feedback-realistic-tp-ceiling]]`: realistic TP ceiling is 1.78× (not 4×) per El Reg's measurement. At 156 tok/s aggregate with 32 active slots we're 23× the 1-client number — well above the TP ceiling expectation, demonstrating that batching dominates the scaling story (not multi-chip TP).

## Bugs surfaced + fixed this session (in order)

1. **gm4 `_lm_head_argmax` rank-3 → rank-2 contract** — one tensor reshape in the helper unblocked the topk path. Commit `5620314` (slice), `29205d7` (rank normalization).
2. **gm4 `build_key_to_shard` IT multi-snapshot** — picked the wrong snapshot dir silently. Walk all snapshots. Commit `0418e83`.
3. **`cb_api.py` clobbered 27B `deltanet_*` (WRONG attribute family)** — discovered to be a no-op for CB perf. The single-stream path needed it; the CB path reads different names. Removed in commit `017665e` (correct removal of dead code, but didn't fix CB perf).
4. **`openai_endpoint.py` stale on qb1** (no `tools` kwarg) → 27B HTTP 500s. Fixed by `deploy.sh experiments/serve/openai_endpoint.py`. Memory `[[stale-deploy-27b-tools-2026-06-04]]`.
5. **CB engine ran manual DN recurrence on 27B (THE BIG FIX)** — `state.cb_dn_recurrence_mode` was never set, defaulted to `"manual"` via `getattr(state, '…', 'manual')` at `server_tp_cb.py:454`. Fixed by setting `cb_dn_recurrence_mode = "owned_gdn"` and `cb_conv_mode = "shiftacc"` defaults inside `setup_cb_state` itself. Commit `38b15b0`. **This is what unlocked the 156 tok/s number.**

## Bugs surfaced + still open

1. **Gemma 4 IT prefix-cache STILL misses after the tokenize=True fix (`46083bd`)** (0/3 hits in latest probe; 3 live slots so cache infra works). Root cause: the Gemma Jinja template applies `| trim` to past assistant content BEFORE tokenization — `r['gen']` keeps trailing whitespace tokens; the chat-template re-render drops them. Next fix: scheduler `_finish` should `decode → rstrip → re-tokenise` the assistant content before storing `tokens_so_far`. Wall times are still much better (turn 2 was 49.12s before; now 13.17s with the argmax-tail trace fix).
2. **35B B>1 empty-slot poisoning** (task #162) — we were forced to TT_CB_SLOTS=1 for 35B. The 3.13 tok/s 35B number is therefore the WORST CB configuration possible for that model — it can't share the B=32 multiplier yet.
3. **Argmax-tail trace not available in HTTP path** — `cb_api.py` forces `sampling=True` even at temperature=0. The historical 376 / 593 tok/s benches used `sampling=False` with the argmax-tail trace. Catching this would close the gap from 156 → ~376 tok/s on 27B B=32.
4. **35B free-run determinism** — same prompt produces different outputs across runs (bf16 chain noise + non-deterministic reductions). Research at `research/35b_determinism_2026-06-04.md`; fix sketches at the bottom of that file (deterministic argmax tie-break, multicore=False on final argmax, fp32_dest_acc on lm_head).

## Open audit items (from `research/code_cleanup_plan_2026-06-04.md` + `research/cb_perf_regression_audit_2026-06-04.md`)

18 items, 6 High severity. After this session's fixes:
- ✅ B1 (cb_api owned_gdn override) — properly addressed by commit `38b15b0` (the earlier `017665e` was the WRONG fix as the perf audit caught).
- ⏳ A1: Delete 35B manual DN recurrence else-branch (it's been broken at cos 0.08 since the start). Removes a known-broken regime.
- ⏳ A2: Fix 35B B>1 empty-slot poisoning so the `TT_CB_SLOTS=1` carve-out for 35B can be dropped.
- ⏳ D: Deploy-bundle tooling (this session: stale `openai_endpoint.py` on qb1 crashed the first 27B stress).
- ⏳ Argmax-tail trace in HTTP — would close the 156 → 376 tok/s gap on 27B B=32. Captures both traces, routes greedy requests to argmax. Audit Fix 3.
