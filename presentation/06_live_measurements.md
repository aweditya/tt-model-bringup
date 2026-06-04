# Live measurements (2026-06-04 session)

Single canonical source for the poster + presentation. Numbers below
are from `presentation/screenshots/stress_*.json` runs through the
HTTP server (traced decode path, B=cb_slots).

## Per-client + aggregate throughput (HTTP /v1/chat/completions, traced)

### Gemma 4 12B Instruct, TT_CB_SLOTS=4, TT_CB_TOPK_K=128, traced

| Clients | Wall (s) | Total completion tokens | Aggregate tok/s | Per-client tok/s | Speedup vs 1 client |
|---|---|---|---|---|---|
| 1 | 18.89 | 64 | 3.39 | 3.39 | 1.00× |
| 2 | 19.29 | 128 | 6.64 | 3.32 | 1.96× |
| 4 | 19.48 | 256 | 13.14 | 3.29 | **3.88×** |

Per-step metric (from `/metrics`): `cb_step_seconds_sum=157.5s / count=728 = 216 ms/step`. 99.7% in device.

### Qwen3.6-27B, TT_CB_SLOTS=4, TT_CB_TOPK_K=128, traced, owned_gdn (after cb_api clobber fix)

| Clients | Wall (s) | Total completion tokens | Aggregate tok/s | Per-client tok/s | Speedup vs 1 client |
|---|---|---|---|---|---|
| 1 | 18.80 | 64 | 3.40 | 3.40 | 1.00× |
| 2 | 19.29 | 128 | 6.64 | 3.32 | 1.95× |
| 4 | 19.25 | 256 | 13.30 | 3.33 | **3.91×** |

Per-step metric: `cb_step_seconds_sum=57.3s / count=250 = 229 ms/step`. 99.5% in device.

### 35B-A3B — TBD (next backend switch)

## Multi-turn HTTP with TT_CB_PREFIX_CACHE=1

### Gemma 4 12B IT — PC misses on chat template

| Turn | prompt_t | gen_t | wall (s) | wall / prompt_t |
|---|---|---|---|---|
| 0 | 36 | 48 | 17.95 | 0.499 |
| 1 | 105 | 48 | 32.89 | 0.313 |
| 2 | 180 | 48 | 49.12 | 0.273 |

Metrics: `cb_prefix_cache_hits_total = 0, cb_prefix_cache_misses_total = 10`. Wall grows linearly in `prompt_t` — Gemma 4 chat template re-renders the prior assistant turn in a way that doesn't byte-equal what was generated, so the matcher (exact-prefix) misses. Documented in `[[prefix-cache-multiturn-miss-2026-06-04]]`.

### Qwen3.6-27B — PC hits on turn 2!

| Turn | prompt_t | gen_t | wall (s) | wall / prompt_t | Notes |
|---|---|---|---|---|---|
| 0 | 32 | 48 | 18.09 | 0.565 | Cold prefill + decode |
| 1 | 105 | 38 | 32.52 | 0.310 | Cold prefill (3.3× prompt) |
| 2 | 172 | 43 | **16.24** | **0.094** | **PC HIT** — turn 2 wall is HALF of turn 1's wall, despite 64% more prompt tokens |

Metrics: `cb_prefix_cache_hits_total = 1, cb_prefix_cache_misses_total = 9`. The Qwen3.6 chat template patches in `experiments/serve/openai_endpoint.py:35-90` (preserve_thinking=True + trailing strip) make turn N+1 re-tokenize to a true prefix of the cached tokens_so_far. Net effect: **6.0× speedup on the PC-hit turn's prefill**.

## Correctness gates passed this session (dev harness, eager)

- **35B teacher-forced cosine ladder**: 7/8 positions argmax-match HF on the 85-tok prompt; cos_L32 ≥ 0.987 at every probed position. The "drift cliff" memory entry is invalidated — drift is gone in current state. `[[35b-drift-resolved-2026-06-04]]`.
- **35B free-run needle-haystack** (L = 100, 200, 300, 460, 1024): per-trial Y/N is a coin flip; bf16 chain noise flips argmax → ~50% retrieval per trial. Failures are coherent ("I don't know"), not gibberish. Successful trials echo the 8-char needle verbatim. `[[35b-needle-haystack-2026-06-04]]`.
- **35B multi-turn Q&A (3 turns)**: 3/3 PASS visual-grade. T2 correctly recalled "Paris" from T0 and answered with a Paris fact. Retention works. Eager ~195 ms/tok. `[[35b-multiturn-qa-2026-06-04]]`.
- **Gemma 4 12B v0.4 trace validator**: 100/100 token-for-token match traced vs eager at 100 free-run steps. Traced **47.5 ms/tok = 21.05 tok/s** single-seq. `[[feedback-p22-gm4-vocab-shard-result]]`.

## Roofline + ceilings

- P150 measured DRAM BW: **404 GB/s on-device** (79% of 512 GB/s peak). `[[feedback-p150-memory-bandwidth-measured]]`.
- Gemma 4 12B bf16 ceiling: 24 GB / 6 GB/chip / 404 GB/s = **14.85 ms/tok = 67.3 tok/s**. We're at 21.05 tok/s single-seq traced = **31% of ceiling**. Headroom **3.2×**.
- 27B dense bf16: scaled ceiling depends on which numbers we trust; current 27B B=1 traced single-seq ≈ 12.93 tok/s (HANDOFF), B=32 CB hits 150.5 tok/s = ~75% scaling efficiency.

## Bugs surfaced + fixed this session

1. **gm4 `_lm_head_argmax` rank-3** → broke cb_scheduler topk path. Fixed at source in commit `29205d7`. (one contract for all backends).
2. **cb_api was clobbering 27B owned_gdn defaults** (re-set to "manual" after bootstrap). Deleted in commit `017665e`. Net: 27B now actually runs the custom kernel path that two weeks of work targeted.
3. **`openai_endpoint.py` stale on qb1** (no `tools` kwarg). Caught by 500 errors on the first 27B stress. Fixed by `scripts/deploy.sh experiments/serve/openai_endpoint.py`. Memory entry `[[stale-deploy-27b-tools-2026-06-04]]`.
4. **gm4 build_key_to_shard** silently killed the harness when picking the wrong snapshot dir (IT variant has two). Walk all snapshots. Commit `0418e83`.
5. **ttnn.slice rank-aware** at gm4 vocab-shard path. Commit `5620314`.

## Bugs surfaced + still open

1. **Gemma 4 IT prefix-cache misses on chat template** (0/10 hits). Needs the equivalent of Qwen3.6's `_messages_to_prompt` patch for Gemma 4. `[[prefix-cache-multiturn-miss-2026-06-04]]`.
2. **B=4 step time scaling 4-5× over B=1 single-seq** is more than BW-bound math predicts. Both 27B and Gemma 4 hit ~220 ms/step at B=4 vs B=1 traced ~50-80 ms. UNDER INVESTIGATION. Most likely: the trace replays the full B=4 forward (matmuls scale with B in activation dim) plus per-slot SDPA work. To confirm: run TT_CB_SLOTS=1 single-client and expect ~20 tok/s.

## Open audit items (from `research/code_cleanup_plan_2026-06-04.md`)

18 items, 6 High severity. Top-3 leverage:
- A1: Delete 35B manual DN recurrence else-branch (it's been broken at cos 0.08 since the start). Removes a known-broken regime and the `TT_DN_STATE_DTYPE` env knob.
- A2: Fix 35B B>1 empty-slot poisoning so the `TT_CB_SLOTS=1` carve-out for 35B can be dropped.
- B1: ✅ DONE (cb_api owned_gdn clobber, commit `017665e`).
