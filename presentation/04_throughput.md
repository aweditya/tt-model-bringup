# Throughput: per-token decode across three model families

All numbers measured on Tenstorrent Blackhole P150 (qb1 or qb2), (1,4) mesh
unless noted. Every cell here is from a sync-bounded benchmark or a memorialized
commit; cells marked TBD are not yet measured. Hardware ceiling is **404 GB/s
on-device DRAM streaming per chip** ([[feedback-p150-memory-bandwidth-measured]],
2026-05-24), not the published 512 GB/s.

## Topline — single-seq decode

| Model | Eager ms/tok | Traced ms/tok | Traced tok/s | BW floor tok/s | % of ceiling | Source |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.6-27B (DENSE, TP (1,4)) | TBD (~250) | **77.0** | **12.93** | ~57 (27 GB / 6.75 GB/chip / 404 GB/s) | ~23% | `HANDOFF.md` "Where the perf is now"; [[feedback-paged-sdpa-shipped-tp]] |
| Qwen3.6-35B-A3B (MoE, TP (1,4)) | 480 (initial, 2026-05-21) | **81.16** | **12.32** | 270 (3 GB active / chip / 404 GB/s) | ~4.6% | `research/35b_perf_milestones.md`; [[feedback-b16-coherent-text-on-device]] |
| Gemma 4 12B (hybrid, TP (1,4)) | 182.7 | **47.5** | **21.05** | 67.3 (12 GB / 3 GB/chip / 404 GB/s) | ~31% | [[feedback-p22-gm4-vocab-shard-result]]; `research/gemma4_perf_briefing_2026-06-04.md` |

Notes:
- **27B eager full-pipeline number is TBD** — we only have the eager batched
  CB number (252.7 ms/step at B=1, [[feedback-cb-batching-free]]), which is the
  same path but instrumented differently.
- **35B "eager"** is the 2026-05-21 first-coherent-text number, pre-trace and
  pre-owned-kernels. The 81.16 ms/tok traced is the cumulative result of
  A002+A003+A004+A008 stacked (`research/35b_perf_workflow_log.md`).
- **35B BW floor counts only active params**: ~3 GB activated per token per
  chip (A3B = 3B active params per token), not the full ~17 GB/chip residency.
  This is why 35B has the highest *theoretical* ceiling despite being the
  biggest model.

## Continuous batching — the 27B story

`experiments/cb/bench/trace.py` on qb1 (1,4) mesh, traced, sync-bounded
(measured 2026-05-29, [[feedback-cb-batching-free]]):

| B  | ms/step (traced) | agg tok/s | vs B=1 |
|---:|---:|---:|---:|
| 1  | 77.15 | 12.96 | 1.00× |
| 8  | 106 | 75 | 5.8× |
| 32 | 212.6 | **150.5** | 11.6× |
| 64 | 348 | 183.5 | 14.2× |
| 64 (after shift-acc conv1d) | — | **593.1** | **45.8×** |

Curve fit: `step_ms ≈ 73 + 4.3 · B` (memory-bound matmul cost ~77 ms is
batch-independent; compute crosses memory-bound around B≈18). Naïve
asymptote = 232 tok/s; after the shift-acc conv1d reformulation
(`cb_conv_mode="shiftacc"`) the per-slot vector slope dropped 3.6 → 0.71
ms/seq and the asymptote moves to ~1400 tok/s. B=64 = **593 tok/s = 45.8×
the B=1 prod number** with the needle test still passing at L=200 and L=500.

**35B CB**: `forward_batch_tp_inner` at B=2 traced is 149.7 ms/step (vs eager
296.7 ms/step = 1.98× speedup, commit `c547419`). Per-slot tok/s ≈ 6.7 at
B=2; higher B requires fixing the B>1 empty-slot poison
([[feedback-35b-batched-forward-empty-slot-poison]], task #162). Production
runs default `TT_CB_SLOTS=1` until that fix lands.

**Gemma 4 CB**: v1.6 validated at B=4 (gates 3a/3b/3c all PASS); B=4 eager
forward ~1.0s/step (= ~250 ms/tok/slot eager, ≈ 16 tok/s aggregate eager).
Traced B=4 aggregate **projected ~55-65 tok/s** from the 3.84× single-seq
trace speedup applied to the B=4 eager step (HANDOFF.md). Not yet measured
end-to-end traced at B>1.

## Why "traced" is 3-4× faster than "eager"

Decode at B=1 is **dispatch-bound**, not compute-bound: the host issues
thousands of small ttnn ops per token, and the gap between dispatches (median
3.4 ms in 35B eager profiling) dominates over the kernel work. Trace mode
captures the dispatch sequence once and replays it without host involvement,
collapsing the inter-op gaps to near-zero. Concretely:

- **Gemma 4 12B**: 182.7 ms/tok eager → 51.3 ms/tok traced = **3.56× speedup**
  out of the box (commit `626c67a`, v0.4). After P1 vocab-shard: 47.5 ms/tok.
- **35B-A3B**: 267 ms/tok eager → 146 ms/tok traced = 1.83× when first captured
  (`research/35b_perf_milestones.md`).
- **27B**: 252.7 ms/step eager at B=1 → 77.0 ms/tok traced = 3.28×.

This is why dispatch-reducing fusions barely move the traced number:
[[feedback-kernel-vs-dispatch-realization]] — kernel-time wins translate
1:1 into trace, dispatch-only fusions land at ~5-10% of their isolated eager
gain. The 35B perf session 2026-05-27 stacked A004 (core_grid 11→110, kernel
time) + A008 (bf8 expert weights, halves DRAM read) for a real
141.79 → 81.16 ms/tok = **+75% tok/s** in one session
(`research/35b_perf_milestones.md`).

## Roofline + remaining headroom

| Model | Active GB / chip | bf16 BW floor (ms/tok) | Ceiling (tok/s) | Current % | Headroom |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B | 6.0 | 14.85 | 67.3 | 31% | 3.2× |
| Qwen3.6-27B (DENSE) | 6.75 | 16.7 | 60 | 22% | 4.6× |
| Qwen3.6-35B-A3B (MoE) | 3.0 (active) | 3.7 | **270** | 4.6% | 22× |

The 35B ceiling looks generous because A3B activates only ~3 GB/token —
on paper, MoE wins. In practice the per-token critical path includes the
*hot* expert weights (bf8-shipped, A008), DN recurrence, and CCLs, and the
real bottleneck is now the batched MoE matmul kernel time rather than DRAM
BW. Gemma 4 at 31% of ceiling has the highest realized efficiency in the
fleet — and the clearest 3.2× runway (P2 distributed RMSNorm + P3 paged
SDPA on global layers, `research/gemma4_perf_briefing_2026-06-04.md`).

Tiny models (Llama 8B, Qwen2.5, SmolLM under `models/`) were brought up earlier
on single-chip Wormhole; not in scope for the (1,4)-mesh decode comparison
above and not re-measured under the current workflow.

## POST-FIX UPDATE 2026-06-04 — HTTP CB stress (after commit `38b15b0`)

The numbers above are pure single-stream / traced. After this session's
`cb_dn_recurrence_mode = "owned_gdn"` fix in `setup_cb_state`
(`38b15b0`), the user-facing HTTP path scales near-linearly to ~160
tok/s aggregate at 32 concurrent clients. Source of truth:
`presentation/06_live_measurements.md` (drives the poster).

### Headline HTTP aggregate (TT_CB_SLOTS=32, traced, paged SDPA)

| Model | 1 client | 8 clients | 16 clients | 32 clients | Scaling 1→32 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B (TP) | 5.36 | 40.87 | 79.59 | **156.59** | **29.23×** |
| Gemma 4 12B IT (unified) | 5.29 | 41.80 | 83.91 | **162.85** | **30.79×** |
| Qwen3.6-35B-A3B (MoE, B=1 only — see #162) | 3.13 | — | — | — | — |

### Pre-fix vs post-fix (the session catch)

| Config | Aggregate tok/s at 4 clients |
|---|---:|
| 27B B=4, TT_CB_TOPK_K=128, `cb_dn_recurrence_mode` unset → manual DN | 13.30 |
| 27B B=32 (post-fix), no TT_CB_TOPK_K, owned_gdn fast path | 156.59 at 32 clients |

The 13.30 number was supposed to be ~44 tok/s; the audit subagent
traced the gap to **the CB forward reading a different attribute
family than the single-stream forward**. A prior "fix" (commit
`017665e`) removed `deltanet_*` overrides but those only gate the
single-stream `forward_token_tp_inner` (`server_tp.py:724`); the CB
forward `forward_batch_tp_inner` (`server_tp_cb.py:454`) reads
`cb_dn_recurrence_mode` / `cb_conv_mode` which were never set →
defaulted to `manual` via `getattr`. The correct fix sets those two
attrs inside `setup_cb_state` itself. Details:
`research/cb_perf_regression_audit_2026-06-04.md`.

### Multi-turn HTTP with prefix cache (27B, post-fix)

| Turn | prompt_t | gen_t | wall (s) | Notes |
|---:|---:|---:|---:|---|
| 0 | 32 | 48 | 10.73 | Cold prefill + decode |
| 1 | 105 | 38 | 19.26 | Cold prefill |
| 2 | 172 | 41 | **9.03** | **PC HIT** — 6.3× speedup vs turn 1 despite 64% more prompt tokens |

`cb_prefix_cache_hits_total = 1`, `cb_prefix_cache_live_slots = 32`.
Gemma 4 PC currently misses on chat-template re-renders (0/60 hits);
needs the equivalent of Qwen3.6's `_messages_to_prompt` patches for
Gemma 4 — open bug.

### Why HTTP 156 tok/s vs historical 376/593 tok/s

The historical `27b_cb_scope.md:687-688` benches used the argmax-tail
trace (single 8-byte readback per step). The HTTP path forces
`sampling=True` in `cb_api.py`, routing through the slower logits-tail
trace. Closing the gap requires capturing both traces and routing
greedy requests through the argmax tail (audit Fix 3).

### Dev-harness vs HTTP — both correct, different things

| Path | Measures | Gemma 4 number |
|---|---|---:|
| Dev harness `step_forward_traced` (B=1, non-paged SDPA, no CB) | Pure model trace speed | **21.05 tok/s** |
| HTTP CB `forward_batch_*_inner` (B=32, paged SDPA, per-slot KV) | What users see | **162.85 tok/s** agg (~5.09 tok/s/slot) |

The per-slot CB rate is ~25% of the dev-harness B=1 because the CB
forward does 32× the work per step. The dev-harness number is the
ceiling we measure against; the HTTP number is the product.
