# Performance summary — all shipped models (2026-06-05)

Single perf-table view of every model brought up in this repo. Sources
are cited inline; this doc **never** introduces new numbers — it
aggregates what is already measured and committed elsewhere.

For the cold-start current production state, read [`../HANDOFF.md`](../HANDOFF.md).
For the design + measurement methodology behind 27B continuous-batching
numbers see [`27b_cb_scope.md`](27b_cb_scope.md); for 35B see
[`35b_perf_milestones.md`](35b_perf_milestones.md).

---

## Table of contents

- [Hardware + measurement conventions](#hardware--measurement-conventions)
- [Headline single-seq numbers](#headline-single-seq-numbers)
- [Continuous-batching scaling](#continuous-batching-scaling)
- [Per-model details](#per-model-details)
- [Roofline ceilings](#roofline-ceilings)
- [Sources](#sources)

---

## Hardware + measurement conventions

All numbers below are on **Tenstorrent QuietBox** workstations
(4× Blackhole P150, `FABRIC_1D` mesh). P150 measured DRAM BW =
**404 GB/s/chip** (79% of 512 GB/s peak; firmware 19.6.0; from
[`feedback-p150-memory-bandwidth-measured`]).

- "Eager" = sync-bounded host-loop timing (`ttnn.synchronize_device`
  before + after each iter); dominated by Python dispatch.
- "Traced" = trace replay via `execute_trace` — the production path.
- "Single-seq" = dev-harness `step_forward_traced` (non-paged SDPA,
  no CB scheduler overhead) — the "how fast is the model at all" floor.
- "B=N (CB)" = the HTTP CB engine (`forward_batch_*_inner`, paged SDPA,
  per-slot KV) at N admitted slots / N concurrent clients.

The two paths answer different questions — see
[`archive/presentation_cs440lx_2026-06-04/06_live_measurements.md`](../archive/presentation_cs440lx_2026-06-04/06_live_measurements.md)
§"Two distinct 'throughput' numbers to keep straight".

---

## Headline single-seq numbers

| Model | Mesh | Eager (ms/tok) | Traced (ms/tok) | Single-seq tok/s | Source |
|---|---|---|---|---|---|
| Qwen3.6-27B-A3B (dense, TP) | (1, 4) | ~225 (approx) | **77** | **12.93** | [HANDOFF §Where the perf is now](../HANDOFF.md) |
| Qwen3.6-35B-A3B (MoE + GDN) | (1, 4) | ~331 | **81.16** | **12.32** | [35b_perf_milestones.md](35b_perf_milestones.md) |
| Gemma 4 12B IT (dense, TP) | (1, 4) | 182.7 | **47.5** | **21.05** | [gemma4_perf_briefing_2026-06-04.md](gemma4_perf_briefing_2026-06-04.md) + [`feedback-p22-gm4-vocab-shard-result`] |
| Nemotron-3 Nano 30B-A3B (Mamba2 hybrid) | (1, 4) | (not measured) | (not measured) | — | [nemotron3_nano_30b_a3b_bringup_plan.md](nemotron3_nano_30b_a3b_bringup_plan.md) — see G1 status row below |
| Llama-3.1-8B Instruct (single chip) | 1× P150 | — | ~56 | **18–19** | [REPRODUCE.md](../REPRODUCE.md) `models/73_llama8b_instruct.py` |
| Llama-3.2-3B (single chip) | 1× P150 | — | ~30 | **33.7** | [REPRODUCE.md](../REPRODUCE.md) `models/67_llama32_3b_port.py` |
| Llama-3.2-1B (single chip) | 1× P150 | — | ~13 | **78.6** | [REPRODUCE.md](../REPRODUCE.md) `models/64_llama32_1b_port.py` |
| Qwen2.5-0.5B (single chip) | 1× P150 | — | ~7 | **142.2** | [REPRODUCE.md](../REPRODUCE.md) `models/60_native_rope_decode.py` |
| SmolLM3-3B (single chip) | 1× P150 | — | ~30 | **33** (approx) | wiki §"Model Zoo Sprint" |

**Headline aggregate (HTTP CB engine, B=32, 32 concurrent clients):**

| Model | TT_CB_SLOTS | Aggregate tok/s at 32 clients | Scaling 1→32 |
|---|---|---|---|
| Gemma 4 12B IT | 32 | **316.12** | 27.73× |
| Qwen3.6-27B-A3B | 32 | **232.12** | 27.89× |
| Qwen3.6-35B-A3B | 1 | 3.13 (blocked on task #162 — B>1 empty-slot poison) | — |

Source: [`archive/presentation_cs440lx_2026-06-04/06_live_measurements.md`](../archive/presentation_cs440lx_2026-06-04/06_live_measurements.md)
"Final headline (after `4506385` argmax-tail trace fix)".

---

## Continuous-batching scaling

### Qwen3.6-27B-A3B (manual-DN, kdim conv — production-stable)

Source: [`27b_cb_scope.md`](27b_cb_scope.md) §"Throughput-vs-B sweep + roofline model".

| B  | step ms (traced) | aggregate tok/s | vs B=1 |
|----|------------------|-----------------|--------|
| 1  | 77.15            | 12.96           | 1.0×   |
| 8  | 106.44           | 75.16           | 5.8×   |
| 16 | 140.57           | 113.82          | 8.8×   |
| 32 | 212.69           | 150.45          | 11.6×  |
| 48 | 279.99           | 171.44          | 13.2×  |
| 64 | 348.79           | 183.49          | 14.2×  |

Linear cost model: `step_ms ≈ 73 + 4.3 · B` → asymptote ≈ **232 tok/s**.

### Qwen3.6-27B-A3B (owned_gdn + 3-column shift-acc conv — DNK-G4 fast path)

Source: [`27b_cb_scope.md`](27b_cb_scope.md) §"DNK-G4 DONE".

| B  | step ms (traced) | aggregate tok/s | vs B=1 |
|----|------------------|-----------------|--------|
| 1  | 63.31            | 15.80           | 1.0×   |
| 32 | 84.90            | **376.92**      | 23.9×  |
| 64 | 107.90           | **593.12**      | 37.5×  |

Cost model collapses to `step_ms ≈ 62.6 + 0.71 · B` →
asymptote ≈ **1400 tok/s** (the SDPA core cap and floor land before
the limit).

### Qwen3.6-27B-A3B (HTTP CB engine, end-user serving path)

Source: [`archive/presentation_cs440lx_2026-06-04/06_live_measurements.md`](../archive/presentation_cs440lx_2026-06-04/06_live_measurements.md).

| TT_CB_SLOTS | 1 client | 8 clients | 16 clients | 32 clients |
|---|---|---|---|---|
| 32 (post argmax-tail trace fix `4506385`) | **8.32 tok/s** | 61.27 | 117.62 | **232.12 tok/s** |
| 32 (pre argmax fix, `38b15b0` only) | 5.36 | 40.87 | 79.59 | 156.59 |

The 232 vs 593 gap (HTTP path vs DNK-G4 bench) is the open
"argmax-tail trace not yet wired into HTTP" lever — see
[`code_cleanup_plan_2026-06-04.md`](code_cleanup_plan_2026-06-04.md).

### Gemma 4 12B IT (unified — sliding + global attention)

Source: [`archive/presentation_cs440lx_2026-06-04/06_live_measurements.md`](../archive/presentation_cs440lx_2026-06-04/06_live_measurements.md).

| TT_CB_SLOTS | 1 client | 8 clients | 16 clients | 32 clients |
|---|---|---|---|---|
| 32 (post argmax-tail trace fix `4506385`) | **11.40 tok/s** | 89.25 | 172.52 | **316.12 tok/s** |

Single-seq dev-harness traced: **21.05 tok/s** (`step_forward_traced`,
post P1 vocab-sharded lm_head, commit reference
[`feedback-p22-gm4-vocab-shard-result`]).

### Qwen3.6-35B-A3B (MoE + GatedDeltaNet) — single-stream perf trail

Source: [`35b_perf_milestones.md`](35b_perf_milestones.md) +
[`archive/superseded_research_2026-06-04/35b_perf_workflow_log.md`](../archive/superseded_research_2026-06-04/35b_perf_workflow_log.md).

| Date | Mode | ms/tok (traced) | tok/s |
|---|---|---|---|
| 2026-05-24 | topk eager baseline | 480 | 2.08 |
| 2026-05-25 | batched Pattern A traced | 146 | 6.85 |
| 2026-05-26 | + fused SwiGLU + DN SILU | 145.1 | 6.89 |
| 2026-05-26 | + qwen36_gdn_decode_owned (first coherent gen) | 143.8 | 6.95 |
| 2026-05-26 | + qwen36_decay_gate_decode_owned | 143.6 | 6.96 |
| 2026-05-27 | + A002 QK L2-norm fusion | 140.66 | 7.11 |
| 2026-05-27 | + A003 SwiGLU on shared expert | 140.41 | 7.12 |
| 2026-05-27 | + A004 batched MoE expert matmul | 110.40 | 9.06 |
| 2026-05-27 | + A008 + A009 kernel-time fusions | **81.16** | **12.32** |

Block attribution post-owned-GDN (eager, per token):
MoE 51.9% · DN 39.7% · Attention 8.5%.

**CB scaling on 35B**: blocked at **TT_CB_SLOTS=1** (3.13 tok/s) by
task #162 — empty-slot poisoning of slot 0 at B>1. B>1 forward is
correct in isolation (CB-v1.5 validators 4/4 PASS) but routes through
the HTTP engine via shape-mismatch path — not the right perf number
to ship. Fix planned in [`35b_cb_bringup_plan.md`](35b_cb_bringup_plan.md).

### Nemotron-3 Nano 30B-A3B (G1 single-core kernel status, not end-to-end perf)

The full-model perf is **not yet measured** — bringup is in Phase 0
(owned Mamba2 SSD kernel). Current G1 status:

| Gate | Verdict | Source |
|---|---|---|
| G0 numpy oracle | PASS (modes 1–5) | [nemotron3_nano_30b_a3b_bringup_plan.md](nemotron3_nano_30b_a3b_bringup_plan.md) Task #183/#184 |
| G0a isolation harness | PASS — multi-step replay + per-head cos/MAD gate | Task #184 (commit `4352baf`) |
| G1 mode=2 (decay-only state update) | state cos = **1.000000** | [HANDOFF §POST-WIN QUICK-START](../HANDOFF.md) |
| G1 mode=3 (full state-update math) | state cos = **0.999707**, y sentinel | HANDOFF |
| G1 mode=4 (y = C·state_in^T + D·x) | state cos = 0.999707, y cos = **0.999998** | HANDOFF (commit `978f23e`) |
| G1 mode=5 (y = C·state_out^T + D·x — production) | state cos = 0.999707, y cos = **0.999852** | HANDOFF (commit `b2c4ccc`) |
| G2 multi-core | not started | next task |

Estimated end-to-end tok/s: **not measured yet**.

---

## Per-model details

For deep-dives, see:

- 27B: [`27b_cb_scope.md`](27b_cb_scope.md), [`27b_prefix_caching_plan.md`](27b_prefix_caching_plan.md), [`27b_prefill_trace_plan.md`](27b_prefill_trace_plan.md).
- 35B: [`35b_perf_milestones.md`](35b_perf_milestones.md), [`35b_tt_perf_report_findings.md`](35b_tt_perf_report_findings.md), [`35b_moe_ffn_kernel_perf_deferrals.md`](35b_moe_ffn_kernel_perf_deferrals.md), [`35b_cb_bringup_plan.md`](35b_cb_bringup_plan.md).
- Gemma 4: [`gemma4_perf_briefing_2026-06-04.md`](gemma4_perf_briefing_2026-06-04.md), [`gemma4_12b_bringup_plan.md`](gemma4_12b_bringup_plan.md).
- Nemotron-3 Nano: [`nemotron3_nano_30b_a3b_bringup_plan.md`](nemotron3_nano_30b_a3b_bringup_plan.md), [`mm7_g1_mamba2_kernel_design.md`](mm7_g1_mamba2_kernel_design.md).
- Legacy single-chip ports (Llama / Qwen2.5 / SmolLM / 8B / MoE-1.5):
  [REPRODUCE.md §Legacy demos](../REPRODUCE.md#reproduce--legacy-single-chip-demos)
  + wiki entries 36/40/42/43/44/45/58/59.

---

## Roofline ceilings

P150 measured peak: **404 GB/s/chip × 4 chips**.

| Model | bf16 weight footprint | bf16 BW floor | bf16 ceiling tok/s | bf8 ceiling tok/s | We are at |
|---|---|---|---|---|---|
| Gemma 4 12B (dense) | 24 GB total / 6 GB/chip | 14.85 ms/tok | **67.3** | 134.6 | 21.05 / 67.3 = **31%** |
| Qwen3.6-27B-A3B (dense) | ~54 GB / 13.5 GB/chip (approx) | ~33 ms/tok | ~30 | ~60 | 12.93 / 30 = **43%** (approx) |
| Qwen3.6-35B-A3B (MoE, ~3 GB active/chip) | ~3 GB active / 0.75 GB/chip | 3.7 ms/tok | **270** | 540 | 12.32 / 270 = **4.6%** |

The 35B "active params/chip" line is what makes its 270 tok/s ceiling
so much higher than its current 12.32 — the MoE only activates ~12 GB
of weights per token, not the full 70 GB. See
[`35b_perf_milestones.md`](35b_perf_milestones.md) for the exact roofline math.

The 27B BW floor is approximate (`(approx)` rather than measured) —
the [`feedback-realistic-tp-ceiling`] entry caps real 4-chip TP speedup
at ~1.78× (El Reg measurement), not the naive 4×, so the per-chip
roofline already over-promises.

---

## Sources

Authoritative sources used to build this table (no number is invented):

- [`../HANDOFF.md`](../HANDOFF.md) — operator-curated cold-start one-pager.
- [`27b_cb_scope.md`](27b_cb_scope.md) — CB design + B-sweep tables.
- [`35b_perf_milestones.md`](35b_perf_milestones.md) — 35B traced perf timeline.
- [`archive/superseded_research_2026-06-04/35b_perf_workflow_log.md`](../archive/superseded_research_2026-06-04/35b_perf_workflow_log.md) — per-attempt A001..A009 detail.
- [`gemma4_perf_briefing_2026-06-04.md`](gemma4_perf_briefing_2026-06-04.md) — Gemma 4 baseline + headroom.
- [`archive/presentation_cs440lx_2026-06-04/06_live_measurements.md`](../archive/presentation_cs440lx_2026-06-04/06_live_measurements.md) — 1/8/16/32-client scaling tables, multi-turn PC measurements.
- [`nemotron3_nano_30b_a3b_bringup_plan.md`](nemotron3_nano_30b_a3b_bringup_plan.md) — current G0..G4 status.
- [`../REPRODUCE.md`](../REPRODUCE.md) — legacy demo expected vs measured.
- Memory entries (referenced via `[[name]]` notation): live in `memory/MEMORY.md`.

If a number here disagrees with one of the above, the upstream source
wins — open a PR to fix this doc, don't change the upstream.
