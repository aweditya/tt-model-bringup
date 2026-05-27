# 35B perf optimization log — strict 6-step workflow

Every optimization attempt MUST go through these 6 steps in order. No
step is skipped without explicit user nod. This doc tracks every attempt
across sessions so context compaction can't lose state.

## The 6 steps

1. **Profile** with tt-perf-report (or Tracy single-block probe) on the
   *current* production state. Roofline math on the candidate hot op.
2. **Hypothesis** — a specific predicted ms/tok gain grounded in step 1.
3. **Isolate + correct** — permanent code in `experiments/`, isolated PCC
   gate vs numpy/HF oracle.
4. **E2E short-context eager** — wire behind a flag, run the existing
   server smoke ("The capital of France is" → " Paris").
5. **Trace A/B** — capture trace, paired A/B (flag on vs off) same harness,
   same session. Measure ms/tok delta + correctness.
6. **Long context** — needle-haystack L=500, L=4k, L=32k. Final verdict.

## Current baseline

- 35B-A3B per-token: 143.6 ms/tok trace-mode (per HANDOFF / commit `dc869c8`)
- Production stack at 2026-05-27:
  - `state.moe_mode = 'pattern_a_batched'` (default)
  - `state.dn_owned_gdn = True` (qwen36_gdn_decode_owned kernel)
  - `state.dn_owned_decay_gate = True` (qwen36_decay_gate_decode_owned)
  - `in_proj_combined` (task 64, 2026-05-27 — bench predicted 1.09 ms/tok,
    pcc=1.0 bit-exact, awaiting trace A/B confirmation)
- Hardware: qb1 (1,4) P150 mesh, 404 GB/s DRAM, 110 Tensix/chip

## Attempts

### A001 — task 64: in_proj fusion (2026-05-27)

| Step | Status | Notes |
|---|---|---|
| 1 Profile | partial | Relied on existing eager profile (research/35b_tt_perf_report_findings.md); didn't redo for DN specifically |
| 2 Hypothesis | done | 4 matmuls -> 1 saves dispatch; bench predicted 0.036 ms/call * 30 = 1.09 ms/tok |
| 3 Isolate | done | `bench_dn_in_proj_fusion.py`, bit-exact pcc=1.0 |
| 4 E2E eager | done | smoke produces " Paris" (commit `ecbffbf`) |
| 5 Trace A/B | **NOT done** | Skipped — no trace-mode bench harness existed |
| 6 Long context | **NOT done** | Skipped |

**Verdict: incomplete by current workflow standards.** Trace A/B is the
test that proves the savings show up in the 143.6 ms/tok number.
Backfilling steps 5-6 is the next chore.

### A002 — DN profile + next candidate (in progress 2026-05-27)

Goal: profile the *current* DN forward (with task 64 fusion + owned
kernels on) to find the next hot op. Output: hypothesis + isolation
plan for one specific optimization.

| Step | Status | Notes |
|---|---|---|
| 1 Profile | blocked | qb1 + qb2 ssh timed out 2026-05-27 ~04:00 PT. Probe ready: `experiments/utils/tracy_profile_one_dn.py` |
| 2+ | pending | unblocked by step 1 |

## Trace-A/B harness — open chore

We don't have a trace-mode A/B bench checked in. We need one before any
optimization can claim a real ms/tok win. Sketch:

  1. Bootstrap state with flag-OFF
  2. Capture trace, run N execute_trace, measure mean ms/tok → T_off
  3. Tear down state
  4. Bootstrap state with flag-ON
  5. Same trace + measure → T_on
  6. Report delta + check next_id sequence matches between runs

Permanent file location: `experiments/serve/bench_traced_decode_ab.py`
(to be written when qb1 returns).

## Hot-op candidates to investigate (pre-profile predictions)

Predictions before measurement — confirm or refute against profile:

- **DN recurrence (owned_gdn)** — fused kernel, expected to be a single
  large kernel call. Already optimized; probably not the biggest gain.
- **DN QK L2 norm** — currently a manual chain (`mul → sum → rsqrt → add → mul`)
  per Q and per K. Per [feedback_qk_normalize_fusion]: `ttnn.rms_norm`
  replaces 11-op manual; 88.6% faster *in isolation* on 27B.
  This pattern hasn't been applied to 35B yet → potential easy win.
- **Q scale by 1/sqrt(d_k)** — single `ttnn.multiply` with a scalar. Cheap.
- **GQA repeat (Q/K 4→8 heads)** — `reshape → repeat → reshape`. Possibly
  free with the right view trick (see memory `[gqa_prerepeat_not_worth]`).
- **conv1d update + silu** — slice/concat for state shift + per-elem mul +
  sum-reduce + silu. Lots of ops. Per memory: conv1d is 21.5% of single-
  chip DN. Could be a single fused kernel target.
- **out_proj + all_reduce** — column-parallel matmul then all_reduce. AR
  defaults to num_links=2 per [p1_num_links_2_shipped].

The profile will rank these by actual time. Then we pick one.
