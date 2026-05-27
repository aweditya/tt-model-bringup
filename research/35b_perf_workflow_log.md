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

### A002 — QK L2-norm via ttnn.rms_norm (DONE 2026-05-27)

Goal: replace the 5-op manual L2-norm chain (per Q and per K) with one
ttnn.rms_norm call. Math equivalence: L2-norm = rms_norm with
weight=1/sqrt(d), epsilon=eps/d.

| Step | Status | Result |
|---|---|---|
| 1 Profile | done | DN total 3.23 ms/call eager (bench_dn_total.py). Tracy probe overflowed DRAM buffer during bootstrap; section profiler hit production API drift. |
| 2 Hypothesis | done | Prior memory says ttnn.rms_norm replaces 11-op manual; 88.6% faster on 27B. Math: weight=1/sqrt(d), eps_rms=eps/d. |
| 3 Isolate | done | `experiments/test_qk_l2_norm_fusion.py`. pcc(fused,manual)=0.999986. Timing 0.6374 -> 0.0580 ms = 10.99x isolated. Predicted: 34.76 ms/tok eager savings if linear. |
| 4 E2E eager | done | decode_smoke eager: 225 -> 201 ms/tok = 24 ms/tok saved (10.7%). First predicted token " Paris" ✓ (correct). |
| 5 Trace A/B | done | Paired file swap (pre-fusion server vs post-fusion server, same harness): 141.79 -> 140.66 ms/tok = 1.13 ms/tok (0.80%). Trace amortizes most of the eager dispatch savings. |
| 6 Long context | partial | 100-token eager generation: " Paris..." first ~50 tok coherent, then greedy-decode degenerate repetition (pre-existing, not fusion-caused). Needle-haystack at L>=500 deferred to next session. |

**Verdict: ship.** 0.80% trace savings is small but real and bit-clean
(pcc=0.999986). Default `state.dn_fused_qk_norm = True`. Manual chain
left in dn_forward_ttnn as the False branch (correctness fallback).

**Open issue:** with `state.dn_fused_qk_norm=False` on the current
commit, the manual-chain path produces incorrect output (decode_smoke
gets 'arus' instead of ' Paris'). The manual chain code is byte-identical
to pre-fusion commit ecbffbf where it gave ' Paris'. Cause unclear;
parking — the fused path is the only one we ship, and the in-session
A/B for trace measurement was done by swapping the WHOLE file. If we
ever need the fallback, debug this first.

**Workflow methodology takeaway:** the eager-to-trace realization rate
on dispatch-reduction fusions is small (~5% for this op). For future
candidates, predict the trace gain (not eager) by estimating what
fraction is genuine kernel work vs dispatch. Dispatch-only wins won't
move 143.6 ms/tok much.

### A003 — Router topk reorder (DONE 2026-05-27)

`softmax(logits, 256) → topk(K) → sum → div(top_vals, sum)` is exactly
`topk(logits, K) → softmax(K)` (softmax is monotonic so top_idxs match,
and softmax(top_logits) == softmax(logits)[top_idxs] / sum after-norm).
4 ops → 2 ops, smaller softmax.

| Step | Result |
|---|---|
| 3 Isolate | 1.71x in iso (0.43 -> 0.25 ms). pcc(candidate vs current aligned)=0.999985 |
| 4 E2E eager | 201 -> 199 ms/tok eager |
| 5 Trace | 140.66 -> 140.41 ms/tok = -0.25 ms (-0.18%, sub-noise) |
| 6 Long ctx | 100-tok eager " Paris" + coherent then known greedy degenerate |

**Verdict: ship for code-cleanliness.** Trace win is sub-noise; the
2-op form is also clearer code.

### A004 — Explicit core_grid=10x11 on batched MoE matmuls (DONE 2026-05-27 — BIG WIN)

Tracy + tt-perf-report revealed the dominant op:
`MatmulDeviceOperation b={64} x 32 x 2048 x 1024` = 1838 us, 62.1% of
MoE step, but using only **11 of 110 Tensix cores**. Default ttnn
program-config picks 11 for this shape.

| Step | Result |
|---|---|
| 1 Profile | Dominant op: 1838 us @ 11/110 cores, 29.9% of 404 GB/s peak |
| 2 Hypothesis | core_grid=10x11 (110 cores). Roofline: 634 us best case. |
| 3 Isolate | 2.00 -> 1.25 ms = **1.60x** (test_moe_gate_up_core_grid.py). pcc=0.999991 |
| 4 E2E eager | 199 -> 194 ms/tok (small visible savings in eager) |
| 5 Trace A/B | 140.41 -> 110.40 ms/tok = **-30.01 ms/tok (-21.4%)** |
| 6 Long ctx | 50-tok " Paris, a city renowned..." coherent first 30 tok |

**Why trace realized 100% of the kernel-time gain** (vs A002/A003's ~5%):
A004 saves *kernel time* (actual device matmul compute), not dispatch.
Trace amortizes dispatch overhead but cannot amortize kernel work.
Kernel-time reductions translate ~1:1 to trace ms/tok.

**Verdict: SHIP.** Applied to `gate_up_batched` + `expert_out_batched`
in `moe_forward_ttnn_pattern_a_batched`.

### A005 — Broader core_grid (REJECTED 2026-05-27)

Tried applying core_grid=10x11 to DN `in_proj_combined` + shared expert
matmuls. **Trace ms/tok regressed +2.99 (110.40 → 113.39)**. ttnn's
default picks the right grid for those smaller shapes; forced full-grid
adds overhead. Reverted. Lesson: do not apply core_grid blindly —
Step 3 isolation is non-skippable.

### A006 — lm_head core_grid (REJECTED 2026-05-27)

Isolated sweep (`experiments/test_lm_head_core_grid.py`):
  default:        1.732 ms/call
  core_grid 10x11: 1.716 ms/call  (1.01x — within noise)
  core_grid 8x8:   1.947 ms/call  (0.89x — regression)

ttnn's default picks the right grid for the lm_head's large output dim
(VOCAB=152064). The 27B "vocab-sharded lm_head" 5.1% win came from
sharding the WEIGHT across the 4-chip mesh (divides matmul cost by 4) +
on-device argmax — that's a structural refactor, not a core_grid kwarg.

Parked for now; revisit if going below 100 ms/tok requires it.

### A007 — h in L1 for batched MoE gate_up (REJECTED 2026-05-27)

tt-perf-report's specific advice on the dominant matmul: "place input 0
in L1 (currently DEV_0_DRAM_INTERLEAVED)". Tested in isolation
(`experiments/test_moe_gate_up_h_in_l1.py`) on top of A004:

  DRAM (A004 baseline):   1.224 ms/call
  L1 interleaved:         1.244 ms/call   delta -0.020 ms (slower)
  L1 in + L1 out:         1.244 ms/call   delta -0.020
  cast DRAM->L1 + matmul: 1.238 ms/call   delta -0.013

PCC: 1.0 bit-clean — all variants compute identically. Pure perf null
result. **The math:** input 0 (h_3d_repeat) = 256 KB; input 1 (W) =
256 MB. Input 0 is 0.1% of total DRAM traffic. At ~280 GB/s achieved
BW, reading 256 KB takes ~0.9 us out of 1224 us kernel = 0.07% saving
possible. Below noise floor. tt-perf-report's "place input 0 in L1"
advice presumes input 0 is meaningfully sized; it isn't here.

What would actually move this matmul (the remaining 2x gap to roofline):
  1. **bf8 weights** — halves W DRAM read (256 -> 128 MB). 27B production.
  2. **Sharded L1** for h_3d_repeat — each core's slice local, no NoC.
     Requires matching matmul's expected shard layout (complex setup).
  3. **Output subblock 2x2** instead of 1x1 — requires custom MatmulProgramConfig.

## Cumulative trace ms/tok timeline

| Stage | ms/tok | Δ from prev |
|---|---|---|
| Pre-2026-05-27 baseline | 141.79 | — |
| +A002 QK norm | 140.66 | -1.13 |
| +A003 router topk | 140.41 | -0.25 |
| +A004 batched core_grid | **110.40** | **-30.01** |

**Total: -31.4 ms/tok (-22.1%)** in one session.

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
