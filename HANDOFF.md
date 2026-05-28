# HANDOFF — read top to bottom (replaces the 900-line version, see `git log HANDOFF.md` for the old narrative)

> **Active work (2026-05-27): maintainability pass on branch `chore/maintainability`** (not merged). Living plan + how to continue: [`research/maintainability_pass.md`](research/maintainability_pass.md). Perf/correctness work below is paused, not abandoned.

## What this project is

35B-A3B (MoE) bringup on Tenstorrent Blackhole (P150 × 4 mesh on `qb1`). Research-driven; goal is **the hardware ceiling**, not parity with someone else's number.

## Where the perf is RIGHT NOW

| Mode | ms/tok | tok/s |
|---|---|---|
| **+A002 + A003 + A004 + A008 bf8 MoE + A009 sampler hook (2026-05-27)** | **81.16** | **12.32** |
| +A002 + A003 + A004 (intermediate) | 110.40 | 9.06 |
| Pre-2026-05-27 baseline (traced) | 141.79 | 7.05 |

**Net: -60.6 ms/tok (-42.8%), tok/s 1.75x in one workflow session.**

Cumulative wins 2026-05-27 (see `research/35b_perf_workflow_log.md`):
- A002 QK L2-norm via ttnn.rms_norm: -1.13 ms/tok
- A003 router softmax-then-topk → topk-then-softmax: -0.25 ms/tok
- **A004 core_grid=10x11 on batched MoE matmuls: -30.01 ms/tok**
- **A008 bf8_b MoE expert weights (halves W DRAM): -29.24 ms/tok**
- A009 DRY+rep_penalty top-K sampler hook (opt-in via state.sampler_topk;
  fast path unaffected): fixes greedy mode collapse for long-context
  coherent generation. +180 ms/tok in sampler mode (eager only).

Both big wins target the SAME dominant op (batched MoE gate_up matmul)
with two orthogonal mechanisms — A004 parallelizes (11→110 cores) and
A008 shrinks the BW footprint (256→128 MB). Stacking is multiplicative.

Rejected: A005 core_grid on non-batched matmuls (+3 ms regression),
A006 lm_head core_grid (no-op), A007 h-in-L1 (no-op, math says input 0
is 0.1% of DRAM traffic).

**Long-context cliff check (cosine ladder vs HF needle100 oracle):**

  Post-A002+A003+A004+A008:  94/97 top-1 (96.9%), median cos_final 0.9970
  Pre-session baseline:       97/100 top-1 (97.0%)

The drift cliff (memory says in (100, 130]) did NOT move earlier. Our
optimizations are correctness-clean — the L≥100 needle-retrieval failure
is the pre-existing DN-recurrent-state drift documented at
`[feedback_35b_a3b_l32_dn_decode_drift]`, not session-induced.

A009 ships the *sampler hook* (eager only); the deeper drift fix is A010.

**A010 progress (open follow-up — DN H_t fp32 fix)**: see
`feedback_35b_dn_h_state_drift_lever.md`. Confirmed: bf16 H_t storage IS
the long-context lever. Owned_gdn kernel is NOT the source (verified
by A/B). But just storing H_t as fp32 doesn't work — mixed-precision
ops break the math. Three scoped fix paths for the next session:
  1. Cast-on-load (smallest): fp32 storage, bf16 arithmetic, fp32 again
  2. fp32-throughout-DN (medium): cast g_b/k_col/k_delta to fp32 too
  3. Kernel-level fp32 accumulator (largest): modify owned_gdn

Hook is in place: `state.dn_state_dtype` in `reset_caches_ttnn`. Default
bf16 (production unchanged). Cosine ladder gate is at L=97 needle:
94/97 top1, median cos_final 0.997. Beat that to claim A010.

Coherent greedy decode + long-context PASS:
- 20/20 tokens: "Paris, a city renowned for its iconic landmarks such as
  the Eiffel Tower, the Louvre Museum"
- 50-token eager: " Paris, a city renowned for its rich history and
  cultural heritage. The Eiffel Tower, an iconic symbol of Paris..."
- Needle-haystack L=100: `N4Y2BWLS` retrieved verbatim.

Production path: `state.moe_mode = "pattern_a_batched"` in `experiments/serve/server_35b_ttnn.py`. Run via `experiments/utils/trace_demo_full_step.py --moe-mode pattern_a_batched` or `experiments/bench_step_forward_traced.py`.

## 27B continuous batching — PARALLEL TRACK (2026-05-27/28)

vLLM-style continuous batching for the DENSE 27B (so decode doesn't waste 31/32
tile rows). **All in `experiments/serve/server_tp_cb.py`** (imports production
`server_tp.py`, which stays byte-for-byte pristine). Full scope + numbers:
`research/27b_cb_scope.md`.

- **CB1 DONE** — batched forward bit-identical to production. `cb_validate_27b.py`
  PASS: per-position logit_cos=1.0, B=4/32 identical slots, B=4 distinct-slot
  isolation (per-slot KV+DN state). Root cause that blocked it: view-decay
  (`ttnn.slice`/`reshape` return VIEWS; deallocating the source corrupts them —
  masked at pos 0). See `feedback_ttnn_slice_view_decay`.
- **CB4 DONE — TRACED throughput measured** (`cb_bench_trace.py`):

  | B  | ms/step | agg tok/s | × |
  |----|---------|-----------|---|
  | 1  | 77.15   | 12.96     | 1.0× (==prod 12.93) |
  | 32 | 212.69  | 150.45    | 11.6× |
  | 64 | 348.79  | 183.49    | 14.2× |

  Cost model `step_ms ≈ 73 + 4.3·B` (memory-bound 73 ms floor + 4.3 ms/seq
  compute; crossover B≈17; aggregate asymptote ~232 tok/s with manual DN).
- **Batched owned-GDN DN kernel DONE** (`experiments/kernel_patches/qwen36_gdn_decode_owned/`):
  owned_gdn is per-slot independent → FOLD batch into slots ([B,NV,K,V]→
  [1,B·NV,K,V]) drives the EXISTING kernel; no ttnn rebuild (device kernels
  JIT-compile). Mode-0 had a high-slot output race (out reads cb_state_out which
  the writer pops) → patched compute kernel adds `debug_mode=10` safe path
  (output via cb_state_next_internal); in-loop conditional (a dup branch
  overflowed the 70656B TENSIX limit); `debug_mode=0` byte-identical so B=1 prod
  untouched. CB uses it via `cb_dn_recurrence_mode="owned_gdn"`. Bit-identical to
  prod at B=1; traced **B=32 168 tok/s (+11.7%), B=64 208 tok/s (+13.5%)**;
  asymptote ~232→~277 tok/s. **B=64 = 16.1× the B=1 prod 12.96 tok/s.**
- **Conv1d shift-accumulate DONE** — profiling found conv1d = 71.8% of the DN
  cost (the K=4→32 tile-padding tax). Reformulated as shift-accumulate on 3
  padding-free [B,C] state columns (`cb_conv_mode="shiftacc"`): **B=32
  168→376.92 tok/s (2.24×), B=64 208→593.12 (2.85×)**; step-slope 3.6→0.71
  ms/seq, asymptote ~277→~1400. **B=64 = 593 tok/s = 45.8× the B=1 prod 12.96.**
  Long-context **needle test (`cb_needle.py`, through cb_scheduler) PASSES**
  (retrieves the code verbatim at L=200 + L=500) — the fast conv is functionally
  long-context-correct despite a 0.9995 logit-cosine. `cb_conv_mode` default
  "kdim" (bit-identical reference); shiftacc opt-in (needle-validated).
- **Kernel methodology**: `research/kernel_design_worksheet.md` (fill-before-code
  + step-0 "do you even need a kernel?") + `research/kernel_dataflow_representation.md`
  (TDG). The conv win came from worksheet step 0 — op-level reformulation, no kernel.
- **NEXT lever past ~593**: the 62.6 ms B-independent floor (2×64 all-reduces,
  248K-vocab lm_head); plus productionization (chunked prefill, sampling, endpoint).
- **CB2 DONE** — ragged per-slot positions + mid-batch admission
  (`cb_validate_ragged.py` PASS). `cb_reset_slots()` clears only the admitted
  slot's DN state (Mamba-style reuse); KV self-overwrites (cur_pos-bounded).
- **CB3 DONE** — Orca iteration-level scheduler (`experiments/serve/cb_scheduler.py`).
  5 reqs / 2 slots all bit-identical to standalone greedy refs, eager AND
  `--trace` (traced ~85 ms/iter @ B=2 vs ~252 eager). Admit/advance/evict +
  queueing all correct. **CB1–CB4 complete: a correct, production-speed
  vLLM-style continuous-batching system for 27B.**
- **NEXT (perf/productionization, not correctness)**: chunked prefill (currently
  one-token/step); sampling (DRY/rep-penalty) vs greedy; OpenAI endpoint (user
  deferred); batched owned-GDN kernel to lift the ~232 tok/s compute ceiling.

## Hardware ceiling (the actual target)

P150 measured: **404 GB/s/chip** DRAM BW, 110 worker cores, 31.81 GB DRAM (`feedback_p150_memory_bandwidth_measured`).

For 35B-A3B with ~3 GB active params per token per chip:
- bf16 BW floor: ~3.7 ms/tok → **270 tok/s ceiling**
- bf8 BW floor: ~1.85 ms/tok → **540 tok/s ceiling**

We're 39× over the bf16 floor. **27B's 12.93 tok/s is not the target** — it's where someone else stopped. Every optimization decision below cites a profiling number, not a comparison to 27B.

## Profile-driven next steps

Latest tracy run on batched MoE post-cleanup + SwiGLU fusion:
- Signposted region kernel: 7.45 ms / 4 MoE calls = 1.86 ms/call kernel time
- Signposted op2op gap: 9.48 ms / 4 = 2.37 ms/call dispatch gap
- Dispatch fraction inside MoE region: 0.56 (down from 0.997 pre-batching)
- Median matmul kernel: 930 μs (batched gate_up + down each ~1.84 ms)

**Empirically rejected (2026-05-26):** HiFi2 on expert matmuls — kernel time
identical at HiFi4 / HiFi2+fp32_dest / HiFi2+fp16_dest. Memory-pattern bound,
not math-bound. See `de9d94d`.

**Active candidates** (priority order; see `research/35b_perf_milestones.md`):

1. **Async all_reduce overlap** — `routed_local`'s all_reduce can hide behind
   the shared-expert 4-matmul block. 35B has the parallel work that 27B
   didn't (see `feedback_async_ccl_negative` for the 27B precedent).
2. **Eliminate `ttnn.concat([h_3d] * E_LOCAL, dim=0)`** — 256 KB copy per
   call; pursue custom matmul broadcast or `ttnn.experimental.broadcast`.
3. **Routing-weight construction fusion** — ~9 ops of pure dispatch.
4. **bf8 expert weights** — halves DRAM BW; bf8 KV neutral and bf8 MLP is
   27B production. Low correctness risk; needs single-layer cos probe.

Workflow per the user directive: isolation test → bench in isolation → cos
gate → integrate → tracy re-profile → commit. No projection without measurement.

## Canonical files

- `experiments/serve/server_35b_ttnn.py` — server. Default `state.moe_mode = "pattern_a_batched"`.
- `experiments/utils/trace_demo_full_step.py` — captures + benches the full traced step.
- `experiments/utils/run_tracy_probe.sh` — one-command tracy + tt-perf-report.
- `experiments/utils/tracy_profile_one_moe.py` — canonical scope (one MoE; full forward overflows the marker buffer).
- `experiments/utils/analyze_ops_perf_results.py` — pandas-free CSV analyzer.
- `experiments/utils/test_batched_expert_matmul_isolated.py` — regression test for the production matmul shape.
- `experiments/utils/test_fused_swiglu_isolated.py` — 5-second isolation harness; template for the next fusion probes.
- `experiments/utils/test_pattern_a_moe_tt.py` — topk vs pattern_a_batched cos gate (run after every MoE edit).
- `experiments/utils/delete_line_range.py` — helper for bulk in-place line deletions (no inline `python -c`).

## Read BEFORE doing anything (in this order)

1. **This file** (you're reading it).
2. `research/profiling-quick-reference.md` — capture + analyze workflow.
3. `research/35b_perf_milestones.md` — perf trajectory.
4. `research/moe-cleanup-plan.md` — six-commit cleanup pass to execute.
5. `research/35b_tt_perf_report_findings.md` — full empirical writeup if you need the data behind the advice.

## Non-negotiables (load-bearing)

- **Profile-driven only**. Every optimization cites a number from a tracy/tt-perf-report run, not "X did Y."
- **Hardware ceiling** is the target. Frame every delta as Δ from BW floor.
- **Correctness gate**: every commit produces "Paris, a city renowned…" with prefill IDs `[2614, 314, 279, 369, 11751]` (bit-identical to topk baseline). 5-token Paris is the canary.
- **Concise**: comments only on non-obvious WHY. Git history has WHAT. No inline narrative of debug attempts.
- **Iterations in git history or `scratch/`**, not in demo scripts. Production code = final version only.
- **No /tmp**, **no inline scripts**, **remote-only on qb1**, **frequent commits**.

## Comments that MUST stay (each cost a multi-day debug)

- **View-decay rule** in `moe_forward_ttnn_pattern_a_batched`: never `ttnn.deallocate` a reshape view; clone when in doubt.
- **+1 zero-centered RMSNorm** offset on `q_norm`/`k_norm` (commit `fd4367f`). Qwen3.6-specific.
- **K-broadcast RoPE workaround** in the SDPA path — sidesteps a ttnn `[1, HEAD_DIM]` slice/concat bug.
- **bf16 KV cache** required by paged SDPA (fp32 hard-rejected by the kernel).

## How NOT to waste tokens after compaction

- Don't grep for the production path → it's `state.moe_mode = "pattern_a_batched"`.
- Don't reinstall profiling → read `research/profiling-quick-reference.md`.
- Don't compare to 27B → cite the BW ceiling above.
- Don't propose changes without a tracy/tt-perf-report number → re-run `bash experiments/utils/run_tracy_probe.sh experiments/utils/tracy_profile_one_moe.py` first.

## Recent commits (newest first, last 10)

```
4d8eabc  cleanup plan + profiling quick-ref + tracy probe targets batched
b5d3364  milestones doc — batched-traced 146 ms/tok is the new floor
961ce7f  Pattern A BATCHED works: 267 ms/tok eager = 3.74 tok/s
e8fcfc1  profiling-cheatsheet.md + run_tracy_probe.sh
ae9b591  tt-perf-report wired up — matmul is 99.7% dispatch-bound
3ebb0b2  isolated batched-matmul suite — 9 variants tested in 5s
3b1457f  batched matmul WIP — three ttnn-op constraints, parked
4cac36a  FULL-STEP TRACE WORKS — 308 ms/tok measured
5fe966e  B17-D + DN state in-place — all trace blockers removed
5f4cff8  --moe-mode flag + Pattern A end-to-end correctness PASS
```
