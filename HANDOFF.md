# HANDOFF — read top to bottom (replaces the 900-line version, see `git log HANDOFF.md` for the old narrative)

## What this project is

35B-A3B (MoE) bringup on Tenstorrent Blackhole (P150 × 4 mesh on `qb1`). Research-driven; goal is **the hardware ceiling**, not parity with someone else's number.

## Where the perf is RIGHT NOW

| Mode | ms/tok | tok/s |
|---|---|---|
| **Batched Pattern A traced + activation fusions** | **145.1** | **6.89** |

Production path: `state.moe_mode = "pattern_a_batched"` in `experiments/serve/server_35b_ttnn.py`. Run via `experiments/utils/trace_demo_full_step.py --moe-mode pattern_a_batched`.

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
