# tt-perf-report findings — 35B traced decode (2026-05-25)

## TL;DR

Profiled an eager forward via Tracy + tt-perf-report. **Every Pattern A
expert matmul is "SLOW" (per tt-perf-report's bound classifier) and is
99.7% dispatch-bound by op-to-op gap.** Median matmul kernel time is
**29.8 μs**; median op-to-op gap is **3411 μs (3.4 ms)**.

**Batched matmul IS the right next step.** Eager dispatch dominates; trace
helps but doesn't eliminate it. Cutting 5120 expert matmuls per token down
to 80 batched ones is the right strategy.

## Setup

- `pipx install tt-perf-report` (1.2.4) on qb1
- Patched `tracy/process_ops_logs.py` (line ~561) — `assert candidates`
  becomes `if not candidates: continue` so the merge step doesn't abort
  on traced-replay ops that don't match host op_ids
- Tracy capture wrapper: `python -m tracy -r -p -v -o <dir> <script>`
  - Tracy spawns a subprocess that calls `python3 -m tracy <script>`;
    must add the venv bin dir to PATH so the subprocess uses the venv
    python (where `tracy` is installed), not `/usr/bin/python3`

Probe: `experiments/utils/tracy_profile_one_moe.py` — bootstraps Pattern A,
calls `moe_forward_ttnn_pattern_a` ONCE inside a `tracy.signpost()` region.
Profiling the full step overflows Tracy's per-RISCV 12000-marker DRAM buffer
("Profiler DRAM buffers were full, markers were dropped!"); one MoE call
fits and gives the same per-op insight (MoE shape repeats across all 40
layers).

## Matmul stats (Pattern A MoE, eager)

From 328 matmul calls with valid device timing:

| percentile | kernel [μs] | op-to-op gap [μs] |
|---|---|---|
| min  | 3.5  | 113 |
| p25  | 29.6 | 3243 |
| **median** | **29.8** | **3411** |
| p75  | 29.9 | 3500 |
| p95  | 30.0 | 33035 |
| max  | 30.4 | 168424 |

- Total kernel time across all measured matmuls: **8.4 ms**
- Total op-to-op gap: **2656 ms**
- **Dispatch fraction = 0.997**

The kernel-time distribution is extremely tight (~30 μs across all
percentiles up to p95), confirming the matmul kernel itself runs
predictably. The tail (p95 33 ms, max 168 ms) is dispatch hiccups
between ops, not kernel variance.

## tt-perf-report's advice (per matmul)

For every MoE matmul shape (`32×2048×1024`, `32×512×2048`, `32×2048×128`):

1. **"Output subblock 1x1 is small, try `out_subblock_h * out_subblock_w >= 2`"**
   — kernels operate on M=1 (one decode token padded to TILE=32). Limits
   per-core throughput.
2. **"HiFi2 may also work, it discards the lowest bit of the activations
   and has 2x the throughput of HiFi4"** — we're currently using HiFi4
   for all matmuls (set in commit `fd4367f`, q/k norm correctness fix).
   For activations specifically, HiFi2 may be fine and ~2× faster.
3. **"If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)"**
   — putting h_tt in L1 instead of DRAM could cut weight-load BW pressure.

## Per-op counts (Pattern A MoE, one call)

The signposted region had 3420 ops total (4 MoE warmup+perf calls):

| op | count | per call |
|---|---|---|
| MatmulDeviceOperation | 532 | 133 |
| SliceDeviceOperation | 1288 | 322 |
| BinaryNgDeviceOperation | 788 | 197 |
| UnaryNgDeviceOperation | 264 | 66 |
| TilizeWithValPaddingDeviceOperation | 252 | 63 |
| UntilizeWithUnpaddingDeviceOperation | 248 | 62 |
| SoftmaxDeviceOperation | 4 | 1 |
| TopKDeviceOperation | 4 | 1 |
| ReduceDeviceOperation | 8 | 2 |
| ReduceScatterDeviceOperation | 8 | 2 |
| AllGatherDeviceOperation | 8 | 2 |
| FillPadDeviceOperation | 12 | 3 |

Pattern A MoE per call = **~133 matmuls + ~322 slice + ~197 binary + …
≈ 1170 device ops per MoE × 40 layers = 46,800 ops per token**. The
ratio matches "5120 matmuls per token" we'd been using as a rough estimate.

The slice/binary/silu count is consistent with the Pattern A loop:
each of 64 experts does (slice gate, slice up, silu, mul, slice
routing_weight, mul, add) = ~7 helper ops × 64 = 448 helpers. Plus per-
expert weight slicing (slice gate_up, reshape, slice down, reshape) =
~4 more × 64 = 256.

## What this means for the next optimization

1. **Batched matmul is the right target.** Reducing 133 matmuls per MoE
   call to 2 (one gate_up batched, one down batched) cuts ~131 × 3.4 ms =
   **445 ms of dispatch per MoE call in eager**. At 40 layers that's the
   entire forward.

2. **Trace doesn't fully eliminate dispatch.** Traced execute_trace at
   308 ms/tok still has substantial inter-op gap (we'd need to profile
   a traced run for the exact number, but if trace cut dispatch from
   3.4 ms to even 0.1 ms per op, we'd save 5120 × 3.3 = 16.9 sec/tok →
   our actual traced is 308 ms, meaning trace already eliminated most
   of it; remaining gap is ~50-60 ms which is probably the floor without
   batching).

3. **HiFi2 → HiFi4 swap for activations** is a free 2× kernel speedup,
   but the kernel is 30 μs of a 3411 μs total per matmul — saves at most
   ~15 μs per matmul = 80 ms per token in eager, much less in trace. Worth
   doing AFTER batching, not before.

4. **L1 input placement** is a similar micro-optimization — meaningful
   if we're already past dispatch, not before.

## Action items

- [x] Set up tt-perf-report (done; patch + capture wrapper documented above)
- [ ] **Solve the in-server batched matmul failure** (current task #55).
  Isolation showed variant H (rank-3 sharded `[256,H,2I]` + h pre-broadcast
  via concat/repeat) works on production-like h state. In-server fails at
  `ttnn.mul(expert_out × routing_weight)` even with pre-broadcast — needs
  more isolation work (extend the isolated suite to mock the 40-layer
  memory pressure or the deep op-chain h state)
- [ ] After batched lands: profile traced batched decode, check the new
  per-op gap. If still dispatch-bound, look at fused ops; if BW-bound,
  look at bf8 weights and L1 placement.

## Files

- `experiments/utils/tracy_profile_one_moe.py` — Tracy probe for one MoE call
- `experiments/utils/tracy_profile_traced_decode.py` — fuller trace probe (overflows DRAM buffer)
- `experiments/utils/_patch_tracy_assertion.py` — idempotent venv patch
- `experiments/utils/analyze_ops_perf_results.py` — pandas-free CSV analyzer
- `.cache/perf_logs/tracy_one_moe/reports/*/ops_perf_results_*.csv` — on qb1
