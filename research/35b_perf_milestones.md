# 35B-A3B perf milestones — single source of truth

Per-token decode latency on Qwen3.6-35B-A3B, qb1 (1,4) P150 mesh. Every number
sync-bounded (`ttnn.synchronize_device` before and after), 10+ steady-state
iterations, prefill+decode bit-identical to topk baseline.

**Target = hardware ceiling, not 27B parity.** P150 measured BW = 404 GB/s/chip
(`feedback_p150_memory_bandwidth_measured`). For 35B-A3B with ~3 GB active
params/token/chip: **bf16 floor ≈ 3.7 ms/tok = 270 tok/s**, bf8 floor ≈ 1.85
ms/tok = 540 tok/s.

| Date | Mode | ms/tok | tok/s | Δ from BW floor (bf16) | Commit |
|---|---|---|---|---|---|
| 2026-05-24 | topk eager (baseline) | 480 | 2.08 | 130x | `fd4367f` |
| 2026-05-24 | looped Pattern A traced | 308 | 3.24 | 83x | `4cac36a` |
| 2026-05-25 | batched Pattern A eager | 267 | 3.74 | 72x | `961ce7f` |
| 2026-05-25 | batched Pattern A TRACED | 146 | 6.85 | 39x | — |
| 2026-05-26 | + cleanup pass (refactor, no perf change) | 146 | 6.85 | 39x | `eb237fd` |
| 2026-05-26 | + fused SwiGLU + DN SILU + shared SIGMOID | 145.1 | 6.89 | 39x | `90b1518` |
| 2026-05-26 | + qwen36_gdn_decode_owned (FIRST COHERENT GEN) | 143.8 | 6.95 | 39x | (this row) |

## The trace-amortization wall (2026-05-26)

Three correct, bit-identical activation fusions landed (SwiGLU on batched MoE
experts, SILU on DN RMSNormGated tail, SIGMOID on shared-expert gating). Each
saves ~1 dispatch per call × 30-40 layers. Isolation showed clean speedups
(1.72x-1.97x). End-to-end traced number moved from 146.0 to 145.1 ms/tok.

Why so little gain in trace? **Trace already eliminates ~all dispatch
latency.** A dispatch-reducing fusion in eager translates to almost nothing
in trace — there's no host-side dispatch loop to shorten. Future per-token
wins must come from one of:

1. Reducing **kernel time per op** — but the dominant ops (gate_up @ 1.84 ms,
   down @ similar) aren't compute or BW bound at these shapes. HiFi2 and bf8
   both no-op'd here.
2. Reducing **op COUNT** in trace — each op still has trace-internal overhead
   (~tens of microseconds). Fewer, larger ops > many tiny ones.
3. Writing **custom fused kernels** that genuinely combine multiple math
   stages (silu+mul+matmul, or DN-step-fused). These DO move kernel time
   per call. Cost: tt-metal C++ work + correctness gate.

## Block-attribution profile (2026-05-26 — eager, post-fusion)

`profile_blocks_35b_ttnn.py` with all 3 MoE variants wrapped:

| Block | total/tok (ms) | per-layer (ms) | layers | share |
|---|---|---|---|---|
| MoE   | 148.0 | 3.70 | 40 | 48.2% |
| DN    | 133.8 | 4.46 | 30 | 43.6% |
| ATTN  |  25.1 | 2.51 | 10 |  8.2% |

Eager total ~331 ms/tok; traced 145.1 ms/tok. Assuming proportional collapse
in trace: MoE ≈ 65 ms/tok, DN ≈ 57 ms/tok, ATTN ≈ 11 ms/tok, other ≈ 12 ms.
Both MoE and DN are major. Single-block 50% reductions (custom kernels) would
move us from 145 → 110-115 ms/tok = 8.7-9.1 tok/s.

The DN per-layer (4.46 ms) is HIGHER than MoE per-layer (3.70 ms) — DN's
~15-op recurrence + L2-norm sequence is the per-layer hot path. 27B has an
owned_gdn kernel (memory `feedback_owned_decay_gate_shipped`, +2.5%) that
fused the analogous chain. Bringing similar for 35B is the next major lever.

## Where the wins came from

1. **Correctness foundation** (`fd4367f`): q/k_norm `+1` zero-centered offset.
2. **Pattern A MoE** (Mixtral-style): on-device top-k mask × all-experts
   compute eliminates the host readback that blocked trace.
3. **Trace capture**: amortizes the 5120-op-per-token dispatch overhead.
4. **Batched expert matmul**: 5120 expert matmuls per token → 80. Biggest single win.
5. **Fused mul+sum reduction** as one matmul: `mul(expert_out, rw) + sum(dim=0)`
   ≡ `matmul(rw_1xK, expert_out_2d)`. Sidesteps view-decay on broadcast.
6. **Fused SwiGLU**: `ttnn.mul(gate, up, input_tensor_a_activations=[SILU])`
   collapses silu+mul into one BinaryNg. 1.87x isolated; ~80 dispatches/token
   saved across MoE + shared expert.

## Empirically rejected

- **HiFi2 on expert matmuls** (2026-05-26, `de9d94d`). tt-perf-report advice
  said 2x kernel speedup; measurement showed kernel-time identical at all
  three configs (HiFi4+fp32_dest=931μs, HiFi2+fp32_dest=931μs, HiFi2+fp16_dest=930μs).
  The MoE expert matmul is memory-pattern bound at shape [64,1,2048]@[64,2048,1024]
  on Blackhole — not math-bound. Dropping fp32_dest_acc additionally risked
  long-context fp16 accumulator drift (user veto, per 27B precedent).
- **Async all_reduce overlap with shared-expert window** (2026-05-26, `dd6b665`).
  Isolation harness on (1,4) mesh with the production-equivalent comm + 4 shadow
  matmuls: cos(serial, async) = 0.99999931, but async was 1.2% slower
  (1.331 vs 1.315 ms/iter). Async setup tax exceeds the overlap savings; the
  shared-expert compute window isn't deep enough to amortize the semaphore
  pool setup. Reaffirms `feedback_async_ccl_negative` for serial-with-side-comm
  patterns.
- **bf8 expert weights** (2026-05-26, reverted same commit). cos(topk, batched)
  = 0.99999668 — clean pass. But tracy showed kernel-time identical at 930 μs
  median (vs bf16 930.6 μs) and signposted op2op within noise (9.13 vs 9.48 ms).
  Back-of-envelope: at gate_up shape, weight DRAM read = 64 MB/chip / 404 GB/s
  ≈ 158 μs vs actual 1837 μs — we're 11x above the BW floor, so the matmul
  isn't BW-bound. The 27B precedent (bf8 MLP in prod) makes correctness
  reliable, but no tt-perf-report-visible perf delta means no ship per
  user's "must be guided by tt-perf-report" gate. Saved as a memory-pressure
  lever for future DRAM-bound contexts (long prefill, vocab-sharded LM head).

## What's left (profile-driven candidates)

Re-profile the post-fusion traced batched path to confirm new bottleneck.
Candidate optimizations ranked by expected leverage:

1. **Async all_reduce overlap** — fire `routed_local`'s all_reduce, run the
   shared-expert 4-matmul block in the shadow, sync at the final add. The
   shared expert is independent compute that can hide the reduce latency.
   Memory note `feedback_async_ccl_negative` says async lost in 27B *for
   serial residual streams* — 35B MoE has the parallel work to win.
2. **Eliminate `ttnn.concat([h_3d] * E_LOCAL, dim=0)`** — 64x duplication of
   h (~256 KB copy per MoE call) into a custom matmul that broadcasts dim 0,
   or use `ttnn.experimental.broadcast` if it exists. Variant-A of the
   isolation suite tried rank-4 broadcast — failed at the time, but
   post-cleanup the matmul API may have moved.
3. **Routing-weight construction fusion** (eq → typecast → reshape → mul →
   reshape → sum → reshape → clone → reshape, ~9 ops) collapsed via custom
   kernel. Lower priority — pure dispatch overhead, kernel time near zero
   per op.
4. **bf8 expert weights** — halves DRAM BW for the expert weight read. Memory
   notes bf8 KV cache was NEUTRAL and bf8 MLP was already in 27B production,
   so correctness risk is low. Worth a careful single-layer cos probe.

Each gets an isolation test → bench → cos gate → integrate. No projection
without measurement.
