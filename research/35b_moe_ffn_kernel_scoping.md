# Custom batched MoE FFN kernel — scoping (2026-05-26)

Goal: replace the current three-op chain in `moe_forward_ttnn_pattern_a_batched`
with a single fused kernel that keeps intermediate state in L1, so trace
kernel-time drops (the trace-amortization wall blocks dispatch-only fusions).

## What we're replacing

```python
# Three back-to-back ttnn ops in moe_forward_ttnn_pattern_a_batched:
gate_up = ttnn.matmul(h_3d_repeat, w["experts_gate_up_local"], ...)   # ~1.84 ms
gate    = ttnn.slice(gate_up, [0,0,0], [E,1,MOE_INTER])               # ~50 us
up      = ttnn.slice(gate_up, [0,0,MOE_INTER], [E,1,2*MOE_INTER])     # ~50 us
mid     = ttnn.mul(gate, up, input_tensor_a_activations=[SILU])       # ~30 us
out     = ttnn.matmul(mid, w["experts_down_local"], ...)              # ~1.84 ms
expert_out_2d = ttnn.reshape(out, [E, HIDDEN])
rw_1xK = ttnn.reshape(routing_weight, [1, E])
routed = ttnn.matmul(rw_1xK, expert_out_2d, ...)                      # tiny
```

Total: ~3.7 ms kernel time per MoE call × 40 layers = 148 ms eager-side.
In trace: ~half of that (the matmuls themselves don't shrink, only dispatch
overhead between them does).

## Shapes (per chip, NCHIPS=4)

- `h`: `[1, HIDDEN=2048]` bf16 — replicated. 1 row × 64 tile-cols (TILE=32).
- `W1 = experts_gate_up_local`: `[E_LOCAL=64, HIDDEN=2048, 2*MOE_INTER=1024]` bf16
  - DRAM bytes: 64 × 2048 × 1024 × 2 = **256 MB/chip**
  - In tiles: 64 experts × 64 × 32 = 131,072 tiles per chip
- `W2 = experts_down_local`: `[E_LOCAL=64, MOE_INTER=512, HIDDEN=2048]` bf16
  - DRAM bytes: 64 × 512 × 2048 × 2 = **128 MB/chip**
- `routing_weight`: `[E_LOCAL=64]` bf16 — small, scalar per expert
- Output `routed`: `[1, HIDDEN=2048]` bf16 — 64 output tile-cols, summed across experts

Total DRAM read per call per chip: 384 MB + 4 KB scalars. At 1.84 ms per
matmul this is 384 MB / 3.68 ms = **104 GB/s** of W1+W2 traffic, vs P150 peak
of 404 GB/s = **26% of BW peak**. Compute peak: 9.25 TFLOPS/chip; we hit ~150
GFLOPS = **1.6% of compute peak**. So the matmul is neither BW-bound nor
compute-bound at these shapes — it's tile-overhead and core-utilization bound.

## Hardware budget (P150 per chip)

- 110 worker Tensix cores
- L1 per core: 1.39 MiB (1408 KB user-allocatable)
- DRAM: 31.83 GB / chip, 8 banks × 3.979 GiB, peak 404 GB/s measured
- HiFi4 with fp32 dest accumulator on every matmul (kept for correctness)

## Patterns we already have to draw from

The owned-GDN kernel (`experiments/owned_ops/qwen36_gdn_decode_owned/`)
defines the canonical template:

- **SPMD work unit**: `(slot, value_tile)`. For GDN: 8 slots × 4 value_tiles =
  32 blocks across 110 cores (each core gets ≤1 block; ~78 cores idle).
- **CB layout**: 18 circular buffers in L1, double-buffered (depth 2) for
  intermediates, depth `key_tiles * 2` for K-stream CBs.
- **Three-RISC split**: reader (data movement in), compute (math), writer
  (data movement out). Compute does internal K transpose via
  `transpose_wh_tile` when k_col not supplied.
- **In-place state writes**: `(state, out)` returned; state buffer mutated
  directly by writer.
- **Microbench result**: fused traced 0.026 ms vs component-chain traced
  0.041 ms — ~38% trace savings on the GDN micro-fixture.

The applicable lessons:

1. Block-level SPMD + work-split-to-cores helper handles uneven distributions.
2. Keep all intermediates in L1 CBs — only DRAM-touch the inputs (h, W1, W2,
   routing_weight) and the final output (routed).
3. Mode-by-mode ablation framework: write the skeleton (read/write/fill only)
   first, then add compute stages, microbench each addition. The mode2-mode9
   ablations in the GDN kernel found that K transpose was the largest sub-cost.

## Design candidates for the MoE FFN kernel

### Candidate A: one core per `(expert, output_col_tile_of_routed)`

```text
work_units = E_LOCAL × (HIDDEN / TILE) = 64 × 64 = 4096
cores = 110
items_per_core = 4096 / 110 ≈ 37
```

Each work unit computes one column-tile of `routed` for one expert. Sequence:

1. **read h**: 1 row × 64 tiles = 64 tiles of bf16, ~4 KB. Read ONCE per core
   (or mcast from a single reader core). Fits trivially in L1.
2. **read W1[expert]**: 2048 × 1024 = 64 × 32 tiles = 2048 tiles × 2 KB =
   4 MB. Streamed through CBs.
3. **compute gate_up_partial**: 1×64 × 64×32 = 32 output tiles for one
   expert's gate_up. Held in L1 as a `[1, 1024]` bf16 vector = 64 tiles ≈ 2 KB.
4. **slice + silu*up**: completely in L1, produces `mid = [1, 512]` = 32 tiles
   ≈ 1 KB.
5. **read W2[expert]**: 512 × 2048 = 16 × 64 tiles = 1024 tiles ≈ 2 MB.
6. **compute expert_out_partial** for this output_col_tile: dot(mid, W2[:,
   output_col]) = 32 reductions for 1 output tile, ~1 KB.
7. **scale by routing_weight[expert]**: 1 op on 1 tile.
8. **partial sum** into a shared `routed_acc` tile in L1, requiring inter-core
   synchronization to accumulate across the 64 experts that contribute to
   the same output tile.

L1 budget per core: ~4 MB W1 streaming (chunked) + ~2 MB W2 streaming + ~10
KB of intermediates. Well within 1408 KB per core IF streaming is done well.

**Risk**: 64 experts × 64 output tiles partition is fine for parallelism
(4096 items / 110 cores), but the cross-core reduction at step 8 needs a
ring-reduce or atomic-add pattern. This is the hard part — owned-GDN never
does cross-core sum (each slot is independent).

### Candidate B: one core per `expert`, sequential through outputs

```text
work_units = E_LOCAL = 64
cores = 64 active (46 idle)
each core does: gate_up → silu*up → down → scale by rw[e] for all 64 output tiles
```

Per core: 4 MB W1 + 2 MB W2 streamed sequentially through CBs. Output is a
per-expert `[1, HIDDEN]` partial vector = 64 tiles in L1 (2 KB).

After all 64 cores finish, ring-reduce or all-gather across cores to sum
the 64 per-expert vectors into the final `routed`. This is one cross-core
collective, much simpler than 64 of them.

**Risk**: only 64/110 cores active = 58% utilization. Doesn't beat the
current matmul kernel's likely 50-70% utilization either, so might not win.

### Candidate C: hybrid — work on `(expert, output_row_tile)` with mcast

Since `h` is `[1, HIDDEN]` (M=1 = ONE row tile), output `routed` is also
[1, HIDDEN] (M=1). Both fit in a single tile-row, so the "row" dimension is
trivial. The real parallelism is over `expert` and `output_col`.

The trick here is to mcast `h` to all cores once, then have each core handle
its (expert, output_col_tile_subset) chunk independently, and at the end
do a single tree-reduce of the 64 partial routed vectors.

This is essentially A + B unified.

### Candidate D: don't fuse the two matmuls; fuse the second-matmul reduction

Looking again at the original chain: the FINAL op is
`routed = matmul(rw_1xK, expert_out_2d)` which is a `[1, 64] × [64, 2048]`
= sum over 64 experts of `rw[e] * expert_out[e]`. This is small (1.8M FLOPs)
and currently dispatches as a separate matmul.

A custom op that fuses `(expert_out_batched, routing_weight) → routed_local`
inside the kernel (using the per-expert results without ever writing to
DRAM) might be the right granularity.

This DOESN'T eliminate the two big matmuls — they stay as ttnn ops. It DOES
eliminate the expert_out_batched DRAM round trip and the explicit reduce
matmul. Smaller scope, lower risk.

## Recommended ladder (G0 → G3, matching the owned-GDN cadence)

Following the owned-GDN G0-G4 staged pattern (memory `feedback_build_kernels_from_scratch`):

- **G0 — Skeleton**: kernel registered, takes inputs, writes correct-shape
  zeros to output. Verify the build, the binding, the program_factory plumbing.
  Estimated: ~250 LOC, no math.
- **G1 — Candidate D (reduce only)**: implement just the
  `(expert_out_batched, routing_weight) → routed_local` reduction in-kernel.
  Lowest-risk first step, gives us a working custom op to extend. Estimated
  +100 LOC compute kernel.
- **G2 — Candidate B (one-core-per-expert with full chain)**: scale up to
  the full gate_up + silu*up + down + scale per expert, with a final ring
  reduce. Validate correctness against the current 3-op Python chain.
  Estimated +300 LOC.
- **G3 — Candidate C (hybrid mcast)**: optimize core utilization by mcasting
  h and partitioning the per-expert work across multiple cores. This is
  where we expect to actually beat the current chain in trace time.
- **G4 — production wire-in**: behind `state.moe_owned_ffn` toggle, default
  off until cos gate + Paris canary + needle-haystack pass.

## Expected savings (very rough)

Hard to predict without microbenchmarking. Lower bound: eliminate the two
intermediate DRAM round-trips (gate_up = 128 KB write+read, expert_out =
256 KB write+read = ~1 ms at 200 GB/s effective BW per chip). Upper bound:
if work-split-to-cores wins over the matmul kernel's M=1 utilization, kernel
time could drop 30-50% = ~1-2 ms saved per MoE call × 40 layers = 40-80 ms
per token IF this generalizes from eager to trace.

Reality check: the GDN kernel's own microbench showed traced fused 0.026 ms
vs traced chain 0.041 ms = 0.015 ms per call. At 30 DN layers that's 0.45
ms/token — which matches the ~0.2-0.5 ms/token we actually measured shipping
owned-GDN + owned-decay-gate (most of the eager win evaporates in trace).

So even the BEST case via this kernel might be ~5-10 ms/token in trace, not
the 40-80 ms upper-bound math suggests. Worth doing, but expectations stay
modest.

## Open questions

1. Does the matmul kernel currently use all 110 cores, or is it constrained
   by M=1? A quick tracy on one matmul (gate_up alone) would answer this.
2. Cross-core reduction primitive: tt-metal has `tt_metal::CoreRangeSet`
   mcast helpers and the existing all-reduce CCL — can the kernel piggy-back
   on these, or do we need to write an inter-core barrier from scratch?
3. Weight DRAM layout: experts_gate_up_local is currently
   `[NCHIPS*E_LOCAL=256, H, 2I]` sharded dim 0. The kernel needs to address
   per-expert tile-rows of W1/W2 efficiently — the existing layout should
   work but worth confirming.
4. Numerics: HiFi4 + fp32_dest_acc must be preserved. The owned-GDN kernel
   uses `init_device_compute_kernel_config(arch, ..., HiFi2, false, true)`
   internally — we'll need HiFi4 since this is the production-correctness
   path, not a B3 SDPA-style relaxation.

## Next step (Session N+1)

Pick D (reduce only) as G1 first because:
- Smallest kernel scope (~100 LOC compute).
- Validates the program_factory + nanobind binding plumbing without taking
  on cross-core reduction or work-split complexity.
- Eliminates one real DRAM round trip (`expert_out_batched` write + reduce
  read) so a measurable trace improvement (even 0.5 ms) tells us the lower
  bound on what the full fusion can deliver.
- If G1 doesn't move trace ms/tok at all, that's strong signal to STOP and
  focus elsewhere — we'd avoid spending weeks on a kernel that the trace
  layer is going to swallow regardless.
