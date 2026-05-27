# MoE FFN kernel — perf deferrals (track-as-we-build)

Running list of correctness-first shortcuts in the qwen36_moe_ffn_decode_owned
kernel. Each row is a place where we picked the simpler/safer option for
faster bring-up; the right column is the perf-window for revisiting once
the kernel is correct.

| ID | Stage | Shortcut | Perf cost (est.) | Revisit lever |
|---|---|---|---|---|
| D-G0-01 | G0 | Single core only (1 of 110 Tensix cores) | ~99% of compute capacity idle | Move to G2 multi-core split; one core per expert. |
| D-G0-02 | G0 | bf16 only (no fp32 path) | Locks downstream to bf16 KV/state; no fallback for long-context fp32 experimentation | Add fp32 branch in validate + compute kernel once correctness is stable. |
| D-G0-03 | G0 | Single device (no mesh sharding) | Can't run on (1,4) production mesh | G4 mesh adapter + ShardTensorToMesh-aware accessors. |
| D-G0-04 | G0 | Reader streams h but compute discards the data | Wasted DRAM read of h (4 KB per call) | Trivial cleanup in G1 once we actually USE h. |
| D-G0-05 | G0 | (Retired) The zero-emit-via-acquire approach failed to compile (`init_sfpu` missing); G0 now emits IDENTITY (copy h to out) using the 27B decay_gate kernel's init pattern. | — | — |
| D-G0-05b | G0 | G0 compute is identity (copy h to output) instead of "fill with zero". Smoke test asserts output == h instead of all-zero. | None — stronger plumbing check (proves data flows reader→compute→writer, not just that writer drained the CB) | Identity gets replaced by real fused FFN math in G1; this row retires there. |
| D-G0-06 | G0 | No CB sizing tuning — all CBs are double-buffered (depth 2), no analysis of which want depth=2 vs key_tiles*2 | Possibly some bubble cycles in the pipeline | Profile via tracy after G1 lands; tune deep buffers for the matmul streams. |
| D-G0-07 | G0 | HiFi4 + fp32_dest_acc_en hardcoded in program_factory | Matches production matmul fidelity (correct for production); can't experiment with HiFi2 without rebuilds | Plumb compute_kernel_config through the op as the friend-repo `qwen36_gdn_decode` does. |
| D-G0-08 | G0 | Compute kernel runs the same loop regardless of `debug_fill` | No actual debug-fill behavior yet | Wire debug_fill = copy h's first tile to output once G1 reads h for real. |
| D-G1a-01 | G1a | Ignore routing_weight entirely (treat rw[e] = 1.0 for all e). Output is plain `sum_e expert_out[e]` instead of weighted sum. | Correctness loss for real model — must be re-enabled before integration. | G1b adds rw; deferral retires there. |
| D-G1a-02 | G1a | Single-core loop over all E experts. 1 of 110 cores used. | ~99% compute idle. | G2 splits work per-expert across cores. |
| D-G1a-03 | G1a | Reader streams W1[e]/W2[e] sequentially per expert (no overlap across experts). | Reader bandwidth not pipelined with compute on expert boundaries. | G2's multi-core split makes per-expert streaming the natural granularity. |
| D-G1a-04 | G1a | CB_PARTIAL is implicitly zero-initialized on the very first expert iteration by NOT calling matmul_tiles' accumulate flag on iteration 0 (separate "first iter" branch). | Branchy compute kernel; one extra conditional per output tile per expert. | Pre-fill CB_PARTIAL with zeros via a small init pass; eliminates the branch. |
| D-G1a-05 | G1a | gate_up resident as 2*MOE_INTER/TILE tiles in L1; split into gate/up by tile-index instead of slice. | None functional — but means we can't easily swap to a streamed gate_up if memory pressure changes. | Re-evaluate when production HIDDEN/MOE_INTER scales up. |
| D-G1a-06 | G1a | Validate restricted: W1 rank-3 [E, HIDDEN, 2*MOE_INTER], W2 rank-3 [E, MOE_INTER, HIDDEN]; no sharded layouts. | Won't run on (1,4) mesh-sharded tensors yet. | G4 mesh adapter. |
| D-G1b-01 | G1b | `routing_weight` is consumed as pre-broadcast `[E, TILE, TILE]` (each `[e]` slab filled with `rw[e]`). Caller (Python wrapper) must broadcast a `[1, E]` rw into this shape before invoking. | One extra ttnn op + ~E*TILE*TILE*2 bytes per call (≈ 128 KB for E=64). | LLK-level row-scalar-to-tile bcast (`mul_tiles_bcast<ROW>` or pack one rw lane to scalar reg + `mul_unary`) would consume `[1, E]` directly. |
| D-G1b-02 | G1b | Compute multiplies `eo[j]` by `rw_broadcast[e]` BEFORE accumulating, requiring one extra mul_tiles per (e, j). | One additional ~32×32 mul per output tile per expert. ~Negligible for E=64, HIDDEN_TILES=64 vs the matmul cost. | Could fuse into matmul_reduce's pack step if mm supports scalar-output-scale. |

**G1b PASS (2026-05-26)**: H=64, I=32, E=2 toy shape, `pcc=0.99998756`,
`max_abs_diff=2.44e-4` vs bf16 numpy oracle. Routing-weight scaling
verified end-to-end. Next stage G2 splits work across cores.

## Deadlocks caught at large-shape (H=256, I=128, E=8) — methodology lesson

**Bug 1 — CB sizing**: `cb_wait_front(cb_w1, hidden_tiles)` requires
CB_W1 depth ≥ hidden_tiles. G0/G1 had CB_W1 depth = 2, which happened
to equal hidden_tiles at the toy shape but deadlocked at production-ish
shapes. Same trap for cb_w2 vs mid_tiles. **Rule:** any `wait_front(cb, N)`
must have CB depth ≥ N — at *runtime* N, not compile-time. This was
mis-classified as "D-G0-06 CB sizing untuned" (perf deferral) when it
was actually a correctness invariant.

**Bug 2 — producer ordering**: Reader pushed `W1 → W2 → rw` per expert.
Compute waits on `cb_rw` *inside* the eo block, after silu*up and at the
start of the W2 drain loop. At large shape: reader fills cb_w2 to depth
(2*mid_tiles=8) while only 1 of 32 W2 tiles for the expert have been
pushed, then blocks on cb_reserve_back(cb_w2). Compute, waiting on
cb_rw which is empty, can't drain cb_w2. **Reader must produce rw
*before* W2** (and ideally before W1 too) per expert.

**Lesson**: toy shapes that satisfy `hidden_tiles ≤ small_CB_depth`
and `mid_tiles*hidden_tiles ≤ small_CB_depth` mask both classes of
deadlock. Future G-stages: stress-test at hidden_tiles ≥ 8 and
mid_tiles*hidden_tiles ≥ 2*small_CB_depth before claiming correctness.

(Rows will be added as G1, G2, G3 land. Use this doc as input to a future
"perf cleanup pass" session once the kernel is correct.)

## Why we deferred each

- **D-G0-01 / D-G0-03 / D-G0-04**: G0 is a build/plumbing check. Real
  multi-core + mesh work needs the device op's validate logic to handle
  sharded tensors and the program factory to call `split_work_to_cores`.
  Adding both at G0 obscures whether the basic dispatch path works.
- **D-G0-02**: bf16-only matches every existing owned kernel (decay_gate,
  conv1d_owned, gdn_owned). Adding fp32 doubles the test matrix.
- **D-G0-05**: This is the only correctness risk in G0. If DST isn't zero
  on acquire, our G0 output is garbage and we get false smoke-test pass
  if the harness happens to compare to zeros and the garbage is also zeros
  due to allocator state. Mitigation: G0 smoke test should fill the output
  buffer with NON-ZERO sentinel before the kernel call, then assert all
  zeros after.
- **D-G0-06**: CB depth tuning is profile-driven; pointless before the
  pipeline does meaningful work.
- **D-G0-07**: Production correctness is the priority; matching 91f's
  HiFi4 + fp32_dest_acc keeps us on the validated numerics path.
- **D-G0-08**: G0 doesn't read h's data so debug_fill has nothing to do.

## Format for new entries

```
| D-Gn-NN | Gn | <one-sentence shortcut> | <perf-cost estimate> | <revisit plan> |
```
