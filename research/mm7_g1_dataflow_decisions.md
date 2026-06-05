# MM7 G1 — explicit dataflow / hardware-mapping decisions

**Owner**: G1 kernel build (task #186).
**Purpose**: every decision we make about how the Mamba2 SSD recursion maps
to Blackhole hardware is **a future optimization lever**. This doc
catalogues each decision **D{N}** with the alternative considered, the
rationale, and the future-revisit trigger. When G2/G3/v3 perf work
fires, this is the menu to revisit.

Companion to `research/mm7_g1_mamba2_kernel_design.md` (the high-level
plan) and `wiki/66_blackhole_kernel_dataflow_anatomy.md` (the pedagogy).

------------------------------------------------------------------------

## D1 — SPMD work unit = `(batch, head)`

**Decision**: Each Tensix core handles ONE (batch, head) pair per
kernel invocation. At B=1 → 64 blocks, 64 cores (46 idle of 110).

**Alternatives considered**:
- `(batch, head, head_dim_chunk)` — finer partition; fits more
  concurrent work per chip but adds reader fan-out.
- `(batch, group=h//8)` — coarser; each core handles 8 heads
  (one B/C group). Saves per-group B/C re-reads but serialises
  more work per core.

**Rationale**: at B=1 the per-(batch, head) compute (~16K FMAs) is
small enough that even 1 head/core leaves headroom. The 46 idle cores
let us scale to B=2 (128 blocks) without contention. Single-head
sharding also has the simplest reader contract: each core reads its
own complete slice from DRAM with no inter-core fan-out.

**Future opportunity**: at B=1 specifically, the
`(batch, head, head_dim_chunk)` split would put 128 blocks on the
mesh (64 heads × 2 head_dim_chunk tiles), saturating cores. Probably
not worth it because the per-block FMAs would drop from 16K to 8K
(below the dispatch-overhead break-even). Confirm with Tracy if v3
demands further speedup.

**Revisit trigger**: if Tracy at B=1 shows cores idle ≥30% of step
time AND `step_ms` is decode-time-critical.

------------------------------------------------------------------------

## D2 — `ssm_state` lives in interleaved DRAM, NOT L1-sharded

**Decision**: Each (batch, head)'s 8-tile fp32 `ssm_state` slice lives
in DRAM, gets read into L1 CB at the start of each kernel invocation,
gets written back to DRAM at the end.

**Alternatives considered**:
- **L1-sharded**: pre-assign each Tensix a permanent L1 slice for its
  head's state. Skip the DRAM ↔ L1 transfer between steps.
- **Hybrid**: state on L1 for the "hot" heads (rare; SSD has uniform
  per-head access), DRAM for the cold ones.

**Rationale**: L1-sharded would tie head → core 1:1 forever. That
locks us out of CB at B>1 (B slots × 64 heads = 32×64 = 2048 slot-heads,
≫ 110 cores). Need to remap each token. DRAM is the right home for
per-slot state in a multi-tenant CB.

**Cost of the choice**: per (batch, head) per step =
8 tiles × 4 KiB × 2 (read+write) = **64 KiB DRAM traffic**.
At B=8 × 64 heads = 512 (batch, head) = **32 MiB / token / layer**.
Across 23 Mamba layers = **736 MiB / token**.
At 404 GB/s DRAM, that's **1.83 ms / token** spent on state DMA
alone — non-trivial. Compare to projected 23 Mamba layers ×
~0.1 ms/layer kernel-compute = 2.3 ms / token kernel. So DRAM
state is **~45% of the per-token bandwidth budget**.

**Future opportunity**: **the single biggest perf lever in the
kernel**. Per-slot L1-shard the state if/when we lock the slot → core
mapping. Saves the 1.83 ms/token state DMA — potential 2-3× speedup
on the Mamba layers. Requires a CB scheduler design change (slot
must persist on the same core for its lifetime, no slot rebalancing).

**Revisit trigger**: at v3 perf pass, when we have the v0..v2 baseline.

------------------------------------------------------------------------

## D3 — `B`/`C` strategy: per-head DRAM replicate (NOT NoC multicast)

**Decision**: Each Tensix reads its head's `B[g, :]` and `C[g, :]`
slice from DRAM. The host pre-computes per-head DRAM addresses via
`group_idx = h // heads_per_group`. No inter-core multicast.

**Alternatives considered**:
- **NoC multicast**: one core in each group of 8 reads B/C from
  DRAM, multicasts to the other 7 via NoC. Saves 7/8 of the B/C
  DRAM bandwidth.
- **L1-replicate**: B/C are tiny (4 tiles × 4 KiB = 16 KiB each per
  group). Could fit all 8 groups' B/C in L1 once per token and
  reference by index. Requires sharded DRAM-to-L1 broadcast at load.

**Rationale**: B/C are tiny (4 tiles each). At B=1, 64 heads × 4
tiles × 2 KiB = **512 KiB / token / layer** for B-replicate. At
404 GB/s, that's 1.27 µs / layer = 0.029 ms / token across 23 Mamba
layers. **Negligible.** No need for multicast complexity.

**Future opportunity**: at high batch (B=32), 32 × 64 × 4 tiles ×
2 KiB = **16 MiB / token / layer** for B-replicate × 23 layers =
**368 MiB / token**. That's ~0.9 ms / token on DRAM. Multicast would
collapse this to ~0.11 ms / token. Worth ~0.8 ms savings if we ever
fight for it.

**Revisit trigger**: at v3 perf pass at B≥16 OR when DRAM-side
profiling shows B/C traffic dominates.

------------------------------------------------------------------------

## D4 — fp32 accumulator for the SSD recursion via `fp32_dest_acc_en`

**Decision**: Compute kernel runs with `fp32_dest_acc_en=true` per
the GDN template. State CBs are typed as bf16 on the DRAM side and
sized for fp32 storage in the dest register file.

**Alternatives considered**:
- **All-bf16**: smaller per-tile L1 footprint (2 KiB vs 4 KiB), faster
  packing. Risk: bf16 recurrent state drifts (verified on 35B's H_t —
  `feedback_35b_dn_h_state_drift_lever`).
- **Mixed**: bf16 state CB, fp32 dest accumulator only for the
  decay-multiply phase. Loses fp32 precision on the input contribution
  outer product.

**Rationale**: the **fp32 SSM state is a hard config requirement**
(`mamba_ssm_cache_dtype="float32"` in Nemotron-3 config). bf16 state
drifts. GDN proved fp32-via-dest-accumulator works on Blackhole at
PCC ≥ 0.99999 (35B owned-GDN bake validated at PCC 0.9999992).

**Future opportunity**: **none on correctness**. There is a 
[[35b-dn-h-state-drift-lever]] precedent suggesting fp32 *inside trace*
caused a 30+ min Blackhole hang on 35B. If reproduced for Mamba2 in
v0.4 trace capture, fall back to bf16 dest + measure drift; if drift
is tolerable accept it as a v0 compromise.

**Revisit trigger**: at v0.4 (trace capture phase).

------------------------------------------------------------------------

## D5 — `decay` (scalar per head) computed once per step, reused over `head_dim` tiles

**Decision**: At the top of each (batch, head) block, compute
`decay = exp(dt_eff * A_per_head)` ONCE, store in a 1-tile CB
(`CB_DECAY`), then reference for each of the 2 `head_dim` tile loops.

**Alternatives considered**:
- **Recompute per head_dim tile**: simpler control flow; wastes ~5 FMAs
  per head_dim tile.
- **Precompute on host**: dt and dt_bias are token-dependent so this
  isn't an option for the kernel.

**Rationale**: head_dim = 64 = 2 tiles. We do the SSD math twice per
(batch, head). Computing decay once saves 1 exp + 1 mul = ~2 cycles
× 2 reuses = trivial. But it also keeps the discretization stage
"one-shot scalar work" up-front, separating it from the inner loop —
cleaner pipeline.

**Future opportunity**: at G3 batching B>1, we could compute decay
for ALL batches' (head=h) at once if they share the same A_per_head[h].
Saves B × 1 cycle per (batch, head) — micro-optimisation. Probably
not worth complicating the kernel.

**Revisit trigger**: if Tracy shows the discretization stage as
non-trivial in the per-step breakdown.

------------------------------------------------------------------------

## D6 — Output reduce `y = C·state + D·x` via `matmul_tile`

**Decision**: The reduce-over-ssm_state for `y[d] = sum_s(C[s] *
state[d, s])` is implemented as a single `matmul_tile` per head_dim
tile (state is `[head_dim, ssm_state]` = 2×4 tiles; C is
`[ssm_state]` = 4 tiles). Two matmul calls per head.

**Alternatives considered**:
- **Manual mul + sum**: 4 tiles of `mul_tiles` + a reduce-tile-along-
  width. Same compute, more LLK boilerplate, less fp32-friendly.
- **`matmul_reduce` helper from GDN** (line 215 of GDN kernel) — same
  pattern; literally fork it.

**Rationale**: forking the GDN `matmul_reduce` is the path of least
resistance. The reduce uses the matrix unit's fp32 accumulator
natively, giving us fp32 numerical stability for the C·state dot
product "for free."

**Future opportunity**: at v3 perf, evaluate if the manual mul +
horizontal reduce is faster on Blackhole's specific matrix unit
geometry. Probably not.

**Revisit trigger**: only if Tracy shows the output reduce stage
dominating step time.

------------------------------------------------------------------------

## D7 — `debug_mode` runtime arg for incremental kernel build-up

**Decision**: Fork GDN's `debug_mode` mechanism. Each mode enables a
subset of the math:
- `0` = full SSD recursion (production)
- `1` = fill_one smoke (skips compute, fills output with 1.0)
- `2` = decay * state only (no input contribution, no output reduce)
- `3` = decay * state + input contribution = `state'` (state correct,
  output garbage)
- `4` = state correct + output = `y = D * x` (state correct, output
  ignores C)
- `5` = state correct + output = `D * x + C @ state` (production
  equivalent of mode 0)

**Rationale**: GDN built up the kernel via this pattern (modes 1-11).
Each mode is independently testable against the numpy oracle — we
can validate decay math (mode 2), state update (mode 3), output
math (mode 4) sequentially. **Without this we'd have to write all
500 LOC of compute before getting our first correctness signal.**

**Future opportunity**: keep `debug_mode` in production. ~2 register
copy + 1 branch per step = nanoseconds. The debugging value at G2/G3/v3
when we fork the kernel for batching is huge.

**Revisit trigger**: never remove.

------------------------------------------------------------------------

## D8 — `softplus_tile` direct call (RESOLVED 2026-06-04 G1 day-3)

**Decision**: use `softplus_tile`, `clamp_tile`, `exp_tile`,
`negative_tile` directly as first-class LLK SFPU primitives. NO
decomposition.

**LLK API survey result** (G1 day-3): all four primitives ship in
`/home/aditya/tenstorrent/tt-metal/tt_metal/hw/inc/api/compute/
eltwise_unary/{softplus,clamp,exp,negative}.h`. Each is a one-line
SFPU dispatch (`MATH(SFPU_UNARY_…)` macros). Per the user's
"don't reinvent the wheel for nonlinearity" directive, we adopt
them as-is.

Specific signatures (verbatim from headers):
- `softplus_tile(idst, beta, beta_recip, threshold)` —
  computes `1/beta * log(1 + exp(beta * x))` (clamps above threshold).
  Standard softplus is beta=1.0, threshold=20.0.
- `clamp_tile(idst, min_bits, max_bits)` — direct fit for our
  `clamp(dt_eff, time_step_floor, time_step_max)`. All bit-cast
  uint32.
- `exp_tile<approx, scale, InputClamping>()` — template-parameterised.
  Default `ClampToNegative` handles inputs < ~-88.5 safely (relevant
  since A = -exp(A_log) can be very negative for large A_log).
- `negative_tile(idst)` — in-place negate. Used for `A = -exp(A_log)`.

**Future opportunity (deferred)**: a fused `softplus_clamp_tile` would
save 2-3 cycles per (batch, head). Upstream-able if Tracy shows the
discretization stage non-trivial at v3.

**Revisit trigger**: at v3 perf pass if discretization shows up in
Tracy as >5% of per-block time.

------------------------------------------------------------------------

## D11 — Split `compute_decay` into two helpers due to SFPU dest-reg lifetime

**Decision (G1 day-3)**: implement decay computation as TWO helpers:
1. `compute_decay(...)` — produces `A = -exp(A_log)` in `cb_decay`.
2. `finalize_decay_with_dt_eff(...)` — multiplies `cb_decay` by
   recomputed `dt_eff`, then `exp`s the result. Stubbed at day-3;
   wired at day-3.5.

**Alternatives considered**:
- **One monolithic helper**: keep `dt_eff` alive in dest reg across
  multiple SFPU stages. Blackhole's dest reg is 8 tiles deep; cross-
  cycle persistence requires careful `tile_regs_*` sequencing.
- **Recompute `dt_eff` inline twice**: one recompute is cheap (~5
  cycles) vs adding a CB store/load round-trip (~30 cycles).

**Rationale**: SFPU primitives consume a single dest tile and produce
into the same slot. To compute `decay = exp(dt_eff * A)` we need both
`dt_eff` and `A` in two dest slots simultaneously. The first pass
fills cb_decay with `A`; the second pass recomputes `dt_eff` (cheap
since dt + dt_bias scalars are still in their CBs) and multiplies by
A in-dest, then exps. The recompute is decision-D5-style (one extra
add + softplus + clamp per block) but keeps each helper as one
`tile_regs_acquire`/`release` cycle — much easier to debug.

**Future opportunity**: when LLK exposes dst-to-dst SFPU mul (or we
write a custom `compute_decay_fused` that stashes `dt_eff` in a
secondary dest slot via `copy_tile_init` patterns), we collapse both
helpers into one. Saves ~30 cycles per (batch, head). Minor at v3.

**Revisit trigger**: if Tracy shows >5% time in discretization stage
at v3 perf pass.

------------------------------------------------------------------------

## D9 — Layout: all per-step inputs interleaved DRAM, no sharding

**Decision**: `x`, `z`, `dt`, `B`, `C` all stored as interleaved-DRAM
tensors. Reader uses standard `interleaved_addr_gen` to fetch tiles.

**Alternatives considered**:
- **Sharded DRAM** per (batch, head) — would let each Tensix's reader
  hit a fixed DRAM bank, reducing addr-gen overhead.
- **Sharded L1** — pre-load all inputs to L1 at the start of the
  layer's forward. Only viable if we have residency per-layer (we
  don't; the forward is one big op).

**Rationale**: per-step input tensors are tiny (single tile each
for dt, multi-tile for x/B/C). Address generation overhead is
negligible vs the compute. The DRAM layout decision lives upstream
in the host-side `Tensor.to_device(...)` call — we just consume
the resulting layout.

**Future opportunity**: at v3 perf, profile whether sharded DRAM
helps on the few-tile inputs. Likely not (they're already in
L1 cache by the time the second use lands).

**Revisit trigger**: only if Tracy shows DRAM contention on B/C.

------------------------------------------------------------------------

## D10 — No `z` consumption inside the kernel

**Decision**: `z` (the gate) is in the kernel signature but the
compute kernel doesn't read it. It's passed through to the caller
who applies `MambaRMSNormGated(y, z)` outside.

**Alternatives considered**:
- **Fuse the norm-gated step into our kernel**: read `z`, compute
  `y_gated = norm(y) * z` inline. Saves ~3 ttnn ops per Mamba layer
  (a separate norm + multiply).

**Rationale**: G1 ships the SSD recursion only — keeping the kernel
focused on novel math. The norm-gated step is a standard `ttnn.rms_norm`
+ `ttnn.multiply` (or potentially a fused `rms_norm_gated` if tt-metal
has one). Fusing into our kernel multiplies the LOC by ~30% and
extends G1 to ~6 days. Not worth it for v0.

**Future opportunity**: a fused `mamba2_decode_with_norm_gated_owned`
at v3 perf. Saves ~0.2 ms/step/Mamba-layer × 23 layers ≈ 4.6 ms/token.
Possibly the second-biggest perf lever after D2.

**Revisit trigger**: at v3 perf pass; absolutely a "should we fuse"
question to answer with Tracy data.

------------------------------------------------------------------------

## Summary — perf-lever ranking (estimated)

| # | Decision | Estimated v3 win |
|---|---|---|
| D2 | L1-resident `ssm_state` | **~1.83 ms/token** (the biggest lever) |
| D10 | Fuse norm-gated + out_proj into kernel | ~4.6 ms/token (across 23 layers) |
| D3 | B/C multicast at B≥16 | ~0.8 ms/token at B=32 |
| D8 | Fused `softplus_clamp_tile` | ~negligible per call, but × 23 layers |
| D5, D6, D9 | Various micro-optimisations | each in the ~0.05 ms/token range |

These are the v3 menu. Phase 1 (v0..v2) ships with the choices
above as-is; we revisit when we have Tracy data to ground the math
in measurement, not estimate.

------------------------------------------------------------------------

## Related

- High-level G1 design: `research/mm7_g1_mamba2_kernel_design.md`
- Pedagogy: `wiki/66_blackhole_kernel_dataflow_anatomy.md`
- Numpy oracle: `experiments/utils/mamba2_numpy_oracle.py`
- Harness: `experiments/utils/test_mamba2_decode_isolated.py`
- Fork base (GDN compute kernel):
  `experiments/owned_ops/qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp`
