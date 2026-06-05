# Pattern A MoE refactor — plan (2026-05-24)

Goal: replace the host-readback expert-dispatch loop in `moe_forward_ttnn`
with Mixtral-style "run all local experts, mask by top-k" so the MoE
becomes trace-clean. Projected end-to-end: **~51 ms/tok = ~19 tok/s** once
all of [Pattern A MoE, DN state-in-place, on-device argmax return,
full-step trace] are landed. Pattern A is the first and biggest piece.

Reference implementation we're copying:
  `experiments/.refs/tt-metal/models/tt_transformers/tt/mixtral_moe.py:94-138`

## Architecture change

### Current sharding (in tree today)

Each chip holds **all 256 experts**, sharded along the intermediate dim:
- `experts_gate_up`: `[256_E, HIDDEN=2048, 2*MOE_INTER_CHIP=256]` per chip
- `experts_down`:    `[256_E, MOE_INTER_CHIP=128, HIDDEN=2048]` per chip
- Top-8 dispatch: Python loop, expert weights sliced by host int, partial matmul on the per-chip slice of intermediate dim, summed across chips via `all_reduce`.

### Pattern A sharding (target)

Each chip holds **64 of 256 experts** (`E_LOCAL = NUM_EXPERTS / NCHIPS`),
each expert with its **full intermediate dim**:
- `experts_gate_up_local`: `[64_E_LOCAL, HIDDEN=2048, 2*MOE_INTER=1024]` per chip
- `experts_down_local`:    `[64_E_LOCAL, MOE_INTER=512, HIDDEN=2048]` per chip
- `local_expert_ids`:      `[64]` per chip — different per chip (chip c owns experts `[c*64, (c+1)*64)`)
- Dispatch: on-device `ttnn.eq(local_expert_ids, top_idxs)` → mask, multiply by top_vals, weighted sum over all 64 local outputs, single `all_reduce` to fuse cross-chip contributions.

Memory check: total per-chip weight count is unchanged (same total params, just resharded). Per-chip footprint stays at ~192 MB MoE weights per layer × 40 layers ≈ 7.7 GB per chip (fits comfortably in 31.8 GB DRAM).

## Compute trade

Pattern A increases per-chip compute by **8×** (each chip runs 64 experts vs the equivalent of ~2 today, 1/4 of 8 active). But:
- Eager today: MoE is 339 ms/tok, ~565× over the bf8 BW floor (0.6 ms). Dispatch overhead dominates by 99.8%.
- Pattern A eager: 8× more matmuls → projected ~2700 ms/tok MoE without trace. **Unusable for production but tolerable for validation runs.**
- Pattern A traced: dispatch overhead amortizes. BW floor with 8× more compute = ~19 ms total MoE (192 MB/chip/layer × 40 layers / 404 GB/s).

The "wasted compute" is real, but the Tensix cores were sitting idle on dispatch overhead today. Mixtral and Grok ship with this trade.

## Phased execution

### Phase 1 — Numpy reference (correctness gate)
- Add a `moe_forward_pattern_a_np` to `experiments/serve/server_35b.py` (or a new module): same input/output as the existing numpy MoE but algorithmically runs all 256 experts and masks by top-8 indices.
- Validate: per-token output of Pattern A numpy == top-8 numpy. Cos > 0.99999. Magnitudes match to fp32 precision. **Must pass before any TT work.**

### Phase 2 — TT re-shard upload
- Update `upload_moe_layer` to produce `experts_gate_up_local`, `experts_down_local`, `local_expert_ids` instead of `experts_gate_up` + `experts_down`.
- Keep shared-expert weights unchanged (already replicated).
- Bootstrap will break until Phase 3 lands — change is atomic with the moe_forward swap.

### Phase 3 — TT `moe_forward_ttnn` rewrite
- New signature: same (`h_tt`, `w`, `mesh`) → same return shape.
- Body:
  1. Router (unchanged): `logits = matmul(h, w_router)`; `probs = softmax(logits)`; `(top_vals, top_idxs) = topk(probs, K=8)`; renormalize.
  2. **Mask**: `mask = ttnn.eq(local_expert_ids[64], top_idxs[8])` → shape `[64, 8]` per chip after reshapes/broadcasts.
  3. **Per-local-expert weight**: `routing_weight = sum(mask * top_vals, dim=-1)` → `[64]` per chip. Zero if expert not in top-8.
  4. **Run all 64 local experts** — start with simple loop over k in 0..63 (compile-time static, trace-friendly):
     - `expert_out_k = down(SiLU(gate(h_tt)) * up(h_tt))` using `gate_up_local[k]`, `down_local[k]`
     - `weighted_k = expert_out_k * routing_weight[k]`
     - Accumulate
  5. `local_sum = sum(weighted_k for k in 0..63)`
  6. `all_reduce(local_sum)` → fused 256-expert output replicated across chips.
  7. Add shared expert (unchanged path).
- **Optimization for later**: replace the 64-iteration loop with a single batched `ttnn.linear` over the stacked expert dim if ttnn supports it. Defer until correctness is in.

### Phase 4 — Correctness validation
- Bootstrap with new weights. Run 5-token "Paris" prompt. Compare top-1 next-token id at every position against HF — must match.
- Run the cosine ladder over the 97-token needle prompt. Median cos_final_norm must stay ≥ 0.997 (current post-fix baseline). Worst-position cos must stay ≥ 0.97.
- Run needle haystack L=100. Must still retrieve "N4Y2BWLS" verbatim.

### Phase 5 — Eager perf measurement
- `profile_35b_ttnn.py` baseline. Expect ~2700 ms/tok decode (8× slowdown vs 480 ms current). **This is acceptable** — eager perf is not the goal; correctness + trace-compatibility is.

### Phase 6 — Trace integration (separate task, P2/P3 in main plan)
- Add on-device argmax return (B17-D, trivial).
- Tackle DN state in-place updates (B17-B-DN, real refactor).
- Capture full-step trace.
- Measure end-to-end traced tok/s. **This is where the projected 51 ms/tok shows up — or fails to.**

## Decision points / risk register

- **Mask shape gymnastics**: `top_idxs` is `[1, 8]` replicated; `local_expert_ids` is `[64]` differing per chip. `ttnn.eq` broadcast rules need verification. If broadcast doesn't work as expected, fall back to explicit reshape + comparison + reduce. Risk: low (well-supported op).
- **Sharding rebalance**: re-uploading all 40 layers of MoE weights in the new format takes ~107s (same as today, just a different reshape on host). Risk: low.
- **Expert top-k could span multiple chips**: today, the router picks 8 of 256 globally. Under new sharding, some of those 8 may live on different chips. Pattern A handles this naturally — every chip computes its local 64 and the all_reduce at the end fuses across chips. No special handling needed. ✓
- **Numerical equivalence with current**: theoretically identical (top-k is the same; gather replaced by mask×output but mathematically equivalent). Should match to bf16 precision. If cos drops, that's a bug, not an algorithm change.
- **64-iter Python loop in trace**: trace can handle long static loops (DN is ~30 ops, demos confirm). 64 expert iterations × ~12 ops each = ~768 ops per MoE layer × 40 layers = ~30k captured ops total. Should fit trace memory; Mixtral has similar order of magnitude.
- **Shared expert path**: stays unchanged. Already trace-clean.

## Files to touch

- `experiments/serve/server_35b_ttnn.py`:
  - `upload_moe_layer` (lines 240-282): replace expert sharding
  - `moe_forward_ttnn` (lines 951-1110): rewrite dispatch
  - Add `local_expert_ids` to State / per-layer-weights dict
- `experiments/serve/server_35b.py` (numpy reference):
  - Add `moe_forward_pattern_a_np` for the Phase 1 cross-check
- New test:
  - `experiments/utils/test_pattern_a_moe_np_vs_topk.py` — Phase 1 numpy correctness gate
- Optional: `experiments/utils/test_pattern_a_moe_tt_vs_hf.py` — Phase 4 TT vs HF validation
