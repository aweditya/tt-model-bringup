# Pattern A batched expert matmul — status (parked 2026-05-25)

Looped Pattern A is the working production path:
  **traced: 308 ms/tok = 3.24 tok/s** (1.56× over 480 ms eager baseline)

Batched matmul attempt is parked. Three ttnn constraints hit:

## What was tried

### Attempt 1: `[1, 1, H] @ [E_LOCAL, H, 2*I]`
Result: `RuntimeError: bmm expects input tensors of the same rank, got a_shape rank: 3 vs b_shape rank: 4`.
The sharded weights tensor reports its **full logical shape** (with leading chip-sharded dim 0 = NCHIPS = 4), so the per-chip view is rank 4 not rank 3.

### Attempt 2: `[1, 1, 1, H] @ [NCHIPS, E_LOCAL, H, 2*I]`
Result: `bmm batch dimension 1 mismatch: a=1 vs b=64 (dimension mismatch only allowed on dim 1 for rank-4 tensors when a[1]=1 and using MatmulMultiCoreReuseMultiCast1DProgramConfig)`.
Decode: dim-0 broadcast is rejected. Only dim-1 broadcast supported.

### Attempt 3: `ttnn.repeat(h_4d, [NCHIPS, 1, 1, 1]) → [NCHIPS, 1, 1, H] @ weights`
Result: `bmm batch dimension 0 mismatch: a=4 vs b=1`.
The weights' per-chip view IS reporting rank 4 with leading chip-shard dim collapsed to 1 — so to match, h needs dim 0 = 1 too. But then dim 1 broadcast (1 vs E_LOCAL=64) is allowed.

### Attempt 4: `ttnn.repeat(h_4d, [1, E_LOCAL, 1, 1]) → [1, E_LOCAL, 1, H] @ weights`
Result: `RuntimeError: Tensor is not allocated` during `ttnn.repeat`.
ttnn.repeat on a bf16 TILE replicated tensor with a target shape that pre-broadcasts on a non-chip dim isn't returning a fully-materialized tensor in this build.

## What to try next

**Refactor at upload time, not forward time.** Instead of letting the weights' logical shape include the chip-sharded leading dim, RESHAPE the uploaded weights to drop that dim:
- Currently: `experts_gate_up_local` shape `[NCHIPS=4, E_LOCAL=64, HIDDEN, 2*MOE_INTER]` sharded on dim 0
- Proposed: upload-time reshape to `[E_LOCAL, HIDDEN, 2*MOE_INTER]` per-chip rank-3, with the shard dim absorbed into the layout metadata rather than the logical shape

Then in the forward:
- h_tt: `[1, HIDDEN]` → reshape `[1, 1, HIDDEN]` (rank 3)
- weights: `[E_LOCAL, HIDDEN, 2*MOE_INTER]` (rank 3 per-chip)
- matmul rank-3: dim 0 broadcast 1→E_LOCAL — might work in rank-3 bmm where the dim-0 broadcast restriction may not apply

If even rank-3 bmm rejects the dim-0 broadcast, fall back to `ttnn.repeat(h_3d, [E_LOCAL, 1, 1])` and matmul without broadcast — same data tiling but at rank 3 which may sidestep the `Tensor is not allocated` issue we hit at rank 4.

## Alternative paths (if upload-time reshape also fails)

- `ttnn.experimental.*` for a dedicated MoE op
- DeepSeek-V3's `all_to_all_dispatch` path (Pattern B) — more refactoring but works in production
- Accept the looped variant; chase other optimizations (vocab-sharded LM head, num_links=2 all_reduce, etc.) — diminishing returns since MoE is the elephant

## Headline

Pattern A batched would unlock another ~5× tok/s if we can get the rank/broadcast plumbing right. It's not blocked on missing ttnn capabilities — it's blocked on the specific tensor-shape contract these matmul kernels require. Worth a focused session, not a continued one-off iteration.
