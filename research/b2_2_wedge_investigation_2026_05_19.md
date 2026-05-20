# B.2.2 Parallel-Attention Wedge Investigation — 2026-05-19

## Status: STUCK on layer 1 DeltaNet wedge across 3 fix attempts

B.2.1 (per_position_list) ships and works (top1 5/5, cos 0.998 median).
B.2.2 (parallel attention via batched [seq_len, HIDDEN] tensors) has a
hard wedge that I can't get past today. Documenting and pivoting.

## Symptom

In `forward_prefill_tp_inner_v3_parallel_attn`:
- Layer 0 (DeltaNet): all 5 positions process correctly (45ms DN body, 19ms MLP).
- Layer 1 (DeltaNet): slice succeeds, `deltanet_step_tp` wedges (no error,
  no output, server stuck at 99% CPU until SIGTERM).

Layer 1 entry x_seq metadata is **bit-identical** to layer 0 entry:
`shape=[5, 5120], layout=TILE, memory_layout=INTERLEAVED, buffer_type=DRAM,
shard_spec=std::nullopt`.

The **only material difference** between layer 0 input and layer 1 input:
- Layer 0: x_seq came from batched `ttnn.embedding` + reshape (validated
  by B.2.1.5a probe: bit-perfect row slicing).
- Layer 1: x_seq came from `mlp_step_tp` output (a chain of matmuls +
  all_reduce + add).

## Fix attempts

### Attempt 1: `to_memory_config(x_pos_view, DRAM)` to "freshen" the slice

Result: `RuntimeError: Tensor is not allocated` inside `rms_norm`.

Diagnosis: `to_memory_config(view, same_mem_config)` returned the view
unchanged. Then `ttnn.deallocate(x_pos_view)` freed x_seq's underlying
storage. The "fresh" tensor was actually still aliased to x_seq.

This is the **same root-cause pattern as today's earlier decay/gate-reshape
bug** (commit `a70ce65` — never deallocate a view of a long-lived tensor).

### Attempt 2: Remove the `ttnn.deallocate(x_pos)` call entirely

Hypothesis: even without the explicit `to_memory_config`, the original
code was deallocating the slice view (treating it as an owned tensor)
and corrupting x_seq for the next iteration.

Result: **Same wedge**, server stuck at 99% CPU. No error message this time.

So either the deallocate wasn't the cause (the view freeing thing wasn't
happening at layer 0 → layer 1 boundary), or the wedge has a different
root cause that's masked by the previous fix.

### Attempt 3: NOT YET TRIED — force fresh allocation of x_seq after MLP

The remaining hypothesis: the MLP output tensor has some internal state
(lazy compute pending? memory aliasing with one of its inputs? kernel
cache state?) that propagates to its slice and makes `deltanet_step_tp`'s
first op (`ttnn.rms_norm`) wedge.

To test: after MLP returns, force a deep copy via either:
- `ttnn.to_memory_config(x_seq, L1)` then `ttnn.to_memory_config(L1, DRAM)`
  (round-trip forces data movement)
- `ttnn.mul(x_seq, 1.0)` or similar to allocate a fresh tensor

If that unwedges layer 1, the issue is confirmed to be inherited state
from the matmul chain in MLP.

## Why this is hard

Bit-identical metadata between two tensors that behave differently means
the metadata API doesn't expose what differs. We'd need to either:
- Read tt-metal source to understand internal tensor state
- File a tt-metal bug for "slice from matmul output wedges in rms_norm"
- Try blind workarounds until one sticks

Given the bootstrap cost (~17 min per attempt), each fix attempt is
expensive. We've burned ~2 hours on this wedge.

## Pivot options

### Option A — Try fix attempt 3 (force fresh alloc after MLP)

~30 min including bootstrap. If it works, B.2.2 unblocks immediately.
If not, we know it's deeper than tensor freshness.

### Option B — Ship B.2.1 (per_position_list) as Phase B and stop here

per_position_list is validated, top1 5/5, top of bf16 noise floor.
It's the SAME wall time as decode-loop (no real prefill speedup) BUT
it's CORRECT.

The original Phase B goal was "real TTFT savings via batched prefill ops".
Without parallel attention, we don't get that. So shipping B.2.1 doesn't
solve the original user problem (long-context coding workload TTFT).

### Option C — Pivot to a DIFFERENT prefill design

The fundamental issue: `slice` from non-embed-output tensors wedges
downstream ops in unpredictable ways. The whole "build multi-row tensor,
slice per position, process, slice_write reassemble" pattern is fragile
on tt-metal.

A safer pattern would be: process per-position throughout but batch the
ATTENTION layer specifically by maintaining BOTH per-position tensors
AND a batched mirror tensor. Attention reads from the batched mirror;
DeltaNet reads from per-position. After each layer, refresh BOTH from
the layer output.

This is complex. ~3-5 days of careful build.

### Option D — Accept that pure-eager batched prefill on tt-metal is hard

Friend's `qwen36` prefill is sequential decode-loop too — they don't
have batched DeltaNet prefill either. Their attention DOES batch but
they use `scaled_dot_product_attention` which we haven't even tested
yet (couldn't get past the DN wedge to even call it).

Maybe the right move is to commit attention-only batching via friend's
exact recipe (we have their reference) rather than building from scratch.
Violates the build-from-scratch principle but reduces risk.

## What I recommend to the user

Try option A (one more fix attempt with bootstrap, ~30 min) — explicit
force-fresh-allocation of MLP output before the next layer's DN body.
If it works, great. If not, pivot conversation about whether to:
- Continue investigating (B.2.2 design has a fundamental tt-metal issue
  we'd need to file a bug for)
- Ship per_position_list as B.2.1 final and move on to long-context
  decode validation (which is a separate axis from prefill)
- Read friend's prefill code more carefully and adopt their attention
  recipe in B.2.2 (loses build-from-scratch purity but reduces risk)

## Commits in this debug arc

- `3efc15a` — B.2.2: initial parallel attention + slice_write DN
- `0a3f817` — diagnostic prints (per-attn substep)
- `901fffd` — per-layer prints
- `1ee127f` — DN-inner per-position prints
- `8b18b28` — fix attempt 1: to_memory_config (made it WORSE with Tensor not allocated error)
- `feb2034` — fix attempt 2: remove view dealloc (no change, still wedges)

## Time spent

~2 hours including bootstraps. Lots learned about tt-metal tensor
semantics but the actual implementation is blocked.
