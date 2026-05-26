# B6 Trace-Safety Question — Resolved by Existing Production Evidence (2026-05-21)

The block plan (`qwen36_35b_a3b_incremental_block_plan_2026_05_21.md`) lists
B6 as a "trace-safety probe" with this central question:

> Can `ttnn.softmax + ttnn.topk + ttnn.sum + ttnn.div` be wrapped in
> `begin_trace_capture/end_trace_capture` on (1,4) mesh with replicated
> weights? Dynamic top-K indices must flow as tensor data, not as graph
> arguments.

**This question is already answered by existing production code** in
`experiments/serve/server_tp.py`, which runs three dynamic-index ops
INSIDE the captured decode trace and uses their tensor outputs to feed
subsequent trace ops:

| Line | Op | Why it's the same pattern as MoE routing |
|---|---|---|
| 1772 | `ttnn.embedding(tok_buf, embed_tt)` | tok_buf is a dynamic [1,1] uint32 index tensor; embedding gathers an embed-vector row from a replicated table. Direct analogue to MoE's "gather expert weight slabs by index." |
| 1780 / 1785 | `ttnn.embedding(rot_idxs_buf, cos/sin_table_tt)` | RoPE cos/sin row-lookup driven by a dynamic index. Same gather-by-index-tensor pattern. |
| 1851 | `ttnn.argmax(rm_logits_tt, dim=-1, keepdim=True, use_multicore=True)` | Produces a tensor of indices INSIDE the trace; the next iteration consumes it as input to embed lookup. |

**Conclusion:** ttnn's trace machinery already supports data-dependent
tensor ops with dynamic indices flowing from one captured op to the next.
The MoE flow `softmax → topk → gather_by_index` is structurally identical
to the production `argmax → embedding` cycle that ships every decode token
in qb2 production today.

## Implications

- **Skip the explicit B6 probe.** The decision gate it was meant to resolve
  (owned router kernel early vs stock ttnn) is decided: **stock ttnn ops
  are sufficient for the router**. Owned `owned_moe_expert_decode` is still
  a real perf optimization (indexed weight read + per-expert SwiGLU fusion),
  but it's a G2.5 perf concern, not a G2 correctness blocker.

- **Stay on the block plan.** Skip to B7 (single-chip ttnn DN port on qb1).

## Remaining trace-safety unknowns (for later)

These ARE genuine open questions, but they're for B11+ when we wire ttnn
expert dispatch into trace, not for B6:

1. Whether `ttnn.topk(k=8)` returns indices in a layout that `ttnn.embedding`
   (or a custom gather) can consume directly, or if we need a reshape /
   typecast in between.
2. Whether `ttnn.softmax(..., dtype=fp32)` (the Qwen router uses fp32
   softmax explicitly) works in a trace alongside bf16 ops. Should — but
   verify when integrating.
3. Whether the per-expert SwiGLU loop K=8 fits in trace memory (each
   iteration captures matmuls referencing different `expert_idx` rows of
   the fused `gate_up_proj[256, 1024, 2048]` tensor — should be cheap, but
   want to measure the trace IR size).

These get answered DURING B11 ttnn implementation, not preemptively in B6.
