# Friend's Qwen3.6-27B Port vs Ours: Comparison Notes

**Source**: `/Users/adityasriram/Labs/stanford/cs440lx/tt-xla/experiments/.refs/tt-qwen-36/` (a fork of `tenstorrent/tt-metal`, with Qwen3.6 added under `models/tt_transformers/`).

**Friend's headline result** (from `models/tt_transformers/PERF.md:50-51` and `QWEN36_README.md:57-60`): Qwen3.6-27B dense at **15.5 tok/s/user on P150x4 (1x4 mesh)**; Qwen3.6-35B-A3B MoE at **14.4 tok/s/user on 2x2 mesh**. There is **no single-chip (1x1) PR-quality config** — they explicitly call out that "Single-device P150 is not a PR-quality real-weight configuration" (`QWEN36_README.md:70-71`). So strictly, "8 / 16 tok/s on multi-chip" maps to 14.4 / 15.5, and we should treat any single-chip comparison as them having punted that case.

## Executive Summary

The friend's port lives inside the upstream `tt_transformers` framework and reuses essentially every existing scaffolding piece (paged KV cache, `Attention` base class, `MLP`, `DistributedNorm`, `TransformerBlock`, `tt_ccl`, `SamplingGenerator`, `begin_trace_capture`/`execute_trace`). Their **novel work is the linear-attention (GDN) block** in `tt/qwen36.py` — a 1700-line custom module that drives a pair of **native C++ ops** (`ttnn.experimental.qwen36_gdn_decode`, `ttnn.experimental.qwen36_gdn_prepare_decode`) for the recurrence and the Q/K/V prep. **Their full-attention layers are subclassed off the stock `Attention`** and just add a Qwen3.6 output-gate matmul + partial RoPE. They run tensor-parallel across 2 or 4 P150s, use paged KV cache with `paged_update_cache` / `paged_scaled_dot_product_attention_decode`, and capture one decode trace via `ttnn.begin_trace_capture`. Compared to us, they (a) have a hand-written C++ GDN kernel we cannot match without writing a Metalium op, (b) lean on multi-chip TP for both memory and bandwidth, (c) use the upstream paged KV path end-to-end, and (d) don't fuse w1+w3 in the MLP. We're doing the same trace-capture story they are, but with a much tighter custom Python kernel on one chip.

## 1. Architecture Differences

### DeltaNet linear-attention vs full-attention split

- **Friend**: Layer types are derived from the HF config and stored as `args.layer_types[layer_num]` (`tt/model_config.py:2796`). Pattern in both 27B and 35B-A3B is `(layer_idx+1) % 4 != 0 → linear_attention`, else `full_attention` (`tt/model_config.py:2955`, `:3020`). `Qwen36TokenMixer` (`tt/qwen36.py:1669-1699`) dispatches per layer to either `Qwen36LinearAttentionDevice` or `Qwen36FullAttentionDevice`. Both are wrapped as `attention_class` and plugged into the stock `TransformerBlock` so RMSNorm/residual/MLP are layer-type-agnostic.
- **Friend's GDN prefill** is *not* using the native prefill op in this branch — `decoder.py:258-321` literally runs prefill as a sequential decode-step loop, calling `ttnn.experimental.slice_write` per token. The README warns "GDN prefill is still a sequential decode-style recurrence loop, so TTFT is not final" (`QWEN36_README.md:78-80`).
- **Us**: Single Python decode kernel that branches DeltaNet vs gated-full-attention per layer. We don't have a native GDN C++ op — we implement the gated delta rule with `ttnn.cumsum` + Neumann series factorization (per Branch C' memory).

### KV cache writes (full-attention layers only)

- **Friend**: `paged_update_cache` or `paged_fused_update_cache` for decode (`tt/attention.py:776-785`); `paged_fill_cache` for prefill (`tt/attention.py:1158-1166`); falls back to `ttnn.fill_cache` if not paged (`tt/attention.py:1169-1174`). All writes go through `update_idxs_tensor=current_pos, page_table=page_table`.
- **Us**: Currently `ttnn.scatter` (validated for bf16 src, refuses fp32+TILE — see `91f_qwen36_27b_full_ondevice.py:349-361`). Swap to `ttnn.kv_cache.update_cache_for_token_` is queued (7.2× faster, per the C'1→C'1.5 note).
- **Assessment**: Friend's `paged_update_cache` with `update_idxs_tensor` is the same family as what we're swapping to; he gets the additional `paged_fused_update_cache` win when fused K+V is available (`tt/attention.py:776`).

### SDPA

- **Friend**: `ttnn.transformer.paged_scaled_dot_product_attention_decode` when a page_table is supplied (`tt/attention.py:794`), `scaled_dot_product_attention_decode` otherwise (`:807`). Prefill uses `chunked_scaled_dot_product_attention` (`tt/attention.py:1190`).
- **Us**: `ttnn.transformer.scaled_dot_product_attention_decode` (non-paged) — and our memory notes the paged variant survives 32k context where the stock cliff fails at MAX_POS=256.
- **Assessment**: We've already validated paged SDPA decode internally; we just haven't switched the production kernel. The friend is on the paged path by default.

### RoPE

- **Friend**: `ttnn.experimental.rotary_embedding` (native), composed with slice+concat for partial RoPE because `partial_rotary_factor=0.25` (`tt/model_config.py:2804-2811`, `tt/qwen36.py:1471-1573` for decode and `:1575-1611` for prefill). The rotary dim is `head_dim * 0.25`; the tail dim is concatenated back via `ttnn.concat(..., dim=3)`. For decode with batch>1 they loop `batch_size_per_device_group` iterations and use sharded reshapes — `tt/qwen36.py:1490-1573`.
- **Us**: Manual rotate-half historically; switched to `ttnn.experimental.rotary_embedding` (2.6× faster per `feedback_native_rope.md`). We tried C'3 native partial-RoPE and abandoned it because `cos_cache.padded_shape` conflicts with TILE_LAYOUT in our setting; we use a Level-1 "identity in passthrough region" trick instead (`feedback_partial_rope_level1_trick.md`).
- **Assessment**: Same op, but the friend pays for slice+concat per call instead of folding the passthrough region into the cos/sin table. Our Level-1 trick (12 → 7 ttnn ops, bit-exact) is genuinely better at op count.

### Trace capture

- **Friend**: One `ttnn.begin_trace_capture` over the full `ttnn_decode_forward` including device-side sampling (`demo/qwen36_decode_smoke.py:208-214`); pattern is identical to ours (warmup → capture → re-execute). The model maintains its own `reset_decode_state()` for GDN conv state and recurrent state (`tt/qwen36.py:591-594` for the linear-attn block, `:1655-1666` for full-attn). They explicitly thread an "in-place" trick for the recurrent state where the native op updates the persistent buffer in place but may return a new Python handle (`tt/qwen36.py:945-960`).
- **Us**: Same single-trace pattern. C'4 v4 lets execute_trace land at 198 ms/tok; in-trace `ttnn.copy(scatter_out, cache_in)` threads autoregressive state (per `feedback_trace_state_threading_works.md`).
- **Assessment**: Architecturally identical. Friend has the advantage of `paged_update_cache` being inherently in-place, so he doesn't need the `ttnn.copy` shim.

## 2. Optimization Choices

### Memory configs

- **Friend** is aggressively sharded: DRAM-sharded weights for all input/output projections (`_load_output_sharded_linear_weight` at `tt/qwen36.py:419-439` uses `create_dram_sharded_mem_config`). Decode activations move L1-sharded ↔ DRAM as needed. He uses `ttnn.create_sharded_memory_config(..., WIDTH, ROW_MAJOR)` for matmul outputs at `tt/qwen36.py:612-620`. Residuals/Norms use `get_residual_mem_config`.
- **Us**: Mix of INTERLEAVED L1 + DRAM. We have not aggressively width-sharded matmul outputs across a fabricated core grid because we're single-chip.

### Datatypes

- **Friend**: Defaults per `PERF.md:11`: "bfp4 MLP and bfp8 attention weights" for most models. For Qwen3.6-27B the table doesn't itemize but the framework drives this via `decoders_optimizations`. CCL dtype is `bfloat16` when `QWEN36_CCL_BF16=1` (default on 1x2 for argmax-tie stability — `tt/model_config.py:1071-1072`). Activation dtype in GDN is hard-coded bfloat16 (`tt/qwen36.py:317`). Recurrent state is **fp32** (`tt/qwen36.py:560-566`) and `neg_A`, `dt_bias` are fp32 — math fidelity matters here.
- **Us**: Full bf8 weights for all 24 layers in our 8B-class testing (per `feedback_bf8_weights.md`); kv_cache bf16. We haven't explored bfp4 for MLP.
- **Assessment**: Friend's `bfp4 MLP + bfp8 attn` mix is a clear precision/speed lever we have not picked up. Our memory notes that for batch=1 we're dispatch-bound, not bandwidth-bound, but Qwen3.6-27B at multi-chip is a different regime.

### Fusions

- **GDN input-projection fusion** (`QWEN36_FUSE_INPUT_PROJECTIONS=1`, default-on, `tt/qwen36.py:327`): merges QKV + Z + B + A into a *single* matmul whose output is sliced into four child tensors via `ttnn.slice + ttnn.clone` (`tt/qwen36.py:668-693`). The pre-pack weight reorder is `reorder_qwen36_input_projection_weight_for_tensor_parallel` (`:154-194`). Substantial win on a hot path.
- **Full-attention QKV fusion**: standard upstream pattern — single `wqkv` matmul (`tt/attention.py:677`). Q-gate is a *separate* matmul (`wq_gate`, `tt/qwen36.py:1297-1305`) because the per-head gated query is Qwen3.6-specific.
- **MLP w1/w3 fusion**: **Not fused**. `tt/mlp.py:145-170` runs two separate `ttnn.linear` calls (w1, w3) then `ttnn.mul` with `input_tensor_a_activations=[self.activation_type]` (SiLU). Each goes through its own all-reduce on TG.
- **Us**: DN-fusion + ATTN-QKV fusion + Level-1 partial RoPE shipped. MLP gate-up explored but rolled back (3% regression per `feedback_fusion_frontier_exhausted.md`).
- **Assessment**: Identical strategy on the GDN block. Both repos don't fuse MLP w1/w3 — confirms our finding that dispatch is amortized post-C'1.

### Per-position / batch-row tricks

- **Friend**: `QWEN36_COMPACT_ACTIVE_RECURRENCE=1` (`tt/qwen36.py:328-334`) keeps the recurrent state sized to the *active* batch rows instead of the tile-padded rows (32). This is a real win at batch=1 since the conv state and recurrent state would otherwise be ×32 too large. Implemented via `_decode_active_rows` (`:640-656`) — slice + clone to L1.
- **Us**: Production batch=1 already, so we don't carry tile-padded recurrence.

## 3. Multi-Chip Approach

- **Pure TP, no SP** (no sequence-parallel). The `Qwen36TensorParallelPlan` (`tt/qwen36.py:30-56`) shards `key_heads` and `value_heads` across `num_devices`. Conv1d, in_proj, out_proj are all split along the head dimension. The recurrent state itself is **replicated** across the mesh (`tt/qwen36.py:558-566`, `ttnn.ReplicateTensorToMesh`) — *the cross-position recurrence stays local-per-head because heads are already sharded across devices*. This is elegant: no cross-device recurrence dependency, just an all-reduce on the post-projection output.
- **CCL pattern**:
  - GDN out-proj: `ttnn.linear` → `tt_all_reduce(..., cluster_axis=0)` (`tt/qwen36.py:1077-1096`). On 2D non-Galaxy meshes (i.e. P150x4 2x2 MoE), it uses `reduce_scatter_minimal_async` + `all_gather` composite instead of the fast TG path (`:1095`, `tt/qwen36.py:1098-1115`).
  - Attention out-proj: `all_gather_async` + `ttnn.linear` for non-Ring, or `all_gather_matmul_async` (fused) for Ring topology (`tt/attention.py:836-892`).
  - MLP: `reduce_scatter` on w1/w3 outputs when dim==8192 or prefill (`tt/mlp.py:181-211`); `tt_all_reduce` otherwise.
- **MoE 35B-A3B on 2x2** uses one mesh axis for TP and the other for MoE expert sharding — that's why MoE runs at 2x2 but dense runs at 1x4 (max-width TP). Confirmed in `QWEN36_HARDWARE_RUN_GUIDE.md:69-71`.
- **Fabric**: They explicitly enable `ttnn.FabricConfig.FABRIC_1D` before opening the mesh and disable on shutdown (`QWEN36_HARDWARE_RUN_GUIDE.md:73-75`) — the same fabric init issue we hit in Phase A6 on qb1.
- **Us**: Currently single-chip on qb2. Our Phase A7 is exactly this — opening the mesh, but collectives still blocked on fabric init. The friend's `tt_ccl.py` is a complete reference for what working fabric+CCL looks like.

## 4. Diff Matrix

| Subsystem | Friend's choice | Our choice | Assessment |
|---|---|---|---|
| DN recurrence | Native C++ op `ttnn.experimental.qwen36_gdn_decode` (`tt/qwen36.py:945`) | Python with `ttnn.cumsum` + Neumann (I-L)^-1 | Friend has hand-written kernel; ours is pure ttnn-composable. Friend is faster on hot path but we don't need a build pipeline. |
| DN prep Q/K/V | Native `qwen36_gdn_prepare_decode` (`tt/qwen36.py:884`) | Python slice/reshape | Same — friend pays compile complexity. |
| DN prefill | Sequential decode-step loop (`decoder.py:258`) | Same in our work (we're decode-first) | Tie; both deferred. |
| Full-attn KV write | `paged_update_cache` / `paged_fused_update_cache` w/ `update_idxs_tensor` (`tt/attention.py:776-785`) | `ttnn.scatter` → swapping to `update_cache_for_token_` | Friend already at the destination; ours is mid-migration. |
| Full-attn SDPA | `paged_scaled_dot_product_attention_decode` (`tt/attention.py:794`) | Non-paged `scaled_dot_product_attention_decode` | Friend handles long context cleanly; we hit the MAX_POS=256 cliff on stock SDPA per `feedback_sdpa_decode_max_pos_256_cliff.md`. |
| RoPE | `ttnn.experimental.rotary_embedding` + slice+concat for partial (`tt/qwen36.py:1504-1531`) | Same op + "Level-1 identity passthrough" trick (`feedback_partial_rope_level1_trick.md`) | We have fewer ops on the hot path. |
| MLP gate/up | Separate `ttnn.linear` calls + `ttnn.mul(SiLU)` (`tt/mlp.py:145-242`) | Same (explored fusion, rolled back) | Tie; both validate fusion isn't a clear win. |
| Trace capture | `ttnn.begin_trace_capture` over full decode incl. sampling (`demo/qwen36_decode_smoke.py:208`) | Same pattern, C'4 v4 | Architecturally identical. |
| Recurrent state | Replicated across mesh, fp32, `compact_active_recurrence=1` (`tt/qwen36.py:558-566, 328`) | Single-chip, fp32 | Friend's compaction is meaningful at batch>1; we're batch=1. |
| Multi-chip | Pure TP across heads, fabric_1D, custom `tt_all_reduce`+`reduce_scatter_minimal_async` (`tt/ccl.py:111`) | Single-chip; Phase A7 mesh open, fabric blocked | This is our C'7 target; `tt/ccl.py` is the reference. |
| Weight dtypes | `bfp4` MLP + `bfp8` attn defaults (`PERF.md:11`); fp32 for `A_log`, `dt_bias` (`tt/qwen36.py:392-405`) | bf8 all weights | We haven't tried bfp4 MLP. |

## 5. Things to Adopt

1. **`paged_scaled_dot_product_attention_decode` for full-attention layers.** We've already validated it survives 32k context (`feedback_paged_sdpa_decode_works_at_32k.md`). Friend uses it by default. This is the unambiguous win — long context is our daily-driver gate.
2. **`paged_fused_update_cache`** when we have K and V together (`tt/attention.py:776`) — single op for both writes. We don't get this from our individual `scatter`/`update_cache_for_token_` calls.
3. **bfp4 MLP weights** for the dense decoder layers (top-1/top-5 don't suffer much in their PERF tables for 70B-class). Worth A/B on Qwen3.6-27B.

## 6. Things We Did Differently That Look Better

1. **Level-1 partial-RoPE trick** — extending cos/sin with identity in the passthrough region (per memory) is bit-exact and drops 12→7 ttnn ops. Friend pays an explicit `slice + concat(dim=3)` per call (`tt/qwen36.py:1504-1531`).
2. **Pure-Python GDN recurrence** (cumsum + Neumann) — we don't need a C++ build pipeline. The friend depends on `_ttnncpp.so` being built from a sibling tree (`QWEN36_HARDWARE_RUN_GUIDE.md:7-9`) which is fragile (the troubleshooting section confirms — `:354-378`).
3. **Single-chip story exists at all.** Friend has explicitly punted single-chip — "Single-device P150 is not a PR-quality real-weight configuration" (`QWEN36_README.md:70-71`). Our 210 ms/tok kernel on one P150 is the only single-chip baseline for this model class.
4. **No prefetcher dependency.** Their full-attention code paths have `prefetcher is not None` branches everywhere (`tt/attention.py:680-686, 836-892`). Ours is simpler to reason about.

## 7. Open Questions Worth Asking

1. **`qwen36_gdn_decode` op contract.** The native op signature in `tt/qwen36.py:945-957` takes `recurrent_state, q, k, value, alpha, beta, active_mask, k_row_mask, normalize_qk_l2`. What's the math fidelity setting inside? Does it use HIFI4 internally regardless of the Python `compute_kernel_config`?
2. **Why is GDN prefill still sequential?** README says "the native GDN prefill op and `QWEN36_GDN_NATIVE_PREFILL` flag are not wired in this phase1 tree" (`QWEN36_README.md:79-81`). Is the native prefill op done in another branch? What does TTFT look like with it?
3. **Recurrent state replication.** State is `ReplicateTensorToMesh` (`tt/qwen36.py:565`). Since heads are sharded, each device's recurrent state is for *its* heads, so replication seems wasteful — is this just for the persistent buffer layout, and the *content* is implicitly per-device after the first step? Worth confirming.
4. **`paged_fused_update_cache` vs two `paged_update_cache` calls** — what's the measured speedup? `tt/attention.py:775-785` switches on `self.use_qk_fused` — what controls that flag?
5. **MoE all-reduce path on 2D non-Galaxy.** Comment at `tt/qwen36.py:1092-1094` says "TG all_gather + fast_reduce path collapses the hidden dimension here; use reduce_scatter + all_gather instead." This is a known bug or a structural limitation?
6. **Why no MLP w1+w3 fusion?** They run two separate linears + `mul`. Was this measured and found neutral, or just inherited from upstream?
7. **`QWEN36_CCL_BF16` default on 1x2 only** — they say it's "for argmax-tie stability" (`tt/model_config.py:1071-1072`). Did they see specific tokens flipping on 1x4 without it, and what made 1x2 special?
8. **Single-chip status.** What's the blocker for a PR-quality 1x1 P150 config? Memory? Or just no one prioritized it?

## Notes On Method

This comparison covered: `models/tt_transformers/QWEN36_README.md`, `QWEN36_HARDWARE_RUN_GUIDE.md`, `PERF.md`, `tt/qwen36.py` (1700 lines), `tt/qwen36_moe.py` (skimmed), `tt/attention.py` (partial), `tt/decoder.py`, `tt/ccl.py`, `tt/mlp.py`, `tt/model.py` (decode path), `tt/model_config.py` (Qwen3.6 layer-type + dtype config), `demo/qwen36_decode_smoke.py` (trace capture path). Bottlenecks not yet explored: the C++ source for `qwen36_gdn_decode`/`qwen36_gdn_prepare_decode` (lives in a sibling `tt-metal-latest` tree per the run guide — not in this checkout).
