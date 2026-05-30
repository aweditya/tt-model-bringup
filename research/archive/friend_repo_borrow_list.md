# Friend (samjett) repo borrow list — Qwen3.6 27B at 15.3 tok/s on 4×P150

Cloned at `experiments/.refs/tt-qwen-36/` (commit `a3d12574`, branch `qwen36-fresh`).

## Summary

| | Friend (samjett) | Ours (server_tp.py) | Ratio |
|---|---|---|---|
| Perf | **15.5 tok/s** (1x4 P150 dense) — `models/tt_transformers/PERF.md:50` and `QWEN36_README.md:59` | 7.02 tok/s | 2.21× gap |
| Mesh | (1, 4), `FABRIC_1D` | (1, 4), `FABRIC_1D` | same |
| MAX_POS | up to 1024 in smoke; 128 default (`QWEN36_HARDWARE_RUN_GUIDE.md:140,254`) | 256 | n/a |
| GDN recurrence | **Custom C++ ttnn op** (`ttnn.experimental.qwen36_gdn_decode`) | Python recurrence body (5 elementwise ops/layer) |
| GDN Q/K/V prep | **Custom C++ ttnn op** (`ttnn.experimental.qwen36_gdn_prepare_decode`) | Python slice+reshape+repeat_interleave |
| conv1d | Python mul+add (NOT the C++ `qwen36_causal_conv_decode`, which exists but is NOT wired) | Python concat+mul+sum+silu+slice |
| RMSNorm | Distributed `rms_norm_pre_all_gather` + `rms_norm_post_all_gather` | Single fused `ttnn.rms_norm` on replicated x (no pre/post split) |
| RoPE | `ttnn.experimental.rotary_embedding` + slice/concat for partial | Manual rotate-only (no rotary_embedding) |
| SDPA decode | `ttnn.transformer.paged_scaled_dot_product_attention_decode` | `ttnn.transformer.paged_scaled_dot_product_attention_decode` (post P18) |
| KV update | `paged_fused_update_cache` (K+V in one op) when `use_qk_fused` | `paged_update_cache` × 2 |
| LM head | **Vocab-sharded, DRAM-sharded multi-split linear, NO final all_reduce on (1,4)** | Replicated lm_head, full all_reduce |
| Embedding | **On-device** `ttnn.embedding(token_ids, embed_weights)` each step | Host `embed_np[token_id]` + `copy_host_to_device_tensor` |
| Position | **On-device** `ttnn.plus_one(current_pos)` | Host `cur_pos += 1` + `copy_host_to_device_tensor` |
| Cos/Sin | Precomputed [max_pos, head_dim] device cache → `ttnn.embedding(rot_idxs, cos_cache)` per step | Host `from_torch(cos_all_np[cur_pos])` + `copy_host_to_device_tensor` per step |
| Argmax | `all_gather(logits)` → `untilize` → `ttnn.argmax(dim=-1)` returning 1 int | Host `ConcatMeshToTensor(logits, dim=0)` then `np.argmax` |

## Architecture comparison: DeltaNet block (linear-attention)

Friend's `Qwen36LinearAttentionDevice._forward_decode_step` (`qwen36.py:1120-1128`) for one block:

1. `_decode_input_projections` — **one** fused `ttnn.linear` of `in_proj_qkv_zba` (`qwen36.py:670-678`, default `QWEN36_FUSE_INPUT_PROJECTIONS=1`), then 4 slice+clone children.
2. `_decode_causal_conv` — Python mul+add accumulation over K=4 kernel taps (`qwen36.py:727-748`), then trace-safe state update via `ttnn.copy(new, conv_states[k])` (`qwen36.py:754-758`). Same shape as our pattern, but with a critical addition: the test `test_qwen36_gdn_decode_native.py:211` asserts `"ttnn.experimental.qwen36_causal_conv_decode" not in model_source` — i.e. the C++ op exists in the tree but is **NOT** wired into production. Their conv1d is Python.
3. `_decode_delta_parameters` — sigmoid(b), softplus(a+dt_bias), neg_A * softplus — `qwen36.py:761-782`. Functionally equivalent to ours.
4. `_decode_gated_delta_rule` — calls **`ttnn.experimental.qwen36_gdn_prepare_decode`** (`qwen36.py:884-890`) to slice+reshape+repeat_interleave Q/K/V in one C++ op, then `_l2_normalize_head_dim` per-head L2 norm (Python, manual mul+sum+rsqrt — `qwen36.py:863-880`), then **`ttnn.experimental.qwen36_gdn_decode`** (`qwen36.py:945-957`) replaces the entire recurrence body + final Q@state matmul.
5. `_decode_gated_rmsnorm` — fused `ttnn.rms_norm` with `ttnn.UnaryOpType.SILU` in the binary op activations kwarg (`qwen36.py:1010-1017`) — single op for `silu(z) * normalized_recurrent_output`.
6. `_decode_output_projection` — DRAM-sharded `ttnn.linear` + `tt_all_reduce` via `reduce_scatter_minimal_async` on the 2D-non-galaxy path (`qwen36.py:1077-1115`).

Our `deltanet_step_tp` (`server_tp.py:400-498`) does the same math but exclusively with Python ops, no fused native ops.

## Architecture comparison: Gated Attention block (full-attention)

Friend's `Qwen36FullAttentionDevice` (`qwen36.py:1205-1652`) is a subclass of the standard `Attention` (`attention.py`):

- Uses `nlp_create_qkv_heads_decode` to split Q/K/V from the fused `xqkv_fused` matmul (`attention.py:743`).
- Uses `paged_fused_update_cache(K, V)` in a single op when `use_qk_fused=True` (`attention.py:776`) vs our two `paged_update_cache` calls.
- Uses `paged_scaled_dot_product_attention_decode` (`attention.py:794-805`) — same as our post-P18 path.
- Uses `nlp_concat_heads_decode` to merge attention output (`attention.py:827`).
- For Qwen3.6 partial RoPE (rotary_dim < head_dim): uses **native `ttnn.experimental.rotary_embedding`** on the rotary slice + `ttnn.concat` with the passthrough slice (`qwen36.py:1504-1531`). Beats our manual rotate-only despite per-batch iteration.
- Adds Qwen3.6 output gate: `wq_gate` linear → `tt_all_reduce` → `sigmoid` → `mul` against attn_output (`qwen36.py:1378-1429, 1613-1628`).

## Borrow list (sorted by ROI for our 7.02 → 15.5 tok/s gap)

The combined headroom from these is bigger than the gap — meaning even partial adoption likely closes it. Sorted by lowest-effort / highest-confidence first.

### 1. On-device embedding lookup + on-device cur_pos increment + cos/sin cache embedding — TOP ROI, LOW EFFORT

**Estimated impact:** ~3-5 ms/tok (most of the 1.9 ms `update_input_buffers` + most of the 1.3 ms cos/sin per-step copies, plus eliminates some host CPU sync overhead).

**What they do:**
- Tokens live in a persistent device buffer; `_transform_decode_inputs_device(x)` calls `self.embd(tokens)` = `ttnn.embedding(x, self.weights, ...)` on device (`model.py:558-561`, `embedding.py:34-36`).
- Sampled token writes back into the **same** `x` buffer (`model.py:716-720` — `self.sampling.sample(tt_logits, tt_out_tok=x)`). No host token readback inside the decode loop.
- Cur_pos lives on device; `_increment_decode_positions_device` calls `ttnn.plus_one(current_pos, skip_negative_entries=True)` inside the trace (`model.py:669-671`).
- Cos/Sin: full `[max_seq_len, head_dim]` cache uploaded once, per-step lookup via `ttnn.embedding(rot_idxs, cos_matrix)` (`rope.py:671-676`).

**What we do:** `update_input_buffers` (`server_tp.py:633-674`) does host `embed_np[token_id]`, host `cur_pos+1`, host `cos_all_np[cur_pos]` slice + 3× `copy_host_to_device_tensor` calls **every decode step**. Per `feedback_tracy_tp_breakdown.md`, this is 1.9 ms/tok directly; plus we incur a logits-readback at 9.4 ms/tok (`server_tp.py:858`) feeding into host argmax which then feeds back into update_input_buffers.

**Gap:** Our host loop is non-traceable; we re-enter the kernel each step paying full HtoD cost. Friend's trace captures cur_pos increment + cos lookup + sampled-token writeback IN the trace, no host sync between steps (only at logits/token readback far less often if `--async-token-read`).

**Effort:** **Medium** — ~150 LOC patch:
- Add `tok_buf` (1-int int32 device tensor), `cur_pos_buf` already exists, plus a `cos_cache_tt` / `sin_cache_tt` of `[MAX_POS, ROTARY_DIM]` on device (already in trace, just unused).
- Replace `state.embed_np[token_id]` with `ttnn.embedding(tok_buf, embed_weights_tt, ...)` at the start of `_traced_forward`.
- Replace `state.cos_all_np[cur_pos]` slice with `ttnn.embedding(cur_pos_buf, cos_cache_tt, ...)`.
- Add `ttnn.plus_one(cur_pos_buf)` inside the trace at the end (Stage E pattern).
- Recapture the trace.

**Custom kernel needed:** No — `ttnn.embedding`, `ttnn.plus_one`, `ttnn.argmax` all exist in our build (`feedback_ttnn_fused_ops_gap_analysis.md` confirmed these).

**Memory note that may be invalidated:** `feedback_tracy_tp_breakdown.md` "logits readback is #2 only 6.6%" — if we do on-device argmax, logits-readback collapses to a 1-int read = ~0.05 ms instead of 9.4 ms.

---

### 2. On-device argmax via all_gather + untilize + ttnn.argmax — UNBLOCKS the 9.4 ms/tok logits readback

**Estimated impact:** ~9 ms/tok saved (eliminates the 152064 fp32 readback bottleneck identified in `feedback_tracy_tp_breakdown.md` row D).

**What they do:** Friend's force-argmax path (`models/common/sampling/tt_sampling.py:423-454`):

```python
if self._force_argmax_sampling:
    if num_devices > 1:
        x = ttnn.experimental.all_gather_async(x, dim=3, cluster_axis=1, ...)  # concat per-chip vocab shards
    x_untilized = ttnn.untilize(x, use_multicore=True)
    tt_out_tok = ttnn.argmax(x_untilized, dim=-1, output_tensor=tt_out_tok, keepdim=False, use_multicore=True)
```

Then writes the 1-int result back to the persistent token buffer (which is also the next step's embedding input — see #1).

**What we do:** `server_tp.py:858` — `ttnn.to_torch(last_logits, mesh_composer=ConcatMeshToTensor(state.mesh, dim=0))` reads the full 152064-fp32 logits to host per step.

**Gap:** This is `feedback_lm_head_argmax_unknown.md` exactly — that note marked `ttnn.argmax(sharded, dim=-1)` "unknown if works per-chip-locally"; friend's evidence: it works ON THE GATHERED tensor (not per-chip-sharded). The fix is `all_gather` first then argmax on the full tensor, untilizing after gather. **THE UNKNOWN IS NOW ANSWERED.**

**Effort:** **Small** — ~30 LOC. Lives behind the on-device sampling path (`tt_sampling.py:404-454`). Bundle with #1.

**Custom kernel needed:** No.

**Memory note that may be invalidated:** `feedback_lm_head_argmax_unknown.md` ("Open question on argmax-on-mesh") — RESOLVED by `tt_sampling.py:423-454`. The recipe is: all_gather then argmax on the full tensor; do NOT try argmax-on-shard.

---

### 3. Distributed RMSNorm via `rms_norm_pre_all_gather` + `rms_norm_post_all_gather` — ALREADY in our plan

**Estimated impact:** 8-18 ms/tok (per `feedback_distributed_rms_norm_corrections.md` projection, and matches our `feedback_ttnn_fused_ops_gap_analysis.md` top recommendation).

**What they do:** `ccl.py:374-418` (`tt_distributed_rmsnorm`):

```python
tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=...)
tt_stats = ttnn.reshape(tt_stats, padded_shape)
tt_stats_gathered = tt_all_gather(tt_stats, dim=3, cluster_axis=1, ...)
tt_out = ttnn.rms_norm_post_all_gather(inp, tt_stats_gathered, epsilon=epsilon, weight=gamma, ...)
```

Used everywhere via `distributed_norm.py:54-148` — pre-attn, pre-ff, lm_head pre-norm all use this with mesh-sharded `gamma` and `is_distributed_norm` toggle.

**What we do:** `server_tp.py:382-397` — single fused `ttnn.rms_norm` on the **replicated** x (after all_reduce). 11 calls of `_rms_norm_manual` per layer × 64 layers = 704/tok, all on replicated 5120-wide vectors.

**Gap:** Friend's pre/post AG split means each chip only normalizes its `dim/num_devices = 5120/4 = 1280`-wide slice. Saves 4× compute + saves the all_reduce that ours implicitly relies on (since after AG you can do the matmul sharded).

**Effort:** **Medium** — exactly `research/integration_distributed_rms_norm.md` plan (5-7h, 270 LOC). Friend's recipe is the production validation we needed.

**Custom kernel needed:** No — both ops exist in our build per `feedback_ttnn_fused_ops_gap_analysis.md`.

**Memory note that may be invalidated:** `feedback_distributed_rms_norm_corrections.md` Step 1 — friend's `ccl.py:374-418` is the line-by-line reference for the production pattern; Step 2 (`fused_rms_minimal` for residual fusion) is **NOT** what friend uses, so our deferral of Step 2 was correct.

---

### 4. Native partial-RoPE via `ttnn.experimental.rotary_embedding` — REPLACES our manual rotate-only

**Estimated impact:** ~3-8 ms/tok at 16 attn layers (rope is 44.3% of attn per `feedback_attn_perop_findings.md`; we'd need to confirm on TP, but native is documented 2.6× faster).

**What they do:** `qwen36.py:1504-1505`:

```python
q_rot_raw = ttnn.experimental.rotary_embedding(q_b[:, :, :, :rotary_dim], cos_b, sin_b, 0)
k_rot_raw = ttnn.experimental.rotary_embedding(k_b[:, :, :, :rotary_dim], cos_b, sin_b, 0)
```

Then concats with the passthrough `q_b[:, :, :, rotary_dim:]` slice (`qwen36.py:1520-1531`). Same partial-rotary semantic.

**What we do:** `server_tp.py:567-575` — manual rotate-only: slice rot/passthru, slice x1/x2, neg + mul + concat + mul + add + concat. 7 ops vs friend's 3 ops (slice+rotary_embedding+concat).

**Gap:** `feedback_c3_native_rope_abandoned.md` said the op's `cos_cache padded_shape` constraints conflict with TILE_LAYOUT padding for partial rotary. **Friend solves this by slicing to rotary_dim BEFORE rotary_embedding** — see `qwen36.py:1504` `q_b[:, :, :, :rotary_dim]`. The shape going into `rotary_embedding` is `[1, 1, n_heads_local, rotary_dim]` not `head_dim`, sidestepping the padding issue. They also iterate per-batch (`B_iter` loop `qwen36.py:1490`) which is fine at batch=1.

**Effort:** **Small** — ~30 LOC change to `gated_attn_step_tp` rope section. The C'3 abandonment may have been too pessimistic.

**Custom kernel needed:** No.

**Memory note that may be invalidated:** `feedback_c3_native_rope_abandoned.md` "don't re-attempt" — partially invalidated. Re-attempt with the friend's recipe: slice to rotary_dim first, then call rotary_embedding on the slice only. The error pattern from C'3 won't repeat because the cache shape will match.

---

### 5. Vocab-sharded LM head with multi-split DRAM-sharded matmul (skip final all_reduce on 1x4) — ALREADY in our menu

**Estimated impact:** ~4-8 ms/tok (lm_head matmul plus eliminating the lm_head all_reduce; this is candidate #6 in our menu).

**What they do:** `lm_head.py:14-211`:
- Vocab dim (152064 padded) split across 4 chips → `size_per_device = 152064/4 = 38016`
- That further split into chunks `<= max_columns_per_device` to avoid L1 OOM (`lm_head.py:38-55`), each chunk a separate `ttnn.linear` call.
- Weights stored as `create_dram_sharded_mem_config(k=args.dim, n=combined_split.shape[-1]/num_devices)` (`lm_head.py:103-104`) — DRAM-sharded.
- **Per `lm_head.py:194-209`: on 2D mesh non-Galaxy (which is our case 1x4 vs they call it 2x2 for MoE; dense 27B 1x4 also matches), the all_reduce is GATED by `if self.args.is_galaxy or not all(dim > 1 for dim in self.args.cluster_shape)`** — meaning for `(1,4)` where `min(cluster_shape)=1`, the condition `all(dim > 1)` is **False**, the LM head output stays sharded along vocab dim. Sampling does the all_gather (#2). One fewer collective per token.

**What we do:** `server_tp.py:830-840` (Stage B) — replicated 152064×5120 matmul on each chip + read full logits.

**Gap:** Vocab-sharded saves both weight memory (152064×5120 = 4× per chip vs replicated) and skips an all_reduce; pair with #2 (argmax-after-all-gather) and it's a net win.

**Effort:** **Medium** — ~200 LOC. Outline already exists in `research/integration_vocab_sharded_lm_head.md`. Friend's `lm_head.py` is the canonical reference.

**Custom kernel needed:** No. **The "DRAM-sharded" trick is critical** — uses `create_dram_sharded_mem_config` not `DRAM_MEMORY_CONFIG` (`lm_head.py:103, 118`); we use plain DRAM today.

**Memory note that may be invalidated:** None — confirms candidate #6 in our menu.

---

### 6. paged_fused_update_cache (K+V in one op) — quick KV write win

**Estimated impact:** ~0.5-1 ms/tok at 16 attn layers (each `paged_update_cache` is 0.02 ms × 32 writes/tok currently).

**What they do:** `attention.py:775-778` — when `use_qk_fused=True`, replaces two separate `paged_update_cache` calls with **one** `paged_fused_update_cache(K_cache, K, V_cache, V, update_idxs_tensor=cur_pos, page_table=...)`.

**What we do:** `server_tp.py:593-598` — two separate calls.

**Effort:** **Trivial** — 1-line replacement, ~5 LOC.

**Custom kernel needed:** No.

**Memory note:** `feedback_update_cache_replaces_scatter.md` is unaffected; this is a further fusion of the post-scatter pattern.

---

### 7. Fused `silu(z) * out` via `input_tensor_b_activations=[ttnn.UnaryOpType.SILU]` — small Python-level fusion

**Estimated impact:** ~0.5-1 ms/tok (eliminates one Python op × 48 DN layers).

**What they do:** `qwen36.py:1010-1017`:
```python
recurrent_output_compact = ttnn.mul(
    recurrent_output_compact, z_heads,
    input_tensor_b_activations=[ttnn.UnaryOpType.SILU],   # silu(z) fused INTO the mul
    output_tensor=recurrent_output_compact,
)
```

**What we do:** `server_tp.py:477-478` — `ttnn.silu(z_per_head)` then `ttnn.mul(out_normed, silu_z)`.

**Effort:** **Trivial** — 1-line replacement. Same trick applies to MLP's silu-gating if we don't already do it.

**Custom kernel needed:** No — kwarg on existing `ttnn.mul`.

---

### 8. Fused input-projection (Q+K+V+Z+B+A in ONE matmul) — `QWEN36_FUSE_INPUT_PROJECTIONS=1`

**Estimated impact:** Hard to estimate; reduces dispatch by 5× per DN layer (1 matmul instead of 5 + 4 slice+clone children). 48 DN layers × dispatch saving... 3-6 ms/tok.

**What they do:** `qwen36.py:355-365, 668-693` — concatenates QKV+Z+B+A weights into a single fused weight tensor (with per-chip channel re-ordering), one `_decode_linear_projection` call, then per-chip slice into 4 children with `ttnn.clone` to make them independent of the trace parent (`qwen36.py:658-666` — explicitly avoiding the trace-replay view bug).

**What we do:** `server_tp.py:417-426` — single matmul `w_in` already covers all (Q|K|V|Z|A|B), then 6 slices. We're already partially here. The fusion is real but mostly cosmetic in our case; the bigger win in friend's code is the **trace-safe clone** of each slice child to avoid the view-in-trace bug.

**Effort:** **Small** — verify our slice path produces independent buffers; add `ttnn.clone` around each slice if needed.

**Custom kernel needed:** No.

---

### 9. Mesh-resident `tt_all_reduce` with persistent buffers + tunable knobs (`chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`) — comms tuning

**Estimated impact:** Hard to estimate; per `tt-metal #33147` (cited in `reference_multi_chip_web_research.md`), CCL scaling is a tunable surface. Each layer's all_reduce is currently using defaults.

**What they do:** `qwen36.py:1099-1113` — explicit `reduce_scatter_minimal_async(...num_links=..., chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)`. The CCL semaphores are cached per cluster axis (`ccl.py:62-75`).

**What we do:** `server_tp.py:625-629` — bare `ttnn.all_reduce(partial)` with no knobs. The fallback path even uses `reduce_scatter+all_gather` composite when all_reduce fails.

**Effort:** **Small to Medium** — wire `TT_CCL` (`ccl.py:34-110`) or copy-paste enough of it. Most of the gain may come from `num_links` and `chunks_per_sync` tuning per layer.

**Custom kernel needed:** No.

**Memory note that may be invalidated:** `reference_multi_chip_web_research.md` cited tt-metal #33147; friend's repo is the production reference.

---

## Custom kernel inventory (friend has these; we'd need them upstream or rebuilt)

Three C++ ttnn ops, all at `experiments/.refs/tt-qwen-36/ttnn/cpp/ttnn/operations/experimental/transformer/`:

1. **`qwen36_gdn_decode`** — `qwen36_gdn_decode/qwen36_gdn_decode.{cpp,hpp}` + `device/qwen36_gdn_decode_{device_operation,program_factory}.cpp` + `device/kernels/{compute,dataflow}/*.cpp`. Replaces the **5-elementwise recurrence body** (15.6% of DeltaNet block, `feedback_deltanet_perop_findings.md` row B8). Signature: `(state, q, k, value, alpha, beta, active_mask=None, k_row_mask=None, normalize_qk_l2=False, ...) → (state_next, recurrent_output)`. State is updated in-place (kernel mutates the persistent buffer). Trace-safe.

2. **`qwen36_gdn_prepare_decode`** — `qwen36_gdn_prepare_decode/`. Replaces the **slice + reshape + repeat_interleave** of Q/K/V from conv_out (14.9% B5 GQA-repeat). Signature: `(conv_out, key_heads_per_device, value_heads_per_device, head_dim=128, ...) → (q, k, value)`.

3. **`qwen36_causal_conv_decode`** — `qwen36_causal_conv_decode/`. Replaces the conv1d 3-tap mul+add+silu (21.5% B4). Signature: `(mixed_qkv, state0..state3, weight0..weight3, ...) → (output, state0..state3)`. **CRITICAL FINDING:** This op exists in the C++ build BUT the test `test_qwen36_gdn_decode_native.py:211` explicitly asserts `"ttnn.experimental.qwen36_causal_conv_decode" not in model_source` — friend built it but is NOT using it. Production conv1d path is still Python mul+add accumulation (`qwen36.py:727-748`). They probably ran out of correctness validation time. Borrow-significance: the Python conv1d in their tree (which DOES match ours) is part of the 15.3 tok/s perf.

**Build process** (from `QWEN36_HARDWARE_RUN_GUIDE.md:357-378`):
- C++ op tree at `ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_*`
- Built into `_ttnncpp.so` via standard `cmake --build build-qwen36-native-py310 --target ttnn -j 8`
- Loaded into Python via `LD_PRELOAD=$TT_BUILD/ttnn/_ttnncpp.so` overlay (`QWEN36_HARDWARE_RUN_GUIDE.md:45`)
- Visible as `ttnn.experimental.qwen36_gdn_decode` / `qwen36_gdn_prepare_decode` / `qwen36_causal_conv_decode`

**Effort to enable on our build:** **Large** — requires building a custom ttnn from source (multi-hour build), the same overlay path friend uses, and matching the kernel ABI. Not feasible without a tt-metal source build. The kernels themselves are ~5 C++ files each (~1500-3000 LOC each across reader/writer/compute) — too large to re-implement quickly. **Realistic path: PR upstream to tt-metal main and wait for a future ttnn release.**

**ROI without the C++ ops:** Friend's TPB gain is **mostly Python-level**: native rope, distributed rmsnorm, on-device embedding/argmax/cur_pos, paged_fused_update_cache, vocab-sharded lm_head, fused input projection. Borrowing items #1-#9 above gets us a large fraction of their headroom without C++ work. The native GDN ops are bonus (likely accounts for ~3-5 ms/tok of the gap).

## Patterns we already have (NOT borrowable wins)

- **Paged SDPA decode** (`paged_scaled_dot_product_attention_decode` with explicit `SDPAProgramConfig`) — we have it post P18 (`feedback_mesh_paged_sdpa_works.md`). Friend uses the same op, same recipe.
- **Trace capture** — both projects use `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`.
- **`ttnn.copy` for in-trace state threading** — both projects use this to mutate persistent state buffers (DeltaNet SSM, conv_state) inside the trace.
- **FABRIC_1D + (1,4) mesh** — same fabric setup.
- **bf8 MLP weights** — `feedback_bf8_mlp_weights.md` says we already ship bf8 MLP. Friend uses `bfloat8_b` for MLP w1/w2/w3 (`mlp.py:97-101`) under accuracy preset.

## Things in friend's code that DON'T translate

- **MoE variant** (`qwen36_moe.py`) — we ship dense 27B only.
- **Prefetcher** (`prefetcher.py` + Galaxy 8x4 mesh) — Galaxy-specific. Our 4×P150 is 1x4 = `is_galaxy=False`. The `prefetcher is None` branches everywhere are what we should follow.
- **`ScaledEmbedding`** (`embedding.py:39-47`) — for Gemma-style scaled embeddings. Qwen3.6 uses plain `Embedding`.
- **Multimodal / vision** — friend explicitly disables (`TT_QWEN36_ENABLE_VISION=0`).
- **`generator_sglang.py` / `generator_vllm.py`** — server adapters for sglang/vLLM. Our Unix-socket server has its own protocol.
- **Chunked prefill** (`attention.py:1190 chunked_scaled_dot_product_attention`) — friend's prefill is "correctness-first sequential decode-style" per `QWEN36_README.md:79`. We have our own prefill path.
- **`fused_rms_minimal`** — NOT in friend's code (confirmed by grep). Our `reference_multi_chip_opt_menu_v2.md` mentioned it; friend doesn't use it, so our deferring it was right.

## Open questions for the friend (if user wants to ping him)

1. **Why is `qwen36_causal_conv_decode` built but not wired?** Did it underperform vs Python? Stability issue? Or just didn't get to validation? (test_qwen36_gdn_decode_native.py:211 explicitly asserts non-wiring)
2. **What's the per-op breakdown that justified writing the C++ `qwen36_gdn_decode` over Python?** Did Python recurrence have a specific scaling pathology we'd see if we hit longer sequence?
3. **`QWEN36_CCL_BF16` flag (`model_config.py:1071`)** — only default-on for 1x2 dense; what's the 1x4 stability story for bf16 CCL?
4. **What's their full host-loop latency, end to end, including `from_torch(tokens)`?** Friend's `--async-token-read` claim (`qwen36_decode_smoke.py:373`) says it reads tokens with a 1-step delayed nonblocking read — does that matter at 65 ms/step?
5. **Are they planning to land any of these as a tt-metal PR?** The C++ ops in `ttnn/cpp/...` look ready for upstreaming.

## Recommended ship order based on friend's choices vs our 14-candidate menu

The friend's evidence re-ranks our menu like this (their priorities, deduced from production wiring):

| Rank | Borrow | LOC | Hours | Confidence | Notes |
|---:|---|---:|---:|---|---|
| 1 | **#1 + #2** combo: on-device embed + plus_one + cos cache + all_gather-then-argmax | ~180 | 4-6 | HIGH | One coherent change; eliminates 11.3 ms/tok of host-loop overhead from `feedback_tracy_tp_breakdown.md` rows C+D combined. Lowest risk because each piece is independently testable. |
| 2 | #3 distributed RMSNorm pre/post AG | ~270 | 5-7 | HIGH | Plan already drafted (`research/integration_distributed_rms_norm.md`). Friend's `ccl.py:374-418` is the production validation. |
| 3 | #5 vocab-sharded LM head with DRAM-sharded multi-split | ~200 | 4-6 | MED-HIGH | Combine with #2 (single all_gather feeds argmax). Plan exists (`research/integration_vocab_sharded_lm_head.md`). DRAM-sharded mem config is the key piece we lack. |
| 4 | #4 native rotary_embedding for gated attention | ~30 | 1-2 | MED | Re-attempt with friend's "slice-first-then-rotary" recipe. Quick win or quick failure. |
| 5 | #6 paged_fused_update_cache | ~5 | 0.5 | HIGH | Drop-in. Almost free. |
| 6 | #7 silu-fused mul | ~5 | 0.5 | HIGH | One-line change × N call sites. |
| 7 | #8 + #9 trace-safe slice clones + CCL knobs | ~80 | 2-3 | LOW-MED | Smaller wins; pursue last. |

Bundle 1+2+5 in one PR gets us through embedding/argmax/normalization without touching attention internals. That alone should close 50-70% of the 7.02 → 15.5 gap.

## Things they're NOT doing that's in our menu (validates we should still ship it)

- **Speculative decoding / MTP head**: friend has neither in `models/tt_transformers/`. Our `feedback_speculative_decoding.md` D'3 path is independent and orthogonal — friend's choices don't invalidate it. If we ship MTP we'd compose multiplicatively with the borrows above.
- **fp32 KV cache**: friend's bf16 KV per `attention.py:175-180` (no fp32 cache wiring). They likely face the same long-context drift; their PERF.md doesn't list 32k context for Qwen3.6. Our `feedback_bf16_prefill_drift_cliff.md` cliff at pos 129 is probably present in their model too — they just haven't validated past 1024.
- **Multi-step Bx fusion (custom mul+sum kernel)**: friend solved this differently — they wrote `qwen36_gdn_decode` to replace the whole recurrence body, not just B8. Their approach is more aggressive (replace 5-op block, not 1 op) but requires C++.
