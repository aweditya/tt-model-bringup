# Multi-chip TP optimizations — v2 addendum

**Built:** 2026-05-14 by Agent O2.
**Purpose:** Extend Agent O's `multi_chip_optimizations_menu.md` with sources O skipped — primarily DeepSeek V3, mesh/CCL tech_reports, deeper Galaxy fused-op exploration, and ttnn op-existence verification.
**Method:** Source review only (local Mac, no device execution). Every claim cites path:line.

All paths under `experiments/.refs/tt-metal/` unless otherwise noted.

---

## New candidates

### [15] `rms_norm_pre_all_gather(..., residual_input_tensor=...)` — fused residual-add inside the pre-AG kernel
- **What:** The unfused distributed-RMSnorm (candidate #1) has an unused kwarg: `residual_input_tensor` accepted directly by `rms_norm_pre_all_gather`. This fuses the residual `h += attn_out` (or `h += ff_out`) into the same kernel that computes Σx² for the norm — saving one extra `ttnn.add` per block-pair.
- **Where:** `ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_nanobind.cpp:43, 82` (kwarg in API), `models/demos/llama3_70b_galaxy/tt/llama_decoder.py:157, 181` (Galaxy already uses this pattern via `attention_norm(x, h, mode)`).
- **Why:** `server_tp.py` currently does separate `ttnn.add` for residual (Galaxy's `unfuse_res_add` path is the *unoptimized* path — it's a fallback). Eliminating that add removes one full hidden-size sized eltwise per block-pair, which is 32 ops/tok (2× per layer × 16 layers).
- **Est. win:** 2–4 ms/tok (under O's #1 already; this is the *strict* version of it).
- **Effort:** L (add `residual_input_tensor=` kwarg once #1 lands).
- **Precondition:** Ships only after #1; cannot be used in the all-reduce-then-norm pattern we have today.

### [16] `ttnn.experimental.all_gather_concat` — fused all_gather + nlp_concat_heads_decode
- **What:** Single experimental kernel that fuses the post-SDPA all-gather of head outputs across chips with `nlp_concat_heads_decode`. Replaces `nlp_concat_heads_decode → all_gather` pair.
- **Where:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/all_gather_concat.hpp:15`; usage `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1266-1284`; canonical call `llama_attention.py:548-555`.
- **Why:** Decode attn output today goes through 2 dispatches (head concat then AG). Galaxy ships this fused — saves one dispatch + one buffer copy per layer × 16 = 16 saved.
- **Est. win:** 2–5 ms/tok.
- **Effort:** M (need persistent inter_tensor buffer + sharded output memcfg).
- **Precondition:** Output of `paged_scaled_dot_product_attention_decode` must be sharded in the shape expected by `all_gather_concat` (`[1, 1, 32, head_dim*num_local_heads]`).

### [17] `ttnn.experimental.llama_rs_matmul` — fused matmul + reduce_scatter (potentially double matmul)
- **What:** Single op that runs an FF1+FF3 matmul pair (or a single matmul) and the immediate reduce_scatter in one kernel, sharing a persistent interim buffer.
- **Where:** `ttnn/cpp/ttnn/operations/experimental/ccl/` (see registration `bind_function<"llama_rs_matmul"` from prior grep); usage `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:840-864` (double form), `:892-916` (single form); driven from `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:123-139`.
- **Why:** TG-Llama uses this for w1+w3 in the MLP. Replaces two `ttnn.linear` calls and a `line_reduce_scatter` (`server_tp.py:464`) with one fused op. Saves dispatch + intermediate buffer copy.
- **Est. win:** 4–8 ms/tok on MLP (combines #3 from O's menu with matmul fusion).
- **Effort:** M.
- **Precondition:** Persistent interim buffer per cluster_axis (Galaxy keeps these in `tt_ccl.reduce_scatter_buffers`); need our sub-device worker id.

### [18] `ttnn.experimental.llama_rs_create_heads` — fused QKV-projection RS + head split
- **What:** QKV matmul output → reduce_scatter → nlp_create_qkv_heads_decode all fused. Returns `(q_pre_rot, k_pre_rot, v)` tuple.
- **Where:** `llama_ccl.py:918-951`. Hard-codes Llama-shape `num_heads=8, num_kv_heads=1`; would need arg-passing for Qwen3.6.
- **Why:** Today server_tp.py does QKV linear → ttnn.all_reduce → 3 manual slices/reshapes (`server_tp.py:531-540`). One fused kernel saves both the collective overhead AND the head-split dispatches.
- **Est. win:** 3–6 ms/tok.
- **Effort:** M-H (must verify the kernel handles Qwen3.6's num_heads=24, num_kv_heads=4 ratio).
- **Precondition:** Qwen3.6's GQA ratio (6:1) — check whether the kernel supports arbitrary ratios.

### [19] `ttnn.experimental.all_to_all_async` — for future MoE / MTP B=2 verify routing
- **What:** All-to-all collective for cross-chip token routing.
- **Where:** Registered `bind_function<"all_to_all_async"` and `all_to_all_dispatch_metadata`; DeepSeek V3 demo uses for MoE expert routing (`deepseek_v3/tt/moe.py` indirectly via `deepseek_moe_reduce_scatter`).
- **Why:** Qwen3.6-27B is dense, but if D' speculative decode adds B=2 verify with per-slot routing across mesh, all-to-all will be the right primitive. Note in MEMORY: D'3 speculative decode probe queued.
- **Est. win:** Enables D'3; no direct decode tok/s win.
- **Effort:** Deferred.

### [20] `ttnn.experimental.matmul_reduce_scatter_async` — generic fused matmul+RS
- **What:** Galaxy's `llama_rs_matmul` is Llama-tuned; the generic version is `matmul_reduce_scatter_async`.
- **Where:** Registered `bind_function<"matmul_reduce_scatter_async"`. May be a simpler integration path than `llama_rs_matmul` since it doesn't bake in head counts.
- **Why:** Same as #17, but lower integration risk (no head-split assumption).
- **Est. win:** 3–6 ms/tok.
- **Effort:** M.
- **Precondition:** Confirm vs `llama_rs_matmul` performance.

### [21] Persistent global semaphores for *all* CCLs — DeepSeek's `CCL` class pattern
- **What:** DeepSeek V3 ships a `CCL` helper class that pre-allocates `sems_per_axis=2` global semaphores per axis per op type (gather, reduce_scatter, barrier) at startup, and cycles through them per call. Eliminates per-call semaphore allocation entirely.
- **Where:** `models/demos/deepseek_v3/tt/ccl.py:22-50` (init), `:60-100` (cycling), `:136-158` (populate_runtime_args).
- **Why:** Galaxy's `llama_ccl.py` does the same thing but in a heavier form. The DeepSeek pattern is cleaner and a closer fit for our `server_tp.py` size. Pre-allocates: `gather_sems, reduce_scatter_sems, barrier_sems` × `num_axes` × `sems_per_axis`.
- **Est. win:** 1–3 ms/tok (eliminating sem allocation latency; subsumed by #2 if we choose async-all-reduce).
- **Effort:** L (50 LOC class).
- **Precondition:** Whether ttnn.create_global_semaphore actually has measurable per-call cost on qb2 (probe first).

### [22] Multi-axis CCL (`cluster_axis=0` vs `cluster_axis=1`) for hybrid TP+DP — already implicit on (1,4) but worth noting
- **What:** On (1,4) we only have one cluster_axis (=1). On a 2D mesh we could combine TP along one axis with replicated-batch DP along the other.
- **Where:** `tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md:890-957` (8.3 Hybrid TP+DP), `tech_reports/LLMs/llms.md:1325-1331` (2D Weight Sharding).
- **Why:** qb2 has 4 chips physically in some mesh shape. If we eventually configure (2,2) we can do TP=2, DP=2.
- **Est. win:** N/A on (1,4); deferred.
- **Effort:** H (full reshape of mesh + activation distribution).
- **Precondition:** qb2 fabric supports (2,2); separate probe needed.

### [23] Use `paged_fused_update_cache` — UPGRADE PATH for O's candidate #11
- **What:** Galaxy already uses `ttnn.experimental.paged_fused_update_cache(keys, k, values, v, ...)` to write K and V in one dispatch.
- **Where:** `models/demos/llama3_70b_galaxy/tt/llama_attention.py:509-511`. Single call updates both caches.
- **Why:** O listed this; sharper recipe below in section "Refinements".
- **Est. win:** 1–2 ms/tok.
- **Effort:** L.
- **Precondition:** Both caches share same page table layout.

### [24] `ttnn.experimental.neighbor_pad_async` — for Mamba/DeltaNet-style state passing
- **What:** New CCL primitive for asynchronous neighbor-of-mesh padding (likely for SSM state propagation across chips).
- **Where:** Registered `bind_function<"neighbor_pad_async"`.
- **Why:** Speculative — could relate to DeltaNet's conv1d state propagation when sharded across chips. Worth probing if/when we re-attempt to split DeltaNet's conv state.
- **Est. win:** Unknown.
- **Effort:** Investigation only (no Qwen3.6 application identified).

### [25] `topology=ttnn.Topology.Ring` — switch from Linear when mesh permits
- **What:** Ring topology cuts CCL data movement by a factor of ~D/log2(D) vs line. `tech_reports/LLMs/llms.md:1346-1353` quantifies it.
- **Where:** Galaxy code uses `self.model_config["CCL_TOPOLOGY"]` which is Ring on TG. Reference table: AllGather DM(Linear) = K·N·DF/D·(D-1)·D vs DM(Ring) = K·N·DF·D·log2(D).
- **Why:** Today `server_tp.py:399, 462, 499` uses `ttnn.all_reduce`/`ttnn.all_gather` which default to whatever topology the build picks. On (1,4) with all P150s connected in a line, Ring may or may not be physically available — but on qb2's actual fabric we don't know. Probe required.
- **Est. win:** 2–5 ms/tok if Ring works on qb2's 4-chip fabric.
- **Effort:** L (single param change) + probe.
- **Precondition:** qb2 fabric supports 4-chip ring (not guaranteed on (1,4) line layout).

---

## Refinements to O's candidates

### #1 (distributed RMSNorm) — additional constraints + sharper recipe
- **Shape constraint** (from `rmsnorm_distributed_nanobind.cpp:76`): "Sharded inputs cannot be height-sharded, padded height must equal TILE_HEIGHT (32)." For B=1 decode, this is naturally satisfied.
- **Stats sharding constraint** (`:157`): when sharded, `stats` tensor must be sharded across **one core**.
- **Cleaner reference:** DeepSeek's `DistributedRMSNorm` (`models/demos/deepseek_v3/tt/rms_norm/distributed_rms_norm.py:222-246`) is ~25 LOC vs Galaxy's macro-laden version — closer fit for `server_tp.py`. Uses a 3-call pattern: `rms_norm_pre_all_gather` → `all_gather_async` → `rms_norm_post_all_gather`. The weight is stored sharded over the dim-1 axis with `shard_dims=(0, -2)` and shape `(num_shards, 1, -1, TILE_SIZE)`.
- **`use_2d_core_grid` kwarg** (`rmsnorm_distributed_nanobind.cpp:47, 86`): set to False for our 1D (1,4) mesh. Galaxy sets `use_2d_grid=False` explicitly (`llama_ccl.py:1366`).

### #2 (`all_reduce_async`) — sharper signature
- **Required persistent buffer:** Galaxy uses `self.persistent_buffers[cluster_axis]` (`llama_ccl.py:711, 714`) — a pre-allocated DRAM tensor of the same shape as the all_reduce input/output. Must be created with the same dtype as the input.
- **`use_optimal_ccl_for_llama=True`** (`llama_ccl.py:726`): selects a tuned kernel. Pass-through; if our shape matches, it accelerates.
- **`use_noc1_only` flag** (`:725`): may help avoid contention with prefetcher (when we add #5).

### #3 (reduce_scatter_minimal_async) — sharper signature
- **`persistent_output_buffers`** (`llama_ccl.py:1059`): a LIST of pre-allocated buffers (one per chunk of the scatter). Pre-allocate at server bootstrap.
- **`num_workers_per_link`** tuning (`:1054`): 1 for short seqlen, 4 for longer. For decode (seqlen=1) use 1.
- **`barrier_semaphore`** (`:1062`): use `get_and_cycle_barrier_semaphore_handle` pattern.

### #4 (DRAM-sharded matmul) — explicit helper functions
- **Recipe** (`tech_reports/LLMs/llms.md:1529-1559`):
  ```python
  weights_memory_config = create_dram_sharded_mem_config(k=K, n=N)
  matmul_program_config = dram_matmul_config(m=M, k=K, n=N, num_cores=core_grid.num_cores)
  output = ttnn.linear(act, weights, program_config=matmul_program_config,
                       memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
  ```
- **HARD CONSTRAINT** (`llms.md:1561`): "Take care that the core grid evenly divides both activations and output. Padding functionality is not implemented for DRAM-Sharded matmuls." Qwen3.6 hidden=5120, intermediate=17408 — check divisibility against compute_with_storage_grid_size at qb2.

### #5 (sub_devices + dram_prefetcher) — sharper recipe location
- **Galaxy `TtLlamaPrefetcherSetup.__init__`** (`prefetcher_common.py:75-99`) is the canonical reference:
  - 2 sub_devices: prefetcher (`SubDeviceId(0)`) and worker (`SubDeviceId(1)`)
  - `global_cb_size = 728 * 1088` (size in tiles)
  - `set_sub_device_stall_group([prefetcher_id, worker_id])`
- **CB created lazily after sub_device_manager loads** (`:104-110`), via `ttnn.create_global_circular_buffer`.
- **Per-matmul plumbing:** every `ttnn.linear` call accepts `global_cb=` and `sub_device_id=` kwargs (`llama_mlp.py:136-137, 183-184`).

### #6 (vocab-sharded lm_head) — RECIPE WAS THIN; DeepSeek's `LMHead1D` is much cleaner
- **DeepSeek 1D lm_head** (`models/demos/deepseek_v3/tt/lm_head1d.py`):
  - Weight shard config: `shard_dims=(None, -2)` over rows of mesh (`:75`); each chip holds `vocab_size / mesh_cols` rows of the [vocab, hidden] weight.
  - Program config: `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0=True` to broadcast the small decode activation to all cores (`:140-150`).
  - Computes `K_tiles = hidden_dim // 32, N_tiles = (vocab_size // mesh_cols) // 32` (`:113-114`).
  - Returns the per-chip slice WITHOUT all-gather; final logits gather happens externally if needed.
- **Galaxy's version** (`lm_head.py:60-105`) uses `ShardTensor2dMesh(dims=(3, 2), mesh_shape=args.cluster_shape)` — 2D shard. For (1,4) prefer DeepSeek's 1D pattern.
- **`max_columns_per_device=128256 // 4`** (Galaxy `lm_head.py:20`) — splitting too wide can OOM/hang; cap shard width.

### #7 (`fused_rms_minimal`) — sharper API + constraints
- **Signature** (`rms_allgather_nanobind.cpp:24-46`):
  ```
  ttnn.fused_rms_minimal(input_tensor, program_config, cluster_axis, mesh_device,
                          global_semaphore,
                          persistent_output_tensor=None, num_links=None,
                          topology=Linear, subdevice_id=None,
                          dtype=None, compute_kernel_config=None, memory_config=None,
                          residual_input_tensor=None,  # FUSES residual add
                          epsilon=1e-12, weight=None, stats=None,
                          use_noc1_only=False)
  ```
- **SHAPE CONSTRAINT** (docstring `rms_allgather_nanobind.cpp:21-22`): "Requires a pre-allocated persistent tensor for the intermediate all gather that is tiled, shape per device (32,32) and sharded with a shard shape (32,32) on core (0,0). Requires that input tensor be of shape (1,1,32,M) where M is a multiple of 32." Our hidden=5120 is a multiple of 32 — safe.
- **Galaxy usage** (`llama_ccl.py:1413-1427`): drives via `tt_sharded_distributed_rmsnorm` wrapper that wires `residual_input_tensor=res`, `weight=gamma`, `stats=persistent_buffer` from `tt_ccl.all_gather_buffers["LAYERNORM"]`.

### #8 (`all_gather_minimal_matmul_async`) — uses different param naming
- **`force_transpose=True`** (`llama_ccl.py:1217, 1258`) — required by Galaxy's W2 path; may need similar for us.
- **`num_workers_per_link`** auto-derives from core_grid (`:1238`).
- **`num_buffers_per_channel=8`** (`:1260`) — Galaxy hardcodes this.
- **Returns list, take [0]** (`:1264`).

### #11 (`paged_fused_update_cache`) — sharper
- **Galaxy call** (`llama_attention.py:509-511`): single dispatch for both K and V cache writes.
  ```python
  ttnn.experimental.paged_fused_update_cache(
      keys, k_heads_1BKD, values, v_heads_1BKD,
      update_idxs_tensor=current_pos, page_table=page_table)
  ```
- **Drop-in replacement** for our `server_tp.py:572, 575` (two separate `paged_update_cache` calls).

### #14 (`nlp_create_qkv_heads_decode`) — exact signature
- **Signature** (`ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.cpp:13`):
  ```
  (q, k, v) = ttnn.experimental.nlp_create_qkv_heads_decode(
      input_tensor, num_heads, num_kv_heads,
      overlap_qk_coregrid=True, batch_offset=None, slice_size=None,
      memory_config=None)
  ```
- **Input layout requirement** (cpp:48-52): `padded_shape[3] % (num_heads + 2*num_kv_heads) == 0`. Inferred head_dim from that. For Qwen3.6 with 24Q+4KV per chip: `(24 + 8) = 32`, `head_dim = qkv_width / 32`.
- **`overlap_qk_coregrid=True`** is default for sharded inputs.

---

## Top 5 recipe snippets (copyable templates)

### #1 Distributed RMSNorm — DeepSeek pattern (cleanest)

`models/demos/deepseek_v3/tt/rms_norm/distributed_rms_norm.py:222-246`:
```python
@classmethod
def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
    program_config = cls._get_pc(x.memory_config())
    # Part 1: compute local stats
    tt_stats = ttnn.rms_norm_pre_all_gather(
        x, program_config=program_config, **cfg["rms_norm_pre_all_gather"])

    # All-gather stats (tiny: just sum-of-squares per chip)
    ccl = cfg["ccl"]
    tt_gathered_stats = ttnn.experimental.all_gather_async(
        tt_stats, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))
    ttnn.deallocate(tt_stats)

    # Part 2: apply normalization with full stats
    tt_out = ttnn.rms_norm_post_all_gather(
        x, tt_gathered_stats, program_config=program_config,
        **cfg["rms_norm_post_all_gather"])
    ttnn.deallocate(tt_gathered_stats)
    return tt_out
```

Galaxy unsharded version `llama_ccl.py:1358-1390` is the bare-bones reference if we want zero plumbing.

### #2 `all_reduce_async` — call site

`models/demos/llama3_70b_galaxy/tt/llama_ccl.py:712-727`:
```python
output_tensor_mesh = ttnn.experimental.all_reduce_async(
    input_tensor_mesh,
    persistent_buffer,                                            # pre-allocated
    cluster_axis=cluster_axis,
    mesh_device=self.mesh_device,
    multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
    num_links=num_links,
    memory_config=memory_config,
    dtype=dtype,
    topology=self.model_config["CCL_TOPOLOGY"],
    subdevice_id=self.worker_sub_device_id,
    use_noc1_only=use_noc1_only,
    use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
)
```

### #6 Vocab-sharded LM head — DeepSeek 1D (simpler than Galaxy 2D)

`models/demos/deepseek_v3/tt/lm_head1d.py:140-150` (program config) and `:200-202` (forward):
```python
# At config time:
program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(grid_size.x, grid_size.y),
    in0_block_w=in0_block_w,       # 32 down to first divisor of K_tiles
    out_subblock_h=1,
    out_subblock_w=out_subblock_w, # min(per_core_N, 4) downstep
    per_core_M=1,
    per_core_N=per_core_N,         # ceil(N_tiles / num_cores) where N_tiles = (vocab/mesh_cols)/32
    fuse_batch=True, fused_activation=None,
    mcast_in0=True,                # broadcast small decode act to all cores
)

# Forward:
return ttnn.linear(x, weight, **cfg["linear"])  # no all-gather; output is per-chip vocab slice
```

Weight shard: `shard_dims=(None, -2)` over column axis (line `:75`).

### #7 `fused_rms_minimal` — Galaxy wrapper (RMSNorm+AG+residual+gamma)

`models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1413-1427`:
```python
tt_out = ttnn.fused_rms_minimal(   # also reachable as ttnn.experimental.fused_rms_minimal
    inp,
    ln_sharded_progcfg,
    cluster_axis,
    tt_ccl.mesh_device,
    semaphore,
    topology=ccl_topology,
    residual_input_tensor=res,     # FUSES residual add (h += attn/ff)
    num_links=1,
    epsilon=epsilon,
    weight=gamma,
    stats=persistent_buffer,        # pre-allocated AG buffer
    memory_config=output_mem_config,
    use_noc1_only=use_noc1_only,
)
```

Input shape must be `(1,1,32,M)` with M%32==0. Persistent stats buffer must be `(32,32)` tiled and sharded on core `(0,0)`.

### #14 `nlp_create_qkv_heads_decode` + #16 `all_gather_concat` — attention recipe

QKV split (`llama_ccl.py:933-949`, wrapper around `ttnn.experimental.llama_rs_create_heads`; if not adopting the all-fused form, use bare `nlp_create_qkv_heads_decode` directly per attention.py):
```python
(q, k, v) = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv_concat,                # output of QKV linear (post-allreduce)
    num_heads=N_Q_LOCAL,
    num_kv_heads=N_KV_LOCAL,
    memory_config=qkv_memory_config,
)
```

Post-SDPA concat+gather (`llama_attention.py:548-555`):
```python
attn_output_cat = self.tt_ccl.all_gather_concat(
    attn_output_1G4D_sharded,
    dim=1, cluster_axis=1, num_links=GALAXY_NUM_LINKS,
    memory_config=SHARDED_ATTN_WO_INPUT_RING_MEMCFG,
    num_heads=self.n_local_heads,
)
# attn_output_cat is now ready for out_proj linear, no manual concat_heads + AG.
```

---

## API existence table

All paths under `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/`. "Python access" is the canonical Python call.

| Op | C++ location | Python access | Notes |
|---|---|---|---|
| `rms_norm_pre_all_gather` | `normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp:15` | `ttnn.rms_norm_pre_all_gather` | Takes `residual_input_tensor` to fuse residual add. |
| `rms_norm_post_all_gather` | `normalization/rmsnorm_distributed/rmsnorm_post_all_gather.hpp:15` | `ttnn.rms_norm_post_all_gather` | Takes gathered stats + weight (gamma). |
| `fused_rms_minimal` | `experimental/ccl/rms_allgather/rms_allgather.hpp:16` | `ttnn.fused_rms_minimal` (also `ttnn.experimental.fused_rms_minimal`) | Fuses pre+AG+post+residual+gamma. Strict shape (1,1,32,M). |
| `all_gather_async` | `experimental/ccl/all_gather_async/` | `ttnn.experimental.all_gather_async` | Persistent-buffer async gather. |
| `all_reduce_async` | `experimental/ccl/all_reduce_async/` | `ttnn.experimental.all_reduce_async` | Persistent-buffer async reduce. |
| `reduce_scatter_minimal_async` | `experimental/ccl/reduce_scatter_minimal_async/` | `ttnn.experimental.reduce_scatter_minimal_async` | Persistent-output-buffers list. |
| `all_gather_minimal_matmul_async` | `experimental/ccl/all_gather_minimal_matmul_async/` | `ttnn.experimental.all_gather_minimal_matmul_async` | Fused AG + matmul. Returns list, [0] is output. |
| `all_gather_concat` | `experimental/ccl/all_gather_concat_heads_fused/all_gather_concat.hpp:15` | `ttnn.experimental.all_gather_concat` | Fused AG + nlp_concat_heads_decode. |
| `all_reduce_create_qkv_heads` | `experimental/transformer/all_reduce_create_qkv_heads/all_reduce_create_qkv_heads.hpp:13` | `ttnn.experimental.all_reduce_create_qkv_heads` | Fused all_reduce + QKV head split. Returns 4-tuple (xqkv, q, k, v). |
| `nlp_create_qkv_heads_decode` | `experimental/transformer/nlp_create_qkv_heads_decode/nlp_create_qkv_heads_decode.cpp:13` | `ttnn.experimental.nlp_create_qkv_heads_decode` | Pure head-split for decode. |
| `nlp_concat_heads_decode` | `experimental/transformer/nlp_concat_heads_decode/` | `ttnn.experimental.nlp_concat_heads_decode` | Used by Galaxy (llama_attention.py:344). |
| `paged_fused_update_cache` | `kv_cache/paged_fused_update_cache/` (existence implied by `llama_attention.py:509` call) | `ttnn.experimental.paged_fused_update_cache` | Single dispatch for K+V cache write. |
| `paged_scaled_dot_product_attention_decode` | `transformer/sdpa_decode/` | `ttnn.transformer.paged_scaled_dot_product_attention_decode` | Already MEMORY-validated at 32k. |
| `llama_rs_matmul` | `experimental/ccl/` | `ttnn.experimental.llama_rs_matmul` | Fused matmul+RS; can do double matmul (w1+w3). |
| `llama_rs_create_heads` | `experimental/ccl/` | `ttnn.experimental.llama_rs_create_heads` | Fused RS+QKV head split. |
| `llama_reduce_scatter` | `experimental/ccl/` | `ttnn.experimental.llama_reduce_scatter` | Llama-tuned RS. |
| `matmul_reduce_scatter_async` | `experimental/ccl/` | `ttnn.experimental.matmul_reduce_scatter_async` | Generic matmul+RS (no head-split). |
| `all_to_all_async` | `experimental/ccl/` | `ttnn.experimental.all_to_all_async` | Future: MoE / B=2 verify. |
| `neighbor_pad_async` | `experimental/ccl/` | `ttnn.experimental.neighbor_pad_async` | Speculative — for SSM/DeltaNet state? |
| `minimal_matmul` | `experimental/minimal_matmul/` | `ttnn.experimental.minimal_matmul` | Galaxy prefill matmul; Galaxy notes "not giving improvement over linear" for lm_head (`lm_head.py:170-178`). |
| `fast_reduce_nc` | (existing) | `ttnn.experimental.fast_reduce_nc` | Used inside Galaxy's all_reduce-via-AG fallback. |
| `create_global_semaphore` | `global_semaphore.hpp` | `ttnn.create_global_semaphore` | Persistent multi-device sem. |
| `create_global_circular_buffer` | `tech_reports/SubDevices/SubDevices.md:160` | `ttnn.create_global_circular_buffer` | For prefetcher (#5). |
| `SubDevice` / `create_sub_device_manager` | `tech_reports/SubDevices/SubDevices.md:46, 50` | `ttnn.SubDevice` / `device.create_sub_device_manager` | Required for #5 dram_prefetcher. |

---

## DeepSeek-borrowable patterns

These are architecturally not directly applicable (Qwen3.6-27B is dense, no MoE, no MLA), but the *infrastructure* patterns transfer:

### A. `CCL` helper class with per-axis semaphore cycling
- **Source:** `models/demos/deepseek_v3/tt/ccl.py:9-189`
- **Pattern:** Pre-allocate `sems_per_axis=2` semaphores per axis per op type at init, cycle through them per call.
- **Apply to server_tp.py:** Replace any per-call `ttnn.create_global_semaphore` with this class. ~50 LOC drop-in.
- **Why DeepSeek's is better than Galaxy's:** half the file size, doesn't bake in 8x4 cluster shape, doesn't require Llama-specific buffer keys.

### B. Page-table aliasing for B=2 verify (speculative decode)
- **Source:** `models/demos/deepseek_v3/tt/generator.py:58-99` (`_build_verify_alias_page_table_host`)
- **Pattern:** During MTP verify, alias rows of the page table so two batch slots point at the same physical pages (one for the draft + verify, one for the rollback if rejected). Lets B=2 decode share KV cache between slots.
- **Apply when:** D'3 MTP speculative decode lands. MEMORY note `feedback_speculative_decoding.md` calls this out — generator.py is the reference.

### C. RunConfig / ModelDecodeConfig / WeightConfig 3-layer config dance
- **Source:** `models/demos/deepseek_v3/tt/rms_norm/distributed_rms_norm.py:71-128`
- **Pattern:** Static-class methods `convert_weights`, `decode_model_config`, `create_state` — separate weight conversion, shape/memcfg config, and runtime state. `RunConfig` merges them at runtime.
- **Apply to server_tp.py:** Probably overkill for our single-server use case. Useful to know if we ever want to add prefill mode cleanly.

### D. Decode block sequencing showing per-op deallocations
- **Source:** `models/demos/deepseek_v3/tt/mtp.py:271-345` (MTP forward_decode)
- **Pattern:** EVERY intermediate tensor is `ttnn.deallocate(...)`-ed immediately after use. Even input copies via `_has_distinct_buffer(in, orig)` guard.
- **Apply to server_tp.py:** Likely we leak L1 buffers across iterations. Audit `deltanet_step_tp` and `gated_attn_step_tp` for missing deallocs.

### E. Sharded RMSnorm input memory config (decode mode)
- **Source:** `distributed_rms_norm.py:104-128`
- **Pattern:** Build a `create_sharded_memory_config` with `shape=(roundup(batch, TILE), roundup(hidden / (num_cores * mesh_cols), TILE))`, `core_grid=ttnn.CoreGrid(x=4, y=7)` (DeepSeek's choice), strategy=WIDTH. Also build a dedicated `rms_norm_stats_memory_config` of shape `[1, 1, TILE_SIZE, TILE_SIZE * mesh_cols]` width-sharded on `CoreGrid(y=1, x=1)`.
- **Apply to server_tp.py:** Required input memcfg setup if we ship #1 with the sharded path.

---

## Surprises / dead-ends

- **`ttnn.minimal_matmul` is NOT auto-better than `ttnn.linear` for lm_head.** Galaxy explicitly comments it out (`lm_head.py:170-178`): "Minimal matmul is not giving any performance improvement over linear". So even though it has the word "minimal" it isn't the right answer for our lm_head.
- **DeepSeek's lm_head DOES NOT all-gather logits at end.** Each chip keeps its vocab slice. Top-k / sampling happens per-chip (or via a final small AG). This is structurally different from O's #6 framing which assumed we'd gather. Saves another AG.
- **`ttnn.experimental.deepseek_moe_reduce_scatter` and `moe_compute` exist** — irrelevant for dense Qwen3.6 but interesting if we ever try GLM-4.5-Air (MoE).
- **The fused `fused_rms_minimal` requires shape `(1,1,32,M)`.** For our B=1 decode this is automatic, but if we add B=2 verify it changes — input becomes `(1,1,32,M)` only with batch=1; for B=2 we'd need batch=2 in a different dim. Verify before MTP/B=2 work.
- **DeepSeek's distributed RMSnorm uses `cluster_axis=1`** even though their mesh shape varies — same as Galaxy. Our (1,4) means cluster_axis=1 covers all 4 chips; we never have a meaningful axis=0 for collectives.
- **Sub-meshes via `mesh_device.create_submeshes(ttnn.MeshShape(2, 4))`** is a real, documented API (`tech_reports/Programming_Mesh_of_Devices:929`). On our (1,4) we'd typically not use it, but if we ever go to (2,2) hybrid TP+DP we can.
- **There is no `ttnn.all_reduce_async` published as a standalone alternative to `ttnn.all_reduce`** — the async form lives under `ttnn.experimental.all_reduce_async` (confirmed `bind_function<"all_reduce_async"`).
- **`use_noc1_only` is a real kwarg, not a typo.** Tells the kernel to use only NOC1 lanes — useful for avoiding contention with the prefetcher on NOC0. Will matter for #5.
