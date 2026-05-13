# 05 — ttnn Memory Configurations: Deep Dive

Background reading for the KV-cache scatter work (`04_update_cache_reference_op.md`) and any custom in-place op we ship. All line numbers cite the cloned tt-metal under `experiments/.refs/tt-metal/`.

---

## 1. Memory-layout taxonomy

A `ttnn.MemoryConfig` is a triple `(memory_layout, buffer_type, optional<ShardSpec>)`. The enum that drives the first axis is in `tt_metal/api/tt-metalium/buffer_types.hpp:11`:

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    HEIGHT_SHARDED = 2,
    WIDTH_SHARDED = 3,
    BLOCK_SHARDED = 4,
    ND_SHARDED = 5,
};
```

The second axis (`buffer_type_`) is in the same file at line 33: `DRAM`, `L1`, `SYSTEM_MEMORY`, `L1_SMALL`, `TRACE`. The default `MemoryConfig()` constructor (in `tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp:38,90-91`) produces interleaved-DRAM:

```cpp
TensorMemoryLayout memory_layout_ = TensorMemoryLayout::INTERLEAVED;
BufferType         buffer_type_   = BufferType::DRAM;
```

Pre-baked Python aliases (from `ttnn/ttnn/tensor.md` "Memory Config" section):

| Alias | layout | buffer | Notes |
|---|---|---|---|
| `ttnn.DRAM_MEMORY_CONFIG` | INTERLEAVED | DRAM | Default for weights, KV caches, anything large. |
| `ttnn.L1_MEMORY_CONFIG` | INTERLEAVED | L1 | Interleaved across cores' SRAM; small intermediates. |
| `ttnn.create_sharded_memory_config(...)` | HEIGHT/WIDTH/BLOCK_SHARDED | L1 | Hand-placed shards. |

Per the metalium doc (`tt_metal/advanced_topics/memory_for_kernel_developers.md:164-272`): interleaved is round-robin across all memory banks at `page_size` granularity — simple, generic, prone to NoC contention. Sharded lets you pin specific tiles to specific cores. The doc states it bluntly (line 272): **"Sharding is usually only done for SRAM buffers."**

Sharding-strategy semantics (`memory_for_kernel_developers.md:268-270`, `ttnn/tensor.md:189`):

| Strategy | Splits along | Typical use |
|---|---|---|
| HEIGHT_SHARDED | rows (dim 0 of folded 2D view) | Per-batch state, one row-band per core; matches our `[B, num_heads, max_seq, head_dim]` cache folded to `[B*num_heads*max_seq, head_dim]` |
| WIDTH_SHARDED | columns (last dim) | Wide matmul activations (column-tiled) |
| BLOCK_SHARDED | 2D grid | Square matmul activations, conv, anywhere with 2D core-grid producers/consumers |

Trade-offs:

| | INTERLEAVED DRAM | INTERLEAVED L1 | SHARDED L1 |
|---|---|---|---|
| Capacity | ~32 GB (Blackhole, 8×4 GB) | ~195 MB total (130 Tensix × 1.5 MB) | same as L1 |
| Bandwidth | DRAM ctrl, NoC contention | SRAM, fewer hops | SRAM, producer/consumer co-located |
| Setup cost | none (default) | none | shard-spec must align with kernel work split |

---

## 2. ShardSpec mechanics

A `ShardSpec` (in `tt_metal/api/tt-metalium/buffer.hpp:58-84`) is three fields:

```cpp
struct ShardSpec {
    CoreRangeSet grid;                 // which cores hold a shard
    std::array<uint32_t, 2> shape;     // shard shape in *elements*, [rows, cols]
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
};
```

Two extra facts you need before you compute one:

- The shard is **per-core**: every core in `grid` owns exactly one shard of `shape` (`ttnn/tensor.md:188` — *"Each core will have a single shard."*).
- The shape is in **elements**, not tiles or bytes. Most kernels then want `page_shape = {TILE_HEIGHT, TILE_WIDTH}` and `tensor2d_shape_in_pages = {height_tiles, width_tiles}` — those go into a `ShardSpecBuffer` (`buffer.hpp:88-114`) when constructing the buffer directly.

ttnn's Python helper computes the shard shape for you when given the tensor shape, grid, and strategy. Signature (from `ttnn/ttnn/api/ttnn.create_sharded_memory_config.md:8`):

```python
ttnn.create_sharded_memory_config(
    shape, core_grid, strategy,
    orientation=None,
    use_height_and_width_as_shard_shape=False
) -> MemoryConfig
```

The doc note says it all (line 23): **"Currently sharding only supports L1 tensors."** That's the API confirmation of the metalium guide claim.

When to actually shard — three rules of thumb from the metalium guide (lines 256-272):

1. The op has a predictable, high-bandwidth access pattern (matmul, conv, sharded SDPA).
2. Producer and consumer can be pinned to the same core grid (no reshard needed).
3. The shard fits in L1 alongside the kernel's CBs and intermediates.

Rule 3 is the binding constraint for our 27B model — see §3.

---

## 3. DRAM vs L1 buffers

L1 = 1.5 MB per Tensix tile on Wormhole/Blackhole (`dram_loopback.md:51`, `memory_for_kernel_developers.md:52`). Blackhole p150: 8 DRAM controllers × 4 GB, 130 Tensix tiles. Lock-step allocation means every L1 alloc is mirrored across all banks (some waste; `memory_for_kernel_developers.md:151-162`).

Where things go:

| Tensor | Where | Why |
|---|---|---|
| Weights, KV cache, inter-layer activations | DRAM, INTERLEAVED | Too big for L1, touched once per token |
| Reused matmul operands (sharded SDPA inputs) | L1, SHARDED | NoC bandwidth dominates dispatch |
| Compute intermediates inside a kernel | L1 via CB | Producer/consumer co-located |
| Anything > ~1 MB per core | DRAM | L1 won't hold it |

For Qwen3.6-27B at `[B=1, n_kv_heads=4, max_seq=32768, head_dim=128]` bf16, each K or V is `4 * 32768 * 128 * 2 = 32 MiB` per layer. Even sharded over all 130 Tensix cores that's ~250 KB/core — feasible but eats most of L1, leaving nothing for CBs. **KV cache must live in DRAM.** Matches `tt_transformers/tt/attention.py:428` which uses `memory_config=ttnn.DRAM_MEMORY_CONFIG`.

---

## 4. The `memory_config` kwarg

Most ttnn ops accept `memory_config=` specifying the **output** layout. If you omit it: creation ops (`from_torch`, `as_tensor`, `zeros`) default to `DRAM_MEMORY_CONFIG`; elementwise/reduction ops inherit from input 0. Layout transitions are explicit calls: `ttnn.to_memory_config`, `ttnn.interleaved_to_sharded`, `ttnn.sharded_to_interleaved`, `ttnn.reshard`.

Hard layout requirements show up as `TT_FATAL`s in `validate(...)`. `paged_update_cache_device_operation.cpp:50-52,160`:

```cpp
TT_FATAL(cache_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
         "Only interleaved cache is supported");
...
TT_FATAL(input_tensor.is_sharded(), "Expect input_tensor to be sharded");
```

So `paged_update_cache` enforces an **asymmetric layout**: cache=INTERLEAVED, new-token input=SHARDED. Not ergonomics — that's what the program factory and kernels were wired for (one core per batch lane reads its own L1-resident shard).

---

## 5. Trace capture interaction

Memory configs are baked into the program at compile time (they appear in `TensorAccessorArgs<>` and the writer's CT-args block — `writer_update_cache_interleaved_start_id.cpp:17-37`). After `ttnn.begin_trace_capture` (`tutorials/ttnn_intro.md:1043`), every intermediate's layout is **frozen for the trace's lifetime**.

Two consequences:

1. No layout changes between replays. `override_runtime_arguments` (`update_cache_multi_core_program_factory.cpp:288-331`) patches buffer addresses and per-core start IDs only — not memory_config.
2. Dynamic values must enter as device-resident tensors. That's why the paged variant moves `cur_pos` into an INT32 ROW_MAJOR `update_idxs_tensor` (`paged_update_cache_device_operation.cpp:112-115`); only the pointer to it is patched per enqueue.

Don't insert an `interleaved_to_sharded` mid-trace just for one op — it captures fine but you pay the reshard cost every replay.

---

## 6. Blackhole #16674: why SHARDED writer hangs

Direct evidence (WebFetch + `04_update_cache_reference_op.md:148-156`):

- Title: "Blackhole: ttnn.experimental.paged_update_cache consistently hanging". Labels P1/blackhole/bug. Assigned `cglagovichTT`. Closed.
- Sweep covered `block_size=32`, `head_dim=128`, `seq_len=32`, users `1024`, `heads ∈ {1,8}`, bf16/bf8_b, `cur_pos ∈ {0,1,127,1057}` — *"the lockup seems pretty independent of parameters."*
- Root cause is not stated in the issue body and closing comments don't surface via WebFetch. Grep for `"16674"` across the cloned repo returns zero hits — no in-source breadcrumb.

Why SHARDED writer is still the most likely trigger:

- The op requires sharded input (`paged_update_cache_device_operation.cpp:160`); only the sharded path runs `UpdateDynamicCircularBufferAddress` (`paged_update_cache_program_factory.cpp:373-375`).
- MEMORY note `paged_sdpa_decode_works_at_32k`: paged SDPA-decode at 32k was unblocked on Blackhole *once the writer went sharded* (different op, explicit note that it's not #16674). So the failure surface isn't "sharded writers generally" — it's specifically `paged_update_cache`'s sharded-input → interleaved-cache `noc_async_write_tile` pattern on Blackhole's NoC.
- Parameter-independence points at structural NoC arbitration / semaphore-init, not tiling. Consistent with Blackhole having more NoC clients than Wormhole (8 DRAM ctrls vs 6).

Best characterization: a Blackhole-specific NoC/semaphore deadlock in `paged_update_cache`'s sharded-input writer. Non-paged `update_cache` takes an interleaved input and doesn't hit it — recommended fallback.

---

## 7. Recommendation for our KV-cache scatter

Cache shape: `[B=1, n_kv_heads=4, max_seq=32768, head_dim=128]`, bf16/bf8_b, TILE_LAYOUT. ~32 MiB per K/V per layer — DRAM-only. Single token write per decode step.

**Cache memory_config:**

```python
ttnn.DRAM_MEMORY_CONFIG
# equivalent: ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
```

Rationale:

1. **Sidesteps #16674.** Hang is correlated with the sharded-input → interleaved-cache writer path. Keeping cache=INTERLEAVED + feeding the writer from an INTERLEAVED input keeps us on the path `ttnn.kv_cache.update_cache_for_token_` exercises today (`grok_attention.py:225-226`).
2. **Maximizes Blackhole DRAM bandwidth.** Round-robin across all 8 controllers. Sharded-DRAM is *"rarely used due to limited DRAM vs NoC bandwidth"* (`memory_for_kernel_developers.md:299`).
3. **Fits trace capture.** No layout transitions in the autoregressive loop. `override_runtime_arguments` patches the in-tile byte offset every step, no recompile.
4. **Matches every reference KV cache.** `tt_transformers/tt/attention.py:428` uses exactly this. Don't deviate without a measured reason.

**New-token input** to our custom scatter: also INTERLEAVED to start (DRAM or L1 — L1 if the producer matmul can place output there cheaply). Move to sharded only if dispatch becomes the bottleneck *and* we have a workaround for #16674's class of failure. The hot path here is the reader untilizing a 32-row cache stripe, not the writer's one-row scatter.

If we later need sharded-input ergonomics, build it in our own program factory (controlled writer core-grid + semaphore init) — not by reusing `paged_update_cache`'s code path.
