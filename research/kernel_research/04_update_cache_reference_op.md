# 04 — `update_cache` / `paged_update_cache`: Reference Op Deep-Dive

Reference reading for our planned in-place KV-cache scatter kernel. Everything below cites the cloned tt-metal at
`experiments/.refs/tt-metal/` (line numbers are from that snapshot, not GitHub `main`).

---

## 1. File-tree map

### `update_cache` (stable, non-paged)
Lives under `ttnn/cpp/ttnn/operations/kv_cache/`:

| Layer | Path |
|------|------|
| User-facing wrapper | `kv_cache.hpp`, `kv_cache.cpp` |
| Python nanobind | `kv_cache_nanobind.cpp` |
| Device op + validation | `device/update_cache_device_operation.{hpp,cpp}` |
| Program factory (host) | `device/update_cache_multi_core_program_factory.{hpp,cpp}` |
| Reader kernel | `device/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp` |
| Writer kernel | `device/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp` |
| Compute kernel | `device/kernels/compute/update_cache.cpp` |

### `paged_update_cache` (experimental)
Lives under `ttnn/cpp/ttnn/operations/experimental/paged_cache/`:

| Layer | Path |
|------|------|
| User-facing wrapper | `paged_cache.{hpp,cpp}` |
| Device op | `device/update_cache/paged_update_cache_device_operation.{hpp,cpp}` |
| Program factory | `device/update_cache/paged_update_cache_program_factory.{hpp,cpp}` |
| Reader / Writer / Compute | `device/kernels/dataflow/{reader,writer}_update_cache_interleaved_start_id.cpp`, `device/kernels/compute/update_cache.cpp` (note: same filenames as non-paged but **different files** under the experimental dir) |

---

## 2. Host-side walk (`update_cache`)

User entry point (`kv_cache.cpp:31-41`) just forwards into the device primitive:

```cpp
ttnn::prim::update_cache(
    cache, input, 0, update_idx, batch_offset, ttnn::prim::UpdateCacheOpType::UPDATE, kernel_config_val);
return cache;
```

Note the return value is `cache` — the op is **in-place semantically**: the cache buffer is both input and output. There's no separate output tensor (`update_cache_multi_core_program_factory.cpp:20`: `Tensor& /*output_tensor*/`).

`UpdateCacheMultiCoreProgramFactory::create` (`update_cache_multi_core_program_factory.cpp:19-286`) does:

1. **Tile arithmetic setup** (lines 44-65):
   ```cpp
   uint32_t Wt = cache_tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
   uint32_t Wbytes = fp32_dest_acc_en ? ... : cache.padded_shape()[-1] * sizeof(::bfloat16);
   uint32_t cache_total_num_tiles = cache_tensor.physical_volume() / TILE_HW;
   uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache.padded_shape()[0];
   uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache.padded_shape()[1];
   uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
   ```
2. **Work split** across cores by `num_batched_heads = B * num_heads / TILE_HEIGHT` (line 63, lines 91-102).
3. **Five CBs** allocated (lines 103-148): `c_0` raw cache tiles, `c_1` raw input tiles, `c_24`/`c_25` untilized intermediates (double-buffered), `c_26` second untilized buffer, `c_16` output (re-tilized) tiles. Double-buffering uses `granularity = 2`.
4. **Three kernels** registered: reader on `NCRISC`, writer on `BRISC`, compute on the Tensix triplet.
5. **Per-core runtime args** (lines 218-271): pre-computes a per-core `cache_start_id = batch_offset_tiles + head_offset_tiles + (update_idx / TILE_HEIGHT) * Wt`. `override_runtime_arguments` (lines 288-331) lets `update_idx` change between calls without re-compiling the program — only `cache_start_id` and the within-tile byte offset are patched.

---

## 3. Kernel-side walk

The op is **read–modify–write at tile granularity** because a tile is 32×W and `cur_pos` usually lands inside an existing tile.

### Reader (`reader_update_cache_interleaved_start_id.cpp`)
For each batched head:
- Reads `Wt` tiles of input from the input tensor into `c_1`.
- Reads `granularity * Wt` tiles of the existing cache slice (one row of tiles per batch entry) into `c_0`.
  ```cpp
  for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
      noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
      cache_l1_write_addr += cache_tile_bytes;
  }
  cache_id += cache_batch_num_tiles;
  ```

### Compute (`compute/update_cache.cpp`)
Pure layout shuffling — **no math**. For each batched head:
- `untilize` the input tile-row into `c_25` (the "untilized input" CB).
- For each granularity step: `untilize` the cache tile-row from `c_0` into `c_24`, **wait for the writer to overwrite the target row inside the untilized buffer**, then `tilize` the modified buffer (from `c_26`) into `c_16`.

This is the key insight: the writer kernel performs the actual scatter **on the untilized data sitting in L1**, between the compute-untilize and compute-tilize steps.

### Writer (`writer_update_cache_interleaved_start_id.cpp`)
The scatter happens at lines 46-53:
```cpp
cb_wait_front(untilized_cache_cb_id, Wt);
cb_reserve_back(untilized_cache2_cb_id, Wt);
uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + offset;
noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
```
That `offset = update_idx % TILE_HEIGHT * Wbytes` lands the input row at the correct sub-tile row. After the compute re-tilizes, the writer pushes the modified tile back to DRAM:
```cpp
for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
    noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
    ...
}
```
The DRAM tile that was read at the start of the step is the same DRAM tile written at the end — **in-place at tile granularity**.

---

## 4. Tile arithmetic for `cur_pos`

Given a TILE layout cache with shape `[B, num_heads, max_seq, head_dim]`:
- `TILE_HEIGHT = TILE_WIDTH = 32`, `Wt = head_dim / 32`, `St = max_seq / 32`.
- The tile containing token `cur_pos` is row `cur_pos / 32`.
- **Tile index offset within a head**: `(cur_pos / TILE_HEIGHT) * Wt`.
- **Within-tile byte offset**: `(cur_pos % TILE_HEIGHT) * Wbytes` — used by the writer to position the scatter inside the untilized L1 buffer.

From `update_cache_multi_core_program_factory.cpp:64,218`:
```cpp
uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
uint32_t cache_tile_idx     = update_idx / tt::constants::TILE_HEIGHT * Wt;
```

`override_runtime_arguments` (lines 303-304) re-derives the same two values from a new `update_idx` and patches per-core runtime args — no recompile.

---

## 5. `paged_update_cache` — what changes

Three additions, all in the reader (`experimental/.../reader_update_cache_interleaved_start_id.cpp:59-100`) and writer (`experimental/.../writer_update_cache_interleaved_start_id.cpp:54-84`):

1. **`update_idxs_tensor`**: a device-side INT32 tensor of per-batch positions (instead of a host-side `vector<uint32_t>`). The kernel reads its own `update_idx = index_ptr[my_batch_idx]`. **This is what makes it traceable** — `cur_pos` no longer needs to be a Python scalar baked into the program.
2. **`page_table`**: optional ROW_MAJOR table mapping virtual block IDs to physical block IDs. When present, `cache_id` is derived via:
   ```cpp
   virtual_block_id  = update_idx / block_size;
   physical_block_id = page_table_ptr[virtual_block_id];
   block_start_id    = physical_block_id * num_heads * block_size_t * Wt;
   block_offset      = ((update_idx % block_size) / TILE_HEIGHT) * Wt;
   cache_id          = block_start_id + block_offset;
   ```
3. **`update_idx == -1` sentinel** (writer, line 60): per-batch skip. Lets a user opt out of an update for some batch lanes.

The paged variant also **requires the input tensor to be sharded** (`paged_update_cache_device_operation.cpp:160`: `TT_FATAL(input_tensor.is_sharded(), ...)`). One core per batch lane. The cache itself must be `INTERLEAVED` (line 51: `"Only interleaved cache is supported"`).

The fundamental tile-RMW pipeline is identical to non-paged — same five CBs, same untilize/scatter/tilize/write cycle.

---

## 6. Issue #16674 — Blackhole `paged_update_cache` hang

**Confirmed from the GitHub issue (`tenstorrent/tt-metal#16674`)**:
- Title: *"Blackhole: ttnn.experimental.paged_update_cache consistently hanging"*.
- The `test_paged_update_cache_decode` test "consistently locks up the machine after device initialization and cache index logging".
- Lockup is parameter-independent: tested `block_size=32`, `head_dim=128`, `seq_len=32`, users `1024`, `heads {1, 8}`, bf16 & bf8_b, `cur_pos ∈ {0, 1, 127, 1057}`. Reporter: *"the lockup seems pretty independent of parameters."*
- Labels: P1, blackhole, bug. Assigned to `cglagovichTT`. Linked to Llama bringup #16013.
- Issue is **closed** (resolved on GitHub).

**Not confirmed from the issue** (root cause was not articulated in the issue body):
- No identified cause, no quoted workaround, no specific PR link surfaced via WebFetch.

**From the cloned repo**: `grep` for `"16674"` across the entire tt-metal tree (sources, tests, docs, commit messages reachable via blob content) returned **zero hits**. No comment or `// FIXME(#16674)` marker is present in this snapshot. Two `TODO ... hangs ... long context` comments exist in `models/tt_transformers/demo/simple_text_demo.py:548` and `models/demos/multimodal/gemma3/demo/text_demo.py:493` but they reference paged *attention*, not paged_update_cache, and don't cite #16674.

**Inferred** (from reading the writer kernel + our prior MEMORY note `paged_sdpa_decode_works_at_32k`):
- The writer-side `noc_async_write_tile` loop over an `INTERLEAVED` cache from a `SHARDED` input is the natural candidate for a Blackhole NoC dispatch hang. Our earlier experiments showed paged SDPA-decode worked at 32k once we put the writer into a sharded memory config — the same shape of fix likely applies to `paged_update_cache`.
- The fact that the lockup is parameter-independent points at a structural issue (e.g., NoC arbitration / semaphore-init order across all cores) rather than a numerical-tiling bug.

Treat the inference as a hypothesis; only the WebFetch summary and the file paths above are direct evidence.

---

## 7. Implications for our custom in-place scatter

What **carries over** from `update_cache`:
- Tile-arithmetic split: `tile_idx = cur_pos / 32`, `within_tile_byte = (cur_pos % 32) * Wbytes`. We need both.
- Read–modify–write is mandatory because the scatter target is a single row inside a 32-row tile. There's no "edit-in-place at sub-tile granularity" path — you must untilize, overwrite one row, retilize.
- The reader→compute (untilize) → writer (scatter) → compute (tilize) → writer (write-back) pipeline is the proven recipe.
- For traceability, the position must enter via a **device-side INT32 tensor** (paged variant's `update_idxs_tensor`), not a Python int.
- `override_runtime_arguments` is how you keep the program-cache hot while changing `cur_pos` between decode steps (`update_cache_multi_core_program_factory.cpp:288-331`).

What's **different** for our case (single-position scatter into our own buffer, no batch/head fan-out):
- We don't need the `num_batched_heads` work split — one core (or a tiny grid) is enough for a single token write.
- We can skip the `c_26` second-intermediate CB and the `granularity` double-buffering if we don't pipeline successive heads.
- No `share_cache`, no `page_table` — single contiguous logical cache.
- **Same tile read and written** — we don't need a distinct output tensor; the `Tensor& /*output_tensor*/` pattern in the program factory carries over cleanly.
- We must decide whether to read the input from `INTERLEAVED` (cheaper to produce upstream) or `SHARDED` (matches `paged_update_cache` but adds shard-spec friction). Given #16674 may be the sharded path, the **INTERLEAVED writer is probably the safer starting point**.

**Bottom line on #16674**: this looks workaround-able, not a fundamental wall. We already have an in-memory note that paged SDPA-decode works at 32k once the writer goes sharded; the inverse may apply here (keep cache INTERLEAVED, avoid the sharded-writer path that exhibits the hang). Worth a small Phase 0 experiment before committing to a layout.
