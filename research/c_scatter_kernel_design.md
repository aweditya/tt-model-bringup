# In-place KV cache scatter kernel — design + research plan

## Why

`ttnn.scatter` at our production shape costs **0.32 ms × 32 calls/tok = 10.4 ms/tok** for the KV-cache slot write. Measured at 0.3% of P150 DRAM bandwidth — so the cost is **dispatch + the rewrite-whole-cache pattern**, not real work. Replacing this with a kernel that writes ONE tile in place should give a 30-60× speedup on this op, saving ~5-9 ms/tok (~2.5-4% of decode time at the current 210 ms/tok).

The same kernel also eliminates the 1.6 ms/tok of in-trace `ttnn.copy` from the C'4 v4 state-threading design. Combined savings: ~7-11 ms/tok = **3.5-5% decode improvement**, on top of all current optimizations.

## What

Single function, narrow scope:

```python
ttnn_local.in_place_scatter_kv(
    cache,    # ttnn.Tensor [1, N_KV, MAX_POS, HEAD_DIM] bf16, TILE_LAYOUT, DRAM
    src,      # ttnn.Tensor [1, N_KV, 1,       HEAD_DIM] bf16, TILE_LAYOUT, DRAM
    cur_pos,  # int (kernel arg, baked at dispatch time) OR int32 scalar tensor
) -> None     # mutates cache in place, returns nothing
```

**v1 constraints (deliberate):**

| Constraint | Why |
|---|---|
| `cur_pos` must be tile-aligned (multiple of 32) | First version doesn't deal with sub-tile partial writes |
| `bf16` only | bf8 needs shared-exponent recompute logic — defer |
| Single chip | No sharded/multi-chip variant |
| Fixed cache shape (this model only) | Hardcode `N_KV=4, MAX_POS=256, HEAD_DIM=256` then generalize later |
| `cur_pos` baked at dispatch time | Eliminates argument parsing complexity; we capture once per trace anyway |

The v1 is the **minimum interesting kernel**. Each constraint is removable in a follow-up if v1 lands.

## How it should work (math)

Cache layout in DRAM (TILE_LAYOUT, row-major over tiles):
- Tile shape: `[32, 32]`
- For our cache `[1, N_KV=4, MAX_POS=256, HEAD_DIM=256]`:
  - Per (batch=0, kv_head=i): 256 × 256 = 8 × 8 = **64 tiles**
  - Total tiles in cache: `4 × 64 = 256 tiles` = `256 × 32 × 32 × 2 bytes = 524 KB`

For `cur_pos = 32 * k` (tile-aligned, v1 constraint):
- Target tile row: `k` (zero-based, i.e., `cur_pos // 32`)
- Tile row contains positions `[k*32, k*32+32)`
- BUT we only want to overwrite ONE row (position `cur_pos`) within that tile — even in v1
- Wait — v1 constraint says cur_pos is tile-aligned, BUT cur_pos still picks ONE position in the tile (position cur_pos % 32 = 0, i.e., the first row of the tile)
- So we ARE doing a sub-tile write even in v1. The "tile-aligned" simplification just means cur_pos is at row 0 of a tile (not row 17, etc.)

**Simpler v1 scope (revised):** `cur_pos = 0, 32, 64, ..., 224`. Write `src` to row 0 of tile k = cur_pos/32. Other rows in the same tile must be preserved.

Per-(kv_head) work:
- Read tile holding `cur_pos`: 8 tiles wide × 1 tile high in tile-coords → actually **8 tiles** (one per HEAD_DIM tile slice)
- Modify row 0 of each
- Write back

For 4 kv_heads × 8 tiles = **32 tile reads + 32 tile writes**. Each tile is 32×32×2 = 2 KB. Total: 32 × 2 KB × 2 = 128 KB of DRAM I/O. At 200 GB/s effective: **0.64 µs** theoretical. Realistic with dispatch + L1 staging: ~5-10 µs. Vs current 320 µs → **30-64× speedup**.

If v1 fully relaxes (any cur_pos), the math is similar — just need to track which row WITHIN the tile to modify.

## Open questions (Phase 1 research must answer)

**Q1. Where does `update_cache` live in tt-metal?**
- Probably `tt-metal/ttnn/cpp/ttnn/operations/kv_cache/update_cache/` or similar
- This is the direct reference — has the exact pattern we need (just functional rather than in-place)

**Q2. Why does `paged_update_cache` hang on Blackhole?**
- GitHub issue #16674 — need to read it
- If it's a writer-NOC issue, our kernel might hit the same wall
- Workarounds: use BRISC for writes (slower path), or wait for tt-metal fix

**Q3. How do we register a custom ttnn op?**
- The op registration system (likely uses nanobind or pybind11)
- Probably need a `register.cpp` + `.hpp` declaration + CMake glue
- Read how `update_cache` is registered as our template

**Q4. Build system: do we need to rebuild ttnn or can we have a separate kernel library?**
- Option A: vendor our kernel into a fork of tt-metal — heavy
- Option B: ship as separate shared library that uses tt-metal's Program API — lighter, ideal for our use
- Option C: contribute upstream once it works — long-term goal

**Q5. What's the "Hello World" kernel example to start from?**
- tt-metal must have a tutorial or programming_examples dir
- Find the simplest "copy A to B" kernel as the starting template

**Q6. How do we run unit tests in-tree (without our model)?**
- We need a Python harness that calls the custom op + compares to numpy reference
- Pattern: `experiments/utils/scatter_inplace_kernel_test.py`
- Same pattern as `trace_state_thread_probe.py` etc.

**Q7. TILE_LAYOUT exact DRAM byte offset given (batch, kv_head, pos, head_dim)?**
- Need to confirm tile-row-major vs tile-col-major
- Need to know if tiles within a row are stored contiguously or interleaved
- The answer is in tt-metal's `tensor_impl.cpp` or similar

## Phased plan

### Phase 1: Research (3-5 days, no code)
1. Clone `tenstorrent/tt-metal` into `experiments/.refs/tt-metal/` (add to .gitignore)
2. Find + read:
   - `update_cache` op source (host dispatch + kernel)
   - `paged_update_cache` op source
   - Simplest "programming example" in `tt_metal/programming_examples/`
   - Op registration glue for one ttnn op
   - L1 buffer / circular buffer documentation
3. Read GitHub #16674 + any related discussion
4. Update this doc with answers to Q1-Q7
5. Decision point: proceed with v1 design, or escalate (ask tt team about #16674)

### Phase 2: Hello-world kernel (2 days)
1. Write the simplest possible custom kernel: `nop_kernel` that takes one tile and writes it back unchanged
2. Build infrastructure to compile + dispatch from Python
3. Unit test: input tile A → output tile A (identity)
4. Output: `experiments/kernels/nop_kernel/` (kernel.cpp + host.cpp + test.py + README.md)
5. **This proves we can ship custom kernels.** Everything else is incremental.

### Phase 3: tile-row-write kernel (3 days)
1. Modify nop_kernel: write `src` row 0 INTO a copy of `cache` tile, preserving other rows
2. Unit test: cache with non-zero rows, write src to row 0, check other rows unchanged
3. Output: `experiments/kernels/scatter_row_kernel/`

### Phase 4: KV scatter kernel v1 (3-5 days)
1. Extend to full cache shape (4 kv_heads × 8 head_dim tiles per kv_head)
2. Add position arithmetic to pick the right tile
3. Unit tests:
   - Write at pos=0, verify pos=0 has src, others 0
   - Sequential writes at pos 0, 32, 64 — check accumulation
   - Two writes at same pos — second overwrites first
   - At pos=224 (last tile row before MAX_POS edge case)
4. Output: `experiments/kernels/in_place_scatter_kv/`

### Phase 5: Integration (1 day)
1. Add `in_place_scatter_kv` as Python wrapper callable from `91f` kernels
2. Modify `gated_attn_step_ondevice` and `gated_attn_step_ondevice_traced` to use it instead of `ttnn.scatter`
3. Cosine validate vs eager
4. Bench against current implementation via persistent server

### Phase 6: Polish + upstream (1-2 weeks, post-validation)
1. Relax constraints: arbitrary cur_pos, bf8 support
2. PR to tt-metal
3. Address reviewer feedback

## Skeleton (illustrative — NOT real code)

```cpp
// kernel.cpp running on Tensix's NCRISC + BRISC trio
// pseudocode level only — real impl uses tt-metal's CB + NOC APIs

void kernel_main() {
    uint32_t cur_pos    = get_arg_val<uint32_t>(0);    // tile-aligned, runtime
    uint32_t cache_addr = get_arg_val<uint32_t>(1);    // DRAM base
    uint32_t src_addr   = get_arg_val<uint32_t>(2);    // DRAM base
    
    uint32_t target_tile_row = cur_pos / 32;
    
    // For each kv_head × head_dim_tile in our cache:
    for (int kv = 0; kv < N_KV; ++kv) {
        for (int hd_tile = 0; hd_tile < N_HD_TILES; ++hd_tile) {
            // Read 1 cache tile + 1 src tile into L1
            noc_async_read_tile(cache_tile_addr(kv, target_tile_row, hd_tile),
                                cache_l1_buffer);
            noc_async_read_tile(src_tile_addr(kv, hd_tile),
                                src_l1_buffer);
            noc_async_read_barrier();
            
            // Overwrite row 0 of cache_l1_buffer with src_l1_buffer's row 0
            // (single 64-byte memcpy in L1 — fast)
            memcpy(cache_l1_buffer.row(0), src_l1_buffer.row(0), HEAD_DIM_PER_TILE * 2);
            
            // Write back
            noc_async_write_tile(cache_l1_buffer,
                                  cache_tile_addr(kv, target_tile_row, hd_tile));
            noc_async_write_barrier();
        }
    }
}
```

The actual implementation has more nuance:
- Circular buffer management (allocating L1 staging space)
- Per-core work split (which Tensix core handles which (kv, hd_tile) pair)
- Compile-time vs runtime args
- BRISC vs NCRISC roles

## Test plan

`experiments/utils/scatter_inplace_kernel_test.py` (DRAM-only, no model needed):

```python
def test_write_at_zero():
    cache = alloc_zero([1, 4, 256, 256], bf16)
    src   = alloc_random([1, 4, 1, 256], bf16)
    in_place_scatter_kv(cache, src, cur_pos=0)
    # Check: cache[:, :, 0, :] == src[:, :, 0, :]
    # Check: cache[:, :, 1:, :] == 0

def test_sequential_writes():
    cache = alloc_zero(...)
    src_a = alloc_random(...)
    src_b = alloc_random(...)
    in_place_scatter_kv(cache, src_a, cur_pos=0)
    in_place_scatter_kv(cache, src_b, cur_pos=32)
    # Check both writes persist

def test_overwrite_same_position():
    # Verify second write completely replaces first

def test_max_pos_edge():
    # Verify pos=224 works (last tile row)

def test_bit_compare_vs_ttnn_scatter():
    # Run both, ensure identical output
```

All tests are **Python harnesses** that call the kernel — same pattern as our existing probes.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Blackhole writer hang affects our kernel too | Phase 1 reads #16674 first; if same bug, document + escalate to tt team |
| Build system more complex than expected | Phase 2's "nop_kernel" exists specifically to derisk the build path |
| Op registration in Python is fragile | Worst case we call via raw program dispatch (lower-level API), skip ttnn registration |
| TILE_LAYOUT offset arithmetic gets us | Phase 3 tests with cache full of known values — any offset bug shows up in the readback |
| Custom kernel slower than current scatter | Surprising but possible — fallback is to upstream the fix to ttnn.scatter itself |

## Why this fits the project

Per CLAUDE.md non-negotiables:
- **Plan first**: this doc + Phase 1 research before any C++ writing ✓
- **Research-driven**: Phase 1 is literally "read the existing kernels" ✓
- **No bloat**: smallest viable v1, generalization deferred ✓
- **Build a wiki**: each phase generates new memory entries / research docs as we learn the tt-metal internals
- **Learning by building**: this is the first kernel-level work in the project — opens the door to all future custom kernels (in-place writers for SSM/conv state, fused ops, etc.)

## What we don't do yet

- Don't write the kernel
- Don't pre-fork tt-metal
- Don't speculate on Phase 4 details before Phase 1 lands

The next concrete action is **Phase 1, step 1**: clone tt-metal locally, set up a reading session. That's a 1-hour task — `git clone`, browse the dir tree, find `update_cache`. Output: append to this doc.
