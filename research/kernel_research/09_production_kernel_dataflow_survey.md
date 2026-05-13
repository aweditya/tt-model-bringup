# 09 — Production Kernel Dataflow Survey

Empirical companion to the CUDA-vs-Tensix mental-model doc. Goal: take 6 real
tt-metal kernels apart, profile how each decomposes a problem onto the hardware,
then extract the recurring dataflow recipes. All line numbers cite the snapshot
under `experiments/.refs/tt-metal/`.

The 6 we tour:

| # | Kernel | Path (under tt-metal) | Why we picked it |
|---|--------|-----------------------|------------------|
| 1 | `eltwise_binary` (add) | `tt_metal/programming_examples/eltwise_binary/` | Single-core minimum-viable R/C/W pipe |
| 2 | `matmul_multi_core` | `tt_metal/programming_examples/matmul/matmul_multi_core/` | Workhorse, output-tile sharding |
| 3 | `update_cache` | `ttnn/cpp/ttnn/operations/kv_cache/` | Reference for our in-place scatter |
| 4 | `sdpa_decode` (non-paged) | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` | Flash-decode with tree reduce + multicast |
| 5 | `sdpa_decode` (paged) | same dir, `is_paged_attention=1` compile-time branch | Reader-side page-table indirection |
| 6 | `layernorm` sharded mcast | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/` | Cross-tile reduction via multicast + sem |

---

## 1. `eltwise_binary` — the baseline pipe

```
- name + path:   eltwise_binary / tt_metal/programming_examples/eltwise_binary/
- input shape:   2 DRAM-interleaved bf16 buffers of n_tiles tiles each (single core, n_tiles=64)
- compute:       FPU add_tiles (one tile per call, write dst reg, pack to cb_out)
- per-core split: NONE — runs only on core {0,0}; the example exists to teach the pipe
- CB layout:     cb_in0 (c_0, 2 tiles), cb_in1 (c_1, 2 tiles), cb_out (c_16, 2 tiles)
- Reader (RISCV_0): for i in 0..n_tiles: cb_reserve_back(in0,1); cb_reserve_back(in1,1);
                   noc_async_read in0[i] AND in1[i] back-to-back, single barrier, push both.
                   (kernels/dataflow/read_tiles.cpp:42-57)
- Compute (TRISCs): mm_init / binary_op_init_common + add_tiles_init once,
                    then per-tile: cb_wait_front(in0,1); cb_wait_front(in1,1);
                    tile_regs_acquire(); add_tiles(...); commit/wait;
                    cb_reserve_back(out,1); pack_tile(0, cb_out); push.
                    (kernels/compute/tiles_add.cpp:34-57)
- Writer (RISCV_1): cb_wait_front(out,1); noc_async_write_tile; barrier; pop.
                    (kernels/dataflow/write_tile.cpp:28-39)
- Per-core runtime args: only n_tiles + buffer addrs — there's nothing to "personalize".
- Key insight:   the CB is the entire synchronization mechanism. Reader pushes,
                 compute waits/pops/pushes, writer waits/pops. No locks, no fences,
                 no shared memory beyond the CB ring. The two-tile depth gives one
                 in-flight tile per stage — that's already double-buffering.
```

Note this kernel is **single-core** by design. It's the recipe that every other
kernel below uses, scaled across cores.

---

## 2. `matmul_multi_core` — output-tile sharding

```
- name + path:   matmul / tt_metal/programming_examples/matmul/matmul_multi_core/
- input shape:   A[Mt, Kt] tiles, B[Kt, Nt] tiles, all interleaved in DRAM, all bf16
- compute:       FPU matmul_tiles accumulating into a single dst reg across kt iterations
- per-core split: PURE OUTPUT-TILE LINEAR SPLIT.
                 split_work_to_cores(grid, Mt*Nt) -> (group_1, work_per_core_1, group_2, work_per_core_2).
                 Each core gets a contiguous range of output-tile IDs
                 [output_tile_start_id, output_tile_start_id + work_per_core).
                 (matmul_multi_core.cpp:121,217-243)
- CB layout:     cb_in0 (A tiles, 2-deep), cb_in1 (B tiles, 2-deep), cb_out (2-deep)
- Reader (RISCV_1): for each output tile owned by this core:
                    out_row = tile_id / Nt; out_col = tile_id % Nt;
                    for k in 0..Kt: read A[out_row, k], read B[k, out_col].
                    (reader_mm_output_tiles_partitioned.cpp:34-63)
- Compute (TRISCs): mm_init once. For each output tile:
                    tile_regs_acquire() (zeros the dst!), for kt in 0..Kt:
                       cb_wait_front(in0/in1,1); matmul_tiles(in0,in1,0,0,0); pop both.
                    pack_tile(0, cb_out); push.
                    (mm.cpp:51-81)
                    Note: `matmul_tiles` ACCUMULATES into dst — the K loop is just
                    the natural tile-MAC sequence, no explicit accumulator.
- Writer (RISCV_0): linear: write owned output tiles to DRAM in start_id..start_id+work_per_core.
- Per-core runtime args: (Mt, Kt, Nt, output_tile_start_id, work_per_core).
                         Everyone gets the same buffer addrs; "personality" is just
                         the start-id and count.
- Key insight:   ONE reader, ONE compute, ONE writer kernel binary — same code on
                 every core — and the entire core diversity collapses into two
                 runtime ints (start_id + count). No A-reuse, no B-reuse, no
                 systolic anything; this is the naive split. Reuse appears in
                 `matmul_multicore_reuse_mcast` via NoC multicast of A across rows.
```

Confirmed by reading the host code: A and B are re-read from DRAM by every core
that needs them — that's intentional for the teaching example. The mcast variant
is the production code; we didn't tour it.

---

## 3. `update_cache` — the reference we copy from

```
- name + path:   ttnn/cpp/ttnn/operations/kv_cache/  (kernels under device/kernels/{dataflow,compute}/)
- input shape:   cache  (B, num_heads, max_seq, head_dim)  tiled
                 input  (1, num_heads, padded_NH, head_dim) where padded_NH = B padded to TILE_HEIGHT
                 update_idxs: scalar OR per-batch vector (in another path)
- compute:       NO MATH. It's untilize -> writer-mutate-block -> tilize. The "compute"
                 kernel is purely the unpacker/packer dancing the data through the FPU's
                 tilize/untilize engines so the writer can do row-aligned NoC writes.
- per-core split: split_work_to_cores over num_batched_heads (=heads * batches / TILE_HEIGHT).
                 (update_cache_multi_core_program_factory.cpp:100)
                 Each core handles a contiguous group of batched-head rows;
                 cache_start_id is computed from batch + head + cache_tile_idx
                 (line 219-238).
- CB layout:     input_cb (c_0), cache_cb (c_1), untilized_cache (c_24),
                 untilized_cache2 (c_25), untilized_input (c_26), out_cb (c_16).
                 Sizes are `2*granularity*Wt` => double-buffered with granularity=2.
- Reader (BRISC/NCRISC): reader_update_cache_interleaved_start_id.cpp:43-74
                 outer h in num_batched_heads: read Wt input tiles into input_cb;
                   inner u in u_count, g in granularity:
                     read a Wt-tile slab from CACHE (read-modify-write source)
                     into cache_cb, advance by cache_batch_num_tiles per row.
                 Note the carefully threaded `(cache_id, b)` so cache_id walks
                 batch-then-head order (line 63-68).
- Compute (TRISCs): update_cache.cpp:28-46
                 For each batched head: untilize the input row once;
                   for each granularity step: untilize cache slab, tilize updated slab.
                 The CB-only sync makes this implicitly cooperate with the writer.
- Writer (BRISC/NCRISC): writer_update_cache_interleaved_start_id.cpp:37-78
                 wait_front(untilized_input_cb, Wt); for u, g:
                   wait on untilized cache slab; noc_async_read FROM input TO cache's
                   L1 buffer at `cache_l1_write_addr + offset`  (the in-place row write);
                   push to untilized_cache2_cb so compute knows it can tilize.
                   Then wait for the tilized result and noc_async_write_tile back to DRAM.
                 ONE barrier at the end (line 77) — write_flushed inside the loop is enough.
- Per-core runtime args: cache_addr, input_addr, Wt, B, num_batched_heads_for_this_core,
                         cache_total_num_tiles, cache_batch_num_tiles, cache_head_num_tiles,
                         cache_start_id, input_start_id, batch_start_id,
                         + writer-only Wbytes, offset, batch_read_offset.
- Key insight:   This kernel does NO arithmetic — the "compute kernel" only runs
                 the tilize/untilize datapath. Mutation lives in the WRITER, which
                 does a NoC read from input-CB into the L1 buffer that the compute
                 will then tilize. That makes update_cache fundamentally a
                 reader/writer-bound op, and explains why our planned in-place
                 scatter doesn't need a real compute kernel either.
```

Important and underlined: compute and writer share an L1 region via two CBs
(`untilized_cache_cb` and `untilized_cache2_cb`) that act as a request/release
hand-off. The writer pops the "untilized cache slab" CB, mutates the row, then
pushes the "updated slab" CB for compute to tilize. That's a producer-consumer
dependency where the producer IS the writer, not the compute.

---

## 4. `sdpa_decode` (non-paged path) — flash attention with tree reduce

```
- name + path:   ttnn/cpp/ttnn/operations/transformer/sdpa_decode/
- input shape:   Q [B, num_q_heads_padded_to_tile, head_dim]   (single seq token, decode)
                 K, V [B_kv, num_kv_heads, max_seq, head_dim]
                 mask, attention_sink optional, cur_pos optional
- compute:       Standard flash-attention math:
                 QK matmul -> mask -> reduce_max -> sub_max -> exp -> reduce_sum ->
                 PV matmul -> accumulator running max/sum (Welford-ish online softmax).
                 (sdpa_flash_decode.cpp)
- per-core split: TWO-LEVEL.
                 1. num_cores_per_batch = num_cores / B  (per-batch core group)
                 2. within a batch group, num_cores_per_head + num_heads_per_core
                    (program_factory.cpp:168-177)
                 3. Within ONE head, K-sequence-length is sharded across cores by
                    chunk: get_workload_for_core(cur_pos, ..., core_num_in_reduce, ...)
                    returns (k_chunk_start, k_chunk_end) for each core
                    (rt_args_common.hpp:35-93).
                 4. One designated output core per batch collects partial (M, L, O)
                    triples via a TREE REDUCE (`num_tree_reduction_rounds` =
                    ceil(log2(num_cores_per_head))).
- CB layout:     20+ CBs. The interesting ones:
                 c_0 Q  | c_1 K  | c_2 V  | c_3 mask
                 c_6 cb_m_in (incoming partial max), c_7 cb_l_in (incoming partial sum)
                 c_24 qk_im, c_25 out_im, c_26 out_accumulate_im (running output)
                 c_27/c_28 cb_max_1/max_2 (running max double-buffered)
                 c_29/c_30 cb_sum_1/sum_2 (running sum)
                 c_31 cb_exp_max_diff (the `exp(prev_max - new_max)` rescale factor)
                 c_20 cb_out_final (only on output cores)
- Reader (RISCV_0): reader_decode_all.cpp:243-352
                 Reads Q once (entire q_chunk_tiles into c_0/c_10).
                 For each head this core owns, for each k_chunk in [k_chunk_start, k_chunk_end):
                   read K-chunk tiles into c_1 (Sk_chunk_t * DHt tiles)
                   read mask chunk into c_3 (if causal/sliding)
                   read V-chunk into c_2.
                 Supports OPTIONAL k_mcast (vertical NoC multicast across cores
                 that share the same K column — see `do_k_mcast` runtime arg).
- Compute (TRISCs): sdpa_flash_decode.cpp:108-130 — many runtime args:
                 do_reduce, do_output, cur_head, cur_batch, core_num_in_reduce,
                 core_num_in_output, is_tree_root, parent_core_in_group,
                 send_at_round, num_children, my_active_rounds,
                 children_per_round[6].
                 The compute kernel itself is the "math" core of flash decode:
                 QK matmul -> online softmax rescale -> PV matmul, running max/sum,
                 and at the end either send partials toward the tree root or do the
                 final divide.
- Writer (RISCV_1): writer_decode_all.cpp. On non-output cores it writes the partial
                 (O, M, L) triple to the PARENT CORE'S L1 via noc_async_write
                 + a semaphore. On the output core it writes the final attention
                 result to the output buffer in DRAM.
- Per-core runtime args: do_reduce, do_output, cur_head, cur_batch,
                 core_num_in_reduce, core_num_in_output, cur_pos_arg,
                 tree-reduction descriptor, plus the all_output_noc_x[] / y[]
                 arrays so any core can find any output core
                 (reader_decode_all.cpp:145-150).
- Key insight:   The K-chunk split lives in `get_workload_for_core`
                 (rt_args_common.hpp:35-93). Each core computes its own
                 [k_chunk_start, k_chunk_end), runs its slice of flash attention,
                 produces a partial (max, sum, output), then ships it L1->L1 to a
                 parent core in a TREE pattern, not an all-reduce. The compute
                 kernel never reads cur_pos directly — the reader stages it via
                 a CB (c_8) shared with compute (lines 100-112).
                 The MAX_POS≥1024 cliff our team is hitting is because Sk_chunk_t
                 (line 4) sets CB allocations; large cur_pos with small Sk_chunk_t
                 means many chunks and the running output CBs (c_26..c_31) overflow L1.
```

The L1 reservation for K/V/intermediate CBs scales as `Sk_chunk_t * (DHt + vDHt)
+ 4*Sq_chunk_t*vDHt` per core. That's the wall.

---

## 5. `sdpa_decode` (paged path) — same kernel, page-table indirection in the reader

The paged variant is the SAME source file, gated on `is_paged_attention =
get_compile_time_arg_val(11) == 1` (reader_decode_all.cpp:29). The compute
kernel is byte-for-byte identical between paged and non-paged.

```
- name + path:   same files as #4. is_paged_attention=1 path
- input shape:   K, V stored as PAGE BLOCKS in DRAM:
                 K_phys [num_blocks, num_kv_heads, block_size_t, DHt]
                 page_table[B, max_pages]: virtual-block -> physical-block ID (u16 or u32)
- compute:       UNCHANGED — same flash-decode math, doesn't know about pages.
- per-core split: Identical to non-paged (chunk-level over K seq, tree reduce).
- CB layout:     Same plus c_9 cb_id_page_table (stores the page-table row for this batch).
- Reader (RISCV_0): The only divergence. Three extras:
                 a) Read the page table row for cur_batch into c_9:
                    `page_table_ptr = read_page_table_for_batch(...)` (line 228).
                    For sharded page tables it's already resident in L1, just take a
                    pointer (lines 235-239).
                 b) Replace linear K/V tile IDs with a page-table lookup:
                    For each k tile in a chunk, compute virtual_seq_tile_id,
                    look up physical block via the page table:
                        virtual_block = seq_tile_idx / block_size_t
                        physical_block = page_table_ptr[virtual_block]
                        physical_tile_id = physical_block * block_stride
                                         + head_offset
                                         + (seq_tile_idx % block_size_t) * Wt
                    (sdpa/device/kernels/dataflow/dataflow_common.hpp:45-58)
                 c) If block_size < TILE_HEIGHT, the compute kernel generates a
                    block-padding mask (sdpa_decode/dataflow_common.hpp:362-377)
                    that the QK softmax uses to ignore the padded slots inside a
                    tile.
- Writer:        Unchanged.
- Per-core runtime args: page_table_addr, page_table_page_size (extra two args).
- Key insight:   The page table is JUST a tile-id translation layer in the reader.
                 Compute pulls K/V from cb_k_in/cb_v_in tile-by-tile; it has no
                 notion of pages. This is the universal pattern: "compute is
                 page-blind, reader is page-aware." We can copy this for our
                 scatter kernel — page indirection happens in the dataflow kernel,
                 never in compute.
                 The `block_size_t < TILE_HEIGHT` case (e.g. block_size=16 with
                 tile=32) is handled by a separately generated padding mask, NOT
                 by changing reader logic. That's the design escape hatch for
                 non-tile-aligned page sizes.
```

---

## 6. `layernorm` sharded multicast — cross-tile reduction

```
- name + path:   ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/
- input shape:   Activations sharded by H across cores (e.g. each core owns block_h
                 tile rows). Width is full (one core sees full row to reduce).
- compute:       Welford or two-pass mean/variance + normalize + gamma/beta.
                 layernorm_sharded.cpp does the full pipeline.
- per-core split: BY ROW-SHARD. Each core owns block_h tile-rows of the input. One
                 designated "sender" core per row-group coordinates the cross-core
                 reduction; the rest are "receivers". Two-stage reduce supported
                 for very wide grids (intra-row then inter-row).
- CB layout (the reduction-specific ones):
                 cb_ex_partial   : per-core partial E[x] tiles (compute output)
                 cb_ex_external  : where SENDER stores partials READ from peers
                 cb_ex           : per-core combined E[x] after summing externals
                 cb_ex_global    : sender's final mcast buffer with all rows' E[x]
                 (and the matching _partial2/ex2 for variance)
- Reader (sender, RISCV_0): reader_mcast_sender_unary_sharded_ln.cpp:106-287
                 1. Wait for own partial reduce (compute writes cb_ex_partial).
                 2. Set reduce_sender_sem to VALID (multicast: "I'm ready").
                 3. Wait on reduce_receiver_sem == num_blocks-1 (all peers ready).
                 4. NoC-read each peer's L1 cb_ex_partial INTO cb_ex_external,
                    one chunk per peer (lines 162-176).
                 5. Push cb_ex_external -> compute reduces locally -> cb_ex.
                 6. Wait for ALL senders' cb_ex to finish (semaphore sync).
                 7. Read every sender's cb_ex region into cb_ex_global.
                 8. async_write_multicast cb_ex_global to all peers' L1 + set
                    sem on completion (lines 263-281).
- Reader (receiver): writes its partial to L1, sets reduce_receiver_sem, waits for
                 the sender's multicast.
- Compute: sees cb_ex_external (raw remote tiles) and runs reduce_init/reduce_tile
                 to combine them locally. Compute NEVER knows about NoC; it just
                 sees CBs full of tiles.
- Writer: writes normalized output back to its L1 shard.
- Per-core runtime args: which mcast group is this core in, mcast_dest_noc rect,
                 the in0_remote_noc_x/y arrays so it knows where peers are,
                 sender-vs-receiver flag.
- Key insight:   The semaphore + multicast dance lives ENTIRELY in the reader/
                 writer. Compute treats the cross-core reduction as "more tiles
                 showing up in cb_ex_external" — same `reduce_tile` API as a
                 within-core reduce. This is the reusable pattern for any
                 collective: aggregate into an "external" CB via NoC reads, then
                 reduce locally; the broadcaster does an async_write_multicast +
                 semaphore-set in a single hop.
```

---

## Synthesis

### Five recurring dataflow patterns

1. **CB-as-the-only-sync.** From the 60-line eltwise example to the 660-line
   flash-decode compute, the protocol between R/C/W is identical:
   `cb_reserve_back -> push_back -> wait_front -> pop_front`. There are no
   atomics, no fences inside the kernel, no shared memory beyond CBs. Every
   "queue" you'd reach for in CUDA is a CB.

2. **One kernel binary per role, per-core diversity is runtime args.** Across
   matmul, update_cache, sdpa_decode: each core runs the SAME compiled reader
   binary, the SAME compute, the SAME writer. The "personality" of a core is a
   handful of runtime integers — `(start_id, count)` for matmul; `(cache_start_id,
   batch_start_id, num_batched_heads)` for update_cache; `(cur_head, cur_batch,
   core_num_in_reduce, is_tree_root)` for sdpa. CUDA's "every thread runs the
   same code, behavior diverges on threadIdx" maps almost 1:1, except the
   per-thread index lives in runtime args, not a hardware register.

3. **Linear output-tile split is the default.** `split_work_to_cores(grid,
   total_output_tiles)` shows up in matmul (`matmul_multi_core.cpp:121`) AND
   update_cache (`update_cache_multi_core_program_factory.cpp:100`). It returns
   two groups so non-evenly-divisible workloads still load-balance to within
   one tile per core. Multi-axis splits (sdpa's batch x kv-head x k-chunk) are
   the exception, only when one axis isn't enough parallelism.

4. **Reader/writer do all NoC, compute does only math.** No production compute
   kernel touches `noc_async_read`. update_cache is the cleanest example: the
   compute kernel is 46 lines of pure tilize/untilize, while the writer
   contains the actual in-place mutation (`noc_async_read` from input-CB into
   cache-CB at a row offset — writer_update_cache_interleaved_start_id.cpp:49).
   This separation is what lets paged sdpa_decode reuse the non-paged compute
   kernel byte-for-byte.

5. **Granularity > 1 is the cheap perf trick.** Both update_cache (granularity=2,
   `update_cache_multi_core_program_factory.cpp:62` — "granularity = 2 best for
   performance") and matmul (2-tile CBs) deliberately double-up the CB depth so
   reader fills tile N+1 while compute eats tile N. The reader/compute/writer
   triplet on three different RISC cores is already pipelined; bumping CB depth
   to ≥2 hides one stage of latency for free.

### Three anti-patterns (production never does these)

1. **Compute doing DRAM I/O.** Never. Across all 6 kernels, `noc_async_read`
   and `noc_async_write_tile` appear exclusively in BRISC/NCRISC dataflow
   kernels. If your compute kernel needs data, the answer is "add a CB" — not
   "read it yourself."

2. **Per-core kernel SOURCE files.** No host file picks a different `.cpp` per
   core. `CreateKernel(..., all_cores, ...)` is the rule. Even sdpa_decode,
   which has wildly different per-core behavior (worker vs reducer vs output
   core, tree-root vs leaf), uses ONE compute source — diversity comes from
   runtime args + early-return branches (e.g. reader_decode_all.cpp:87,
   `if (q_addr == 0) return;` is the "idle core" path).

3. **Free-standing semaphores without a CB pair.** Every cross-core signal we
   saw is paired with a CB/L1 region whose readiness it advertises. Bare
   semaphores would mean a separate L1 layout convention; the production code
   makes every signal point at a CB read/write pointer, which keeps the
   producer/consumer contract local to one file. The receiver semaphore in
   layernorm (reader_mcast_sender:126,215) always gates a corresponding
   `cb_ex_external` or `cb_ex_global` consume.

### Three most reusable building blocks for our in-place KV-scatter kernel

1. **The update_cache reader/writer skeleton** is already 90% of what we want.
   Same input/cache CB pair (c_0, c_1), same untilize-mutate-tilize dance.
   Our scatter only needs to (a) replace the linear `cache_id` walk with a
   page-table lookup (steal `virtual_seq_tile_id_to_physical_tile_id` from
   `sdpa/device/kernels/dataflow/dataflow_common.hpp:45-58`), and (b) drop the
   "input read by batch" assumption — for our case it's "scatter row r at
   page_table[r]".

2. **Reader-side page-table indirection.** The paged sdpa pattern is the
   blueprint: load the page table into a dedicated CB once (`c_9`, lines
   222-241), then translate every tile id at read time. The compute kernel
   stays paged-blind. For our scatter, the WRITER is the paged actor — it
   writes mutated rows to `physical_block * block_stride + ...`, same formula.

3. **split_work_to_cores over num_batched_heads.** Lifts directly from
   update_cache. Output-tile linear partition, two core groups for residual,
   per-core args = (start_batched_head_idx, num_batched_heads). Don't reinvent
   the split.

---

### What surprised us

The paged variant of sdpa_decode and the non-paged variant share **the same
compute kernel binary, byte-for-byte.** The entire paging feature lives in
about 40 lines of the reader (the page-table CB load and the tile-id
translation in `virtual_seq_tile_id_to_physical_tile_id`). That is the strongest
argument we've seen for the "reader does DMA + indirection, compute is
data-layout-agnostic" doctrine — and it tells us our scatter kernel can be
hosted on the same compute binary as a non-paged in-place update, with paging
gated entirely in the writer/reader.
