# 08 — Tensix vs CUDA: A Mental Model for Kernel Designers

**Audience**: anyone on tt-model-bringup who has written CUDA and now needs to think in Tensix. This is the conceptual doc — what to internalize before designing a custom op, not an API reference.

**Stance**: empirical, not aspirational. Tensix is a different machine, not a different syntax for the same machine.

**Sources** (local, plus Marty's blog):
- `tt_docs_corpus/.../advanced_topics/compute_engines_and_dataflow_within_tensix.md`
- `tt_docs_corpus/.../advanced_topics/memory_for_kernel_developers.md`
- `tt_docs_corpus/.../advanced_topics/tiles.md`
- `tt_docs_corpus/.../examples/eltwise_binary.md`, `examples/matmul_multi_core.md`
- `research/kernel_research/01_tensix_architecture_primer.md`, `03_hello_world_kernel_walkthrough.md`, `05_memory_configs_deep_dive.md`, `06_trace_capture_internals.md`
- Marty (clehaxze): `2025/10-26-building-the-rope-operation-for-tensorrent-hardware.gmi`, `2025/04-21-programming-tensotrrent-processors.gmi`

---

## 1. The unit of work

The first reflex to retrain. In CUDA you start by asking "how many threads?" In Tensix you start by asking "how many tiles, on how many cores, and which RISC-V does what?"

| Layer | CUDA | Tensix (Blackhole P150) |
| --- | --- | --- |
| Atomic compute unit | scalar lane in a thread | **32×32 tile** (one math instruction operates on a whole tile) |
| Programmable lockstep group | warp = 32 SIMT threads | SFPU `LReg` = 32 elements; FPU works on full tiles |
| Programmable independent group | block / CTA (cooperative) | **Tensix core** (~100 per chip; Blackhole P150 exposes 130) |
| Top-level scheduling | grid (oversubscribed; HW scheduler swaps blocks) | **CoreRangeSet** (static; one program slot per core) |
| Programs per "block" | 1 | **5 — one per RISC-V** (2 dataflow + 3 compute, MIMD) |

Read those last two rows again. CUDA gives you one program over many threads (SPMD). A Tensix core runs **five concurrent programs on five RISC-V cores**, each from a different `.cpp`. From `compute_engines_and_dataflow_within_tensix.md:45`:

> "Although a compute kernel is written as a single piece of code, it is compiled into three separate binaries, each running on a different RISC-V core (T0-2) within the Tensix."

Plus BRISC and NCRISC (the two data-movement RISC-Vs), and you have a five-binary core. The FPU, SFPU, unpacker, and packer are not cores — `compute_engines_and_dataflow_within_tensix.md:47`:

> "Furthermore, the unpacker, packer, SFPU, and FPU are not processing cores and cannot make control flow decisions on their own. The RISC-V cores issue commands to the compute engines and manage data flow between them."

Across the chip, scheduling is **static**. `matmul_multi_core.md:155-157`:

> "In contrast, Metalium's parallelism model is static. The number of parallel tasks you can launch is limited to the number of available Tensix cores on the device. Each core is assigned a specific portion of the work at launch, and there is no dynamic scheduling or oversubscription: once a core finishes its assigned work, it remains idle until the next task is launched."

No latency hiding by oversubscription. You hide latency yourself, **inside one core**, with the reader/compute/writer trinity and circular buffers.

## 2. The compute primitive

| | CUDA | Tensix |
| --- | --- | --- |
| Elementwise unit | thread executes scalar op | `add_binary_tile`, `sigmoid_tile`, etc. — **tile-level SFPU op** |
| Matmul unit | warp/MMA via mma.sync (16×16 etc.) | `matmul_tiles(cb_a, cb_b, …, dst_idx)` — **one instruction per 32×32 tile-pair on the FPU** |
| Where ops land | register file | `Dst` register set (8 or 16 tile slots; 16-bit or 32-bit storage) |
| Operand staging | load from shared/global into registers | unpacker reads CBs into `SrcA`/`SrcB`/`Dst`; packer writes `Dst` to a CB |

A few things to internalize:
- **The FPU and the SFPU are separate engines** with different strengths. FPU = matmul, broadcast-add, accumulate. SFPU = transcendentals, comparisons, masked ops; lane-parallel over 32 elements at a time.
- **`Dst` is shared between unpacker, FPU/SFPU, and packer.** Ownership transfers explicitly: `tile_regs_acquire()` → math → `tile_regs_commit()` → `tile_regs_wait()` → `pack_tile()` → `tile_regs_release()`. The pattern is mandatory; `compute_engines_and_dataflow_within_tensix.md:119` warns: *"Even if a kernel does not pack any data, `tile_regs_commit` and `tile_regs_release` must still be called in sequence after computation to correctly manage the register state. Failure to do so results in undefined behavior."*
- **`Dst` capacity depends on host config**: 8 tiles (16-bit, double-buffered) up to 16 tiles (16-bit, no double-buffer). With `fp32_dest_acc_en=true`, halve it.
- **SFPU is not a generic vector ALU.** `compute_engines_and_dataflow_within_tensix.md:19`: *"the SFPU treats the register as holding 32 elements of at most 32 bits each, regardless of the actual data type."* Lane position inside the tile is meaningful (faces, §9), and the API surface is verb-per-op (`exp_tile`, `sin_tile`, `add_binary_tile`), not a free-form ISA.

## 3. Memory hierarchy

| | CUDA | Tensix |
| --- | --- | --- |
| Per-thread | registers | per-RISC-V private memory (stack/locals) |
| Per-block / per-core | shared memory (~64–100 KB, software-managed) | **L1 SRAM, 1.5 MB per Tensix** (Wormhole and Blackhole) |
| Chip-wide cache | L2 (~tens of MB, HW-managed) | **None.** No global cache. |
| Off-chip | HBM | DRAM, 8 × 4 GB = 32 GB on Blackhole, addressed as `(x, y, local_addr)` over the NoC |
| Cross-block / cross-core | atomics + L2; or kernel barriers | **Explicit NoC DMA**: `noc_async_read / write` |

From `memory_for_kernel_developers.md:8`:

> "Instead of a single address space shared by all cores, memory is addressed by an (`x`, `y`, `local_address`) tuple. This is due to the mesh-based design, where each node on the NoC has its own local resources."

There is no flat pointer-dereference path off-core. A RISC-V can read its own L1 and its own private region. **Everything else — another Tensix's L1, any DRAM tile, the PCIe block — is a NoC packet.** And there is no L2 to bail you out: Marty puts it this way: *"There is no cache hierarchy on Tenstorrent chips — this is a deliberate design choice... This approach provides deterministic performance and eliminates the unpredictability of cache evictions."* (Marty 2025-04-21.)

Two consequences that bite people coming from CUDA:
1. **Stack pointers are not DMA-addressable.** `memory_for_kernel_developers.md:80-93`: *"Stack variables cannot be used as DMA source or destination."* You can't `cudaMemcpyAsync(&local_var, ...)`-style move private data.
2. **Lock-step allocation hides bank count.** A `BufferType::DRAM` allocation returns one `uint32_t` address that means "the same offset in every bank, round-robined by `page_size`." `TensorAccessor(buffer_addr, tile_id)` resolves the per-bank NoC coordinate. This is the Tensix equivalent of "the HW interleaves your global memory" — except the kernel sees the math.

For the 27B model, this hierarchy is binding: KV cache `[1, 4, 32768, 128]` bf16 is 32 MiB per K/V per layer. Across 130 Tensix cores that is ~250 KB/core if sharded, which crowds out CB space. So caches live in **DRAM, interleaved**, and you scatter into them through the NoC, one tile at a time (`05_memory_configs_deep_dive.md:108`).

## 4. The dataflow primitive: circular buffers

CUDA has no direct analog. The closest you ever get is "producer threads stage tiles in shared memory, consumer threads read them, sync with `__syncthreads()`." Tensix formalizes that pattern as a first-class object: the **circular buffer**.

A CB is a paginated FIFO in L1, owned by one Tensix, used by some subset of the five RISC-V kernels on that core. Up to 32 CBs per core. Each CB is typed (its `DataFormat` configures the unpacker/packer when compute reads or writes it). From `eltwise_binary.md:151`:

> "Here we introduce a new concept: circular buffers. They are communication channels between the different kernel on a Tensix. Conceptually they act as pipes between different kernels."

Producer side:
```cpp
cb_reserve_back(cb, 1);                  // wait for free slot
noc_async_read_tile(i, src, get_write_ptr(cb));
noc_async_read_barrier();
cb_push_back(cb, 1);                     // publish to consumer
```

Consumer side:
```cpp
cb_wait_front(cb, 1);                    // wait for produced tile
matmul_tiles(cb_in0, cb_in1, 0, 0, dst_idx);
cb_pop_front(cb_in0, 1);                 // free the slot
```

The CB is **why all five RISC-Vs can run concurrently without explicit locks**. BRISC pushes input tiles; TRISC0/1/2 (the compute trinity) consumes them, writes results to an output CB; NCRISC writes the output CB tiles out to DRAM. Each stage stalls only when its CB is empty (consumer) or full (producer). It is producer-consumer pipelining baked into the hardware.

A typical eltwise kernel CB layout for one core:
```
DRAM in0 --BRISC--> CB c_0 --TRISC--> Dst tile --TRISC--> CB c_16 --NCRISC--> DRAM out
DRAM in1 --BRISC--> CB c_1 ----'
```

Page size = tile size. Total size = `tiles_per_cb * tile_size` (commonly 2 tiles for double-buffering between BRISC and TRISC). Conventional indices: `c_0..c_15` for inputs, `c_16..c_31` for outputs (`eltwise_binary.md:88-105`).

A pure data-movement kernel (the `dram_loopback` shape and our scatter kernel) can **skip CBs entirely** — BRISC reads to a shared L1 slot, NCRISC writes from it, two barriers between. CBs only become necessary once compute joins the pipeline or when you want two RISC-Vs (over two NoCs) to overlap.

## 5. Multi-core decomposition

CUDA: one program, `blockIdx.x/y/z` and `threadIdx.x/y/z` self-identify within a launched grid. Oversubscribe and let the HW scheduler decide.

Tensix: **per-core runtime arguments**, single launch. From `eltwise_binary.md:258`:

> "Unlike OpenCL/CUDA. Each kernel (reader, compute and writer) can have it's own set of arguments. Furthermore, on a multi cored program (i.e. using more then 1 Tensix core), kernels within each core can have different arguments. This enables Metalium to exploit the grid like nature of the Tenstorrent processors to achieve high performance."

The host loops over cores and calls `SetRuntimeArgs(program, kernel_id, core, {…})` with that core's slice. From `matmul_multi_core.md:384-423`, the canonical pattern:
```cpp
uint32_t work_offset = 0;
for (const auto& [ranges, work_per_core] : work_groups) {
  for (const auto& range : ranges.ranges()) {
    for (const auto& core : range) {
      SetRuntimeArgs(program, reader_id,  core, {src0_addr, src1_addr, Mt, Kt, Nt, work_offset, work_per_core});
      SetRuntimeArgs(program, writer_id,  core, {dst_addr, work_per_core, work_offset});
      SetRuntimeArgs(program, compute_id, core, {work_per_core, Kt});
      work_offset += work_per_core;
    }
  }
}
```

Kernels on-device read with `get_arg_val<uint32_t>(slot)`. There is no `blockIdx`. The host is the scheduler. Convenience helper `tt::tt_metal::split_work_to_cores(grid, num_work)` returns `(num_cores, all_cores, group_1, group_2, work_per_core_1, work_per_core_2)` so you can write the loop above generically (`matmul_multi_core.md:135-149`).

Mental shift: **the host knows the static work assignment before launch and bakes it in as args; the device knows only what its args say.**

## 6. "Shape it like the hardware" — Marty's philosophy

Marty (clehaxze) has published two long pieces on programming Tenstorrent. The recurring discipline:

> *"Even though the chip looks like a systolic array. Flexibility is paramount. There is no one stopping you from using it like a CPU with a SPMD pattern."* (2025-04-21, "Programming Tenstorrent processors.")

> *"RoPE also needs each pair of elements to be rotated by a different amount, based on their index. So, which element lands in which SFPU lane actually matters (RoPE is not a simple element wise operation), and the internal format of a tile must be taken into account."* (2025-10-26, "Building the RoPE operation.")

That second quote is the operative principle: **when an op is not lane-invariant, you design around the tile's face layout and the SFPU lane mapping, not around the math written on the page.** The 32×32 tile is stored as four 16×16 faces (`tiles.md:25-29`), and `dst_reg[0:3]` is face 0, `[4:7]` is face 1, and so on (`compute_engines_and_dataflow_within_tensix.md:298`). RoPE pairs (`x_i`, `x_{i+d/2}`) cross face boundaries; Marty's implementation has to account for which lane sees which element. Ignoring this is how "correct on paper" kernels produce wrong outputs.

Three named Marty techniques worth adopting:

**(M-1) The reader / compute / writer trinity is the default, not a fancy optimization.** From the 04-21 post: *"There are 3 written kernels. The reader, the writer, and the compute kernel… The compute kernel waits data to be made available by the reader, performs add."* The trinity decouples NoC latency from compute throughput. Marty also dismisses the "you have to write 5 programs" fear: *"It looks like a single operation is 5 programs that developers need to write, no? No, unless if you hell bent to."* — the compute file is one source compiled three times, so practically you write three sources, not five.

**(M-2) Abuse `Dst` as scratch.** From 10-26: *"There is one place that we could store data into — the Dst registers… Position indices will be stored in tile 2. And since each tile is 32 LReg wide, tile 2 starts at offset 64."* `Dst` is not just the output of `matmul_tiles`; it is a register file. Stash constants and intermediates there if it saves a CB round-trip.

**(M-3) Precompute on the host, pull constants in via the reader.** The 10-26 post hoists expensive things (RoPE frequency tables, division-via-softfp) into host-side math, pre-tilizes them, and the reader simply ships them in as another CB. The RISC-Vs don't have hardware FP division; you don't compute it on-device.

## 7. CUDA → Tensix conversion exercises

### 7a. RMSNorm

CUDA shape:
- 1 thread block per row of `[rows, hidden]`.
- Threads cooperatively load the row to shared memory, do warp-shuffle reduction for `sum(x^2)`, broadcast, divide, multiply by weight.
- One kernel, many threads, `__syncthreads()` between phases.

Tensix shape (single-row decode, `hidden = 4096` = 128 tiles wide along one row of tiles):
- One Tensix core handles one row, or you height-shard across cores. Pick based on row count.
- BRISC streams `hidden / 32` tiles from the row into CB `c_0`, plus weight tiles into CB `c_1`.
- TRISC trinity per tile: `copy_tile(c_0, …, dst=0)`, `square_tile(0)`, accumulate into a running-sum tile via `add_binary_tile`. After all tiles consumed: SFPU `reduce_tile` to a scalar, `rsqrt_tile`, broadcast back via `bcast_mul_h`. Then re-stream the row through `mul_tiles` with the weight CB. Push results to `c_16`.
- NCRISC pops `c_16` and writes back via `noc_async_write_tile`.
- Three programs, one core. Or shard the rows across N cores with `SetRuntimeArgs(start_row, num_rows)`.

The CUDA mental error to avoid: trying to make TRISC0/1/2 act like 32 cooperating threads. They are not. They are *unpacker, math, packer*. The reduction lives inside the FPU/SFPU operating on tiles, not across threads.

### 7b. KV cache scatter (the op we actually need)

Cache `[1, 4, 32768, 128]` bf16 in DRAM. New token `[1, 4, 1, 128]`. Write the new row into `cache[:, :, cur_pos, :]`.

CUDA shape: launch `(heads, head_dim/4)` threads. Each thread does one `cache[..., cur_pos, j] = new[..., j]` store. ~512 stores. Trivial.

Tensix shape (`03_hello_world_kernel_walkthrough.md` + `05_memory_configs_deep_dive.md`):
- Cache and input both INTERLEAVED DRAM (rationale: `05:163-180` — sidesteps Blackhole #16674 and matches the reference path).
- One Tensix per KV head — 4 cores total, `CoreRangeSet{(0,0)..(3,0)}`.
- **No compute kernel.** Pure data movement.
- BRISC: `noc_async_read_tile(src_tile_id, src_accessor, l1_slot); noc_async_read_barrier(); cb_push_back(cb, 1);`
- NCRISC: `cb_wait_front(cb, 1); noc_async_write_tile(dst_tile_id, dst_accessor, l1_slot); noc_async_write_barrier(); cb_pop_front(cb, 1);`
- Tile id math: `cur_pos` falls *inside* a tile row (32 rows per tile in TILE_LAYOUT). Either (a) read-modify-write one tile per row of cache or (b) live with sub-tile writes through `noc_async_write` byte-addressed. **Open: needs experiment** — `03:155`.
- Runtime args per core: `{src_addr, dst_addr, head_idx, cur_pos, head_dim_tiles}`.

Why this matters: today `ttnn.scatter` costs ~10 ms/token of dispatch overhead (memory: `feedback_ttnn_scatter_for_kv_cache`). A 4-core, no-compute, no-CB kernel is the floor — its only cost is one NoC read and one NoC write per head per token plus dispatch of a much smaller program.

### 7c. RoPE (Marty just wrote this — defer to him)

CUDA shape: thread per `(seq, head, dim_pair)`. Each thread reads two elements `x_i, x_{i+d/2}`, multiplies by `(cos, sin)` from a precomputed table, writes both back. Embarrassingly parallel.

Tensix shape (Marty 2025-10-26):
- Host precomputes the cos/sin tables once and tilizes them as constant inputs.
- BRISC streams the activation tile and the cos/sin tile(s) into CBs.
- Compute kernel: `copy_tile` activation into `Dst[0]`, cos into `Dst[1]`, sin into `Dst[2]`. Use SFPU `LReg` views (`dst_reg[i]`) to fish out the right halves of each face and do the rotation pairwise.
- Critical: face-aware indexing. Marty: *"Which element lands in which SFPU lane actually matters."* The (i, i+d/2) pairing has to be mapped to lane addressing inside the tile's face layout.
- Pack to `c_16`, NCRISC writes out.

The team's memory note `feedback_c3_native_rope_abandoned` says the native `ttnn.experimental.rotary_embedding` op fights us on `cos_cache padded_shape` constraints under partial-rotary. A custom kernel built Marty-style is the principled fallback if we go back to it.

## 8. Common pitfalls when porting a CUDA mental model

- **One-giant-kernel reflex.** "Just write a big kernel and parallelize threads." There is no thread fanout per core; you split the work across kernels (reader/compute/writer) and across cores (runtime args). The single-kernel idea collapses the pipeline into serial NoC waits.
- **Forgetting CBs (or sizing them = 1 tile).** A 1-tile CB serializes producer and consumer. Standard is 2-tile (double-buffered), or 4+ when the compute is much slower than the read. CBs are how you hide NoC latency *within a core*.
- **Single-core thinking.** Blackhole P150 exposes 130 Tensix cores. A kernel that uses 1 of them is a 1.3% kernel. `split_work_to_cores` is one line. Use it from kernel #1.
- **Over-sharding.** L1 is 1.5 MB per core and shared with kernel binaries, CBs, semaphores. Aggressive width-sharding leaves no room for the CB pipeline; on Blackhole the sharded-input writer in `paged_update_cache` deadlocks (#16674; `05:142-157`). When in doubt, INTERLEAVED in DRAM, and shard only when bandwidth measurements demand it.
- **Treating SFPU like a vector ALU.** It is a tile-shaped vector unit with face-aware lane mapping and a fixed verb set. Don't reach for "and then I'll do some bit-twiddling per lane" — the right primitive is probably already named (`sigmoid_tile`, `exp_tile`, `rsqrt_tile`, etc.), and if it isn't, build it from the named ones or write low-level SFPI.
- **Python scalars baked into traces.** Memory: `feedback_trace_capture`. Anything that must vary across replays goes through device-resident tensors, not Python ints captured into runtime args. The `paged_update_cache` op carries `cur_pos` as an INT32 ROW_MAJOR `update_idxs_tensor` for exactly this reason (`05:131-138`).

## 9. The Tensix idioms with no CUDA equivalent

| Idiom | What it is | Why it has no CUDA mirror |
| --- | --- | --- |
| **TILE_LAYOUT** | 32×32 tile stored as four 16×16 faces, address order `face0, face1, face2, face3` (`tiles.md:25-39`) | CUDA tensors are row-major; tile layout exists only inside cooperative MMA fragments, never at the buffer level |
| **Reader / compute / writer trinity** | Three programs per core, communicating through CBs, scheduled by the producer-consumer protocol | CUDA has one kernel; oversubscription hides latency. Tensix exposes pipelining directly. |
| **NoC routing (unicast / multicast)** | `noc_async_*` packets explicitly addressed `(x, y, addr)`; multicast for one→many broadcasts | CUDA has L2 + atomics + cooperative groups, all HW-managed. Tensix exposes the wire. |
| **Producer-consumer via CBs** | `cb_reserve_back / cb_push_back / cb_wait_front / cb_pop_front` as the only inter-RISC-V sync within a core | Closest CUDA analog is shmem + `__syncthreads()`, but that is barrier-style, not flow-control style |
| **Interleaved vs sharded placement** | INTERLEAVED round-robins pages across banks transparently; SHARDED pins specific tiles to specific cores | CUDA HBM is HW-interleaved; sharding (e.g. cudaMemcpyPeer) is for multi-GPU, not within a chip |
| **Per-core runtime args** | The same program receives a different `{a, b, c, …}` on each core, set host-side at launch | CUDA threads compute their own index; Tensix cores are told their index |
| **Lock-step DRAM allocation** | One `uint32_t` address shared across N banks, with `TensorAccessor` computing per-bank NoC coords | CUDA allocations return one device pointer; banking is invisible |

## 10. Where to go from here

If after reading this you can answer the following without looking back, you have the model:
- *"For an op X, which RISC-V (BRISC, NCRISC, TRISC0/1/2) does what, and what CBs connect them?"*
- *"How many tiles per core, and how is the work split across cores via runtime args?"*
- *"Where does each tensor live (DRAM interleaved? L1 sharded? L1 interleaved?) and why?"*
- *"What's the tile-layout subtlety — does my op care about face boundaries or SFPU lane mapping?"*

When those four questions have crisp answers before you write a line of C++, you're designing kernels the way Tensix wants. The Marty discipline in one sentence: **the math you wrote on paper has to be shaped to the tile, the face, the lane, the CB, the core grid — not the other way around.**
