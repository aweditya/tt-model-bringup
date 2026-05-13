# 01: Tensix Architecture Primer (for the in-place KV-scatter kernel)

**Audience**: team about to write a custom Tensix kernel that does
1-tile-read + 1-tile-write per KV head, replacing `ttnn.scatter`.

**Sources** (all paths are local to this repo):

- `tt_docs_corpus/.../tt_metal/programming_model/index.md` — pointer to the upstream `METALIUM_GUIDE.md`. The real content lives in `advanced_topics/`.
- `tt_docs_corpus/.../advanced_topics/compute_engines_and_dataflow_within_tensix.md`
- `tt_docs_corpus/.../advanced_topics/memory_for_kernel_developers.md`
- `tt_docs_corpus/.../advanced_topics/tiles.md`
- `tt_docs_corpus/.../apis/host_apis.md`, `.../apis/kernel_apis.md`
- `tt_docs_corpus/.../examples/dram_loopback.md` and `.../examples/eltwise_binary.md` (best end-to-end walkthrough of a 3-kernel program)
- `tt_docs_corpus/.../tools/watcher.md` (only place the names BRISC/NCRISC/TRISC0/1/2 appear by name)

---

## 1. The five RISC-V cores per Tensix

From `examples/eltwise_binary.md:120`: *"The Tensix core in fact contains 5 RISC-V cores. 2 of them are the data movement cores ... The other 3 are compute cores, which operate cooperatively and run a single compute kernel."*

The convention (from `tools/watcher.md:96, 228-232`) is:

| RISC-V | Tt-Metal name | Role | Host config knob |
| --- | --- | --- | --- |
| 0 | **BRISC** | Data movement core 0 — typically the *reader*: issues `noc_async_read*` to bring DRAM/L1 tiles into circular buffers | `DataMovementProcessor::RISCV_0`, default NoC `NOC::RISCV_0_default` (NoC 0) |
| 1 | **NCRISC** | Data movement core 1 — typically the *writer*: `noc_async_write*` from CBs to DRAM/L1 | `DataMovementProcessor::RISCV_1`, NoC 1 |
| 2 | **TRISC0** | Compute "Unpack" core — drives the **unpacker** that pulls tiles from CBs into `SrcA`/`SrcB`/`Dst` | one `ComputeConfig` produces all three TRISC binaries |
| 3 | **TRISC1** | Compute "Math" core — issues FPU/SFPU instructions | (same) |
| 4 | **TRISC2** | Compute "Pack" core — drives the **packer** that writes `Dst` tiles back into a CB in L1 | (same) |

Key facts the team must internalize:

- A single `.cpp` `ComputeConfig` source is **compiled three times** into three separate binaries (`compute_engines_and_dataflow_within_tensix.md:45`). The `tile_regs_acquire / commit / wait / release` calls are how those three TRISC binaries hand the `Dst` register set back and forth.
- BRISC and NCRISC are independent `.cpp` files with their own `kernel_main()`. Convention: BRISC = NoC 0, NCRISC = NoC 1, so reader + writer use different NoCs in parallel.
- The FPU/SFPU/unpacker/packer are **not** processing cores. They do not run instructions; the TRISC RISC-Vs issue commands to them (`compute_engines_and_dataflow_within_tensix.md:47`).

**Scatter-kernel implication**: a single-Tensix scatter where each KV head does `read tile → write tile` does **not need the compute pipeline at all**. The pure-RISC-V `dram_loopback.md` shape — one data-movement kernel, no `ComputeConfig`, no CBs — is the minimal viable form. CBs only become necessary if the writer hands work to a separate NCRISC.

## 2. L1 SRAM layout

From `memory_for_kernel_developers.md:16` and `dram_loopback.md:51`:

- **Per Tensix L1 size: 1.5 MB on Wormhole *and* Blackhole** (Grayskull was 1 MB). This is **shared SRAM**, not cache.
- Each RISC-V also has a **small private region** for its stack/locals, mapped at the *same address* on every core (so a stack pointer value is meaningless across cores; you cannot pass a stack address to another core or to the NoC).
- Each RISC-V has a tiny per-core **instruction cache: 0.5–2 KiB (128–512 instructions)** that fronts L1 reads of the kernel binary.
- Binaries themselves live in L1. RISC-V loads/stores to L1 are **bandwidth-limited and several cycles of latency** — the bulk-data path is the NoC and the unpacker/packer, not the RISC-V's own ld/st.
- All other clients on the Tensix (NoC RX/TX, packer, unpacker, RISC-Vs) contend for the same L1.

L1 contains, in practice: kernel code, circular buffers, semaphores, and any `BufferType::L1` allocations. Stack lives in private memory; **you cannot DMA into a stack array** (`memory_for_kernel_developers.md:80-93`).

## 3. DRAM access via NoC and the role of circular buffers

The address model is **not** a flat space. From `memory_for_kernel_developers.md:8`: memory is `(x, y, local_address)` — a NoC coordinate plus an offset into that tile's local memory.

- RISC-V cores directly address only their own private memory + the local shared SRAM. **Any** access to another Tensix's L1, to DRAM, or to a peripheral goes through the NoC as an async DMA (`memory_for_kernel_developers.md:70-115`).
- DRAM is exposed as DRAM tiles on the NoC, each backed by a memory controller. **Wormhole: 6 controllers × 2 GB GDDR6. Blackhole: 8 controllers × 4 GB** (`memory_for_kernel_developers.md:153`).
- The standard `noc_async_read / noc_async_write / noc_async_read_tile / noc_async_write_tile` calls are non-blocking and may complete out of order. Use `noc_async_read_barrier()` / `noc_async_write_barrier()` to fence.
- `TensorAccessor`/`TensorAccessorArgs` is the recommended abstraction for both interleaved and sharded buffers — given `(tile_index, buffer, dst_l1_addr)` it computes the right `(bank_id, offset)`. The KV cache is almost certainly an interleaved DRAM (or sharded L1) buffer reachable through this.

**Circular buffers (CBs)**:

- CBs are SRAM-backed, paginated, producer/consumer queues **inside one Tensix's L1**, used to pipe tiles between BRISC, NCRISC, and the TRISCs (`eltwise_binary.md:85`).
- Up to 32 CBs per Tensix. Each is configured with `{index, total_size, page_size, data_format}` on the host via `CircularBufferConfig` / `CreateCircularBuffer`.
- Kernel API: `cb_reserve_back / cb_push_back` (producer side), `cb_wait_front / cb_pop_front` (consumer side), `get_read_ptr / get_write_ptr` to get an L1 address you can pass to `noc_async_*` (`apis/kernel_apis.md:13-20`).
- The CB's `data_format` metadata is also what drives unpacker/packer reconfiguration in compute kernels — i.e. the CB is the typed pipe.

**Why CBs exist**: they decouple the reader (BRISC), the compute pipeline (TRISC0/1/2), and the writer (NCRISC) so all four can run concurrently. The synchronization is structural: a producer must `cb_reserve_back` before writing, a consumer must `cb_wait_front` before reading, and they communicate via internal semaphores managed by the runtime.

**Scatter-kernel implication**: the cheapest scatter form is BRISC-only — issue `noc_async_read_tile` for the source page, `noc_async_read_barrier`, then `noc_async_write_tile` to the destination cache page, `noc_async_write_barrier`. No CBs, no compute kernel. If the source and destination live on different NoCs (e.g. cache in DRAM, source in L1), we can split into BRISC reader + NCRISC writer with one 1-tile CB between them to overlap the two transfers — this is the standard pattern from `dram_loopback.md` upgraded with a second mover.

## 4. Host-side Program / dispatch model

From `host_apis.md`, `dram_loopback.md`, `eltwise_binary.md`:

The host-side flow per program is:

1. `MeshDevice::create_unit_mesh(device_id)` — even a single P150 is a 1×1 mesh.
2. `mesh_device->mesh_command_queue()` — the FIFO of commands (uploads, downloads, program launches) is the only ordering primitive.
3. `Program program = CreateProgram();` — empty container.
4. `MeshBuffer::create(...)` with `DeviceLocalBufferConfig{page_size, buffer_type=DRAM|L1}` and a `ReplicatedBufferConfig{size}`. The `page_size` controls round-robin bank assignment ("interleaved"); the address returned is a single `uint32_t` thanks to **lock-step allocation** (`memory_for_kernel_developers.md:151-162`).
5. `CircularBufferConfig(...).set_page_size(...)` + `CreateCircularBuffer(program, core, cfg)` for any CBs.
6. `CreateKernel(program, "path/to/kernel.cpp", core_or_corerange, DataMovementConfig{...} | ComputeConfig{...})` for each of the up-to-5 kernels (1 BRISC + 1 NCRISC + 1 ComputeConfig, the latter producing 3 TRISC binaries). Compile-time args are baked in here.
7. `SetRuntimeArgs(program, kernel_id, core, {a, b, c, ...})` — per-core vector of `uint32_t`s, retrieved on-device via `get_arg_val<uint32_t>(i)`.
8. Wrap in a `MeshWorkload`, then `EnqueueMeshWorkload(cq, workload, blocking)` and `Finish(cq)`.

Compilation: kernels are looked up by `TT_METAL_KERNEL_PATH`, then `TT_METAL_HOME`, then absolute, then CWD (`dram_loopback.md:128-136`). Each kernel is compiled by Metal's own RISC-V toolchain; what reaches the device is a per-core binary placed in that core's L1.

**Important for our scatter kernel**: a single host launch sends the same `Program` to every core in the `CoreRangeSet`, but `SetRuntimeArgs` can vary per core — so a multi-core scatter (one core per KV head) is just a single kernel + N rows of runtime args, not N programs. This is the cheapest path to amortize the ~10 ms `ttnn.scatter` dispatch overhead.

## 5. Blackhole vs Wormhole — what the docs actually say

The docs draw very few hard distinctions. Confirmed deltas:

| Aspect | Wormhole (B0) | Blackhole |
| --- | --- | --- |
| L1 per Tensix | 1.5 MB | **1.5 MB** (same) — `dram_loopback.md:51` |
| DRAM controllers | 6 × 2 GB GDDR6 (12 GB) | **8 × 4 GB (32 GB)** — `memory_for_kernel_developers.md:153` |
| SFPU `LReg` width | 32 elements | **32 elements** — `compute_engines_and_dataflow_within_tensix.md:283` ("on Wormhole and Blackhole, LReg is 32 elements wide") |
| Tile / face shape | 32×32, 4×(16×16) | **same** — `tiles.md` |

Everything else (`Dst`-register semantics, the 16-bit/32-bit Dst capacity table, CB API, NoC API, BRISC/NCRISC convention) is presented as architecture-wide. The docs do **not** call out Blackhole-specific scatter pitfalls. The published Tt-ISA reference link in the docs is the *Wormhole* one — Blackhole-specific bit-level behavior is not in this corpus.

From our existing project memory (not from these docs): on Blackhole P150, `WormholeComputeKernelConfig` must be passed all-or-nothing or it corrupts ops; `paged_update_cache` has layout conflicts that forced us to cast src to bf16 in TILE_LAYOUT. Worth keeping in mind when designing the kernel's host-side config struct.

## 6. Specifically for a 1-tile-read + 1-tile-write KV scatter

Putting (1)–(5) together, the minimum-viable kernel shape is:

- **Topology**: one Tensix per scattered KV head (or one Tensix doing all of them in a loop) — `CoreRangeSet` chosen on host.
- **Kernels per core**: 1 BRISC reader + 1 NCRISC writer + 0 compute kernels (no FPU/SFPU touch).
- **CBs**: one 1-tile CB (page_size = tile_size = 2 KB for bfloat16, 4 KB for fp32) sized at 2 pages for double-buffering between BRISC and NCRISC.
- **Per-head work**:
  - BRISC: `cb_reserve_back(cb,1); noc_async_read_tile(src_tile_idx, src_accessor, get_write_ptr(cb)); noc_async_read_barrier(); cb_push_back(cb,1);`
  - NCRISC: `cb_wait_front(cb,1); noc_async_write_tile(dst_tile_idx, dst_accessor, get_read_ptr(cb)); noc_async_write_barrier(); cb_pop_front(cb,1);`
- **Runtime args per core**: source buffer addr, source tile index, dest buffer addr, dest tile index — that's it. The `TensorAccessor` from `TensorAccessorArgs` handles bank math.
- **Data type**: tile sizes are 2 KB (bf16/bfloat16_b) or 4 KB (fp32). If the cache is paged_update_cache TILE_LAYOUT bf16 (per project memory), use the bf16 tile size.

### Open questions / gaps in this corpus

- The docs **do not** quantify L1 bandwidth in bytes/cycle for the NoC vs the RISC-V load/store ports; just qualitative "limited bandwidth, several cycles of latency". Numeric ceilings need the upstream `WormholeB0/TensixTile/L1.md` ISA doc.
- The corsix Wormhole blog series and clehaxze's article (in the project resource list) are not in this local corpus — would be worth a follow-up note to scrape them for NoC packet sizing and the exact RISC-V → packer issue-rate, which will matter for whether the kernel is dispatch-bound or transfer-bound.
- Nothing here documents the cost of `Program` launch itself (the `EnqueueMeshWorkload` overhead). Project memory has the symptom (`ttnn.scatter` = 10 ms/token of dispatch); a separate experiment must measure the floor of a hand-rolled metal kernel before we know how big the win can be.
