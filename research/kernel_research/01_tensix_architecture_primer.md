# 01: Tensix Architecture Primer (for the in-place KV-scatter kernel)

**Audience**: team writing a custom Tensix kernel that does 1-tile-read + 1-tile-write per KV head, replacing `ttnn.scatter` (~10 ms/token of dispatch overhead).

**Sources** (all local to this repo):
- `tt_docs_corpus/.../advanced_topics/compute_engines_and_dataflow_within_tensix.md`
- `tt_docs_corpus/.../advanced_topics/memory_for_kernel_developers.md`
- `tt_docs_corpus/.../advanced_topics/tiles.md`
- `tt_docs_corpus/.../examples/{dram_loopback,eltwise_binary}.md` — best end-to-end walkthroughs
- `tt_docs_corpus/.../tools/watcher.md` — the *only* file in the corpus that names BRISC/NCRISC/TRISC0/1/2

The upstream `programming_model/` doc is just a pointer to `METALIUM_GUIDE.md`; real content lives in `advanced_topics/`.

---

## 1. The five RISC-V cores per Tensix

From `examples/eltwise_binary.md:120`: *"The Tensix core in fact contains 5 RISC-V cores. 2 of them are the data movement cores ... The other 3 are compute cores, which operate cooperatively and run a single compute kernel."* Names from `tools/watcher.md:96`:

| RISC-V | Name | Role | Host knob |
| --- | --- | --- | --- |
| 0 | **BRISC** | Data-movement: typically *reader* (`noc_async_read*`) | `DataMovementProcessor::RISCV_0`, NoC 0 |
| 1 | **NCRISC** | Data-movement: typically *writer* (`noc_async_write*`) | `DataMovementProcessor::RISCV_1`, NoC 1 |
| 2 | **TRISC0** | Compute "Unpack" — drives unpacker into `SrcA/SrcB/Dst` | one `ComputeConfig` produces all 3 TRISC binaries |
| 3 | **TRISC1** | Compute "Math" — issues FPU/SFPU instructions | (same) |
| 4 | **TRISC2** | Compute "Pack" — drives packer from `Dst` back to a CB | (same) |

Key facts:
- A single compute `.cpp` is **compiled three times** into three TRISC binaries (`compute_engines_and_dataflow_within_tensix.md:45`). The `tile_regs_acquire/commit/wait/release` calls hand `Dst` register ownership between them.
- BRISC and NCRISC are independent `.cpp` files. Convention: BRISC = NoC 0, NCRISC = NoC 1, so reader/writer use different NoCs in parallel.
- FPU, SFPU, unpacker, packer are **not** processing cores — they execute commands issued by the TRISC RISC-Vs (`compute_engines_and_dataflow_within_tensix.md:47`).

**Scatter-kernel implication**: pure tile-copy needs **no compute pipeline**. The `dram_loopback.md` shape — one data-movement kernel, no CBs, no `ComputeConfig` — is the minimum.

## 2. L1 SRAM layout

From `memory_for_kernel_developers.md:16` and `dram_loopback.md:51`:

- **Per-Tensix L1 = 1.5 MB on both Wormhole and Blackhole** (Grayskull was 1 MB). This is **shared SRAM scratchpad**, not a cache.
- Each RISC-V has a **small private region** for stack/locals, mapped at the same address on every core. Stack addresses are meaningless cross-core; **you cannot DMA into a stack array** (`memory_for_kernel_developers.md:80-93`).
- Each RISC-V has a 0.5–2 KiB **instruction cache** (~128–512 instructions) fronting L1.
- RISC-V loads/stores to L1 are **bandwidth-limited, multi-cycle latency** — bulk data goes through NoC/unpacker/packer, not via RISC-V ld/st.
- L1 holds: kernel binaries, circular buffers, semaphores, and any `BufferType::L1` allocations. All clients (NoC RX/TX, packer, unpacker, RISC-Vs) contend for the same L1 bandwidth.

## 3. DRAM access via NoC and circular buffers

Memory is addressed as `(x, y, local_address)` — a NoC coordinate plus an offset (`memory_for_kernel_developers.md:8`). There is **no flat address space**.

- RISC-Vs directly address only private memory + local L1. Everything else (other Tensix L1, DRAM, peripherals) is an async DMA through the NoC.
- DRAM appears as DRAM tiles on the NoC. **Wormhole: 6 × 2 GB GDDR6 (12 GB). Blackhole: 8 × 4 GB (32 GB)** (`memory_for_kernel_developers.md:153`).
- `noc_async_read[_tile] / noc_async_write[_tile]` are non-blocking, may complete out of order. Fence with `noc_async_read_barrier()` / `noc_async_write_barrier()`.
- `TensorAccessor` + `TensorAccessorArgs` is the recommended abstraction: given `(tile_index, buffer, l1_addr)` it handles bank math for interleaved or sharded buffers.

**Circular buffers (CBs)** (`eltwise_binary.md:85`):
- L1-backed paginated FIFOs piping tiles **between kernels on the same Tensix** (BRISC ↔ TRISCs ↔ NCRISC).
- Up to 32 CBs per Tensix, configured host-side via `CircularBufferConfig{index, total_size, page_size, data_format}` + `CreateCircularBuffer`.
- Kernel API: `cb_reserve_back / cb_push_back` (producer), `cb_wait_front / cb_pop_front` (consumer); `get_read_ptr / get_write_ptr` returns an L1 address you pass to `noc_async_*`.
- The CB's `data_format` metadata is what configures unpacker/packer in compute kernels — the CB is the *typed pipe*.

**Why CBs exist**: they decouple reader (BRISC), compute (TRISC0/1/2), writer (NCRISC) so all four run concurrently. Synchronization is structural via runtime-managed semaphores.

**Scatter implication**: cheapest form is BRISC-only — `noc_async_read_tile` → barrier → `noc_async_write_tile` → barrier, **no CBs, no compute kernel**. To overlap read and write across NoCs, split into BRISC reader + NCRISC writer with a 1-tile CB between them (standard `dram_loopback` extended with a second mover).

## 4. Host-side Program / dispatch model

From `host_apis.md`, `dram_loopback.md`, `eltwise_binary.md`:

1. `MeshDevice::create_unit_mesh(device_id)` — even a single P150 is a 1×1 mesh.
2. `mesh_device->mesh_command_queue()` — FIFO of uploads/downloads/launches.
3. `Program program = CreateProgram();`
4. `MeshBuffer::create(...)` with `DeviceLocalBufferConfig{page_size, buffer_type=DRAM|L1}` + `ReplicatedBufferConfig{size}`. **Lock-step allocation** returns a single `uint32_t` address even though storage is round-robined across all banks (`memory_for_kernel_developers.md:151-162`).
5. `CreateCircularBuffer(program, core, cfg)` for any CBs.
6. `CreateKernel(program, "path/kernel.cpp", core_or_corerange, DataMovementConfig{...} | ComputeConfig{...})` per kernel (≤ 1 BRISC + 1 NCRISC + 1 ComputeConfig per core; the latter produces the 3 TRISC binaries). Compile-time args bake in here.
7. `SetRuntimeArgs(program, kernel_id, core, {a,b,c,...})` — per-core `uint32_t` vector, read on-device via `get_arg_val<uint32_t>(i)`. **Per-core args can differ within a single program.**
8. Wrap in a `MeshWorkload`, then `EnqueueMeshWorkload(cq, workload, blocking)` and `Finish(cq)`.

Kernel source resolution: `TT_METAL_KERNEL_PATH` → `TT_METAL_HOME` → absolute → CWD (`dram_loopback.md:128-136`). Metal's own RISC-V toolchain compiles to a per-core binary placed in that core's L1.

**Crucial for our scatter**: a single host launch sends the same program to every core in the `CoreRangeSet`, but `SetRuntimeArgs` varies per core. So multi-core scatter (one core per KV head) is **one kernel + N rows of runtime args**, not N programs — that's how we amortize the `ttnn.scatter` dispatch overhead.

## 5. Blackhole vs Wormhole — what the docs actually say

The docs draw very few hard distinctions. Confirmed deltas in this corpus:

| Aspect | Wormhole B0 | Blackhole |
| --- | --- | --- |
| L1 per Tensix | 1.5 MB | **1.5 MB** (same) — `dram_loopback.md:51` |
| DRAM controllers | 6 × 2 GB GDDR6 (12 GB) | **8 × 4 GB (32 GB)** — `memory_for_kernel_developers.md:153` |
| SFPU `LReg` width | 32 elements | **32 elements** — `compute_engines_and_dataflow_within_tensix.md:283` |
| Tile / face shape | 32×32, 4 × 16×16 faces | **same** — `tiles.md` |

Everything else (`Dst` semantics, the 8/16-tile Dst capacity table, CB API, NoC API, BRISC/NCRISC convention) is presented as architecture-wide. The docs do **not** call out Blackhole-specific scatter pitfalls. The Tt-ISA reference linked from the docs is the *Wormhole* one — Blackhole bit-level behavior is **not documented here**.

From project memory (not these docs): on Blackhole P150, `WormholeComputeKernelConfig` must be all-or-nothing; `paged_update_cache` has layout conflicts (we cast src to bf16 in TILE_LAYOUT to work around them). Worth keeping in mind when designing this kernel's host-side config.

## 6. Minimum viable shape for a 1-tile-read + 1-tile-write KV scatter

Putting (1)–(5) together:

- **Topology**: one Tensix per KV head (or one Tensix looping over heads). `CoreRangeSet` chosen on host.
- **Kernels per core**: 1 BRISC reader + 1 NCRISC writer + **no compute kernel** (FPU/SFPU never touched).
- **CBs**: one 1-tile CB, `page_size = tile_size` (2 KB bf16 / 4 KB fp32), `total_size = 2 × page_size` for double-buffering between BRISC and NCRISC.
- **Per-head work**:
  - BRISC: `cb_reserve_back(cb,1); noc_async_read_tile(src_idx, src_accessor, get_write_ptr(cb)); noc_async_read_barrier(); cb_push_back(cb,1);`
  - NCRISC: `cb_wait_front(cb,1); noc_async_write_tile(dst_idx, dst_accessor, get_read_ptr(cb)); noc_async_write_barrier(); cb_pop_front(cb,1);`
- **Runtime args per core**: src buffer addr, src tile idx, dst buffer addr, dst tile idx. `TensorAccessor` handles bank math.
- **Data type**: 2 KB tile for bf16 (matches the paged_update_cache TILE_LAYOUT bf16 we already use).

### Open questions / gaps

- **L1 / NoC bandwidth numbers are not in this corpus** — only qualitative "limited, several cycles". Need upstream `WormholeB0/TensixTile/L1.md` ISA doc for byte/cycle ceilings.
- **Corsix Wormhole blog + clehaxze's article are not local** — both would help us pin down NoC packet sizing and RISC-V → packer issue rates (relevant for whether the kernel is dispatch-bound or transfer-bound).
- **No documented number for `EnqueueMeshWorkload` overhead** — we have the *symptom* (`ttnn.scatter` = 10 ms/token), but the floor of a hand-rolled metal kernel must be measured before we know how big the win can be.
