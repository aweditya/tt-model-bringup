# Command Queue, Dispatch, and the Five RISC-V Cores

## Q: What are the 5 RISC-V cores inside each Tensix tile?

**A:** Each Tensix tile contains five RV32IM ("Baby RISC-V") cores. They are deliberately minimal -- no interrupts, no user/kernel modes, no hypervisor, bare-metal only. Their job is *control and instruction dispatch*, not heavy computation (that is done by the matrix engine, vector engine, and pack/unpack hardware units).

| Core | Also known as | Primary role |
|------|--------------|--------------|
| **BRISC** | Data Movement 0, RISCV_0, "B" core | Reader kernel (data movement in via NoC 0) |
| **NCRISC** | Data Movement 1, RISCV_1, "NC" core | Writer kernel (data movement out via NoC 1) |
| **TRISC0** | Compute core 0, "T0" | Unpack -- reads data from L1 into SrcA/SrcB registers |
| **TRISC1** | Compute core 1, "T1" | Math -- drives the matrix engine (FPU) and vector engine (SFPU) |
| **TRISC2** | Compute core 2, "T2" | Pack -- moves results from Dst registers back to L1 |

Key details:
- Each core has 32 GPRs (32-bit) and a 32-bit program counter.
- BRISC and NCRISC each "own" one of the two NoC interfaces (NoC 0 and NoC 1 respectively). They issue DMA read/write commands to move data between DRAM, other tiles' L1, and the local L1.
- TRISC0/1/2 share a single Tensix instruction pipeline but drive different *pipes* within it. A single compute kernel source file is compiled three times with different preprocessor defines (`-DUCK_CHLKC_UNPACK`, `-DUCK_CHLKC_MATH`, `-DUCK_CHLKC_PACK`), producing three separate ELF binaries (trisc0.elf, trisc1.elf, trisc2.elf).
- The T cores use a clever encoding trick: instructions with low bits != `0b11` are "rotated right by two and treated as data" written to MMIO address `0xFFE40000`, which injects them into the Tensix instruction pipeline. This is how RISC-V cores issue Tensix-native instructions.

## Q: How do the three kernel types map to cores?

**A:** Every TT-Metalium operation is decomposed into three cooperating kernels:

```
Reader kernel  --> BRISC  (Data Movement 0, NoC 0)
Compute kernel --> TRISC0 + TRISC1 + TRISC2 (Unpack + Math + Pack)
Writer kernel  --> NCRISC (Data Movement 1, NoC 1)
```

The reader kernel runs on BRISC and issues NoC 0 DMA reads to pull data from DRAM (or other tiles) into the local L1 SRAM. The writer kernel runs on NCRISC and issues NoC 1 DMA writes to push results out. The compute kernel is a single source file but compiles into three binaries -- one per TRISC core -- each handling its pipeline stage.

This decomposition is fundamental: it allows data movement and computation to be *overlapped*. While TRISC0/1/2 are computing on tile N, BRISC can be prefetching tile N+1 and NCRISC can be writing out tile N-1. This is the Tensix equivalent of a software pipeline.

## Q: How do the cores synchronize?

**A:** Through **circular buffers** backed by **hardware semaphores**.

Circular buffers (CBs) live in L1 SRAM and act as producer-consumer queues between kernels:

```
BRISC (reader) --[CB0]--> TRISC0 (unpack) --> TRISC1 (math) --> TRISC2 (pack) --[CB1]--> NCRISC (writer)
```

The API is simple:
- **Producer side**: `cb_reserve_back(cb, n)` blocks until n pages are free, then `cb_push_back(cb, n)` marks them as ready.
- **Consumer side**: `cb_wait_front(cb, n)` blocks until n pages are available, then `cb_pop_front(cb, n)` frees them.

Under the hood, each Tensix has 8 hardware mutexes and 8 hardware semaphores that enforce these waits at the hardware level -- no spinning in software loops. This is what makes the 3-kernel decomposition efficient despite running on 5 separate processors.

Between the three TRISC cores specifically, synchronization also happens through the Tensix instruction pipeline itself. The Macro-Op Expander and Replay Expander stages in the pipeline coordinate unpack/math/pack sequencing so that TRISC1 (math) does not fire until TRISC0 (unpack) has loaded the source registers.

## Q: What is the TT-Metal command queue?

**A:** The command queue is the mechanism by which the host CPU sends work to the device. In **fast dispatch** mode (the default), it works like this:

1. The host writes commands into a **memory region accessible to both host and device** (likely mapped via PCIe BAR or hugepages).
2. A **dedicated RISC-V core on the device** -- the "dispatch core" -- continuously reads from this queue and processes commands.
3. This dispatch core is typically placed on an **unused Ethernet tile** (not a Tensix tile), so it does not consume any compute resources.
4. Each device supports **up to 2 command queues**, each with its own dedicated dispatch core. Queue 0 is typically used for compute dispatch, Queue 1 for data transfer, though both support any command type.

Commands in the queue include:
- **WriteBuffer**: Transfer data from host to device DRAM/L1
- **ReadBuffer**: Transfer data from device to host
- **EnqueueProgram**: Load and execute kernels on specified Tensix cores
- **Events**: Synchronization primitives between the two queues

The queues are strictly in-order (FIFO). To coordinate between queues (e.g., "don't start computing until the input write on queue 1 is done"), you use **events** as cross-queue synchronization barriers.

### Slow dispatch (debug mode)

Setting `TT_METAL_SLOW_DISPATCH_MODE=1` bypasses the command queue entirely. The host CPU directly performs each operation synchronously -- writing buffers with `WriteToBuffer()`, launching programs, and blocking until completion. This is useful for debugging but has massive overhead since every operation is a blocking host-device round-trip.

## Q: What happens step-by-step when you call `ttnn.matmul()`?

**A:** Here is the full path from Python to silicon:

### 1. Python layer (TT-NN)
```python
z = ttnn.matmul(x, y)
```
TT-NN looks up the matmul operation, selects an optimized implementation (program config, sharding strategy, data format), and prepares a **Program** object containing the three kernel binaries + their runtime arguments.

### 2. Program compilation
The matmul op specifies:
- Which Tensix cores to use (a rectangular grid from the compute grid)
- Reader kernel source (C++) -- compiled to BRISC binary
- Compute kernel source (C++) -- compiled to TRISC0/1/2 binaries
- Writer kernel source (C++) -- compiled to NCRISC binary
- Circular buffer allocations (how much L1 per CB, how many pages)
- Runtime arguments (tensor addresses, dimensions, strides)

Compilation happens once; results are cached in the **program cache** so subsequent calls skip this step.

### 3. Dispatch via command queue
The host enqueues an `EnqueueProgram` command into the command queue. This command contains:
- Kernel binary data (or references to already-loaded binaries)
- Per-core runtime arguments
- CB configurations
- The set of target Tensix cores

### 4. Dispatch core processes the command
The dispatch core (running on an Ethernet tile) reads the command from the queue and:
- **Writes kernel binaries** to each target Tensix tile's L1 SRAM via NoC DMA (each core's binary goes to a specific address in L1)
- **Writes runtime arguments** to each core's designated argument region in L1
- **Configures circular buffers** by writing CB metadata to L1
- **Releases cores from soft-reset** (or signals them to start) -- each RISC-V core on the Tensix tile begins execution at address 0 in its instruction memory

### 5. Kernel execution on Tensix cores
All five RISC-V cores on each target Tensix tile begin executing simultaneously:
- **BRISC**: Runs the reader kernel -- DMA reads input tiles from DRAM into local L1 circular buffers
- **TRISC0**: Runs unpack -- reads tiles from L1 CBs, converts data format, loads into SrcA/SrcB registers
- **TRISC1**: Runs math -- issues matrix multiply instructions to the FPU (8x16 @ 16x16 primitive, tiled to build 32x32 output)
- **TRISC2**: Runs pack -- reads results from Dst register, converts data format, writes back to L1 CBs
- **NCRISC**: Runs the writer kernel -- DMA writes output tiles from L1 to DRAM

The circular buffers keep everyone synchronized. BRISC stays ahead of TRISC0, which stays ahead of TRISC1, etc. This is a classic software pipeline.

### 6. Completion
When all cores finish, the dispatch core detects completion (via semaphores or mailbox polling) and updates the command queue status. If the host called with `blocking=True`, it was waiting for this signal.

## Q: How does `execute_trace` work at the hardware level?

**A:** Metal Trace is an optimization that eliminates host-side dispatch overhead by recording and replaying command sequences entirely on-device.

### Capture phase
```python
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
output = model(input_tensor)  # ops are recorded, not fully executed
ttnn.end_trace_capture(device, trace_id, cq_id=0)
```

During capture, TT-Metal intercepts all the commands that would normally be written to the command queue (EnqueueProgram, buffer configurations, etc.) and instead writes them into a **dedicated DRAM buffer on the device**. This buffer holds the exact byte sequence of dispatch commands.

Key constraint: all operation parameters are **statically baked** into the trace -- tensor shapes, memory addresses, data types, CB configs. Nothing can change between replays.

### Replay phase
```python
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
```

When you replay, the dispatch core does not need to receive new commands from the host. Instead, it reads the pre-recorded command sequence directly from the DRAM buffer and re-executes it. The host sends a single "replay trace X" command, and the device does everything else.

This eliminates:
- **Host-side op construction**: No Python/C++ overhead to build each op's arguments
- **Host-to-device command transfer**: No PCIe latency per op
- **Kernel recompilation/relookup**: Binaries are already in the trace buffer

What it does NOT eliminate:
- **Actual device execution time**: The kernels still run
- **Data transfer for new inputs**: You still need to `copy_host_to_device_tensor()` before replaying

### Why this matters
From our experiment 12 results, trace replay gives 2-3x speedup on dispatch-dominated workloads. For a 10-op chain on small tensors, per-op cost drops from 25us (eager) to 9us (traced) -- the 16us savings per op is pure dispatch overhead eliminated.

The trace buffer occupies DRAM space (configured via `trace_region_size`), trading memory for speed. This is the same trade-off as CUDA Graphs.

## Q: How does this compare to GPU dispatch?

**A:**

| Aspect | GPU (CUDA) | Tenstorrent (Metalium) |
|--------|-----------|----------------------|
| Dispatch unit | GPU command processor (fixed-function HW) | Dedicated RISC-V core on Ethernet tile (programmable) |
| Command queue | Ring buffer in system memory, GPU reads via PCIe | Similar: host writes, device RISC-V core reads |
| Kernel loading | Driver loads to GPU instruction memory | Dispatch core DMA-writes binaries to each Tensix L1 |
| Kernel launch | Single warp scheduler starts all thread blocks | Dispatch core releases each Tensix from soft-reset |
| Trace/graph | CUDA Graphs: record + replay command stream | Metal Trace: record commands to DRAM buffer + replay |
| Parallelism | 1000s of threads hide latency (SIMT) | 5 cores per tile, deterministic pipeline (no latency hiding) |

The key difference: Tenstorrent's dispatch is *software-defined* (a RISC-V core running firmware), not fixed-function hardware. This means it can be updated and optimized in firmware, but it also means dispatch overhead is higher than a dedicated hardware scheduler -- which is exactly why Metal Trace matters so much.

## Q: What is the BRISC firmware loop?

**A:** Each RISC-V core on a Tensix tile runs a firmware loop provided by TT-Metal. The general pattern is:

1. **Boot**: Core comes out of soft-reset, begins executing at address 0
2. **Init**: Configure the Macro-Op Expander, Replay Expander, Tensix Scalar GPRs, and relevant configuration registers
3. **Wait for work**: Poll a mailbox/semaphore for the dispatch core to signal "kernel is ready"
4. **Run kernel**: Execute the user's kernel code (reader/compute/writer)
5. **Signal completion**: Write to a mailbox/semaphore to indicate the kernel finished
6. **Loop**: Go back to step 3, waiting for the next kernel

The profiler distinguishes between `BRISC-FW` (the entire firmware loop iteration) and `BRISC-KERNEL` (just the user kernel execution within one iteration). The difference is firmware overhead -- init, mailbox polling, cleanup.

This firmware loop means that kernel binaries don't need to include boot code or hardware init -- they are just the "main function" that the firmware calls. The dispatch core loads the binary, signals the firmware, and the firmware calls into the kernel.

## Q: What are the key memory regions in a Tensix tile's L1?

**A:** The 1.5 MB L1 SRAM per Tensix tile is partitioned (not cached -- fully software-managed):

| Region | Address range | Size | Purpose |
|--------|--------------|------|---------|
| Shared L1 SRAM | Base of tile | ~1464 KiB | Circular buffers, tensor data, kernel binaries |
| Core-local RAM | `0xFFB00000` | 2-4 KiB per core | Per-core scratch space |
| NC instruction RAM | `0xFFC00000` | 16 KiB | NCRISC instruction memory |
| Tensix Scalar GPRs | `0xFFE00000` | 64 per pipe x 3 pipes | Configuration and control registers |
| Tensix instruction inject | `0xFFE40000` | N/A | MMIO write target for Tensix instructions |
| Soft-reset register | `0xFFB121B0` | 4 bytes | Controls which cores are in reset (bit per core) |

The soft-reset register at `0xFFB121B0` is critical for dispatch: writing `0x47800` puts all 5 cores in reset; clearing individual bits releases specific cores to start execution.

## Summary for tt-xla

Understanding this dispatch pipeline is essential for building a JAX backend:

1. **Minimum viable path**: Map StableHLO ops to TT-NN calls, wrap in trace capture, replay per invocation. This is "Level 1" from wiki entry 12.
2. **The dispatch overhead is real**: ~25us per op in eager mode. For a 100-op model, that is 2.5ms of pure overhead. Trace cuts this dramatically.
3. **The 3-kernel decomposition is mandatory**: Any custom kernels we write (if we go beyond TT-NN) must follow the reader/compute/writer pattern.
4. **Sharding decisions matter more than "put it in L1"**: As we learned in wiki 08, interleaved L1 is slower than DRAM. The dispatch system needs to set up correct shard configs.

## Sources

- Corsix blog Part 5 (T tile internals, RISC-V cores, Tensix pipeline): https://www.corsix.org/content/tt-wh-part5
- Corsix blog Part 3 (soft-reset, cycle counters): https://www.corsix.org/content/tt-wh-part3
- clehaxze blog (programming model, fast dispatch): https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi
- TT-Metalium Guide: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- TT-Metal Advanced Performance Optimizations: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md
- TT-Metalium Compute Engines and Data Flow: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/compute_engines_and_dataflow_within_tensix.html
- TT-Metalium Watcher docs (core naming): https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/watcher.html
- Our research notes: `research/01_tenstorrent_hardware.md`, `research/02_tenstorrent_software_stack.md`
- Our experiment results: wiki entries 08 (memory hierarchy) and 12 (trace capture)
