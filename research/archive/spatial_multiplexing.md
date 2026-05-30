# Spatial Multiplexing and Tile-Level Partitioning on Tenstorrent Blackhole

Research date: 2026-04-22
Context: Stanford CS440LX — TT-XLA project
Target hardware: Blackhole P150 (currently 11x10 = 110 usable Tensix cores after harvesting)

**Note on core counts:** Blackhole was designed with 140 Tensix cores. Our P150 reports 11x10 = 110 cores via `compute_with_storage_grid_size()`. As of January 2026, Tenstorrent firmware v19.5.0+ downgrades all P150 cards to 120 Tensix cores (from 140), with further harvesting reducing usable compute cores. The exact grid geometry depends on which rows/columns are harvested.

---

## 1. What Is Spatial Multiplexing on Blackhole?

### The Core Idea

Spatial multiplexing means partitioning the physical core grid of a single Blackhole chip so that different subsets of cores run different, independent workloads simultaneously. Instead of all 110 cores working on the same matmul or the same model forward pass, you divide the grid into regions — say 60 cores for workload A and 50 cores for workload B — each operating independently.

This is fundamentally enabled by Blackhole's architecture:

- **Independent cores.** Each Tensix core is a self-contained compute unit with 5 RISC-V processors, its own 1.5MB L1 SRAM, matrix engine, vector engine, and pack/unpack units. There is no hardware warp scheduler or shared execution unit that couples cores together. Each core operates independently unless the program explicitly coordinates them.
- **Explicit work partitioning.** Unlike CUDA where a GPU kernel launches and the hardware schedules warps across SMs, TT-Metalium requires the programmer to explicitly assign work to specific cores via `CoreRange` and `CoreRangeSet`. This is both a burden and an opportunity — it means you have fine-grained control over which cores do what.
- **2D torus NoC.** The two NoCs (NoC 0 and NoC 1, traversing in opposite directions) provide full connectivity. Any core can communicate with any other core. There is no architectural barrier to having core (0,0)-(5,9) running one program while core (6,0)-(10,9) runs another, because the NoC allows both groups to independently access DRAM.

### Why This Matters

Our current approach uses all 110 cores for a single inference workload (e.g., Llama-3.1-8B decode). This is correct for maximizing single-request latency. But several scenarios demand spatial multiplexing:

1. **Multi-tenant serving** — running two different models (or model sizes) on one chip
2. **Prefill/decode disaggregation** — separating compute-bound prefill from memory-bound decode on the same chip
3. **MoE expert parallelism** — assigning different experts to different core groups
4. **Utilization recovery** — if a workload only saturates 60 cores, using the remaining 50 for something else

### Architectural Comparison: Blackhole vs. GPU

On a GPU, spatial multiplexing is hard because:
- SMs share L2 cache, memory controllers, and warp schedulers
- The hardware decides which SMs run which warps
- Isolation requires firmware/hardware support (MIG on Ampere+)

On Blackhole, spatial multiplexing is architecturally natural because:
- Each core has private L1 with no shared cache hierarchy
- Work assignment to cores is explicit in the programming model
- The NoC provides uniform connectivity without shared bottlenecks (beyond DRAM bandwidth)
- DRAM is interleaved across 24 controllers, so different core groups can access different banks

The challenge is not hardware — it is software. The TT-Metalium and TT-NN software stacks are primarily designed for single-program-all-cores execution patterns.

---

## 2. TT-NN / Metalium APIs for Core Grid Partitioning

### CoreRange, CoreRangeSet, CoreCoord

These are the fundamental building blocks for specifying which cores execute a kernel:

```python
# A single core
core = ttnn.CoreCoord(x=3, y=5)

# A rectangular range of cores (inclusive)
core_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),   # top-left
    ttnn.CoreCoord(5, 9)    # bottom-right: 6x10 = 60 cores
)

# A set of (possibly non-contiguous) core ranges
core_range_set = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4)),  # left half, 30 cores
    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(10, 4)), # right half, 25 cores
])
```

Every kernel in TT-Metalium is created with a `core_spec` parameter (`CoreCoord`, `CoreRange`, or `CoreRangeSet`) that controls which cores execute it. This is the foundation for spatial partitioning.

### compute_with_storage_grid_size()

This device method returns the maximum usable compute grid:

```python
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
# Returns CoreCoord(x=11, y=10) on our P150 — 110 usable cores
```

This accounts for harvested cores. It defines the upper bound of the grid you can address. All our current workloads use this full grid.

### Sharded Memory Configs with Core Grids

When creating sharded memory configurations, you specify exactly which cores hold the data:

```python
# Shard data across a 6x5 subgrid (30 cores) instead of all 110
sub_grid = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 4))
])
mem_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, shard_w),
    core_grid=sub_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True
)
```

This is already a form of spatial partitioning — you are telling TT-NN to only use 30 cores for this tensor's storage and computation.

### Matmul Program Configs

Matmul operations accept explicit grid specifications:

```python
config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(6, 5),  # only use a 6x5 subgrid
    # ... other params
)
result = ttnn.matmul(a, b, program_config=config)
```

The `core_grid=ttnn.CoreGrid(x, y)` shorthand on `ttnn.matmul` also controls this. Our Experiment 40 demonstrated that default matmul sometimes only uses 22-24 cores even on P150, and explicitly specifying the full grid gives speedups.

### SubDeviceId (Emerging API)

The TT-Metalium API includes a `SubDeviceId` parameter on `CreateBuffer` and `ttnn.to_device`:

```python
# Allocate buffer on a specific sub-device partition
buffer = tt_metal.CreateBuffer(config, sub_device_id=sub_device_id)

# Move tensor to a specific sub-device
tensor = ttnn.to_device(tensor, device, sub_device_ids=[sub_device_id])
```

This is the most direct API for true spatial multiplexing — it allows you to define sub-device partitions (subsets of the core grid) and allocate resources independently to each. However, the documentation for this API is sparse and it appears to be under active development. The exact mechanism for creating `SubDeviceId` instances (via a `sub_device_manager` or similar) is not well-documented in public docs as of April 2026.

### num_cores_to_corerangeset Utility

```python
crs = ttnn.num_cores_to_corerangeset(
    num_cores=32,
    grid_size=ttnn.CoreCoord(11, 10),
    row_wise=True
)
```

This utility maps a desired core count to a `CoreRangeSet`, handling non-rectangular layouts. Useful for splitting work across a subset of cores.

---

## 3. Multi-Program Multi-Data (MPMD)

### Can Different Cores Run Different Programs?

**Yes, architecturally.** Each Tensix core runs its own set of 3 kernels (reader, compute, writer) independently. There is no constraint that all cores must run the same kernel binary. The `CreateKernel` API takes a `core_spec` parameter:

```cpp
// Metalium C++ API
auto kernel_a = CreateKernel(program, "kernel_a.cpp",
    CoreRange({0,0}, {5,9}));   // cores 0-5 x 0-9 run kernel A

auto kernel_b = CreateKernel(program, "kernel_b.cpp",
    CoreRange({6,0}, {10,9}));  // cores 6-10 x 0-9 run kernel B
```

Within a single `Program`, you can assign different kernels to different core ranges. Different cores can also receive different runtime arguments:

```cpp
// Each core gets unique arguments
SetRuntimeArgs(program, kernel_id, core_coord, {arg0, arg1, ...});
```

### Single Program, Multiple Kernels

The standard pattern in TT-Metalium is one `Program` containing multiple kernel assignments across different core groups. The program is dispatched as a unit via `EnqueueProgram`. All assigned cores begin execution, but each runs its own kernel with its own arguments.

This is inherently MPMD within a single program dispatch — the "M" programs are different kernel binaries assigned to different core ranges.

### Multiple Programs Sequentially

The command queue (`EnqueueProgram`) is a FIFO. You can enqueue multiple programs back-to-back:

```python
ttnn.execute_trace(device, trace_id_a)  # program A on all cores
ttnn.execute_trace(device, trace_id_b)  # program B on all cores
```

But these execute **sequentially**, not concurrently. The command queue serializes execution.

### True Concurrent Programs: The Open Question

Running two fully independent programs concurrently on non-overlapping core subsets of the same device is the key open question. The architecture supports it (cores are independent), but the current software stack appears to enforce a single-program-at-a-time model through the command queue.

Potential paths to concurrent execution:
1. **Multiple command queues** — `ttnn.open_device(device_id=0, num_command_queues=2)` supports opening two queues. If each queue can dispatch to a non-overlapping core subset, this enables concurrency. However, documentation suggests this is primarily for overlapping compute and data transfer, not for running two independent compute programs.
2. **SubDevice partitions** — The emerging `SubDeviceId` API may enable independent dispatch to different core partitions.
3. **Single program, divergent kernels** — Pack both workloads into one program with different kernels on different core ranges. This gives spatial multiplexing within the program dispatch model but requires careful orchestration.

---

## 4. Use Cases

### 4.1 Running Two Different Models on the Same Chip

**Scenario:** Serve Qwen-0.5B (small, fast) and Llama-8B (large, slow) on one P150.

**Feasibility analysis:**
- Qwen-0.5B weights: ~1GB in bf16. Could fit in 30 cores' L1 partially, but weights are in DRAM regardless.
- Llama-8B weights: ~16GB in bf16. Also in DRAM.
- Both models share the same 32GB DRAM and 24 DRAM controllers.
- The bottleneck is DRAM bandwidth (450 GB/s), not compute cores.

**Approach:**
- Assign cores (0,0)-(5,9) = 60 cores to Model A, (6,0)-(10,9) = 50 cores to Model B.
- Each model uses its own `CoreRangeSet` for sharded memory and matmul configs.
- DRAM bandwidth is shared — this is the critical constraint. Two memory-bound decode workloads would compete for the same 450 GB/s.

**Verdict:** Possible at the kernel level, but DRAM bandwidth contention limits the benefit. Most useful when one model is idle (e.g., waiting for user input) while the other is generating.

### 4.2 Prefill and Decode on Separate Core Groups

**Scenario:** Disaggregate prefill (compute-bound, large batch of input tokens) from decode (memory-bound, one token at a time) on the same chip.

**Why this is compelling:**
- Prefill is compute-bound: large matmuls with many tokens, high arithmetic intensity. Benefits from many cores doing parallel compute.
- Decode is memory-bound: tiny activations (batch=1), streaming weights from DRAM. Limited by DRAM bandwidth, not core count.
- When mixed, prefill operations block decode, causing latency spikes (the "prefill-decode interference" problem well-studied in GPU serving).

**On Blackhole:**
- Give 80 cores to prefill (compute-heavy), 30 cores to decode (bandwidth-limited).
- Decode only needs enough cores to pipeline DRAM reads; using all 110 cores for batch-1 decode doesn't help since the bottleneck is DRAM bandwidth (our Experiment 40 showed diminishing returns past ~30 cores for small matmuls).
- Prefill benefits from more cores (more parallel compute on the large token batch).

**Challenge:** Still requires concurrent program execution (two programs running simultaneously on non-overlapping cores), which the command queue may serialize. The single-program divergent-kernel approach could work but is complex to orchestrate for full model forward passes.

### 4.3 Expert Parallelism for MoE

**Scenario:** For Qwen-2.5-MoE or Mixtral, assign different experts to different core groups.

**This is the most natural fit for spatial multiplexing on Blackhole:**
- MoE models activate only k of N experts per token (e.g., top-2 of 8 experts in Mixtral).
- Each expert is an independent MLP (gate + up + down projections).
- With 110 cores and 8 experts, assign ~13 cores per expert.
- The router selects which 2 experts to activate; those core groups compute while others are idle.
- Inter-expert communication is minimal (just the router output and the weighted sum of expert outputs).

**Implementation sketch:**
1. All cores run the shared attention layers using the full grid (standard mode).
2. At the MoE layer, the router (small matmul) runs on a few cores and produces top-k expert indices.
3. Each expert's MLP weights are sharded across its assigned core group's L1 or DRAM region.
4. Only the selected expert core groups activate for each token.
5. Results are gathered via NoC and combined.

**This maps well to TT-Metalium's MPMD model:** a single program with different expert kernels assigned to different `CoreRange` groups, with runtime arguments selecting which experts are active.

**Key concern:** With top-2 routing, at any given time only 2 of 8 core groups are active (26 of 110 cores). The other 84 cores are idle. This is acceptable because MoE's entire premise is conditional computation — you trade utilization for capacity. But it means spatial multiplexing for MoE does not improve hardware utilization; it enables the MoE computation pattern itself.

---

## 5. Profiling and Utilization Tools

### 5.1 Device Program Profiler

The built-in device profiler provides per-core, per-RISC-V visibility into execution timelines.

**Setup:**
```bash
# Enable profiling
export TT_METAL_DEVICE_PROFILER=1

# Run your program
python my_model.py

# Output: generated/profiler/.logs/profile_log_device.csv
```

**What it captures:**
- Start and end cycle counts for marked code zones on every RISC-V core (all 5 baby RISCs per Tensix)
- Per-core execution duration
- Can identify which cores are active/idle during each operation
- Works on Ethernet cores too, not just Tensix

**Key metrics:**
- `DEVICE FW START CYCLE` — earliest RISC of earliest core
- `DEVICE FW END CYCLE` — latest RISC of latest core
- `DEVICE FW DURATION` — wall-clock cycles across all cores
- `DEVICE KERNEL DURATION` — kernel-only cycles

The CSV output enables automated analysis to identify which cores are underutilized.

### 5.2 Tracy Profiler Integration

TT-Metalium integrates with Tracy, a real-time nanosecond-resolution profiler.

**Capabilities:**
- Interactive timeline visualization of every RISC-V core on every Tensix
- Real-time streaming of device profiling data to a Tracy client
- Per-core execution traces showing exactly when each core starts/finishes each kernel
- CPU-side profiling (host dispatch overhead, PCIe transfer times)
- Memory allocation tracking

**Usage:**
```bash
# Build with Tracy support
# Run with Tracy client connected
# See per-core timelines in the Tracy UI
```

Tracy is the best tool for visually identifying spatial utilization patterns — you can literally see which cores are active at each point in time and identify idle core regions.

### 5.3 tt-npe: NoC Performance Estimator

[tt-npe](https://github.com/tenstorrent/tt-npe) is a lightweight NoC simulator developed by Tenstorrent.

**What it does:**
- Simulates NoC workloads on Tensix-based devices (supports Blackhole)
- Models bandwidth derating from congestion between concurrent transfers
- Outputs per-transfer timing, congestion analysis, and bandwidth utilization

**Integration with profiling:**
```bash
# Collect NoC traces during profiling
tt-perf-report --collect-noc-traces my_trace.json
```

This adds `DRAM BW UTIL` and `NOC UTIL` columns to the ops performance report — directly answering "are we bandwidth-limited?"

**Visualization:**
The `--collect-noc-traces` option creates timeline files in an `npe_viz/` subdirectory, viewable in the TT-NN Visualizer's NPE tab. This shows per-transfer NoC traffic, congestion zones, and bandwidth saturation across the chip.

### 5.4 tt-perf-report

[tt-perf-report](https://github.com/tenstorrent/tt-perf-report) is the top-level performance analysis tool.

**Key columns:**
- `Device Time` — time on device in microseconds
- `Op-to-op Gap` — host overhead between operations
- `Total %` — percentage of total execution time
- `Cores` — number of cores used by each operation (max 64 on Wormhole, higher on Blackhole)
- `DRAM BW UTIL` — DRAM bandwidth utilization (with tt-npe)
- `NOC UTIL` — NoC utilization (with tt-npe)

**Optimization guidance:**
The tool classifies each matmul as DRAM-bound, FLOP-bound, or SLOW, and provides tailored advice:
- DRAM-bound: use DRAM-sharded configs
- FLOP-bound: increase core count
- SLOW: optimize memory layouts and block sizes

### 5.5 TT-NN Visualizer

[ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer) is a comprehensive GUI tool for visualizing model execution:
- Interactive operation flow graphs
- Memory plots showing buffer allocation across cores
- Tensor detail views
- NoC traffic visualization (via tt-npe integration)
- Multi-instance support for comparing runs

---

## 6. Hardware Utilization Saturation — Are We Using All 110 Cores?

### The Uncomfortable Truth

For batch-1 decode (our primary workload), we are almost certainly **not** effectively using all 110 cores. Here is why:

**Batch-1 decode is memory-bound, not compute-bound.** Each matmul reads a full weight matrix from DRAM (e.g., 4096x4096 = 32MB for bf16) but only computes a tiny dot product (1x4096 @ 4096x4096). The arithmetic intensity is ~0.5 FLOPs/byte — far below the machine's compute-to-bandwidth ratio.

**What this means for core utilization:**
- The 110 cores collectively have much more compute capacity than needed for batch-1 matmuls.
- Adding more cores does not help because the bottleneck is DRAM bandwidth (450 GB/s), not compute.
- Our Experiment 40 confirmed this: for small matmuls (decode-sized), using 22 vs 88 cores showed modest differences because both are bandwidth-limited.

**When do we saturate all 110 cores?**
- Large batch sizes (batch=8, 16, 32+) where activations are big enough to distribute meaningful work across all cores.
- Prefill with long sequences (seq_len=512, 1024+) where the matmul is genuinely compute-bound.
- Large matmuls where the arithmetic intensity exceeds the bandwidth limit.

### How to Measure

1. **tt-perf-report `Cores` column** — shows how many cores each op actually uses. If your decode matmuls report 22 cores, you have 88 idle cores.
2. **Tracy timeline** — visually shows idle cores as gaps in the per-core timeline.
3. **Back-of-envelope calculation:**
   - Decode-1 weight bandwidth: ~315MB per forward pass (all weight matrices).
   - At 450 GB/s DRAM BW: ~0.7ms theoretical minimum per token.
   - Our measured: ~7ms per token (with trace).
   - The 10x gap is dispatch overhead, memory layout inefficiency, and serialized op execution — not compute starvation.

### Implications for Spatial Multiplexing

This analysis actually **argues for** spatial multiplexing: if batch-1 decode only needs 30-40 cores to saturate DRAM bandwidth, the remaining 70-80 cores are wasted. Running a second workload on them (another decode stream, a prefill, or a different model) would improve total chip utilization without significantly impacting the first workload — as long as DRAM bandwidth is shared fairly.

---

## 7. Comparison to NVIDIA GPU Partitioning

### NVIDIA Multi-Process Service (MPS)

**What it is:** A runtime service that enables multiple CUDA processes to share a single GPU concurrently via spatial sharing of SMs.

**How it works:**
- Processes share the same GPU context (no context switching overhead)
- Kernels from different processes can run on different SMs simultaneously
- No memory isolation — one process crash resets the entire GPU
- No bandwidth partitioning — all processes share L2 cache and memory controllers

**Comparison to Blackhole spatial multiplexing:**
| Aspect | NVIDIA MPS | Blackhole Spatial |
|--------|-----------|-------------------|
| Isolation | None (shared context) | Strong (private L1 per core) |
| Memory protection | None | Inherent (separate L1 SRAM) |
| Failure blast radius | Entire GPU | Potentially per-core-group |
| Setup | Runtime service, transparent | Explicit core assignment in code |
| Granularity | SM-level (implicit) | Core-level (explicit) |
| DRAM bandwidth sharing | Shared, unpartitioned | Shared, but interleaved across 24 controllers |

### NVIDIA Multi-Instance GPU (MIG)

**What it is:** Hardware-level GPU partitioning (Ampere+) that creates fully isolated GPU instances with dedicated compute, memory, and bandwidth.

**How it works:**
- Physically partitions SMs, L2 cache slices, and memory controllers
- Each instance is a separate GPU from the software's perspective
- Fixed partition profiles (e.g., A100 supports 1/2/3/4/7 instances)
- Full error isolation between instances

**Comparison to Blackhole:**
| Aspect | NVIDIA MIG | Blackhole Spatial |
|--------|-----------|-------------------|
| Isolation | Full hardware isolation | Software-enforced via CoreRange |
| Flexibility | Fixed profiles only | Arbitrary core groupings |
| Memory partitioning | Dedicated memory per instance | Shared DRAM, private L1 |
| Bandwidth guarantee | Dedicated memory controllers | No bandwidth QoS (shared DRAM) |
| Maximum instances | 7 (A100) | Theoretically up to 110 (one per core) |
| Reconfiguration | Requires GPU reset | Could be per-program (no reset) |

### The Blackhole Advantage

Blackhole's architecture is inherently more flexible than GPU partitioning:
- **Arbitrary geometry.** You can create any rectangular (or non-contiguous) partition, not just fixed profiles.
- **No reconfiguration cost.** Changing which cores run which workload is a software decision, not a hardware reconfiguration.
- **Private L1 SRAM.** Each core has 1.5MB of private, directly-addressable SRAM. There is no shared cache to partition or contend over.
- **NoC vs. bus.** The 2D torus NoC provides more uniform communication than a bus-based GPU interconnect.

### The Blackhole Disadvantage

- **No DRAM bandwidth partitioning.** MIG guarantees each instance its own memory controllers. Blackhole's 24 DRAM controllers are shared across all core groups. Two bandwidth-hungry workloads will contend.
- **Software maturity.** MIG and MPS are production-grade with years of tooling. Blackhole's spatial multiplexing APIs (SubDeviceId, etc.) are nascent.
- **No hardware QoS.** There is no hardware mechanism to guarantee a core group gets a minimum share of DRAM bandwidth. A badly-behaved workload on one core group can starve others.

---

## 8. Key Open Questions and Experiments

### Experiment Ideas (to run on remote host)

1. **Core grid subsetting.** Run the same matmul on a 5x5 grid vs 11x10 grid. Measure throughput ratio. If 5x5 gives >45% of full-grid throughput for decode-sized matmuls, spatial multiplexing is viable.

2. **Concurrent kernel divergence.** In a single program, assign kernel A to cores (0,0)-(5,9) and kernel B to cores (6,0)-(10,9). Verify both execute and produce correct results. This proves MPMD within a single dispatch.

3. **DRAM bandwidth contention.** Run two independent matmul streams on non-overlapping core groups. Measure if total throughput exceeds single-stream throughput (it should if compute-bound) or matches it (if bandwidth-bound).

4. **Dual command queue.** Open device with `num_command_queues=2`. Attempt to dispatch different programs to different queues targeting non-overlapping cores. Measure if they execute concurrently.

5. **SubDeviceId exploration.** Investigate the SubDevice API — create two sub-devices, allocate buffers on each, run operations on each independently.

### Research Questions

- Does the TT-Metalium dispatcher support true concurrent program execution on non-overlapping core ranges, or does it serialize all dispatches?
- What is the DRAM bandwidth degradation when two core groups issue independent read streams?
- Can trace capture work with sub-device partitions? (Trace is critical for our decode performance.)
- Is there a way to pin specific DRAM banks to specific core groups for bandwidth isolation?
- How does the firmware handle program dispatch to partial core grids — does it barrier on all cores or only the assigned ones?

---

## 9. Summary

| Feature | Status on Blackhole |
|---------|-------------------|
| Per-core kernel assignment | Supported (CoreRange + CreateKernel) |
| Different runtime args per core | Supported (SetRuntimeArgs) |
| MPMD within single program | Supported (different kernels on different CoreRanges) |
| True concurrent programs | Uncertain (command queue may serialize; SubDevice API emerging) |
| Sharded memory on core subsets | Supported (CoreRangeSet in memory configs) |
| Matmul on core subsets | Supported (core_grid / program config) |
| SubDevice partitioning API | Exists but poorly documented |
| DRAM bandwidth isolation | Not supported (shared controllers) |
| Per-core profiling | Supported (Device Profiler, Tracy) |
| NoC bandwidth monitoring | Supported (tt-npe) |

**Bottom line:** Blackhole's architecture is a natural fit for spatial multiplexing — independent cores, explicit work assignment, private L1 SRAM, and uniform NoC connectivity. The hardware enables it. The software stack is catching up, with `CoreRange`-level control already available and `SubDeviceId` APIs emerging. The main limitation is shared DRAM bandwidth, which cannot be partitioned or guaranteed. For our project, the most immediately actionable use case is MoE expert parallelism, where different experts naturally map to different core groups within a single program dispatch.

---

## Sources

- [TT-Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metalium Multi-Core Matmul Example](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core.html)
- [TT-Metalium CreateKernel API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis/kernels/CreateKernel.html)
- [TT-Metalium CreateBuffer API](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/host_apis/buffers/CreateBuffer.html)
- [TT-NN API Reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html)
- [TT-NN Sharded Memory Config](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.create_sharded_memory_config.html)
- [Device Program Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
- [Tracy Profiler for TT-Metalium](https://docs.tenstorrent.com/tt-metal/v0.55.0/tt-metalium/tools/tracy_profiler.html)
- [tt-npe: NoC Performance Estimator](https://github.com/tenstorrent/tt-npe)
- [tt-perf-report](https://github.com/tenstorrent/tt-perf-report)
- [TT-NN Profiling Operations](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/profiling_ttnn_operations.html)
- [ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer)
- [Blackhole & TT-Metalium — Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf)
- [Dissecting the Tenstorrent Blackhole Architecture via Microbenchmarking (ASPLOS)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Programming Tenstorrent Processors (clehaxze)](https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi)
- [Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [TT-Metal Multi-Device Programming](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md)
- [TT-Fabric Architecture](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Fabric/TT-Fabric-Architecture.md)
- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/concepts.html)
- [NVIDIA MPS Documentation](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
- [Tenstorrent P150 Core Count Update (Tom's Hardware)](https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop)
- [DeepWiki: TTNN Python API](https://deepwiki.com/tenstorrent/tt-metal/4.1-ttnn-python-api-and-configuration)
