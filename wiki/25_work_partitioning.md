# Wiki 25: How TT-NN Distributes Work Across Blackhole Cores

## Q: What is the physical core layout on our Blackhole device?

**A:** Our device reports `compute_with_storage_grid_size()` = 11x10 = **110 usable Tensix cores**.

Blackhole's physical silicon has a 14x10 grid (140 Tensix cores). The UMD logs show:
```
Harvesting masks for chip 0 tensix: 0xc0 dram: 0x0 eth: 0x120
```

The harvesting mask `0xc0` = `0b11000000` means 2 Tensix columns are harvested (disabled for yield). Additionally, one column is reserved for dispatch, leaving 14 - 2 - 1 = 11 usable columns, times 10 rows = **110 compute cores**.

Other resources on the chip:
- **8 DRAM banks** (`dram_grid_size()` = 8x1), positioned along the chip edges
- **1.5 MB L1 SRAM per core** (confirmed via `l1_size_per_core()` on earlier firmware; not directly accessible on current MeshDevice API, but documented as 1.5 MB)
- **2 NoC networks** (NOC0 and NOC1) connecting all cores in a 2D mesh
- Each core has 5 RISC-V processors: RISC0 (reader/NoC), RISC1 (writer/NoC), and 3 compute RISCs (unpack, math, pack)

Note: As of firmware 19.5+, all p150 cards are being standardized to 120 cores (harvesting 20). Our device still reports 110, which may reflect a different harvesting configuration.

## Q: What memory layout do our operations use by default?

**A:** Experimentally confirmed -- all our ops use **INTERLEAVED DRAM** by default:

```
a mem:         MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM)
matmul out:    MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM)
relu out:      MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM)
softmax out:   MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED, buffer_type=BufferType::DRAM)
```

**INTERLEAVED** means tensor tiles are round-robined across all 8 DRAM banks at page-size granularity. This ensures balanced bandwidth utilization across DRAM controllers -- no single bank gets hot-spotted. But it means every operation must stream data from DRAM to L1 and back.

The alternative memory layouts are:
| Layout | Buffer | Description |
|--------|--------|-------------|
| `INTERLEAVED` | `DRAM` | Tiles round-robin across 8 DRAM banks (our default) |
| `INTERLEAVED` | `L1` | Tiles round-robin across all 110 core L1s |
| `HEIGHT_SHARDED` | `L1` | Rows of tiles distributed across cores (1D partition along M) |
| `WIDTH_SHARDED` | `L1` | Columns of tiles distributed across cores (1D partition along N) |
| `BLOCK_SHARDED` | `L1` | 2D block partition: rows along core-y, columns along core-x |

Sharded L1 is where the real performance lives -- data is pre-placed in each core's local SRAM, eliminating DRAM round-trips between ops. Our transformer currently does not use sharding.

## Q: How does TT-NN parallelize matmul across cores?

**A:** TT-NN has multiple matmul "program factories" that implement different parallelization strategies. The key ones exposed in Python:

1. **`MatmulMultiCoreReuseProgramConfig`** -- Basic multi-core with data reuse
2. **`MatmulMultiCoreReuseMultiCastProgramConfig`** -- 2D grid with NoC multicast (the workhorse)
3. **`MatmulMultiCoreReuseMultiCast1DProgramConfig`** -- 1D multicast variant
4. **`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`** -- DRAM-sharded variant

When we call `ttnn.matmul(A, B)` without specifying a program config, TT-NN auto-selects based on tensor shapes, memory layout, and available cores.

### The 2D Multicast Pattern (MatmulMultiCoreReuseMultiCast)

For C = A @ B where A is [M, K] and B is [K, N] (in tiles of 32x32):

1. **Output is partitioned across a 2D core grid.** The output C is divided into blocks, with each core responsible for computing a `per_core_M x per_core_N` tile block of the output. For example, if M=128 (4 tiles), N=512 (16 tiles), the output might be distributed across a 4x16 grid where each core computes one output tile.

2. **Data flows via 2D multicast.** For each K-block iteration:
   - A row-block of A (tiles along M for this K-slice) is **multicast along the row** of cores -- all cores in the same row receive the same A tiles
   - A column-block of B (tiles along N for this K-slice) is **multicast along the column** of cores -- all cores in the same column receive the same B tiles
   - Each core locally multiplies its received A and B blocks and accumulates into its output partial sum

3. **The `transpose_mcast` flag** swaps which dimension uses row vs. column multicast. This matters because NoC multicast efficiency depends on the physical layout.

4. **Blocking parameters** control memory pressure:
   - `in0_block_w`: How many K-tiles to process at once (larger = fewer iterations, but more L1 needed)
   - `out_subblock_h/w`: Subblock dimensions for the FPU's accumulation registers
   - `out_block_h/w`: Block size in tiles per core (defaults to per_core_M/N)

### Concrete Example: Our Transformer's Matmul

For our transformer's attention QKV projection with shapes like [1,1,32,64] @ [1,1,64,64]:
- M=32 (1 tile), K=64 (2 tiles), N=64 (2 tiles)
- Output is 1x2 tiles -- only 2 cores are needed
- This is **massively underutilizing** the 110-core grid
- At these small sizes, dispatch overhead and DRAM streaming latency dominate, not compute

This is why our Wiki 24 showed raw matmul reaching 95 TFLOPS at 2048x2048 but the transformer only hitting 2.5 TFLOPS -- the matrices are too small to fill the core grid.

## Q: How does TT-NN parallelize elementwise ops (add, relu, exp, etc.)?

**A:** Elementwise operations use a simple **SPMD (Single Program, Multiple Data)** strategy:

1. The total number of output tiles is computed
2. `split_work_to_cores(core_grid, num_tiles)` divides tiles across available cores as evenly as possible
3. Each core runs the same kernel but operates on a different subset of tiles
4. If tiles don't divide evenly, some cores get one extra tile (split into `core_group_1` and `core_group_2`)

For **INTERLEAVED DRAM** inputs (our case):
- Each core uses its RISC0 (reader) to issue NoC reads, pulling its assigned tiles from DRAM into L1
- The compute RISC applies the elementwise function (FPU for simple ops, SFPU for transcendentals like exp, rsqrt)
- RISC1 (writer) sends results back to DRAM via NoC

For **SHARDED L1** inputs:
- Data is already in each core's L1 -- no DRAM read needed
- The compute kernel runs directly on local L1 data
- Results stay in L1 (no DRAM write if next op is also sharded)
- This is **embarrassingly parallel** with zero NoC traffic

The key insight: for elementwise ops, DRAM bandwidth is usually the bottleneck, not compute. The FPU/SFPU can process tiles far faster than DRAM can deliver them. Sharding eliminates this bottleneck entirely.

## Q: How does TT-NN parallelize reduction ops (sum, max)?

**A:** Reductions are more complex because they require cross-tile communication. The strategy depends on the reduction dimension.

### Reduction along the last dimension (e.g., `ttnn.sum(x, dim=-1)`)

For a tensor of shape [1,1,M,N] reducing along N:
1. Each row of tiles is assigned to a core (or group of cores)
2. Within each core, tiles along the N dimension are summed locally using the compute engine
3. For INTERLEAVED inputs, tiles are streamed from DRAM, accumulated in L1, and results written back
4. No cross-core communication is needed because each row reduces independently

### Reduction across the first dimension (e.g., `ttnn.sum(x, dim=0)`)

This requires combining data from tiles assigned to different cores. TT-NN handles this via:
1. Local partial reductions on each core
2. Cross-core data movement via NoC to combine partials

### What we don't know yet

We were unable to determine the exact reduction tree topology (is it a binary tree? A flat all-reduce? A ring?). The C++ source for the reduction program factories (`HCSumReduceProgramFactory` etc.) is compiled into `_ttnncpp.so` and not shipped as readable source. The documentation describes the `split_work_to_cores` function for dividing work, but doesn't detail the inter-core reduction pattern.

## Q: How does softmax work across cores?

**A:** Softmax requires three passes: max (reduction), subtract + exp (elementwise), sum (reduction) + divide (elementwise). TT-NN has dedicated program configs:

- **`SoftmaxDefaultProgramConfig`**: Auto-configured, uses INTERLEAVED memory
- **`SoftmaxShardedMultiCoreProgramConfig`**: For sharded tensors, with parameters:
  - `compute_with_storage_grid_size`: Core grid to use
  - `subblock_w`, `block_h`, `block_w`: Tiling parameters

For softmax along the last dimension (the common case, and what our transformer uses):
1. Each row of the tensor can be processed independently (the reductions are per-row)
2. Rows are distributed across cores
3. Within each core, the three-pass algorithm runs locally:
   - Pass 1: Find max across the row (local reduction)
   - Pass 2: Subtract max, compute exp (elementwise)
   - Pass 3: Sum exp values (local reduction), divide (elementwise)
4. No cross-core communication needed because each row is self-contained

This makes softmax surprisingly parallelizable -- for `softmax(x, dim=-1)`, the number of independent work items equals the number of rows, which scales with batch_size * seq_len * num_heads.

## Q: What profiling tools exist for seeing per-core activity?

**A:** TT-Metalium has two profiling systems:

### 1. Device Program Profiler
- Enable with: `TT_METAL_DEVICE_PROFILER=1`
- Uses `DeviceZoneScopedN(zone_name)` macros in C++ kernel code
- Generates CSV at `${TT_METAL_HOME}/generated/profiler/.logs/profile_log_device.csv`
- Shows start/end timestamps for each zone on each RISC-V core on each Tensix

### 2. Tracy Profiler (Tenstorrent fork)
- Captures both host-side Python/C++ and device-side RISC-V execution
- Shows execution timeline per-RISC per-core
- Requires building with Tracy support (not available in our pip-installed package)
- GitHub: `tenstorrent/tracy`

### 3. Python-level perf API
- `ttnn.get_latest_programs_perf_data()` -- returns perf data for recently executed programs
- `ttnn.get_all_programs_perf_data()` -- returns all perf data
- In our testing, these returned 0 entries, likely because profiling must be enabled at build time or via environment variable

### What we tested

We called `ttnn.get_latest_programs_perf_data()` after running matmul, relu, and softmax. It returned 0 entries. The profiling infrastructure exists but requires `TT_METAL_DEVICE_PROFILER=1` to be set before launching, and the Tracy-based profiler requires a special build.

## Q: What would it take to make our transformer use sharded memory?

**A:** Currently our transformer uses all-DRAM-interleaved memory. Every op streams tiles from DRAM, computes, and writes back to DRAM. The sequence looks like:

```
DRAM -> [matmul QKV] -> DRAM -> [reshape] -> DRAM -> [matmul QK^T] -> DRAM -> [softmax] -> DRAM -> [matmul AV] -> DRAM -> [matmul proj] -> DRAM -> [add residual] -> DRAM -> [layernorm] -> DRAM -> [matmul ff1] -> DRAM -> [relu] -> DRAM -> [matmul ff2] -> DRAM -> [add residual] -> DRAM -> [layernorm] -> DRAM
```

That is **16+ DRAM round-trips** per transformer layer. With sharded intermediates:

```
DRAM -> [matmul QKV] -> L1 -> [reshape] -> L1 -> [matmul QK^T] -> L1 -> [softmax] -> L1 -> [matmul AV] -> L1 -> [matmul proj] -> L1 -> [add residual] -> L1 -> [layernorm] -> L1 -> [matmul ff1] -> L1 -> [relu] -> L1 -> [matmul ff2] -> L1 -> [add residual] -> L1 -> [layernorm] -> DRAM
```

Requirements:
1. Specify `memory_config=ttnn.L1_MEMORY_CONFIG` or a sharded config for each op's output
2. For the multicast matmul to work with sharded inputs, the shard spec must match what the matmul program expects (HEIGHT_SHARDED for in0, WIDTH_SHARDED for in1, or BLOCK_SHARDED for both)
3. The shard dimensions must be tile-aligned (multiples of 32)
4. Total L1 usage per core must not exceed 1.5 MB (including input buffers, output buffers, and circular buffers for streaming)

This is the next big optimization opportunity. Our experiment 09 showed that L1 block-sharded elementwise ops can be significantly faster than DRAM-interleaved.

## Q: What is the data movement pattern for INTERLEAVED mode?

**A:** For an INTERLEAVED DRAM tensor:
1. Tiles are round-robined across 8 DRAM banks at page granularity
2. When an op runs, each core is assigned a range of tile indices
3. RISC0 (reader) on each core issues NoC read requests to the appropriate DRAM bank for each tile
4. Tiles arrive in L1 via the NoC into circular buffers
5. The compute pipeline processes tiles as they arrive (pipelined with data movement)
6. RISC1 (writer) sends output tiles back to DRAM via NoC

For INTERLEAVED L1 tensors, it's similar but tiles are round-robined across L1 of all 110 cores. A core may need to read tiles from other cores' L1 via NoC.

The critical bottleneck: **DRAM bandwidth**. With 8 DRAM banks, total DRAM bandwidth is the main constraint for bandwidth-bound ops. Each Tensix core has enough compute to process tiles far faster than DRAM can deliver them. This is why utilization is low for small tensors -- there isn't enough data to amortize the streaming overhead.

## Q: Summary -- where are the optimization opportunities?

**A:** Our transformer at 2,564 fwd/sec is leaving significant performance on the table:

| Issue | Impact | Fix |
|-------|--------|-----|
| Small matrices (32x64, 64x64) | Most cores idle, <3% of grid utilized | Batch multiple inputs, or increase model size |
| All DRAM-interleaved | 16+ DRAM round-trips per layer | Use sharded L1 for intermediates |
| No op fusion | Each op is a separate program dispatch | Use trace capture (already done) + sharded chains |
| No fused activations | relu after matmul is a separate op | Use `fused_activation` in matmul program config |
| Default program configs | TT-NN auto-selects, may not be optimal | Experiment with explicit program configs |

The biggest wins would come from:
1. **Sharded memory** -- eliminate DRAM round-trips between ops (potentially 2-4x for elementwise chains)
2. **Larger batch sizes** -- fill the core grid to improve utilization
3. **Fused activations** -- combine relu/gelu with preceding matmul

---

*Sources:*
- [Matmul Multi Core Example (TT-Metalium docs)](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core.html)
- [Data Reuse in matmul_multicore_reuse](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core_optimizations/data_reuse.html)
- [MatmulMultiCoreReuseMultiCastProgramConfig](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.MatmulMultiCoreReuseMultiCastProgramConfig.html)
- [Memory for Kernel Developers](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html)
- [Device Program Profiler](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/device_program_profiler.html)
- [Corsix: Tenstorrent Wormhole Part 7 - Bits of the MatMul](https://www.corsix.org/content/tt-wh-part7)
- [Tensor Layouts Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md)
- Experiments run on device 0 of our Blackhole p150 host
