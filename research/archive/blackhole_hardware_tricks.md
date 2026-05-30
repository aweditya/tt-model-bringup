# Blackhole Hardware Tricks: Actionable Optimizations for LLM Inference

**Date:** April 2026
**Hardware:** Tenstorrent Blackhole P150 (11x10 = 110 Tensix cores, 32 GB GDDR6, ~450 GB/s measured DRAM BW)
**Current performance:** ~136 tok/s (Qwen 0.5B), ~19 tok/s (Llama 8B), ~20 tok/s (MoE)

---

## 1. DRAM Bandwidth Tricks

### 1.1 DRAM-Sharded Weight Layout (HIGH PRIORITY)

The default interleaved layout round-robins tiles across all DRAM banks. For bandwidth-bound decode matmuls, this causes suboptimal access patterns. The fix is DRAM-sharded weights.

**How it works:**
- Shard weight tensors across DRAM banks so each reader core reads from exactly one bank
- Each DRAM bank has a dedicated reader kernel on an adjacent Tensix core
- Eliminates cross-bank NOC congestion from interleaved reads

**Key rules from TT's DRAM bandwidth tech report:**
- **One reader per bank maximum.** Multiple readers per bank cause "serious NOC congestion"
- **Place readers adjacent to their DRAM bank** to minimize return-route NOC hops
- **Use different NOC virtual channels** for readers on the same row to avoid route overlap
- **Pipeline NOC transaction tags:** Assign different tags to consecutive read blocks, barrier on completed tags while issuing new reads. Never back-to-back barrier the same block.

**API:**
```python
# Create DRAM-sharded memory config for weight matrix
dram_mem_config = ttnn.create_dram_sharded_mem_config(weight_shape, shard_dims)

# Use MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(...)
result = ttnn.matmul(act, weight, program_config=program_config, memory_config=dram_mem_config)
```

**Expected impact:** 10-20% improvement on bandwidth-bound matmuls. TT's own benchmarks show 83-92% DRAM bandwidth utilization with proper sharding vs. ~60-70% with interleaved.

**Experiment:** Convert one layer's Q/K/V/O projection weights to DRAM-sharded, benchmark before/after. Start with the largest matmul (gate_proj: 4096x14336 for 8B).

### 1.2 ROW_MAJOR Weight Storage (EASY WIN)

TT's LLM tech report recommends: "Converting weights to ROW_MAJOR_LAYOUT wrapped in TILE_WIDTH sticks eliminates padding overhead and improves DRAM access efficiency without runtime cost."

This is a weight preprocessing step during model loading -- zero runtime cost.

**Experiment:** Convert all weight tensors with `layout=ttnn.ROW_MAJOR_LAYOUT` at load time, verify matmul still works (it should automatically tilize during compute).

### 1.3 Alignment to DRAM Page Boundaries

Blackhole GDDR6 has native page sizes. NOC transactions up to 16 KB on Blackhole (vs 8 KB on Wormhole). Aligning tensor pages to these boundaries avoids partial page reads.

**Key numbers:**
- Blackhole max NOC transaction: 16 KB
- Tile size: 32x32 x 2 bytes (BF16) = 2 KB per tile
- Optimal: pack 8 tiles per NOC transaction (8 x 2 KB = 16 KB)

**Experiment:** Check if our tensor page sizes align to 16 KB boundaries. If not, pad shard shapes to match.

### 1.4 BFP4 MLP Weights (HIGHEST PRIORITY)

TT's reference implementations use BFP4 (4-bit block floating point) for MLP weights (gate, up, down projections). This halves the bytes read from DRAM for the largest weight matrices.

For Llama 8B:
- MLP weights per layer: 3 x (4096 x 14336) x 2 bytes = 336 MB
- With BFP4: 3 x (4096 x 14336) x 0.5 bytes = 84 MB
- Across 32 layers: saves 8 GB per decode step

**Expected impact:** 20-30% speedup. Moves from 16 GB to ~8 GB effective weight reads per step.

TT uses BFP4 for all MLP layers except the last decoder layer (which gets BFP8 for quality).

**Experiment:** Already partially tested (exp 83). Need full 8B validation with cosine similarity checks.

---

## 2. L1/SRAM Utilization Tricks

### 2.1 L1 Residency Between Ops (MEDIUM PRIORITY)

If consecutive ops use compatible sharding, intermediate tensors stay in L1 (1.5 MB per core, 165 MB total) with zero DRAM round-trips. This is the single most impactful architectural advantage of Tenstorrent.

**How to exploit it:**
- Use `memory_config=ttnn.L1_MEMORY_CONFIG` on op outputs that feed into the next op
- Ensure shard shapes are compatible (same core grid, same shard strategy)
- Chain: rms_norm (L1 out) -> matmul (L1 in, DRAM weight) -> silu (L1 in/out) -> matmul

**What stays in L1 (tiny, should always be L1):**
- Activations during decode: [1, 1, 1, hidden_dim] = 8 KB for 4096 hidden
- RMSNorm gamma weights: [1, 1, 1, hidden_dim] = 8 KB
- RoPE cos/sin tables (per-position): small

**What must stream from DRAM (too large for L1):**
- Weight matrices: 4096x4096 = 32 MB each
- KV cache: grows with sequence length

**Key constraint:** Shard sizes must be tile-aligned (multiples of 32 in both dims).

### 2.2 Pre-allocated Tensor Buffers for Trace (ALREADY USED, OPTIMIZE)

Trace replays the exact same memory allocation pattern. Pre-allocating all buffers ensures no fragmentation.

**Tricks:**
- Set `trace_region_size` explicitly when opening device (e.g., 256 MB)
- Use `optional_output_tensor=` on ops to write into pre-allocated buffers
- Deallocate intermediate tensors explicitly with `tensor.deallocate()` to prevent GC delays

### 2.3 Weight Pinning in L1 (SPECULATIVE)

For small models (Qwen 0.5B: 980 MB weights), we have 165 MB of L1 total. We cannot fit all weights, but we CAN fit:
- All RMSNorm weights (~200 KB total for 0.5B)
- All bias terms (if any)
- Embedding table subset (frequently used tokens)

These stay resident across all decode steps, eliminating repeated DRAM reads.

**Experiment:** Move all RMSNorm gammas to L1_MEMORY_CONFIG at load time, verify they persist across trace replays.

### 2.4 Double Buffering for Compute-Memory Overlap

The Tensix architecture supports double-buffered circular queues where reader, compute, and writer kernels execute asynchronously on separate RISC-V cores. At the TT-NN level, this is handled automatically by matmul program configs, but you can influence it:

**Key insight from Corsix Part 7:** "At fp8 precision, the bottleneck becomes data transfer (16 cycles to multiply a 32x32 tile, but 18 cycles to get the data in or out)." This means compute is NOT the bottleneck for low-precision formats -- data movement is. Double buffering hides transfer latency.

---

## 3. NOC Optimization Tricks

### 3.1 Multicast for Activation Broadcast (SHOULD VERIFY)

In decode, the activation vector [1, hidden_dim] is tiny. It should be multicast to all cores, not unicast to each:

```python
# In matmul program config:
mcast_in0=True  # Broadcast activation (in0), each core has weight shard (in1)
```

This uses NOC multicast which sends one packet to a rectangular region of cores simultaneously. Verify our matmul configs use `mcast_in0=True` for decode matmuls.

### 3.2 NOC Routing Awareness

From Corsix's measurements:
- Tile-to-tile propagation: 9 clock cycles (~9 ns at 1 GHz)
- Same-row round-trip: 90 cycles
- Cross-row-and-column: 198 cycles

**Implication:** Place reader cores adjacent to their DRAM banks. The physical tile interleaving handles this somewhat, but explicit core placement via CoreRangeSet can improve it.

### 3.3 Avoid NOC Congestion in Reductions

FlashDecode's reduction phase (combining partial attention results) uses NOC writes + semaphores. With too many cores per KV head, NOC traffic dominates compute. TT caps this at `max_cores_per_head_batch=16`.

**For our 8B model:** 8 KV heads x batch=1 = 8 work units across 110 cores. That is ~13 cores per work unit, which is under the cap. Good.

**For our 0.5B model:** 2 KV heads x batch=1 = 2 work units. That is ~55 cores per work unit -- way over the cap. FlashDecode will limit to 16 cores per head, leaving 78 cores idle during SDPA. This explains why SDPA is 42% of our 0.5B decode time!

**Experiment:** Profile SDPA on 0.5B to confirm core utilization. Test with explicit `max_cores_per_head_batch` values.

### 3.4 Use Both NOCs Simultaneously

Blackhole has two independent NOCs (NOC#0: east+south, NOC#1: west+north). Using both simultaneously doubles effective NOC bandwidth. The reader kernel can issue reads on NOC#0 while the writer uses NOC#1.

This is handled at the kernel level (data movement kernels use `noc_async_read` on one NOC, `noc_async_write` on the other). Verify that our matmul program configs exploit both NOCs.

---

## 4. Compute Pipeline Tricks

### 4.1 Math Fidelity Selection (ALREADY PARTIALLY USED)

| Fidelity | Matmul TFLOPS (BH) | When to use |
|----------|-------------------|-------------|
| LoFi     | ~5.4              | BFP4/BFP8 weights (bandwidth-bound, extra compute is free) |
| HiFi2    | ~2.7              | BF16 decode matmuls (DRAM-sharded, otherwise flop-bound) |
| HiFi4    | ~1.35             | Final layer, embedding lookups (quality-sensitive) |

TT's attention module uses HiFi2 for DRAM-sharded matmuls: "Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision."

**Key insight:** When using BFP4 weights with LoFi, compute throughput is 5.4 TFLOPS per core x 110 cores = 594 TFLOPS chip-wide. At this rate, a 4096x14336 matmul takes ~0.1 ms of compute, while reading the BFP4 weights takes ~0.8 ms. So we are firmly bandwidth-bound and LoFi wastes nothing.

### 4.2 Packer L1 Accumulation

`packer_l1_acc=True` enables in-SRAM accumulation: accumulate partial products in Float32, convert to BF16 only on final output. This avoids writing partial sums back to DRAM between matmul tiles.

```python
compute_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    packer_l1_acc=True,
    fp32_dest_acc_en=True,  # safe on Blackhole (bug fixed vs Wormhole)
    math_approx_mode=True   # faster SFPU ops (exp, gelu, sqrt)
)
```

### 4.3 Math Approximation Mode

`math_approx_mode=True` uses faster polynomial approximations for SFPU operations (SiLU, GELU, exp, sqrt, tanh). These are used in activation functions and softmax.

Trade-off: ~1-2 ULP error vs. 1.5-2x speedup on SFPU-bound ops. For inference this is almost always worthwhile.

### 4.4 MOP (Macro-Op) Expansion

The Tensix ISA has a MOP instruction that expands into programmable template sequences. This is used internally by TT-Metal's LLK layer. Not directly controllable from TT-NN, but relevant to understanding why some kernels are faster than others.

### 4.5 SFPU Pipeline: Insert NOPs for Dependencies

From Corsix Part 6: SFPU multiply/add operations require 2 cycles before results are available. An SFPNOP must be inserted if the next instruction would consume the result. TT-Metal's compiler handles this, but custom kernels need to account for it.

---

## 5. Op Fusion Opportunities

### 5.1 Matmul + Activation Fusion (EASY WIN)

```python
# Instead of:
g = ttnn.matmul(h, gate_w)
g = ttnn.silu(g)

# Fused (eliminates 1 L1 read/write cycle + 1 dispatch):
g = ttnn.matmul(h, gate_w, activation="silu")
```

Also works with GELU, ReLU. The fused_activation is handled inside the matmul kernel's pack stage.

### 5.2 Fused RoPE: Q+K in One Call (MEDIUM PRIORITY)

```python
# Instead of separate RoPE on Q and K (2 calls):
q = ttnn.experimental.rotary_embedding(q, cos, sin, ...)
k = ttnn.experimental.rotary_embedding(k, cos, sin, ...)

# Fused (1 call, both Q and K):
q, k = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos_cache, sin_cache, trans_mat)
```

Saves 1 kernel launch per layer = 32 fewer dispatches for 8B.

### 5.3 Fused KV Cache Update (MEDIUM PRIORITY)

```python
# Instead of separate K and V cache updates:
ttnn.experimental.paged_update_cache(k_cache, k_new, update_idxs)
ttnn.experimental.paged_update_cache(v_cache, v_new, update_idxs)

# Fused (parallel K+V update):
ttnn.experimental.paged_fused_update_cache(k_cache, v_cache, k_new, v_new, update_idxs)
```

Halves cache update dispatch overhead. With sub-devices, K and V updates can run on different core groups simultaneously.

### 5.4 Fused RMSNorm + Residual Add

```python
# Instead of:
x = ttnn.add(residual, attn_out)
h = ttnn.rms_norm(x, weight=gamma, epsilon=eps)

# Fused:
h = ttnn.experimental.dit_rms_norm_unary_fused(attn_out, residual_input_tensor=residual, weight=gamma)
```

Saves 1 add op + 1 memory read per layer = 64 fewer ops for 8B.

### 5.5 Fused QKV Projection

For GQA models (Q_dim != K_dim != V_dim), full fusion is not possible with equal-chunk splitting. But K+V can be fused:

```python
# Instead of 3 matmuls:
q = ttnn.matmul(h, q_w)     # [4096, 4096]
k = ttnn.matmul(h, k_w)     # [4096, 1024]
v = ttnn.matmul(h, v_w)     # [4096, 1024]

# Fuse K+V into 1 matmul (2 matmuls total):
kv = ttnn.experimental.minimal_matmul_split(h, kv_weight, chunks=2)  # [4096, 2048] -> k, v
q = ttnn.matmul(h, q_w)
```

Reduces 3 matmuls to 2 per layer = 32 fewer matmuls for 8B. The matmul dispatch overhead savings are significant.

### 5.6 Fused MLP Gate: mul with Activation

```python
# Instead of:
g = ttnn.silu(ttnn.matmul(h, gate_w))
u = ttnn.matmul(h, up_w)
m = ttnn.mul(g, u)

# TT's MLP fuses the mul + activation:
m = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
```

This applies SiLU to tensor_a during the element-wise multiply, saving one separate SiLU kernel.

---

## 6. Metal Trace Optimization

### 6.1 Reduce Op Count in Trace (HIGHEST LEVERAGE)

Our 8B trace has ~960 ops. TT's reference has significantly fewer because of:
- Fused RoPE (4 ops -> 1)
- Native GQA in flash_decode (5 ops -> 0, eliminates split/concat)
- Fused KV updates (4 ops -> 2)
- Fused activation in matmul (2 ops -> 1)

At ~5 us per inter-op gap in trace replay, reducing from 960 to ~600 ops saves ~1.8 ms per decode step.

**Experiment:** Implement fusions one at a time, measure trace time delta after each.

### 6.2 Dual Command Queue + Trace (MEDIUM PRIORITY)

```python
device = ttnn.open_device(device_id=0, num_command_queues=2)

# CQ0: execute trace (compute)
# CQ1: upload next token's embedding (I/O)

ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
event = ttnn.record_event(device, cq_id=0)  # mark when trace finishes
ttnn.wait_for_event(cq_id=1, event=event)
ttnn.copy_host_to_device_tensor(next_embed, device_embed, cq_id=1)
```

This overlaps the PCIe upload of the next token's embedding with the current trace execution. Saves ~1-2 ms per step (the PCIe round-trip time).

**Better:** With on-device argmax + on-device embedding lookup, there is NO PCIe round-trip at all during decode. The entire token pipeline stays on-device.

### 6.3 On-Device Token Pipeline (HIGH PRIORITY for small models)

```python
# Inside trace:
logits = decode_forward(embed)
_, idx = ttnn.topk(logits, k=1)           # on-device argmax
next_embed = ttnn.embedding(idx, embed_w)  # on-device lookup
# next_embed feeds directly into next trace iteration
```

Eliminates PCIe readback of logits (~256 KB for 128K vocab) and PCIe upload of embedding (~8 KB). For Qwen 0.5B where PCIe is 4% of decode time, this saves ~0.3 ms. For 8B where it is ~1-2 ms, more significant.

**Caveat:** ttnn.topk has a width limit of 65536, and our vocab is 128256. Workaround: split logits into 2x64K chunks, topk each, compare winners.

### 6.4 trace_region_size Pre-allocation

```python
# Pre-allocate large trace region to avoid fragmentation
device = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
```

This reserves 256 MB of DRAM for trace command buffers upfront, avoiding runtime reallocation.

---

## 7. Multi-Device Tricks (2 Blackhole Chips)

### 7.1 Tensor Parallelism (BEST FOR 8B)

Split weight matrices across both chips. Each chip holds half the weights and processes the full activation. An all-reduce (or all-gather + local matmul) combines results.

**For Llama 8B:**
- Each chip holds ~8 GB weights instead of 16 GB
- Effective DRAM bandwidth doubles: 900 GB/s combined
- Theoretical ceiling: 900 / 8 = 112 tok/s (vs. 28 tok/s single-chip)
- Realistic: ~40-50 tok/s (accounting for Ethernet overhead)

**Ethernet bandwidth:** Blackhole uses 400 Gbps Ethernet = 50 GB/s. For tensor parallelism, each layer requires an all-reduce of the activation vector (8 KB for 4096 hidden). At 50 GB/s, this is negligible (~0.16 us). The overhead is in the synchronization, not bandwidth.

### 7.2 Pipeline Parallelism (SIMPLER, LESS OPTIMAL)

Chip 0 runs layers 0-15, chip 1 runs layers 16-31. Activation is forwarded between chips.

**Problem:** Each chip only uses half its DRAM bandwidth (it only reads half the weights). The theoretical speedup is limited to ~1.5x due to pipeline bubble.

**When it makes sense:** For MoE models where each chip holds different experts, pipeline parallelism avoids duplicating the shared attention layers.

### 7.3 Data Parallelism (BEST FOR BATCH)

Both chips run the full model independently on different batch elements. Doubles batch throughput with zero communication overhead.

For batch serving, this is the simplest and most effective approach.

### 7.4 Multi-Chip API

```python
# Create mesh device
mesh = ttnn.open_mesh_device(
    mesh_shape=(1, 2),  # 1 row, 2 columns
    dispatch_core_config=ttnn.DispatchCoreConfig(...)
)

# Shard weights across devices
weight_sharded = ttnn.distribute_tensor(weight, mesh, strategy="column_shard")

# Execute with all-reduce
output = ttnn.matmul(activation, weight_sharded)
output = ttnn.all_reduce(output, mesh)
```

**Experiment:** First test: open both devices, run same model independently (data parallel). Measure any PCIe contention. Then try tensor parallel on a single layer.

---

## 8. Profiling Tools

### 8.1 Tracy Integration

TT-Metal integrates with the Tracy profiler for fine-grained timing:

```bash
# Build tt-metal with profiling enabled
export TT_METAL_ENABLE_PROFILER=1

# Run with Tracy capture
tt-profiler run -c "python3 my_script.py"

# View results
tt-profiler view
```

Tracy shows:
- Per-op execution time on device
- Inter-op gaps (dispatch overhead)
- NOC transfer times
- Kernel launch/completion timestamps
- Memory allocation events

### 8.2 Python-Level Op Timing

```python
import ttnn

# Enable op-level timing
ttnn.enable_program_cache(device)

# After running ops:
ttnn.dump_device_profiler(device)
```

### 8.3 Environment Variable Controls

```bash
# Detailed DRAM bandwidth profiling
export TT_METAL_DPRINT_CORES=0,0  # Print from specific core
export TT_METAL_DPRINT_ENABLED=1   # Enable device-side prints

# Performance counters
export TT_METAL_DEVICE_PROFILER=1   # Enable hardware counters

# Op-level CSV output
export TTNN_CONFIG_OVERRIDES='{"enable_logging": true, "enable_graph_report": true}'
```

### 8.4 What to Measure First

1. **Op-to-op gap histogram:** Identify which ops have the largest gaps (these are dispatch overhead candidates for fusion)
2. **DRAM bandwidth per matmul:** Are we hitting 450 GB/s or leaving bandwidth on the table?
3. **Core utilization during SDPA:** How many of 110 cores are actually active during flash_decode?
4. **Trace replay overhead:** Time from `execute_trace` call to first op dispatch on device

**Experiment:** Run Tracy profile on one decode step of Llama 8B. Export op timeline. Identify top-5 time consumers.

---

## 9. Prioritized Action Plan

### Tier 1: Immediate (1-2 days each, highest ROI)

| # | Trick | Expected Impact | Section |
|---|-------|----------------|---------|
| 1 | BFP4 MLP weights (gate/up/down) | 20-30% | 1.4 |
| 2 | Fused SiLU in gate matmul | 3-5% | 5.1 |
| 3 | Fused RMSNorm + residual add | 3-5% | 5.4 |
| 4 | On-device topk + embedding | 3-5% (small models) | 6.3 |

### Tier 2: Medium effort (3-5 days each)

| # | Trick | Expected Impact | Section |
|---|-------|----------------|---------|
| 5 | DRAM-sharded weight matmuls | 10-20% | 1.1 |
| 6 | Fused QK RoPE | 2-3% | 5.2 |
| 7 | Fused KV cache update | 2-3% | 5.3 |
| 8 | Dual command queue | 3-5% | 6.2 |
| 9 | Tracy profiling (measure before optimizing) | diagnostic | 8.1 |

### Tier 3: Significant effort (1-2 weeks)

| # | Trick | Expected Impact | Section |
|---|-------|----------------|---------|
| 10 | Multi-device tensor parallelism | ~1.8x | 7.1 |
| 11 | Sub-device parallel ops | 3-5% | 3.3 |
| 12 | Custom DRAM-sharded matmul configs per layer | 5-10% | 1.1 |

### Projected 8B Performance Stack

```
Baseline:                              52.0 ms  (19 tok/s)
+ BFP4 MLP weights:                   ~40.0 ms  (25 tok/s)  -- halve MLP weight reads
+ DRAM-sharded matmul configs:         ~35.0 ms  (29 tok/s)  -- better BW utilization
+ Op fusion (RoPE, KV, activation):    ~33.0 ms  (30 tok/s)  -- ~100 fewer ops in trace
+ Dual CQ / on-device token pipeline:  ~32.0 ms  (31 tok/s)  -- hide PCIe
+ Multi-device (2 chips):              ~18.0 ms  (56 tok/s)  -- double bandwidth
```

---

## 10. Blackhole-Specific Notes

### 10.1 Blackhole vs. Wormhole Differences

| Feature | Wormhole | Blackhole |
|---------|----------|-----------|
| DRAM | 12 GB GDDR6, 12 banks | 32 GB GDDR6, 8+ banks |
| DRAM BW | ~250 GB/s measured | ~450 GB/s measured |
| Tensix cores | 80 (64-72 usable) | 140 (110 usable) |
| L1 per core | 1.5 MB | 1.5 MB |
| NOC transaction max | 8 KB | 16 KB |
| FPU clock | ~1 GHz | ~1.35 GHz |
| fp32 dest accumulation | Bug (rare corruption) | Fixed |
| Vector unit width | 32 lanes | 32 lanes |
| Ethernet | 100 Gbps x 16 | 400 Gbps x 4 |

### 10.2 Blackhole Grid Layout

Our P150 reports `compute_with_storage_grid_size() = (11, 10)`:
- 11 columns x 10 rows = 110 usable Tensix cores
- DRAM controllers are on specific columns (need to check exact positions)
- Harvested rows reduce from 140 to 110

### 10.3 WormholeComputeKernelConfig on Blackhole

Despite the misleading name, `ttnn.WormholeComputeKernelConfig` is the correct config class for Blackhole. There is no `BlackholeComputeKernelConfig`.

**CRITICAL (from memory):** Must use ALL-or-nothing config. Do not mix configured and unconfigured ops in the same trace -- it corrupts results. Always pass `compute_kernel_config=` to every matmul, rms_norm, etc.

---

## Sources

- [TT-Metal: Saturating DRAM Bandwidth](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md)
- [TT-Metal: Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)
- [TT-Metal: FlashDecode](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashDecode.md)
- [TT-Metal: GEMM FLOPS](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md)
- [TT-Metal: LLM Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md)
- [TT-Metal: Tensor Layouts](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md)
- [TT-Metal: Tensor Sharding](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_sharding/tensor_sharding.md)
- [TT-Metal: Sub-Devices](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/SubDevices/SubDevices.md)
- [TT-Metal: PCIe Bandwidth](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/PCIe_bandwidth/PCIe_bandwidth.md)
- [TT-Metal: Matrix Engine](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md)
- [TT-Metal: Data Formats](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/data_formats.md)
- [TT-Metal: tt_transformers/tt/attention.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/attention.py)
- [TT-Metal: tt_transformers/tt/mlp.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/mlp.py)
- [TT-Metal: tt_transformers/tt/generator.py](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/tt/generator.py)
- [Corsix: Tenstorrent Wormhole Part 1-8](https://www.corsix.org/content/tt-wh-part1)
- [Programming Tenstorrent Processors](https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi)
- [ASPLOS Blackhole Microbenchmark Paper](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
