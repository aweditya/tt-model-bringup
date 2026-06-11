# Blackhole Memory Hierarchy: Surprises

## Q: Is L1 SRAM faster than DRAM on Blackhole?

**A: Not with the default interleaved memory configs.** In fact, L1_MEMORY_CONFIG is *slower* than DRAM_MEMORY_CONFIG for matmul — sometimes by 3-4x. This was our most surprising result so far.

## Results

### Host ↔ Device Transfer

| Size | H→D (ms) | D→H (ms) | H→D BW | D→H BW |
|------|-----------|-----------|--------|--------|
| 32×32 | 0.096 | 0.052 | 0.02 GB/s | 0.04 GB/s |
| 128×128 | 0.114 | 0.054 | 0.29 GB/s | 0.61 GB/s |
| 512×512 | 0.586 | 0.162 | 0.89 GB/s | 3.24 GB/s |
| 1024×1024 | 2.225 | 0.746 | 0.94 GB/s | 2.81 GB/s |
| 2048×2048 | 9.767 | 2.938 | 0.86 GB/s | 2.86 GB/s |
| 4096×4096 | 74.81 | 36.68 | 0.45 GB/s | 0.91 GB/s |

**Observations:**
- Peak ~3.2 GB/s D→H, ~0.94 GB/s H→D. PCIe x16 Gen4 should deliver 32 GB/s — we're only seeing 3-10% of theoretical. This suggests `from_torch`/`to_torch` includes format conversion (tiling, dtype) on the host side, not just raw DMA.
- D→H is consistently ~3x faster than H→D — likely because H→D includes tile layout conversion.
- Bandwidth drops at 4096×4096 (32 MB bf16) — possibly hitting host-side bottlenecks or memory allocation overhead.
- **Key takeaway for XLA**: Minimizing host↔device transfers is critical. A JIT-compiled graph that stays on-device will massively outperform eager dispatch.

### Elementwise Add: L1 vs DRAM

| Size | DRAM (ms) | L1 (ms) | Speedup |
|------|-----------|---------|---------|
| 32×32 | 0.060 | 0.062 | 0.96x |
| 128×128 | 0.060 | 0.057 | 1.05x |
| 256×256 | 0.065 | 0.062 | 1.06x |
| 512×512 | 0.059 | 0.066 | 0.89x |
| 1024×1024 | 0.066 | 0.066 | 1.01x |

**No difference.** All times are ~60µs regardless of size or memory config. We're measuring dispatch overhead, not actual compute or memory latency. The elementwise add is too fast to be the bottleneck.

### Matmul Output: L1 vs DRAM (the surprise)

| Size | DRAM (ms) | L1 (ms) | Speedup |
|------|-----------|---------|---------|
| 256×256×256 | 0.057 | 0.057 | 0.99x |
| 512×512×512 | 0.067 | 0.113 | **0.59x** |
| 1024³ | 0.090 | 0.350 | **0.26x** |
| 2048³ | 0.214 | 0.625 | **0.34x** |

**L1 is 2-4x SLOWER than DRAM for matmul.**

### matmul→relu Chain: L1 vs DRAM Intermediate

| Size | DRAM (ms) | L1 (ms) | Speedup |
|------|-----------|---------|---------|
| 512³ | 0.084 | 0.113 | **0.74x** |
| 1024³ | 0.105 | 0.340 | **0.31x** |

Same pattern — DRAM intermediate is faster.

## Q: Why is L1 slower than DRAM?

**A:** `ttnn.L1_MEMORY_CONFIG` and `ttnn.DRAM_MEMORY_CONFIG` are both **interleaved** configs. "Interleaved" means data is striped across all memory banks (all DRAM channels, or all cores' L1). The key difference:

1. **DRAM interleaved**: Data is striped across 8 GDDR6 channels. TT-NN's matmul kernels are **optimized for this pattern** — they stream tiles from DRAM through the NoC to compute cores. The DRAM bandwidth is 400+ GB/s across all channels.

2. **L1 interleaved**: Data is scattered across 110 cores' L1 (1.5 MB each). When a core needs a tile that lives in another core's L1, it must fetch it over the NoC. This creates **many small NoC transfers** between cores — essentially turning L1 into a distributed, non-local memory.

The real benefit of L1 comes from **sharded** memory configs (`ttnn.create_sharded_memory_config(...)`) where you carefully place each core's input data in that core's own L1, eliminating cross-core traffic. This is what TT-NN's optimized program configs do — they shard the weight matrix across cores so each core has its local data in L1.

## Q: What does this mean for XLA fusion?

**A:** The naive assumption "keep intermediates in L1 for fusion" is wrong on Tenstorrent. Unlike GPUs (where shared memory/L1 is local to an SM), Blackhole's L1 is per-core and accessed via NoC. The correct approach for fusion is:

1. **Shard intermediates** — place each tile on the core that produced it and will consume it
2. **Use program configs** — TT-NN's matmul has `program_config` options that control sharding
3. **The fusion benefit is real but requires work** — tt-xla/tt-mlir must generate the correct sharding plan, not just "put it in L1"

This partially explains why "MLIR passes are hard" — the sharding/placement decisions are non-trivial and hardware-specific.

## Experiment

`experiments/08_memory_hierarchy.py` — run on Blackhole p150a device 0, 2026-04-21.

## Sources
- Experiment 08 results
- TT-NN memory config docs
- Blackhole specs: 8 GDDR6 channels, 110 Tensix cores with 1.5 MB L1 each
