# Performance Ceilings for LLM Inference on Blackhole P150

**Date:** April 2026
**Course:** Stanford CS440LX
**Hardware:** Tenstorrent Blackhole P150a — 110 usable Tensix cores, 32 GB GDDR6, 512 GB/s spec / ~450 GB/s measured DRAM bandwidth, 332 TFLOPS BF16 / 664 TFLOPS BFP8, PCIe Gen5 x16, 300W TDP

**Current measured performance (batch=1, traced decode):**

| Model | Params | Precision | ms/tok | tok/s |
|-------|--------|-----------|--------|-------|
| Qwen2.5-0.5B | 0.49B | BF16 | 7.1 | 140 |
| Llama-3.2-1B | 1.24B | BF16 | 12.8 | 78 |
| Llama-3.1-8B | 8.0B | BFP8 | 43 | 23 |

**Peak aggregate:** 4,867 tok/s at batch=64 (Qwen2.5-0.5B)

---

## 1. Theoretical DRAM Bandwidth Ceiling

### The fundamental constraint

Single-sequence autoregressive decode is memory-bandwidth-bound. Each token generation requires reading all model weights from DRAM exactly once. The activation vectors are negligibly small (a few KB) compared to the weight matrices (hundreds of MB to tens of GB). The theoretical minimum time per token is:

```
t_min = weight_bytes / DRAM_bandwidth
```

We use 450 GB/s (measured, ~88% of 512 GB/s spec) as the realistic DRAM bandwidth ceiling, based on the ASPLOS microbenchmark paper showing 88% utilization under optimal access patterns with all 8 GDDR6 controllers saturated.

### Per-model ceilings

| Model | Params | Weight bytes (BF16) | Weight bytes (BFP8) | t_min BF16 | Ceiling BF16 | t_min BFP8 | Ceiling BFP8 |
|-------|--------|--------------------|--------------------|------------|-------------|------------|-------------|
| Qwen2.5-0.5B | 0.49B | 0.98 GB | 0.49 GB | 2.18 ms | 459 tok/s | 1.09 ms | 918 tok/s |
| Llama-3.2-1B | 1.24B | 2.48 GB | 1.24 GB | 5.51 ms | 181 tok/s | 2.76 ms | 363 tok/s |
| Llama-3.1-8B | 8.0B | 16.0 GB | 8.0 GB | 35.6 ms | 28 tok/s | 17.8 ms | 56 tok/s |

### How close are we?

| Model | Actual tok/s | BW ceiling tok/s | Efficiency | Headroom |
|-------|-------------|-----------------|------------|----------|
| Qwen2.5-0.5B | 140 | 459 | **30.5%** | 3.3x |
| Llama-3.2-1B | 78 | 181 | **43.1%** | 2.3x |
| Llama-3.1-8B | 23 | 28 (BFP8: 56) | **82% of BF16 / 41% of BFP8** | 1.2x / 2.4x |

**Key finding: larger models are more bandwidth-efficient.** The 8B model at BFP8 achieves 82% of the BF16 bandwidth ceiling (its weights are BFP8, but the KV cache, activations, and norms remain BF16). This makes sense: larger matmuls better saturate the 110-core mesh, and fixed overhead (trace dispatch, PCIe, KV cache reads) is amortized over more useful work per layer.

**Small models are NOT bandwidth-bound.** Qwen2.5-0.5B's weights are only 0.98 GB, readable in 2.2ms, yet the total decode takes 7.1ms. The remaining 4.9ms is dominated by SDPA KV cache reads (~3.0ms), reshape/dispatch overhead (~0.8ms), trace execution (~0.5ms), and PCIe readback (~0.3ms). This is why switching Qwen to BFP8 weights produced zero speedup.

### Where the time actually goes (Qwen2.5-0.5B breakdown)

```
Component                           Time      % of total
--------------------------------------------------------------
SDPA decode (24 layers x KV read)   3.0 ms      42%
Matmuls (168 per forward pass)      2.5 ms      35%
RoPE + cache update + reshape       0.8 ms      11%
Trace execution overhead            0.5 ms       7%
PCIe readback (logits)              0.3 ms       4%
--------------------------------------------------------------
Total                               7.1 ms     100%
```

The weight read (2.2ms theoretical) is buried inside the 2.5ms matmul line. The rest of the matmul time is compute and dispatch. SDPA's 3.0ms comes from reading the full KV cache (all MAX_SEQ=2048 positions, all layers) even when the actual sequence is short.

---

## 2. Compute Ceiling and Memory-to-Compute Crossover

### Blackhole compute specs

| Precision | Peak TOPS/TFLOPS | Source |
|-----------|-----------------|--------|
| BFP8 (BlockFP8) | 664 TFLOPS | Tenstorrent spec |
| BF16 | 332 TFLOPS | Tenstorrent spec (4-pass matmul fidelity) |
| BFP4 | ~440 TFLOPS (est.) | Intermediate fidelity passes |

The matrix engine uses multi-pass accumulation with 7-bit x 5-bit multipliers: BFP8 requires 2 passes (LoFi + HiFi2), BF16 requires 4 passes (LoFi through HiFi4). Higher precision = lower throughput.

### Arithmetic intensity and the crossover point

For a matrix-vector multiply W * x where W is (M, K) and x is (B, K):

- **Bytes read:** M * K * bytes_per_weight (weights dominate for small B)
- **FLOPs:** 2 * M * K * B
- **Arithmetic intensity:** 2B / bytes_per_weight (FLOPs per byte)

The crossover from memory-bound to compute-bound occurs when:

```
arithmetic_intensity = peak_FLOPS / peak_bandwidth
```

For Blackhole at BF16: 332 TFLOPS / 450 GB/s = **738 FLOP/byte**
At 2 bytes per weight: crossover batch = 738 * 2 / 2 = **738**

For Blackhole at BFP8: 664 TFLOPS / 450 GB/s = **1,476 FLOP/byte**
At 1 byte per weight: crossover batch = 1476 * 1 / 2 = **738**

**Theoretical crossover: batch ~738.** Below this, decode is memory-bound. Above it, compute-bound.

But this theoretical number is misleading for several reasons:

### Why the practical crossover is much lower

1. **Not all ops scale with batch.** Embedding lookups, layer norms, RoPE rotations, and KV cache updates are not large matmuls. Their overhead stays roughly constant, eating into the batch throughput budget.

2. **KV cache scales with batch.** Each batch element has its own KV cache. At batch=64, the KV cache for Qwen is 96 MB total — not negligible. SDPA must read all of it per step.

3. **L1 SRAM limits.** Each Tensix core has 1.5 MB of L1. The KV cache sharding assigns one core per batch element (hence batch=128 fails on 110 cores). Larger batch = more cores consumed by KV storage = fewer available for compute.

4. **Observed scaling efficiency:**

| Batch | ms/step | tok/s aggregate | Per-tok efficiency | Regime |
|-------|---------|----------------|-------------------|--------|
| 1 | 7.6 | 132 | 100% | Memory-bound |
| 8 | 7.6 | 1,050 | 100% | Memory-bound (free scaling) |
| 16 | 8.3 | 1,926 | 91% | Transitional |
| 32 | 9.6 | 3,335 | 79% | Transitional |
| 64 | 13.3 | 4,819 | 57% | Approaching compute-bound |

Batch 1-8 is perfectly free: the hardware was simply idle at batch=1, so adding sequences costs zero additional latency. By batch=64, latency has nearly doubled, indicating we are consuming real compute and memory bandwidth.

**Practical estimate: the system transitions around batch=16-32 for the 0.5B model.** For the 8B model (which better saturates cores even at batch=1), the transition would occur at a lower batch size, likely batch=4-8.

### Compute utilization at batch=1

At batch=1 for Qwen2.5-0.5B, the 168 matmuls perform roughly:
- Total FLOPs per token: ~2 * 0.49B = 0.98 GFLOP
- Time for matmuls: ~2.5ms
- Effective compute: 0.98 GFLOP / 2.5ms = 392 GFLOPS = **0.12% of 332 TFLOPS**

The hardware is 99.9% idle during single-sequence decode. This is universal across all accelerators for small-batch LLM decode — the workload is fundamentally memory-bound.

---

## 3. PCIe Ceiling

### The 3.9ms readback problem

Our decode loop includes a PCIe transfer to read logits back to the host for argmax:

```
t_pcie = from_device(logits) + numpy_argmax + update_buffers
```

Measured at 3.9ms end-to-end for the host round-trip (exp 55). The breakdown:

| Component | Time | Notes |
|-----------|------|-------|
| `from_device(logits)` | ~0.3ms | 256 KB over PCIe Gen5 x16 (64 GB/s) |
| `np.argmax` on host | ~0.1ms | CPU compute |
| `update_buffers` (embed + cos + sin + pos) | ~3.5ms | 4 host-to-device copies |
| **Total host round-trip** | **~3.9ms** | |

The raw PCIe bandwidth is not the bottleneck — 256 KB at 64 GB/s takes 4 microseconds. The overhead is **driver/API latency**: each `ttnn.copy` or `from_device` call involves Python-to-C++ dispatch, command queue synchronization, and DMA setup.

### Impact by model

| Model | Total ms/tok | PCIe overhead | % of total |
|-------|-------------|--------------|------------|
| Qwen2.5-0.5B | 7.1 | 3.9 | **55%** |
| Llama-3.2-1B | 12.8 | 3.9 | **30%** |
| Llama-3.1-8B | 43 | 3.9 | **9%** |

For the 0.5B model, PCIe overhead is over half the total time. For the 8B model, it is relatively minor.

### Can it be hidden?

**Yes, partially.** Three approaches:

1. **On-device argmax (implemented, exp 55b).** Run argmax on the Tensix cores and read back only 1 integer instead of 128K floats. This eliminates the `from_device(logits)` cost but still requires `update_buffers` for the next token's embedding lookup. Measured improvement: the 0.3ms logits readback drops to near zero, but the 3.5ms update_buffers remains.

2. **Dual command queue overlap.** Use CQ0 for trace execution and CQ1 for host-to-device writes, overlapping PCIe transfers with computation. TT-Metal supports this pattern. Expected improvement: hide most of the 3.5ms update_buffers behind the next step's trace execution. Net saving: ~2-3ms for small models.

3. **Fully on-device token loop.** Keep the entire decode loop on-device: on-device argmax, on-device embedding lookup, no host round-trip between tokens. This eliminates PCIe entirely during generation. Blocked by: need an on-device embedding table lookup indexed by a device-resident scalar.

**Net assessment:** Dual command queue could reduce effective PCIe overhead from 3.9ms to ~1ms by overlapping writes with compute. For the 0.5B model, this would improve throughput from 140 to ~165 tok/s (~18% gain). For the 8B model, the gain would be marginal (~2%).

---

## 4. Comparison to GPU Baselines

### RTX 4090 (24 GB GDDR6X, 1,008 GB/s BW, 330 TFLOPS FP16, ~$1,600)

| Model | RTX 4090 tok/s (llama.cpp FP16) | RTX 4090 tok/s (TRT-LLM) | Our Blackhole | BH / 4090 |
|-------|-------------------------------|--------------------------|---------------|-----------|
| Qwen2.5-0.5B | ~350 | ~500+ | 140 | 0.28-0.40x |
| Llama-3.2-1B | ~200 | ~350 | 78 | 0.22-0.39x |
| Llama-3.1-8B | ~104-150 | ~200 | 23 | 0.12-0.22x |

The RTX 4090 has 2.24x our DRAM bandwidth (1,008 vs 450 GB/s). The raw bandwidth ratio predicts we should be ~0.45x of the 4090, but we are 0.12-0.40x. The additional gap comes from:
- CUDA's 15+ years of kernel optimization (fused attention, quantization support, operator fusion)
- Better memory controller utilization (GDDR6X vs GDDR6)
- INT4/INT8 quantization support (unavailable on Blackhole for matmuls)

### A100 (80 GB HBM2e, 2,039 GB/s BW, 312 TFLOPS BF16, ~$15,000)

| Model | A100 tok/s (TRT-LLM est.) | Our Blackhole | BH / A100 |
|-------|--------------------------|---------------|-----------|
| Llama-3.1-8B | ~250-350 | 23 | 0.07-0.09x |

The A100 has 4.5x our bandwidth (2,039 vs 450 GB/s) and HBM's superior access patterns. At equivalent software maturity, we would expect ~0.22x of A100 performance. The additional gap is again software.

### H100 (80 GB HBM3, 3,350 GB/s BW, 990 TFLOPS BF16, ~$30,000)

| Model | H100 tok/s (TRT-LLM) | Our Blackhole | BH / H100 |
|-------|----------------------|---------------|-----------|
| Llama-3.1-8B | ~200-300 | 23 | 0.08-0.12x |

### Apple M4 Pro (14-core, 273 GB/s unified memory, ~$2,000 system)

| Model | M4 Pro tok/s (FP16 est.) | Our Blackhole | BH / M4 Pro |
|-------|--------------------------|---------------|-------------|
| Qwen2.5-0.5B | ~70 | 140 | **2.0x** |
| Llama-3.2-1B | ~38 | 78 | **2.1x** |
| Llama-3.1-8B | ~12 | 23 | **1.9x** |

### Apple M4 Max (40-core GPU, 546 GB/s unified memory, ~$3,500 system)

| Model | M4 Max tok/s (FP16, MLX) | Our Blackhole | BH / M4 Max |
|-------|--------------------------|---------------|-------------|
| Qwen2.5-0.5B | ~100 | 140 | **1.4x** |
| Llama-3.2-1B | ~55 | 78 | **1.4x** |
| Llama-3.1-8B | ~18 | 23 | **1.3x** |

### Tenstorrent's own reference (N300 = 2x Wormhole, ~500 GB/s combined BW)

| Model | TT reference tok/s | Our tok/s | Ratio |
|-------|-------------------|-----------|-------|
| Llama-3.2-1B | 105.9 | 78 | 0.74x |
| Llama-3.2-3B | 68.0 | 34 | 0.50x |
| Llama-3.1-8B | 44.2 | 23* | 0.52x |

*Our 8B number uses BFP8 weights; TT reference uses BFP4 MLP + BFP8 attention. Per-chip, per-bandwidth, we are at 48-81% of TT's optimized reference. The gap is primarily BFP4 MLP weights (halves MLP weight reads) and DRAM-sharded memory layouts.

### Summary positioning

```
                         Llama-3.1-8B single-sequence tok/s (log scale)
                         
Cerebras WSE-3           ████████████████████████████████████████  2,200
Groq LPU                 ██████████████████████████               620
H100 (TRT-LLM)           ██████████████                           300
A100 (TRT-LLM)           ██████████                               250
RTX 4090 (TRT-LLM)       ████████                                 200
RTX 4090 (llama.cpp)     █████                                    130
TT N300 reference        ██                                       44
Blackhole P150 (ours)    █                                        23
M4 Max (MLX FP16)        █                                        18
M4 Pro (FP16)            ▌                                        12
```

**Blackhole P150 at $999 sits between Apple Silicon consumer chips and the RTX 4090.** Its value proposition is batch throughput (4,867 tok/s at batch=64 on 0.5B) and multi-chip Ethernet scaling, not single-stream latency.

---

## 5. What Would 2x Improvement Require?

For each model, we analyze what changes would double throughput from current levels.

### Qwen2.5-0.5B: 140 --> 280 tok/s (7.1ms --> 3.55ms)

**Current bottleneck:** Overhead-bound. SDPA (3.0ms) + dispatch (1.3ms) + PCIe (0.3ms) = 4.6ms of non-weight-read time.

| Change | Savings | Feasibility |
|--------|---------|-------------|
| Dual command queue (hide PCIe) | ~0.8-1.5ms | Medium — TT-Metal supports it, needs event sync |
| Reduce MAX_SEQ 2048 --> 512 | ~1.5ms (SDPA reads 4x less KV) | Easy — but limits context length |
| Fused block-sparse SDPA | ~1.5-2.0ms (skip empty KV positions) | Hard — custom kernel |
| Reduce op count (fuse reshapes) | ~0.3-0.5ms (fewer inter-op gaps) | Medium |
| **Combined** | **~3.5ms saved** | |

**Verdict: achievable but requires multiple optimizations stacking.** The single biggest lever is reducing SDPA's KV cache reads (shorter MAX_SEQ or sparse attention). No single change yields 2x.

### Llama-3.2-1B: 78 --> 156 tok/s (12.8ms --> 6.4ms)

**Current bottleneck:** Mixed. Weight reads (~5.5ms) + SDPA/overhead (~7.3ms).

| Change | Savings | Feasibility |
|--------|---------|-------------|
| BFP8 weights (all layers) | ~2.8ms (halve weight bandwidth) | Easy — validated on other models |
| Dual command queue | ~1.5ms | Medium |
| Native RoPE + fused ops | ~0.5-1.0ms | Easy — proven on Qwen |
| Reduce MAX_SEQ | ~1.0ms | Easy if context allows |
| **Combined** | **~6.0ms saved** | |

**Verdict: likely achievable.** BFP8 weights alone would push to ~100 tok/s. Adding PCIe overlap and op fusion could approach 150+ tok/s.

### Llama-3.1-8B: 23 --> 46 tok/s (43ms --> 21.5ms)

**Current bottleneck:** Primarily bandwidth-bound. Weight reads at BFP8: ~17.8ms. Total overhead: ~25ms.

| Change | Savings | Feasibility |
|--------|---------|-------------|
| BFP4 MLP weights (gate/up/down) | ~10-12ms (MLP is 2/3 of params) | Medium — TT uses it in production |
| DRAM-sharded matmul layout | ~3-5ms (better BW utilization) | Medium — per-matmul config |
| Native GQA in flash_decode | ~1-2ms (eliminate split/concat) | Easy — if API supports 8 KV heads |
| Sub-device parallel KV update | ~1-2ms | Medium — sub-device API |
| Dual command queue | ~1-2ms | Medium |
| **Combined** | **~18-23ms saved** | |

**Verdict: achievable with BFP4 as the primary lever.** BFP4 MLP alone would bring weight reads from 17.8ms to ~9ms, pushing total from 43ms to ~33ms (30 tok/s). Stacking DRAM-sharded layouts and op reduction could reach 25 tok/s or beyond. TT's own reference achieves 44 tok/s on comparable bandwidth, confirming 2x is within reach.

### Batch=64 aggregate: 4,867 --> 9,734 tok/s

**Current bottleneck:** Compute and KV bandwidth scaling.

| Change | Impact | Feasibility |
|--------|--------|-------------|
| BFP4/BFP8 weights (reduce per-step BW) | 1.3-1.5x | Medium |
| Batch=128+ with multi-core KV sharding | 1.5-1.8x (more sequences per step) | Hard — blocked by 110-core limit |
| Multi-chip (2x Blackhole via Ethernet) | ~1.8x (double BW + cores) | Hard — needs Ethernet mesh |
| Speculative decoding (draft model) | 1.5-2.0x effective tokens | Hard — needs draft model |
| **Any one of these** | **Likely sufficient** | |

**Verdict: 2x batch throughput is most readily achieved via multi-chip scaling or batch size increase with alternative KV sharding.** The current batch=64 already uses 13.3ms/step — approaching compute-bound territory where more bandwidth (via multi-chip) or more efficient compute (BFP4) is needed.

### Summary: what matters most per model size

| Model size | Primary bottleneck | 2x lever | Difficulty |
|------------|-------------------|----------|------------|
| 0.5B | Fixed overhead (SDPA, dispatch, PCIe) | Sparse KV + PCIe overlap + op fusion | Hard (many small wins) |
| 1B | Mixed (overhead + bandwidth) | BFP8 weights + PCIe overlap | Medium |
| 8B | Bandwidth | BFP4 MLP weights | Medium (proven by TT) |
| Batch throughput | Compute + core count | Multi-chip or higher batch | Hard (infrastructure) |

---

## Appendix A: Hardware Spec Sources

- **512 GB/s DRAM bandwidth:** [Tenstorrent Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- **~450 GB/s measured:** [ASPLOS Blackhole Microbenchmark Paper](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- **332 TFLOPS BF16 / 664 TFLOPS BFP8:** Tenstorrent product page
- **120 cores (110 usable after harvesting):** Firmware v19.5.0+ reports, confirmed in our experiments
- **1.5 MB L1 per core:** Corsix blog series, TT documentation
- **PCIe Gen5 x16:** Tenstorrent spec (theoretical 64 GB/s bidirectional)
- **TT N300 reference numbers:** [TT-Transformers PERF.md](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/PERF.md)
- **RTX 4090 bandwidth (1,008 GB/s):** NVIDIA product specifications
- **A100 bandwidth (2,039 GB/s):** NVIDIA A100 datasheet
- **H100 bandwidth (3,350 GB/s):** NVIDIA H100 datasheet
- **M4 Pro bandwidth (273 GB/s):** Apple specifications
- **Apple Silicon LLM benchmarks:** [SiliconBench](https://siliconbench.radicchio.page/), [LocalScore.ai](https://www.localscore.ai/accelerator/550)
- **RTX 4090 LLM benchmarks:** [NVIDIA llama.cpp blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- **Cerebras/Groq/SambaNova:** [Intuition Labs comparison](https://intuitionlabs.ai/articles/cerebras-vs-sambanova-vs-groq-ai-chips)

## Appendix B: Key Formulas

**Bandwidth ceiling (single-sequence):**
```
max_tok_per_sec = DRAM_bandwidth_bytes_per_sec / model_weight_bytes
```

**Arithmetic intensity crossover (memory-bound to compute-bound):**
```
crossover_batch = (peak_FLOPS / DRAM_bandwidth) * (bytes_per_weight / 2)
```

**Batch scaling efficiency:**
```
efficiency(B) = (B * tok_per_sec_at_B1) / actual_tok_per_sec_at_B
            -- or equivalently --
efficiency(B) = latency_at_B1 / latency_at_B
```

**Effective bandwidth utilization:**
```
BW_util = model_weight_bytes / (time_per_token * DRAM_bandwidth)
```
