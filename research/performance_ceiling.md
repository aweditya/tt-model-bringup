# Performance Ceiling Analysis: Blackhole P150 Decode Throughput

**Date:** April 2026
**Hardware:** Tenstorrent Blackhole P150a (120 Tensix cores, 32 GB GDDR6, 512 GB/s DRAM BW, 332 TFLOPS BF16)

---

## 1. Hardware Specifications (Verified)

| Spec | Value | Source |
|------|-------|--------|
| DRAM capacity | 32 GB GDDR6 | [Tenstorrent docs](https://docs.tenstorrent.com/aibs/blackhole/specifications.html) |
| DRAM bandwidth (theoretical) | 512 GB/s | Tenstorrent spec sheet |
| DRAM bandwidth (measured) | ~450 GB/s | [ASPLOS microbenchmark paper](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf) — 88% utilization |
| Compute (BF16) | 332 TFLOPS | Tenstorrent spec (664 TFLOPS BlockFP8 / 2) |
| Compute (BlockFP8) | 664 TFLOPS | Tenstorrent spec |
| Tensix cores | 120 (110 usable after harvesting) | Wiki 35, firmware v19.5.0+ |
| L1 SRAM per core | 1.5 MB (180 MB total) | Research/01 |
| TDP | 300W | Tenstorrent spec |

**Key correction:** Our wiki files variously cite 200, 400, and 512 GB/s for DRAM bandwidth. The **spec is 512 GB/s**, with **~450 GB/s measured** under optimal access patterns (all 8 GDDR6 controllers saturated). For theoretical ceiling calculations below we use 450 GB/s (measured) as the realistic upper bound.

---

## 2. Theoretical Decode Ceiling (Bandwidth-Bound)

Single-sequence autoregressive decode is **memory-bandwidth-bound**: each token requires reading all model weights from DRAM once (the activation vectors are tiny by comparison). The theoretical ceiling is:

```
max_tok/sec = DRAM_bandwidth / model_weight_bytes
```

### Model Size Estimates

Weight bytes = parameters x bytes_per_param. We use BF16 (2 bytes) as our actual precision for most weights, and note BF8 (1 byte) where applicable.

| Model | Parameters | BF16 size | BF8 size | Ceiling (BF16, 450 GB/s) | Ceiling (BF8, 450 GB/s) |
|-------|-----------|-----------|----------|--------------------------|-------------------------|
| Qwen2.5-0.5B | 0.49B | 0.98 GB | 0.49 GB | **459 tok/s** | 918 tok/s |
| Qwen3-0.6B | 0.6B | 1.2 GB | 0.6 GB | **375 tok/s** | 750 tok/s |
| Llama-3.2-1B | 1.24B | 2.48 GB | 1.24 GB | **181 tok/s** | 363 tok/s |
| Llama-3.2-3B | 3.2B | 6.4 GB | 3.2 GB | **70 tok/s** | 141 tok/s |
| SmolLM3-3B | 3.0B | 6.0 GB | 3.0 GB | **75 tok/s** | 150 tok/s |
| Llama-3.1-8B | 8.0B | 16.0 GB | 8.0 GB | **28 tok/s** | 56 tok/s |

**Note:** These ceilings assume perfect streaming of weights with zero overhead — no KV cache reads, no activation compute latency, no dispatch overhead. Real throughput will always be lower.

---

## 3. Our Efficiency: Actual vs. Theoretical

| Model | Actual tok/s | BW ceiling tok/s | **Efficiency** | Gap |
|-------|-------------|-----------------|----------------|-----|
| Qwen2.5-0.5B | 140 | 459 | **30.5%** | 3.3x headroom |
| Qwen3-0.6B | 76 | 375 | **20.3%** | 4.9x headroom |
| Llama-3.2-1B | 78 | 181 | **43.1%** | 2.3x headroom |
| Llama-3.2-3B | 34 | 70 | **48.6%** | 2.1x headroom |
| SmolLM3-3B | 38 | 75 | **50.7%** | 2.0x headroom |
| Llama-3.1-8B | 19 | 28 | **67.9%** | 1.5x headroom |

### Key Observations

1. **Larger models are more efficient.** Llama-8B achieves 68% of bandwidth ceiling vs. Qwen-0.5B at 31%. This makes sense: larger matmuls better saturate the 110-core mesh, and the fixed overhead (trace dispatch, PCIe readback, KV cache reads) is amortized over more compute per layer.

2. **Small models are bottlenecked by overhead, not bandwidth.** Qwen-0.5B's 7.1ms is dominated by SDPA decode (~3.0ms reading KV cache) and trace overhead (~0.5ms), not weight reads. The weights are only 0.98 GB — reading them takes ~2.2ms at 450 GB/s, yet the total is 7.1ms. The remaining 4.9ms is KV cache, dispatch, reshapes, and PCIe.

3. **The KV cache is a hidden bandwidth consumer.** SDPA decode reads the entire KV cache each step (all positions, all layers). For Qwen2.5-0.5B with MAX_SEQ=2048, 2 KV heads, 64 head_dim, 24 layers: `2048 x 2 x 64 x 24 x 2 x 2 = 25 MB` per step for K+V. This adds ~6% to the bandwidth load for the 0.5B model, but scales with sequence length.

4. **The "NOT bandwidth-bound" finding (Wiki 42) is consistent.** We proved that switching to BF8 weights (halving weight size) did NOT improve throughput. This confirms the bottleneck is elsewhere — SDPA latency, trace dispatch overhead, and the serial nature of the 24-layer pipeline.

---

## 4. CPU Baselines: What Would These Run at on a CPU?

### Apple M4 Pro (14-core, 273 GB/s memory bandwidth)

Best available data from community benchmarks (llama.cpp Q4_K_M quantization, which is ~4.5 bits/param):

| Model | M4 Pro tok/s (Q4) | M4 Pro tok/s (FP16 est.) | Source |
|-------|-------------------|--------------------------|--------|
| Llama-3.2-1B | ~112 | ~35-40 | [localscore.ai](https://www.localscore.ai/accelerator/550) |
| Llama-3.2-3B | ~55-65 | ~18-22 | Estimated from bandwidth scaling |
| Llama-3.1-8B | ~33-40 | ~10-13 | [SiliconBench](https://siliconbench.radicchio.page/), community reports |
| Qwen2.5-0.5B | ~180+ | ~60-80 | Estimated from bandwidth scaling |

### Apple M4 Max (40-core GPU, 546 GB/s memory bandwidth)

With MLX (20-87% faster than llama.cpp on Apple Silicon):

| Model | M4 Max tok/s (Q4, MLX) | M4 Max tok/s (FP16, MLX est.) |
|-------|------------------------|-------------------------------|
| Llama-3.2-1B | ~150-180 | ~50-60 |
| Llama-3.2-3B | ~70-90 | ~25-35 |
| Llama-3.1-8B | ~45-55 | ~15-20 |

### Key CPU Comparison

**Important caveat:** CPU benchmarks typically use Q4 quantization (4-bit), which halves or quarters the memory footprint vs. our BF16. For fair comparison, we should compare against FP16 CPU numbers, which are much lower.

| Model | Our Blackhole (BF16) | M4 Pro (FP16 est.) | M4 Max (FP16, MLX est.) | **BH vs M4 Pro** | **BH vs M4 Max** |
|-------|---------------------|---------------------|--------------------------|-------------------|-------------------|
| Qwen2.5-0.5B | 140 | ~70 | ~100 | **2.0x** | **1.4x** |
| Llama-3.2-1B | 78 | ~38 | ~55 | **2.1x** | **1.4x** |
| Llama-3.2-3B | 34 | ~20 | ~30 | **1.7x** | **1.1x** |
| Llama-3.1-8B | 19 | ~12 | ~18 | **1.6x** | **1.1x** |

**Verdict:** Blackhole P150 provides a **1.5-2x advantage over M4 Pro at equivalent precision**, and is roughly **comparable to M4 Max** at FP16. However, when Apple Silicon uses Q4 quantization (which Blackhole currently cannot — no INT4 matmul support in TT-NN), Apple devices are competitive or faster in raw tok/s.

The real Blackhole advantage is **batch throughput**: our 4,867 tok/s at batch=64 on Qwen2.5-0.5B is unreachable on any Apple Silicon device (no batched decode support in MLX/llama.cpp consumer setups).

---

## 5. GPU Baselines: NVIDIA Comparison

### RTX 4090 (24 GB GDDR6X, 1,008 GB/s bandwidth, 330 TFLOPS FP16)

| Model | RTX 4090 tok/s (llama.cpp) | RTX 4090 tok/s (TensorRT-LLM est.) | Our Blackhole | **BH vs 4090** |
|-------|---------------------------|--------------------------------------|---------------|-----------------|
| Qwen2.5-0.5B | ~300+ | ~500+ | 140 | **0.3-0.5x** |
| Llama-3.2-1B | ~200+ | ~350+ | 78 | **0.2-0.4x** |
| Llama-3.2-3B | ~120+ | ~200+ | 34 | **0.2-0.3x** |
| Llama-3.1-8B | ~104-150 | ~200+ | 19 | **0.1-0.2x** |

**The RTX 4090 has 2x our memory bandwidth (1,008 vs. 512 GB/s) and a far more mature software stack.** Its CUDA ecosystem has had 15+ years of optimization. Our efficiency gap vs. the 4090 is 3-5x, which breaks down as:
- ~2x from raw bandwidth disadvantage
- ~1.5-2.5x from software maturity (CUDA kernels, operator fusion, quantization support)

### Datacenter AI Accelerators (Llama 3.1 8B, single-sequence)

| Accelerator | tok/s (8B) | Notes |
|-------------|-----------|-------|
| Cerebras WSE-3 | ~2,200 | Wafer-scale, ~$2M+ system |
| SambaNova SN40L | ~1,000 | Dataflow architecture, rack-scale |
| Groq LPU | ~620 | SRAM-only, no DRAM bottleneck |
| NVIDIA H100 (TensorRT-LLM) | ~200-300 | 3.35 TB/s HBM3 bandwidth |
| NVIDIA RTX 4090 | ~104-150 | Consumer GPU |
| **Blackhole P150 (ours)** | **19** | Single chip, BF16, early software |
| Apple M4 Max (MLX) | ~18-20 | Consumer laptop/desktop |
| Apple M4 Pro (llama.cpp) | ~12-13 | Consumer laptop |

### Tenstorrent's Own Reference Numbers (N300 = 2x Wormhole)

| Model | TT Reference tok/s | Our tok/s | Ratio |
|-------|-------------------|-----------|-------|
| Llama-3.2-1B | 105.9 | 78 | 0.74x |
| Llama-3.2-3B | 68.0 | 34 | 0.50x |
| Qwen2.5-7B | 24.6 | N/A | — |

We are 0.5-0.74x of Tenstorrent's reference, which uses 2 Wormhole chips with height-sharded memory layouts and fully optimized TT-NN kernels. Per-chip, we are likely at parity or ahead for the 1B case, and ~1x for 3B (since N300 has 2 chips with combined bandwidth).

---

## 6. Where the Time Actually Goes

From Wiki 42's breakdown of Qwen2.5-0.5B at 7.1ms/tok:

```
Component                         Time      % of total    Bottleneck type
─────────────────────────────────────────────────────────────────────────
SDPA decode (24 layers x KV read)  3.0ms      42%         Memory (KV cache)
Matmuls (168 per forward)          2.5ms      35%         Compute + memory (weights)
RoPE + cache update + reshape      0.8ms      11%         Dispatch overhead
Trace execution overhead           0.5ms       7%         Fixed cost
PCIe readback (logits)             0.3ms       4%         PCIe bandwidth
─────────────────────────────────────────────────────────────────────────
Total                              7.1ms     100%
```

**The weight read is NOT the bottleneck.** At 450 GB/s, reading 0.98 GB of weights takes ~2.2ms. But SDPA alone takes 3.0ms because it reads the full KV cache (all MAX_SEQ positions) even for short sequences. This is why BF8 weights didn't help.

For the 8B model at 52ms/tok, the weight read IS a larger fraction: 16 GB / 450 GB/s = 35.6ms, which is 68% of the 52ms total. This explains why larger models show higher bandwidth efficiency.

---

## 7. Realistic Optimization Targets

### What could close the gap to bandwidth ceiling?

| Optimization | Expected impact | Feasibility |
|-------------|----------------|-------------|
| HEIGHT_SHARDED matmul output | 10-30% (eliminate DRAM round-trips between ops) | Medium — requires layout changes |
| Fused SDPA with block-sparse KV | 20-40% (skip empty KV positions) | Hard — custom kernel work |
| Reduce MAX_SEQ (e.g., 512 → shorter KV reads) | 15-25% for small models | Easy but limits context |
| INT4/INT8 weight quantization | 1.5-2x (if TT-NN adds support) | Blocked on TT-NN support |
| Multi-chip (2x Blackhole) | ~1.8x (linear bandwidth scaling) | Requires Ethernet mesh setup |

### Realistic single-sequence targets (with feasible optimizations)

| Model | Current | Realistic target | % of ceiling |
|-------|---------|-------------------|-------------|
| Qwen2.5-0.5B | 140 | ~200 | 44% |
| Llama-3.2-1B | 78 | ~120 | 66% |
| Llama-3.2-3B | 34 | ~50 | 71% |
| Llama-3.1-8B | 19 | ~24 | 86% |

The 8B model is already close to its ceiling. The biggest gains are in smaller models where fixed overhead dominates.

---

## 8. Summary

### Is Blackhole P150 providing value as an accelerator?

**For single-sequence decode at BF16:** Marginal. Blackhole is 1.5-2x faster than a CPU (M4 Pro) at equivalent precision, but comparable to a high-end laptop chip (M4 Max). It is 3-5x slower than an RTX 4090.

**For batch decode:** Yes, definitively. At batch=64, Blackhole delivers 4,867 tok/s on Qwen2.5-0.5B — a regime Apple Silicon and consumer GPUs cannot easily reach without serving infrastructure.

**For the $1,000 price point:** Blackhole P150 offers competitive value for batch inference workloads. An RTX 4090 costs $1,600+ and has superior single-stream performance, but Blackhole's 32 GB memory and Ethernet mesh scaling provide a different value proposition for multi-chip deployment.

### Efficiency scorecard

| Model | Bandwidth efficiency | vs. M4 Pro (FP16) | vs. RTX 4090 | vs. TT reference |
|-------|---------------------|-------------------|-------------|-------------------|
| Qwen2.5-0.5B | 31% | 2.0x faster | 0.3-0.5x | N/A |
| Llama-3.2-1B | 43% | 2.1x faster | 0.2-0.4x | 0.74x (per-system) |
| Llama-3.2-3B | 49% | 1.7x faster | 0.2-0.3x | 0.50x (per-system) |
| Llama-3.1-8B | 68% | 1.6x faster | 0.1-0.2x | N/A |

### The fundamental insight

Our small models (0.5-1B) are **overhead-bound**, not bandwidth-bound. SDPA KV cache reads and trace dispatch dominate. Our large models (3-8B) are approaching the **bandwidth ceiling** — 8B is at 68% efficiency with only 1.5x headroom left. The most impactful optimizations would target the overhead for small models (sparse KV, lower MAX_SEQ) and INT4 quantization for large models (doubles effective bandwidth).

---

*Analysis based on experiments 41-74. Hardware: Blackhole P150a, firmware v19.6.0, tt-metal SDK.*

Sources:
- [Tenstorrent Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [ASPLOS Blackhole Microbenchmark Paper](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Tenstorrent Blackhole Product Page](https://tenstorrent.com/hardware/blackhole)
- [llama.cpp Apple Silicon Benchmarks](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [SiliconBench Apple Silicon LLM Benchmarks](https://siliconbench.radicchio.page/)
- [LocalScore.ai M4 Pro Results](https://www.localscore.ai/accelerator/550)
- [NVIDIA llama.cpp RTX Blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- [NVIDIA Llama 3.2 Optimizations](https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/)
- [Cerebras/Groq/SambaNova Comparison](https://intuitionlabs.ai/articles/cerebras-vs-sambanova-vs-groq-ai-chips)
- [TensorRT-LLM Benchmarks](https://www.jan.ai/post/benchmarking-nvidia-tensorrt-llm)
- [Videocardz Blackhole Downgrade Report](https://videocardz.com/newz/tenstorrent-downgrades-blackhole-p150-pcie-cards-specs-from-140-to-120-cores)
