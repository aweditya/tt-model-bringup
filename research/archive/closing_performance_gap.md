# Closing the Performance Gap: 18 to 25+ tok/s on Llama-3.1-8B

**Date:** April 2026
**Hardware:** Tenstorrent Blackhole P150a (120 Tensix cores, 32 GB GDDR6, ~450 GB/s measured DRAM BW)
**Current:** 19 tok/s (52ms/tok) | **Ceiling:** 28 tok/s (35.8ms/tok) | **Efficiency:** 68%

---

## 1. Where Is the 16ms Overhead Going?

Our decode step takes 52ms. Weight reads alone should take 35.8ms (16.1 GB / 450 GB/s). The remaining ~16ms breaks down as follows:

### 1.1 Op Dispatch Overhead Within Trace (~4-6ms estimated)

Even with Metal Trace, each op replay has nonzero cost. Our `decode_forward()` for Llama-8B executes per layer:
- 1 rms_norm + 4 matmuls (Q/K/V/O projections) + 2 reshapes + RoPE ops (mul, matmul, add x2) + 4 slice + 4 to_memory_config + 4 paged_update_cache + 2 sdpa_flash_decode + 1 concat + 1 add + 1 rms_norm + 3 matmuls (gate/up/down) + 1 silu + 1 mul + 1 add
- That is **~30 ops per layer x 32 layers = ~960 ops** in the trace, plus final rms_norm + lm_head matmul

From TT's profiling tools, "op-to-op gap" (time between ops including dispatch) is measured in microseconds. At ~5us per op gap in trace replay, 960 ops = ~4.8ms of inter-op gaps alone. This is likely our single largest overhead source.

**Key insight:** TT's reference implementation reduces op count by:
- Using fused RoPE (`ttnn.experimental.rotary_embedding`) instead of our 4-op RoPE (mul + matmul + mul + add)
- Using native GQA in flash_decode instead of our split-concat workaround (eliminates 4 slices + 1 concat per layer)
- DRAM-sharded matmul program configs that avoid explicit `to_memory_config` reshards

### 1.2 KV Cache Update Overhead (~2-3ms estimated)

Per layer we do: 4 slice + 4 to_memory_config (reshard) + 4 paged_update_cache = 12 ops for KV update. That is 384 ops across 32 layers. Even at 5us/op, this is ~2ms.

TT's planned optimization: fuse K and V cache updates to run in parallel on separate sub-devices (V sharded on cores [0-8], K on cores [8-16]).

### 1.3 SDPA Flash Decode (~3-5ms estimated)

Flash decode is memory-bound (reads full KV cache per step). At position 100:
- KV data per layer: 2 x (1 batch x 4 kv_heads x 100 pos x 128 dim x 2 bytes) = 0.2 MB
- Total across 32 layers: 6.4 MB = 0.014ms at 450 GB/s

So the raw data read is negligible. The overhead comes from:
- **Split SDPA:** We split into 2 groups (4 KV heads each) because flash_decode requires power-of-2 KV heads. This doubles the kernel launches (64 flash_decode calls instead of 32).
- **Kernel launch overhead:** Each flash_decode involves core allocation, NoC setup, and reduction across cores. At 50-100us per kernel, 64 launches = 3-6ms.
- **Inter-core reduction:** FlashDecode requires cores to coordinate via NOC writes + semaphores, adding latency proportional to core count.

TT's reference achieves 180 GB/s bandwidth utilization in flash_decode (70% of peak). Our split approach likely achieves less due to doubled launches and smaller per-call work.

### 1.4 PCIe Readback + Host Processing (~1-2ms)

- `from_dev(logits_ref)`: reading vocab_size=128256 values (256 KB) over PCIe
- `np.argmax` on CPU
- `update_buffers`: 3 `ttnn.copy` calls for embed, cos, sin + 1 for pos
- Each copy involves host-to-device PCIe transfer

### 1.5 Summary of 16ms Overhead

| Source | Estimated time | % of overhead |
|--------|---------------|---------------|
| Inter-op gaps in trace (960 ops) | 4-6ms | 30-40% |
| SDPA flash decode launches (64 calls) | 3-5ms | 20-30% |
| KV cache update ops (384 ops) | 2-3ms | 15-20% |
| PCIe + host (copy, argmax) | 1-2ms | 10-15% |
| Memory allocation/resharding | 1-2ms | 5-10% |
| **Total overhead** | **~16ms** | **100%** |

---

## 2. What TT's Reference Implementation Does Differently

TT's tt-metal repo contains `models/tt_transformers/` with optimized Llama implementations. Their Llama-3.1-8B achieves **44.2 t/s on N300** (2 Wormhole chips). Key differences from our implementation:

### 2.1 BFP4 MLP Weights (BIGGEST WIN)

TT's reference uses **BFP4 weights for FF1 (gate) and FF3 (down) in the MLP**, with BFP8 for attention weights. This improved 8B decode from ~23 to ~28 t/s/u — a **22% speedup**.

- MLP weights are 3 x (4096 x 14336) = ~516M params per layer, 32 layers = ~16.5B params
- At BFP4 (0.5 bytes) vs BF16 (2 bytes): saves 24.8 GB of weight reads per step
- Total weight reads drop from 16.1 GB to ~8.1 GB (attention stays BF16/BFP8)
- New ceiling: 450 GB/s / 8.1 GB = **55.6 tok/s** (or ~18ms/tok)

**For us:** This is directly applicable. Our bfloat4_b microbenchmark showed 56.7% error on random data, but MLP weights are NOT random — they are trained and have structured distributions. TT's production use confirms it works. We should:
1. Convert gate_w, up_w, down_w to bfloat4_b with LoFi math fidelity
2. Keep attention weights (Q/K/V/O) at bfloat16 or bfloat8_b
3. Validate quality with cosine similarity vs fp32 reference

**Expected impact: 20-30% speedup (52ms to 38-42ms)**

### 2.2 DRAM Prefetching for Matmuls

TT's April 2025 update mentions "DRAM prefetching to remove memory bottlenecks for matmuls" as a key optimization for their 70B model. This is a hardware feature where:
- The DRAM controller prefetches weight tiles into the NoC pipeline before the compute engine needs them
- Overlaps weight read latency with compute from previous tiles
- Requires DRAM-sharded memory configs with specific shard shapes

**For us:** We use default interleaved DRAM layout for all weights. Switching to DRAM-sharded with proper program_config for matmuls could hide DRAM latency.

### 2.3 Sub-Devices for Parallel Ops

TT uses "Sub-Devices to run multiple ops in parallel" — a feature where the 120-core mesh is partitioned into independent sub-devices that can execute different ops simultaneously.

Example: K cache update on cores [0-8] and V cache update on cores [8-16] run in parallel instead of sequentially. This halves the KV update time.

**For us:** We run all ops serially. Sub-device parallelism could save 1-2ms on KV updates alone.

### 2.4 Native GQA in Flash Decode

TT's flash_decode natively supports GQA by taking KV cache with shape `[b x nkv x S x d]` and handling the Q-to-KV head mapping internally. This eliminates our split-concat workaround.

**For us:** We split 8 KV heads into 2 groups of 4 because flash_decode required power-of-2 KV heads. If the API now supports 8 KV heads natively, we can eliminate 5 ops per layer (4 slices + 1 concat) = 160 ops total, saving ~1ms.

### 2.5 Non-Uniform Precision Per Layer

TT supports "non-uniform data format configurations in different decoder layers via json files." For Llama-8B specifically, they use BFP8 MLP in the 32nd (last) decoder layer and BFP4 MLP elsewhere.

This suggests the last layer is more sensitive to quantization — a pattern seen in other quantization literature where early/late layers need higher precision.

---

## 3. Quantization Paths on Blackhole

### 3.1 Block Floating Point (Available Now)

| Format | Bits/element | Matmul TFLOPS | Quality | Status |
|--------|-------------|---------------|---------|--------|
| bfloat16 | 16 | 332 | Baseline | In use |
| bfloat8_b | 8 | ~370 | 0.999+ cosine | In use (tested exp 61-62) |
| bfloat4_b | 4 | ~440 | Needs calibration | **Available, untested on 8B** |

BFP4 is the clear path. TT's own models use it for MLP. No calibration framework needed — just convert weight dtype.

### 3.2 INT8/INT4 Quantization (Partial Support)

TT-NN has `ttnn.quantize`, `ttnn.dequantize`, and `ttnn.requantize` in the API, supporting per-tensor and per-channel scaling. However:
- No native INT8 matmul in the FPU pipeline — Tensix is BF16-native
- INT8 would go through SFPU (slower) or require dequant-to-BFP before matmul
- The practical path is BFP, not INT

**Verdict:** BFP4 for weights, BFP8 for activations is the Blackhole-native quantization strategy. Do not pursue INT4/INT8.

### 3.3 What About GPTQ/AWQ Calibration?

BFP4 with naive conversion (just truncate) showed 56.7% error on random data. But:
- MLP weights have learned structure, not random distributions
- TT uses BFP4 in production without issues
- The block floating point format preserves relative magnitudes well within each 32-element block

If naive BFP4 conversion causes quality issues, we could:
1. Use SmoothQuant-style per-channel scaling before BFP4 conversion
2. Apply GPTQ optimal rounding (round each weight to minimize output error)
3. Use per-layer calibration to find which layers tolerate BFP4

---

## 4. Actionable Optimization Plan

### Tier 1: Quick Wins (1-2 days each)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| 1 | **BFP4 MLP weights** (gate, up, down) | 20-30% (52ms to 38-42ms) | Low — just change dtype on upload |
| 2 | **Native RoPE** (ttnn.experimental.rotary_embedding) | 5-8% (remove 4 ops/layer = 128 ops) | Low — already tested on Qwen |
| 3 | **Test 8 KV heads directly** in flash_decode | 5-10% if supported (remove split/concat) | Low — one API call change |
| 4 | **Dual command queue** for I/O overlap | 3-5% (hide PCIe transfers) | Medium — event synchronization |

### Tier 2: Medium Effort (3-5 days each)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| 5 | **DRAM-sharded weight layout** with matmul program_config | 10-20% (better BW utilization) | Medium — needs per-matmul config |
| 6 | **Fuse KV cache updates** (sub-device parallelism) | 3-5% (halve KV update time) | Medium — sub-device API learning |
| 7 | **BFP4 + BFP8 mixed precision** (BFP4 MLP, BFP8 attn) | Additive with #1 | Medium — quality validation |
| 8 | **Reduce op count** (fuse reshape chains, eliminate redundant to_memory_config) | 5-10% (fewer inter-op gaps) | Medium — trace profiling needed |

### Tier 3: Significant Effort (1-2 weeks)

| # | Optimization | Expected Impact | Effort |
|---|-------------|----------------|--------|
| 9 | **Speculative decoding** (draft model) | 1.5-2x effective throughput | High — need draft model + verify |
| 10 | **C++ dispatch path** (bypass Python) | 10-20% (eliminate Python overhead in non-traced path) | High — C++ compilation |
| 11 | **Multi-chip** (2 Blackhole P150s) | ~1.8x (linear BW scaling) | High — Ethernet mesh setup |

### Projected Performance Stack

Starting from 52ms/tok (19 tok/s):

```
Baseline:                          52.0 ms  (19 tok/s)
+ BFP4 MLP weights (#1):          40.0 ms  (25 tok/s)  ← halve MLP weight reads
+ Native RoPE (#2):                38.5 ms  (26 tok/s)  ← remove 128 trace ops
+ Native 8-head SDPA (#3):         36.5 ms  (27 tok/s)  ← remove 160 trace ops + halve SDPA calls
+ DRAM-sharded matmuls (#5):       34.0 ms  (29 tok/s)  ← better BW utilization
+ Dual CQ I/O overlap (#4):       33.0 ms  (30 tok/s)  ← hide PCIe transfers
```

**Realistic target: 25-30 tok/s (85-107% of BF16 ceiling, exceeding it via BFP4)**

Note: BFP4 MLP changes the effective bandwidth ceiling. With ~8 GB of weight reads instead of 16 GB, the new ceiling is ~55 tok/s. At 30 tok/s we would be at 54% of this new ceiling — room for further gains from DRAM prefetching and op fusion.

---

## 5. TT Reference Performance Comparison

| Model | Our tok/s | TT Reference tok/s | TT Device | TT Precision |
|-------|-----------|-------------------|-----------|-------------|
| Llama-3.2-1B | 78 | 105.9 | N300 (2 chips) | BFP4 MLP, BFP8 attn |
| Llama-3.2-3B | 34 | 68.0 | N300 (2 chips) | BFP4 MLP, BFP8 attn |
| Llama-3.1-8B | 19 | 44.2 | N300 (2 chips) | BFP4 MLP, BFP8 attn |

N300 has 2 Wormhole chips with combined ~500 GB/s bandwidth (similar to our single Blackhole at 512 GB/s spec). So per-bandwidth, TT's reference achieves:
- 1B: 105.9 tok/s at ~500 GB/s = 0.21 tok/s per GB/s — we get 0.17 (81% of TT)
- 8B: 44.2 tok/s at ~500 GB/s = 0.088 tok/s per GB/s — we get 0.042 (48% of TT)

The 8B gap is largest because TT uses BFP4 MLP weights (halving effective weight size) and optimized DRAM-sharded layouts. With BFP4 MLP, our effective weight size would match theirs, and the gap would narrow from 48% to ~70-80% of TT's per-bandwidth efficiency.

---

## 6. Key Unknowns to Resolve Experimentally

1. **Does BFP4 work for 8B MLP without quality loss?** TT uses it in production, but we need to validate on our specific decode path with cosine similarity.

2. **Does flash_decode now support 8 KV heads?** The GQA improvements in tt-metal may have fixed the power-of-2 restriction. Test: `sdpa_flash_decode(q_all_32, k_cache_8, v_cache_8)`.

3. **What is the actual per-op gap in our trace?** Use `ttnn.profiler` or the `tt-perf-report` tool to get precise microsecond-level breakdown.

4. **Does DRAM-sharded weight layout help for our sizes?** The DRAM-sharded matmul has known stability issues (issue #24681). Test on a single layer first.

5. **Can we use dual command queues with trace?** The advanced perf guide shows CQ0 for trace execution + CQ1 for writes. Need to verify this works with `ttnn.execute_trace`.

---

## Sources

- [TT-Metal Advanced Performance Optimizations](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md)
- [TT-Transformers PERF.md](https://github.com/tenstorrent/tt-metal/blob/main/models/tt_transformers/PERF.md)
- [TT-Metal Model Updates](https://github.com/tenstorrent/tt-metal/blob/main/models/docs/MODEL_UPDATES.md)
- [FlashDecode Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashDecode.md)
- [FlashAttention Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashAttention.md)
- [TT-NN Quantize API](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.quantize.html)
- [Quantized Ops Issue #15934](https://github.com/tenstorrent/tt-metal/issues/15934)
- [Blackhole QuietBox Review (The Register)](https://www.theregister.com/2025/11/27/tenstorrent_quietbox_review/)
- [ASPLOS Blackhole Microbenchmark Paper](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [Corsix Wormhole Part 7: MatMul](https://www.corsix.org/content/tt-wh-part7)
- [TT-Metal Performance Optimization (DeepWiki)](https://deepwiki.com/tenstorrent/tt-metal/7.5-performance-optimization-techniques)
- [TT Data Formats Documentation](https://docs.tenstorrent.com/pybuda/latest/dataformats.html)
- [TT-NN Matmul Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.matmul.html)
- [vLLM Integration Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/vLLM_integration.md)
