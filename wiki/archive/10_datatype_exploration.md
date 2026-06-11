# Datatype Exploration on Blackhole

## Q: What data types does Blackhole support, and how do they affect throughput?

**A:** Blackhole supports BFloat16, Float32, BFloat8_b (block FP8), and BFloat4_b (block FP4) for matmul. The matrix engine is **bf16-native** — lower precision formats are faster, and fp32 is ~3x slower.

## Matmul Throughput by Dtype (4096x4096x4096)

| Dtype | Time (ms) | TFLOPS | % of 372 TFLOPS | vs bf16 |
|-------|-----------|--------|-----------------|---------|
| **BFloat4_b** | 0.589 | **233.2** | **62.7%** | 1.32x |
| **BFloat8_b** | 0.622 | **220.9** | **59.4%** | 1.25x |
| BFloat16 | 0.776 | 177.1 | 47.6% | 1.00x |
| Float32 | 2.253 | 61.0 | 16.4% | 0.34x |

## Q: Why is fp32 so much slower?

**A:** Blackhole's Tensix matrix engine computes in bf16 natively. A single bf16 matmul takes one cycle on the matrix engine. For fp32, the engine likely decomposes each operation into multiple bf16 multiply-accumulates to maintain the wider mantissa, resulting in ~3x overhead.

This means the "372 TFLOPS" spec is for bf16. The fp32 peak is effectively ~120 TFLOPS (matching the ~61 TFLOPS at 16.4% utilization, which scales similarly to bf16's utilization pattern).

## Q: What are BFloat8_b and BFloat4_b?

**A:** These are Tenstorrent's **block floating point** formats:
- **BFloat8_b**: 8-bit values sharing a per-block exponent. Each block (typically 16 or 32 values) has one shared exponent + individual 8-bit mantissas.
- **BFloat4_b**: Same idea, 4-bit mantissas per element.

They're faster than bf16 because:
1. Less data to move through the memory hierarchy (2x or 4x less bandwidth needed)
2. The matrix engine can potentially process more elements per cycle with smaller operands

These formats are highly relevant for LLM inference where 8-bit and 4-bit quantization is standard practice.

## Numerical Accuracy (256x256 matmul vs fp32 reference)

| Dtype | Mean Abs Error | Max Abs Error | Mean Rel Error |
|-------|---------------|---------------|----------------|
| Float32 | 0.031 | 0.19 | 1.7% |
| BFloat16 | 0.082 | 0.87 | 7.4% |
| BFloat8_b | 0.257 | 1.62 | 9.5% |
| BFloat4_b | 3.301 | 17.74 | 56.7% |

- **fp32**: Best accuracy but 3x slower. Use for training or when precision matters.
- **bf16**: The sweet spot — good accuracy, full throughput.
- **bfp8**: Usable for inference. ~3x error vs bf16, but 1.25x faster.
- **bfp4**: Very lossy (mean error 3.3), but 1.32x faster. Only useful with quantization-aware training.

## Q: What about elementwise operations?

| Dtype | Add time (2048x2048) | Effective BW |
|-------|---------------------|--------------|
| BFloat16 | 0.118 ms | 213 GB/s |
| Float32 | 0.181 ms | 278 GB/s |

Float32 elementwise is 1.5x slower wall-clock, but actually achieves *higher effective bandwidth* (278 vs 213 GB/s) because it moves 2x the bytes. The vector engine handles both dtypes efficiently.

## Key Takeaways for XLA Backend

1. **Default to bf16** — it's the native format and the best speed/accuracy tradeoff
2. **Support bfp8 for inference** — 1.25x speedup with acceptable accuracy loss
3. **fp32 accumulation matters** — matmul likely accumulates in fp32 internally even with bf16 inputs (standard practice). The fp32 *output* path is what's slow.
4. **Quantization is a lever** — An XLA backend that can automatically quantize to bfp8 for inference would get meaningful speedups

## Experiment

`experiments/10_datatype_exploration.py` — run on Blackhole p150a device 0, 2026-04-21.
