# Wiki 55: flash_decode JIT Build Bug on Blackhole

## The Problem

`ttnn.transformer.scaled_dot_product_attention_decode` fails to JIT compile on Blackhole (11x10 = 110 cores) for multiple KV head configurations.

## Test Results (Exp 86)

| KV Heads | Q Heads | GQA Ratio | Result |
|----------|---------|-----------|--------|
| 1 | 32 | 32:1 | FAIL: "Tree reduction max 6 rounds (64 cores/head), got 110 cores/head" |
| 2 | 32 | 16:1 | FAIL: trisc1 JIT build failure (softmax exponential kernel) |
| 4 | 32 | 8:1 | FAIL: same JIT build failure |
| 8 | 32 | 4:1 | FAIL: same JIT build failure |
| 4 | 16 | 4:1 | **WORKS** (our split workaround) |

## Root Cause

Two separate bugs:

1. **Tree reduction overflow (1 KV head):** With 110 cores and 1 KV head, all 110 cores participate in reduction for that head. The kernel only supports up to 64 cores/head (6 rounds of tree reduction, 2^6=64).

2. **JIT compile failure (2/4/8 heads):** The `sdpa_flash_decode` kernel fails to compile for `trisc1` on Blackhole. The error is in `ckernel_sfpu_exp.h` — "cannot write sfpu vector to memory". This is a Blackhole-specific compiler issue in the SFPI backend.

## Why Our Workaround Works

Our split approach (exp 64) works because:
- Split 8 KV heads → 2 groups of 4
- Split 32 Q heads → 2 groups of 16
- **GQA ratio per call: 16Q / 4KV = 4:1** (not 32Q / 4KV = 8:1)
- The lower GQA ratio triggers a different kernel configuration that compiles successfully

## Impact

The split workaround adds per layer:
- 4 slice ops (split Q and KV into lo/hi)
- 2 flash_decode calls (instead of 1)
- 4 to_memory_config reshards
- 1 concat (merge attention outputs)

That's ~11 extra ops per layer × 32 layers = **352 extra ops** in the trace. At ~5μs/op gap, this adds ~1.8ms of overhead.

## Upstream Bug Report

Should include:
- Hardware: Blackhole P150a (11x10 grid, 2 harvested cols)
- ttnn version: 0.68.0
- Error: trisc1 build failure in `ckernel_sfpu_exp.h:388`
- Minimal reproducer: `sdpa_decode(Q=[1,1,32,128], KV=[1,8,512,128])`
- Note: Wormhole (8x8) likely works since flash_decode was designed for it
