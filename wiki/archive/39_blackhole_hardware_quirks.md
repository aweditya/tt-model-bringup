# Wiki 39: Blackhole P150 Hardware Quirks — Measured on Real Silicon

## Q: How much does the program cache matter?

**A:** The program cache gives a **4031x speedup** on a 256x256 matmul:

| Metric | Value |
|--------|-------|
| Cold (first execution) | 257,344 us (257ms) |
| Warm (cached) | 64 +/- 6 us |
| Speedup | **4031x** |

The cold execution includes kernel compilation, memory allocation, and program setup. After the first run, the compiled program is cached and reused. This is why the first forward pass in our models (161ms in 54b) is so much slower than sustained decode (7.6ms).

**Implication:** Always warm up before benchmarking. The program cache is per-shape — changing tensor shapes triggers recompilation.

## Q: What TFLOPS does Blackhole P150 actually achieve?

**A:** Measured across three matmul sizes with bfloat16 and bfloat8_b:

| Size | bfloat16 | bfloat8_b | bf8 speedup |
|------|----------|-----------|-------------|
| 256x256x256 | 0.5 TFLOPS (63us) | 0.5 TFLOPS (63us) | 1.0x |
| 896x896x896 | 16.3 TFLOPS (88us) | 18.0 TFLOPS (80us) | 1.10x |
| 896x4864x896 | **38.3 TFLOPS** (204us) | **41.6 TFLOPS** (188us) | 1.08x |

Key observations:
- **Small matmuls (256x256) are dispatch-dominated** — actual compute is negligible, so dtype doesn't matter
- **Large matmuls approach peak utilization** — 41.6 TFLOPS at 896x4864x896 is likely near the practical ceiling
- **bfloat8_b is ~10% faster** than bfloat16 at model-relevant sizes, with potential precision trade-off
- The 896x4864 size matches Qwen's MLP gate/up projection — this is the hottest matmul in our model

## Q: What is the Python dispatch overhead per op?

**A:** ~**39 microseconds per op** for lightweight elementwise operations:

| Metric | Value |
|--------|-------|
| 1 op (neg, 32x64 tensor) | 57 us |
| 5 ops (neg, relu chain) | 215 us |
| Per-op marginal cost | **39 us** |

This means a 24-layer model with ~15 ops per layer = 360 ops = **~14ms** of pure Python dispatch overhead. That's exactly why trace capture (which eliminates dispatch) drops us from 21ms to 7.6ms — it removes ~13ms of dispatch overhead.

**The dispatch-overhead-per-layer math:**
- 15 ops/layer x 39us = 585us/layer dispatch
- 24 layers x 585us = 14ms total dispatch
- Non-traced decode: 21ms (14ms dispatch + 7ms compute)
- Traced decode: 7.6ms (0ms dispatch + 7.6ms compute)
- Difference: 13.4ms ≈ 14ms estimated dispatch. The numbers check out.

## Q: Does L1 SRAM help for matmul inputs?

**A:** Barely, and it **hurts** at larger sizes:

| Size | DRAM | L1 | L1/DRAM ratio |
|------|------|----|----|
| 32x32 | 64us | 59us | **1.10x** (L1 wins) |
| 64x64 | 63us | 57us | **1.10x** (L1 wins) |
| 128x128 | 56us | 56us | 1.00x (tie) |
| 256x256 | 61us | 61us | 1.01x (tie) |
| 512x512 | 67us | 97us | **0.69x** (DRAM wins!) |

At small sizes (32-64), L1 gives a modest 10% improvement. At 128+, they're identical. At 512x512, L1 INTERLEAVED is **31% slower** than DRAM.

**Why?** The matmul kernel streams data from its source regardless. For small tensors, the entire operand fits in L1 with no DRAM round-trip. For larger tensors, L1 INTERLEAVED spreads data across all cores' local SRAM — the matmul kernel may need to fetch from remote cores' L1, which can be slower than a DRAM read.

This confirms exp 54d's finding: for our decode tensors (1x32x896), memory layout isn't the bottleneck.

## Q: How much does synchronize_device cost?

**A:** **37 microseconds** per call:

| Mode | Latency |
|------|---------|
| With sync | 55 us |
| Without sync (fire and forget) | 19 us |
| **Sync overhead** | **37 us** |

The "without sync" time (19us) measures only Python dispatch — the op is queued but we don't wait for completion. The sync forces a round-trip to the device to confirm completion.

In traced decode, we sync once per step (after `execute_trace`) — so this adds 37us to our 7.6ms total, less than 0.5%.

## Q: What are the known Blackhole-specific bugs and gotchas?

**A:** Documented through experiments 43-54:

### 1. Kernel config state leak (exp 46, Wiki 36)
Applying `WormholeComputeKernelConfig(HiFi4)` to SDPA but not to subsequent matmuls causes the kernel configuration to "leak" — corrupting the matmul output. **Fix:** Apply the same config to ALL ops.

### 2. `ttnn.split` fails on Blackhole (exp 51)
Tile padding (32x32 tiles) makes `ttnn.split` produce incorrect results. **Workaround:** Rotation matrix trick for RoPE.

### 3. `L1_HEIGHT_SHARDED_MEMORY_CONFIG` convenience constant fails (exp 53)
The constant has no ShardSpec and causes "bad optional access". **Fix:** Always use explicit `ttnn.create_sharded_memory_config()`.

### 4. HEIGHT_SHARDED Q fails for batch>1 at high head counts (exp 54c)
`batch=8 * n_q_heads=14 = 112 > 110 available cores`. SDPA decode can't shard Q across more cores than exist. **Workaround:** Use INTERLEAVED Q for batched decode.

### 5. `rotary_embedding_llama` requires HEIGHT_SHARDED trans_mat (exp 54)
Even for INTERLEAVED Q input, the transformation matrix must be HEIGHT_SHARDED. And the op implements interleaved rotation, not Qwen's half-format.

### 6. L1 HEIGHT_SHARDED is slower than DRAM for matmul chains (exp 54d)
171us vs 59us — 2.9x slower. The overhead is from layout conversion between matmul (which outputs INTERLEAVED) and sharded ops.

---

*Hardware quirk measurements from Blackhole P150, firmware 19.6.0, April 2026.*
