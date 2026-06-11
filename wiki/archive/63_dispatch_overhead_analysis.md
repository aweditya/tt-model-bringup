# Wiki 63: Dispatch Overhead Analysis — Where Does MoE Time Go?

## Q: What is dispatch overhead?

**A:** Every `ttnn` op call from Python traverses the host-side stack (Python -> C++ bindings -> command queue -> device) before any device compute begins. This takes ~30us per op regardless of the operation's actual compute cost. For small tensors (like MoE decode with 1x2048 activations and 1408-wide experts), device compute is near-instant, making dispatch the dominant cost.

## Q: How was the MoE decode profiled? (Exp 94)

**A:** Instrumented `decode_step` with per-phase `time.perf_counter()` timing across 20 steps. Each phase was timed independently:

| Phase | ms/tok | % of total | ms/layer |
|-------|--------|-----------|----------|
| Attention (24 layers) | 11.4 | 24% | 0.47 |
| Expert dispatch (4x24 layers) | 15.6 | 32% | 0.65 |
| Routing sync+read (24 layers) | 6.7 | 14% | 0.28 |
| Routing dispatch (24 layers) | 4.9 | 10% | 0.20 |
| Shared expert (24 layers) | 5.0 | 10% | 0.21 |
| Residual (24 layers) | 2.6 | 5% | 0.11 |
| LM head + final sync | 1.3 | 3% | 1.3 |
| Embedding + RoPE + pos | 0.8 | 2% | 0.8 |
| **Total** | **48.3** | **100%** | — |

## Q: How many ops are dispatched per token?

**A:** Estimated op count per layer:

| Component | Ops/layer |
|-----------|----------|
| Attention (rms_norm, 3x linear, RoPE, KV cache update, SDPA, O proj, residual) | 15 |
| Router (rms_norm, matmul, softmax, topk) | 4 |
| 4 routed experts (linear_silu + matmul + mul + matmul + multiply + add) | 16 |
| Shared expert (linear_silu + matmul + mul + matmul + sigmoid + mul + add) | 7 |
| Misc (residual add, rms_norm) | 2 |
| **Per layer total** | **44** |

Total per token: 44 x 24 layers + 3 (embed + final_norm + lm_head) = **1059 ops** (post-fusion).

At 30us/op: **~32ms of pure dispatch overhead** — 66% of the 48ms total. The remaining ~16ms is actual device compute + sync stalls.

## Q: Where is the biggest optimization opportunity?

**A:** Expert dispatch at 32% of total time. Each routed expert requires 4-6 ops, and we dispatch 4 experts x 24 layers = 96 expert forward passes per token. This is the cost of data-dependent routing: we cannot trace these ops because the expert indices change every token.

Three potential approaches to reduce expert dispatch cost:

### 1. All-experts-traced (run 60, mask 56)
Execute all 60 expert MLPs in a single trace, zero-mask unused outputs. Eliminates all expert dispatch overhead but increases bandwidth: 60 expert weight reads x 24 layers = ~12.5 GB/step at BFP8. At 450 GB/s = ~28ms for expert weights alone.

**Trade-off:** ~28ms bandwidth vs ~15ms dispatch + ~5ms compute = ~20ms current. All-experts is slower for batch=1, but becomes faster at higher batch sizes (bandwidth cost amortized, dispatch cost is not).

### 2. Expert group tracing
Profile which experts are commonly co-activated and pre-group them into traceable clusters. Reduces the number of trace captures needed while avoiding full 60-expert bandwidth.

### 3. Custom kernel
Write a single Metalium kernel that does router + expert dispatch on-device without host round-trips. Eliminates both dispatch overhead and sync stalls but requires writing custom RISC-V code.

## Q: Why does routing sync cost so much?

**A:** 6.7ms for 24 layers = 0.28ms/layer to read 32 bytes (4 indices + 4 probabilities). The cost is not bandwidth — it is the round-trip latency of `ttnn.synchronize_device()`:

1. Python calls `synchronize_device` (~5us)
2. Command queue flushes all pending commands (~50us)
3. Device executes any queued but not-yet-completed ops
4. Host polls for completion (~100-200us per poll cycle)
5. `from_dev` copies result to host (~10us)

The 0.28ms/layer is dominated by sync latency, not data transfer. This is why multi-CQ (exp 96) failed to help — `synchronize_device` drains both queues, defeating the purpose of pipelining.

## Q: What is the theoretical minimum latency for MoE decode?

**A:** Assuming zero dispatch overhead (fully traced or custom kernel):

| Component | Bandwidth | Compute | Latency |
|-----------|-----------|---------|---------|
| Attention weights (bf16) | 34 MB | — | 0.08ms |
| SDPA (flash decode) | — | — | ~0.5ms |
| 4 expert weights (BFP8) | 4 x 8.6 MB = 34 MB | — | 0.08ms |
| Shared expert weights (BFP8) | 33 MB | — | 0.07ms |
| Expert + shared compute | — | — | ~0.2ms |
| KV cache read (256 seq) | 24 x 2 x 256 x 128 x 2B = 3 MB | — | ~0.01ms |
| **Total per layer** | **~104 MB** | — | **~0.9ms** |
| **24 layers** | **~2.5 GB** | — | **~5.6ms** |
| + LM head | ~580 MB | — | ~1.3ms |
| **Total** | **~3.1 GB** | — | **~7ms** |

At 450 GB/s DRAM bandwidth: **~7ms = ~143 tok/s** theoretical ceiling for batch=1 MoE decode. Current 44ms is 6.3x off the ceiling, with dispatch overhead being the primary gap.

---

*Experiment 94. Qwen1.5-MoE-A2.7B-Chat profiling on Blackhole P150. April 2026.*
