# Sharded Memory and Dispatch Overhead

## Q: Does block-sharding fix the L1 performance problem from Experiment 08?

**A: No — for elementwise ops, block-sharding is about equal to DRAM interleaved.** The operations are so fast (~60-120µs) that we're dominated by dispatch overhead, not memory latency. Block-sharding at 1024×1024 was actually 0.74x (slower) and at 2048×2048 was 0.97x (equal).

However, **L1 interleaved intermediates DO help for larger elementwise chains** — a 3-op chain (add→relu→add) at 2048×2048 with L1 intermediates was **1.38x faster** than DRAM. This suggests XLA fusion keeping intermediates in L1 is beneficial for bandwidth-bound elementwise sequences, but only at larger sizes.

## Results

### Block-Sharded Elementwise

| Size | DRAM (ms) | Block-Sharded L1 (ms) | Speedup |
|------|-----------|----------------------|---------|
| 1024×1024 | 0.067 | 0.091 | 0.74x |
| 2048×2048 | 0.114 | 0.117 | 0.97x |

The overhead of setting up shard specs and the data redistribution costs negate any L1 latency advantage at these sizes.

### Elementwise Chain (add→relu→add)

| Size | DRAM (ms) | L1 Interleaved (ms) | Block-Sharded (ms) |
|------|-----------|---------------------|---------------------|
| 1024×1024 | 0.119 | 0.122 (0.98x) | 0.208 (0.57x) |
| 2048×2048 | 0.223 | 0.162 (**1.38x**) | 0.215 (1.04x) |

At 2048×2048, L1 interleaved intermediates save ~0.061 ms across 3 ops — that's ~20µs saved per intermediate by avoiding a DRAM round-trip. Block-sharded doesn't help here, likely due to redistribution overhead.

### Per-Op Dispatch Overhead (the key finding)

| Measurement | Time |
|-------------|------|
| Single add dispatch (no sync) | **0.021 ms** (21µs) |
| Single add dispatch + sync | 0.055 ms |
| 10-op chain, sync at end | 0.267 ms (27µs/op) |

**Each TT-NN op call costs ~21µs in Python dispatch overhead.** This means:
- A 100-op neural network forward pass wastes **~2.1 ms** just in dispatch
- A 1000-op transformer layer wastes **~21 ms** — potentially more than the actual compute

This is the "interpretation tax" — the cost of walking a graph node-by-node and dispatching each operation from Python. XLA compilation eliminates this by compiling the entire graph into a single device program.

## Q: What does this tell us about the value of XLA on Tenstorrent?

**A:** The case for XLA compilation on Tenstorrent has three legs:

1. **Dispatch elimination** (measured: ~21µs/op) — For deep graphs, this alone could give 2-10x speedup. This is the same motivation as `torch.compile` on CUDA.

2. **L1 intermediate placement** (measured: up to 1.38x for elementwise chains) — Keeping intermediates in L1 between fused ops avoids DRAM round-trips. The benefit scales with data size and chain length.

3. **Kernel fusion** (not yet measured) — Fusing multiple elementwise ops into a single Metalium kernel would eliminate dispatch overhead AND intermediate storage entirely. This is what tt-mlir should eventually do.

The dispatch overhead alone makes a compelling case. Even without sophisticated fusion, just batching ops into a single compiled program would help.

## Q: Why does block-sharding hurt performance here?

**A:** Two likely reasons:

1. **Setup overhead**: Creating ShardSpec, redistributing data across cores — this has a fixed cost that dominates for fast elementwise ops.

2. **Kernel compatibility**: Not all TT-NN kernels are optimized for sharded inputs. The elementwise add kernel may fall back to a generic path when it receives sharded data.

Block-sharding is designed for large matmuls and attention patterns where you carefully partition the computation across cores. For elementwise ops, the data parallelism is trivial and DRAM interleaving works just fine.

## Experiment

`experiments/09_sharded_memory.py` — run on Blackhole p150a device 0, 2026-04-21.

## Sources
- Experiment 09 results
- Experiment 08 results (L1 interleaved slower than DRAM for matmul)
- TT-NN documentation on memory configs and sharding
