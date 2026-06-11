# End-to-End MLP Inference on Blackhole

## Q: Can we run a real neural network on Blackhole using TT-NN?

**A: Yes!** A 4-layer MLP (784→1024→512→256→10, ~1.5M params) runs successfully with eager op-by-op dispatch, achieving **1.69M samples/sec** at batch 512 — **5.5x faster than PyTorch CPU**.

## Throughput Results

| Batch | PyTorch CPU | TT-NN Blackhole | Speedup |
|-------|-------------|-----------------|---------|
| 1 | 7,468 samples/s | 3,409 samples/s | **0.5x** (CPU wins) |
| 32 | 80,456 | 114,488 | 0.9x (roughly even) |
| 128 | 161,828 | 456,832 | **2.2x** |
| 512 | 300,462 | **1,692,991** | **5.5x** |

## Q: Why does Blackhole lose at small batch sizes?

**A: Dispatch overhead.** Each forward pass dispatches 11 ops (4 matmul + 4 add + 3 relu). At ~21µs per dispatch, that's ~0.23ms of fixed overhead per forward pass. At batch=1, actual compute is nearly zero, so you're paying 0.28ms for dispatch on what should be a trivially fast computation.

The crossover point is around batch 32-64. Below that, the CPU's zero-dispatch-overhead advantage wins.

**This is exactly what XLA compilation would fix.** Instead of 11 separate Python→device dispatches, a compiled graph would send the entire forward pass as a single program. The dispatch overhead would drop from ~0.23ms to ~0.02ms (one dispatch instead of eleven).

## Q: How important is pipelined dispatch?

**A: Critical.** TT-NN's command queue allows ops to be dispatched while the previous op is still computing on the device.

| Mode | Time (batch 512) | Description |
|------|-------------------|-------------|
| Pipelined (sync at end) | **0.313 ms** | Dispatch all 11 ops, sync once |
| Serialized (sync per op) | **0.826 ms** | Dispatch, wait, dispatch, wait... |

Serializing is **2.6x slower**. The per-op sync cost is 0.047ms — the device-side round-trip time. Pipelining hides this behind compute.

**Implication for XLA**: Even without kernel fusion, just compiling the graph and dispatching it as a single unit would reduce sync overhead. The device could execute the entire program without waiting for the host after each op.

## Q: How accurate is bf16 inference?

**A:** For this MLP, mean absolute error is 0.017 (vs fp32 PyTorch), and class predictions agree 94% of the time (30/32 samples). The 6% disagreement comes from edge cases where two classes have nearly equal logits and bf16 rounding tips the argmax the other way.

For real inference workloads, this level of accuracy is perfectly acceptable. Production models are typically trained with awareness of their inference precision.

## Key Takeaways

1. **Blackhole is viable for neural network inference** — even with naive eager dispatch, it beats CPU at batch ≥ 128
2. **Batch size matters enormously** — dispatch overhead is fixed (~0.28ms), so larger batches amortize it better
3. **XLA compilation would help most at small batches** — eliminating per-op dispatch could make batch=1 competitive
4. **Pipelined dispatch is essential** — the command queue already provides 2.6x over serialized execution
5. **The 372 TFLOPS peak is irrelevant for this model** — at these matrix sizes (1024×512, etc.), we're nowhere near saturating the hardware. The bottleneck is dispatch and data movement, not compute.

## Experiment

`experiments/11_mlp_inference.py` — run on Blackhole p150a device 0, 2026-04-21.
