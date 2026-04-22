# Wiki 54: Performance Optimization Experiments (82-84)

## Summary

Three experiments testing optimizations from research/ttnn_advanced_features.md:

| Optimization | Speedup | Quality | Verdict |
|---|---|---|---|
| Fused SiLU in matmul (exp 82) | 1.00x | Perfect | No effect — ops pipelined in trace |
| On-device topk/argmax (exp 82) | N/A | Correct | Too slow: 1890ms/124ms for vocab > 65K |
| BFP4 MLP weights (exp 83) | 1.02x (0.5B) | 0/20 match | Catastrophic without calibration |
| **BFP8 MLP + HiFi2 (exp 84)** | **1.20x (8B)** | **8/8 match** | **First real win: 18→21 tok/s** |

## Experiment 82: Fused SiLU + On-Device Topk

### Fused SiLU
Changed `g = silu(matmul(h, gate_w))` to `g = matmul(h, gate_w, activation="silu")`.
Result: 7.58ms → 7.58ms (no change). The separate silu kernel was already hidden behind other operations in the trace pipeline. At batch=1, 0.5B is not compute-bound.

### On-Device Topk
`ttnn.topk(logits, k=1)` on vocab=151936: **1890ms**. Falls back to single-core since width > 65536 (multicore requires < 65536). `ttnn.argmax`: **124ms**. Both unusably slow vs 3.9ms PCIe readback.

The on-device token pipeline (topk + embedding) is not viable for any of our models — all have vocab > 65K.

### On-Device Embedding
`ttnn.embedding` requires UINT32 index input (we passed INT32). Moot since topk/argmax are too slow to produce the index on-device.

## Experiment 83: BFP4 MLP Weights on 0.5B

Per-layer MLP cosine similarity vs bf16:
- Layer 0: 0.987, Layer 12: 0.974, Layer 23: 0.992 (looks survivable)
- End-to-end first-token logit cosine: **0.469** (catastrophic)
- Token match: **0/20**

The ~3% per-layer error compounds across 24 layers into total output destruction. BFP8 per-layer cosine is 0.999+ (safe). BFP4 requires GPTQ/AWQ calibration to be viable.

Speed: 7.58ms → 7.41ms (1.02x) — 0.5B is not bandwidth-bound, as expected.

## Experiment 84: BFP8 MLP on Llama-3.1-8B (THE WIN)

BFP8 weights for gate_proj, up_proj, down_proj. Attention weights stay bf16.

### Results
```
BF16:        trace=52.0ms  e2e=56.1ms  (18 tok/s)
BFP8+HiFi4: trace=43.4ms  e2e=47.4ms  (21 tok/s)  1.20x
BFP8+HiFi2: trace=43.0ms  e2e=47.0ms  (21 tok/s)  1.21x
```

- **8/8 token match** vs bf16: "The capital of France is Paris."
- **40% weight memory reduction**: 14.0 GB → 8.3 GB
- HiFi2 vs HiFi4: negligible difference, HiFi2 is the better default

### Why It Works
- 8B model IS bandwidth-bound (52ms vs 36ms ceiling)
- MLP weights are 80% of total weight reads (11.3 GB / 14.0 GB)
- BFP8 halves MLP reads: 11.3 GB → 5.6 GB
- Total reads: 14.0 GB → 8.3 GB → new ceiling ~18.4ms

### Why 0.5B Didn't Benefit
- 0.5B total weights: ~1 GB at bf16 → 2.2ms at 450 GB/s
- Actual trace time: 7.6ms (3.5x above ceiling)
- Bottleneck is trace overhead + SDPA, not bandwidth
- BFP8 saves weight reads but doesn't change the bottleneck

## Key Takeaways

1. **Bandwidth optimizations only help bandwidth-bound models** (8B+, not 0.5B)
2. **Fused ops don't help when ops are already pipelined in trace** at batch=1
3. **BFP8 is safe across architectures** (0.999+ cosine, proven on Qwen and Llama)
4. **BFP4 needs calibration** — naive conversion destroys quality
5. **On-device token selection is a dead end** for large vocabularies (> 65K)

## Next Steps

- [ ] Apply BFP8 MLP to all models > 1B (where bandwidth matters)
- [ ] Test BFP8 attention weights too (Q/K/V/O projections)
- [ ] Investigate DRAM-sharded matmul configs for better bandwidth utilization
- [ ] Look into reducing op count (fused RoPE, native 8-head GQA) to cut trace overhead
