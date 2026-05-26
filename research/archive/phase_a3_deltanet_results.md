# Phase A3 Results — Gated DeltaNet Isolated Kernel

Test rig: `experiments/82_gated_deltanet.py` on qb1, ttnn 0.69, device 0.

## Correctness (gate: cosine ≥ 0.99)

| | Value | Gate |
|---|---:|:---:|
| **cosine(out_np, out_ttnn)** | **0.999995** | ≥ 0.99 ✓ |
| **cosine(H_new_np, H_new_ttnn)** | **0.999999** | ≥ 0.99 ✓ |
| max-abs-diff (out) | 0.000881 | |
| max-abs-diff (H_new) | 0.000514 | |

**PASS on first try.** Q,K,V,g,β stored in bf16; recurrent state H in fp32 (per `mamba_ssm_dtype: float32`). bf16×fp32 mixed-precision math worked without explicit casts; ttnn handled it cleanly.

## Performance — per single-token decode step

```
Inputs:  Q,K,V,g,β bf16; H_prev fp32 [B=1, n_v_heads=32, d_k=128, d_v=128]
State size: 4.2 MB
Memory ceiling on 1× P150 (450 GB/s): 9.3 µs  (read + write H)
```

| Mode | Median µs | p90 µs | % of ceiling | Speedup vs eager |
|---|---:|---:|---:|:---:|
| Eager  | 1325 | 1788 | 0.70 % | 1.00× |
| Traced |  251 |  252 | 3.72 % | **5.29×** |

## Per-chip utilization analysis

At decode time, the recurrence has ~15 individual ttnn ops:
- 4 mul (L2 norm sub-pieces × 2 for Q and K)
- 2 rsqrt
- 1 exp
- 4 reshape (for broadcasting)
- 1 outer-product mul
- 2 sum reductions
- 1 sub, 2 add

**Eager (1325 µs)**: 15 ops × ~80-90 µs dispatch each → matches. Pure dispatch-bound.

**Traced (251 µs)**: still 15 device-side kernel launches in the trace, but no Python→C++ dispatch round-trip per op. ~17 µs per op average — much better, but still bounded by kernel launch on-device.

**To approach the 9.3 µs memory ceiling on 1 chip we'd need:**
- **Fusion**: combine the elementwise sequence into a single kernel (e.g., outer + add + decay → one custom kernel). Could plausibly land at ~30-50 µs.
- **Multi-chip TP**: shard the 32 v-heads across chips, getting ~4× the bandwidth. Doesn't reduce per-step time on a *single* chip but lets us push more tokens through.

Per the kickoff invariant: **multi-chip is NOT a workaround for poor single-chip util**. Current 3.7% single-chip util is honest "decomposed kernel" territory — the path to higher util is fusion, not more chips.

## What this means for Branch III timeline

A whole-model DeltaNet pass (30 layers): 30 × 251 µs = **7.5 ms per token** just for the recurrences (excluding projections, output gate, conv1d). Plus 10 Gated Attention layers and 40 MoE blocks. Realistic Phase B per-token estimate at single-chip: 30-50 ms. Matches the ceiling-floor analysis in A1.

## Why this kernel was easy

Mostly because the recurrence is a sequence of independent elementwise ops + reductions — no novel primitives needed. The hard parts I'd flagged in the plan turned out fine:

- **fp32 state on Blackhole**: works as-is via `dtype=ttnn.float32`.
- **bf16 × fp32 mixed**: handled by ttnn without explicit casts.
- **Broadcasting `k[..., :, None] * delta[..., None, :]` for outer product**: explicit reshape via `ttnn.reshape` and `ttnn.mul` broadcasts correctly to [..., d_k, d_v].
- **L2 normalize**: composed manually from `sum(x*x) → rsqrt → mul`. No ttnn.l2_normalize needed.

## What's still pending for Phase B integration

- In-projections (`in_proj_qkv` / `_z` / `_a` / `_b`): standard `ttnn.linear`, nothing new.
- Conv1d kernel=4 (only for prefill; decode uses a small state update on the conv cache, also standard).
- Output `silu(z)` gate + `out_proj`: standard.
- GQA repeat from 16 k-heads to 32 v-heads: just `ttnn.repeat` or implicit broadcast.

None of those should be hard — A3 isolated the actually-new physics.

## Stretch experiments to consider (not blocking)

1. **Fusion**: write a custom ttnn kernel that combines decay + outer-product + add into one launch. Could approach 50-100 µs.
2. **Wider batches**: how does the recurrence scale with B>1? Per-token cost should stay roughly constant (the ops are batched over the first dim).
3. **Longer sequences (prefill)**: this is Phase A6. The recurrence is serial-over-time without a scan algorithm.

## Status

✅ Phase A3 complete.
→ Next: Phase A4 — Gated Attention isolated.
