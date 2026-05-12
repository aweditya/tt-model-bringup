# Phase A0 — MoE Regression Microbench Findings

Date: 2026-05-11 / 2026-05-12 (qb1 was flapping; data is from two short windows)

## Method

`experiments/81_moe_regression_micro.py` — times each MoE-decode-loop surface in isolation on ttnn 0.69. We don't have ttnn 0.68 installed for a side-by-side; this is "what's expensive now" data. We compare to the per-op envelope our PJRT benchmarks (`research/pjrt_phase5_benchmarks.md`) already established.

200-iter median + p90 per op, with `synchronize_device` after each call (so we measure submit + execute, not just submit).

## Numbers (ttnn 0.69, Blackhole P150, decode shapes, batch=1)

| Op | shape | median µs | p90 µs |
|---|---|---:|---:|
| matmul `h @ w_h2i` | `[1,2048] @ [2048,1408]` | **64.5** | 66.5 |
| matmul `h @ w_i2h` | `[1,1408] @ [1408,2048]` | **92.7** | 94.5 |
| matmul `h @ w_router` | `[1,2048] @ [2048,60]` | **53.6** | 55.4 |
| silu(g) | `[1,1408]` | 68.0 | 73.4 |
| **mul(silu_g, u)** | `[1,1408] × [1,1408]` | **137.5** ⚠️ | 148.6 |
| add(h, h) | `[1,2048]` | 87.1 | 94.9 |
| multiply(g, 0.5) (scalar) | `[1,1408] × scalar` | 84.5 | 91.9 |
| softmax | `[1,60]` | 45.3 | 50.4 |
| **topk (k=4)** | `[1,60]` | **101-153** ⚠️ | 110-168 |
| synchronize_device (noop after sync) | — | 20.8 | 22.5 |

(`to_torch / one_expert / 4-expert / shared_expert` rows haven't completed yet — ssh broke before they ran.)

## Where the 20 ms regression went (best hypothesis)

Existing Qwen1.5-MoE 0.69 decode = 64 ms/tok (was 44 ms/tok on 0.68). Gap = +20 ms / 24 layers = **+0.83 ms/layer**.

Per-layer eager MoE op count (4 active experts):
- 1× rms_norm
- 4× `mul(silu_g, u)` — **the standout, ~50µs over a bare `multiply`**
- 1× shared expert (same SwiGLU pattern)
- 12× big matmuls (3 per expert × 4)
- 3× shared expert matmuls
- 1× router (matmul + softmax + topk)
- 1× sync + 2× to_torch readback
- ~4× add (accumulator + residual)

**Standout: `ttnn.mul(a, b)` of two tensors costs 137 µs vs `ttnn.multiply(a, scalar)` at 85 µs.** That's a 50µs penalty for the binary op vs the unary-with-scalar variant. Per layer:

- 4 active experts × 1 `mul` each + 1 shared expert × 1 `mul` = **5 binary muls per layer**
- 5 × ~50 µs penalty = **250 µs / layer overhead**
- × 24 layers = **6 ms per token from binary-mul cost alone**

Add the `topk` cost (101-153 µs × 24 = 2.4-3.7 ms) and we're already at ~9 ms of the 20 ms gap. The rest is presumably distributed across matmul dispatch (which itself crept up — we measured 64-92 µs, prior eager bench memory had ~50 µs for similar shapes).

## Diagnosis

Two distinct issues:

1. **Binary mul is expensive** in ttnn 0.69 — almost 2× a unary multiply. Either a kernel regression or a missing fast path for elementwise-binary at small shapes. Workarounds:
   - Fuse mul into the preceding matmul via `linear(activation="silu")` pattern (already done for `g = silu(matmul(h, w))`)
   - Compose `silu(g) * u` into a fused `swiglu` op if ttnn has one — `ttnn.swiglu` exists per memory of exp 97 BUT crashed on 4D shapes on Blackhole. May work at 2D shapes.

2. **topk(k=4) is 100-150 µs** — high. For 60 logits this should be fast. Worth a separate check of `ttnn.topk` vs hand-rolled `argmax` repeated 4 times.

## Recommendations

For our Branch III plan:

a. **Don't fight this regression for Qwen1.5-MoE** — fixing 0.69 internals isn't our job. Document it, move on.

b. **For Qwen3.6 MoE** — use these findings:
   - **Fuse SwiGLU**: try `ttnn.swiglu` at the 2D shapes we'll use. If it works, save ~50 µs per expert per layer.
   - **Re-evaluate topk**: 256 experts top-8 will be even more expensive on raw `ttnn.topk`. May need to look at sharded topk or device-side argmax-based alternatives.
   - **Asymmetric matmul cost** (1408→2048 is 1.4× slower than 2048→1408): worth investigating with our Phase B integration. May be related to layout/sharding.

c. **For PJRT** — these per-op numbers match what `pjrt_phase5_benchmarks.md` reported. Confirms our PJRT-traced speedup is real and the floor is set by ttnn dispatch itself.

## Limitations of this measurement

- No 0.68 baseline. Differences vs prior tok/s are inferred.
- `to_torch / one_expert / 4-expert / shared_expert` composites not yet measured (qb1 ssh dropped).
- Single op-at-a-time bench includes per-call sync. Real decode runs many ops between syncs, which Both pipeline (cheaper than measured here) and queue up (potentially slower if back-pressure).

## Next steps

1. Re-run when qb1 stabilizes to capture composites + readback time.
2. Test `ttnn.swiglu` at decode shapes — could be a 100 ms/sec improvement for free.
3. Phase A3 — these numbers feed into the DeltaNet ceiling analysis (mul/add costs apply).
