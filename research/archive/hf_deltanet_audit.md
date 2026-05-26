# HF DeltaNet Recurrence Audit — Math is Identical

**Date**: 2026-05-12
**Status**: Audit complete. No bug found in the recurrence math itself.

## What we read

`/home/aditya/tt-xla/.venv/lib/python3.10/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`:
- `torch_recurrent_gated_delta_rule` (lines 314-355): single-token sequential form
- `torch_chunk_gated_delta_rule` (lines 234-313): chunked parallel form (used in prefill)
- `l2norm` (lines 228-233): standard `x * rsqrt(sum(x²)+eps)`

## Line-by-line comparison vs `91f.deltanet_step_ondevice`

| Step | HF | Ours | Match |
|---|---|---|---|
| L2 norm formula | `x * rsqrt((x²).sum(-1, keepdim=True) + 1e-6)` | same | ✓ |
| L2 norm dim | over k_head_dim (per-head) | over K_DIM (per-head) | ✓ |
| Cast to fp32 in recurrence | yes | we run fp32 throughout (B'9) | ✓ |
| Decay computation | `g_t.exp()` | `ttnn.exp(g)` | ✓ |
| State decay | `state * g_t` (broadcast over K, V) | `H_4d * decay` (same broadcast) | ✓ |
| `kv_mem` | `(state * k.unsqueeze(-1)).sum(dim=-2)` | `sum(H_decayed * k_col, dim=-2)` | ✓ |
| `delta` | `(v - kv_mem) * beta_t.unsqueeze(-1)` | `(v_3d - kv_mem) * beta_reshaped` | ✓ |
| State update | `state + k.unsqueeze(-1) * delta.unsqueeze(-2)` | `H_decayed + k_col * delta_reshaped` | ✓ |
| Output | `(state * q.unsqueeze(-1)).sum(dim=-2)` | `sum(H_new * q_col, dim=-2)` | ✓ |
| Q/K split from conv_out | `torch.split([key_dim, key_dim, value_dim])` | `ttnn.slice` at same offsets | ✓ |
| Conv1d direction | `nn.Conv1d` depthwise with causal left-padding | concat(conv_state, mixed_col) × weight + sum | ✓ (verified by tracing pos 0..4) |
| GQA repeat | `repeat_interleave(N_REP, dim=2)` | `gqa_interleave` (B'9.5 fix) | ✓ |
| Beta | `b.sigmoid()` | `ttnn.sigmoid(b_tt)` | ✓ |
| `g = -A_log.exp() * softplus(a + dt_bias)` | exact | same with `log(exp(x)+1)` for softplus | ✓ (softplus stable for x in [-7, +1]) |

## The one difference: Q scaling

```python
# HF (lines 326-327):
scale = 1 / (query.shape[-1] ** 0.5)   # = 1/sqrt(128) ≈ 0.0884
query = query * scale
```

We don't do this. **But it's cosine-invariant**: scaling Q by a constant α scales the output by α, which `RMSNorm` immediately normalizes away. So this doesn't explain the cosine drop.

(We should still add it for output magnitude correctness; it just doesn't fix the 0.508 cosine.)

## What this rules out

- Recurrence math sign/order errors
- Softplus instability (probe confirmed max|Δ|=0)
- L2 norm formula differences
- Conv1d direction
- GQA broadcast (we fixed earlier)
- Sigmoid vs silu (we use the right ones)
- Beta application
- Q split offsets
- Cast order

## What remains in the hypothesis space

1. **Input mismatch before recurrence**: one of (q, k, v, g, beta) differs from HF when we hand it to the recurrence. Substep dump will catch this.
2. **A ttnn op that behaves unexpectedly at our specific shape/dtype**: e.g., `ttnn.sum(x * y, dim=-2)` for shape [1, 48, 128, 128] fp32 may have a bug we haven't caught.
3. **Some subtle dtype demotion** somewhere in the recurrence chain.

All three require finer-grained substep capture for layer 2 to localize.

## Why this audit was still worth it

Even though we didn't find the bug, the audit confirms our math is correct. The bug is NOT in the conceptual algorithm — it's in some narrower spot (input prep or a specific ttnn op call). That's a much smaller search space for next session.

## Recommended next investigation

Write `experiments/91s_layer2_deltanet_substep.py`:
- Mirror HF's substep capture for layer 2 but on the ttnn side
- Capture: q_pre_l2, k_pre_l2, v, q_post_l2, k_post_l2, g, beta, then state at each of 5 positions, then recurrence_out per position
- Compare each tensor to HF's equivalents
- The first one that doesn't match within 0.99 cosine localizes the bug

This was the original Plan B. Now that the ttnn JIT bug is patched, it should run.
