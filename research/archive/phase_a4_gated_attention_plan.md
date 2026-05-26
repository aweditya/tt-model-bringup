# Phase A4 — Gated Attention (Isolated Kernel)

## Same rules as A3

Plan first → numpy fp32 reference → ttnn impl → cosine ≥ 0.99 → perf vs ceiling. Single device. Permanent file at `experiments/83_gated_attention.py`.

## What we're isolating

The SDPA-decode path inside Qwen3.6's `Qwen3_5MoeAttention` for a single token (decode case). Per A2:

```
# Q-projection is 2× width: produces (query, gate) split in half.
# K, V are 8× narrower (GQA 16/2 ratio).
q_packed = q_proj(x)               # [B, T, n_q_heads*head_dim*2 = 8192]
q, gate = chunk(q_packed, 2, dim=-1)   # each [B, T, 4096]
k = k_proj(x)                      # [B, T, n_kv_heads*head_dim = 512]
v = v_proj(x)                      # [B, T, 512]

# Reshape to heads
q: [B, T, n_q_heads=16, head_dim=256]
k, v: [B, T, n_kv_heads=2, head_dim=256]

# Partial RoPE — apply rotation to first 64 of 256 head dims; rest pass through
q[..., :64], k[..., :64] = apply_rope(q[..., :64], k[..., :64], cos, sin)

# SDPA over (Q, K, V, KV-cache)
attn_output = scaled_dot_product_attention(q, k, v, kv_cache, cur_pos)

# Output gate: multiply by sigmoid of the gate channel
attn_output = attn_output * sigmoid(gate)

# Output projection 4096 → 2048
output = o_proj(attn_output)
```

## What we exclude from A4

- The projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) — standard linears, no new physics
- The KV cache `paged_update_cache` — we have this working in `demos/generate_moe.py`
- The exact `ttnn.transformer.scaled_dot_product_attention_decode` call — assume we use this for the SDPA part

A4 tests the **interesting glue** in this attention block:
1. Partial RoPE on first 64 dims of head_dim 256
2. The output gate (sigmoid of the Q-projection's gate-half)
3. SDPA-decode call with our exact GQA shape (16 Q heads, 2 KV heads, head_dim 256)

These three are what make Qwen3.6's attention different from what we've ported before.

## Test plan

Build a self-contained kernel that:
1. Takes Q (already projected & reshaped, with gate appended), K, V, cos, sin, KV-cache, pos
2. Splits Q/gate, applies partial RoPE, runs SDPA-decode, applies sigmoid gate, returns output (no o_proj)
3. Compares cosine ≥ 0.99 vs numpy fp32 reference of the same math

Skip the upstream projections — assume Q has shape `[B, 1, 16, 256]` (post-split into Q+gate), K and V are `[B, 1, 2, 256]`, with pre-populated KV cache from random history.

## Memory ceiling

Per decode-step in this block (single token, KV cache length N=128 say):
- Read Q: 1×16×256×2 bytes = 8 KB
- Read K, V cache: 2×(128 × 2 × 256 × 2) = 256 KB  (already on device, may not move)
- Output: 1×16×256×2 = 8 KB
- gate sigmoid: 1×16×256×2 = 8 KB
- SDPA intermediate: dominated by KV-cache scan

For KV-cache length N=128: total ~270 KB I/O → 0.6 µs floor at 450 GB/s.

For longer caches the floor scales linearly: N=4K tokens → ~10 ms floor for full attention. This is why DeltaNet's O(D²) constant-cost is so attractive for long context.

## Open questions

1. **Partial RoPE in ttnn**: `ttnn.experimental.rotary_embedding` rotates the FULL last dim. For partial rotary (first 64 of 256), do we slice → rotate → concat back? Or is there a `partial_rotary_factor` argument?
2. **SDPA shape constraints**: `ttnn.transformer.scaled_dot_product_attention_decode` was tested in our Qwen1.5-MoE port at head_dim=128, 16 Q / 16 KV (no GQA). Need to verify it works at head_dim=256 with GQA 16/2.
3. **Sigmoid gate application**: we measured binary mul in A0 is 137 µs. Avoid where possible by using `ttnn.sigmoid` then `ttnn.mul` — or check if SDPA has an output-gate fused option.

## Verdict criteria

- Cosine ≥ 0.99 against numpy ref
- µs/call measured (eager and traced)
- % of memory ceiling reported
- One-line: pass / fail / blocked-on-X

## Stretch

If A4 passes quickly, sketch A5 (MoE block) plan and start.
