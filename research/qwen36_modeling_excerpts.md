# Qwen3.6-35B-A3B — Modeling Code Excerpts (Phase A2)

Source: `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (HF transformers main branch). The Qwen3.6-35B-A3B repo uses this code (model_type `qwen3_5_moe` — same family as 3.5).

This doc resolves the four open questions from A1 and gives us the exact equations to port.

---

## Question 1 — Delta-rule update (the BIG one)

### Gated DeltaNet block structure

The block has **FOUR input projections**, not three:

```python
mixed_qkv = self.in_proj_qkv(hidden_states)   # packed Q+K+V (then conv'd, then split)
z         = self.in_proj_z(hidden_states)     # Mamba-style output gate
b         = self.in_proj_b(hidden_states)     # scalar gate per timestep (→ beta)
a         = self.in_proj_a(hidden_states)     # scalar gate per timestep (→ decay g)
```

Projection sizes (from config):
- `in_proj_qkv`: hidden=2048 → 2*key_dim + value_dim = 2*(16*128) + (32*128) = **8192**
- `in_proj_z`: hidden=2048 → value_dim = **4096**
- `in_proj_a`, `in_proj_b`: hidden=2048 → num_value_heads = **32** (one scalar per head)

So each DeltaNet layer has 4 input projections totaling ~28M params, then a 4096→2048 output projection (~8M), plus conv1d kernel and learned `A_log`, `dt_bias` params (negligible).

### Conv1d step

```python
if use_precomputed_states and seq_len == 1:
    mixed_qkv = self.causal_conv1d_update(...)
else:
    # prefill path: causal conv along seq dim with kernel=4
    mixed_qkv = causal_conv1d_fn(mixed_qkv, conv_weight, conv_bias, activation="silu")
```

Then split:
```python
query, key, value = torch.split(mixed_qkv,
    [self.key_dim, self.key_dim, self.value_dim], dim=-1)
# key_dim = 16*128 = 2048 (16 heads × 128 head_dim)
# value_dim = 32*128 = 4096
```

### The decay and gate

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)   # state decay
beta = b.sigmoid()                                                      # delta rate
```

- `g` is **strictly negative** (decay rate, will be exponentiated to get multiplicative factor)
- `A_log` is a learned PER-HEAD constant (shape: `[num_value_heads]`)
- `dt_bias` is also per-head, learned
- All computed in **float32** (per config: `mamba_ssm_dtype: float32`)

### Delta-rule recurrence (the core math)

Pseudocode from HF source:

```python
for i in range(sequence_length):                          # serial over time
    q_t = query[:, :, i]                                  # [B, n_v_heads, d_k]
    k_t = key[:, :, i]                                    # [B, n_v_heads, d_k]
    v_t = value[:, :, i]                                  # [B, n_v_heads, d_v]
    g_t = g[:, :, i]                                      # [B, n_v_heads]
    beta_t = beta[:, :, i]                                # [B, n_v_heads]

    # 1) decay state
    H = H * g_t.unsqueeze(-1).unsqueeze(-1).exp()         # [B, n_v_heads, d_k, d_v]

    # 2) read current V via K from state
    kv_mem_t = (H * k_t.unsqueeze(-1)).sum(dim=-2)        # [B, n_v_heads, d_v]

    # 3) delta correction
    delta_t = (v_t - kv_mem_t) * beta_t.unsqueeze(-1)     # [B, n_v_heads, d_v]

    # 4) update state: H += outer(K_t, delta_t)
    H = H + k_t.unsqueeze(-1) * delta_t.unsqueeze(-2)     # [..., d_k, d_v]

    # 5) read output via Q
    out[:, :, i] = (H * q_t.unsqueeze(-1)).sum(dim=-2)    # [B, n_v_heads, d_v]
```

**Key implementation notes:**
- State H shape: `[batch, n_v_heads=32, key_head_dim=128, value_head_dim=128]` = **4096 fp32 values per head × 32 heads = 524K fp32 = 2 MB per layer per batch**
- All 30 DeltaNet layers each carry their own H → **60 MB of fp32 state** at batch=1 (significant!)
- Q and K are L2-normalized inside the kernel (`use_qk_l2norm_in_kernel=True`)
- For DECODE: just one iteration of this loop per call
- For PREFILL: the loop is serial — Mamba family uses chunked scan / Heinsen scan for parallelism, ttnn likely doesn't have this kernel ready

### Output gate (the `z` channel)

After the recurrence:
```python
# z came from in_proj_z(hidden_states)
out = out * silu(z)   # Mamba-style output gating (per channel of value_dim)
out = self.out_proj(out)
```

So DeltaNet has *two* gating mechanisms: `beta` for the state update AND `silu(z)` for the output. They serve different purposes.

---

## Question 2 — Partial RoPE and M-RoPE for text-only

The forward applies standard `apply_rotary_pos_emb(q, k, cos, sin)`. The "partial" part is encoded in `cos`/`sin` themselves: the first 64 dims (`partial_rotary_factor=0.25 × head_dim=256`) have rotation values, the remaining 192 dims have cos=1, sin=0 (pass-through).

**For text-only inference**: M-RoPE collapses to 1D RoPE. The `mrope_section=[11, 11, 10]` only matters when we have (text, height, width) position triplets. For pure text generation, we feed plain integer positions and the rope_init_fn gives us the standard 1D RoPE values. Confirmed by the HF source — there's a branch for `position_ids.dim() == 2` vs 3D.

**Port consequence**: same `ttnn.experimental.rotary_embedding` we already use, but the cos/sin tables must be sized 64 (not 256). The 192 pass-through dims we handle by NOT applying RoPE to them.

---

## Question 3 — Output gate in Gated Attention

```python
# q_proj output is sized head_dim * 2 — split into Q and gate
qg = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
query_states, gate = torch.chunk(qg, 2, dim=-1)
# ... standard SDPA on (query_states, key_states, value_states) ...
attn_output = attn_output * torch.sigmoid(gate)
attn_output = self.o_proj(attn_output)
```

**So the gate is computed by DOUBLING the Q-projection's output channels** (4096 → 8192), then splitting half-and-half. The gate is per-head and per-channel. No separate gate-projection matmul.

This adds zero overhead vs computing Q normally — we just allocate 2× the projection columns.

**Port consequence**: `q_proj` is hidden(2048) → 16*256*2 = 8192. Slice to get Q and gate.

---

## Question 4 — MoE routing

```python
class Qwen3_5MoeTopKRouter:
    def forward(x):
        router_logits = self.weight(x)                       # [B*T, num_experts=256]
        router_probs = F.softmax(router_logits, dtype=fp32, dim=-1)
        top_value, top_idx = torch.topk(router_probs, k=8, dim=-1)
        top_value /= top_value.sum(dim=-1, keepdim=True)     # renormalize
        return router_logits, top_value, top_idx
```

Standard softmax + top-k + renormalize. Same pattern we have in Qwen1.5-MoE.

Then in the block:

```python
shared_expert_output = self.shared_expert(hidden_states_reshaped)
_, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

# the SHARED expert has its own sigmoid gate!
shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

# combine
final = expert_output + shared_expert_output
```

**Note on shared expert gate**: matches my prior memory (`feedback_shared_expert_gate.md`) — the shared expert has a [hidden_size → 1] linear `shared_expert_gate` that produces a per-token scalar through sigmoid. The same pattern as Qwen1.5-MoE.

---

## Summary table — every projection we'll need

| Layer | Projection | In dim | Out dim | Bias? | Params |
|---|---|---|---|---|---|
| DeltaNet | `in_proj_qkv` | 2048 | 8192 | no | 16.8M |
| DeltaNet | `in_proj_z` | 2048 | 4096 | no | 8.4M |
| DeltaNet | `in_proj_a` | 2048 | 32 | no | 66K |
| DeltaNet | `in_proj_b` | 2048 | 32 | no | 66K |
| DeltaNet | `out_proj` | 4096 | 2048 | no | 8.4M |
| DeltaNet | conv1d weight | (8192, 4) | — | yes | 32K |
| DeltaNet | `A_log`, `dt_bias` | — | (32,) each | — | 64 |
| **DeltaNet TOTAL** | | | | | **~34M / layer** |
| Full-attn | `q_proj` | 2048 | 8192 | no | 16.8M (Q + gate) |
| Full-attn | `k_proj` | 2048 | 512 | no | 1.0M |
| Full-attn | `v_proj` | 2048 | 512 | no | 1.0M |
| Full-attn | `o_proj` | 4096 | 2048 | no | 8.4M |
| **Full-attn TOTAL** | | | | | **~27M / layer** |
| MoE | `gate` (router) | 2048 | 256 | no | 0.5M |
| MoE | `experts[i].gate_proj` × 256 | 2048 | 512 | no | 1.0M × 256 = 256M |
| MoE | `experts[i].up_proj` × 256 | 2048 | 512 | no | 256M |
| MoE | `experts[i].down_proj` × 256 | 512 | 2048 | no | 256M |
| MoE | `shared_expert.gate_proj` | 2048 | 512 | no | 1.0M |
| MoE | `shared_expert.up_proj` | 2048 | 512 | no | 1.0M |
| MoE | `shared_expert.down_proj` | 512 | 2048 | no | 1.0M |
| MoE | `shared_expert_gate` | 2048 | 1 | no | 2K |
| **MoE TOTAL** | | | | | **~772M / layer** |

A1's back-of-envelope was right: MoE dominates by ~30×.

---

## Implementation order for Phase A

Now that we have the math, the isolation order makes sense:

1. **A2 (done — this doc).** All equations resolved.
2. **A0/A2.5 — MoE regression diagnosis.** Before doubling-down on MoE, find the Qwen1.5-MoE perf regression.
3. **A3 — Gated DeltaNet isolated.** Implement the decode path first (1-iter loop, easier), validate cosine vs numpy ref, measure µs/call vs memory ceiling.
4. **A4 — Gated Attention isolated.** Apply partial RoPE, output gate, run SDPA, validate cosine.
5. **A5 — MoE block isolated.** 256-expert routing + 1 shared expert with its own gate. Smaller test (e.g., 8 experts) first to debug.
6. **A6 — Conv1d kernel isolated** (just for DeltaNet's pre-conv).

Then **Phase B integration**.

---

## What's still uncertain

1. **Chunked scan for prefill.** Decode is easy (1-iter recurrence). Prefill is serial in time over up to 256K tokens — naive serial is unusable. Mamba uses chunked/Heinsen scan for parallelism. We'll need to either:
   - Implement chunked scan ourselves on ttnn
   - Limit prefill to a manageable context (e.g., 1024-token chunks scanned serially)
   - For Phase B, accept slow prefill and focus on decode quality
2. **Fp32 state H on Blackhole.** ttnn defaults to bf16. We need to verify that `dtype=ttnn.float32` works for the state tensor and that mixed-precision math (bf16 weights × fp32 state) is supported. Probably fine, just needs verification.
3. **conv1d with kernel=4.** ttnn has `ttnn.conv1d` for inference but the 1D-along-channel pattern (one kernel per channel, independent) is sometimes called depthwise conv1d. Check ttnn API.
