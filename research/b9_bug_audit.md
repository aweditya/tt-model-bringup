# Phase B'9 — HF Source Audit: Four Bugs in Our Qwen3.5 Implementation

**Date**: 2026-05-12
**Trigger**: 91o (HF oracle) revealed missing `linear_attn.norm.weight`. Deeper audit of `transformers/models/qwen3_5/modeling_qwen3_5.py` exposed three more.

## How we missed these

Our numpy reference (B'2) was written from architecture intuition + partial HF code reading. We never enumerated the safetensors keys per layer and cross-checked against our loader. The B'4-B'7 ttnn-vs-numpy cosines (0.997+) were comparing ttnn to a buggy numpy reference, so they masked the bugs.

**Process improvement**: in every future model bringup, the FIRST sanity check is "list all weight keys in safetensors for a representative layer; ensure the loader requests every one of them." We'll add this to the wiki.

## The four bugs

### Bug #1 — Missing `linear_attn.norm.weight` (DeltaNet RMSNormGated)

`Qwen3_5GatedDeltaNet.forward`, line 540:
```python
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
z = z.reshape(-1, self.head_v_dim)
core_attn_out = self.norm(core_attn_out, z)    # ← MISSING IN OUR IMPL
core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
output = self.out_proj(core_attn_out)
```

Where `self.norm` is `Qwen3_5RMSNormGated`:
```python
class Qwen3_5RMSNormGated(nn.Module):
    def forward(self, hidden_states, gate=None):
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states                     # standard w*x
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)
```

We do `out * silu(z)` directly. HF does `RMSNorm(out) * silu(z)`. Critically: the RMSNorm is **per-head** (operates over `head_v_dim = 128` dimensions, not the full output).

### Bug #2 — Missing `self_attn.q_norm.weight`

`Qwen3_5Attention.forward`, line ~660:
```python
query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
```

Q is RMSNorm'd **per-head** (`head_dim=256`) BEFORE RoPE. We skip this entirely.

### Bug #3 — Missing `self_attn.k_norm.weight`

```python
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
```

Same as #2 but for K.

### Bug #4 — `Qwen3_5RMSNorm` uses `(1.0 + weight)`, not `weight`

```python
class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))    # init at ZERO, not ONE

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # ← 1 + w
        return output.type_as(x)
```

vs. standard RMSNorm: `output * weight` with weight init at one.

Qwen3.5 parameterizes the learned scale as an **offset from 1**, so trained weights are roughly mean-zero (small perturbations to the identity scale). Standard RMSNorm parameterizes as the scale itself, so trained weights are roughly mean-one.

**Affected norms** (use `Qwen3_5RMSNorm`):
- `final_norm` (mean +0.96 → effective scale 1.96)
- `input_layernorm` per layer (mean -0.024 → effective scale 0.98)
- `post_attention_layernorm` per layer (mean -0.21 → effective scale 0.79)
- `q_norm` per attention layer (mean +0.22 → effective scale 1.22)
- `k_norm` per attention layer (mean +0.21 → effective scale 1.21)

**NOT affected** (use `Qwen3_5RMSNormGated`):
- `linear_attn.norm` per DeltaNet layer (mean +0.88, used as standard `w * x`)

### Weight stats that confirm bug #4

| Norm | mean | std | min | max | type |
|---|---:|---:|---:|---:|---|
| `final_norm` | +0.962 | 0.136 | -0.27 | +1.76 | RMSNorm (1+w) |
| `layers.0.input_layernorm` | -0.024 | 0.043 | -0.13 | +0.20 | RMSNorm (1+w) |
| `layers.0.post_attn_layernorm` | -0.211 | 0.038 | -0.99 | +0.01 | RMSNorm (1+w) |
| `layers.0.linear_attn.norm` | +0.877 | 0.031 | +0.76 | +0.94 | RMSNormGated (w) |
| `layers.3.input_layernorm` | +0.244 | 0.058 | +0.01 | +0.89 | RMSNorm (1+w) |
| `layers.3.post_attn_layernorm` | -0.103 | 0.079 | -1.00 | +0.11 | RMSNorm (1+w) |
| `layers.3.self_attn.q_norm` | +0.222 | 0.072 | -0.20 | +0.44 | RMSNorm (1+w) |
| `layers.3.self_attn.k_norm` | +0.211 | 0.135 | -0.60 | +0.71 | RMSNorm (1+w) |

The clean separation between mean ≈ 0 (RMSNorm) and mean ≈ 1 (RMSNormGated) is the smoking gun for bug #4.

## Combined effect

For every `Qwen3_5RMSNorm` (161 sites total across the 27B), our `ttnn.rms_norm(x, w)` computes `output * w_raw` where `w_raw` is centered near 0. The correct math is `output * (1 + w_raw)` centered near 1. We're effectively scaling every normalized hidden state by a near-zero quantity at the input to every layer body. Combined with the two missing per-head Q/K RMSNorms and the missing per-head DeltaNet RMSNormGated, the whole forward pass produces hidden states with corrupted scale and missing structure.

The fact that our hidden state norms still grew across 64 layers (7.7 → 150) is because the residual stream skips the layer body via the residual connection — `x_out = x_in + body_out` — so even when `body_out ≈ 0`, `x` accumulates from the embedding. But the SEMANTIC content of `x` is dominated by the un-processed embedding, not the layer computations.

## Fix plan (incremental, gated on cosine)

1. **Weight loader**: add the 3 missing keys (linear_attn.norm, q_norm, k_norm); pre-process all Qwen3_5RMSNorm weights as `(1.0 + raw)` so existing `ttnn.rms_norm(x, w_loaded)` calls produce correct math without changes to call sites
2. **DeltaNet kernel**: add per-head RMSNorm before silu-gate
3. **Gated Attn kernel**: add per-head q_norm and k_norm before RoPE
4. **Validation**: write 91p (single-layer-0 ttnn forward) and compare to HF reference from 91o. Gate at cosine ≥ 0.999.
5. **Full model**: re-run 91l with 60 tokens. Eyeball coherence; expect `'Paris'` in top-5 at minimum.

## Future-proofing

Adding to wiki: a `bringup_checklist.md` that every model port must satisfy:
- [ ] Enumerate all weight keys in safetensors for layer 0 and layer N (where N spans all distinct layer types)
- [ ] Cross-check loader requests against safetensors keys; assert equality
- [ ] Compare numpy reference output AGAINST AN EXTERNAL ORACLE (HF or vendor reference), not against our own implementation
- [ ] Audit every RMSNorm/LayerNorm class's exact formula (`w*x` vs `(1+w)*x` vs `w*(x-mean)/std`)
- [ ] Audit every gating operation's activation (sigmoid vs silu vs tanh) and order (norm-then-gate vs gate-then-norm)
