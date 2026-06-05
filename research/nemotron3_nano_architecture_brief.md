# Nemotron-3 Nano 30B-A3B (BF16) — architecture brief for Tenstorrent Blackhole bringup

Self-contained research brief for planning a from-scratch bringup of
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on Blackhole. Sourced from
primary HF artifacts + transformers PR + Nvidia release + the original
Nemotron-H paper (arXiv 2504.03624). Cross-referenced where possible;
disagreements flagged in §2.

> **Bringup posture**: this model is the **biggest single departure**
> from the 27B / 35B / Gemma 4 lineage we've shipped so far. It is a
> **Mamba2-Transformer hybrid** (NOT a pure transformer), with a custom
> MoE block that has a sigmoid router + group-restricted routing, NO
> RoPE on the attention layers, and an `MEMEM*EMEMEM*...` layer
> dispatch pattern. The recipe in `research/model_bringup_recipe.md`
> applies, but **a Mamba2 SSM kernel is brand new code on Blackhole**
> and is the gating concern.

------------------------------------------------------------------------

## 1. TL;DR

- **Architecture class**: `NemotronHForCausalLM` (`model_type=nemotron_h`).
  This is **Nemotron-H** family (arXiv 2504.03624 lineage), **NOT** a
  Qwen-MoE / Mixtral / DeepSeek variant. Custom remote code
  (`trust_remote_code=True` required); also upstreamed in transformers
  `transformers/models/nemotron_h/`.
- **Total / active**: ~30B total, ~3.5B active per token
  (model card claim; another Nvidia source quotes 31.6B total / 3.2B-3.6B
  active — see §2). 13 safetensors shards, 63.2 GB on disk.
- **52 layers, dispatched via `hybrid_override_pattern`**:
  `"MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"`
  → **23 Mamba2** + **23 MoE** + **6 Attention** layers. Attention
  layers sit at indices `[5, 12, 19, 26, 33, 42]` (the six `*`
  positions — verified by counting the pattern string).
- **Attention shape**: 32 Q heads, **2 KV heads** (16:1 GQA),
  `head_dim=128`, **NO RoPE applied** (config has `rope_theta`/
  `partial_rotary_factor` fields but the modeling file does not consume
  them; positional information lives entirely in the Mamba2 layers).
- **MoE shape**: 128 routed experts + 1 shared, top-6 routed,
  `moe_intermediate_size=1856` per routed expert,
  `moe_shared_expert_intermediate_size=3712`. **Sigmoid router** (not
  softmax) with **group-restricted routing** (n_group=8, topk_group=1)
  and `routed_scaling_factor=2.5`.
- **Mamba2 shape**: 64 heads × 64 head_dim = 4096 SSM "intermediate",
  `ssm_state_size=128`, `n_groups=8` (NB: config has both `n_group=1`
  and `n_groups=8` — `n_group` is for MoE, `n_groups` is the Mamba
  matrix grouping), `conv_kernel=4`, `expand=2`, `chunk_size=128`.
- **Vocab**: 131072 (smallest of any model we've bringup'd; 27B=152064/
  248320, 35B=248320, Gemma 4=262144). `tie_word_embeddings=False`.
  Hidden=2688 (also smallest).
- **Context**: `max_position_embeddings=262144` (256K). Model card claims
  up to 1M is supported.
- **No logit softcap**, no embed scale, no per-layer learned scalar, no
  q_norm/k_norm/v_norm, NO RoPE. Compared to Gemma 4 this is much
  simpler in the dense parts and much harder in the recurrent parts.

------------------------------------------------------------------------

## 2. Source table (claim → URL → confidence)

| Claim | Source | Confidence | Notes |
|---|---|---|---|
| `model_type=nemotron_h`, `NemotronHForCausalLM` | HF raw `config.json` | HIGH | Verbatim |
| 52 layers, hybrid `MEMEM*…` pattern | HF raw `config.json` (`num_hidden_layers`, `hybrid_override_pattern`) | HIGH | Verbatim |
| Pattern char map `M→mamba, *→attention, -→mlp, else→moe` | HF raw `configuration_nemotron_h.py` `layers_block_type` property | HIGH | Direct quote from source |
| 23 Mamba + 23 MoE + 6 Attention | HF model card + manual count of pattern string | HIGH | Pattern has 23 'M', 23 'E', 6 '*' |
| 30B total / 3.5B active | HF model card | MEDIUM | OpenRouter listing says "31.6B total / 3.2B-3.6B active" — likely "with vs without embeddings" framing. Both sources agree on the OOM. Bringup posture: budget for 31.6B-equivalent weights. |
| 128 routed experts + 1 shared, top-6 | `config.json` `n_routed_experts=128`, `n_shared_experts=1`, `num_experts_per_tok=6` | HIGH | Verbatim |
| Hidden=2688, intermediate=1856 (dense MLP slot, unused here), moe_intermediate=1856, moe_shared_intermediate=3712 | `config.json` | HIGH | Verbatim. Pattern has no 'mlp' slots so dense `intermediate_size` is unused. |
| 32 Q heads, 2 KV heads, head_dim=128 | `config.json` `num_attention_heads=32`, `num_key_value_heads=2`, `head_dim=128` | HIGH | Verbatim |
| Mamba2: 64 heads × 64 head_dim, ssm_state=128, n_groups=8, conv_kernel=4, expand=2 | `config.json` `mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`, `n_groups=8`, `conv_kernel=4`, `expand=2` | HIGH | Verbatim |
| vocab=131072 | `config.json` `vocab_size=131072` | HIGH | Verbatim |
| `tie_word_embeddings=False`, no softcap | `config.json` | HIGH | Verbatim; `_tied_weights_keys=["lm_head.weight"]` in modeling but config says False (see §4.5) |
| **NO RoPE applied to attention** | `modeling_nemotron_h.py` — grep for `apply_rotary_pos_emb`/`rotary_emb`/`rope`/`cos`/`sin` returns NOTHING in `NemotronHAttention.forward`; commented `#TODO position_embeddings`; `position_ids` accepted but unused | **HIGH** | This is a critical finding — config fields `rope_theta=10000`, `partial_rotary_factor=1.0` are present but the modeling does not consume them. Cross-check: `transformers/docs/.../nemotron_h` lists them as config fields without confirming usage. |
| RMSNorm formula `y = (x/sqrt(var+eps)) * w` (Llama-style, no +1, no bias) | `modeling_nemotron_h.py:1042-1059` `NemotronHRMSNorm.forward` | HIGH | Direct quote |
| One pre-norm per layer, no post-norm | `modeling_nemotron_h.py:1065-1121` `NemotronHBlock.forward` | HIGH | Direct quote; residual = h; h = norm(h); h = mixer(h); return residual + h |
| MoE: sigmoid router, top-k via groups, norm_topk + routed_scaling | `modeling_nemotron_h.py:1264-1269` `NemotronHTopkRouter` | HIGH | `scores = router_logits.sigmoid()`; group-restricted top-k; if norm_topk: `topk_weights /= sum`; then `topk_weights *= routed_scaling_factor` |
| Shared expert added AFTER routed sum | `modeling_nemotron_h.py:1225` | HIGH | `hidden_states + self.shared_experts(residuals)` |
| 25T training tokens, WSD schedule, peak lr=1e-3 | HF model card | MEDIUM | Public claim; bringup-irrelevant but contextually useful |
| Released 2025-12-14 | OpenRouter / NVIDIA blog | MEDIUM | Contextual |
| Open license (Nvidia Open Model License), not gated | HF tree listing | HIGH | No HF login required for download (verified by raw-file fetch succeeding without token) |
| Nemotron-H paper architecture parallel | arXiv 2504.03624 abstract (full paper not fetched) | MEDIUM | Abstract confirms hybrid Mamba-Transformer + roughly 8% attention layers. 6/52 = 11.5% for this model — close. |

**Disagreement flag**: 30B/3.5B (HF card) vs 31.6B/3.2B-3.6B (OpenRouter
listing). Likely "model.safetensors weights only" vs "with embedding
parameters". For Blackhole memory planning, treat as ~32 GB BF16 weights
total (matches the 63.2 GB on-disk total minus embedding amortization).

------------------------------------------------------------------------

## 3. Verbatim config.json

```json
{
  "architectures": ["NemotronHForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModel": "modeling_nemotron_h.NemotronHForCausalLM",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM"
  },
  "bos_token_id": 1,
  "chunk_size": 128,
  "conv_kernel": 4,
  "eos_token_id": 2,
  "expand": 2,
  "head_dim": 128,
  "hidden_dropout": 0.0,
  "hidden_size": 2688,
  "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
  "initializer_range": 0.02,
  "intermediate_size": 1856,
  "layer_norm_epsilon": 1e-05,
  "mamba_head_dim": 64,
  "mamba_hidden_act": "silu",
  "mamba_num_heads": 64,
  "mamba_proj_bias": false,
  "mamba_ssm_cache_dtype": "float32",
  "max_position_embeddings": 262144,
  "mlp_bias": false,
  "mlp_hidden_act": "relu2",
  "model_type": "nemotron_h",
  "moe_intermediate_size": 1856,
  "moe_shared_expert_intermediate_size": 3712,
  "n_group": 1,
  "n_groups": 8,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "norm_eps": 1e-05,
  "norm_topk_prob": true,
  "num_attention_heads": 32,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 52,
  "num_key_value_heads": 2,
  "num_logits_to_keep": 1,
  "pad_token_id": 0,
  "partial_rotary_factor": 1.0,
  "rescale_prenorm_residual": true,
  "residual_in_fp32": false,
  "rope_theta": 10000,
  "routed_scaling_factor": 2.5,
  "sliding_window": null,
  "ssm_state_size": 128,
  "tie_word_embeddings": false,
  "time_step_floor": 0.0001,
  "time_step_max": 0.1,
  "time_step_min": 0.001,
  "topk_group": 1,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.55.4",
  "use_bias": false,
  "use_cache": true,
  "use_conv_bias": true,
  "use_mamba_kernels": true,
  "vocab_size": 131072
}
```

### 3.1 Pattern decomposition (verified by hand-count of the 52-char string)

```
idx : 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
char:  M  E  M  E  M  *  E  M  E  M  E  M  *  E  M  E  M  E  M  *  E  M  E  M  E  M
idx : 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51
char:  *  E  M  E  M  E  M  *  E  M  E  M  E  M  E  M  *  E  M  E  M  E  M  E  M  E
```

- Attention (`*`) at indices **5, 12, 19, 26, 33, 42** → **6 layers** ✓
- Mamba (`M`) at 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50 → **23 layers** ✓
- MoE (`E`) → **23 layers** ✓
- No 'mlp' (`-`) layers in this model — `intermediate_size=1856` field is unused.

------------------------------------------------------------------------

## 4. Architecture by component

### 4.1 Embedding & LM head

`NemotronHModel.embeddings = nn.Embedding(131072, 2688)`, plain lookup
(NO sqrt-hidden scale — diff vs Gemma 4). `NemotronHForCausalLM.lm_head
= nn.Linear(2688, 131072, bias=False)`.

`config.tie_word_embeddings = False` (verbatim) but `_tied_weights_keys =
["lm_head.weight"]` in modeling. **Audit the safetensors checkpoint
keys** before assuming a separate `lm_head.weight` shard exists — Gemma 4
hit this exact gotcha. Likely the safetensors index has both
`model.embeddings.weight` and `lm_head.weight` because config says
untied; verify with `safe_open(...).keys()`.

**No logit softcap** in the forward (diff vs Gemma 4's
`final_logit_softcapping=30.0`). Logits go straight to argmax.

**Vocab=131072** is a clean power of 2 — vocab-sharded LM head over 4
chips gives 32768/chip (compare 27B's 248320/4=62080/chip).
**Production reuse**: P22 vocab-sharded LM head pattern from
`server_tp.py:1680-1687` ports directly.

**Comparison**: matches Llama-style embedding layer (simpler than Gemma
4 by far). NEW vs 27B/35B: smaller hidden (2688 vs 5120/2048), smaller
vocab (131072 vs 248320).

### 4.2 Attention layer (6 total)

`NemotronHAttention.forward` (modeling lines ~1262-1330):

```
q = q_proj(h).view(B, S, 32, 128).transpose(1,2)        # 32 Q heads
k = k_proj(h).view(B, S,  2, 128).transpose(1,2)        #  2 KV heads
v = v_proj(h).view(B, S,  2, 128).transpose(1,2)
# NO q_norm, k_norm, v_norm
# NO RoPE — query_states / key_states pass through unchanged
if past_key_value is not None:
    k, v = past_key_value.update(k, v, layer_idx)
k = repeat_kv(k, 16)   # 16:1 GQA group expansion
v = repeat_kv(v, 16)
attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
# Default SDPA scale = 1 / sqrt(head_dim) = 1 / sqrt(128)
out = o_proj(attn.transpose(1,2).reshape(B, S, 32*128))
```

**Critical**: NO positional encoding applied. Config has `rope_theta`
and `partial_rotary_factor=1.0` but modeling explicitly leaves
`position_embeddings` as `#TODO` and never computes cos/sin/RoPE on
q/k. **Positional information lives entirely in the Mamba2 SSM state.**
Cross-confirmed by grepping the modeling file for `apply_rotary_pos_emb`,
`rotary_emb`, `cos`, `sin`, `rope_theta`, `partial_rotary_factor` →
all absent from the forward path.

**Comparison**:
- vs 27B: 27B has Q-gate + Q/K-norm + full RoPE on partial dims.
  Nemotron has **none of this** — just plain GQA SDPA.
- vs 35B: 35B has Q-gate + Q/K-norm + RoPE + GatedDeltaNet recurrence.
  Nemotron's attention is much closer to **vanilla Llama-3 GQA**.
- vs Gemma 4: Gemma has q/k/v_norm + full RoPE (sliding) / p-RoPE
  (global) + scale=1.0. Nemotron has none of these; standard
  `1/sqrt(d_k)` SDPA scale (PyTorch default).

**Bringup recipe**: this is the **simplest attention block of any
model we've brought up**. Paged SDPA with `scale=1/sqrt(128)` and no
RoPE table; just `paged_update_cache` + `paged_scaled_dot_product_attention_decode`
on a `[NUM_BLOCKS, 2, BLOCK, 128]` KV cache per attention layer.

### 4.3 Mamba2 layer (23 total) — **NEW for this codebase**

`NemotronHMamba2Mixer` (modeling lines ~830-1000). Selective SSM block,
NOT a transformer attention or DeltaNet (35B's GatedDeltaNet is a DN /
linear-attn family op; Mamba2 is a structured-state-space family op —
related but distinct math). Key shapes (per layer):

```
in_proj: 2688 → (z_dim + x_dim + B_dim + C_dim + dt_dim)
         where x_dim = expand * hidden = 2 * 2688 = 5376    (but
         actually d_inner = mamba_num_heads * mamba_head_dim = 64 * 64 = 4096
         — config.expand is overridden by the explicit heads spec)
         z_dim = 4096 (gate)
         B_dim = n_groups * ssm_state = 8 * 128 = 1024
         C_dim = n_groups * ssm_state = 8 * 128 = 1024
         dt_dim = mamba_num_heads = 64
         total in_proj output ≈ 10304 (verify against safetensors)

conv1d:  causal Conv1d, kernel_size=4, groups=x_dim (depthwise)
         operates on (x, B, C) concatenated along feature axis
         conv state: [B, x_dim+B_dim+C_dim, 4]

SSM state: [B, num_heads=64, head_dim=64, ssm_state=128]   per layer
           dtype = float32 (config.mamba_ssm_cache_dtype)
           per-step compute reads state, applies discretized
           A/B/C/dt/D recursion, writes back

learned params per layer:
  - A_log: [64]  (init: torch.log(arange(1, 65)) — S4D init)
  - dt_bias: [64]
  - D: [64]
  - in_proj.weight, conv1d.weight, conv1d.bias, dt_proj.weight,
    out_proj.weight (all unbiased per `mamba_proj_bias=false`)
  - norm: MambaRMSNormGated (group_size = head_dim = 64, fused
    norm + gate via mamba-ssm Triton kernel)

out_proj: 4096 → 2688
```

Decode forward (single token, ignoring kernels):
```
x_in = h                                                # [B, 1, 2688]
zxbcdt = in_proj(x_in)                                  # [B, 1, ≈10304]
z, xbc, dt = split(zxbcdt, ...)                         # gate / state / time
xbc_conv = conv1d_step(xbc, conv_state)                 # SiLU after
x, B, C = split(xbc_conv, ...)
dt = softplus(dt + dt_bias).clamp(time_step_floor, ...) # discretization
# SSD recursion at chunk_size=1 for decode:
#   A = -exp(A_log)
#   ssm_state = exp(dt*A) * ssm_state + dt*B*x        per head/group
y = (C @ ssm_state) + D * x                              # per head
y = norm_gated(y, z)                                     # fused gate
out = out_proj(y)                                        # [B, 1, 2688]
```

This is **brand-new kernel territory on Blackhole**. tt-metal does not
ship a Mamba2 SSD kernel; we will need to author one. Prefix path is
even harder (chunked SSD scan).

**Comparison**: 35B's GatedDeltaNet (also a recurrent linear-attn-style
mixer) shares the *conceptual* pattern — fixed-size recurrent state
updated per step, no quadratic attention — and 35B's `qwen36_gdn_decode_owned`
kernel solves an analogous problem. But the math differs (DN: structured
matrix recurrence on K/V; Mamba2: SSD with selective time-step). We
will need a fresh kernel; the DN one is *not* directly portable.

### 4.4 MoE layer (23 total)

`NemotronHMOE` + `NemotronHTopkRouter` (modeling lines ~1176-1270).

**Routed experts**: 128 × `(gate_proj, up_proj, down_proj)` with
intermediate=1856, activation=`mlp_hidden_act="relu2"` (i.e.
`relu(x).pow(2)` — squared ReLU, **NEW** vs SiLU/GELU). Bias-free.

**Shared expert**: 1 × `(gate, up, down)` with intermediate=**3712**
(2× routed expert size). Same `relu2` activation. **Always active** for
every token. Added AFTER the routed-expert weighted sum:

```
h_routed = sum_{k in top6} weight_k * Expert_{idx_k}(h)
h_out    = h_routed + SharedExpert(h)
```

**Routing** (`NemotronHTopkRouter`):
1. `router_logits = h @ W_router`  → `[B, S, 128]`
2. `scores = router_logits.sigmoid()`  — **NOT softmax**
3. Group-restricted top-k:
   - Reshape scores into `[n_group=8, experts_per_group=16]`
   - For each group, sum top-2 expert scores → group score
   - Select top `topk_group=1` group(s) per token, mask others
4. Global top-k=6 from the remaining (16 candidates after masking)
5. If `norm_topk_prob=True`: `topk_weights /= (sum + 1e-20)`
6. Final: `topk_weights *= routed_scaling_factor (2.5)`

**Comparison**:
- vs 35B Qwen3-MoE: 35B has 256 routed + 1 shared, top-8, softmax
  router, no group restriction, no `routed_scaling_factor`,
  expert intermediate ≈ 1024.
- vs DeepSeek-V3 (canonical reference): identical *shape* of
  sigmoid + group routing + routed_scaling, but DeepSeek-V3 has 256
  experts (not 128) and a different topk_group config.
- The **sigmoid + group-restricted + scaling** trio is the
  DeepSeek-V3 routing recipe; Nemotron-3 Nano adopts it wholesale.

**Bringup recipe**: 35B's `moe_forward_ttnn_pattern_a_batched`
(`server_35b_ttnn.py:1225`) is the closest analog and uses `ttnn.matmul`
broadcasts. Need changes for:
- 128 experts instead of 256 (smaller per-chip slice E_LOCAL=32).
- **Sigmoid gating** instead of softmax (1-line swap).
- **Group-restricted topk** — fresh code; pre-mask scores before topk.
  ~30 LOC.
- **Squared ReLU activation** — `ttnn.relu` then `ttnn.square` (or use
  ttnn fused if available). Drops in where `SILU` lives in 35B.
- **Shared expert is 2× wider** than routed (3712 vs 1856) — different
  matmul shapes; needs its own program config.
- **routed_scaling_factor=2.5** applied to topk weights — scalar mul.

### 4.5 RMSNorm

`NemotronHRMSNorm.forward` (modeling lines 1042-1059):

```python
def forward(self, h):
    in_dtype = h.dtype
    h = h.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + self.variance_epsilon)
    return (self.weight.to(torch.float32) * h).to(in_dtype)
```

**Llama-style** — `y = (x/rms) * w`, NO `+1.0`, NO bias.
This matches Gemma 4 and DIFFERS from Qwen3.6 (27B/35B use `(1+w)`).

**Placement** (`NemotronHBlock.forward`, lines 1083-1121):
- **One pre-norm per layer** (before the mixer), then residual add.
  No post-norm.
- Differs from Gemma 4's **four** norms per layer (pre+post for both
  attn and FFN). Closer to 27B/35B's two-norm pattern, but with only
  ONE norm because the FFN/MoE/Attention/Mamba mixer is the only
  sub-block (no separate "attn then MLP" layering inside a block).

**Final norm**: `self.norm_f` on model output before lm_head — also
`NemotronHRMSNorm` (same Llama convention).

**Bringup recipe**: `ttnn.rms_norm` with `weight=w` (no `+1.0`).
Identical to the Gemma 4 `[[gemma4-v-norm]]` upload pattern, NOT the
Qwen3.6 `(1+w)` pattern.

### 4.6 Per-layer learned scalars

**NONE**. No equivalent of Gemma 4's `layer_scalar` per-layer multiply.
Mamba2 layers do have per-head learned scalars (`A_log`, `dt_bias`, `D`)
but those are internal to the SSM math, not at the layer-residual level.

### 4.7 Cache structure

`HybridMambaAttentionDynamicCache` — unified container for:
- KV cache for the 6 attention layers (`[K_per_layer, V_per_layer]`)
- Conv state for the 23 Mamba layers (`[B, conv_dim, kernel=4]`)
- SSM state for the 23 Mamba layers (`[B, 64, 64, 128]` float32)

The mamba_ssm_cache_dtype=float32 is significant — `ssm_state` must be
kept in fp32 on device. bf16 would drift just like 35B's H_t lever
([[35b-dn-h-state-drift-lever]]). Already proven on Blackhole that fp32
state in trace works for 35B's DN.

------------------------------------------------------------------------

## 5. Tokenizer & chat template

### 5.1 Special tokens (verified from `special_tokens_map.json` +
`tokenizer_config.json` partial dump)

- `<unk>` = 0
- `<s>` = 1 (BOS)
- `</s>` = 2
- `[INST]` = 3
- `[/INST]` = 4
- `[AVAILABLE_TOOLS]` = 5, `[/AVAILABLE_TOOLS]` = 6
- `[TOOL_RESULTS]` = 7, `[/TOOL_RESULTS]` = 8
- `[TOOL_CALLS]` = 9
- `<|im_start|>` = 10
- `<|im_end|>` = 11
- `<think>` = 12, `</think>` = 13
- `<tool_call>` = 14, `</tool_call>` = 15
- `<tool_response>` = 16, `</tool_response>` = 17
- `<SPECIAL_18>` … `<SPECIAL_565+>` reserved placeholders

**EOS configuration** (from `generation_config.json`):
```json
{
  "bos_token_id": 1,
  "eos_token_id": [2, 11],
  "pad_token_id": 0,
  "do_sample": true,
  "temperature": 1.0,
  "top_p": 1.0
}
```
**Two-EOS frozenset**: `{2, 11}` = `{</s>, <|im_end|>`. Our existing
`openai_endpoint.py` EOS-set logic must accept a list of EOS IDs (27B/
35B already pass a list). Verify.

`special_tokens_map.json` declares **bos=`<s>`, eos=`<|im_end|>`** (NOT
`</s>`); generation_config exits on EITHER. The "primary" chat-EOS is
`<|im_end|>` (id 11); `</s>` (id 2) is the legacy fallback.

Tokenizer class: **not yet captured** (the `tokenizer_config.json`
WebFetch hit the partial-content truncation issue and the tail with
`tokenizer_class`/`model_max_length` wasn't surfaced). Likely
`PreTrainedTokenizerFast` per `tokenizer.json` presence. **needs
verification** by reading the file end on qb1.

### 5.2 Chat template (`chat_template.jinja`, verbatim retrieved)

Header / per-message format is **ChatML** with `<|im_start|>{role}\n…<|im_end|>\n`
— identical to Qwen3 / Qwen3.6. **No system-prompt wrap is forced**
unless caller provides one.

**Active prompt suffix** (the `add_generation_prompt=True` branch):

```jinja
{%- if add_generation_prompt %}
    {%- if enable_thinking %}
        {{- '<|im_start|>assistant\n<think>\n' }}     ← default
    {%- else %}
        {{- '<|im_start|>assistant\n<think></think>' }}
    {%- endif %}
{%- endif %}
```

**Key implications for our `_active_prompt_suffix` stripper in
`openai_endpoint.py`**:

1. With `enable_thinking=True` (default): suffix is
   `<|im_start|>assistant\n<think>\n` (4 tokens). Active prompt ends
   *inside* the `<think>` block; the model is expected to generate
   reasoning, close with `</think>`, then emit the final answer.
2. With `enable_thinking=False`: suffix is
   `<|im_start|>assistant\n<think></think>` (also 4 tokens, slightly
   different layout). `<think></think>` is emitted **closed-empty** as
   a marker that thinking is disabled.

**This is EXACTLY the same hazard Qwen3.6 had** with the active-prompt
asymmetry: the prior assistant turn re-renders with closing structure
that doesn't byte-equal what was generated. Our existing
`preserve_thinking=True` + trailing-strip patch
(`[[qwen36-preserve-thinking]]`) covers the case where current-turn
suffix differs from history rendering. **Likely just works** but
**needs verification** via a `chat_template_invariant.py`-style test
once tokenizer is loaded.

The template also has a `truncate_history_thinking` knob (default True)
that strips `<think>...</think>` from PRIOR assistant turns in
multi-turn history. This is *additional* asymmetry vs single-turn
generation. Test plan: replay a 3-turn conversation through the
template, verify our prefix-cache exact-match doesn't tank.

### 5.3 Tool-call format

Different from Qwen3.6 — uses XML-style `<tool_call>` / `<function=…>`
/ `<parameter=…>` blocks per the template. This is the "Nemotron-3
nano agent format". Our `openai_endpoint.py` does not currently parse
tool-calls so this is a v2/v3 concern, not v0 bringup.

------------------------------------------------------------------------

## 6. Reuse map (the critical output)

| Existing component | Reuse for Nemotron-3 Nano | Confidence |
|---|---|---|
| `experiments/cb/_runner.py` `project_root()` / `log()` / `bootstrap_*_cb()` | Fork → `bootstrap_nemotron3_cb()`; ~30 LOC | HIGH |
| `experiments/utils/ttnn_introspect.py` | Use as-is for kernel signature lookups (Mamba2 ops, paged SDPA) | HIGH |
| `experiments/utils/hf_reference_35b.py` | Fork → `hf_reference_nemotron3_nano.py`; **must add Mamba2 layer hooks** (in_proj, conv1d, ssm_state, out_proj per-step), drop DN-specific hooks, keep attn + MoE hooks. **HF AutoModel will likely OOM on CPU bf16** at 30B; consider using `device_map="auto"` to CPU + 1 GPU or do per-layer hooks streaming | HIGH (oracle) / MEDIUM (memory risk) |
| `experiments/utils/cosine_ladder_*.py` | Fork shape; swap n_layers 40→52, oracle path | HIGH |
| `experiments/cb/dev/cb35_dev_harness.py` → `cb_n3_dev_harness.py` | Fork; same shape; expect long bootstrap (32 GB BF16 / 4 chips = 8 GB/chip upload) | HIGH |
| `experiments/cb/isolate/paged_sdpa.py` | Reuse as-is for the 6 attention layers (simpler than 35B/Gemma 4: no Q-gate, no q/k_norm, no RoPE) | HIGH |
| `experiments/cb/isolate/paged_update_cache.py` | Reuse as-is — NKV_PER_CHIP=2/4 chips = 1 → contracts match cleanly (no Gemma 4 split-SDPA issue: NUM_KV_HEADS=2, NCHIPS=4 means 2 chips get 1 KV head each and 2 chips replicate; or sub-shard) — **VERIFY** during bringup how 2 KV heads shard over 4 chips. Option: replicate KV across chip-pairs. See §7 risks. | MEDIUM |
| `experiments/serve/server_35b_ttnn.py` `bootstrap()` skeleton | Fork → `server_nemotron3_nano_ttnn.py`; swap MODEL_ID, NUM_LAYERS, HIDDEN, per-layer dispatch on `state.layer_types[L]` (use 4 types: mamba/attention/moe — no 'mlp' in this model). Keep the MoE Pattern A batched matmul code with sigmoid + group-mask + relu² swaps. | HIGH (skeleton) / MEDIUM (per-layer dispatch) |
| `experiments/serve/server_tp.py:1680-1687` P22 vocab-sharded LM head + on-device argmax | Reuse verbatim with VOCAB=131072 | HIGH |
| `experiments/serve/cb_api.py` `BACKENDS` dict + `cb_scheduler.py` `_BACKEND_MODULES` | Add `"nemotron3_nano": ("server_nemotron3_nano_ttnn", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")` entries; preserve `[[cb-backend-dispatch-holes]]` audit discipline | HIGH |
| `experiments/serve/openai_endpoint.py` `_messages_to_prompt` + `_active_prompt_suffix` | Use as-is for ChatML; verify against Nemotron's `<think>\n` vs `<think></think>` active-suffix variants | HIGH (template) / MEDIUM (think handling needs test) |
| `experiments/utils/needle_haystack_*.py` | Reuse for v0.3.3 long-context validation at 8K-32K — Mamba2's selective state should handle long context well | HIGH |
| 35B's `moe_forward_ttnn_pattern_a_batched` (`server_35b_ttnn.py:1225`) | Fork: sigmoid in place of softmax, group-restricted top-k pre-mask, routed_scaling_factor mul, expert FFN with relu² activation, 2×-wide shared expert. ~150 LOC delta | HIGH (pattern) / MEDIUM (delta) |
| 35B's owned-GDN kernel (`qwen36_gdn_decode_owned`) | **DO NOT REUSE** — different math (DN matrix recurrence vs Mamba2 SSD). The plumbing pattern (recurrent state in/out, kernel-side fp32 acc) is transferable to a new `nemotron3_mamba2_decode_owned` kernel; the per-step math is NOT. | LOW (math) / MEDIUM (plumbing) |
| `experiments/utils/test_fused_*.py` | Use as-is for activation probes (relu², MambaRMSNormGated) | HIGH |
| `scripts/run_harness_tmux.sh` | Use as-is, register new harness name `nemotron3` | HIGH |
| `scripts/deploy.sh` / `serve_cb.sh` | Use as-is (deploy auto-globs `experiments/serve/*.py`) | HIGH |
| `experiments/cb/validate/cb_validate_*.py` | Fork pattern for 3a/3b/3c gates | HIGH |

### 6.1 Three biggest reuse wins (copy-able as-is)

1. **CB framework (`cb_engine`/`cb_scheduler`/`cb_api`/`openai_endpoint`/
   `live_slot_store`)** — entire prod CB chat path ports verbatim;
   register a new backend in two dicts.
2. **Paged SDPA + vocab-sharded LM head + on-device argmax** —
   Nemotron's attention has FEWER quirks than 27B/35B/Gemma 4 (no
   Q-gate, no q/k_norm, no RoPE), so paged SDPA is literally a
   `1/sqrt(128)` scale call with no preprocessing on q/k.
3. **HF oracle + cosine ladder + per-layer drift + needle haystack +
   dev harness** — the entire validation infra is model-agnostic;
   only constants change.

### 6.2 Three biggest NEW things (need fresh code)

1. **Mamba2 SSD decode kernel** — `tt-metal` does not ship one. We need
   to author a Blackhole kernel that performs the per-token SSD
   recursion (or a manual ttnn composite if perf is tolerable). This is
   THE blocker for the whole bringup. Reference math in the original
   Nemotron-H paper (arXiv 2504.03624) + the mamba-ssm GitHub repo
   (state-spaces/mamba). 23/52 layers go through this path.
2. **Mamba2 conv1d (causal depthwise, kernel=4) decode step** —
   maintain a `[B, conv_dim, 4]` conv state, apply causal Conv1d-step
   semantics per token. ttnn has Conv1d but not the per-step caching
   form. Fresh code.
3. **DeepSeek-V3-style MoE router** (sigmoid + group-restricted top-k
   + routed_scaling) — 35B's top-k softmax router is the wrong starting
   point. Fork and replace the gating math. ~150-200 LOC.

------------------------------------------------------------------------

## 7. Risks & unknowns

### 7.1 BLOCKER: Mamba2 SSD kernel does not exist on Blackhole

23 of 52 layers go through Mamba2. We have no precedent in this repo
(35B's GatedDeltaNet is a sibling family but the math is different).
Two paths:
1. **Manual ttnn composite**: replicate the SSD recursion with `ttnn.mul`
   / `ttnn.add` / `ttnn.exp` over the per-head loop. Will be SLOW
   (probably > 200 ms/tok for 23 layers; per-token recursion has poor
   compute intensity). Acceptable for v0 correctness.
2. **Owned kernel**: author `nemotron3_mamba2_decode_owned` from scratch
   following the 35B `qwen36_gdn_decode_owned` build pattern (G0..G4
   staging per [[build-kernels-from-scratch]]). 1-3 week effort.

**Plan**: v0 correctness via Path 1; v3 perf via Path 2. Same v0..v4
ladder as 35B with Mamba2-decode swapped in.

### 7.2 KV cache sharding with 2 KV heads on 4 chips

`num_key_value_heads=2`, mesh is (1,4). Cannot cleanly shard 2 KV heads
across 4 chips. Options:
- **Replicate**: each chip-pair gets one KV head; both pairs replicate
  across each other. Wastes 2× memory but trivial. 6 attn × 256K
  context = budget: `262144 * 2 * 128 * 2(K+V) * 2(bf16) * 6 = 1.5 GB`
  total → ~750 MB/chip if replicated 2×. **Fits**.
- **Sub-shard a head**: split the head_dim across chips. Adds
  collective traffic per SDPA call. Avoid for v0.
- **Use existing 35B pattern**: 35B has NKV=1 over 4 chips by some
  contract — re-read its KV sharding. Likely it replicates K across
  chips and shards V (or vice versa).

**Decision**: replicate for v0; revisit at v3 perf pass.

### 7.3 SSM state in fp32 inside trace

`mamba_ssm_cache_dtype="float32"` per config + research-confirmed need
([[35b-dn-h-state-drift-lever]] showed bf16 recurrent state drifts).
The 35B fp32 H_t experiment ([commits `35ea58f` + `7c3ede6` +
`1c650b7`]) had a **30+ min trace capture hang** when fp32 was
introduced into the trace. **Risk**: same Blackhole bug bites us with
Mamba2's fp32 SSM state inside trace.

**Mitigation plan**: validate the Mamba2 fp32 state path in EAGER first
(v0); only attempt trace capture (v0.4) after eager passes. If trace
hangs, fall back to bf16 SSM state + accept the drift (we can measure
it; 35B fp32 H_t was a CORRECTNESS lever, not a hard requirement).

### 7.4 No RoPE = attention has no built-in positional signal

Theoretically problematic — if we set up the attention layers
incorrectly with a stale `position_ids` argument or some leftover RoPE
table, the model will silently degrade. **Triple-check** during
bringup that q/k go from `q_proj`/`k_proj` directly into SDPA with NO
intermediate transform. **No RoPE table allocation at all**.

### 7.5 Squared ReLU (`relu²`) activation in expert FFN

`mlp_hidden_act="relu2"` — `relu(x).pow(2)`. ttnn has `relu` and
`square` and `pow`; no fused `relu_squared` UnaryOp known. Need to
verify: `ttnn.square(ttnn.relu(x))` is the right composition; can
also try fused `UnaryOpType.RELU_SQUARED` if it exists (introspect
on qb1).

Per Step 0.2 of Gemma 4 bringup, GELU fused gave WORSE precision
than a two-op decomposition. Likely same lesson applies: prefer
two-op for correctness; revisit fusion at perf pass. Plan: probe
both with `experiments/utils/test_fused_*.py`-style isolation.

### 7.6 30B model load + weight-upload time

13 safetensors shards × ~5 GB each. With 35B's ~14-min bootstrap as
reference and Gemma 4's ~80s reference, Nemotron sits in between:
expect **~10-12 min bootstrap**. Dev harness mandatory from day 1
([[use-dev-harness-for-iteration]]).

### 7.7 Tokenizer end-of-file not yet captured

`tokenizer_class`, `model_max_length`, `clean_up_tokenization_spaces`,
and inline `chat_template` fallback in `tokenizer_config.json` were
not surfaced by the WebFetch (file truncation). **Needs on-qb1 verify**:
```
ssh qb1 'tail -50 ~/.cache/huggingface/.../tokenizer_config.json'
```
to confirm `tokenizer_class` (expected `PreTrainedTokenizerFast`) and
that no inline `chat_template` overrides the `.jinja` we already have.

### 7.8 Custom code requires `trust_remote_code=True`

HF AutoModel/AutoTokenizer for this checkpoint **requires
`trust_remote_code=True`** (the auto_map points to
`modeling_nemotron_h.py` shipped IN the repo). For our HF-oracle
script: include the flag. For our TT bringup: we don't load HF code,
just read safetensors directly via `safe_open`, so no flag needed there.

### 7.9 Param count discrepancy (30B vs 31.6B)

HF model card says "30B total, 3.5B active". OpenRouter/blog says
"31.6B total / 3.2B-3.6B active". 13 × ~5 GB shards ≈ 63 GB BF16
=> 31.5B params, confirming the 31.6B number. **For memory planning,
use 32 GB BF16 weights total** (8 GB/chip on a 4-chip mesh).

### 7.10 Long context

`max_position_embeddings=262144`. We have NEVER shipped a model at
> 8K context (35B caps at 8192). With Mamba2's constant-state-per-token,
Nemotron has a **structural advantage** for long context — but our
infrastructure (`MAX_POS=8192` in `server_tp.py:53`) caps at 8K. v0
ships at 8K; long-context push is a separate workstream. **Bonus**:
because attention is only 6/52 layers and has no RoPE, the long-context
drift cliff that hit us at 35B (bf16 chain drift over many attn layers)
is *less* of a risk here — 6 attention layers can't accumulate the
same noise budget.

### 7.11 No license-gating, but commercial license is Nvidia-specific

NVIDIA Nemotron Open Model License — not Apache/MIT. For research use
this is fine; no download blocker.

------------------------------------------------------------------------

## 8. Local files for reuse-map grounding (read before planning)

For first-time planners, these three files together give the entire
ladder + reuse philosophy:

- [`research/35b_cb_bringup_plan.md`](35b_cb_bringup_plan.md) —
  closest precedent (MoE + recurrent mixer + (1,4) mesh + CB v0..v4
  ladder + dev harness pattern). The MoE Pattern A and CB scheduler
  pieces port DIRECTLY.

- [`research/gemma4_12b_bringup_plan.md`](gemma4_12b_bringup_plan.md) —
  hybrid-attention precedent (sliding + global dispatch). The
  per-layer-type dispatch pattern, the bootstrap skeleton, the v0.1.x
  sub-staging table, and the kernel-source-first discipline all carry
  over. The Llama-style RMSNorm (`w`, not `(1+w)`) is the same here.

- [`research/model_bringup_recipe.md`](model_bringup_recipe.md) — the
  meta-recipe that took Gemma 4 from oracle to HTTP chat in 36 hours.
  Read first. The 12-bug catalog (§1) is mostly applicable; Mamba2-
  specific bugs not yet in the catalog (will populate during bringup).

Other useful references:
- `archive/superseded_research_2026-06-04/27b_continuous_batching_plan.md`
  — CB1/2/3/4 origin design (archived 2026-06-04).
- `archive/superseded_research_2026-06-04/35b_a3b_correctness_plan.md`
  — the diagnosis recipe for per-layer drift (archived 2026-06-04;
  will be needed if a v0.x cosine gate fails).
- `research/multi_chip_optimizations_menu.md` (+ `v2`) — perf-pass
  candidate list; useful AFTER v0..v0.4 ships.

------------------------------------------------------------------------

## 9. Suggested bringup ladder (sketch, refine in a separate plan)

| Stage | Adds | Gate |
|---|---|---|
| v0.0 | HF oracle (`hf_reference_nemotron3_nano.py`) — needs Mamba2 layer hooks + custom-code import | `prompt_ids.npy`, `hidden_states.npy[53, S, 2688]` exist |
| v0.1.0 | Bootstrap on (1,4): mesh, weights upload, embed lookup | bootstrap < 15 min; embed lookup cos ≥ 0.999 vs HF |
| v0.1.1 | L0 forward — Mamba2 mixer (manual ttnn composite) with pre-norm + residual | cos ≥ 0.999 on L0 output vs HF |
| v0.1.2 | L1 (MoE) — sigmoid router + group top-6 + relu² experts + shared expert + scale | cos ≥ 0.999 on L1 output vs HF |
| v0.1.3 | L5 (Attention) — paged SDPA, NO RoPE, GQA repeat_kv | cos ≥ 0.999 on L5 output vs HF |
| v0.2 | All 52 layers + final_norm + lm_head + argmax | argmax matches HF at pos 0 |
| v0.3 | Multi-step decode: KV cache + Mamba2 state (conv + ssm) in fp32, eager | TT tokens 0..5 match HF token-for-token |
| v0.3.3 | Long-context smoke at L=8192 (`needle_haystack_nemotron3.py`) | password retrieved ≥ 75% at L=8192 |
| v0.4 | Trace capture (CHECK fp32-in-trace risk per §7.3) | 100 traced == 100 eager tokens |
| v1 | CB B=4 (`server_nemotron3_nano_cb.py`) | 3a/3b/3c PASS |
| v2 | HTTP wire-up | curl /v1/chat/completions returns sensible English |
| v3 | Owned Mamba2 SSD decode kernel (perf pass) | step_ms reduced ≥ 30% over v0.4 baseline |

Estimate: **3-4 weeks** to v2 if Mamba2 manual composite is acceptable;
**6-8 weeks** if an owned kernel is required upfront.

------------------------------------------------------------------------

## 10. Memory entries to write during bringup (predicted)

- `feedback_nemotron3_no_rope_silent_drift` — bug where stray RoPE
  application degrades cosine without crashing (predicted)
- `feedback_mamba2_ssm_fp32_state_in_trace` — Blackhole trace
  behavior with fp32 recurrent state (predicted, mirror of 35B
  fp32-H experience)
- `feedback_nemotron3_moe_sigmoid_group_router` — diff from softmax
  routing
- `feedback_nemotron3_relu_squared` — activation fused vs split
  decision
- `feedback_nemotron3_thinking_template_double_branch` — `<think>\n`
  vs `<think></think>` active-prompt variants vs our PC matcher

------------------------------------------------------------------------

## Appendix A: minimum-viable HF-reference invocation

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="bfloat16",
    device_map="cpu",          # 30B bf16 fits in ~62 GB CPU RAM; check qb1 host RAM
)
```

`device_map="cpu"` is likely required for the oracle (qb1 host RAM check
needed). If CPU RAM is insufficient, stream layers from disk via
`accelerate.load_checkpoint_and_dispatch`.

------------------------------------------------------------------------

## Sources

- HF model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- HF raw config: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/config.json
- HF raw modeling: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/modeling_nemotron_h.py
- HF raw configuration: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/configuration_nemotron_h.py
- HF raw chat template: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/chat_template.jinja
- HF raw generation_config: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/generation_config.json
- HF raw special_tokens_map: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/raw/main/special_tokens_map.json
- Transformers docs: https://huggingface.co/docs/transformers/model_doc/nemotron_h
- Nemotron-H paper (arch lineage): https://arxiv.org/abs/2504.03624
- NVIDIA research page: https://research.nvidia.com/labs/nemotron/Nemotron-3/
- NVIDIA build catalog (model card): https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard
- OpenRouter listing (param counts cross-ref): https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b:free
