# Qwen3.6-35B-A3B — Architecture Notes (text-only path)

Source: `Qwen/Qwen3.6-35B-A3B/config.json` fetched 2026-05-12.
We skip the vision tower entirely. This doc is the contract for Phase A
isolation work and Phase B integration.

## At-a-glance

```
hidden_size:         2048
num_hidden_layers:   40   (10 patterns × 4 layers)
vocab_size:          248320 (padded; tie_word_embeddings = false)
max_position:        262144
rms_norm_eps:        1e-6
mtp_num_hidden:      1     (multi-token-prediction head — we ignore for now)
```

### Layer interleaving — EXACT from config.layer_types

```
Index:  0  1  2  3  4  5  6  7  8  9 ... 36 37 38 39
Type:   L  L  L  F  L  L  L  F  L  L     L  L  L  F

L = linear_attention (Gated DeltaNet)
F = full_attention   (Gated Attention)
full_attention_interval = 4 → every 4th layer
```

30 DeltaNet layers + 10 full-attention layers. Pattern is rigid, repeats 10×.

---

## Block 1 — Linear-attention layer (Gated DeltaNet)

This is the new thing. We have no prior port to crib from.

```
linear_conv_kernel_dim:    4            (Mamba-style local 1D conv before linear-attn)
linear_num_key_heads:      16           (== num_attention_heads)
linear_num_value_heads:    32           (2× more V heads than K)
linear_key_head_dim:       128
linear_value_head_dim:     128
mamba_ssm_dtype:           float32      (state must stay fp32 for stability)
```

**Dimensions:**
- Q dim (= K dim): 16 × 128 = 2048   (matches hidden)
- V dim: 32 × 128 = 4096              (2× hidden — outputs more channels)
- Output back-projects 4096 → 2048

**Sub-ops (decomposition for ttnn):**
1. RMSNorm(x) → x_norm
2. Linear: Q, K, V projections (each from 2048; Q,K to 2048; V to 4096) — 3 matmuls
3. **1D causal conv (kernel=4)** on Q, K, V along sequence dim (independent per channel)
4. Activation (likely silu) on Q, K, V (typical Mamba pattern)
5. **Delta-rule update** of recurrent state H:
   - `H_t = H_{t-1} * decay(K_t) + outer(V_t, K_t)` (rough form; exact gating from paper)
   - State H has shape [batch, num_value_heads, value_head_dim, key_head_dim] kept in fp32
6. **Linear attention readout**: `Y_t = H_t @ Q_t`
7. Output projection 4096 → 2048
8. Add residual

**Decode-time cost (per token):**
- Conv updates: 4 prior K/V tokens (cheap)
- State H is ALREADY accumulated — no quadratic dependence on context length
- State update is O(D_k × D_v × n_v_heads) = 128 × 128 × 32 ≈ 524K mul-adds
- State readout is O(D_k × D_v × n_v_heads) = same
- Projections dominate cost

**This is the key benefit:** decode cost is O(D²) per token, independent of how long the context is.

---

## Block 2 — Full-attention layer (Gated Attention)

This is closer to what we've ported, with some new wrinkles.

```
num_attention_heads:        16    (Q heads)
num_key_value_heads:        2     (KV heads — aggressive GQA, ratio 8:1)
head_dim:                   256
partial_rotary_factor:      0.25  → only 64 dims of 256 get RoPE
attn_output_gate:           true  → sigmoid gate on attention output
rope_theta:                 10,000,000
mrope_interleaved:          true, section [11, 11, 10]  (multi-axis RoPE)
```

**Dimensions:**
- Q dim: 16 × 256 = 4096
- KV dim: 2 × 256 = 512  (very small — heavy compression)
- Output 4096 → 2048

**Sub-ops:**
1. RMSNorm
2. Linear: Q (2048→4096), K (2048→512), V (2048→512)
3. Apply RoPE — but only to the first 64 dims of head_dim 256, rest pass through
4. SDPA over (Q, K, V, KV-cache)
5. **Output gate**: `out = SDPA(Q,K,V) * sigmoid(gate_proj(x))` — a sigmoid mask on output
6. Output projection 4096 → 2048
7. Residual

**RoPE specifics:**
- M-RoPE (multimodal): section [11, 11, 10] for (text, height, width)
- For text-only inference, we use a 1D RoPE on the first 64 dims
- The remaining 192 dims of head_dim 256 are passed through unrotated
- This is partial rotary — common in newer models

**Decode-time cost:** standard SDPA over KV cache. KV is 512-dim per token (2 heads × 256), so cache is cheap per token even at long context.

---

## Block 3 — MoE layer

Both layer types are followed by an MoE block.

```
num_experts:                   256
num_experts_per_tok:           8     (routed)
shared_expert_intermediate:    512   (1 shared expert, always active)
moe_intermediate_size:         512   (per routed expert)
router_aux_loss_coef:          0.001  (training-only, ignored at inference)
```

**Per-token cost:**
- Router: linear 2048 → 256, softmax, top-k=8
- 8 routed experts × (gate 2048→512, up 2048→512, down 512→2048) = 8 × 3 matmuls
- 1 shared expert × (gate 2048→512, up 2048→512, down 512→2048) = 3 matmuls
- Total: **27 matmuls per MoE block**
- Active params per token: 8×3×(2048×512) + 1×3×(2048×512) = 9×3×1048576 ≈ 28M params per layer
- Over 40 layers: ~1.1B active per token from MoE — matches the ~3B active total (rest is attention + embeddings)

This is double the routing of Qwen1.5-MoE (4 routed there, 8 here) and 4× the experts (60 → 256).

---

## Tokens / vocab

```
vocab_size:        248320 (padded)
image_token_id:    248056  ← skip (no vision)
video_token_id:    248057  ← skip
bos / eos / pad:   248044 / 248044 / null
tie_word_embeddings: false  → input and output embeddings are SEPARATE 2048-d × 248320 matrices
```

Two distinct embedding tables of 2048 × 248320 = 508M params each. ~1B in embeddings alone.

---

## Total parameter budget (back-of-envelope)

| Component | Per layer | × layers | Total |
|---|---|---|---|
| DeltaNet (Q,K,V proj + conv + output) | ~12.5M | × 30 | ~376M |
| Full-attention (Q,K,V,O,gate) | ~14M | × 10 | ~141M |
| MoE block | 256 × (3 × 2048×512) = ~805M | × 40 | ~32.2B |
| RMSNorm × 2 per layer | trivial | × 40 | ~0.1M |
| Embeddings (in + out) | — | — | ~1.0B |
| **TOTAL** | | | **~33.7B** |

Reported 35B includes the MTP head + bias terms.

**Active per token** (sparse — only the 9 of 256 experts that fire):
- DeltaNet/Attn: 376M + 141M = ~517M (all layers fire every token)
- MoE: 9 of 256 experts × 805M / 256 ≈ ~28M/layer × 40 ≈ 1.1B
- Embeddings: ~1 row read per token (negligible compute)
- **Active total: ~1.6B compute-active** — close to reported 3B (the rest are scratch + routing).

---

## Hardware ceiling — Blackhole P150 baseline

To know if our kernels are good, we need targets.

```
Cores:                  110 (11×10 tensix grid)
DRAM bandwidth:         ~450 GB/s
Compute peak (HiFi4):   TODO — measure empirically (see Phase A2)
L1 per core:            1.5 MB
DRAM capacity:          ~32 GB
```

**Memory bandwidth bounds per decode step:**

| Surface | Bytes / token (bf8) | Bandwidth-limited time |
|---|---|---|
| DeltaNet QKVO weights | ~12.5M | 12.5 MB / 450 GB/s = **28 µs** |
| Full-attn QKVOG weights | ~14M | 31 µs |
| MoE: 9 active experts | 9 × 3 × 1MB = 27M | 60 µs |
| Per layer total (DeltaNet+MoE) | ~40M | **89 µs** |
| Whole model 1 token | 40 layers × ~40M = ~1.6 GB | **3.6 ms** |

**Theoretical token ceiling: ~280 tok/s** if perfectly memory-bound. With dispatch overhead, real targets:
- Phase B (eager, dispatch-bound): 5-15 tok/s expected
- Phase B (traced, dispatch eliminated): 50-100 tok/s expected — closer to bandwidth limit
- Phase C (4-chip TP, 4× bandwidth): 150-250 tok/s

These are my honest expectations. We'll know more after measuring.

**Compute-bound check** for the biggest matmul (router-MoE down 512→2048):
- 512 × 2048 = 1M ops per token
- At HiFi4 peak (~1 TFLOPS/core × 110 cores = ~110 TFLOPS), 1M ops = 0.01 µs
- We're FAR memory-bound at batch=1.

This is why MoE is hard at batch=1 — every expert read costs DRAM bandwidth and we use very little of the compute.

---

## What "isolation testing" means for each kernel

Per user's instinct: each component validated alone before integration.

| Kernel | Isolation test | Hardware ceiling target |
|---|---|---|
| **Gated DeltaNet** | Random Q/K/V input → cosine ≥ 0.99 vs numpy. Time per token. | Memory: ~28 µs. Target ≤ 100 µs eager. |
| **Gated Attention** | Random Q/K/V/cache → cosine vs numpy. Time per token. | Memory: ~31 µs. Target ≤ 80 µs eager. |
| **MoE block** | Random routing + experts → cosine. Time per token. | Memory: ~60 µs. Target ≤ 200 µs eager. |
| **RMS norm** | Random → cosine. Time. | Bandwidth: ~5 µs. We know this from prior work. |
| **RoPE (partial 64-dim)** | Random Q → cosine. Time. | Native ttnn RoPE: ~50 µs prior. |

Each kernel gets a `experiments/82_<name>.py` script. Each prints:
- Cosine vs numpy fp32 reference
- Time per call (eager + traced)
- % of memory-bandwidth ceiling
- Bottleneck identification (memory / compute / dispatch)

---

## Open questions to resolve in Phase A

1. **Delta-rule exact form.** Several Mamba-family models use different decay gates. Need to read Qwen3 paper section on Gated DeltaNet or peek at HF modeling code.
2. **M-RoPE for text-only.** Can we just use 1D RoPE on the first 64 dims, or does Qwen3.6 require the multi-axis form even for pure text? Probably the former.
3. **Output gate exact form.** `attn_output_gate: true` — is gate computed from Q? From hidden? Likely from hidden via a separate linear.
4. **Routing temperature.** 8 of 256 is sharp. Likely just `softmax(router_logits).topk(8)`.

I'll answer these in Phase A2 by reading the HuggingFace `modeling_qwen3_5_moe.py` source. Pasting key snippets back here.

---

## Files Phase A produces

```
research/qwen36_arch_notes.md         ← THIS FILE (A1)
research/qwen36_modeling_excerpts.md  ← A2: HF modeling code key snippets
experiments/82_gated_deltanet.py      ← A3: isolated DeltaNet kernel + perf
experiments/83_gated_attention.py     ← A4: isolated Gated Attention + perf
experiments/84_moe_block.py           ← A5 (added): isolated MoE block + perf
experiments/81_moe_regression_micro.py ← A0: diagnose Qwen1.5-MoE slowdown
```
