# Gemma 4 Architecture Research

Research date: 2026-04-22
Sources: HuggingFace model cards and config.json files for google/gemma-4-{E2B,E4B,26B-A4B,31B}{,-it}

## 1. Model Family Overview

Gemma 4 is Google's latest open-weight model family. All variants are natively multimodal
(text + image; smaller models also support audio). Both base and instruction-tuned (-it)
variants are released under Apache 2.0.

| Model | Total Params | Active Params | Layers | Hidden | Context | MoE | Architecture Class |
|-------|-------------|--------------|--------|--------|---------|-----|-------------------|
| E2B   | 5.1B        | ~2.3B (eff.) | 35     | 1536   | 128K    | No  | Gemma4ForConditionalGeneration |
| E4B   | 8.0B        | ~4.5B (eff.) | 42     | 2560   | 128K    | No  | Gemma4ForConditionalGeneration |
| 26B-A4B | 25.2B     | 3.8B         | 30     | 2816   | 256K    | Yes | Gemma4ForConditionalGeneration |
| 31B   | 30.7B       | 30.7B        | 60     | 5376   | 256K    | No  | Gemma4ForConditionalGeneration |

## 2. Detailed Architecture Parameters

### Dense Models (E2B, E4B, 31B)

| Parameter | E2B | E4B | 31B |
|-----------|-----|-----|-----|
| hidden_size | 1536 | 2560 | 5376 |
| num_hidden_layers | 35 | 42 | 60 |
| num_attention_heads | 8 | 8 | 32 |
| num_key_value_heads | 1 | 2 | 16 |
| num_global_key_value_heads | N/A | N/A | 4 |
| head_dim (sliding) | 256 | 256 | 256 |
| global_head_dim (full) | 512 | 512 | 512 |
| intermediate_size (MLP) | 6144 | 10240 | 21504 |
| vocab_size | 262144 | 262144 | 262144 |
| sliding_window | 512 | 512 | 1024 |
| attention_k_eq_v | false | false | true |
| num_kv_shared_layers | 20 | 18 | 0 |
| hidden_size_per_layer_input | 256 | 256 | 0 |
| use_double_wide_mlp | true | false | false |
| tie_word_embeddings | true | true | true |

### MoE Model (26B-A4B)

| Parameter | Value |
|-----------|-------|
| hidden_size | 2816 |
| num_hidden_layers | 30 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| num_global_key_value_heads | 2 |
| head_dim (sliding) | 256 |
| global_head_dim (full) | 512 |
| intermediate_size (dense MLP) | 2112 |
| num_experts | 128 |
| top_k_experts | 8 |
| moe_intermediate_size | 704 |
| vocab_size | 262144 |
| sliding_window | 1024 |
| attention_k_eq_v | true |
| num_kv_shared_layers | 0 |
| hidden_size_per_layer_input | 0 |

The MoE variant uses 128 experts with top-8 routing. Each expert has a small FFN with
intermediate_size=704. The dense MLP intermediate_size is 2112. No shared expert is
mentioned in the config (unlike Qwen's shared_expert_gate pattern), though the model
card mentions "1 shared expert" -- this may be implemented as the dense MLP path.

## 3. Memory Footprint and Blackhole Fit

Weight sizes in BF16 (from HuggingFace safetensors metadata):

| Model | BF16 Size | BF8 Estimate | Fits 32GB BH at BF8? |
|-------|-----------|-------------|----------------------|
| E2B   | 5.1 GB    | ~2.6 GB     | Yes (easily)         |
| E4B   | 8.0 GB    | ~4.0 GB     | Yes (easily)         |
| 26B-A4B | 26.5 GB | ~13.3 GB    | Yes (but tight with KV cache) |
| 31B   | 32.7 GB   | ~16.4 GB    | Borderline -- weights alone fit, but KV cache for 256K context won't |

For Blackhole with 32 GB DRAM:
- **E2B and E4B**: Fit trivially. Best candidates for initial bring-up.
- **26B-A4B**: Fits at BF8 (~13.3 GB) but only 128 experts x 704 intermediate = expert
  weights are small. The 8 active experts per token means forward pass touches only
  ~3.8B params. Memory for all 128 expert weight sets is the bottleneck.
- **31B**: At BF8 ~16.4 GB for weights. Leaves ~15.6 GB for KV cache and activations.
  Feasible for short contexts but cannot support full 256K context.

### Recommended target: E4B (8 GB BF16, ~4 GB BF8)

Similar parameter budget to Qwen 0.5B-class models we already run but much more capable.
42 layers is manageable. GQA with 8 Q heads and 2 KV heads is straightforward.

## 4. Attention Architecture

### Hybrid Sliding Window + Full Attention

All Gemma 4 models use a repeating pattern of sliding window attention and full global
attention layers:

- **E2B (35 layers)**: 4 sliding + 1 full, repeating 7x = 35 layers
- **E4B (42 layers)**: 5 sliding + 1 full, repeating 7x = 42 layers  
- **26B-A4B (30 layers)**: 5 sliding + 1 full, repeating 5x = 30 layers
- **31B (60 layers)**: 5 sliding + 1 full, repeating 10x = 60 layers

Sliding window sizes: 512 (E2B, E4B) or 1024 (26B-A4B, 31B).

### Dual Head Dimensions

A unique Gemma 4 feature: sliding attention layers use head_dim=256, while full attention
layers use global_head_dim=512. This means:
- Sliding layers: Q is [batch, n_heads, seq, 256], KV is [batch, n_kv_heads, seq, 256]
- Full layers: Q is [batch, n_heads, seq, 512], KV is [batch, n_kv_heads, seq, 512]

This is different from Llama/Qwen where head_dim is constant across all layers.

**Implementation impact**: Need separate KV cache allocation per layer type, or a unified
cache sized to the larger head_dim with masking.

### Grouped Query Attention (GQA)

All models use GQA, not MHA or MQA (except E2B which has 1 KV head = MQA for sliding):

| Model | Q Heads | KV Heads (sliding) | KV Heads (full/global) | GQA Ratio |
|-------|---------|--------------------|-----------------------|-----------|
| E2B   | 8       | 1                  | N/A                   | 8:1       |
| E4B   | 8       | 2                  | N/A                   | 4:1       |
| 26B-A4B | 16    | 8                  | 2                     | 2:1 / 8:1 |
| 31B   | 32      | 16                 | 4                     | 2:1 / 8:1 |

The larger models (26B-A4B, 31B) have a separate `num_global_key_value_heads` that is
smaller than `num_key_value_heads` -- the full attention layers use fewer KV heads than
the sliding layers. This saves memory for the full-context KV cache.

### attention_k_eq_v

The 26B-A4B and 31B models set `attention_k_eq_v=true`. This means keys and values share
the same projection (K == V), halving the KV projection parameters. This is a novel
parameter-efficiency trick not seen in Llama or Qwen.

### Unified Keys and Values

The 31B model card specifically mentions "Unified Keys and Values in global layers" as a
memory optimization for long context. Combined with the reduced global KV head count (4
vs 16), the full-attention layers are heavily optimized for memory.

## 5. Novel Architecture Features

### 5a. Per-Layer Embeddings (PLE)

The E2B and E4B models have `hidden_size_per_layer_input=256`, which enables Per-Layer
Embeddings. Instead of a single large embedding table, each layer gets a smaller
per-layer embedding input (256-dim) that is projected up to the full hidden size. This
dramatically reduces the embedding table size, which is why these models have "Effective"
parameter counts much lower than total: the 262K vocab with full hidden_size embeddings
would be huge, but PLE keeps it compact.

This is the "E" in E2B and E4B -- "Effective" parameter count after PLE optimization.

### 5b. KV Cache Sharing Across Layers

E2B has `num_kv_shared_layers=20` (out of 35 layers), E4B has `num_kv_shared_layers=18`
(out of 42 layers). This means groups of consecutive layers share the same KV cache,
drastically reducing KV cache memory. This is not present in the larger models (31B and
26B-A4B have num_kv_shared_layers=0).

**Implementation impact**: When implementing the KV cache, we need to track which layers
share KV projections and only compute/store KV once per shared group.

### 5c. Dual RoPE Configuration

Every model uses different RoPE settings for sliding vs full attention layers:

| Attention Type | RoPE Type | Theta | Partial Rotary Factor |
|---------------|-----------|-------|----------------------|
| Sliding       | default   | 10,000 | 1.0 (full rotation) |
| Full          | proportional | 1,000,000 | 0.25 |

The full attention layers only apply RoPE to 25% of the head dimension
(partial_rotary_factor=0.25). With global_head_dim=512, that means RoPE is applied to
only 128 dimensions, and the remaining 384 are position-independent. This is a technique
to improve long-context generalization.

"Proportional" RoPE type likely means the frequencies are scaled proportionally to extend
the effective context window.

### 5d. Logit Softcapping

All models use `final_logit_softcapping=30.0`. The final logits are capped via:
```
logits = 30.0 * tanh(logits / 30.0)
```
This prevents extreme logit values and was introduced in Gemma 2. Must be applied before
sampling.

### 5e. GELU Activation (not SiLU)

All Gemma 4 models use `gelu_pytorch_tanh` activation in the MLP, not SiLU/swish used by
Llama and Qwen. This means the MLP is: up_proj -> GELU -> down_proj (or gate_proj *
GELU(up_proj) for gated variants).

### 5f. Double-Wide MLP (E2B only)

E2B has `use_double_wide_mlp=true`. This likely means the MLP uses a gated architecture
with double the intermediate size split between gate and up projections, similar to
SwiGLU but with GELU.

### 5g. Bidirectional Attention for Vision

The 31B and 26B-A4B models set `use_bidirectional_attention="vision"`, meaning image
tokens use bidirectional (non-causal) attention while text tokens use standard causal
attention. This is only relevant if we implement the vision path.

## 6. MoE Details (26B-A4B)

| Parameter | Value |
|-----------|-------|
| Total experts | 128 |
| Active experts (top-k) | 8 |
| Expert intermediate size | 704 |
| Dense MLP intermediate size | 2112 |
| Shared expert | 1 (per model card; likely the dense MLP) |

Comparison with Qwen MoE:
- Qwen uses 60 experts with top-8, Gemma uses 128 experts with top-8
- Qwen has an explicit shared_expert_gate; Gemma's "shared expert" appears to be the
  dense MLP path (intermediate_size=2112)
- Gemma's per-expert FFN (704) is smaller than Qwen's
- Both route 8 experts per token

**Implementation note**: 128 experts is a lot of weight matrices. Each expert has
in: [2816, 704] and out: [704, 2816] projections. Total expert params:
128 * 2 * 2816 * 704 = ~509M params just for expert FFNs (about 1 GB at BF16).
The routing and gating adds negligible params.

## 7. Weight Format and Tokenizer

### Weights
- **Format**: Safetensors (all models)
- **Precision**: BF16
- All hosted on HuggingFace under the `google/` namespace
- Base and -it variants available for all sizes

### Tokenizer
- **Type**: GemmaTokenizer (SentencePiece-based)
- **Vocabulary size**: 262,144 tokens (262K) -- 2x larger than Llama's 128K
- **Special tokens**: <bos>, <eos>, <pad>, <unk>
- **Multimodal tokens**: <|image|>, <|audio|>, <|tool_call|>, <|think|>
- **Padding side**: left

## 8. Comparison with Models We Already Support

### vs Qwen 0.5B (our fastest model, 93 tok/s)

| Feature | Qwen 0.5B | Gemma 4 E2B | Notes |
|---------|-----------|-------------|-------|
| Params | 0.5B | 5.1B (2.3B eff.) | E2B is ~4.6x larger |
| Hidden | 896 | 1536 | 1.7x wider |
| Layers | 24 | 35 | 1.5x deeper |
| Heads (Q/KV) | 14/2 | 8/1 | E2B uses MQA |
| Head dim | 64 | 256/512 | 4-8x larger heads |
| MLP | SiLU, 4864 | GELU, 6144 | Different activation |
| Vocab | 151936 | 262144 | 1.7x larger vocab |
| RoPE | Standard | Dual (default + proportional) | More complex |
| Sliding window | No | Yes (512) | New requirement |
| KV sharing | No | Yes (20 layers) | New requirement |
| Per-layer embed | No | Yes (256-dim) | New requirement |
| Context | 32K | 128K | 4x longer |

### vs Llama 8B (our largest model, 22 tok/s)

| Feature | Llama 8B | Gemma 4 E4B | Notes |
|---------|----------|-------------|-------|
| Params | 8.0B | 8.0B (4.5B eff.) | Same total, fewer effective |
| Hidden | 4096 | 2560 | Smaller hidden |
| Layers | 32 | 42 | More layers |
| Heads (Q/KV) | 32/8 | 8/2 | Fewer heads, larger dim |
| Head dim | 128 | 256/512 | 2-4x larger heads |
| MLP | SiLU, 14336 | GELU, 10240 | Different activation |
| Vocab | 128256 | 262144 | 2x vocab |
| RoPE | Standard | Dual | More complex |
| Sliding window | No | Yes (512) | New requirement |
| KV sharing | No | Yes (18 layers) | New requirement |
| Per-layer embed | No | Yes (256-dim) | New requirement |
| Logit softcap | No | Yes (30.0) | New requirement |

## 9. Implementation Considerations for Tenstorrent Blackhole

### New features needed (vs existing Llama/Qwen support):

1. **Hybrid attention**: Must handle both sliding window (local) and full (global)
   attention in the same model. Different head dims per attention type. Need to route
   layers to the correct attention implementation.

2. **Dual head dimensions**: head_dim=256 for sliding, 512 for full. The SDPA kernel
   must handle both. 256 should work with flash decode (power-of-2). 512 is large and
   may need special handling.

3. **Partial RoPE**: Only 25% of head dimensions get rotary embeddings in full attention
   layers. Need to split the head dim, apply RoPE to the first quarter, and concatenate.

4. **KV cache sharing** (E2B/E4B): Groups of layers share KV projections. Reduces memory
   but requires careful cache management.

5. **Per-layer embeddings** (E2B/E4B): Input projection from 256-dim per-layer space to
   full hidden size. Need to understand the exact mechanism.

6. **GELU activation**: ttnn supports GELU, so this should be straightforward.

7. **Logit softcapping**: Simple tanh-based capping, easy to implement.

8. **262K vocabulary**: Larger embedding table (262144 x hidden_size). At BF16:
   - E2B: 262144 * 1536 * 2 = ~805 MB (but PLE reduces this)
   - E4B: 262144 * 2560 * 2 = ~1.34 GB (but PLE reduces this)

9. **attention_k_eq_v** (larger models): K and V share projections. Need to duplicate
   the single KV projection output for both K and V paths.

### Recommended bring-up order:

1. **Gemma 4 E4B** -- 8 GB BF16, ~4 GB BF8. Similar total params to Llama 8B but with
   Per-Layer Embeddings reducing effective compute. Good test of all novel features.
   42 layers with GQA (8Q/2KV) is close to our existing patterns.

2. **Gemma 4 E2B** -- 5.1 GB BF16, ~2.6 GB BF8. Smallest model, fastest iteration.
   MQA (8Q/1KV) and double-wide MLP add variety. Good for validation.

3. **Gemma 4 26B-A4B** -- If MoE support is ready. Tests 128-expert routing on top of
   all the novel attention features.

4. **Gemma 4 31B** -- Only if we solve the memory budget at 32 GB. Borderline fit.
