# Open-Source LLM Landscape: Targets for Tenstorrent Blackhole (April 2026)

**Context:** We have Qwen2.5-0.5B running at 4,819 tok/sec aggregate on Blackhole P150 with batch decode. This document surveys the open-source model landscape to identify the best next targets.

**Hardware constraints (Blackhole P150):**
- 120 Tensix cores (each with 1.5 MB L1 SRAM = 180 MB total on-chip)
- 32 GB GDDR6 at 512 GB/s bandwidth
- 664 TFLOPS (BlockFP8)
- Our proven TT-NN op set: matmul, SDPA, RMSNorm, SiLU/SwiGLU, embedding, RoPE, softmax, argmax

---

## 1. Small But Mighty (0.5B-3B) -- Immediate Targets

These models fit comfortably in GDDR6 at BF16 and could reuse most of our Qwen2.5-0.5B pipeline.

### Qwen3-0.6B
- **Architecture:** 28 layers, hidden=1024, 16 query heads, 8 KV heads (GQA), vocab=151,936, SwiGLU, RMSNorm, RoPE, QK-Norm
- **Why interesting:** Direct successor to our Qwen2.5-0.5B. Adds thinking/non-thinking mode toggle and QK-Norm. Nearly identical architecture to what we already run. Trained on 36T tokens (vs Qwen2.5's ~18T).
- **TT-NN mapping:** Almost identical to our current pipeline. QK-Norm is a new op (just RMSNorm on Q and K before attention). Trivial port.
- **Buzz:** Part of the massive Qwen3 family. Strong multilingual (119 languages). Thinking-mode gives chain-of-thought at 0.6B scale.
- **Weight size (BF16):** ~1.2 GB. Fits trivially.

### Qwen3-4B
- **Architecture:** 36 layers, hidden=2560, 32 query heads, 8 KV heads (GQA), vocab=151,936, tied embeddings, SwiGLU, RMSNorm, RoPE, QK-Norm
- **Why interesting:** Sweet spot -- 4B params but with tied embeddings (saves memory). 36 layers is deeper than typical for this size. Thinking mode included.
- **TT-NN mapping:** Same ops as Qwen2.5. Larger matmuls will better utilize the 120 Tensix cores. KV cache at 128K context would be the main memory challenge.
- **Weight size (BF16):** ~8 GB. Comfortable fit.
- **Buzz:** Outperforms many 7B models on reasoning benchmarks.

### Phi-4-mini (3.8B)
- **Architecture:** 32 layers, hidden=3072, 24 query heads, 8 KV heads (GQA), vocab=200,064, SwiGLU, RMSNorm, RoPE (fractional -- 25% position-agnostic), tied input/output embeddings
- **Why interesting:** Microsoft's data-quality-over-scale philosophy. Beats GPT-4o on MATH and GPQA benchmarks. The fractional RoPE is architecturally novel -- only 75% of head dim gets positional encoding, improving long-context generalization. Massive 200K vocab for multilingual.
- **TT-NN mapping:** Standard transformer ops. The fractional RoPE needs a small modification to our RoPE kernel (zero out rotations for last 25% of head dim). The 200K vocab means a larger embedding table (~1.2 GB just for embeddings at BF16).
- **Weight size (BF16):** ~7.6 GB. Fits easily.
- **Buzz:** Enormous community adoption. The Phi-4-mini-reasoning variant adds chain-of-thought. Microsoft's flagship small model.

### SmolLM3-3B (Hugging Face)
- **Architecture:** 36 layers, hidden=2048, 16 query heads, 4 KV heads (GQA), intermediate=11008, NoPE (removes RoPE every 4th layer)
- **Why interesting:** Fully open -- Hugging Face published the complete training recipe, data mixture, and post-training methodology. The NoPE innovation (selectively removing positional embeddings from every 4th layer) improves long-context without hurting short-context. Depth-over-width design philosophy (36 layers at only 2048 hidden).
- **TT-NN mapping:** Almost identical to our pipeline. The NoPE layers are actually simpler (skip RoPE application). Very GQA-heavy (4:1 ratio) which is memory-efficient.
- **Weight size (BF16):** ~6 GB. Easy fit.
- **Buzz:** Outperforms Llama-3.2-3B and Qwen2.5-3B. Fully reproducible training. Community darling.

### Llama 3.2 (1B and 3B)
- **Architecture (3B):** 28 layers, hidden=3072, 24 query heads, 8 KV heads (GQA), vocab=128,256, SwiGLU, RMSNorm, RoPE
- **Why interesting:** Meta's flagship small models. Knowledge-distilled from Llama 3.1 8B and 70B during pretraining (logit-level distillation). The 1B variant is interesting as a speed demon target.
- **TT-NN mapping:** Standard transformer. Very similar to Qwen architecture. Direct port.
- **Weight size (BF16):** ~6 GB (3B), ~2 GB (1B).
- **Buzz:** Huge ecosystem. Ollama/llama.cpp support. Reference model for the industry.

### Gemma 3 (1B text-only)
- **Architecture:** 26 layers, hidden=1152, 4 query heads, 1 KV head (very aggressive GQA), vocab=262,144, GeGLU activation, RMSNorm, RoPE with local/global attention pattern (5 local sliding-window layers per 1 global layer)
- **Why interesting:** Google's local/global attention interleaving -- 5 layers of sliding-window (1024 tokens) then 1 layer of full global attention. This is architecturally distinct from everything else we've run. Extremely aggressive GQA (4:1) makes KV cache tiny.
- **TT-NN mapping:** The sliding-window attention is a new primitive for us. Would need to mask or limit the attention span in SDPA. The 262K vocab is the largest of any model here (~600 MB embedding table at BF16). GeGLU is slightly different from SwiGLU (uses GELU instead of SiLU).
- **Weight size (BF16):** ~2 GB. Fits trivially.
- **Buzz:** Google's answer to small models. Multimodal at 4B+. Strong on multilingual tasks.

---

## 2. Medium Models (3B-14B) -- Stretch Goals on Single Chip

These models fit in 32 GB GDDR6 at BF16 but leave less room for KV cache. BF8/INT8 quantization would help.

### Qwen3-8B
- **Architecture:** 36 layers, hidden=4096, 32 query heads, 8 KV heads (GQA), vocab=151,936, no tied embeddings, SwiGLU, RMSNorm, RoPE, QK-Norm, 128K context
- **Why interesting:** Same family as our 0.5B. The jump from 4B to 8B is mostly wider (4096 vs 2560 hidden) not deeper (same 36 layers). Untied embeddings means a separate output projection.
- **TT-NN mapping:** Identical ops. The 4096-wide matmuls should utilize cores very efficiently. At BF16 weights are ~16 GB, leaving 16 GB for KV cache and activations.
- **Weight size (BF16):** ~16 GB. Tight but feasible. INT8 weights would be ~8 GB.
- **Buzz:** Strong reasoning model. Thinking mode. Part of the dominant Qwen ecosystem.

### Qwen3-14B
- **Architecture:** 40 layers, hidden=5120, 40 query heads, 8 KV heads (GQA), vocab=151,936
- **Why interesting:** The largest Qwen3 dense model that could possibly fit on single chip.
- **TT-NN mapping:** Same ops. At BF16, weights are ~28 GB -- too tight. INT8 weights (~14 GB) would work with room for KV cache.
- **Weight size (BF16):** ~28 GB. Requires quantization.
- **Buzz:** Sweet spot for many deployment scenarios. Good quality/cost ratio.

### Gemma 3-4B (Multimodal)
- **Architecture:** 34 layers, hidden=2560, local/global attention interleaving (5:1), SigLIP vision encoder, 128K context
- **Why interesting:** First multimodal target. The vision encoder (SigLIP) processes images into tokens that get concatenated with text tokens. Would let us demonstrate image understanding on Blackhole.
- **TT-NN mapping:** Text decoder is standard transformer. Vision encoder is a ViT (Vision Transformer) -- needs conv2d for patch embedding, then standard attention. New territory for us.
- **Weight size (BF16):** ~8 GB (text) + ~400 MB (vision). Fits well.
- **Buzz:** Google's multimodal play at small scale. Strong community adoption.

### Mistral Small 3.1 (24B)
- **Architecture:** Dense transformer, 24B params, 131K vocab (Tekken tokenizer), 128K context, vision-capable
- **Why interesting:** Dense 24B is the largest dense model anyone would want to run on a single accelerator. Apache 2.0 license. Known for fast inference (150 tok/sec on RTX 4090).
- **TT-NN mapping:** Standard transformer ops. At BF16 (~48 GB) it won't fit. INT4 quantization (~12 GB) would be required.
- **Weight size (BF16):** ~48 GB. Does NOT fit. INT4 required.
- **Buzz:** Mistral's flagship open model. Strong agent/function-calling capabilities. Competes with GPT-4o mini.

### Qwen3-30B-A3B (MoE)
- **Architecture:** 48 layers, 30B total params, 3B active per token, 128 experts per layer with top-8 routing + 1 shared expert, GQA, SwiGLU, RMSNorm, RoPE
- **Why interesting:** This is the gateway MoE model. Only 3B params active per token (similar compute to our current models) but with 30B total knowledge. The routing mechanism selects 8 of 128 experts per token.
- **TT-NN mapping:** This is where it gets interesting. Each expert is a small FFN. The router is a linear layer + softmax + top-k. We'd need to implement sparse expert dispatch -- only run the 8 selected FFNs per token. On Blackhole, we could potentially assign different experts to different Tensix cores (120 cores, 128 experts -- almost 1:1!). The shared expert runs on every token.
- **Weight size (BF16):** ~60 GB. Does NOT fit at BF16. INT4 (~15 GB) makes it feasible. But the active parameters per forward pass are only ~6 GB.
- **Buzz:** Qwen's answer to efficient scaling. Thinking mode at MoE scale.

---

## 3. MoE Architectures -- The Big Opportunity

MoE models are the most interesting architectural target for Blackhole because the 120-core mesh naturally maps to expert parallelism.

### How MoE Works

1. **Router:** A learned linear layer produces logits over N experts for each token
2. **Top-K selection:** Pick the K experts with highest affinity scores (typically K=2 or K=8)
3. **Dispatch:** Route each token to its selected experts
4. **Compute:** Each expert is a standard FFN (linear -> activation -> linear)
5. **Combine:** Weighted sum of expert outputs (weights from router softmax)

### DeepSeek V3 / R1 (671B total, 37B active)
- **Architecture:** 61 layers, 256 routed experts + 1 shared expert per MoE layer, 8 experts activated per token, Multi-head Latent Attention (MLA), intermediate dim 2048 per expert
- **MLA innovation:** Instead of standard KV cache, MLA compresses keys and values into a low-rank latent space, dramatically reducing KV cache size. This is a significant architectural departure from standard GQA.
- **Auxiliary-loss-free load balancing:** Instead of adding a loss term to balance expert usage (which hurts quality), DeepSeek adds a bias term to routing scores that's manually adjusted. The bias is only used for routing decisions, not training loss. This was a breakthrough.
- **Why interesting for Blackhole:** Way too large for single chip (671B params), but the architectural innovations (MLA, auxiliary-loss-free routing) are worth studying. A distilled/small DeepSeek-MoE variant would be ideal.
- **Buzz:** The model that proved open-source can match frontier closed models. R1's reasoning capabilities shocked the industry in January 2025. V3.2 and R2 are the latest iterations.

### Llama 4 Scout (109B total, 17B active)
- **Architecture:** 16 experts per MoE layer, MoE in every layer, 17B active params, iRoPE (interleaved attention layers without positional embeddings), 10M context
- **Why interesting:** Meta's first MoE. The iRoPE architecture alternates layers with and without positional embeddings, enabling extreme context lengths (10M tokens). 16 experts is much more manageable than DeepSeek's 256.
- **TT-NN mapping:** At INT4, total weights ~27 GB, which fits in 32 GB GDDR6. 16 experts per layer could map to groups of ~7 Tensix cores each. Active compute per token is ~17B params, similar to running a 17B dense model.
- **Weight size:** ~218 GB (BF16), ~27 GB (INT4). INT4 fits!
- **Buzz:** Meta's MoE debut. The 10M context is industry-leading.

### Llama 4 Maverick (400B total, 17B active)
- **Architecture:** 128 experts + 1 shared expert, MoE and dense layers alternating (MoE in half the layers), 17B active, 1M context
- **Why interesting:** 128 experts maps beautifully to 120 Tensix cores. Alternating dense/MoE layers means half the layers are standard transformers (easy) and half are MoE (interesting). Same active params as Scout but much more total knowledge.
- **TT-NN mapping:** At INT4, total weights ~100 GB -- does NOT fit on single chip. Would need multi-chip or aggressive pruning. But the architecture is a dream for expert parallelism study.
- **Buzz:** Meta's flagship. Used internally for WhatsApp, Messenger, Instagram.

### Qwen3.6-35B-A3B
- **Architecture:** 35B total, 3B active per token, latest Qwen MoE generation (2026)
- **Why interesting:** Only 3B active parameters. If this fits in memory (INT4 ~9 GB), we get frontier-quality outputs at small-model compute costs.
- **Buzz:** Qwen 3.6 scores 73.4% on SWE-bench Verified while activating only 3B parameters. Remarkable efficiency.

### How MoE Maps to Blackhole's 120-Core Architecture

The key insight: **each expert is an independent FFN computation**. With 120 Tensix cores:

| Experts per layer | Cores per expert | Utilization |
|---|---|---|
| 8 | 15 | Perfect -- all cores used |
| 16 | 7-8 | Good -- slight waste |
| 128 | ~1 | Each core hosts ~1 expert. Only activated cores compute. |
| 256 | <1 | Experts must time-share cores |

For models with 128 experts (Maverick, Qwen3-30B-A3B), the mapping is elegant: assign ~1 expert per core, then only the 8 selected cores actually compute per token. The other 112 cores sit idle for the MoE layers -- but this matches how MoE is supposed to work (sparse activation).

The challenge is **expert weight residency**. If each expert's FFN is ~50 MB, 128 experts = 6.4 GB per MoE layer. With 48+ MoE layers, that's 300+ GB -- doesn't fit in GDDR6. Solutions:
1. INT4 quantization (4x reduction)
2. Expert offloading (stream from host memory)
3. Smaller MoE models (Qwen3-30B-A3B, Llama 4 Scout)

---

## 4. Multimodal Models -- Vision + Language

### Architecture Pattern

All modern VLMs follow the same basic pattern:
1. **Vision encoder** (usually a ViT variant) processes image into patch embeddings
2. **Projection layer** maps vision embeddings to LLM's hidden dimension
3. **LLM decoder** processes interleaved text + vision tokens

The LLM decoder is a standard transformer -- our existing pipeline handles it. The new work is the vision encoder.

### Qwen3-VL (2B / 8B / 32B / 235B-A22B)
- **Vision encoder:** DFN-based ViT with dynamic resolution (processes images at native resolution, not fixed 224x224)
- **Architecture:** Standard Qwen3 decoder + vision encoder + projection MLP
- **Why interesting:** Dynamic resolution means variable-length vision token sequences. The 2B variant is a natural next step from our Qwen text models.
- **TT-NN mapping:** The ViT needs: patch embedding (conv2d or reshape+linear), standard attention, LayerNorm, GELU. The projection is just a 2-layer MLP. Then the decoder is our existing pipeline.
- **Buzz:** Rivals GPT-5 and Gemini-2.5-Pro on multimodal benchmarks (at the 235B scale). The small variants are competitive with much larger models.

### Phi-4-Multimodal (3.8B)
- **Architecture:** Unified framework handling vision + audio + text. Single transformer backbone with modality-specific encoders feeding into shared hidden space.
- **Why interesting:** True multimodal (not just vision-language). Audio understanding at 3.8B parameters. Could enable speech-to-text on Blackhole.
- **TT-NN mapping:** The audio encoder (likely Whisper-based) needs 1D convolutions and attention. Vision encoder is standard ViT. Both project into the shared transformer.
- **Weight size (BF16):** ~8-10 GB total (all modalities). Fits well.
- **Buzz:** Most capable multimodal model under 4B params.

### LLaVA-OneVision-1.5
- **Architecture:** RICE-ViT (cluster discrimination vision encoder) + LLM decoder. Focuses on region-aware visual understanding and OCR.
- **Why interesting:** Fully open-source training recipe. Strong on document understanding and OCR -- practical use cases.
- **TT-NN mapping:** Standard ViT + transformer. The RICE-ViT modifications are minor architectural changes to the vision encoder.
- **Buzz:** Academic community favorite. Outperforms Qwen2.5-VL on some benchmarks.

### Vision Encoder Implementation on Blackhole

A typical ViT vision encoder for VLMs:
1. **Patch embedding:** Split image into 14x14 or 16x16 patches, linear projection to hidden dim. This is a single matmul (or conv2d).
2. **Positional embedding:** Add learned position embeddings (simple tensor addition).
3. **Transformer blocks:** Standard attention + FFN, with LayerNorm (not RMSNorm). We have all these ops.
4. **Output projection:** Linear layer to match LLM hidden dim.

The main new ops needed: LayerNorm (instead of RMSNorm), GELU (instead of SiLU), and potentially conv2d for patch embedding. All are available in TT-NN.

---

## 5. Architecturally Novel Models -- Beyond Standard Transformers

### Mamba / State Space Models (SSM)

**Core idea:** Replace attention (O(n^2) in sequence length) with a state-space model (O(n) linear complexity).

**How it works:**
- Maintains a fixed-size hidden state (like an RNN)
- Each new token updates the state via learned matrices (A, B, C, D)
- Mamba's innovation: the state transition matrices are **input-dependent** (selective), unlike classic SSMs which are fixed
- Hardware-aware implementation uses a scan operation (prefix sum) instead of sequential RNN steps

**Falcon Mamba 7B:**
- Pure Mamba architecture (no attention at all)
- 7B parameters, permissive license
- Significantly faster inference and lower memory than transformer equivalents at long sequences
- Constant memory regardless of sequence length (no KV cache!)

**TT-NN mapping challenge:** The selective scan is fundamentally different from attention. It's a sequential scan with element-wise operations -- more like a reduction than a matmul. TT-NN may not have a native selective scan op. We'd need to either:
1. Implement it as a custom kernel on Tensix RISC-V cores
2. Decompose it into primitive ops (multiply, add, scan)
3. Use the SFPU (SIMD engine) for element-wise operations

**Why interesting for Blackhole:** No KV cache means infinite context at constant memory. The scan operation could potentially parallelize across Tensix cores (parallel prefix sum). But the lack of large matmuls means we wouldn't be utilizing the tensor math units as efficiently as with transformers.

### Jamba (Hybrid Transformer + Mamba + MoE)

**Architecture:** Interleaved blocks of Transformer attention layers and Mamba SSM layers, combined with MoE.

**Jamba 1.5 family:**
- Jamba-1.5-Large: 94B active params (398B total), 72 layers
- Jamba-1.5-Mini: 12B active params, 256K context
- Ratio: ~90% Mamba layers, ~10% Transformer layers (the few attention layers handle tasks Mamba struggles with, like exact copying and in-context learning)

**Key insight:** Pure Mamba struggles with exact token copying and few-shot learning. Adding just 10% attention layers fixes this while keeping 90% of the efficiency gains.

**TT-NN mapping:** This is a mixed workload:
- Transformer layers: Our existing pipeline (matmul, SDPA, RMSNorm)
- Mamba layers: Selective scan (new primitive needed)
- MoE routing: Expert dispatch (new for us)
- All three paradigms in one model -- ultimate stress test

**KV cache comparison (256K context):**
| Model | KV cache size |
|---|---|
| Jamba 1.5 | 4 GB |
| Mixtral 8x7B | 32 GB |
| Llama-2-70B | 128 GB |

**Buzz:** AI21 Labs model. Demonstrates that hybrid architectures are the future -- pure transformers may not be optimal.

### RWKV-7 "Goose"

**Architecture:** RNN that can be trained like a transformer (parallelizable). Linear time, constant space, no KV cache.

**Key innovation:** Generalized delta rule with vector-valued gating and in-context learning rates. Can perform state tracking and recognize all regular languages.

**How it differs from Mamba:**
- RWKV is an RNN with channel-wise time mixing and token mixing
- Mamba uses continuous state spaces with selective mechanisms
- RWKV-7 adds expressive dynamic state evolution (the state itself evolves based on input)

**TT-NN mapping:** Similar challenges to Mamba -- needs sequential state update operations rather than large matmuls. The parallelizable training mode uses a different computation graph than inference mode.

**Buzz:** Active open-source community. RWKV Foundation drives development. Interesting for infinite-context applications. Apache 2.0 license.

### Falcon-H1 (Hybrid Mamba-Transformer)

**Architecture:** Hybrid with Mamba SSM layers + Transformer attention layers. Released by TII (Abu Dhabi).

**Falcon-H1R 7B (2026):** Reasoning-focused variant combining hybrid architecture with chain-of-thought training. Linear-time sequence processing with lower memory usage.

**TT-NN mapping:** Same hybrid challenge as Jamba but at a more manageable 7B scale. Good candidate for testing hybrid architectures on Blackhole.

**Buzz:** TII has deep pockets and ships regularly. The H1 series proves hybrid architectures work at production quality.

---

## 6. Recommended Target Progression

Based on our current capabilities (Qwen2.5-0.5B at 4,819 tok/sec) and Blackhole P150 constraints:

### Phase 1: Direct ports (weeks, not months)
| Model | Why | New ops needed | Weight size (BF16) |
|---|---|---|---|
| **Qwen3-0.6B** | Direct successor, same architecture + QK-Norm | RMSNorm on Q,K | 1.2 GB |
| **Llama 3.2-1B** | Validates generality of our pipeline | None (same ops) | 2 GB |
| **Llama 3.2-3B** | First "bigger" model | None | 6 GB |

### Phase 2: Interesting small models (moderate effort)
| Model | Why | New ops needed | Weight size (BF16) |
|---|---|---|---|
| **Phi-4-mini (3.8B)** | Fractional RoPE, huge vocab, top benchmarks | Modified RoPE | 7.6 GB |
| **SmolLM3-3B** | NoPE layers, depth-over-width, fully open | Conditional RoPE skip | 6 GB |
| **Qwen3-4B** | Thinking mode, strong benchmarks | QK-Norm | 8 GB |

### Phase 3: Scale up (significant effort)
| Model | Why | New ops needed | Weight size (BF16) |
|---|---|---|---|
| **Qwen3-8B** | Proves Blackhole can handle "real" models | None new | 16 GB |
| **Gemma 3-4B** | Local/global attention, multimodal-capable | Sliding-window attention, GeGLU, LayerNorm | 8 GB |

### Phase 4: Architectural frontiers (research projects)
| Model | Why | New ops needed | Weight size |
|---|---|---|---|
| **Qwen3-30B-A3B (MoE)** | Gateway to MoE on Blackhole | Expert routing, sparse dispatch | ~15 GB (INT4) |
| **Llama 4 Scout** | MoE + iRoPE, fits in INT4 | Expert routing, iRoPE | ~27 GB (INT4) |
| **Falcon-H1R 7B** | Hybrid Mamba+Transformer | Selective scan kernel | ~14 GB |
| **Qwen3-VL 2B** | First multimodal | ViT encoder, conv2d, LayerNorm, GELU | ~4 GB |

### Phase 5: Moonshots
| Model | Why | Challenge |
|---|---|---|
| **Llama 4 Maverick** | 128 experts = 120 cores, perfect match | 100 GB at INT4, needs multi-chip |
| **Jamba 1.5 Mini** | Transformer + Mamba + MoE hybrid | Three paradigms in one model |
| **DeepSeek V3 (distilled)** | MLA attention, auxiliary-loss-free routing | Novel attention mechanism |

---

## 7. Key Architectural Innovations to Watch

### Multi-head Latent Attention (MLA) -- DeepSeek
Compresses KV cache into a low-rank latent space. Instead of caching full K and V tensors, cache a compressed representation and reconstruct on-the-fly. Dramatically reduces memory for long contexts. Would require new TT-NN ops for the compression/decompression.

### iRoPE -- Llama 4
Alternates layers with and without positional embeddings. Layers without RoPE can attend to any position equally, enabling extreme context lengths (10M tokens). Simple to implement -- just conditionally skip RoPE.

### QK-Norm -- Qwen3, Gemma 3
Apply RMSNorm to Q and K tensors before computing attention scores. Stabilizes training and inference at scale. Trivial to add (one extra RMSNorm call per attention layer).

### NoPE (No Positional Embedding) layers -- SmolLM3
Remove RoPE from every Nth layer. Improves long-context extrapolation. Even simpler than iRoPE -- just skip RoPE.

### Thinking Mode -- Qwen3
Model can internally "think" (generate hidden chain-of-thought tokens) before producing the final answer. The thinking budget is controllable. Doesn't change the architecture -- it's a prompting/generation strategy. But it means our inference pipeline needs to support generating many tokens (thinking) and then extracting the final answer.

### Expert Parallelism on Mesh Architectures
The most unexplored opportunity. Blackhole's 120-core mesh with NoC interconnect is almost purpose-built for MoE routing. Each core can host expert weights in L1 SRAM (1.5 MB per core). A small expert FFN (e.g., in Qwen3-30B-A3B) might be ~500 KB at INT4, fitting entirely in one core's L1. This would enable **fully on-chip expert computation** with zero DRAM round-trips for the MoE layers.

---

## Sources

- [BentoML: Best Open-Source SLMs 2026](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [NVIDIA Blog: Qwen3-Next Hybrid MoE](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
- [DeepSeek V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
- [Fireworks: DeepSeek V3/R1 Architecture](https://fireworks.ai/blog/deepseek-model-architecture)
- [Epoch AI: DeepSeek Transformer Improvements](https://epoch.ai/gradient-updates/how-has-deepseek-improved-the-transformer-architecture/)
- [Phi-4-mini Technical Report (arXiv:2503.01743)](https://arxiv.org/abs/2503.01743)
- [Phi-4-mini on HuggingFace](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [Llama 3.2 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
- [Llama 4 Blog Post](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786)
- [Google Developers: What's New in Gemma 3](https://developers.googleblog.com/gemma-explained-whats-new-in-gemma-3/)
- [SmolLM3 Blog Post](https://huggingface.co/blog/smollm3)
- [SmolLM3-3B on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [Mistral Small 3.1](https://mistral.ai/news/mistral-small-3-1)
- [Jamba: Hybrid Transformer-Mamba (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887)
- [RWKV-7 "Goose" (OpenReview)](https://openreview.net/forum?id=ayB1PACN5j)
- [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM)
- [Falcon Mamba (arXiv:2410.05355)](https://arxiv.org/abs/2410.05355)
- [Tenstorrent Blackhole Specifications](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)
- [Tenstorrent Blackhole Microbenchmarking (ASPLOS 2025)](https://asplos.dev/wordpress/wp-content/uploads/2025/09/TT_bench-1.pdf)
- [BentoML: Open-Source Vision Language Models 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [HuggingFace: VLMs 2025](https://huggingface.co/blog/vlms-2025)
- [Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6)
- [Sebastian Raschka: Qwen3 From Scratch](https://magazine.sebastianraschka.com/p/qwen3-from-scratch)
