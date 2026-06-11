# Home LLM Landscape — 2026-06

Snapshot of the open-source LLM ecosystem relevant to a personal/home chat
server on Tenstorrent Blackhole (4× P150, ~32 GB DRAM/chip). Audience already
has Qwen3.6-27B (dense FFN, hybrid attn+DeltaNet) and Qwen3.6-35B-A3B (MoE)
brought up; the question is "what's worth doing next?"

All facts dated; speculative items marked "not found" or "unconfirmed".

---

## 1. DeepSeek family

**Released (open weights) as of mid-2026:**

| Model            | Date          | Total / Active | Notes |
|------------------|---------------|----------------|-------|
| DeepSeek-V3      | Dec 2024      | 671B / 37B MoE | MLA + MoE, original |
| DeepSeek-R1      | Jan 2025      | 671B / 37B MoE | RL reasoning over V3-base; the model that broke into mainstream news |
| R1 distills      | Jan 2025      | 1.5B–70B dense | SFT-only distills onto Qwen2.5 / Llama3 backbones (1.5/7/8/14/32/70B) |
| DeepSeek-V3.1    | Aug 2025      | 671B / 37B MoE | Merged chat + reasoner ("thinking mode" toggle) |
| DeepSeek-V3.2-Exp| Sep 2025      | 671B / 37B MoE | First model with **DeepSeek Sparse Attention (DSA)** on top of MLA |
| DeepSeek-V3.2    | Dec 2025      | 671B / 37B MoE | Full V3.2 release; DSA + improved training mix ([arxiv 2512.02556](https://arxiv.org/abs/2512.02556)) |
| DeepSeek-V4      | Apr 24 2026   | not disclosed  | "ultra-long context usable in production", 32T tokens, 1M native context |
| DeepSeek-V4-Pro  | Apr 24 2026   | larger variant | Same family |

**R2 status:** *not released as of 2026-06.* Repeatedly rumoured but never
shipped; DeepSeek instead pushed the V3.x line plus distills.

**Chat-friendly variants:** the R1 distill onto Qwen2.5-32B is the most
home-friendly: dense, 32B fits TP across 4× P150 trivially, Apache-2.0
licensed via the Qwen2.5 base, and *delivered the largest "small reasoning
model" jump of 2025*. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` on HF.

**Reception:** R1 was *the* viral local-LLM event of early 2025 (the trigger
for the now-infamous NVIDIA stock drop). V3.x is too big for a single home
host without aggressive quantization. The distills are what people actually
run.

---

## 2. Google — Gemma 3 and Gemma 4

**Gemma 3** (Mar 12 2025): 1B / 4B / 12B / 27B. 4B/12/27B are
**vision-capable** (SigLIP encoder); 1B is text-only. 128K context. License
is Gemma's own (not Apache). Widely adopted as the multimodal default at home
through 2025-26.

**Gemma 4** (Apr 2 2026, [blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)):
**Apache 2.0** (big change from prior Gemma licenses). Four SKUs:

| SKU      | Architecture           | Total / Active | Notes |
|----------|------------------------|----------------|-------|
| E2B      | dense, edge-tuned      | 5.1B / 2.3B    | Native audio input |
| E4B      | dense, edge-tuned      | ~9B / 4B       | Native audio input |
| 26B MoE  | 128 experts, top-8 + 1 shared | 26B / 3.8B | Fast token/s for size |
| 31B Dense| dense                  | 31B            | Currently #3 on Arena open leaderboard |

All four are multimodal (text + image + audio + video input). At launch
Gemma 4 31B ranked #3 open and 26B-MoE ranked #6 on Arena. Adoption is
growing fast — Gemma 4 E4B has become a common "small daily driver" recommendation
([XDA](https://www.xda-developers.com/found-open-source-local-llm-that-competes-with-cloud-ai-gemma-4/)).

---

## 3. Other notable open releases (1–50B sweet spot)

**Mistral** ([news page](https://mistral.ai/news/mistral-3/)):
- **Mistral Small 3 / 3.1 / 3.2** (Jan–Jun 2025): 24B dense, GQA (32Q/8KV),
  Apache-2.0, 128K context, Pixtral vision encoder in 3.1+
  (`mistralai/Mistral-Small-3.2-24B-Instruct-2506`). Widely cited as the best
  pure-dense 24B at home in 2025.
- **Mistral 3** (Dec 2025 / early 2026): dense 3B, 8B, 14B + **Mistral Large 3**
  (675B total / 41B active sparse MoE), Apache-2.0.
- **Mistral Small 4** (Mar 2026): 119B total / 6B active MoE; instruct +
  reason + vision + code in one model.
- **Mistral Medium 3.5** (Apr 29 2026): 128B **dense**, agentic features.
- **Voxtral TTS** (Mar 2026): open-source text-to-speech, 9 languages.

**Microsoft Phi:**
- **Phi-4** (Dec 2024, 14B dense, MIT)
- **Phi-4-mini** (3.8B dense)
- **Phi-4-multimodal** (5.6B, unified text+speech+vision)
- **Phi-4-reasoning-vision-15B** (Mar 2026) — current latest
- **Phi-5: not found** as of 2026-06.

**NVIDIA Nemotron 3** (GTC 2026): hybrid latent-MoE architecture, three
sizes — Nano (4B), Super (120B), Ultra. Nemotron 3 Nano Omni adds
multimodal (text + image + audio + video). Nemotron 4 announced as the
"coalition" follow-up but **not yet shipped**.

**Zhipu / Z.ai GLM** ([HF org](https://huggingface.co/zai-org)):
GLM-4.5 → 4.5V → 4.6 → 4.7 (Jul–Dec 2025), then **GLM-5** (Feb 2026,
flagship 744B) and **GLM-5.1** (Apr 8 2026, also open).
GLM-4.6V is the open native-tool-calling vision model. The 32B-active /
355B-total GLM-4.6 became a popular home target after late-2025 Cambricon
FP8/INT4 quantizations dropped its footprint.

**Moonshot Kimi:**
- **Kimi K2.5** (Jan 27 2026, 1T-param MoE, open-source, agent coder)
- **Kimi K2.6** (Apr 20 2026, 1T total / 32B active MoE, Modified MIT,
  256K context, scores 58.6 on SWE-Bench Pro — ahead of GPT-5.4/Claude
  Opus 4.6 on that benchmark)
- Too large for 4× P150 without serious quantization; mentioned for
  completeness.

**HuggingFace SmolLM3-3B** (Jul 8 2025): fully open (data + code + weights),
dual-mode reasoning, 64K context (128K via YaRN), GQA + NoPE. Outperforms
Llama-3.2-3B and Qwen2.5-3B; competitive with 4B-class. `HuggingFaceTB/SmolLM3-3B`.

**Reasoning-specific:**
- DeepSeek-R1 distills (above)
- Qwen3 / Qwen3.6 with `/think` toggle (already covered)
- SmolLM3 dual-mode (small reasoner)
- Phi-4-reasoning-vision-15B

**Coder models** (from PromptQuorum / Pinggy 2026 surveys):
- **Qwen3-Coder-30B**: SWE-bench Verified 58.7%, 256K context, Apache-2.0;
  current top recommended "single-GPU coder" at home.
- **DeepSeek-Coder-V3**: heavyweight, 48 GB+ VRAM, longest context.
- **Codestral 22B**, **StarCoder 2**, **Granite Code** — also present.

---

## 4. What r/LocalLLaMA actually runs at home (2026-Q2)

From the survey roundups
([codersera May 2026](https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/),
[HF blog](https://huggingface.co/blog/daya-shankar/open-source-llms),
[XDA](https://www.xda-developers.com/found-open-source-local-llm-that-competes-with-cloud-ai-gemma-4/),
[StationX local-LLM monthly](https://app.stationx.net/articles/best-local-llm)):

| Tier         | Default daily-driver                          |
|--------------|------------------------------------------------|
| ~3B (edge)   | SmolLM3-3B, Phi-4-mini, Gemma 4 E4B           |
| ~7–9B        | Qwen3-8B, Llama 3.1-8B-Instruct (still!)      |
| ~12–14B      | Phi-4 (14B), Gemma 3-12B                       |
| ~24–32B      | **Mistral Small 3.2 24B**, Qwen3-32B, Gemma 4 31B-dense |
| ~70B         | Llama 3.3-70B-Instruct, R1-Distill-Llama-70B  |
| MoE 100B+    | Qwen3-235B-A22B, Gemma 4 26B-MoE              |

**Consensus default daily-driver (2026):** Qwen3 family + Gemma 4 are the two
"safe defaults"; Mistral Small 3.2 24B is the dense crowd-favourite that
"fits in 32 GB once quantized". The Qwen3.6 family (which the user already
has) is the **bleeding edge** dense+hybrid story.

**Quantization vs full precision:** Q4_K_M / Q5_K_M GGUF for llama.cpp/Ollama
remains the home default (~50% size, <1% benchmark loss). bf16/fp16 is reserved
for serving stacks and hardware that supports it natively (Tenstorrent
Blackhole, NVIDIA, MLX). FP8 is now standard on Hopper/Blackwell GPUs and is
the *server-side* default at LM Studio + vLLM.

---

## 5. Recommendation for our bringup

The user already covers:
- **Hybrid attn + DeltaNet + dense FFN** (Qwen3.6-27B)
- **Hybrid attn + DeltaNet + sparse MoE FFN** (Qwen3.6-35B-A3B)

For framework generalization the highest-value gaps are:
- **Pure dense attention** (no DeltaNet / Mamba) — to validate generic
  decoder-only GQA paths and to remove an experimental kernel from the
  critical path
- **A different vendor's attention layout** (Gemma's per-head bias /
  alternating local-global, Mistral's exact 32Q/8KV, etc.)
- **A multimodal SKU** (different tokenizer, different vision tower) —
  generalises the embedding / prefill paths

### Top 3 concrete suggestions

#### A. Mistral Small 3.2 24B (`mistralai/Mistral-Small-3.2-24B-Instruct-2506`)

**Pros:**
- **Pure dense GQA** — vanilla decoder-only transformer; no DeltaNet, no MoE,
  no scan kernels. Forces the framework to be solid on the boring path.
- Different attention shape than Qwen3.6 (32 Q heads / 8 KV, vs Qwen3.6-27B's
  layout). Confirms generality of TP-attention plumbing.
- 24B fits cleanly in single-chip bf16 (~48 GB → split 2-chip) or 4-chip TP
  with massive headroom for long context.
- 128K context, Pixtral vision encoder available (3.1+) for a future
  multimodal milestone.
- Apache 2.0; widely used at home → easy to find reference traces and
  cosine-validate against `transformers`.

**Cons:**
- Smaller than what the hardware can comfortably hold (4× P150 is 128 GB
  total) — won't stress TP. Possibly an argument *for* it: clean ports first,
  scale later.
- Not the freshest model; Mistral Medium 3.5 (128B dense, Apr 2026) would be
  more topical but doesn't fit cleanly.

#### B. Gemma 4 31B Dense (Apache 2.0, Apr 2026)

**Pros:**
- Brand-new (Apr 2026), genuinely strong (#3 open on Arena).
- **Pure dense**, 31B fits comfortably across 4× P150 with TP=4.
- **Multimodal** (text + image + audio + video) — first real test of the
  framework's vision / audio embedding paths.
- Different architecture lineage (DeepMind's Gemini-3-derived). Generalisation
  win: not another Qwen-family model.
- Apache 2.0, active community, ongoing updates expected.

**Cons:**
- Gemma's softcap / sliding-window-attention alternation is its own family
  of kernel quirks — not free.
- Vision encoder integration is real engineering, not a bringup-friendly
  Hello World.

#### C. DeepSeek-R1-Distill-Qwen-32B (Jan 2025, Apache via Qwen2.5 base)

**Pros:**
- **Pure dense, Qwen2.5 architecture** — different enough from Qwen3.6
  (no DeltaNet, no hybrid layers) that it exercises the dense-only code
  path, but close enough to the Qwen tokenizer/weights that bringup
  shouldn't surprise us.
- **Reasoning-tuned**: getting `/think` style decoding working end-to-end
  is real signal that the serving stack handles long output streams + sampling
  cleanly. Good fit for the existing continuous-batching work.
- 32B → fits 1-chip aggressive quant or 2/4-chip bf16.
- Community traction is enormous; lots of evals/datasets exist.

**Cons:**
- Architectural delta from Qwen3.6-27B is real but smaller than (A) or (B).
- Distill is a 2025 model; less "shiny" than Gemma 4 or DeepSeek-V4.

### Recommended ordering

1. **Mistral Small 3.2 24B first** — cleanest, fastest port; validates the
   "boring dense" path and the existing TP plumbing on a non-Qwen tokenizer.
2. **Gemma 4 31B-dense next** — the multimodal stretch; biggest community
   payoff in 2026.
3. **DeepSeek-R1-Distill-Qwen-32B** — exercise reasoning + continuous
   batching on long outputs without building a new tokenizer.

Avoid for now: Llama-4-Scout / Maverick (MoE, not what the user wants more
of), Kimi K2.6 (1T params), DeepSeek-V4 (too big), GLM-5 (744B). DeepSeek-V3.2
is interesting *only* if we want to attempt a DSA kernel — that's a separate
research project.

---

## 6. Multi-modal

**Vision-language (open weights, 2026):** dominated by Qwen2.5-VL, InternVL3,
Llama-4 multimodal, Pixtral (Mistral), Phi-4-multimodal, Molmo, and NVLM.
Qwen2.5-VL-72B leads open-weight VLMs at ~70.2% MMMU and ~888 OCRBench
(May 2026). The newer **Qwen3-VL** (235B-A22B MoE and 30B-A3B MoE) is the
state of the art — both have FP8 variants. 256K context, interleaved-MRoPE,
DeepStack ViT fusion.
([HF Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL))

**Smallest competent VLM (2026):** `Qwen/Qwen2.5-VL-7B-Instruct` — retains
most OCR quality; or `microsoft/Phi-4-multimodal` (5.6B, unified speech+
vision+text). For ultra-small: SmolVLM (HF) at sub-3B.

**Audio-text for chat:** Gemma 4 E2B/E4B have **native audio input**
(speech-recognition + understanding) — would be the most "interesting" small
multimodal pick if we also wanted voice in. Mistral's Voxtral TTS (Mar 2026)
covers the output side but is a separate model.

**For our bringup:** if multimodal is a priority, **Gemma 4 31B dense** doubles
as the multimodal milestone (covers text + image + audio + video in one
weight file). Otherwise defer multimodal; it's a real engineering line item.

---

## 7. Tooling landscape

From [codersera 2026 runtime survey](https://codersera.com/blog/ollama-vs-lm-studio-vs-vllm-vs-llama-cpp-vs-mlx-2026/)
and [BIZON's inference engine guide](https://bizon-tech.com/blog/best-llm-inference-engines):

| Runtime    | Layer            | When people use it |
|------------|------------------|--------------------|
| **llama.cpp** | engine (C++)    | embedded / weird hardware / raw control |
| **MLX**       | engine (Apple)  | Apple Silicon best path; Ollama 0.19+ uses MLX under the hood on M-series |
| **Ollama**    | experience layer over llama.cpp/MLX | single-user "Docker for LLMs"; easiest local install |
| **LM Studio** | GUI on llama.cpp/MLX | desktop GUI for browsing + running models |
| **vLLM**      | serving system  | multi-user production; 14–24× HF transformers throughput via PagedAttention + continuous batching |
| **SGLang**    | serving system  | structured-gen + agentic flows; growing share |
| **exo**       | distributed serving | runs one model split across many home devices |

**For our project specifically:**
- The continuous-batching work (CB1–CB4) directly mirrors what vLLM does
  with PagedAttention. The user's `cb_scheduler.py` (Orca-style) is the same
  iteration-level admit/advance pattern. There's prior art worth reading in
  vLLM's scheduler if we hit edge cases (e.g. preemption, prefix sharing).
- Tenstorrent maintains a [`tt-inference-server`](https://github.com/tenstorrent/tt-inference-server)
  with vLLM integration shims (issue [#2491](https://github.com/tenstorrent/tt-inference-server/issues/2491)
  references `--max-num-batched-tokens` on TT hardware). That repo is the
  natural "where would this work upstream" target.
- For comparing the user's tok/s numbers to community references, the
  cleanest apples-to-apples is vLLM bf16 on H100/H200 at the same batch
  size — most numbers on /r/LocalLLaMA are GGUF Q4 on consumer GPUs and
  aren't directly comparable.

**Worth integrating / learning from:**
- **vLLM scheduler** — for CB3 edge cases (preemption, fair scheduling).
- **SGLang's RadixAttention** — prefix-tree KV sharing across requests; the
  next obvious win after continuous batching for multi-turn chat.
- **llama.cpp's GGUF format** — if we ever want to release the bringup as
  a drop-in for the home-LLM crowd, GGUF export is the lingua franca.

---

## Quick reference: HF URLs for top recommendations

- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `google/gemma-4-31b-it` (Gemma 4 31B-dense Instruct — verify exact slug at
  release time)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- `Qwen/Qwen2.5-VL-7B-Instruct` (smallest competent VLM)
- `HuggingFaceTB/SmolLM3-3B` (smallest fully-open daily-driver reasoner)
- `mistralai/Mistral-Medium-3.5` (128B dense, the "if we want a stretch dense" choice)

## Sources

- [DeepSeek-V3.2 paper, arxiv 2512.02556](https://arxiv.org/abs/2512.02556)
- [DeepSeek API change log](https://api-docs.deepseek.com/updates)
- [DeepSeek complete guide (BentoML)](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
- [DeepSeek roadmap, rumors, confirmed](https://chat-deep.ai/guide/deepseek-roadmap-rumors/)
- [DeepSeek-R1-Distill-Qwen-32B (HF)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Gemma releases (Google AI)](https://ai.google.dev/gemma/docs/releases)
- [Gemma 4 announcement (blog.google)](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 DeepMind page](https://deepmind.google/models/gemma/gemma-4/)
- [Gemma 4 guide (Codersera)](https://codersera.com/blog/gemma-4-complete-guide-2026/)
- [Mistral news index](https://mistral.ai/news/)
- [Mistral 3 announcement](https://mistral.ai/news/mistral-3/)
- [Mistral Small 4 announcement](https://mistral.ai/news/mistral-small-4/)
- [Mistral Small 3 announcement](https://mistral.ai/news/mistral-small-3/)
- [Mistral Small 3.1 announcement](https://mistral.ai/news/mistral-small-3-1/)
- [Mistral Small 3.2 review (ChatForest)](https://chatforest.com/reviews/mistral-small-3-2-24b-instruct-refinement-llm-review/)
- [Mistral Small 3.2 HF](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)
- [Mistral 2026 models guide (Serenities)](https://serenitiesai.com/articles/mistral-ai-models-2026-complete-guide)
- [Microsoft Phi (Azure)](https://azure.microsoft.com/en-us/products/phi)
- [PhiCookBook (GitHub)](https://github.com/microsoft/PhiCookBook)
- [Phi-4 HF](https://huggingface.co/microsoft/phi-4)
- [NVIDIA Nemotron 3 announcement](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models)
- [NVIDIA Nemotron 3 Nano Omni](https://blogs.nvidia.com/blog/nemotron-3-nano-omni-multimodal-ai-agents/)
- [NVIDIA Nemotron developer page](https://developer.nvidia.com/nemotron)
- [Z.ai GLM lineage 2026 (Presenc)](https://presenc.ai/research/zhipu-glm-model-lineage-2026)
- [GLM-V (GitHub)](https://github.com/zai-org/GLM-V)
- [GLM-5 (HF blog)](https://huggingface.co/blog/mlabonne/glm-5)
- [Kimi K2.6 announcement (Moonshot via DeepInfra)](https://deepinfra.com/blog/kimi-k2-6-model-overview)
- [Kimi K2.5 (SiliconANGLE)](https://siliconangle.com/2026/01/27/moonshot-ai-releases-open-source-kimi-k2-5-model-1t-parameters/)
- [Kimi K2.6 (CnTechPost)](https://cntechpost.com/2026/04/21/moonshot-releases-open-sources-latest-model-kimi-k2-6-2/)
- [Llama 4 multimodal launch (Meta AI)](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Qwen3 technical report, arxiv 2505.09388](https://arxiv.org/abs/2505.09388)
- [Qwen3 complete guide (InsiderLLM)](https://insiderllm.com/guides/qwen3-complete-guide/)
- [Qwen3-VL (GitHub)](https://github.com/QwenLM/Qwen3-VL)
- [Qwen2.5-VL-7B HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Best open-weight VLMs 2026 (Presenc)](https://presenc.ai/research/best-open-weight-vision-language-models-2026)
- [Best local coding models 2026 (PromptQuorum)](https://www.promptquorum.com/power-local-llm/best-local-coding-models-2026)
- [Best open-source self-hosted coding LLMs (Pinggy)](https://pinggy.io/blog/best_open_source_self_hosted_llms_for_coding/)
- [SmolLM3 announcement (HF blog)](https://huggingface.co/blog/smollm3)
- [SmolLM3-3B HF](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [Best open-source LLM May 2026 (Codersera)](https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/)
- [Best open-source SLMs 2026 (BentoML)](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [Open-source LLM models to run locally (HF blog)](https://huggingface.co/blog/daya-shankar/open-source-llm-models-to-run-locally)
- [Local LLM runtimes update May 2026 (Codersera)](https://codersera.com/blog/local-ai-runtimes-may-2026-update/)
- [Ollama vs LM Studio vs vLLM vs llama.cpp vs MLX 2026 (Codersera)](https://codersera.com/blog/ollama-vs-lm-studio-vs-vllm-vs-llama-cpp-vs-mlx-2026/)
- [LLM inference engines 2026 (BIZON)](https://bizon-tech.com/blog/best-llm-inference-engines)
- [Tenstorrent tt-inference-server vLLM issue #2491](https://github.com/tenstorrent/tt-inference-server/issues/2491)
- [Best local LLM monthly (StationX)](https://app.stationx.net/articles/best-local-llm)
