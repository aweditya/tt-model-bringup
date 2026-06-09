---
title: Precision choices and long-context correctness — Typecast strategy for Gemma 4 12B on Blackhole
date: 2026-06-08
scope: Decide whether we can drop the 38% TypecastDeviceOperation cost (fp32 accumulator → bf16 pack) without breaking long-context generation. Companion to the Tracy device-kernel profile of `server_gemma4_unified_ttnn.py`.
---

## Context

Tracy reports `TypecastDeviceOperation` at **38% of forward kernel time**. We currently run matmul with `fp32_dest_acc_en=True`; the packer typecast back to bf16 for downstream consumption is what shows up as Typecast. We need to know whether that 38% buys us actual long-context correctness, or whether it is a holdover. We were burned on 35B without `fp32_dest_acc` (per-layer cosine ladder collapsed), so we are cautious.

---

## 1. Does fp32_dest_acc actually matter at long context?

**Yes — but the mechanism is concrete, not vague.** BF16 has 7 mantissa bits; the dot-product reduction over a contraction dimension K accumulates a biased rounding error that grows roughly as O(sqrt(K)·eps_bf16). Two recent results put numbers on it:

- The 2026 vLLM blog on FP8 KV cache documents a Flash Attention 3 regression where "intermediate accumulation loses precision when the contraction dimension is large." A 128k needle-in-haystack went from **91% → 13%** until they implemented two-level accumulation, recovering to 89% ([vllm.ai/blog/2026-04-22-fp8-kvcache](https://vllm.ai/blog/2026-04-22-fp8-kvcache)). This is the exact failure mode of dropping the wide accumulator at long context.
- "Why Low-Precision Transformer Training Fails" (arxiv 2510.04212) shows BF16 dot products develop **biased** rounding error on the non-random distributions transformers produce, and that this is what blows up Flash Attention BF16 training. The same mechanism applies at inference for the K-dim of attention and the hidden-dim of MLP.

For matmuls where K is small (≤256 ish) the gap is invisible in cosine; for `o_proj` (K = num_heads · head_dim), `lm_head` (K = hidden), and **attention's S·V where K = sequence length**, the gap matters and gets worse with context. Our 35B per-layer cosine ladder collapse without `fp32_dest_acc` is the same story.

**Verdict:** `fp32_dest_acc` is not a holdover; it is real, but it is **K-dependent**. Cheap (small-K) matmuls don't need it.

---

## 2. What KV cache precision do production stacks use?

| Stack | Default | Lower-precision option | Stance on long context |
|---|---|---|---|
| vLLM | `auto` (matches model dtype, bf16/fp16) | `fp8_e4m3`, `fp8_e5m2` via `--kv-cache-dtype` ([docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)) | Cautious: 2026-04 blog admits long-context regressions on Hopper FA3 until two-level accum fix; e4m3 generally safer than e5m2; per-layer sensitivity varies ([vllm#22195](https://github.com/vllm-project/vllm/issues/22195) — Llama-3 middle layers more sensitive). |
| llama.cpp | f16 K and V | `q8_0`, `q4_0`, `q5_0`, plus experimental TurboQuant 3-4 bit | Perplexity within 1-2% of f16 on 35B+ for q8_0; at 128k context q4_0 degrades **34% generation tok/s** from dequant cost ([discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)). Common guidance: `k:q8_0 / v:f16` for general use, `k:q4_0 / v:q4_0` only when memory beats quality. |
| HF Transformers (eager + SDPA) | model dtype (bf16/fp16) | `cache_implementation="quantized"` via quanto/HQQ (int4 by KIVI paper) ([docs](https://huggingface.co/docs/transformers/main/en/kv_cache)) | Treated as opt-in; default keeps model dtype. No special long-context guidance. |
| DeepSeek-V3 official | bf16 (MLA latent compressed) | FP8 for weights/activations; **MLA latent kept in original precision** | Technical report (arxiv 2412.19437) lists embeddings, output head, MoE gate, norms, **attention operators** as kept in BF16/FP32 in their FP8 framework. KV is the MLA latent in original precision. |
| Llama 3 reference | bf16 (training) / fp16 (original inference) | Community FP8 deployments use TransformerEngine | Llama 3 paper (arxiv 2407.21783) keeps attention/softmax/norm in fp32 even when matmul runs in bf16/fp8. |
| Qwen 3 / Qwen2.5 | bf16 | Same as HF default | No bespoke KV precision stance. |

**Pattern:** every production stack defaults to **model dtype (bf16)** for KV and treats lower precision as a memory-vs-quality opt-in. None default to fp32 KV. FP8 KV is gaining traction in 2026 specifically because long-context inference is memory-bound, but the FA3 regression above shows it isn't free.

---

## 3. Attention sinks / long-context numerical issues — bf16 only, or also fp16/fp32?

**Both bf16 and fp16 hit it; fp32 doesn't.** Two distinct findings:

- StreamingLLM (arxiv 2309.17453) — attention sinks are an **architectural** property of softmax, present at any precision. The paper doesn't analyze precision specifically.
- "The Spike, the Sparse and the Sink" (arxiv 2603.05498) and Flash Attention low-precision analyses (arxiv 2510.04212) — massive activations and outlier-driven softmax overflow are precision-dependent. **FP16 overflows** (5-bit exponent → max ~65504), so `Q·K^T` can blow past it on outlier heads; bf16 has fp32's exponent range so it doesn't overflow, but it still **rounds the accumulator badly** at long K. fp32 has neither problem.

So our risk on Blackhole's bf16 path is mantissa-rounding error, not overflow. Keeping softmax in fp32 (which is the universal convention — Llama 3, HF SDPA, Flash Attention all do this) is sufficient defense against the overflow side. The mantissa-rounding side is exactly what `fp32_dest_acc` defends against.

---

## 4. FP8 inference literature on long-context drift

- NVIDIA TransformerEngine: on Hopper, FP8 matmul still **accumulates in fp32**, and quantization scales are recomputed per-tensor or per-row. The 2026 NVIDIA NVFP4 KV cache blog claims <1% accuracy loss on RULER 64K when KV is moved to NVFP4 on Blackwell.
- Llama 3.3 70B FP8 deployments (per NVIDIA) preserve MMLU near baseline; long-context wasn't the headline number.
- DeepSeek-V3 (arxiv 2412.19437): "the relative loss error of the FP8-training model remains consistently below 0.25%" vs BF16 baseline. Their precision recipe **explicitly keeps embeddings, output head, MoE gate, normalization, and attention operators in BF16/FP32**. This is the strongest signal: even the most aggressive production FP8 deployment carves out attention and norms.
- vLLM 2026-04 blog: 128k needle-in-haystack regressed 91% → 13% from a hardware-level accumulator-precision bug, fixed by two-level accumulation. **Long-context FP8 needs explicit attention from the kernel author.**

---

## 5. Is fp32_dest_acc a Tenstorrent quirk, or do GPUs do it?

It's universal, but the **cost structure differs**.

- NVIDIA tensor cores: BF16 inputs **only accumulate to FP32** (no BF16-accum option). FP16 inputs can pick FP16 or FP32 accum. The FP32→BF16 cast on output happens as a fused PTX instruction on the way to shared memory; it does **not** show up as a separate kernel ([NVIDIA forums](https://forums.developer.nvidia.com/t/numerics-of-tensor-core-instructions/244609)). Net cost ~0 vs ~2-3% throughput dip per Stosic's ECCV 2020 tutorial.
- Tenstorrent Tensix: per [tt-metal advanced docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html) and Corsix Part 7, FP32 accum doubles the Dst register footprint and the pack-back to bf16 is a **separate Tensix instruction issued by the packer**, which Tracy currently surfaces as `TypecastDeviceOperation`. That's the architectural difference — NVIDIA hides the cast inside the MMA epilogue; Tensix exposes it as a distinct op.

So our 38% Typecast is structurally inherent to the Tenstorrent FP32-accum path. NVIDIA's 0% on the same logical operation is the GPU folding it; we don't get that fold.

---

## 6. Recommendation for Gemma 4 12B on Blackhole

Ranking from safest+highest-leverage to riskiest:

**(b) Reduce matmul count via QKV concat-fuse + gate+up SwiGLU fuse — DO FIRST.**
Pure win. Halves the matmul count in attention's projections and MLP's gate/up, which directly halves the number of typecast events (one typecast per matmul output). No precision change, no long-context risk. Standard fusion every production stack already does ([fused QKV/gate-up references](https://github.com/ml-explore/mlx-lm/issues/956)). Expected: meaningful chunk of the 38% Typecast cost disappears because we issue fewer matmul→typecast pairs.

**(d) Turn off fp32_dest_acc on small-K matmuls only — DO SECOND, GATED.**
K-sensitivity argument from §1 gives a clean shortlist of safe candidates:
- Safe to disable: Q/K/V/gate/up projections where K = hidden_dim ≤ 4096. BF16 dot products of width 4096 are within tolerance; HF eager runs all of these in bf16 with no fp32 accum and passes RULER.
- Keep enabled: `o_proj` (K = num_heads · head_dim, can be large), `down_proj` (K = intermediate_size, up to 14336 for Gemma 4), `lm_head` (K = hidden, but feeds argmax/topk — magnitude matters), and **attention's S·V where K = sequence length** (this is the long-context blast radius).
- Validation: re-run the per-layer cosine ladder + 4K and 32K needle-in-haystack on a candidate change. We have all the infrastructure (`experiments/utils/needle_haystack_*`, [[reference-ruler-long-context-benchmark]]). Bisect by op, not by layer.

**(a) Confirm matmul output dtype is bfloat16 — VERIFY, NOT A CHANGE.**
Already done in our config but worth grepping. If any matmul is writing fp32 to L1 unintentionally we'd be paying double bandwidth on the downstream consumer. Low-cost check.

**(c) Keep selected chains (norm → matmul → norm) in fp32 — AVOID FOR NOW.**
Theoretically amortizes one typecast across multiple ops, but: (i) doubles L1 footprint for the chain, conflicting with our trace memory budget; (ii) tt-metal RMSNorm in fp32 has a known TRISC hang risk (per 35B precedent in MEMORY.md); (iii) we'd be reinventing what NVIDIA gets for free as an MMA epilogue, with extra plumbing. Defer until (b) and (d) are exhausted.

### Concrete next step (cheapest experiment)

1. Pick one matmul known to have small K (e.g. Q projection, K = 3840 in Gemma 4 12B). Flip `fp32_dest_acc_en=False` on **only that op**.
2. Run per-layer cosine ladder at pos 0 and pos 4 ([[reference-teacher-forced-ladder-method]]).
3. Run 4k + 32k needle ([[reference-ruler-long-context-benchmark]], with retry per [[feedback-35b-needle-haystack-2026-06-04]]).
4. Tracy-profile to confirm the Typecast slice for that op disappeared.
5. If clean: extend to all small-K projections in one commit. If cos < 0.999 at any L: revert that op and try the next K bucket.

This gives a measured tradeoff curve instead of a guess. Combined with (b) the realistic ceiling is reducing 38% Typecast to ~10-15% without touching the accuracy-critical ops; that's the 47ms → ~35ms ballpark we want.

---

## Sources

- [vLLM Quantized KV Cache docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM blog: State of FP8 KV-Cache and Attention Quantization (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [vLLM issue #22195: layer-sensitive KV precision](https://github.com/vllm-project/vllm/issues/22195)
- [llama.cpp discussion #20969: TurboQuant KV cache](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [HuggingFace KV cache strategies docs](https://huggingface.co/docs/transformers/main/en/kv_cache)
- [DeepSeek-V3 Technical Report (arxiv 2412.19437)](https://arxiv.org/pdf/2412.19437)
- [Llama 3 Herd of Models (arxiv 2407.21783)](https://arxiv.org/pdf/2407.21783)
- [StreamingLLM / attention sinks (arxiv 2309.17453)](https://arxiv.org/abs/2309.17453)
- [Why Low-Precision Transformer Training Fails (arxiv 2510.04212)](https://arxiv.org/pdf/2510.04212)
- [The Spike, the Sparse and the Sink (arxiv 2603.05498)](https://arxiv.org/pdf/2603.05498)
- [Tenstorrent FP32 Accuracy docs](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html)
- [Corsix tt-wh Part 7: MatMul](https://www.corsix.org/content/tt-wh-part7)
- [NVIDIA Tensor Core numerics forum](https://forums.developer.nvidia.com/t/numerics-of-tensor-core-instructions/244609)
- [NVIDIA NVFP4 KV cache blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
- [mlx-lm issue #956: fused gate/up](https://github.com/ml-explore/mlx-lm/issues/956)
