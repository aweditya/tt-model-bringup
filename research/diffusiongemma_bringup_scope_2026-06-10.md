# DiffusionGemma 26B-A4B-IT — Tenstorrent Blackhole bringup feasibility (2026-06-10)

Owner framing: *"It's a diffusion model — fundamentally different from autoregressive — but
the core ops should be stuff we've already done. Worth researching the reusability
before scoping."* This document validates that claim and scopes the effort.

Sources at the bottom. All numbers from the HF/Google model card + the vLLM blog of
the same release day (2026-06-10) unless noted.

---

## 1. Model fact-sheet

DiffusionGemma is Google's first open-weights **discrete-diffusion language model**.
It is **not** a separately-trained encoder-decoder transformer in the T5 sense — it is
**one Gemma 4 MoE tower used in two attention modes that share weights**:

- **Encoder mode** — ordinary causal attention; runs twice per block (prefill of the
  prompt, then a "commit" pass that bakes a finished canvas into the KV cache).
- **Decoder mode** — bidirectional attention over a fixed-size canvas of token IDs;
  reads the encoder's KV cache, never writes to it. This is the denoising step.

| Field | Value | Source |
|---|---|---|
| Architecture class | `DiffusionGemmaForBlockDiffusion` | HF model card |
| Backbone | Gemma 4 26B-A4B MoE | Google blog |
| Total params | 25.2B | model card |
| Active params per token | 3.8B (~4B) | model card |
| `num_hidden_layers` | 30 (shared across both modes) | config.json |
| `hidden_size` | 2816 | config.json |
| `num_attention_heads` | 16 | config.json |
| `num_key_value_heads` | 8 (GQA, 2:1) | config.json |
| `head_dim` | 256 | config.json |
| `intermediate_size` (dense MLP / shared expert) | 2112 | config.json |
| `moe_intermediate_size` (per-expert) | 704 | config.json |
| MoE | 128 routed experts, top-8 + 1 shared | model card |
| Vocab | 262 144 | config.json |
| Max position embeddings | 262 144 | config.json |
| Sliding window | 1024 (same as Gemma 4 12B) | config.json |
| Layer types | mix of `sliding_attention` + `full_attention` | config.json |
| RoPE θ | 1e6 (global), 1e4 (sliding) | config.json |
| RMSNorm ε | 1e-6 | config.json |
| Canvas length | 256 | model card |
| Max denoising steps | 48 | model card |
| Sampler | entropy-bound (≤0.1), temperature linearly decayed 0.8 → 0.4 | model card |
| Adaptive stop | mean entropy < 0.005 AND stable top-1 | model card |
| Training corruption | uniform-random-token (no `[MASK]` absorbing state) | NeMo guide |
| Tokens / forward | 15–20 committed per pass | model card |
| Throughput claim | >1100 tok/s @ H100 FP8, ~4× vs AR Gemma 4 | model card / Google blog |
| License | Apache 2.0, ungated | HF card |
| HF support | `transformers >= 5.11.0` ships native modeling code | NeMo guide |
| vLLM support | first dLLM with native vLLM backend (2026-06-10) | vLLM blog |

What's different from autoregressive Gemma 4: there is **no per-token next-token
sampling**. Generation produces a 256-token "canvas" of full-vocab random initial
tokens, then runs up to 48 forward passes through the decoder. Each pass produces
logits at every canvas position; the sampler keeps the lowest-entropy positions as
"committed" tokens, renoises the rest, and repeats. When entropy stalls or 48 steps
elapse, the canvas is sent through the encoder once more to commit its KV, and the
next canvas begins.

---

## 2. Op inventory

| Op | Already in repo? | Location | Gap |
|---|---|---|---|
| Token embed | yes | `server_gemma4_unified_ttnn.py`, `server_35b_ttnn.py` | none |
| RMSNorm (ε=1e-6) | yes | `tt_rms_norm.py` mirror, plus our `_layer_pos0_*` paths | none |
| RoPE (dual-θ: 1e6 + 1e4) | yes | Gemma 4 12B server ships exactly this | none |
| GQA SDPA (nqh=16, nkv=8, dh=256, **causal**, sliding=1024) | yes (encoder mode) | `server_gemma4_unified_ttnn.py:2561,2630` `is_causal=True scale=1.0 sliding_window_size=1024` | encoder mode is byte-for-byte reusable |
| GQA SDPA (**bidirectional**, sliding=1024) | **NO** — never built | `ttnn.transformer.scaled_dot_product_attention` supports `is_causal=False` in upstream, never exercised by us | new contract; needs an isolation probe at our shapes |
| Paged fill cache (commit pass writes KV) | yes | Gemma 4 `paged_fill_cache` two-call workaround | reusable for the encoder "commit" pass |
| Paged SDPA decode (single token) | yes | `_layer_pos0_sliding_paged` | **not used** in DiffusionGemma — there is no single-token decode step |
| Cross-attention K/V read from a *different* cache than the current pass's queries | partial | We always read+write the same cache. Decoder must read encoder KV but not write. | minor refactor — gate the `paged_fill_cache` call |
| MoE Expert-Parallel routing (128 experts top-8 + 1 shared) | yes (35B has 128 experts top-8) | `server_35b_ttnn.py` + owned-kernel router | **shared expert is new**; 35B has no "always-on" shared expert path. ~1 day to add a parallel dense MLP and sum. |
| Host TopK on router logits | yes | DeepSeek-V3 pattern, `feedback_ttnn_topk_tie_break_drift` | none |
| Embed → unembed (tied or untied lm_head, 262144 vocab) | yes (vocab-sharded lm_head) | Gemma 4 12B P1 vocab-shard | reusable |
| Entropy-bound sampler (per-position entropy, temperature decay, partial commit) | **NO** | n/a | **host code, trivial**; <50 lines of NumPy after argmax → logsoftmax → entropy |
| Uniform-random renoise of un-committed positions | **NO** | n/a | host-side `np.random.randint(0, vocab)` mask write into the next canvas. Trivial. |
| Multimodal (vision encoder ~550M, video) | partial | We have Gemma 4 12B text-only; vision tower not bringup'd | **OUT OF SCOPE** for v0; text-only canvas is fully usable |

Six ops are byte-identical reuse from Gemma 4 12B. Two ops are new behaviour on
existing TTNN kernels (bidirectional SDPA, decoupled read/write cache). One op is a
small shared-expert addition. The sampler and renoise are pure host code.

---

## 3. Real reusability assessment — is the owner right?

**Yes, with two caveats.** The framing "core ops are stuff we've already done" is
accurate at the matmul/norm/RoPE/MoE layer. The actual delta is concentrated in two
places:

**Caveat 1 — bidirectional SDPA at canvas shape is a new kernel contract.**
Upstream `ttnn.transformer.scaled_dot_product_attention` documents `is_causal=False`
as supported, but **we have never exercised it** — every SDPA call in
`server_gemma4_unified_ttnn.py`, `server_tp.py`, `server_35b_ttnn.py` passes
`is_causal=True`. The shape is also unusual: Q is `[B=1, nqh=16, L_canvas=256, dh=256]`
and K/V come from a separate cache of length up to ~256K (full-attention layers) or
1024 (sliding layers). The full-attention case at long context with bidirectional
attention has never been benchmarked at our head dim. **Risk: medium.** An isolation
probe on day 2 of phase 1 settles it. If the upstream kernel hangs or has a PCC bug
at our shapes, the fallback is the Gemma 4 12B `_layer_pos0_sliding_paged` pattern
adapted to take a square attention mask — that's ~100 LOC.

**Caveat 2 — the inference *loop* is fundamentally new.** No reuse from any of our
four bringups; all of them run "advance one position, sample one token, append to
cache, loop". DiffusionGemma's loop is "fill canvas with random tokens, run decoder
forward, score entropy per position, commit low-entropy positions, renoise the rest,
goto step 1". This is ~150–200 LOC of host orchestration and an entropy-bound
sampler. It is straightforward but it has zero existing scaffolding in our repo —
`cb_scheduler` / `cb_engine` / `cb_api` are all single-token-step-shaped. **The CB
path is unusable; we'd ship an eager non-CB server first.**

The encoder mode is essentially "run our existing Gemma 4 12B forward, on a 26B-A4B
weight set, with 128 experts top-8 + 1 shared". That's the 35B MoE recipe with the
12B sliding-attention recipe glued together. We already have both pieces. The
attention shapes and kv-cache primitive are identical between 12B (our reference)
and 26B-A4B (target), and the head dim 256 matches. The MoE topology — 128
experts top-8 plus 1 shared expert — is one expert wider than 35B's top-8 and adds
a permanent dense MLP, both small additions.

**Honest verdict: ~5–7 weeks of dedicated effort** if the bidirectional SDPA probe
passes on week 1. Compare to:
- 27B (dense AR, copy-paste from existing patterns): ~6 weeks
- 12B Gemma 4 IT (dense AR + sliding window + scale=1 + v_norm + layer_scalar): ~36h
- 35B-A3B (MoE + EP routing, blessed primitives): ~4 weeks
- Nemotron-3 Nano (Mamba2 SSD + custom kernel): ~8 weeks (kernel work)

DiffusionGemma sits between 12B and 35B in op novelty, but the **inference loop is
new from scratch** and **CB is off the table for v0**. Net: more like 35B-class
effort.

---

## 4. Recommended phased plan

### Phase 0 — feasibility & code-read (1 week)
- Read upstream `modeling_diffusion_gemma.py` end-to-end (HF transformers
  `>=5.11.0`).
- Confirm config: 30 layers shared between encoder + decoder modes, sliding+full
  layer mix matches our Gemma 4 12B handling.
- Isolation probe: `ttnn.transformer.scaled_dot_product_attention(is_causal=False)`
  at Q=[1,16,256,256], K/V=[1,8,L,256] for L ∈ {1024, 8192, 32768}. PCC vs
  PyTorch reference on qb1. **Gate for phase 1: cos ≥ 0.999.**
- Weight introspect: confirm `safetensors` shard layout, dtype, presence of a
  single tied embed/lm_head, shared-expert weights.
- Numpy ground-truth: build a small reference of canvas[256] → logits[256, 262144]
  for one canvas init, one denoising step.

### Phase 1 — single denoising-step forward correctness (1–2 weeks)
- Fork `server_gemma4_unified_ttnn.py` → `server_diffusiongemma_ttnn.py`.
- Encoder mode = identity reuse of Gemma 4 12B prefill at 26B-A4B weights + 128 expert
  router + shared expert. Validate cos ≥ 0.999 per layer vs numpy oracle. The 35B
  shared-MoE-router path with `+1 shared expert` is the new addition.
- Decoder mode = encoder forward with `is_causal=False`, no `paged_fill_cache`,
  rank-3 input `[1, 256, 2816]`. The canvas is just a length-256 token sequence with
  a custom embedding lookup of "noise" tokens — no special preprocessing.
- Owned kernels: none new (we are NOT building an SDPA kernel; we are calling the
  upstream `is_causal=False` path).
- **Exit criterion:** one denoising step on real prompt + 256 random canvas tokens,
  PCC ≥ 0.999 vs HF transformers eager reference. Use the per-layer ladder.

### Phase 2 — full inference loop + sampler (1–2 weeks)
- Host code: implement the entropy-bound sampler (per-position log-softmax →
  entropy → keep ≤0.1, drop above; temperature schedule 0.8→0.4 over 48 steps;
  adaptive stop on mean entropy < 0.005 + stable argmax).
- Host code: uniform-random renoise of un-committed positions (`np.random.randint`
  over vocab; reserve no special tokens — the model was trained without an
  absorbing-state MASK token).
- Encoder "commit" pass: re-use the same forward but with paged-fill-cache write
  enabled, taking the now-committed canvas.
- Multi-canvas: loop until response token cap or stop sequence.
- Validate end-to-end vs HF `generate()` on three prompts of known answer.

### Phase 3 — HTTP wire-up + chat smoke (3–5 days)
- New server endpoint `server_diffusiongemma_ttnn.py` exposing `/v1/chat/completions`.
- **No CB.** Single-stream eager. Document this in the server module docstring.
- Smoke through `scripts/chat.py` to confirm parity with our other servers' UX.
- Streaming: emit committed positions in left-to-right order as they appear (vLLM
  does this; mimic).

**Total: 4–6 weeks elapsed time** assuming the bidirectional-SDPA probe passes.
Add 2 weeks if it fails and we have to write a fallback bidirectional kernel.

---

## 5. Risks + open questions

1. **Memory at long context.** Both the encoder *and* decoder forwards run on full
   sequence (encoder = past tokens via cache, decoder = full canvas + cross-attn to
   cache). Canvas is fixed at 256, so the decoder's intra-canvas attention is
   `O(256²)` per layer — small. But encoder KV cache is normal-shaped (`L × 8 × 256`
   bf16 per layer × 30 layers). At L=32k that's ~3.9 GiB KV alone, fits easily on
   (1,4) Blackhole; at L=256k (advertised max) it's ~31 GiB, will not fit a single
   chip but is fine TP=4. **Bidirectional decoder attention at canvas=256, kv=32k
   never benchmarked on TT** — see Phase 0 probe. Practical v0 cap: L=16k.
2. **Gating.** Apache 2.0, ungated. Greenlight.
3. **HF inference framework.** Native `transformers >= 5.11.0` ships
   `DiffusionGemmaForBlockDiffusion` and a custom `AutoProcessor`. vLLM
   `2026-06-10` blog post claims first-class dLLM support. We can use both as
   oracles. NVIDIA NeMo Automodel has a fine-tuning recipe — useful as a second
   reference implementation if HF modeling code is sparse.
4. **Continuous batching.** Off the table for v0. The denoising loop's per-canvas
   structure does not map onto `cb_engine`'s per-token step abstraction
   (`feedback_dev_harness_vs_cb_engine_gap`). A future CB v1 would need a "step =
   one denoising pass on N canvases of one batch each" abstraction.
5. **Determinism.** Per-step renoise uses random tokens; the sampler keeps the
   lowest-entropy positions, which are bf16-sensitive at boundaries. Expect
   `feedback_bf16_chain_drift_at_B_gt_1`-class non-determinism between runs. The
   model card already acknowledges adaptive stopping is non-deterministic. We
   should fix a seed in v0 for reproducibility tests.
6. **Shared expert + 128 routed experts.** Our 35B router supports 128 experts
   top-8 and `qwen36_*` owned kernels are batch=1 hard-asserted. Re-use is
   straightforward in eager but kernel hot-path needs revalidation at this expert
   count.
7. **Vision tower out of scope for v0.** ~550M params, deferred. Text-only canvas
   is the contract for chat.
8. **`canvas_length` is a hard constant 256.** All shapes are static — good for
   trace capture in a future v1, but trace is a phase-4 optimization.

---

## Sources

- [HF model card — google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- [Google AI model card — DiffusionGemma](https://ai.google.dev/gemma/docs/diffusiongemma/model_card)
- [Google AI overview — DiffusionGemma](https://ai.google.dev/gemma/docs/diffusiongemma)
- [Google Developers blog — DiffusionGemma: The Developer Guide](https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/)
- [Google Blog — DiffusionGemma: 4x faster text generation](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- [vLLM blog — DiffusionGemma: The First dLLM Natively Supported in vLLM (2026-06-10)](https://vllm-project.github.io/2026/06/10/diffusion-gemma)
- [HF transformers — model_doc/diffusion_gemma.md](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/diffusion_gemma.md)
- [NVIDIA NeMo Automodel — DiffusionGemma guide](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/dllm/diffusiongemma.md)
- [TTNN docs — scaled_dot_product_attention](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.transformer.scaled_dot_product_attention.html)
- [MarkTechPost — DiffusionGemma release](https://www.marktechpost.com/2026/06/10/google-ai-releases-diffusiongemma-a-26b-moe-open-model-using-text-diffusion-for-up-to-4x-faster-generation/)

Internal cross-references:

- `experiments/serve/server_gemma4_unified_ttnn.py:2561,2630` — current is_causal=True SDPA call site (reuse pattern for encoder mode).
- `experiments/serve/server_35b_ttnn.py` — MoE EP router precedent (128 experts top-8).
- `.cache/gemma4_branch_diff/` — Tenstorrent's `arg/gemma4_optimizations` Gemma 4 12B reference (no DiffusionGemma code present; confirmed via grep).
- `[feedback-cb-backend-dispatch-holes]` — DiffusionGemma will not ship behind `TT_BACKEND` because CB doesn't apply; document this in `cb_api.py` if we add it to the models registry.
