# DiffusionGemma 26B-A4B-IT — Phase 0 feasibility code-read (2026-06-10)

Inputs: scope doc `research/diffusiongemma_bringup_scope_2026-06-10.md`; HF transformers
`modeling_diffusion_gemma.py` + `generation_diffusion_gemma.py` (`main` branch, file
length 2562 lines + 1062 lines respectively); HF config.json of
`google/diffusiongemma-26B-A4B-it`; TT-Metal `ttnn/cpp/ttnn/operations/transformer/sdpa/`.

Outcome: **Phase 1 greenlit**. SDPA op accepts `is_causal=False` with an
attention mask at our shapes — no kernel work needed. The denoising loop is
pure host code on top of an encoder/decoder forward we mostly already have.

## 1. Config (verified vs HF `config.json`)

`text_config`: `num_hidden_layers=30`, `hidden_size=2816`, `num_attention_heads=16`,
`num_key_value_heads=8`, `head_dim=256`, `global_head_dim=512`,
`num_global_key_value_heads=2`, `intermediate_size=2112` (dense MLP / shared expert),
`moe_intermediate_size=704` (per-routed-expert), `num_experts=128`, `top_k_experts=8`,
`vocab_size=262144`, `sliding_window=1024`, `rms_norm_eps=1e-06`, `canvas_length=256`,
`tie_word_embeddings=true`, `final_logit_softcapping=30.0`,
`hidden_activation=gelu_pytorch_tanh`. Layer types: 30 entries — 5 sliding then
1 full × 5 cycles. RoPE: `sliding={theta=1e4, type=default}`,
`full={theta=1e6, type=proportional, partial_rotary_factor=0.25}`. Top-level:
`canvas_length=256`. Scope-doc claims of `mask_token_id` / `max_denoising_steps`
in config are wrong — both live in the generation config / sampler defaults
(scope numbers 48 steps + 0.8→0.4 temp + 0.1 entropy bound still correct;
sourced from the model card, not config.json). **`use_bidirectional_attention="vision"`**
gates the text decoder to bidirectional via the `DiffusionGemmaDecoderTextAttention`
class which hard-sets `self.is_causal = False`.

## 2. Op-by-op mapping (the heart of this doc)

All file:line refs are in our `experiments/serve/server_gemma4_unified_ttnn.py`
unless noted. Status: **R** = byte-identical reuse, **A** = adapt (config swap or
small refactor), **N** = new code.

| DiffusionGemma op (HF class : line) | Our equivalent (file:line) | Status |
|---|---|---|
| token embed + `sqrt(hidden)` scale | `_embed_and_lookup_rope_seq` 2471-2506 (`ttnn.embedding` 2482 + scale 2486) | R (vocab + hidden differ — alloc-time only) |
| `DiffusionGemmaRMSNorm` (eps=1e-6, Gemma `(1+w)` form) | `ttnn.rms_norm(…, epsilon=EPS)` throughout; e.g. 991, 2523-2525 | R |
| `DiffusionGemmaTextRotaryEmbedding`, dual-theta + partial-rotary | sliding+global `cos/sin` build 893-918; `_apply_full_rope_seq` 2456-2468 | R |
| q_norm/k_norm/v_norm (per-head RMSNorm; v_norm `with_scale=False`) | sliding: 2523-2525; global: 2593-2595; `state.ones_head_dim_sliding` for v_norm | R |
| `DiffusionGemmaEncoderTextAttention` GQA SDPA, **is_causal=True** | sliding-prefill: `_layer_prefill_sliding` 2509-2577 (2-call split + `sliding_window_size=SLIDING_WINDOW`); global-prefill: `_layer_prefill_global` 2580-2643 | R (encoder mode == our prefill, byte-identical) |
| `DiffusionGemmaDecoderTextAttention` GQA SDPA, **is_causal=False, bidirectional** with mask | NEW: same body as `_layer_prefill_*`, swap `is_causal=False`, pass an `attn_mask` (canvas-canvas square + canvas-to-encoder-KV strip), DO NOT call `paged_fill_cache` | A (fork `_layer_prefill_sliding/global`, ~120 LOC each) |
| `paged_fill_cache` (encoder commit pass, writes K/V) | 2556-2557 (sliding), 2616-2621 (global) | R |
| Decoder reads encoder KV from cache, does NOT write | NEW: fork the prefill helpers, delete `paged_fill_cache` calls, pass cache `kc/vc` directly as K/V (whole prefix length) | A (~30 LOC delta) |
| MoE router (softmax → top-8 of 128) | host-topk DeepSeek pattern; router weight upload at `server_35b_ttnn.py:230-275` (`upload_moe_layer`); router matmul + host topk shipped (35B has 256-expert/top-8; we strip to 128) | A (config swap) |
| 128 routed experts × `moe_intermediate_size=704` | `upload_moe_layer` / `upload_moe_layer_pattern_a` `server_35b_ttnn.py:230-360`; Pattern A `experts_gate_up_local` + `experts_down_local` | A (E=128, INTER=704; vs 35B E=256, INTER=512 — shape swap) |
| **shared expert** (always-on dense MLP, `intermediate_size=2112`) | `_moe_shared_expert` `server_35b_ttnn.py:1153-1168` + `shared_gate / shared_up / shared_down / shared_expert_gate` weight upload (262-272) | R (35B already has it — scope doc's "new in DiffusionGemma" claim is wrong; just a `intermediate_size` swap) |
| 4 norms / decoder layer + `layer_scalar` per-layer scalar buf | `input/post_attention/pre_feedforward/post_feedforward_layernorm` upload 941-944; `layer_scalar` scalar load 956 + multiply at end of layer | R |
| Final RMSNorm + tied lm_head + `30·tanh(x/30)` softcap + argmax | `lm_head_tt` upload 732 (vocab-sharded dim=1); softcap+all-gather+argmax pattern at `server_35b_ttnn.py:1151-1180` | R |
| `gelu_pytorch_tanh` activation in MLP/expert | `ttnn.gelu(fast_and_approximate_mode=False)` per file docstring line 17-20 | R |
| Entropy-bound sampler (per-pos H, temp 0.8→0.4, accept low-H) | NEW host code — `EntropyBoundSampler.accept_canvas` `generation_diffusion_gemma.py:556-575`, `LinearTemperatureScheduleLogitsProcessor.__call__` 692-709 | N (~80 LOC NumPy) |
| Renoise un-committed positions (uniform `randint(0, vocab)`) | NEW host code — `EntropyBoundSampler.renoise_canvas` `generation_diffusion_gemma.py:577-593` | N (~30 LOC) |
| Adaptive stop (stable argmax + mean H < 0.005) | NEW host code — `StableAndConfidentStoppingCriteria.__call__` 625-659 | N (~40 LOC) |
| Outer canvas loop + encoder commit-pass invocation | NEW host orchestration — `DiffusionGemmaGenerationMixin.generate()` 677-1062 outer loop | N (~150 LOC) |

Net new TT-side code: **two helpers (`_layer_decoder_sliding`, `_layer_decoder_global`)
of ~120 LOC each**, both forks of existing `_layer_prefill_*`. All other model ops
already ship.

## 3. Bidirectional SDPA op-fitness — VERDICT: FITS

Reading `sdpa.cpp` lines 21-77 (entry point) and `sdpa_device_operation.cpp`
lines 19-358 (validation). Op signature accepts `is_causal=False` with an
optional `attn_mask` tensor — that's the **regular mode** branch (`validate_regular_mode`,
lines 56-159). Critical TT_FATALs we'd hit at our shapes:

- L57-61: `is_causal && attn_mask` mutually exclusive — we use `is_causal=False` + mask, **OK**.
- L78-81: mask must be **DRAM**, BFLOAT16/8/4, TILE — easy.
- L86-94: mask shape `[1|B, 1|NQH, Sq, Sk]` — for us `[1, 1, 256, 256]` (canvas-canvas)
  or `[1, 1, 256, L_enc]` (canvas → encoder KV). Sq must equal Q[2]=256, Sk must equal
  K[2]=L_enc, **OK**.
- L107-110: `Sq == Sk` only enforced **when `is_causal=True`** — explicitly NOT for
  bidirectional. **OK** at canvas=256, kv_enc up to 32k.
- L138-142: `nqh >= nkv && nqh % nkv == 0` — Q[1]=16, K[1]=8, ratio 2, **OK**.
- L148-157: `q_chunk_size`/`k_chunk_size` must be % 32 — pick `q=256, k=512`, **OK**.
- L355-357: padding only on seq dim; batch/heads/head_dim must equal padded — fine.

**No TT_FATAL blocks us.** GQA in non-causal mode is supported on the same code
path as causal — the assertion is independent of `is_causal`. This matches
`[[ttnn-prefill-sdpa-gqa-native]]` extended to non-causal.

Two practical caveats:
1. The "decode-mirror 2-call SDPA split" we use in `_layer_prefill_sliding`
   (lines 2546-2565, comment 2447-2453) was because `paged_fill_cache` requires
   one KV-head per call — that constraint does NOT apply to plain SDPA. In
   decoder mode (no fill_cache) we can issue a **single** non-paged SDPA call per layer.
2. `sliding_window_size` kwarg works in both causal and non-causal modes (passed
   straight through `prim::sdpa`, line 69 sdpa.cpp). For sliding layers, the
   decoder still gets a 1024-window bidirectional pattern for free.

**Probe (week 1):** Q `[1,16,256,256]`, K/V `[1,8,L,256]` for L ∈ {1024, 8192, 32768},
`is_causal=False`, scale=1.0, full-ones mask. PCC vs PyTorch `F.scaled_dot_product_attention(is_causal=False)`. **Gate:** cos ≥ 0.999.

## 4. Denoising loop schedule (HF `generation_diffusion_gemma.py`)

- **Canvas init:** `torch.randint(low=0, high=vocab_size, size=(B,256))`
  (`EntropyBoundSampler.initialize_canvas` 926-932). **No MASK token.** Trained on
  uniform-random corruption, not absorbing-state mask.
- **Per step:** encoder is run ONCE per canvas (746-756); inner loop iterates
  steps `N..1` (784-813); `_denoising_step` (883-948) runs decoder forward → logits
  → temperature divide → entropy → accept low-entropy positions → renoise rest →
  next step uses output logits for self-conditioning.
- **Temperature:** `t = t_min + (t_max - t_min) * (cur_step / max_steps)`
  (692-709) — linear 0.8→0.4 over `max_denoising_steps` (default 48 per model card).
- **Acceptance:** sort positions by entropy ascending; accept top-k where
  `cumulative_H - max_H ≤ entropy_bound (0.1)` (556-575). Rejected positions
  re-randomised via `torch.where` against a fresh `randint` canvas (577-593).
- **Stop:** `stable AND confident` — argmax canvas identical across last K steps
  AND mean per-position H < `confidence_threshold (0.005)` (625-659). Hard cap 48 steps.
- **Outer loop:** after canvas converges, run encoder once more on the
  finalized canvas to write its KV into cache, then start the next canvas.

## 5. Memory budget @ L=16k

KV per layer (sliding, NKV=8, HD=256, bf16): 8 × 16384 × 256 × 2 = 64 MiB.
Global layers (NKV=2, HD=512, bf16): 2 × 16384 × 512 × 2 = 32 MiB.
Layout: 25 sliding + 5 global. Total KV: 25·64 + 5·32 = **1760 MiB ≈ 1.72 GiB**.
On (1,4) Blackhole TP=4 with NKV/4: 440 MiB/chip — easily fits. At advertised
L=256k: 27.5 GiB total / 6.9 GiB/chip — still fits. Weights at 25.2B bf16
≈ 50.4 GiB / 4 = 12.6 GiB/chip; routed-expert MoE shard like 35B. Practical
v0 cap: **L=16k** (matches scope doc §5.1).

## 6. Refined Phase 1 plan (5-7 sub-stages, each gated)

- **v0.0** — numpy oracle: build `experiments/utils/hf_oracle_diffusiongemma.py`
  forking `hf_oracle_gemma4_assistant.py`. Capture: one encoder forward at L=64
  (per-layer h ladder) + one decoder forward at canvas=256 with L_enc=64
  (per-layer h ladder) + final logits[256, 262144]. **Gate:** numpy file
  round-trip vs HF eager cos ≥ 0.9999.
- **v0.1.0** — bootstrap + L0 single token: fork `server_gemma4_unified_ttnn.py`
  → `server_diffusiongemma_ttnn.py`. Strip dense MLP, add 35B MoE upload
  (E=128, INTER=704, +shared expert). Reuse mesh/embed/layernorm path
  verbatim. **Gate:** L0 `input_layernorm` cos ≥ 0.9999 vs oracle.
- **v0.1.1** — single layer encoder mode (causal): fork
  `_layer_prefill_sliding/global` unchanged, swap weight upload for 128-expert
  MoE + shared. **Gate:** L0 encoder forward cos ≥ 0.999 at L=64.
- **v0.1.2** — single layer decoder mode (bidirectional): NEW helpers
  `_layer_decoder_sliding/global`. Pass mask `[1,1,256,256+L_enc]`,
  `is_causal=False`. **Gate:** L0 decoder forward cos ≥ 0.999 vs oracle.
  This is also the SDPA op-fitness gate.
- **v0.2** — full 30-layer forward both modes. Per-layer ladder
  (`feedback_teacher_forced_ladder_method`). **Gate:** every layer cos ≥
  0.997; pos-0 argmax-match vs HF on 3 prompts.
- **v0.3** — single denoising step end-to-end: encoder once + decoder once +
  logits[256, 262144]. **Gate:** logits cos ≥ 0.999 vs HF oracle one-step.
- **v0.4** — host-side denoising loop: entropy sampler + temperature
  schedule + renoise + adaptive stop. **Gate:** HF `generate()` parity on
  3 prompts with fixed seed; entropy-trace match.

## 7. Top 3 risks + open questions

1. **`sliding_window` semantics under bidirectional attention.** SDPA op
   passes `sliding_window_size` through to the kernel regardless of
   `is_causal`. Need to confirm the window is symmetric (±512 around each
   query) and not "previous 1024" — open question for the probe. Fallback:
   pass a hand-built `attn_mask` of shape `[1,1,256,L_enc]` and skip
   `sliding_window_size`; mild perf cost.
2. **MoE owned-kernels assume B=1, token-by-token.** Our `qwen36_*` owned
   kernels are hard-asserted batch=1 (see CLAUDE.md §5). The decoder runs
   on canvas=256 — that's L=256 prefill-style MoE, not B=256 decode. The
   35B prefill MoE path on these kernels has never been exercised at L>32.
   Need to verify in v0.1.2 — if it hangs, fall back to the per-token MoE
   loop (slow but correct).
3. **HF reference availability.** `transformers >= 5.11.0` ships
   `modeling_diffusion_gemma.py` (verified raw fetch). qb1 venv currently
   pins an older version — need `scripts/setup_venv_diffusiongemma.sh`
   mirroring the Gemma 4 oracle venv pattern (CLAUDE.md note line 56-57).

Open questions: (a) how does HF handle the encoder-commit pass at the
boundary between canvases — single `forward()` with `decoder_input_ids=None`?
(b) is the lm_head softcap applied during the denoising loop or only at the
final commit? Both resolvable in v0.0 oracle build.

## Sources

- HF model card: <https://huggingface.co/google/diffusiongemma-26B-A4B-it>
- HF config.json (raw): <https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/config.json>
- HF modeling: <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py>
- HF generation: <https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py>
- TT-Metal SDPA entry: `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp:21-77`
- TT-Metal SDPA validation: `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp:19-358`
- Our Gemma 4 server SDPA call sites: `experiments/serve/server_gemma4_unified_ttnn.py:2559-2563, 2628-2632`
- Our 35B shared-expert path: `experiments/serve/server_35b_ttnn.py:1153-1168`
- Our 35B MoE upload (128 expert template at E=256): `experiments/serve/server_35b_ttnn.py:230-275`
- Our final lm_head + softcap: `experiments/serve/server_35b_ttnn.py:1151-1180`
- Scope doc: `research/diffusiongemma_bringup_scope_2026-06-10.md`
