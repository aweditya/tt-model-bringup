# S2 chunked prefill — prior-art audit (2026-05-30)

Background: CB engine prefills 1 tok/step today → 200-tok prompt = ~52s TTFT at B=32.
We need real prefill. Two structural blockers were identified in
[`27b_chunked_prefill_plan.md`](27b_chunked_prefill_plan.md): multi-query paged
SDPA and shiftacc⇄per-position conv state. Before designing, audit what
Tenstorrent + vLLM mainline + fla-org have already shipped.

## Sources (cite-worthy, click-through)

- **TT vLLM fork**: https://github.com/tenstorrent/vllm (default `main`, dev `dev` @ `b4dd0fa695`, last push 2026-05-29)
  - Scheduler design: `plugins/vllm-tt-plugin/docs/SCHEDULING.md`
  - Scheduler code: `plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py`
  - Model runner: `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:251` (`assert enable_chunked_prefill is False`); `:1940-1997` (`submit_prefill` calls `model.prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos, …)`).
- **TT chunked SDPA op** (the missing primitive — turns out it exists):
  - Header: https://github.com/tenstorrent/tt-metal/blob/main/ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.hpp#L33 (scalar idx) and L45 (tensor idx, trace-safe).
  - Docs: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.transformer.chunked_scaled_dot_product_attention.html
  - Tracking issues: tt-metal#15876 (op), tt-metal#15873 (Llama chunked prefill, closed).
  - Existence-proof callers:
    - `models/tt_transformers/tt/attention.py:870` (`forward_prefill`), `:1039-1040` (`paged_fill_cache` for multi-user CB prefill), `:1072` / `:1083` (the two branches of chunked SDPA — tensor-idx for trace).
    - `models/tt_transformers/tt/generator.py` (chunked-prefill driver — `use_chunked_prefill = seq_len > max_prefill_chunk_size`).
    - `models/demos/llama3_70b_galaxy/tt/llama_attention.py`, `models/demos/t3000/llama2_70b/tt/llama_attention_optimized.py`, `models/demos/qwen25_vl/tt/vision_attention.py`, `models/demos/qwen3_vl/tt/attention.py`.
  - Unit tests: `tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py`, `test_sdpa_prefill.py`.
- **vLLM mainline Mamba2/Jamba chunked prefill**:
  - `vllm/model_executor/layers/mamba/mamba_mixer2.py` L35-38 (import `mamba_chunk_scan_combined_varlen`); L772-783 (split mixed batch into `[num_decode_tokens, num_prefill_tokens]`); L815-826 (varlen chunked SSD on prefill slice); L854-883 (SSM cache write at chunk-block boundaries).
- **vLLM mainline GatedDeltaNet (our model family)** — discovered late, key:
  - `vllm/model_executor/layers/mamba/gdn/{base.py, qwen_gdn_linear_attn.py}` — exposes `torch.ops.vllm.qwen_gdn_attention_core` with `_warmup_prefill_kernels`. Release v0.22.0 notes mention "KDA chunk-prefill exp2 semantics".
- **fla-org reference (the math)**:
  - https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py L44-108 — op order: `gdn_gate_chunk_cumsum → chunk_gated_delta_rule_fwd_intra (WY block-Neumann) → chunk_gated_delta_rule_fwd_h (cross-chunk state propagation) → chunk_fwd_o (output matmul)`.
- **TT Mamba demo (only TT-side recurrent prior art)**:
  - `models/demos/wormhole/mamba/README.md` — *"The prefill graph is not currently integrated into the demo. Therefore we currently process the prompt a single token at a time using the decode graph."* I.e. they have the same problem we have.
  - `models/demos/wormhole/mamba/tt/mamba_ssm.py:110-112` (`to_prefill()` swaps A/D weights), `:188-206` prefill branch (calls `ttnn.experimental.prefix_scan`).
  - `models/demos/wormhole/mamba/tt/mamba_conv.py` — stateless `ttnn.conv1d()` for prefill; no shift register; no CB integration.

## Per-blocker verdicts

| Blocker | Verdict | Lift |
|---|---|---|
| Multi-query paged SDPA (attn prefill) | **SOLVED upstream.** | `ttnn.transformer.chunked_scaled_dot_product_attention` — drop-in, trace-safe variant exists, validated by Llama/Qwen-VL prefill code. Constraints: `chunk_start_idx` multiple of `q_chunk_size`; no sliding-window (fine for Qwen3.6). |
| shiftacc ⇄ per-position conv state (GDN prefill) | **No TT prior art — we'd be first.** | None upstream. Closest reference is fla-org's `chunk_gated_delta_rule_fwd` op order. |
| Mixed prefill+decode batches | **TT vLLM intentionally rejects them.** | Adopt their pattern: alternate PREFILL-only and DECODE-only steps. |

## What this changes about the design

[`27b_chunked_prefill_plan.md`](27b_chunked_prefill_plan.md) framed the choice as
"path 1 = CB-native chunked prefill (big, two new kernels)" vs "path 2 = state
transplant (smaller, just a converter)". With these findings, path 2 sharpens:

**Recommended structure — single-seq chunked prefill + state transplant + TT-style alternating scheduler:**

1. **Scheduler** alternates PREFILL-only and DECODE-only steps (TT vLLM pattern). No mixed-batch admission code needed.
2. **PREFILL step** reuses our working S1a (chunked attention) + S1b (block-Neumann GDN) on a temp full state.
3. **S1a swaps internal primitive** to `ttnn.transformer.chunked_scaled_dot_product_attention` — gets paged KV "for free" so end-of-prefill KV is already in paged-block layout.
4. **State converter** at end of prefill: end-of-prefill kdim conv state → 3-col shiftacc slot state; end-of-prefill GDN H/K → slot-s GDN state slot.
5. **DECODE step** is the current CB engine unchanged.

This is the smallest correct thing. Zero new kernels for attention. No batched-GDN-prefill kernel needed. TTFT drops from `L × decode_step` to `(L/C) × prefill_step`.

## Optional future work (not for S2)

- True mixed-batch prefill+decode (vLLM mainline's mamba_mixer2 pattern). Needs a batched chunked-GDN kernel — fla-org's op order applied across CB slots. Bigger; only if alternating-step scheduler ends up bottlenecked.

## First concrete experiment (unchanged from prior plan)

Single-slot proof at small L (8–32 tokens):
- Prefill via S1a+S1b on temp state → transplant into slot 0 → 4 decode steps via CB engine.
- Reference: same model, same prompt, 1-tok-per-iter CB prefill all the way through.
- Gate: output token sequence matches exactly for the first 4 decode tokens; final state cos ≥ 0.999.
