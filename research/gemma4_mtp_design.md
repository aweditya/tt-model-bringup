# Gemma 4 12B IT — MTP / speculative decoding design

Status: feasibility scoping. Research-only doc. Author: agent (session 2026-06-07).
Companion to `research/speculative_decoding_plan_2026-06-04.md` (which scoped the 35B + Qwen-3B *generic* spec-dec path); this doc is specifically about the **Google-shipped Gemma 4 MTP drafters**.

## 1. What is `google/gemma-4-E2B-it-assistant`?

WebFetch of the model card and the `config.json`:

> "This model card is for the Multi-Token Prediction (MTP) drafters for the Gemma 4 models. MTP is implemented by extending the base model with a smaller, faster draft model. When used in a Speculative Decoding pipeline, the draft model predicts several tokens ahead, which the target model then verifies in parallel."

It is therefore **not** a DeepSeek-V3-style MTP head bolted onto the target's residual stream (which would predict t+2, t+3, … using shared trunk activations). It is the **vanilla Leviathan small-draft-model speculative-decoding** pattern, packaged as a tiny independent autoregressive model that shares the Gemma 4 tokenizer (262144-vocab) and produces *one* draft token per forward, called K times to generate K candidates.

> "This results in significant decoding speedups (up to 3x) while guaranteeing the exact same quality as standard generation, making these checkpoints perfect for low-latency and on-device applications."

The transformers integration is the standard `target_model.generate(assistant_model=...)` path; no MTP head wiring required at the target's side.

### Naming / pairing

- The user provided **`google/gemma-4-E2B-it-assistant`** as a candidate. The model card text is generic ("the Gemma 4 models" / "the target model") and the HF code example pairs it with `google/gemma-4-E2B-it` (the 2.3B-effective sibling), **not 12B IT**.
- HF also ships **`google/gemma-4-12b-it-assistant`** — confirmed via WebFetch — 0.4B params, same `gemma4_assistant` architecture, Apache 2.0, ungated. *That* is the matched drafter for our production target `google/gemma-4-12B-it`.
- **Conclusion**: use `gemma-4-12b-it-assistant` (0.4B) as the drafter for our Gemma 4 12B IT path. The E2B-it-assistant is the wrong pair; using it as a drafter for 12B would tokenize-match but acceptance rate would be lower because the draft's training distribution doesn't match the 12B target's.

### `config.json` for the 12B assistant family (gleaned from the E2B-assistant config; 12B-assistant is the same arch class)

```
model_type      : gemma4_assistant
architectures   : ["Gemma4AssistantForCausalLM"]
hidden_size     :  256   (E2B-asst; 12B-asst likely larger — to confirm at bringup)
num_hidden_layers : 4
num_attention_heads : 4
num_key_value_heads : 1
head_dim        : 256
intermediate_size : 2048
vocab_size      : 262144   (== Gemma 4 12B IT)
sliding_window  : 512
layer_types     : ["sliding_attention" × 3, "full_attention"]
tie_word_embeddings : true
use_ordered_embeddings : true        ← unusual
num_centroids        : 2048          ← unusual: centroid-based embedding
centroid_intermediate_top_k : 32     ← unusual
backbone_hidden_size : 1536          ← unusual: two-stack design
```

**The four "unusual" fields are a real architectural delta.** This is not a vanilla 4-layer transformer. The drafter appears to factor its embedding/forward through a 2048-codebook + 32-way centroid topk projection from a 1536-dim backbone, then a 256-dim 4-layer trunk. That's a custom forward we'd have to reimplement in ttnn — non-trivial.

Param count: **78M for E2B-it-assistant, 0.4B (400M) for 12B-it-assistant**. Both Apache 2.0, both ungated.

## 2. Tenstorrent precedent for speculative decoding / MTP

`ssh qb1 'find ~/tenstorrent/tt-metal/models/demos -type d \( -iname "*spec*" -o -iname "*draft*" -o -iname "*mtp*" \)'`:

- `models/demos/deepseek_v3_b1/weights/specs/` (weights subdir, not code)

`grep -li "mtp\|spec" ~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/*.py`:

- **`tt/mtp.py` (419 lines)** — class `MTP2D(AbstractModule)`, "Second-token predictor for DeepSeek-R1 using the dedicated MTP layer weights." This is the DeepSeek-V3 MTP head pattern (predicts t+2 from the target's trunk hidden states), **not** the small-draft-model pattern.
- **`tt/generator.py` (2977 lines)** — contains the full MTP orchestration. Relevant identifiers:
  - `enable_mtp: bool` constructor flag
  - `_MtpPromptLayout`, `_MtpDecodeBootstrap`, `_MtpDecodeLoopResult(mtp_accept_rate, mtp_accepts, mtp_verifies)`
  - `_mtp_verify_trace_id`, `_mtp_predict_trace_id` — **two separate captured traces** (one for the K-batched target verify, one for the draft predict)
  - `_build_verify_alias_page_table_host` — page-table aliasing across `verify_offset` slots (KV cache trick for B=K+1 verify)
  - Hard constraint: `if enable_mtp and sample_on_device: raise SystemExit("MTP with sampling on device is not supported. Disable MTP or sample on host.")`

**This is the real precedent.** The pattern: capture two traces (predict at B=1, verify at B=K+1), use an aliased page-table to map verify's K+1 KV reads onto the same logical slot, sample on host, walk the K accepts, fall back to predict on the first reject.

The DeepSeek pattern is for an **MTP head attached to the target**, not for a separate small-draft model. But the *trace orchestration + accept/reject machinery* is reusable for Gemma 4 vanilla spec-dec: we just replace "predict trace = MTP head forward" with "predict trace = drafter model forward". The verify side is identical (target at B=K+1 with a K-batched single page-table slot alias).

There is **no other speculative-decoding demo** in `tt-metal/models/demos`. Gemma 4 spec-dec is greenfield in our tree, with DeepSeek-V3 MTP as the closest in-house pattern.

## 3. Architecture sketch — integration paths

### Path A (recommended): spec-dec wrapper around cb_engine, drafter as a second ttnn model

```
┌──────────────────────────────────────────────────────────────┐
│  cb_api.py        (one FastAPI process; chat endpoint)        │
│         │                                                      │
│         ▼                                                      │
│  spec_dec_scheduler  — extends cb_scheduler                    │
│         │                                                      │
│         ├── draft_step(slot, K) → K tokens from D (predict)    │
│         ├── target_verify(slot, K) → 1 fwd at B=K+1 on T       │
│         └── accept_or_correct → emit accepted + correction     │
├──────────────────────────────────────────────────────────────┤
│  T = server_gemma4_unified_ttnn  (sharded on (1,4) qb2)        │
│  D = server_gemma4_12b_assistant (sharded or single-chip)       │
│         both share one mesh_device handle                      │
└──────────────────────────────────────────────────────────────┘
```

- Fork `cb_scheduler.py` to a `spec_dec_scheduler.py` that wraps the existing
  `cb_scheduler` step interface. The wrapper is the accept/reject loop; the
  underlying scheduler still drives one model's forward.
- Add `DRAFT_BACKEND=gemma4_12b_assistant` env so the cb engine loads both T
  and D modules on bootstrap (per `[[multi-model-serving-plan]]` MM1).
- **Verify trace** = a new T capture at `B=K+1`. We already capture B=1 in
  `server_gemma4_unified_cb.py`; a second capture is bounded by `trace_region_size`
  (default 50 MB) — likely needs lifting to ~150 MB for two traces.
- **Aliased page-table**: fork `_build_verify_alias_page_table_host` from
  DeepSeek-V3 `tt/generator.py:43-101`. Maps K+1 logical batch rows onto
  the same physical KV slot so the verify forward reads the slot's history
  K+1 times in parallel.
- Sample on **host** (DeepSeek-V3 precedent's hard constraint; matches
  `[[feedback-ttnn-topk-tie-break-drift]]` finding that bf16 device topk is
  unstable — we want the accept walk to be deterministic).

### Path B (rejected): DeepSeek-style MTP head attached to the target

Would require:
1. Training (or finetuning) MTP heads onto the existing Gemma 4 12B IT weights — Google has NOT shipped MTP heads for the 12B IT model. The `-assistant` ckpt is a separate model, not a head.
2. New ttnn modules for the MTP layers.

Path B is moot because no MTP-head weights exist for Gemma 4 12B. Path A is the only path.

### Files touched (Path A)

| File | Touch | LOC est. |
|---|---|---|
| **NEW** `experiments/serve/server_gemma4_12b_assistant_ttnn.py` | drafter model, follows `[[reference-model-bringup-recipe]]` ladder | ~1500 (forks server_gemma4_unified_ttnn) |
| **NEW** `experiments/serve/server_gemma4_12b_assistant_cb.py` | CB shim around drafter | ~300 |
| **NEW** `experiments/serve/spec_dec_scheduler.py` | accept/reject loop, verify-page-table alias | ~500 |
| `experiments/serve/cb_api.py` | add `BACKENDS["gemma4_spec"] = (target, drafter)` | +20 |
| `experiments/serve/cb_scheduler.py` | route `TT_BACKEND=gemma4_spec` to `spec_dec_scheduler` | +30 |
| `experiments/serve/server_gemma4_unified_ttnn.py` | add a `_capture_verify_trace(B=K+1)` entry; expose alias page-table builder | +100 |
| **NEW** `experiments/cb/isolate/gemma4_assistant_smoke.py` | bringup probe (forks `[[reference-gm4-dev-harness]]`) | ~150 |

**Total new code: ~2500 lines, ~150 lines edits.**

## 4. Dev hour estimate

Hard items (cost-driving):

1. **Bringing up the drafter (`gemma4_12b_assistant`) on ttnn.** The drafter
   is NOT a vanilla transformer — it has `gemma4_assistant` model_type with
   `num_centroids=2048`, `centroid_intermediate_top_k=32`,
   `use_ordered_embeddings=true`, and a `backbone_hidden_size=1536` distinct
   from `hidden_size=256`. We have to read the HF modeling code to figure out
   the forward, then implement it in ttnn. Per `[[reference-model-bringup-recipe]]`
   the v0.1→v2 ladder for a known architecture is 1–2 days. For an
   *unknown* architecture (centroid embeddings are new) add 1 day. **~3 days.**

2. **Adding a B=K+1 verify trace to the existing 12B server.** The current
   server captures only B=1. Adding B=K+1 means re-allocating page-table
   buffers at the larger shape, re-running two-phase warmup
   (`[[ttnn-multi-trace-two-phase-warmup]]`), and validating that the B=K+1
   forward produces the K+1 logits we expect (vs HF reference). Reuse
   `experiments/cb/isolate/paged_sdpa.py` pattern. **~1 day** assuming no
   sliding-window+sliding-attention surprise at B>1.

3. **`spec_dec_scheduler` build.** Fork Leviathan Algorithm 1, port the
   DeepSeek-V3 `_build_verify_alias_page_table_host` + accept-walk logic.
   Test through dev-harness first. **~1.5 days.**

4. **End-to-end HTTP integration + perf measurement.** Wire DRAFT_BACKEND
   into cb_api, validate against `/v1/chat/completions`, measure tok/s
   at K∈{3,4,5} and α empirical. **~0.5 day.**

5. **Risk buffer** for chat-template alignment between drafter and target
   (centroid-embedding asymmetry could mean the drafter doesn't actually
   share the tokenizer cleanly), bf16 acceptance-rate weirdness, paged-cache
   alias bugs at K>3. **~1 day.**

**Total: ~7 days of focused build.** With current Gemma 4 12B perf at 47 ms/tok
traced and an honest acceptance rate of α≈0.7 (Google quotes "up to 3×" =
α≈0.85; realistic for matched-family drafter), the expected gain is **~1.7–2.2×
decode throughput** (similar to the 35B spec-dec analysis in
`speculative_decoding_plan_2026-06-04.md` §7 — ratio dominated by drafter
forward time, not drafter quality).

## 5. Risks (in priority order)

1. **Centroid-embedding architecture in the drafter** — reverse-engineering
   `Gemma4AssistantForCausalLM` from HF modeling code. If the centroid path
   is just an optimization (lookup table + projection) we can reproduce in
   ttnn cleanly; if it's a learned vector-quantizer with custom kernels, we
   may need to dequantize offline and ship the dequantized weights (loses
   the speed advantage but preserves correctness). **Probe at v0.0**: read
   `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
   (assuming it exists in transformers 5.7+) before committing to a ttnn impl.

2. **Tokenizer / chat-template alignment.** Both models claim vocab=262144,
   but the drafter has unusual special-token IDs in its config
   (`image_token_id=258880`, `audio_token_id=258881`). These should be
   identical to 12B IT's; verify by `cmp` of `tokenizer.json`. If they
   differ, draft proposals don't map cleanly to target's vocab and we need
   a remap step (~10 LOC).

3. **B=K+1 verify trace on Gemma 4 sliding+global attention pattern.** Gemma 4
   12B has the 5:1 sliding:global pattern with NKV_PER_CHIP_SLIDING=2,
   NUM_KV_HEADS_GLOBAL=1. The two-call paged SDPA workaround
   (`[[reference-gemma4-two-call-paged-decode]]`) was authored for B=1; we
   need to confirm it generalizes to B>1. Likely yes (the workaround
   addresses NKV_PER_CHIP>1, not B>1), but verify.

4. **bf16 acceptance-rate floor.** Spec-dec acceptance requires
   `argmax(T_logits[i]) == draft[i]`. Per `[[feedback-bf16-chain-drift-at-B-gt-1]]`
   and `35b_determinism_2026-06-04.md` §3, bf16 + 48-layer chain
   accumulates ULP noise that flips near-tie argmaxes. Less catastrophic at
   12B than 35B (fewer layers), but expect α to be ~5% lower than what an
   fp32 CPU run would measure. Mitigation: ship the A+B+D determinism
   patches (host-side deterministic argmax tie-break, `argmax(use_multicore=False)`,
   `fp32_dest_acc` on lm_head) on the target before measuring α. These are
   free wins regardless.

5. **Prefix-cache interaction.** Per `[[feedback-prefix-cache-multiturn-miss-2026-06-04]]`,
   our exact-prefix matcher fails on multi-turn because chat-template
   re-rendering of past assistant turns introduces separator bytes that
   don't match generated tokens. Spec-dec on top of prefix-caching needs
   the cache to hit the *target*'s KV; the *drafter*'s KV is independent
   and can be rebuilt from scratch each turn (it's tiny). Recommendation:
   in v0, run spec-dec WITHOUT prefix caching to isolate measurement;
   layer prefix-caching back on top of spec-dec in a separate iteration.

6. **KV cache duplication on the (1,4) mesh.** Drafter on a (1,4) mesh
   with NKV=1 + 4 layers + 8K context ≈ 64 MB total — negligible. The
   verify B=K+1 trace needs B=K+1 worth of page-table indirection but
   reuses the same physical slots (alias trick). No bulk cache duplication.

7. **Trace region size.** Two large traces (decode-B=1 + verify-B=K+1)
   plus the drafter's two traces (predict-B=1 + drafter-verify-B=K+1 if
   we go that deep). Likely fits in 100 MB but should pre-budget by
   running `trace_region_size=150*1024*1024` (3× headroom).

## 6. Order of operations (if greenlit)

1. **Determinism fixes A+B+D** from `35b_determinism_2026-06-04.md` ported
   to Gemma 4 12B. ~2 hours. Independent value. Measure trial-flip rate
   on the 12B path (should be much lower than 35B given fewer layers).
2. **Read `Gemma4AssistantForCausalLM` modeling code in transformers**, write
   a numpy oracle for the drafter forward. ~half day. Decide ttnn
   feasibility before committing to the bringup ladder.
3. **Bring up `server_gemma4_12b_assistant_ttnn.py`** following
   `[[reference-model-bringup-recipe]]`. v0.1 → v0.3 → v0.5 free-run greedy
   match HF on 100 tokens. ~3 days.
4. **Add B=K+1 verify trace to the 12B server.** Two-phase warmup; smoke
   that K+1 logits agree with K+1 independent B=1 forwards (per
   `[[feedback-deploy-serve-files-too]]`). ~1 day.
5. **Build `spec_dec_scheduler`** with greedy accept walk. Reference:
   Leviathan Algorithm 1 + DeepSeek-V3 `tt/generator.py:600-900` (the MTP
   decode-loop and trace orchestration). Bench in dev-harness first. ~1.5 days.
6. **HTTP wire-up via DRAFT_BACKEND env**. Bench `/v1/chat/completions`
   at K∈{3,5,7}, log accept_rate, measure tok/s. ~0.5 day.

**Total: ~7 days build + 1 day risk buffer = ~8 days.**

## 7. Decision frame

Compared to the 35B + Qwen-3B spec-dec plan in
`speculative_decoding_plan_2026-06-04.md`:

| | Gemma 4 12B + 12b-assistant | 35B + Qwen-3B (other plan) |
|---|---|---|
| Drafter availability | Apache 2.0, ungated, **ships from Google** | Need to bring up Qwen-3B from scratch |
| Drafter bringup | 3 days (unknown centroid arch) | 1–2 days (known Qwen3 arch) |
| Target bringup | DONE | DONE (35B) |
| Acceptance rate (est.) | **0.7–0.85** (matched family + officially trained) | 0.6–0.8 (cross-family approximation) |
| Speedup ceiling | "up to 3×" claimed; realistic 1.7–2.2× | 1.5–2.0× realistic |
| Production blockers | Determinism fixes, B>1 verify trace | Determinism, task #162 B>1, Qwen-3B bringup |
| Risk profile | Centroid-embedding reverse-engineering | DN-state rewind on rejected drafts |

The **Gemma 4 path has a higher realistic ceiling** (Google trained the
drafter on the target's exact distribution) and a **lower bringup risk profile**
(no DN-state rewind problem — Gemma 4 is dense, no recurrent state to undo).
The one open question is whether the `gemma4_assistant` centroid forward
is implementable in ttnn without custom kernels — that's a 4-hour HF-code
read-and-decide gate at v0.0.

## 8. Honest limits

- The model card text was read; the *exact* `12b-it-assistant`/`config.json`
  was not fetched (E2B-it-assistant config was used as a proxy for arch).
  Re-fetch before bringup; numeric shapes for 12B-asst will differ (likely
  hidden_size=512, num_hidden_layers=6, backbone_hidden_size=2304 or
  similar — speculation).
- `Gemma4AssistantForCausalLM` modeling code in transformers must be read
  to confirm the centroid forward; this hasn't been done in this session.
  If it doesn't exist in transformers' main branch, the model is
  uninstantiable today and we wait for an upstream release.
- Acceptance-rate estimates (0.7–0.85) are inferred from Google's "up to 3×"
  claim and matched-family precedents (Llama-3.2-1B drafter for Llama-3.1-70B
  measured ~0.78 in published vLLM benchmarks). The honest α is whatever a
  4-prompt benchmark shows on day 1 of measurement.
- The 47 ms/tok target baseline is from 2026-06-04; current numbers may have
  shifted (P1 vocab-shard +8% landed since). Re-baseline before claiming
  spec-dec speedup.

## 9. Related memory / files

- `research/speculative_decoding_plan_2026-06-04.md` — 35B + Qwen-3B variant
- `research/multi_model_serving_plan.md` — MM1 `TT_BACKEND` selector
- `research/model_bringup_recipe.md` — drafter bringup ladder
- `research/35b_determinism_2026-06-04.md` — A+B+D fixes (required prereq)
- `experiments/serve/server_gemma4_unified_ttnn.py` (2017 lines) — target
- `experiments/serve/server_gemma4_unified_cb.py` (528 lines) — target CB
- `~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/mtp.py` (419 lines)
  — class `MTP2D`, weight conversion for MTP head
- `~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/generator.py` (2977 lines)
  — `enable_mtp` flag, two captured traces (predict + verify),
  `_build_verify_alias_page_table_host` (lines 43–101), accept-walk
  (`_MtpDecodeLoopResult`)
- `[[reference-model-bringup-recipe]]` — v0.1→v2 ladder
- `[[reference-gm4-dev-harness]]` — dev harness pattern for fast iteration
- `[[ttnn-multi-trace-two-phase-warmup]]` — two-phase warmup for B=1 + B=K+1
- `[[feedback-ttnn-topk-tie-break-drift]]` — why sample on host (matches
  DeepSeek-V3's hard constraint)
- `[[reference-gemma4-two-call-paged-decode]]` — workaround if NKV_PER_CHIP>1
  generalizes to B>1
- Leviathan et al. 2023, *Fast Inference from Transformers via Speculative
  Decoding*, arXiv 2211.17192 — Algorithm 1
- Google MTP docs: https://ai.google.dev/gemma/docs/mtp/mtp
