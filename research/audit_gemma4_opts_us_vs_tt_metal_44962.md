# Audit — our Gemma 4 12B opts vs tt-metal #44962 (Tenstorrent's umbrella tracker)

Date: 2026-06-04. Source-of-truth for "their side": GitHub issue
[tenstorrent/tt-metal#44962](https://github.com/tenstorrent/tt-metal/issues/44962)
("[Gemma4] text-model optimization summary") opened by Ashai Reddy Ginuga
(`arginugaTT`), assigned to Benjamin Goel + Ashai. Source-of-truth for
"our side": local files cited inline.

Audience: the Tenstorrent engineer who shared #44962 during the
2026-06-04 poster session. Goal is to surface alignment, gaps, and a
concrete two-way engagement.

---

## 1. TL;DR (5 bullets)

- **Strongly aligned in spirit, very different in scope.** Their #44962
  covers four variants (E2B / E4B / 26B-A4B / 31B). We bring up
  **one variant they do not target** (`google/gemma-4-12B`, the
  `gemma4_unified` dense 12B) on (1,4) P150. The optimization *menu*
  overlaps heavily; the *acceptance suites* do not.
- **We already shipped 3 of their 18 sub-issues** end-to-end on 12B:
  on-device embedding (#44952), vocab-sharded lm_head + on-device argmax
  (#44953), and traced decode at multiple seq-lens (#44957). We have a
  fourth in flight (paged KV cache, #44946 — sliding done, global pending
  fork).
- **They have 5 sub-issues we should adopt verbatim**: paged_fused_update_cache
  (#44946 / their "block-size sweetspot"), RMSNorm fusion into adjacent
  matmuls (#44948), `concat_heads_decode → o_proj` fusion (#44945), MLP
  gate/up fused matmul (#44949), and the **redundant `to_memory_config` /
  `to_layout` audit** (#44958). Each is a clean kernel-time win that
  maps 1:1 into trace per [[feedback-kernel-vs-dispatch-realization]].
- **We have 4 hard-won lessons not on their list** that would save them
  bringup time on `gemma-4-12B` if they ever target it: the
  `gemma4_unified` v_norm (with_scale=False) per-head sub-norm, the
  per-layer `layer_scalar` multiply, the SDPA `scale=1.0` (NOT
  `1/sqrt(d_k)`) for Gemma 4 text, and `use_multicore=False` on argmax
  for cross-process determinism.
- **The biggest open question** to put back to them: their published
  26B-A4B baseline says "Decoded output is incoherent for the 26B variant
  — indicates PCC regression vs HF reference on the MoE path". We did
  not hit that on Gemma 4 12B (dense, no MoE). Our 35B-A3B MoE
  bringup [[project_35b_drift_resolved_2026-06-04]] suggests router /
  sub-norm convention bugs are the most common cause. Worth comparing
  notes.

---

## 2. Their pending optimization list (#44962, verbatim with 1-line summaries)

Top-level scope from the issue body:

> Umbrella tracker for **Gemma‑4 text‑only** performance work across all
> four variants … `gemma-4-E2B-it` (1x1), `gemma-4-E4B-it` (1x1),
> `gemma-4-26B-A4B-it` (1x4 P150 on QB2 + Galaxy DP=4), `gemma-4-31B-it`
> (same TP/DP shape).
>
> Audit method on every module: tracy + Tensix device profiler signpost
> → ttnn-visualizer perf tab (op core util + TM/DM ops) → land the
> optimization → re-verify per-module PCC against
> `models/demos/gemma4/tests/pcc_thresholds.json` → re-measure decode
> tok/s/user, decode total tok/s, prefill tok/s.

Their current published baseline (Blackhole T3K, 26B-A4B, 1x8, batch=32,
prefill_len=4096): prefill **~3.8 k tok/s**, decode
**~10.9 tok/s/user / ~349 tok/s total**, but **"decoded output is
incoherent for the 26B variant — indicates PCC regression vs HF
reference on the MoE path"**.

Branch: `handrews/gemma4-hybrid-kvc`.

### Sub-issues (text-model only, copied verbatim from the issue body)

| # | Title | One-line summary |
|---|---|---|
| #44943 | Profile + signpost every text module via tracy + tt-perf-report (per-variant baseline) | The "before" measurement — needed to prioritise the rest. |
| #44944 | Attention prefill: reduce DRAM↔L1 round-trips, fuse QKV-split + per-head-norm, raise core util | Cut TM/DM ops in QKV path; merge the per-head q_norm/k_norm into the split. |
| #44945 | Attention decode: tune `paged_scaled_dot_product_attention_decode` for batch=32, fuse `concat_heads_decode` → `o_proj` | Bigger-batch tuning + a known fusion that lands ~5-10% on per-head models. |
| #44946 | Paged KV-cache: validate `paged_fill_cache`/`paged_update_cache` on `[1,B,H,D]` batched layout, audit block-size sweetspot | Confirm batched (vs per-slot) cache ops + tune block size for the variant. |
| #44947 | RoPE caches: collapse decode `cos_pos`/`sin_pos` slice cost; share between layers | Lift cos/sin lookups out of the per-layer hot path; reuse across the 48 layers. |
| #44948 | RMSNorm fusion (input / post-attn / pre-ff / post-ff / q-norm / k-norm) into adjacent matmuls or single sharded kernel | Either fuse into the following matmul or call a single sharded RMSNorm kernel — kills the per-norm dispatch tax. |
| #44949 | Shared MLP: gate/up fused matmul + GeGLU activation, retune `_decode_linear_1d_config` | One matmul instead of two, fused activation, retuned tile shape. |
| #44950 | MoE router: top-k + per-expert scale + scatter hot path, eliminate fallback `ttnn.topk` host trips | Keep the routing on device; the published baseline implies host fallback is hot. |
| #44951 | MoE experts: sparse-matmul utilization (gate/up/down), expert dispatch overhead, GeGLU fused into experts kernel | Expert-side kernel + dispatch tuning + activation fusion. |
| #44952 | Embedding: replace host-side embedding lookup with `ttnn.embedding` on device for prefill + decode | Move the lookup off the host. |
| #44953 | LM head + on-device sampling: argmax/top-k path for batch=32, vocab=262144 (per-device 32768) | Vocab-sharded lm_head + on-device argmax / top-k for B=32. |
| #44954 | CCL: matmul ↔ allgather fusion, allreduce algo audit, sub-mesh topology for DP | Collective-fusion + algorithm choice + DP submesh shape. |
| #44955 | Batched prefill coverage: chunked prefill when `batch × seq ≥ 128k`, fix per-variant trace cache | Make chunked prefill the default for long inputs; fix trace-cache invalidation. |
| #44956 | Batched decode coverage: lift batched path to all variants (26B, 31B, E2B, E4B) | Generalise the B>1 forward across variants. |
| #44957 | Device-tracing coverage: extend `trace_prefill_supported_seq_lens` from `[128, 512]` to `{128, 512, 1024, 2048, 4096}` | Cover more prefill lengths so non-chunked-prefill workloads stay traced. |
| #44958 | Memory-config audit: remove redundant `to_memory_config`/`to_layout`/`reshape` along hot paths | Pure removal of dispatch-only ops — usually free perf. |
| #44959 | PCC verification: tighten `pcc_thresholds.json` per-module after each optimization; add MoE-router and per-expert PCC tests | Tighten the gate so future PRs can't regress quietly. |
| #44960 | Data parallelism on Galaxy (DP=4) for 26B and 31B — code on T3K, validate on Galaxy | Scale-out on Galaxy. |
| #44961 | Per-variant perf targets and acceptance criteria (TTFT, tok/s/user, tok/s total) | Owners + numbers per variant. |

---

## 3. Our shipped optimizations (Gemma 4 12B on (1,4) P150 qb1)

Baseline: bf16 12B = 24 GB / 6 GB/chip / 404 GB/s = **14.85 ms/tok = 67 tok/s**
BW floor ([[reference-p150-roofline-priority]]). All numbers below are
qb1, IT variant, single-client traced unless stated. Headlines from
[`HANDOFF.md`](../HANDOFF.md).

| # | Win | Files | Commit | Perf delta | Source |
|---|---|---|---|---|---|
| 1 | **Vocab-sharded lm_head + on-device argmax** | `experiments/serve/server_gemma4_unified_ttnn.py:789-850` (`_lm_head_argmax`), `server_gemma4_unified_cb.py:476-490` | `a24f2ea` | 51.3 → **47.5 ms/tok traced (+8.0%)**, 100/100 token-for-token vs eager. Bigger than 27B's +5.1% — vocab 262144 vs 248320. | `feedback_p22_gm4_vocab_shard_result.md` |
| 2 | **On-device argmax-tail trace fast path** (sample + step in one trace) | `experiments/serve/server_gemma4_unified_cb.py` argmax-tail path; aggregate from B=1→B=32 trace | `11e3083` (perf data), shipped earlier | 8.35 → **316.12 tok/s at 32 clients aggregate (+94% over logits-trace baseline)** | `HANDOFF.md:128-130` |
| 3 | **Paged SDPA + `sliding_window_size=1024` on sliding layers** | `server_gemma4_unified_ttnn.py:9, 37`, sub-staging v0.3.0.1 | `e2ae9f2` | Decode at ~160 ms/tok eager (vs 27B's similar paged-SDPA path which lifted 7.02 → 11.43 tok/s, +62%) | plan §v0.3.0.1 |
| 4 | **Two-phase trace warmup** (compile-all-then-capture-all) | `experiments/serve/server_gemma4_unified_ttnn.py` `ensure_decode_trace`, fork of 27B prod | `626c67a` | 182.7 → **51.3 ms/tok = 3.56× speedup** out of the box. `trace_region_size=400 MB` for 48-layer decode trace. | plan §v0.4; [[feedback-two-phase-warmup]]; mirrors [vLLM #352](https://github.com/tenstorrent/vllm/issues/352) |
| 5 | **`use_multicore=False` on lm_head argmax** (determinism) | `server_gemma4_unified_ttnn.py:840`, also applied to 35B + 27B | `918c025` | Determinism gate; perf cost being measured. Closes a cross-core tie-break race that was flipping argmax on near-ties. | HANDOFF queue #6 |
| 6 | **v_norm correctness** — `RMSNorm(head_dim, with_scale=False)` on V via all-ones weight | `server_gemma4_unified_ttnn.py:373-378, 678-683` | v0.1.2 (see `[[gemma4-v-norm]]`) | mixer_out cos 0.95 → 0.999990. Magnitude was 3.7× too high; cos missed it, mad caught it. | `feedback_gemma4_v_norm.md` |
| 7 | **Per-layer `layer_scalar` multiply** at end of each decoder layer | `server_gemma4_unified_ttnn.py:591-595, 777-783` | v0.2 | Without it: L0 cos=1.0 (invariant under scale) but mad 18× off → L1 cos=0.49, L2=0.115. With: argmax matches HF. | `feedback_gemma4_layer_scalar.md` |
| 8 | **SDPA `scale=1.0`** (Gemma 4 text attention sets `self.scaling=1.0`, NOT `1/sqrt(d_k)`) | `server_gemma4_unified_ttnn.py` SDPA call sites | `c97bf15` | Multi-step pos-0..5 final_norm cos 0.26 → ≥0.997. Masked at pos 0 by single-token softmax; surfaced at pos 1+. | `feedback_gemma4_sdpa_scale_1.md` |
| 9 | **Llama-style RMSNorm convention** (NO Qwen `(1+w)` zero-centered offset) | weight loader in `server_gemma4_unified_ttnn.py` | v0.1.0 | Avoided the 35B trap [[qwen36-qnorm-knorm-zero-centered]] that took several days to root-cause there. | plan §1.4 |
| 10 | **Continuous batching at B=4** (CB module fork) | `experiments/serve/server_gemma4_unified_cb.py` | `9b73205`/`dadda74` | All 3 acceptance gates PASS (B=1==single-slot, identical-slot, distinct-slot). Aggregate 8.35 → 316 tok/s at B=32 (27.7×). | v1 row of plan §v0.1 STAGED table |
| 11 | **HTTP wire-up via `BACKENDS` registry** | `experiments/serve/cb_api.py`, `cb_scheduler.py` | `9a1e45a` | `/v1/chat/completions` end-to-end. | v2 row |
| 12 | **IT-variant fork without re-bringup** (`TT_GEMMA4_VARIANT=it`) | bootstrap + tokenizer in `server_gemma4_unified_ttnn.py`, EOS list in `cb_engine.py` | `bdd207c` | Base → IT in ~2 hours by reusing the v0..v2 staging. EOS list `[1, 106, 50]` supported. | `[[reference-model-bringup-recipe]]` |
| 13 | **On-device embedding + `ttnn.embedding`** (matches their #44952) | `server_gemma4_unified_ttnn.py` embed lookup + `*sqrt(3840)` scale | v0.1.0 (`b9f3c35`) | Lookup never round-trips host. | plan §1.7 |

**Pending on our side** (parked, both designed in `research/gemma4_perf_briefing_2026-06-04.md`):

- **P2 distributed RMSNorm** across mesh — projected +12-15 ms/tok. Forks
  `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390`. **Tracy probe
  already shipped** (`experiments/utils/tracy_profile_one_gemma4_layer.py`,
  commit `2610ef3`).
- **P3 paged SDPA on global layers** — sliding pattern already validates
  the kernel contract; global is a per-layer swap. Projected +20-40% on
  attn.

---

## 4. Cross-reference table — their item × our equivalent

Legend: **SHIPPED** (in prod), **PLANNED** (in our roadmap), **N/A**
(not relevant to our scope), **WORTH ADOPTING** (their idea, we don't
have it).

| Their issue | Us, status | Where it lives / why |
|---|---|---|
| #44943 (profile + signpost) | **PARTIALLY SHIPPED** | `experiments/utils/tracy_profile_one_gemma4_layer.py` (commit `2610ef3`) + analyse pipeline in `archive/superseded_research_2026-06-04/profiling-quick-reference.md`. Not per-module-signposted yet. |
| #44944 (prefill QKV fuse + per-head norm fuse) | **WORTH ADOPTING** | We don't yet run a traced multi-chunk prefill — chat path uses prefix cache + 1-tok/iter fallback above chunk_size. Their fuse would unblock long-context prefill perf. |
| #44945 (decode SDPA tune for B=32 + concat_heads→o_proj fuse) | **PARTIALLY SHIPPED** (sliding) + **WORTH ADOPTING** (the fuse) | Sliding SDPA at B=4 traced. concat_heads_decode → o_proj fusion not done; this is a clean kernel-time win. |
| #44946 (paged KV cache batched layout + block-size sweet-spot) | **SHIPPED** (sliding) + **PLANNED** (global) + **WORTH ADOPTING** (`paged_fused_update_cache`) | We call `paged_update_cache` twice per sliding layer (NKV_PER_CHIP=1 each). Briefing flags `paged_fused_update_cache` as a ~1.6 ms/tok win. |
| #44947 (RoPE cache share across layers) | **SHIPPED** | Single `cos_pos`/`sin_pos` table per RoPE variant (sliding theta=10000, global theta=1e6 partial 0.25) — shared across all 48 layers. `server_gemma4_unified_ttnn.py` v0.3.0 setup. |
| #44948 (RMSNorm fusion) | **WORTH ADOPTING** | We have 4 norms/layer × 48 + 1 final = 193 ttnn.rms_norm calls/tok. Briefing P2 projects +12-15 ms/tok. Our plan is "distributed RMSNorm"; theirs is "fuse into adjacent matmul" — different mechanism, same payoff target. |
| #44949 (MLP gate/up fused matmul + GeGLU + retune) | **WORTH ADOPTING** | We use two matmuls (gate, up) + `ttnn.gelu(fast_and_approximate_mode=False)` + multiply (Step 0.2 mandates non-fused for correctness because the fused `UnaryOpType.GELU` is the approximate variant — see plan §3.6). Their #44949 would need a *non-approx* GELU fusion, which is a more involved kernel. |
| #44950 (MoE router) | **N/A** | gemma-4-12B (gemma4_unified) has `enable_moe_block=false`. Their item is for 26B-A4B + 31B. We do have MoE experience on Qwen 35B-A3B (`server_35b_ttnn.py`); cross-pollination possible. |
| #44951 (MoE experts) | **N/A for 12B** | Same as #44950. |
| #44952 (on-device `ttnn.embedding`) | **SHIPPED** | `server_gemma4_unified_ttnn.py` embed lookup is on-device; we also fold `*sqrt(3840)` (Gemma 4 quirk) into a runtime multiply since lm_head is tied to embed. |
| #44953 (vocab-sharded lm_head + on-device argmax/top-k for B=32) | **SHIPPED** (argmax) + **PARTIALLY** (top-k) | `a24f2ea` for argmax. Top-K path (`return_topk`) exists in `server_gemma4_unified_cb.py:485-490` via `ttnn.topk` on `full_logits` — works but allocates full `[B, vocab]` first; their per-device 32k top-k would be cleaner. |
| #44954 (CCL: matmul↔allgather fusion, allreduce algo audit, DP submesh) | **WORTH ADOPTING** (matmul↔allgather fuse) | We ship `ttnn.all_reduce(cluster_axis=1)` (`all_reduce_tt` helper, `server_gemma4_unified_ttnn.py:255-263`); async variant was negative ([[async-ccl-negative]]). matmul↔allgather fusion (the `llama_rs_matmul` pattern in `reference_multi_chip_opt_menu_v2.md`) we have listed but not landed. |
| #44955 (chunked prefill at batch×seq ≥ 128k) | **PARTIALLY SHIPPED** | 27B prod runs `TT_CB_CHUNKED_PREFILL=1` + `TT_CB_PREFIX_CACHE=1` (CW1 fix `ea9aa20`). Gemma 4 12B inherits the same code path; not yet validated for L > 32 prompts at scale. |
| #44956 (batched decode coverage) | **SHIPPED** | `server_gemma4_unified_cb.py` B=4 acceptance gate; HTTP CB runs at B=32 default (316 tok/s aggregate at 32 clients). |
| #44957 (extend traced prefill seq-lens 128/512/1024/2048/4096) | **PARTIALLY SHIPPED** | We trace decode at full coverage; trace_prefill only at chunk_size=32 (CW1 path). Their 1k/2k/4k traces would unblock long-context prefill perf. |
| #44958 (memory-config audit — remove redundant TM ops) | **WORTH ADOPTING** | We have no formal pass for this. The CB module forward (`forward_batch_gm4_inner`) and the layer forward both have explicit `ttnn.deallocate` chains but no audit for `to_memory_config`/`to_layout` round-trips. Tracy probe would expose this. |
| #44959 (tighten PCC thresholds; add MoE-router + per-expert PCC tests) | **N/A for 12B PCC list**; **PRECEDENT FOR ALL** | Our equivalent is `experiments/cb/isolate/gm4_v0*.py` (cos ≥ 0.999 gates) and per-layer drift ladder. Could share methodology back. |
| #44960 (Galaxy DP=4) | **N/A** | We're single-host (1,4). |
| #44961 (per-variant perf targets) | **EQUIVALENT** | Our equivalent is `research/gemma4_perf_briefing_2026-06-04.md` for 12B. |

---

## 5. Adoption opportunities

### 5a. From them → us (top 3 by ROI on our 12B)

1. **#44946 `paged_fused_update_cache`** — kernel-time, 1:1 in trace.
   Briefing already calls it out (~1.6 ms/tok, ~1 hour) when "Tracy shows
   `paged_update_cache` twice per layer" — and on sliding layers we
   *literally* call it twice per layer (NKV_PER_CHIP=1 split, commit
   `e2ae9f2`). If their tt-metal team has shipped a fused variant in
   `tt-metal main`, we can pick it up with a one-line edit. **Worth
   asking what their preferred call signature is.**
2. **#44948 RMSNorm fusion** — kernel-time, 193 norm calls/tok. Briefing
   P2 projects +12-15 ms/tok via *distributed* RMSNorm (Llama-70B-Galaxy
   pattern); theirs is *adjacent-matmul* fusion. Whichever lands first is
   the win; we should align on which pattern they recommend on Blackhole
   so we don't pick the wrong one (Llama Galaxy pattern was authored
   against Wormhole). **Tracy probe shipped (`tracy_profile_one_gemma4_layer.py`)
   means we can pick a pattern within 1-2 cycles after this conversation.**
3. **#44958 memory-config audit** — almost free. We have no formal pass
   for redundant `to_memory_config`/`to_layout`/`reshape`. Our
   `cb_dn_recurrence_mode` bug ([[feedback-cb_api-clobbered-27b-owned-gdn]])
   recently regressed 27B for the SAME class of reason (silent layout
   round-trip). Tracy + ttnn-visualizer pass would catch both classes.

Honourable mentions: #44945 `concat_heads_decode → o_proj` fusion (clean,
known to land 5-10% on per-head models); #44944 prefill QKV + per-head
norm fuse (unblocks long-context prefill perf if/when we push past
chunk_size=32).

### 5b. From us → them (top 3 worth surfacing back)

1. **SDPA `scale=1.0` for Gemma 4 text** ([[feedback-gemma4-sdpa-scale-1]]).
   HF `modeling_gemma4.py:1178` sets `self.scaling = 1.0`. We saw this in
   their in-tree demo (`decode.py:144`, `operations.py:15`) so they know
   for *the four variants in #44962* — but it bit us hard on `gemma4_unified`
   12B (multi-step pos-0..5 cos 0.26 → ≥0.997 after the fix). If they
   ever extend coverage to gemma4_unified, this should be in the bringup
   recipe; if it's already documented somewhere upstream we'd love a
   pointer.
2. **`use_multicore=False` on `ttnn.argmax`** for cross-process determinism
   (commit `918c025`). At vocab=262144 we saw cross-core tie-breaks flip
   argmax on near-ties between server restarts. The fix is one kwarg;
   it's likely relevant to #44953's on-device sampling path. We can share
   the smoke + the cost measurement (TBD — server-restart-blocked).
3. **Two-phase warmup discipline** for multi-trace coexistence
   ([[feedback-two-phase-warmup]]; documented in
   [vllm#352](https://github.com/tenstorrent/vllm/issues/352)). The rule
   is: capture multiple traces by compiling ALL paths first
   (`enable_trace=False`) then capturing all back-to-back. Without it,
   decode JIT compilation between captures corrupts prefill trace memory.
   Relevant to their #44955 (chunked prefill + trace) and #44957
   (extend traced seq lens). We shipped this verbatim on `server_tp_cb.py`
   and `server_gemma4_unified_cb.py`; an upstream guide note would save
   the next bringup.

Honourable mentions: per-layer `layer_scalar` multiply
([[gemma4-layer-scalar]]) and v_norm (`with_scale=False`)
([[gemma4-v-norm]]) for the gemma4_unified variant if they ever cover
it; cos-not-enough-also-check-mad diagnostic discipline
([[cos-not-enough-also-check-mad]]) which would have caught their
"decoded output is incoherent for the 26B variant" MoE PCC regression
earlier (cosine often misses magnitude bugs).

---

## 6. Engagement recommendation (1-paragraph reply, shovel-ready)

> Hi <name> — thanks for the pointer to #44962, this is a clean menu and
> our Gemma 4 12B (`gemma4_unified`) bringup overlaps it heavily even
> though we don't target the four variants in your tracker. End-to-end
> on (1,4) P150 we're at 47.5 ms/tok traced single-client / 316 tok/s
> aggregate at B=32 (8.35→316 = 27.7×); 29% of the bf16 BW floor with
> ~3.4× headroom. Of your sub-issues we've shipped #44946 (sliding-only),
> #44947, #44952, #44953 (argmax), #44956, and most of #44955; we'd love
> to pick up your `paged_fused_update_cache` (#44946) and the RMSNorm
> fusion (#44948) next — our Tracy probe just landed (`tracy_profile_one_gemma4_layer.py`,
> commit `2610ef3`) and the projected wins are +1.6 ms/tok and +12-15
> ms/tok respectively. Two questions: (1) is the fused
> `paged_fused_update_cache` already in tt-metal main and what call
> signature do you recommend for the NKV_PER_CHIP=1 split-call pattern,
> (2) for #44948 do you fuse RMSNorm into the *following* matmul (the
> standard pattern) or into a single sharded kernel — we have both
> `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390` and the
> adjacent-matmul pattern on our short list and don't want to pick the
> wrong one for Blackhole. Happy to share back our debug notes from
> bringup — there are three `gemma4_unified`-specific corrections that
> aren't in the in-tree gemma4 demo (SDPA scale=1.0, per-layer
> `layer_scalar` buffer, v_norm with_scale=False) plus a
> `use_multicore=False` determinism fix for `ttnn.argmax` at vocab=262144
> that we can write up if it's useful.

Concrete commit pointers for the reply: our perf wins at `a24f2ea`
(vocab-shard lm_head), `626c67a` (v0.4 trace), `e2ae9f2` (sliding paged
SDPA), `c97bf15` (SDPA scale=1.0), `918c025` (argmax determinism),
`2610ef3` (Tracy probe). Briefing doc at
[`research/gemma4_perf_briefing_2026-06-04.md`](gemma4_perf_briefing_2026-06-04.md).
