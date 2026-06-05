# Audit — our Qwen3.6 work vs Tenstorrent `qwen9b-p150` branch

Date: 2026-06-04. Scope: read-only comparison. Read first:
[`HANDOFF.md`](../HANDOFF.md) for our headline perf,
[`research/27b_prefix_caching_plan.md`](27b_prefix_caching_plan.md) for our PC
design, [`archive/superseded_research_2026-06-04/27b_chunked_prefill_prior_art.md`](../archive/superseded_research_2026-06-04/27b_chunked_prefill_prior_art.md)
for the upstream chunked-SDPA audit (archived 2026-06-04). Their branch:
https://github.com/tenstorrent/tt-metal/tree/qwen9b-p150 (branch HEAD
`14be5b9` as of fetch).

A Tenstorrent engineer (2026-06-04 poster-session attendee) shared the
branch as their WIP supporting "Qwen3.5/3.6 with 256K context + all
other features". This doc cross-checks that claim against what's
actually on the branch and against our shipped 27B / 35B-A3B work.

---

## 1 — TL;DR (5 bullets)

1. **Their headline model is Qwen3.5-9B** (`models/demos/blackhole/qwen3_5_9b/`).
   The branch ships a full hybrid attention + Gated DeltaNet stack with
   masked-fixed-bucket prefill, chunk-outer trace prefill, vLLM hooks,
   and Qwen27B "integration" landing on `2026-06-04` (`e61d05c`). It
   targets one host (P150 single-node).
2. **The "256K" claim is closer to 128K in code.** Their `model_config`
   computes `max_seq_len = MAX_NUM_BLOCKS(2048) * BLOCK_SIZE(64) =
   131,072 tokens` (per `text_demo.py` extract). Long-prompt JSON
   fixtures top out at `input_data_long_64k.json`. No 256K-specific
   code path is on the branch.
3. **Where we're ahead**: continuous-batching scheduler that's
   production-live, slot-level prefix caching with measured 5.1×/8.0×
   per-token TTFT speedup at turn 2/3, an OpenAI HTTP server, a
   universal active-prompt-suffix stripper that works across Qwen3.6,
   Gemma 4 IT, and any future template, AND a working 35B-A3B MoE
   bringup their branch does not have.
4. **Where they're ahead**: an actually-implemented end-to-end
   chunk-outer traced prefill at 2048-token chunk size with a 5-bucket
   (128/256/512/1024/2048) masked fixed-bucket fallback for short
   prompts and a chunked-trace tail. Today we run S2 chunked prefill at
   `PREFILL_CHUNK_SIZE=32` and fall back to 1-tok/iter for `L >
   chunk_size`. **That's the largest cap on our chat TTFT for cold
   prompts.**
5. **Surprise**: their decode-time GDN does NOT use the
   `qwen36_gdn_decode_owned` ttnn-experimental kernel we built and ship.
   They use a different composed op named
   `recurrent_gated_delta_rule_decode_ttnn` that internally fuses "L2
   norm + scale + delta step". The two kernels are independent
   implementations of the same recurrence; comparing their kernel-time
   numbers would be a high-value experiment for both teams.

---

## 2 — What `qwen9b-p150` is

**Primary model**: Qwen3.5-9B dense + hybrid attention/DeltaNet, 32 layers
(8 full-attention + 24 DeltaNet per `models/demos/blackhole/qwen3_5_9b/tt/model.py`).
Target: Blackhole P150 single-node, mesh up to 4 chips (`P150x4`).

**Recent commits, with technical theme** (last 30, branch HEAD `14be5b9`):

| Date | SHA | Theme |
|---|---|---|
| 2026-06-04 | `14be5b9`,`ba06e5a` | code cleanup, name-cache fix |
| 2026-06-04 | `e61d05c` | **Qwen27B integrated into Qwen9B codebase** (multi-config via `HF_MODEL` env-var + `Qwen35ModelArgs.load_state_dict`) |
| 2026-06-04 | `07e4d5f` | **decode conforms to tt_transformers `Generator`**, masked-bucket prefill consolidation |
| 2026-06-03 | `79f3697` | **masked fixed-bucket prefill for short prompts** (5 buckets 128/256/512/1024/2048) |
| 2026-06-02 | `b513e06`,`78f9fbb`,`55a1052`,`122cdf0`,`84ec593` | deleting dead "whole-sequence paged prefill trace" paths — only chunk-outer prefill survives |
| 2026-06-01 | `7f31122` | conform to `tt_transformers` Generator + mesh-aware blocks |
| 2026-06-01 | `7d431c8`,`f262990` | split into `tt/gdn/`, `tt/attention/` package layout |
| 2026-06-01 | `2687380` | **pin `transformers>=5.2.0` for `qwen3_5` arch loader** |
| 2026-05-31..06-01 | `2fd603f`..`a35f10c` | `Qwen35ModelArgs` subclasses `tt_transformers.ModelArgs`; weight-loading refactor |

**What the branch ships** (from URLs above):
- `tt/attention/decode.py` — paged decode through `gated_attention_forward_ttnn` with `cur_pos_tensor`, paged KV write + paged SDPA.
- `tt/attention/prefill.py` — two branches: paged-prefill via chunked SDPA + a concat-prefill fallback.
- `tt/gdn/_experimental_path.py` — sys-path shim to `experimental/gated_attention_gated_deltanet`.
- `tt/gdn/tp.py` — calls `recurrent_gated_delta_rule_decode_ttnn` for decode, `chunk_gated_delta_rule_seq_adapter` for prefill, `tt_all_reduce` finalizes the row-parallel output. Recurrent state is **bf16**.
- `tt/generator_interface.py` — `prefill_traced_chunked` is THE traced prefill entry for "every input length up to 128k" (per commit `07e4d5f` notes); `prefill_paged` for the non-traced path; `decode_forward` with optional trace.
- `tt/qwen35_vllm.py` — vLLM `initialize_vllm_model` hook. **`supports_prefix_caching=False, supports_async_decode=False`** — both explicitly disabled.
- `demo/text_demo.py` — orchestrator. `_run_traced_generation` (decode trace with GDN save/restore), `_run_paged_generation` (untraced paged), `_run_tp_generation` (multi-device).
- `demo/sample_prompts/input_data_long_{4k,8k,16k,32k,64k}.json` — long-context fixtures; **no 128k JSON**.
- `tests/test_prefill_masked_bucket.py`, `tests/test_prefill_trace_chunked.py`, `tests/unit/test_generator_contract.py`, etc.

---

## 3 — Side-by-side table

| Feature | Us (Qwen3.6-27B + 35B-A3B) | Them (`qwen9b-p150`) |
|---|---|---|
| **Primary model(s)** | Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE+GDN, Gemma 4 12B (base+IT). Multi-backend dispatch in `cb_api.BACKENDS` (`experiments/serve/cb_api.py:346`). | Qwen3.5-9B (hybrid), Qwen27B integration landed 2026-06-04 (`e61d05c`) via `HF_MODEL`-driven config. |
| **MoE** | **Shipped** — 35B-A3B Pattern-A broadcast MoE (B>1 bit-validated; `experiments/serve/server_35b_ttnn.py`, `server_35b_cb.py`; `archive/superseded_research_2026-06-04/35b_moe_pattern_a_plan.md`). | Not present in this branch's `qwen3_5_9b/` tree. (35B-A3B is dense MoE, 9B is dense GDN.) |
| **Gated DeltaNet decode kernel** | `ttnn.experimental.qwen36_gdn_decode_owned` — fused owned kernel (`server_tp.py:783`, `server_35b_ttnn.py:598`). Default on for 27B since 2026-05-18; recently un-clobbered for CB (commit `017665e`, `38b15b0`). | **`recurrent_gated_delta_rule_decode_ttnn`** — a different fused op composing "L2 norm + scale + delta step" internally (per `tt/gdn/tp.py` extract). Independent implementation of the same math. |
| **GDN recurrent state dtype** | bf16 for owned-gdn path; fp32 path exists but routes through broken manual recurrence (`[[feedback-35b-manual-recurrence-path-broken]]`). | bf16 only — `torch.zeros(*shape, dtype=torch.bfloat16)`. |
| **MAX_POS / max_seq_len** | **8192** (`server_tp.py:53`, bumped from 2048 for L=4000/8000). 35B same. Long-context working at L=7312 (`feedback_qb2_tp_long_context_works.md`). | **131,072** = `MAX_NUM_BLOCKS(2048) × BLOCK_SIZE(64)`. Demo fixtures max at 64k JSON; no 256K-specific code path verified. |
| **KV cache layout** | `[NUM_BLOCKS, N_KV=4, BLOCK_SIZE=32, HEAD_DIM]` sharded on dim=1 across mesh → per-chip `[NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM]` (`server_tp.py:327-335`). bf16. | `[MAX_NUM_BLOCKS=2048, num_kv_heads, BLOCK_SIZE=64, head_dim]` bf16, identity page table for requests. |
| **Chunked prefill** | `TT_CB_CHUNKED_PREFILL=1` ships at `PREFILL_CHUNK_SIZE=32` (`server_tp.py:527`). **L > 32 falls back to 1-tok/iter** through the decode trace (CW1 fix `ea9aa20`). | `prefill_traced_chunked` chunk-outer at **2048 tokens/chunk**, replayed per chunk; survives to 128k. Single entry-point handles every length (`07e4d5f`). |
| **Short-prompt prefill** | Same 1-tok/iter through the decode trace if `L > 32`. | **Masked fixed-bucket prefill** — 5 buckets (128, 256, 512, 1024, 2048) with masked GDN state. Prevents request-time program compile + parked-trace clobber (`tests/test_prefill_masked_bucket.py`). |
| **Paged SDPA op** | `ttnn.transformer.paged_scaled_dot_product_attention_decode` (B3 HiFi2 + paged page-table-driven; `feedback_paged_sdpa_shipped_tp.md`). | `ttnn.transformer.scaled_dot_product_attention_decode` + separate `paged_update_cache`; **not** the paged_scaled_dot_product_attention_decode variant per `attention/tp.py` extract. |
| **Continuous batching** | Production. Orca-style iteration-level scheduler (`cb_scheduler.py`), B=32 traced = 232 tok/s aggregate (27.9× scaling); admission/eviction/queueing; multi-EOS support. | **Not present in this codebase.** `qwen35_vllm.py` exposes a vLLM-side hook only — vLLM does scheduling externally. |
| **Prefix cache** | **Slot-level content-keyed, live.** Measured T1 5.1×, T2 8.0× per-token speedup (HANDOFF.md). Handles Qwen3.6 + Gemma 4 IT via universal `_active_prompt_suffix` detector. | **Explicitly disabled.** `model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}` in `qwen35_vllm.py`. |
| **OpenAI HTTP endpoint** | Yes — `openai_endpoint.py`, `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`, `/bootstrap`. | No. Delegated to vLLM. |
| **Chat template patches** | `preserve_thinking=True` Qwen3.6 jinja kwarg + trailing strip + universal `_active_prompt_suffix` for active-prompt-only suffixes (`openai_endpoint.py:47`). | None in-tree (model doesn't serve chat itself). |
| **Trace pattern** | Two-phase warmup (compile-all → capture-all), per `[[ttnn-multi-trace-two-phase-warmup]]`. Decode trace + S2 chunked-prefill trace coexist. `trace_region_size=400_000_000` for Gemma 4 48-layer decode. | Chunk-outer prefill trace (capture ONE 2048-token chunk's all-layer forward, replay per chunk to fit "4 GiB ceiling at long context"). Decode trace warmed via `prime_decode_trace`. |
| **Sliding-window attention** | Gemma 4 only (`server_gemma4_unified_ttnn.py`). Qwen3.6 does not use it. | Not in 9B (no sliding in arch); branch references `cache_position_modulo` for bounded sliding only in upstream tt-metal (`#45193`). |
| **vLLM integration** | None directly — we built our own CB stack. `archive/superseded_research_2026-06-04/27b_chunked_prefill_prior_art.md` documents the vLLM-TT plugin design we deliberately did not adopt. | Lightweight model-side hook (`qwen35_vllm.py`); scheduling delegated to vLLM externally. |
| **35B / MoE bringup** | Production. `server_35b_ttnn.py` 103 KB. | Not on this branch. |
| **Tokenizer / chat handling** | `apply_chat_template(tokenize=True)` direct path; `_normalise_template_output` handles both list-of-int (Qwen) and dict (Gemma); multi-EOS (`[1, 106, 50]` for Gemma IT). | `HF_MODEL` env-var → `transformers.from_pretrained`; chat template not handled in the model code. |
| **Multi-turn measured perf** | T0 8.69s wall (cold), T1 5.84s (PC HIT, 5.1× per-tok), T2 5.92s (PC HIT, 8.0× per-tok). HANDOFF.md. | Demo only; no multi-turn perf claim on this branch. |
| **Single-seq decode** | 27B: 12.93 tok/s (77 ms/tok) traced. 35B: 81.16 ms/tok = 12.32 tok/s. Gemma 4: 51.3 ms/tok traced. | Not stated in branch; `_run_traced_generation` tracks average decode latency at runtime. |

---

## 4 — Their long-context approach

**Headline mechanism**: chunk-outer trace prefill at 2048-token chunks +
masked-fixed-bucket short-prompt path + bf16 paged KV.

### KV layout (capped at 128K, not 256K)

From `text_demo.py` extract and `model_config.py` extract:

- `MAX_NUM_BLOCKS = 2048`, `BLOCK_SIZE = 64`, → `max_seq_len = 131,072` tokens.
- Paged KV cache shape: `[2048, num_kv_heads, 64, head_dim]` bf16.
- Page table is "an identity page table mapping request block indices to physical block indices" — single-request paging only on this branch's demo path.
- vLLM handles per-request paging when integrated through `qwen35_vllm.py`.

### Prefill: single traced entry point for all lengths

From the `07e4d5f` commit notes:

> "The traced path has a single entry, `prefill_traced_chunked`, for EVERY input length up to 128k"

The path:

1. **`prefill_traced_chunked`** captures ONE chunk's all-layer forward at `chunk_size=2048`, then replays it per chunk for full 2048-token chunks (per `text_demo.py` extract: "capture ONE chunk's all-layer forward and replay it per chunk (chunk-outer), keeping the captured trace under tt-metal's 4 GiB ceiling at long context").
2. **Tail** is processed via masked fixed-bucket prefill — buckets `{128, 256, 512, 1024, 2048}` (`test_prefill_masked_bucket.py`). The tail's bucket is chosen by rounding L_tail up; GDN state and conv state are masked to reflect actual tail length.
3. **GDN state carry-over**: end-of-chunk GDN recurrent state is threaded into the next chunk; masked-bucket tail receives the carried state.

### Why 5 buckets (the masked-bucket point)

The motivation isn't speed alone — it's avoiding device hang:

> "Short prompts (< the 2048 chunk size) take the on-demand eager prefill path, which compiles a fresh program for each distinct prompt length." (`test_prefill_masked_bucket.py` quote)

The 5 fixed buckets bound the kernel program cache so the parked prefill
trace doesn't get clobbered by a freshly-compiled program from a novel
prompt length.

### Compute kernel choices

- Decode SDPA: `ttnn.transformer.scaled_dot_product_attention_decode` + separate `paged_update_cache` (`attention/tp.py` extract). Not the fused paged variant we use.
- Prefill SDPA: `ttnn.transformer.scaled_dot_product_attention(is_causal=True)`.
- GDN decode: `recurrent_gated_delta_rule_decode_ttnn`, internally "L2 norm + scale + delta step" fused, bf16 state.
- All-reduce: `tt_all_reduce` after o_proj (no `num_links` shown in extract; we observed +1.65% from `num_links=2` per `feedback_p1_num_links_2_shipped.md`).

### "256K" verdict

**No 256K-specific code on the branch as of `14be5b9`.** The architectural
cap is **128K** (131,072 tokens). The engineer may be planning to bump
`MAX_NUM_BLOCKS` to 4096 for 256K, or there is a different branch we
should check. Worth asking them directly.

---

## 5 — Adoption opportunities

### From them → us

1. **Chunk-outer 2048-token prefill trace.** This is the largest TTFT
   lift available to us. Today our `PREFILL_CHUNK_SIZE=32` traced
   prefill is fine for the prefix-cache hot path but **anything beyond
   32 cold tokens drops to 1-tok/iter at ~80 ms/tok** (HANDOFF.md
   "deferred T3 multi-chunk traced prefill"). Pulling their
   chunk-outer design — capture one 2048-block all-layer trace,
   replay per chunk — likely cuts cold-start TTFT on 1k+ prompts by an
   order of magnitude. The op exists in upstream tt-metal
   (`ttnn.transformer.chunked_scaled_dot_product_attention`, audited
   in `archive/superseded_research_2026-06-04/27b_chunked_prefill_prior_art.md`); their branch is
   the concrete reference implementation.
2. **Masked fixed-bucket prefill for short prompts.** A 5-bucket
   {128, 256, 512, 1024, 2048} masked path with state masking is a
   clean solution to the "novel-length program-compile clobbers
   parked trace" failure we hit during chunked-prefill bringup
   ([`research/27b_prefill_trace_plan.md`](27b_prefill_trace_plan.md)).
   Their `test_prefill_masked_bucket.py` is a small file we can fork.
3. **`recurrent_gated_delta_rule_decode_ttnn` bake-off.** They built
   an independent decode-time GDN kernel that fuses L2-norm into the
   recurrence. Ours (`qwen36_gdn_decode_owned`) is separate and bf16
   state-only. Kernel-time A/B on the same 27B/35B weights would tell
   us whether (a) we should adopt theirs, or (b) ours is faster and
   we should upstream it.

### From us → them

1. **Slot-level content-keyed prefix cache.** Our measured **8.0×
   per-token TTFT speedup at turn 3** is bigger than any prefill
   optimization. Their `qwen35_vllm.py` explicitly sets
   `supports_prefix_caching=False` — they're paying for it. Our
   `research/27b_prefix_caching_plan.md` is the design doc, and
   `experiments/serve/live_slot_store.py` + `cb_scheduler.py` (lines
   313-617) is the ~700 LOC implementation. The non-obvious insight
   is that block-level KV caching gives zero TTFT win on a hybrid
   GDN+attention model (DN state at pos N requires the full
   sequence 0..N), so slot-level capture-the-whole-slot is the only
   working strategy until Marconi-style DN checkpoints ship
   (vllm#26201 still open).
2. **Universal active-prompt-suffix detector.** The 30-LOC
   `_active_prompt_suffix` in `experiments/serve/openai_endpoint.py:47`
   handles Qwen3.6's `<think>\n\n</think>\n\n` AND Gemma 4 IT's
   `<|channel>thought\n<channel|>` AND any future chat template with
   the same active-vs-past asymmetry, with one detection cost per
   tokenizer. They'll hit this exact problem the moment they wire
   chat into their `qwen35_vllm.py` for any thinking model.
3. **Dev harness with `importlib.reload` + trigger files.** Our
   `scripts/run_harness_tmux.sh` + `experiments/cb/dev/*_dev_harness.py`
   keeps the bootstrap resident and cuts iteration from ~14 min
   (35B) to ~30 sec per probe. Their `text_demo.py` re-bootstraps
   per run. For 9B their bootstrap is shorter, but for 27B it adds
   up; for 35B-A3B + MoE it's the difference between getting work
   done and not. Pattern is documented in
   `[[reference-gm4-dev-harness]]` and applies cleanly to a vLLM
   plugin worker too.

---

## 6 — Engagement recommendation

> **Reply to send the engineer.** Thanks for sharing the branch — really
> useful comparison. Two specific things on our side that might transfer
> to your 9B path: (1) we have a working slot-level content-keyed prefix
> cache with measured 5.1× / 8.0× per-token speedup at chat turns 2 / 3
> ([`experiments/serve/live_slot_store.py`](../experiments/serve/live_slot_store.py)
> + [`cb_scheduler.py:313-617`](../experiments/serve/cb_scheduler.py);
> design at [`research/27b_prefix_caching_plan.md`](27b_prefix_caching_plan.md));
> (2) a universal active-prompt-suffix detector that handles Qwen's
> `<think>` block AND Gemma's `<|channel>thought` AND should drop into
> any future thinking template
> ([`experiments/serve/openai_endpoint.py:47`](../experiments/serve/openai_endpoint.py)).
> Two things from your branch we'd love to learn from: (a) your
> `prefill_traced_chunked` chunk-outer pattern at 2048-token chunks —
> we run S2 chunked prefill at chunk_size=32 today and `L > 32` falls
> through to 1-tok/iter; your design is the clear path forward
> ([`models/demos/blackhole/qwen3_5_9b/tt/generator_interface.py`](https://github.com/tenstorrent/tt-metal/blob/qwen9b-p150/models/demos/blackhole/qwen3_5_9b/tt/generator_interface.py)
> + [`tests/test_prefill_masked_bucket.py`](https://github.com/tenstorrent/tt-metal/blob/qwen9b-p150/models/demos/blackhole/qwen3_5_9b/tests/test_prefill_masked_bucket.py));
> (b) your `recurrent_gated_delta_rule_decode_ttnn` op vs our
> `ttnn.experimental.qwen36_gdn_decode_owned` — independent decode-GDN
> kernels, both bf16, would be a great kernel-time A/B if you're up
> for it. Side note on the "256K" framing: the branch as of `14be5b9`
> looks like 128K (`MAX_NUM_BLOCKS=2048 * BLOCK_SIZE=64`) — is there a
> separate branch for 256K, or is that on the roadmap via
> `MAX_NUM_BLOCKS=4096`?

---

## Method notes

- Branch HEAD at fetch: `14be5b9128bd9a0f92f44ba1328a20070f0a4b99`
  (2026-06-04). Recursive tree fetched via
  `gh api repos/tenstorrent/tt-metal/git/trees/qwen9b-p150?recursive=true`
  (20,702 items; 336 qwen-related).
- Commits page 1: 30 entries spanning 2026-05-31..2026-06-04 are the
  branch-specific work; commits page 2 onward are merged-base activity.
- Source files inspected via raw github URLs on the branch:
  `tt/{model.py, model_config.py, attention/decode.py,
  attention/prefill.py, attention/tp.py, gdn/decode.py, gdn/tp.py,
  gdn/_experimental_path.py, generator_interface.py, qwen35_vllm.py}`,
  `demo/text_demo.py`, `tests/test_prefill_masked_bucket.py`. Two
  commits read in full diff: `79f3697`, `e61d05c`, `07e4d5f`.
- Files we did NOT manage to inspect (404 on raw): `tt/attention/kv_cache.py`
  doesn't exist (KV setup lives in `attention/decode.py`); no draft PR
  found for merging the branch (`gh pr list` against the branch was
  not run — only the commit log).
- Our side: read `HANDOFF.md`, `experiments/serve/{server_tp.py,
  server_35b_ttnn.py, openai_endpoint.py, cb_engine.py, cb_scheduler.py,
  cb_api.py}`, `research/27b_prefix_caching_plan.md`,
  `archive/superseded_research_2026-06-04/27b_chunked_prefill_prior_art.md`. Direct `grep` for
  MAX_POS, NUM_BLOCKS, BLOCK_SIZE, chunked_prefill, prefix_cache,
  owned_gdn confirmed file:line citations above.
