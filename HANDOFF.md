# HANDOFF — cold-start one-pager

What this project is, where the perf is now, what to run, and what is next.
Read top to bottom; everything else is linked.

## Project

Qwen3.6-family bringup on Tenstorrent Blackhole (P150 × 4). Two production paths:

- **27B dense, 4× P150 TP** — `experiments/serve/server_tp.py`.
- **27B continuous batching** — `experiments/serve/cb_api.py` + `cb_engine.py`,
  served by `experiments/serve/scripts/serve_cb.sh`. **This is the canonical
  chat path.** Both production paths run on `qb1` and `qb2`.
- **35B-A3B MoE** — `experiments/serve/server_35b_ttnn.py` (in-progress
  perf work; default `state.moe_mode = "pattern_a_batched"`).

Hosts: `qb1` and `qb2`, both 4× Blackhole P150 with working `FABRIC_1D`.

## Where the perf is now

| Path | Number | Source |
|---|---|---|
| 27B TP single-seq (steady-state, traced) | **12.93 tok/s** (77 ms/tok) | `serve_tp` on qb2 |
| 27B CB B=1 (traced) | 12.96 tok/s (==prod) | `experiments/cb/bench/trace.py` |
| 27B CB B=32 (traced, aggregate) | **150.5 tok/s** (11.6×) | same |
| 27B CB B=64 (traced, aggregate, shift-acc conv1d) | **593 tok/s** (45.8×) | same |
| 35B-A3B traced decode (qb1, after A002+A003+A004+A008+A009) | **81.16 ms/tok = 12.32 tok/s** | `research/35b_perf_milestones.md` |

CB SLO (qb1, 8 clients × 60 s, P5 gate 2026-05-30):
0 errors / 36 requests / 15 tok/s aggregate / **TTFT p99 = 176 ms**
(`experiments/cb/load/concurrent_chat.py`).

## Hardware ceiling

P150 measured: **404 GB/s/chip** DRAM BW, 110 worker cores, 31.81 GB DRAM
(`feedback_p150_memory_bandwidth_measured` in MEMORY.md).
For 35B-A3B with ~3 GB active params/token/chip:
bf16 BW floor ≈ 3.7 ms/tok → **270 tok/s ceiling**;
bf8 BW floor ≈ 1.85 ms/tok → **540 tok/s ceiling**.

The target is the hardware ceiling, not parity with someone else's number.

## Chat path (production)

```bash
# Recommended config:
TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready
bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # SIGTERM → graceful drain → mesh release
```

Env knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics` (Prometheus).
See README §"Chat server (production)" for `curl` + `openai` client examples.

**Current status (2026-06-02): 27B server is STOPPED.** qb1's mesh is
consumed by the 35B dev harness (CB35-1 v0 work — see Roadmap §3). The
27B production stack is feature-complete (CB + prefix caching + sampling
+ OpenAI endpoint + metrics, all validated). Restart any time with the
command above — exclusive of running the 35B harness on the same host.

## What's next

**Prefix caching — END-TO-END VALIDATED (2026-06-01).** Slot-level
content-keyed prefix cache for the CB scheduler. Returning chats reclaim
their live slot at `cur_pos = len(matched_prefix)`, skipping re-prefill of
the history. qb1 smoke test:
- Turn 1 (cold, miss): **5.33s**
- Turn 2 (warm, **HIT**): **2.73s** — 1.96× speedup, 1 cache hit, 0 misses
- Qualitative: turn 2 correctly continued conversation (France → Germany / Berlin)

**Recommended runtime config (2026-06-02):** `TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1`.
CW1 fix (`ea9aa20`) makes both flags coexist by skipping the eager
chunked-prefill fallback for L > chunk_size — the L > chunk_size path
takes the legacy 1-tok/iter route via the decode trace, which is
allocation-free and safe alongside captured traces. Cold-start L > 32
prompts pay 1-tok/iter latency (~80 ms/tok), but with prefix caching
catching turn-2+ this is rare in chat workloads. Verified on qb1 smoke
2026-06-02: 1.97× turn-2 speedup preserved, no wedge.

Gated by `TT_CB_PREFIX_CACHE=0/1` (default 0). `TT_CB_PREFIX_TTL_S=300` for
stale-slot cleanup. Metrics in `/metrics`: `cb_prefix_cache_{hits,misses,
evictions}_total`, `cb_prefix_cache_live_slots`, `cb_prefix_cache_enabled`.

Load-bearing fixes from the smoke debug chain:
- `mark_live` fires on `max_tokens` cap (not just EOS) — `4acc955`
- `tokens_so_far` keeps trailing EOS for chat-template compat — `184f00e`
- `_messages_to_prompt`: `preserve_thinking=True` + trailing-only `<think>` strip — `2cad663`

Plan + per-milestone status: [`research/27b_prefix_caching_plan.md`](research/27b_prefix_caching_plan.md).
Research:
- [`research/vllm_prefix_caching_audit.md`](research/vllm_prefix_caching_audit.md) — APC design
- [`research/vllm_chat_template_handling.md`](research/vllm_chat_template_handling.md) — Qwen3.6 quirks + upstream-blessed `preserve_thinking` fix

Regression gate: `experiments/cb/isolate/chat_template_invariant.py`
(7 cases including 239/239 long-prompt, unicode, 3-turn compound).
Memory: [[prefix-caching-design]], [[qwen36-preserve-thinking]].

**S2 — chunked prefill — LIVE in production (2026-06-01).** CB serves with
`TT_CB_CHUNKED_PREFILL=1`: traced chunked prefill at chunk_size=32 for L ≤ 32,
legacy 1-tok/iter fallback for L > 32. Two-phase warmup (compile-all-then-capture-all)
solves the multi-trace coexistence wedge per [vLLM #352](https://github.com/tenstorrent/vllm/issues/352).
Plan + post-mortem: [`research/27b_prefill_trace_plan.md`](research/27b_prefill_trace_plan.md).

Deferred / superseded: T3 multi-chunk traced prefill (chat win comes from skipping
re-prefill via prefix caching, not making re-prefill faster). Bigger chunk_size
(same reasoning). Both revisitable for long single-prompt cases (no prior cache
to match) after prefix caching ships.

## Roadmap (priority order, 2026-06-02)

1. **DONE — `chunked_prefill=1 + prefix_cache=1` coexist** (CW1, commit `ea9aa20`).
   Run prod with both flags on. Long L > chunk_size prompts use 1-tok/iter
   fallback (slow but allocator-safe). Validated 2026-06-02: 1.97× turn-2
   speedup preserved + no wedge.
2. **DONE — `TT_BACKEND` env selector** (MM1, commit `418f9cc`). cb_api.py
   has a `BACKENDS` registry; `TT_BACKEND={27b,35b}` switches at boot.
3. **IN PROGRESS — 35B-A3B CB bringup (CB35-v0..v4)**. Plan:
   [`research/35b_cb_bringup_plan.md`](research/35b_cb_bringup_plan.md).
   Research basis: [`research/tt_metal_moe_cb_patterns.md`](research/tt_metal_moe_cb_patterns.md)
   (DeepSeek-V3 + Llama-70B-Galaxy patterns).
   - **v0 BIT-VALIDATED 2026-06-02** (commits `112d72a`, `49778b3`):
     `cb35_v0_smoke` 3/3 + `cb35_v0_chat` (8-tok decode bit-identical).
     Critical fix: `base.reset_caches_ttnn` leaks per-layer tensors;
     `cb_reset_states` explicit-deallocs first.
     `return_logits=True` raises (35B [1,VOCAB] readback broken, #149).
     cb_engine routes 35B through topk via `TT_CB_TOPK_K=64`.
   - **v1.0 alloc + v1.1 embed/RoPE BIT-VALIDATED 2026-06-02** (commits
     `ec01052`, `cf211a4`). `setup_cb_state(B>1)` allocates B-leading
     `cb_dn`/`cb_kv` + per-slot `cb_page_table_tt`. `_batched_prelude`
     reads `cb_tok_buf` / `cb_rot_idxs_buf`. Gate: 3/3 alloc + 4/4 prelude.
   - **v1.2 batched DN BIT-VALIDATED 2026-06-02** (commit `4546d29`).
     `dn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2 slot 0. Rank-3
     `[B, N, D]` rms_norm fix ([[ttnn-rms-norm-shape-drift]]).
   - **v1.3 batched GatedAttention BIT-VALIDATED 2026-06-02** (commit
     `e8f2d82`). `attn_step_batched_35b` cos=1.0/mad=0.0 at B=1 and B=2
     slot 0 — first-try pass. Rank-3 `_apply_partial_rope_b` sidesteps
     base's K-broadcast workaround. Paged KV write + SDPA decode use
     `cb_cur_pos_buf`/`cb_page_table_tt` for per-slot context.
     `setup_cb_paged_cfgs` builds B-sized HEIGHT_SHARDED L1 mem cfg +
     SDPA progcfg (B=1 reuses base's `state.paged_*`).
   - **v1.4 batched MoE SHIPPED 2026-06-02** (commit `2d0f582`).
     `moe_step_batched_35b` — B=1 bit-identical to base
     `moe_forward_ttnn_pattern_a_batched`. B>1 is per-slot sequential
     loop with KNOWN ~13% rel drift between identical-input slots at
     small magnitudes (layer-0 MoE-only output, |out|~3e-4). Likely
     ttnn-internal MoE matmul accumulation-order non-determinism. The
     true broadcast port (B-leading Pattern A) is v1.4b, task #156.
   - **v1.5 full B>1 forward SHIPPED 2026-06-02** (commit `bc96651`).
     `forward_batch_tp_inner_batched` + `layer_forward_batched_35b` —
     drives full 40-layer chain at B>1 across multi-step decode without
     errors. Bit-equiv to B=1 reference FAILS (slot 0 sequence diverges
     from B=1 ref) because the v1.4 per-slot MoE drift compounds across
     30 MoE layers and 4 steps. **v1.5 bit-equiv is GATED on v1.4b.**
     Infrastructure shippable for B=1 production; B>1 production needs
     v1.4b.
   - v1 (true B>1 batched forward): next. ~3-5 days.
   - v2 (trace capture at B=N): ~1-2 days.
   - v3 (owned-GDN batched FOLD trick): optional.
   - v4 (prefix cache for attn layers only): LOW PRIORITY (vllm#36493 reports
     ~0.1% hit rate on this arch class — DN layers can't be cached).
   - **Dev iteration harness** (`experiments/cb/dev/cb35_dev_harness.py`):
     bootstrap-once long-running python on qb1, watches `/tmp/cb35_trig/`,
     reloads via `importlib`. Cuts fix-test cycle from ~14 min to seconds.
     **MUST launch via `bash scripts/run_harness_tmux.sh`** — nohup + disown
     and setsid + double-fork both fail to keep the python alive after SSH
     disconnect on qb1 (it dies before completing bootstrap). tmux
     survives by design. See `research/35b_cb_bringup_plan.md` for usage.
4. **Multi-model fleet** — plan: [`research/multi_model_serving_plan.md`](research/multi_model_serving_plan.md).
   Once 35B is live, MM5 (Mistral Small 3.2 24B) is the strongest
   framework-generalization test (different vendor, different tokenizer,
   pure dense GQA, no DN). Candidate research:
   [`research/home_llm_landscape_2026.md`](research/home_llm_landscape_2026.md).
5. **Long-context concurrent stress test (MM3)** — validate PC works at
   L=1000+ prompts under realistic concurrency. Reuses
   `experiments/cb/load/concurrent_chat.py`. Final-validation step after
   the multi-model work is in place.

**35B perf** (parallel track). Next levers tracked in
[`research/35b_perf_milestones.md`](research/35b_perf_milestones.md):
async all_reduce overlap, expert-broadcast elimination, routing-weight
fusion, bf8 expert weights.

## Repo entry points

- README — install + demos.
- CONTRIBUTING — dev loop, canary gate, code style.
- `research/` — design docs + living plans (index: `research/README.md`).
- `wiki/` — Q&A wiki, learning-by-building.
- `models/` — multi-model demos (Llama, Qwen2.5, SmolLM, 8B).

## Read order when resuming work

1. This file.
2. [`research/profiling-quick-reference.md`](research/profiling-quick-reference.md) — Tracy + tt-perf-report capture/analyze.
3. [`research/35b_perf_milestones.md`](research/35b_perf_milestones.md) — 35B perf trajectory.
4. [`research/27b_cb_scope.md`](research/27b_cb_scope.md) — CB design + numbers (CB0–CB4).
5. [`research/35b_tt_perf_report_findings.md`](research/35b_tt_perf_report_findings.md) — empirical writeup behind 35B advice.

## Load-bearing rules (each cost a multi-day debug)

- **View-decay**: `ttnn.slice` / `ttnn.reshape` return views. Never
  `ttnn.deallocate` the source while a view is live; clone when in doubt.
- **+1 zero-centered RMSNorm offset** on `q_norm` / `k_norm` /
  `input_layernorm` / `post_attention_layernorm` / `final_norm` (Qwen3.6).
- **K-broadcast RoPE workaround** in the SDPA path — sidesteps a ttnn
  `[1, HEAD_DIM]` slice/concat bug.
- **bf16 KV cache** required by paged SDPA (fp32 hard-rejected).
- **HiFi4 + `fp32_dest_acc_en`** on every matmul (the 91f recipe); mixing
  fidelities corrupts ops silently on Blackhole.

## Workflow

- Profile-driven only. Cite a Tracy / tt-perf-report number for any
  optimization claim. Frame deltas as Δ from BW floor.
- Correctness gate: 5-token Paris (`"The capital of France is" → " Paris..."`)
  on prefill IDs `[2614, 314, 279, 369, 11751]`.
- Iterations in git history or `scratch/`, never in demo scripts.
- Remote-only execution (`ssh qb1` / `ssh qb2`); no device code locally.
