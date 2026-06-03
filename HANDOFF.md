# HANDOFF — cold-start one-pager

What this project is, where the perf is now, what to run, and what is next.
Read top to bottom; everything else is linked.

## Live session state (2026-06-02 late evening)

- **27B HTTP smoke PASSED** end-to-end on qb1 — `/v1/chat/completions` returns
  "The capital of France is Paris." in 2.4s with `finish_reason=stop`.
  Three regressions fixed this session: cb_scheduler dispatch (commit `97abfab`),
  35B bootstrap log-signature (`8111d70`), 27B bootstrap log-signature (`a7ea0fe`),
  deploy gap (workflow), 35B tokenizer alias (`73fd269`).
- **35B HTTP bootstrap PASSED** end-to-end on qb1 — 14:29 bootstrap, all 40
  layers + setup buffers, `/health` reports `{"ok":true,"ready":true,
  "model":"Qwen/Qwen3.6-35B-A3B","slots":2,"sampling":true}`.
- **35B first inference step CRASHES** — task #160 `CB35-bug`. Server log:
  `[cb-engine] step failed: TypeError: only length-1 arrays can be converted
  to Python scalars`. Suspect site: `server_35b_cb.py:268`
  (`int(token_ids[0])`). The dev-harness validator (`cb35_prod_topk`, 4/4 PASS)
  drives `_step_sampled_topk` directly and skips the cb_scheduler.advance
  layer where this fires. Memory: `feedback_dev_harness_vs_cb_engine_gap.md`.
- The earlier "35B bootstrap hangs at enumerate shards" hypothesis was wrong.
  Side-file was frozen because only the FIRST log line was routed through
  the new `log` callable; layer-upload logs DID emit ("layer 10/40 uploaded"
  etc.). Bootstrap completed in ~14 min the first time we let it run to
  completion (then crashed at lifespan post-bootstrap on `state.tok`).

## Project

Qwen3.6-family bringup on Tenstorrent Blackhole (P150 × 4). Production paths:

- **27B dense, 4× P150 TP** — `experiments/serve/server_tp.py` (single-stream
  Unix socket) or via CB (`serve_cb.sh start`, default backend).
- **27B continuous batching** — `experiments/serve/cb_api.py` + `cb_engine.py`,
  served by `experiments/serve/scripts/serve_cb.sh` (the canonical chat path).
- **35B-A3B MoE continuous batching** — same `serve_cb.sh` with
  `TT_BACKEND=35b`. Routes through `server_35b_ttnn.py` (model) + `server_35b_cb.py`
  (batched forward). Production-ready 2026-06-02.

Backend selection: `BACKENDS` registry in `cb_api.py` + identical dispatch
in `cb_scheduler.py`. Adding a new backend = drop a `server_<name>_ttnn.py`
+ `server_<name>_cb.py` + register both in `BACKENDS`/`_BACKEND_MODULES`.

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
# 27B (default backend):
TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready

# 35B-A3B MoE:
TT_BACKEND=35b TT_CB_SLOTS=2 TT_CB_TOPK_K=64 \
  bash experiments/serve/scripts/serve_cb.sh start   # ~6-14 min bootstrap
# verify via:  curl http://qb1:8000/v1/models  (must report Qwen/Qwen3.6-35B-A3B)
# 35B contract: TT_CB_TOPK_K must be >0 (logits-readback broken, #149);
#               cb_api defaults TT_CB_TOPK_K=64 when TT_BACKEND=35b.
bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # SIGTERM → graceful drain → mesh release
```

Env knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics` (Prometheus), `/bootstrap` (stage + elapsed_s).
See README §"Chat server (production)" for `curl` + `openai` client examples.

**Bootstrap observability** (commit `3e150c0`, 2026-06-02). Lifespan
startup blocks uvicorn from accepting HTTP until it yields, so HTTP
health probes can't see the 14-min 35B bootstrap. Three channels:
- `/bootstrap` (JSON): stage + elapsed_s + ready. Only reachable AFTER
  lifespan yields — useful for "is it ready yet" but not for "is it
  stuck during boot".
- `/health` enriched: 503 payload includes `{bootstrap: {…}}` once
  reachable.
- **Side file `~/tt-xla/.cache/server_cb.bootstrap.log`** — appended
  with explicit fsync from the bootstrap thread. Tail-able during the
  lifespan startup phase, when no HTTP endpoint is reachable yet.
  This is the canonical "is it making progress?" probe.

**Current status (2026-06-02 late evening): 35B HTTP server (re)bootstrapping
on qb1**. Background poll task `bz5lcsa9n` is watching for ready. v0/v1/v2
device primitives + cb35_prod_topk all PASS via the dev harness; the HTTP
wrap is what's flaky to bring up (uvicorn lifespan + 14-min boot + worker-
thread stdout buffering). Once the side-file shows `[harness ready]`, the
chat TUI / curl should work end-to-end.

## Deploy hygiene before any `serve_cb.sh start`

Always sync the entire `experiments/serve/*.py` before starting the server.
The dev harness uses `importlib.reload` per test so it only needs the file
under test; the production server boots a fresh process and reads qb1's
filesystem ONCE.
- One MM1 commit (`418f9cc`) sat in local git but never reached qb1; the
  server kept loading 27B for hours of debugging until we caught it.
- See memory `[[deploy-serve-files-too]]`.
- Quick command: `bash scripts/deploy.sh experiments/serve/*.py`

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
   - **v1.4b broadcast MoE BIT-VALIDATED 2026-06-02** (commit `a6ac640`).
     True broadcast Pattern A — `[E_LOCAL, B, HIDDEN]` middle-dim bump
     in batched expert matmul. mad=0.000000 vs base at B=2 slot 0.
     Fixed the 13% per-slot drift from v1.4 loop. Critical bug in init
     port: used `MOE_INTER_CHIP (128)` instead of `MOE_INTER (512)`.
   - **v2 trace capture SHIPPED 2026-06-02** (commit `c547419`).
     Two-phase warmup + `begin_trace_capture`/`end_trace_capture`
     around `forward_batch_tp_inner` works at B=2. Replay 149.7 ms/step
     vs eager 296.7 ms/step = **1.98× speedup**. cb_scheduler trace
     plumbing inherits automatically (calls the unified entry).
     Higher B → larger speedups.
   - **CB35-prod wire-up GATED 2026-06-02** (commit `f1f7a61`).
     Unified `forward_batch_tp_inner` dispatches B=1→base (v0 bit-id) /
     B>1→v1 batched. Now supports `return_topk=K` at both Bs. cb_scheduler
     can drop in without changes. `cb35_prod_topk.py` 4/4 PASS:
     B=1 top-1 = 8 (matches v0), B=2 distinct prompts produce distinct
     top-1 tokens. Ready for cb_api/cb_engine end-to-end at
     `TT_CB_SLOTS=2`.
   - **v1.5 full B>1 forward FUNCTIONAL PASS 2026-06-02** (commit `0a50e97`).
     `forward_batch_tp_inner_batched` + `layer_forward_batched_35b` —
     40-layer chain at B>1 runs end-to-end. v1_chat results:
     - ✓ slot 0 != slot 1 with distinct prompts (per-slot independence)
     - ✓ slot 0 == slot 1 with same prompt (determinism)
     - ⚠ slot 0 != B=1 ref by argmax tokens — bf16 chain precision drift
       compounding across 40 layers (every individual op verified bit-id
       at B=2 slot 0; the chain noise is what flips argmax tokens). See
       [[bf16-chain-drift-at-B-gt-1]] for the lesson. **Production-shippable** —
       each slot's generation is valid; use cosine for benches, not
       exact tokens.
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
