# HANDOFF — cold-start one-pager

What this project is, where the perf is now, what to run, and what is next.
Read top to bottom; everything else is linked.

## Live session state (2026-06-03 — Gemma 4 12B is the active priority)

**Pivot 2026-06-03**: paused 35B drift (#163) and pivoted to Gemma 4 12B
bringup (#165). Driver: 14-min 35B bootstrap per harness restart was
severely rate-limiting iteration AND the harness itself hung silently
mid-investigation (task #166 captures the harden-it-before-next-bootstrap
work). Gemma 4 12B is dense, 12B, dual-attention-type — smaller weights
(~5-7 min bootstrap), structurally interesting (sliding+global), and
exercises position-dependent paths in isolation. If a positional-state
bug lives in our codebase, v0.3 surfaces it without MoE/DN confounders.

### Active — Gemma 4 12B bringup (#165)

- **Plan**: [`research/gemma4_12b_bringup_plan.md`](research/gemma4_12b_bringup_plan.md)
  — start at §"REUSE MANDATE" (always grep for an existing pattern
  before writing new code) → §0 Step 0 pre-flight (DONE) → §2
  code-reuse map at `file:line` → §3 novel items → §4 sub-task
  breakdown with cosine gates.
- **Step 0 DONE 2026-06-03 (commit `4395b28`)**:
  - §6.1 `sliding_window_size` kwarg confirmed on
    `ttnn.transformer.paged_scaled_dot_product_attention_decode`
    — sliding decode is a kwarg flip, not a new kernel. Plan §3.3
    risk closed.
  - §6.3 GELU variant probe (`experiments/utils/gemma4_gelu_variant_probe.py`):
    `ttnn.gelu(x, fast_and_approximate_mode=False)` matches
    `gelu_pytorch_tanh` at cos=0.999998. The fused-activation path
    (`ttnn.mul(.., [UnaryOpType.GELU])`) uses the APPROXIMATE kernel
    — DO NOT mirror 35B's SwiGLU fused pattern in Gemma 4 v0.
  - Bonus: SDPA doc confirms `cur_pos=-1` skips compute for that
    batch slot — may resolve 35B #162 if backported.
- **v0 oracle DONE 2026-06-03**: `experiments/utils/hf_reference_gemma4_12b.py`
  (commit `156dc9f`) produced `.cache/hf_oracle_gemma4_12b/` with
  6-token "The capital of France is" forward; HF predicts " a" at pos 5;
  all 6 L0 sub-captures present (in/post_attn/pre_ff/mlp/post_ff norms
  + mixer_out).
- **v0.1.0 DONE 2026-06-03 commit `b9f3c35`** — bootstrap + embed
  scale + L0 input_layernorm. cos 0.999996 / 0.999991 vs HF oracle.
  Bootstrap **74-84 sec** on qb1 (vs 35B's 14 min — 11× faster).
- **v0.1.1 DONE 2026-06-03 commit `a35525e`** — Q/K/V projections +
  q_norm/k_norm. 7/7 sub-steps PASS at cos ≥ 0.99997. Hit a sharder
  gotcha en route (`ShardTensor2dMesh` vs the correct
  `ShardTensorToMesh(dim=0)`) — one-line fix; memory entry
  `[[ttnn-shard-1d-vs-2d]]`.
- **v0.1.2 DONE** — attention output at pos 0 + o_proj. cos=0.999990
  mad=0.0625. Found via numpy reproducer that Gemma 4 has v_norm =
  `RMSNorm(head_dim, with_scale=False)` applied to V after v_proj
  (memory `[[gemma4-v-norm]]`). 27B/35B don't have this — reading the
  HF source caught what guessing wouldn't. v0.1.2 also surfaced a
  ttnn `all_reduce_async` signature change vs the simple
  `ttnn.all_reduce(cluster_axis=1)` used by 35B (one-line fix in
  `all_reduce_tt`).
- **v0.1.3 DONE 2026-06-03 commit `7f9f396`** — 13/13 sub-steps PASS;
  full L0 forward bit-id to HF at cos ≥ 0.999958. L0 output (vs HF
  hidden_states[1, 0, :]) cos=0.999975. Two-op GELU works per Step 0.2;
  both Gemma 4 post-norms (post_attention + post_feedforward) correct.
- **v0.2 DONE 2026-06-03** — all 48 layers (sliding + global dispatch)
  + final_norm + tied lm_head + 30·tanh logit softcap. Greedy top-1
  matches HF at pos 0: TT argmax=258882 == HF argmax=258882 (`<image|>`).
  final_norm cos=0.999563, logits cos=0.999137. Surfaced two more
  Gemma 4 novelties: per-layer `layer_scalar` ([[gemma4-layer-scalar]])
  multiplied at end of each decoder layer, AND the cosine-is-not-enough
  diagnostic discipline ([[cos-not-enough-also-check-mad]]) — direction
  passed at L0 but magnitude was 18× off, propagating to L1 collapse.
- **v0.3 IN FLIGHT — setup DONE 2026-06-03 commit `cb4e299`**:
  KV caches per layer (sliding [num_blocks, 8, 32, 256] sharded over
  mesh dim=1; global [num_blocks, 1, 32, 512] replicated), SDPA
  program + memory + compute configs (35B B3 recipe — HiFi2 +
  fp32_dest_acc=False per [[fp32-sdpa-cliff-probe]]), RoPE tables
  (sliding theta=10000 full-rotate, global p-RoPE theta=1e6 partial
  0.25). v0.2 probe re-runs with these in place: STILL PASSES.
  Only FORWARD changes remain. Plan §"v0.3 sub-staging" has detailed
  design notes with 35B `file:line` references.
  - v0.3.0 ARGMAX PASS commit `01c88d6` (batch-0 hack)
  - v0.3.0.1 **FULL PASS** commit `e2ae9f2` (2 SDPA calls per
    sliding layer, NKV=1 each, proper GQA). final_norm cos 0.999601,
    logits cos 0.999370, argmax matches HF.
    Memorialized: [[paged-update-cache-nkv-per-chip]],
    [[read-kernel-source-first]], [[use-existing-isolation-probes]].
  - **v0.3.1 FIXED 2026-06-03 commit `c97bf15`** — root cause was
    SDPA `scale=1.0/sqrt(head_dim)` (wrong); Gemma 4 text attention
    sets `self.scaling = 1.0` (HF `modeling_gemma4.py:1178`),
    confirmed in Tenstorrent's in-tree demo (`decode.py:144`). The
    wrong scale was MASKED at pos 0 because a single-token softmax
    is 1.0 regardless of scale. After-fix multi-step: pos 0..5
    cos_final all ≥ 0.997 (was 0.26 at pos 1); 5/6 argmax PASS (pos
    4 cos=0.9984 but argmax differs — bf16 tie noise per
    [[bf16-chain-drift-at-B-gt-1]]). Per-layer drift ladder
    L0-L46 cos > 0.996 at both pos 0 and pos 1. Memory rule:
    [[feedback-gemma4-sdpa-scale-1]].

    The debug ladder remains useful — keep the env knobs
    (GM4_DEBUG_POS, GM4_ROPE_ZERO, GM4_SKIP_SLIDING, GM4_SKIP_GLOBAL)
    and the four isolation probes (gm4_sliding_write_read,
    gm4_global_write_read, gm4_rope_lookup, gm4_per_layer_drift_pos1)
    for future bringup work.

    Full debug writeup in memory `[[project-gm4-pos1-cliff]]` — the
    bisection burned several "masked-at-pos-0" hypotheses (view-decay,
    in-place buffer update, canonical SDPA config) BEFORE landing on
    scale=1.0. None of those earlier fixes moved the 3/6 number, but
    each closed a real Tenstorrent anti-pattern that would have masked
    the real bug, so they ship.
  - **v0.3.2 DONE 2026-06-03 commit `acb20a6`** — 16-token free-run
    coherent: "The capital of France is a city of art, culture, and
    history." End-to-end forward composition validated. Probe:
    `gm4_v032_freerun.py`.
  - **v0.3.3 long-context validation IN FLIGHT** — mirrors 27B/35B's
    needle-haystack + bf16 prefill drift gates. Three sub-probes:
    (a) per-pos cosine ladder at L=128 vs an extended HF oracle, (b)
    sliding-window correctness at pos > 1024 (invariance to pre-window
    tokens), (c) needle-haystack retrieval at L=100, 500, 1024. Plan
    table has the fork map + concrete gates. Reuse: extend
    `hf_reference_gemma4_12b.py`, fork `needle_haystack_35b_ttnn.py`
    and `gm4_v031_multistep_cos.py`.

  **~5-6 days of focused work remaining** to ship `TT_BACKEND=gemma4_12b
  serve_cb.sh start` chat working end-to-end:
  v0.3.3 long-context (~1 day) + v0.4 trace (~1 day) + v1 CB (~2-3 days)
  + v2 HTTP (~1 day).
- **All computation on (1,4) P150 mesh on qb1**; readback only for
  cosine compare against the HF oracle (matches 27B/35B pattern).
- **Reuse mandate (user-set 2026-06-03)**: every new file must cite the
  existing file it forks (or "no prior art, here's why") in its commit
  message. Deep utility shelf exists: `experiments/cb/_runner.py`,
  `experiments/utils/{ttnn_introspect,hf_reference_35b,cosine_ladder_*,
  test_fused_*_isolated,needle_haystack_*,tracy_*}.py`,
  `experiments/cb/isolate/{paged_sdpa,paged_update_cache,chunked_sdpa,
  owned_gdn,...}.py`, `experiments/serve/{server_35b_ttnn,server_35b_cb,
  server_tp_cb,cb_api,cb_scheduler}.py`. Plan §"REUSE MANDATE" has
  the full table.
- **Step 0 — pre-flight hardware probes (no model upload, ~5 min)**:
  1. **§6.1**: confirm qb1's installed ttnn exposes `sliding_window_size`
     on `ttnn.experimental.paged_scaled_dot_product_attention_decode`.
     If missing, rebuild ttnn or use manual K/V slice fallback before v0.3.
  2. **§6.3**: 1D pointwise check — which ttnn UnaryOp matches
     `torch.nn.functional.gelu(approximate="tanh")` over `[-5, 5]`.
- **v0 staging** (mirrors 35B `research/35b_cb_bringup_plan.md`):
  v0.1 L0-only forward → v0.2 all 48 layers → v0.3 KV cache with
  sliding-window kwarg → v0.4 trace capture → v1 CB B=4 → v2 server
  wire-up + chat smoke.
- **Top NOVEL items** (full detail in plan §3):
  dual head_dim (256 sliding / 512 global), four norms per layer
  (Llama RMSNorm `w`, NOT Qwen `(1+w)` — bit us hard on 35B
  [[qwen36-qnorm-knorm-zero-centered]]), tied embed + sqrt(hidden)
  embed-scale + 30·tanh(x/30) logit softcap, GELU_tanh activation,
  p-RoPE = standard partial RoPE with global head_dim divisor,
  `attention_k_eq_v` on global layers only.

### Parked — 35B drift cliff (#163, #164, #162)

- **#163**: full staging notes in
  [`research/35b_drift_next_session_plan.md`](research/35b_drift_next_session_plan.md)
  §"REAL findings 2026-06-03". Cliff between pos 1 (cos_L32=0.99) and
  pos 5 (cos_L32=0.32); flavor = positional-state bug. Step 1 probe
  wrapper `cb35_drift_cliff_search` deployed but never executed
  (harness restart from hang was killed when we pivoted).
  Cross-pollination: if Gemma 4 v0.3 surfaces a positional-state bug,
  it may share mechanism with this cliff.
- **#164**: manual recurrence path structurally broken
  (cos 0.08 @ pos 0 with owned_gdn=OFF). Orthogonal to cliff;
  fp32 H_t fix (`92b442f`/`8010b3c`) on main routes through broken path.
- **#162**: B>1 batched forward empty-slot poison. Default
  TT_CB_SLOTS=1 for 35B masks it.
- **#166** (NEW): harness hang hardening — line-buffered Python log
  write, 30s heartbeat, top-level try/except. ~12 LOC; ship BEFORE
  next 35B bootstrap.

### Earlier in this session (still active in prod)

- **27B HTTP smoke PASSED** end-to-end on qb1 (commits `97abfab`,
  `a7ea0fe`, `73fd269`). `/v1/chat/completions` returns
  "The capital of France is Paris." in 2.4s.
- **35B HTTP COHERENT at TT_CB_SLOTS=1** (cb_api default for 35B).
  Sample: "Hello" → "Hello! How can I help you today?" with
  `finish_reason=stop`. 35B B>1 broken (see #162 above).
- **35B first inference step crash fixed** in `39f4663` (cb_scheduler
  3-D readback squeeze).

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
     bootstrap-once long-running python on qb1, watches `~/tt-xla/.cache/cb35_runtime/trig/`,
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
