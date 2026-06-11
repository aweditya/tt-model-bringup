# HANDOFF — cold-start one-pager

Read top to bottom. Everything else is linked.

## Table of contents

- [Demo (2026-06-10)](#demo-2026-06-10)
- [Where the perf is now](#where-the-perf-is-now)
- [Hardware ceiling](#hardware-ceiling)
- [Production chat path](#production-chat-path)
- [Deploy hygiene](#deploy-hygiene)
- [Open workstreams](#open-workstreams)
- [Read order when resuming work](#read-order-when-resuming-work)
- [Load-bearing rules](#load-bearing-rules)
- [Workflow](#workflow)

---

## Demo (2026-06-10)

Two backends live in parallel; pick whichever is hot.

| Host | Model | Demo path | Status |
|---|---|---|---|
| qb1 | Gemma 4 12B IT | OpenAI HTTP (multi-client) → `scripts/chat.py --tools` | READY |
| qb2 | Nemotron-3 Nano 30B-A3B | dev harness → trigger probes | bringup in flight |

### qb1 chat demo (primary)

```bash
# server — TT_CB_SLOTS=1 mandatory for correctness today (#313)
ssh qb1 'TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it \
        TT_CB_CHUNKED_PREFILL=1 TT_CB_SLOTS=1 TT_CB_TOPK_K=0 \
        bash ~/tt-xla/experiments/serve/scripts/serve_cb.sh start'

# tunnel + chat client (laptop, after /health returns 200)
ssh -L 8000:localhost:8000 qb1
python3 scripts/chat.py --tools
```

**Why slots=1, not 4?** `paged_fill_cache(..., batch_idx=0)` is hardcoded
at `server_gemma4_unified_ttnn.py:2556-2557, 2616-2617` — every chunked
prefill K/V write lands in slot 0's pages regardless of which CB slot the
request targets. Slots 1+ decode from empty pages. Bug **#313**, fix in
flight.

### qb2 Nemotron-3 backup (not ready)

Weights downloading (~60 GB). Once landed:
```bash
bash scripts/run_harness_tmux.sh nm3 qb2     # ~108s bootstrap
ssh qb2 'touch ~/tt-xla/.cache/nm3_runtime/trig/v033_nstep_chain_smoke'
```
Warm decode ~260 ms/tok eager (~3.8 tok/s) post-vocab-shard. No HTTP server;
harness is the demo surface.

### Long-decode collapse — kernel-side bug confirmed (#314)

Gemma 4 chat collapses to `####` / `***` ~100-200 tokens in. HF reference
(`experiments/utils/hf_long_decode_gemma4_it.py`) produces coherent
300-token output with identical seed/temperature → model is fine, our
impl drifts.

Ladder verdict (`gemma4_long_decode_vs_hf_ladder.py`):
- first cos<0.99 at step=0 layer=0 cos=0.913 (not bf16 drift — too early)
- per-layer hidden cos near zero (0.05-0.2) for layers 1-47 — TT hidden
  orthogonal to HF
- argmax-match 58/99 (58.5%) — top-1 still tracks because logits have
  sharp peaks, but state is wrong

Hypothesis: layer 0 RoPE/Q/K projection diverges at decode time. Next:
extend `gemma4_chunked_prefill_ladder.py` per-sub-op pattern to decode.

---

## Where the perf is now

| Path | Number | Source |
|---|---|---|
| 27B TP single-seq (traced) | **12.93 tok/s** (77 ms/tok) | `serve_tp` on qb2 |
| 27B CB B=32 (traced, aggregate) | **150.5 tok/s** (11.6×) | `experiments/cb/bench/trace.py` |
| 27B CB B=64 (traced, shift-acc conv1d) | **593 tok/s** (45.8×) | same |
| 35B-A3B traced decode (qb1) | **81.16 ms/tok = 12.32 tok/s** | post A002–A009 |
| Gemma 4 12B traced decode | **47.5 ms/tok** (21.05 tok/s) | post P1 vocab-shard |
| Gemma 4 12B chunked prefill L=2048 | **100×** speedup (2.9 s vs 294 s) | #290 P2 |

CB SLO (qb1, 8 clients × 60 s): 0 errors / 36 requests / **TTFT p99 = 176 ms**.

## Hardware ceiling

P150: **404 GB/s/chip** DRAM BW, 110 worker cores, 31.81 GB DRAM. For
35B-A3B with ~3 GB active params/token/chip: bf16 floor 3.7 ms/tok →
**270 tok/s ceiling**; bf8 floor 1.85 ms/tok → **540 tok/s ceiling**.

The target is the hardware ceiling, not parity with anyone else's number.

---

## Production chat path

```bash
# Gemma 4 12B IT (CURRENT PRIMARY)
TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it TT_CB_CHUNKED_PREFILL=1 \
  TT_CB_SLOTS=1 bash experiments/serve/scripts/serve_cb.sh start

# 27B (default backend)
TT_CB_CHUNKED_PREFILL=1 TT_CB_PREFIX_CACHE=1 \
  bash experiments/serve/scripts/serve_cb.sh start

# 35B-A3B MoE
TT_BACKEND=35b TT_CB_SLOTS=2 TT_CB_TOPK_K=64 \
  bash experiments/serve/scripts/serve_cb.sh start

bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # graceful drain
```

Knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics`, `/bootstrap`. Tail
`~/tt-xla/.cache/server_cb.bootstrap.log` to watch the 6-14 min boot.

## Deploy hygiene

Always sync `experiments/serve/*.py` to the host before `serve_cb.sh start`:
```bash
bash scripts/deploy.sh experiments/serve/*.py
```
The dev harness reloads on every test; the production server reads the
filesystem ONCE at boot. One MM1 commit sat in local git but never reached
qb1 — hours of debugging until caught. See `[[deploy-serve-files-too]]`.

---

## Open workstreams

| ID | Scope | Status |
|---|---|---|
| #290 | Gemma 4 chunked prefill (L=128/2048/4032) | P1–P3 + P5 HTTP SHIPPED; P4 trace deferred |
| #313 | Gemma 4 multi-slot K/V routing bug (slots=1 today) | 4 root causes localised, fix in flight |
| #314 | Gemma 4 long-decode coherence ladder | systematic bug at layer 0 decode RoPE/Q/K |
| #307 | Tool calls via `<\|tool_call\|>` token | `scripts/chat.py --tools` ready, untested live |
| Nemotron-3 | Mamba2 SSD owned kernel + bringup | G2 multi-core PASS; v0.5 perf in flight |

Plan docs:
- `research/gemma4_chunked_prefill_plan_2026-06-08.md`
- `research/gemma4_step2_fp32_acc_plan_2026-06-09.md`
- `research/gemma4_layout_op_elimination_plan_2026-06-08.md`
- `research/nemotron3_nano_30b_a3b_bringup_plan.md`
- `research/diffusiongemma_bringup_scope_2026-06-10.md`
- `research/agentic_harness_scope_2026-06-10.md`

---

## Read order when resuming work

1. This file.
2. [`README.md`](README.md) — install + demos.
3. [`research/model_bringup_recipe.md`](research/model_bringup_recipe.md) — the staging ladder.
4. [`research/27b_cb_scope.md`](research/27b_cb_scope.md) — CB design + numbers.
5. The active plan doc for whichever workstream you're picking up.

## Load-bearing rules

Each cost a multi-day debug; never skip.

- **View-decay**: `ttnn.slice` / `ttnn.reshape` return views. Never
  `ttnn.deallocate` the source while a view is live; clone when in doubt.
- **+1 zero-centered RMSNorm offset** on `q_norm` / `k_norm` /
  `input_layernorm` / `post_attention_layernorm` / `final_norm` (Qwen3.6).
- **K-broadcast RoPE workaround** in the SDPA path — sidesteps a ttnn
  `[1, HEAD_DIM]` slice/concat bug.
- **bf16 KV cache** required by paged SDPA (fp32 hard-rejected).
- **HiFi4 + `fp32_dest_acc_en`** on every matmul (the 91f recipe); mixing
  fidelities corrupts ops silently on Blackhole.
- **Gemma 4 SDPA `scale=1.0`** (NOT `1/sqrt(d_k)`). Wrong scale is masked
  at pos 0; surfaces at pos > 0. See [[feedback-gemma4-sdpa-scale-1]].
- **TP weight concat-fuse order**: shard each weight first, then concat
  per-chip. Naive concat-then-shard gives chip 0 the first 2048 cols of Q
  only (no K, no V). See [[feedback-tp-concat-fuse-order]].

## Workflow

- **Profile-driven only.** Cite a Tracy or tt-perf-report number for any
  optimization claim. Frame deltas as Δ from BW floor.
- **Correctness gate**: 5-token Paris (`"The capital of France is" → " Paris..."`)
  on prefill IDs `[2614, 314, 279, 369, 11751]`.
- **Remote-only execution** (`ssh qb1` / `ssh qb2`); no device code
  locally. No `python -c`; no `/tmp`; permanent files only.
- **Bug-finding technique that wins**: teacher-forced per-sub-op ladder
  (capture per-sub-op output, cosine against HF reference, find the first
  divergence). Use it first when any forward is broken at pos > 0.
- **Dev harness** for iteration. `scripts/run_harness_tmux.sh` keeps the
  model resident; trigger via `touch trig/<name>`. Per-iter ~5-10 s vs
  6-14 min per `serve_cb.sh` restart.
