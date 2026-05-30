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
bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready
bash experiments/serve/scripts/serve_cb.sh status
bash experiments/serve/scripts/serve_cb.sh stop    # SIGTERM → graceful drain → mesh release
```

Env knobs: `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64`. Over-cap requests → HTTP 429.

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/metrics` (Prometheus).
See README §"Chat server (production)" for `curl` + `openai` client examples.

## What's next

**S2 — chunked prefill** (productionization step 1). Today CB prefills one
token per scheduler iteration; chunked prefill processes the prompt in C-token
chunks for realistic TTFT + end-to-end task times. WIP on branch
`feature/chunked-prefill`. Plan: [`research/27b_chunked_prefill_plan.md`](research/27b_chunked_prefill_plan.md).

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
