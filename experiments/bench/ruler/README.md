# RULER long-context benchmark — Gemma 4 12B / 27B / Nemotron-3

Drives NVIDIA RULER (`https://github.com/NVIDIA/RULER`) against our CB OpenAI
server (`experiments/serve/cb_api.py`). Replaces the ad-hoc IT-template-
sensitive needle test with the benchmark every paper-table cites.

Design doc: `research/ruler_gemma4_integration_plan.md`.
Memory note: `reference_ruler_long_context_benchmark.md`.

## Status

**Scaffold only — not yet runnable.** Phase 0 = this directory exists +
points to the design doc + lists the run command shape. Phase 1 wiring (clone
RULER on qb2, smoke a single NIAH task) is the next-session entry point.

## Architecture (when wired)

```
[ scripts/chat.py style HTTP client ]
        |
        v
[ NVIDIA/RULER scripts/pred/call_api.py --server_type openai ]
        |  POSTs /v1/chat/completions  (model, messages, max_tokens, temp, top_p, seed, stop)
        v
[ cb_api.py @ qb2:8000  --  experiments/serve/cb_api.py ]
        |
        v
[ CB engine -> Gemma 4 12B forward -> token stream ]
```

RULER lives at `~/RULER/` on qb2 (cloned once, pinned SHA in this README).
We do not vendor it. Our adapter is the model-config entry in RULER's
`scripts/config_models.sh` plus the wrapper script in `scripts/run_ruler.sh`.

Pinned RULER SHA: `TBD` (set in Phase 1; record here for reproducibility).

## Run (documentation stub — phase 1 wires this up for real)

Prerequisites:

- Gemma 4 12B CB server up on qb2:8000 (`serve_cb.sh start`, `TT_BACKEND=gemma4_12b`).
- RULER cloned to `~/RULER/` on qb2.
- Forwarded tunnel: `ssh -L 8000:localhost:8000 qb2` (only needed for client-
  side validation — the RULER run itself happens on qb2 against localhost).

```bash
# v0.5 accept gate (3 tasks x 4 lengths x 50 samples) ~3h wall
bash scripts/run_ruler.sh gemma4_12b v0_5_accept

# full sweep (13 tasks x 6 lengths x 500 samples) ~24h — perf-round only
bash scripts/run_ruler.sh gemma4_12b v0_5_bench_full

# smoke (1 task x 1 length x 10 samples) ~3min
bash scripts/run_ruler.sh gemma4_12b v0_5_accept_smoke
```

Under the hood:

```bash
TT_BACKEND=gemma4_12b \
OPENAI_API_BASE=http://localhost:8000/v1 \
OPENAI_API_KEY=dummy \
bash ~/RULER/scripts/run.sh gemma4_12b synthetic
```

Outputs land in `~/RULER/benchmark_root/gemma4_12b/synthetic/{4096,8192,...}/`
(RULER's layout). Our `_runner.py` mirrors the score JSON into
`experiments/bench/ruler/results/gemma4_12b/<YYYY-MM-DD>.json` for repo-
visible diffs across sessions.

## Task set (v0.5 accept)

Defined in `tasks.yaml`. Summary:

- `niah_single_1` — basic single-needle retrieval, noise haystack. Floor task.
- `niah_multikey_2` — needle haystack with 4 keys, 1 numeric value. Hard.
- `vt` — variable tracking, 1-chain x 4-hop. Reasoning task.
- Lengths: 4096, 8192, 16384, 32768.
- Samples: 50/task/length.

Acceptance numbers (from design doc):

- `niah_single_1 @ 4k ≥ 0.85`
- `niah_multikey_2 @ 16k ≥ 0.50`
- `vt @ 8k ≥ 0.30`

## Other backends

Same wrapper, different `TT_BACKEND`:

```bash
bash scripts/run_ruler.sh 27b v0_5_accept           # Qwen3.6-27B
bash scripts/run_ruler.sh nemotron3_nano v0_5_accept # Nemotron-3 Nano 30B-A3B
```

Three-model A/B/C is the standing model-quality gate going forward.

## Why this lives in `experiments/bench/`

Mirrors `experiments/cb/bench/` (perf benches) and `experiments/cb/load/`
(SLO benches). RULER is the correctness/long-context bench — a third sibling.
