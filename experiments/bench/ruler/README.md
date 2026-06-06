# RULER long-context benchmark — Gemma 4 12B / 27B / Nemotron-3

Drives NVIDIA RULER (`https://github.com/NVIDIA/RULER`) against our CB OpenAI
server (`experiments/serve/cb_api.py`). Replaces the ad-hoc IT-template-
sensitive needle test with the benchmark every paper-table cites.

Design doc: `research/ruler_gemma4_integration_plan.md`.
Memory note: `reference_ruler_long_context_benchmark.md`.

## Status

**Scaffold + RULER cloned on qb2 — smoke BLOCKED by device-lock contention as
of 2026-06-06.** Phase 0 done (directory exists, design doc, run-command shape
documented). Phase 1 partial:

- Cloned `NVIDIA/RULER` → `qb2:~/RULER/`, pinned SHA
  `ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13` (also in `tasks.yaml`).
- Inspected RULER's `OpenAIClient` (`~/RULER/scripts/pred/client_wrappers.py:188`).
  It DOES NOT respect `OPENAI_API_BASE` — `OpenAI(api_key=…)` is called with no
  `base_url`. Needs a ~5-line patch (see "Required RULER patches" below).
- Server bring-up BLOCKED: qb2 has a long-running `experiments/cb/dev/gm4_dev_harness.py`
  (PID 1126437 in tmux session `gm4`, started 2026-06-06 15:44) holding the
  `CHIP_IN_USE_*_PCIe` lock on chips 0..3. Our `serve_cb.sh start` spun on the
  UMD `robust_mutex.cpp:417 Waiting for lock` and was SIGKILLed cleanly (no
  fabric corruption — only mesh-init had started). The dev harness is not
  ours to kill.
- Also surfaced and fixed: qb2's `~/tt-xla/.venv` was missing `fastapi`,
  `uvicorn`, `openai`. Installed via
  `~/.local/bin/uv pip install --python ~/tt-xla/.venv/bin/python "fastapi>=0.115" "uvicorn[standard]>=0.30" openai`.
  Versions landed: fastapi 0.136.3, uvicorn 0.49.0, openai 2.41.0. Next session
  does NOT need to redo this. NOTE: qb2's `~/tt-xla` is not a git repo (rsync'd
  from local), so `uv sync --extra serve` doesn't work — manual `uv pip install`
  into the venv is the only path until the host is re-bootstrapped.

Next-session entry point:

1. Confirm the gm4 dev harness is no longer needed (check with the user, or
   that tmux session `gm4` is empty: `ssh qb2 'tmux ls'`).
2. `ssh qb2 'tmux kill-session -t gm4'` (releases the device).
3. Verify chip locks clear: `ssh qb2 'ls /tmp/*CHIP_IN_USE* 2>/dev/null'`
   (lock files come and go; alternatively just retry start).
4. Start server (uvicorn deps are already installed):
   `ssh qb2 'cd ~/tt-xla && TT_BACKEND=gemma4_12b TT_GEMMA4_VARIANT=it bash experiments/serve/scripts/serve_cb.sh start'`
5. Wait ~14 min for bootstrap (`tail -f ~/tt-xla/.cache/server_cb.log`).
6. Apply the `OpenAIClient` patch (see below), then run smoke.

## Required RULER patches

RULER's `OpenAIClient` (`scripts/pred/client_wrappers.py:188`) needs minimal
edits to work against our `cb_api.py` HTTP server:

1. `_create_client` does `OpenAI(api_key=…)` with no `base_url`. Add
   `base_url=os.environ.get("OPENAI_API_BASE")` so `OPENAI_API_BASE=http://localhost:8000/v1`
   is honoured. (Upstream OpenAI Python SDK also respects this env var
   automatically — verify behaviour, may already be a no-op fix.)
2. `model2length` (`client_wrappers.py:194`) is OpenAI/Azure-only. Either:
   - add a `model_name.lower().startswith(("google/", "qwen/", "nvidia/"))`
     fallback that returns 131072, OR
   - read it from `OPENAI_MODEL_MAX_TOKENS` env var.
3. `_count_tokens` uses tiktoken `cl100k_base` — wrong for Gemma 4 but only
   used to clamp `tokens_to_generate`. Safe to leave (over-estimate is fine
   since RULER caps decode at ~150 tokens for NIAH).

Patches will land at `experiments/bench/ruler/patches/0001-openai-base-url.patch`
and be applied via `git -C ~/RULER apply` in our wrapper script.

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
