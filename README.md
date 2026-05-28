# tt-model-bringup

Tenstorrent Blackhole LLM bringup for the Qwen3.6 family — direct TT-Metal
model graphs plus custom owned_* compute kernels.

> **Pivot note.** This repo was originally scoped as a JAX/XLA PJRT backend
> (hence the legacy `tt-xla` directory name). The work pivoted to direct
> TT-Metal model bringup + custom compute kernels for Qwen3.6 on Tenstorrent
> Blackhole. The PJRT plugin sources under `pjrt_plugin/` are retained for
> reference but are not on the active path. Current production:
> Qwen3.6-27B on qb2 4× P150 mesh @ 12.93 tok/s with custom `owned_gdn` and
> `owned_decay_gate` kernels. Upcoming: Qwen3.6-35B-A3B MoE bringup.

GitHub: `aweditya/tt-model-bringup`.

---

## Quickstart

On a host with Tenstorrent P150s and a tt-metal build (see [Setup](#setup)):

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla && cd ~/tt-xla
make setup                              # uv sync — Python deps into .venv
# (one-time: build tt-metal + owned_ops kernels + set HF_TOKEN — see Setup)
make run PY=experiments/cb_validate_27b.py     # run a script on the TT host
```

`make help` lists targets. Device runs go through `scripts/run_remote.sh`
(the single source of truth for the ttnn env + mesh reset); set `TT_HOST=qb2`
to target the 4-chip box. `make dr PY=...` deploys then runs (the edit loop).

---

## Host matrix

There are two reference hosts; the right host to use depends on the demo.

| Host  | Hardware                | Inter-chip fabric | Use it for                                |
|-------|-------------------------|-------------------|-------------------------------------------|
| `qb1` | 4× Blackhole P150       | **No fabric**     | Demo A (single-chip serve)                |
| `qb2` | 4× Blackhole P150       | **FABRIC_1D**     | Demo B (4-chip Tensor-Parallel serve)     |

Demo B (`server_tp.py`) calls `ttnn.set_fabric_config(FABRIC_1D)` during
bootstrap and will **hang on qb1**. Stick to the matrix above.

---

## Setup

### 1. Build TT-Metal from source

Follow [Tenstorrent's TT-Metal install guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
for Blackhole. Once the build completes, export:

```bash
export TT_METAL_HOME=$HOME/tenstorrent/tt-metal
export TT_BUILD_DIR=$TT_METAL_HOME/build_Release   # or your build variant (e.g. build_tracy_gcc12_nodist)
export ARCH_NAME=blackhole
```

The exact `TT_BUILD_DIR` name depends on which CMake preset you ran
(`./build_metal.sh` defaults to `build_Release` on Blackhole; profiler
builds produce `build_tracy_gcc12_nodist`). Both qb1 and qb2 currently
use `build_Release`.

> **TODO (follow-up PR):** pin a known-good `tt-metal` SHA here. Today's
> `ttnn` build on qb2 is the reference; the SHA will be captured by a
> `scripts/setup.sh` aggregator in the next reproducibility PR.

### 2. Build owned_ops kernels

Each custom kernel under `experiments/owned_ops/<name>/` ships an
`INTEGRATION.md` plus an `integrate_into_ttmetal.py` script. Run the
integration step for every kernel, then **rebuild `ttnn`** so the new C++
ops are registered.

Kernels currently shipped (all required for production decode):

- `experiments/owned_ops/qwen36_gdn_decode_owned/INTEGRATION.md`
- `experiments/owned_ops/qwen36_gdn_prediction/INTEGRATION.md`
- `experiments/owned_ops/qwen36_gdn_delta/INTEGRATION.md`
- `experiments/owned_ops/qwen36_gdn_decay_state/INTEGRATION.md`
- `experiments/owned_ops/qwen36_gdn_outer_update/INTEGRATION.md`
- `experiments/owned_ops/qwen36_gdn_output/INTEGRATION.md`
- `experiments/owned_ops/qwen36_decay_gate_decode_owned/INTEGRATION.md`
- `experiments/owned_ops/qwen36_conv1d_decode_owned/INTEGRATION.md`

> **TODO (follow-up PR):** `scripts/build_owned_ops.sh` will loop over all
> eight kernels and rebuild `ttnn` once at the end.

### 3. Install Python dependencies with `uv`

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and provisions `.venv/` with the
pinned versions of `torch`, `transformers`, `huggingface_hub`,
`safetensors`, and `numpy`.

**You must then install `ttnn` (Tenstorrent's Python runtime) into the
same venv** — `pyproject.toml` cannot pin it because it has no PyPI
release. Use the editable install from your tt-metal checkout:

```bash
# build deps for ttnn's setup.py (one-time)
uv pip install setuptools_scm

# install ttnn + its runtime deps (loguru, pandas, seaborn, graphviz, ml-dtypes, tracy, …)
uv pip install -e $TT_METAL_HOME --no-build-isolation
```

The `-e` install registers a `ttnn` package in `.venv/lib/.../site-packages/`
that points at `$TT_METAL_HOME/ttnn/ttnn/`. A bare `sys.path.insert(...,
$TT_METAL_HOME/ttnn)` is **not** sufficient — ttnn's `__init__.py` imports
the bundled `tracy` profiler module, which only ships via the
`uv pip install -e .` path.

Verified 2026-05-21 on qb1: a fresh `uv sync` resolves to
`torch==2.12.0+cu130` (qb1 prod runs `torch==2.11.0+cu130`); the prebuilt
`ttnn==0.69.0` C extension imports cleanly against both — no torch ABI
break across that minor bump.

### 4. Configure HuggingFace access (HF_TOKEN)

You need a HuggingFace token to download the Qwen3.6 weights — bootstrap
dies in `AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")` without it.
qb1 and qb2 currently have **no token configured**, and the first cold
download is rate-limited on the anonymous path.

You have two equivalent options. Pick one.

**Option A — `hf auth login` (recommended).**
The `hf` CLI ships as part of the `huggingface_hub` Python package, so
it is installed automatically by `uv sync`. No extra install step is
needed. (Note: `huggingface-cli` is the legacy name and is deprecated as
of `huggingface_hub` ≥1.10 — `hf` is its drop-in replacement.)

```bash
uv sync
uv run hf auth login    # writes ~/.cache/huggingface/token
```

`transformers` and `huggingface_hub` both auto-detect that token; no
environment variable is required afterwards.

**Option B — `.env` file with `HF_TOKEN`.**

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_…
```

Then `source .env` (or use your shell's auto-loader) before running
`serve.sh` / `serve_tp.sh`.

---

## Demo A — single P150 on qb1

```bash
ssh qb1
cd ~/tt-xla
bash experiments/serve/scripts/serve.sh start   # ~11 min bootstrap

# Wait for bootstrap to finish (server pings the Unix socket when ready).
tail -f .cache/server.log

# Once "server ready" appears in the log:
uv run python -m experiments.serve.client status
uv run python -m experiments.serve.client generate \
    --prompt "The capital of France is" --max-tokens 32

# Graceful shutdown:
bash experiments/serve/scripts/serve.sh stop
```

Reference: `experiments/serve/scripts/serve.sh`,
`experiments/serve/server.py`, `experiments/serve/client.py`.

## Demo B — 4× P150 Tensor-Parallel on qb2

```bash
ssh qb2
cd ~/tt-xla
bash experiments/serve/scripts/serve_tp.sh start   # ~17 min bootstrap

# Wait for bootstrap to finish:
tail -f .cache/server_tp.log

# Once ready:
uv run python -m experiments.serve.client_tp status
uv run python -m experiments.serve.client_tp generate_tp \
    --prompt "The capital of France is" --max-tokens 32

# Graceful shutdown (important — see Troubleshooting below):
bash experiments/serve/scripts/serve_tp.sh stop
```

Reference: `experiments/serve/scripts/serve_tp.sh`,
`experiments/serve/server_tp.py`, `experiments/serve/client_tp.py`.

Steady-state throughput on qb2 as of 2026-05-20: **12.93 tok/s** with
`num_links=2` all_reduce + `owned_gdn` + `owned_decay_gate` kernels.

---

## Verified demos

End-to-end client runs against the persistent servers on 2026-05-21,
prompt `"The capital of France is"`, `--max-tokens 32`. Both demos
generate the literal token `Paris` and continue coherently into a
`<think>...</think>` block followed by a short factual answer.

| Demo | Host | Prefill | Decode | Total wall |
|------|------|---------|--------|------------|
| A — Qwen3.6-27B single P150 | qb1 | 1926 ms (5 prompt tokens) | 194.63 ms/tok = **5.14 tok/s** | 9.6 s |
| B — Qwen3.6-27B 4× P150 TP | qb2 | 382 ms (5 prompt tokens) | 77.02 ms/tok = **12.98 tok/s** | 2.9 s |

Demo A first 32 generated tokens (qb1):

```
 Paris.

<think>

</think>

That is correct. Paris is the capital and most populous city of France. It is located in the north-central part of the
```

Demo B first 32 generated tokens (qb2):

```
 Paris.

<think>

</think>

That is correct. **Paris** is the capital and most populous city of France. It is located in the north-central part
```

Demo B's measured 12.98 tok/s matches the 2026-05-20 steady-state
12.93 tok/s within run-to-run variance. Demo A's 5.14 tok/s is consistent
with the qb1 single-chip baseline (5.19 tok/s in the QK-rms_norm-shipped
memory note). Both servers had been bootstrapped before measurement, so
the numbers reflect steady-state traced decode (not cold-start).

### Legacy 8B-era demos (re-verified on qb1, 2026-05-21)

`REPRODUCE.md` documents six experiments from the pre-pivot Llama /
Qwen2.5 era. All six were authored on the now-deprecated `ssh tenstorrent`
host (per `CLAUDE.md` non-negotiable #4) and were **re-verified on qb1**
on 2026-05-21 against the current `~/tt-xla/.venv` install
(`torch==2.11.0+cu130`, `ttnn==0.69.0`, firmware 19.6.0). All six PASS
within 2-7% of the historical baselines:

| Script | Baseline | qb1 (2026-05-21) |
|--------|----------|------------------|
| `experiments/60_native_rope_decode.py` | 140 tok/s | **142.2 tok/s** |
| `experiments/64_llama32_1b_port.py`    |  78 tok/s | **78.6 tok/s**  |
| `experiments/67_llama32_3b_port.py`    |  34 tok/s | **33.7 tok/s**  |
| `experiments/73_llama8b_instruct.py`   |  19 tok/s | **19 tok/s**    |
| `experiments/76b_8b_correctness_check.py` | cos > 0.997, 8/8 | cos **0.997327**, **8/8** |
| `experiments/80_8b_diverse_qa_demo.py` | 18 tok/s, 9/10 EOS | **18 tok/s, 9/10 EOS** |

See `REPRODUCE.md` for the run recipe (stop the prod `serve.sh` first
so device 0 is free), and `~/tt-xla/.cache/legacy_demos_2026_05_21/`
on qb1 for the captured stdout logs. The fresh public-clone setup
recipe (Setup steps 3+4 above) was also re-verified the same day:
anonymous `git clone` → `uv sync` →
`uv pip install -e $TT_METAL_HOME --no-build-isolation` produces a
working `.venv/` with `ttnn==0.69.0`, and the fresh-clone
`experiments.serve.client generate` talks cleanly to the persistent
qb1 prod server's Unix socket and decodes 16 tokens at 5.12 tok/s.

### Experienced-user shortcut

If you already have a working `.venv/` on qb1/qb2 with `torch`,
`transformers`, `huggingface_hub`, `safetensors`, and `numpy` installed,
you can skip the `uv sync` step in Setup. The serve scripts only need
`$VENV_PY` to point at an executable Python interpreter (default
`$PROJECT_ROOT/.venv/bin/python`); they do **not** invoke `uv`
themselves. Both Demo A and Demo B above were measured on hosts whose
`.venv/` pre-dates the new `pyproject.toml` and they ran without any
`uv sync` step.

---

## Long context

As of 2026-05-21, the qb2 4-chip TP path is validated at **L=4000** with
**verbatim 8-char needle-haystack retrieval** at all needle positions
(0.25/0.5/0.75 frac), using the B3 HiFi2 SDPA recipe. Sweep history:
L=460 (6/6), L=1024 (6/6), L=1990 (6/6), L=4000 (3/3), L=8000 (in
flight). Probe: `experiments/utils/needle_haystack_qb2_tp.py` (the
predecessor qb1 single-chip probe lives in
`experiments/utils/needle_haystack_b3_probe.py`).

---

## Troubleshooting

- **`server_tp.sh start` hangs on qb1.** qb1 has no inter-chip fabric.
  Use qb2 for any multi-chip workload.
- **Server hangs or fabric is wedged after a hard-kill.** Reset the
  devices and retry:
  ```bash
  tt-smi -r 0,1,2,3
  bash experiments/serve/scripts/serve_tp.sh start
  ```
  Always prefer `serve_tp.sh stop` (graceful socket shutdown) over
  `kill -9` — hard-killing a mesh process can leave qb2 fabric wedged.
- **`AutoTokenizer.from_pretrained` fails with 401.** No HF token. See
  "Configure HuggingFace access" above.
- **`HF_HOME` fills up `$HOME`.** Override:
  ```bash
  export HF_HOME=/path/with/space/.cache/hf
  ```
- **First cold weight download is slow.** ~6 minutes on qb2 — this is the
  HuggingFace cold-cache hit, not a project bug. A
  `scripts/prefetch_weights.py` helper will be added in a follow-up PR.
- **Legacy 8B-era notes (Llama-3.1-8B, Llama-3.2-1B/3B, Qwen2.5-0.5B).**
  See `REPRODUCE.md`. Those experiments pre-date the Qwen3.6-27B pivot
  and are kept for historical reference only.

---

## Follow-up PRs (per `research/repro_packaging_plan_2026_05_21.md`)

- `scripts/setup.sh` — preflight + `uv sync` aggregator.
- `scripts/build_owned_ops.sh` — loop owned_ops integration + rebuild ttnn.
- `scripts/prefetch_weights.py` — `snapshot_download(Qwen/Qwen3.6-27B)` to
  `$HF_HOME` for offline bootstrap.
- Pin `tt-metal` SHA in `docs/SETUP.md`.
- De-hardcode `~/tt-xla` paths in `experiments/serve/server.py` and
  `server_tp.py`.
