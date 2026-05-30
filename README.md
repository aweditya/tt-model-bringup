# tt-model-bringup

Tenstorrent Blackhole LLM bringup for the Qwen3.6 family — direct TT-Metal
model graphs plus custom owned_* compute kernels.

> **Pivot note.** This repo was originally scoped as a JAX/XLA PJRT backend
> (hence the legacy `tt-xla` directory name). The work pivoted to direct
> TT-Metal model bringup + custom compute kernels for Qwen3.6 on Tenstorrent
> Blackhole. The PJRT plugin sources under `archive/legacy/pjrt_plugin/` are
> retained for reference but are not on the active path. Current production:
> Qwen3.6-27B on qb2 4× P150 mesh @ 12.93 tok/s with custom `owned_gdn` and
> `owned_decay_gate` kernels; Qwen3.6-35B-A3B MoE bringup is underway (see
> `HANDOFF.md`).

GitHub: `aweditya/tt-model-bringup`.

---

## Quickstart

On a host with Tenstorrent P150s and a tt-metal build (see [Setup](#setup)):

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla && cd ~/tt-xla
make setup                              # uv sync — Python deps into .venv
# (one-time: build tt-metal + owned_ops kernels + set HF_TOKEN — see Setup)
make run PY=experiments/cb/validate/forward.py   # run a script on the TT host
```

`make help` lists targets. Device runs go through `scripts/run_remote.sh`
(the single source of truth for the ttnn env + mesh reset); set `TT_HOST=qb2`
to target the 4-chip box. `make dr PY=...` deploys then runs (the edit loop).

---

## Host matrix

There are two reference hosts. Both have working inter-chip fabric and can run
either demo; the split below is operational — keep qb2's TP server up as the
"production" path and use qb1 for experimentation.

| Host  | Hardware                | Inter-chip fabric    | Use it for                                |
|-------|-------------------------|----------------------|-------------------------------------------|
| `qb1` | 4× Blackhole P150       | **FABRIC_1D** ✓      | Experimental TP / CB / new kernels        |
| `qb2` | 4× Blackhole P150       | **FABRIC_1D** ✓      | Production Demo B (single-seq TP serve)   |

qb1's fabric works as of **2026-05-21**; both single-chip and multi-chip
workloads run there. Earlier versions of this README claimed "no fabric on qb1"
— stale; ignore.

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

**tt-metal SHA pin**: this repo targets the tt-metal SHA in
[`tt-metal-sha.txt`](tt-metal-sha.txt) (the build qb1 + qb2 run). The
owned_ops integrate scripts (`experiments/owned_ops/*/integrate_into_ttmetal.py`)
are layout-sensitive; on a mismatched tt-metal SHA, `scripts/build_owned_ops.sh`
prints a loud warning and the patches may fail to apply. After checking out
tt-metal, verify with `git -C $TT_METAL_HOME rev-parse HEAD`.

### 2. Build owned_ops kernels

The 27B serving path calls two custom TT-NN ops (`qwen36_gdn_decode_owned`,
`qwen36_decay_gate_decode_owned`). Install them into your tt-metal checkout and
rebuild `ttnn` with the orchestrator — **run it on the TT host**:

```bash
scripts/build_owned_ops.sh            # install the 2 production ops + rebuild ttnn
scripts/build_owned_ops.sh --all      # install every owned op (matches qb1/qb2)
scripts/build_owned_ops.sh --dry-run  # preview the source changes
```

See [`experiments/owned_ops/README.md`](experiments/owned_ops/README.md) for the
full op index (GDN sub-ops, experimental kernels, the JIT compute-kernel patch
that needs no rebuild) and each op's `INTEGRATION.md` for its validation gate.

### 3. Install Python dependencies with `uv`

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and provisions `.venv/` with the
pinned versions of `torch`, `transformers`, `huggingface_hub`,
`safetensors`, `numpy`, plus ttnn's pure-Python runtime deps (`loguru`,
`pandas`, `seaborn`, `graphviz`, `pyyaml`, `click`, `networkx`,
`ml_dtypes`) declared so `uv sync` doesn't strip them from the venv.

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

### Legacy multi-model demos

Smaller models brought up before the Qwen3.6 pivot (Llama 1B/3B/8B, SmolLM3,
Qwen2.5/3-small), re-verified on qb1 2026-05-21 within 2-7% of baseline. Index:
[`models/`](models/README.md); run recipes: `REPRODUCE.md` (stop the prod server
first so device 0 is free).

| Script | Baseline | qb1 (2026-05-21) |
|--------|----------|------------------|
| `models/60_native_rope_decode.py` | 140 tok/s | **142.2 tok/s** |
| `models/64_llama32_1b_port.py`    |  78 tok/s | **78.6 tok/s**  |
| `models/67_llama32_3b_port.py`    |  34 tok/s | **33.7 tok/s**  |
| `models/73_llama8b_instruct.py`   |  19 tok/s | **19 tok/s**    |
| `models/76b_8b_correctness_check.py` | cos > 0.997, 8/8 | cos **0.997327**, **8/8** |
| `models/80_8b_diverse_qa_demo.py` | 18 tok/s, 9/10 EOS | **18 tok/s, 9/10 EOS** |

---

## OpenAI-compatible endpoint

A host-side HTTP proxy (`experiments/serve/openai_endpoint.py`) exposes
`/v1/chat/completions` + `/v1/completions` over the persistent TP server (greedy
by default; `temperature`/`top_p`/`top_k` use the server's sampler). On the TT host:

```bash
bash experiments/serve/scripts/serve_tp.sh start      # the model server (~17 min bootstrap)
uv sync --extra serve                                 # fastapi + uvicorn
uv run --extra serve uvicorn experiments.serve.openai_endpoint:app --host 0.0.0.0 --port 8000
# then, e.g.:
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"capital of France?"}],"temperature":0.7,"stream":true}'
```

It applies the model's chat template to `messages`, forwards `generate_tp` over the
Unix socket, and streams OpenAI SSE (or returns one JSON for `stream:false`). The
translation helpers are unit-tested (`experiments/serve/tests/test_openai_endpoint.py`).

---

## Long context

As of 2026-05-21, the qb2 4-chip TP path is validated at **L=4000** with
**verbatim 8-char needle-haystack retrieval** at all needle positions
(0.25/0.5/0.75 frac), using the B3 HiFi2 SDPA recipe. Sweep history:
L=460 (6/6), L=1024 (6/6), L=1990 (6/6), L=4000 (3/3), L=8000 (in
flight). Probe: `experiments/utils/needle_haystack_qb2_tp.py` (the
predecessor qb1 single-chip probe lives in
`experiments/utils/archive/needle_haystack_b3_probe.py`).

---

## Troubleshooting

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
- **First cold weight download is slow.** ~6 minutes on qb2 — the HuggingFace
  cold-cache hit, not a project bug.
- **Legacy 8B-era notes (Llama-3.1-8B, Llama-3.2-1B/3B, Qwen2.5-0.5B).**
  See `REPRODUCE.md`. Those experiments pre-date the Qwen3.6-27B pivot
  and are kept for historical reference only.
