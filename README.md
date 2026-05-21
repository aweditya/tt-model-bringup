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
export TT_BUILD_DIR=$TT_METAL_HOME/build_tracy_gcc12_nodist   # or your variant
export ARCH_NAME=blackhole
```

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

### 4. Configure HuggingFace access (HF_TOKEN)

You need a HuggingFace token to download the Qwen3.6 weights — bootstrap
dies in `AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")` without it.
qb1 and qb2 currently have **no token configured**, and the first cold
download is rate-limited on the anonymous path.

You have two equivalent options. Pick one.

**Option A — `huggingface-cli login` (recommended).**
The `huggingface-cli` binary ships as part of the `huggingface_hub`
Python package, so it is installed automatically by `uv sync`. No extra
install step is needed.

```bash
uv sync
uv run huggingface-cli login    # writes ~/.cache/huggingface/token
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
