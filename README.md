# tt-model-bringup

Tenstorrent Blackhole LLM bringup for the Qwen3.6 family — direct TT-Metal
model graphs plus custom `owned_*` compute kernels. Production paths:
Qwen3.6-27B dense on (1, 4) P150 mesh; Qwen3.6-35B-A3B MoE in-progress
(see [`HANDOFF.md`](HANDOFF.md)).

> The repo was originally scoped as a JAX/XLA PJRT backend (hence the legacy
> `tt-xla` directory name). The pivoted work targets TT-Metal directly; the
> PJRT plugin sources live in `archive/legacy/pjrt_plugin/` and are not on the
> active path.

GitHub: `aweditya/tt-model-bringup`.

---

## Quickstart

On a host with Tenstorrent P150s and a tt-metal build (see [Setup](#setup)):

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla && cd ~/tt-xla
make setup                              # uv sync — Python deps into .venv
make install-ttnn                       # editable ttnn from $TT_METAL_HOME
make check                              # sanity-check the setup (no device open)
make kernels                            # build the owned_ops custom kernels
bash experiments/serve/scripts/serve_cb.sh start    # boot the chat server (~6 min)
```

Then talk to it: `curl http://localhost:8000/v1/chat/completions ...`
(see [Chat server](#chat-server-production) for full examples + OpenAI client).

`make help` lists targets. Device runs go through `scripts/run_remote.sh`
(the single source of truth for the ttnn env + mesh reset); set `TT_HOST=qb2`
to target the 4-chip box. `make dr PY=...` deploys then runs (the edit loop).

---

## Host matrix

| Host  | Hardware                | Inter-chip fabric    | Use it for                                |
|-------|-------------------------|----------------------|-------------------------------------------|
| `qb1` | 4× Blackhole P150       | **FABRIC_1D** ✓      | Experimental TP / CB / new kernels        |
| `qb2` | 4× Blackhole P150       | **FABRIC_1D** ✓      | Production Demo B (single-seq TP serve)   |

Both hosts can run either demo. Operational split: keep qb2's TP server up as
the production path; use qb1 for experimentation.

---

## Setup

### 1. Build TT-Metal from source

Follow [Tenstorrent's TT-Metal install guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
for Blackhole. Export:

```bash
export TT_METAL_HOME=$HOME/tenstorrent/tt-metal
export TT_BUILD_DIR=$TT_METAL_HOME/build_Release   # or build_tracy_gcc12_nodist for profiler builds
export ARCH_NAME=blackhole
```

`build_metal.sh` defaults to `build_Release` on Blackhole; both qb1 and qb2 use
`build_Release`.

**tt-metal SHA pin**: this repo targets the SHA in [`tt-metal-sha.txt`](tt-metal-sha.txt)
(the build qb1 + qb2 run). The owned_ops integrate scripts
(`experiments/owned_ops/*/integrate_into_ttmetal.py`) are layout-sensitive;
`scripts/build_owned_ops.sh` warns loudly on a SHA mismatch.

### 2. Build owned_ops kernels

The 27B serving path calls two custom TT-NN ops (`qwen36_gdn_decode_owned`,
`qwen36_decay_gate_decode_owned`). Install them into your tt-metal checkout and
rebuild `ttnn` with the orchestrator — run it on the TT host:

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

Reads `pyproject.toml` + `uv.lock`, provisions `.venv/` with pinned versions of
`torch`, `transformers`, `huggingface_hub`, `safetensors`, `numpy`, plus ttnn's
pure-Python runtime deps (`loguru`, `pandas`, `seaborn`, `graphviz`, `pyyaml`,
`click`, `networkx`, `ml_dtypes`).

Then install `ttnn` (Tenstorrent's Python runtime) into the same venv — it has
no PyPI release. Use the wrapper script:

```bash
make install-ttnn        # or: bash scripts/install_ttnn.sh
```

`install_ttnn.sh` passes `--no-deps` so tt-metal's pyproject does NOT shadow
the `torch` / `transformers` / `numpy` pins from `uv.lock`. It finishes by
printing `ttnn.__file__` plus the two owned-op availability checks
(`qwen36_gdn_decode_owned`, `qwen36_decay_gate_decode_owned`).

Re-run `make install-ttnn` after any `uv sync` (which prunes unmanaged
packages) or after rebuilding tt-metal.

Sanity-check the whole setup before booting the server:

```bash
make check        # or: bash scripts/check_setup.sh
```

Verifies the venv, the tt-metal SHA pin, ttnn imports, the owned kernels are
built in, `tt-smi` sees the devices, and `Qwen/Qwen3.6-27B` is reachable.
No device is opened — safe to run anytime.

Verified torch ABI: a fresh `uv sync` resolves to `torch==2.12.0+cu130` while
qb1 prod runs `torch==2.11.0+cu130`; the prebuilt `ttnn==0.69.0` C extension
imports cleanly against both.

### 4. Configure HuggingFace access (HF_TOKEN)

You need a HuggingFace token to download the Qwen3.6 weights — bootstrap dies
in `AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")` without it.

Pick one of two options:

**Option A — `hf auth login` (recommended).** The `hf` CLI ships as part of
`huggingface_hub` (installed by `uv sync`). `huggingface-cli` is the legacy
name and is deprecated as of `huggingface_hub` ≥ 1.10.

```bash
uv sync
uv run hf auth login    # writes ~/.cache/huggingface/token
```

`transformers` and `huggingface_hub` auto-detect that token; no env var needed.

**Option B — `.env` file with `HF_TOKEN`.**

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_…
```

Then `source .env` (or use your shell's auto-loader) before running
`serve.sh` / `serve_tp.sh` / `serve_cb.sh`.

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

Reference: `experiments/serve/scripts/serve.sh`, `experiments/serve/server.py`,
`experiments/serve/client.py`.

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

Steady-state throughput on qb2: **12.93 tok/s** with `num_links=2` all_reduce +
`owned_gdn` + `owned_decay_gate` kernels.

---

## Verified demos

End-to-end client runs against the persistent servers, prompt
`"The capital of France is"`, `--max-tokens 32`. Both demos generate the literal
token `Paris` and continue coherently into a `<think>...</think>` block followed
by a short factual answer.

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

### Legacy multi-model demos

Smaller models brought up before the Qwen3.6 pivot (Llama 1B/3B/8B, SmolLM3,
Qwen2.5/3-small), re-verified on qb1 within 2-7% of baseline. Index:
[`models/`](models/README.md); run recipes: [`REPRODUCE.md`](REPRODUCE.md)
(stop the prod server first so device 0 is free).

| Script | Baseline | qb1 |
|--------|----------|------|
| `models/60_native_rope_decode.py` | 140 tok/s | **142.2 tok/s** |
| `models/64_llama32_1b_port.py`    |  78 tok/s | **78.6 tok/s**  |
| `models/67_llama32_3b_port.py`    |  34 tok/s | **33.7 tok/s**  |
| `models/73_llama8b_instruct.py`   |  19 tok/s | **19 tok/s**    |
| `models/76b_8b_correctness_check.py` | cos > 0.997, 8/8 | cos **0.997327**, **8/8** |
| `models/80_8b_diverse_qa_demo.py` | 18 tok/s, 9/10 EOS | **18 tok/s, 9/10 EOS** |

---

## Chat server (production)

`experiments/serve/cb_api.py` is the production chat server: an
OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/health`, `/metrics`) running in the **same process** as a
continuous-batching engine (`experiments/serve/cb_engine.py`) over the
validated Orca scheduler (CB1–CB4) and the logits-traced sampling-mode
forward (~125 ms/step at B≤4 on qb1). Multi-client by design — concurrent
requests share the engine's slots and are sampled per-request with their own
`temperature` / `top_p` / `top_k` / `seed`. Slot is freed automatically on
client disconnect (Starlette cancellation → `engine.cancel(rid)`).

### Run it

On the TT host (qb1 or qb2), after `make setup && make install-ttnn && make check`:

```bash
bash experiments/serve/scripts/serve_cb.sh start          # nohup uvicorn, logs in .cache/server_cb.log
# bootstrap ~6 min; /health returns 503 until the engine is up
bash experiments/serve/scripts/serve_cb.sh status         # shows /health code
bash experiments/serve/scripts/serve_cb.sh stop           # SIGTERM → graceful drain → mesh release
```

Knobs (env): `TT_CB_PORT=8000`, `TT_CB_SLOTS=4`, `TT_CB_MAX_NEW=1024`,
`TT_CB_MAX_INFLIGHT=64` (over-cap requests get HTTP 429).

### Talk to it

```bash
# non-stream
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi! What can you do?"}],"max_tokens":200}'

# streaming SSE
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a haiku about silicon."}],"max_tokens":100,"stream":true}'

# sampled (temperature>0 → per-request rng; pass seed for reproducibility)
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Pick a random fruit."}],"max_tokens":50,"temperature":0.8,"top_p":0.95,"seed":7}'
```

From the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="Qwen/Qwen3.6-27B",
    messages=[{"role": "user", "content": "Hi! What can you do?"}],
    max_tokens=300,
)
print(r.choices[0].message.content)
```

### Observability

`GET /metrics` returns Prometheus text exposition: counters
(`cb_requests_{submitted,done,cancelled,rejected}_total`,
`cb_tokens_generated_total`), histograms (`cb_step_seconds`,
`cb_ttft_seconds`, `cb_request_duration_seconds`), gauges
(`cb_engine_{slots_total,slots_active,queue_depth,inflight,max_inflight,sampling}`).
Scrape with any Prometheus-compatible tool; gauge values are sampled at scrape
time.

### Load test

`experiments/cb/load/concurrent_chat.py` fires N concurrent SSE chat clients
for `duration` seconds and reports aggregate tok/s, TTFT p50/p99, request-wall
p50/p99, per-client fairness, and `/metrics` deltas:

```bash
.venv/bin/python -m experiments.cb.load.concurrent_chat \
    --clients 8 --duration 60 --max-tokens 32 --sampling
```

P5 gate (2026-05-30, qb1, 8×60s): 0 errors / 36 requests / 15 tok/s aggregate
/ **TTFT p99=176ms**.

### Tests

- `experiments/serve/tests/test_cb_api_routing.py` — pure routing probe with a
  fake engine; runs in milliseconds, no device. **7/7 PASS**.
- `experiments/cb/validate/engine.py` / `engine_sampling.py` / `engine_api.py`
  — qb1 e2e validators for the engine + sampling + HTTP layer + /metrics.

### Legacy proxy (frozen)

`experiments/serve/openai_endpoint.py` is the older path: a host-side proxy
that forwards `/v1/chat/completions` over the **Unix-socket TP server**
(`experiments/serve/server_tp.py` started via
`experiments/serve/scripts/serve_tp.sh`). Single-seq, no continuous batching.
Kept as the frozen production reference; use `cb_api.py` for new work.

---

## Long context

The qb2 4-chip TP path is validated to **L=4000** with verbatim 8-char
needle-haystack retrieval at all needle positions (0.25 / 0.5 / 0.75 frac),
using the B3 HiFi2 SDPA recipe. Sweep coverage: L=460 / 1024 / 1990 / 4000.
Probe: `experiments/utils/needle_haystack_qb2_tp.py` (the predecessor qb1
single-chip probe lives in `experiments/utils/archive/needle_haystack_b3_probe.py`).

---

## Troubleshooting

- **Server hangs or fabric is wedged after a hard-kill.** Reset the devices
  and retry:
  ```bash
  tt-smi -r 0,1,2,3
  bash experiments/serve/scripts/serve_tp.sh start
  ```
  Always prefer `serve_tp.sh stop` (graceful socket shutdown) over `kill -9` —
  hard-killing a mesh process can leave qb2 fabric wedged.
- **`AutoTokenizer.from_pretrained` fails with 401.** No HF token. See
  "Configure HuggingFace access" above.
- **`HF_HOME` fills up `$HOME`.** Override:
  ```bash
  export HF_HOME=/path/with/space/.cache/hf
  ```
- **First cold weight download is slow.** ~6 minutes on qb2 — the HuggingFace
  cold-cache hit, not a project bug.
- **Legacy 8B-era notes (Llama-3.1-8B, Llama-3.2-1B/3B, Qwen2.5-0.5B).** See
  [`REPRODUCE.md`](REPRODUCE.md). These experiments pre-date the Qwen3.6-27B
  pivot and are kept for historical reference only.
