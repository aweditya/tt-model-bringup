# tt-model-bringup

Bringing up modern open-weight LLMs on Tenstorrent Blackhole P150 hardware:
direct TT-Metal model graphs, custom `owned_*` compute kernels, and a
continuous-batching OpenAI-compatible HTTP server.

For the current perf number, production code path, and the active bringup
target, read [`HANDOFF.md`](HANDOFF.md) first — it's the one-page cold-start
entry point.

> Origin: a Stanford CS440LX research project, originally scoped as a JAX/XLA
> PJRT backend (legacy `tt-xla` directory name). The PJRT sources live in
> `archive/legacy/pjrt_plugin/` and are off the active path.

---

## Repo layout

| Path           | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `experiments/` | Production servers, owned kernels, CB engine + validators, probes.      |
| `research/`    | Design docs and living plans (index: `research/README.md`).             |
| `wiki/`        | Q&A wiki — learning-by-building notes on JAX/XLA + TT-Metal internals.  |
| `scripts/`     | Dev-loop scripts: `deploy.sh`, `run_remote.sh`, `build_owned_ops.sh`, chat TUI. |
| `models/`      | Legacy multi-model demos (Llama, SmolLM, Qwen2.5/3) kept as references. |
| `archive/`     | Retired probes, the PJRT plugin, and the CS440LX poster + measurements. |

Top-level entry points:

- [`HANDOFF.md`](HANDOFF.md) — current perf, production paths, next bringup target.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev loop, canary gates, code style.
- [`REPRODUCE.md`](REPRODUCE.md) — reproduce the chat server + legacy demos.
- [`CLAUDE.md`](CLAUDE.md) — project non-negotiables.

---

## Host matrix

| Host  | Hardware           | Inter-chip fabric | Use it for                                |
|-------|--------------------|-------------------|-------------------------------------------|
| `qb1` | 4× Blackhole P150  | FABRIC_1D         | Experimental TP / CB / new kernels        |
| `qb2` | 4× Blackhole P150  | FABRIC_1D         | Production-style 4× P150 TP serving       |

All device work runs on a Tenstorrent host over ssh — never locally.

---

## Quickstart

On a host with Tenstorrent P150s and a `tt-metal` source build:

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla && cd ~/tt-xla
make setup                                            # uv sync — Python deps into .venv
make install-ttnn                                     # editable ttnn from $TT_METAL_HOME
make check                                            # sanity-check setup (no device open)
make kernels                                          # build the owned_ops custom kernels
bash experiments/serve/scripts/serve_cb.sh start      # boot the CB chat server (~6 min)
```

Then talk to it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":200}'
```

Stop with `bash experiments/serve/scripts/serve_cb.sh stop` (graceful drain —
hard-killing can wedge the fabric).

`make help` lists the dev-loop targets (`deploy`, `run`, `dr`, `reset`, `lint`).
Set `TT_HOST=qb2` to target the 4-chip box.

---

## Setup

1. **Build TT-Metal from source** per [Tenstorrent's guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
   for Blackhole. Export `TT_METAL_HOME`, `TT_BUILD_DIR=$TT_METAL_HOME/build_Release`,
   and `ARCH_NAME=blackhole`. The repo pins a SHA in
   [`tt-metal-sha.txt`](tt-metal-sha.txt); `scripts/build_owned_ops.sh` warns on mismatch.
2. **Install Python + ttnn**: `make setup && make install-ttnn && make check`.
   `install_ttnn.sh` passes `--no-deps` so tt-metal's pyproject does not shadow
   pins in `uv.lock`. Re-run after any `uv sync` or tt-metal rebuild.
3. **Configure HuggingFace access**: `uv run hf auth login`, or copy
   `.env.example` to `.env` and set `HF_TOKEN=hf_…` before starting a server.

---

## Chat server

`experiments/serve/cb_api.py` is the production HTTP server: an
OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/health`, `/metrics`) hosting a continuous-batching engine
(`cb_engine.py`) on top of the Orca scheduler (`cb_scheduler.py`) and the
logits-traced forward. Multi-client by design; per-request sampling
(`temperature` / `top_p` / `top_k` / `seed`); slot freed on client disconnect.

Backend selection via `TT_BACKEND` (`27b`, `35b`, `gemma4_12b`). Knobs:
`TT_CB_PORT`, `TT_CB_SLOTS`, `TT_CB_MAX_NEW`, `TT_CB_MAX_INFLIGHT`,
`TT_CB_PREFIX_CACHE`, `TT_CB_CHUNKED_PREFILL` — see
`experiments/serve/scripts/serve_cb.sh` for the full list.

A Claude-Code-style chat TUI lives at [`scripts/chat.py`](scripts/chat.py)
(README: [`scripts/CHAT_TUI.md`](scripts/CHAT_TUI.md)).

---

## Models brought up

| Model                          | Params       | Architecture                        | Status                                          |
|--------------------------------|--------------|-------------------------------------|-------------------------------------------------|
| Qwen3.6-27B                    | 27 B dense   | Hybrid attention + GatedDeltaNet    | Production CB + TP (qb1/qb2), prefix-cache live |
| Qwen3.6-35B-A3B                | 35 B / 3B-A  | Hybrid + GatedDeltaNet + MoE        | CB shipped; B>1 blocked on slot-poisoning fix   |
| Gemma 4 12B (base + IT)        | 12 B dense   | Dual sliding/global attention       | End-to-end CB + HTTP chat                       |
| Nemotron-3 Nano 30B-A3B        | 30 B / 3B-A  | Mamba2-Transformer hybrid MoE       | In progress — owned Mamba2 SSD kernel (G0→G4)   |
| Llama 1B/3B/8B, SmolLM3, Qwen2.5/3 small | up to 8 B | Decoder-only Transformer  | Legacy single-chip demos (see `models/`)        |

Per-token throughput numbers drift — see [`HANDOFF.md`](HANDOFF.md) for the
current figures.

---

## Long context

The qb2 4-chip TP path is validated to `L=4000` with verbatim 8-char
needle-haystack retrieval at all needle positions (0.25 / 0.5 / 0.75 frac),
using the B3 HiFi2 SDPA recipe. Probe:
`experiments/utils/needle_haystack_qb2_tp.py`.

---

## Troubleshooting

- **Fabric wedged after a hard-kill.** `make reset` (`tt-smi -r 0,1,2,3`) and
  restart the server. Always prefer the `serve_*.sh stop` path.
- **`AutoTokenizer` fails with 401.** No HF token configured — see
  "Configure HuggingFace access" above.
- **`HF_HOME` fills `$HOME`.** Override: `export HF_HOME=/path/with/space/.cache/hf`.
- **Legacy multi-model demos.** See [`REPRODUCE.md`](REPRODUCE.md); stop the
  prod server first so device 0 is free.
