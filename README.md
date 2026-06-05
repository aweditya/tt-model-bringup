# tt-model-bringup

> Direct TT-Metal bringup of modern open-weight LLMs on Tenstorrent Blackhole,
> with custom `owned_*` compute kernels and a continuous-batching
> OpenAI-compatible HTTP server.

**What:** Hand-written TT-Metal graphs (no PJRT, no JAX) for Qwen3.6-27B,
Qwen3.6-35B-A3B MoE, Gemma 4 12B, Nemotron-3 Nano 30B-A3B, plus a model zoo
of single-chip Llama / Qwen2.5 / SmolLM ports.
**Why:** Squeeze production-grade per-token throughput out of Blackhole P150s
by owning the compute graph end-to-end — kernels, scheduler, HTTP server.
**Who:** Stanford CS440LX research project; pivoted from a JAX/XLA PJRT
backend (`tt-xla`) to direct bringup.
**How:** Custom fused-op kernels under `experiments/owned_ops/`, a
continuous-batching engine under `experiments/serve/`, and a learning-wiki
documenting every design decision.

> **Cold-start?** Read [`HANDOFF.md`](HANDOFF.md) first — the one-page
> entry point with current perf, production paths, and what's next.

---

## Table of contents

- [Hardware](#hardware)
- [Quickstart](#quickstart)
- [Models brought up](#models-brought-up)
- [Chat server](#chat-server)
- [Repo layout](#repo-layout)
- [Long context](#long-context)
- [Troubleshooting](#troubleshooting)
- [Related projects](#related-projects)
- [Top-level docs](#top-level-docs)

---

## Hardware

Bringups run on **[Tenstorrent QuietBox][quietbox]** workstations
(4× Blackhole P150 per box) with `FABRIC_1D` inter-chip links. A single
P150 is enough for the legacy 8B-class demos under [`models/`](models/).
27B / 35B / Gemma 4 12B require a `(1, 4)` mesh.

| Model class | Mesh | Path |
|---|---|---|
| Llama 1B/3B/8B, SmolLM3, Qwen2.5/3 small | single P150 | `models/*.py` |
| Qwen3.6-27B, Gemma 4 12B | `(1, 4)` 4×P150 | `experiments/serve/server_*.py` |
| Qwen3.6-35B-A3B MoE | `(1, 4)` 4×P150 | `experiments/serve/server_35b_*.py` |
| Nemotron-3 Nano 30B-A3B | `(1, 4)` 4×P150 (target) | `experiments/owned_ops/nemotron3_mamba2_decode_owned/` |

[quietbox]: https://tenstorrent.com/hardware/quietbox

> **No local device execution.** Every command in this repo assumes you
> are on the QuietBox; the dev loop expects ssh into a host with TT-Metal
> built from source.

---

## Quickstart

On a QuietBox with TT-Metal built from source:

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla
cd ~/tt-xla

make setup                                          # uv sync — Python deps into .venv
make install-ttnn                                   # editable ttnn from $TT_METAL_HOME
make check                                          # sanity-check setup (no device open)
make kernels                                        # build the owned_ops custom kernels
bash experiments/serve/scripts/serve_cb.sh start    # boot the CB chat server (~6 min)
```

Talk to it once `/health` returns 200:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":200}'
```

Stop gracefully (hard-killing can wedge the fabric):

```bash
bash experiments/serve/scripts/serve_cb.sh stop
```

> The repo's working directory is `~/tt-xla` for historical reasons (the
> project was originally a JAX/XLA PJRT backend). Renaming breaks too many
> rsync paths to be worth it.

See [`REPRODUCE.md`](REPRODUCE.md) for the full reproducible-on-any-QuietBox
recipe, tested-versions matrix, and per-demo expected numbers.

---

## Models brought up

| Model | Params | Architecture | Status |
|---|---|---|---|
| Qwen3.6-27B | 27 B dense | Hybrid attention + GatedDeltaNet | Production CB + TP, prefix-cache live |
| Qwen3.6-35B-A3B | 35 B / 3 B-A | Hybrid + GatedDeltaNet + MoE | CB shipped; B>1 blocked on slot-poisoning fix |
| Gemma 4 12B (base + IT) | 12 B dense | Dual sliding/global attention | End-to-end CB + HTTP chat |
| Nemotron-3 Nano 30B-A3B | 30 B / 3 B-A | Mamba2-Transformer hybrid MoE | In progress — owned Mamba2 SSD kernel (G0→G4) |
| Llama 1B/3B/8B, SmolLM3, Qwen2.5/3 small | up to 8 B | Decoder-only Transformer | Legacy single-chip demos (see [`models/`](models/)) |

Per-token throughput numbers drift across kernel work — see
[`HANDOFF.md`](HANDOFF.md) for the current figures.

---

## Chat server

[`experiments/serve/cb_api.py`](experiments/serve/cb_api.py) is the production
HTTP server: an OpenAI-compatible API hosting a continuous-batching engine
on top of the Orca scheduler and the logits-traced forward.

- **Endpoints:** `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
  `/health`, `/metrics`
- **Multi-client** by design; per-request sampling (`temperature` / `top_p` /
  `top_k` / `seed`); slot freed on client disconnect
- **Backend selection** via `TT_BACKEND` (`27b`, `35b`, `gemma4_12b`)
- **Tuning knobs:** `TT_CB_PORT`, `TT_CB_SLOTS`, `TT_CB_MAX_NEW`,
  `TT_CB_MAX_INFLIGHT`, `TT_CB_PREFIX_CACHE`, `TT_CB_CHUNKED_PREFILL` —
  full list in [`experiments/serve/scripts/serve_cb.sh`](experiments/serve/scripts/serve_cb.sh)

A Claude-Code-style chat TUI lives at
[`scripts/chat.py`](scripts/chat.py) — see
[`scripts/CHAT_TUI.md`](scripts/CHAT_TUI.md) for the slash-command reference.

---

## Repo layout

| Path | Purpose |
|---|---|
| `experiments/` | Production servers, owned kernels, CB engine + validators, probes |
| `research/` | Design docs and living plans (index: [`research/README.md`](research/README.md)) |
| `wiki/` | Q&A wiki — learning-by-building notes on JAX/XLA + TT-Metal internals |
| `scripts/` | Dev-loop scripts: `deploy.sh`, `run_remote.sh`, `build_owned_ops.sh`, chat TUI |
| `models/` | Legacy single-chip multi-model demos (Llama, SmolLM, Qwen2.5/3) |
| `archive/` | Retired probes, the original PJRT plugin (`legacy/`), the CS440LX poster + measurements, and dated cleanup buckets |

---

## Long context

The 4-chip TP path is validated to `L=4000` with verbatim 8-character
needle-haystack retrieval at all needle positions (`0.25` / `0.5` / `0.75`
frac), using the B3 HiFi2 SDPA recipe.

Probe: [`experiments/utils/needle_haystack_qb2_tp.py`](experiments/utils/needle_haystack_qb2_tp.py).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Fabric wedged after a hard-kill | `make reset` (`tt-smi -r 0,1,2,3`) and restart the server; always prefer the `serve_*.sh stop` path. |
| `AutoTokenizer` fails with 401 | No HF token configured — `uv run hf auth login` or set `HF_TOKEN` in `.env`. |
| `HF_HOME` fills `$HOME` | Override: `export HF_HOME=/path/with/space/.cache/hf`. |
| Legacy demos won't open device 0 | Prod server holds it — `bash experiments/serve/scripts/serve_cb.sh stop` first. |

---

## Related projects

- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — the
  device runtime + LLK that everything here links against.
- [tenstorrent/tt-xla](https://github.com/tenstorrent/tt-xla) — Tenstorrent's
  official PJRT backend (different from this repo's archived `archive/legacy/pjrt_plugin/`).
- [tenstorrent/vllm](https://github.com/tenstorrent/vllm) — Tenstorrent's
  vLLM fork; the continuous-batching design here lines up with its API.
- [Corsix's Tenstorrent Wormhole blog series](https://www.corsix.org/content/tt-wh-part1) —
  best architectural reference outside the TT docs.

---

## Top-level docs

- [`HANDOFF.md`](HANDOFF.md) — current perf, production paths, next bringup target.
- [`REPRODUCE.md`](REPRODUCE.md) — reproduce the chat server + legacy demos on a fresh QuietBox.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev loop, canary gates, code style.
- [`CLAUDE.md`](CLAUDE.md) — project non-negotiables (host-specific).

> **Origin:** the repo was originally a JAX/XLA PJRT backend — hence the
> `tt-xla` directory name. PJRT sources live in
> [`archive/legacy/pjrt_plugin/`](archive/legacy/pjrt_plugin/) and are off
> the active path.
