# Contributing

How to work on this repo. For first-time setup + demos see [`README.md`](README.md);
for the current perf number + production code path see [`HANDOFF.md`](HANDOFF.md).

## Repo map

```
experiments/serve/         production servers — server_tp.py (27B TP), server.py
                           (single-chip), server_35b_ttnn.py (35B MoE), ondevice_27b.py
                           + generate_27b.py (shared 27B kernels + loaders),
                           cb_engine.py, cb_api.py, cb_scheduler.py, protocol.py, clients
experiments/cb/            continuous-batching validate/bench/profile/isolate suite
experiments/owned_ops/     custom TT-NN ops (see its README)
experiments/kernel_patches/  JIT device-kernel patches (no rebuild)
experiments/utils/         re-usable diagnostics
models/                    multi-model bringup demos (Llama / Qwen / SmolLM / 8B)
research/                  design notes + living plans (index: research/README.md)
wiki/                      learning-by-building Q&A wiki
archive/                   retired probes + bringup intermediates (not maintained)
scratch/                   legacy demos kept for reference
scripts/                   run_remote.sh, deploy.sh, build_owned_ops.sh,
                           check_setup.sh, install_ttnn.sh, ci_check_deploy_sync.py
```

## Setup

```bash
make setup            # uv sync — Python deps into .venv
make install-ttnn     # editable ttnn from $TT_METAL_HOME (run on the TT host)
make check            # sanity-check setup (no device open)
make kernels          # build the owned_ops custom kernels
```

`make help` lists targets. See README §Setup for the env block
(`TT_METAL_HOME`, `TT_BUILD_DIR`, `ARCH_NAME`, `HF_TOKEN`) and tt-metal SHA pin
([`tt-metal-sha.txt`](tt-metal-sha.txt)).

## Dev loop

All device work runs on a Tenstorrent host over ssh — never locally.

```bash
make deploy                                   # rsync code to $TT_HOST (default qb1; TT_HOST=qb2)
make run PY=experiments/cb/validate/forward.py
make dr  PY=...                               # deploy + run in one step (the edit→test loop)
make lint                                     # ruff (host-independent)
make reset                                    # tt-smi -r 0,1,2,3 if the mesh wedges
```

`scripts/run_remote.sh` is the single source of truth for the ssh + ttnn-env +
mesh-reset incantation; don't hand-roll it.

## Layered docs

- [`README.md`](README.md) — install + demos + chat server.
- [`HANDOFF.md`](HANDOFF.md) — current perf number + production paths + what's next.
- [`REPRODUCE.md`](REPRODUCE.md) — reproduce the chat server + the legacy multi-model demos.
- [`research/`](research/) — design notes + living plans. Index in
  [`research/README.md`](research/README.md); completed plans live in
  [`research/archive/`](research/archive/).
- [`wiki/`](wiki/) — Q&A wiki + the kernel-architecture deep-dives
  ([`research/kernel_research/`](research/kernel_research/)).

## Validating server changes (the canary)

A change to `server_tp.py` / `ondevice_27b.py` / `server_tp_cb.py` must pass
the bootstrap + forward canary on a TT host before merge:

```bash
make run PY=experiments/cb/validate/forward.py    # use --max-pos 6 for a short run
```

Expect `verdict: PASS` with CB B=1 vs production B=1 `logit_cos = 1.000000`
(bit-identical). (A pre-CB mesh-free import sanity smoke once lived at
`experiments/serve/import_smoke.py`; it was archived to
`archive/pre_cb_server_stack_2026-06-04/import_smoke.py` on 2026-06-04
after the CB stack subsumed its coverage.)

## Adding a custom kernel

Drop a self-contained op dir under `experiments/owned_ops/<name>/` (mirror an
existing one: `.cpp/.hpp` + nanobind, `sources.cmake`,
`integrate_into_ttmetal.py`, `INTEGRATION.md`, `test_*.py`). Then
`make kernels OPS=<name>` installs it + rebuilds ttnn. Device kernels JIT-compile,
so a compute-kernel-only change can go through `experiments/kernel_patches/` with
no rebuild. See [`experiments/owned_ops/README.md`](experiments/owned_ops/README.md).

## Code style

`make lint` (ruff, config in `pyproject.toml`) gates CI. Run
`uv run pre-commit install` once for auto-fix + format on commit. Formatting
lands incrementally on touched files. Comments are load-bearing only — a
`# GOTCHA:` for a hidden constraint or `Why:` for a multi-day-debug rationale;
no narrative.

## Non-negotiables

1. **Think first.** Plan before implementing; ground perf claims in
   arithmetic / roofline, not hand-waving. Isolate, then integrate.
2. **Remote execution only** — `ssh qb1` / `ssh qb2`. No device code locally.
3. **No `/tmp`.** Outputs, logs, caches, scratch → project dirs (`.cache/`, etc.).
4. **No inline scripts** (`python -c`). Write a permanent file under `scripts/`,
   `experiments/`, or the relevant package. Reusable helpers live in
   `experiments/utils/`.
5. **Single device by default** — saturate one P150 before reaching for the
   mesh; multi-chip is a memory / throughput decision, not a util workaround.
6. **`server_tp.py` is canary-gated.** Edits must pass the canary above.
7. **Frozen names:** the repo dir stays `tt-xla` locally / `~/tt-xla` on the
   hosts (renaming breaks Claude settings + every rsync path).
8. **Commit early + often.** Conventional prefixes (`feat` / `fix` / `refactor` /
   `chore` / `docs`), why-not-what messages.
