# Contributing

Onboarding for working *on* this repo. For first-time setup + demos see
[`README.md`](README.md); for the current perf number + production code path see
[`HANDOFF.md`](HANDOFF.md) (read it first).

## Repo map

```
experiments/serve/   production servers — server_tp.py (27B TP, qb2 prod),
                     server.py (single-chip), server_35b_ttnn.py (35B MoE),
                     ondevice_27b.py / generate_27b.py (shared 27B kernels +
                     loaders), cb_scheduler.py, protocol.py + clients
experiments/cb/      continuous-batching validation/bench/profile/isolate suite
experiments/owned_ops/   custom TT-NN ops (see its README)
experiments/kernel_patches/   JIT device-kernel patches (no rebuild)
models/              multi-model bringup demos (Llama/Qwen/SmolLM/8B)
research/, wiki/     design notes + the learning-by-building Q&A wiki
archive/             retired probes + bringup intermediates (not maintained)
scripts/             run_remote.sh, deploy.sh, build_owned_ops.sh, strip_functions.py
```

## Dev loop

All device work runs on a Tenstorrent host over ssh — **never locally**.

```bash
make deploy                 # rsync code to $TT_HOST (default qb1; TT_HOST=qb2)
make run PY=experiments/cb/validate/forward.py
make dr  PY=...             # deploy + run in one step (the edit→test loop)
make lint                   # ruff (host-independent)
make reset                  # tt-smi -r 0,1,2,3 if the mesh wedges
```
`scripts/run_remote.sh` is the single source of truth for the ssh + ttnn-env +
mesh-reset incantation; don't hand-roll it.

## Non-negotiables

1. **Think first.** Plan before implementing; ground perf claims in
   arithmetic/roofline, not hand-waving. Isolate, then integrate.
2. **Remote execution only** — `ssh qb1`/`ssh qb2`. No device code runs locally.
3. **No `/tmp`.** Outputs, logs, caches, scratch → project dirs (`.cache/`, etc.).
4. **No inline scripts** (`python -c`). Write a permanent file under `scripts/`,
   `experiments/`, or the relevant package. Reusable helpers live in
   `experiments/utils/`.
5. **Never break the qb1/qb2 production server.** `server_tp.py`'s path + name
   are frozen (live rsync target + prod). Edits to it are canary-gated (below).
6. **Single device by default** — saturate one P150 before reaching for the mesh;
   multi-chip is a memory/throughput decision, not a util workaround.
7. **Frozen names:** the repo dir stays `tt-xla` locally / `~/tt-xla` on the
   hosts (renaming breaks Claude settings + every rsync path).
8. **Commit early + often.** Conventional prefixes (`feat`/`fix`/`refactor`/
   `chore`/`docs`), why-not-what messages.

## Validating server changes (the canary)

A change to `server_tp.py` / `ondevice_27b.py` / `server_tp_cb.py` must pass the
bootstrap + forward canary on a TT host before merge:

```bash
make run PY=experiments/cb/validate/forward.py   # --max-pos 6 for a short run
```
Expect `verdict: PASS` with CB B=1 vs production B=1 `logit_cos = 1.000000`
(bit-identical). Mesh-free sanity (catches import/dispatch breakage in ~30s):
`scripts/run_remote.sh --no-reset -m experiments.serve.import_smoke`.

## Adding a custom kernel

Drop a self-contained op dir under `experiments/owned_ops/<name>/` (mirror an
existing one: `.cpp/.hpp` + nanobind, `sources.cmake`, `integrate_into_ttmetal.py`,
`INTEGRATION.md`, `test_*.py`), then `make kernels OPS=<name>` to install + rebuild
ttnn. Device kernels JIT-compile, so a compute-kernel-only change can go through
`experiments/kernel_patches/` with no rebuild. See `experiments/owned_ops/README.md`.

## Code style

`make lint` (ruff, config in `pyproject.toml`) gates CI. Run
`uv run pre-commit install` once for auto-fix + format on commit. The format
sweep is intentionally not yet applied repo-wide — formatting lands incrementally
on touched files. Comments: lead-bearing only (a `# GOTCHA:` for a hidden
constraint), no narrative; the diff and commit message carry the rest.
