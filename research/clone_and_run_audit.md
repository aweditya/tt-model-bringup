# Clone-and-run audit (2026-05-30)

Audit of what a fresh `git clone … && cd … && follow README` user hits.
Pure audit; no code changes. All evidence cites paths + line numbers.

## Summary — most plausible "garbage output" + "needed specific versions" causes

1. **`server_tp.py` requires `experiments/utils/full_layer_tp_probe.py` +
   `tp_attn_traced_probe.py` but `scripts/deploy.sh` (default arg set) does NOT
   sync them** (`scripts/deploy.sh:12-17`). 12 active `from full_layer_tp_probe`
   call sites in `experiments/serve/server_tp.py` (e.g. lines 208, 212, 653, 686,
   866, 1083, 1321, 1425, 2088). On a fresh box `make dr` boots a TP server that
   crashes mid-bootstrap with `ModuleNotFoundError`; if the user happened to
   `rsync` everything manually it works. This is also why "specific versions
   needed" is suspicious — the dep is on a sibling Python file, not a package.
2. **`experiments/serve/scripts/serve.sh` (Demo A — the FIRST command a fresh
   user runs) does not export *any* TT env vars** (`serve.sh:24-55`). Only
   `serve_tp.sh`/`serve_35b.sh`/`serve_cb.sh` do. If the user's interactive
   shell doesn't already have `TT_METAL_HOME` / `PYTHONPATH` / `LD_LIBRARY_PATH`
   / `ARCH_NAME` set, `import ttnn` either fails outright or imports a stale
   wheel — exactly the "wrong-numerics / garbage tokens" failure mode.
3. **`TT_BUILD_DIR` default is inconsistent across the repo.** README says
   `build_Release` (`README.md:59,65`), `.env.example` says
   `build_tracy_gcc12_nodist` (`.env.example:24`), `serve_tp.sh` defaults to
   `build_tracy_gcc12_nodist` (`serve_tp.sh:37`), but
   `serve.sh`/`serve_35b.sh`/`serve_cb.sh`/`run_remote.sh`/`build_owned_ops.sh`
   default to `build_Release`. If the user built only `build_Release`,
   `serve_tp.sh` silently picks an empty `LD_LIBRARY_PATH` segment (Demo B
   boots against the wrong libs). The owned-op INTEGRATION.md (line 19) is
   stale on the same axis.
4. **`README.md:40-46` says qb1 has "No fabric"** — directly contradicted by
   the maintainability_pass + memory notes (qb1 inter-chip fabric works as of
   2026-05-21 and was used for owned-op work). A fresh user who wants TP gets
   wrongly told to find a second machine.
5. **No `Qwen3.6` HF-access gating in code.** `server_tp.py:199`
   `AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")` works only because the
   model is currently public. If HF flips it to gated, the failure mode is a
   401 inside bootstrap, with no upfront check.

## Punch list

### P0 — blockers / silent-wrong-output risks

1. **(P0) `deploy.sh` default omits the two `full_layer_tp_probe.py` +
   `tp_attn_traced_probe.py` Python deps that `server_tp.py`/`server_tp_cb.py`
   need.** Evidence: `scripts/deploy.sh:12-17` lists 8 paths under
   `experiments/serve/` and `experiments/cb/`, none from `experiments/utils/`;
   `experiments/serve/server_tp.py:208,212,653,686,866,1083,1321,1425,2088` all
   do `from full_layer_tp_probe import ...`. Fix: add
   `experiments/utils/full_layer_tp_probe.py` and `experiments/utils/tp_attn_traced_probe.py`
   to the default arg list in `deploy.sh`, OR move them into `experiments/serve/`
   (where they semantically belong now) and update the imports.
2. **(P0) `serve.sh` does not export `TT_METAL_HOME` / `PYTHONPATH` /
   `LD_LIBRARY_PATH` / `ARCH_NAME` before launching the server.** Evidence:
   `experiments/serve/scripts/serve.sh:24-55` only sets `HF_HOME` +
   `PYTHONUNBUFFERED`. Compare `serve_tp.sh:36-42` which sets the full block.
   Demo A on a fresh box silently picks whatever `ttnn` your shell happens to
   resolve — often nothing, sometimes a stale install. Fix: copy the
   `TT_METAL_HOME`/`TT_BUILD_DIR`/`PYTHONPATH`/`LD_LIBRARY_PATH`/`ARCH_NAME`
   block from `serve_tp.sh` into `serve.sh` (and verify against
   `scripts/run_remote.sh:22-28`).
3. **(P0) tt-metal SHA not pinned anywhere.** Evidence: `README.md:66-67`
   explicitly says "A known-good `tt-metal` SHA is not yet pinned — the current
   qb1/qb2 `ttnn` build is the reference." Friend would have to ask the user.
   Owned-op nanobind/CMake patches in `experiments/owned_ops/*/integrate_into_ttmetal.py`
   are pinned to a specific tt-metal layout (e.g. `ttnn/cpp/ttnn/operations/experimental/transformer/`
   in `qwen36_gdn_decode_owned/integrate_into_ttmetal.py:16`); any tt-metal
   refactor since the dev box was built will break `--all`/`scripts/build_owned_ops.sh`.
   Fix: pin a commit SHA in README §1 and add it to a `tt-metal-sha.txt` file
   so CI/scripts can assert. Failing that, the integrate scripts should pretty-
   print expected-vs-actual file paths so the failure mode is "tt-metal moved
   X to Y" not "garbage output".
4. **(P0) Owned-op calls have no graceful degradation if the op isn't built
   into ttnn.** Evidence: `experiments/serve/server_tp.py:762,789` and
   `experiments/serve/server_tp_cb.py:346` unconditionally call
   `ttnn.experimental.qwen36_decay_gate_decode_owned` /
   `qwen36_gdn_decode_owned` based on `state.deltanet_decay_gate_mode` /
   `state.deltanet_recurrence_mode` (defaults `"owned_decay_gate"` /
   `"owned_gdn"` set at `server_tp.py:135,138`). If the user skips
   `scripts/build_owned_ops.sh`, decode raises `AttributeError` mid-token
   instead of failing fast at bootstrap with a helpful message. Manual
   fallbacks already exist in the same files (e.g. `server_tp.py:768-776`).
   Fix: add a bootstrap-time `hasattr(ttnn.experimental, "qwen36_gdn_decode_owned")`
   check that either flips the mode to `"manual"` with a loud warning or exits
   with `"run scripts/build_owned_ops.sh"`.
5. **(P0) `README.md:40-46` host matrix is wrong about qb1's fabric.**
   Evidence: the same README's `## Host matrix` says qb1 has "**No fabric**"
   and Demo B "will hang on qb1"; but `research/maintainability_pass.md:139-141`
   and the user MEMORY entries (qb1 HAS fabric since 2026-05-21; owned kernels
   verified on qb1 2026-05-28) say the opposite. A fresh user takes the README
   at face value and avoids qb1 for TP. Fix: rewrite the table to "qb1: fabric
   works as of 2026-05-21" and drop the "will hang" line, or add the date and
   tt-metal SHA at which the change happened.

### P1 — high friction

6. **(P1) README Setup §3 tells the user to `uv pip install setuptools_scm`
   then `uv pip install -e $TT_METAL_HOME --no-build-isolation`** (`README.md:103,106`).
   That `-e` install pulls a fresh `torch`/`transformers` from `tt-metal`'s
   build deps that can shadow the `uv.lock` pins — `uv.lock` resolves
   `torch==2.12.0`, `transformers==5.9.0`, but qb1 prod runs `torch==2.11.0`
   per `README.md:117`. Fix: document the exact `--no-deps` invocation, or
   provide a one-line `scripts/install_ttnn.sh` that does the install with
   `--no-deps` and prints the resulting `ttnn.__version__`.
7. **(P1) `.env.example` defaults `TT_BUILD_DIR=$TT_METAL_HOME/build_tracy_gcc12_nodist`**
   (`.env.example:24`) but README says `build_Release` (`README.md:59`). The
   user copies `.env.example` to `.env`, sources it, then runs
   `bash experiments/serve/scripts/serve.sh start` and gets the wrong libs.
   Fix: set `.env.example` to `build_Release` and add a comment about the
   profiler-build alternative.
8. **(P1) `experiments/owned_ops/qwen36_gdn_decode_owned/INTEGRATION.md:19-21`
   tells the user to build into `build_tracy_gcc12_nodist`** but
   `scripts/build_owned_ops.sh:22` defaults to `build_Release`. If the user
   reads INTEGRATION.md and runs the cmake by hand they build to the wrong
   dir and the `.so` copies in `build_owned_ops.sh:65-66` overwrite nothing
   useful. Fix: align INTEGRATION.md with `build_owned_ops.sh`'s default.
9. **(P1) README §4 Option A says `uv run hf auth login` but `.env.example:12`
   says `uv run huggingface-cli login`.** Same binary, but the README itself
   notes the latter is deprecated (`README.md:132-133`). Fix: update
   `.env.example:12-13` to `hf auth login`.
10. **(P1) No upfront sanity-check script.** No `scripts/check_setup.sh` /
    `make check` that verifies (a) `ttnn` imports, (b)
    `ttnn.experimental.qwen36_gdn_decode_owned` is callable, (c) `tt-smi -s`
    sees the right number of chips, (d) `HF_TOKEN` is set or `~/.cache/huggingface/token`
    exists, (e) `Qwen/Qwen3.6-27B` config.json downloads. A fresh user
    discovers each missing piece serially over ~30 min of bootstrap retries.
    Fix: add `scripts/check_setup.sh` (host-only, no device open) wired to
    `make check`.
11. **(P1) `experiments/utils/p22_vocab_sharded_lm_head_probe.py:77`
    still hardcodes `/home/aditya/tt-xla/experiments/91l_fp32_residual_generate.py`**
    — that file was renamed in M3.0 (`research/maintainability_pass.md:62-63`
    even calls this out as a known leftover). Fix: repoint to
    `from experiments.serve import generate_27b`, or move the probe to
    `archive/`.
12. **(P1) `README.md:293` points to `experiments/utils/needle_haystack_b3_probe.py`,
    file actually lives at `experiments/utils/archive/needle_haystack_b3_probe.py`.**
    Fix: drop the broken link or update to the archive path.

### P2 — polish

13. **(P2) `requirements.txt` is retained but its constraints (`>=`, no upper
    bounds on torch/transformers) drift from `uv.lock` and don't match the
    `REPRODUCE.md:13-22` "tested environment" table.** Fresh users who
    `pip install -r` (rather than `uv sync`) get a different stack. Fix:
    either (a) regenerate `requirements.txt` from `uv export --frozen` per
    release, or (b) delete it and have `REPRODUCE.md` point at `uv.lock`.
14. **(P2) `serve_tp.sh` documents a `~17 min bootstrap` but does not print a
    `READY` marker** beyond the implicit Unix socket appearing. A fresh user
    `tail -f`s the log and guesses when to fire the client. Fix: have
    `server_tp.py` write a `READY` line on socket bind; have `serve_tp.sh`
    optionally wait for it.
15. **(P2) `scripts/build_owned_ops.sh` succeeds silently if the `.so` copy at
    lines 65-66 (`cp $BUILD_DIR/ttnn/_ttnn.so $TT_METAL/ttnn/ttnn/_ttnn.so`)
    overwrites a wheel-installed file** — exactly the "stale wheel shadowing
    the rebuilt source-package" case that
    `experiments/owned_ops/qwen36_gdn_decode_owned/INTEGRATION.md:30+` warns
    about. Fix: print `ttnn.__file__` + `getattr(ttnn.experimental, "qwen36_gdn_decode_owned", None)`
    at the end of the script to confirm the new op resolves.
16. **(P2) CI runs `compileall` but does not exercise
    `from experiments.serve.import_smoke import *`** (`.github/workflows/ci.yml:19-23`).
    The 12 `full_layer_tp_probe` imports in `server_tp.py` are lazy (inside
    `bootstrap`), so `compileall` doesn't catch the missing-deploy bug. Fix:
    add a CI step that does `python -c "from experiments.serve import server_tp;
    server_tp.MeshServerState()"` after stubbing ttnn (or just run the
    `import_smoke.py` static-import check from `experiments/serve/import_smoke.py`
    even without device).

## Quick-fix priority order

1. **Ship items #1 + #2 + #4 first** (deploy.sh deps, serve.sh env block,
   bootstrap-time op detection). Each is one-screen of change, each unblocks a
   real failure mode the friend likely hit. #1 alone is the most likely root
   cause of "garbage output" — server bootstrapped against a partial sync.
2. **Then #5 + #7 + #9** (fix the README host-matrix + .env.example
   inconsistencies). Pure docs; zero risk.
3. **Then #3 + #10** (pin tt-metal SHA + add `scripts/check_setup.sh` wired to
   `make check`). These are the durable fixes — they make every subsequent
   fresh-clone fail loudly and early.
4. The remaining items are polish; batch them into a follow-up commit after #1-3.

## What would have prevented this

- **CI: a "fresh-clone-deploy dry-run" step.** Take `scripts/deploy.sh` with
  its default args, run it against a local fake tree, and assert every
  `from X import Y` in `experiments/serve/server*.py` resolves. Would have
  caught #1.
- **CI: `import_smoke.py` runs without device.** Stub `ttnn` and import the
  three server modules; assert no `ImportError`. Catches structural drift
  between `serve/` and `utils/` early. Catches #11 and any future stale-path
  reference.
- **A `scripts/check_setup.sh` invariant**: README claims "clone-and-run", so
  add an executable that says it. Run it from CI in a docker that mocks ttnn;
  run it from a host bootstrap to confirm.
- **Single source of truth for `TT_BUILD_DIR`**. Either drop the variable
  entirely (build_Release everywhere) or have all scripts source one
  `scripts/env.sh`. Cuts #2/#3/#7/#8 down to one edit.
- **A `tt-metal-sha.txt` plus a `scripts/build_owned_ops.sh` assertion** that
  `git -C $TT_METAL rev-parse HEAD` matches. Catches "wrong tt-metal SHA"
  before it surfaces as numeric garbage.
