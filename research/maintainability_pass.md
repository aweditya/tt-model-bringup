# Maintainability pass — living plan (read this to continue after compaction)

**Branch:** `chore/maintainability` (NOT merged — review before merging to main).
**Goal (user's words):** the repo must be *lean*; someone clones it onto a host
with P150s + a tt-metal build and *just runs things*. Good-but-proportionate SWE
practices (packaging, CI, docs). Experiments **preserved, not the focus**.
Multi-model bringup results (Llama/Qwen/SmolLM/8B) are **valuable — surface, don't
bury**. Persona: lean 10x engineer; no code bloat; comments to-the-point.

## HARD CONSTRAINTS (do not violate)
- **Repo dir name frozen** — `tt-xla` local / `~/tt-xla` on qb1/qb2 (renaming
  breaks Claude settings + every rsync path; CLAUDE.md).
- **`experiments/serve/server*.py` paths + names FROZEN** — live qb2 prod server
  + rsync targets. Only *low-risk, canary-gated* edits (dead-debug excision).
- **`91f`/`91l` are load-bearing**: `server_tp.py` does
  `importlib.spec_from_file_location("_91f", "experiments/91f_qwen36_27b_full_ondevice.py")`
  (~line 208) for `load_layer_weights_all`, and `_91l` (~line 350) for
  `load_embed_lm_head_weights`. Hardcoded `experiments/<name>.py` paths.
- **Run-validate every refactor step** (user: "make sure it runs"). Structural
  moves: prove no active importer breaks (grep). server_tp.py edits: qb2/qb1
  bootstrap + Paris canary (`make run PY=experiments/cb_validate_27b.py`).
- **No /tmp. Remote device exec only (ssh qb1/qb2). Frequent commits.**
- **`git mv` segfaults on large arg lists** in this shell — use a per-file
  `while read` loop. A crash can leave a stale `.git/index.lock` — verify no git
  process is running, then `rm .git/index.lock`.

## STRUCTURE (target, partly done)
```
tt-xla/
  README.md (Quickstart), HANDOFF.md, Makefile, pyproject.toml, .python-version
  scripts/        run_remote.sh, deploy.sh   (device-run incantation)  [DONE]
  experiments/    ACTIVE only: serve/, owned_ops/, kernel_patches/, utils/,
                  the validation/bench suite (cb_*), 91f, 91l            [in progress]
  models/         multi-model bringup RESULTS (10 demos)                [DONE]
  archive/        learning probes + bringup intermediates (173 files)   [DONE]
  research/, wiki/  docs (78 + 73 md — to be indexed in M6)
```

## PHASES + STATUS
- **M1 clone-and-run rails — DONE** (commit `e2b8088`): Makefile, scripts/run_remote.sh
  + deploy.sh (validated on qb1), .python-version=3.10.12, README Quickstart.
- **M3.1 archive learning — DONE** (`19894b8`): 105 probes → archive/ + index.
- **M3.2 surface/archive — DONE** (`fd0ef38`): 10 → models/, 68 → archive/bringup/,
  README links fixed. experiments/ root 221→38 .py.
- **M3.0 untangle 91f/91l — DONE (device-validated on qb1).** Reality was deeper
  than "2 loaders": `91f` is a shared on-device op LIBRARY (upload, hifi4,
  mlp/deltanet/gated_attn step ops + load_layer_weights_all) used by BOTH
  `server_tp.py` and `server.py`; the importlib hack existed only because the
  filenames started with a digit. Fix: `git mv` to valid module names in serve/
  (`91f_…`→`serve/ondevice_27b.py`, `91l_…`→`serve/generate_27b.py`) and replace
  every `importlib.spec_from_file_location` with `from experiments.serve import
  ondevice_27b as _91f` (the package-import form the servers already use for
  `protocol`, since they launch as `-m experiments.serve.server_tp` from repo
  root — bare `import` would NOT resolve). Kept the `state._91f`/`_91l`
  attributes so all 50+ `_91f.<sym>` call sites are untouched. Repointed:
  server_tp, server (incl. its now-`importlib.reload`-based `handle_reload_kernels`,
  dropped `_load_kernel_module` + `_91F/L_PATH`), generate_27b's own internal 91f
  load, demo_qwen36_27b (now run via `-m experiments.demo_qwen36_27b`). Added
  `serve/import_smoke.py` (mesh-free package-import smoke) + the two modules to
  deploy.sh. VALIDATED: `run_remote.sh --no-reset -m experiments.serve.import_smoke`
  on qb1 → "IMPORT SMOKE OK". NOT archived (they're the live library). LEFTOVER:
  `experiments/utils/p22_vocab_sharded_lm_head_probe.py` still hardcodes the old
  91l path (utils is pending archive triage; remote-only probe — fix when triaged).
  Old `91f_…`/`91l_…` files may still exist on remote hosts from prior rsyncs
  (orphaned, harmless; a clean redeploy / `--delete` removes them).
- **M2 CI/ruff — DONE** (commit `c38a939`): ruff lint config in pyproject
  (select E+F; ignore the compact-style E7xx/E4xx + E501; exclude archive/scratch/
  pjrt_plugin/experiments/utils/tt_jax), `[dependency-groups].dev` (ruff,
  pre-commit), `.pre-commit-config.yaml` (ruff + ruff-format incremental + safety
  hooks), `.github/workflows/ci.yml` (uvx ruff check + compileall smoke; NO heavy
  deps, NO device — TT runs stay on qb1/qb2). Curated tree is now `ruff check`
  clean: autofixed F401/F541 (148), hand-fixed F841×6 + the NCHIPS F811 in active
  files. Makefile lint/fmt pinned to `ruff@0.14.0`.
  **DEFERRED:** the big `ruff format` sweep — it splits the deliberate `a; b`
  one-liners and would be a huge, hard-to-review, risky diff across the frozen
  servers. Formatting lands incrementally via pre-commit on touched files; do a
  scoped sweep later if desired. So CI does NOT run `ruff format --check`.
- **M3 finish — PARTLY DONE**: archived `jax_qwen05b_*` (9) + `qwen05b_*` (2) →
  `archive/legacy/` (no active importers; done during M2 to clear lint). STILL
  PENDING: rename `cb_*` suite to verb_noun (e.g. validate_forward / bench_decode
  / profile_dn / needle_haystack); their `PROJECT_ROOT =
  Path(__file__).resolve().parents[1]` + `sys.path.insert(.../experiments/serve)`
  assumes top-level — fix `parents[N]` if moved into a subdir, and update
  scripts/deploy.sh defaults. Archive `experiments/tt_jax/` + `pjrt_plugin/` →
  archive/legacy/. Run-validate imports. (NOTE: when files leave
  `experiments/utils`, they re-enter the ruff lint scope — re-run `make lint`.)
- **M4 lean code + comments — PENDING (canary-gated)**: `# GOTCHA:` convention
  for load-bearing comments (view-decay, +1 RMSNorm, K-broadcast RoPE, bf16-KV —
  the HANDOFF must-keeps); delete narrative/debug-log comments. Excise dead debug
  blocks + unused probe endpoints from server_tp.py (7913 lines), each gated by
  the canary. Trim verbose cb_* comments.
  **Detailed line-level audit** (dead prefill variants `forward_prefill_tp_inner_v2_*`
  / `_v3_parallel_attn` ~1300 lines, the dead-state-flag + CLI-flag cull table,
  server_35b_ttnn comment trims, the load-bearing-looks-removable list) lives in
  `research/repo-cleanup-plan.md` — use it for the de-bloat specifics. NOTE: that
  doc's *folder-reorg* section (scratch/, experiments/utils/archive/) predates the
  actual M3.1/M3.2 layout (top-level `archive/` + `models/`) — follow the real
  layout for structure; only mine it for the de-bloat/flag specifics.
- **M5 kernels installable — PENDING**: organize owned_ops/ + kernel_patches/
  under `kernels/`; one `install_kernels.py` (+ Makefile target) integrating into
  tt-metal; README on the JIT-vs-rebuild model + per-kernel commit flow.
- **M6 docs — PENDING**: CONTRIBUTING.md (ssh/env/commit workflow); index the
  78 research + 73 wiki docs; consolidate research/27b_cb_scope.md → an
  architecture overview. Keep HANDOFF.md as "read first".

## HOW TO CONTINUE
`git checkout chore/maintainability`. Pick the next PENDING phase. M2 is the
safe next (no device). The untangle (M3.0) + M4 server_tp.py edits need the
device canary — run `make run PY=experiments/cb_validate_27b.py` (expects
logit_cos≈1.0 vs prod) after each. Commit per phase. Tooling reference: see the
companion `kernel_design_worksheet.md` for the TT hard constraints.
