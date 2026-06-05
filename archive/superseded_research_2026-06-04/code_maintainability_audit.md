# Code Maintainability Audit — 2026-05-30

Companion to the earlier `clone_and_run_audit.md`, scoped to the **code** in
the IN-scope set (`experiments/serve/*.py` for the production + CB stack,
`experiments/cb/{validate,bench,load,needle.py}`, `scripts/`, `Makefile`,
`pyproject.toml`). The goal is "easier to read and maintain for a new
contributor," not redesigns. Punch list is grouped by theme; each item carries
**severity** (P0 / P1 / P2), file:line evidence, and a concrete one-line fix.

Out of scope (untouched): `experiments/utils/`, `experiments/tt_jax/`,
`archive/`, `pjrt_plugin/`, `models/`, `wiki/`, `research/`, `tt_docs_corpus/`,
C++ owned-op sources, `server_35b*`, `server.py`, `ondevice_27b.py`,
`generate_27b.py`, `client*.py`, `protocol.py`.

---

## Summary — top themes

1. **`server_tp.py` carries 100+ lines of dead/debug noise** behind
   `getattr(state, 'ccl_debug', False)` / `debug_mlp_resid` /
   `debug_layer_boundary` flags (lines 596-637, 1457-1486, 1704-1740). Nothing
   in-tree sets these. They were one-off probes; the audit value is `0`. **Strip
   these blocks**.

2. **Validators / benches duplicate ~10 lines of bootstrap boilerplate verbatim**
   (`PROJECT_ROOT` discovery, `sys.stdout.reconfigure`, `def log(msg)`,
   `MeshServerState() if hasattr(...) else State()`, and the three `state.delta*`
   mode overrides). 15 files repeat the same five lines. A two-helper module
   (`experiments/cb/_runner.py` with `bootstrap_27b_cb_manual(extra=None)` +
   `log()`) removes ~150 LOC and makes test files actually scannable.

3. **Narrative / commit-SHA / "Agent X" / dated-decision comments belong in git**.
   `server_tp.py` has dozens (lines 91, 131, 564, 589, 666, 1317, 1457, 1704,
   1723, 1742, 2144...). Two examples: line 564 cites a savings projection by
   "Agent N's gap analysis (5a9808d)"; line 1742 cites "Agent X's resolution at
   feedback_lm_head_argmax_unknown.md". Compress to the actual invariant or
   delete.

4. **Three different `PROJECT_ROOT` patterns coexist in the same directory**
   (`server_tp.py` uses `os.environ.get("TT_XLA_ROOT")`, `server_tp_cb.py` /
   `cb_scheduler.py` use `parents[2]`, `cb_engine.py` / `cb_api.py` walk parents
   until they find `experiments/serve/`). Standardise on the `parents-walk` form
   the new CB stack uses — it's robust to symlinks and clone location.

5. **Stale "legacy" fields kept "for safety"** (`MeshServerState.x_buf`,
   `cos_buf`, `sin_buf`, `traced_logits_tt`, `trace_x_buf`, `trace_logits_buf`
   — server_tp.py:93-118). Every one is None or unread by current code paths.
   The `# legacy: ...` comments are clear evidence they're unused. Delete the
   fields and the allocation block at lines 509-524.

---

## Punch list

### `experiments/serve/server_tp.py` (frozen production, fix only as P2 unless flagged)

- **P1** — lines 596-637, `_tp_all_reduce` carries 42 lines of `ccl_debug` /
  `_diag_output` instrumentation behind `getattr(state, 'ccl_debug', False)`.
  No code in-tree sets `state.ccl_debug = True`. The legacy print is dead.
  Fix: delete both blocks; the function body becomes the 9-line
  collective-dispatch tail.
- **P1** — lines 1457-1486, `mlp_step_tp` has 30 lines of
  `debug_mlp_resid` print scaffolding (B.2.2 Test 10 / Test 8 / Test 7).
  Same story: nothing sets `state.debug_mlp_resid` in the repo. Fix: delete.
- **P1** — lines 1704-1740, `forward_token_tp_inner` carries two more
  `debug_layer_boundary` print blocks (B.2.2 Test 7 / 8). Fix: delete.
- **P1** — line 1742, "P22 vocab-sharded LM head (see Agent X's resolution at
  feedback_lm_head_argmax_unknown.md)". Strip the Agent X / wiki cross-ref; keep
  the operative one-liner: "vocab-sharded LM head; per-chip linear →
  all_gather → slice → untilize → argmax".
- **P1** — line 564, in-line projection citing "Agent N's gap analysis
  (5a9808d)" with a saved-ms estimate. Delete — that's a commit message, not a
  source comment.
- **P1** — lines 53-58, `MAX_POS = 8192` carries six lines of dated context
  ("2026-05-21 bumped from 2048 → 8192… History: 256 → 512 → 2048 → 8192"). Keep
  the one-line constraint (`# 64 KB/token/chip; raise when long-context needs it`)
  and drop the history.
- **P2** — line 93, `self.traced_logits_tt = None  # legacy field, retained for
  safety; unused after vocab-sharded LM head ship`. Confirmed by grep — no read
  sites. Delete the field.
- **P2** — lines 100-103 + 509-524, `self.x_buf` / `cos_buf` / `sin_buf` all
  documented `# legacy` and allocated each bootstrap but never read on the hot
  path (the P25 swap to `tok_buf` / `rot_idxs_buf` made them dead). Delete the
  fields + the four `from_torch` calls.
- **P2** — lines 117-118, `self.trace_x_buf = None` / `self.trace_logits_buf =
  None`. Zero readers. Delete.
- **P2** — line 225, `from experiments.serve import ondevice_27b as _91f  # was
  the 91f importlib hack`. The "was the … hack" half is dead trivia. Drop it.
  Same on line 363 for `_91l`.
- **P2** — line 666, `REFACTOR (2026-05-20, task #77 prep): inner body moved to
  _deltanet_step_tp_from_inproj so the v4 chunked stub can call the inner body
  with pre-computed batched in_proj output. Behaviorally identical to the
  pre-refactor function.` Refactor commentary belongs in the PR / commit. Keep
  the docstring's first sentence ("One DeltaNet TP step on the mesh.") only.
- **P2** — lines 1300-1336, `deltanet_chunked_neumann_tp` docstring is 35 lines
  of design-doc / Stage history. Compress to a one-paragraph contract and link
  the design doc once.
- **P2** — lines 543-595 (`_rms_norm_manual`, `_tp_all_reduce` docstrings) carry
  multi-paragraph deal-history ("B.2.2 wedge fix 2026-05-19", "DIAG
  (2026-05-20)…", citing commits 3822293 / 1fabf07 / 2d30af7). Keep the
  one-line invariant (rms_norm wrapper; `all_reduce` with `num_links=2` +
  `Ring` topology). Drop the rest.
- **P2** — `MeshServerState.__init__` has 30+ `# P<num>:` comments tagging
  fields with the plan-phase they were introduced. New contributor doesn't need
  the plan numbers. Replace with a one-line purpose where non-obvious; delete
  otherwise.

### `experiments/serve/server_tp_cb.py`

- **P1** — line 93 (top of `setup_cb_state`), the conv-state docstring runs to
  line 248 of in-function comments explaining `shiftacc` vs `kdim` op-orders,
  ending with "DEFAULT 'kdim' is BIT-IDENTICAL to production… 'shiftacc' is the
  fast (28.76× isolated) path but its bf16 op-order DRIFTS over a sequence". The
  invariant is fine; the 12-line essay isn't. Fix: collapse to two lines —
  "`kdim` is bit-identical to production; `shiftacc` is fast but drifts past pos
  ~30, gate via needle test before enabling."
- **P1** — `gated_attn_step_batched` is **80 lines** (lines 397-478) with
  ~30 lines of view-decay warning comments interleaved (421-425, 450-457,
  474-475, 482-484). The warnings are correct and necessary, but currently they
  obscure the math. Fix: lift a single `# view-decay invariant:` paragraph at
  top of function (one location), drop the inline restatements at each site.
- **P1** — `deltanet_step_batched` is **185 lines** doing 9 numbered phases.
  Acceptable for now, but: lines 223-227 + 251-254 + 326-330 + 377-378
  reference profiling-only `cb_dn_skip` / `cb_skip_blocks` / `cb_conv_mode` /
  `cb_dn_recurrence_mode` flags read via `getattr`. Each guard is "skip op X
  with a passthrough" for `experiments/cb/profile/*.py`. The current
  `cb_profile/*` files DO use them, so this is real. But each comment block
  re-explains what passthrough means. Fix: one top-of-file comment "5
  profiling-only `cb_*` getattrs are read by `experiments/cb/profile/{dn,blocks}.py`
  to ablate sub-ops; all default to empty / `'manual'` / `'kdim'` so the prod
  path is unaffected", then keep each guard line bare.
- **P2** — line 24, "This file is CB1+CB2. The Orca scheduler is CB3
  (separate)." References the plan-phase names. Drop — the docstring summary
  already identifies the design.
- **P2** — line 545, "Mirrors server_tp.forward_token_tp_inner:1742-1748".
  Brittle line-number reference — line numbers in server_tp.py change with
  every edit. Drop the line range, keep "mirrors
  server_tp.forward_token_tp_inner".
- **P2** — `TILE_H = 32` (line 413) appears as a bare local in
  `gated_attn_step_batched`. `BLOCK_SIZE = 32` is imported from `base` (line
  56). Same constant, two names. Reuse `base.TILE_HEIGHT` (defined in
  server_tp.py:64).
- **P2** — `forward_batch_tp_inner` uses `assert NKV_PER_CHIP == 1`
  (line 414). That's a load-bearing invariant; OK to keep as `assert` but
  promote to a one-line constants block at top of file rather than buried in
  the function. Same goes for "paged SDPA per-chip assumes 1 KV head".

### `experiments/serve/cb_engine.py`

- **P2** — module docstring (lines 1-20) references the production-plan phases
  ("P0 of the production server (research/production_server_plan.md)") and
  walks through the design ("Greedy decode for P0 — per-request sampling is
  P1"). Plan-phase tags are journey artifacts. Compress to: "Thread-safe front
  end to the CB scheduler. One device-owning thread runs `drain inbound → drain
  cancels → step → stream`. Callers interact via queues only."
- **P2** — line 73, `# sampling=True → eager per-slot temp/top-p/top-k (the
  chat-API mode); sampling=False → greedy argmax trace (P0 fast path). See
  Scheduler.` Drop "(the chat-API mode)" and "(P0 fast path)" — the rest is
  good.
- **P2** — line 76, `# Backpressure: cap on total in-flight requests
  (queued+active). When at cap, submit() raises queue.Full → API maps to HTTP
  429. Default unlimited.` Good content but it's documenting the public
  `max_inflight` parameter — move into the `submit()` docstring or
  `__init__`'s parameter doc instead of an inline comment.
- **P2** — line 185, `except BaseException as e:  # surface bootstrap/capture
  failure to start()`. `BaseException` swallows `KeyboardInterrupt` and
  `SystemExit`; either tighten to `Exception` or note explicitly why
  `BaseException` is intended (mesh init can raise `SystemExit` on some
  Tenstorrent failures — but verify).

### `experiments/serve/cb_scheduler.py`

- **P2** — line 94-102, the `sampling` /  `use_trace` switch matrix is 8 lines
  of "argmax trace (P0 fast path) | logits trace | eager logits forward (slow;
  kept for non-traced testing)." Move to a 3-line table at top of class or the
  `step()` docstring, since `step` is where the dispatch happens.
- **P2** — line 117, `_capture_trace` docstring spans 8 lines and ends with
  "the per-slot host argmax/sample loop in _step_sampled reads that handle." —
  fine. The "(CB4 pattern)" parenthetical is plan-jargon; drop.
- **P2** — line 286, `state = base.MeshServerState() if hasattr(base,
  "MeshServerState") else base.State()`. The `hasattr` fallback to `base.State`
  is for an old code path; `server_tp.py` only defines `MeshServerState`.
  `hasattr(base, "MeshServerState")` is always True. Same pattern is duplicated
  in **15 files** across `experiments/cb/`. Delete the fallback everywhere;
  it's cargo cult.
- **P2** — lines 287-290, the three `state.deltanet_*_mode = "manual"` /
  `"native_softplus"` lines are repeated **9 times** across the CB stack
  (validate + bench + load + needle). Extract a `_force_cb_manual_modes(state)`
  helper in `server_tp_cb.py` (one-line setter) and call it from every entry
  point.

### `experiments/serve/cb_api.py`

- **P0** — `_build_app_with_default_lifespan` builds an engine inside the
  lifespan but does **NOT** wire the engine's shutdown into the cancel path
  reliably: `engine.stop()` is called in the `finally`, good. However, the
  `state` dict captured by the closure is shadowed by the validator (line 171
  of `engine_api.py` reassigns the local `state` to a fresh dict). Not a bug
  here — but the **closure-over-mutable-dict pattern** is fragile and the
  validator-vs-lifespan reuse depends on _build_app being called twice. Worth
  a one-line module comment: "`state` is a dict so the lifespan can fill it
  after FastAPI captures it — `_build_app` MUST be called BEFORE the handlers
  run."
- **P1** — line 49 `_try_get` is a 3-line helper used twice; OK. But line 100
  `_drain_handle` has a `while handle.final is None: ... if m[0] != "tok":
  handle.final = m[0]; break` post-cancel drain loop. The `_drain_handle`
  generator's contract isn't obvious from the call sites. Fix: 1-line comment
  on `yield payload` saying "tokens stream to the SSE caller; final markers
  set handle.final and stop the generator".
- **P2** — lines 17-25, docstring's "P2 of the production server", "Greedy and
  sampled requests share one engine (sampling-mode); temperature<=0 normalises
  to greedy." Compress and drop "P2 of the production server"; the file path is
  the identifier.
- **P2** — lines 195-225, `_build_app_with_default_lifespan` docstring says
  "Bootstrap is sync + slow (~350s on qb1) — runs in the default executor so
  the event loop stays responsive." That's a real invariant; keep. The rest is
  scaffolding — line 198 imports `server_tp as base` and `cb_engine` inside the
  function, which is fine for the lazy-on-import reason.
- **P2** — `app = _build_app_with_default_lifespan() if os.environ.get(...
  "1") == "1" else None`. The env-guarded module-level app is a known smell.
  Comment is OK ("keeps unit tests free of fastapi/transformers imports") but
  the env-var name `TT_CB_API_BUILD_APP` collides conceptually with
  `TT_OPENAI_BUILD_APP`. Document both in one place (e.g.
  `experiments/serve/__init__.py` or a README).

### `experiments/serve/cb_metrics.py`

- **P2** — small and clean. Two nits: (a) `Counter.get()` (line 42) is not
  locked; if a reader interleaves with `inc()` they get either pre- or
  post-value, which is fine for monotonic counters — but worth a one-line
  comment "intentionally unlocked: reader sees monotonic value either side of
  inc". (b) `DEFAULT_BUCKETS` (line 25) is documented well; no change needed.

### `experiments/cb/validate/forward.py`

- **P1** — lines 138-152, three different `# CB uses manual recurrence + …`
  comment blocks restating the same invariant. Compress to one.
- **P1** — lines 162-170, "Two FRESH passes (each consumes its own state once;
  no stale re-run — the earlier logit-check bug re-ran prod over already-
  consumed KV/DN)." Lines 167-172 add another "NOTE: a standalone hidden-state
  ladder here would re-run prod…". Both warn against the same earlier bug. One
  comment, one location.
- **P2** — line 65 has `prompt = ("The capital of France is the city of
  Paris, which has long been a center of art, science, philosophy, …")`
  duplicated in `prefill_generate.py:71` and `long_context.py` (variant).
  Promote to a single `experiments/cb/_prompts.py` constants file.

### `experiments/cb/validate/engine_api.py`

- **P1** — line 43, `import os` after the other imports (after the `# noqa:
  E402` block) because `os.environ` mutation has to happen before
  `from cb_api import _build_app` on line 45. Real lazy-init reason — but the
  ordering will trip future readers. Fix: hoist `import os` to top of file and
  inline-comment the mutation `os.environ["TT_CB_API_BUILD_APP"] = "0"  # must
  precede cb_api import` so the order isn't accidentally regrouped.
- **P1** — line 273, `# give the server > DISCONNECT_POLL_S (+ drain margin)`
  references a constant `DISCONNECT_POLL_S` that does NOT exist anywhere in
  the repo (`grep` confirms — no definition). The actual poll period in
  `cb_api.py:108` is a hardcoded `1.0`. Either define the constant in `cb_api.py`
  and reference it here, or drop the symbolic reference in favor of "1 sec
  poll".

### `experiments/cb/load/concurrent_chat.py`

- **P2** — clean. Note: `_pct` (line 149) reimplements percentiles by hand
  even though `statistics.quantiles` is imported indirectly. Not worth changing
  for the saved 5 LOC.

### `experiments/cb/bench/trace.py` and `bench/throughput.py`

- **P2** — both define a `def log(msg)` boilerplate + the
  `MeshServerState() if hasattr(...)` fallback. Covered by the shared-helper
  fix in the scheduler entry above.
- **P2** — `bench/trace.py:128-131` and `bench/throughput.py:72-75` have the
  same `try: ttnn.deallocate(...); except Exception: pass` cleanup loop for
  freeing the KV pool between batches. Lift to a `_free_cb_kv(state)` helper
  in `server_tp_cb.py`.

### `experiments/cb/needle.py`

- **P2** — clean and self-contained.

### `experiments/cb/validate/sampling.py` (60 lines)

- **P2** — clean. The pattern repeats the boilerplate from above.

### `experiments/cb/validate/prefill.py`, `prefill_generate.py`, `long_context.py`

- **P2** — all three repeat the long-prompt string (a 200-char Paris sentence).
  See punch-list item under `forward.py`: extract to `experiments/cb/_prompts.py`.

### `experiments/serve/import_smoke.py`

- **P2** — clean. The "untangle" docstring (line 4) references "91f/91l" rename
  history; trim to "Smoke-imports `ondevice_27b` + `generate_27b` and checks
  the symbols `server_tp.py` calls into."

### `experiments/serve/openai_endpoint.py` (frozen legacy)

- **P1** — line 105, `@app.on_event("startup")` is deprecated in current
  FastAPI; the new path is `lifespan` (which `cb_api.py` uses). If this proxy
  is intended to keep running, swap to `lifespan` to silence the warning. If
  it's frozen, fine.
- **P2** — lines 1-13, docstring runs to "The translation helpers
  (_messages_to_prompt, _chat_completion, _chat_chunk) are pure + unit-tested
  in experiments/serve/tests/test_openai_endpoint.py." The unit-tested fact is
  good; keep. The "Run on the TT host (after `serve_tp.sh start`)" block: keep
  one line max.

### `experiments/serve/tests/test_cb_api_routing.py`

- **P2** — clean and fast. Module docstring is a model of how this audit wants
  docstrings to read.

### `scripts/`

- **P1** — `scripts/deploy.sh:13-18` hardcodes the default sync list. The new
  CB stack files (`cb_engine.py`, `cb_metrics.py`, `cb_api.py`,
  `experiments/cb/*`) are NOT in the default `set --` block. `make dr` syncs
  only `server_tp{,_cb}.py` + a handful of older modules; running `make dr
  PY=experiments/cb/validate/engine.py` on a fresh host will fail because
  `cb_engine.py` was never deployed. Fix: extend the default list (or use a
  globbed `experiments/serve/*.py experiments/cb/**/*.py` rsync).
  `scripts/ci_check_deploy_sync.py` only checks `server*.py` imports so it
  won't catch this gap.
- **P2** — `scripts/check_setup.sh:31`, `fail=$((fail + 1))` increments a
  counter shared across subshells. Works because bash arithmetic is in-process
  here; OK.
- **P2** — `scripts/run_remote.sh:23-28`, the env block is hand-written.
  Could be cleaner with a `bash -c "$(printf 'A=1 B=2 …\n')"` but it's read
  twice a session by humans; leave alone.
- **P2** — `scripts/install_ttnn.sh:32`, "`>>> uv pip install setuptools_scm
  (ttnn's setup.py needs it)…`" is fine; no fix.
- **P2** — `scripts/chat.py:37-41`, `BOLD = lambda s: _ansi("1", s)` (noqa
  E731). 5 lambdas where a single helper `def color(code, s)` reads better;
  not worth the churn.
- **P2** — `scripts/strip_functions.py` is good — small, well-named,
  dry-run-by-default. No changes.
- **P2** — `scripts/build_owned_ops.sh:96-102`, the verification heredoc
  inlines a Python script. Acceptable here because it's a 7-line sanity
  print, not a permanent helper.

### `Makefile`, `pyproject.toml`

- No issues worth flagging.

---

## Patterns worth standardising

| Concern | Current variants | Recommend |
|---|---|---|
| `PROJECT_ROOT` discovery | env-var (server_tp.py); `parents[2]` (server_tp_cb, cb_scheduler); `parents`-walk-for-`experiments/serve` (cb_engine, cb_api) | the parents-walk form — robust to renames and clone path |
| Validator boilerplate | 15 files redo PROJECT_ROOT + `sys.stdout.reconfigure` + `def log()` + `state = …MeshServerState()` + 3 mode setters | `experiments/cb/_runner.py` with `bootstrap_cb_manual()` + `log` |
| `MeshServerState` fallback (`hasattr(base, "MeshServerState") else base.State()`) | 15 files | Delete — `base.State` doesn't exist any more |
| DN mode override block (`"manual"` / `"manual"` / `"native_softplus"`) | 9 files | `server_tp_cb.force_cb_manual_modes(state)` one-liner |
| `def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)` | ~15 copies | move to shared helper |
| Long prompts (Paris paragraph) | 3 verbatim copies | `experiments/cb/_prompts.py` |
| `# legacy:` fields kept "for safety" | 6 in `MeshServerState` | delete (grep proves no readers) |
| `ttnn.deallocate(...) try/except Exception: pass` cleanup loop | 2 (bench files) | `server_tp_cb._free_cb_kv(state)` |
| Plan-phase comments (P0/P1/P2/P4/P14/P22/P25/CB1-4) | scattered | the file IS the artifact; delete the phase tag |

---

## Quick-fix priority order

1. **(P0 / blocking)** Fix `scripts/deploy.sh` so `make dr` on a fresh host
   actually deploys the CB stack — currently the default arg list omits
   `cb_engine.py`, `cb_metrics.py`, `cb_api.py`, and `experiments/cb/**`.
2. **(P1 / 1-hour win)** Strip the three debug-print blocks in
   `server_tp.py` (lines 596-637, 1457-1486, 1704-1740). ~100 LOC, zero
   behaviour change.
3. **(P1 / 1-hour win)** Extract the validator boilerplate to
   `experiments/cb/_runner.py` (PROJECT_ROOT + log + bootstrap + DN-mode
   setters). 9-15 LOC × 15 files saved.
4. **(P1)** Compress the narrative docstrings in `server_tp_cb.py`'s
   `deltanet_step_batched` (lines 240-292 conv1d essay) and
   `gated_attn_step_batched`'s view-decay warnings (one top-of-fn note
   instead of 4 inline restatements).
5. **(P1)** Drop the dead `MeshServerState` legacy fields (`x_buf`, `cos_buf`,
   `sin_buf`, `traced_logits_tt`, `trace_x_buf`, `trace_logits_buf`) + their
   bootstrap allocations.
6. **(P1)** Fix `experiments/cb/validate/engine_api.py:273` — reference to
   undefined `DISCONNECT_POLL_S`. Either define the constant in `cb_api.py` or
   say "1 sec poll".
7. **(P1)** Strip Agent-X / commit-SHA / dated-decision comments from
   `server_tp.py` (lines 91, 131, 564, 666, 1317, 1742, 2144, 2899; not
   exhaustive — search for `commit [0-9a-f]{6,}`, `Agent [A-Z]`, `task #`,
   `# 202[56]-`).
8. **(P2)** Remaining items above.

---

## What this audit did NOT touch (with reasons)

- **`server_tp.py` is frozen production code** — the audit only flags P0/P1
  items that are dead code / actively misleading, never anything that touches
  the forward path. The 3095-line size is acknowledged but no split is
  proposed; a split is a refactor, not a cleanup.
- **`experiments/utils/full_layer_tp_probe.py` and friends** — excluded per
  scope; the probe layer is pending its own archive pass.
- **`server_35b*.py`, `server.py`, `ondevice_27b.py`, `generate_27b.py`,
  `client*.py`, `protocol.py`** — out of scope.
- **Owned-op C++ kernels** — only the Python integrate scripts would be
  audit-able; they are clean as far as the build script driver pattern goes.
- **`research/`, `wiki/`** — not source code.
- **The actual algorithmic correctness of CB scheduling / batched DN / paged
  attention** — this audit is a maintainability pass, not a correctness
  re-verification. The validators (`forward.py`, `ragged.py`, `engine.py`,
  `engine_api.py`) exist and pass; trust them.
