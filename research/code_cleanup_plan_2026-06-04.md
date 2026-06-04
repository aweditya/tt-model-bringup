# Code cleanup plan — 2026-06-04

Targeted, surgical bug/inconsistency/workaround inventory for
`experiments/serve/` + `experiments/cb/dev/` + `scripts/`. The user has
explicitly directed: no bloat, no leaving bugs as known-broken-with-flag-
workaround, no introducing more options. Every entry below either fixes
the underlying bug or deletes the workaround code path entirely.

Scope drawn from: `HANDOFF.md`, MEMORY.md `feedback_*` files for
known-broken regimes (35B manual recurrence, 35B B>1 empty-slot poison,
cb_reset no-op, dispatch holes, deploy gap, dev-harness gap, harness
hang), recent commit history (last ~50 commits), and a grep audit of
`experiments/serve/` for `TODO|FIXME|HACK|workaround|broken|backend ==`.

---

## 1. Executive summary

- **18 cleanup items found across A–F.** 6 are silent-correctness (High),
  7 are maintenance tax (Medium), 5 are cosmetic (Low).
- **Total scoped effort: ~22 hours** if all High+Medium are done. The
  Low items are deletions and amount to ~1 hour combined.
- **Top-3 leverage** (in order):
  1. **#A1 — Fix or delete the 35B manual DN recurrence path**
     (`server_35b_ttnn.py:608-672`). It's been broken since the start
     (cos@L32 pos 0 = 0.08), and it papers over the broken `TT_DN_STATE_DTYPE=fp32`
     hook in `bootstrap()`. Deleting it removes the broken else-branch,
     an environment knob, and an entire stale code path (~80 LOC).
  2. **#A2 — Fix or delete the 35B B>1 empty-slot poison workaround**
     (`cb_api.py:325-333`, default `TT_CB_SLOTS=1` for `TT_BACKEND=35b`).
     This carve-out exists only because `forward_batch_tp_inner_batched`
     poisons slot 0 when slot 1 is empty. Closing it deletes the
     backend-specific default and unblocks 35B continuous-batching
     parity with 27B.
  3. **#B1 — Unify the 27B-only `deltanet_*_mode` writes in
     `cb_api.py:292-295`**. Lines that set `st.deltanet_recurrence_mode`
     etc. are 27B-only feature flags being plumbed from a generic
     bootstrap — they should live in `server_tp.MeshServerState.__init__`
     (where they already partly do) and `cb_api` should never know
     about them.

---

## 2. Per-issue table

| ID | Title | Cat | Sev | Effort | Fix sketch |
|---|---|---|---|---|---|
| A1 | 35B manual DN recurrence path is structurally broken | A | High | l | Delete `server_35b_ttnn.py:608-672` else-branch, drop `TT_DN_STATE_DTYPE` env knob (`:1729-1740`), drop `dn_state_dtype` field from State. The fp32 H_t experiment is closed-negative. |
| A2 | 35B B>1 empty-slot poison → forced `TT_CB_SLOTS=1` | A | High | l | Probe `forward_batch_tp_inner_batched` (`server_35b_cb.py:760-840+`) with cur_pos=-1 ragged input, identify the op that bleeds slot 1's garbage into slot 0 (most-likely SDPA mask or batched MoE Pattern A); fix it; delete the `_slots_default = "1" if TT_BACKEND == "35b"` branch in `cb_api.py:325-333`. |
| A3 | 35B `return_logits` is silently broken (issue #149) | A | High | m | `forward_batch_tp_inner` (`server_35b_cb.py:760-777`) raises `NotImplementedError` and `cb_api.py:336-344` defaults `TT_CB_TOPK_K=64` to route around it. Fix the bulk-readback path on `[1, VOCAB]` (cb35_v0 already does correct on-device argmax — fold that into the readback), then delete the backend-specific `_topk_default` carve-out in cb_api. |
| C1 | `state.dn_caches_tt`/`kv_caches_tt` re-alias in `cb_reset_states` is double-bookkeeping | C | Medium | s | `server_35b_cb.py:202-237`: at B=1 we dealloc then re-alias the lists; the comment admits the existing `reset_caches_ttnn` "rebinds the list, leaking the old tensors". Fix `state.reset_caches_ttnn` itself to dealloc-then-rebind (per `[[ttnn-list-rebinding-leaks]]`), then drop the explicit dealloc loop here. |
| C2 | 35B B=1 setup_cb_state aliases vs B>1 separate allocations is two code paths for one contract | C | Medium | m | `server_35b_cb.py:105-185` splits into `if B==1: alias // else: alloc`. Merge to one alloc path: at B=1 the alias is a perf micro-opt worth ~80 MB; pay it. Removes a class of bugs where edits to one branch don't land in the other (#162 caught this). |
| B1 | 27B-only `deltanet_*_mode` writes in cb_api bootstrap | B | Medium | xs | `cb_api.py:289-295`: `if TT_BACKEND == "27b": st.deltanet_recurrence_mode = "manual"…`. These belong in `server_tp.MeshServerState.__init__` (which already sets them, lines 122-132) — the cb_api write OVERWRITES the State default, which is itself the result of A/B tuning. Delete the cb_api block. |
| B2 | Two separate `_BACKEND_MODULES` registries (cb_api + cb_scheduler) | B | Medium | s | `cb_api.py:50-59` and `cb_scheduler.py:50-61` define overlapping dispatch tables. The memory `[[cb-backend-dispatch-holes]]` documents a multi-hour debug from these drifting. Promote to one shared module (e.g. `experiments/serve/backends.py`) imported by both. |
| B3 | `cb_scheduler._warmup_prefill` is 27B-only but gated by `hasattr` checks | B | Medium | s | `cb_scheduler.py:377-403`: `if not hasattr(st, 'prefill_chunk_size')` is implicit "is this 27B?" Hide behind the backend dispatch table — add a `supports_chunked_prefill` flag to `BACKENDS` entries and gate the call site (`cb_scheduler.py:199-209`) on that, not on `hasattr`. |
| B4 | `chunked_prefill=True` writes 27B-internal state fields from cb_scheduler | B | Medium | xs | `cb_scheduler.py:168-170`: scheduler reaches into `state.cb_conv_mode` / `state.cb_dn_recurrence_mode` (27B DN internals) to force "kdim"/"manual". Either move into `cb.setup_cb_state(state, B)` (the 27B-specific setup is the right home), or behind a `cb.prepare_for_chunked_prefill(state)` hook. |
| C3 | Profiling-flag dead branches in `server_tp_cb.py` | C | Low | s | `cb_dn_skip`, `cb_skip_blocks`, `cb_conv_mode='shiftacc'` branches (`server_tp_cb.py:345-415, 653-665`) exist only for `experiments/cb/profile/{dn,blocks}.py` to attribute timing. Keep them but make them a single explicit `class ProfileFlags` dataclass attached to State at debug time only — drop the `getattr(state, 'cb_dn_skip', None) or set()` defensive idiom that bloats every iteration of the forward. |
| E1 | Defensive `getattr(state, "X", None)` lazy-inits left over from harness-reload skew | E | Low | s | 35B/Gemma 4 forwards use `getattr(state, "dn_owned_gdn", False)` (`server_35b_ttnn.py:1080-1084`, etc.). These were added per `[[harness-state-version-skew]]` so existing harness sessions wouldn't break when new fields landed. For fresh sessions they're dead. Replace each with direct `state.dn_owned_gdn` once the corresponding field is initialised in `__init__`/`bootstrap`. |
| E2 | `getattr(state.tok, "eos_token_id", None)` in two callsites of server_tp | E | Low | xs | `server_tp.py:2691, 2796`. The tokenizer always has `eos_token_id`; the defensive `getattr` predates the explicit eos plumbing. Direct attribute access. |
| F1 | Stale TODO comments in `server_35b_cb.py` (cb_reset, return_logits, prefill stub) | F | Low | xs | `server_35b_cb.py:240-273` (`v1 will use masked multiply`), `:760-777` (#149), `:783-786` (`No-op stub for v0`). These were placeholders; either they shipped (delete TODO), or they're still open (link to the issue and stop apologising in code). |
| F2 | `ondevice_27b.py:405-411` C'3 native RoPE post-mortem comment | F | Low | xs | A 7-line "we tried X and it didn't work" comment in the hot forward path. Move to `research/` (it's research notes, not production code documentation). |
| D1 | Dev-harness trigger-dir contract drift | D | Medium | s | `cb35_dev_harness.py` uses `.cache/cb35_runtime/trig/`; `gm4_dev_harness.py` uses `.cache/gm4_runtime/trig/`; `scripts/run_harness_tmux.sh` picks the runtime dir from `$MODEL`. No drift today, but the contract isn't documented anywhere — promote to a constant in a shared module (e.g. `experiments/cb/dev/_harness.py`) that both harness files import. |
| D2 | `cb35_dev_harness` hang hardening (#166) referenced as "shipped 84efe50" but not verified after pivot | D | Medium | xs | `feedback_cb35_dev_harness_hung_2026-06-03.md` says "HARDENED commit `84efe50`". Verify the heartbeat + try/except actually log + survive the next idle period. If yes, drop the warning from HANDOFF and the memory entry's "ship before next bootstrap" call-out. |
| D3 | `scripts/run_harness_tmux.sh` PASS_THROUGH_ENV hand-maintained allowlist | D | Low | xs | Currently 4 env vars hard-coded. Use `env | grep '^TT_'` to forward all `TT_*` env vars; one-line change in `scripts/run_harness_tmux.sh:30-35`. |
| F3 | Manual recurrence comment in `server_35b_ttnn.py:608-618` references the WRONG memory entry | F | Low | xs | The comment cites `[[35b-batched-forward-empty-slot-poison]]` then corrects itself to `[[35b-dn-h-state-drift-lever]]` mid-sentence. This is git-blame archaeology, not docstring. After A1 lands, this block is gone. If A1 is deferred, clean the comment. |

---

## 3. Top-10 fixes worth doing first

### #1 — A1: Delete the 35B manual DN recurrence else-branch + `TT_DN_STATE_DTYPE` knob

- **Files** `experiments/serve/server_35b_ttnn.py:608-672` (else branch of `dn_forward_ttnn`); `:1080-1091` (the use-of-flag site in `step_forward_inner`); `:1720-1740` (the `TT_DN_STATE_DTYPE` bootstrap hook); `:1435-1450` (the `dn_state_dtype` propagation).
- **The bug**: `dn_forward_ttnn` `use_owned_gdn=False` path produces cos@L32 pos 0 = **0.08** vs HF oracle (owned_gdn=ON gives 0.99). Per `[[35b-manual-recurrence-path-broken]]`. The fp32 H_t fix (`TT_DN_STATE_DTYPE=fp32`) auto-disables `dn_owned_gdn` and routes through this broken path — that's why it didn't move the drift number.
- **The proper fix**: nobody uses the manual path correctly. Delete the entire else-branch (lines 608-672), the `dn_owned_gdn` getattr (`:1080`), and the `TT_DN_STATE_DTYPE` env hook (`:1720-1740`). Keep `dn_owned_gdn` as a hard-coded `True` constant ONLY if anyone calls `dn_forward_ttnn(use_owned_gdn=False)` from a probe (`grep -rn 'use_owned_gdn=False' experiments/`). If no callers, drop the parameter.
- **Why high-leverage**: deletes ~80 LOC of dead code, removes a state field, removes an env-var feature flag, kills the entire "fp32 H_t is the drift lever" hypothesis from the codebase (`[[35b-dn-h-state-drift-lever]]` is closed-negative). Also resolves the inconsistency where probes documented in `research/35b_perf_milestones.md` (A010) reference a path that no longer works.
- **Test**: `cb35_drift_ladder` probe via the dev harness — pos 0 cos@L32 must remain ≥ 0.99 (current owned_gdn=ON number). Tag CB35-v0 smoke 3/3 still passes (`experiments/cb/validate/cb35_v0_smoke.py`).

### #2 — A2: Fix the 35B B>1 empty-slot poison, delete the `TT_CB_SLOTS=1` default

- **Files** `experiments/serve/server_35b_cb.py:760-967` (`forward_batch_tp_inner_batched` + `layer_forward_batched_35b` + SDPA call sites); `cb_api.py:325-333` (the `_slots_default = "1" if TT_BACKEND == "35b" else "4"` block).
- **The bug**: when slot 1 has `cur_pos=-1` and slot 0 has a real prompt, slot 0's argmax output collapses to a deterministic Chinese-char loop independent of the prompt. Per `[[35b-batched-forward-empty-slot-poison]]`.
- **The proper fix**: per the existing memory entry, the cheapest probe is to dump `h_tt` at L=0 for both slots after the prelude and compare TT_CB_SLOTS=1 vs TT_CB_SLOTS=2-with-empty-slot-1. Three likely loci: (a) batched SDPA where `cur_pos_buf[1] = -1` is not properly masked (Gemma 4 v0.3 memory `[[paged-update-cache-nkv-per-chip]]` notes that `cur_pos=-1` documents a "skip" semantics — verify 35B's SDPA call respects it); (b) batched MoE Pattern A's `[E_LOCAL, B, HIDDEN]` matmul (`[[ttnn-moe-per-slot-drift]]` is related); (c) batched DN recurrence cross-slot reduction. **The Gemma 4 SDPA doc check** (HANDOFF §"Bonus") suggests `cur_pos=-1` skip-semantics may already cover the SDPA-mask case — try plumbing that first.
- **Why high-leverage**: removes the only place where cb_api branches on TT_BACKEND for an operational default (other branches are correctness-tested defaults). Unlocks 35B serving B=4 like the rest of the fleet.
- **Test**: `cb35_prod_topk.py` with TT_CB_SLOTS=2 and one admit must produce coherent English. End-to-end HTTP smoke at TT_CB_SLOTS=4 over `/v1/chat/completions` with 1, 2, and 3 concurrent clients.

### #3 — B1: Move `deltanet_*_mode` defaults out of cb_api into the backend State

- **Files** `experiments/serve/cb_api.py:289-295`; `experiments/serve/server_tp.py:122-132`.
- **The bug**: `cb_api.py:292-295` writes 27B-specific feature flags AFTER `bootstrap()` returns — the same flags that `MeshServerState.__init__` (`server_tp.py:122-132`) already sets, but to different values (`__init__` sets `"owned_gdn"` and `"owned_decay_gate"`; `cb_api` overwrites with `"manual"`). The cb_api overwrite is wrong — it disables P1+G4 perf wins (`[[p1-num-links-2-shipped]]`, `[[owned-decay-gate-shipped]]`).
- **The proper fix**: delete `cb_api.py:289-295`. The State default is the right value; if anyone needs `manual` for a profile probe, they set it explicitly (which they already do — `experiments/cb/profile/dn.py:86` does `state.cb_dn_recurrence_mode = "owned_gdn" if args.owned_gdn else "manual"`).
- **Why high-leverage**: removes a generic-infrastructure file (cb_api) from knowing 27B internals; restores P1+G4 perf wins that are currently being clobbered on every prod boot (verify in a server log — current bootstrap log MUST be reporting `manual`, contradicting `[[branchC-perf-state]]`). Also makes cb_api `if TT_BACKEND == "27b"` clean it from the file entirely.
- **Test**: `tools/serve_cb.sh start` + `tools/client_tp.py` → 5-token Paris greedy check; tok/s number from `/metrics` should match the 12.93 tok/s headline in HANDOFF.

### #4 — A3: Fix 35B `return_logits` (#149), delete `TT_CB_TOPK_K=64` default

- **Files** `experiments/serve/server_35b_cb.py:760-777`; `experiments/serve/cb_api.py:336-344`; `experiments/serve/server_35b_ttnn.py` lm_head/argmax callsites.
- **The bug**: 35B's `[1, VOCAB]` ttnn.to_torch readback returns garbage that varies per run; on-device argmax of the same tensor finds the right answer. The CB system routes around this by forcing topk-mode (TT_CB_TOPK_K=64) for 35B.
- **The proper fix**: the on-device argmax + 8-byte readback path that cb35_v0 uses (`[[b16i-full-ondevice-35b]]`) works correctly. Plumb it into `forward_batch_tp_inner` as the default sample path — `return_logits=False` returns argmax_tt of shape `[B, ?, 1]`. For sampling, the on-device top-k path is already there. Delete the `_topk_default = "64" if TT_BACKEND == "35b"` carve-out.
- **Why high-leverage**: removes a second cb_api/TT_BACKEND backend-specific default; closes #149.
- **Test**: `cb35_prod_topk.py` 4/4 PASS without `TT_CB_TOPK_K=64`. Server smoke with `TT_CB_TOPK_K=0` (full logits) on 35B should still return coherent text.

### #5 — B2: Unify the two `_BACKEND_MODULES` registries

- **Files** `cb_api.py:50-59`; `cb_scheduler.py:50-61`. New file `experiments/serve/backends.py`.
- **The bug**: both files hold near-identical backend dispatch tables. Per `[[cb-backend-dispatch-holes]]` and `[[deploy-serve-files-too]]`, a one-hour debug already happened from them drifting.
- **The proper fix**: extract a single `BACKENDS: dict[str, BackendSpec]` registry into `experiments/serve/backends.py`. `BackendSpec` is a small dataclass: `(base_module: str, cb_module: str, default_model_id: str, supports_chunked_prefill: bool, supports_logits_readback: bool, default_slots: int, default_topk_k: int | None)`. Both cb_api and cb_scheduler import it. Adding a fourth backend = one entry in one file.
- **Why high-leverage**: kills the entire dispatch-hole bug class. Naturally subsumes A2's `default_slots` and A3's `default_topk_k` as registry fields rather than `if TT_BACKEND ==` carve-outs.
- **Test**: `gm4_v2_wireup_smoke.py` should still pass (it asserts both `cb_api.TT_BACKEND` and `cb_scheduler._TT_BACKEND` agree); add a similar smoke for 27B + 35B.

### #6 — B3+B4: Move chunked-prefill setup into the 27B-specific module

- **Files** `cb_scheduler.py:168-170, 199-209, 377-403, 388-403`; `experiments/serve/server_tp_cb.py` (new function `cb_prepare_for_chunked_prefill`).
- **The bug**: scheduler reaches into 27B DN internals (`state.cb_conv_mode`, `state.cb_dn_recurrence_mode`) and gates calls on `hasattr(st, 'prefill_chunk_size')` (an implicit "is this 27B" check).
- **The proper fix**: add `supports_chunked_prefill: bool` to the `BackendSpec` from #5; in cb_scheduler `__init__` only enable the chunked-prefill warmup if the spec says yes, and call a `cb.prepare_for_chunked_prefill(state)` hook that lives in `server_tp_cb.py` (Gemma 4 + 35B can `def prepare_for_chunked_prefill(state): raise NotImplementedError`). Drop the `hasattr` checks.
- **Why high-leverage**: cb_scheduler stops being a 27B/35B/Gemma 4 polyglot — it sees only the abstract contract.
- **Test**: cb_scheduler unit tests + 27B prefix-cache smoke (1.97× turn-2 speedup gate from HANDOFF "Recommended runtime config").

### #7 — C1+C2: Collapse 35B `setup_cb_state` B=1/B>1 into one allocation path

- **Files** `experiments/serve/server_35b_cb.py:105-185, 202-237`.
- **The bug**: two code paths for what should be one shape contract. B=1 aliases existing single-stream caches; B>1 allocates fresh. The cb_reset_states then has a `B==1: dealloc+re-alias` vs `B>1: masked-zero-multiply` split (`server_35b_cb.py:202-237`). Per `[[35b-cb-reset-slots-b-gt-1-noop]]` the original B>1 reset shipped as a no-op.
- **The proper fix**: allocate per-slot caches always (even B=1). The ~80 MB cost at B=1 is dwarfed by the bug-class savings. cb_reset_states then only has the masked-zero-multiply path. The base.reset_caches_ttnn list-rebinding leak (`[[ttnn-list-rebinding-leaks]]`) should be fixed at the source (`server_35b_ttnn.py:reset_caches_ttnn`) — dealloc before rebind.
- **Why high-leverage**: removes a B==1/B>1 fork in two functions; closes the dispatch-skew class that produced #162 and a previous `cb_reset_slots` no-op.
- **Test**: cb35_v0_smoke (3/3 B=1) + cb35_v1_chat (B=2) both pass without regression on cos/argmax.

### #8 — E1: Replace defensive `getattr(state, ...)` with direct attribute access

- **Files** `server_35b_ttnn.py:1080-1084, 1611, 1702`; `server_35b_cb.py:73, 87, 90, 93, 98, 121, 682-684, 722`; `server_gemma4_unified_ttnn.py:339, 824, 1461, 1505`; `server_tp_cb.py:349, 381, 454, 656`; `server_tp.py:140, 173, 1098, 1897, 2691, 2796`.
- **The bug**: `[[harness-state-version-skew]]` documents these were temporary band-aids so existing harness sessions don't crash when a field is added to State. For fresh sessions they're dead-code defensiveness that obscures the actual State contract.
- **The proper fix**: for each State field referenced via `getattr(state, "X", default)`, initialise `X` to `default` in `State.__init__` (or `MeshServerState.__init__`), then change the access to direct `state.X`. Each is ~2-line edit.
- **Why high-leverage**: makes the State schema explicit in one place per backend. Tools like ruff / mypy then catch new bugs at edit time. Drops ~35 LOC of noise.
- **Test**: bootstrap each backend from a cold State and run the existing smoke gate. Run dev harness `_reload` trigger and confirm no harness-skew crash on a re-iteration.

### #9 — F1+F2+F3: Delete stale TODO/NOTE comments

- **Files** `server_35b_cb.py:240 (`v1 will use masked multiply` — landed)`, `:760-777 (#149 reference — track in issue tracker not source)`, `:783-786 (`No-op stub for v0`)`; `server_35b_ttnn.py:608-618` (wrong-memory-entry comment); `ondevice_27b.py:405-411` (C'3 RoPE archaeology).
- **The bug**: each comment is a snapshot of a debug session whose result has shipped; the comment is left as in-source archaeology.
- **The proper fix**: delete each. Replace with a one-line link to `research/` doc if the why is non-obvious. `ondevice_27b.py:405-411` archaeology moves to `research/c3_native_rope_post_mortem.md` (or just `[[c3-native-rope-abandoned]]` memory link).
- **Why high-leverage**: pure deletion; ~30 LOC. Hot paths get readable.
- **Test**: `git grep -n 'TODO\|FIXME\|HACK' experiments/serve/` returns 0 lines after.

### #10 — D1: Promote dev-harness trigger-dir contract to a shared module

- **Files** new `experiments/cb/dev/_harness.py`; `cb35_dev_harness.py:62-64`; `gm4_dev_harness.py` (likely lines ~60-70); `scripts/run_harness_tmux.sh:20-22`.
- **The bug**: trigger-dir path (`.cache/<model>_runtime/trig/`) and log path (`.cache/<model>_runtime/harness.log`) are repeated in two harness files and the shell script. Drift between them is silent.
- **The proper fix**: `experiments/cb/dev/_harness.py` exports `runtime_dir_for(model: str) -> Path`, `trig_dir_for(model: str) -> Path`, `log_path_for(model: str) -> Path`. Both harness files import. Shell script reads them via a one-line `python -c` introspection at startup (one of the few legit `python -c` exceptions; or a tiny CLI helper `experiments/cb/dev/_harness.py --print-trig-dir cb35`).
- **Why high-leverage**: closes the local-vs-deployed drift class for harness paths; future harnesses are <10 LOC.
- **Test**: existing dev harness triggers continue to fire (`touch ~/tt-xla/.cache/cb35_runtime/trig/v0_smoke` runs the test).

---

## 4. Don't-do list (looks like cleanup; actually load-bearing)

- **Multi-EOS `frozenset` in `cb_engine.py:104-112`**. This is a real contract (Gemma 4 IT has 3 EOS, Qwen3.6 chat path emits both `<|im_end|>` and `<|endoftext|>`). Don't collapse to a single int. Memory `[[deploy-serve-files-too]]` covers the smoke-debug chain.
- **Two-phase warmup** (compile-all-then-capture-all) in `cb_scheduler._warmup_decode` / `_capture_decode_trace_only`. Looks like four functions doing what could be one — but `[[ttnn-multi-trace-two-phase-warmup]]` is mandatory; interleaving warmup+capture causes the 99% CPU wedge. Document the constraint better (HANDOFF already does); don't refactor.
- **K-broadcast workaround in attn RoPE** (`server_35b_ttnn.py:986`). Sidesteps the ttnn `[1, HEAD_DIM]` slice/concat bug `[[qwen36-attn-rope-single-row-ttnn-bug]]`. Until that ttnn bug is fixed upstream, this stays.
- **`_rms_norm_manual`** (`server_tp.py:608` + callers). Not the same as `ttnn.rms_norm(weight=...)`. The 27B Qwen RMSNorm has +1 zero-centred offset (`[[qwen36-qnorm-knorm-zero-centered]]`). Keep.
- **Per-layer `layer_scalar`** in Gemma 4 forward. Real (`[[gemma4-layer-scalar]]`). Not bloat.
- **`prefix_cache` slot-level live-slot store** (`live_slot_store.py`). Real contract — `[[prefix-caching-design]]` documents why slot-level beats block-level for hybrid DN models.

---

## 5. Order of operations

Dependencies between fixes — sequence them so earlier work enables later deletions:

1. **#5 (B2 unify registries)** FIRST. It's prerequisite for clean #1/#2/#3/#4 (those become "delete the branch on TT_BACKEND" instead of "delete + add a new field to State"). After #5, the registry is the only `if TT_BACKEND ==` site to maintain.
2. **#3 (B1 deltanet defaults)**. xs-effort cleanup; verifies the registry pattern works in cb_api.
3. **#6 (B3+B4 chunked-prefill backend hook)**. Once the registry exists, this becomes a 1-field addition + 1 method hook. Resolves the cb_scheduler `hasattr` checks.
4. **#1 (A1 delete manual DN path)**. Independent of the registry work but should land before #2 (#2 may need to touch dn_forward in a debug session and we don't want a broken branch confusing the debug).
5. **#4 (A3 fix 35B return_logits)**. Independent; can interleave with #2. Once fixed, the TT_CB_TOPK_K backend default deletes itself via #5.
6. **#2 (A2 35B B>1 empty-slot poison)**. The hard one. Requires a probe of `forward_batch_tp_inner_batched`. Estimate ~4 hours. Lands after #1 cleans up the DN path noise.
7. **#7 (C1+C2 collapse setup_cb_state)**. After #2 — the per-slot allocation path becomes the only path, no B=1 fork.
8. **#8 (E1 defensive getattr cleanup)**. After all the State-schema changes from #1/#3/#5/#7 land, sweep the codebase once.
9. **#9 (F1+F2+F3 delete stale comments)**. Cleanup pass. After #1 the wrong-memory-entry comment auto-deletes.
10. **#10 (D1 dev-harness shared module)**. Independent; can land any time. Cheap.
11. **D2 / D3 / C3** — opportunistic, anywhere in the schedule.

---

## 6. Out-of-scope (explicit non-goals)

- The pending tasks in HANDOFF (#63 DN per-op profile, #66 batched-expert FFN kernel, #109 chunked SDPA swap, #118 multi-chunk loop, #134 long-context stress, #138 Mistral Small bringup, #146 attn-only prefix cache, #148 owned-GDN FOLD trick) are FEATURE work, not cleanup. They expand the codebase rather than tightening it. Track separately.
- `experiments/cb/profile/{dn,blocks}.py` profiling-flag mechanism (C3 above is a partial cleanup; the full mechanism stays — it's the only way to attribute per-block trace cost without Tracy).
- 27B-specific perf knobs (`cb_conv_mode='shiftacc'` etc.) — those are real correctness/perf trade-offs already documented in their memory entries.
- Anything that requires upstream ttnn kernel changes (paged_update_cache NKV>1 contract, fp32 KV, `[1, HEAD_DIM]` slice bug) — out of cleanup scope.

---

## Quick stats post-cleanup (projection)

- `server_35b_ttnn.py`: ~2037 LOC → ~1950 LOC (#1 + #8 + #F3).
- `server_35b_cb.py`: ~967 LOC → ~880 LOC (#7 + #2 + #F1).
- `cb_api.py`: ~372 LOC → ~340 LOC (#3 + #5 + #4).
- `cb_scheduler.py`: ~753 LOC → ~720 LOC (#5 + #6).
- `ondevice_27b.py`: ~−7 LOC (#F2).
- Net deletion: ~250 LOC, 4 env-var feature flags (`TT_DN_STATE_DTYPE`, the implicit-via-default `TT_CB_TOPK_K`, the implicit `TT_CB_SLOTS` carve-out, plus reducing the surface of `cb_dn_skip`/`cb_skip_blocks` from prod hot paths).
- 0 new env vars. 0 new options. 0 new flag knobs.
