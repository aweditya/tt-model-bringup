# CB perf regression audit — 2026-06-04

## Root cause (one sentence)

The historical CB throughput numbers (B=32 = 150.5 → 376.92 tok/s, B=64 = 593.12 tok/s)
were measured in `experiments/cb/bench/trace.py` with `sampling=False` (the **argmax
trace** P0 fast path, single 8-byte readback), while the HTTP server hardcodes
`sampling=True` and the engine layer in `experiments/serve/cb_api.py:342–346` forces
the **top-k trace path with TT_CB_TOPK_K=128** — a path whose own docstring records
"~75% slower at B=4 (step grew 131 → 232 ms in measurement)", which is a near-exact
match for the 229 / 216 ms/step we now observe.

The clobber-fix at commit `017665e` removed the `deltanet_*` overrides, which gate
the **single-stream** `forward_token_tp_inner` path (`server_tp.py:724, 774`), but it
did **not** touch the CB-only flags `state.cb_dn_recurrence_mode` and
`state.cb_conv_mode` consulted in `server_tp_cb.py:381, 454`. The CB forward still
defaults to manual DN recurrence (the historical `--owned_gdn` flag in `bench/trace.py`
is unset). The 593 tok/s number additionally required `--shiftacc` (3-column conv).

Net result: the HTTP path runs (a) a **slower per-step kernel mix** (manual DN, sampling-tail topk-128) than the bench, and (b) a **structurally different scheduler step** (per-step host readback of `[B, 128]` values + indices, per-slot host sample loop), versus the bench's pure `execute_trace` + 8-byte argmax readback.

## Evidence table

| Claim | File / commit | Lines |
|---|---|---|
| Bench used argmax-tail trace (no kwargs to `forward_batch_tp_inner`) | `experiments/cb/bench/trace.py` | 65 (`am = cb.forward_batch_tp_inner(state)`), 69 (capture handle), 85 (8-byte readback) |
| Bench called with manual+kdim by default; `--owned-gdn --shiftacc` opt-in | `experiments/cb/bench/trace.py` | 135–146 |
| HTTP engine is forced sampling=True | `experiments/serve/cb_api.py` | 341–346 (`engine = CBEngine(..., sampling=True, topk_k=topk_k, ...)`) |
| HTTP topk_k default for 27B/Gemma4 = `os.environ["TT_CB_TOPK_K"]` or `"0"` (= None for these backends per the `or None` check) — **but** the live measurements run with TT_CB_TOPK_K=128 (per the live header) | `experiments/serve/cb_api.py` | 336–337; `archive/presentation_cs440lx_2026-06-04/06_live_measurements.md:9,19` |
| Topk path at low B is the cost | `experiments/serve/cb_scheduler.py` | 173–176 ("topk adds ~100ms of fixed device cost that HURTS at low B (e.g. B=4 step grew 131→232ms in measurement)"); 641–682 (`_step_sampled_topk`) |
| Argmax fast path: in-trace `_argmax_handle`, no per-step `_step_sampled_*` host loop | `experiments/serve/cb_scheduler.py` | 240–256, 567–574 |
| `cb_dn_recurrence_mode` defaults to `'manual'` everywhere except where explicitly set | `experiments/serve/server_tp_cb.py` | 454 (`getattr(state, 'cb_dn_recurrence_mode', 'manual')`); never set in `server_tp.MeshServerState.__init__` or in `cb_api.lifespan` |
| `cb_conv_mode` defaults to `'shiftacc'` (good); kdim is opt-in | `experiments/serve/server_tp_cb.py` | 381 |
| `017665e` removed `deltanet_*` overrides (single-stream path), not CB flags | `git show 017665e` | `experiments/serve/cb_api.py` only; deleted 7 lines that set `st.deltanet_recurrence_mode = "manual"` etc. |
| `deltanet_*` flags read by `forward_token_tp_inner`, NOT `forward_batch_tp_inner` | `experiments/serve/server_tp.py` | 724, 774, 821 (all inside the single-stream decode); CB code reads only `cb_dn_recurrence_mode` |
| Historical 593.12 tok/s = B=64 + owned_gdn + 3-col shiftacc | `research/27b_cb_scope.md` | 682–693 (table; the 593.12 row is "owned_gdn + 3-col conv" benchmark only) |
| Live measurements log already calls this out as an open bug | `archive/presentation_cs440lx_2026-06-04/06_live_measurements.md` | 77 ("B=4 step time scaling 4-5× over B=1 single-seq … UNDER INVESTIGATION") |
| User-recalled "370/600 tok/s" = `B=32 376.92 / B=64 593.12` in `27b_cb_scope.md:687–688` | `research/27b_cb_scope.md` | 687–688 |
| `feedback_cb_batching_free.md`'s 150.5 / 183.5 numbers are the **older** owned_gdn+kdim (no shiftacc) sweep; the 376.92 / 593.12 are after the 3-col conv refactor | `feedback_cb_batching_free.md` (memory) | lines 32–45, 66–77 |

## Reconciliation of the three sources

1. `feedback_cb_batching_free.md` — measured **2026-05-29**, before the 3-col shiftacc refactor (DNK-G4-FINAL). Numbers: B=1=12.96, B=8=75, B=32=150.5, B=64=183.5 (with owned_gdn) → 208.30 after batched owned_gdn. **Argmax-tail trace, no per-step sampling.**
2. The 370 / 593 tok/s the user remembers — from `research/27b_cb_scope.md:687-688`, table "owned_gdn + 3-col conv": B=32=**376.92**, B=64=**593.12**. Measured later same day. **Argmax-tail trace, no per-step sampling.**
3. Presentation `04_throughput.md` reproduces the 593.1 number with the caveat "after shift-acc conv1d" — consistent.

Note: `04_throughput.md` lines 38–40 mix the *pre*-shiftacc numbers (150.5 / 183.5) with the *post*-shiftacc 593.1 in the same table without explicitly labelling — a minor doc bug, but the numbers themselves are real.

## Verification commands

Run on qb1 (with the current process or freshly via `serve_cb.sh`):

```bash
# 1) Confirm engine sampling mode + topk_k via /metrics
curl -s http://127.0.0.1:8000/metrics | grep -E 'cb_engine_sampling|cb_engine_slots_total'
# Expected: cb_engine_sampling 1.0, cb_engine_slots_total 4.
# (Argmax fast path would show cb_engine_sampling 0.0.)

# 2) Reproduce the argmax-tail bench numbers (no HTTP, direct bench)
ssh qb1 'cd ~/tt-xla && tt-smi -r 0,1,2,3 && \
  TT_METAL_HOME=$HOME/tenstorrent/tt-metal TT_BUILD_DIR=$TT_METAL_HOME/build_Release \
  ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME/ttnn \
  LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \
  .venv/bin/python -u experiments/cb/bench/trace.py --batches 4,32 --steps 50 --owned-gdn --shiftacc'
# Expected: B=4 ~80-90 ms/step, B=32 ~85 ms/step (= 376.92 tok/s).

# 3) Confirm 017665e is deployed on qb1
ssh qb1 'grep -n "deltanet_recurrence_mode\\|deltanet_decay_gate_mode" /home/aditya/tt-xla/experiments/serve/cb_api.py'
# Expected: NO matches (the 7 lines are gone).
# If matches present, the file is stale — re-deploy.

# 4) Confirm CB-mode flags on the live state
curl -s http://127.0.0.1:8000/debug/state 2>/dev/null  # if a debug endpoint exists
# OR via the engine: inspect server logs for cb_dn_recurrence_mode/cb_conv_mode.

# 5) Confirm the cost split for the topk path: cb_step_seconds vs device vs sample
curl -s http://127.0.0.1:8000/metrics | grep -E 'cb_step_seconds_sum|cb_step_seconds_count|cb_step_device_seconds_sum|cb_step_sample_seconds_sum'
# Expected (with topk_k=128 at B=4):
#   step_seconds ≈ device_seconds + sample_seconds + ε
#   device_seconds >> sample_seconds (live notes say device ≈ 99.5%)
# So the 229 ms is DEVICE time inside execute_trace+readback, NOT the host sample loop.
# That means the topk *kernel* in the trace tail (not the host post-processing) is the cost.
```

## Concrete fixes (ranked by confidence + reversibility)

### Fix 1 — drop topk_k for low-concurrency serving (HIGH confidence, trivial)

Change in `experiments/serve/cb_api.py:336` (the relevant code is `_topk_default = "64" if TT_BACKEND == "35b" else "0"`):

The default for non-35B is already `"0"` → `topk_k=None` → the **logits trace path**. The 229 ms number was measured with `TT_CB_TOPK_K=128` set by the launch wrapper. **First action: launch without `TT_CB_TOPK_K=128`.**

- 27B: drop env, fall through to logits trace (P3.5). Expected step_ms at B=4: ~131 ms (per the scheduler docstring), recovering the "65% of the right number" range.
- Even better: drop `sampling=True` entirely when no caller sets `temperature > 0`. The fastest path is `sampling=False` (the argmax handle pre-captured in the trace). This is the bench/trace.py path. But it requires either (a) routing through TWO engine instances (one greedy, one sampling) or (b) capturing both `_argmax_handle` and `_logits_handle` in the same trace. (b) is straightforward.

### Fix 2 — set the CB-mode flags explicitly in `cb_api.lifespan` (HIGH confidence)

Add right after the deleted clobber lines:

```python
if TT_BACKEND == "27b":
    st.cb_dn_recurrence_mode = "owned_gdn"  # required for 27B fast path; bit-identical at B=1
    st.cb_conv_mode = "shiftacc"            # needle-validated; already the code default
```

Without this, the HTTP path runs `manual` DN even after `017665e` — because the clobber-fix removed the wrong-attribute override but never set the right-attribute. (Defaults via `getattr(state, 'cb_dn_recurrence_mode', 'manual')` keep manual on.)

Caveats:
- `owned_gdn` hard-asserts `state_logical[0] == 1` in the kernel (per `feedback_cb_batching_free.md` line 88–91, `qwen36_gdn_decode_owned_device_operation.cpp:118`). For B>1, server_tp_cb folds batch into slots, so the kernel still sees per-slot inputs — but verify with `cb_validate_27b.py --owned-gdn` after enabling.
- `shiftacc` is not bit-identical to kdim (logit_cos 0.9995, 0.963 worst at 32 pos) but passes the needle test at L=200 and L=500. Acceptable.

### Fix 3 — capture an argmax-tail trace alongside the logits trace (MEDIUM confidence, moderate change)

Today the engine captures exactly one trace (`_trace_id`) shaped by `_decode_kw()`. Capture two: one with `return_logits=True` (for sampling requests) and one with no kwargs (argmax). At step time:

```python
if any_active_slot_wants_sampling():
    execute_trace(self._logits_trace)
    # per-slot sample loop on [B, vocab]
else:
    execute_trace(self._argmax_trace)
    # 8-byte readback only
```

This recovers the bench/trace.py step shape (~85 ms at B=32) for all greedy requests, including the HTTP path's default `temperature=0` chats.

Risk: dual-trace memory budget on Blackhole; the two-phase warmup note in `MEMORY.md` ("multi-trace = two-phase warmup", `feedback_two_phase_warmup.md`) applies here — must compile both paths first, then capture both back-to-back to avoid the 99% CPU hang documented in tenstorrent/vllm#352.

## Re-baseline plan

Once Fix 1 (drop TT_CB_TOPK_K) AND Fix 2 (set cb_dn_recurrence_mode='owned_gdn', cb_conv_mode='shiftacc') are in:

1. Restart `serve_cb.sh` with no `TT_CB_TOPK_K` env. Confirm `/metrics` shows `cb_engine_sampling 1.0` and the engine reports `topk_k=None` in startup logs.
2. Run `scripts/stress_concurrent_chat.py` (or the equivalent stress harness already in the tree — historical JSONs exist at `archive/presentation_cs440lx_2026-06-04/screenshots/stress_*.json`) at 1, 2, 4 concurrent clients. Record `cb_step_seconds_sum/count` and aggregate tok/s.
3. Compare to `experiments/cb/bench/trace.py --batches 1,4,32 --owned-gdn --shiftacc` (argmax-tail). At B=4 both should now be in the same neighborhood (≤ 1.5× delta is "sampling-mode overhead"; > 2× means another bug).
4. Update `archive/presentation_cs440lx_2026-06-04/06_live_measurements.md` with the new numbers AND a new column "engine sampling tail" so future runs are unambiguous (argmax / logits / topk-128).
5. If Fix 3 lands as well: target ≤ 1.05× of bench/trace.py at all greedy clients (the only remaining cost is per-step scheduler bookkeeping in `step()` lines 559–594, which is pure Python and trivial).
6. Update `MEMORY.md` `feedback_cb_batching_free.md` entry with a "HTTP-path caveat" note pointing at this audit, so the next session doesn't repeat the question.

## Open follow-ups (not blocking)

- `archive/presentation_cs440lx_2026-06-04/04_throughput.md:38–40` interleaves pre- and post-shiftacc CB numbers in one table. Add an explicit "shiftacc" column or split into two tables.
- The "B=4 step time scaling 4-5× over B=1 single-seq" line in `06_live_measurements.md:77` should be marked RESOLVED once Fix 1/2/3 land.
- The unintentional ambiguity in `_topk_default = "64" if TT_BACKEND == "35b" else "0"` (where `"0"` becomes `None` via `int(...) or None`) reads cleanly but is brittle; `code_cleanup_plan_2026-06-04.md` A3 already proposes folding it into a per-backend registry.
