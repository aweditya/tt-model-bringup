# Trace S1a chunked prefill — milestone plan (2026-05-31)

Living plan. Status per milestone; commit hash on completion.

## Root cause being fixed

Eager `forward_prefill_chunked_tp` allocates tensors after the decode trace is captured at bootstrap. ttnn warns about this at startup (`"Allocating device buffers is unsafe due to existence of an active trace"`); under sustained concurrent traffic the chunked-prefill temporary allocations land at addresses the decode trace's reserved scratch reads/writes — trace replay reads garbage, engine wedges. Legacy 1-tok/iter prefill avoids this because it just replays the decode trace L times (zero new allocations).

Fix: trace S1a too. Both forwards become trace replays; no allocations happen after bootstrap = no collisions.

## Design

- **One fixed chunk_size = 128** (Llama's middle sweet spot). One trace, looped for L > 128.
- For chunk c (c = 0, 1, ...): Q at prompt positions `[c*128, c*128+128)`, `chunk_start_idx = c*128`. Paged KV accumulates across chunks via `paged_fill_cache`. The trace replays with `chunk_start_idx` updated via a host-side buffer write between calls.
- For L < 128: pad to 128, run one trace, slice output to L.
- For L > 128: loop `ceil(L/128)` times.

| Real L | Trace replays | Padded compute | Waste |
|---|---|---|---|
| 20  | 1 | 128 | 6.4× |
| 100 | 1 | 128 | 1.3× |
| 200 | 2 | 256 | 1.28× |
| 500 | 4 | 512 | 1.02× |
| 1000 | 8 | 1024 | 1.02× |

## Milestones

| ID | What | Status | Gate result | Commit |
|---|---|---|---|---|
| T0 | Padded fixed-L=128 forward matches legacy stub argmax | ✅ DONE | argmax match (token 279 ' the') | `fa908fb` |
| T1 | Capture trace + replay at L=128, bit-equivalent to eager | ✅ DONE | cos=1.000000 vs eager, **3.4× speedup** | `b8d67c8` |
| T2 | Same trace replayed across L ∈ {5,8,9,11,15} all padded to 128 | ✅ DONE | all replays cos=1.000000 vs same-input eager | `523b141` |
| T3 | Multi-chunk for L > 128 | 🟡 DEFERRED | needs per-chunk page_table + DN-state persistence | — |
| T4 | TTFT bench legacy 1-tok/iter vs traced | ✅ DONE | crossover ~L=32; **9.76× at L=128** | `9ee90e4` |
| T5 | Scheduler integration | ✅ **DONE** | 4/4 turns / 0 errors under 2-client concurrent gate (after two-phase warmup) | `98450ec` |
| T6 | Production wire-up + real concurrent load | ✅ **DONE** | server up with `TT_CB_CHUNKED_PREFILL=1`; user validated multi-window chat | live on qb1 |

### T5 blocker detail (5 attempts, all hung — pattern is two-trace coexistence)

Symptom: scheduler init runs prefill JIT warmup + capture (succeeds — dbg
output visible), then decode JIT warmup allocates → ttnn warns `"Allocating
device buffers is unsafe due to the existence of an active trace"`, then
python burns 99% CPU forever with no log progress.

Attempts:

| Ver | trace_region | chunk_size | sync after end-capture | Outcome |
|---|---|---|---|---|
| T5a | default (~50 MB) | 128 | no | Hung at warning |
| T5b | 200 MB | 128 | no | Crashed: `ARC core failed to start` (mesh hardware) |
| T5c | 200 MB | 128 | no | Explicit OOM: trace buffer needs 1.76 GB |
| T5d | 2.5 GB | 128 | no | OOM during model load (635 MB/bank tensor doesn't fit) |
| T5e | 1.5 GB | 64 | no | Same model-load OOM |
| T5f | 800 MB | 32 | **yes** (Llama pattern) | Hung at warning again, same as T5a |

Conclusions:
- `trace_region_size` AND chunk_size AND Llama's sync pattern all applied;
  hang persists. So at least one of these is true:
  - Two-trace coexistence isn't supported in our ttnn build (despite docs)
  - There's another Llama-side setup we're missing (SubDeviceManager? L1 buffer
    type? device init flags?)
  - The hang isn't allocator-related; could be in trace replay vs eager
    interaction during decode warmup

Production stays on legacy 1-tok/iter prefill (chunked_prefill=False default).
T0–T4 stand as proven isolation tests — primitives work; integration needs
a dedicated session with deeper ttnn debugging (py-spy on the hung process,
ttnn allocator stats, look at how Llama 70B Galaxy handles it).

Until resolved, production runs on legacy 1-tok/iter prefill (chunked path
gated behind `chunked_prefill=False` default; uncommitted state-buffer
allocations in `server_tp.py` are harmless — only triggered by `chunked_prefill=True`).

### Standalone trace primitives are PROVEN

T0-T4 are gated, validated, and committed. The bug is integration only.
Once two-trace coexistence is solved, `_step_prefill_chunked` in
`cb_scheduler.py` (commit `530d259`) flips on and we ship.

### ROOT CAUSE FOUND (post-mortem, 2026-06-01) — two-phase warmup

[tenstorrent/vllm#352](https://github.com/tenstorrent/vllm/issues/352)
documents the exact symptom and fix. Capturing prefill trace first then
running decode JIT warmup compiles ops that prefill didn't touch; those
compilations allocate kernel-cache buffers that can land on prefill
trace's reserved memory → trace replay reads garbage → our 99% CPU hang.

**Fix is order, not size**. TT vLLM's default trace_region_size is just
50 MB and works fine for both traces — once warmup order is right.

Two-phase warmup pattern (TT vLLM `model_runner.py:2538-2592`):

```
phase 1 (compile only, no captures):
  warmup_prefill(enable_trace=False)
  warmup_decode (enable_trace=False)
  ttnn.synchronize_device(mesh)
phase 2 (capture, no JIT — cache already populated):
  warmup_prefill(enable_trace=True)
  warmup_decode (enable_trace=True)
```

Code change required in `experiments/serve/cb_scheduler.py`:
- Split `_capture_prefill_trace` and `_capture_trace` into separate
  `_warmup_X(enable_trace=False)` and `_capture_X` methods.
- In `__init__`, call both warmups first, then both captures.

Diagnostic env var: `TT_METAL_TRACE_ALLOC_TRACKING=1` (tt-metal commit
5043de3df5) makes the warning into an UnsafeAllocationTracker that names
the offending op.

Saved in memory: `feedback_two_phase_warmup`. T5 unblocked for next session.

## Production state (2026-06-01, post-T6)

Server on qb1 runs **traced chunked prefill** (chunk_size=32, trace_region=800 MB,
TT_CB_CHUNKED_PREFILL=1). For:
- L ≤ 32 → traced replay (~1s) — turn-1 messages.
- L > 32 → legacy 1-tok/iter fallback (~80 ms/tok) — turn-2+ with history.

User-validated: 4 concurrent chat windows; turn-1 fast, turn-2 stalls briefly during
re-prefill of full history (expected — TT vLLM alternating PREFILL_ONLY / DECODE_ONLY
scheduler design). Once prefill finishes, all slots resume.

## Next levers for chat speed (deferred, in priority order)

1. **T3 multi-chunk** (~2 hrs): loop the same trace at advancing `chunk_start_idx`
   so L > chunk_size doesn't fall to legacy. Needs per-chunk page_table for
   `paged_fill_cache` and DN-state persistence across chunks.
2. **Bigger chunk_size** (chunk_size=64 or 128): single-chunk traced covers
   more turns. Needs re-checking memory budget; T5d hit model-load OOM with
   trace_region=2.5 GB. Now with two-phase warmup, retry budgeting.
3. **Prefix caching** (architectural): turn-N prefill = new tokens only, not
   re-prefill entire history. The real chat win. Needs per-client session +
   slot-state-persistence-across-requests + prefix matching.

## Risks

1. **Chunked-SDPA precision at q_chunk_size=128**: S2.2 attempts at q_chunk_size=32 drifted to cos=0.43. Two reasons q_chunk_size=128 likely fixes it: Llama uses 64-256 in prod, and our S2.1 isolation at q_chunk_size=32 passed vs numpy (drift was integration-level). T1 catches this.
2. **L-dependent branches in `forward_prefill_chunked_tp`**: `if L <= C` for Neumann DN path. T0 must extract a branch-free `_traced_prefill_chunk` at fixed L=chunk_size.
3. **Pre-allocated input buffers**: need `prefill_tok_buf`, `prefill_pos_buf`, `chunk_start_idx_buf` sized for chunk_size=128, allocated at bootstrap, mutated host-side between replays.
4. **Tokenizer stays outside trace** — host-side, before replay loop.

## Reference points

- Llama prefill recipe: `models/tt_transformers/tt/attention.py:1083-1091` (chunked SDPA call signature).
- Llama generator chunk loop: `models/tt_transformers/tt/generator.py:1084-1137` (per-chunk forward calls with advancing `chunk_start_idx`).
- Our existing decode trace pattern: `cb_scheduler.Scheduler._capture_trace`.

## Out of scope

- Prefix caching (skip re-prefill of prior turn's history). Future work; this plan only makes single-request prefill fast.
- S2.2 swap of attention SDPA (chunked SDPA inside S1a). T0 will need to make a call: either keep S1a's `is_causal=True` non-paged SDPA (safe; the wedge fix is at the trace level, not the SDPA op), or also swap to chunked SDPA. Default: keep `is_causal=True` for T1 simplicity; revisit in T4 if perf demands it.
