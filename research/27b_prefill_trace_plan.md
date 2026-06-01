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

| ID | What | Status | Gate | Commit |
|---|---|---|---|---|
| T0 | Refactor `forward_prefill_chunked_tp` static-shape + extract `_traced_prefill_chunk(state, chunk_start_idx)` (no Python branches at fixed L=128) | ⏳ | code compiles | — |
| T1 | Isolation: `experiments/cb/isolate/prefill_trace.py` — capture trace at L=128, replay with L=128 prompt, compare to legacy stub | ⏳ | last-pos cos ≥ 0.99 | — |
| T2 | Padding: T1 extended to L=8, L=64 (padded to 128) | ⏳ | first-L cos ≥ 0.99 | — |
| T3 | Multi-chunk: T1 extended to L=256 (2 chunks), L=512 (4 chunks) | ⏳ | last-pos cos ≥ 0.99 | — |
| T4 | Perf bench: extend `experiments/cb/bench/ttft.py` legacy vs traced at L ∈ {32, 128, 256, 512, 1024} | ⏳ | ≥ 5× at L=200 | — |
| T5 | Integration: rewire `cb_scheduler._step_prefill_chunked` to use trace; re-run `cb_alternating_scheduler.py`, `engine_chunked_prefill.py`, `chunked_concurrent.py` | ⏳ | concurrent wedge test PASS | — |
| T6 | Production: `serve_cb.sh` with TT_CB_CHUNKED_PREFILL=1; 4-client concurrent_chat.py load probe | ⏳ | no wedge under load | — |

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
