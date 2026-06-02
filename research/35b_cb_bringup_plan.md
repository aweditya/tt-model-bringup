# Qwen3.6-35B-A3B → CB chat server — milestone plan (2026-06-02)

Living plan. CB35-0..CB35-7 mirror the original 27B CB story (CB0..CB5)
plus MoE-specific additions. **Heavy reuse of the 27B CB stack — we ONLY
write 35B-specific bits.**

## Goal

Bring 35B-A3B online as a CB chat backend behind the existing
`/v1/*` endpoint via `TT_BACKEND=35b` (MM1 already shipped). Users get:
- OpenAI-compatible chat (same as 27B)
- Slot-level prefix caching (same as 27B)
- Sampling, metrics, two-phase warmup, all 27B's hard-won goodies
- Inherits future improvements to the shared CB stack for free

Non-goal for v1: prefix-cache-style chat speedups on 35B's TTFT. That
comes free once CB+PC is wired; no separate work.

## Why this is bigger than MM1 promised

`server_35b_ttnn.py` is single-stream (B=1). Audit (commit `MM1`-era):

| File | Class / API | 27B has | 35B has |
|---|---|---|---|
| `server_tp.py` | `MeshServerState` | ✓ | — |
| `server_35b_ttnn.py` | `State` | — | ✓ |
| `server_tp_cb.py` | `setup_cb_state(state, B, blocks_per_seq)` | ✓ | — |
| `server_tp_cb.py` | `cb_reset_states / cb_reset_slots` | ✓ | — |
| `server_tp_cb.py` | `cb_prefill_transplant` | ✓ | — |
| `server_tp_cb.py` | `update_input_buffers_batched(state, toks, curs)` | ✓ | — |
| `server_tp_cb.py` | `forward_batch_tp_inner(state, ...)` | ✓ | — |
| `server_35b_ttnn.py` | `update_input_buffers(state, tok, pos)` (B=1) | — | ✓ |
| `server_35b_ttnn.py` | `step_forward_inner(state)` (B=1) | — | ✓ |

So: 35B is a fully-working B=1 chat brain; we need to wrap it in the
CB B=N harness analogous to what `server_tp_cb.py` does for 27B.

## What's reused vs new

### Reused as-is (model-agnostic, zero changes)
- `experiments/serve/cb_engine.py` — threaded engine wrapper
- `experiments/serve/cb_scheduler.py` — Orca scheduler + slot-level prefix cache
- `experiments/serve/cb_metrics.py` — Prometheus registry
- `experiments/serve/cb_api.py` — `/v1/*` endpoints (post-MM1 backend selector)
- `experiments/serve/live_slot_store.py` — prefix cache index
- `experiments/serve/openai_endpoint.py` — chat-template renderer
  (Qwen3.6-35B shares the Qwen3.6 chat template — verify in CB35-0)
- All test infra: `prefix_cache_store.py`, `prefix_cache_lifecycle.py`,
  `chat_template_invariant.py`, `prefix_cache_smoke.py`

### New (35B-specific)
- `experiments/serve/server_35b_ttnn_cb.py` — analogue of
  `server_tp_cb.py` but for 35B. Per-slot state setup, cb_reset, batched
  forward, MoE-aware DN/attn/MoE batched step functions.
- Minor refactor of `server_35b_ttnn.py` to expose state as a class that
  `cb_api.py` can construct (already has `class State` ✓; may need
  `MeshServerState` alias).

## Milestones

| ID | What | Gate | Status |
|---|---|---|---|
| CB35-0 | Audit + design | This doc + tt-metal MoE pattern recon | 🟡 in progress |
| CB35-1 | Batched state class + minimal batched forward (B=N, MoE per-slot loop) | B=4 forward, slot 0 = real request, others idle; slot 0 output matches B=1 reference within bf16 noise | ⏳ |
| CB35-2 | Per-slot ragged state (cur_pos, KV, DN per slot) | 2 slots with different cur_pos; slot isolation holds (slot 0 gen != slot 1 gen) | ⏳ |
| CB35-3 | Plug into existing `cb_scheduler.Scheduler` (no changes) | 5 reqs through 4 slots, all generations bit-identical to standalone B=1 refs | ⏳ |
| CB35-4 | Two-phase warmup + trace capture for batched forward | Traced forward bit-correct vs eager; no wedge | ⏳ |
| CB35-5 | MoE inside the traced forward (B=N) | Routing top-k per-slot works in trace; bit-correct vs eager | ⏳ |
| CB35-6 | Prefix caching on 35B (mostly free) | `chat_template_invariant.py` 7/7 with 35B tokenizer; smoke shows turn-2 cache hit | ⏳ |
| CB35-7 | Production wire-up via `TT_BACKEND=35b` | Real chat through `chat.py` on TT_BACKEND=35b; PC hits; no wedge | ⏳ |

## Per-milestone detail

### CB35-0 — Audit + design (this doc)

Tasks:
- Audit `server_35b_ttnn.py` vs `server_tp.py` (done above).
- Confirm Qwen3.6-35B uses the same Jinja chat template as 27B (high
  probability; verify via `chat_template_invariant.py` with
  `TT_MODEL_ID=Qwen/Qwen3.6-35B-A3B`).
- Confirm DN block shapes (27B vs 35B): if identical, `cb_dn_step` from
  `server_tp_cb.py` may be a starting point — port + adapt.
- Reference research: `research/tt_metal_moe_cb_patterns.md` (research
  agent in flight) for tt-metal MoE + CB prior art (Llama-70B-Galaxy
  patterns, vLLM MoE batched routing, Mixtral/DeepSeek demos if any).

Gate: this plan ready for execution.

### CB35-1 — Batched state + minimal batched forward

Create `experiments/serve/server_35b_ttnn_cb.py` mirroring
`server_tp_cb.py`:

```python
def setup_cb_state(state, B, blocks_per_seq=None):
    """Allocate per-slot state on top of single-stream 35B state.
       - cb_B = B
       - cb_kv[li]   per-slot KV blocks (paged_update_cache layout)
       - cb_dn[li]   per-slot DN ssm + conv_cols (same as 27B)
       - cb_cur_pos_buf [B]
       - cb_rot_idxs_buf [B,1]
       MoE has no per-slot weights (experts are shared); intermediate
       activations are per-token, allocated transiently inside forward."""
```

```python
def update_input_buffers_batched(state, token_ids, cur_positions):
    # Same pattern as 27B's: host→device copy into pre-allocated buffers.
```

```python
def forward_batch_tp_inner(state, return_logits=False, return_topk=None):
    # Batched embed + RoPE.
    # For each layer:
    #   if attention: paged SDPA (batched, like 27B)
    #   if DN: cb_dn_step (batched, like 27B's masked recurrence)
    #   MoE: top-k router across [B, HIDDEN], then loop OR scatter-gather
    #        (v1: Python loop over B — slow but correct)
    # Final norm + LM head + per-slot argmax/topk/logits.
```

**Key MoE design decision (CB35-1 v1)**: handle batched routing by
**iterating over B in Python inside the MoE forward**. Each iter does
the existing single-stream MoE forward for that slot's hidden state.
Slow (B× expert dispatch overhead) but correct. v2/v3 optimize.

Reference: 27B's `cb_dn_step` in `server_tp_cb.py:333` uses a masked
recurrence to update per-slot DN state. The mask pattern (multiply by 0
for slots being reset, 1 for slots being preserved) is the cleanest way
to do "per-slot stateful ops" in a single batched forward without
branching.

Gate: B=4 forward, slot 0 has a real request, slots 1-3 idle (DUMMY_TOK
at cur_pos=0). Slot 0's output (last layer activations, or first
generated token) matches standalone B=1 35B reference within bf16 noise
(cos ≥ 0.999).

### CB35-2 — Per-slot ragged state

What changes from CB35-1:
- Per-slot cur_pos in `cb_cur_pos_buf` (different positions per slot —
  each request is at its own point in the conversation).
- Per-slot KV reads/writes via paged tables (already correct in 27B —
  port verbatim).
- Per-slot DN reset on admit (mirror `cb_reset_slots` masked-mul).

Gate: 2 slots run different prompts in parallel; slot 0's gen tokens are
identical to standalone B=1 reference for prompt 0, slot 1's gen tokens
are identical to standalone B=1 reference for prompt 1. (Slot isolation
test — same as CB2 was for 27B.)

### CB35-3 — Plug into `cb_scheduler.Scheduler`

`cb_scheduler.py` already calls into `base.forward_batch_tp_inner` etc.
via the engine. Once CB35-1+2 are done, the existing scheduler should
just work with 35B.

Gate: `cb_engine.CBEngine(state=35b_state, ...)` runs 5 requests through
4 slots, sampling=True. Each rid's generation matches the standalone
greedy-from-B=1 reference. Same gate as CB3 for 27B.

### CB35-4 — Two-phase warmup + traced forward

Two-phase warmup is already proven (`feedback_two_phase_warmup`,
`research/27b_prefill_trace_plan.md`). Just:
- Phase 1: warmup the 35B batched forward without trace
- Phase 2: capture as a trace
- Replay via `ttnn.execute_trace` per step

Risk: 35B trace may need a bigger `trace_region_size`. Two-phase warmup
itself doesn't change memory budget, but the batched MoE inside the trace
may pin more intermediates. Test with current 50-800MB region first; bump
if OOM.

Gate: traced forward at B=N bit-correct vs eager forward (cos ≥ 0.9999
on slot outputs). No wedge over 50+ traced step replays.

### CB35-5 — MoE inside the trace

The Python-B-loop MoE from CB35-1 won't directly trace (data-dependent
dispatch inside Python). Two options:
- **Option A**: trace the per-slot MoE forward N times (one trace per
  active slot count). Memory-prohibitive at scale.
- **Option B**: refactor MoE to be batched-tensor-only. For each slot
  s, compute `top_k_experts[s]` on-device, then for each expert e,
  compute `mask[B] = (any slot routed to e)`, run expert on
  masked input, scatter results. This is roughly what Pattern A does
  already at B=1.
- **Option C**: precompute the router decision OUTSIDE the trace (host),
  then inside the trace use deterministic expert dispatch.

Reference: the existing P3 task ("MoE trace refactor — top-k
data-dependent dispatch") was about THIS exact problem for B=1. We
already solved data-dependent dispatch for B=1; the question is whether
that solution generalizes to B=N.

Gate: traced batched forward with MoE inside == eager batched forward
(cos ≥ 0.999 on logits, sampled-token equality after argmax).

### CB35-6 — Prefix caching on 35B

This one is mostly free given CB35-1..5 are done. The cb_scheduler
already has the live-slot cache; we just need the chat-template
roundtrip to work for 35B.

Tasks:
- Run `experiments/cb/isolate/chat_template_invariant.py` with
  `TT_MODEL_ID=Qwen/Qwen3.6-35B-A3B`. Expect 5/5 must-pass cases (same
  template family as 27B). If anything diverges, MM4 (per-model template
  config) is blocking.
- Verify `_messages_to_prompt`'s `preserve_thinking=True + trailing
  strip` works correctly for 35B's tokenizer.
- Run `experiments/cb/validate/prefix_cache_smoke.py` against the live
  35B server: turn-2 cache hit.

Gate: smoke test on 35B: turn-2 cache hit ≥ 1, turn-2 latency <<
turn-1 latency, no wedge.

### CB35-7 — Production wire-up

MM1 already added `TT_BACKEND=35b` selection. Just need:
- Restart `serve_cb.sh` with `TT_BACKEND=35b TT_CB_PREFIX_CACHE=1`
- Real chat smoke through `chat.py`
- Verify `/v1/models` advertises Qwen3.6-35B-A3B
- 4-tab concurrent chat smoke (multi-slot)

Gate: real chat works, PC hits visible in /metrics, no regressions.

## Risk catalog

1. **DN shape mismatch 27B vs 35B**: 27B's `cb_dn_step` may not directly
   port. CB35-0 audit will check. If shapes differ, CB35-1 grows the
   batched-DN-step work.
2. **MoE in trace at B=N**: hardest unknown. Option B (batched-tensor
   dispatch) is the right answer but may need significant work. Falling
   back to eager (no trace) keeps CB working at slow tok/s.
3. **Memory budget for batched B + MoE intermediates + traces**: 35B is
   bigger than 27B. (1,4) P150 has 31.81 GB/chip. Batching B=4 ×
   MoE intermediates may push tight. Trace_region_size may need tuning.
4. **Chat template asymmetry beyond what we know**: 35B is same family
   as 27B so likely fine, but the invariant test will catch.

## Reuse summary (the cheaper case)

Total new code estimate: ~600-1200 LOC for `server_35b_ttnn_cb.py`
(mirror of `server_tp_cb.py` which is ~700 LOC). ~10-20 LOC for any
state class alias/cleanup in `server_35b_ttnn.py`. Zero changes to
cb_engine, cb_scheduler, cb_metrics, cb_api, openai_endpoint,
live_slot_store, all test files.

All of CB35's correctness gates piggyback on the existing test infra
(`prefix_cache_store.py`, `prefix_cache_lifecycle.py`,
`chat_template_invariant.py`, `prefix_cache_smoke.py`) — they're
backend-agnostic.

## Sequence

CB35-0 (this doc, today) → CB35-1 → CB35-2 → CB35-3 → CB35-4 → CB35-5
→ CB35-6 → CB35-7.

Each milestone ships independent value. CB35-1+2+3 alone = working CB
on 35B (slow, no trace). CB35-4 adds the trace speedup. CB35-5 makes
MoE+trace work properly. CB35-6+7 = chat with 35B and PC.

## Reference points
- `experiments/serve/server_tp_cb.py` — the canonical 27B template to
  port.
- `research/27b_cb_scope.md` — original 27B CB design doc.
- `research/27b_prefix_caching_plan.md` — PC pattern (re-used as-is).
- `feedback_two_phase_warmup` — multi-trace coexistence rule.
- `research/tt_metal_moe_cb_patterns.md` (research agent in flight) —
  external prior art on MoE+CB.
