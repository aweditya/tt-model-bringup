# Multi-model chat-serving fleet — milestone plan (2026-06-02)

Living plan. Status per milestone; commit hash on completion.

## Motivation

The CB + PC stack we built around 27B is model-agnostic at almost every layer.
The hardcoded binding is `experiments/serve/cb_api.py:192` — `import server_tp
as base`. Three reasons to lift this:

1. **Framework validation**: if prefix caching, sampling, two-phase warmup,
   `/v1/*` OpenAI endpoint, metrics, etc. all transfer to a different model
   (35B-A3B MoE, or a Llama-style dense), our stack is genuinely re-usable.
2. **Different models for different jobs**: small models for quick replies,
   big models for harder questions. Today's chat is locked to one.
3. **Research breadth**: hybrid attention+DN (Qwen3.6-27B), MoE (Qwen3.6-35B-
   A3B), pure dense (Llama, Qwen2.5) each exercise the CB layer differently.
   Catches generalization bugs early.

## Architecture (current state of the world)

```
┌──────────────────────────────────────────────────────────────┐
│  cb_api.py        — FastAPI /v1/* endpoints (model-agnostic) │
│  cb_engine.py     — thread-safe engine wrapping Scheduler    │
│  cb_scheduler.py  — Orca CB + slot-level prefix cache        │
│  cb_metrics.py    — Prometheus registry                       │
│  openai_endpoint.py — chat template renderer (Qwen3.6 today) │
│  live_slot_store.py — prefix cache index                      │
├──────────────────────────────────────────────────────────────┤
│  server_tp.py            ← 27B Qwen3.6 (prod)                │  ← cb_api.py:192 hardcodes THIS one
│  server_35b_ttnn.py      ← 35B-A3B MoE  (in-progress perf)   │
│  server_35b.py           ← 35B variant  (older / archived?)  │
│  server.py               ← legacy single-chip                 │
└──────────────────────────────────────────────────────────────┘
```

Each `server_*.py` exports the same shape:
  - `bootstrap(state, log)` — load weights, build state on mesh
  - `forward_token_tp_inner(state)` — one-token decode through the model
  - `forward_prefill_chunked_traced_inner(state)` — chunked prefill (if used)

What's model-specific: weights layout, attention/DN block structure, chat
template (different tokenizers!), KV/DN state shapes.

What's model-agnostic: the scheduler, the slot lifecycle, prefix caching,
the API, metrics, sampling, two-phase warmup pattern.

## Milestones

| ID | What | Gate | Status |
|---|---|---|---|
| MM1 | `TT_BACKEND` env selector in cb_api | `TT_BACKEND=27b` selects server_tp.py; `=35b` selects server_35b_ttnn.py | ⏳ |
| MM2 | 35B-A3B end-to-end via CB | Bootstrap, 2-turn chat, prefix cache hit observed | ⏳ |
| MM3 | Long-context concurrent stress test | `experiments/cb/load/concurrent_chat.py` adapted: long prompts (L=1000+), multi-tab; cache hit rate measured at scale | ⏳ |
| MM4 | Chat template integration for non-Qwen3.6 models | `_messages_to_prompt` parameterized by tokenizer's quirks; new model picks up its own template kwargs | ⏳ |
| MM5 | Bringup of an additional model | Candidate: Llama 70B (matches TP=(1,4)) or Qwen2.5-32B (already in `models/`). Wired as `server_*.py`; smoke test passes | ⏳ |
| MM6 | (stretch) Multi-model concurrency | Two server processes on the same host (1 chip each) OR cross-host (qb1 model A, qb2 model B) | ⏳ |

## MM1 — `TT_BACKEND` env selector

**Scope**: one-line config switch. Make the model selection a runtime decision.

**Implementation**:
```python
# experiments/serve/cb_api.py:192 area
backend = os.environ.get("TT_BACKEND", "27b")
if backend == "27b":
    import server_tp as base
elif backend == "35b":
    import server_35b_ttnn as base
else:
    raise ValueError(f"unknown TT_BACKEND={backend}")
```

Plus matching env var in `serve_cb.sh` for plumbing.

**Gate**: `TT_BACKEND=27b bash serve_cb.sh start` still serves 27B as today;
`TT_BACKEND=35b bash serve_cb.sh start` boots the 35B server and `/health`
responds.

**Risks**:
- 35B's bootstrap may have different env requirements (different model_id,
  different tt-metal kernel needs).
- 35B has owned MoE kernels; verify they're loaded.
- `DEFAULT_MODEL_ID` in cb_api should adapt to the chosen backend (different
  model IDs for `/v1/models` advertisement).

## MM2 — 35B-A3B end-to-end via CB

**Scope**: confirm the CB/PC stack works for the MoE hybrid model.

**Pre-requisites**: MM1 done; 35B kernel build present on the host.

**Gate**: equivalent of our 27B PC-P5 smoke test but driving the 35B model.
Turn 1 latency baseline, turn 2 cache hit, server stays up. Documented in
`research/27b_prefix_caching_plan.md`'s style → `research/35b_chat_server.md`.

**Risks**:
- **35B chat template is different from 27B's** — Qwen3.6-35B has its own
  Jinja with potentially different `<think>` injection / `preserve_thinking`
  behavior. The invariant test (`chat_template_invariant.py`) needs to be
  re-run with `TT_MODEL_ID=Qwen/Qwen3.6-35B-A3B`.
- The MoE forward is more complex; trace memory may differ. Two-phase warmup
  applies same way.
- DN state in 35B is a different shape; the slot lifecycle in `cb_reset_slots`
  is correct in principle but needs verification.

## MM3 — Long-context concurrent stress test

**Scope**: validate prefix caching at realistic chat scale (L=1000-2000+
prompt tokens, 4+ concurrent tabs) on the *current* 27B server.

**Pre-requisites**: none — pure load test on what we have.

**Implementation**: extend `experiments/cb/load/concurrent_chat.py` (or
fork to `concurrent_chat_long.py`) with:
- System prompts containing 1000-2000 tokens of context (e.g., RAG-style)
- 3-turn conversations per client
- Assertion: turn-2+ TTFT < turn-1 TTFT × 0.3 for each client (cache hit)
- Metrics scrape post-run: `cb_prefix_cache_hits_total ≥ N_clients × 2`,
  `cb_prefix_cache_evictions_total == 0` (if N_clients ≤ N_SLOTS)

**Gate**: 4 clients × 3 turns × L=1500 prompt → cache hit rate ≥ 75%, no
server wedge, all responses coherent.

**Risks**:
- L=1500 prompts exceed our current chunk_size=32 + chunked_prefill_disabled
  cold-start latency budget (1500 × 80ms = 2 minutes). Mitigates if (1)
  chunked_prefill wedge is fixed first OR (2) we accept the slow cold start
  and only measure the warm path.
- LRU eviction edge cases — verify slot reuse doesn't race with
  in-flight requests.

## MM4 — Chat template per-model parameterization

**Scope**: `_messages_to_prompt` currently has Qwen3.6 specifics baked in
(`preserve_thinking=True`, trailing `<think>\n\n</think>\n\n` strip). Other
models won't have these.

**Implementation**: add a per-model "chat_template_config" object loaded
alongside the tokenizer. Each backend's bootstrap registers its template
kwargs and any post-processing hooks. Default = upstream behavior (no
patches).

**Gate**: `chat_template_invariant.py` passes with `TT_MODEL_ID` pointed at:
- Qwen/Qwen3.6-27B (current)
- Qwen/Qwen3.6-35B-A3B
- Llama-3.1-70B or similar

For each model, the invariant holds across the must-pass cases.

## MM5 — Bringup of an additional model family

**Scope**: take an existing bring-up script in `models/` (e.g., Llama 3.1
70B) and wire it as a CB server.

**Recommended candidate**: Llama 3.1 70B — dense attention only, no DN
quirks, similar size to 27B for a TP fit. The cleanest "does our framework
generalize" test.

**Implementation**:
1. Create `experiments/serve/server_llama3_70b.py`
2. Lift bootstrap + forward from the existing bring-up
3. Verify `cb_reset_slots`, KV cache layout, paged SDPA all map
4. Plug into TT_BACKEND selector
5. Run the full PC test suite

**Gate**: `TT_BACKEND=llama70b bash serve_cb.sh start` → 2-turn chat works,
prefix cache hits.

**Risks**: trace capture, kernel needs, memory layout could all differ
non-trivially. This is the biggest milestone, save for last.

## MM6 — Multi-model concurrency (stretch)

**Scope**: serve TWO different models simultaneously.

**Approach options**:
- **Cross-host**: qb1 hosts 35B, qb2 hosts 27B. Two FastAPI processes.
  Client picks via URL. Trivial; just choose what to serve where.
- **Same-host, single-chip-per-model**: harder. Would require breaking
  TP=(1,4) and running each model on a (1,1) mesh. Significant scheduler
  rework.

**Recommendation**: cross-host first. Free if MM5 succeeds.

## Test plan / regression gates

For every new model added:

1. `chat_template_invariant.py` with that model's tokenizer (the 7-case
   suite already in place).
2. `prefix_cache_smoke.py` adapted to drive the new server.
3. The full lifecycle test suite (`prefix_cache_store.py`, `prefix_cache_
   lifecycle.py`) is model-agnostic and runs as-is.
4. Concurrent load test with the new model.

## Open questions

1. **Multi-model in one server process?** Today the engine owns ONE mesh +
   ONE set of state. Lifting to N models in one process means N×mesh —
   N=2 fits on qb1 only if each model uses (1,2) instead of (1,4). Big
   surgery on bootstrap.
2. **Per-request model selection in `/v1/chat/completions`?** OpenAI API
   already has `body["model"]`. We could route requests to the right
   backend engine if multiple are loaded. Future work.
3. **Smaller models on single chips?** Qwen3 0.6B, Llama 1B-3B fit on a
   single chip. Could run 4 different small models on qb1's 4 chips
   simultaneously. Different bringup path — single-chip CB.

## Sequence

1. **First**: finish 27B story — chunked_prefill+prefix_cache wedge fix
   (orthogonal but blocks long-context test).
2. **Then**: MM3 (long-context stress test) on 27B.
3. **Then**: MM1+MM2 (35B via CB) — biggest leverage step.
4. **Then**: MM5 (Llama 70B or similar) — proves generalization.
5. **Stretch**: MM6 (cross-host fleet) — once we have 2 servable models.

This is several weeks of work. Each step ships independent value.
