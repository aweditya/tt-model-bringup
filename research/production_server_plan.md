# Production CB inference server — plan of action

**Goal:** a production-grade, OpenAI-compatible inference server where many
concurrent users chat with Qwen3.6-27B, served via continuous batching, with
correctness under load, robustness, observability, and a clean lifecycle. Then
resume prefill optimization (S2).

**Definition of done (SLOs to validate, not assert):**
- N concurrent chat clients (target N≥32) sustained with **no crashes, no
  cross-request leakage, no fabric wedge**.
- Aggregate throughput tracks the CB curve (B=64 ≈ 593 tok/s measured; ~150 @ B=32).
- TTFT bounded + reported (v1 uses the current 1-tok/iter prefill; S2 improves it).
- Graceful shutdown (drain + clean mesh close); restart recovers.
- `/health` + `/metrics`; structured logs; OpenAI `/v1/chat/completions` parity
  (streaming, sampling, stop, usage, cancellation).

## Current assets (grounded — reuse, don't reinvent)
- **CB engine core**: `cb_scheduler.Scheduler` (Orca submit/step/run, B slots,
  trace) over `server_tp_cb` (batched forward, per-slot paged KV + DN state,
  ragged CB2, trace CB4). Validated: bit-identical to prod B=1, per-slot
  isolation, B=64 = 593 tok/s.
- **Concurrency prototype**: `experiments/cb/serving_demo.py` — device-owning
  engine thread (scheduler + select loop) + concurrent socket clients. This is
  the architecture seed for P0.
- **OpenAI HTTP**: `serve/openai_endpoint.py` (chat template + SSE + sampling
  params) — currently routes to single-seq `generate_tp`; **re-point at the CB
  engine** in P2.
- **Sampling**: `_sample_from_logits` (host temp/top-p/top-k) — make **per-request
  in CB** (P1).
- **Single-seq prod** `server_tp.py` @ 12.93 tok/s — FROZEN, stays as
  reference/fallback. The production CB server is a NEW path (no regression).

## Target architecture
- **Engine** (one thread, owns the mesh): continuous scheduler loop. Thread-safe
  inbound queue (admission + backpressure); per-request outbound token stream;
  per-request sampling; eviction (EOS / max_tokens / client-cancel + slot reset).
  Metrics. Device ops single-threaded here only.
- **API** (FastAPI + uvicorn, async): OpenAI `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`, `/metrics`. Chat template, SSE
  stream, sampling params, stop sequences, usage; cancel on client disconnect.
- **Bridge**: async API ↔ sync engine via thread-safe queues + a per-request
  asyncio stream. (Engine = thread; uvicorn = asyncio; never touch the device
  from the async side.)
- **Lifecycle**: `serve_cb.sh` (start/stop/status/restart); readiness after
  bootstrap; SIGTERM → stop admitting → drain in-flight → `close_mesh_device`
  cleanly (hard-kill wedges fabric — see serve_tp.sh).

## Hard problems + approach (the parts that aren't glue)
1. **Per-request sampling in CB.** `Scheduler.step()` returns per-slot argmax
   (greedy, fast traced). Production needs per-slot temp/top-p/top-k/seed → the
   batched forward must expose **per-slot logits or on-device top-k** so the
   engine samples per slot. Cost: B×vocab readback (mitigate with on-device top-k
   = B×k). Greedy stays the argmax fast path. ISOLATE + validate per-slot sampling
   (correctness + cost) before integrating. [[validate-against-ground-truth-not-a-weaker-tt-path]]
2. **TTFT / prefill.** Scheduler prefills 1-tok/iter (slow for long prompts).
   **v1 ships with this** (document TTFT); the real fix is **S2** (chunked prefill
   into a CB slot — the deferred major effort: multi-query paged SDPA + shiftacc
   conv). Sequencing per the user: production server first, prefill opt after.
3. **Concurrency correctness.** Per-slot isolation is CB2-validated; re-validate
   under sustained load (no leakage). Cancellation must free the slot + reset its
   DN state (`cb_reset_slots`); KV self-overwrites (cur_pos-bounded).
4. **Graceful shutdown / device hygiene.** SIGTERM handler: drain, release trace,
   `ttnn.close_mesh_device`. A wedged mesh needs `tt-smi -r 0,1,2,3` (make reset).
5. **Robustness/error isolation.** Bad request → 4xx (don't crash engine); a
   per-request device error → fail that request, keep serving others; max
   context/tokens caps; queue-full → 429 (or bounded wait).
6. **Observability.** `/metrics`: aggregate tok/s, queue depth, slot utilization,
   TTFT + decode latency p50/p99, request/error counts. Structured logs.

## Staged plan (each stage = a validated increment, committed)
- **P0 — Engine module. DONE (2026-05-29).** `serve/cb_engine.py`: `CBEngine`
  (device-owning thread runs the scheduler loop: drain inbound → drain cancels →
  step → stream) + `RequestHandle.tokens()`; thread-safe `submit(prompt,max_new)`
  / `cancel(rid)` / `start()` / `stop()`; greedy. Added `Scheduler.cancel(rid)`
  (frees slot; next `_admit` resets DN state — same path as eviction, no device
  op). **Gate PASS** (`experiments/cb/validate/engine.py`, qb1 traced, 4 slots):
  6 concurrent clients → each stream == its B=1 greedy ref; cancel mid-flight →
  `cancelled` + freed slot recycles correctly; per-request max_new exact; clean
  start/stop, no wedge.
- **P1 — Per-request sampling in CB. DONE (2026-05-29).** Engine grew a
  construction-time `sampling=False|True` mode (de-risked: no mixing eager
  forwards with execute_trace; no unproven two-trace gamble). `sampling=False`
  keeps the P0 argmax-trace fast path byte-identical. `sampling=True` runs the
  eager logits forward (`forward_batch_tp_inner(return_logits=True)` already
  existed → `[B,vocab]` replicated) each step; greedy slots take host argmax,
  sampled slots use `_sample_from_logits` with their own seeded rng (reused
  from `server_tp`). Wired through `Scheduler.submit(prompt, sampling=…)` +
  `CBEngine.submit(…, sampling=…)`; greedy-via-host-argmax is exact. **Gate
  PASS** (`experiments/cb/validate/engine_sampling.py`, qb1, 4 slots): (A)
  greedy-via-sampling-engine == device-argmax ref exactly; (B) mixed batch:
  greedy slots unaffected, sampled seeds differ + coherent; (C) determinism
  (same seed → identical). **Measured cost**: ~318 ms/eager-step at B≤4 (vs
  ~106 ms traced in P0; ~3× per-step penalty, mostly Python dispatch + ~15 MB
  [B,vocab] readback at B=32). A traced-logits fast path is a deferred perf
  increment (revisit if P5 shows it matters at production B).
- **P2 — Async OpenAI API over the engine. DONE (2026-05-29).** New
  `serve/cb_api.py` in the SAME process as the device-owning engine: a
  closure-over-`state` FastAPI app whose handlers `submit()` to the engine and
  bridge the blocking token queue to async SSE via `loop.run_in_executor`. On
  client disconnect, Starlette cancels the SSE generator → `CancelledError` →
  `engine.cancel(rid)` → slot recycles (CB2 admission). Reuses the unit-tested
  pure helpers from `openai_endpoint`. (Design note: `request: Request` is not
  used — FastAPI 0.13x stopped auto-injecting Request, treats it as a query
  param; closure-over-state is the durable pattern.) A unit-level routing probe
  `experiments/serve/tests/test_cb_api_routing.py` (TestClient + fake engine,
  no device) catches signature/parsing bugs in milliseconds, complementing the
  qb1 e2e validator. **Gate PASS** (`experiments/cb/validate/engine_api.py`,
  qb1, 4 slots, sampling-mode engine): (a) non-stream /v1/chat/completions 200
  with correct OpenAI body + usage; (b) SSE streaming — 18 events, role +
  finish + [DONE], streamed text bit-identical to non-stream; (c) 6 concurrent
  /v1/completions vs serial refs — **each==ref=6/6**, all 6 distinct, 13.15s
  through 4 slots — HTTP layer multiplexes the engine with zero crosstalk;
  (d) cancel-on-disconnect — partial SSE, slot recycled, post-cancel request
  matches. Clean lifecycle, no wedge.
- **P3 — Lifecycle + robustness. PARTIAL DONE (2026-05-29).** SHIPPED:
  `experiments/serve/scripts/serve_cb.sh` — start/stop/status/restart for
  `uvicorn experiments.serve.cb_api:app`; SIGTERM → uvicorn graceful drain →
  lifespan `__aexit__` → `engine.stop()`; SIGKILL fallback only after 10s.
  Backpressure: `CBEngine(max_inflight=…)` via `threading.BoundedSemaphore`
  (acquired in `submit`, released on `done`/`cancel`); over-cap submits raise
  `queue.Full` → `cb_api` maps to HTTP 429. Readiness already in place
  (`/health` 503 until engine ready, 200 after). Routing probe extended (6/6
  PASS): backpressure → 429 verified end-to-end. DEFERRED (P3.5): explicit
  error-isolation try/except in the engine loop (so one bad device call doesn't
  kill the engine thread); qb1 e2e of the daemon (start → /health → request →
  stop → re-start → no fabric wedge). These need their own ~350s qb1 cycles
  and don't block productionization at low traffic.
- **P4 — Observability. DONE (2026-05-30).** `experiments/serve/cb_metrics.py`
  — handrolled Counter / Gauge / Histogram + Registry rendering Prometheus text
  exposition (no `prometheus_client` dep). `CBEngine` instruments per-step
  latency (`cb_step_seconds`), per-request TTFT (`cb_ttft_seconds`) + end-to-end
  duration (`cb_request_duration_seconds`), counters for submitted / done /
  cancelled / rejected / tokens-generated, gauges for slots-active / queue-depth
  / inflight / max-inflight / sampling. `cb_api` exposes `GET /metrics` returning
  `text/plain; version=0.0.4`. **Gate PASS** end-to-end on qb1:
  test_cb_api_routing 7/7 (no device) + engine_api 5/5 (real engine), with
  `cb_requests_submitted_total=16`, `cb_tokens_generated_total=244`,
  `cb_step_seconds_count=292` advancing live during (a)–(d). Bonus: (c) ran in
  3.59 s vs 13.15 s pre-P3.5-perf — the logits-trace win lands in concurrent
  throughput too (~3.6×). Structured-logs scaffold deferred to P5 (the load
  harness needs it more directly).
- **P5 — Load / SLO validation.** Load-test harness (M sustained concurrent
  users); throughput, p50/p99 latency, slot utilization, zero leakage/crash over
  a sustained run. **Gate:** SLOs met.
- **P6 — Deploy + docs.** README/CONTRIBUTING run guide, config, OpenAI client
  examples; the production server is the documented serving path.

- **P3.5-perf — Logits trace for sampling mode. DONE (2026-05-30).** Captures a
  single logits-returning trace when `sampling=True and use_trace=True`
  (replaces the per-step eager forward). Same forward graph as the P0 argmax
  trace, just stops one op earlier (`return_logits=True` was already plumbed
  through `forward_batch_tp_inner`). Per-slot host argmax/sample loop unchanged.
  The two trace modes never coexist (one engine = one mode), so there's no
  eager-vs-trace interleave hazard. **Gate PASS** (re-ran
  `engine_sampling.py`): (A) greedy-via-sampling-engine == device-argmax ref
  exactly, (B) mixed batch (seeds differ + coherent), (C) determinism. **2.54×
  faster** measured: `64 tok / 2.99s ≈ 125 ms/step` (P1 eager was 318 ms/step);
  the remaining ~19 ms over P0's traced argmax (106 ms/step) is the [B,vocab]
  readback + host sample, exactly as predicted.

**Then:** resume **prefill optimization (S2)** — chunked prefill into CB slots for
TTFT under load (multi-query paged SDPA + shiftacc conv; see
`27b_chunked_prefill_plan.md`).

## Risks / constraints
- One mesh ⇒ engine is single-threaded; concurrency is request-level (CB), not
  device-parallel. Prod host = qb2 (TP) — but keep its existing single-seq prod
  server up; bring the CB server up on qb1 first, then qb2.
- Never break `server_tp` (frozen) — CB server is additive/new path.
- ttnn device ops must stay on the engine thread; async API never touches device.
- Validate functionally (chat coherence / needle) + by load, not weak-ref cosine.
