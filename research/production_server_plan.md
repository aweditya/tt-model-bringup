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
- **P1 — Per-request sampling in CB.** Per-slot logits/top-k from the batched
  forward; engine samples per slot. **Gate:** per-slot sampling correctness +
  readback cost; greedy unchanged.
- **P2 — Async OpenAI API over the engine.** Re-point `openai_endpoint` (or a new
  `serve/cb_api.py`) at `cb_engine`; SSE, cancellation, sampling, stop, usage.
  **Gate:** concurrent OpenAI clients (real chat), correctness + streaming.
- **P3 — Lifecycle + robustness.** `serve_cb.sh`; readiness/health; SIGTERM drain
  + clean mesh close; backpressure (429); validation; error isolation. **Gate:**
  shutdown clean (no wedge) + chaos (disconnects, bad requests, queue-full).
- **P4 — Observability.** `/metrics` + structured logs + latency histograms.
  **Gate:** metrics accurate under load.
- **P5 — Load / SLO validation.** Load-test harness (M sustained concurrent
  users); throughput, p50/p99 latency, slot utilization, zero leakage/crash over
  a sustained run. **Gate:** SLOs met.
- **P6 — Deploy + docs.** README/CONTRIBUTING run guide, config, OpenAI client
  examples; the production server is the documented serving path.

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
