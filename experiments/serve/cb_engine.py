#!/usr/bin/env python3
"""CB inference engine — device-owning thread + thread-safe submit/cancel/stream.

P0 of the production server (research/production_server_plan.md). Wraps the
validated Orca scheduler (cb_scheduler.Scheduler — pure orchestration over the
CB1/CB2/CB4 device primitives, no new device risk) behind a clean, thread-safe
API so an async HTTP layer (P2) can drive it without ever touching the mesh:

    engine = CBEngine(state, slots=32, max_new_cap=512, eos_id=...).start()
    h = engine.submit(prompt_ids, max_new=128)   # returns immediately
    for token_id in h.tokens():                   # blocks on a thread-safe queue
        ...
    engine.cancel(h.rid)                          # free the slot mid-flight
    engine.stop()                                 # drain in-flight + release trace

ONE engine thread owns the mesh and runs the scheduler loop:
    drain inbound (submit) → drain cancels → step → stream tokens out.
Callers interact only via queues; scheduler state (reqs/slots/waiting) is touched
by the engine thread alone. Greedy decode for P0 — per-request sampling is P1.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

from cb_metrics import Registry  # noqa: E402
from cb_scheduler import Scheduler  # noqa: E402


class RequestHandle:
    """One in-flight request. Two queue modes:
      - loop=None: sync queue.Queue; use .tokens() (blocks, no async req).
      - loop=<asyncio loop>: asyncio.Queue; use `await handle.aget()` in handlers.
        Engine pushes via loop.call_soon_threadsafe so no executor thread is burned.
    `final` ∈ {'done','cancelled','error'} once the request terminates."""

    __slots__ = ("rid", "prompt_len", "final", "error", "_q_sync", "_q_async", "_loop")

    def __init__(self, rid, prompt_len, loop=None):
        self.rid = rid
        self.prompt_len = prompt_len
        self.final = None
        self.error = None
        self._loop = loop
        if loop is not None:
            import asyncio as _asyncio
            self._q_async = _asyncio.Queue()
            self._q_sync = None
        else:
            self._q_sync = queue.Queue()
            self._q_async = None

    def _push(self, msg):
        """Cross-thread-safe push from engine thread."""
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._q_async.put_nowait, msg)
            except RuntimeError:
                pass  # loop closed (server shutting down) — drop silently
        else:
            self._q_sync.put(msg)

    def tokens(self, timeout=None):
        """Sync iterator (tests/CLI demos). Raises if handle is async-mode."""
        if self._q_sync is None:
            raise RuntimeError("async handle; use `await handle.aget()`")
        while True:
            kind, payload = self._q_sync.get(timeout=timeout)
            if kind == "tok":
                yield payload
            else:
                self.final = kind
                if kind == "error":
                    self.error = payload
                return

    async def aget(self):
        """Async one-shot get (FastAPI handlers)."""
        if self._q_async is None:
            raise RuntimeError("sync handle; use .tokens()")
        return await self._q_async.get()


class CBEngine:
    """Thread-safe front end to the CB scheduler. One owned thread runs the loop."""

    def __init__(self, state, slots, max_new_cap, eos_id, use_trace=True,
                 sampling=False, max_inflight=None, topk_k=None,
                 chunked_prefill=False, prefix_cache=False, prefix_ttl_s=300.0,
                 idle_sleep=0.001):
        self.state = state
        self.slots = slots
        self.max_new_cap = int(max_new_cap)
        self.eos_id = eos_id
        self.use_trace = use_trace
        # sampling=True → per-slot host temp/top-p/top-k each step.
        #   topk_k=None  → logits trace + full-vocab sample (default; best for
        #                  low concurrency / solo chat at slots ≤ ~8).
        #   topk_k=K     → W2 on-device top-k trace + per-slot K-elem sample
        #                  (~6× total step at B=32; ~75% slower at B=4 — use
        #                  when slots ≥ ~16). Typical K=128.
        # sampling=False → argmax trace (P0 fast path).
        self.sampling = sampling
        self.topk_k = int(topk_k) if topk_k else None
        # chunked_prefill=True → admit via S1a chunked prefill + on-device
        # transplant (S2); alternating PREFILL_ONLY / DECODE_ONLY steps. Wins
        # over 1-tok/iter at all L >= ~64 (2-2.5x at L=64-200; bigger at
        # longer L). Forces cb_conv_mode='kdim' inside the scheduler so the
        # post-transplant decode math is bit-identical to S1a's path.
        self.chunked_prefill = bool(chunked_prefill)
        # prefix_cache=True → slot-level content-keyed prefix caching. Completed
        # CB slots are held under hash(tokens_so_far); a returning chat reclaims
        # its slot at cur_pos=n_matched, skipping re-prefill. DN+KV state stays
        # in-place — no Marconi-style checkpointing. Plan: 27b_prefix_caching_plan.md.
        self.prefix_cache = bool(prefix_cache)
        self.prefix_ttl_s = float(prefix_ttl_s)
        # Backpressure: cap on total in-flight requests (queued+active). When at
        # cap, submit() raises queue.Full → API maps to HTTP 429. Default unlimited.
        self.max_inflight = max_inflight
        self._inflight_sem = (
            threading.BoundedSemaphore(max_inflight) if max_inflight else None
        )
        self.idle_sleep = idle_sleep

        self._inbound = queue.Queue()      # (ext_rid, prompt, max_new, sampling, submit_time, out_q)
        self._cancel_q = queue.Queue()     # ext_rid
        self._idlock = threading.Lock()
        self._next_id = 0
        self._stop = threading.Event()
        self.started = threading.Event()
        self._err = None
        self._thread = None
        self._sched = None

        # engine-thread-only:
        self._meta = {}            # sched_rid -> {ext, q, max_new, sent, submit_time, first_tok_time}
        self._ext_to_sched = {}    # ext_rid -> sched_rid

        # ---- P4 metrics (Prometheus text format via /metrics) ----
        self.metrics = Registry()
        M = self.metrics
        self.m_submitted = M.counter("cb_requests_submitted_total",
                                      "Requests accepted onto the engine queue.")
        self.m_done = M.counter("cb_requests_done_total",
                                 "Requests that completed (eos/max_new).")
        self.m_cancelled = M.counter("cb_requests_cancelled_total",
                                      "Requests cancelled (client disconnect or per-req cap).")
        self.m_rejected = M.counter("cb_requests_rejected_total",
                                     "Requests rejected by backpressure (max_inflight reached).")
        self.m_tokens = M.counter("cb_tokens_generated_total",
                                   "Tokens emitted to client streams.")
        self.m_step_seconds = M.histogram("cb_step_seconds",
                                           "Wall time of one scheduler step (one batched forward).")
        # Sub-step split (sampling mode only — populated by _step_sampled). Lets
        # us answer "host-loop vs device" at any B without re-running Tracy.
        self.m_step_device_seconds = M.histogram("cb_step_device_seconds",
            "Step time spent in device forward + readback (execute_trace -> to_torch -> upcast).")
        self.m_step_sample_seconds = M.histogram("cb_step_sample_seconds",
            "Step time spent in the per-slot host sample/argmax loop after readback.")
        self.m_ttft_seconds = M.histogram("cb_ttft_seconds",
                                           "Time-to-first-token from submit() to first stream emit.")
        self.m_request_seconds = M.histogram("cb_request_duration_seconds",
                                              "End-to-end request wall time (submit() to done/cancel).")
        M.gauge("cb_engine_slots_total", "Configured CB scheduler slots.",
                fn=lambda: self.slots)
        M.gauge("cb_engine_slots_active",
                "Slots currently holding an in-flight request.",
                fn=lambda: 0 if self._sched is None
                else sum(1 for s in self._sched.slots if s is not None))
        M.gauge("cb_engine_queue_depth",
                "Inbound + waiting depth (admitted/queued, not yet active).",
                fn=lambda: self._inbound.qsize() + (
                    0 if self._sched is None else len(self._sched.waiting)))
        M.gauge("cb_engine_inflight",
                "Total in-flight (queued + active); compare to cb_engine_max_inflight.",
                fn=lambda: self._inbound.qsize() + (
                    0 if self._sched is None else (len(self._sched.waiting) + len(self._meta))))
        M.gauge("cb_engine_max_inflight",
                "Backpressure cap; 0 = unlimited.",
                fn=lambda: self.max_inflight or 0)
        M.gauge("cb_engine_sampling",
                "1 if engine is in sampling mode (eager/logits-trace), else 0.",
                fn=lambda: 1.0 if self.sampling else 0.0)
        # ---- PC-P6 prefix-cache metrics ----
        # Scheduler increments pc_hits/pc_misses/pc_evictions; we expose them
        # as counters via lambda observation. live_slots gauge reflects current
        # cached slot count (visible idle vs active).
        self.m_pc_hits = M.counter("cb_prefix_cache_hits_total",
            "Requests admitted via prefix-cache hit (skipped re-prefill).")
        self.m_pc_misses = M.counter("cb_prefix_cache_misses_total",
            "Requests that fell through to cold prefill (no match in cache).")
        self.m_pc_evictions = M.counter("cb_prefix_cache_evictions_total",
            "Cached slots evicted to make room for new requests (LRU + TTL).")
        M.gauge("cb_prefix_cache_live_slots",
            "Slots currently in the live prefix cache (queued for reuse).",
            fn=lambda: 0 if (self._sched is None or self._sched.live_slots is None)
                else len(self._sched.live_slots))
        M.gauge("cb_prefix_cache_enabled",
            "1 if prefix caching is enabled on this engine.",
            fn=lambda: 1.0 if self.prefix_cache else 0.0)

    # ---- public API (any thread) ----
    def start(self):
        self._thread = threading.Thread(target=self._run, name="cb-engine", daemon=True)
        self._thread.start()
        self.started.wait()
        if self._err is not None:
            raise self._err
        return self

    def submit(self, prompt_ids, max_new=None, sampling=None, loop=None):
        """Returns a RequestHandle. If `loop` is an asyncio event loop, the
        handle uses an asyncio.Queue (handler awaits via `await handle.aget()`,
        engine pushes via call_soon_threadsafe → no executor thread burned).
        If `loop=None`, falls back to a sync queue.Queue (handle.tokens())."""
        if self._stop.is_set():
            raise RuntimeError("engine is stopping; not accepting new requests")
        if sampling is not None:
            if not self.sampling:
                raise RuntimeError("engine was not built with sampling=True")
            if sampling.get("temperature", 0.0) <= 0.0:
                sampling = None  # greedy
        if self._inflight_sem is not None and not self._inflight_sem.acquire(blocking=False):
            self.m_rejected.inc()
            raise queue.Full(f"engine in-flight cap reached (max_inflight={self.max_inflight})")
        mn = self.max_new_cap if max_new is None else min(int(max_new), self.max_new_cap)
        with self._idlock:
            rid = self._next_id
            self._next_id += 1
        handle = RequestHandle(rid, len(prompt_ids), loop=loop)
        self._inbound.put((rid, [int(t) for t in prompt_ids], mn, sampling, time.time(), handle))
        self.m_submitted.inc()
        return handle

    def cancel(self, rid):
        self._cancel_q.put(rid)

    def stop(self, timeout=60.0):
        """Stop admitting, drain in-flight, release trace, join the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ---- engine thread ----
    def _run(self):
        try:
            self._sched = Scheduler(self.state, self.slots, self.max_new_cap,
                                    self.eos_id, use_trace=self.use_trace,
                                    sampling=self.sampling, topk_k=self.topk_k,
                                    chunked_prefill=self.chunked_prefill,
                                    prefix_cache=self.prefix_cache)
            # Attach sub-step histograms so _step_sampled can record the split.
            self._sched.m_device = self.m_step_device_seconds
            self._sched.m_sample = self.m_step_sample_seconds
            self._pc_last_hits = 0
            self._pc_last_misses = 0
            self._pc_last_evictions = 0
            self._pc_ttl_last_check = time.monotonic()
        except BaseException as e:  # surface bootstrap/capture failure to start()
            self._err = e
            self.started.set()
            return
        self.started.set()
        try:
            self._loop()
        finally:
            if self._sched is not None:
                self._sched.release()

    def _loop(self):
        sched = self._sched
        while True:
            self._drain_inbound()
            self._drain_cancels()
            have_work = bool(sched.waiting) or any(s is not None for s in sched.slots)
            if self._stop.is_set() and not have_work and self._inbound.empty():
                break
            if have_work:
                t0 = time.perf_counter()
                try:
                    sched.step()
                except BaseException as e:
                    self._fail_active_requests(e)
                    continue
                self.m_step_seconds.observe(time.perf_counter() - t0)
                try:
                    self._stream()
                except BaseException as e:
                    self._fail_active_requests(e)
                    continue
            else:
                time.sleep(self.idle_sleep)
            if self.prefix_cache:
                self._pc_sync_metrics()
                self._pc_ttl_sweep()

    def _pc_sync_metrics(self):
        """Sync scheduler-side counters to Prometheus counters via delta. Cheap;
        called every iter — pure-Python int ops."""
        sched = self._sched
        if sched.pc_hits > self._pc_last_hits:
            self.m_pc_hits.inc(sched.pc_hits - self._pc_last_hits)
            self._pc_last_hits = sched.pc_hits
        if sched.pc_misses > self._pc_last_misses:
            self.m_pc_misses.inc(sched.pc_misses - self._pc_last_misses)
            self._pc_last_misses = sched.pc_misses
        if sched.pc_evictions > self._pc_last_evictions:
            self.m_pc_evictions.inc(sched.pc_evictions - self._pc_last_evictions)
            self._pc_last_evictions = sched.pc_evictions

    def _pc_ttl_sweep(self):
        """Periodically free live-cached slots that have been idle > prefix_ttl_s.
        Prevents disconnected/stale chats from indefinitely hoarding slots."""
        now = time.monotonic()
        if now - self._pc_ttl_last_check < 30.0:
            return
        self._pc_ttl_last_check = now
        if self._sched.live_slots is None:
            return
        expired = self._sched.live_slots.expire_stale(self.prefix_ttl_s)
        if expired:
            self.m_pc_evictions.inc(len(expired))
            self._pc_last_evictions = self._sched.pc_evictions  # don't double-count

    def _fail_active_requests(self, exc):
        """Push an error to every in-flight handle's queue + clear state, so
        FastAPI handlers don't block forever on a dead engine. The thread keeps
        running; the next admitted request gets a fresh slot pool."""
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[cb-engine] step failed: {msg}", file=sys.stderr, flush=True)
        for sched_rid in list(self._meta.keys()):
            m = self._meta.pop(sched_rid, None)
            if m is None:
                continue
            m["handle"]._push(("error", msg))
            self._ext_to_sched.pop(m["ext"], None)
            if self._inflight_sem is not None:
                try: self._inflight_sem.release()
                except ValueError: pass
        self._sched.waiting.clear()
        self._sched.slots = [None] * self._sched.B

    def _drain_inbound(self):
        while True:
            try:
                ext, prompt, mn, sampling, submit_t, handle = self._inbound.get_nowait()
            except queue.Empty:
                return
            sched_rid = self._sched.submit(prompt, sampling=sampling)
            self._meta[sched_rid] = {"ext": ext, "handle": handle, "max_new": mn,
                                      "sent": 0, "submit_time": submit_t,
                                      "first_tok_time": None}
            self._ext_to_sched[ext] = sched_rid

    def _drain_cancels(self):
        while True:
            try:
                ext = self._cancel_q.get_nowait()
            except queue.Empty:
                return
            sched_rid = self._ext_to_sched.get(ext)
            if sched_rid is None or sched_rid not in self._meta:
                continue
            self._sched.cancel(sched_rid)
            m = self._meta.pop(sched_rid)
            self._ext_to_sched.pop(ext, None)
            m["handle"]._push(("cancelled", None))
            self.m_cancelled.inc()
            self.m_request_seconds.observe(time.time() - m["submit_time"])
            if self._inflight_sem is not None:
                self._inflight_sem.release()

    def _stream(self):
        sched = self._sched
        now = time.time()
        for sched_rid in list(self._meta.keys()):
            m = self._meta[sched_rid]
            r = sched.reqs[sched_rid]
            gen = r["gen"]
            while m["sent"] < len(gen) and m["sent"] < m["max_new"]:
                if m["sent"] == 0:
                    m["first_tok_time"] = now
                    self.m_ttft_seconds.observe(now - m["submit_time"])
                m["handle"]._push(("tok", gen[m["sent"]]))
                m["sent"] += 1
                self.m_tokens.inc()
            if r["status"] == "DONE" or m["sent"] >= m["max_new"]:
                if r["status"] != "DONE":   # per-request cap before global cap
                    sched.cancel(sched_rid)
                m["handle"]._push(("done", None))
                self.m_done.inc()
                self.m_request_seconds.observe(now - m["submit_time"])
                self._meta.pop(sched_rid, None)
                self._ext_to_sched.pop(m["ext"], None)
                if self._inflight_sem is not None:
                    self._inflight_sem.release()
