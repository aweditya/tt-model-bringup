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

from cb_scheduler import Scheduler  # noqa: E402


class RequestHandle:
    """Caller-side handle to one in-flight request. `tokens()` blocks on the
    request's outbound queue, yielding generated token ids until the request
    terminates; `final` is then one of 'done' | 'cancelled' | 'error'."""

    __slots__ = ("rid", "prompt_len", "final", "error", "_q")

    def __init__(self, rid, q, prompt_len):
        self.rid = rid
        self.prompt_len = prompt_len
        self.final = None
        self.error = None
        self._q = q

    def tokens(self, timeout=None):
        while True:
            kind, payload = self._q.get(timeout=timeout)
            if kind == "tok":
                yield payload
            else:
                self.final = kind
                if kind == "error":
                    self.error = payload
                return


class CBEngine:
    """Thread-safe front end to the CB scheduler. One owned thread runs the loop."""

    def __init__(self, state, slots, max_new_cap, eos_id, use_trace=True,
                 sampling=False, idle_sleep=0.001):
        self.state = state
        self.slots = slots
        self.max_new_cap = int(max_new_cap)
        self.eos_id = eos_id
        self.use_trace = use_trace
        # sampling=True → eager per-slot temp/top-p/top-k (the chat-API mode);
        # sampling=False → greedy argmax trace (P0 fast path). See Scheduler.
        self.sampling = sampling
        self.idle_sleep = idle_sleep

        self._inbound = queue.Queue()      # (ext_rid, prompt, max_new, out_q)
        self._cancel_q = queue.Queue()     # ext_rid
        self._idlock = threading.Lock()
        self._next_id = 0
        self._stop = threading.Event()
        self.started = threading.Event()
        self._err = None
        self._thread = None
        self._sched = None

        # engine-thread-only:
        self._meta = {}            # sched_rid -> {ext, q, max_new, sent}
        self._ext_to_sched = {}    # ext_rid -> sched_rid

    # ---- public API (any thread) ----
    def start(self):
        self._thread = threading.Thread(target=self._run, name="cb-engine", daemon=True)
        self._thread.start()
        self.started.wait()
        if self._err is not None:
            raise self._err
        return self

    def submit(self, prompt_ids, max_new=None, sampling=None):
        """sampling: None → greedy; else {temperature, top_p, top_k, seed}.
        temperature<=0 is normalised to greedy. Requires the engine to have been
        built with sampling=True."""
        if self._stop.is_set():
            raise RuntimeError("engine is stopping; not accepting new requests")
        if sampling is not None:
            if not self.sampling:
                raise RuntimeError("engine was not built with sampling=True")
            if sampling.get("temperature", 0.0) <= 0.0:
                sampling = None  # greedy
        mn = self.max_new_cap if max_new is None else min(int(max_new), self.max_new_cap)
        with self._idlock:
            rid = self._next_id
            self._next_id += 1
        q = queue.Queue()
        self._inbound.put((rid, [int(t) for t in prompt_ids], mn, sampling, q))
        return RequestHandle(rid, q, len(prompt_ids))

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
                                    sampling=self.sampling)
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
                sched.step()
                self._stream()
            else:
                time.sleep(self.idle_sleep)

    def _drain_inbound(self):
        while True:
            try:
                ext, prompt, mn, sampling, q = self._inbound.get_nowait()
            except queue.Empty:
                return
            sched_rid = self._sched.submit(prompt, sampling=sampling)
            self._meta[sched_rid] = {"ext": ext, "q": q, "max_new": mn, "sent": 0}
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
            m["q"].put(("cancelled", None))

    def _stream(self):
        sched = self._sched
        for sched_rid in list(self._meta.keys()):
            m = self._meta[sched_rid]
            r = sched.reqs[sched_rid]
            gen = r["gen"]
            while m["sent"] < len(gen) and m["sent"] < m["max_new"]:
                m["q"].put(("tok", gen[m["sent"]]))
                m["sent"] += 1
            if r["status"] == "DONE" or m["sent"] >= m["max_new"]:
                if r["status"] != "DONE":   # per-request cap before global cap
                    sched.cancel(sched_rid)
                m["q"].put(("done", None))
                self._meta.pop(sched_rid, None)
                self._ext_to_sched.pop(m["ext"], None)
