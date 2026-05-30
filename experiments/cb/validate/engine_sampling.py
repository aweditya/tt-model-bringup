#!/usr/bin/env python3
"""P1 gate — per-request sampling in the CB engine.

The chat API (P2) runs the engine in sampling mode: every step does the eager
logits forward ([B,vocab]) and samples each slot with its own temp/top-p/top-k/
seed; greedy slots (temperature<=0) take the host argmax. This validates:

  (A) greedy-via-sampling-engine is exact — a greedy request through the sampling
      engine (host argmax on the logits) == its standalone device-argmax ref.
  (B) mixed batch — greedy slots are unaffected by concurrent sampling slots
      (== refs); two sampled slots (same params, different seeds) produce
      coherent, DIFFERENT text.
  (C) determinism — same prompt + same seed → identical output.

Also reports the per-step eager cost (the [B,vocab] readback the plan flagged).
Refs are computed B=1 (device argmax) BEFORE the engine starts. Run on qb1:
  make run PY=experiments/cb/validate/engine_sampling.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base               # noqa: E402
from cb_engine import CBEngine          # noqa: E402
from cb_scheduler import greedy_ref     # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NEW = 16
PROMPTS = [
    "The capital of France is the city of",
    "Once upon a time there lived a young",
]
SAMP = {"temperature": 0.8, "top_p": 0.95, "top_k": 0}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _drain(eng, prompt, **kw):
    """Submit, fully consume, return (token ids, final-state)."""
    h = eng.submit(prompt, max_new=NEW, **kw)
    return list(h.tokens()), h.final


def _concurrent(eng, jobs):
    """jobs: list of (prompt, kwargs). Returns list of token-id lists (by index)."""
    out = [None] * len(jobs)

    def _run(i):
        h = eng.submit(jobs[i][0], max_new=NEW, **jobs[i][1])
        out[i] = list(h.tokens())

    ts = [threading.Thread(target=_run, args=(i,)) for i in range(len(jobs))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def main():
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    tok = state.tok
    eos_id = getattr(tok, "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else -1
    pid = [tok.encode(p) for p in PROMPTS]

    log(f"=== standalone B=1 greedy refs (device argmax, max_new={NEW}) ===")
    refs = [greedy_ref(state, p, NEW, eos_id) for p in pid]
    for i, r in enumerate(refs):
        log(f"  ref {i}: {r}")

    eng = CBEngine(state, slots=4, max_new_cap=256, eos_id=eos_id, sampling=True)
    eng.start()
    log("=== engine up: 4 slots, SAMPLING mode (logits trace) ===")

    # (A) greedy through the sampling engine == device-argmax ref
    log("--- (A) greedy via sampling engine == device-argmax ref ---")
    a_ok = True
    for i in range(len(pid)):
        g, fin = _drain(eng, pid[i])
        match = (g == refs[i]) and (fin == "done")
        a_ok = a_ok and match
        log(f"  req {i}: {'OK' if match else 'MISMATCH'}  {g}")

    # (B) mixed batch: greedy unaffected; two seeds differ + coherent
    log("--- (B) mixed concurrent batch (2 greedy + 2 sampled) ---")
    jobs = [
        (pid[0], {}),                                  # greedy
        (pid[0], {"sampling": {**SAMP, "seed": 0}}),   # sampled s0
        (pid[0], {"sampling": {**SAMP, "seed": 1}}),   # sampled s1
        (pid[1], {}),                                  # greedy
    ]
    t0 = time.time()
    o = _concurrent(eng, jobs)
    dt = time.time() - t0
    greedy_ok = (o[0] == refs[0]) and (o[3] == refs[1])
    varied = (o[1] != o[2])
    coherent = all(len(tok.decode(o[k], skip_special_tokens=True).strip()) > 0 for k in (1, 2))
    log(f"  greedy slot0 vs ref0: {'OK' if o[0] == refs[0] else 'MISMATCH'}")
    log(f"  greedy slot3 vs ref1: {'OK' if o[3] == refs[1] else 'MISMATCH'}")
    log(f"  sampled s0: {tok.decode(o[1], skip_special_tokens=True)!r}")
    log(f"  sampled s1: {tok.decode(o[2], skip_special_tokens=True)!r}")
    log(f"  sampled differ across seeds: {varied}; both coherent: {coherent}")
    b_ok = greedy_ok and varied and coherent

    # (C) determinism: same prompt + same seed → identical
    log("--- (C) determinism (same prompt + same seed) ---")
    d = _concurrent(eng, [(pid[0], {"sampling": {**SAMP, "seed": 7}}),
                          (pid[0], {"sampling": {**SAMP, "seed": 7}})])
    c_ok = (d[0] == d[1])
    log(f"  seed=7 twice identical: {c_ok}  {d[0]}")

    # eager cost note: 4 slots × NEW tok in dt → per-(active-step) wall time
    steps = max(NEW + max(len(p) for p in pid), 1)
    log(f"  [cost] mixed batch (B≤4): {sum(len(x) for x in o)} tok / {dt:.2f}s "
        f"≈ {dt / steps * 1000:.0f} ms/step (incl. [B,vocab] readback + host sample)")

    eng.stop()
    log("=== engine stopped cleanly ===")

    ok = a_ok and b_ok and c_ok
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log(f"  (A) greedy-exact={a_ok}  (B) mixed/varied/coherent={b_ok}  (C) deterministic={c_ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
