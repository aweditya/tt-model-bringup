#!/usr/bin/env python3
"""P0 gate — CBEngine: concurrent submit / per-slot isolation / cancel / max_new.

The engine (experiments/serve/cb_engine.py) is the device-owning thread that the
production HTTP layer will sit on. This validates the new code the engine adds on
top of the CB3-validated scheduler:

  1. Isolation under concurrency — fire 6 client THREADS through 4 slots (so the
     waiting queue + mid-run admission are exercised); each request's streamed
     output must equal its standalone B=1 greedy reference.
  2. Cancel + slot reuse — cancel a long request mid-flight; its stream must
     terminate 'cancelled', and a NEW request admitted into the freed slot must
     still match its reference (proves the freed slot is clean).
  3. Per-request max_new — a request capped at K must stream exactly K tokens,
     and they must be the K-token prefix of its full reference.
  4. Clean lifecycle — start()/stop() with no fabric wedge.

Refs are computed B=1 BEFORE the engine starts (shared mesh). Run on qb1:
  make run PY=experiments/cb/validate/engine.py        # traced (production path)
  scripts/run_remote.sh experiments/cb/validate/engine.py --eager
"""
from __future__ import annotations

import argparse
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

ISO_NEW = 16
PROMPTS = [
    "The capital of France is the city of",
    "Once upon a time there lived a young",
    "The largest planet in our solar system is",
    "Water boils at a temperature of one hundred",
    "The quick brown fox jumps over the",
    "Photosynthesis is the process by which",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--eager", action="store_true", help="run engine eager (default: traced)")
    args = ap.parse_args()

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

    log(f"=== standalone B=1 greedy refs (max_new={ISO_NEW}) ===")
    refs = [greedy_ref(state, p, ISO_NEW, eos_id) for p in pid]
    for i, r in enumerate(refs):
        log(f"  ref {i}: {r}")

    eng = CBEngine(state, args.slots, max_new_cap=256, eos_id=eos_id, use_trace=not args.eager)
    eng.start()
    log(f"=== engine up: {args.slots} slots, {'eager' if args.eager else 'traced'} ===")

    # ---- 1. concurrent isolation: 6 clients through `slots` slots ----
    log(f"--- (1) {len(pid)} concurrent clients / {args.slots} slots ---")
    out = [None] * len(pid)

    def _client(i):
        h = eng.submit(pid[i], max_new=ISO_NEW)
        out[i] = list(h.tokens())

    ts = [threading.Thread(target=_client, args=(i,)) for i in range(len(pid))]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    iso_ok = all(out[i] == refs[i] for i in range(len(pid)))
    for i in range(len(pid)):
        log(f"  req {i}: {'OK' if out[i] == refs[i] else 'MISMATCH'}  {out[i]}")
    log(f"  isolation: {'PASS' if iso_ok else 'FAIL'} "
        f"({sum(len(o) for o in out)} tok / {dt:.2f}s)")

    # ---- 2. cancel mid-flight + slot reuse ----
    log("--- (2) cancel mid-flight, then reuse the freed slot ---")
    h_long = eng.submit(pid[1], max_new=256)
    g = h_long.tokens()
    got = [next(g) for _ in range(5)]
    eng.cancel(h_long.rid)
    got += list(g)
    cancel_ok = (h_long.final == "cancelled") and (len(got) < 256)
    log(f"  long req: streamed {len(got)} tok then final={h_long.final!r} "
        f"→ {'OK' if cancel_ok else 'FAIL'}")
    h_reuse = eng.submit(pid[2], max_new=ISO_NEW)
    reuse = list(h_reuse.tokens())
    reuse_ok = (reuse == refs[2]) and (h_reuse.final == "done")
    log(f"  reuse req (slot recycled): {'OK' if reuse_ok else 'MISMATCH'}  {reuse}")

    # ---- 3. per-request max_new ----
    log("--- (3) per-request max_new ---")
    K = 5
    h_k = eng.submit(pid[0], max_new=K)
    short = list(h_k.tokens())
    maxnew_ok = (len(short) == K) and (short == refs[0][:K]) and (h_k.final == "done")
    log(f"  capped at {K}: got {len(short)} tok = {short}  → "
        f"{'OK' if maxnew_ok else 'FAIL'} (ref[:{K}]={refs[0][:K]})")

    eng.stop()
    log("=== engine stopped cleanly ===")

    ok = iso_ok and cancel_ok and reuse_ok and maxnew_ok
    log(f"\n=== verdict: {'PASS' if ok else 'FAIL'} ===")
    log(f"  isolation={iso_ok}  cancel={cancel_ok}  reuse={reuse_ok}  max_new={maxnew_ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
