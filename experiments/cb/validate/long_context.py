#!/usr/bin/env python3
"""S1a long-context gate: chunked prefill of a long prompt -> needle retrieval.

Confirms chunked prefill is usable for chat-like apps: prefills a long prompt with
an embedded needle (a distinctive code), decodes the answer, and checks retrieval
via chunked prefill (and the stub for comparison). Also validates the NON-CAPTURE
return path (capture vs non-capture first-token match) — that path is what
production uses through the `state.prefill_chunked` flag.

Run on qb1 (from repo root):
  make run PY=experiments/cb/validate/long_context.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _chip0_logits(state, rm):
    import ttnn
    t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
    return t.float().numpy()[0][:state.vocab_size]


def _decode(state, first_tid, start_pos, n):
    import ttnn
    gen = [int(first_tid)]
    tid, pos = int(first_tid), start_pos
    for _ in range(n - 1):
        base.update_input_buffers(state, tid, pos)
        rm = base.forward_token_tp_inner(state, return_logits=True)
        logits = _chip0_logits(state, rm)
        ttnn.deallocate(rm)
        tid = int(np.argmax(logits))
        gen.append(tid)
        pos += 1
    return gen


def main():
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-len", type=int, default=128, help="approx prompt token count")
    ap.add_argument("--decode", type=int, default=12)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    tokenizer = state.tok
    state.deltanet_recurrence_mode = "manual"

    NEEDLE = "7X9Q2"
    needle_s = f" Important: the Aurora project access code is {NEEDLE}. Keep it confidential."
    filler = (" The quarterly report covers routine administrative matters and general "
              "operational updates across the various regional offices and field teams.")
    sents = []
    while len(tokenizer.encode("".join(sents) + needle_s)) < args.target_len - 25:
        sents.append(filler)
    sents.insert(len(sents) // 2, needle_s)  # needle in the middle of the haystack
    prompt = ("".join(sents)
              + "\n\nQuestion: What is the Aurora project access code?\nAnswer: The access code is")
    ids = tokenizer.encode(prompt)
    L = len(ids)
    log(f"prompt L={L} tok; needle={NEEDLE!r}")

    # 1. Chunked prefill (capture) -> first token + populated state -> decode + retrieval.
    base._reset_state_buffers(state)
    cap = base.forward_prefill_chunked_tp(state, ids, capture_logits=True)
    first_cap = int(np.argmax(cap[-1]))
    chk_txt = tokenizer.decode(_decode(state, first_cap, L, args.decode))
    chk_found = NEEDLE in chk_txt
    log(f"chunked answer: {chk_txt!r}  -> needle {'FOUND' if chk_found else 'MISSING'}")

    # 2. Non-capture return-path check (what the prefill_chunked flag uses).
    base._reset_state_buffers(state)
    rm = base.forward_prefill_chunked_tp(state, ids, capture_logits=False)
    first_nc = int(np.argmax(_chip0_logits(state, rm)))
    ttnn.deallocate(rm)
    nc_ok = first_cap == first_nc
    log(f"non-capture path: first-token capture={first_cap} non-capture={first_nc} "
        f"{'OK' if nc_ok else 'MISMATCH'}")

    # 3. Stub prefill -> decode + retrieval (reference).
    base._reset_state_buffers(state)
    stub = base.forward_prefill_tp_inner(state, ids, capture_logits=True)
    stub_txt = tokenizer.decode(_decode(state, int(np.argmax(stub[-1])), L, args.decode))
    log(f"stub    answer: {stub_txt!r}  -> needle {'FOUND' if NEEDLE in stub_txt else 'MISSING'}")

    log(f"=== verdict: {'PASS' if (nc_ok and chk_found) else 'CHECK'} "
        f"(non-capture path {'OK' if nc_ok else 'BAD'}; chunked needle "
        f"{'retrieved' if chk_found else 'MISSING'}) ===")


if __name__ == "__main__":
    main()
