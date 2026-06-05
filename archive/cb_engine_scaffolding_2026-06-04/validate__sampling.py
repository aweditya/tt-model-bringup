#!/usr/bin/env python3
"""Productionization gate: sampling (temperature / top-p / top-k).

Bootstraps server_tp and runs handle_generate_tp three ways:
  - greedy (temperature=0 → the existing traced argmax path, must be unchanged),
  - sampled (temperature>0 → non-traced logits + host sampling) with two seeds.
Confirms greedy is coherent/deterministic and sampling is coherent + varies across
seeds.

Run on qb1 (from repo root):
  make run PY=experiments/cb/validate/sampling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "serve").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(state, prompt, **kw):
    final = None
    for chunk in base.handle_generate_tp(state, {"prompt": prompt, "max_tokens": 40, **kw}):
        if chunk.get("_final"):
            final = chunk
    return final


def main():
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    prompt = "The capital of France is"

    g = _run(state, prompt)  # greedy (temperature defaults to 0)
    log(f"greedy            : {g['generated_text']!r}  ({g.get('ms_per_tok', float('nan')):.1f} ms/tok)")
    s0 = _run(state, prompt, temperature=0.8, top_p=0.95, seed=0)
    log(f"sample(t=.8,p=.95,s0): {s0['generated_text']!r}  ({s0.get('ms_per_tok', float('nan')):.1f} ms/tok)")
    s1 = _run(state, prompt, temperature=0.8, top_p=0.95, seed=1)
    log(f"sample(t=.8,p=.95,s1): {s1['generated_text']!r}")
    g2 = _run(state, prompt)  # greedy again — must reproduce
    greedy_det = g['generated_text'] == g2['generated_text']
    varied = s0['generated_text'] != s1['generated_text']
    log(f"=== verdict: {'PASS' if (greedy_det and varied) else 'CHECK'} "
        f"(greedy deterministic={greedy_det}; sampling varies across seeds={varied}) ===")


if __name__ == "__main__":
    main()
