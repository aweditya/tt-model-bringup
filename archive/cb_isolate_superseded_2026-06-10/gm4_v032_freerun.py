#!/usr/bin/env python3
"""v0.3.2 free-run greedy decode — feed TT's own argmax as next input.

v0.3.1 multi-step (teacher-forced) gates pos 0..5 against HF with cos
≥ 0.997 at every position and 5/6 argmax match (pos 4 is a near-tie
bf16 noise per [[bf16-chain-drift-at-B-gt-1]]). v0.3.2 validates the
forward composition holds when TT generates its own continuation.

Plan:
  - pos 0..seq_len-1: teacher-forced (feed prompt_ids[pos]).
  - pos seq_len..seq_len+15: free-run (feed previous-step argmax).
  - Decode all tokens to text and print. Look for a coherent
    English continuation (no hard PASS gate — bf16 will diverge
    from HF after a few tokens).

Fork shape: `experiments/cb/isolate/gm4_v031_multistep_cos.py`.

Run (qb1):  bash scripts/run_remote.sh experiments/cb/isolate/gm4_v032_freerun.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"
FREE_RUN_TOKENS = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    seq_len = int(meta["seq_len"])
    log(f"oracle prompt: {prompt_ids.tolist()} ({meta['prompt']!r})")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta["model_id"])

    owned_state = state is None
    if owned_state:
        log("bootstrapping Gemma 4 12B server (~80 sec)…")
        t0 = time.time()
        state = srv.State()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("using pre-bootstrapped state from harness")

    log(f"teacher-forced prefill (pos 0..{seq_len-1}) + free-run "
        f"(pos {seq_len}..{seq_len+FREE_RUN_TOKENS-1})…")
    generated = list(prompt_ids.tolist())
    prev_argmax = None
    for pos in range(seq_len + FREE_RUN_TOKENS):
        if pos < seq_len:
            in_tok = int(prompt_ids[pos])  # teacher-forced
            phase = "TF "
        else:
            in_tok = prev_argmax           # free-run
            phase = "FR "
            generated.append(in_tok)
        argmax_tt = srv.step_forward_v031(state, in_tok, pos)
        prev_argmax = argmax_tt
        log(f"  {phase} pos={pos:>2d} in_tok={in_tok:>6d} "
            f"({tok.decode([in_tok])!r}) → {argmax_tt:>6d} "
            f"({tok.decode([argmax_tt])!r})")
    generated.append(prev_argmax)  # final step's output

    text = tok.decode(generated)
    log("=" * 78)
    log("Generated text:")
    log(f"  {text!r}")
    log("=" * 78)

    if owned_state:
        import ttnn
        ttnn.close_device(state.mesh)


if __name__ == "__main__":
    main()
