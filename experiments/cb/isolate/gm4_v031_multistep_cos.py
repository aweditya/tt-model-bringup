#!/usr/bin/env python3
"""v0.3.1 multi-step decoder validation.

Teacher-forced: for each pos 0..N-1, feed HF's prompt_ids[pos] to the TT
forward + compare the TT argmax to HF argmax_per_position[pos]. KV cache
accumulates across calls. RoPE rotates non-trivially at pos > 0.

Gate: TT argmax matches HF for ALL positions 0..5 (6-token canonical
"The capital of France is" prompt).

Fork shape: `experiments/cb/isolate/gm4_v030_paged_cos.py` (single-pos)
extended to a loop. Same oracle, same gates per step.

Run (qb1):  bash scripts/run_remote.sh experiments/cb/isolate/gm4_v031_multistep_cos.py
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        return 1

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_argmax  = np.load(ORACLE_DIR / "argmax.npy")
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    seq_len = int(meta["seq_len"])
    log(f"oracle: prompt_ids={prompt_ids.tolist()} seq_len={seq_len}")
    log(f"oracle: HF argmax_per_position={hf_argmax.tolist()} "
        f"text={meta['argmax_text_per_position']}")

    log("bootstrapping Gemma 4 12B server (~80 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    # HF hidden_states shape [n_layers+1, seq, HIDDEN]. Last index is the
    # post-final-norm hidden state at each sequence position.
    hf_hidden = np.load(ORACLE_DIR / "hidden_states.npy")
    hf_final_per_pos = hf_hidden[-1]  # [seq, HIDDEN]

    def cos_(a, b):
        a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if (na and nb) else 0.0

    log("running teacher-forced multi-step decode…")
    results = []
    for pos in range(seq_len):
        tok_id = int(prompt_ids[pos])
        cap = {}
        argmax_tt = srv.step_forward_v031(state, tok_id, pos, capture=cap)
        hf_arg = int(hf_argmax[pos])
        match = (argmax_tt == hf_arg)
        results.append((pos, tok_id, argmax_tt, hf_arg, match))
        c_final = cos_(cap["final_norm"], hf_final_per_pos[pos])
        log(f"  pos={pos} in_tok={tok_id:>6d} TT_argmax={argmax_tt:>6d} "
            f"HF_argmax={hf_arg:>6d} cos_final={c_final:.4f} "
            f"{'PASS' if match else 'FAIL'}")

    log("=" * 64)
    log("Gemma 4 12B v0.3.1 multi-step teacher-forced gate")
    log("=" * 64)
    n_pass = sum(1 for r in results if r[4])
    log(f"  {n_pass}/{seq_len} positions match HF")
    all_pass = (n_pass == seq_len)
    log(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")

    import ttnn
    ttnn.close_device(state.mesh)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
