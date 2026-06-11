#!/usr/bin/env python3
"""v0.3.0 cosine validator — full forward via paged SDPA at pos 0.

Fork shape: `experiments/cb/isolate/gm4_v02_full_cos.py`. Calls
`step_forward_v03` (paged SDPA on sliding; v0.2 V-routing on global)
instead of `step_forward_v02`. Same gates: cos ≥ 0.999 on final_norm
+ logits + argmax matches HF at pos 0 (= 258882 `<image|>`).

Goal: prove the paged_update_cache + paged_scaled_dot_product_attention_decode
+ sliding_window_size=1024 pipeline works at pos 0. Then v0.3.1 extends
to pos > 0 with non-trivial RoPE.

Run (qb1):  bash scripts/run_remote.sh experiments/cb/isolate/gm4_v030_paged_cos.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"
PASS_THRESH = 0.999


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def mad(a, b):
    return float(np.abs(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)).max())


def main():
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        return 1

    prompt_ids   = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    hf_logits    = np.load(ORACLE_DIR / "logits.npy")
    hf_argmax    = np.load(ORACLE_DIR / "argmax.npy")
    hf_final     = hidden_states[-1, 0, :]

    log(f"oracle: HF predicts argmax[0] = {int(hf_argmax[0])}")

    log("bootstrapping Gemma 4 12B server (~75 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    log("running v0.3.0 PAGED forward at pos 0…")
    cap = {}
    tok_id0 = int(prompt_ids[0])
    argmax_tt = srv.step_forward_v03(state, tok_id=tok_id0, capture=cap)
    log(f"TT argmax (v0.3.0 paged) = {argmax_tt}")

    c_final  = cos(cap["final_norm"], hf_final)
    c_logits = cos(cap["logits"],     hf_logits[0])
    m_final  = mad(cap["final_norm"], hf_final)
    m_logits = mad(cap["logits"],     hf_logits[0])
    argmax_match = (argmax_tt == int(hf_argmax[0]))

    log("=" * 64)
    log(f"Gemma 4 12B v0.3.0 cosine ladder vs HF oracle (gate: cos ≥ {PASS_THRESH})")
    log("=" * 64)
    log(f"  final_norm : cos={c_final:.6f}  mad={m_final:.4e}  "
        f"[{'PASS' if c_final  >= PASS_THRESH else 'FAIL'}]")
    log(f"  logits     : cos={c_logits:.6f} mad={m_logits:.4e}  "
        f"[{'PASS' if c_logits >= PASS_THRESH else 'FAIL'}]")
    log(f"  argmax     : TT={argmax_tt} HF={int(hf_argmax[0])}  "
        f"[{'PASS' if argmax_match else 'FAIL'}]")
    log("=" * 64)
    all_pass = (c_final >= PASS_THRESH and c_logits >= PASS_THRESH and argmax_match)
    log(f"VERDICT: {'PASS' if all_pass else 'FAIL'}")

    import ttnn
    ttnn.close_device(state.mesh)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
