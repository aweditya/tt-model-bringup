#!/usr/bin/env python3
"""v0.2 cosine validator — full 48-layer forward + final_norm + lm_head + softcap + argmax.

Fork shape: `experiments/cb/isolate/gm4_v01_L0_cos.py`. Same HF oracle,
new gates:
  - final_norm vs HF hidden_states[-1, 0, :]  (cos ≥ 0.999)
  - logits     vs HF logits[0, :]              (cos ≥ 0.999)
  - argmax     vs HF argmax_per_position[0]   (exact match — should be 258882 '<image|>')

Bootstraps the server (~76 sec) then runs step_forward_v02 at pos 0.

Run (qb1):
    bash scripts/run_remote.sh experiments/cb/isolate/gm4_v02_full_cos.py
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
        log(f"FATAL: oracle missing at {ORACLE_DIR}; run hf_reference_gemma4_12b.py first")
        return 1

    prompt_ids   = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [49, seq, 3840]
    hf_logits    = np.load(ORACLE_DIR / "logits.npy")          # [seq, 262144]
    hf_argmax    = np.load(ORACLE_DIR / "argmax.npy")
    hf_final     = hidden_states[-1, 0, :]                     # HF post-final-norm at pos 0
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    log(f"oracle: prompt_ids={prompt_ids.tolist()}")
    log(f"oracle: HF predicts argmax_per_position[0] = {int(hf_argmax[0])} "
        f"({meta['argmax_text_per_position'][0]!r})")

    log("bootstrapping Gemma 4 12B server (~76 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    log("running v0.2 forward at pos 0 (all 48 layers + final_norm + lm_head + softcap)…")
    cap = {"per_layer": True}
    tok_id0 = int(prompt_ids[0])
    argmax_tt = srv.step_forward_v02(state, tok_id=tok_id0, capture=cap)
    log(f"TT argmax = {argmax_tt}")

    # Per-layer cos to localize where the chain breaks.
    log("Per-layer cos vs HF hidden_states[L+1, 0, :] (layer_type, cos):")
    for L in range(48):
        key = f"layer_{L}"
        if key in cap:
            hf_L = hidden_states[L + 1, 0, :]
            c = cos(cap[key], hf_L)
            lt = state.layer_types[L][:3]  # 'sli' or 'ful'
            tag = " *FAIL*" if c < PASS_THRESH else ""
            log(f"  L={L:2d} ({lt})  cos={c:.6f}{tag}")

    c_final  = cos(cap["final_norm"], hf_final)
    c_logits = cos(cap["logits"],     hf_logits[0])
    m_final  = mad(cap["final_norm"], hf_final)
    m_logits = mad(cap["logits"],     hf_logits[0])
    argmax_match = (argmax_tt == int(hf_argmax[0]))

    log("=" * 64)
    log(f"Gemma 4 12B v0.2 cosine ladder vs HF oracle (gate: cos ≥ {PASS_THRESH})")
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
