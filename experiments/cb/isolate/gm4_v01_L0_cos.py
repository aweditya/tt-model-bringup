#!/usr/bin/env python3
"""v0.1.0 cosine validator — embed_scaled + L0 input_layernorm vs HF oracle.

Fork base: shape from `experiments/cb/dev/cb35_drift_ladder.py` (per
REUSE MANDATE) — cosine ladder + headline metric pattern.

Loads `.cache/hf_oracle_gemma4_12b/` (`hf_reference_gemma4_12b.py`
output) and runs `server_gemma4_unified_ttnn.step_forward_v01` on the
prompt's first token. Compares:
  - TT `embed_scaled` vs HF `hidden_states[0, pos=0, :]` (post-embed,
    post-sqrt-3840 scale).
  - TT `in_norm`       vs HF `L0_in_norm[pos=0, :]`.

Gate: per-sub-step cos ≥ 0.999. Both PASS → bootstrap + embed scale +
RMSNorm `w` convention are correct. Headline printed at end.

Run (qb1):
    bash scripts/run_remote.sh experiments/cb/isolate/gm4_v01_L0_cos.py
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
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def mad(a, b):
    return float(np.abs(a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)).max())


def main():
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}; run hf_reference_gemma4_12b.py first")
        return 1

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [49, seq, 3840]
    hf_embed_scaled = hidden_states[0, 0, :]  # post-embed-scale at pos 0
    hf_in_norm = np.load(ORACLE_DIR / "L0_in_norm.npy")  # [seq, 3840]
    hf_in_norm_pos0 = hf_in_norm[0, :]

    log(f"oracle: prompt_ids={prompt_ids.tolist()} hidden_shape={hidden_states.shape}")
    log(f"oracle: hf_embed_scaled[:4]={hf_embed_scaled[:4]} rms={np.sqrt(np.mean(hf_embed_scaled**2)):.4f}")
    log(f"oracle: hf_in_norm_pos0[:4]={hf_in_norm_pos0[:4]} rms={np.sqrt(np.mean(hf_in_norm_pos0**2)):.4f}")

    log("bootstrapping Gemma 4 12B server (eats ~3-5 min on first run)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    log("running v0.1.0 forward at pos 0 (tok_id = prompt_ids[0])…")
    cap = {}
    tok_id0 = int(prompt_ids[0])
    srv.step_forward_v01(state, tok_id=tok_id0, capture=cap)

    log(f"tt: embed_scaled[:4]={cap['embed_scaled'][:4]} rms={np.sqrt(np.mean(cap['embed_scaled']**2)):.4f}")
    log(f"tt: in_norm[:4]     ={cap['in_norm'][:4]} rms={np.sqrt(np.mean(cap['in_norm']**2)):.4f}")

    # Cosines
    c_embed = cos(cap["embed_scaled"], hf_embed_scaled)
    c_in    = cos(cap["in_norm"],      hf_in_norm_pos0)
    m_embed = mad(cap["embed_scaled"], hf_embed_scaled)
    m_in    = mad(cap["in_norm"],      hf_in_norm_pos0)

    log("=" * 64)
    log(f"v0.1.0 cosine ladder vs HF oracle (gate: cos ≥ {PASS_THRESH})")
    log("=" * 64)
    log(f"  embed_scaled : cos={c_embed:.6f} mad={m_embed:.4e} "
        f"[{'PASS' if c_embed >= PASS_THRESH else 'FAIL'}]")
    log(f"  in_norm      : cos={c_in:.6f}    mad={m_in:.4e} "
        f"[{'PASS' if c_in    >= PASS_THRESH else 'FAIL'}]")
    log("=" * 64)

    import ttnn
    ttnn.close_device(state.mesh)
    return 0 if (c_embed >= PASS_THRESH and c_in >= PASS_THRESH) else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
