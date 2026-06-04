#!/usr/bin/env python3
"""Per-layer h cosine ladder at pos 0 vs pos 1 — bisect the v0.3.1 cliff.

v0.3.1 multi-step probe shows pos 0 cos_final=0.9995 (bit-clean) and pos 1
cos_final=0.2618 (catastrophic). All isolation primitives PASS. This probe
captures the per-layer hidden state at pos 0 and pos 1 and compares each
to HF `hidden_states[L+1, pos, :]`, finding where TT diverges from HF.

HF hidden_states shape: [NUM_LAYERS + 1, seq, HIDDEN] = [49, 6, 3840].
hidden_states[L+1] = output of layer L (1-indexed; index 0 = embed input).

Expected outcomes:
1. Both pos 0 and pos 1 stay cos > 0.99 through all 48 layers, then final
   diverges → bug is in final_norm or LM head.
2. pos 0 stays clean; pos 1 has a sharp cliff at layer K → bug originates
   in layer K (first sliding, first global, or somewhere specific).
3. Both posthems compounding cleanly across layers → bf16 numeric noise.

Run on qb1:  bash scripts/run_remote.sh experiments/cb/isolate/gm4_per_layer_drift_pos1.py
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def main():
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_hidden  = np.load(ORACLE_DIR / "hidden_states.npy")  # [49, seq, HIDDEN]
    seq_len = hf_hidden.shape[1]
    log(f"oracle: {seq_len}-tok prompt; HF hidden_states {hf_hidden.shape}")

    log("bootstrapping Gemma 4 12B server (~80 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    # Step pos 0: PASS reference. Capture per-layer h.
    log("running pos 0 (PASS reference)…")
    cap0 = {"per_layer": True}
    tok0 = int(prompt_ids[0])
    srv.step_forward_v031(state, tok0, 0, capture=cap0)

    # Step pos 1: FAIL.
    log("running pos 1 (FAIL — cliff)…")
    cap1 = {"per_layer": True}
    tok1 = int(prompt_ids[1])
    srv.step_forward_v031(state, tok1, 1, capture=cap1)

    log("=" * 78)
    log("per-layer h cosine ladder — TT[L] vs HF[L+1, pos, :]")
    log("L | layer_type | pos0 cos | pos0 PASS | pos1 cos | pos1 PASS | Δ(pos0-pos1)")
    log("=" * 78)
    first_cliff_layer = None
    for L in range(srv.NUM_LAYERS):
        lt = state.layer_types[L]
        lt_short = "SL" if lt == "sliding_attention" else "GL"
        tt0 = cap0["layer_h"][L]
        tt1 = cap1["layer_h"][L]
        hf0 = hf_hidden[L + 1, 0, :]
        hf1 = hf_hidden[L + 1, 1, :]
        c0 = cos(tt0, hf0)
        c1 = cos(tt1, hf1)
        delta = c0 - c1
        p0 = "PASS" if c0 >= 0.99 else "FAIL"
        p1 = "PASS" if c1 >= 0.99 else "FAIL"
        if first_cliff_layer is None and delta > 0.2 and c1 < 0.9:
            first_cliff_layer = L
            tag = "  ← FIRST CLIFF"
        else:
            tag = ""
        log(f"  L{L:2d} | {lt_short} | {c0:.5f} | {p0} | {c1:.5f} | {p1} | {delta:+.4f}{tag}")

    log("=" * 78)
    if first_cliff_layer is not None:
        lt = state.layer_types[first_cliff_layer]
        log(f"VERDICT: drift cliff first appears at L{first_cliff_layer} ({lt}). "
            f"This is the bisection target — next debug step is to instrument "
            f"that layer's sub-ops (Q/K/V proj, norms, mixer, post_attn, MLP) "
            f"and run the same cos ladder at pos 0 vs pos 1.")
    else:
        log("VERDICT: no single-layer cliff — drift is gradual. Likely "
            "compounding bf16 noise or a global issue (embed, LM head).")

    import ttnn
    ttnn.close_device(state.mesh)


if __name__ == "__main__":
    main()
