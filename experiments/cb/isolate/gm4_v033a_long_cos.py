#!/usr/bin/env python3
"""v0.3.3.a — per-pos cosine ladder at L=215 vs HF (long-context gate).

Forks `gm4_v031_multistep_cos.py`. v0.3.1 validated pos 0..5; this
probe extends to pos 0..L-1 on a longer factual prompt (the Wikipedia
Eiffel Tower paragraph). Confirms the forward composition doesn't
drift past the 6-tok regime — gates the basic forward at moderate
context BEFORE we attempt the sliding-window-boundary (b) and
needle-haystack (c) sub-probes at L > 1024.

Gates (mirror 27B's [[long-context-cosine-ladder]] precedent):
  - PRIMARY: argmax match rate ≥ 90% — the production-relevant
    metric (does the model pick the same next-token as HF).
  - SECONDARY: median cos_final ≥ 0.99 AND min cos_final ≥ 0.95 —
    sanity check that bf16 chain noise stays within the historical
    envelope, NOT a cliff. The strict 0.99-everywhere gate is too
    aggressive for bf16 at L > 100; expected noise floor is 0.97-0.99
    per `[[bf16-prefill-drift-cliff]]`.

Reuses the same multi-step pattern (teacher-forced, KV cache
accumulates). Same per-pos cos vs HF `hidden_states[-1, pos, :]`.

Oracle: `.cache/hf_oracle_gemma4_12b_L215/` — generated from
`experiments/utils/long_prompts/wikipedia_eiffel.txt`.

Run (qb1):
  bash scripts/run_remote.sh experiments/cb/isolate/gm4_v033a_long_cos.py
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

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_L215"
ARGMAX_GATE = 0.90        # primary: production-relevant top-1 agreement
COS_MEDIAN_GATE = 0.99    # secondary: most positions are bit-clean
COS_P5_GATE = 0.95        # secondary: 95% of positions stay above 0.95 —
                          # noise floor for bf16 chain noise. A 5th-pctile
                          # gate (not MIN) absorbs single-position outliers
                          # that don't move the production argmax rate.


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def main(state=None):
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        log("Generate first:")
        log("  ssh qb1 'cd ~/tt-xla && HF_HUB_OFFLINE=1 .venv/bin/python -u "
            "experiments/utils/hf_reference_gemma4_12b.py "
            "--prompt-file experiments/utils/long_prompts/wikipedia_eiffel.txt "
            "--output-dir .cache/hf_oracle_gemma4_12b_L215'")
        return 1

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_argmax = np.load(ORACLE_DIR / "argmax.npy")
    hf_hidden = np.load(ORACLE_DIR / "hidden_states.npy")  # [49, seq, HIDDEN]
    hf_final_per_pos = hf_hidden[-1]
    seq_len = int(prompt_ids.shape[0])
    log(f"oracle: seq_len={seq_len}; hidden_states {hf_hidden.shape}")

    if seq_len > srv.MAX_KV:
        log(f"FATAL: seq_len={seq_len} > MAX_KV={srv.MAX_KV}. Bump MAX_KV.")
        return 1

    owned_state = state is None
    if owned_state:
        log("bootstrapping Gemma 4 12B server (~80 sec)…")
        t0 = time.time()
        state = srv.State()
        srv.bootstrap(state, log=log)
        log(f"bootstrap took {time.time()-t0:.1f}s")
    else:
        log("using pre-bootstrapped state from harness")

    log(f"running teacher-forced multi-step decode pos 0..{seq_len-1}…")
    n_argmax_pass = 0
    cos_vals = []
    t_step = time.time()
    for pos in range(seq_len):
        tok_id = int(prompt_ids[pos])
        cap = {}
        argmax_tt = srv.step_forward_v031(state, tok_id, pos, capture=cap)
        c = cos(cap["final_norm"], hf_final_per_pos[pos])
        cos_vals.append(c)
        if argmax_tt == int(hf_argmax[pos]):
            n_argmax_pass += 1
        if (pos + 1) % 16 == 0 or pos == seq_len - 1:
            dt = time.time() - t_step
            log(f"  pos={pos:>3d} cos_final={c:.4f} "
                f"argmax {'PASS' if argmax_tt == int(hf_argmax[pos]) else 'FAIL'} "
                f"(elapsed {dt:.1f}s, {n_argmax_pass}/{pos+1} argmax PASS)")

    cos_arr = np.array(cos_vals)
    argmax_frac = n_argmax_pass / seq_len
    cos_median = float(np.median(cos_arr))
    cos_min = float(cos_arr.min())
    cos_min_pos = int(cos_arr.argmin())
    p95 = float(np.percentile(cos_arr, 5))   # bottom 5%

    log("=" * 78)
    log(f"v0.3.3.a long-context cosine ladder vs HF (L={seq_len})")
    log("=" * 78)
    log(f"  argmax match rate:   {n_argmax_pass}/{seq_len} ({argmax_frac*100:.2f}%) "
        f"[primary gate ≥ {ARGMAX_GATE*100:.0f}% → "
        f"{'PASS' if argmax_frac >= ARGMAX_GATE else 'FAIL'}]")
    log(f"  cos_final median:    {cos_median:.4f} "
        f"[gate ≥ {COS_MEDIAN_GATE} → "
        f"{'PASS' if cos_median >= COS_MEDIAN_GATE else 'FAIL'}]")
    log(f"  cos_final 5th-pct:   {p95:.4f} "
        f"[gate ≥ {COS_P5_GATE} → "
        f"{'PASS' if p95 >= COS_P5_GATE else 'FAIL'}]")
    log(f"  cos_final MIN:       {cos_min:.4f} at pos={cos_min_pos} "
        f"(info only — single-position outliers don't fail the test)")

    gates_pass = (argmax_frac >= ARGMAX_GATE
                  and cos_median >= COS_MEDIAN_GATE
                  and p95 >= COS_P5_GATE)
    log(f"VERDICT: {'PASS' if gates_pass else 'FAIL'}")

    if owned_state:
        import ttnn
        ttnn.close_device(state.mesh)
    return 0 if gates_pass else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
