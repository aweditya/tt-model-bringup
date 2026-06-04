#!/usr/bin/env python3
"""v0.3.3.a — per-pos cosine ladder at L=215 vs HF (long-context gate).

Forks `gm4_v031_multistep_cos.py`. v0.3.1 validated pos 0..5; this
probe extends to pos 0..L-1 on a longer factual prompt (the Wikipedia
Eiffel Tower paragraph). Confirms the forward composition doesn't
drift past the 6-tok regime — gates the basic forward at moderate
context BEFORE we attempt the sliding-window-boundary (b) and
needle-haystack (c) sub-probes at L > 1024.

Gate: cos_final ≥ 0.99 at ≥ 95% of positions (allows for occasional
bf16 chain-noise dips on near-tie positions, per
[[bf16-chain-drift-at-B-gt-1]]). Argmax match rate reported but not
gated — argmax can flip on near-ties even when cos is high.

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
COS_THRESH = 0.99
PASS_FRAC = 0.95


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def main():
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

    log("bootstrapping Gemma 4 12B server (~80 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    log(f"running teacher-forced multi-step decode pos 0..{seq_len-1}…")
    n_cos_pass = 0
    n_argmax_pass = 0
    cos_low_positions = []
    t_step = time.time()
    for pos in range(seq_len):
        tok_id = int(prompt_ids[pos])
        cap = {}
        argmax_tt = srv.step_forward_v031(state, tok_id, pos, capture=cap)
        c = cos(cap["final_norm"], hf_final_per_pos[pos])
        if c >= COS_THRESH:
            n_cos_pass += 1
        else:
            cos_low_positions.append((pos, c))
        if argmax_tt == int(hf_argmax[pos]):
            n_argmax_pass += 1
        # Progress every 16 steps.
        if (pos + 1) % 16 == 0 or pos == seq_len - 1:
            dt = time.time() - t_step
            log(f"  pos={pos:>3d} cos_final={c:.4f} "
                f"argmax {'PASS' if argmax_tt == int(hf_argmax[pos]) else 'FAIL'} "
                f"(elapsed {dt:.1f}s, {n_cos_pass}/{pos+1} cos PASS, "
                f"{n_argmax_pass}/{pos+1} argmax PASS)")

    cos_frac = n_cos_pass / seq_len
    argmax_frac = n_argmax_pass / seq_len
    log("=" * 78)
    log(f"v0.3.3.a long-context cosine ladder vs HF (L={seq_len})")
    log("=" * 78)
    log(f"  cos_final ≥ {COS_THRESH}:  {n_cos_pass}/{seq_len} ({cos_frac*100:.1f}%) "
        f"[gate ≥ {PASS_FRAC*100:.0f}% → {'PASS' if cos_frac >= PASS_FRAC else 'FAIL'}]")
    log(f"  argmax match (info only):  {n_argmax_pass}/{seq_len} ({argmax_frac*100:.1f}%)")
    if cos_low_positions:
        log(f"  low-cos positions (cos < {COS_THRESH}): "
            f"{cos_low_positions[:20]}{'…' if len(cos_low_positions) > 20 else ''}")

    verdict = "PASS" if cos_frac >= PASS_FRAC else "FAIL"
    log(f"VERDICT: {verdict}")

    import ttnn
    ttnn.close_device(state.mesh)
    return 0 if cos_frac >= PASS_FRAC else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
