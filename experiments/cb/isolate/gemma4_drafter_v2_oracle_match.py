#!/usr/bin/env python3
"""R-5 / R-6 sanity probe: replay v2 oracle inputs through our ttnn drafter.

Loads the R-2 v2 oracle artifacts for prompt 0 (HF's ground-truth K=5
drafter trajectory), feeds them through our ttnn drafter via the
CORRECT inputs_embeds construction, and checks per-round argmax
matches HF.

If our drafter doesn't reproduce v2 oracle's [496, 5464, 236772, 2084, 3207]
trajectory, the issue isn't just IT/BASE variant — there's a deeper bug
in either the drafter forward or how the scheduler feeds it.

Run on qb1:
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \\
      experiments/cb/isolate/gemma4_drafter_v2_oracle_match.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_12b_assistant_ttnn as drf  # noqa: E402

V2_ORACLE = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant_v2"
PROMPT_IDX = 0
K = 5

# HF v2 oracle's expected K=5 trajectory for prompt 0 ("The capital of France is").
EXPECTED_ARGMAXES = [496, 5464, 236772, 2084, 3207]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 64)
    log(f"R-5 sanity: replay v2 oracle prompt_{PROMPT_IDX} K={K} trajectory")
    log(f"  Expected argmaxes: {EXPECTED_ARGMAXES}")
    log("=" * 64)

    log("STAGE 1: bootstrap drafter (~8s)…")
    drf_state = drf.State()
    t0 = time.time()
    drf.bootstrap(drf_state, log=log)
    log(f"  drafter bootstrap took {time.time()-t0:.1f}s")

    # Load v2 oracle artifacts for prompt 0.
    pd = V2_ORACLE / f"prompt_{PROMPT_IDX}"
    target_h_last = np.load(pd / "target_h_last.npy").astype(np.float32)
    K_sl = np.load(pd / "shared_kv_sliding_K.npy").astype(np.float32)
    V_sl = np.load(pd / "shared_kv_sliding_V.npy").astype(np.float32)
    K_fl = np.load(pd / "shared_kv_full_K.npy").astype(np.float32)
    V_fl = np.load(pd / "shared_kv_full_V.npy").astype(np.float32)
    input_ids = np.load(pd / "input_ids.npy")

    log(f"target_h_last shape: {target_h_last.shape}")
    log(f"shared_kv shapes: sliding K={K_sl.shape}, full K={K_fl.shape}")
    log(f"input_ids: {input_ids.flatten().tolist()}")

    shared_kv = {
        "sliding_attention": (K_sl, V_sl),
        "full_attention": (K_fl, V_fl),
    }

    # We need the target's embed table. For an isolated drafter test, we
    # can load it from HF (slow, ~24 GB), OR use the v2 oracle's
    # inputs_embeds directly (skips embed lookup).
    # Use v2 oracle's stored inputs_embeds — that's the most direct test.
    log("STAGE 2: feed each round's exact v2 oracle inputs_embeds → drafter")
    log("─" * 64)
    rc = 0
    for round_r in range(K):
        round_dir = pd / f"K{K}" / f"round_{round_r}"
        if not round_dir.is_dir():
            log(f"  ✗ round {round_r}: artifact dir missing at {round_dir}")
            rc = 1; continue
        inputs_embeds = np.load(round_dir / "inputs_embeds.npy").astype(np.float32)
        expected_argmax = int(np.load(round_dir / "drafter_argmax.npy").flatten()[0])
        log(f"  round {round_r}: inputs_embeds shape {inputs_embeds.shape}, "
            f"expected argmax {expected_argmax}")

        t = time.time()
        out = drf.drafter_forward(drf_state, inputs_embeds, shared_kv)
        wall_ms = (time.time() - t) * 1000
        actual_argmax = int(out["argmax"].flatten()[0])
        match = "✓" if actual_argmax == expected_argmax else "✗"
        log(f"    {match} ttnn argmax={actual_argmax}  HF={expected_argmax}  "
            f"wall={wall_ms:.0f}ms")
        if actual_argmax != expected_argmax:
            rc = 1
            # Also print top-3 ttnn predictions to see how far off we are.
            logits = out["logits"].flatten()
            top3 = np.argsort(logits)[-3:][::-1]
            log(f"    ttnn top-3: {top3.tolist()}, "
                f"logits={[f'{logits[t]:.2f}' for t in top3]}")

    log("=" * 64)
    if rc == 0:
        log("VERDICT: PASS — ttnn drafter reproduces v2 oracle trajectory")
    else:
        log("VERDICT: FAIL — ttnn drafter does NOT match v2 oracle")
        log("  → bug is in our ttnn drafter forward, NOT the scheduler chain")
    log("=" * 64)
    ttnn.close_mesh_device(drf_state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main())
