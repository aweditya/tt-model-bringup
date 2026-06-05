#!/usr/bin/env python3
"""MM7 v0.3.1.a — quadratic-prefix multi-step decode smoke.

Validates that TT predicts the same N-token continuation as HF, BEFORE
adding real KV cache + ssm_state. Each step re-runs the full forward on
[prompt + already_generated_tokens] and takes argmax of the last position.

This is O(n²) (re-computes prefix every step) — fine for an 8-token
correctness gate but won't scale. v0.3.1.b will replace this with a
proper decode loop carrying state on device.

Gate: TT next-tokens for steps 0..7 match HF token-for-token.

Requires .cache/hf_oracle_nemotron3_nano with --gen 8 (so full_ids has
prompt + 8 tokens). If only prompt+0 is available, gates 1..7 are
skipped with a warning.

REUSE: forks `nemotron3_v030_resident_smoke.py`'s forward helper.
Harness-aware: accepts `state=None`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
N_LAYERS = 52
LOG_DIR = PROJECT_ROOT / ".cache" / "smoke_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"v031a_quadratic_{time.strftime('%Y%m%d_%H%M%S')}.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def run_forward_argmax(state, ids_np, srv, ttnn) -> int:
    """Run one all-resident forward, return argmax at last position."""
    h_np = srv.embed_lookup(state, ids_np[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next_tt = srv.attn_block_eager_tt(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        else:
            raise NotImplementedError(f"L{L} kind={kind!r}")
        ttnn.deallocate(h_tt)
        h_tt = h_next_tt
    h_np = ttnn.to_torch(
        h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    ttnn.deallocate(h_tt)
    h_final = srv.apply_final_norm(state, h_np)
    _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
    if argmax_np.ndim == 2:
        argmax_np = argmax_np[0]
    return int(argmax_np[-1])


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle…")
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    full_ids = np.asarray(meta.get("full_ids", []), dtype=np.int64)
    n_gen = int(meta.get("gen", 0))
    log(f"  prompt_ids ({len(prompt_ids)}): {prompt_ids.tolist()}")
    log(f"  oracle gen={n_gen}; full_ids ({len(full_ids)}): {full_ids.tolist()}")
    if n_gen < 1:
        log("WARNING: oracle has gen=0 — only step 0 has HF ground truth.")
        log("Re-run experiments/utils/hf_reference_nemotron3_nano.py --gen 8")

    import server_nemotron3_nano_ttnn as srv
    import ttnn
    t_boot = 0.0
    if state is None:
        log("bootstrap (all-resident)…")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log)
        t_boot = time.time() - t0
        log(f"  bootstrap in {t_boot:.1f}s")
    else:
        log("[harness] reusing live state ✓")

    try:
        N_STEPS = min(8, max(1, len(full_ids) - len(prompt_ids)))
        if n_gen == 0:
            N_STEPS = 1  # only the prompt→first-token gate works
        log(f"running {N_STEPS} decode step(s) via quadratic prefix growth…")

        cur_ids = list(int(x) for x in prompt_ids.tolist())
        gate_results = []
        for step in range(N_STEPS):
            t0 = time.time()
            ids_np = np.asarray(cur_ids, dtype=np.int64)
            tt_next = run_forward_argmax(state, ids_np, srv, ttnn)
            elapsed = time.time() - t0
            target_pos = len(cur_ids)
            if target_pos < len(full_ids):
                hf_next = int(full_ids[target_pos])
                ok = tt_next == hf_next
                gate_results.append(ok)
                log(f"step {step}: prefix_len={len(cur_ids)}  TT={tt_next}  "
                    f"HF={hf_next}  {'PASS ✓' if ok else 'FAIL ✗'}  "
                    f"({elapsed:.1f}s)")
            else:
                log(f"step {step}: prefix_len={len(cur_ids)}  TT={tt_next}  "
                    f"(no HF reference at pos {target_pos}; ran {elapsed:.1f}s)")
            cur_ids.append(tt_next)

        n_pass = sum(gate_results)
        n_total = len(gate_results)
        log("")
        log(f"v0.3.1.a {'PASS ✓' if n_pass == n_total else 'FAIL ✗'} "
            f"({n_pass}/{n_total} gates green)")
        if n_pass < n_total:
            log("  ⚠ TT diverges from HF at some step — bf16 chain drift or a")
            log("    real bug. Inspect the first divergent step.")
        return 0 if n_pass == n_total else 1
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
