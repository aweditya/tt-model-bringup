#!/usr/bin/env python3
"""MM7 v0.1.3.a — L1 MoE router (sigmoid + e_score_bias + topk).

Smoke gates the router output (indices + weights) against HF oracle
hooks:

  T — topk_indices    EXACT set match vs HF L1_moe_router_idx
  W — topk_weights    per-token cos vs HF L1_moe_router_w

The "exact set match" on T is the right gate (HF doesn't sort topk
results — order can differ). On W we sort by indices for the
comparison.

REUSE: forks v0.1.2.a smoke.

Run on the QuietBox:
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=1 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v013_moe_router_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
COS_GATE = 0.999
L1 = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    import os
    if "NEMOTRON3_UPLOAD_LAYERS" not in os.environ:
        os.environ["NEMOTRON3_UPLOAD_LAYERS"] = str(L1)

    log("loading HF oracle artifacts…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    idx_hf = np.load(ORACLE_DIR / "L1_moe_router_idx.npy")
    w_hf = np.load(ORACLE_DIR / "L1_moe_router_w.npy")
    log(f"  hidden_states[1] shape: {hidden_states[L1].shape}")
    log(f"  idx_hf shape: {idx_hf.shape}, dtype={idx_hf.dtype}")
    log(f"  w_hf   shape: {w_hf.shape}")

    log("bootstrapping (uploading L1)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        h_input = hidden_states[L1]
        log("running TT pre-norm + router matmul + sigmoid + host topk…")
        res = srv.moe_router_only(state, h_input, L1)

        idx_tt = res["topk_indices"]   # [S, top_k]
        w_tt = res["topk_weights"]
        log(f"  idx_tt: shape={idx_tt.shape}\n    {idx_tt.tolist()}")
        log(f"  idx_hf:\n    {idx_hf.astype(np.int32).tolist()}")
        log(f"  w_tt: shape={w_tt.shape}\n    {w_tt.tolist()}")
        log(f"  w_hf:\n    {w_hf.tolist()}")

        # ── Gate T — exact set match per token ─────────────────
        idx_hf_int = idx_hf.astype(np.int32)
        per_tok_match = []
        for t in range(idx_tt.shape[0]):
            set_tt = set(idx_tt[t].tolist())
            set_hf = set(idx_hf_int[t].tolist())
            per_tok_match.append(set_tt == set_hf)
        all_match = all(per_tok_match)
        log(f"Gate T per-token set match: {per_tok_match}")
        log(f"Gate T topk_indices: {'PASS ✓' if all_match else 'FAIL ✗'}")

        # ── Gate W — weights match (after aligning by indices) ─
        # Sort both by index ascending so the same expert weights line up.
        sort_tt = np.argsort(idx_tt, axis=-1)
        sort_hf = np.argsort(idx_hf_int, axis=-1)
        rows = np.arange(idx_tt.shape[0])[:, None]
        idx_tt_sorted = idx_tt[rows, sort_tt]
        w_tt_sorted = w_tt[rows, sort_tt]
        idx_hf_sorted = idx_hf_int[rows, sort_hf]
        w_hf_sorted = w_hf[rows, sort_hf]
        # Sanity: after sorting both by their own indices, sets are aligned
        # only if Gate T already passed. Print for visibility.
        log(f"  idx_tt sorted: {idx_tt_sorted.tolist()}")
        log(f"  idx_hf sorted: {idx_hf_sorted.tolist()}")
        cos_w, mad_w = cos_and_mad(w_tt_sorted, w_hf_sorted)
        gate_w = cos_w >= COS_GATE
        log(f"Gate W topk_weights vs HF (sorted-by-idx): "
            f"cos={cos_w:.6f}  mad={mad_w:.4e}  "
            f"{'PASS ✓' if gate_w else 'FAIL ✗'}")

        all_pass = all_match and gate_w
        n_pass = int(all_match) + int(gate_w)
        log("")
        log(f"v0.1.3.a moe-router smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/2 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
