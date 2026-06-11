#!/usr/bin/env python3
"""MM7 v0.1.1.b — L5 full attention block (pre-norm + qkv + SDPA + o_proj + residual).

Smoke gates for the FULL L5 attention forward. Inherits v0.1.1.a's
projection sub-gates and adds:

  O — o_proj output      cos vs HF L5_attn_o_proj.npy
  M — full mixer output  cos vs HF L5_attn_mixer_out.npy
      (= attention block output before residual add)
  B — block output       cos vs HF hidden_states[6]
      (= L5 block output after residual; this is the input to L6)

NB on SDPA: runs in numpy fp32 for v0.1.1. NKV=2 × NCHIPS=4 doesn't
shard cleanly; on-device prefill SDPA is a v0.5 perf concern. The math
is validated here against the most accurate reference.

REUSE: forks `nemotron3_v011_attn_projections_smoke.py`.

Run on the QuietBox:
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=5 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v011_attn_block_smoke.py
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
L5 = 5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    import os
    if "NEMOTRON3_UPLOAD_LAYERS" not in os.environ:
        os.environ["NEMOTRON3_UPLOAD_LAYERS"] = str(L5)

    log("loading HF oracle artifacts…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [53, 5, 2688]
    q_hf = np.load(ORACLE_DIR / "L5_attn_q_proj.npy")
    k_hf = np.load(ORACLE_DIR / "L5_attn_k_proj.npy")
    v_hf = np.load(ORACLE_DIR / "L5_attn_v_proj.npy")
    o_hf = np.load(ORACLE_DIR / "L5_attn_o_proj.npy")
    mixer_out_hf = np.load(ORACLE_DIR / "L5_attn_mixer_out.npy")
    for arr in (q_hf, k_hf, v_hf, o_hf, mixer_out_hf):
        if arr.ndim == 3 and arr.shape[0] == 1:
            pass  # leave as [1, S, ...]
    log(f"  hidden_states[5] (input)   shape: {hidden_states[L5].shape}")
    log(f"  hidden_states[6] (target)  shape: {hidden_states[L5+1].shape}")
    log(f"  mixer_out_hf  shape: {mixer_out_hf.shape}")
    log(f"  o_proj_hf     shape: {o_hf.shape}")

    log("bootstrapping (uploading L5)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        h_input = hidden_states[L5]  # [S, HIDDEN]
        log("running TT pre-norm + qkv + numpy SDPA + TT o_proj + residual…")
        res = srv.attn_block_eager(state, h_input, L5)

        # Sub-gates from v0.1.1.a (re-validated, should still PASS)
        cos_q, _ = cos_and_mad(res["q"], q_hf[0] if q_hf.ndim == 3 else q_hf)
        cos_k, _ = cos_and_mad(res["k"], k_hf[0] if k_hf.ndim == 3 else k_hf)
        cos_v, _ = cos_and_mad(res["v"], v_hf[0] if v_hf.ndim == 3 else v_hf)
        log(f"  (regression) q/k/v_proj cos: {cos_q:.6f} / {cos_k:.6f} / {cos_v:.6f}")

        # ── Gate O — o_proj output ─────────────────────────────
        o_target = o_hf[0] if o_hf.ndim == 3 else o_hf
        o_pred = res["o_proj_out"]
        if o_pred.ndim == 3 and o_target.ndim == 2:
            o_pred = o_pred[0]
        cos_o, mad_o = cos_and_mad(o_pred, o_target)
        gate_o = cos_o >= COS_GATE
        log(f"Gate O o_proj vs HF: cos={cos_o:.6f}  mad={mad_o:.4e}  "
            f"{'PASS ✓' if gate_o else 'FAIL ✗'}")

        # ── Gate M — full mixer output (= o_proj_out, since HF
        # mixer returns o_proj output for attention layers) ───
        m_target = mixer_out_hf[0] if mixer_out_hf.ndim == 3 else mixer_out_hf
        cos_m, mad_m = cos_and_mad(o_pred, m_target)
        gate_m = cos_m >= COS_GATE
        log(f"Gate M mixer_out vs HF: cos={cos_m:.6f}  mad={mad_m:.4e}  "
            f"{'PASS ✓' if gate_m else 'FAIL ✗'}")

        # ── Gate B — block output (= hidden_states[6]) ───────
        b_target = hidden_states[L5 + 1]
        b_pred = res["block_out"]
        if b_pred.ndim == 3 and b_target.ndim == 2:
            b_pred = b_pred[0]
        cos_b, mad_b = cos_and_mad(b_pred, b_target)
        gate_b = cos_b >= COS_GATE
        log(f"Gate B block_out vs hidden_states[6]: cos={cos_b:.6f}  "
            f"mad={mad_b:.4e}  {'PASS ✓' if gate_b else 'FAIL ✗'}")

        all_pass = gate_o and gate_m and gate_b
        n_pass = sum([gate_o, gate_m, gate_b])
        log("")
        log(f"v0.1.1.b attn-block smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/3 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
