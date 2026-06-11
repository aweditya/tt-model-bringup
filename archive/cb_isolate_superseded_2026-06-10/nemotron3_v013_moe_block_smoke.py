#!/usr/bin/env python3
"""MM7 v0.1.3.b — L1 MoE full block forward.

Smoke gates the full L1 MoE block end-to-end vs HF oracle hooks:

  S — shared_out  cos vs HF L1_moe_shared_out
  M — mixer_out   cos vs HF L1_moe_mixer_out  (= routed + shared, pre-residual)
  B — block_out   cos vs HF hidden_states[2]   (post-residual)

REUSE: forks v0.1.3.a smoke.

Run on the QuietBox:
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=1 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v013_moe_block_smoke.py
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
    shared_hf = np.load(ORACLE_DIR / "L1_moe_shared_out.npy")
    mixer_hf = np.load(ORACLE_DIR / "L1_moe_mixer_out.npy")
    log(f"  shared_hf shape: {shared_hf.shape}")
    log(f"  mixer_hf  shape: {mixer_hf.shape}")

    log("bootstrapping (L1 full MoE — ~1.3 GB upload)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        h_input = hidden_states[L1]
        log("running TT MoE block (pre-norm + router + per-token dispatch + "
            "shared + combine + residual)…")
        t_fwd = time.time()
        res = srv.moe_block_eager(state, h_input, L1)
        log(f"  forward in {time.time() - t_fwd:.1f}s")
        for k in ["shared_out", "mixer_out", "block_out"]:
            log(f"  {k} shape: {res[k].shape}")

        # Gate S — shared_out
        s_target = shared_hf[0] if shared_hf.ndim == 3 else shared_hf
        s_pred = res["shared_out"]
        if s_pred.ndim == 3 and s_target.ndim == 2:
            s_pred = s_pred[0]
        cos_s, mad_s = cos_and_mad(s_pred, s_target)
        gate_s = cos_s >= COS_GATE
        log(f"Gate S shared_out vs HF: cos={cos_s:.6f}  mad={mad_s:.4e}  "
            f"{'PASS ✓' if gate_s else 'FAIL ✗'}")

        # Gate M — mixer_out
        m_target = mixer_hf[0] if mixer_hf.ndim == 3 else mixer_hf
        m_pred = res["mixer_out"]
        if m_pred.ndim == 3 and m_target.ndim == 2:
            m_pred = m_pred[0]
        cos_m, mad_m = cos_and_mad(m_pred, m_target)
        gate_m = cos_m >= COS_GATE
        log(f"Gate M mixer_out vs HF: cos={cos_m:.6f}  mad={mad_m:.4e}  "
            f"{'PASS ✓' if gate_m else 'FAIL ✗'}")

        # Gate B — block_out vs hidden_states[2]
        b_target = hidden_states[L1 + 1]
        b_pred = res["block_out"]
        if b_pred.ndim == 3 and b_target.ndim == 2:
            b_pred = b_pred[0]
        cos_b, mad_b = cos_and_mad(b_pred, b_target)
        gate_b = cos_b >= COS_GATE
        log(f"Gate B block_out vs hidden_states[2]: cos={cos_b:.6f}  "
            f"mad={mad_b:.4e}  {'PASS ✓' if gate_b else 'FAIL ✗'}")

        all_pass = gate_s and gate_m and gate_b
        n_pass = sum([gate_s, gate_m, gate_b])
        log("")
        log(f"v0.1.3.b moe-block smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/3 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
