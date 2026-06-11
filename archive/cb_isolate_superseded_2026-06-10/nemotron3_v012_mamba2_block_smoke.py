#!/usr/bin/env python3
"""MM7 v0.1.2.c — L0 Mamba2 full block forward.

Smoke gates the END-TO-END L0 Mamba2 block vs the HF oracle hooks:

  N — norm_out      cos vs HF L0_norm.npy            (MambaRMSNormGated)
  O — o_proj_out    cos vs HF L0_out_proj.npy
  M — mixer_out     cos vs HF L0_mamba2_mixer_out.npy (= o_proj_out here)
  B — block_out     cos vs HF hidden_states[1]       (post-residual)

Plus regression gates from v0.1.2.b (conv1d) for completeness.

REUSE: forks v0.1.2.b smoke.

Run on the QuietBox:
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=0 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v012_mamba2_block_smoke.py
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
L0 = 0


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
        os.environ["NEMOTRON3_UPLOAD_LAYERS"] = str(L0)

    log("loading HF oracle artifacts…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [53, 5, 2688]
    conv1d_hf = np.load(ORACLE_DIR / "L0_conv1d.npy")
    norm_hf = np.load(ORACLE_DIR / "L0_norm.npy")
    o_proj_hf = np.load(ORACLE_DIR / "L0_out_proj.npy")
    mixer_out_hf = np.load(ORACLE_DIR / "L0_mamba2_mixer_out.npy")
    for arr_name, arr in [("norm", norm_hf), ("o_proj", o_proj_hf),
                            ("mixer_out", mixer_out_hf)]:
        log(f"  {arr_name}_hf shape: {arr.shape}")
    log(f"  hidden_states[0] (input) shape: {hidden_states[L0].shape}")
    log(f"  hidden_states[1] (target) shape: {hidden_states[L0+1].shape}")

    log("bootstrapping (uploading L0)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        h_input = hidden_states[L0]
        log("running TT full L0 Mamba2 forward (pre-norm + in_proj + conv1d "
            "+ silu + split + SSD loop + norm-gated + out_proj + residual)…")
        t_fwd = time.time()
        res = srv.mamba2_block_eager(state, h_input, L0)
        log(f"  forward in {time.time() - t_fwd:.1f}s")
        for k in ["conv1d_out", "y_post_ssd", "norm_out", "o_proj_out", "block_out"]:
            log(f"  {k} shape: {res[k].shape}")

        # Regression — conv1d output (squeeze to [B, C, S+pad] for HF compare).
        conv_tt = res["conv1d_out"]
        if conv_tt.ndim == 4 and conv_tt.shape[1] == 1:
            conv_tt = conv_tt.squeeze(1)
        if conv_tt.ndim == 3 and conv_tt.shape[-1] == srv.CONV_DIM_M:
            conv_tt = conv_tt.transpose(0, 2, 1)
        cos_c, _ = cos_and_mad(conv_tt, conv1d_hf)
        log(f"  (regression) conv1d cos = {cos_c:.6f}")

        # Gate N — norm_out vs HF L0_norm
        n_target = norm_hf[0] if norm_hf.ndim == 3 else norm_hf
        n_pred = res["norm_out"]
        if n_pred.ndim == 3 and n_target.ndim == 2:
            n_pred = n_pred[0]
        cos_n, mad_n = cos_and_mad(n_pred, n_target)
        gate_n = cos_n >= COS_GATE
        log(f"Gate N norm_out vs HF L0_norm: cos={cos_n:.6f}  mad={mad_n:.4e}  "
            f"{'PASS ✓' if gate_n else 'FAIL ✗'}")

        # Gate O — o_proj_out vs HF L0_out_proj
        o_target = o_proj_hf[0] if o_proj_hf.ndim == 3 else o_proj_hf
        o_pred = res["o_proj_out"]
        if o_pred.ndim == 3 and o_target.ndim == 2:
            o_pred = o_pred[0]
        cos_o, mad_o = cos_and_mad(o_pred, o_target)
        gate_o = cos_o >= COS_GATE
        log(f"Gate O o_proj_out vs HF L0_out_proj: cos={cos_o:.6f}  mad={mad_o:.4e}  "
            f"{'PASS ✓' if gate_o else 'FAIL ✗'}")

        # Gate M — mixer_out vs HF L0_mamba2_mixer_out
        # The HF hook captures the FULL mixer output (post-out_proj, pre-residual),
        # which for an attention/Mamba2 layer = o_proj output.
        m_target = mixer_out_hf[0] if mixer_out_hf.ndim == 3 else mixer_out_hf
        cos_m, mad_m = cos_and_mad(o_pred, m_target)
        gate_m = cos_m >= COS_GATE
        log(f"Gate M mixer_out vs HF L0_mixer_out: cos={cos_m:.6f}  mad={mad_m:.4e}  "
            f"{'PASS ✓' if gate_m else 'FAIL ✗'}")

        # Gate B — block_out vs hidden_states[1]
        b_target = hidden_states[L0 + 1]
        b_pred = res["block_out"]
        if b_pred.ndim == 3 and b_target.ndim == 2:
            b_pred = b_pred[0]
        cos_b, mad_b = cos_and_mad(b_pred, b_target)
        gate_b = cos_b >= COS_GATE
        log(f"Gate B block_out vs hidden_states[1]: cos={cos_b:.6f}  "
            f"mad={mad_b:.4e}  {'PASS ✓' if gate_b else 'FAIL ✗'}")

        all_pass = gate_n and gate_o and gate_m and gate_b
        n_pass = sum([gate_n, gate_o, gate_m, gate_b])
        log("")
        log(f"v0.1.2.c full-mamba2-block smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/4 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
