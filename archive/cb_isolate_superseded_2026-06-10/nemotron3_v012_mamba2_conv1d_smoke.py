#!/usr/bin/env python3
"""MM7 v0.1.2.b — L0 Mamba2 pre-norm + in_proj + split + conv1d.

Adds conv1d to the v0.1.2.a chain (depth-wise, K=4, sym pad=3 via
`ttnn.conv1d` with groups=conv_dim). The HF L0_conv1d hook captures
the FULL conv output [1, 6144, 8] (pre-causal-slice).

Gates:
  H — pre-norm    cos vs numpy fp32
  I — in_proj     cos vs HF L0_in_proj
  C — conv1d_out  cos vs HF L0_conv1d (squeeze/transpose to match)

REUSE: forks v0.1.2.a smoke.

Run on the QuietBox:
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=0 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v012_mamba2_conv1d_smoke.py
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


def numpy_rms_norm(x_np, w_np, eps):
    x_fp32 = x_np.astype(np.float32)
    var = (x_fp32 ** 2).mean(axis=-1, keepdims=True)
    return x_fp32 * np.float32(1.0) / np.sqrt(var + eps) * w_np.astype(np.float32)


def main() -> int:
    import os
    if "NEMOTRON3_UPLOAD_LAYERS" not in os.environ:
        os.environ["NEMOTRON3_UPLOAD_LAYERS"] = str(L0)

    log("loading HF oracle artifacts…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [53, 5, 2688]
    in_proj_hf = np.load(ORACLE_DIR / "L0_in_proj.npy")
    conv1d_hf = np.load(ORACLE_DIR / "L0_conv1d.npy")  # [1, 6144, 8]
    if in_proj_hf.ndim == 3 and in_proj_hf.shape[0] == 1:
        in_proj_hf = in_proj_hf[0]
    log(f"  hidden_states[0] shape: {hidden_states[L0].shape}")
    log(f"  in_proj_hf shape: {in_proj_hf.shape}")
    log(f"  conv1d_hf shape: {conv1d_hf.shape}  (matches [B, conv_dim, S+pad])")

    log("bootstrapping (uploading L0)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        from safetensors import safe_open
        snap = next(srv.SNAPSHOT_ROOT.glob("*"))
        norm_w_np = None
        for s in sorted(snap.glob("*.safetensors")):
            with safe_open(s, framework="pt") as f:
                if f"backbone.layers.{L0}.norm.weight" in f.keys():
                    norm_w_np = f.get_tensor(
                        f"backbone.layers.{L0}.norm.weight").float().numpy()
                    break
        h_input = hidden_states[L0]
        h_norm_np = numpy_rms_norm(h_input, norm_w_np, srv.EPS)

        log("running TT pre-norm + in_proj + split + conv1d…")
        res = srv.mamba2_in_proj_split_conv1d(state, h_input, L0)
        log(f"  h_norm:        {res['h_norm'].shape}")
        log(f"  in_proj_out:   {res['in_proj_out'].shape}")
        log(f"  z / x_BC / dt: {res['z'].shape} / {res['x_BC'].shape} / {res['dt'].shape}")
        log(f"  conv1d_out:    {res['conv1d_out'].shape}")

        # Gate H
        cos_h, mad_h = cos_and_mad(res["h_norm"], h_norm_np)
        gate_h = cos_h >= COS_GATE
        log(f"Gate H pre-norm vs numpy fp32: cos={cos_h:.6f}  mad={mad_h:.4e}  "
            f"{'PASS ✓' if gate_h else 'FAIL ✗'}")

        # Gate I
        cos_i, mad_i = cos_and_mad(res["in_proj_out"], in_proj_hf)
        gate_i = cos_i >= COS_GATE
        log(f"Gate I in_proj vs HF: cos={cos_i:.6f}  mad={mad_i:.4e}  "
            f"{'PASS ✓' if gate_i else 'FAIL ✗'}")

        # Gate C — conv1d output. HF stores [1, 6144, 8]; our TT output
        # from ttnn.conv1d in NHWC is [1, 1, 8, 6144] (or [B, S_out, C]
        # after reshape). Squeeze + transpose to compare.
        conv1d_tt = res["conv1d_out"]
        log(f"  conv1d_tt raw shape: {conv1d_tt.shape}")
        # Normalize to [B, C, S_out] for the HF comparison.
        if conv1d_tt.ndim == 4 and conv1d_tt.shape[1] == 1:
            # [B, 1, S_out, C] → [B, S_out, C]
            conv1d_tt = conv1d_tt.squeeze(1)
        if conv1d_tt.ndim == 3 and conv1d_tt.shape[-1] == 6144:
            # [B, S_out, C] → [B, C, S_out]
            conv1d_tt = conv1d_tt.transpose(0, 2, 1)
        log(f"  conv1d_tt for compare: {conv1d_tt.shape}")
        cos_c, mad_c = cos_and_mad(conv1d_tt, conv1d_hf)
        gate_c = cos_c >= COS_GATE
        log(f"Gate C conv1d vs HF L0_conv1d: cos={cos_c:.6f}  mad={mad_c:.4e}  "
            f"{'PASS ✓' if gate_c else 'FAIL ✗'}")

        all_pass = gate_h and gate_i and gate_c
        n_pass = sum([gate_h, gate_i, gate_c])
        log("")
        log(f"v0.1.2.b mamba2-conv1d smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/3 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
