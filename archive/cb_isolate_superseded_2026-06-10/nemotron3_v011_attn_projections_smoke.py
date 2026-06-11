#!/usr/bin/env python3
"""MM7 v0.1.1.a — L5 attention pre-norm + q/k/v projections.

Smoke gates for the FIRST attention-layer hooks: confirm our weight
upload + pre-norm + q/k/v_proj matmul chain matches HF at cos ≥ 0.999.
SDPA + o_proj + residual land at v0.1.1.b.

Gates (all per-position, cos vs numpy fp32 strict; HF as soft sanity):

  H — pre-norm output cos vs numpy fp32 (rms_norm(hidden_states[5], norm))
  Q — q_proj output    cos vs HF L5_attn_q_proj.npy
  K — k_proj output    cos vs HF L5_attn_k_proj.npy
  V — v_proj output    cos vs HF L5_attn_v_proj.npy

Mesh-replicated forward; no sharding yet (v0.5 perf concern).

REUSE: forks `nemotron3_v010_bootstrap_smoke.py`.

Run on the QuietBox (env-gates the layer upload):
    cd ~/tt-xla && NEMOTRON3_UPLOAD_LAYERS=5 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v011_attn_projections_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))

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


def numpy_rms_norm(x_np, w_np, eps):
    """Pure-numpy fp32 reference RMSNorm (Llama-style; no +1.0)."""
    x_fp32 = x_np.astype(np.float32)
    var = (x_fp32 ** 2).mean(axis=-1, keepdims=True)
    return x_fp32 * np.float32(1.0) / np.sqrt(var + eps) * w_np.astype(np.float32)


def main() -> int:
    import os
    if "NEMOTRON3_UPLOAD_LAYERS" not in os.environ:
        os.environ["NEMOTRON3_UPLOAD_LAYERS"] = str(L5)

    log("loading HF oracle artifacts…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [53, 5, 2688]
    q_hf = np.load(ORACLE_DIR / "L5_attn_q_proj.npy")          # [1, S, NQ*HD]
    k_hf = np.load(ORACLE_DIR / "L5_attn_k_proj.npy")
    v_hf = np.load(ORACLE_DIR / "L5_attn_v_proj.npy")
    # Squeeze the leading B=1 dim to match server output shape.
    if q_hf.ndim == 3 and q_hf.shape[0] == 1:
        q_hf, k_hf, v_hf = q_hf[0], k_hf[0], v_hf[0]
    log(f"  hidden_states[5] input shape: {hidden_states[L5].shape}")
    log(f"  q_hf {q_hf.shape}  k_hf {k_hf.shape}  v_hf {v_hf.shape}")

    log("bootstrapping (uploading L5)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        # Pre-norm weight — load from safetensors directly (the TT
        # readback path on a 1D tensor would scalar-out via [0]; loading
        # from disk is also closer to ground truth).
        from safetensors import safe_open
        snap = next(srv.SNAPSHOT_ROOT.glob("*"))
        norm_w_np = None
        for s in sorted(snap.glob("*.safetensors")):
            with safe_open(s, framework="pt") as f:
                if f"backbone.layers.{L5}.norm.weight" in f.keys():
                    norm_w_np = f.get_tensor(
                        f"backbone.layers.{L5}.norm.weight").float().numpy()
                    break
        assert norm_w_np is not None
        assert norm_w_np.shape == (srv.HIDDEN,)

        # Numpy fp32 reference for pre-norm
        h_input = hidden_states[L5]  # [S, HIDDEN]
        h_norm_np = numpy_rms_norm(h_input, norm_w_np, srv.EPS)
        log(f"  h_norm numpy ref shape: {h_norm_np.shape}")

        log("running TT pre-norm + q/k/v projections…")
        result = srv.attn_projections_only(state, h_input, L5)

        # ── Gate H — pre-norm output ────────────────────────────
        cos_h, mad_h = cos_and_mad(result["h_norm"], h_norm_np)
        log(f"Gate H pre-norm vs numpy fp32: cos={cos_h:.6f}  mad={mad_h:.4e}  "
            f"{'PASS ✓' if cos_h >= COS_GATE else 'FAIL ✗'}")

        # ── Gate Q/K/V — projections vs HF sub-hooks ───────────
        cos_q, mad_q = cos_and_mad(result["q"], q_hf)
        log(f"Gate Q q_proj vs HF: cos={cos_q:.6f}  mad={mad_q:.4e}  "
            f"{'PASS ✓' if cos_q >= COS_GATE else 'FAIL ✗'}")
        cos_k, mad_k = cos_and_mad(result["k"], k_hf)
        log(f"Gate K k_proj vs HF: cos={cos_k:.6f}  mad={mad_k:.4e}  "
            f"{'PASS ✓' if cos_k >= COS_GATE else 'FAIL ✗'}")
        cos_v, mad_v = cos_and_mad(result["v"], v_hf)
        log(f"Gate V v_proj vs HF: cos={cos_v:.6f}  mad={mad_v:.4e}  "
            f"{'PASS ✓' if cos_v >= COS_GATE else 'FAIL ✗'}")

        all_pass = all(g >= COS_GATE for g in [cos_h, cos_q, cos_k, cos_v])
        n_pass = sum(g >= COS_GATE for g in [cos_h, cos_q, cos_k, cos_v])
        log("")
        log(f"v0.1.1.a attn-projections smoke "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} ({n_pass}/4 gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
