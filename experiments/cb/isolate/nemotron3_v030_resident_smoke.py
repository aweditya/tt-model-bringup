#!/usr/bin/env python3
"""MM7 v0.3.0 — all-layers-resident forward smoke.

Bootstraps all 52 layers up-front (mirrors server_35b_ttnn.py:1818
pattern) and runs ONE forward without per-layer upload/deallocate.
Verifies the v0.2 argmax regression (TT == HF == 6993) and measures
per-iter time on the resident path.

P150 has 32 GB DRAM/chip; full Nemotron-3 Nano load ≈ 21 GB/chip — fits.
The streaming pattern from v0.2.b was over-engineered for a non-problem;
this is the correct architecture for v0.3+.

Expected timing:
  • Bootstrap:  ~3 min cold (52 layers × ~3s each + top-level)
  • Forward:    ~13s (just the TT compute, no upload cost)

Run on qb1:
    NEMOTRON3_UPLOAD_LAYERS=all NEMOTRON3_MOE_MODE=ep \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v030_resident_smoke.py
"""
from __future__ import annotations

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
LOG_FILE = LOG_DIR / f"v030_resident_{time.strftime('%Y%m%d_%H%M%S')}.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass  # don't crash the smoke on log-write failure


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def run_forward(state, prompt_ids, srv, ttnn) -> tuple[int, float]:
    """Run one all-resident forward → final_norm → lm_head → argmax."""
    t0 = time.time()
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        # Layer weights ALREADY RESIDENT — no upload or dealloc per layer.
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
    return int(argmax_np[-1]), time.time() - t0


def main() -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle…")
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_argmax_last = int(np.load(ORACLE_DIR / "argmax.npy").flatten()[-1])
    log(f"  prompt_ids {prompt_ids.shape}  HF argmax_last={hf_argmax_last}")

    log("bootstrap (ALL 52 layers resident — expect ~2-3 min)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    t_boot = time.time() - t0
    log(f"  bootstrap in {t_boot:.1f}s")

    try:
        # Iter 1 — first forward (JIT compilation included)
        log("=" * 60)
        log("ITER 1 — first resident forward (JIT cold)")
        log("=" * 60)
        argmax1, t1 = run_forward(state, prompt_ids, srv, ttnn)
        log(f"  argmax_last={argmax1}  time={t1:.1f}s")

        # Iter 2 — warm (JIT cached, pure compute)
        log("=" * 60)
        log("ITER 2 — warm (JIT cached)")
        log("=" * 60)
        argmax2, t2 = run_forward(state, prompt_ids, srv, ttnn)
        log(f"  argmax_last={argmax2}  time={t2:.1f}s")

        log("")
        log(f"Summary:")
        log(f"  bootstrap={t_boot:.1f}s  iter1={t1:.1f}s  iter2={t2:.1f}s")
        log(f"  HF argmax_last={hf_argmax_last}")
        log(f"  Iter1 argmax: {'PASS ✓' if argmax1 == hf_argmax_last else 'FAIL ✗'}")
        log(f"  Iter2 argmax: {'PASS ✓' if argmax2 == hf_argmax_last else 'FAIL ✗'}")
        log(f"  v0.2.5 streaming was 76s/iter; v0.2.6 warm was 61.8s/iter")
        log(f"  Speedup: vs v0.2.6 warm = {61.8/t2:.2f}×")
        all_ok = argmax1 == hf_argmax_last and argmax2 == hf_argmax_last
        log(f"v0.3.0 {'PASS ✓' if all_ok else 'FAIL ✗'}")
        return 0 if all_ok else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
