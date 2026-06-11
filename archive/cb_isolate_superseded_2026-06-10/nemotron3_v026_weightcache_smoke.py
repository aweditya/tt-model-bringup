#!/usr/bin/env python3
"""MM7 v0.2.6 — host weight cache smoke (cold vs warm).

Runs the v0.2.5 on-device forward TWICE in the same process:
  Iter 1 (cold)  — first reads of safetensors files; populates cache
  Iter 2 (warm)  — all weights served from RAM

Gates:
  • argmax_last (both iters)  == HF == 6993
  • iter 2 time ≤ 25 s  (target 4× warm speedup vs iter 1's ~80s)

REUSE: forks v0.2.5 smoke; just loops the forward twice.
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
WARM_TARGET = 25.0  # seconds


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_forward(state, prompt_ids, srv, ttnn) -> tuple[int, float]:
    """Run one full 52-layer streaming forward + final_norm + lm_head.
    Returns (argmax_last, elapsed_sec).
    """
    t0 = time.time()
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        state.per_layer_tt[L] = srv.upload_one_layer(state, L, lambda *_a, **_k: None)
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
        srv.deallocate_layer(state, L)
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
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle…")
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_argmax_last = int(np.load(ORACLE_DIR / "argmax.npy").flatten()[-1])
    log(f"  prompt_ids {prompt_ids.shape}  HF argmax_last={hf_argmax_last}")

    log("bootstrap (top-level only)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")
    if not state.per_layer_tt:
        state.per_layer_tt = [None] * N_LAYERS

    try:
        # Iter 1 — cold
        log("=" * 60)
        log("ITER 1 (cold — first disk reads, populates cache)")
        log("=" * 60)
        argmax1, t1 = run_forward(state, prompt_ids, srv, ttnn)
        cache_keys_after_cold = len(state.weight_np_cache)
        log(f"  argmax_last={argmax1}  time={t1:.1f}s  "
            f"cache_keys={cache_keys_after_cold}")

        # Iter 2 — warm
        log("=" * 60)
        log("ITER 2 (warm — all weights from RAM cache)")
        log("=" * 60)
        argmax2, t2 = run_forward(state, prompt_ids, srv, ttnn)
        log(f"  argmax_last={argmax2}  time={t2:.1f}s")

        log("")
        log(f"Summary: iter1={t1:.1f}s  iter2={t2:.1f}s  "
            f"speedup={t1/t2:.2f}×")
        log(f"  HF argmax_last={hf_argmax_last}")
        log(f"  Iter1 argmax match: {'PASS ✓' if argmax1 == hf_argmax_last else 'FAIL ✗'}")
        log(f"  Iter2 argmax match: {'PASS ✓' if argmax2 == hf_argmax_last else 'FAIL ✗'}")
        log(f"  Iter2 ≤ {WARM_TARGET}s target: "
            f"{'PASS ✓' if t2 <= WARM_TARGET else 'FAIL ✗'}")
        all_ok = (argmax1 == hf_argmax_last and
                  argmax2 == hf_argmax_last and
                  t2 <= WARM_TARGET)
        log(f"v0.2.6 {'PASS ✓' if all_ok else 'FAIL ✗'}")
        return 0 if all_ok else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
