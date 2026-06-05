#!/usr/bin/env python3
"""MM7 v0.2.5 — on-device tensor-flow regression smoke.

Same gate as v0.2.b (TT argmax_last == HF argmax_last == 6993), but
through the on-device `_tt` block variants. h_tt persists across all
52 blocks — no inter-block host round trips.

Pipeline:
  embed → ttnn.from_torch (replicated) → h_tt
  for L in 0..52:
      upload_one_layer(L)
      h_tt = block_eager_tt(state, h_tt, L)   # NEW: tt-flow
      deallocate_layer(L)
  final_norm + lm_head + argmax(last)

Regression gate: argmax_last == 6993 (matches v0.2.b).
Perf gate: per-step time should drop substantially vs v0.2.b's 77s
(target ≥5× faster; theoretical ceiling = remove 52 round trips ×
~22 KB each ≈ ~1-2s recovered).
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle…")
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    hf_argmax_last = int(np.load(ORACLE_DIR / "argmax.npy").flatten()[-1])
    log(f"  prompt_ids {prompt_ids.shape}  HF argmax_last={hf_argmax_last}")

    log("bootstrap (top-level only — streaming layers)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")
    if not state.per_layer_tt:
        state.per_layer_tt = [None] * N_LAYERS

    try:
        # Embed → h_np → upload as h_tt (replicated, TILE)
        log("embed lookup → upload h_tt (replicated)…")
        h_np = srv.embed_lookup(state, prompt_ids[None, :])  # [1, S, HIDDEN]
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

        # 52-layer stream, ON DEVICE
        log("52-layer tt-flow stream…")
        t_chain = time.time()
        for L in range(N_LAYERS):
            kind = state.layer_types[L]
            t_layer = time.time()
            state.per_layer_tt[L] = srv.upload_one_layer(state, L, log)
            t_upload = time.time() - t_layer
            t_fwd = time.time()
            if kind == "attention":
                h_next_tt = srv.attn_block_eager_tt(state, h_tt, L)
            elif kind == "mamba2":
                h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
            elif kind == "moe":
                h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
            else:
                raise NotImplementedError(f"L{L} kind={kind!r}")
            t_fwd = time.time() - t_fwd
            ttnn.deallocate(h_tt)
            h_tt = h_next_tt
            srv.deallocate_layer(state, L)
            log(f"L{L:>2d} ({kind:>9s}) upload={t_upload:.1f}s fwd={t_fwd:.2f}s "
                f"total={time.time() - t_layer:.1f}s")
        t_chain = time.time() - t_chain
        log(f"  52-layer tt-flow stream in {t_chain:.1f}s")

        # Final norm + lm_head + argmax (still numpy-readback at the boundary)
        log("final_norm + lm_head + argmax…")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        fn_cos, fn_mad = cos_and_mad(h_final[0], hidden_states[-1])
        log(f"  final_norm cos={fn_cos:.6f} mad={fn_mad:.4e}")
        logits_np, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        if argmax_np.ndim == 2:
            argmax_np = argmax_np[0]
        tt_last_argmax = int(argmax_np[-1])
        log(f"  TT argmax_last={tt_last_argmax}  HF argmax_last={hf_argmax_last}")
        ok = tt_last_argmax == hf_argmax_last
        log(f"  Gate: {'PASS ✓' if ok else 'FAIL ✗'}")
        log(f"  Total chain time: {t_chain:.1f}s (v0.2.b was 77.7s)")
        return 0 if ok else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
