#!/usr/bin/env python3
"""MM7 v0.3.1.c step 3a — attn_prefill_tt regression smoke.

Runs the full 52-layer forward but uses `attn_prefill_tt` (writes K/V
into the paged cache via Gemma 4 two-call) for the 6 attention layers
instead of `attn_block_eager_tt`. Forward result should be IDENTICAL —
the cache write is a side effect. If the argmax changes, that's a real
regression. If it stays 6993, the prefill path is correct + the cache
is populated for step 3b (decode).

Gate: argmax_last == HF == 6993

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v031c_prefill_smoke'
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


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    import json
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    prompt_ids = np.asarray(meta["prompt_ids"], dtype=np.int64)
    hf_argmax_last = 6993  # from v0.2.b oracle
    log(f"prompt_ids ({len(prompt_ids)}): {prompt_ids.tolist()}")
    log(f"HF argmax_last = {hf_argmax_last}")

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
        log("calling reset_decode_state(state)…")
        srv.reset_decode_state(state, B=1, log=log)
        log(f"  cur_pos before prefill = {state.cur_pos}")

        log("running full forward with attn_prefill_tt for attention layers…")
        t0 = time.time()
        h_np = srv.embed_lookup(state, prompt_ids[None, :])
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        for L in range(N_LAYERS):
            kind = state.layer_types[L]
            if kind == "attention":
                h_next_tt = srv.attn_prefill_tt(state, h_tt, L)  # ← NEW
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
        tt_last_argmax = int(argmax_np[-1])
        forward_t = time.time() - t0

        log(f"  forward in {forward_t:.1f}s")
        log(f"  cur_pos after prefill = {state.cur_pos}  "
            f"(expected {6 * len(prompt_ids)} for 6 attn layers × S={len(prompt_ids)})")
        log(f"  TT argmax_last={tt_last_argmax}  HF argmax_last={hf_argmax_last}")
        ok = tt_last_argmax == hf_argmax_last
        log(f"  Gate (argmax match): {'PASS ✓' if ok else 'FAIL ✗'}")
        return 0 if ok else 1
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
