#!/usr/bin/env python3
"""MM7 v0.3.1.c step 3b — first true single-token decode step.

Pipeline:
  1. reset_decode_state(state)
  2. Prefill: forward on 5-token prompt using attn_prefill_tt for the 6
     attention layers (writes K/V into paged caches at slots [0..4])
  3. Decode step: forward on the single predicted token (6993 = " Paris")
     using attn_decode_step_tt (paged_update_cache writes K/V at slot 5,
     paged_sdpa_decode reads from cache[0..5])
  4. Argmax of decode_step output vs HF gen[6] = 1063 (", ")

Gates:
  • Prefill argmax == 6993 ✓ (same as v0.3.1.c step 3a)
  • Decode step argmax == HF gen[6] (= 1063)

If gate 2 passes: full decode pipeline correct + ssm_state carry + paged
KV cache work end-to-end.
If gate 2 fails with bf16-drift-level error (off by few tokens): same
phenomenon as v0.3.1.a (7/8 with recovery); plumbing is right.
If gate 2 fails with garbage: real bug in decode-step plumbing.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v031c_decode_smoke'
"""
from __future__ import annotations

import json
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


def _forward_layers(state, h_tt, srv, ttnn, *, attn_fn_name: str):
    """Forward through all 52 layers using the named attn function
    (attn_prefill_tt or attn_decode_step_tt) for attention layers."""
    attn_fn = getattr(srv, attn_fn_name)
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next_tt = attn_fn(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        else:
            raise NotImplementedError(f"L{L} kind={kind!r}")
        ttnn.deallocate(h_tt)
        h_tt = h_next_tt
    return h_tt


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    prompt_ids = np.asarray(meta["prompt_ids"], dtype=np.int64)
    full_ids = np.asarray(meta["full_ids"], dtype=np.int64)
    hf_after_prompt = int(full_ids[len(prompt_ids)])  # 6993 (Paris)
    hf_after_decode = int(full_ids[len(prompt_ids) + 1])  # 1063 (', ')
    log(f"prompt ({len(prompt_ids)}): {prompt_ids.tolist()}")
    log(f"HF after prompt = {hf_after_prompt}  (' Paris')")
    log(f"HF after 1-step decode = {hf_after_decode}  (', ')")

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
        log("reset_decode_state(state)…")
        srv.reset_decode_state(state, B=1, log=log)

        # ── PREFILL ────────────────────────────────────────────────
        log("PREFILL: full forward on prompt via attn_prefill_tt…")
        t0 = time.time()
        h_np = srv.embed_lookup(state, prompt_ids[None, :])
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt = _forward_layers(state, h_tt, srv, ttnn,
                               attn_fn_name="attn_prefill_tt")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        if argmax_np.ndim == 2:
            argmax_np = argmax_np[0]
        prefill_argmax = int(argmax_np[-1])
        prefill_t = time.time() - t0
        log(f"  prefill in {prefill_t:.1f}s  argmax={prefill_argmax}  "
            f"{'PASS ✓' if prefill_argmax == hf_after_prompt else 'FAIL ✗'}")

        # ── DECODE STEP ────────────────────────────────────────────
        # The position of the new token is exactly len(prompt_ids) — that's
        # where attn_decode_step_tt writes K/V (slot 5). Update state.cur_pos
        # (host-side); device cur_pos_buf is updated inside attn_decode_step_tt.
        # NOTE: state.cur_pos was bumped by attn_prefill_tt to 30 (5 × 6 attn
        # layers), but for paged-cache semantics we need it = 5 (the position
        # to write the new token at). Override here.
        state.cur_pos = int(len(prompt_ids))
        log(f"DECODE: new-token position = {state.cur_pos}; "
            f"input token = {prefill_argmax}")
        t0 = time.time()
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[prefill_argmax]], dtype=np.int64),
        )  # [1, 1, HIDDEN]
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np_dec.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt = _forward_layers(state, h_tt, srv, ttnn,
                               attn_fn_name="attn_decode_step_tt")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        if argmax_np.ndim == 2:
            argmax_np = argmax_np[0]
        decode_argmax = int(argmax_np[-1])
        decode_t = time.time() - t0
        log(f"  decode step in {decode_t:.1f}s  argmax={decode_argmax}  "
            f"(HF = {hf_after_decode})  "
            f"{'PASS ✓' if decode_argmax == hf_after_decode else 'FAIL ✗'}")

        log("")
        ok = (prefill_argmax == hf_after_prompt
              and decode_argmax == hf_after_decode)
        log(f"v0.3.1.c step 3b {'PASS ✓' if ok else 'FAIL ✗'}")
        return 0 if ok else 1
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
