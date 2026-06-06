#!/usr/bin/env python3
"""MM7 v0.3.1.c step 3c — per-layer teacher-forced decode ladder.

Same correctness method we used for 35B and Gemma 4: at each layer L,
feed HF's hidden_states[L][decode_pos] as input (teacher forcing) and
compare TT output to HF hidden_states[L+1][decode_pos]. If a layer
fails, drift after that layer is not its fault — the locus is THIS
layer's compute.

Pipeline:
  1. Reset state (zero ssm_state per mamba2 layer + clear KV caches).
  2. PREFILL: full 5-token forward through all 52 layers using
     attn_prefill_tt for attention (populates state.ssm_state_np[L]
     and writes K/V to paged caches at slots [0..4]).
  3. LADDER at decode_pos = 5:
       for L in 0..51:
         h_in = HF hidden_states[L][decode_pos]   (teacher force)
         h_out = TT layer_L_decode(h_in)
         report cos(h_out, HF hidden_states[L+1][decode_pos])

Gate: per-layer cos ≥ 0.99 OR identifying the first divergence point.

NOTE: this is a DIAGNOSTIC, not a regression — we expect failures
to surface the bug location.

REUSE: forks the per-layer ladder pattern from
`cb35_per_layer_drift_pos1.py` + `gm4_per_layer_drift_pos1.py`.
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
DECODE_POS = 5  # position 5 = first decode (HF: prompt → 6993 → decode at 5 → 1063)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle…")
    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    prompt_ids = np.asarray(meta["prompt_ids"], dtype=np.int64)
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")  # [53, 13, 2688]
    assert hidden_states.shape == (N_LAYERS + 1, 13, 2688), \
        f"unexpected hidden_states shape {hidden_states.shape}"
    log(f"  prompt ({len(prompt_ids)}): {prompt_ids.tolist()}")
    log(f"  HF hidden_states[*, {DECODE_POS}, :] = teacher-force inputs")
    log(f"  ladder at decode_pos = {DECODE_POS}")

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

    HIDDEN = srv.HIDDEN
    try:
        log("reset_decode_state…")
        srv.reset_decode_state(state, B=1, log=log)

        # ── PREFILL: populate state.ssm_state_np[L] + KV caches ─────
        log("PREFILL: full 5-token forward via attn_prefill_tt…")
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
                h_next_tt = srv.attn_prefill_tt(state, h_tt, L)
            elif kind == "mamba2":
                h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
            elif kind == "moe":
                h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
            ttnn.deallocate(h_tt)
            h_tt = h_next_tt
        ttnn.deallocate(h_tt)
        log(f"  prefill in {time.time() - t0:.1f}s")

        # Override host-side cur_pos: prefill bumped it to 30 (5×6 attn);
        # for decode we need it to equal the seq position of the new token = 5.
        state.cur_pos = DECODE_POS

        # ── LADDER ────────────────────────────────────────────────
        log(f"LADDER at decode_pos = {DECODE_POS} "
            f"(teacher-forcing each layer's input from HF)…")
        results = []
        for L in range(N_LAYERS):
            kind = state.layer_types[L]
            hf_in = hidden_states[L, DECODE_POS, :]    # [HIDDEN]
            hf_out_target = hidden_states[L + 1, DECODE_POS, :]
            # Upload teacher-forced input as [1, 1, HIDDEN].
            h_in_tt = ttnn.from_torch(
                torch.from_numpy(hf_in.reshape(1, 1, HIDDEN).astype(np.float32)),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            )
            t0 = time.time()
            try:
                if kind == "attention":
                    # cur_pos_buf now caller-updated (was inline in
                    # attn_decode_step_tt; moved out for trace compat).
                    srv.update_cur_pos_buf(state, int(state.cur_pos))
                    h_out_tt = srv.attn_decode_step_tt(state, h_in_tt, L)
                elif kind == "mamba2":
                    h_out_tt = srv.mamba2_block_eager_tt(state, h_in_tt, L)
                elif kind == "moe":
                    h_out_tt = srv.moe_block_eager_ep_tt(state, h_in_tt, L)
                else:
                    raise NotImplementedError(f"L{L} {kind!r}")
            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - t0
                log(f"L{L:>2d} ({kind:>9s}) FAIL: {type(e).__name__}: {e} "
                    f"({elapsed:.1f}s)")
                ttnn.deallocate(h_in_tt)
                results.append((L, kind, None, None, str(e)))
                continue
            elapsed = time.time() - t0
            h_out_np = ttnn.to_torch(
                h_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )[:1].float().numpy().reshape(-1)
            ttnn.deallocate(h_out_tt)
            ttnn.deallocate(h_in_tt)
            cos, mad = cos_and_mad(h_out_np, hf_out_target)
            marker = "PASS ✓" if cos >= 0.99 else f"FAIL ✗ (cos<0.99)"
            log(f"L{L:>2d} ({kind:>9s}) cos={cos:.6f}  mad={mad:.3e}  "
                f"{marker}  ({elapsed:.1f}s)")
            results.append((L, kind, cos, mad, None))

        # ── SUMMARY ────────────────────────────────────────────────
        log("")
        log("=" * 60)
        log("SUMMARY (first divergence + per-kind histogram)")
        log("=" * 60)
        first_fail = None
        passed = {"attention": 0, "mamba2": 0, "moe": 0}
        failed = {"attention": 0, "mamba2": 0, "moe": 0}
        errored = []
        for L, kind, cos, mad, err in results:
            if err is not None:
                errored.append((L, kind, err))
                continue
            if cos < 0.99:
                if first_fail is None:
                    first_fail = (L, kind, cos, mad)
                failed[kind] += 1
            else:
                passed[kind] += 1
        for kind in ("mamba2", "moe", "attention"):
            log(f"  {kind:>9s}: {passed[kind]:>2d} PASS / {failed[kind]:>2d} FAIL")
        if errored:
            log(f"  errored layers: {len(errored)}")
        if first_fail is not None:
            L, kind, cos, mad = first_fail
            log(f"\n  FIRST DIVERGENCE: L{L} ({kind})  cos={cos:.6f}  mad={mad:.3e}")
            log(f"  → drill into this layer's sub-ops next")
        else:
            log(f"\n  all layers cos ≥ 0.99 ✓")
        log("")
        return 0
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
