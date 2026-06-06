#!/usr/bin/env python3
"""MM7 v0.3.3 — N-step decode chain via state carry.

Builds on v0.3.1's verified single-step decode. Tests that state
(ssm_state, conv_state, KV cache, cur_pos) carries correctly across
N sequential decode_step calls.

Pipeline:
  1. Reset state.
  2. PREFILL on 5-token prompt → next_token (= 6993).
  3. Loop for N=8 steps:
       a. Feed prev_token through the full 52-layer single-token
          decode (attn_decode_step_tt + mamba2 with state +
          moe_ep with no state).
       b. Take argmax; compare to HF gen[5+step+1].
       c. Advance state.cur_pos by 1.
  4. Report per-step pass/fail; verdict matches v0.3.1.a baseline
     (≥7/8 expected since bf16 drift floor is identical).

Gates:
  • PREFILL argmax = HF[5] = 6993
  • DECODE chain: TT[5+i+1] == HF[5+i+1] for at least 7/8 steps
  • No crashes / TT_FATAL across the 8 steps

If all 8 steps PASS: constant-time decode is fully validated; v0.4
trace is unblocked.
If 7/8 (drift recovery like v0.3.1.a): expected behavior, also unblocks
v0.4 (drift is bf16 floor, not state-carry bug).
If <7/8 or any state-carry-related crash: bug in state advance.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v033_nstep_chain_smoke'
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
N_DECODE_STEPS = 8  # matches HF gen=8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _forward_layers(state, h_tt, srv, ttnn, *, attn_fn_name: str):
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
    prompt_len = len(prompt_ids)
    log(f"prompt ({prompt_len}): {prompt_ids.tolist()}")
    log(f"HF full sequence ({len(full_ids)}): {full_ids.tolist()}")
    log(f"  → testing chain of {N_DECODE_STEPS} decode steps after prefill")

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
        log("reset_decode_state…")
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
        prefill_argmax = int(argmax_np.flatten()[-1])
        prefill_t = time.time() - t0
        hf_first = int(full_ids[prompt_len])
        prefill_ok = prefill_argmax == hf_first
        log(f"  prefill in {prefill_t:.1f}s  TT={prefill_argmax}  "
            f"HF={hf_first}  {'PASS ✓' if prefill_ok else 'FAIL ✗'}")

        # Override cur_pos: prefill bumps it to 30 (5×6 attn);
        # set to len(prompt_ids) for paged-cache semantics.
        state.cur_pos = prompt_len

        # ── DECODE CHAIN ───────────────────────────────────────────
        log(f"DECODE CHAIN: {N_DECODE_STEPS} steps, state-carried…")
        tt_chain = []
        step_results = []
        prev_token = prefill_argmax  # = 6993 = HF[5]
        for step in range(N_DECODE_STEPS):
            target_pos = prompt_len + step + 1  # position of THIS step's output
            if target_pos >= len(full_ids):
                log(f"step {step}: out of HF reference range, stopping")
                break
            hf_expected = int(full_ids[target_pos])
            t0 = time.time()
            # cur_pos_buf is now caller-updated (moved out of attn_decode_step_tt
            # for trace compatibility — see commit history around 2026-06-06).
            srv.update_cur_pos_buf(state, int(state.cur_pos))
            # Embed prev_token
            h_np_dec = srv.embed_lookup(
                state, np.asarray([[prev_token]], dtype=np.int64),
            )
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
            tt_next = int(argmax_np.flatten()[-1])
            elapsed = time.time() - t0
            ok = tt_next == hf_expected
            step_results.append(ok)
            tt_chain.append(tt_next)
            log(f"step {step}: cur_pos={state.cur_pos}  TT={tt_next}  "
                f"HF={hf_expected}  {'PASS ✓' if ok else 'FAIL ✗'}  "
                f"({elapsed:.1f}s)")
            # Advance state for next iteration
            state.cur_pos += 1
            prev_token = tt_next

        # ── SUMMARY ────────────────────────────────────────────────
        log("")
        log("=" * 60)
        n_pass = sum(step_results)
        n_total = len(step_results)
        log(f"PREFILL: {'PASS ✓' if prefill_ok else 'FAIL ✗'}")
        log(f"DECODE CHAIN: {n_pass}/{n_total} steps PASS")
        log(f"  TT chain: {tt_chain}")
        log(f"  HF chain: {full_ids[prompt_len + 1 : prompt_len + 1 + n_total].tolist()}")
        # Acceptance: prefill must pass AND chain must be at least 7/8 like v0.3.1.a
        overall_ok = prefill_ok and (n_pass >= max(1, n_total - 1))
        log("")
        log(f"v0.3.3 N-step chain {'PASS ✓' if overall_ok else 'FAIL ✗'} "
            f"(prefill + chain ≥ {max(1, n_total - 1)}/{n_total})")
        return 0 if overall_ok else 1
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
