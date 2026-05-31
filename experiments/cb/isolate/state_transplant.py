#!/usr/bin/env python3
"""S2.3 isolation — cb_prefill_transplant copies production state into CB slot.

Runs S1a (forward_prefill_chunked_tp) + S1b (chunked DN inside it) on a fresh
state buffers, captures the post-prefill production tensors, runs
`cb_prefill_transplant(state, slot_s=0, L=len(prompt))`, then reads CB slot-0
back from device and compares per-layer:

  - KV: cb_kv[li]['kc']/['vc'] blocks [0:n_used_blocks] vs production attn[*]
        blocks [0:n_used_blocks] — bit-equal (no math, just a tensor copy).
  - DN ssm: cb_dn[li]['ssm'][slot=0] vs production dn['ssm'][0] — bit-equal.
  - DN conv_cols: cb_dn[li]['conv_cols'][k][slot=0] vs production
                  dn['conv_st'][:, k] — bit-equal.

We expect cosine == 1.0 / max-abs-diff == 0 on every layer; any miss means the
transplant code mis-maps a dim or a shard. End-to-end (decode-after-transplant)
gate is S2.4.

Run on qb1 (from repo root):
  make run PY=experiments/cb/isolate/state_transplant.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

from _runner import bootstrap_27b_cb, log  # noqa: E402
import server_tp_cb as cb                    # noqa: E402


def _cos(a, b):
    a = a.astype(np.float64).reshape(-1); b = b.astype(np.float64).reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 1.0


def main():
    import ttnn
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=64, help="prompt token count")
    ap.add_argument("--slot", type=int, default=0, help="CB slot to transplant into")
    ap.add_argument("--cos-threshold", type=float, default=0.9999)
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state, base = bootstrap_27b_cb()
    tok = state.tok

    prompt_text = ("The capital of France is the city of Paris, which has long been a "
                   "center of art, science, philosophy, and political history in Europe, "
                   "drawing scholars and travelers from every corner of the wider world "
                   "for many centuries of recorded human civilization and culture, "
                   "blending tradition and reinvention across countless generations.")
    ids = tok.encode(prompt_text)[:args.length]
    L = len(ids)
    log(f"prompt L={L} tokens")

    # 1. Production prefill — populates dn['ssm'], dn['conv_st'], attn['kc'/'vc'].
    base._reset_state_buffers(state)
    base.forward_prefill_chunked_tp(state, ids, capture_logits=False)
    ttnn.synchronize_device(state.mesh)
    log("S1a prefill done; production state populated")

    # Snapshot post-prefill production state, layer-by-layer.
    from full_layer_tp_probe import NV_PER_CHIP, CONV_DIM_CHIP
    N_V_TOTAL = NV_PER_CHIP * 4
    CONV_DIM = CONV_DIM_CHIP * 4
    cfg = state.cfg
    BLOCK_SIZE = base.BLOCK_SIZE
    n_used = (L + BLOCK_SIZE - 1) // BLOCK_SIZE
    prod_snapshot = {}
    for li, layer in enumerate(state.layers):
        if layer['type'] == 'full_attention':
            kc = ttnn.to_torch(layer['attn']['kc'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            vc = ttnn.to_torch(layer['attn']['vc'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            prod_snapshot[li] = {'kc': kc[:n_used].copy(), 'vc': vc[:n_used].copy()}
        else:
            ssm = ttnn.to_torch(layer['dn']['ssm'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            conv = ttnn.to_torch(layer['dn']['conv_st'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).float().numpy()
            prod_snapshot[li] = {'ssm': ssm[0].copy(), 'conv_st': conv.copy()}
    log(f"snapshot taken for {len(prod_snapshot)} layers")

    # 2. CB setup + transplant.
    B = 4
    cb.setup_cb_state(state, B)
    cb.cb_reset_states(state)
    log(f"CB state ready (B={B}); transplanting slot {args.slot} (L={L})")
    cb.cb_prefill_transplant(state, args.slot, L)
    ttnn.synchronize_device(state.mesh)
    log("transplant done; verifying slot state matches production…")

    # 3. Read CB slot back; per-layer compare.
    any_fail = False
    blocks_per_seq = state.cb_blocks_per_seq
    slot_s = args.slot
    for li, layer in enumerate(state.layers):
        if layer['type'] == 'full_attention':
            kv = state.cb_kv[li]
            cb_kc = ttnn.to_torch(kv['kc'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            cb_vc = ttnn.to_torch(kv['vc'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            lo = slot_s * blocks_per_seq
            slot_kc = cb_kc[lo:lo + n_used]
            slot_vc = cb_vc[lo:lo + n_used]
            c_kc = _cos(slot_kc, prod_snapshot[li]['kc'])
            c_vc = _cos(slot_vc, prod_snapshot[li]['vc'])
            ok = c_kc >= args.cos_threshold and c_vc >= args.cos_threshold
            any_fail = any_fail or not ok
            # Print first 2 layers + any FAILs (rest abbreviated to keep log readable).
            if not ok or li < 2 or li == len(state.layers) - 1:
                log(f"  L{li:02d} attn:  cos_kc={c_kc:.6f}  cos_vc={c_vc:.6f}  "
                    f"{'OK' if ok else 'FAIL'}")
        else:
            slot = state.cb_dn[li]
            cb_ssm = ttnn.to_torch(slot['ssm'],
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
            slot_ssm = cb_ssm[slot_s]
            c_ssm = _cos(slot_ssm, prod_snapshot[li]['ssm'])
            conv_cos = []
            for k in range(cfg['conv_kernel'] - 1):
                cb_col = ttnn.to_torch(slot['conv_cols'][k],
                    mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
                slot_col = cb_col[slot_s]
                conv_cos.append(_cos(slot_col, prod_snapshot[li]['conv_st'][:, k]))
            ok = c_ssm >= args.cos_threshold and all(c >= args.cos_threshold for c in conv_cos)
            any_fail = any_fail or not ok
            if not ok or li < 2:
                log(f"  L{li:02d} dn:    cos_ssm={c_ssm:.6f}  cos_conv={conv_cos}  "
                    f"{'OK' if ok else 'FAIL'}")

    # Verify slot cur_pos was set.
    cur = ttnn.to_torch(state.cb_cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)).int().numpy()[:B]
    pos_ok = int(cur[slot_s]) == L
    log(f"  cur_pos[{slot_s}] = {int(cur[slot_s])} (expected {L}) "
        f"{'OK' if pos_ok else 'FAIL'}")
    if not pos_ok:
        any_fail = True

    if any_fail:
        log("FAIL: transplant did not bit-match production state on all layers.")
        raise SystemExit(1)
    log(f"PASS: cb_prefill_transplant copied production state into slot {slot_s} "
        f"at cos >= {args.cos_threshold} on all {len(prod_snapshot)} layers. "
        f"S2.3 gate green; S2.4 (end-to-end decode-after-transplant) unblocked.")


if __name__ == "__main__":
    main()
