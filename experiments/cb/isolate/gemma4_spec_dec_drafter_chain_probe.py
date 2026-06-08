#!/usr/bin/env python3
"""Spec-dec drafter chain probe: compare scheduler's per-round drafter
predictions to v2 oracle for the canonical prompt 0.

Bootstrap target + drafter, prefill prompt 0, then run scheduler's
_drafter_autoregressive_K with the real device-side state (target's
stashed last hidden + device shared_kv) and dump per-round drafter
argmaxes. Compare to HF v2 oracle's [496, 5464, 236772, 2084, 3207].

If our scheduler argmaxes MATCH v2 oracle → drafter+scheduler perfect,
problem is target verify (verify trace produces different argmaxes than
expected).

If our argmaxes DIFFER → target's device-side hidden state diverges
from HF (bf16 chain drift) enough to throw the drafter off.

Run on qb1:
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \\
      experiments/cb/isolate/gemma4_spec_dec_drafter_chain_probe.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as tgt  # noqa: E402
import server_gemma4_12b_assistant_ttnn as drf  # noqa: E402
import spec_dec_scheduler as sched  # noqa: E402

V2_ORACLE = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant_v2"
K = 5
HF_V2_EXPECTED = [496, 5464, 236772, 2084, 3207]  # prompt_0 K=5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if (na and nb) else 0.0


def main():
    log("=" * 64)
    log("Spec-dec drafter chain probe — compare to v2 oracle prompt_0 K=5")
    log(f"  HF v2 expected: {HF_V2_EXPECTED}")
    log("=" * 64)

    # Load v2 oracle hidden state for sanity comparison.
    pd = V2_ORACLE / "prompt_0"
    hf_target_h_last = np.load(pd / "target_h_last.npy").astype(np.float32)
    hf_kv_sl_K = np.load(pd / "shared_kv_sliding_K.npy").astype(np.float32)
    hf_kv_sl_V = np.load(pd / "shared_kv_sliding_V.npy").astype(np.float32)
    hf_kv_fl_K = np.load(pd / "shared_kv_full_K.npy").astype(np.float32)
    hf_kv_fl_V = np.load(pd / "shared_kv_full_V.npy").astype(np.float32)
    log(f"  HF target_h_last shape {hf_target_h_last.shape}, "
        f"|hf_h|={np.abs(hf_target_h_last).max():.3f}")

    log("STAGE 1: bootstrap target")
    tgt_state = tgt.State()
    t0 = time.time()
    tgt.bootstrap(tgt_state, log=log)
    log(f"  target bootstrap took {time.time()-t0:.1f}s")

    log("STAGE 1: bootstrap drafter co-loaded")
    drf_state = drf.State()
    drf_state.mesh = tgt_state.mesh
    _orig_open = ttnn.open_mesh_device
    _orig_fab = ttnn.set_fabric_config
    ttnn.open_mesh_device = lambda *a, **kw: tgt_state.mesh
    ttnn.set_fabric_config = lambda *a, **kw: None
    t0 = time.time()
    try:
        drf.bootstrap(drf_state, log=log)
    finally:
        ttnn.open_mesh_device = _orig_open
        ttnn.set_fabric_config = _orig_fab
    drf_state.mesh = tgt_state.mesh
    log(f"  drafter bootstrap took {time.time()-t0:.1f}s")

    log("STAGE 2: prefill 'The capital of France is' (NO BOS — matches HF v2 oracle)")
    # HF v2 oracle did NOT add BOS. Use raw 5-token prefix.
    prompt_ids = [818, 5279, 529, 7001, 563]
    L = len(prompt_ids)
    last_argmax = None
    for pos in range(L):
        last_argmax = tgt.step_forward_v031(tgt_state, tok_id=prompt_ids[pos], pos=pos)
    log(f"  prefill done; target predicted t_{L} = {last_argmax}")

    # Compare our target_h_last to HF's.
    our_h_last = tgt_state.last_target_hidden_cur
    log(f"  our target_h_last shape: {our_h_last.shape}, "
        f"|our_h|={np.abs(our_h_last).max():.3f}")
    c = cos(our_h_last, hf_target_h_last)
    mad = float(np.max(np.abs(our_h_last.flatten() - hf_target_h_last.flatten())))
    log(f"  cos(our_h, hf_h) = {c:.6f}  mad = {mad:.4f}")

    # Compare our shared_kv to HF's.
    cur_pos = L - 1
    shared_kv_ours = tgt.read_shared_kv_for_drafter(tgt_state, L_kv=L)
    our_Ksl, our_Vsl = shared_kv_ours["sliding_attention"]
    our_Kfl, our_Vfl = shared_kv_ours["full_attention"]
    log(f"  our K_sliding shape {our_Ksl.shape} vs HF {hf_kv_sl_K.shape}")
    log(f"  our K_full shape    {our_Kfl.shape} vs HF {hf_kv_fl_K.shape}")
    # Only compare if shapes match; otherwise log size mismatch.
    if our_Ksl.shape == hf_kv_sl_K.shape:
        c_ksl = cos(our_Ksl, hf_kv_sl_K)
        c_kfl = cos(our_Kfl, hf_kv_fl_K)
        log(f"  cos(K_sliding) = {c_ksl:.6f}, cos(K_full) = {c_kfl:.6f}")
    else:
        log(f"  ⚠ KV shape mismatch — skipping cos compare")

    log("=" * 64)
    log(f"STAGE 3: run scheduler._drafter_autoregressive_K (K={K})")
    log("=" * 64)
    cfg = sched.SpecDecConfig(K=K)
    scheduler = sched.SpecDecScheduler(target_state=tgt_state,
                                         drafter_state=drf_state, config=cfg)
    # Use base_token = last prompt token = 563 (matches v2 oracle)
    base_token = prompt_ids[-1]
    log(f"  base_token = {base_token}")

    # Run two variants:
    # (A) feed OUR device-side target_h_last
    log("─" * 64)
    log("VARIANT A: with OUR device-side target_h_last")
    log("─" * 64)
    draftA = scheduler._drafter_autoregressive_K(
        base_token=base_token,
        target_h_last_np=our_h_last,
        shared_kv_np=shared_kv_ours,
    )
    log(f"  our argmaxes: {draftA}")
    log(f"  HF expected:  {HF_V2_EXPECTED}")
    matches_A = sum(1 for a, b in zip(draftA, HF_V2_EXPECTED) if a == b)
    log(f"  ✓ MATCH {matches_A}/{K}" if matches_A == K else
        f"  ⚠ MATCH {matches_A}/{K} — mismatches at rounds "
        f"{[r for r,(a,b) in enumerate(zip(draftA, HF_V2_EXPECTED)) if a!=b]}")

    # (B) feed HF's target_h_last (from v2 oracle) with our device shared_kv
    log("─" * 64)
    log("VARIANT B: with HF's target_h_last + our device shared_kv")
    log("─" * 64)
    draftB = scheduler._drafter_autoregressive_K(
        base_token=base_token,
        target_h_last_np=hf_target_h_last,
        shared_kv_np=shared_kv_ours,
    )
    log(f"  our argmaxes: {draftB}")
    matches_B = sum(1 for a, b in zip(draftB, HF_V2_EXPECTED) if a == b)
    log(f"  ✓ MATCH {matches_B}/{K}" if matches_B == K else
        f"  ⚠ MATCH {matches_B}/{K}")

    # (C) feed HF's target_h_last + HF's shared_kv
    log("─" * 64)
    log("VARIANT C: with HF's target_h_last + HF's shared_kv")
    log("─" * 64)
    shared_kv_hf = {
        "sliding_attention": (hf_kv_sl_K, hf_kv_sl_V),
        "full_attention": (hf_kv_fl_K, hf_kv_fl_V),
    }
    draftC = scheduler._drafter_autoregressive_K(
        base_token=base_token,
        target_h_last_np=hf_target_h_last,
        shared_kv_np=shared_kv_hf,
    )
    log(f"  our argmaxes: {draftC}")
    matches_C = sum(1 for a, b in zip(draftC, HF_V2_EXPECTED) if a == b)
    log(f"  ✓ MATCH {matches_C}/{K}" if matches_C == K else
        f"  ⚠ MATCH {matches_C}/{K}")

    log("=" * 64)
    log("DIAGNOSIS")
    log("=" * 64)
    log(f"  A (our h, our kv):  {matches_A}/{K} match")
    log(f"  B (HF h, our kv):   {matches_B}/{K} match")
    log(f"  C (HF h, HF kv):    {matches_C}/{K} match")
    if matches_C == K:
        if matches_B == K:
            log(f"  → device shared_kv is fine, target_h device drift is the issue")
        else:
            log(f"  → device shared_kv breaks drafter (KV layout/precision)")
    else:
        log(f"  → drafter has SOME bug even with all-HF inputs")

    ttnn.close_mesh_device(tgt_state.mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
