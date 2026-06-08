#!/usr/bin/env python3
"""Spec-dec KV cache-invariance regression probe.

The off-by-one in the cache-advance loop (commit 5aa1550) silently
corrupted the KV cache and only surfaced via mangled output text
("Paris Paris**, and** and**"). This probe locks down the invariant:

    Running spec-dec for N tokens MUST produce the same KV cache
    state as running plain B=1 target decode for the same token
    sequence.

Strategy:
1. Bootstrap target + drafter, prefill a fixed prompt → cache @ pos L-1.
2. Spec-dec path: run rounds until ≥N tokens emitted. Capture:
     - emitted_sd (list[int])
     - cur_pos_sd after the last round
     - probe_argmax_sd: feed an arbitrary "probe" token at cur_pos_sd+1
       and capture target's argmax — this exercises the cache reads
       all the way back to position 0.
3. Reset: re-prefill same prompt (overwrites cache positions 0..L-1
   with the same data, leaving cur_pos = L-1).
4. Baseline path: feed emitted_sd[i] at positions L, L+1, ..., L+len-1.
   Capture:
     - cur_pos_bl after the last step
     - probe_argmax_bl: same probe token at cur_pos_bl+1.
5. Assert: cur_pos_sd == cur_pos_bl AND probe_argmax_sd == probe_argmax_bl.

If both match, the cache state must agree at every cached position
the SDPA touched (any divergence would change the probe's argmax).

Gates printed and exit-coded:
  GATE A: len matches (sanity)
  GATE B: cur_pos matches
  GATE C: probe argmax matches  ← the real invariance check

Run on qb2 (drafter cache is only there):
  ssh qb2 'cd ~/tt-xla && tt-smi -r 0,1,2,3 >/dev/null 2>&1 && \\
    TT_GEMMA4_VARIANT=it TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$HOME/tenstorrent/tt-metal/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$HOME/tenstorrent/tt-metal/ttnn \\
    LD_LIBRARY_PATH=... \\
    .venv/bin/python -u experiments/cb/isolate/gemma4_spec_dec_cache_invariant_probe.py'
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

K = 3
N_TOKENS = 12        # ≥ N tokens emitted via spec-dec, then replayed
PROBE_TOKEN = 818     # arbitrary "The" — any common token works
PROMPT = "The capital of France is"  # short enough to bootstrap fast
BOS = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def co_load():
    log("STAGE 1: bootstrap target (~90s)…")
    tgt_state = tgt.State()
    t0 = time.time()
    tgt.bootstrap(tgt_state, log=log)
    log(f"  target bootstrap took {time.time()-t0:.1f}s")
    log("STAGE 1: bootstrap drafter co-loaded…")
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
    return tgt_state, drf_state


def _cur_pos(tgt_state):
    return int(ttnn.to_torch(
        tgt_state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
    ).flatten()[0].item())


def prefill(tgt_state, ids):
    """Walks step_forward_v031 over the prompt. Cache ends at pos L-1."""
    for pos, tok in enumerate(ids):
        tgt.step_forward_v031(tgt_state, tok_id=int(tok), pos=pos)


def run_spec_dec(tgt_state, drf_state, ids, n_target):
    """Spec-dec for ≥n_target emitted tokens. Returns (emitted, cur_pos)."""
    tgt.setup_verify_kp1_state(tgt_state, K=K, log=lambda *a, **k: None)
    cp = _cur_pos(tgt_state)
    tgt.update_verify_inputs(tgt_state, current_pos=cp,
                              candidate_token_ids=[0] * (K + 1))
    tgt.ensure_verify_trace_kp1(tgt_state, log=lambda *a, **k: None)
    cfg = sched.SpecDecConfig(K=K, max_new=n_target + K)
    sd = sched.SpecDecScheduler(target_state=tgt_state,
                                  drafter_state=drf_state, config=cfg)
    base_token = int(ids[-1])
    emitted = []
    rounds = 0
    while len(emitted) < n_target:
        cp = _cur_pos(tgt_state)
        res = sd.step(
            base_token=base_token,
            target_h_prev_np=tgt_state.last_target_hidden_prev,
            target_h_last_np=tgt_state.last_target_hidden_cur,
            cur_pos=cp,
        )
        emitted.extend(res.accepted_tokens)
        base_token = res.accepted_tokens[-1]
        rounds += 1
        if rounds > n_target * 2:
            log(f"  ⚠ safety bound; emitted={len(emitted)}, rounds={rounds}")
            break
    # Return ALL emitted tokens (no truncation). The scheduler already
    # wrote K/V for every emitted token; truncating here would make the
    # baseline replay feed fewer tokens and mis-detect a cache divergence
    # at the LENGTH level instead of at the CONTENTS level.
    return emitted, _cur_pos(tgt_state)


def run_baseline_seq(tgt_state, ids, seq):
    """Re-prefill `ids`, then feed `seq` tokens via target B=1 traced
    (overwrites cache at positions 0..L-1, then writes seq at L..L+len-1).
    Returns cur_pos after the last step."""
    prefill(tgt_state, ids)
    tgt.ensure_decode_trace(tgt_state, log=lambda *a, **k: None)
    cp = _cur_pos(tgt_state)
    feed_pos = cp + 1
    for tok in seq:
        tgt.step_forward_traced(tgt_state, token_id=int(tok),
                                  cur_pos=feed_pos)
        feed_pos += 1
    return _cur_pos(tgt_state)


def probe_argmax_at_next(tgt_state):
    """Feed PROBE_TOKEN at cur_pos+1 (one decode step) and capture
    target's argmax. The argmax depends on cache contents at every
    cached position via SDPA; this is the cache-invariance witness.

    Returns: (argmax_int, new_cur_pos). The cache advances by 1; caller
    is responsible for the after-state if it matters.
    """
    cp = _cur_pos(tgt_state)
    argmax = tgt.step_forward_traced(tgt_state, token_id=int(PROBE_TOKEN),
                                       cur_pos=cp + 1)
    return int(argmax), _cur_pos(tgt_state)


def main():
    log("=" * 72)
    log("Spec-dec KV cache-invariance probe")
    log(f"  K={K}, N_TOKENS={N_TOKENS}, prompt={PROMPT!r}")
    log("=" * 72)
    tgt_state, drf_state = co_load()

    # Tokenize via drafter's tokenizer (shared vocab).
    ids = [BOS] + list(drf_state.tokenizer.encode(
        PROMPT, add_special_tokens=False))
    L = len(ids)
    log(f"prompt: {L} tokens, ids[:6]={ids[:6]}")

    # Warm decode trace ONCE — both probes need it. Trace warmup writes
    # K/V at positions 0..2 (BOS), but the immediately-following prefill
    # overwrites positions 0..L-1 with the real prompt, so no residual
    # pollution.
    tgt.ensure_decode_trace(tgt_state, log=log)

    # ── Phase A: spec-dec path. ────────────────────────────────────
    log("─" * 72)
    log("PHASE A: spec-dec path")
    log("─" * 72)
    prefill(tgt_state, ids)
    emitted_sd, cp_sd_after = run_spec_dec(tgt_state, drf_state, ids, N_TOKENS)
    log(f"  emitted_sd ({len(emitted_sd)}): {emitted_sd}")
    log(f"  cur_pos_sd after spec-dec: {cp_sd_after}")
    probe_sd, cp_sd_probe = probe_argmax_at_next(tgt_state)
    log(f"  probe argmax (sd): {probe_sd} (cache advanced to {cp_sd_probe})")

    # ── Phase B: baseline path on the same emitted sequence. ───────
    log("─" * 72)
    log("PHASE B: baseline path on emitted_sd")
    log("─" * 72)
    cp_bl_after = run_baseline_seq(tgt_state, ids, emitted_sd)
    log(f"  cur_pos_bl after baseline: {cp_bl_after}")
    probe_bl, cp_bl_probe = probe_argmax_at_next(tgt_state)
    log(f"  probe argmax (bl): {probe_bl} (cache advanced to {cp_bl_probe})")

    # ── Gates ──────────────────────────────────────────────────────
    log("=" * 72)
    log("GATES")
    log("=" * 72)
    gate_a = len(emitted_sd) >= N_TOKENS
    gate_b = cp_sd_after == cp_bl_after
    gate_c = probe_sd == probe_bl
    log(f"  GATE A (emitted length)  : "
        f"len={len(emitted_sd)} target≥{N_TOKENS} "
        f"{'✓ PASS' if gate_a else '✗ FAIL'}")
    log(f"  GATE B (cur_pos match)   : "
        f"sd={cp_sd_after} bl={cp_bl_after} "
        f"{'✓ PASS' if gate_b else '✗ FAIL'}")
    log(f"  GATE C (probe argmax)    : "
        f"sd={probe_sd} bl={probe_bl} "
        f"{'✓ PASS' if gate_c else '✗ FAIL'}")
    all_pass = gate_a and gate_b and gate_c
    log("=" * 72)
    log(f"VERDICT: {'✓ PASS' if all_pass else '✗ FAIL — KV cache invariant violated'}")
    log("=" * 72)
    ttnn.close_mesh_device(tgt_state.mesh)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
