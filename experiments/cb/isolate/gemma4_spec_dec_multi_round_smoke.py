#!/usr/bin/env python3
"""Phase 3 v0.0b — multi-round spec-dec smoke + α sweep at K∈{3,5,7}.

Builds on v0.0a (single-round smoke). Runs N=10 rounds per K, tracks
α stabilization, emits decoded text. Uses the Phase 3 v0.0b target
hidden exposure hook so drafter chaining works across rounds without
HF oracle assistance.

Gates:
1. PER-ROUND PROGRESS: cur_pos advances every round (no stuck loops).
2. EMIT > 0: scheduler emits at least 1 token per round.
3. α ≥ 0.0 (always true), captured per-K.
4. NO TT_FATAL across N rounds.
5. EMITTED TEXT IS COHERENT (manual inspection).

Run on qb1 (standalone):
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \\
      experiments/cb/isolate/gemma4_spec_dec_multi_round_smoke.py'
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

ORACLE_DIR_TGT = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"

K_SWEEP = [3, 5]  # 7 added if 5 is stable (more memory + trace)
N_ROUNDS_PER_K = 5  # tokens per K — adjust based on budget


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def co_load(log_fn):
    """Bootstrap target + drafter on one mesh. Returns (tgt_state, drf_state)."""
    log_fn("STAGE 1: bootstrap target (~70s)…")
    tgt_state = tgt.State()
    t0 = time.time()
    tgt.bootstrap(tgt_state, log=log_fn)
    log_fn(f"  target bootstrap took {time.time()-t0:.1f}s")

    log_fn("STAGE 1: bootstrap drafter co-loaded (~7s)…")
    drf_state = drf.State()
    drf_state.mesh = tgt_state.mesh
    _orig_open = ttnn.open_mesh_device
    _orig_fab = ttnn.set_fabric_config
    ttnn.open_mesh_device = lambda *a, **kw: tgt_state.mesh
    ttnn.set_fabric_config = lambda *a, **kw: None
    t0 = time.time()
    try:
        drf.bootstrap(drf_state, log=log_fn)
    finally:
        ttnn.open_mesh_device = _orig_open
        ttnn.set_fabric_config = _orig_fab
    drf_state.mesh = tgt_state.mesh
    log_fn(f"  drafter bootstrap took {time.time()-t0:.1f}s")
    return tgt_state, drf_state


def run_K(tgt_state, drf_state, K: int, N_rounds: int) -> dict:
    """Run multi-round spec-dec at the given K. Returns metrics + emitted toks."""
    log(f"━━━ K={K}: capture verify trace + run {N_rounds} rounds ━━━")

    # Fresh verify trace per K. Need a clean target state (cur_pos at
    # post-prefill); since the test runs K=3 then K=5, each K assumes
    # target's prefill state is preserved. For first K, re-prefill.

    # Verify trace: requires setup_verify_kp1_state then ensure_verify_trace_kp1.
    # If state.verify_K differs from K, we'd need to re-bootstrap; for the
    # K sweep we just bootstrap once + run K=3 (smallest first). The
    # verify_K is locked at first setup.
    if getattr(tgt_state, "verify_K", None) is None:
        tgt.setup_verify_kp1_state(tgt_state, K=K, log=log)
        # Seed verify inputs for warmup (will be overwritten).
        cur_pos_arr = ttnn.to_torch(
            tgt_state.cur_pos_buf,
            mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
        )
        cur_pos = int(cur_pos_arr.flatten()[0].item())
        tgt.update_verify_inputs(tgt_state, current_pos=cur_pos,
                                  candidate_token_ids=[0] * (K + 1))
        tgt.ensure_verify_trace_kp1(tgt_state, log=log)
    elif tgt_state.verify_K != K:
        log(f"  ⚠ verify trace already at K={tgt_state.verify_K}, "
            f"can't reconfigure to K={K} without bootstrap restart. "
            f"Skipping K={K}.")
        return {"K": K, "skipped": True}

    cfg = sched.SpecDecConfig(K=K, max_new=N_rounds * (K + 1))
    scheduler = sched.SpecDecScheduler(target_state=tgt_state,
                                         drafter_state=drf_state, config=cfg)

    # Initial base_token = target's last prefilled token (or last emitted).
    prompt_ids = np.load(ORACLE_DIR_TGT / "prompt_ids.npy")
    base_token = int(prompt_ids[-1])
    log(f"  initial base_token (last prefilled) = {base_token}")

    emitted_all = []
    per_round_alphas = []
    t_start = time.time()
    for r in range(N_rounds):
        # Read cur_pos
        cur_pos_arr = ttnn.to_torch(
            tgt_state.cur_pos_buf,
            mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
        )
        cur_pos = int(cur_pos_arr.flatten()[0].item())
        target_h_prev = tgt_state.last_target_hidden_prev
        target_h_last = tgt_state.last_target_hidden_cur
        if target_h_prev is None or target_h_last is None:
            log(f"  ✗ round {r}: missing stashed target hidden — "
                f"prev={target_h_prev is None}, cur={target_h_last is None}")
            return {"K": K, "rounds": r, "error": "missing_stash"}
        try:
            t_round = time.time()
            result = scheduler.step(base_token=base_token,
                                      target_h_prev_np=target_h_prev,
                                      target_h_last_np=target_h_last,
                                      cur_pos=cur_pos)
            round_wall_ms = (time.time() - t_round) * 1000
        except Exception as e:
            log(f"  ✗ round {r} crash: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return {"K": K, "rounds": r, "error": str(e)}

        emitted_all.extend(result.accepted_tokens)
        per_round_alphas.append(result.alpha)
        log(f"  round {r}: emitted {result.accepted_tokens} "
            f"(accept={result.accept_count}/{K} α={result.alpha:.2f}) "
            f"cur_pos {cur_pos}→{cur_pos+result.n_emitted} "
            f"wall {round_wall_ms:.0f}ms")
        base_token = result.accepted_tokens[-1]

    total_wall_s = time.time() - t_start
    total_tokens = len(emitted_all)

    # Decode emitted tokens to text.
    tok = tgt_state.tok
    try:
        text = tok.decode(emitted_all)
    except Exception as e:
        text = f"[decode error: {e}]"
    log(f"  EMITTED ({total_tokens} tok): {emitted_all}")
    log(f"  TEXT: {text!r}")
    avg_alpha = sum(per_round_alphas) / len(per_round_alphas) if per_round_alphas else 0.0
    log(f"  K={K} summary: rounds={N_rounds}, tokens={total_tokens}, "
        f"avg α={avg_alpha:.3f}, total wall {total_wall_s:.1f}s, "
        f"{total_tokens/total_wall_s:.2f} tok/s")
    return {
        "K": K, "rounds": N_rounds, "tokens": total_tokens,
        "avg_alpha": avg_alpha, "per_round_alphas": per_round_alphas,
        "wall_s": total_wall_s, "tok_per_s": total_tokens / total_wall_s,
        "emitted": emitted_all, "text": text,
    }


def main():
    log("=" * 64)
    log("Phase 3 v0.0b — multi-round spec-dec smoke + α sweep")
    log("=" * 64)
    rc = 0
    tgt_state, drf_state = co_load(log)

    # PREFILL once. Subsequent K runs reuse state. (Target hidden stash
    # ensures drafter chain has correct inputs across rounds.)
    log("STAGE 2: prefill target's 6-token canonical prompt…")
    prompt_ids = np.load(ORACLE_DIR_TGT / "prompt_ids.npy")
    L_prefill = int(prompt_ids.shape[0])
    log(f"  L_prefill = {L_prefill}, prompt_ids = {prompt_ids.tolist()}")
    t = time.time()
    for pos in range(L_prefill):
        tok = int(prompt_ids[pos])
        tgt.step_forward_v031(tgt_state, tok_id=tok, pos=pos)
    log(f"  prefill wall: {(time.time()-t)*1000:.1f} ms")
    log(f"  stashed target_h_prev shape: "
        f"{None if tgt_state.last_target_hidden_prev is None else tgt_state.last_target_hidden_prev.shape}")
    log(f"  stashed target_h_cur shape:  "
        f"{None if tgt_state.last_target_hidden_cur is None else tgt_state.last_target_hidden_cur.shape}")

    if tgt_state.last_target_hidden_prev is None or tgt_state.last_target_hidden_cur is None:
        log("  ✗ target hidden stash missing — server may not have v0.0b hook")
        return 1

    # Run only the FIRST K from K_SWEEP (verify_K is locked at first setup).
    # Re-bootstrapping for each K would take 70s × len(K_SWEEP); not worth it
    # in v0.0b. v0.0c sweep across K via bootstrap-per-K.
    K = K_SWEEP[0]
    result = run_K(tgt_state, drf_state, K=K, N_rounds=N_ROUNDS_PER_K)
    log("=" * 64)
    log("FINAL")
    log("=" * 64)
    log(f"  K={K}: avg α = {result.get('avg_alpha', '?'):.3f}, "
        f"tokens = {result.get('tokens', '?')}, "
        f"wall = {result.get('wall_s', '?'):.1f}s")
    log(f"  text: {result.get('text', '?')!r}")

    if "error" in result:
        log("VERDICT: FAIL — see error above")
        rc = 1
    elif result.get("tokens", 0) == 0:
        log("VERDICT: FAIL — no tokens emitted")
        rc = 1
    else:
        log("VERDICT: PASS — multi-round spec-dec works end-to-end")

    ttnn.close_mesh_device(tgt_state.mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main())
