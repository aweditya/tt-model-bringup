#!/usr/bin/env python3
"""Phase 3 v0.0c — multi-prompt α distribution.

Bootstrap target + drafter ONCE, then loop over the 5 HF oracle prompts:
re-prefill each, run N=5 spec-dec rounds at K=3, log per-prompt α +
emitted text. Aggregate at end.

Goal: characterize REAL α across prompts. v0.0b's prompt_0 result of α=0
was clean drafter/target disagreement (validated bit-equiv to HF), not
a bug — but other prompts may show α > 0 where the two models agree.

Run on qb1:
  ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \\
      experiments/cb/isolate/gemma4_spec_dec_multi_prompt_smoke.py'
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

ORACLE_DRF = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"

K = 3
N_ROUNDS = 5
N_PROMPTS = 5

# BOS is prepended for target prefill (matches Gemma 4 convention).
BOS = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def co_load():
    log("STAGE 1: bootstrap target (~70s)…")
    tgt_state = tgt.State()
    t0 = time.time()
    tgt.bootstrap(tgt_state, log=log)
    log(f"  target bootstrap took {time.time()-t0:.1f}s")
    log("STAGE 1: bootstrap drafter co-loaded (~8s)…")
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


def prefill_prompt(tgt_state, prompt_ids: list) -> int:
    """Prefill target with prompt_ids starting at position 0. Returns
    the last predicted argmax (= what target predicts as t_{L}).
    """
    last_argmax = None
    for pos, tok in enumerate(prompt_ids):
        last_argmax = tgt.step_forward_v031(tgt_state, tok_id=int(tok), pos=pos)
    return int(last_argmax)


def run_prompt(tgt_state, drf_state, prompt_idx: int) -> dict:
    """Re-prefill prompt + run N spec-dec rounds. Returns metrics."""
    pd = ORACLE_DRF / f"prompt_{prompt_idx}"
    raw_ids = np.load(pd / "input_ids.npy").flatten().tolist()
    # Prepend BOS to match target prefill convention.
    prompt_ids = [BOS] + raw_ids
    L = len(prompt_ids)
    hf_drafter_argmax = int(np.load(pd / "drafter_argmax.npy").flatten()[0])
    log(f"━━━ prompt_{prompt_idx} ━━━")
    log(f"  L={L}, ids[:8]={prompt_ids[:8]}{'…' if L > 8 else ''}")
    log(f"  HF drafter argmax @ t_{L-1}'s prediction: {hf_drafter_argmax}")

    t = time.time()
    last_target_argmax = prefill_prompt(tgt_state, prompt_ids)
    log(f"  prefill {L} tokens in {(time.time()-t)*1000:.0f} ms; "
        f"target predicted t_{L} = {last_target_argmax}")

    if (tgt_state.last_target_hidden_prev is None or
            tgt_state.last_target_hidden_cur is None):
        log("  ✗ target hidden stash missing — abort")
        return {"prompt": prompt_idx, "error": "missing_stash"}

    cfg = sched.SpecDecConfig(K=K, max_new=N_ROUNDS * (K + 1))
    scheduler = sched.SpecDecScheduler(target_state=tgt_state,
                                         drafter_state=drf_state, config=cfg)
    base_token = int(prompt_ids[-1])
    emitted = []
    alphas = []
    accept_counts = []
    t_start = time.time()
    for r in range(N_ROUNDS):
        cur_pos = int(ttnn.to_torch(
            tgt_state.cur_pos_buf,
            mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
        ).flatten()[0].item())
        try:
            result = scheduler.step(
                base_token=base_token,
                target_h_prev_np=tgt_state.last_target_hidden_prev,
                target_h_last_np=tgt_state.last_target_hidden_cur,
                cur_pos=cur_pos,
            )
        except Exception as e:
            log(f"  ✗ round {r} crash: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return {"prompt": prompt_idx, "rounds": r, "error": str(e)}
        emitted.extend(result.accepted_tokens)
        alphas.append(result.alpha)
        accept_counts.append(result.accept_count)
        log(f"    round {r}: emitted {result.accepted_tokens} "
            f"(accept={result.accept_count}/{K} α={result.alpha:.2f})")
        base_token = result.accepted_tokens[-1]
    total_wall_s = time.time() - t_start

    avg_alpha = sum(alphas) / len(alphas) if alphas else 0.0
    # Decode emitted to text.
    try:
        text = tgt_state.tok.decode(emitted)
    except Exception as e:
        text = f"[decode error: {e}]"
    log(f"  TOTAL: {len(emitted)} tok in {total_wall_s:.1f}s, "
        f"avg α = {avg_alpha:.3f}")
    log(f"  TEXT: {text!r}")
    return {
        "prompt": prompt_idx,
        "L": L,
        "rounds": N_ROUNDS,
        "tokens": len(emitted),
        "avg_alpha": avg_alpha,
        "alphas": alphas,
        "accept_counts": accept_counts,
        "wall_s": total_wall_s,
        "emitted": emitted,
        "text": text,
    }


def main():
    log("=" * 64)
    log(f"Phase 3 v0.0c — multi-prompt α distribution (N={N_PROMPTS} prompts, "
        f"K={K}, {N_ROUNDS} rounds each)")
    log("=" * 64)

    tgt_state, drf_state = co_load()

    # Prefill prompt_0 ONCE (1) to populate state.last_target_hidden_prev/cur
    # before the first verify-trace warmup, and (2) so we can warmup at a
    # valid cur_pos that matches the verify-trace's expected shape contract.
    log("STAGE 2a: warmup prefill with prompt_0 (to seed target hidden + setup verify trace)…")
    pd0 = ORACLE_DRF / "prompt_0"
    raw_ids_0 = np.load(pd0 / "input_ids.npy").flatten().tolist()
    prefill_prompt(tgt_state, [BOS] + raw_ids_0)

    log(f"STAGE 2b: capture verify trace at K={K}…")
    tgt.setup_verify_kp1_state(tgt_state, K=K, log=log)
    cur_pos = int(ttnn.to_torch(
        tgt_state.cur_pos_buf,
        mesh_composer=ttnn.ConcatMeshToTensor(tgt_state.mesh, dim=0),
    ).flatten()[0].item())
    tgt.update_verify_inputs(tgt_state, current_pos=cur_pos,
                              candidate_token_ids=[0] * (K + 1))
    tgt.ensure_verify_trace_kp1(tgt_state, log=log)

    # Run all 5 prompts.
    results = []
    for i in range(N_PROMPTS):
        results.append(run_prompt(tgt_state, drf_state, i))
        # Reset state for next prompt: cur_pos and cache slots will be
        # overwritten by next prefill (step_forward_v031 writes K/V at
        # the specified pos). No explicit reset needed.

    # ── Aggregate ──
    log("=" * 64)
    log("AGGREGATE")
    log("=" * 64)
    valid = [r for r in results if "error" not in r]
    if not valid:
        log("✗ NO valid runs")
        return 1
    log(f"prompt | L  | α    | accept_counts        | text")
    log("-" * 78)
    for r in valid:
        ac_str = "[" + ", ".join(str(c) for c in r["accept_counts"]) + "]"
        text_short = r["text"][:30].replace("\n", "\\n")
        log(f"  {r['prompt']:5d} | {r['L']:2d} | {r['avg_alpha']:.3f}| {ac_str:21s}| {text_short!r}")
    total_alpha = sum(r["avg_alpha"] for r in valid) / len(valid)
    max_alpha = max(r["avg_alpha"] for r in valid)
    nonzero_count = sum(1 for r in valid if r["avg_alpha"] > 0)
    log("-" * 78)
    log(f"OVERALL: mean α = {total_alpha:.3f}, max α = {max_alpha:.3f}, "
        f"{nonzero_count}/{len(valid)} prompts had α > 0")

    log("=" * 64)
    if max_alpha > 0:
        log(f"VERDICT: PASS — spec-dec demonstrates α > 0 ({max_alpha:.2f}) "
            f"on at least one prompt. Framework + drafter/target alignment "
            f"working end-to-end.")
    else:
        log("VERDICT: PARTIAL — framework runs cleanly across all prompts but "
            "α = 0 throughout. May indicate drafter/target distributional "
            "mismatch needs more investigation, or larger N is required.")
    log("=" * 64)
    ttnn.close_mesh_device(tgt_state.mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
