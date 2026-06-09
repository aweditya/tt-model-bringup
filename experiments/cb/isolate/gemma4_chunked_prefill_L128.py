#!/usr/bin/env python3
"""#290 Phase 1 — chunked prefill for Gemma 4 at L=128 (SCAFFOLD).

Current state of our prefill is N × `step_forward_v031` (sequential
decode-style, one token at a time). At 47 ms/tok this means a 200-token
prompt costs ~9.4 s TTFT — unusable for chat. Real prefill processes L
tokens through matmul layers in parallel, with causal SDPA over L
tokens, in one pass.

This probe is the FIRST step: build `step_forward_prefill(state,
token_ids, start_pos=0)` that processes L=128 in parallel. Gates:
  1. cos ≥ 0.999 vs L=128 × `step_forward_v031` (ground truth)
  2. eager TTFT ≥ 2× faster than sequential
  3. HF-equivalent argmax at the last position

Reuse map (per non-negotiable):
- `experiments/serve/server_tp.py:forward_prefill_chunked_tp` (line 1945)
  — 27B's chunked prefill: embed all L tokens, RoPE per position, loop
  layers in parallel. Forks the OUTER STRUCTURE only — Gemma 4 has
  different layer types (sliding + global attention) and different
  attention contracts (dual head_dim, p-RoPE on global, v_norm).
- `experiments/cb/isolate/gm4_v033c_*` family — Gemma 4 isolation
  pattern (bootstrap + capture).
- `server_gemma4_unified_ttnn._layer_pos0_sliding_paged` /
  `_layer_pos0_global_paged` — per-layer decode. The prefill versions
  need to (a) accept multi-token Q, (b) use causal SDPA over L instead
  of paged_decode, (c) write K/V for ALL L positions via paged_update.

TODO work-list for the next session (gated on user direction):

  [TODO P1.1] Multi-token embed + RoPE rebuild
    Embed [L] tokens into [L, HIDDEN] (vs current [1, HIDDEN]). Build
    cos/sin tables for positions [start_pos..start_pos+L-1] in one
    `ttnn.embedding` lookup on rot_idxs_buf at multiple indices.

  [TODO P1.2] _layer_prefill_sliding (forks _layer_pos0_sliding_paged)
    - Q matmul over [L, HIDDEN] → [L, NQ_PER_CHIP * HEAD_DIM_SLIDING]
    - K/V matmul over [L, HIDDEN] → [L, NKV_PER_CHIP_SLIDING * HEAD_DIM_SLIDING]
    - Per-head reshape + q_norm/k_norm/v_norm (norms apply to last dim,
      should work over [L, NH, HD]).
    - RoPE on Q+K over all L positions (cos/sin shaped [L, head_dim]).
    - Causal SDPA: ttnn.transformer.scaled_dot_product_attention with
      is_causal=True, q_len = L, k_len = L. (NOT paged_sdpa_decode.)
    - paged_fused_update_cache writing K/V at positions
      [start_pos..start_pos+L-1] in ONE call.
    - o_proj + all_reduce (works at [L, HIDDEN]; matmul is
      leading-dim agnostic).
    - Validation: cos ≥ 0.999 vs sequential.

  [TODO P1.3] _layer_prefill_global
    Same as sliding but:
    - head_dim = HEAD_DIM_GLOBAL = 512
    - NKV_PER_CHIP_GLOBAL = 1 (replicated across mesh)
    - p-RoPE (partial 0.25; cos/sin tables have inv_freq=0 on non-rotated)
    - attention_k_eq_v: V aliases K_raw pre-norm

  [TODO P1.4] _layer_prefill_mlp (mostly reuse)
    pre_ff_norm → gate_proj matmul → up_proj matmul → mul → down_proj
    + all_reduce. All matmuls are leading-dim agnostic; the existing
    code at line 1994-1996 (or DRAM-sharded variant at 1957-1962)
    should work as-is by passing a [L, HIDDEN] activation.

  [TODO P1.5] step_forward_prefill orchestrator
    Bootstrap → embed all L → per-layer forward over [L, HIDDEN] →
    final_norm → lm_head on last row → argmax.

  [TODO P1.6] Gates
    A. cos ≥ 0.999: compare last-position hidden vs sequential.
    B. eager TTFT measurement: time chunked vs sequential at L=128.
    C. argmax: compare last-position prediction to HF / sequential.

  [TODO P1.7] Run on qb1, iterate per failure mode.

Run skeleton (will fail with NotImplementedError until TODOs land):
  ssh qb1 'cd ~/tt-xla && tt-smi -r 0,1,2,3 >/dev/null 2>&1 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$HOME/tenstorrent/tt-metal/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$HOME/tenstorrent/tt-metal/ttnn \\
    LD_LIBRARY_PATH=... \\
    .venv/bin/python -u experiments/cb/isolate/gemma4_chunked_prefill_L128.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

L = 128
N_DECODE_CHECK = 1   # only check last-position argmax for now


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_forward_prefill(state, token_ids, start_pos=0):
    """[SCAFFOLD] Multi-token parallel prefill.

    See TODOs P1.1-P1.5 in the file docstring. Will raise
    NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "step_forward_prefill not yet implemented. See P1.1-P1.5 in "
        "the docstring for the work-list. Each TODO is a self-contained "
        "fork from an existing decode helper."
    )


def baseline_sequential(state, token_ids):
    """Reference path: process token_ids one at a time via step_forward_v031.
    Returns (last_argmax, total_wall_s, last_hidden_np)."""
    t0 = time.time()
    last_argmax = None
    for pos, tok in enumerate(token_ids):
        last_argmax = srv.step_forward_v031(state, tok_id=int(tok), pos=pos)
    elapsed = time.time() - t0
    # state.last_target_hidden_cur is the post-final-norm hidden at last
    # position, stashed by the v0.3 hook (spec-dec convenience).
    last_hidden = state.last_target_hidden_cur
    return int(last_argmax), elapsed, last_hidden


def main():
    log("=" * 72)
    log(f"#290 Phase 1 SCAFFOLD — chunked prefill at L={L}")
    log("=" * 72)

    log("STAGE 1: bootstrap target (~90s)…")
    state = srv.State()
    srv.bootstrap(state, log=log)

    log(f"STAGE 2: pick {L}-token prompt (reuse the SEED_PARAGRAPH from "
        f"the long-context gate)")
    from gemma4_long_context_argmax_gate import (build_prompt, BOS)
    token_ids = build_prompt(state.tokenizer, target_len=L)
    log(f"  prompt: {len(token_ids)} tokens, first 6 = {token_ids[:6]}")

    log("STAGE 3: baseline sequential prefill (ground truth)")
    base_argmax, base_wall, base_hidden = baseline_sequential(state, token_ids)
    log(f"  baseline last-argmax = {base_argmax}, wall = {base_wall:.1f}s, "
        f"hidden shape = {base_hidden.shape if base_hidden is not None else 'None'}")

    log("STAGE 4: chunked prefill (NOT YET IMPLEMENTED)")
    try:
        chunk_argmax = step_forward_prefill(state, token_ids, start_pos=0)
    except NotImplementedError as e:
        log(f"  ✗ SCAFFOLD STOP: {e}")
        log("=" * 72)
        log("This scaffold is a structural placeholder; the chunked path "
            "needs implementation per TODOs P1.1-P1.5 in the docstring.")
        return 2

    # When implemented, the gates below run.
    log(f"  chunked last-argmax = {chunk_argmax}")
    pass_argmax = (chunk_argmax == base_argmax)
    log(f"  GATE C (argmax): {'✓ PASS' if pass_argmax else '✗ FAIL'} "
        f"(chunk={chunk_argmax} base={base_argmax})")

    # Cos vs baseline hidden — needs chunked path to expose last hidden.
    log("  GATE A (cos): TBD — chunked path needs to expose last_hidden_np")

    return 0 if pass_argmax else 1


if __name__ == "__main__":
    sys.exit(main())
