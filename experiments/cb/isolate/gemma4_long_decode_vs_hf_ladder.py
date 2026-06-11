#!/usr/bin/env python3
"""#314 — teacher-forced long-decode ladder: TT vs HF, per-step per-layer.

Loads the HF reference artifacts written by
`experiments/utils/cosine_ladder_hf_gemma4_it.py`, teacher-forces the
SAME token sequence through our ttnn impl, captures per-layer hidden
after each decoder layer at each decode step, and reports the FIRST
(step, layer) where cos(tt_hidden, hf_hidden) < threshold.

That (step, layer) IS the drift onset — where to start bisecting.

REUSE:
- `step_forward_v03(state, tok_id, capture=...)` already supports
  `capture["per_layer"] = True` → `capture["layer_h"][L]`,
  `capture["final_norm"]`, `capture["logits"]`, `capture["argmax"]`.
  Zero code-change needed to instrument the forward.
- `forward_prefill_chunked_tp` writes the prompt K/V cache + advances
  `cur_pos_buf` so decode picks up at position L.
- Pattern + cosine helpers forked from
  `experiments/cb/isolate/gemma4_chunked_prefill_ladder.py`.

Inputs (CLI): --hf-npz path to .npz from cosine_ladder_hf_gemma4_it.py.
              --max-steps N caps how many decode steps we compare.

Run on qb1:
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/cb/isolate/gemma4_long_decode_vs_hf_ladder.py \\
        --hf-npz .cache/cosine_ladder_hf_gemma4_it/<ts>/ladder.npz \\
        --max-steps 100'

Output → .cache/gm4_long_decode_ladder/<ts>/ — cos heatmap (steps×layers),
verdict with the first sub-threshold (step, layer), and a json with
the full per-(step, layer) cos matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-npz", required=True,
                    help="path to ladder.npz from cosine_ladder_hf_gemma4_it.py")
    ap.add_argument("--max-steps", type=int, default=100,
                    help="cap on decode steps to ladder (default: all)")
    ap.add_argument("--out-dir", default=None,
                    help="default: .cache/gm4_long_decode_ladder/<ts>")
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="cos < threshold = divergence (default 0.99)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT_ROOT / ".cache" / "gm4_long_decode_ladder" / str(int(time.time()))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"out_dir = {out_dir}")

    log(f"loading HF reference {args.hf_npz}")
    hf = np.load(args.hf_npz)
    prompt_ids = hf["prompt_ids"].astype(np.int64).tolist()
    decode_ids = hf["decode_ids"].astype(np.int64).tolist()
    decode_hidden = hf["decode_hidden"]    # [N, L+1, HIDDEN] float32
    decode_logits = hf["decode_logits"]    # [N, VOCAB] float32
    N_HF = len(decode_ids)
    L_prompt = len(prompt_ids)
    n_layers_plus_1, HIDDEN = decode_hidden.shape[1], decode_hidden.shape[2]
    N = min(args.max_steps, N_HF - 1)  # we use decode_ids[0..N-1] as inputs to TT
    log(f"  L_prompt={L_prompt} N_HF={N_HF} n_layers+1={n_layers_plus_1} "
        f"HIDDEN={HIDDEN}; ladder up to N={N} steps")

    log("bootstrapping gemma4…")
    state = type("S", (), {})()
    srv.bootstrap(state, log=log)
    # step_forward_v03 + forward_prefill_chunked_tp both stash
    # `last_target_hidden_cur`/`_prev` into state (Phase 3 spec-dec
    # path). Direct callers (us) must pre-initialise to None or the
    # first read raises AttributeError.
    if not hasattr(state, "last_target_hidden_cur"):
        state.last_target_hidden_cur = None
    if not hasattr(state, "last_target_hidden_prev"):
        state.last_target_hidden_prev = None
    log("bootstrap done")

    # Prefill the prompt via the production chunked-prefill path. This
    # writes K/V for positions 0..L_prompt-1 and advances cur_pos_buf
    # to L_prompt — so the first decode step lands at the right
    # position.
    log(f"prefill via forward_prefill_chunked_tp (L={L_prompt})…")
    t0 = time.time()
    tt_prefill_argmax = srv.forward_prefill_chunked_tp(state, prompt_ids)
    log(f"  prefill {time.time() - t0:.1f}s; tt next-token argmax = "
        f"{tt_prefill_argmax}  hf decode_ids[0] = {decode_ids[0]}  "
        f"{'MATCH' if tt_prefill_argmax == decode_ids[0] else 'DIFFER'}")

    # Teacher-force decode_ids[0..N-1] through step_forward_v03.
    # capture["per_layer"]=True saves layer_h[layer_idx] per step.
    # decode_hidden[k] in HF corresponds to feeding decode_ids[k-1]
    # (k>=1) — TT step k feeds decode_ids[k-1] too, producing logits
    # whose argmax should be decode_ids[k].
    NUM_LAYERS = srv.NUM_LAYERS
    cos_layer_h = np.zeros((N, NUM_LAYERS), dtype=np.float32)
    cos_final_norm = np.zeros((N,), dtype=np.float32)
    cos_logits = np.zeros((N,), dtype=np.float32)
    argmax_match = np.zeros((N,), dtype=bool)
    tt_argmax_seq = np.zeros((N,), dtype=np.int32)

    log(f"ladder: {N} teacher-forced decode steps…")
    t0 = time.time()
    for k in range(N):
        input_tok = decode_ids[k]
        cap = {"per_layer": True}
        # cur_pos is implicit — step_forward_v03 + _set_pos handles it.
        # Production decode chains step_forward_v03 calls, which read
        # state.cur_pos_buf. The cur_pos was advanced by the prefill;
        # subsequent step_forward_v03 calls do not auto-advance, so
        # the production server's chat loop must call _set_pos between
        # steps. We mirror that.
        srv._set_pos(state, L_prompt + k)
        tt_argmax = srv.step_forward_v03(state, input_tok, capture=cap)
        tt_argmax_seq[k] = tt_argmax

        # HF reference for THIS step's output: decode_hidden[k+1, ...]
        # and decode_logits[k+1] (since HF step 0 reflects the prefill's
        # last position; step k+1 reflects feeding decode_ids[k]).
        hf_idx = k + 1 if k + 1 < N_HF else k
        hf_hidden_per_layer = decode_hidden[hf_idx]   # [L+1, HIDDEN]
        hf_logits_step = decode_logits[hf_idx]        # [VOCAB]

        # cap["layer_h"][L] is each layer's POST-layer hidden state
        # (HF idx 1..n_layers is also post-layer; idx 0 is embed).
        tt_layers = cap.get("layer_h", {})
        for L in range(NUM_LAYERS):
            tt_h = np.asarray(tt_layers[L]).reshape(-1).astype(np.float32)
            hf_h = hf_hidden_per_layer[L + 1].astype(np.float32).reshape(-1)
            cos_layer_h[k, L] = cosine(tt_h, hf_h)
        # final_norm = post-rms-norm pre-lm_head (HF's final layer
        # hidden state IS the final-norm output for Gemma 4).
        tt_final = np.asarray(cap["final_norm"]).reshape(-1).astype(np.float32)
        cos_final_norm[k] = cosine(tt_final, hf_hidden_per_layer[-1].reshape(-1))
        tt_logits = np.asarray(cap["logits"]).reshape(-1).astype(np.float32)
        cos_logits[k] = cosine(tt_logits, hf_logits_step)
        # argmax match: TT's next-token vs HF's decode_ids[k+1].
        hf_next = decode_ids[k + 1] if k + 1 < N_HF else decode_ids[-1]
        argmax_match[k] = (tt_argmax == hf_next)

        if k < 20 or k % 10 == 0:
            min_layer_cos = float(cos_layer_h[k].min())
            log(f"  step {k:3d}  "
                f"min_layer_cos={min_layer_cos:.6f}  "
                f"cos_final_norm={cos_final_norm[k]:.6f}  "
                f"cos_logits={cos_logits[k]:.6f}  "
                f"argmax tt={tt_argmax} hf={hf_next} "
                f"{'✓' if argmax_match[k] else '✗'}")
    log(f"ladder done in {time.time() - t0:.1f}s")

    # Verdict.
    print()
    print("══ Divergence onset " + "═" * 50)
    sub_threshold = np.where(cos_layer_h < args.threshold)
    first_step = first_layer = None
    if len(sub_threshold[0]) > 0:
        # Find the FIRST (step, layer) by step then layer ordering.
        first_idx = np.lexsort((sub_threshold[1], sub_threshold[0]))[0]
        first_step = int(sub_threshold[0][first_idx])
        first_layer = int(sub_threshold[1][first_idx])
        print(f"  FIRST cos < {args.threshold}:"
              f" step={first_step} layer={first_layer}"
              f" cos={cos_layer_h[first_step, first_layer]:.6f}")
        print(f"  argmax matches HF up to step "
              f"{int(np.argmin(argmax_match)) if not argmax_match.all() else N - 1}")
    else:
        print(f"  no per-layer cos sub-{args.threshold} in {N} steps — "
              f"the TT path tracks HF; bug is elsewhere (sampling? prefill?)")
    print(f"  cos_logits range = "
          f"[{cos_logits.min():.6f}, {cos_logits.max():.6f}]")
    print(f"  argmax-match rate = {int(argmax_match.sum())}/{N}")

    # Persist.
    np.savez(out_dir / "ladder.npz",
             cos_layer_h=cos_layer_h, cos_final_norm=cos_final_norm,
             cos_logits=cos_logits, argmax_match=argmax_match,
             tt_argmax_seq=tt_argmax_seq,
             hf_decode_ids=np.asarray(decode_ids, dtype=np.int32))
    verdict = {
        "N_steps": int(N),
        "threshold": float(args.threshold),
        "first_divergence_step": first_step,
        "first_divergence_layer": first_layer,
        "first_divergence_cos": (
            float(cos_layer_h[first_step, first_layer])
            if first_step is not None else None),
        "argmax_match_rate": int(argmax_match.sum()) / int(N),
        "cos_logits_min": float(cos_logits.min()),
        "cos_logits_max": float(cos_logits.max()),
        "min_layer_cos_per_step": cos_layer_h.min(axis=1).tolist(),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    log(f"wrote {out_dir}/ladder.npz + verdict.json")
    return 0 if first_step is None else 1


if __name__ == "__main__":
    sys.exit(main())
