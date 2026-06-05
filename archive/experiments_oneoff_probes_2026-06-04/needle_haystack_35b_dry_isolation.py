#!/usr/bin/env python3
"""A009 Step 3 isolation — DRY+rep_penalty sampler on top-K logits.

Hypothesis: bf16 + greedy creates a fixed-point on "the" at long context.
Sampling with DRY + repetition_penalty breaks the fixed point. Memory
[feedback_drift_dry_rep_penalty] reports this fix +57% coherent chars on
27B (`--dry-multiplier 0.8 --repetition-penalty 1.1`).

Uses the new step_forward_ttnn_topk added to server_35b_ttnn.py:
  - returns top-K (K=64) logit values + indices per decode step
  - DRY + rep_penalty applied on host over the K candidates
  - argmax of the penalized values gives the next token

Compared to the broken greedy needle test:
  greedy:  output "The question is the the the the the..." (grade N)
  dry:     <hopefully retrieves the 'W79QHBGJ' needle>

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/needle_haystack_35b_dry_isolation.py
"""
from __future__ import annotations

import argparse
import random as _r
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


DISTRACTOR = (
    "The history of computing spans many centuries from the abacus to "
    "modern silicon chips. Early mechanical calculators like the "
    "Pascaline gave way to electromechanical machines and eventually "
    "to fully electronic computers. The transistor revolutionized the "
    "field in the late 1940s enabling much smaller and faster devices. "
    "Integrated circuits packed thousands then millions of transistors "
    "onto a single chip. Today modern processors contain billions of "
    "transistors and execute instructions in parallel across many cores. "
)
ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_needle(seed):
    rng = _r.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(8))


def apply_chat_no_think(tok, user):
    msgs = [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)


def build_prompt(tok, target_tokens, frac, needle):
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count(user):
        return len(tok.encode(apply_chat_no_think(tok, user), add_special_tokens=False))

    n_para = max(1, target_tokens // 220 + 2)
    distractor = DISTRACTOR * n_para
    for _ in range(40):
        body = distractor + " " + needle_sentence + " " + question
        n = count(body)
        if n < target_tokens - 5:
            distractor = distractor + DISTRACTOR
        elif n > target_tokens + 5:
            cut = max(40, int(len(distractor) * 0.04))
            distractor = distractor[:-cut]
            if len(distractor) < 80:
                break
        else:
            break

    d_ids = tok.encode(distractor, add_special_tokens=False)
    idx = int(len(d_ids) * frac)
    prefix = tok.decode(d_ids[:idx], skip_special_tokens=True)
    suffix = tok.decode(d_ids[idx:], skip_special_tokens=True)
    user_body = prefix + " " + needle_sentence + " " + suffix + question
    rendered = apply_chat_no_think(tok, user_body)
    actual = len(tok.encode(rendered, add_special_tokens=False))
    return rendered, actual


def topk_sampler_pick(
    top_vals, top_idxs,
    history,
    repetition_penalty=1.1,
    dry_multiplier=0.8,
    dry_base=1.75,
    dry_allowed_length=2,
):
    """Pick next token id from top-K (values + indices) with DRY +
    repetition_penalty applied. Deterministic (greedy argmax of penalized
    values). Mirrors server.py:_pick_next_token but restricted to top-K.
    """
    vals = top_vals.astype(np.float64).copy()
    idxs = top_idxs.astype(np.int64)

    # Step 1: repetition_penalty on top-K indices that appear in history.
    if repetition_penalty != 1.0 and history:
        recent = set(int(x) for x in history)
        for i, tid in enumerate(idxs):
            if int(tid) in recent:
                v = vals[i]
                # CTRL: |v| moves toward 0.
                vals[i] = v * repetition_penalty if v < 0 else v / repetition_penalty

    # Step 1c: DRY — for each candidate t in top-K, find max L such that
    # history[-L:] + [t] appears in history. If L >= allowed_length, penalize.
    if dry_multiplier > 0.0 and history and len(history) >= dry_allowed_length:
        L_max = len(history)
        h = list(history)
        # For each candidate token t in top-K, compute its DRY penalty:
        # Walk every position i in [0, L_max-2], compute longest k such that
        # h[i-k+1 : i+1] == h[L_max-k : L_max] AND h[i+1] == t.
        # If k+1 >= allowed: ext = (k+1) - allowed; penalty = mult * base^ext.
        cand_set = set(int(x) for x in idxs)
        token_match_len = {}
        suffix_cap = min(L_max, 50)
        for i in range(L_max - 1):
            k = 0
            while k < suffix_cap and (i - k) >= 0 and h[i - k] == h[L_max - 1 - k]:
                k += 1
            if k >= dry_allowed_length:
                t = int(h[i + 1])
                if t in cand_set:
                    ext = (k + 1) - dry_allowed_length
                    if ext > token_match_len.get(t, 0):
                        token_match_len[t] = ext
        if token_match_len:
            id_to_pos = {int(x): pos for pos, x in enumerate(idxs)}
            for t, ext in token_match_len.items():
                penalty = dry_multiplier * (dry_base ** ext)
                vals[id_to_pos[t]] -= penalty

    # Greedy argmax of penalized values.
    best = int(np.argmax(vals))
    return int(idxs[best])


def classify(output_text, needle):
    if needle in output_text:
        return "Y"
    for n in range(8, 3, -1):
        for i in range(0, 9 - n):
            sub = needle[i:i+n]
            if sub in output_text:
                return "P"
    return "N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="100")
    ap.add_argument("--fracs", default="0.5")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-gen", type=int, default=24)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--dry-multiplier", type=float, default=0.8)
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs   = [float(x) for x in args.fracs.split(",")]

    import ttnn

    log(f"bootstrap (bf8 MoE + A004 core_grid + DRY sampler K={args.k})…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    srv.bootstrap(state, log)
    tok = state.tokenizer

    for L in lengths:
        for frac in fracs:
            for trial in range(args.trials):
                seed = L * 1000 + int(frac * 100) * 10 + trial
                needle = make_needle(seed)
                prompt, actual_L = build_prompt(
                    tok, L, frac, needle) if not args.no_chat_template else (
                    None, None)
                if args.no_chat_template:
                    # Build raw body (no chat render).
                    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
                    distractor = DISTRACTOR * max(1, L // 220 + 2)
                    d_ids = tok.encode(distractor, add_special_tokens=False)
                    idx = int(len(d_ids) * frac)
                    prefix = tok.decode(d_ids[:idx], skip_special_tokens=True)
                    suffix = tok.decode(d_ids[idx:], skip_special_tokens=True)
                    prompt = (prefix + " " + needle_sentence + " " + suffix +
                              "\n\nBased only on the document above, what is the magic "
                              "password? Answer with only the 8-character password.\n\nAnswer: ")
                    actual_L = len(tok.encode(prompt, add_special_tokens=False))
                log(f"\n  L={L}  frac={frac}  trial={trial}  needle={needle!r}  actual_L={actual_L}")

                state.reset_caches_ttnn()
                prompt_ids = tok.encode(prompt, add_special_tokens=False)

                # Prefill (still greedy/argmax — only the last decoded token
                # matters as the start of generation).
                t0 = time.time()
                last_argmax = None
                for p, tid in enumerate(prompt_ids):
                    last_argmax = srv.step_forward_ttnn(state, tid, p)
                prefill_dt = time.time() - t0

                # Decode with top-K + DRY + rep_penalty.
                generated = []
                # We seed `history` for DRY with the prompt tokens — gives
                # DRY visibility into chat-template "the the" patterns.
                history = list(prompt_ids)
                pos = len(prompt_ids)

                # First, evaluate the first decoded token (last_argmax) with sampler.
                # But step_forward_ttnn already consumed it; we have last_argmax.
                # Re-pick last_argmax via top-K sampler at position pos-1?
                # Simpler approach: generated[0] is the model's first answer
                # after the prompt. To incorporate DRY/rep_penalty we'd need
                # to re-run that step. For now, start sampling from step 1.
                generated.append(last_argmax)
                history.append(last_argmax)

                t0 = time.time()
                # Decode max_gen - 1 more tokens using top-K sampler.
                for _ in range(args.max_gen - 1):
                    top_vals, top_idxs = srv.step_forward_ttnn_topk(
                        state, generated[-1], pos, k=args.k)
                    next_id = topk_sampler_pick(
                        top_vals, top_idxs, history,
                        repetition_penalty=args.rep_penalty,
                        dry_multiplier=args.dry_multiplier,
                    )
                    generated.append(next_id)
                    history.append(next_id)
                    pos += 1
                decode_dt = time.time() - t0
                out_text = tok.decode(generated)
                grade = classify(out_text, needle)
                log(f"    prefill {prefill_dt:.1f}s  decode {decode_dt:.1f}s  "
                    f"({decode_dt*1000/(args.max_gen-1):.0f} ms/tok)")
                log(f"    output: {out_text!r}")
                log(f"    grade:  {grade}")

    ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    main()
