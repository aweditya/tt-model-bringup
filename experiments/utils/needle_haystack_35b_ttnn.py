#!/usr/bin/env python3
"""Needle-haystack long-context probe for Qwen3.6-35B-A3B via server_35b_ttnn
(direct import, no socket). Tests whether per-position cosine drift translates
to user-facing UX loss — i.e. can the model retrieve a known fact from a long
chat-rendered prompt?

Direct port of `experiments/utils/needle_haystack_35b.py` but bypasses the
persistent `server_35b.py` socket; runs `server_35b_ttnn.step_forward_ttnn`
directly. Honors state.attn_mode (default sdpa).

Run (qb1, ttnn env exported, no resident server):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/needle_haystack_35b_ttnn.py \\
      --lengths 100,256 --fracs 0.5 --trials 1 \\
      [--attn-mode sdpa]   # default
"""
import argparse
import json
import os
import random as _r
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
sys.stdout.reconfigure(line_buffering=True)

import server_35b_ttnn as srv  # noqa: E402

OUT_DIR = PROJECT_ROOT / ".cache" / "needle_haystack_35b_ttnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "log.txt"
RESULTS_JSON = OUT_DIR / "results.json"

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
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


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


def score(text, needle):
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in text:
            return "P"
    return "N"


def generate_one(state, prompt_text, max_new_tokens):
    """Prefill the prompt then autoregressively decode max_new_tokens.
    Returns (generated_text, prefill_seconds, decode_seconds, n_prompt).
    """
    prompt_ids = state.tokenizer.encode(prompt_text)
    state.reset_caches_ttnn()

    t0 = time.time()
    last_argmax = None
    for p, tid in enumerate(prompt_ids):
        last_argmax = srv.step_forward_ttnn(state, tid, p)
    prefill_s = time.time() - t0

    generated = [last_argmax]
    pos = len(prompt_ids)
    t0 = time.time()
    for _ in range(max_new_tokens - 1):
        next_id = srv.step_forward_ttnn(state, generated[-1], pos)
        generated.append(next_id)
        pos += 1
    decode_s = time.time() - t0

    text = state.tokenizer.decode(generated, skip_special_tokens=True)
    return text, prefill_s, decode_s, len(prompt_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="100",
                    help="comma-separated target prompt token counts")
    ap.add_argument("--fracs", default="0.5",
                    help="comma-separated needle positions (0..1)")
    ap.add_argument("--trials", type=int, default=1,
                    help="trials per (length, frac); each uses a different needle seed")
    ap.add_argument("--max-new", type=int, default=24,
                    help="tokens to decode after the prompt")
    ap.add_argument("--attn-mode", choices=["manual", "sdpa"], default="sdpa")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-prompt-file", default=None,
                    help="if set, write the constructed prompt to this path and exit "
                         "(useful for feeding to hf_reference_35b.py via --prompt-file). "
                         "Requires --lengths and --fracs to be a single value each.")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    if args.save_prompt_file is not None:
        # Tokenizer-only path: no mesh bootstrap. Construct prompt, save, exit.
        # This lets us feed bit-identical prompt to hf_reference_35b.py for
        # per-layer drift localization.
        if len(lengths) != 1 or len(fracs) != 1 or args.trials != 1:
            raise ValueError("--save-prompt-file requires single --lengths, --fracs, --trials=1")
        from transformers import AutoTokenizer
        log(f"tokenizer-only mode for --save-prompt-file…")
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")
        needle = make_needle(args.seed + 1000*0 + 100*int(fracs[0]*100) + 0)
        prompt, n_prompt = build_prompt(tok, lengths[0], fracs[0], needle)
        from pathlib import Path as _P
        _P(args.save_prompt_file).write_text(prompt)
        # Also save the needle + metadata next to it for downstream scoring
        _P(args.save_prompt_file + ".needle").write_text(needle)
        log(f"  needle={needle} prompt_tokens={n_prompt}")
        log(f"  wrote prompt → {args.save_prompt_file}")
        log(f"  wrote needle → {args.save_prompt_file}.needle")
        return

    log(f"bootstrap (attn_mode={args.attn_mode})…")
    state = srv.State()
    state.attn_mode = args.attn_mode
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"bootstrap in {time.time()-t0:.1f}s")

    # Warmup to populate JIT (SDPA mode has a known JIT race on the first call;
    # see feedback_35b_a3b_sdpa_swap_result.md).
    state.reset_caches_ttnn()
    log("warmup forward to populate JIT caches…")
    t0 = time.time()
    _ = srv.step_forward_ttnn(state, int(state.tokenizer.encode("Hello")[0]), 0)
    log(f"warmup done in {time.time()-t0:.1f}s")

    results = []
    for L in lengths:
        for f in fracs:
            for t in range(args.trials):
                needle = make_needle(args.seed + 1000*lengths.index(L) + 100*int(f*100) + t)
                prompt, n_prompt = build_prompt(state.tokenizer, L, f, needle)
                log(f"\n=== L={L} frac={f} trial={t} needle={needle} (rendered={n_prompt} toks) ===")
                text, p_s, d_s, n_prompt_actual = generate_one(state, prompt, args.max_new)
                verdict = score(text, needle)
                log(f"  prefill {p_s:.1f}s ({p_s*1000/n_prompt_actual:.0f} ms/tok), "
                    f"decode {d_s:.1f}s ({d_s*1000/args.max_new:.0f} ms/tok)")
                log(f"  generated: {text!r}")
                log(f"  verdict: {verdict}")
                results.append({
                    "length": L, "frac": f, "trial": t,
                    "needle": needle, "prompt_tokens": n_prompt_actual,
                    "max_new": args.max_new,
                    "generated_text": text,
                    "verdict": verdict,
                    "prefill_seconds": p_s,
                    "decode_seconds": d_s,
                })

    # Aggregate
    by_length = {}
    for r in results:
        by_length.setdefault(r["length"], []).append(r["verdict"])
    log("\n=== SUMMARY ===")
    for L, vs in sorted(by_length.items()):
        ys = vs.count("Y"); ps = vs.count("P"); ns = vs.count("N")
        log(f"  L={L}: Y={ys} P={ps} N={ns} / {len(vs)}")

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log(f"wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
