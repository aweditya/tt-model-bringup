#!/usr/bin/env python3
"""v0.4 — EAGER needle-haystack retrieval at L=128/512/1024.

Round 9 escalation companion to `gm4_v04_needle_haystack_traced.py`.
Step A revealed that the long-context regression is INDEPENDENT of the
Round-8 bfp8 swap (a full bf16 revert produces bit-identical
attractor-loop output to Round 8 at L=128). The remaining unexamined
differences between the failing v0.4 traced probe and the passing pre-
Round-8 v0.3.3.c baseline (`gm4_v033c_needle_haystack.py`, 3/3 Y @
L=100/256/512) are:
  1. EAGER (step_forward_v031) vs TRACED (step_forward_traced)
  2. BASE model (`google/gemma-4-12B`) vs IT (`google/gemma-4-12B-it`)
  3. 1 trial/length vs 3 trials/length
  4. add_special_tokens=True (both probes use this — not a difference)

This probe holds (2), (3), (4) constant — IT model, 3 trials, special
tokens on — and flips ONLY (1) back to EAGER. If retrieval recovers,
the regression is the trace path. If it doesn't, it's IT-model behaviour
(prompt understanding / template parroting at long context, not a
precision bug).

Run (qb2 stand-alone — bootstraps fresh):
  TT_GM4_NEEDLE_OUT_SUBDIR=round9_eager_ablation \\
    bash scripts/_needle_haystack_qb2_runner.sh --probe eager
      (the wrapper script doesn't pass --probe; invoke this file directly
       via `bash scripts/run_remote_qb2.sh
            experiments/cb/isolate/gm4_v04_needle_haystack_eager.py
            --lengths 128,512,1024 --trials 3`)
"""
from __future__ import annotations

import json
import os
import random as _r
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402

# Reuses the v04 traced probe's output-subdir hook.
_OUT_BASE = PROJECT_ROOT / "research" / "gemma4_perf_qb2_2026-06-05" / "needle_haystack"
_subdir = os.environ.get("TT_GM4_NEEDLE_OUT_SUBDIR", "").strip()
OUT_DIR = _OUT_BASE / _subdir if _subdir else _OUT_BASE
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


def build_prompt(tok, target_tokens, frac, needle):
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count(s):
        return len(tok.encode(s, add_special_tokens=True))

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
    body = prefix + " " + needle_sentence + " " + suffix + question
    n_actual = count(body)
    return body, n_actual


def score(text, needle):
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in text:
            return "P"
    return "N"


def generate_one_eager(state, tok, prompt_text, max_new_tokens):
    """Teacher-force then free-run via the EAGER step path."""
    prompt_ids = tok.encode(prompt_text, add_special_tokens=True)
    t0 = time.time()
    last_argmax = None
    for p, tid in enumerate(prompt_ids):
        last_argmax = srv.step_forward_v031(state, int(tid), p)
    prefill_s = time.time() - t0

    generated = [last_argmax]
    pos = len(prompt_ids)
    t0 = time.time()
    for _ in range(max_new_tokens - 1):
        next_id = srv.step_forward_v031(state, int(generated[-1]), pos)
        generated.append(next_id)
        pos += 1
    decode_s = time.time() - t0
    text = tok.decode(generated, skip_special_tokens=True)
    return text, prefill_s, decode_s, len(prompt_ids)


def main(state=None, lengths=(128, 512, 1024), fracs=(0.5,), trials=3,
         max_new=24, seed=1337):
    from transformers import AutoTokenizer
    # IT variant — same as the traced probe; only the decode path differs.
    tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")

    owned_state = state is None
    if owned_state:
        log("bootstrapping Gemma 4 12B IT server (~85s)…")
        t0 = time.time()
        state = srv.State()
        srv.bootstrap(state, log=log)
        log(f"bootstrap in {time.time()-t0:.1f}s")
    else:
        log("using pre-bootstrapped state from harness")

    log("EAGER path — no trace capture")

    results = []
    for L in lengths:
        for f in fracs:
            for t in range(trials):
                needle = make_needle(seed + 1000*list(lengths).index(L) + 100*int(f*100) + t)
                prompt, n_prompt = build_prompt(tok, L, f, needle)
                log(f"=== L={L} frac={f} trial={t} needle={needle} (rendered={n_prompt} toks) ===")
                if n_prompt + max_new > srv.MAX_KV:
                    log(f"  ! SKIP — n_prompt+max_new={n_prompt+max_new} > MAX_KV={srv.MAX_KV}")
                    continue
                text, p_s, d_s, n_prompt_actual = generate_one_eager(state, tok, prompt, max_new)
                verdict = score(text, needle)
                log(f"  prefill {p_s:.1f}s ({p_s*1000/n_prompt_actual:.0f} ms/tok), "
                    f"decode {d_s:.1f}s ({d_s*1000/max_new:.0f} ms/tok)")
                log(f"  generated: {text!r}")
                log(f"  verdict: {verdict}")
                results.append({
                    "length": L, "frac": f, "trial": t,
                    "needle": needle, "prompt_tokens": n_prompt_actual,
                    "max_new": max_new, "generated_text": text,
                    "verdict": verdict,
                    "prefill_seconds": p_s, "decode_seconds": d_s,
                })

    by_length = {}
    for r in results:
        by_length.setdefault(r["length"], []).append(r["verdict"])
    log("=== SUMMARY ===")
    for L, vs in sorted(by_length.items()):
        ys = vs.count("Y"); ps = vs.count("P"); ns = vs.count("N")
        log(f"  L={L}: Y={ys} P={ps} N={ns} / {len(vs)}")

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log(f"wrote {RESULTS_JSON}")
    log("done")

    if owned_state:
        import ttnn
        ttnn.close_device(state.mesh)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="128,512,1024",
                    help="comma-separated context lengths to test")
    ap.add_argument("--fracs", default="0.5",
                    help="comma-separated needle insertion fractions")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    sys.exit(main(
        lengths=tuple(int(x) for x in args.lengths.split(",")),
        fracs=tuple(float(x) for x in args.fracs.split(",")),
        trials=args.trials,
        max_new=args.max_new,
        seed=args.seed,
    ) or 0)
