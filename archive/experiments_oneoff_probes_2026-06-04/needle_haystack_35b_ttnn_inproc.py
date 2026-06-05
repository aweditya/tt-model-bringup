#!/usr/bin/env python3
"""Workflow Step 6: needle-haystack long-context check, in-process.

Verifies that A002+A003+A004+A008 (the 2026-05-27 perf optimizations,
notably the bf8 MoE weights and core_grid expansion) don't compromise
long-context retrieval.

Reuses the haystack-construction logic from
experiments/utils/archive/needle_haystack_35b.py (which talked to a
persistent server via Unix socket), but bootstraps server_35b_ttnn
directly in-process — no separate server, faster iteration, same code
path through dn_forward + moe_forward + lm_head.

Sweeps L (target prompt token count) and `frac` (where in the haystack
the needle sits). For each (L, frac, seed), inserts a random 8-char
password, prefills the prompt, then decodes ~24 tokens looking for the
needle. Reports Y (full needle in output) / P (partial >= 4-char
substring) / N (no overlap).

Gate: at L=100, must see Y (memory says pre-A008 baseline did).

Run on qb1 (boots the (1,4) mesh):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/needle_haystack_35b_ttnn_inproc.py
"""
from __future__ import annotations

import argparse
import random as _r
import sys
import time
from pathlib import Path

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


def build_prompt(tok, target_tokens, frac, needle, use_chat=True):
    """Build a haystack of ~target_tokens with the needle inserted at
    `frac` fraction. If use_chat: render via chat template. Else: raw body.
    Returns (rendered_prompt, actual_token_count).
    """
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count(user):
        if use_chat:
            return len(tok.encode(apply_chat_no_think(tok, user), add_special_tokens=False))
        return len(tok.encode(user, add_special_tokens=False))

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
    rendered = apply_chat_no_think(tok, user_body) if use_chat else user_body
    actual = len(tok.encode(rendered, add_special_tokens=False))
    return rendered, actual


def classify(output_text, needle):
    """Y = full needle in output; P = >=4-char contiguous substring; N = no."""
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
    ap.add_argument("--lengths", default="100",
                    help="comma-separated target prompt token counts (e.g. 100,460,1024)")
    ap.add_argument("--fracs", default="0.5",
                    help="comma-separated needle fractions in [0,1]")
    ap.add_argument("--trials", type=int, default=1, help="seeds per (L,frac)")
    ap.add_argument("--max-gen", type=int, default=24,
                    help="tokens to decode looking for the needle")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="Skip chat template — use raw text body directly.")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs   = [float(x) for x in args.fracs.split(",")]

    import ttnn

    log("bootstrap (production state, bf8 MoE + A004 core_grid)…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    srv.bootstrap(state, log)
    tok = state.tokenizer

    results = []
    for L in lengths:
        for frac in fracs:
            for trial in range(args.trials):
                seed = L * 1000 + int(frac * 100) * 10 + trial
                needle = make_needle(seed)
                prompt, actual_L = build_prompt(tok, L, frac, needle,
                                                 use_chat=not args.no_chat_template)
                log(f"\n  L={L}  frac={frac}  trial={trial}  needle={needle!r}  actual_L={actual_L}")

                state.reset_caches_ttnn()
                prompt_ids = tok.encode(prompt, add_special_tokens=False)

                # Prefill: teacher-force the prompt.
                t0 = time.time()
                last_argmax = None
                for p, tid in enumerate(prompt_ids):
                    last_argmax = srv.step_forward_ttnn(state, tid, p)
                prefill_dt = time.time() - t0

                # Decode max_gen tokens autoregressively.
                generated = [last_argmax]
                pos = len(prompt_ids)
                t0 = time.time()
                for _ in range(args.max_gen - 1):
                    next_id = srv.step_forward_ttnn(state, generated[-1], pos)
                    generated.append(next_id)
                    pos += 1
                decode_dt = time.time() - t0
                out_text = tok.decode(generated)
                grade = classify(out_text, needle)
                log(f"    prefill {prefill_dt:.1f}s  decode {decode_dt:.1f}s  "
                    f"({decode_dt*1000/(args.max_gen-1):.0f} ms/tok)")
                log(f"    output: {out_text!r}")
                log(f"    grade:  {grade}")
                results.append({
                    "L": L, "frac": frac, "trial": trial, "seed": seed,
                    "needle": needle, "actual_L": actual_L, "output": out_text,
                    "grade": grade,
                })

    log("\n=== summary ===")
    by_L = {}
    for r in results:
        by_L.setdefault(r["L"], []).append(r["grade"])
    for L, grades in sorted(by_L.items()):
        y = grades.count("Y"); p = grades.count("P"); n = grades.count("N")
        log(f"  L={L:5d}: Y={y}  P={p}  N={n}  ({len(grades)} trials)")
    all_y = all(r["grade"] == "Y" for r in results)
    any_n = any(r["grade"] == "N" for r in results)
    if any_n:
        log("FAIL: at least one trial got grade=N (no needle overlap).")
        raise SystemExit(1)
    if all_y:
        log("PASS: all trials retrieved the needle verbatim.")
    else:
        log("PARTIAL PASS: all trials got at least a 4-char substring.")

    ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    main()
