#!/usr/bin/env python3
"""B15 — Needle-haystack long-context probe for 35B-A3B server_35b on qb1.

Ports `experiments/utils/needle_haystack_qb2_tp.py` to point at the
35B-A3B persistent server. Builds a chat-rendered haystack of L tokens
with an 8-char alphanumeric password inserted at fractional position,
then asks the model for the password. Y = full needle in output;
P = ≥4-char substring; N = no.

Usage:
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/needle_haystack_35b.py --lengths 100 --fracs 0.5 --trials 1'

For long-context sweep:
    --lengths 100,460,1024 --fracs 0.25,0.5,0.75 --trials 2
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import protocol as P  # noqa: E402

from transformers import AutoTokenizer

SOCK = PROJECT_ROOT / ".cache" / "server_35b.sock"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
OUT_DIR = PROJECT_ROOT / ".cache" / "needle_haystack_35b"
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
    import random as _r
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


def read_lines(s):
    buf = bytearray()
    while True:
        while True:
            nl = buf.find(b"\n")
            if nl < 0: break
            line = bytes(buf[:nl]); del buf[:nl+1]
            yield line
        chunk = s.recv(65536)
        if not chunk:
            if buf: yield bytes(buf)
            return
        buf.extend(chunk)


def call_generate(prompt, max_tokens, timeout=3600.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(SOCK))
    s.sendall(P.pack_request("generate_35b", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop_on_eos": False,  # needle haystack — keep generating to capture full needle
    }))
    final = None; chunks = []
    try:
        for raw in read_lines(s):
            if not raw: continue
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                final = {"error": obj.get("msg")}; break
            if t == "chunk":
                chunks.append(obj.get("data", {}).get("token_text", ""))
            elif t == "result":
                final = obj.get("data", {}); break
    finally:
        s.close()
    if final is None:
        final = {"error": "server closed before final"}
    if "generated_text" not in final:
        final["generated_text"] = "".join(chunks)
    return final


def score(text, needle):
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in text:
            return "P"
    return "N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="100")
    ap.add_argument("--fracs", default="0.5")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    log(f"needle_haystack_35b lengths={lengths} fracs={fracs} trials={args.trials}")
    log(f"loading tokenizer {MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    results = {"meta": {"model": MODEL_ID, "max_tokens": args.max_tokens,
                        "lengths": lengths, "fracs": fracs,
                        "trials": args.trials, "started": time.time()},
               "cells": []}

    for L in lengths:
        for frac in fracs:
            for trial in range(args.trials):
                seed = trial * 1000 + int(frac * 100) + L
                needle = make_needle(seed)
                prompt, actual = build_prompt(tok, L, frac, needle)
                log(f"\n=== L={L} frac={frac} trial={trial} seed={seed} ===")
                log(f"needle={needle}  actual_tokens={actual}")
                t0 = time.time()
                final = call_generate(prompt, args.max_tokens)
                wall = time.time() - t0
                gen = final.get("generated_text", "")
                err = final.get("error")
                s = score(gen, needle)
                log(f"wall={wall:.1f}s  score={s}  err={err}")
                log(f"  generated: {gen[:200]!r}")
                results["cells"].append({
                    "L": L, "frac": frac, "trial": trial, "seed": seed,
                    "needle": needle, "actual_tokens": actual,
                    "score": s, "wall_s": wall, "error": err,
                    "generated": gen,
                })
                with open(RESULTS_JSON, "w") as f:
                    json.dump(results, f, indent=2)

    log("\n=== SUMMARY ===")
    by = {}
    for c in results["cells"]:
        by.setdefault((c["L"], c["frac"]), []).append(c["score"])
    for (L, frac), scores in sorted(by.items()):
        y = sum(1 for s in scores if s == "Y")
        p = sum(1 for s in scores if s == "P")
        n = sum(1 for s in scores if s == "N")
        log(f"L={L} frac={frac}: Y={y} P={p} N={n}  ({''.join(scores)})")


if __name__ == "__main__":
    main()
