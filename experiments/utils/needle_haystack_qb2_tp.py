#!/usr/bin/env python3
"""Needle-haystack retrieval probe — qb2 4-chip TP.

Validates that the B3 paged SDPA recipe (HiFi2 + no fp32_dest_acc, shipped
in state.sdpa_compute_kernel_config) actually kills the bf16 prefill drift
cliff on qb2 TP, the same way it did on qb1 single-chip (see
feedback_fp32_sdpa_cliff_probe.md).

Probe builds a haystack of TARGET_L total tokens (chat-rendered), inserts
a needle sentence "REMEMBER THIS: The magic password is XXXXXXXX." at
fractional position `frac`, then asks the model for the password. Y = full
8-char needle in output; P = ≥4-char contiguous substring; N = no.

Constraints:
  - MAX_POS=512 on qb2 → keep TARGET_L ≤ 460 and max_tokens=40 to fit.
  - Uses generate_tp endpoint (traced one-position-at-a-time prefill +
    decode). Both prefill and decode hit the same paged SDPA + B3 config.

Run:
  ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \\
      experiments/utils/needle_haystack_qb2_tp.py --lengths 460 --fracs 0.25,0.5,0.75 --trials 2'
"""
import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from experiments.serve import protocol as P  # noqa: E402

try:
    from transformers import AutoTokenizer
except Exception:
    print("transformers not importable; run via ~/tt-xla/.venv/bin/python", file=sys.stderr)
    sys.exit(2)

SOCK = os.path.expanduser("~/tt-xla/.cache/server_tp.sock")
MODEL_ID = "Qwen/Qwen3.6-27B"
OUT_DIR = os.path.expanduser("~/tt-xla/.cache/needle_haystack_qb2_tp")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
LOG_PATH = os.path.join(OUT_DIR, "log.txt")

DISTRACTOR_PARA = (
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


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def make_needle(seed: int) -> str:
    import random as _r
    rng = _r.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(8))


def apply_chat_no_think(tok, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)


def build_prompt(tok, total_tokens: int, frac: float, needle: str):
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count_rendered_tokens(user_text: str) -> int:
        return len(tok.encode(apply_chat_no_think(tok, user_text), add_special_tokens=False))

    n_para = max(1, total_tokens // 220 + 2)
    distractor = DISTRACTOR_PARA * n_para
    for _ in range(40):
        body = distractor + " " + needle_sentence + " " + question
        n = count_rendered_tokens(body)
        if n < total_tokens - 5:
            distractor = distractor + DISTRACTOR_PARA
        elif n > total_tokens + 5:
            cut = max(40, int(len(distractor) * 0.04))
            distractor = distractor[:-cut]
            if len(distractor) < 80:
                break
        else:
            break

    distractor_ids = tok.encode(distractor, add_special_tokens=False)
    insert_tok_idx = int(len(distractor_ids) * frac)
    prefix = tok.decode(distractor_ids[:insert_tok_idx], skip_special_tokens=True)
    suffix = tok.decode(distractor_ids[insert_tok_idx:], skip_special_tokens=True)
    user_body = prefix + " " + needle_sentence + " " + suffix + question
    rendered = apply_chat_no_think(tok, user_body)
    actual = len(tok.encode(rendered, add_special_tokens=False))
    return rendered, actual


def read_lines(sock):
    buf = bytearray()
    while True:
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            yield line
        chunk = sock.recv(65536)
        if not chunk:
            if buf:
                yield bytes(buf)
            return
        buf.extend(chunk)


def call_generate_tp(prompt: str, max_tokens: int, timeout: float = 900.0) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    s.sendall(P.pack_request("generate_tp", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "chunk_size": 1,
    }))
    final = None
    chunks = []
    try:
        for raw in read_lines(s):
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                final = {"error": obj.get("msg")}
                break
            if t == "chunk":
                chunks.append(obj.get("data", {}).get("token_text", ""))
            elif t == "result":
                final = obj.get("data", {})
                break
    finally:
        s.close()
    if final is None:
        final = {"error": "server closed before final"}
    if "generated_text" not in final:
        final["generated_text"] = "".join(chunks)
    return final


def score(text: str, needle: str) -> str:
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k + 4] in text:
            return "P"
    return "N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="460")
    ap.add_argument("--fracs", default="0.25,0.5,0.75")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    log(f"needle_haystack_qb2_tp — lengths={lengths} fracs={fracs} trials={args.trials}")
    log(f"loading tokenizer {MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    log("ok")

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
                final = call_generate_tp(prompt, args.max_tokens)
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
    by_L_frac = {}
    for c in results["cells"]:
        k = (c["L"], c["frac"])
        by_L_frac.setdefault(k, []).append(c["score"])
    for (L, frac), scores in sorted(by_L_frac.items()):
        y = sum(1 for s in scores if s == "Y")
        p = sum(1 for s in scores if s == "P")
        n = sum(1 for s in scores if s == "N")
        log(f"L={L} frac={frac}: Y={y} P={p} N={n}  ({''.join(scores)})")


if __name__ == "__main__":
    main()
