#!/usr/bin/env python3
"""Print a sample constructed needle-haystack prompt for inspection.

Used to debug prompt construction independently of running the full grid.

Run:
    cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/needle_haystack_inspect.py --L 256 --frac 0.5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from transformers import AutoTokenizer  # noqa: E402
from experiments.utils.needle_haystack_probe import build_prompt, make_needle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--show-chars", type=int, default=400)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    seed = args.L * 1000 + int(args.frac * 100)
    needle = make_needle(seed)
    print(f"NEEDLE: {needle}")
    prompt, ofs, tot = build_prompt(tok, args.L, args.frac, needle)
    print(f"TOTAL TOKENS (chat-templated): {tot}")
    print(f"NEEDLE OFFSET TOKENS: {ofs} ({ofs/tot:.0%})")
    print(f"PROMPT LENGTH (chars): {len(prompt)}")
    print(f"NEEDLE IN PROMPT?  {needle in prompt}")
    if needle in prompt:
        i = prompt.find(needle)
        print(f"  found at char {i}: {prompt[max(0,i-50):i+50]!r}")
    else:
        # Maybe needle was split during tokenize+decode roundtrip; search for shorter prefix
        for k in [6, 5, 4, 3]:
            for start in range(len(needle) - k + 1):
                sub = needle[start:start+k]
                if sub in prompt:
                    i = prompt.find(sub)
                    print(f"  partial '{sub}' (k={k}) found at char {i}: "
                          f"{prompt[max(0,i-50):i+50]!r}")
                    break
            else:
                continue
            break
    print("=" * 70)
    print(f"FIRST {args.show_chars} CHARS:")
    print(prompt[:args.show_chars])
    print("=" * 70)
    print(f"LAST {args.show_chars} CHARS:")
    print(prompt[-args.show_chars:])
    print("=" * 70)
    # build_prompt already returns the chat-rendered string (enable_thinking=False).
    print(f"CHAT-RENDERED FIRST 300 CHARS (no-think):")
    print(prompt[:300])
    print("=" * 70)
    print(f"CHAT-RENDERED LAST 300 CHARS (no-think):")
    print(prompt[-300:])


if __name__ == "__main__":
    main()
