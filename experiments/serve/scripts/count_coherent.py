#!/usr/bin/env python3
"""Count coherent tokens before drift in a generate_long output file.

A run is 'coherent' until the model collapses into a short repeating cycle.
Heuristics, in order, looking at the streamed token text + decoded text:

  1. Find the body between the prompt header and the result-summary footer.
  2. Walk character by character keeping a 60-char rolling window — flag the
     first window where ≥75% of chars are in {space, newline, '/'} OR a single
     2-char string fills > 30 chars of the window. That's the drift point.
  3. Report: collapse_char_pos, approx coherent text length, snippet around
     collapse, first 300 chars of generated body.

Plus we extract n_generated_tokens / ms_per_tok from the final summary if
present.

Usage:
  python experiments/serve/scripts/count_coherent.py <file> [<file>...]
"""
import re
import sys


HEADER_RE = re.compile(r"^-{20,}$", re.M)
FOOTER_RE = re.compile(r"^=+\n\s*prompt:", re.M)  # not used; we slice differently


def extract_body(text: str) -> tuple[str, dict]:
    """Return (generated_body_text, summary_dict)."""
    # Body is between the second '---' line and the '===' footer before stats.
    lines = text.splitlines()
    # find indices of separator lines
    sep_idx = [i for i, ln in enumerate(lines) if ln.startswith("------") or ln.startswith("======")]
    if len(sep_idx) < 3:
        return text, {}
    body_start = sep_idx[1] + 1  # after first '---' (which is below 'prompt: ...')
    body_end = sep_idx[2]
    body = "\n".join(lines[body_start:body_end])
    # Stats lines come after the footer ===
    summary = {}
    for ln in lines[body_end + 1:]:
        m = re.search(r"prompt:\s+(\d+)\s+tokens,\s+prefill\s+([\d.]+)\s+ms", ln)
        if m:
            summary["n_prompt_tokens"] = int(m.group(1))
            summary["prefill_ms"] = float(m.group(2))
        m = re.search(r"generated:\s+(\d+)\s+tokens,\s+decode\s+([\d.]+)\s+ms/tok\s+=\s+([\d.]+)\s+tok/s", ln)
        if m:
            summary["n_generated_tokens"] = int(m.group(1))
            summary["ms_per_tok"] = float(m.group(2))
            summary["tok_per_sec"] = float(m.group(3))
        m = re.search(r"total wall:\s+([\d.]+)\s+ms", ln)
        if m:
            summary["total_ms"] = float(m.group(1))
    return body, summary


def find_collapse(body: str, win: int = 60) -> int:
    """Return character index where collapse begins, or len(body) if no collapse."""
    if len(body) < win:
        return len(body)
    deg_chars = set(" \n\t/")
    for start in range(len(body) - win):
        window = body[start:start + win]
        # Test 1: ≥75% of window is whitespace+/
        deg_count = sum(1 for c in window if c in deg_chars)
        if deg_count >= int(0.75 * win):
            return start
        # Test 2: a single 2-char substring fills > 30 chars
        bigrams = {}
        for i in range(len(window) - 1):
            bg = window[i:i+2]
            bigrams[bg] = bigrams.get(bg, 0) + 1
        if max(bigrams.values()) > 30:
            return start
        # Test 3: a single 3-char substring fills > 20 windows
        trigrams = {}
        for i in range(len(window) - 2):
            tg = window[i:i+3]
            trigrams[tg] = trigrams.get(tg, 0) + 1
        if max(trigrams.values()) > 18:
            return start
    return len(body)


def analyze(path: str) -> None:
    with open(path) as f:
        text = f.read()
    body, summary = extract_body(text)
    n_gen = summary.get("n_generated_tokens", "?")
    ms_per_tok = summary.get("ms_per_tok", "?")
    tps = summary.get("tok_per_sec", "?")
    collapse = find_collapse(body)
    print(f"=== {path} ===")
    print(f"  generated_text chars: {len(body)}")
    print(f"  collapse char pos:    {collapse}")
    print(f"  approx coherent chars: {collapse if collapse < len(body) else 'no collapse detected'}")
    print(f"  n_generated_tokens:   {n_gen}   ms/tok={ms_per_tok}   tok/s={tps}")
    print(f"  --- first 300 chars of body ---")
    print(body[:300].replace("\n", "\\n"))
    print(f"  --- chars around collapse [{max(0, collapse-50)}:{collapse+50}] ---")
    if collapse < len(body):
        print(body[max(0, collapse - 50):collapse + 50].replace("\n", "\\n"))
    else:
        print("(no collapse detected; full body coherent)")
    print()


def main():
    if len(sys.argv) < 2:
        print("usage: count_coherent.py <file> [<file>...]")
        sys.exit(2)
    for path in sys.argv[1:]:
        try:
            analyze(path)
        except FileNotFoundError:
            print(f"  (not found): {path}")


if __name__ == "__main__":
    main()
