#!/usr/bin/env python3
"""
Needle-in-a-haystack retrieval probe for qb1 Qwen3.6-27B persistent server.

Question: at what (total_length, needle_position) does the model lose the
ability to retrieve a single short fact buried in distractor context?

Distinct from:
  - Agent I's cosine-ladder probe (teacher-forced precision)
  - Agent A's sampling probe (drift in autoregressive generation)
This probe measures: when generation is unconstrained but the answer is
short and copy-able from context, does the model attend to the right
context tokens?

Design:
  - Distractor = repetitive but coherent Wikipedia-geography prose
  - Needle    = "The magic password is XXXXXXXX." (XXXXXXXX = unique
                8-char alphanumeric, NEVER a real word, chosen so the
                tokenizer turns it into 4+ tokens that the model must
                copy verbatim)
  - Query     = "What is the magic password? Reply with just the
                 8-character password."
  - Insertion at fraction frac of distractor by token count
  - Total prompt token target = L ∈ {256, 512, 1024, 2048, 4096}
  - frac ∈ {0.25, 0.50, 0.75, 0.95}

For each (L, frac):
  1. Tokenize-iterate to land prompt within ±3% of L tokens
  2. Send via `generate_long` with --chat, DRY+rp sampling, max_tokens=20,
     max_pos = round_up(L + 64, 64)
  3. Parse generated text for the needle
  4. Record Y / P / N + wall time + n_prompt_tokens + n_generated_tokens

Drives the qb1 server over its UDS socket — same chip-lock-avoidance
pattern as cosine_ladder_tt_probe.py.

Run:
    cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/needle_haystack_probe.py
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
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# --- Output paths -----------------------------------------------------------
OUT_DIR = os.path.expanduser("~/tt-xla/.cache")
RESULTS_JSON = os.path.join(OUT_DIR, "needle_haystack_results.json")
LOG_PATH = os.path.join(OUT_DIR, "needle_haystack_log.txt")


# --- Distractor corpus ------------------------------------------------------
# Coherent Wikipedia-style geography prose. Repeats are deliberate so we
# can build any token length; the model has surely seen this style before
# so the "background" is low-perplexity (good — we want the model NOT to
# be confused by garbled text; we want it to find a sharp needle in
# familiar prose).
DISTRACTOR_PARA = (
    "France is a country located in Western Europe. Its capital and largest "
    "city is Paris, situated on the Seine River. The country is bordered by "
    "Belgium and Luxembourg to the north, Germany and Switzerland to the "
    "east, Italy and Monaco to the southeast, and Spain and Andorra to the "
    "south. France has a long coastline along the Atlantic Ocean to the west "
    "and the Mediterranean Sea to the south. The country covers an area of "
    "approximately 643,801 square kilometers, including its overseas "
    "territories. The terrain is varied, with mountains in the Alps and "
    "Pyrenees, plains in the north, and a Mediterranean coastline in the "
    "south. The climate ranges from oceanic in the west to continental in "
    "the east, with a Mediterranean climate along the southern coast. "
    "France has a population of over 67 million people, making it one of "
    "the most populous countries in Europe. The official language is "
    "French, and the currency is the Euro. The country is a founding "
    "member of the European Union and the United Nations. "
)


# --- Needle generation ------------------------------------------------------
# Deterministic per-trial needle: hash of (L, frac, trial_idx) → 8 chars
# from a restricted alphabet that the Qwen BPE tokenizer is unlikely to
# merge with neighbors. We avoid lowercase 'e','t','a','o','i' which form
# common BPE pairs; we use uppercase letters + digits.
ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"  # 28 chars

def make_needle(seed: int) -> str:
    import random as _r
    rng = _r.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(8))


# --- Token budgeting --------------------------------------------------------
def apply_chat_no_think(tok, user_content: str) -> str:
    """Render the chat template with enable_thinking=False so the model
    answers directly without emitting a `<think>...</think>` block.
    Qwen3.6's template emits `<think>\\n\\n</think>\\n\\n` as the generation
    prefix in that mode — the model continues from there as if it's done
    thinking. Returns a STRING (not token ids); we send raw text with
    chat=False to the server."""
    msgs = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)


def build_prompt(tok, total_tokens: int, frac: float, needle: str
                  ) -> tuple[str, int, int]:
    """Build a chat-rendered prompt of approximately `total_tokens` tokens
    with the needle sentence at fractional position `frac` of the distractor.

    Uses enable_thinking=False so the model answers directly without a
    long <think>...</think> preamble (which would chew the 20-token budget
    on filler).

    Returns (rendered_prompt_text, needle_token_pos_estimate, actual_tokens).
    Token count is from `tok.encode(rendered, add_special_tokens=False)` —
    the rendered string already contains all special tokens.
    """
    # NIAH-style: present a document, then ask the question explicitly.
    # The needle is a distinctive sentence; the question explicitly references
    # "the magic password" so the model knows what to retrieve.
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count_rendered_tokens(user_text: str) -> int:
        rendered = apply_chat_no_think(tok, user_text)
        return len(tok.encode(rendered, add_special_tokens=False))

    # Build a distractor body sized so the FULL chat-templated prompt
    # (distractor + needle + question + template overhead) lands near
    # `total_tokens`. We iterate (the chat template adds ~17 tokens and
    # the question adds ~25, so we shrink the distractor accordingly).
    # Estimate paragraphs needed for the distractor body: roughly 220 tokens
    # per paragraph. We start oversized and shrink.
    n_para = max(1, total_tokens // 220 + 2)
    distractor = DISTRACTOR_PARA * n_para
    # Iteratively grow / shrink distractor until full prompt lands near target.
    iters = 0
    while iters < 40:
        body = distractor + " " + needle_sentence + " " + question
        n = count_rendered_tokens(body)
        if n < total_tokens - 5:
            distractor = distractor + DISTRACTOR_PARA
        elif n > total_tokens + 5:
            # Trim 4% of characters; works because tokens-per-char is roughly
            # uniform in this prose.
            cut = max(40, int(len(distractor) * 0.04))
            distractor = distractor[:-cut]
            if len(distractor) < 80:
                break
        else:
            break
        iters += 1

    # Now we have a `distractor` string sized to fill total_tokens.
    # Insert needle at fraction `frac` of the distractor's tokens.
    distractor_ids = tok.encode(distractor, add_special_tokens=False)
    insert_tok_idx = int(len(distractor_ids) * frac)
    # Split distractor by tokens, then decode pieces back to text.
    prefix_ids = distractor_ids[:insert_tok_idx]
    suffix_ids = distractor_ids[insert_tok_idx:]
    prefix_text = tok.decode(prefix_ids, skip_special_tokens=True)
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)

    # Final user-message body: prefix + needle + suffix + question
    user_body = (prefix_text + " " + needle_sentence + " "
                  + suffix_text + question)
    # Apply chat template (enable_thinking=False) to get the actual prompt
    # we send to the server as raw text.
    rendered = apply_chat_no_think(tok, user_body)
    actual_tokens = len(tok.encode(rendered, add_special_tokens=False))
    # Position of needle (as a fraction of full prompt) for logging
    rendered_until_needle = apply_chat_no_think(
        tok, prefix_text + " " + needle_sentence)
    needle_offset_tokens = len(tok.encode(rendered_until_needle,
                                           add_special_tokens=False))
    return rendered, needle_offset_tokens, actual_tokens


# --- Server RPC -------------------------------------------------------------
def _read_all_lines(sock):
    """Buffered line-iterator over a stream socket. Yields one JSON line at a
    time. Robust to multiple lines per recv. The repo's protocol.read_line
    has a bug: it discards everything after the first newline in a single
    recv, which loses tokens when the server packs chunk+result tightly."""
    buf = bytearray()
    while True:
        # Drain any complete lines from the buffer first.
        while True:
            nl = buf.find(b"\n")
            if nl < 0:
                break
            line = bytes(buf[:nl])
            del buf[:nl + 1]
            yield line
        # Buffer empty of complete lines; read more.
        chunk = sock.recv(65536)
        if not chunk:
            # Flush any final partial line, then stop.
            if buf:
                yield bytes(buf)
            return
        buf.extend(chunk)


def stream_generate_long(prompt: str, max_pos: int, max_tokens: int,
                          dry_multiplier: float = 0.0,
                          repetition_penalty: float = 1.0,
                          chat: bool = False,
                          timeout: float = 1200.0) -> dict:
    """Stream-call the server's generate_long. Returns the final result dict."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(P.SOCKET_PATH)
    req = P.pack_request("generate_long", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "max_pos": max_pos,
        "block_size": 64,
        "chunk_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": 0,
        "dry_multiplier": dry_multiplier,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "seed": 0,
        "chat": chat,
        "system": "",
    })
    sock.sendall(req)
    final = None
    chunks_text = []
    try:
        for raw in _read_all_lines(sock):
            if not raw:
                continue
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                final = {"error": obj.get("msg")}
                break
            if t == "chunk":
                chunks_text.append(obj.get("data", {}).get("token_text", ""))
            elif t == "result":
                final = obj.get("data", {})
                break
    finally:
        sock.close()
    if final is None:
        final = {"error": "server closed before final"}
    if "generated_text" not in final:
        final["generated_text"] = "".join(chunks_text)
    return final


# --- Needle scoring ---------------------------------------------------------
def score_needle(generated_text: str, needle: str) -> str:
    """Return 'Y' (full match), 'P' (partial: 4+ chars in order), 'N' (no)."""
    if needle in generated_text:
        return "Y"
    # Partial: longest common ordered subsequence ≥ 4 chars.
    # Simpler: check any 4-char substring of needle in generated.
    for k in range(len(needle) - 3):
        if needle[k:k+4] in generated_text:
            return "P"
    return "N"


# --- Main -------------------------------------------------------------------
def round_up_to_block(n: int, block: int) -> int:
    return ((n + block - 1) // block) * block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=str, default="256,512,1024,2048,4096",
                     help="comma-separated total prompt token targets")
    ap.add_argument("--fracs", type=str, default="0.25,0.50,0.75,0.95",
                     help="comma-separated needle fractional positions")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--time-budget-sec", type=float, default=600.0,
                     help="if any single trial exceeds this, mark FAIL and "
                          "skip larger L values for the same frac")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    os.makedirs(OUT_DIR, exist_ok=True)
    log_f = open(LOG_PATH, "w")
    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log("=" * 72)
    log(f"needle_haystack_probe — Qwen3.6-27B qb1 server")
    log(f"  lengths: {lengths}")
    log(f"  fracs:   {fracs}")
    log(f"  max_tokens={args.max_tokens}  chat=True  dry=0.8 rp=1.1")
    log("=" * 72)

    # Load tokenizer locally (CPU only).
    from transformers import AutoTokenizer
    log("[loading tokenizer Qwen/Qwen3.6-27B …]")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    log(f"[tokenizer loaded, vocab_size={tok.vocab_size}]")

    results = []
    skip_at_length: int = 10**9  # if a trial at L exceeds budget, skip all L'>=L

    for L in lengths:
        if L >= skip_at_length:
            for frac in fracs:
                log(f"\n[SKIP L={L} frac={frac:.2f} — L >= {skip_at_length} (budget exceeded)]")
                results.append({"L_target": L, "frac": frac, "skipped": True})
            continue
        for frac in fracs:

            # Deterministic per-cell needle.
            seed = L * 1000 + int(frac * 100)
            needle = make_needle(seed)

            log(f"\n--- trial L={L} frac={frac:.2f} needle={needle} ---")
            prompt, needle_offset, actual_tok = build_prompt(tok, L, frac, needle)
            max_pos = round_up_to_block(actual_tok + args.max_tokens + 16, 64)
            log(f"  built prompt: {actual_tok} tokens (target {L}), "
                f"needle at ~token {needle_offset} ({needle_offset/actual_tok:.0%})")
            log(f"  calling generate_long max_pos={max_pos} max_tokens={args.max_tokens}")

            t0 = time.time()
            try:
                final = stream_generate_long(
                    prompt, max_pos=max_pos, max_tokens=args.max_tokens)
            except socket.timeout:
                final = {"error": "client timeout"}
            wall = time.time() - t0

            if "error" in final:
                log(f"  ERROR: {final['error']}  wall={wall:.1f}s")
                cell = {
                    "L_target": L, "L_actual": actual_tok, "frac": frac,
                    "needle": needle, "needle_offset_tokens": needle_offset,
                    "max_pos": max_pos, "score": "ERR",
                    "generated": "",
                    "error": final["error"], "wall_sec": wall,
                }
                results.append(cell)
                if wall > args.time_budget_sec:
                    skip_at_length = min(skip_at_length, L + 1)
                continue

            gen = final.get("generated_text", "")
            score = score_needle(gen, needle)
            log(f"  generated ({final.get('n_generated_tokens', 0)} tok): "
                f"{gen[:120]!r}")
            log(f"  score: {score}  wall={wall:.1f}s  "
                f"prefill={final.get('prefill_ms', 0):.0f}ms  "
                f"decode={final.get('ms_per_tok', 0):.1f}ms/tok")

            cell = {
                "L_target": L,
                "L_actual": actual_tok,
                "frac": frac,
                "needle": needle,
                "needle_offset_tokens": needle_offset,
                "max_pos": max_pos,
                "score": score,
                "generated": gen,
                "n_generated_tokens": final.get("n_generated_tokens", 0),
                "wall_sec": wall,
                "prefill_ms": final.get("prefill_ms", 0),
                "ms_per_tok": final.get("ms_per_tok", 0),
                "stopped_on_eos": final.get("stopped_on_eos", False),
            }
            results.append(cell)
            if wall > args.time_budget_sec:
                log(f"  BUDGET: trial exceeded {args.time_budget_sec}s — "
                    f"skipping all subsequent L > {L}")
                skip_at_length = min(skip_at_length, L + 1)

    # --- Print 5×4 grid ---
    log("\n" + "=" * 72)
    log("RECALL GRID (Y=full, P=partial 4+ chars in order, N=miss, ERR=error)")
    log("=" * 72)
    header = "frac \\ L  |" + "".join(f" {L:>5d} |" for L in lengths)
    log(header)
    log("-" * len(header))
    for frac in fracs:
        row = f"  {frac:.2f}    |"
        for L in lengths:
            cells = [c for c in results
                      if c.get("L_target") == L and c.get("frac") == frac]
            if not cells:
                row += f"   -   |"
            else:
                c = cells[0]
                if c.get("skipped"):
                    row += f"  SKP  |"
                else:
                    row += f"   {c['score']:>2s}  |"
        log(row)
    log("=" * 72)

    # Wall-time grid
    log("\nWALL TIME (sec) GRID")
    log("=" * 72)
    log(header)
    log("-" * len(header))
    for frac in fracs:
        row = f"  {frac:.2f}    |"
        for L in lengths:
            cells = [c for c in results
                      if c.get("L_target") == L and c.get("frac") == frac]
            if not cells or cells[0].get("skipped"):
                row += f"   -   |"
            else:
                row += f" {cells[0].get('wall_sec', 0):5.1f} |"
        log(row)
    log("=" * 72)

    with open(RESULTS_JSON, "w") as f:
        json.dump({"lengths": lengths, "fracs": fracs,
                    "results": results}, f, indent=2)
    log(f"\n[results written to {RESULTS_JSON}]")
    log(f"[full log: {LOG_PATH}]")
    log_f.close()


if __name__ == "__main__":
    main()
