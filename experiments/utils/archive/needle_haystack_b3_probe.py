#!/usr/bin/env python3
"""
Needle-haystack retrieval probe — B3 SDPA path (HiFi2 + no fp32_dest_acc).

Background: feedback_fp32_sdpa_cliff_probe.md showed that B3 eliminates the
bf16 prefill cliff that Agent K hit (0/4 retrieval at L=500). The qb1 server
is currently running with sentinel `~/.cache/p21_sdpa_variant.txt = B3` and
already validated retrieval on one (L=500, frac=0.5) cell.

This probe extends that single-cell validation to a real grid:
    L    ∈ {500, 1024, 2048, 4096, 8192}
    frac ∈ {0.25, 0.50, 0.75}
    3 trials per cell (distinct seeds → distinct needles)

Key differences from K's original needle_haystack_probe.py:
  - Sentinel pre-check (refuses to run if not B3)
  - Multi-trial per cell
  - Incremental JSON write after every trial (resilient to SIGINT)
  - Global wall-time budget (default 3 hours)
  - Per-trial wall budget cascades skips to larger L
  - Abort if 0/3 Y at the smallest L (a sanity gate; B3 already passed, so
    this should never trigger — but if it does, we save time)

Outputs to `.cache/needle_haystack_b3/`:
  - results.json — full grid with one entry per trial
  - log.txt     — line-buffered transcript

Run:
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/needle_haystack_b3_probe.py'

Typical wall (with 3-trial cells, B3 prefill ~196 ms/tok):
  L=500   trial ≈ 100 s; cell ≈ 300 s
  L=1024  trial ≈ 200 s; cell ≈ 600 s
  L=2048  trial ≈ 400 s; cell ≈ 1200 s
  L=4096  trial ≈ 800 s; cell ≈ 2400 s
  L=8192  trial ≈ 1600 s; cell ≈ 4800 s
Total naive: ~7.5 h. Global-budget cap will stop early if needed.
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
OUT_DIR = os.path.expanduser("~/tt-xla/.cache/needle_haystack_b3")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")
LOG_PATH = os.path.join(OUT_DIR, "log.txt")
SENTINEL_PATH = os.path.expanduser("~/tt-xla/.cache/p21_sdpa_variant.txt")


# --- Distractor corpus (identical to K's probe) -----------------------------
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
ALPHABET = "BCDFGHJKLMNPQRSTVWXYZ23456789"


def make_needle(seed: int) -> str:
    import random as _r
    rng = _r.Random(seed)
    return "".join(rng.choice(ALPHABET) for _ in range(8))


# --- Token budgeting --------------------------------------------------------
def apply_chat_no_think(tok, user_content: str) -> str:
    msgs = [{"role": "user", "content": user_content}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                    tokenize=False, enable_thinking=False)


def build_prompt(tok, total_tokens: int, frac: float, needle: str
                  ) -> tuple[str, int, int]:
    needle_sentence = f"REMEMBER THIS: The magic password is {needle}. END REMEMBER."
    question = ("\n\nBased only on the document above, what is the magic "
                "password? Answer with only the 8-character password.\n\nAnswer: ")

    def count_rendered_tokens(user_text: str) -> int:
        rendered = apply_chat_no_think(tok, user_text)
        return len(tok.encode(rendered, add_special_tokens=False))

    n_para = max(1, total_tokens // 220 + 2)
    distractor = DISTRACTOR_PARA * n_para
    iters = 0
    while iters < 40:
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
        iters += 1

    distractor_ids = tok.encode(distractor, add_special_tokens=False)
    insert_tok_idx = int(len(distractor_ids) * frac)
    prefix_ids = distractor_ids[:insert_tok_idx]
    suffix_ids = distractor_ids[insert_tok_idx:]
    prefix_text = tok.decode(prefix_ids, skip_special_tokens=True)
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)

    user_body = (prefix_text + " " + needle_sentence + " "
                  + suffix_text + question)
    rendered = apply_chat_no_think(tok, user_body)
    actual_tokens = len(tok.encode(rendered, add_special_tokens=False))
    rendered_until_needle = apply_chat_no_think(
        tok, prefix_text + " " + needle_sentence)
    needle_offset_tokens = len(tok.encode(rendered_until_needle,
                                           add_special_tokens=False))
    return rendered, needle_offset_tokens, actual_tokens


# --- Server RPC -------------------------------------------------------------
def _read_all_lines(sock):
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


def stream_generate_long(prompt: str, max_pos: int, max_tokens: int,
                          timeout: float = 1800.0) -> dict:
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
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "dry_multiplier": 0.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "seed": 0,
        "chat": False,  # prompt already chat-rendered
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
    """Y (full needle in output), P (any 4-char contiguous substring of
    needle present in output, in order), N (no)."""
    if needle in generated_text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in generated_text:
            return "P"
    return "N"


# --- Incremental persistence ------------------------------------------------
def write_results(state: dict):
    """Atomic-ish write: tmp file + rename."""
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, RESULTS_JSON)


# --- Main -------------------------------------------------------------------
def round_up_to_block(n: int, block: int) -> int:
    return ((n + block - 1) // block) * block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=str, default="500,1024,2048,4096,8192")
    ap.add_argument("--fracs", type=str, default="0.25,0.50,0.75")
    ap.add_argument("--trials", type=int, default=3,
                     help="trials per cell (distinct needles via seed)")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--per-trial-budget-sec", type=float, default=1800.0,
                     help="single-trial wall cap; on overflow skip larger L")
    ap.add_argument("--global-budget-sec", type=float, default=10800.0,
                     help="overall wall cap (default 3h); stop probe after")
    ap.add_argument("--require-sentinel", type=str, default="B3",
                     help="refuse to run if sentinel file != this")
    ap.add_argument("--resume", action="store_true",
                     help="if .cache/needle_haystack_b3/results.json exists, "
                          "skip trials with non-ERR scores and re-run only "
                          "ERR / missing trials; append a run-id suffix to log")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    os.makedirs(OUT_DIR, exist_ok=True)
    # In resume mode, append to log + preserve old results file
    log_mode = "a" if args.resume and os.path.exists(LOG_PATH) else "w"
    log_f = open(LOG_PATH, log_mode)

    def log(msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_f.write(line + "\n")
        log_f.flush()

    # --- Sentinel check ---
    try:
        with open(SENTINEL_PATH) as f:
            sentinel = f.read().strip()
    except FileNotFoundError:
        sentinel = "<missing>"
    log(f"sentinel: {sentinel}")
    if sentinel != args.require_sentinel:
        log(f"ABORT: sentinel != {args.require_sentinel}")
        log_f.close()
        sys.exit(2)

    log("=" * 72)
    log("needle_haystack_b3_probe — Qwen3.6-27B qb1 server (SDPA B3)")
    log(f"  lengths: {lengths}")
    log(f"  fracs:   {fracs}")
    log(f"  trials/cell: {args.trials}")
    log(f"  max_tokens={args.max_tokens}  chat=False (pre-rendered) greedy")
    log(f"  per-trial budget {args.per_trial_budget_sec}s; "
        f"global budget {args.global_budget_sec}s")
    log("=" * 72)

    from transformers import AutoTokenizer
    log("[loading tokenizer Qwen/Qwen3.6-27B …]")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    log(f"[tokenizer loaded, vocab_size={tok.vocab_size}]")

    # --- Resume support: pre-load existing results, skip non-ERR trials ---
    prior_keep: list[dict] = []  # already-successful trials to preserve
    prior_skip_keys: set[tuple] = set()  # (L, frac, trial) to skip
    if args.resume and os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                old = json.load(f)
            for c in old.get("results", []):
                if c.get("skipped"):
                    continue
                score = c.get("score")
                if score in ("Y", "P", "N"):
                    prior_keep.append(c)
                    prior_skip_keys.add(
                        (int(c["L_target"]), float(c["frac"]),
                         int(c["trial"])))
            log(f"[resume] preloaded {len(prior_keep)} non-ERR trials")
        except Exception as e:
            log(f"[resume] WARNING failed to load prior results: {e}")
    state = {
        "lengths": lengths,
        "fracs": fracs,
        "trials_per_cell": args.trials,
        "sentinel": sentinel,
        "start_unix": time.time(),
        "results": list(prior_keep),
        "aborted_reason": None,
    }
    write_results(state)

    skip_at_length = 10**9
    t_start = time.time()
    aborted = False
    abort_reason = None

    # Track per-cell scores for L=min(lengths) sanity gate.
    # Pre-populate from prior_keep so the gate counts ALL successful trials,
    # not just trials from this resumed run.
    smallest_L = min(lengths)
    smallest_L_scores: list[str] = [
        c["score"] for c in prior_keep if c.get("L_target") == smallest_L
    ]

    for L in lengths:
        if aborted:
            break
        if L >= skip_at_length:
            for frac in fracs:
                log(f"\n[SKIP L={L} frac={frac:.2f} — over budget]")
                for trial in range(args.trials):
                    state["results"].append({
                        "L_target": L, "frac": frac, "trial": trial,
                        "skipped": True,
                    })
                write_results(state)
            continue

        for frac in fracs:
            if aborted:
                break

            for trial in range(args.trials):
                # Resume: skip trials already successfully completed
                if (L, frac, trial) in prior_skip_keys:
                    log(f"[resume] skip L={L} frac={frac:.2f} trial={trial} "
                        f"(prior non-ERR result)")
                    continue
                # Global budget check
                elapsed = time.time() - t_start
                if elapsed > args.global_budget_sec:
                    log(f"\nGLOBAL BUDGET EXHAUSTED after {elapsed:.0f}s — "
                        f"stopping probe.")
                    aborted = True
                    abort_reason = "global_budget"
                    break

                # Deterministic per-(L, frac, trial) seed → unique needle
                seed = L * 10_000 + int(frac * 100) * 100 + trial
                needle = make_needle(seed)

                log(f"\n--- L={L} frac={frac:.2f} trial={trial} "
                    f"needle={needle} ---")
                prompt, needle_offset, actual_tok = build_prompt(
                    tok, L, frac, needle)
                max_pos = round_up_to_block(
                    actual_tok + args.max_tokens + 16, 64)
                log(f"  built prompt: {actual_tok} tok (target {L}), "
                    f"needle@{needle_offset} ({needle_offset/actual_tok:.0%})")
                log(f"  generate_long max_pos={max_pos} "
                    f"max_tokens={args.max_tokens}")

                t0 = time.time()
                try:
                    final = stream_generate_long(
                        prompt, max_pos=max_pos,
                        max_tokens=args.max_tokens,
                        timeout=args.per_trial_budget_sec + 60.0)
                except socket.timeout:
                    final = {"error": "client_timeout"}
                except Exception as e:
                    final = {"error": f"client_exception: {type(e).__name__}: {e}"}
                wall = time.time() - t0

                if "error" in final:
                    log(f"  ERROR: {final['error']}  wall={wall:.1f}s")
                    cell = {
                        "L_target": L, "L_actual": actual_tok, "frac": frac,
                        "trial": trial, "needle": needle,
                        "needle_offset_tokens": needle_offset,
                        "max_pos": max_pos, "score": "ERR",
                        "generated": "", "error": final["error"],
                        "wall_sec": wall,
                    }
                    state["results"].append(cell)
                    write_results(state)
                    if wall > args.per_trial_budget_sec:
                        skip_at_length = min(skip_at_length, L + 1)
                        log(f"  BUDGET: wall {wall:.0f}s > "
                            f"{args.per_trial_budget_sec:.0f}s → "
                            f"skip L > {L}")
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
                    "trial": trial,
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
                state["results"].append(cell)
                write_results(state)

                if L == smallest_L:
                    smallest_L_scores.append(score)

                if wall > args.per_trial_budget_sec:
                    skip_at_length = min(skip_at_length, L + 1)
                    log(f"  BUDGET: wall {wall:.0f}s > "
                        f"{args.per_trial_budget_sec:.0f}s → "
                        f"skip L > {L}")

        # Sanity gate: after the smallest L has fully completed all trials × fracs,
        # check Y rate. If 0% Y → abort (B3 should give high Y here).
        if L == smallest_L and len(smallest_L_scores) >= args.trials * len(fracs):
            y_count = sum(1 for s in smallest_L_scores if s == "Y")
            log(f"\n[sanity gate] smallest L={smallest_L}: "
                f"Y rate {y_count}/{len(smallest_L_scores)}")
            if y_count == 0:
                log(f"ABORT: 0/{len(smallest_L_scores)} Y at smallest L — "
                    f"B3 regression?")
                aborted = True
                abort_reason = "sanity_gate_smallest_L"
                break

    state["end_unix"] = time.time()
    state["wall_sec"] = state["end_unix"] - state["start_unix"]
    state["aborted_reason"] = abort_reason
    write_results(state)

    # --- Print result grids ---
    def grid_value(L: int, frac: float) -> str:
        cells = [c for c in state["results"]
                  if c.get("L_target") == L and c.get("frac") == frac
                  and not c.get("skipped")]
        if not cells:
            return "  -  "
        y = sum(1 for c in cells if c.get("score") == "Y")
        p = sum(1 for c in cells if c.get("score") == "P")
        n = sum(1 for c in cells if c.get("score") == "N")
        e = sum(1 for c in cells if c.get("score") == "ERR")
        # Compact format e.g. "3/0/0" or "2/0/1/ERR"
        if e:
            return f"{y}/{p}/{n}/{e}e"
        return f" {y}/{p}/{n} "

    log("\n" + "=" * 72)
    log("Y/P/N grid (per-cell, n=trials)")
    log("=" * 72)
    header = "frac \\ L  |" + "".join(f"  {L:>5d}  |" for L in lengths)
    log(header)
    log("-" * len(header))
    for frac in fracs:
        row = f"  {frac:.2f}    |"
        for L in lengths:
            row += f" {grid_value(L, frac):>7s} |"
        log(row)
    log("=" * 72)

    # Wall-time grid
    def mean_wall(L: int, frac: float) -> float:
        cells = [c for c in state["results"]
                  if c.get("L_target") == L and c.get("frac") == frac
                  and not c.get("skipped")
                  and "wall_sec" in c and c.get("score") != "ERR"]
        if not cells:
            return -1.0
        return sum(c["wall_sec"] for c in cells) / len(cells)

    log("\nMean wall sec per trial")
    log("=" * 72)
    log(header)
    log("-" * len(header))
    for frac in fracs:
        row = f"  {frac:.2f}    |"
        for L in lengths:
            mw = mean_wall(L, frac)
            row += f" {mw:7.1f} |" if mw > 0 else "    -    |"
        log(row)
    log("=" * 72)

    total_wall = state["wall_sec"]
    log(f"\nTotal wall: {total_wall:.0f}s ({total_wall/60:.1f} min)")
    log(f"Aborted reason: {abort_reason!r}")
    log(f"Results: {RESULTS_JSON}")
    log(f"Log: {LOG_PATH}")
    log_f.close()


if __name__ == "__main__":
    main()
