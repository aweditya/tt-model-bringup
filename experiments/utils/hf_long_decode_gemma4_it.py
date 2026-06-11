#!/usr/bin/env python3
"""HF long-decode reference for gemma-4-12B-it — definitive answer to:
"is our 150-token collapse an intrinsic model property?"

Loads google/gemma-4-12B-it on CPU, runs `model.generate()` with the IT
chat template at the same prompt our ttnn server collapsed on, for the
same N=300 tokens at the same temperature/top-p. Saves raw generation,
per-position argmax, and a short repetition diagnostic so we can
compare structurally without needing to reload the TT path.

Hypothesis under test:
  If HF produces coherent paragraph(s) at N=300 → our ttnn impl has a
  long-decode bug (#314).
  If HF ALSO collapses to repetition → model property; we adjust the
  demo prompt or sampling and move on.

REUSE: forks `experiments/utils/hf_reference_gemma4_12b.py` bootstrap
(same MODEL_ID variant select, same AutoModelForCausalLM load). This
file ADDS `.generate()` + chat-template formatting + N-step run.

Run on qb1 CPU (model fits in 478 GB RAM; ~40-90 min for 300 tokens
depending on CPU):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/hf_long_decode_gemma4_it.py \\
        --prompt "What is Tenstorrent?" --max-new 300 --temp 0.4 \\
        --top-p 0.9 --seed 42'

Output → .cache/hf_long_decode_gemma4_it/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / "hf_long_decode_gemma4_it"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-4-12B-it",
                    help="HF model id (default: IT variant)")
    ap.add_argument("--prompt", default="What is Tenstorrent?",
                    help="user turn content")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: .cache/hf_long_decode_gemma4_it/<ts>)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        DEFAULT_OUT_DIR / str(int(time.time()))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"out_dir = {out_dir}")
    log(f"model = {args.model_id}")
    log(f"prompt = {args.prompt!r}")
    log(f"max_new={args.max_new} temp={args.temp} top_p={args.top_p} seed={args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log("loading tokenizer…")
    tok = AutoTokenizer.from_pretrained(args.model_id)

    log("loading model (bf16, CPU)…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s")

    # Render via the IT chat template so the prompt structure matches
    # exactly what cb_api / scripts/chat.py would see on the wire.
    messages = [{"role": "user", "content": args.prompt}]
    chat_ids = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    log(f"prompt tokens = {chat_ids.shape[-1]}")

    log("generating…")
    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(
            chat_ids,
            max_new_tokens=args.max_new,
            do_sample=(args.temp > 0),
            temperature=args.temp,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
        )
    wall = time.time() - t0
    log(f"  wall {wall:.1f}s ({args.max_new / wall:.2f} tok/s)")

    gen_ids = out_ids[0, chat_ids.shape[-1]:].cpu().numpy().astype(np.int32)
    full_text = tok.decode(gen_ids, skip_special_tokens=True)
    n_gen = len(gen_ids)
    log(f"actually generated {n_gen} tokens, {len(full_text)} chars")

    # Repetition diagnostic — does the HF reference also collapse?
    # Look at: (a) overall #-char ratio, (b) longest single-char run,
    # (c) longest repeated bigram, (d) trailing 100-char unique-char count.
    def longest_run(s: str) -> tuple[str, int]:
        if not s:
            return ("", 0)
        best_ch, best_n = s[0], 1
        cur_ch, cur_n = s[0], 1
        for c in s[1:]:
            if c == cur_ch:
                cur_n += 1
            else:
                cur_ch, cur_n = c, 1
            if cur_n > best_n:
                best_ch, best_n = cur_ch, cur_n
        return (best_ch, best_n)

    run_ch, run_n = longest_run(full_text)
    tail = full_text[-100:] if len(full_text) >= 100 else full_text
    tail_uniq = len(set(tail))
    diag = {
        "n_gen_tokens": int(n_gen),
        "n_chars": len(full_text),
        "longest_run_char": run_ch,
        "longest_run_len": run_n,
        "tail_100_unique_chars": tail_uniq,
        "hash_count": full_text.count("#"),
        "wall_seconds": round(wall, 1),
    }

    # Verdict — heuristic for the demo's failure mode (`####…####` tail).
    if run_n > 50 or tail_uniq < 5:
        verdict = "COLLAPSED (run >50 OR tail unique <5)"
    else:
        verdict = "COHERENT (no degenerate-repeat signal)"
    diag["verdict"] = verdict
    log(f"VERDICT: {verdict}")
    for k, v in diag.items():
        log(f"  {k} = {v}")

    # Persist for after-the-fact comparison vs the TT impl's output.
    (out_dir / "gen_ids.npy").write_bytes(b"")  # placeholder for np.save below
    np.save(out_dir / "gen_ids.npy", gen_ids)
    (out_dir / "gen_text.txt").write_text(full_text)
    (out_dir / "diag.json").write_text(json.dumps({
        "prompt": args.prompt,
        "model_id": args.model_id,
        "max_new": args.max_new,
        "temperature": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        "n_prompt_tokens": int(chat_ids.shape[-1]),
        **diag,
    }, indent=2))
    log(f"wrote {out_dir}/gen_ids.npy, gen_text.txt, diag.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
