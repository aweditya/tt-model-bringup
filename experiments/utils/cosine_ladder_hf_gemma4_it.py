#!/usr/bin/env python3
"""HF teacher-forced long-decode reference for gemma-4-12B-it.

#314 — long-decode coherence collapse. HF was already proven coherent
at 300 tokens (see `experiments/utils/hf_long_decode_gemma4_it.py`).
This probe captures **per-step, per-layer hidden states + lm_head
logits** so a companion ttnn ladder can teacher-force the same token
sequence through our impl and cosine against HF at each (step, layer).
The first (step, layer) where cos drops below 0.99 IS the drift onset.

REUSE:
- `experiments/utils/cosine_ladder_hf_ref.py` — greedy decode + KV
  cache + per-step logits (Qwen 27B variant). This file forks the
  decode loop verbatim and adds gemma4 model id + per-step per-layer
  hidden capture.
- `experiments/utils/hf_reference_gemma4_12b.py` — per-layer
  `output_hidden_states=True` capture pattern. Reused as-is.
- `experiments/utils/hf_long_decode_gemma4_it.py` — chat-template
  rendering + .venv-gemma4 contract.

Output (single .npz at .cache/cosine_ladder_hf_gemma4_it/<ts>.npz):
- prompt_ids:      [L_prompt]               int32
- decode_ids:      [N]                      int32   teacher-force order
- decode_hidden:   [N, n_layers+1, HIDDEN]  float32 hidden AFTER each
                                                    layer at the NEW
                                                    token position
- decode_logits:   [N, VOCAB]               float32 post-softcap
- decode_argmax:   [N]                      int32   confirms greedy
- meta.json:       params + verdict

Run on qb1 CPU via .venv-gemma4 (the HF venv with transformers 5.10+):
    ssh qb1 'cd ~/tt-xla && .venv-gemma4/bin/python -u \\
        experiments/utils/cosine_ladder_hf_gemma4_it.py \\
        --prompt "What is Tenstorrent?" --max-tokens 100'

~2.5 tok/s on qb1 CPU per the previous run — 100 tokens = ~40-50 min.
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
DEFAULT_MODEL_ID = "google/gemma-4-12B-it"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / "cosine_ladder_hf_gemma4_it"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--prompt", default="What is Tenstorrent?")
    ap.add_argument("--max-tokens", type=int, default=100,
                    help="number of decode steps to teacher-force-record")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch RNG seed (greedy doesn't sample, but "
                         "fixed for repeatability of any non-determinism)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        DEFAULT_OUT_DIR / str(int(time.time()))
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"out_dir = {out_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log(f"loading tokenizer ({args.model_id})…")
    tok = AutoTokenizer.from_pretrained(args.model_id)

    log("loading model (bf16, CPU)…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"  loaded in {time.time() - t0:.1f}s")

    # Render via the IT chat template — exact prompt structure cb_api
    # uses on the wire.
    messages = [{"role": "user", "content": args.prompt}]
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    chat_ids = enc["input_ids"] if hasattr(enc, "input_ids") else (
        enc["input_ids"] if isinstance(enc, dict) else enc)
    L_prompt = int(chat_ids.shape[-1])
    log(f"prompt tokens = {L_prompt}")

    # Probe shape inference — one tiny forward to learn n_layers + HIDDEN
    # + VOCAB.
    log("probing shapes (prompt forward, hidden capture)…")
    t0 = time.time()
    with torch.no_grad():
        probe = model(
            input_ids=chat_ids, use_cache=True,
            output_hidden_states=True, return_dict=True,
        )
    n_layers_plus_1 = len(probe.hidden_states)
    HIDDEN = probe.hidden_states[0].shape[-1]
    VOCAB = probe.logits.shape[-1]
    past = probe.past_key_values
    log(f"  n_layers+1={n_layers_plus_1} HIDDEN={HIDDEN} VOCAB={VOCAB} "
        f"(prefill {time.time() - t0:.1f}s)")

    # Decode N steps greedy, capturing per-layer hidden at the NEW
    # token position. Forked from cosine_ladder_hf_ref.py.
    log(f"greedy decode for {args.max_tokens} tokens with per-layer capture…")
    decode_ids = np.zeros((args.max_tokens,), dtype=np.int32)
    decode_argmax = np.zeros((args.max_tokens,), dtype=np.int32)
    decode_hidden = np.zeros(
        (args.max_tokens, n_layers_plus_1, HIDDEN), dtype=np.float32)
    decode_logits = np.zeros((args.max_tokens, VOCAB), dtype=np.float32)

    # Step 0: pick from the prefill's last position.
    last_logits = probe.logits[0, -1, :].to(torch.float32).cpu().numpy()
    next_id = int(last_logits.argmax())
    decode_ids[0] = next_id
    decode_argmax[0] = next_id
    decode_logits[0] = last_logits
    for li, h in enumerate(probe.hidden_states):
        decode_hidden[0, li] = h[0, -1, :].to(torch.float32).cpu().numpy()
    tok_str = tok.decode([next_id])
    log(f"  step   0 id={next_id:6d} tok={tok_str!r}")

    t_decode = time.time()
    cur_ids = torch.tensor([[next_id]], dtype=torch.long)
    for step in range(1, args.max_tokens):
        t_s = time.time()
        with torch.no_grad():
            out = model(
                input_ids=cur_ids, use_cache=True,
                past_key_values=past,
                output_hidden_states=True, return_dict=True,
            )
        past = out.past_key_values
        last_logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
        next_id = int(last_logits.argmax())
        decode_ids[step] = next_id
        decode_argmax[step] = next_id
        decode_logits[step] = last_logits
        for li, h in enumerate(out.hidden_states):
            decode_hidden[step, li] = h[0, -1, :].to(torch.float32).cpu().numpy()
        cur_ids = torch.tensor([[next_id]], dtype=torch.long)
        if step % 10 == 0:
            tok_str = tok.decode([next_id])
            log(f"  step {step:3d} id={next_id:6d} tok={tok_str!r} "
                f"({time.time() - t_s:.1f}s/step)")
    decode_wall = time.time() - t_decode
    log(f"decode wall {decode_wall:.1f}s "
        f"({(args.max_tokens - 1) / max(decode_wall, 1e-6):.2f} tok/s)")

    full_text = tok.decode(decode_ids.tolist(), skip_special_tokens=True)
    log(f"continuation ({len(full_text)} chars): {full_text[:300]!r}"
        + ("…" if len(full_text) > 300 else ""))

    # Single .npz for easy ttnn-side load.
    npz_path = out_dir / "ladder.npz"
    np.savez(
        npz_path,
        prompt_ids=chat_ids[0].cpu().numpy().astype(np.int32),
        decode_ids=decode_ids,
        decode_hidden=decode_hidden,
        decode_logits=decode_logits,
        decode_argmax=decode_argmax,
    )
    meta = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "L_prompt": L_prompt,
        "max_tokens": args.max_tokens,
        "n_layers_plus_1": int(n_layers_plus_1),
        "HIDDEN": int(HIDDEN),
        "VOCAB": int(VOCAB),
        "decode_wall_s": round(decode_wall, 1),
        "decode_tokens_s": round(
            (args.max_tokens - 1) / max(decode_wall, 1e-6), 3),
        "continuation_chars": len(full_text),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "gen_text.txt").write_text(full_text)
    log(f"wrote {npz_path} + meta.json + gen_text.txt")
    log(f"npz size = {npz_path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
