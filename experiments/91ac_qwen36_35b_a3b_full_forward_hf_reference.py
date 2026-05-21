#!/usr/bin/env python3
"""B5 — Qwen3.6-35B-A3B full 40-layer forward HF reference: "Paris" test.

The grand finale of the G0 reference set. Loads the full HF model into
CPU RAM (qb2 has 503 GB so this is fine at ~70 GB bf16), tokenizes
"The capital of France is", runs the full forward through embed → 40
layers → final norm → lm_head, and argmaxes the next token.

If this returns "Paris" or similar coherent token, the model + our load
machinery (state dict prefix resolution, MRoPE config, layer ordering)
are all correct end-to-end. This is the gold reference any ttnn server
implementation must eventually reproduce.

Run:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python \\
        experiments/91ac_qwen36_35b_a3b_full_forward_hf_reference.py'

Output: `~/tt-xla/.cache/qb2_35b_moe/b5_full_forward_reference.npz`
(stores logits + final hidden + selected top-K tokens)

Expected wall: ~5-15 min for CPU forward over 35B params (a few seconds
per layer matmuls, 40 layers, plus expert dispatch overhead).
"""
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = Path.home() / "tt-xla" / ".cache" / "qb2_35b_moe"
OUT_PATH = OUT_DIR / "b5_full_forward_reference.npz"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
PROMPT = "The capital of France is"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output: {OUT_PATH}")
    print(f"prompt: {PROMPT!r}")

    print("[1] load tokenizer…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long)
    print(f"  tokens: {input_ids[0].tolist()}")
    print(f"  decoded back: {tok.decode(input_ids[0])!r}")
    print(f"  vocab_size: {tok.vocab_size}")

    print("[2] load full model (bf16 CPU)… expect ~5 min download verify + load")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    load_wall = time.time() - t0
    print(f"  loaded in {load_wall:.1f} s")
    print(f"  model class: {model.__class__.__name__}")

    # Some models nest under .model.language_model — try to find the language model
    print(f"  module tree (top-level): {[n for n, _ in model.named_children()]}")

    print("[3] full forward (CPU, may be slow)…")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    fwd_wall = time.time() - t0
    print(f"  forward wall: {fwd_wall:.1f} s")

    if hasattr(out, "logits"):
        logits = out.logits  # [1, seq_len, vocab]
    else:
        logits = out[0]
    print(f"  logits: shape={list(logits.shape)}, dtype={logits.dtype}")

    # Last-position logits give the prediction for the next token
    last_logits = logits[0, -1, :].float()
    print(f"  last-pos logits: min={last_logits.min():.3f}, max={last_logits.max():.3f}, "
          f"argmax_id={last_logits.argmax().item()}")

    print("[4] top-5 next-token predictions:")
    topk_vals, topk_ids = last_logits.topk(5)
    for rank, (v, i) in enumerate(zip(topk_vals.tolist(), topk_ids.tolist())):
        decoded = tok.decode([i])
        print(f"  {rank+1}. id={i}  token={decoded!r}  logit={v:.3f}")

    top1_id = int(topk_ids[0].item())
    top1_token = tok.decode([top1_id])
    print(f"\n  top-1 next token: {top1_token!r}")
    paris_match = "paris" in top1_token.lower() or "Paris" in top1_token
    if paris_match:
        print("  ✓ COHERENT — matches expected 'Paris' completion")
    else:
        print(f"  ⚠ UNEXPECTED top-1; investigate. (Acceptable if it's still a "
              f"reasonable city completion or a leading-space variant of Paris.)")

    print("[5] save npz…")
    np.savez(
        OUT_PATH,
        prompt=np.array([PROMPT]),
        input_ids=input_ids.numpy(),
        last_pos_logits=last_logits.numpy(),
        top5_ids=topk_ids.numpy(),
        top5_logits=topk_vals.float().numpy(),
        load_wall_s=np.array([load_wall]),
        forward_wall_s=np.array([fwd_wall]),
    )
    print(f"  wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")

    print("\nB5 DONE.")
    print(f"\nSUMMARY: prompt={PROMPT!r}, top-1 next-token={top1_token!r}, "
          f"forward_wall={fwd_wall:.1f}s")


if __name__ == "__main__":
    main()
