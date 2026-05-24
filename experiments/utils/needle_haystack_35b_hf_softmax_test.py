#!/usr/bin/env python3
"""Verify the fp32-softmax hypothesis: run HF needle-haystack with the
attention-softmax forced to bf16 (no fp32 upcast). If HF then drifts like
TT, the kernel patch (Float16_b → Float32 for im_df in tt-metal's
sdpa_decode kernel) is the correct intervention.

Two modes:
  --mode fp32  → HF default (line 623 dtype=torch.float32 — current HF)
  --mode bf16  → monkey-patches eager_attention_forward to skip the upcast

Both modes use attn_implementation="eager" to force the Qwen3.5 MoE attention
function we patched (not PyTorch SDPA).

Run (qb1, ~5 min on CPU):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/needle_haystack_35b_hf_softmax_test.py \\
      --lengths 100 --fracs 0.5 --trials 1 --max-new 30 --mode bf16
"""
import argparse
import json
import random as _r
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(line_buffering=True)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
OUT_DIR = PROJECT_ROOT / ".cache" / "needle_haystack_35b_hf_softmax_test"
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


def score(text, needle):
    if needle in text:
        return "Y"
    for k in range(len(needle) - 3):
        if needle[k:k+4] in text:
            return "P"
    return "N"


def _patched_eager_attention_forward_bf16(module, query, key, value, attention_mask,
                                           scaling, dropout=0.0, **kwargs):
    """Drop-in replacement for qwen3_5_moe.eager_attention_forward that
    uses bf16 softmax (no fp32 upcast). Mirrors the original at line 619-626
    of modeling_qwen3_5_moe.py except for the softmax dtype kwarg."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    # NO dtype upcast — this is the test:
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="100")
    ap.add_argument("--fracs", default="0.5")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=30)
    ap.add_argument("--mode", choices=["fp32", "bf16"], required=True,
                    help="fp32 = HF default; bf16 = monkey-patch softmax to bf16 (no upcast)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.mode == "bf16":
        import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qm
        log("MONKEY-PATCH: replacing eager_attention_forward with bf16-softmax version")
        qm.eager_attention_forward = _patched_eager_attention_forward_bf16

    log(f"loading HF model on CPU ({MODEL_ID}) with attn_implementation=eager, mode={args.mode}…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",  # force the patchable path
    )
    model.eval()
    log(f"  loaded in {time.time()-t0:.0f}s; attn_impl={model.config._attn_implementation if hasattr(model.config, '_attn_implementation') else 'unknown'}")

    lengths = [int(x) for x in args.lengths.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]
    results = []
    for L in lengths:
        for f in fracs:
            for t in range(args.trials):
                needle = make_needle(args.seed + 1000*lengths.index(L) + 100*int(f*100) + t)
                prompt, n_prompt = build_prompt(tok, L, f, needle)
                log(f"\n=== mode={args.mode} L={L} frac={f} trial={t} needle={needle} (rendered={n_prompt} toks) ===")
                input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=False)
                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        input_ids, max_new_tokens=args.max_new,
                        do_sample=False, use_cache=True,
                        pad_token_id=tok.eos_token_id,
                    )
                gen_s = time.time() - t0
                generated_ids = out[0, input_ids.shape[1]:].tolist()
                text = tok.decode(generated_ids, skip_special_tokens=True)
                verdict = score(text, needle)
                log(f"  generate {gen_s:.1f}s ({gen_s*1000/args.max_new:.0f} ms/tok)")
                log(f"  generated: {text!r}")
                log(f"  verdict: {verdict}")
                results.append({
                    "mode": args.mode, "length": L, "frac": f, "trial": t,
                    "needle": needle, "prompt_tokens": n_prompt,
                    "max_new": args.max_new, "generated_text": text,
                    "verdict": verdict, "generate_seconds": gen_s,
                })

    log("\n=== SUMMARY ===")
    by_mode = {}
    for r in results:
        by_mode.setdefault(r["mode"], []).append(r["verdict"])
    for m, vs in sorted(by_mode.items()):
        ys, ps, ns = vs.count("Y"), vs.count("P"), vs.count("N")
        log(f"  mode={m}: Y={ys} P={ps} N={ns} / {len(vs)}")

    out_path = OUT_DIR / f"results_{args.mode}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
