"""mtp_head_probe_hfcpu.py - HF-CPU-based MTP head probe (no device).

Replaces mtp_head_probe.py for cases when device contention with the running
qb1 server triggers SIGBUS (mesh buffer dispatch crash during weight upload).

Strategy:
  - Use HF Qwen3Next on CPU (bf16) for the verifier. Get per-layer hidden states
    for each prompt via output_hidden_states=True.
  - For each prompt of length P:
      For each t in 0 .. P-3:
        h_t = hidden_states[-1][t]                   # post-final-RMSNorm hidden state
        token_t1 = prompt_ids[t+1]                    # actual next token (verifier's choice
                                                       in greedy-decode of this corpus)
        token_t2 = prompt_ids[t+2]                    # actual next-next token
        logits_mtp = mtp_forward_numpy(h_t, embed(token_t1), position=t+1)
        pred_t2 = argmax(logits_mtp)
        match = (pred_t2 == token_t2)

  Note on ground truth: with a real text prompt, the "verifier" is HF itself.
  For most positions of a well-tokenized natural prompt, HF's top-1 prediction
  at position t IS prompt_ids[t+1] (the actual next token). This means the
  natural prompt's actual sequence is a faithful proxy for the verifier's
  greedy continuation. So pred_t2 == prompt_ids[t+2] iff MTP matches what
  the verifier would have chosen (assuming verifier is greedy).

Runtime estimate: ~1-3 min per prompt (CPU prefill of Qwen3.6-27B in bf16).

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/mtp_head_probe_hfcpu.py
"""
import argparse
import gc
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.expanduser("~"))

# Reuse numpy MTP forward + weight loader from mtp_head_probe.py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_mtp", os.path.expanduser("~/tt-xla/experiments/utils/mtp_head_probe.py"))
_mtp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mtp)

MODEL_ID = "Qwen/Qwen3.6-27B"

# Slightly longer prompts that exercise multiple natural token transitions.
DEFAULT_PROMPTS = [
    "The capital of France is Paris, which is famous for the Eiffel Tower.",
    "Albert Einstein was a German-born theoretical physicist best known for his theory of relativity.",
    "The largest planet in our solar system is Jupiter. It is a gas giant orbiting the Sun.",
    "Water boils at one hundred degrees Celsius under standard atmospheric pressure at sea level.",
    "One plus one equals two. Two plus two equals four. Three plus three equals six.",
]


def build_hf_model(dtype=torch.bfloat16):
    """Load Qwen3Next text-only model on CPU. Mirrors hf_full_model_oracle.py."""
    print(f"[hf] loading config + modeling module…")
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, "text_config", full_cfg)
    config_mod = type(text_cfg).__module__
    modeling_mod = config_mod.replace("configuration_", "modeling_")
    mqn = importlib.import_module(modeling_mod)
    ModelClass = None
    for name in ("Qwen3NextModel", "Qwen3_5TextModel", "Qwen3_5Model"):
        if hasattr(mqn, name):
            ModelClass = getattr(mqn, name)
            print(f"[hf] using {name}")
            break
    if ModelClass is None:
        raise RuntimeError("No suitable Qwen3Next model class found in HF transformers")

    print(f"[hf] constructing empty model…")
    t0 = time.time()
    model = ModelClass(text_cfg)
    print(f"[hf]   constructed in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # Load weights from safetensors (only text-model state, NO mtp / lm_head here)
    idx = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]

    # Determine prefix used by HF state_dict (Qwen3.6 stores 'model.language_model....')
    model_state_keys = set(model.state_dict().keys())
    # Pick a sample key from weight_map matching the model side
    text_prefix = None
    for candidate in ("model.language_model.", "model.", ""):
        sample = next(iter(model_state_keys))
        if candidate + sample in weight_map:
            text_prefix = candidate
            break
    if text_prefix is None:
        # Try the reverse: HF stores with prefix, model expects unprefixed
        for cand in ("model.language_model.", "model.", ""):
            if any(k.startswith(cand) and "embed_tokens" in k for k in weight_map):
                text_prefix = cand
                break
    has_prefix = (text_prefix != "")
    print(f"[hf] weight prefix detected: {text_prefix!r}")

    # Build needed_keys (text model only, exclude mtp/lm_head)
    needed_keys = []
    for k in weight_map:
        if k.startswith("mtp"):
            continue
        if k == "lm_head.weight":
            continue
        # Only load keys that have the text_prefix
        local = k[len(text_prefix):] if has_prefix else k
        # Some loaders include "language_model." in the local key
        # We accept either; load_state_dict(strict=False) handles
        needed_keys.append(k)

    by_shard = {}
    for k in needed_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    print(f"[hf] loading {len(needed_keys)} text-model tensors from {len(by_shard)} shards…")
    state = {}
    t0 = time.time()
    for shard_idx, (shard, keys) in enumerate(sorted(by_shard.items())):
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k in keys:
                local = k[len(text_prefix):] if has_prefix else k
                state[local] = f.get_tensor(k).to(dtype)
        if shard_idx % 3 == 0 or shard_idx == len(by_shard) - 1:
            print(f"[hf]   shard {shard_idx+1}/{len(by_shard)} ({time.time()-t0:.1f}s)")
        gc.collect()
    print(f"[hf] total weight load: {time.time()-t0:.1f}s")
    info = model.load_state_dict(state, strict=False)
    print(f"[hf] state_dict missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    if info.missing_keys[:3]:
        print(f"[hf] first missing: {info.missing_keys[:3]}")
    if info.unexpected_keys[:3]:
        print(f"[hf] first unexpected: {info.unexpected_keys[:3]}")
    model = model.to(dtype).eval()
    del state
    gc.collect()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--out-path", default=os.path.expanduser(
        "~/tt-xla/.cache/mtp_head_probe_hfcpu_results.json"))
    args = ap.parse_args()

    prompts = args.prompts or DEFAULT_PROMPTS
    print("=" * 72)
    print(f"MTP head probe (HF CPU verifier + numpy MTP)")
    print(f"  dtype={args.dtype}  n_prompts={len(prompts)}")
    for p in prompts:
        print(f"   * {p!r}")
    print("=" * 72)

    # Config
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)["text_config"]
    cfg = {
        "hidden":      text_cfg["hidden_size"],
        "n_q_heads":   text_cfg["num_attention_heads"],
        "n_kv_heads":  text_cfg["num_key_value_heads"],
        "head_dim":    text_cfg["head_dim"],
        "partial_rotary_factor": text_cfg["partial_rotary_factor"],
        "intermediate_size": text_cfg["intermediate_size"],
    }
    ROTARY_DIM = int(cfg["head_dim"] * cfg["partial_rotary_factor"])

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n[main] loading MTP weights + embed + lm_head…")
    mtp_w = _mtp.load_mtp_weights()
    elm = _mtp.load_embed_lm_head()
    embed_np = elm["embed"]
    lm_head_np = elm["lm_head"]

    print("\n[main] building HF text-only model…")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = build_hf_model(dtype=dtype)

    all_results = []
    agg_matches = 0
    agg_total = 0
    for pidx, prompt in enumerate(prompts):
        print(f"\n{'=' * 72}")
        print(f"Prompt {pidx+1}/{len(prompts)}: {prompt!r}")
        print("=" * 72)
        prompt_ids = tok.encode(prompt)
        P = len(prompt_ids)
        print(f"  {P} tokens: {prompt_ids[:8]}{' ...' if P > 8 else ''}")
        if P < 4:
            print(f"  SKIP (too short for MTP probe)")
            continue

        # HF prefill with hidden_states
        t0 = time.time()
        with torch.no_grad():
            inp = torch.tensor([prompt_ids])
            out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
        print(f"  HF prefill: {time.time()-t0:.1f}s")

        if not hasattr(out, "hidden_states") or out.hidden_states is None:
            print(f"  ERROR: HF did not return hidden_states (model class may not support it)")
            continue
        hidden_states = out.hidden_states  # tuple of len num_layers+1
        # last_hidden_state = post-final-norm
        h_last = hidden_states[-1][0].float().numpy()  # [P, hidden]
        print(f"  hidden_states tuple len = {len(hidden_states)}  h_last shape={h_last.shape}")

        # For each t in 0..P-3 predict token_t2 from (h_last[t], embed(prompt_ids[t+1]))
        matches = 0
        total = 0
        details = []
        for t in range(P - 2):
            tok_t1 = int(prompt_ids[t + 1])
            tok_t2 = int(prompt_ids[t + 2])
            logits = _mtp.mtp_forward_numpy(
                h_last[t], embed_np[tok_t1], position_t1=t + 1,
                w=mtp_w, lm_head_np=lm_head_np, cfg=cfg, rotary_dim=ROTARY_DIM)
            pred_t2 = int(np.argmax(logits))
            # Also compute the rank of the true t+2 token for diagnostics
            true_logit = logits[tok_t2]
            rank = int((logits > true_logit).sum() + 1)
            match = (pred_t2 == tok_t2)
            matches += int(match)
            total += 1
            details.append({
                "t": t,
                "tok_t1": tok_t1,
                "tok_t1_str": tok.decode([tok_t1]),
                "actual_t2": tok_t2,
                "actual_t2_str": tok.decode([tok_t2]),
                "pred_t2": pred_t2,
                "pred_t2_str": tok.decode([pred_t2]),
                "rank_of_actual": rank,
                "match": match,
            })

        rate = matches / max(total, 1)
        avg_rank = sum(d["rank_of_actual"] for d in details) / max(total, 1)
        med_rank = sorted(d["rank_of_actual"] for d in details)[len(details) // 2] if details else None
        print(f"  MTP match: {matches}/{total} = {rate*100:.1f}%  (avg rank of actual: {avg_rank:.1f}, median: {med_rank})")
        print(f"  {'t':>3s} {'tok_t1':>18s} {'actual_t2':>18s} {'pred_t2':>18s} {'rank':>6s} {'OK':>3s}")
        for d in details[:20]:
            t1s = (d["tok_t1_str"] or "")[:16]
            a2s = (d["actual_t2_str"] or "")[:16]
            p2s = (d["pred_t2_str"] or "")[:16]
            ok = "Y" if d["match"] else " "
            print(f"  {d['t']:3d} {t1s!r:>18s} {a2s!r:>18s} {p2s!r:>18s} {d['rank_of_actual']:6d} {ok:>3s}")
        if len(details) > 20:
            print(f"  ... ({len(details) - 20} more)")

        all_results.append({
            "prompt": prompt,
            "prompt_ids": [int(x) for x in prompt_ids],
            "match_count": matches,
            "total": total,
            "match_rate": rate,
            "avg_rank_of_actual": avg_rank,
            "median_rank_of_actual": int(med_rank) if med_rank else None,
            "details": details,
        })
        agg_matches += matches
        agg_total += total

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    for r in all_results:
        print(f"  {r['match_rate']*100:5.1f}%  ({r['match_count']:3d}/{r['total']:3d})  "
              f"med_rank={r['median_rank_of_actual']:>4d}  {r['prompt']!r}")
    agg_rate = agg_matches / max(agg_total, 1)
    print("-" * 72)
    print(f"  TOTAL: {agg_matches}/{agg_total} = {agg_rate*100:.1f}% match rate")

    if agg_rate >= 0.5:
        verdict = "STRONGLY RECOMMEND D'3 - high accept rate suggests ~1.5-2x speedup"
    elif agg_rate >= 0.3:
        verdict = "RECOMMEND D'3 with caveat - modest speedup, integration cost still ~1000 LOC"
    else:
        verdict = "DEFER D'3 - draft quality too low to justify ~1000 LOC integration"
    print(f"  VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "dtype": args.dtype,
            "method": "hf_cpu_verifier + numpy_mtp",
            "prompts": prompts,
            "results": all_results,
            "aggregate_match_count": agg_matches,
            "aggregate_total": agg_total,
            "aggregate_match_rate": agg_rate,
            "verdict": verdict,
        }, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer,)) else o)
    print(f"  results -> {args.out_path}")


if __name__ == "__main__":
    main()
