#!/usr/bin/env python3
"""
Permanent utility — run HF's full Qwen3.5 model on CPU for a single forward.

Used to establish GROUND TRUTH: what would the model produce in fp32/bf16
on a reference CPU run? Saves top-K predictions for our prompt so we
have a target to aim our ttnn implementation at.

Avoids AutoModel.from_pretrained (known to crash per auto-memory).
Constructs the model class directly and loads weights from safetensors
shard-by-shard.

Run on qb2 (CPU only, ~10-15 min for 27B in bf16):
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/hf_full_model_oracle.py [--prompt P] [--top-k 100]
"""
import os, sys, json, time, gc, argparse
import importlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_PROMPT = "The capital of France is"
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/hf_oracle_topk.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--dump-hidden-states", action="store_true",
                   help="Also dump per-layer hidden states for Plan A diff")
    args = p.parse_args()

    print("=" * 64)
    print(f"HF full-model oracle — prompt {args.prompt!r}, dtype={args.dtype}")
    print("=" * 64)
    t_total = time.time()
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # ----- Config + modeling module -----
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, 'text_config', full_cfg)
    config_module_path = type(text_cfg).__module__
    modeling_module_path = config_module_path.replace('configuration_', 'modeling_')
    print(f"importing {modeling_module_path}")
    mqn = importlib.import_module(modeling_module_path)

    # Find the right text-only model class
    ModelClass = None
    for name in ('Qwen3_5TextModel', 'Qwen3_5ForCausalLM', 'Qwen3NextModel',
                 'Qwen3_5Model'):
        if hasattr(mqn, name):
            ModelClass = getattr(mqn, name)
            ModelClass_name = name
            print(f"using {name}")
            break
    if ModelClass is None:
        print("No suitable model class found"); sys.exit(1)

    # ----- Tokenize -----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids])
    print(f"prompt ids: {prompt_ids}")
    print(f"decoded:    {[tok.decode([t]) for t in prompt_ids]}")

    # ----- Construct empty model -----
    print(f"\nConstructing {ModelClass_name}({type(text_cfg).__name__}) (empty)…")
    t0 = time.time()
    model = ModelClass(text_cfg)
    print(f"  constructed in {time.time()-t0:.1f}s (param count uninitialized)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params")

    # ----- Identify weight key namespace -----
    # HF stores under "model.language_model.*" for the text backbone but the
    # model's state_dict expects local keys (no prefix). We need to strip.
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    text_prefix = "model.language_model."
    # Determine likely state_dict prefix by inspecting a model parameter
    sd_keys = list(model.state_dict().keys())[:3]
    print(f"  sample model state_dict keys: {sd_keys}")
    # Most TextModel classes have state_dict keys like 'embed_tokens.weight',
    # 'layers.0.input_layernorm.weight', etc. (no model.language_model. prefix)
    # If keys come prefixed, we adjust.
    has_prefix = any(k.startswith("language_model") or k.startswith("model.") for k in sd_keys)

    # Gather all keys we need
    needed_keys = [k for k in weight_map.keys() if k.startswith(text_prefix)]
    # Group by shard for efficient loading
    by_shard = {}
    for k in needed_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    print(f"\nLoading {len(needed_keys)} weight tensors from {len(by_shard)} shards…")

    state = {}
    t_load_start = time.time()
    for shard_idx, (shard, keys) in enumerate(sorted(by_shard.items())):
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k in keys:
                local = k[len(text_prefix):]  # strip prefix
                # If model expects prefixed, add it back
                if has_prefix:
                    local = "language_model." + local
                state[local] = f.get_tensor(k).to(torch_dtype)
        print(f"  shard {shard_idx+1}/{len(by_shard)} loaded ({time.time()-t_load_start:.1f}s elapsed)")
        gc.collect()

    print(f"  total weight load: {time.time()-t_load_start:.1f}s")

    # ----- Load into model -----
    print(f"\nLoading state_dict (strict=False)…")
    info = model.load_state_dict(state, strict=False)
    print(f"  missing keys ({len(info.missing_keys)}): {info.missing_keys[:5]}{'…' if len(info.missing_keys) > 5 else ''}")
    print(f"  unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:5]}{'…' if len(info.unexpected_keys) > 5 else ''}")
    model = model.to(torch_dtype).eval()
    del state
    gc.collect()

    # ----- Load lm_head separately -----
    lm_head_key = "lm_head.weight"
    print(f"\nLoading lm_head weight…")
    lm_path = hf_hub_download(MODEL_ID, weight_map[lm_head_key])
    with safe_open(lm_path, framework="pt") as f:
        lm_head_w = f.get_tensor(lm_head_key).to(torch_dtype)
    print(f"  lm_head shape: {tuple(lm_head_w.shape)}")

    # ----- Forward -----
    print(f"\nForward pass (CPU, may take 1-3 minutes)…")
    t_fwd = time.time()
    with torch.no_grad():
        kwargs = dict(input_ids=input_ids)
        if args.dump_hidden_states:
            kwargs['output_hidden_states'] = True
        out = model(**kwargs)
        if hasattr(out, 'last_hidden_state'):
            hidden = out.last_hidden_state
        elif hasattr(out, 'logits'):
            # ForCausalLM path: already has logits
            hidden = None
            logits = out.logits
        else:
            hidden = out[0] if isinstance(out, tuple) else out
        if hidden is not None:
            # Apply lm_head manually
            logits = hidden.to(torch.float32) @ lm_head_w.to(torch.float32).t()
    print(f"  forward took {time.time()-t_fwd:.1f}s")
    print(f"  logits shape: {tuple(logits.shape)}")

    # ----- Optional: dump per-layer hidden states -----
    if args.dump_hidden_states and hasattr(out, 'hidden_states') and out.hidden_states is not None:
        per_layer_path = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")
        hidden_states = out.hidden_states  # tuple of (n_layers + 1) tensors
        print(f"\nDumping {len(hidden_states)} hidden states (embed + {len(hidden_states)-1} layer outputs)…")
        npz_data = {}
        for i, h in enumerate(hidden_states):
            arr = h.float().cpu().numpy()  # [1, seq, hidden]
            npz_data[f"hidden_{i}"] = arr
        npz_data["prompt_ids"] = np.array(prompt_ids)
        # Also save final lm_head output for downstream reference
        npz_data["logits_last"] = logits[0, -1].float().cpu().numpy()
        np.savez(per_layer_path, **npz_data)
        total_mb = sum(v.nbytes for v in npz_data.values()) / 1e6
        print(f"  saved per-layer hidden states ({total_mb:.1f} MB) → {per_layer_path}")

    # ----- Top-K of LAST token's logits -----
    last = logits[0, -1].float()
    top_vals, top_ids = torch.topk(last, args.top_k)

    # Softmax over full vocab (in fp32 to avoid overflow)
    probs = torch.softmax(last, dim=-1)

    records = []
    for rank in range(args.top_k):
        tid = int(top_ids[rank])
        records.append({
            "rank": rank + 1,
            "token_id": tid,
            "token": tok.decode([tid]),
            "logit": float(top_vals[rank]),
            "prob": float(probs[tid]),
        })

    print(f"\nTop {min(args.top_k, 20)} predictions for prompt {args.prompt!r}:")
    print(f"{'rank':>4s} {'tok_id':>7s} {'token':>25s} {'logit':>10s} {'prob':>10s}")
    for r in records[:20]:
        print(f"{r['rank']:4d} {r['token_id']:7d} {repr(r['token']):>25s} "
              f"{r['logit']:10.4f} {r['prob']:10.6f}")

    # Save
    out_data = {
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "dtype": args.dtype,
        "model": MODEL_ID,
        "top_k": records,
        "logit_max": float(last.max()),
        "logit_min": float(last.min()),
        "logit_mean": float(last.mean()),
        "logit_std": float(last.std()),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved top-{args.top_k} → {OUT_PATH}")

    # Also dump a quick sanity check: where is 'Paris'?
    paris_ids = tok.encode(" Paris", add_special_tokens=False)
    if len(paris_ids) == 1:
        ptid = paris_ids[0]
        prank = (last > last[ptid]).sum().item() + 1
        print(f"\n' Paris' (id {ptid}): rank {prank}/{len(last)}  "
              f"logit={float(last[ptid]):.4f}  prob={float(probs[ptid]):.6f}")

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
