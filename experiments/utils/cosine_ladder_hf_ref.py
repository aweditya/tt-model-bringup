#!/usr/bin/env python3
"""
Cosine-ladder HF reference generator (Agent task — long-context bf16 drift).

Generates a greedy continuation from a fixed prompt under HuggingFace's
own Qwen3_5 modeling code, in fp32 or bf16 on CPU. Saves:
  - the generated token sequence (prompt + continuation)
  - per-step logits for EACH generated token (the position of the
    *next-token logits*, i.e. the last-prompt-token logit for step 0,
    then each subsequent generated-token logit)

The companion script `cosine_ladder_tt_probe.py` teacher-forces this exact
sequence through our TT path on device 3 and compares cosines.

Output: ~/tt-xla/.cache/cosine_ladder_hf_ref.npz with keys
  - prompt_ids:        [P]              int32
  - generated_ids:     [M]              int32   (what HF greedy-picked)
  - logits_at_step:    [M, vocab]       float32
                       logits_at_step[i] = logits for token index P+i-1
                       (i.e. the distribution over which generated_ids[i]
                       was argmax-sampled)
  - top1_ids:          [M]              int32   (argmax confirmation)
  - dtype:             str
  - prompt:            str
  - model_id:          str

Run on qb1 (CPU only; ~20-90 minutes for 27B at fp32, 100 tokens):
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/cosine_ladder_hf_ref.py \
        --prompt "The quick brown fox" --max-tokens 100 --dtype fp32
"""
import os, sys, json, time, gc, argparse
import importlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

# Force line-buffered stdout for SSH-piped runs (auto-memory note).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_PROMPT = "The quick brown fox"
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_hf_ref.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Number of tokens HF will greedy-generate.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32",
                   help="HF compute dtype. fp32 is the gold reference but slow.")
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()

    print("=" * 64, flush=True)
    print(f"HF cosine-ladder reference  dtype={args.dtype}  "
          f"max_tokens={args.max_tokens}  prompt={args.prompt!r}", flush=True)
    print("=" * 64, flush=True)
    t_total = time.time()
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # ----- Config + dynamic modeling import -----
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, "text_config", full_cfg)
    config_module_path = type(text_cfg).__module__
    modeling_module_path = config_module_path.replace("configuration_", "modeling_")
    print(f"importing {modeling_module_path}", flush=True)
    mqn = importlib.import_module(modeling_module_path)

    ModelClass = None
    for name in ("Qwen3_5TextModel", "Qwen3_5ForCausalLM", "Qwen3NextModel",
                 "Qwen3_5Model"):
        if hasattr(mqn, name):
            ModelClass = getattr(mqn, name)
            ModelClass_name = name
            print(f"using {name}", flush=True)
            break
    if ModelClass is None:
        print("No suitable model class found", flush=True); sys.exit(1)

    # ----- Tokenize -----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(args.prompt)
    print(f"prompt ids: {prompt_ids}", flush=True)
    print(f"decoded:    {[tok.decode([t]) for t in prompt_ids]}", flush=True)

    # ----- Construct empty model -----
    print(f"\nConstructing {ModelClass_name}({type(text_cfg).__name__}) (empty)…",
          flush=True)
    t0 = time.time()
    model = ModelClass(text_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.2f}B params constructed in {time.time()-t0:.1f}s",
          flush=True)

    # ----- Load weights -----
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    text_prefix = "model.language_model."
    sd_keys = list(model.state_dict().keys())[:3]
    has_prefix = any(k.startswith("language_model") or k.startswith("model.")
                     for k in sd_keys)

    needed_keys = [k for k in weight_map.keys() if k.startswith(text_prefix)]
    by_shard = {}
    for k in needed_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    print(f"\nLoading {len(needed_keys)} tensors from {len(by_shard)} shards…",
          flush=True)

    state = {}
    t_load_start = time.time()
    for shard_idx, (shard, keys) in enumerate(sorted(by_shard.items())):
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for k in keys:
                local = k[len(text_prefix):]
                if has_prefix:
                    local = "language_model." + local
                state[local] = f.get_tensor(k).to(torch_dtype)
        print(f"  shard {shard_idx+1}/{len(by_shard)} loaded "
              f"({time.time()-t_load_start:.1f}s)", flush=True)
        gc.collect()
    print(f"  total weight load: {time.time()-t_load_start:.1f}s", flush=True)

    info = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(info.missing_keys)}, "
          f"unexpected: {len(info.unexpected_keys)}", flush=True)
    if info.missing_keys[:5]:
        print(f"  e.g. missing: {info.missing_keys[:5]}", flush=True)
    model = model.to(torch_dtype).eval()
    del state
    gc.collect()

    # ----- Load lm_head separately -----
    lm_head_key = "lm_head.weight"
    lm_path = hf_hub_download(MODEL_ID, weight_map[lm_head_key])
    with safe_open(lm_path, framework="pt") as f:
        lm_head_w = f.get_tensor(lm_head_key).to(torch_dtype)
    print(f"  lm_head shape: {tuple(lm_head_w.shape)}", flush=True)

    VOCAB = lm_head_w.shape[0]
    print(f"  vocab: {VOCAB}", flush=True)

    # ----- Greedy continuation (autoregressive, use_cache=True) -----
    print(f"\nGreedy decode for {args.max_tokens} tokens…", flush=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    past = None
    generated_ids = []
    logits_at_step = np.empty((args.max_tokens, VOCAB), dtype=np.float32)
    top1_ids = []

    cur_ids = input_ids  # first call: full prompt
    t_decode = time.time()
    for step in range(args.max_tokens):
        t_s = time.time()
        with torch.no_grad():
            kwargs = dict(input_ids=cur_ids, use_cache=True)
            if past is not None:
                kwargs["past_key_values"] = past
            out = model(**kwargs)

            # out is a BaseModelOutput / CausalLMOutputWithPast — get hidden
            if hasattr(out, "logits") and out.logits is not None:
                last_logits = out.logits[0, -1, :].to(torch.float32)
            else:
                # ModelClass returns hidden states — apply lm_head manually
                if hasattr(out, "last_hidden_state"):
                    hidden = out.last_hidden_state
                else:
                    hidden = out[0]
                last_hidden = hidden[0, -1, :].to(torch.float32)
                last_logits = last_hidden @ lm_head_w.to(torch.float32).t()

            past = getattr(out, "past_key_values", None)

        next_id = int(last_logits.argmax().item())
        generated_ids.append(next_id)
        top1_ids.append(next_id)
        logits_at_step[step] = last_logits.cpu().numpy()

        # Print every step (so we can monitor progress live)
        tok_str = tok.decode([next_id])
        dt = time.time() - t_s
        print(f"  step {step+1:3d}/{args.max_tokens}  "
              f"id={next_id:6d}  tok={tok_str!r:>20s}  ({dt:.1f}s)", flush=True)

        # Next iteration: feed only the new token
        cur_ids = torch.tensor([[next_id]], dtype=torch.long)

    print(f"\nDecode complete: {time.time()-t_decode:.1f}s "
          f"({(time.time()-t_decode)/args.max_tokens:.2f}s/tok)", flush=True)

    generated_text = tok.decode(generated_ids, skip_special_tokens=False)
    print(f"\nFull continuation: {generated_text!r}", flush=True)

    np.savez(
        args.out,
        prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
        generated_ids=np.asarray(generated_ids, dtype=np.int32),
        logits_at_step=logits_at_step,
        top1_ids=np.asarray(top1_ids, dtype=np.int32),
        dtype=np.asarray([args.dtype]),
        prompt=np.asarray([args.prompt]),
        model_id=np.asarray([MODEL_ID]),
    )
    print(f"\nSaved → {args.out}  "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)
    print(f"Total: {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
