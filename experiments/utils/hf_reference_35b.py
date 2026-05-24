#!/usr/bin/env python3
"""B16-oracle — HuggingFace transformers reference for Qwen3.6-35B-A3B.

Generates a layer-by-layer activation snapshot that `server_35b_ttnn.py`
can use as ground truth for cosine comparison.

Why: the on-device server runs end-to-end on (1,4) mesh but produces
gibberish ('ythe' at token 0) because of conv1d state-shift / RoPE /
KV-cache / GQA chip-mapping placeholders. To debug the math we need
an independent oracle to compute layer-wise cosine against.

What it saves (under .cache/hf_oracle_35b/):
  - meta.json: prompt, prompt_ids, predicted_token, predicted_text, layer_types
  - prompt_ids.npy: [seq] int32 tokens
  - hidden_states.npy: [41, seq, HIDDEN] fp32 — index 0 is embed output,
                      indices 1..40 are after each decoder layer
  - logits.npy: [seq, VOCAB] fp32 final logits (post lm_head)
  - final_norm.npy: [seq, HIDDEN] fp32 (post final norm, pre lm_head)
  - argmax.npy: [seq] int32 — greedy argmax at every position

Memory: HF Qwen3_5MoeForCausalLM in bfloat16 on CPU uses ~70 GB.
qb1 has 503 GB total / ~478 GB available so this fits comfortably.

Run (qb1):
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/hf_reference_35b.py
"""
import json
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_35b"

# Same prompt server_35b_ttnn.py smoke uses (5 tokens). Override with --prompt
# for longer ladders; --output-dir lets you keep parallel oracles without
# clobbering this default B16 oracle.
DEFAULT_PROMPT = "The capital of France is"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None,
                    help="text to tokenize (overrides --prompt-file)")
    ap.add_argument("--prompt-file", default=None,
                    help="path to a UTF-8 file whose contents are the prompt")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                    help="directory to save oracle artifacts")
    ap.add_argument("--no-special-tokens", action="store_true",
                    help="pass add_special_tokens=False to the tokenizer. Use when the prompt is "
                         "already chat-template-rendered (contains <|im_start|> etc.) — otherwise "
                         "tokenizer double-adds BOS-like prefix tokens.")
    args = ap.parse_args()

    if args.prompt is not None:
        prompt_text = args.prompt
    elif args.prompt_file is not None:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt_text = DEFAULT_PROMPT

    global PROMPT, OUT_DIR
    PROMPT = prompt_text
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading config + tokenizer ({MODEL_ID})…")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text_cfg = cfg.text_config
    layer_types = list(text_cfg.layer_types)
    n_layers = text_cfg.num_hidden_layers
    log(f"  {n_layers} layers; types: "
        f"{sum(1 for t in layer_types if t == 'linear_attention')} DN, "
        f"{sum(1 for t in layer_types if t == 'full_attention')} attn")

    log("loading HF model on CPU in bfloat16 (may take a few minutes)…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    log(f"  model loaded in {time.time()-t0:.0f}s")

    log(f"tokenize prompt ({'no special tokens' if args.no_special_tokens else 'add special tokens'}): "
        f"{PROMPT[:80]!r}{'…' if len(PROMPT) > 80 else ''}")
    prompt_ids = tok.encode(PROMPT, return_tensors="pt",
                            add_special_tokens=not args.no_special_tokens)  # [1, seq]
    seq = prompt_ids.shape[1]
    log(f"  prompt_ids shape={list(prompt_ids.shape)} = {prompt_ids[0].tolist()}")

    # Register forward hooks on Layer 0 sub-modules so we can capture
    # intra-layer activations (matching server_35b_ttnn's sub_capture points).
    # Hook output is the module's OUTPUT tensor at the given submodule.
    intra = {}
    handles = []
    L0 = model.model.layers[0]

    def make_hook(name):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            intra[name] = t.detach().float().cpu().numpy()
        return hook

    # Match server_35b_ttnn's sub_capture names: in_norm, mixer_out, post_attn_norm, moe_out
    handles.append(L0.input_layernorm.register_forward_hook(make_hook("in_norm")))
    # Layer 0 is linear_attention; module is `linear_attn` (DN)
    if layer_types[0] == "linear_attention":
        handles.append(L0.linear_attn.register_forward_hook(make_hook("mixer_out")))
        # DN sub-module hooks: 7 hookable points within DN
        dn = L0.linear_attn
        handles.append(dn.in_proj_qkv.register_forward_hook(make_hook("dn_in_proj_qkv")))
        handles.append(dn.in_proj_z.register_forward_hook(make_hook("dn_in_proj_z")))
        handles.append(dn.in_proj_a.register_forward_hook(make_hook("dn_in_proj_a")))
        handles.append(dn.in_proj_b.register_forward_hook(make_hook("dn_in_proj_b")))
        handles.append(dn.conv1d.register_forward_hook(make_hook("dn_conv1d")))
        handles.append(dn.norm.register_forward_hook(make_hook("dn_norm")))
        handles.append(dn.out_proj.register_forward_hook(make_hook("dn_out_proj")))

        # PRE-hook on norm captures the input: (core_attn_out, gate=z).
        # core_attn_out shape [B*S*NV_HEADS, head_v_dim]; z same.
        def pre_norm_hook(_module, inp):
            args = inp if isinstance(inp, tuple) else (inp,)
            if len(args) >= 1:
                intra["dn_core_attn_out"] = args[0].detach().float().cpu().numpy()
            if len(args) >= 2:
                intra["dn_norm_gate_z"] = args[1].detach().float().cpu().numpy()
        handles.append(dn.norm.register_forward_pre_hook(pre_norm_hook))
    else:
        handles.append(L0.self_attn.register_forward_hook(make_hook("mixer_out")))
    handles.append(L0.post_attention_layernorm.register_forward_hook(make_hook("post_attn_norm")))
    handles.append(L0.mlp.register_forward_hook(make_hook("moe_out")))

    log("HF forward pass with output_hidden_states=True + L0 sub-hooks…")
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids,
                output_hidden_states=True,
                use_cache=False,  # we only need a single forward, no incremental
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()
    log(f"  forward took {time.time()-t0:.1f}s")
    log(f"  L0 sub-captures: {sorted(intra.keys())} "
        f"shapes: {{ {', '.join(f'{k}={list(v.shape)}' for k, v in intra.items())} }}")

    hidden_states = out.hidden_states  # tuple of (n_layers+1) tensors [1, seq, HIDDEN]
    logits = out.logits  # [1, seq, VOCAB]
    log(f"  hidden_states: {len(hidden_states)} tensors, "
        f"shape per layer: {list(hidden_states[0].shape)}")
    log(f"  logits shape: {list(logits.shape)}")

    # Stack hidden states: [n_layers+1, seq, HIDDEN] in fp32
    hs_stack = torch.stack([h[0] for h in hidden_states]).float().numpy()
    logits_np = logits[0].float().numpy()  # [seq, VOCAB]

    # Final norm output (pre lm_head) — last hidden state IS post-final-norm
    # in HF Qwen3_5Moe model (verified by inspection: see modeling_qwen3_5_moe.py).
    final_norm_np = hs_stack[-1].copy()  # [seq, HIDDEN]

    # Greedy argmax at every position
    argmax_np = logits_np.argmax(axis=-1).astype(np.int32)
    predicted_token = int(argmax_np[-1])
    predicted_text = tok.decode([predicted_token])
    log(f"  predicted next token at pos {seq-1}: id={predicted_token} "
        f"text={predicted_text!r}")

    log("saving artifacts…")
    np.save(OUT_DIR / "prompt_ids.npy", prompt_ids[0].numpy().astype(np.int32))
    np.save(OUT_DIR / "hidden_states.npy", hs_stack)
    np.save(OUT_DIR / "logits.npy", logits_np)
    np.save(OUT_DIR / "final_norm.npy", final_norm_np)
    np.save(OUT_DIR / "argmax.npy", argmax_np)
    # L0 intra-layer captures: normalize all to shape [seq, D_flat] so the
    # cosine probe can do a uniform `arr[p]` lookup per position.
    seq_local = seq
    for k, v in intra.items():
        if v.ndim >= 2 and v.shape[0] == 1:
            v = v[0]
        if k == "dn_conv1d":
            # HF shape [CONV_DIM, S+pad] — take valid seq slots and transpose
            v = v[:, :seq_local].T  # [seq, CONV_DIM]
        elif k in ("dn_norm", "dn_core_attn_out", "dn_norm_gate_z"):
            # HF shape [B*S*NV_HEADS, head_v_dim] = [seq*NV_HEADS, head_v_dim]
            # Reshape to [seq, NV_HEADS * head_v_dim] to match our reassembled
            # per-chip→full V_DIM layout. Note: pre-hook captures haven't had
            # batch stripped yet so they're [seq*NV_HEADS, head_v_dim] directly.
            assert v.ndim == 2, f"unexpected {k} shape {v.shape}"
            v = v.reshape(seq_local, -1)  # [seq, NV_HEADS * head_v_dim]
        np.save(OUT_DIR / f"L0_{k}.npy", v.astype(np.float32))
    meta = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "prompt_ids": prompt_ids[0].tolist(),
        "seq_len": seq,
        "n_layers": n_layers,
        "layer_types": layer_types,
        "hidden": int(hs_stack.shape[-1]),
        "vocab": int(logits_np.shape[-1]),
        "predicted_token": predicted_token,
        "predicted_text": predicted_text,
        "argmax_per_position": argmax_np.tolist(),
        "argmax_text_per_position": [tok.decode([int(t)]) for t in argmax_np],
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  written under {OUT_DIR}")
    log("HF oracle ready.")


if __name__ == "__main__":
    main()
