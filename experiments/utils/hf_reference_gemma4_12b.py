#!/usr/bin/env python3
"""HF reference oracle for Gemma 4 12B (google/gemma-4-12B).

Fork of `experiments/utils/hf_reference_35b.py` per REUSE MANDATE:
- KEEP: argparse, log helper, meta.json shape, hidden_states/logits/final_norm/
  argmax save layout, HF forward with output_hidden_states + use_cache=False.
- REMOVE: DN-specific hooks (no linear_attention), MoE router hooks (dense).
- ADD: pre_feedforward_layernorm + post_feedforward_layernorm hooks (Gemma 4's
  4-norm decoder structure, plan §1.5), `attn_layer_type` per-layer dump
  in meta.json (sliding vs global), `--hook-rope-layer` flag, gemma4_unified
  model-structure handling (`model.language_model.*` not `model.model.*`).

Generates a layer-by-layer activation snapshot that
`server_gemma4_unified_ttnn.py` v0.1 will cosine against.

What it saves (under .cache/hf_oracle_gemma4_12b/):
  - meta.json: prompt, prompt_ids, predicted_token, predicted_text,
               layer_types (per-layer sliding/full), hidden, vocab
  - prompt_ids.npy: [seq] int32
  - hidden_states.npy: [49, seq, 3840] fp32 — idx 0 = embed (post-scale),
                       1..48 = post each decoder layer
  - logits.npy: [seq, 262144] fp32 (POST final_logit_softcapping)
  - final_norm.npy: [seq, 3840] fp32 (post final norm, pre lm_head)
  - argmax.npy: [seq] int32
  - L0_<sub>.npy: per-sub-step captures at L0 (input_layernorm, mixer_out,
                  post_attention_layernorm, pre_feedforward_layernorm,
                  mlp_out, post_feedforward_layernorm)

Memory: 12B × bfloat16 = ~24 GB on CPU; qb1 has ~478 GB available so this is
trivial. Bootstrap should be ~3-5 min vs 35B's ~14 min.

Run (qb1):
  source ~/tt-xla/.venv-gemma4/bin/activate
  cd ~/tt-xla
  python -u experiments/utils/hf_reference_gemma4_12b.py
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
MODEL_ID = "google/gemma-4-12B"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"

# Same canonical 5-tok smoke prompt the 27B/35B oracles use, for symmetry.
DEFAULT_PROMPT = "The capital of France is"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_text_layers(model):
    """Locate the per-layer ModuleList irrespective of HF naming.

    gemma4_unified wraps the text model under `model.language_model`; older
    structures put it at `model.model`. Try both and raise loudly if neither.
    """
    # Gemma4UnifiedForConditionalGeneration → .model is Gemma4UnifiedModel
    # → .language_model is Gemma4UnifiedTextModel → .layers
    candidates = [
        ("model.model.language_model", lambda m: m.model.language_model),
        ("model.language_model",        lambda m: m.language_model),
        ("model.model",                 lambda m: m.model),
    ]
    for name, fn in candidates:
        try:
            sub = fn(model)
            if hasattr(sub, "layers"):
                return name, sub
        except AttributeError:
            continue
    raise RuntimeError(
        f"could not locate .layers on Gemma 4 model; tried "
        f"{[c[0] for c in candidates]}. Try `print(model)` to inspect."
    )


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
                         "already chat-template-rendered (contains <start_of_turn> etc.) — otherwise "
                         "tokenizer double-adds BOS-like prefix tokens.")
    ap.add_argument("--hook-attn-layer", type=int, default=None,
                    help="ALSO hook q_proj/k_proj/v_proj/q_norm/k_norm/o_proj outputs on this "
                         "decoder layer. Saves as L<N>_attn_<sub>.npy. Works for sliding OR full layers.")
    ap.add_argument("--hook-rope-layer", type=int, default=None,
                    help="ALSO hook the rotary embedding cos/sin tables actually USED at this layer. "
                         "Captures (cos, sin) at the layer's RoPE module input. Useful for "
                         "sliding-vs-global RoPE-config validation (plan §1.3 dual-rope_theta).")
    args = ap.parse_args()

    prompt_text = (args.prompt if args.prompt is not None
                   else Path(args.prompt_file).read_text(encoding="utf-8").strip()
                   if args.prompt_file is not None
                   else DEFAULT_PROMPT)

    global PROMPT, OUT_DIR
    PROMPT = prompt_text
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading config + tokenizer ({MODEL_ID})…")
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text_cfg = cfg.text_config
    layer_types = list(text_cfg.layer_types)
    n_layers = text_cfg.num_hidden_layers
    log(f"  {n_layers} layers; types: "
        f"{sum(1 for t in layer_types if t == 'sliding_attention')} sliding, "
        f"{sum(1 for t in layer_types if t == 'full_attention')} full")
    log(f"  hidden={text_cfg.hidden_size} vocab={text_cfg.vocab_size} "
        f"head_dim={text_cfg.head_dim} global_head_dim={getattr(text_cfg, 'global_head_dim', '?')}")

    log("loading HF model on CPU in bfloat16 (~3-5 min)…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
    )
    model.eval()
    log(f"  model loaded in {time.time()-t0:.0f}s")

    base_name, base = get_text_layers(model)
    log(f"  text layers located at: {base_name} (n_layers={len(base.layers)})")

    log(f"tokenize prompt ({'no special tokens' if args.no_special_tokens else 'add special tokens'}): "
        f"{PROMPT[:80]!r}{'…' if len(PROMPT) > 80 else ''}")
    prompt_ids = tok.encode(PROMPT, return_tensors="pt",
                            add_special_tokens=not args.no_special_tokens)  # [1, seq]
    seq = prompt_ids.shape[1]
    log(f"  prompt_ids shape={list(prompt_ids.shape)} = {prompt_ids[0].tolist()}")

    # Register forward hooks on L0 sub-modules. Gemma 4's decoder has FOUR
    # norms per layer (plan §1.5): input_layernorm, post_attention_layernorm,
    # pre_feedforward_layernorm, post_feedforward_layernorm. Hook all four +
    # self_attn + mlp.
    intra = {}
    handles = []
    L0 = base.layers[0]

    def make_hook(name):
        def hook(_module, _inp, output):
            t = output[0] if isinstance(output, tuple) else output
            intra[name] = t.detach().float().cpu().numpy()
        return hook

    # L0 baseline capture set — matches what server_gemma4_unified_ttnn v0.1
    # will produce as `capture` dict keys.
    handles.append(L0.input_layernorm.register_forward_hook(make_hook("in_norm")))
    handles.append(L0.self_attn.register_forward_hook(make_hook("mixer_out")))
    handles.append(L0.post_attention_layernorm.register_forward_hook(make_hook("post_attn_norm")))
    if hasattr(L0, "pre_feedforward_layernorm"):
        handles.append(L0.pre_feedforward_layernorm.register_forward_hook(make_hook("pre_ff_norm")))
    handles.append(L0.mlp.register_forward_hook(make_hook("mlp_out")))
    if hasattr(L0, "post_feedforward_layernorm"):
        handles.append(L0.post_feedforward_layernorm.register_forward_hook(make_hook("post_ff_norm")))

    if args.hook_attn_layer is not None:
        N = args.hook_attn_layer
        attn_L = base.layers[N].self_attn
        for sub_name in ("q_proj", "k_proj", "q_norm", "k_norm", "o_proj"):
            if hasattr(attn_L, sub_name):
                sub_mod = getattr(attn_L, sub_name)
                handles.append(sub_mod.register_forward_hook(make_hook(f"attn_L{N}_{sub_name}")))
        # v_proj may be None on global layers when attention_k_eq_v=true (§1.10)
        if getattr(attn_L, "v_proj", None) is not None:
            handles.append(attn_L.v_proj.register_forward_hook(make_hook(f"attn_L{N}_v_proj")))
        log(f"  hooking attn submodules on layer {N} ({layer_types[N]})")

    if args.hook_rope_layer is not None:
        N = args.hook_rope_layer
        # The rotary module location varies; try common attribute names.
        attn_L = base.layers[N].self_attn
        for rope_attr in ("rotary_emb", "rope"):
            if hasattr(attn_L, rope_attr):
                rope_mod = getattr(attn_L, rope_attr)
                def rope_hook(_module, _inp, output, _N=N):
                    cos, sin = (output[0], output[1]) if isinstance(output, tuple) else (None, None)
                    if cos is not None:
                        intra[f"rope_L{_N}_cos"] = cos.detach().float().cpu().numpy()
                        intra[f"rope_L{_N}_sin"] = sin.detach().float().cpu().numpy()
                handles.append(rope_mod.register_forward_hook(rope_hook))
                log(f"  hooking rope at layer {N}.{rope_attr} ({layer_types[N]})")
                break

    log("HF forward pass with output_hidden_states=True + L0 sub-hooks…")
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model(
                input_ids=prompt_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()
    log(f"  forward took {time.time()-t0:.1f}s")
    log(f"  L0 sub-captures: {sorted(intra.keys())}")
    for k, v in intra.items():
        log(f"    {k}: shape={list(v.shape)}")

    hidden_states = out.hidden_states  # tuple of (n_layers+1) tensors [1, seq, HIDDEN]
    logits = out.logits  # [1, seq, VOCAB] — POST final_logit_softcapping
    log(f"  hidden_states: {len(hidden_states)} tensors, "
        f"shape per layer: {list(hidden_states[0].shape)}")
    log(f"  logits shape: {list(logits.shape)}")

    hs_stack = torch.stack([h[0] for h in hidden_states]).float().numpy()
    logits_np = logits[0].float().numpy()
    final_norm_np = hs_stack[-1].copy()
    argmax_np = logits_np.argmax(axis=-1).astype(np.int32)
    predicted_token = int(argmax_np[-1])
    predicted_text = tok.decode([predicted_token])
    log(f"  predicted next token at pos {seq-1}: id={predicted_token} text={predicted_text!r}")

    log("saving artifacts…")
    np.save(OUT_DIR / "prompt_ids.npy", prompt_ids[0].numpy().astype(np.int32))
    np.save(OUT_DIR / "hidden_states.npy", hs_stack)
    np.save(OUT_DIR / "logits.npy", logits_np)
    np.save(OUT_DIR / "final_norm.npy", final_norm_np)
    np.save(OUT_DIR / "argmax.npy", argmax_np)
    seq_local = seq
    for k, v in intra.items():
        if v.ndim >= 2 and v.shape[0] == 1:
            v = v[0]
        np.save(OUT_DIR / f"L0_{k}.npy", v.astype(np.float32))
    meta = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "prompt_ids": prompt_ids[0].tolist(),
        "seq_len": seq,
        "n_layers": n_layers,
        "layer_types": layer_types,  # sliding_attention vs full_attention per layer
        "hidden": int(hs_stack.shape[-1]),
        "vocab": int(logits_np.shape[-1]),
        "head_dim": text_cfg.head_dim,
        "global_head_dim": getattr(text_cfg, "global_head_dim", None),
        "sliding_window": getattr(text_cfg, "sliding_window", None),
        "final_logit_softcapping": getattr(text_cfg, "final_logit_softcapping", None),
        "tie_word_embeddings": getattr(text_cfg, "tie_word_embeddings", None),
        "hidden_activation": getattr(text_cfg, "hidden_activation", None),
        "predicted_token": predicted_token,
        "predicted_text": predicted_text,
        "argmax_per_position": argmax_np.tolist(),
        "argmax_text_per_position": [tok.decode([int(t)]) for t in argmax_np],
        "base_attr_path": base_name,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  written under {OUT_DIR}")
    log("Gemma 4 12B HF oracle ready.")


if __name__ == "__main__":
    main()
