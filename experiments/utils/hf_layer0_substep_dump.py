#!/usr/bin/env python3
"""
Permanent utility — dump HF layer 0 substep activations for diff-vs-ours.

Used when the final-layer cosine is close (e.g., 0.997) but not at gate
(0.999), and we need to localize WHICH substep introduces drift. HF is
the authoritative oracle; our impl matches it where we got the equations
right and diverges where we got them wrong.

Registers PyTorch forward hooks on every named submodule of
Qwen3_5DecoderLayer for layer 0 (linear_attention variant for layer 0 of
Qwen3.6-27B; works for any DeltaNet layer index passed via --layer).

Each hook captures `output` (and where useful, `input` too). All captures
are saved as npz to ~/tt-xla/.cache/hf_layer{N}_substeps.npz so a future
ttnn-side diff script can load and compare per-substep.

Submodule paths captured (for linear_attention layer):
  input_layernorm
  linear_attn (the whole module's output)
  linear_attn.conv1d
  linear_attn.in_proj_qkv, in_proj_z, in_proj_a, in_proj_b
  linear_attn.norm (RMSNormGated)
  linear_attn.out_proj
  post_attention_layernorm
  mlp.gate_proj, mlp.up_proj, mlp.down_proj
  mlp (whole module)

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/utils/hf_layer0_substep_dump.py [--layer N] [--prompt P]
"""
import os, sys, json, time, argparse
import importlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.6-27B"
DEFAULT_PROMPT = "The capital of France is"


def to_np(t):
    """Detach a torch tensor and return fp32 numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    if isinstance(t, (tuple, list)):
        return [to_np(x) for x in t]
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--input-from-hidden", action="store_true",
                   help="Use hidden_N from ~/tt-xla/.cache/hf_per_layer_hidden_states.npz "
                        "as input (= real layer N input in production), instead of feeding "
                        "embed(prompt) through this isolated layer. Required for cross-check "
                        "against ttnn substep dumps that use the same input.")
    args = p.parse_args()

    print("=" * 64)
    print(f"HF substep dump — layer {args.layer}, prompt {args.prompt!r}")
    print("=" * 64)

    # ----- Find correct modeling module -----
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, 'text_config', full_cfg)
    config_module_path = type(text_cfg).__module__
    modeling_module_path = config_module_path.replace('configuration_', 'modeling_')
    print(f"text config class: {type(text_cfg).__name__}")
    print(f"importing: {modeling_module_path}")
    mqn = importlib.import_module(modeling_module_path)

    # Find the right DecoderLayer + RotaryEmbedding class
    DecoderLayer = None
    for name in ('Qwen3_5DecoderLayer', 'Qwen3NextDecoderLayer'):
        if hasattr(mqn, name):
            DecoderLayer = getattr(mqn, name)
            break
    if DecoderLayer is None:
        print("No DecoderLayer class found"); sys.exit(1)
    RotaryEmb = None
    for name in ('Qwen3_5TextRotaryEmbedding', 'Qwen3_5RotaryEmbedding',
                 'Qwen3NextRotaryEmbedding'):
        if hasattr(mqn, name):
            RotaryEmb = getattr(mqn, name)
            break

    # ----- Construct + load layer + embed -----
    layer = DecoderLayer(text_cfg, layer_idx=args.layer).float().eval()
    print(f"layer.layer_type = {layer.layer_type}")

    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    # Load layer weights
    prefix = f"model.language_model.layers.{args.layer}."
    layer_keys = sorted(k for k in weight_map.keys() if k.startswith(prefix))
    state = {}
    by_shard = {}
    for k in layer_keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    for shard, ks in by_shard.items():
        shard_path = hf_hub_download(MODEL_ID, shard)
        with safe_open(shard_path, framework="pt") as f:
            for k in ks:
                state[k.replace(prefix, "")] = f.get_tensor(k).float()
    info = layer.load_state_dict(state, strict=False)
    print(f"layer.load_state_dict: missing={info.missing_keys}, unexpected={info.unexpected_keys}")

    # Load embed
    embed_key = "model.language_model.embed_tokens.weight"
    embed_path = hf_hub_download(MODEL_ID, weight_map[embed_key])
    with safe_open(embed_path, framework="pt") as f:
        embed_w = f.get_tensor(embed_key).float()
    embed = torch.nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
    embed.weight.data.copy_(embed_w)
    embed.eval()

    # ----- Register forward hooks on EVERY named submodule -----
    intermediates = {}  # name → numpy
    handles = []
    for name, mod in layer.named_modules():
        if name == "":  # skip the layer itself; we'll capture its output via main forward
            continue
        def make_hook(n):
            def hook(module, inp, out):
                # input may be a tuple
                if isinstance(inp, tuple) and len(inp) >= 1:
                    intermediates[f"{n}.in"] = to_np(inp[0]) if isinstance(inp[0], torch.Tensor) else None
                intermediates[f"{n}.out"] = to_np(out)
            return hook
        handles.append(mod.register_forward_hook(make_hook(name)))

    # ----- Forward pass -----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = torch.tensor([tok.encode(args.prompt)])
    seq_len = input_ids.shape[1]
    print(f"prompt ids: {input_ids.tolist()}, seq_len={seq_len}")

    with torch.no_grad():
        if args.input_from_hidden:
            # Load real layer-N input from full-forward dump
            per_layer_path = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")
            if not os.path.exists(per_layer_path):
                print(f"ERROR: {per_layer_path} missing. Run hf_full_model_oracle.py "
                      f"--dump-hidden-states first.")
                sys.exit(1)
            hf_data = np.load(per_layer_path)
            hidden_states = torch.from_numpy(hf_data[f"hidden_{args.layer}"]).float()
            print(f"  using REAL layer {args.layer} input from {per_layer_path}")
            print(f"  hidden_states shape: {tuple(hidden_states.shape)}, "
                  f"‖·‖={hidden_states.norm().item():.4f}")
        else:
            hidden_states = embed(input_ids).float()
        intermediates['__embed__.out'] = to_np(hidden_states)

        # Build forward kwargs
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        forward_kwargs = {
            'position_ids': position_ids,
            'attention_mask': None,
            'past_key_values': None,
        }
        if RotaryEmb is not None:
            rot = RotaryEmb(config=text_cfg).float().eval()
            cos, sin = rot(hidden_states, position_ids)
            forward_kwargs['position_embeddings'] = (cos.float(), sin.float())
            intermediates['__rope__.cos'] = to_np(cos)
            intermediates['__rope__.sin'] = to_np(sin)

        t0 = time.time()
        out = layer(hidden_states, **forward_kwargs)
        dt = time.time() - t0

    intermediates['__layer__.out'] = to_np(out)
    intermediates['__layer__.in'] = to_np(hidden_states)
    print(f"layer forward took {dt*1000:.0f} ms")

    # ----- Remove hooks -----
    for h in handles:
        h.remove()

    # ----- Save -----
    out_dir = os.path.expanduser("~/tt-xla/.cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hf_layer{args.layer}_substeps.npz")

    # filter None values (input hooks that weren't tensors)
    filtered = {k: v for k, v in intermediates.items() if v is not None}
    # Replace tuple values (rare) — keep only first element
    flat = {}
    for k, v in filtered.items():
        if isinstance(v, list):
            for i, vv in enumerate(v):
                if isinstance(vv, np.ndarray):
                    flat[f"{k}[{i}]"] = vv
        elif isinstance(v, np.ndarray):
            flat[k] = v

    np.savez(out_path, **flat)
    print(f"\nSaved {len(flat)} substep tensors → {out_path}")
    print("\nSubstep summary:")
    for name in sorted(flat.keys()):
        a = flat[name]
        s = f"  {name:>55s}: shape={str(a.shape):>30s}  ‖·‖={np.linalg.norm(a):10.4f}"
        print(s)


if __name__ == "__main__":
    main()
