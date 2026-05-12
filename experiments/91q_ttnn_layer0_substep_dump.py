#!/usr/bin/env python3
"""
Experiment 91q — Minimal substep dump (post-deltanet, post-mlp) for layer 0.

The aggressive inline capture in the v1 of this file triggered fresh
ttnn JIT kernel variants that hit the bf16/fp32 weight JIT bug
(`cannot convert 'int' to 'sfpi::RoundMode'`). Workaround: capture only
at the outer boundaries of 91f's existing (known-working) kernels.

Capture points:
  - pre_layer:    x just after embed lookup
  - post_deltanet: x after deltanet_step_ondevice (= residual + linear_attn.out)
  - post_mlp:     x after mlp_step_ondevice (= residual + mlp.out) — this is
                  what HF reports as __layer__.out

If post-deltanet cosine drops vs HF, the bug is in DeltaNet.
If post-mlp cosine drops further vs post-deltanet, the bug is in MLP.

Saves to ~/tt-xla/.cache/ttnn_layer0_substeps.npz (5 positions × 3 captures = 15
tensors). Use experiments/utils/substep_compare.py for the diff.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91q_ttnn_layer0_substep_dump.py
"""
import os, sys, json, time
import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_91f", os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py"))
_91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91f)
deltanet_step_ondevice = _91f.deltanet_step_ondevice
mlp_step_ondevice = _91f.mlp_step_ondevice
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
PROMPT = "The capital of France is"
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/ttnn_layer0_substeps.npz")


def to_np(t_tt):
    return ttnn.to_torch(t_tt).float().cpu().numpy()


def main():
    print("=" * 64)
    print("Experiment 91q — minimal substep dump for layer 0")
    print("=" * 64)
    t_total = time.time()

    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    cfg = {
        'hidden':      text_cfg['hidden_size'],
        'n_k_heads':   text_cfg['linear_num_key_heads'],
        'n_v_heads':   text_cfg['linear_num_value_heads'],
        'k_dim':       text_cfg['linear_key_head_dim'],
        'v_dim':       text_cfg['linear_value_head_dim'],
        'conv_kernel': text_cfg['linear_conv_kernel_dim'],
        'n_q_heads':   text_cfg['num_attention_heads'],
        'n_kv_heads':  text_cfg['num_key_value_heads'],
        'head_dim':    text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
    }
    HIDDEN = cfg['hidden']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(PROMPT)
    print(f"prompt: {PROMPT!r}  ids={prompt_ids}")

    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    embed_path = hf_hub_download(MODEL_ID, weight_map["model.language_model.embed_tokens.weight"])
    with safe_open(embed_path, framework="pt") as f:
        embed_full = f.get_tensor("model.language_model.embed_tokens.weight").float().numpy()

    print("\nOpening device + loading layer 0 weights…")
    t0 = time.time()
    device = ttnn.open_device(device_id=0)
    w_np = load_layer_weights_all(0, 'linear_attention')
    w_tt = {}
    for k, arr in w_np.items():
        if k == 'conv1d_weight' and arr.ndim == 3:
            arr = arr.squeeze(1)
        if 'proj' in k or k == 'conv1d_weight':
            dt = ttnn.bfloat8_b
        elif k in ('A_log', 'dt_bias'):
            dt = ttnn.float32
        else:
            dt = ttnn.bfloat16
        w_tt[k] = upload(arr, device, dtype=dt)
    print(f"  weights uploaded in {time.time()-t0:.1f}s")

    ssm_state = upload(
        np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
        device, dtype=ttnn.float32)
    conv_state = upload(
        np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
        device, dtype=ttnn.float32)

    # IMPORTANT: defer all to_np calls until AFTER the full forward to avoid
    # triggering ttnn JIT kernel variants that hit the bf16/fp32 JIT bug.
    # Hold device tensor references throughout; read them at the end.
    captures_tt = {}
    print(f"\nForwarding {len(prompt_ids)} prompt tokens (no host reads until end)…")
    for pos, tok_id in enumerate(prompt_ids):
        x_np = embed_full[tok_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        prefix = f"pos{pos}"
        captures_tt[f"{prefix}.pre_layer"] = x_tt

        x_tt, ssm_state, conv_state = deltanet_step_ondevice(
            x_tt, w_tt, ssm_state, conv_state, cfg)
        captures_tt[f"{prefix}.post_deltanet"] = x_tt

        x_tt = mlp_step_ondevice(x_tt, w_tt)
        captures_tt[f"{prefix}.post_mlp"] = x_tt
        print(f"  pos {pos} forward done")

    ttnn.synchronize_device(device)
    print(f"\nReading {len(captures_tt)} captured tensors to host…")
    caps = {k: to_np(v) for k, v in captures_tt.items()}
    for pos in range(len(prompt_ids)):
        print(f"  pos {pos}  "
              f"‖pre‖={np.linalg.norm(caps[f'pos{pos}.pre_layer']):.4f}  "
              f"‖post_deltanet‖={np.linalg.norm(caps[f'pos{pos}.post_deltanet']):.4f}  "
              f"‖post_mlp‖={np.linalg.norm(caps[f'pos{pos}.post_mlp']):.4f}")

    np.savez(OUT_PATH, **{k: v for k, v in caps.items()})
    print(f"\nSaved {len(caps)} tensors → {OUT_PATH}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
