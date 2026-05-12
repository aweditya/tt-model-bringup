#!/usr/bin/env python3
"""
Experiment 91r — Plan A: per-layer intrinsic cosine vs HF.

The naive "run our 64-layer forward and compare to HF" sees the cosine
compound (0.997^64 ≈ 0.82 at the end), hiding WHERE the drift originates.

This script decomposes the drift: for each layer N in a sampled set,
  - load HF's hidden_states[N] (input to layer N)
  - run ttnn layer N in isolation (single layer, fresh state)
  - compare ttnn output to HF's hidden_states[N+1]
  - report cosine per position

The first layer with notably-lower cosine is the bug source. In
particular, we want to compare:
  - DeltaNet layers (linear_attention; we validated layer 0 → 0.997)
  - Full Attention layers (we added q_norm/k_norm but never validated layer 3)

Layers tested by default: [0, 1, 2, 3, 7, 11, 15, 31, 47, 63]

Prerequisites:
  - HF per-layer hidden states at ~/tt-xla/.cache/hf_per_layer_hidden_states.npz
    (run: experiments/utils/hf_full_model_oracle.py --dump-hidden-states)

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91r_per_layer_diff.py [--layers 0,3,7,...]
"""
import os, sys, json, time, gc, argparse
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
gated_attn_step_ondevice = _91f.gated_attn_step_ondevice
mlp_step_ondevice = _91f.mlp_step_ondevice
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
# Match 91l's KV cache size so we reuse the same JIT-cached kernels.
# (Smaller sizes like 64 trigger fresh kernel variants which hit the
# upstream ttnn LLK 'int32_to_float' int→RoundMode bug.)
MAX_POS = 256
HF_HIDDEN_PATH = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")
DEFAULT_LAYERS = [0, 1, 2, 3, 7, 11, 15, 31, 47, 63]

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def upload_weights(w_np, device, proj_dtype=ttnn.bfloat8_b):
    """Apply the 91l dtype policy when uploading per-layer weights.
    proj_dtype: override for projection + conv1d weights (default bf8,
    can be bf16 / fp32 to ablate quantization-noise hypothesis)."""
    w_tt = {}
    for k, arr in w_np.items():
        if k == 'conv1d_weight' and arr.ndim == 3:
            arr = arr.squeeze(1)
        if 'proj' in k or k == 'conv1d_weight':
            dt = proj_dtype
        elif k in ('A_log', 'dt_bias'):
            dt = ttnn.float32
        else:
            dt = ttnn.bfloat16
        w_tt[k] = upload(arr, device, dtype=dt)
    return w_tt


_PROJ_DTYPE = ttnn.bfloat8_b  # set in main from CLI


def test_linear_attention_layer(layer_idx, hf_in, hf_out, cfg, device):
    """Forward ttnn DeltaNet+MLP for layer_idx, return per-position outputs."""
    HIDDEN = cfg['hidden']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    w_np = load_layer_weights_all(layer_idx, 'linear_attention')
    w_tt = upload_weights(w_np, device, proj_dtype=_PROJ_DTYPE)

    # Fresh state
    ssm_state = upload(
        np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
        device, dtype=ttnn.float32)
    conv_state = upload(
        np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
        device, dtype=ttnn.float32)

    # hf_in shape: [seq, hidden]. Feed each position sequentially.
    seq = hf_in.shape[0]
    outputs = []
    for pos in range(seq):
        x_np = hf_in[pos]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        x_tt, ssm_state, conv_state = deltanet_step_ondevice(
            x_tt, w_tt, ssm_state, conv_state, cfg)
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        outputs.append(ttnn.to_torch(x_tt).float().cpu().numpy().flatten()[:HIDDEN])
    return np.stack(outputs)


def test_full_attention_layer(layer_idx, hf_in, hf_out, cfg, device):
    """Forward ttnn Gated Attention + MLP for layer_idx, return per-position outputs."""
    HIDDEN = cfg['hidden']

    w_np = load_layer_weights_all(layer_idx, 'full_attention')
    w_tt = upload_weights(w_np, device, proj_dtype=_PROJ_DTYPE)

    # Fresh KV cache
    kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)

    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    def rope_for_pos(pos):
        angles = pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        return (upload(cos_np, device, dtype=ttnn.float32),
                upload(sin_np, device, dtype=ttnn.float32))

    seq = hf_in.shape[0]
    outputs = []
    for pos in range(seq):
        x_np = hf_in[pos]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt, sin_tt = rope_for_pos(pos)
        cur_pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
        x_tt, kv_k, kv_v = gated_attn_step_ondevice(
            x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, pos, cos_tt, sin_tt, cfg, device)
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        outputs.append(ttnn.to_torch(x_tt).float().cpu().numpy().flatten()[:HIDDEN])
    return np.stack(outputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=str, default=",".join(map(str, DEFAULT_LAYERS)),
                   help="Comma-separated layer indices to test")
    p.add_argument("--weight-dtype", choices=["bf8", "bf16", "fp32"], default="bf8",
                   help="Override projection + conv1d weight dtype to ablate "
                        "quantization-noise hypothesis (default bf8, matches 91l)")
    args = p.parse_args()
    layers_to_test = [int(x) for x in args.layers.split(",")]
    global _PROJ_DTYPE
    _PROJ_DTYPE = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}[args.weight_dtype]

    print("=" * 64)
    print(f"Experiment 91r — per-layer intrinsic cosine "
          f"(layers {layers_to_test}, weight_dtype={args.weight_dtype})")
    print("=" * 64)
    t_total = time.time()

    if not os.path.exists(HF_HIDDEN_PATH):
        print(f"\nERROR: HF hidden states missing at {HF_HIDDEN_PATH}")
        print("Run: experiments/utils/hf_full_model_oracle.py --dump-hidden-states")
        sys.exit(1)

    print(f"\nLoading HF hidden states from {HF_HIDDEN_PATH}…")
    hf_data = np.load(HF_HIDDEN_PATH)
    n_hidden = sum(1 for k in hf_data.keys() if k.startswith("hidden_"))
    print(f"  found {n_hidden} hidden states (embed + {n_hidden-1} layer outputs)")

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

    print("\nOpening device…")
    device = ttnn.open_device(device_id=0)

    results = []
    for layer_idx in layers_to_test:
        layer_type = 'linear_attention' if layer_idx % 4 != 3 else 'full_attention'
        print(f"\n--- Layer {layer_idx} ({layer_type}) ---")
        hf_in = hf_data[f"hidden_{layer_idx}"][0]      # [seq, hidden]
        hf_out = hf_data[f"hidden_{layer_idx+1}"][0]  # [seq, hidden]

        t_layer = time.time()
        if layer_type == 'linear_attention':
            ttnn_out = test_linear_attention_layer(layer_idx, hf_in, hf_out, cfg, device)
        else:
            ttnn_out = test_full_attention_layer(layer_idx, hf_in, hf_out, cfg, device)
        dt = time.time() - t_layer

        cos_per_pos = [cosine(hf_out[p], ttnn_out[p]) for p in range(hf_in.shape[0])]
        norm_pairs = [(float(np.linalg.norm(hf_out[p])), float(np.linalg.norm(ttnn_out[p])))
                      for p in range(hf_in.shape[0])]
        results.append({
            'layer': layer_idx,
            'type': layer_type,
            'cosines': cos_per_pos,
            'norms_hf': [p[0] for p in norm_pairs],
            'norms_ttnn': [p[1] for p in norm_pairs],
            'dt_sec': dt,
        })
        print(f"  cosines per pos: {[f'{c:.5f}' for c in cos_per_pos]}")
        print(f"  norms HF / ttnn: {[f'{a:.2f}/{b:.2f}' for a, b in norm_pairs]}")
        print(f"  layer test took {dt:.1f}s")
        gc.collect()

    ttnn.close_device(device)

    # Final table
    print("\n" + "=" * 80)
    print("PER-LAYER COSINE SUMMARY (last position)")
    print("=" * 80)
    print(f"{'layer':>6s} {'type':>20s}  {'pos 0':>9s} {'pos 1':>9s} {'pos 2':>9s} "
          f"{'pos 3':>9s} {'pos 4':>9s}  {'worst':>9s}")
    print("-" * 80)
    for r in results:
        cs = r['cosines']
        line = f"{r['layer']:6d} {r['type']:>20s}  "
        for c in cs:
            line += f"{c:9.5f} "
        line += f" {min(cs):9.5f}"
        print(line)

    out_path = os.path.expanduser("~/tt-xla/.cache/per_layer_diff_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
