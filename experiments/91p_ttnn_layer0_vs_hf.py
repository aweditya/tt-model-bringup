#!/usr/bin/env python3
"""
Experiment 91p — Phase B'9.5 validation: ttnn layer 0 vs HF reference.

After the bug fixes in 91f (load linear_attn.norm/q_norm/k_norm, apply 1+w
to Qwen3_5RMSNorm weights, add per-head RMSNorm in DeltaNet + Gated Attn),
this script runs ttnn layer 0 only and compares to HF reference from 91o.

Loads just one layer's weights (~30s) so iteration is fast if more bugs
hide.

For each prompt token, calls deltanet_step_ondevice + mlp_step_ondevice
on device with the fixed kernels, saves x_after_mlp, compares to
~/tt-xla/.cache/qwen36_27b_hf_layer0_ref.npz from 91o.

Gate: cosine ≥ 0.999. If pass → run 91l with 60 tokens.

CLI flags:
  --weight-dtype {bf8,bf16,fp32}  (default: bf8)
    Override the projection/conv1d_weight dtype. Use to ablate whether
    bf8 quantization is the source of remaining drift. bf16 doubles
    weight memory but matches HF reference at higher precision.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91p_ttnn_layer0_vs_hf.py [--weight-dtype bf16]
"""
import os, sys, json, time, argparse
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
HF_REF_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_hf_layer0_ref.npz")


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight-dtype", choices=["bf8", "bf16", "fp32"], default="bf8",
                   help="dtype for projection and conv1d weights")
    args = p.parse_args()

    proj_dtype = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "fp32": ttnn.float32}[args.weight_dtype]

    print("=" * 64)
    print(f"Experiment 91p — ttnn layer 0 vs HF reference (proj_dtype={args.weight_dtype})")
    print("=" * 64)
    t_total = time.time()

    # ----- Config -----
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

    # ----- Tokenize -----
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(PROMPT)
    print(f"prompt: {PROMPT!r}  ids={prompt_ids}")

    # ----- Load embed rows we need -----
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    embed_key = "model.language_model.embed_tokens.weight"
    embed_shard = weight_map[embed_key]
    embed_path = hf_hub_download(MODEL_ID, embed_shard)
    with safe_open(embed_path, framework="pt") as f:
        embed_full = f.get_tensor(embed_key).float().numpy()
    print(f"embed loaded: shape={embed_full.shape}")

    # ----- Open device -----
    print("\nOpening device + loading layer 0 weights…")
    t0 = time.time()
    device = ttnn.open_device(device_id=0)

    # ----- Load layer 0 (linear_attention) weights using fixed loader -----
    w_np = load_layer_weights_all(0, 'linear_attention')
    print(f"layer 0 weight keys loaded: {sorted(w_np.keys())}")
    # Sanity: confirm linear_attn_norm is present
    assert 'linear_attn_norm' in w_np, "linear_attn_norm MISSING — fix not applied"
    assert 'input_layernorm' in w_np, "input_layernorm MISSING"
    assert 'post_attention_layernorm' in w_np, "post_attention_layernorm MISSING"
    # Sanity: confirm (1+w) was applied to input_layernorm (mean should now be near 1)
    print(f"  input_layernorm stats:        "
          f"mean={w_np['input_layernorm'].mean():+.4f} "
          f"std={w_np['input_layernorm'].std():.4f}")
    print(f"  post_attention_layernorm:    "
          f"mean={w_np['post_attention_layernorm'].mean():+.4f} "
          f"std={w_np['post_attention_layernorm'].std():.4f}")
    print(f"  linear_attn_norm (no 1+w):   "
          f"mean={w_np['linear_attn_norm'].mean():+.4f} "
          f"std={w_np['linear_attn_norm'].std():.4f}")

    w_tt = {}
    for k, arr in w_np.items():
        if k == 'conv1d_weight' and arr.ndim == 3:
            arr = arr.squeeze(1)
        if 'proj' in k or k == 'conv1d_weight':
            dt = proj_dtype  # CLI override; default bf8
        elif k in ('A_log', 'dt_bias'):
            dt = ttnn.float32
        else:
            dt = ttnn.bfloat16
        w_tt[k] = upload(arr, device, dtype=dt)
    print(f"  uploaded {len(w_tt)} weight tensors in {time.time()-t0:.1f}s "
          f"(proj_dtype={args.weight_dtype})")

    # ----- Initialize DeltaNet state (one layer) -----
    ssm_state = upload(
        np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
        device, dtype=ttnn.float32)
    conv_state = upload(
        np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
        device, dtype=ttnn.float32)

    # ----- Forward each prompt token through embed + deltanet + mlp -----
    print(f"\nForwarding {len(prompt_ids)} prompt tokens through embed + layer 0…")
    outputs = []  # post-MLP hidden state per token
    for pos, tok_id in enumerate(prompt_ids):
        x_np = embed_full[tok_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        x_tt, ssm_state, conv_state = deltanet_step_ondevice(
            x_tt, w_tt, ssm_state, conv_state, cfg)
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)
        x_np_out = ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]
        outputs.append(x_np_out)
        print(f"  pos {pos}  ‖x‖={np.linalg.norm(x_np_out):.4f}")

    ttnn_layer0 = np.stack(outputs, axis=0)  # [seq, hidden]
    ttnn.close_device(device)

    # ----- Compare to HF reference -----
    if not os.path.exists(HF_REF_PATH):
        print(f"\nHF reference not found at {HF_REF_PATH}")
        print("Run experiments/91o_hf_reference_layer0.py first.")
        sys.exit(1)

    ref = np.load(HF_REF_PATH)
    hf_layer0 = ref['hf_layer0']  # [seq, hidden]
    print(f"\nHF reference: shape={hf_layer0.shape}")
    print(f"ttnn output:  shape={ttnn_layer0.shape}")

    if hf_layer0.shape != ttnn_layer0.shape:
        print(f"  SHAPE MISMATCH — cannot compare directly")
        sys.exit(1)

    # ----- Cosine per token + overall -----
    print("\nPer-token comparison:")
    print(f"  pos | ‖HF‖     ‖ttnn‖    cosine     max|Δ|")
    print(f"  ----+----------------------------------------")
    for pos in range(len(prompt_ids)):
        hf_v = hf_layer0[pos]
        tt_v = ttnn_layer0[pos]
        c = cosine(hf_v, tt_v)
        md = float(np.abs(hf_v.astype(np.float64) - tt_v.astype(np.float64)).max())
        print(f"   {pos}  | {np.linalg.norm(hf_v):8.4f}  {np.linalg.norm(tt_v):8.4f}  "
              f"{c:9.6f}  {md:9.4f}")

    # Verdict (last token cosine — closest to what matters for decode)
    final_cos = cosine(hf_layer0[-1], ttnn_layer0[-1])
    print(f"\n  ┌─────────────────────────────────────")
    print(f"  │ Final-token cosine: {final_cos:.6f}")
    print(f"  │")
    if final_cos >= 0.999:
        print(f"  │ ✓ GATE PASSED. Run 91l for full generation test.")
    elif final_cos >= 0.99:
        print(f"  │ ~ close but not 0.999. Investigate remaining drift.")
    elif final_cos >= 0.9:
        print(f"  │ ✗ partial fix. Some bug remains.")
    else:
        print(f"  │ ✗ still broken. Major bug remains.")
    print(f"  └─────────────────────────────────────")

    print(f"\nTotal elapsed: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
