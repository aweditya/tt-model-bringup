#!/usr/bin/env python3
"""
Experiment 91c — Phase B′3 ttnn implementation of Qwen3.6-27B layer 0.

Wires RMSNorm → DeltaNet → residual → RMSNorm → SwiGLU MLP → residual
on Blackhole device 0 in bf16. Loads real layer-0 weights from HF.
Compares post-layer-0 hidden state cosine vs the numpy fp32 reference
saved by experiment 91b.

GATE: cosine ≥ 0.99. Per feedback_correctness_first.md.

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91c_qwen36_27b_layer0_ttnn.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL_ID = "Qwen/Qwen3.6-27B"
REF_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_layer0_3_ref.npz")
EPS = 1e-6

# All-or-nothing HiFi4 on Blackhole per feedback_compute_kernel_config
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def load_layer0_weights(cfg):
    """Same loader as B′2, with bf16 → torch → fp32 → numpy."""
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    base = f"model.language_model.layers.0"
    needed = {
        'input_layernorm':   f"{base}.input_layernorm.weight",
        'in_proj_qkv':       f"{base}.linear_attn.in_proj_qkv.weight",
        'in_proj_z':         f"{base}.linear_attn.in_proj_z.weight",
        'in_proj_a':         f"{base}.linear_attn.in_proj_a.weight",
        'in_proj_b':         f"{base}.linear_attn.in_proj_b.weight",
        'out_proj':          f"{base}.linear_attn.out_proj.weight",
        'conv1d_weight':     f"{base}.linear_attn.conv1d.weight",
        'A_log':             f"{base}.linear_attn.A_log",
        'dt_bias':           f"{base}.linear_attn.dt_bias",
        'post_attention_layernorm': f"{base}.post_attention_layernorm.weight",
        'gate_proj':         f"{base}.mlp.gate_proj.weight",
        'up_proj':           f"{base}.mlp.up_proj.weight",
        'down_proj':         f"{base}.mlp.down_proj.weight",
    }
    by_shard = {}
    for key, tname in needed.items():
        if tname not in weight_map:
            continue
        by_shard.setdefault(weight_map[tname], []).append((key, tname))

    weights = {}
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for key, tname in items:
                t = f.get_tensor(tname).float().numpy()
                if any(k in key for k in ['proj']):
                    t = t.T   # [out, in] → [in, out] for x @ W
                weights[key] = t.copy()
    return weights


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def upload(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


# ============================================================
# DeltaNet ttnn (decode-step, layer 0 with H=0)
# ============================================================

def deltanet_step_ttnn(x_tt, w_tt, ssm_state_tt, conv_input_pre_tt, cfg, device):
    """
    Single-token DeltaNet step. Inputs already uploaded.

    x_tt:            [1, hidden]   bf16
    w_tt:            dict of weight tensors (on device, bf16)
    ssm_state_tt:    [n_v_heads, k_dim, v_dim]   fp32
    conv_input_pre_tt: [conv_dim, kernel-1]   bf16   (3 prior tokens for conv1d)
    cfg:             constants

    Returns: (out_post_residual_tt [1, hidden] bf16,
              ssm_state_tt updated [..., fp32],
              conv_state_new_tt [conv_dim, kernel-1] bf16)
    """
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KERNEL = cfg['conv_kernel']
    KEY_DIM = N_K_HEADS * K_DIM       # 2048
    VAL_DIM = N_V_HEADS * V_DIM       # 6144
    CONV_DIM = 2 * KEY_DIM + VAL_DIM  # 10240

    # 1) Pre-norm
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    # 2) Four input projections
    mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)  # [1, conv_dim]
    z_tt     = ttnn.linear(h_tt, w_tt['in_proj_z'], compute_kernel_config=hifi4)      # [1, val_dim]
    a_tt     = ttnn.linear(h_tt, w_tt['in_proj_a'], compute_kernel_config=hifi4)      # [1, n_v_heads]
    b_tt     = ttnn.linear(h_tt, w_tt['in_proj_b'], compute_kernel_config=hifi4)      # [1, n_v_heads]

    # 3) Conv1d via numpy (correctness path; trace later)
    mixed_qkv_np = ttnn.to_torch(mixed_qkv).float().numpy().flatten()[:CONV_DIM]
    conv_state_np = ttnn.to_torch(conv_input_pre_tt).float().numpy()
    # conv_state may have been padded; trim to [CONV_DIM, KERNEL-1]
    conv_state_np = conv_state_np.reshape(-1)[:CONV_DIM * (KERNEL-1)].reshape(CONV_DIM, KERNEL-1)
    conv_input = np.concatenate([conv_state_np, mixed_qkv_np[:, None]], axis=-1)   # [CONV_DIM, KERNEL]
    cw = w_tt['_conv1d_weight_np']  # already squeezed [CONV_DIM, KERNEL]
    conv_out_np = np.sum(conv_input * cw, axis=-1)
    conv_out_np = conv_out_np * (1.0 / (1.0 + np.exp(-conv_out_np)))   # silu
    conv_state_new_np = conv_input[:, 1:]   # drop oldest

    # 4) Split, reshape, GQA-repeat, L2-normalize Q + K
    q_flat = conv_out_np[:KEY_DIM]
    k_flat = conv_out_np[KEY_DIM:2*KEY_DIM]
    v_flat = conv_out_np[2*KEY_DIM:]

    q = q_flat.reshape(N_K_HEADS, K_DIM)
    k = k_flat.reshape(N_K_HEADS, K_DIM)
    v = v_flat.reshape(N_V_HEADS, V_DIM)
    q = np.repeat(q, N_V_HEADS // N_K_HEADS, axis=0)   # [N_V_HEADS, K_DIM]
    k = np.repeat(k, N_V_HEADS // N_K_HEADS, axis=0)

    def _l2(x):
        return x / (np.sqrt(np.sum(x*x, axis=-1, keepdims=True)) + EPS)
    q = _l2(q)
    k = _l2(k)

    # 5) Compute decay g and beta on host
    A_log_np = w_tt['_A_log_np']
    dt_bias_np = w_tt['_dt_bias_np']
    a_np = ttnn.to_torch(a_tt).float().numpy().flatten()[:N_V_HEADS]
    b_np = ttnn.to_torch(b_tt).float().numpy().flatten()[:N_V_HEADS]
    g = -np.exp(A_log_np) * np.log1p(np.exp(a_np + dt_bias_np))
    beta = 1.0 / (1.0 + np.exp(-b_np))

    # 6) Recurrent state update (per A3 math, all on host for now — bf16 device
    #    version comes in B′6 trace)
    H = ttnn.to_torch(ssm_state_tt).float().numpy().reshape(N_V_HEADS, K_DIM, V_DIM)
    decay = np.exp(g)[:, None, None]
    H = H * decay
    kv_mem = (H * k[:, :, None]).sum(axis=-2)
    delta = (v - kv_mem) * beta[:, None]
    H = H + k[:, :, None] * delta[:, None, :]
    out_heads = (H * q[:, :, None]).sum(axis=-2)
    out_flat = out_heads.reshape(-1)

    # 7) Output gate (silu(z) per output_gate_type "swish")
    z_np = ttnn.to_torch(z_tt).float().numpy().flatten()[:VAL_DIM]
    z_silu = z_np * (1.0 / (1.0 + np.exp(-z_np)))
    out_flat = out_flat * z_silu

    # 8) Output projection (back on device for the matmul)
    out_flat_tt = upload(out_flat, device, dtype=ttnn.bfloat16)
    out_proj_tt = ttnn.linear(out_flat_tt, w_tt['out_proj'], compute_kernel_config=hifi4)

    # 9) Residual
    x_out_tt = ttnn.add(x_tt, out_proj_tt)

    # Update state buffers (return as numpy/tt mix; B′3 only runs ONE step so
    # we just hand back the new H + conv state as numpy)
    return x_out_tt, H, conv_state_new_np


# ============================================================
# Dense MLP ttnn
# ============================================================

def mlp_step_ttnn(x_tt, w_tt):
    """SwiGLU MLP: silu(gate) * up, then down. Pre-RMSNormed. Residual added."""
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
    g_tt = ttnn.linear(h_tt, w_tt['gate_proj'], activation="silu",
                       compute_kernel_config=hifi4)
    u_tt = ttnn.linear(h_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
    inter = ttnn.mul(g_tt, u_tt)
    out = ttnn.linear(inter, w_tt['down_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out)


# ============================================================
# Driver
# ============================================================

def main():
    print("=" * 64)
    print("Phase B′3 — Qwen3.6-27B layer 0 ttnn forward + cosine gate")
    print("=" * 64)

    # Load gold reference
    print(f"\n[1/5] Loading numpy reference from {REF_PATH}")
    ref = np.load(REF_PATH)
    x_np = ref['input_x']
    gold_post_layer0 = ref['post_layer0']
    print(f"  input x: norm={np.linalg.norm(x_np):.4f}")
    print(f"  gold post-layer-0: norm={np.linalg.norm(gold_post_layer0):.4f}")

    # Config
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
    }

    print("\n[2/5] Loading layer-0 weights (bf16 → fp32 numpy)…")
    w_np = load_layer0_weights(cfg)
    print(f"  {len(w_np)} tensors loaded")

    # Open device
    print("\n[3/5] Opening Blackhole device + uploading bf16 weights…")
    device = ttnn.open_device(device_id=0)
    w_tt = {}
    for key, arr in w_np.items():
        if key == 'conv1d_weight':
            cw = arr
            if cw.ndim == 3:
                cw = cw.squeeze(1)
            w_tt['_conv1d_weight_np'] = cw  # keep host copy for conv
        elif key == 'A_log':
            w_tt['_A_log_np'] = arr
        elif key == 'dt_bias':
            w_tt['_dt_bias_np'] = arr
        else:
            w_tt[key] = upload(arr, device, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)
    print("  weights on device")

    # Upload input and zero states
    print("\n[4/5] Running layer 0 forward on device…")
    x_tt = upload(x_np.reshape(1, cfg['hidden']), device, dtype=ttnn.bfloat16)
    ssm_state_tt = upload(
        np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
        device, dtype=ttnn.float32)
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    conv_state_tt = upload(
        np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
        device, dtype=ttnn.bfloat16)

    # DeltaNet
    x_after_dn_tt, H_new, conv_state_new = deltanet_step_ttnn(
        x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg, device)
    ttnn.synchronize_device(device)

    # MLP
    x_after_mlp_tt = mlp_step_ttnn(x_after_dn_tt, w_tt)
    ttnn.synchronize_device(device)

    # Read back
    print("\n[5/5] Cosine vs numpy fp32 gold")
    ttnn_post = ttnn.to_torch(x_after_mlp_tt).float().numpy().flatten()[:cfg['hidden']]
    print(f"  ttnn post-layer-0: norm={np.linalg.norm(ttnn_post):.4f}")

    cos_v = _cosine(gold_post_layer0, ttnn_post)
    max_abs = float(np.max(np.abs(gold_post_layer0 - ttnn_post)))
    print(f"  cosine(ttnn, gold) = {cos_v:.6f}")
    print(f"  max-abs-diff       = {max_abs:.6f}")
    PASS = cos_v >= 0.99
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}  (gate: cosine ≥ 0.99)")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
