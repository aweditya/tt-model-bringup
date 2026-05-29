#!/usr/bin/env python3
"""
Experiment 91d — Phase B′4 FULLY ON-DEVICE layer 0 of Qwen3.6-27B.

Replaces the host-side recurrence/conv/norm in 91c with ttnn equivalents.
Zero numpy on the forward path except where ttnn doesn't expose the op
(weight upload only). Gates cosine ≥ 0.99 vs the same numpy fp32 reference.

Layer 0 forward, on-device:
  RMSNorm
  → 4 input projections (in_proj_qkv / _z / _a / _b)        ttnn.linear
  → conv1d kernel=4 depthwise (state + matmul + silu)        ttnn.mul/sum
  → split Q/K/V; GQA-repeat 16→48                            ttnn.repeat (or reshape+broadcast)
  → L2-normalize Q, K                                        ttnn.mul/sum/rsqrt
  → g = -exp(A_log) * softplus(a + dt_bias)                  ttnn.exp/log1p/add/mul
  → beta = sigmoid(b)                                         ttnn.sigmoid
  → recurrent H update (Phase A3 math)                       ttnn.mul/sum/reshape/add
  → out = sum(H * Q.unsqueeze(-1), dim=-2)                   ttnn.mul/sum/reshape
  → out = out * silu(z)                                       ttnn.silu/mul
  → out_proj                                                  ttnn.linear
  → residual add                                              ttnn.add
RMSNorm → SwiGLU MLP → residual                              (already on-device in 91c)

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91d_qwen36_27b_layer0_ondevice.py
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

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def load_layer0_weights():
    """Same as 91c but no reshape — caller picks fp32 vs upload."""
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    base = "model.language_model.layers.0"
    needed = {
        'input_layernorm':            f"{base}.input_layernorm.weight",
        'in_proj_qkv':                f"{base}.linear_attn.in_proj_qkv.weight",
        'in_proj_z':                  f"{base}.linear_attn.in_proj_z.weight",
        'in_proj_a':                  f"{base}.linear_attn.in_proj_a.weight",
        'in_proj_b':                  f"{base}.linear_attn.in_proj_b.weight",
        'out_proj':                   f"{base}.linear_attn.out_proj.weight",
        'conv1d_weight':              f"{base}.linear_attn.conv1d.weight",
        'A_log':                      f"{base}.linear_attn.A_log",
        'dt_bias':                    f"{base}.linear_attn.dt_bias",
        'post_attention_layernorm':   f"{base}.post_attention_layernorm.weight",
        'gate_proj':                  f"{base}.mlp.gate_proj.weight",
        'up_proj':                    f"{base}.mlp.up_proj.weight",
        'down_proj':                  f"{base}.mlp.down_proj.weight",
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
                if 'proj' in key:
                    t = t.T   # HF stores Linear as [out, in]; we want [in, out]
                weights[key] = t.copy()
    return weights


def upload(arr, device, dtype=ttnn.bfloat16):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ============================================================
# Fully on-device DeltaNet step (extends A3 with input/output projections)
# ============================================================

def deltanet_step_ondevice(x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg, device):
    """
    Single-token DeltaNet step entirely on device.

    All ttnn ops; numpy nowhere in this function.
    """
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']      # 16
    N_V_HEADS = cfg['n_v_heads']      # 48
    K_DIM = cfg['k_dim']               # 128
    V_DIM = cfg['v_dim']               # 128
    KERNEL = cfg['conv_kernel']        # 4
    KEY_DIM = N_K_HEADS * K_DIM        # 2048
    VAL_DIM = N_V_HEADS * V_DIM        # 6144
    CONV_DIM = 2 * KEY_DIM + VAL_DIM   # 10240
    N_REP = N_V_HEADS // N_K_HEADS     # 3 (GQA replicate)

    # 1) Pre-norm
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)

    # 2) Four input projections
    mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)  # [1, conv_dim]
    z_tt     = ttnn.linear(h_tt, w_tt['in_proj_z'], compute_kernel_config=hifi4)      # [1, val_dim]
    a_tt     = ttnn.linear(h_tt, w_tt['in_proj_a'], compute_kernel_config=hifi4)      # [1, n_v_heads]
    b_tt     = ttnn.linear(h_tt, w_tt['in_proj_b'], compute_kernel_config=hifi4)      # [1, n_v_heads]

    # 3) Conv1d (depthwise per channel, kernel=4) on device:
    #    state [conv_dim, kernel-1] concat new [conv_dim, 1] → [conv_dim, kernel]
    #    out  = silu(sum(state * weight, dim=-1))
    #    Reshape mixed_qkv from [1, conv_dim] to [conv_dim, 1] to concat.
    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM, 1])
    conv_input = ttnn.concat([conv_state_tt, mixed_col], dim=-1)   # [conv_dim, kernel]
    # conv_weight shape on device: [conv_dim, kernel]
    conv_prod = ttnn.mul(conv_input, w_tt['conv1d_weight'])         # [conv_dim, kernel]
    conv_out = ttnn.sum(conv_prod, dim=-1)                          # [conv_dim]
    conv_out = ttnn.silu(conv_out)                                  # [conv_dim]
    conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])  # drop oldest

    # 4) Split conv_out into Q [key_dim], K [key_dim], V [val_dim]
    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM])
    k_flat = ttnn.slice(conv_out, [KEY_DIM], [2*KEY_DIM])
    v_flat = ttnn.slice(conv_out, [2*KEY_DIM], [CONV_DIM])

    # 5) Reshape to heads; GQA-repeat Q,K from 16 to 48 heads
    q = ttnn.reshape(q_flat, [N_K_HEADS, K_DIM])      # [16, 128]
    k = ttnn.reshape(k_flat, [N_K_HEADS, K_DIM])
    v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])      # [48, 128]
    # GQA repeat: each Q/K head pairs with 3 V heads. Replicate along head dim.
    q = ttnn.repeat(q, ttnn.Shape([N_REP, 1]))        # [48, 128]
    k = ttnn.repeat(k, ttnn.Shape([N_REP, 1]))

    # 6) L2 normalize Q, K (per HF use_qk_l2norm_in_kernel=True)
    qq = ttnn.mul(q, q)
    q_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS))
    q = ttnn.mul(q, q_norm)
    kk = ttnn.mul(k, k)
    k_norm = ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS))
    k = ttnn.mul(k, k_norm)

    # 7) Compute decay g = -exp(A_log) * softplus(a + dt_bias)
    #    and beta = sigmoid(b)
    a_plus_bias = ttnn.add(a_tt, w_tt['dt_bias'])
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_plus_bias), 1.0))
    exp_A_log = ttnn.exp(w_tt['A_log'])
    g = ttnn.mul(ttnn.neg(exp_A_log), softplus_a)         # [1, n_v_heads]
    beta = ttnn.sigmoid(b_tt)                              # [1, n_v_heads]

    # 8) Recurrent H update (Phase A3 math)
    decay = ttnn.exp(g)                                    # [1, n_v_heads]
    decay_4d = ttnn.reshape(decay, [1, N_V_HEADS, 1, 1])
    # H shape [n_v_heads, k_dim, v_dim] — reshape to [1, n_v_heads, k_dim, v_dim] for broadcast
    H_4d = ttnn.reshape(ssm_state_tt, [1, N_V_HEADS, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay_4d)

    # kv_mem = sum(H * k[..., :, None], dim=-2)  →  [1, n_v_heads, v_dim]
    k_col = ttnn.reshape(k, [1, N_V_HEADS, K_DIM, 1])
    kv_mem = ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2)        # [1, n_v_heads, 1, v_dim]
    kv_mem = ttnn.reshape(kv_mem, [1, N_V_HEADS, V_DIM])

    # delta = (v - kv_mem) * beta
    v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
    diff = ttnn.sub(v_3d, kv_mem)
    beta_3d = ttnn.reshape(beta, [1, N_V_HEADS, 1])
    delta = ttnn.mul(diff, beta_3d)                              # [1, n_v_heads, v_dim]

    # H_new = H_decayed + outer(k, delta)
    delta_row = ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])
    outer = ttnn.mul(k_col, delta_row)                           # [1, n_v_heads, k_dim, v_dim]
    H_new = ttnn.add(H_decayed, outer)

    # out = sum(H_new * Q.unsqueeze(-1), dim=-2)
    q_col = ttnn.reshape(q, [1, N_V_HEADS, K_DIM, 1])
    out_heads = ttnn.sum(ttnn.mul(H_new, q_col), dim=-2)         # [1, n_v_heads, 1, v_dim]
    out_heads = ttnn.reshape(out_heads, [1, N_V_HEADS * V_DIM])   # [1, val_dim]

    # 9) Output gate: out *= silu(z)
    z_silu = ttnn.silu(z_tt)
    out_gated = ttnn.mul(out_heads, z_silu)

    # 10) Output projection + residual
    out_proj = ttnn.linear(out_gated, w_tt['out_proj'], compute_kernel_config=hifi4)
    x_out = ttnn.add(x_tt, out_proj)

    # Reshape H back to [n_v_heads, k_dim, v_dim] for state carry
    H_new_3d = ttnn.reshape(H_new, [N_V_HEADS, K_DIM, V_DIM])
    return x_out, H_new_3d, conv_state_new


def mlp_step_ondevice(x_tt, w_tt):
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
    g_tt = ttnn.linear(h_tt, w_tt['gate_proj'], activation="silu", compute_kernel_config=hifi4)
    u_tt = ttnn.linear(h_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
    inter = ttnn.mul(g_tt, u_tt)
    out = ttnn.linear(inter, w_tt['down_proj'], compute_kernel_config=hifi4)
    return ttnn.add(x_tt, out)


def main():
    print("=" * 64)
    print("Phase B′4 — Qwen3.6-27B layer 0 FULLY ON-DEVICE + cosine gate")
    print("=" * 64)

    # Gold
    print(f"\n[1/5] Loading numpy reference from {REF_PATH}")
    ref = np.load(REF_PATH)
    x_np = ref['input_x']
    gold_post_layer0 = ref['post_layer0']
    print(f"  input x norm = {np.linalg.norm(x_np):.4f}")
    print(f"  gold post-L0 norm = {np.linalg.norm(gold_post_layer0):.4f}")

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

    # Weights
    print("\n[2/5] Loading layer-0 weights from HF…")
    w_np = load_layer0_weights()
    print(f"  {len(w_np)} tensors loaded")

    # Open device
    print("\n[3/5] Opening device + uploading weights (bf16 + fp32 state)…")
    device = ttnn.open_device(device_id=0)

    # Upload weights. conv1d_weight gets squeezed from [conv_dim, 1, kernel].
    w_tt = {}
    for key, arr in w_np.items():
        if key == 'conv1d_weight':
            if arr.ndim == 3:
                arr = arr.squeeze(1)   # [conv_dim, kernel]
            w_tt[key] = upload(arr, device, dtype=ttnn.bfloat16)
        else:
            w_tt[key] = upload(arr, device, dtype=ttnn.bfloat16)
    ttnn.synchronize_device(device)

    # Input + zero states
    print("\n[4/5] Running fully on-device forward…")
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

    x_after_dn, H_new, conv_new = deltanet_step_ondevice(
        x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg, device)
    x_after_mlp = mlp_step_ondevice(x_after_dn, w_tt)
    ttnn.synchronize_device(device)

    # Compare
    print("\n[5/5] Cosine vs numpy fp32 gold")
    ttnn_out = ttnn.to_torch(x_after_mlp).float().numpy().flatten()[:cfg['hidden']]
    print(f"  ttnn post-L0 norm = {np.linalg.norm(ttnn_out):.4f}")
    cos_v = _cosine(gold_post_layer0, ttnn_out)
    max_abs = float(np.max(np.abs(gold_post_layer0 - ttnn_out)))
    print(f"  cosine = {cos_v:.6f}")
    print(f"  max-abs-diff = {max_abs:.6f}")
    PASS = cos_v >= 0.99
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
