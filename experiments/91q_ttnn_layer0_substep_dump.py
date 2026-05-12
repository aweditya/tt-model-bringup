#!/usr/bin/env python3
"""
Experiment 91q — Instrumented ttnn DeltaNet + MLP forward for layer 0.

Mirrors HF's substep capture boundaries (from utils/hf_layer0_substep_dump.py)
so we can diff per-substep and localize the remaining ~0.3% drift on layer 0.

Inlines 91f's deltanet_step_ondevice + mlp_step_ondevice with explicit
capture-after-each-ttnn-op so this file is the canonical authority on
what ttnn computes at each boundary. Don't pollute 91f's production
kernels with debug hooks.

For each prompt token, runs ttnn through embed + layer 0 with all 5 bug
fixes applied (from 91f) and saves intermediate tensors keyed by
(substep_name, position) to ~/tt-xla/.cache/ttnn_layer0_substeps.npz

Use with utils/substep_compare.py to find the divergent substep.

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
load_layer_weights_all = _91f.load_layer_weights_all
upload = _91f.upload

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
PROMPT = "The capital of France is"
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/ttnn_layer0_substeps.npz")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def to_np(t_tt):
    """Pull a ttnn tensor to numpy fp32 with a flat-but-recoverable shape."""
    return ttnn.to_torch(t_tt).float().cpu().numpy()


def deltanet_with_captures(x_tt, w_tt, ssm_state_tt, conv_state_tt, cfg, caps, prefix):
    """Inlined deltanet_step_ondevice with capture at every meaningful boundary.

    caps: dict[str, np.ndarray] — captures appended here keyed by f"{prefix}.<name>"
    """
    HIDDEN = cfg['hidden']
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KERNEL = cfg['conv_kernel']
    KEY_DIM = N_K_HEADS * K_DIM
    VAL_DIM = N_V_HEADS * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    N_REP = N_V_HEADS // N_K_HEADS

    caps[f"{prefix}.input_layernorm.in"] = to_np(x_tt)
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
    caps[f"{prefix}.input_layernorm.out"] = to_np(h_tt)

    mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)
    z_tt     = ttnn.linear(h_tt, w_tt['in_proj_z'],   compute_kernel_config=hifi4)
    a_tt     = ttnn.linear(h_tt, w_tt['in_proj_a'],   compute_kernel_config=hifi4)
    b_tt     = ttnn.linear(h_tt, w_tt['in_proj_b'],   compute_kernel_config=hifi4)
    caps[f"{prefix}.in_proj_qkv.out"] = to_np(mixed_qkv)
    caps[f"{prefix}.in_proj_z.out"] = to_np(z_tt)
    caps[f"{prefix}.in_proj_a.out"] = to_np(a_tt)
    caps[f"{prefix}.in_proj_b.out"] = to_np(b_tt)

    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM, 1])
    conv_input = ttnn.concat([conv_state_tt, mixed_col], dim=-1)
    caps[f"{prefix}.conv1d.in"] = to_np(conv_input)
    conv_prod = ttnn.mul(conv_input, w_tt['conv1d_weight'])
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
    caps[f"{prefix}.conv1d.out"] = to_np(conv_out)
    conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])

    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM])
    k_flat = ttnn.slice(conv_out, [KEY_DIM], [2*KEY_DIM])
    v_flat = ttnn.slice(conv_out, [2*KEY_DIM], [CONV_DIM])

    # GQA interleave (B'9.5 fix)
    def gqa_interleave(t_flat, n_kh, d):
        t = ttnn.reshape(t_flat, [n_kh, 1, d])
        t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t, [n_kh * N_REP, d])
    q = gqa_interleave(q_flat, N_K_HEADS, K_DIM)
    k = gqa_interleave(k_flat, N_K_HEADS, K_DIM)
    v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])
    caps[f"{prefix}.q_pre_l2"] = to_np(q)
    caps[f"{prefix}.k_pre_l2"] = to_np(k)
    caps[f"{prefix}.v"] = to_np(v)

    qq = ttnn.mul(q, q)
    q = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
    kk = ttnn.mul(k, k)
    k = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
    caps[f"{prefix}.q_post_l2"] = to_np(q)
    caps[f"{prefix}.k_post_l2"] = to_np(k)

    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, w_tt['dt_bias'])), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(w_tt['A_log'])), softplus_a)
    beta = ttnn.sigmoid(b_tt)
    caps[f"{prefix}.g"] = to_np(g)
    caps[f"{prefix}.beta"] = to_np(beta)

    decay = ttnn.reshape(ttnn.exp(g), [1, N_V_HEADS, 1, 1])
    H_4d = ttnn.reshape(ssm_state_tt, [1, N_V_HEADS, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay)
    k_col = ttnn.reshape(k, [1, N_V_HEADS, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [1, N_V_HEADS, V_DIM])
    v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, N_V_HEADS, 1]))
    H_new = ttnn.add(H_decayed,
                     ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
    q_col = ttnn.reshape(q, [1, N_V_HEADS, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM])
    caps[f"{prefix}.recurrence_out"] = to_np(out)

    # B'9.5 fix: per-head RMSNormGated
    out_per_head = ttnn.reshape(out, [N_V_HEADS, V_DIM])
    caps[f"{prefix}.norm.in"] = to_np(out_per_head)
    out_normed = ttnn.rms_norm(out_per_head, weight=w_tt['linear_attn_norm'], epsilon=EPS)
    caps[f"{prefix}.norm.out_pre_gate"] = to_np(out_normed)
    out_normed = ttnn.reshape(out_normed, [1, VAL_DIM])
    out_gated = ttnn.mul(out_normed, ttnn.silu(z_tt))
    caps[f"{prefix}.gated"] = to_np(out_gated)

    out_proj = ttnn.linear(out_gated, w_tt['out_proj'], compute_kernel_config=hifi4)
    caps[f"{prefix}.out_proj.out"] = to_np(out_proj)
    x_out = ttnn.add(x_tt, out_proj)
    caps[f"{prefix}.linear_attn.out"] = to_np(x_out)  # post-residual hidden

    H_new_3d = ttnn.reshape(H_new, [N_V_HEADS, K_DIM, V_DIM])
    return x_out, H_new_3d, conv_state_new


def mlp_with_captures(x_tt, w_tt, caps, prefix):
    """Inlined mlp_step_ondevice with capture at every boundary."""
    caps[f"{prefix}.post_attention_layernorm.in"] = to_np(x_tt)
    h_tt = ttnn.rms_norm(x_tt, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
    caps[f"{prefix}.post_attention_layernorm.out"] = to_np(h_tt)
    g_tt = ttnn.linear(h_tt, w_tt['gate_proj'], activation="silu", compute_kernel_config=hifi4)
    caps[f"{prefix}.gate_proj_silu"] = to_np(g_tt)
    u_tt = ttnn.linear(h_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
    caps[f"{prefix}.up_proj.out"] = to_np(u_tt)
    gated = ttnn.mul(g_tt, u_tt)
    caps[f"{prefix}.gate_x_up"] = to_np(gated)
    out = ttnn.linear(gated, w_tt['down_proj'], compute_kernel_config=hifi4)
    caps[f"{prefix}.down_proj.out"] = to_np(out)
    x_out = ttnn.add(x_tt, out)
    caps[f"{prefix}.layer.out"] = to_np(x_out)
    return x_out


def main():
    print("=" * 64)
    print("Experiment 91q — ttnn layer 0 substep dump")
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

    caps = {}
    print(f"\nForwarding {len(prompt_ids)} prompt tokens with capture…")
    for pos, tok_id in enumerate(prompt_ids):
        x_np = embed_full[tok_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        prefix = f"pos{pos}"
        caps[f"{prefix}.embed"] = to_np(x_tt)
        x_tt, ssm_state, conv_state = deltanet_with_captures(
            x_tt, w_tt, ssm_state, conv_state, cfg, caps, prefix)
        x_tt = mlp_with_captures(x_tt, w_tt, caps, prefix)
        ttnn.synchronize_device(device)
        print(f"  pos {pos}  ‖layer.out‖={np.linalg.norm(caps[f'{prefix}.layer.out']):.4f}")

    np.savez(OUT_PATH, **{k: v for k, v in caps.items()})
    print(f"\nSaved {len(caps)} substep tensors → {OUT_PATH}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
