#!/usr/bin/env python3
"""
Experiment 91s — Capture EVERY intermediate of layer 2's ttnn forward.

Plan A.5 — substep dump for the bad layer.

Layer 2 collapses to cosine 0.508 at pos 2 (per 91r). Audit of HF
recurrence math showed our DeltaNet equations are identical to HF's.
Therefore the bug is either:
  (a) an intermediate value differs from HF before reaching the recurrence
  (b) a specific ttnn op produces wrong output at some shape/dtype combo

This script captures every meaningful intermediate value in layer 2's
forward (DeltaNet substeps + MLP substeps) and saves them per-position.
Compared against HF's substep dump (~/tt-xla/.cache/hf_layer2_substeps.npz),
the first one that diverges from HF (cosine < 0.99) localizes the bug.

CRITICAL design choice: defer ALL host reads (`to_np`) to the END of
the forward. Holding ttnn tensor references across many ops can trigger
fresh JIT kernel variants and was the source of 91q v1 crashes. With
the LLK header patch in place we *might* be safe to read mid-forward,
but deferring is strictly safer.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91s_layer2_full_substep_dump.py
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
HF_HIDDEN_PATH = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/ttnn_layer2_substeps_full.npz")
LAYER_IDX = 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def main():
    print("=" * 64)
    print(f"Experiment 91s — full substep dump for layer {LAYER_IDX}")
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
    N_K_HEADS = cfg['n_k_heads']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KERNEL = cfg['conv_kernel']
    KEY_DIM = N_K_HEADS * K_DIM
    VAL_DIM = N_V_HEADS * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    N_REP = N_V_HEADS // N_K_HEADS

    # ----- HF reference inputs -----
    if not os.path.exists(HF_HIDDEN_PATH):
        print(f"ERROR: missing {HF_HIDDEN_PATH}")
        sys.exit(1)
    hf = np.load(HF_HIDDEN_PATH)
    hidden_in = hf[f"hidden_{LAYER_IDX}"][0]   # [seq, HIDDEN] — input to layer N
    print(f"input shape: {hidden_in.shape}  per-pos ‖·‖: "
          f"{[f'{np.linalg.norm(hidden_in[p]):.3f}' for p in range(hidden_in.shape[0])]}")

    # ----- Device + layer 2 weights -----
    print(f"\nOpening device + loading layer {LAYER_IDX} weights…")
    t0 = time.time()
    device = ttnn.open_device(device_id=0)
    w_np = load_layer_weights_all(LAYER_IDX, 'linear_attention')
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

    # ----- Fresh state -----
    ssm_state = upload(np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32),
                       device, dtype=ttnn.float32)
    conv_state = upload(np.zeros((CONV_DIM, KERNEL - 1), dtype=np.float32),
                        device, dtype=ttnn.float32)

    # All captures stored as ttnn tensors first (refs held), to_np at end.
    captures = {}    # name (with pos prefix) → ttnn tensor (keep ref alive)
    seq = hidden_in.shape[0]

    print(f"\nForwarding {seq} positions through layer {LAYER_IDX} with full capture…")
    for pos in range(seq):
        p = f"pos{pos}"
        # Upload input as fp32 (matches 91l's B'9 residual-stream policy)
        x_tt = upload(hidden_in[pos].reshape(1, HIDDEN), device, dtype=ttnn.float32)
        captures[f"{p}.x_in"] = x_tt

        # === DeltaNet step (inlined from 91f, with captures) ===
        h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
        captures[f"{p}.h_after_input_norm"] = h_tt

        mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)
        z_tt     = ttnn.linear(h_tt, w_tt['in_proj_z'],   compute_kernel_config=hifi4)
        a_tt     = ttnn.linear(h_tt, w_tt['in_proj_a'],   compute_kernel_config=hifi4)
        b_tt     = ttnn.linear(h_tt, w_tt['in_proj_b'],   compute_kernel_config=hifi4)
        captures[f"{p}.in_proj_qkv"] = mixed_qkv
        captures[f"{p}.in_proj_z"]   = z_tt
        captures[f"{p}.in_proj_a"]   = a_tt
        captures[f"{p}.in_proj_b"]   = b_tt

        mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM, 1])
        conv_input = ttnn.concat([conv_state, mixed_col], dim=-1)
        captures[f"{p}.conv_input"] = conv_input
        conv_prod = ttnn.mul(conv_input, w_tt['conv1d_weight'])
        conv_pre_silu = ttnn.sum(conv_prod, dim=-1)
        captures[f"{p}.conv_pre_silu"] = conv_pre_silu
        conv_out = ttnn.silu(conv_pre_silu)
        captures[f"{p}.conv_out"] = conv_out
        conv_state = ttnn.slice(conv_input, [0, 1], [CONV_DIM, KERNEL])

        q_flat = ttnn.slice(conv_out, [0], [KEY_DIM])
        k_flat = ttnn.slice(conv_out, [KEY_DIM], [2 * KEY_DIM])
        v_flat = ttnn.slice(conv_out, [2 * KEY_DIM], [CONV_DIM])
        captures[f"{p}.q_flat"] = q_flat
        captures[f"{p}.k_flat"] = k_flat
        captures[f"{p}.v_flat"] = v_flat

        # GQA interleave (B'9.5 fix)
        def gqa_interleave(t_flat, n_kh, d):
            t = ttnn.reshape(t_flat, [n_kh, 1, d])
            t = ttnn.repeat(t, ttnn.Shape([1, N_REP, 1]))
            return ttnn.reshape(t, [n_kh * N_REP, d])
        q = gqa_interleave(q_flat, N_K_HEADS, K_DIM)
        k = gqa_interleave(k_flat, N_K_HEADS, K_DIM)
        v = ttnn.reshape(v_flat, [N_V_HEADS, V_DIM])
        captures[f"{p}.q_pre_l2"] = q
        captures[f"{p}.k_pre_l2"] = k
        captures[f"{p}.v"] = v

        qq = ttnn.mul(q, q)
        q = ttnn.mul(q, ttnn.rsqrt(ttnn.add(ttnn.sum(qq, dim=-1, keepdim=True), EPS)))
        kk = ttnn.mul(k, k)
        k = ttnn.mul(k, ttnn.rsqrt(ttnn.add(ttnn.sum(kk, dim=-1, keepdim=True), EPS)))
        captures[f"{p}.q_post_l2"] = q
        captures[f"{p}.k_post_l2"] = k

        softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, w_tt['dt_bias'])), 1.0))
        g = ttnn.mul(ttnn.neg(ttnn.exp(w_tt['A_log'])), softplus_a)
        beta = ttnn.sigmoid(b_tt)
        captures[f"{p}.softplus_a"] = softplus_a
        captures[f"{p}.g"] = g
        captures[f"{p}.beta"] = beta

        decay = ttnn.reshape(ttnn.exp(g), [1, N_V_HEADS, 1, 1])
        captures[f"{p}.decay"] = decay
        H_4d = ttnn.reshape(ssm_state, [1, N_V_HEADS, K_DIM, V_DIM])
        H_decayed = ttnn.mul(H_4d, decay)
        captures[f"{p}.H_decayed"] = H_decayed
        k_col = ttnn.reshape(k, [1, N_V_HEADS, K_DIM, 1])
        kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                              [1, N_V_HEADS, V_DIM])
        captures[f"{p}.kv_mem"] = kv_mem
        v_3d = ttnn.reshape(v, [1, N_V_HEADS, V_DIM])
        delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, N_V_HEADS, 1]))
        captures[f"{p}.delta"] = delta
        H_new = ttnn.add(H_decayed,
                         ttnn.mul(k_col, ttnn.reshape(delta, [1, N_V_HEADS, 1, V_DIM])))
        captures[f"{p}.H_new"] = H_new
        q_col = ttnn.reshape(q, [1, N_V_HEADS, K_DIM, 1])
        out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM])
        captures[f"{p}.recurrence_out"] = out

        # Per-head RMSNormGated
        out_per_head = ttnn.reshape(out, [N_V_HEADS, V_DIM])
        captures[f"{p}.norm_in"] = out_per_head
        out_normed = ttnn.rms_norm(out_per_head, weight=w_tt['linear_attn_norm'], epsilon=EPS)
        captures[f"{p}.norm_out_pre_gate"] = out_normed
        out_normed = ttnn.reshape(out_normed, [1, VAL_DIM])
        silu_z = ttnn.silu(z_tt)
        captures[f"{p}.silu_z"] = silu_z
        out_gated = ttnn.mul(out_normed, silu_z)
        captures[f"{p}.gated"] = out_gated

        out_proj = ttnn.linear(out_gated, w_tt['out_proj'], compute_kernel_config=hifi4)
        captures[f"{p}.out_proj"] = out_proj
        x_after_deltanet = ttnn.add(x_tt, out_proj)
        captures[f"{p}.post_deltanet"] = x_after_deltanet

        # Update SSM state for next position
        ssm_state = ttnn.reshape(H_new, [N_V_HEADS, K_DIM, V_DIM])

        # === MLP step ===
        h2_tt = ttnn.rms_norm(x_after_deltanet, weight=w_tt['post_attention_layernorm'], epsilon=EPS)
        captures[f"{p}.post_attn_norm"] = h2_tt
        g_tt = ttnn.linear(h2_tt, w_tt['gate_proj'], activation="silu",
                           compute_kernel_config=hifi4)
        u_tt = ttnn.linear(h2_tt, w_tt['up_proj'], compute_kernel_config=hifi4)
        captures[f"{p}.mlp_gate_silu"] = g_tt
        captures[f"{p}.mlp_up"] = u_tt
        gated = ttnn.mul(g_tt, u_tt)
        captures[f"{p}.mlp_gate_x_up"] = gated
        mlp_out = ttnn.linear(gated, w_tt['down_proj'], compute_kernel_config=hifi4)
        captures[f"{p}.mlp_down"] = mlp_out
        x_after_mlp = ttnn.add(x_after_deltanet, mlp_out)
        captures[f"{p}.post_mlp"] = x_after_mlp

        print(f"  pos {pos} forward + capture done")

    ttnn.synchronize_device(device)

    # ----- to_np everything, save -----
    print(f"\nReading {len(captures)} captured tensors to host…")
    np_caps = {}
    for name, t in captures.items():
        try:
            np_caps[name] = ttnn.to_torch(t).float().cpu().numpy()
        except Exception as e:
            print(f"  WARN: could not read {name}: {e}")

    np.savez(OUT_PATH, **np_caps)
    print(f"\nSaved {len(np_caps)} tensors → {OUT_PATH}")
    print(f"Total elapsed: {time.time()-t_total:.1f}s")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
