#!/usr/bin/env python3
"""
Experiment 91i — Phase B′8 pre-flight: shape contract validation in ~30s.

Catches shape bugs in the on-device decode path WITHOUT paying the 10-minute
27B weight upload cost. Builds random-init weights for ONE layer of each type
(DeltaNet, Gated Attention, MLP, final norm, lm_head), allocates KV cache +
DeltaNet state at the production MAX_POS, and runs ONE full decode step.

Inputs:
  --max-pos N   (default 256)  — KV cache size to validate
  --layer-of-each            — only test one DeltaNet + one GatedAttn (default)

What it does NOT validate:
  - Numerical correctness (we use random weights)
  - 64-layer stack behavior (we test one of each type)
  - Cumulative bf16 drift

What it DOES validate:
  - All ttnn calls compile + execute at production shapes
  - KV cache reshape contract (the bug B′8 hit)
  - DeltaNet state H + conv_state shapes round-trip cleanly
  - logits has expected vocab size

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91i_shape_preflight.py
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download

# Reuse production kernels
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_91f", os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py"))
_91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_91f)
deltanet_step_ondevice = _91f.deltanet_step_ondevice
gated_attn_step_ondevice = _91f.gated_attn_step_ondevice
mlp_step_ondevice = _91f.mlp_step_ondevice
upload = _91f.upload

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def rand(shape, scale=0.02):
    """Small-magnitude random init — avoids exp() overflow in DeltaNet."""
    return (np.random.randn(*shape) * scale).astype(np.float32)


def build_deltanet_weights(cfg, device):
    """Random-init weights for one DeltaNet layer at production shapes."""
    H = cfg['hidden']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    INTER = cfg['intermediate']
    N_V = cfg['n_v_heads']
    KERNEL = cfg['conv_kernel']

    w = {
        'input_layernorm': np.ones(H, dtype=np.float32),
        'post_attention_layernorm': np.ones(H, dtype=np.float32),
        # MLP (loader transposes proj → [in, out])
        'gate_proj': rand((H, INTER)),
        'up_proj':   rand((H, INTER)),
        'down_proj': rand((INTER, H)),
        # DeltaNet projections (loader transposes proj → [in, out])
        'in_proj_qkv': rand((H, CONV_DIM)),
        'in_proj_z':   rand((H, VAL_DIM)),
        'in_proj_a':   rand((H, N_V)),
        'in_proj_b':   rand((H, N_V)),
        'out_proj':    rand((VAL_DIM, H)),
        # Non-proj weights
        'conv1d_weight': rand((CONV_DIM, KERNEL)),
        'A_log':         rand((N_V,), scale=0.1),   # small so exp() doesn't blow up
        'dt_bias':       rand((N_V,), scale=0.01),
    }
    w_tt = {}
    for k, arr in w.items():
        dt = ttnn.bfloat8_b if ('proj' in k or k == 'conv1d_weight') else ttnn.bfloat16
        w_tt[k] = upload(arr, device, dtype=dt)
    return w_tt


def build_gated_attn_weights(cfg, device):
    """Random-init weights for one Gated Attention layer at production shapes."""
    H = cfg['hidden']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    HD = cfg['head_dim']
    INTER = cfg['intermediate']

    w = {
        'input_layernorm': np.ones(H, dtype=np.float32),
        'post_attention_layernorm': np.ones(H, dtype=np.float32),
        'gate_proj': rand((H, INTER)),
        'up_proj':   rand((H, INTER)),
        'down_proj': rand((INTER, H)),
        # Q is packed with output gate → 2*N_Q*HD output
        'q_proj': rand((H, N_Q * HD * 2)),
        'k_proj': rand((H, N_KV * HD)),
        'v_proj': rand((H, N_KV * HD)),
        'o_proj': rand((N_Q * HD, H)),
    }
    w_tt = {}
    for k, arr in w.items():
        dt = ttnn.bfloat8_b if 'proj' in k else ttnn.bfloat16
        w_tt[k] = upload(arr, device, dtype=dt)
    return w_tt


def rope_tables_for_pos(pos, cfg, device):
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half).astype(np.float32) / half))
    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    return (upload(cos_np, device, dtype=ttnn.bfloat16),
            upload(sin_np, device, dtype=ttnn.bfloat16))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-pos", type=int, default=256)
    args = p.parse_args()

    np.random.seed(0)

    print("=" * 64)
    print(f"Phase B′8 pre-flight — shape contract validation (MAX_POS={args.max_pos})")
    print("=" * 64)
    t_start = time.time()

    # 1) Config
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    cfg = {
        'hidden':      text_cfg['hidden_size'],
        'intermediate': text_cfg['intermediate_size'],
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
    VOCAB = text_cfg['vocab_size']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    print(f"  hidden={HIDDEN}  inter={cfg['intermediate']}  vocab={VOCAB}")
    print(f"  n_q={cfg['n_q_heads']}  n_kv={cfg['n_kv_heads']}  head_dim={cfg['head_dim']}")
    print(f"  KEY_DIM={KEY_DIM}  VAL_DIM={VAL_DIM}  CONV_DIM={CONV_DIM}")

    # 2) Device
    print(f"\n[1/5] Opening device…")
    device = ttnn.open_device(device_id=0)

    # 3) Random-init weights for ONE DeltaNet + ONE GatedAttn + final
    print(f"[2/5] Building random-init weights for one DeltaNet + one GatedAttn…")
    t = time.time()
    w_dn = build_deltanet_weights(cfg, device)
    w_ga = build_gated_attn_weights(cfg, device)
    final_norm_tt = upload(np.ones(HIDDEN, dtype=np.float32), device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(rand((HIDDEN, VOCAB)), device, dtype=ttnn.bfloat8_b)
    print(f"  weights built in {time.time()-t:.1f}s")

    # 4) State
    print(f"[3/5] Allocating KV cache + DeltaNet state at MAX_POS={args.max_pos}…")
    kv_init = np.zeros((1, cfg['n_kv_heads'], args.max_pos, cfg['head_dim']), dtype=np.float32)
    kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    ssm_state = upload(
        np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
        device, dtype=ttnn.float32)
    conv_state = upload(
        np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
        device, dtype=ttnn.bfloat16)

    # 5) One forward step: DeltaNet → MLP → GatedAttn → MLP → final → lm_head
    print(f"[4/5] Running one decode step through each layer type…")
    t = time.time()
    cur_pos = 0
    cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
    cos_tt, sin_tt = rope_tables_for_pos(cur_pos, cfg, device)

    x_tt = upload(rand((1, HIDDEN), scale=0.1), device, dtype=ttnn.bfloat16)

    # DeltaNet layer
    print("  → deltanet_step…")
    x_tt, ssm_state, conv_state = deltanet_step_ondevice(
        x_tt, w_dn, ssm_state, conv_state, cfg)
    x_tt = mlp_step_ondevice(x_tt, w_dn)

    # Gated Attn layer
    print("  → gated_attn_step…")
    x_tt, kv_k, kv_v = gated_attn_step_ondevice(
        x_tt, w_ga, kv_k, kv_v, None, cur_pos_tt, cur_pos,
        cos_tt, sin_tt, cfg, device)
    x_tt = mlp_step_ondevice(x_tt, w_ga)

    print("  → final_norm + lm_head…")
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
    logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)
    print(f"  forward completed in {time.time()-t:.2f}s")

    # 6) Validate output shape
    print(f"[5/5] Validating output shape…")
    logits_np = ttnn.to_torch(logits_tt).float().numpy().flatten()
    assert logits_np.size >= VOCAB, f"logits size {logits_np.size} < vocab {VOCAB}"
    print(f"  ✓ logits has ≥ {VOCAB} elements ({logits_np.size} actual)")

    ttnn.close_device(device)

    elapsed = time.time() - t_start
    print()
    print("=" * 64)
    print(f"PRE-FLIGHT PASS — all shapes valid for MAX_POS={args.max_pos}")
    print(f"Total time: {elapsed:.1f}s   (vs ~700s for full 27B weight load)")
    print("=" * 64)


if __name__ == "__main__":
    main()
