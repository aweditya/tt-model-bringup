#!/usr/bin/env python3
"""
Experiment 91g — Phase B′7 Full Qwen3.6-27B forward across all 64 layers.

Streams weights layer-by-layer (host RAM can't hold all 27 GB at once),
runs forward through the complete [L L L F] × 16 pattern.

Gates:
  - Cosine ≥ 0.99 at layer 7 (after 2 full patterns) vs numpy fp32 reference.
    Layers 0-3 already validated by B′6. If layer 7 is also good, the pattern
    generalizes; full-model cosine drift through 64 layers is expected
    (B′8 will gate on greedy token match, the strict end-to-end signal).
  - All 64 layers must execute without crashing.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91g_qwen36_27b_full_model.py
"""
import os, sys, json, gc, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Import the reusable kernels from B′6
sys.path.insert(0, os.path.join(os.path.expanduser("~/tt-xla"), "experiments"))
# We need the *_step_ondevice helpers from 91f. Import via importlib so the
# script remains self-contained when re-deployed.
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
REF_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_layers0_7_ref.npz")
EPS = 1e-6
MAX_POS = 128
CHECKPOINT_LAYER = 7  # layer index for the cosine gate (after 2 [L L L F] patterns)


def _cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_numpy_ref_through(checkpoint, x_init, cfg):
    """fp32 numpy forward through layers 0..checkpoint (inclusive).
    Cached to disk if available."""
    if os.path.exists(REF_PATH):
        cached = dict(np.load(REF_PATH))
        if f'post_layer{checkpoint}' in cached:
            print(f"  using cached reference (has post_layer{checkpoint})")
            return cached

    # Import the np layer helpers from 91e
    _spec2 = importlib.util.spec_from_file_location(
        "_91e", os.path.expanduser("~/tt-xla/experiments/91e_qwen36_27b_layers0_3.py"))
    _91e = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(_91e)

    print(f"  computing fresh numpy ref through layer {checkpoint}…")
    HIDDEN = cfg['hidden']
    N_V_HEADS = cfg['n_v_heads']
    K_DIM = cfg['k_dim']
    V_DIM = cfg['v_dim']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = N_V_HEADS * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    ssm_state = np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32)
    conv_state = np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32)
    kv_cache = {
        'k': np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
        'v': np.zeros((cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32),
    }
    cur_pos = 0
    cos, sin = _91e.make_rope_tables_np(cur_pos,
        int(cfg['head_dim'] * cfg['partial_rotary_factor']))

    states = {'input_x': x_init.copy()}
    x = x_init.copy()
    for i in range(checkpoint + 1):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        print(f"    layer {i:2d} ({layer_type})…")
        w = _91e.load_layer_weights_all(i, layer_type)
        if layer_type == 'linear_attention':
            x, ssm_state, conv_state = _91e.deltanet_layer_np(x, w, ssm_state, conv_state, cfg)
        else:
            x, kv_cache = _91e.gated_attn_layer_np(x, w, kv_cache, cur_pos, cos, sin, cfg)
        x = _91e.mlp_layer_np(x, w)
        states[f'post_layer{i}'] = x.copy()

    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    np.savez(REF_PATH, **states)
    return states


def main():
    print("=" * 64)
    print(f"Phase B′7 — Qwen3.6-27B full 64-layer forward on device")
    print(f"  cosine gate at layer {CHECKPOINT_LAYER}; all 64 layers must run")
    print("=" * 64)

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
        'n_q_heads':   text_cfg['num_attention_heads'],
        'n_kv_heads':  text_cfg['num_key_value_heads'],
        'head_dim':    text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
    }
    NUM_LAYERS = text_cfg['num_hidden_layers']  # 64
    HIDDEN = cfg['hidden']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    # Numpy reference up to checkpoint
    print(f"\n[1/4] Numpy fp32 reference through layer {CHECKPOINT_LAYER}…")
    rng = np.random.default_rng(42)
    x_init = rng.standard_normal(HIDDEN).astype(np.float32) * 0.05
    gold = build_numpy_ref_through(CHECKPOINT_LAYER, x_init, cfg)
    gold_checkpoint = gold[f'post_layer{CHECKPOINT_LAYER}']
    print(f"  gold post-layer-{CHECKPOINT_LAYER} norm = {np.linalg.norm(gold_checkpoint):.4f}")

    # Device + initial states
    print("\n[2/4] Opening device + initial states…")
    device = ttnn.open_device(device_id=0)

    # DeltaNet states (one per layer); 48 of them since [L L L F]×16 → 48 DeltaNet
    n_deltanet = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_deltanet
    print(f"  {n_deltanet} DeltaNet states + {n_attn} attention layers")

    # Pre-allocate state buffers (cheap; small)
    ssm_states = [
        upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
               device, dtype=ttnn.float32)
        for _ in range(n_deltanet)
    ]
    conv_states = [
        upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
               device, dtype=ttnn.bfloat16)
        for _ in range(n_deltanet)
    ]

    # Single KV cache for all attention layers? No — each layer has its own.
    # 16 layers × 2 (k,v) × n_kv=4 × MAX_POS=128 × head_dim=256 × 2 bytes = 4 MB total.
    # Tiny — just keep them all on device.
    kv_caches = []
    kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                 device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                 device=device, layout=ttnn.TILE_LAYOUT)
        kv_caches.append([kv_k, kv_v])

    # cur_pos + RoPE tables for full-attn layers (all use same pos=0 at first token)
    cur_pos = 0
    cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    angles = cur_pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    cos_tt = upload(cos_np, device, dtype=ttnn.bfloat16)
    sin_tt = upload(sin_np, device, dtype=ttnn.bfloat16)

    # ── Forward through all 64 layers, streaming weights ────────
    print(f"\n[3/4] Streaming weights + forward through all {NUM_LAYERS} layers…")
    x_tt = upload(x_init.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)
    t_start = time.time()
    dn_idx = 0  # which DeltaNet state to use
    attn_idx = 0  # which KV cache to use
    cos_at_checkpoint = None

    for i in range(NUM_LAYERS):
        t_layer = time.time()
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == 'conv1d_weight' and arr.ndim == 3:
                arr = arr.squeeze(1)
            # bf8 weights for footprint — except A_log/dt_bias which are small fp32-stable
            dt = ttnn.bfloat8_b if 'proj' in k or k == 'conv1d_weight' or 'gate' in k else ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        del w_np
        gc.collect()

        if layer_type == 'linear_attention':
            x_tt, H_new, c_new = deltanet_step_ondevice(
                x_tt, w_tt, ssm_states[dn_idx], conv_states[dn_idx], cfg)
            ssm_states[dn_idx] = H_new
            conv_states[dn_idx] = c_new
            dn_idx += 1
        else:
            kv_k, kv_v = kv_caches[attn_idx]
            x_tt, kv_k, kv_v = gated_attn_step_ondevice(
                x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                cos_tt, sin_tt, cfg, device)
            kv_caches[attn_idx] = [kv_k, kv_v]
            attn_idx += 1
        x_tt = mlp_step_ondevice(x_tt, w_tt)
        ttnn.synchronize_device(device)

        # Release weight tensors (release_trace etc; rely on python gc)
        del w_tt
        gc.collect()

        dt = time.time() - t_layer
        if i == CHECKPOINT_LAYER:
            ttnn_post = ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]
            cos_at_checkpoint = _cosine(gold_checkpoint, ttnn_post)
            print(f"  ★ layer {i:2d} ({layer_type:18s}) [{dt:5.2f}s]  cosine = {cos_at_checkpoint:.6f}")
        elif i % 8 == 0 or i == NUM_LAYERS - 1:
            x_norm = float(np.linalg.norm(
                ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN]))
            print(f"    layer {i:2d} ({layer_type:18s}) [{dt:5.2f}s]  ‖x‖ = {x_norm:.3f}")

    elapsed = time.time() - t_start
    print(f"\n[4/4] VERDICT")
    print(f"  All {NUM_LAYERS} layers executed in {elapsed:.1f}s ({elapsed/NUM_LAYERS:.2f}s/layer)")
    gate = "✓" if cos_at_checkpoint and cos_at_checkpoint >= 0.99 else "✗"
    print(f"  cosine @ layer {CHECKPOINT_LAYER} = {cos_at_checkpoint:.6f}  {gate}")
    print(f"  Final hidden state ‖x‖ = {float(np.linalg.norm(ttnn.to_torch(x_tt).float().numpy().flatten()[:HIDDEN])):.3f}")
    PASS = cos_at_checkpoint and cos_at_checkpoint >= 0.99
    print(f"  VERDICT: {'PASS ✓' if PASS else 'FAIL ✗'}")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
