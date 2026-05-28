#!/usr/bin/env python3
"""
Experiment 91j — Phase B′8 post-mortem diagnostics.

After B'8 produced a fixed-point "FRFRFR..." output, this script
distinguishes between two failure modes:

  (A) bf16 drift collapse: 64 layers of bf16 accumulation distort the
      final hidden state until argmax becomes input-independent.
      Predicts: ‖x‖ at intermediate layers DIFFERS across decode steps,
      but final-layer logits stay clustered around the same token.

  (B) State propagation bug: DeltaNet/conv/KV cache state not flowing
      across decode steps.
      Predicts: ‖x‖ at intermediate layers is IDENTICAL across decode
      steps (state never updates) → same logits → same token.

For each decode step, prints:
  - ‖x‖ at layer indices [0, 16, 32, 48, 63, final_norm]
  - top-5 logits + decoded tokens
  - chosen token

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91j_decode_diagnostics.py
"""
import os, sys, json, time, gc
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

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

_spec2 = importlib.util.spec_from_file_location(
    "_91h", os.path.expanduser("~/tt-xla/experiments/91h_qwen36_27b_generate.py"))
_91h = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_91h)
load_embed_lm_head_weights = _91h.load_embed_lm_head_weights

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256
PROMPT = "The capital of France is"
NUM_DIAG_STEPS = 5
CHECKPOINT_LAYERS = [0, 16, 32, 48, 63]  # indices to capture ‖x‖

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def x_norm(x_tt):
    """Compute ‖x‖_2 from a device tensor — small host transfer for diagnostics only."""
    a = ttnn.to_torch(x_tt).float().numpy().flatten()
    return float(np.linalg.norm(a))


def main():
    print("=" * 64)
    print("Phase B′8 — Decode Diagnostics")
    print("=" * 64)
    print(f"  Prompt: {PROMPT!r}")
    print(f"  Decode steps to instrument: {NUM_DIAG_STEPS}")
    print(f"  Checkpoint layers: {CHECKPOINT_LAYERS}")

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
    NUM_LAYERS = text_cfg['num_hidden_layers']
    HIDDEN = cfg['hidden']
    VOCAB = text_cfg['vocab_size']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    print(f"\n[1/5] Loading tokenizer + tokenizing prompt…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(PROMPT)
    print(f"  {len(prompt_ids)} prompt tokens: {prompt_ids}")

    print(f"\n[2/5] Loading embedding + lm_head…")
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']
    final_norm_np = eweights['final_norm']
    lm_head_np = eweights['lm_head']

    print(f"\n[3/5] Opening device + loading all {NUM_LAYERS} layers (bf8)…")
    device = ttnn.open_device(device_id=0)
    final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    t_load = time.time()
    layer_weights = []
    for i in range(NUM_LAYERS):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == 'conv1d_weight' and arr.ndim == 3:
                arr = arr.squeeze(1)
            dt = ttnn.bfloat8_b if 'proj' in k or k == 'conv1d_weight' or 'gate' in k else ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d} loaded ({time.time()-t_load:.1f}s elapsed)")
    print(f"  ✓ all {NUM_LAYERS} layers loaded in {time.time()-t_load:.1f}s")

    # Initial state
    n_deltanet = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_deltanet
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
    kv_caches = []
    kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_caches.append([kv_k, kv_v])

    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    def rope_tables_for_pos(pos):
        angles = pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        return (upload(cos_np, device, dtype=ttnn.bfloat16),
                upload(sin_np, device, dtype=ttnn.bfloat16))

    def forward_with_checkpoints(token_id, cur_pos, capture):
        """Returns (logits, dict of checkpoint norms).

        capture=True: record ‖x‖ at CHECKPOINT_LAYERS and final.
        capture=False: just forward (used for prefill).
        """
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)
        norms = {'embed': x_norm(x_tt)} if capture else None

        cos_tt, sin_tt = rope_tables_for_pos(cur_pos)
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32), device=device)

        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
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
            if capture and i in CHECKPOINT_LAYERS:
                norms[f'layer_{i}'] = x_norm(x_tt)

        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        if capture:
            norms['final_norm'] = x_norm(x_tt)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        logits = ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]
        return logits, norms

    print(f"\n[4/5] Prefill ({len(prompt_ids)} tokens, no diagnostics)…")
    t0 = time.time()
    for pos, tid in enumerate(prompt_ids):
        _, _ = forward_with_checkpoints(tid, pos, capture=False)
    print(f"  prefill done in {time.time()-t0:.1f}s")

    print(f"\n[5/5] Diagnostic decode — {NUM_DIAG_STEPS} steps with instrumentation:")
    print("=" * 64)
    all_ids = list(prompt_ids)
    diag_records = []
    for step in range(NUM_DIAG_STEPS):
        cur_pos = len(all_ids)
        last_token = all_ids[-1]
        t0 = time.time()
        logits, norms = forward_with_checkpoints(last_token, cur_pos - 1, capture=True)
        dt = time.time() - t0

        # Top-5 logits with decoded tokens
        top5_idx = np.argsort(logits)[::-1][:5]
        top5 = [(int(i), tok.decode([int(i)]), float(logits[i])) for i in top5_idx]
        next_id = top5_idx[0]
        all_ids.append(int(next_id))

        rec = {'step': step, 'cur_pos': cur_pos, 'input_token': last_token,
               'input_token_str': tok.decode([last_token]),
               'norms': norms, 'top5': top5,
               'logit_margin_top1_top2': top5[0][2] - top5[1][2],
               'logit_max': float(logits.max()),
               'logit_min': float(logits.min()),
               'logit_mean': float(logits.mean()),
               'logit_std': float(logits.std()),
               'dt_sec': dt}
        diag_records.append(rec)

        print(f"\n┌─ step {step} ─── cur_pos={cur_pos}  input={last_token!r} "
              f"({tok.decode([last_token])!r})  dt={dt*1000:.0f}ms")
        print(f"│ ‖x‖ checkpoints:")
        for k, v in norms.items():
            print(f"│   {k:>14s}: {v:10.4f}")
        print(f"│ logit stats: max={rec['logit_max']:.3f}  min={rec['logit_min']:.3f}  "
              f"mean={rec['logit_mean']:.3f}  std={rec['logit_std']:.3f}")
        print(f"│ top-5 (margin top1-top2 = {rec['logit_margin_top1_top2']:.3f}):")
        for tid_, s, lg in top5:
            mark = " ← chosen" if tid_ == int(next_id) else ""
            print(f"│   {tid_:6d}  {s!r:>15s}  logit={lg:8.3f}{mark}")
        print(f"└─")

    # Cross-step analysis: are the norms STATIC or DRIFTING?
    print("\n" + "=" * 64)
    print("CROSS-STEP ANALYSIS")
    print("=" * 64)
    if len(diag_records) >= 2:
        keys = list(diag_records[0]['norms'].keys())
        print(f"  step→  " + "  ".join(f"step{i}".rjust(10) for i in range(NUM_DIAG_STEPS)))
        for k in keys:
            row = [r['norms'][k] for r in diag_records]
            spread = (max(row) - min(row)) / (max(abs(v) for v in row) + 1e-9)
            print(f"  {k:>14s}: " +
                  "  ".join(f"{v:10.4f}" for v in row) +
                  f"  (rel spread {spread*100:.2f}%)")

        # Are top-1 tokens identical across all steps?
        top1s = [r['top5'][0][0] for r in diag_records]
        all_same = len(set(top1s)) == 1
        print(f"\n  top-1 tokens across steps: {top1s}")
        print(f"  all identical?  {all_same}")
        if all_same:
            margin0 = diag_records[0]['logit_margin_top1_top2']
            print(f"  → fixed-point detected. top1-top2 margin at step 0: {margin0:.3f}")
            # Classify: are the norms changing?
            norm_spreads = [
                (max(r['norms'][k] for r in diag_records) -
                 min(r['norms'][k] for r in diag_records)) /
                (max(abs(r['norms'][k]) for r in diag_records) + 1e-9)
                for k in keys
            ]
            max_spread = max(norm_spreads)
            print(f"  max ‖x‖ relative spread across checkpoints: {max_spread*100:.2f}%")
            if max_spread < 0.001:
                print(f"  → DIAGNOSIS: norms IDENTICAL → state propagation bug.")
            else:
                print(f"  → DIAGNOSIS: norms DRIFTING but argmax fixed → bf16 collapse.")

    # Persist a JSON dump for later inspection
    out_dir = os.path.expanduser("~/tt-xla/.cache")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "b8_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(diag_records, f, indent=2)
    print(f"\nDiagnostic records → {out_path}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
