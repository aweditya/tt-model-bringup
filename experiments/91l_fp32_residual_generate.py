#!/usr/bin/env python3
"""
Experiment 91l — Phase B′9: Qwen3.6-27B with fp32 residual stream.

Hypothesis: bf16 residual stream accumulates rounding across 64 layers,
collapsing the lm_head's logit landscape to near-fixed-points (B'8 saw
the 'FR' fixed-point with top1-top2 margin 0.062 and junk subwords
populating top-5). Promoting the residual to fp32 should restore the
margin without changing matmul math (HiFi4 + fp32 DEST is unchanged).

Modes:
    --mode diagnose   instrumented forward (like 91j) for 5 decode steps
    --mode generate   generate N tokens with greedy argmax (default)

The bf16 baseline ran with this exact same prompt; outputs are stored
under ~/tt-xla/.cache/b8_diagnostics.json for direct comparison.

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91l_fp32_residual_generate.py --mode diagnose
"""
import os, sys, json, time, gc, argparse
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# Reuse the now-fp32-aware kernels from 91f (typecasts around SDPA were added)
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
MAX_POS = 256
PROMPT_DEFAULT = "The capital of France is"
NUM_TOKENS_DEFAULT = 60
CHECKPOINT_LAYERS = [0, 16, 32, 48, 63]

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def x_norm(x_tt):
    return float(np.linalg.norm(ttnn.to_torch(x_tt).float().numpy().flatten()))


def load_embed_lm_head_weights():
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']
    needed = {
        'embed':        "model.language_model.embed_tokens.weight",
        'final_norm':   "model.language_model.norm.weight",
        'lm_head':      "lm_head.weight",
    }
    by_shard = {}
    for key, tname in needed.items():
        if tname in weight_map:
            by_shard.setdefault(weight_map[tname], []).append((key, tname))
    weights = {}
    for shard, items in by_shard.items():
        path = hf_hub_download(MODEL_ID, shard)
        with safe_open(path, framework="pt") as f:
            for key, tname in items:
                t = f.get_tensor(tname).float().numpy()
                if key == 'lm_head':
                    t = t.T
                # B'9.5 fix: final_norm is Qwen3_5RMSNorm with (1.0 + w) formula
                if key == 'final_norm':
                    t = t + 1.0
                weights[key] = t.copy()
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', default=PROMPT_DEFAULT)
    p.add_argument('--tokens', type=int, default=NUM_TOKENS_DEFAULT)
    p.add_argument('--mode', choices=['generate', 'diagnose'], default='generate')
    p.add_argument('--diag-steps', type=int, default=5)
    args = p.parse_args()

    print("=" * 64)
    print("Phase B'9 — Qwen3.6-27B with fp32 residual stream")
    print(f"  Mode: {args.mode}   Prompt: {args.prompt!r}   Tokens: {args.tokens}")
    print("=" * 64)

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

    # Tokenize
    print("\n[1/5] Tokenizer + prompt encode…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(args.prompt)
    print(f"  {len(prompt_ids)} prompt tokens: {prompt_ids}")

    # Embed + lm_head (host-side; fp32 numpy)
    print("\n[2/5] Loading embedding + lm_head + final_norm…")
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']
    final_norm_np = eweights['final_norm']
    lm_head_np = eweights['lm_head']

    # Device + per-layer weights
    print(f"\n[3/5] Opening device + loading all {NUM_LAYERS} layers…")
    device = ttnn.open_device(device_id=0)
    # final_norm and lm_head — keep weights in their cheap formats; fp32 propagation comes from x_tt
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
            # B'9 dtype policy:
            #   - projection / matmul weights stay bf8/bf16 (most of the memory)
            #   - conv1d_weight stays bf8 (multiplies with fp32 x → fp32 propagates)
            #   - norm weights stay bf16 (rms_norm(fp32, bf16) → fp32 per probe)
            #   - SMALL scalar/bias weights (A_log, dt_bias) → fp32 to avoid mul/add demotion
            if 'proj' in k or k == 'conv1d_weight':
                dt = ttnn.bfloat8_b
            elif k in ('A_log', 'dt_bias'):
                dt = ttnn.float32        # B'9 change vs 91h
            else:
                dt = ttnn.bfloat16       # norm weights
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d} loaded ({time.time()-t_load:.1f}s elapsed)")
    print(f"  ✓ all {NUM_LAYERS} layers loaded in {time.time()-t_load:.1f}s")

    # Initial states — B'9 changes: conv_state is fp32 now (was bf16)
    n_deltanet = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_deltanet
    ssm_states = [
        upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
               device, dtype=ttnn.float32)
        for _ in range(n_deltanet)
    ]
    conv_states = [
        upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
               device, dtype=ttnn.float32)         # B'9 change
        for _ in range(n_deltanet)
    ]
    # KV cache stays bf16 (storage; single-write per slot doesn't compound drift)
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

    # C'0.6: precompute the full RoPE table once at startup; slice one row per
    # step on-device. Eliminates per-token host np.cos/np.sin + PCIe upload
    # (~100 µs per token saved). Math-identical to the prior per-position
    # recompute. Memory: 2 × MAX_POS × rotary_dim × 4 B; trivial at any
    # MAX_POS ≤ 128k. Hard prereq for C'4 trace capture (no per-step host
    # compute allowed in a trace).
    positions = np.arange(MAX_POS).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)
    sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)

    def forward_one_token(token_id, cur_pos, capture=False):
        x_np = embed_np[token_id]
        # B'9 change: upload embed as fp32 (was bf16)
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        norms = {'embed': x_norm(x_tt)} if capture else None

        cos_tt = ttnn.slice(cos_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        sin_tt = ttnn.slice(sin_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
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
            norms['x_final_dtype'] = str(x_tt.dtype)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        logits = ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]
        return logits, norms

    # Prefill (no diagnostics)
    print(f"\n[4/5] Prefill ({len(prompt_ids)} tokens)…")
    t0 = time.time()
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(tid, pos)
    prefill_time = time.time() - t0
    print(f"  prefill: {prefill_time:.1f}s ({prefill_time/len(prompt_ids)*1000:.0f} ms/tok)")

    if args.mode == 'diagnose':
        print(f"\n[5/5] Diagnostic decode — {args.diag_steps} steps:")
        print("=" * 64)
        all_ids = list(prompt_ids)
        diag_records = []
        for step in range(args.diag_steps):
            cur_pos = len(all_ids)
            last_token = all_ids[-1]
            t0 = time.time()
            logits, norms = forward_one_token(last_token, cur_pos - 1, capture=True)
            dt = time.time() - t0
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
            print(f"│ x dtype at final: {norms.get('x_final_dtype', '?')}")
            print("│ ‖x‖ checkpoints:")
            for k, v in norms.items():
                if isinstance(v, (int, float)):
                    print(f"│   {k:>14s}: {v:10.4f}")
            print(f"│ logit stats: max={rec['logit_max']:.3f} mean={rec['logit_mean']:.3f} "
                  f"std={rec['logit_std']:.3f}")
            print(f"│ top-5 (margin top1-top2 = {rec['logit_margin_top1_top2']:.3f}):")
            for tid_, s, lg in top5:
                mark = " ← chosen" if tid_ == int(next_id) else ""
                print(f"│   {tid_:6d}  {s!r:>18s}  logit={lg:8.3f}{mark}")
            print("└─")

        # Cross-step analysis
        print("\n" + "=" * 64)
        print("CROSS-STEP ANALYSIS — B'9 (fp32 residual)")
        print("=" * 64)
        keys = [k for k in diag_records[0]['norms'].keys()
                if isinstance(diag_records[0]['norms'][k], (int, float))]
        print("  step→  " + "  ".join(f"step{i}".rjust(10) for i in range(args.diag_steps)))
        for k in keys:
            row = [r['norms'][k] for r in diag_records]
            spread = (max(row) - min(row)) / (max(abs(v) for v in row) + 1e-9)
            print(f"  {k:>14s}: " +
                  "  ".join(f"{v:10.4f}" for v in row) +
                  f"  (rel spread {spread*100:.2f}%)")

        top1s = [r['top5'][0][0] for r in diag_records]
        all_same = len(set(top1s)) == 1
        print(f"\n  top-1 tokens across steps: {top1s}")
        print(f"  decoded:               {[r['top5'][0][1] for r in diag_records]}")
        print(f"  all identical?  {all_same}")
        margins = [r['logit_margin_top1_top2'] for r in diag_records]
        print(f"  margins:               {[f'{m:.3f}' for m in margins]}")

        out_path = os.path.expanduser("~/tt-xla/.cache/b9_diagnostics.json")
        with open(out_path, "w") as f:
            json.dump(diag_records, f, indent=2)
        print(f"\nDiagnostic records → {out_path}")
    else:
        print(f"\n[5/5] Generating {args.tokens} tokens (greedy)…")
        all_ids = list(prompt_ids)
        t_decode = time.time()
        for step in range(args.tokens):
            cur_pos = len(all_ids)
            last_token = all_ids[-1]
            logits, _ = forward_one_token(last_token, cur_pos - 1)
            next_id = int(np.argmax(logits))
            all_ids.append(next_id)
            if step < 3 or step % 10 == 0 or step == args.tokens - 1:
                elapsed = time.time() - t_decode
                rate = (step + 1) / elapsed
                print(f"    step {step:2d}: tok {next_id} → "
                      f"{tok.decode([next_id])!r}  "
                      f"[{1000*elapsed/(step+1):.0f} ms/tok, {rate:.2f} tok/s]")
        total = time.time() - t_decode
        print(f"\n  decode: {args.tokens} tokens in {total:.1f}s = {args.tokens/total:.2f} tok/s")
        text = tok.decode(all_ids)
        print("\n[Generated text]:")
        print("  ┌" + "─" * 60)
        for line in text.split("\n"):
            print(f"  │ {line}")
        print("  └" + "─" * 60)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
