#!/usr/bin/env python3
"""
Qwen3.6-27B Demo — coherent text generation on a single Blackhole P150.

Standalone demo: loads the 27B model once, runs prompts, greedy-decodes.

**FAST DEV LOOP TIP**: this script pays the full ~11 min weight load every
time. For iterative work, use the persistent inference server instead:

    # On qb1 or qb2:
    cd ~/tt-xla && .venv/bin/python -m experiments.serve.server &
    # Then from a separate shell on the same host:
    cd ~/tt-xla && .venv/bin/python -m experiments.serve.client bench_decode
    cd ~/tt-xla && .venv/bin/python -m experiments.serve.client bench_decode_paged \\
        --max-pos 8192

The persistent server amortizes weight load across many requests and supports
hot-reload of kernel changes via `reload_kernels`.

Run this script directly only for end-to-end "show me text generation":
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \\
        experiments/demo_qwen36_27b.py

    # Custom prompts:
    .venv/bin/python experiments/demo_qwen36_27b.py \\
        --prompts "The largest planet is" "In machine learning," \\
        --tokens 40

Current single-chip perf (validated 2026-05-13 on qb1, with QK rms_norm shipped):
  - Eager non-paged decode:  192.81 ms/tok (5.19 tok/s) at MAX_POS=256
  - Eager paged decode:      ~215 ms/tok (4.65 tok/s) at MAX_POS=8192+ (unlimited context)
  - Traced decode:           ~198 ms/tok (5.04 tok/s) — execute_trace alone

Per-layer cosine vs HF: ≥ 0.99973 (Branch III gate; QK rms_norm slight drift
from 0.99997 baseline within bf16 precision noise — see feedback_qk_rms_norm_shipped.md)

Multi-chip TP (qb2, 4× P150 with fabric, in development):
  - REAL measured (C'7.6.1 + C'7.7, 2026-05-13):
      * Traced TP one block (DeltaNet + MLP, real Qwen3.6 shapes): 1.21 ms
      * Layer-0 TP forward cos vs numpy gold: 0.999997 (math correct)
  - NOT YET MEASURED (do not cite as fact):
      * End-to-end multi-chip ms/tok with all 64 blocks chained + trace
      * Per-tok overhead (embedding, lm_head, sampling, KV writes)
      * Gated Attention TP latency (only MLP TP + DN TP measured)
  - Until C'7.8 (persistent multi-chip server) lands, multi-chip is
    correctness-validated but not perf-measured end-to-end.

Optimization stack landed in 91f:
  - C'1: in-place update_cache_for_token_ for KV slot writes (7.2× scatter)
  - C'2: bf16 residual stream
  - C'4 v4: trace capture for full single-chip decode step
  - V2 RoPE: rotate-only path with ROTARY_DIM-wide server slice
  - QK rms_norm: 11-op manual sequence → 2-op fused ttnn.rms_norm
  - DeltaNet 4-linear in_proj fusion (qkv|z|a|b concat)
  - ATTN-QKV fusion in gated_attn_step (q_proj|k_proj|v_proj concat)

This script uses the eager non-paged path. For long context support use
`bench_decode_paged` via the server (which exercises `gated_attn_step_ondevice_paged`).
"""
import os, sys, json, time, gc, argparse
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# Reuse 91f's production kernels (all 7 bug fixes baked in)
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

# Reuse 91l's embed/lm_head loader
_spec2 = importlib.util.spec_from_file_location(
    "_91l", os.path.expanduser("~/tt-xla/experiments/91l_fp32_residual_generate.py"))
_91l = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_91l)
load_embed_lm_head_weights = _91l.load_embed_lm_head_weights

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256
DEFAULT_PROMPTS = [
    "The capital of France is",                       # canonical sanity check
    "In a galaxy far, far away,",                     # creative continuation
    "The three laws of thermodynamics are:",          # factual list
]
DEFAULT_TOKENS = 40
SANITY_PROMPT = "The capital of France is"
SANITY_FIRST_TOKEN = " Paris"

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                   help="tokens to generate per prompt")
    p.add_argument("--skip-sanity", action="store_true",
                   help="skip the 'Paris' sanity check on the canonical prompt")
    args = p.parse_args()

    print("=" * 64)
    print("Qwen3.6-27B Demo — single Blackhole P150 chip")
    print("=" * 64)
    print(f"  prompts: {len(args.prompts)}  tokens each: {args.tokens}")

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Tokenizer, embed, lm_head
    # --------------------------------------------------------
    print(f"\n[1/4] Loading tokenizer + embedding + lm_head…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']
    final_norm_np = eweights['final_norm']
    lm_head_np = eweights['lm_head']

    # --------------------------------------------------------
    # Device + all 64 layers
    # --------------------------------------------------------
    print(f"\n[2/4] Opening device + loading all {NUM_LAYERS} layers (bf8) — ~10 min…")
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
            if 'proj' in k or k == 'conv1d_weight':
                dt = ttnn.bfloat8_b
            elif k in ('A_log', 'dt_bias'):
                dt = ttnn.float32
            else:
                dt = ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d}  ({time.time()-t_load:.0f}s elapsed)")
    print(f"  ✓ all {NUM_LAYERS} layers loaded in {time.time()-t_load:.0f}s")

    # --------------------------------------------------------
    # Reusable forward closure
    # --------------------------------------------------------
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    # C'0.6: precompute the full RoPE table at startup; slice one row per step.
    positions = np.arange(MAX_POS).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)
    sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)

    def fresh_state():
        n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
        n_attn = NUM_LAYERS - n_dn
        ssm = [upload(np.zeros((cfg['n_v_heads'], cfg['k_dim'], cfg['v_dim']), dtype=np.float32),
                      device, dtype=ttnn.float32) for _ in range(n_dn)]
        cvs = [upload(np.zeros((CONV_DIM, cfg['conv_kernel']-1), dtype=np.float32),
                      device, dtype=ttnn.float32) for _ in range(n_dn)]
        kvc = []
        kv_init = np.zeros((1, cfg['n_kv_heads'], MAX_POS, cfg['head_dim']), dtype=np.float32)
        for _ in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kvc.append([kv_k, kv_v])
        return ssm, cvs, kvc

    def forward_token(token_id, cur_pos, ssm, cvs, kvc):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(cos_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        sin_tt = ttnn.slice(sin_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = layer_weights[i]
            if layer_type == 'linear_attention':
                x_tt, ssm[dn_idx], cvs[dn_idx] = deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt, kv_k, kv_v = gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kvc[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        logits = ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]
        return logits

    def generate(prompt, n_tokens):
        ssm, cvs, kvc = fresh_state()
        prompt_ids = tok.encode(prompt)
        # Prefill
        t0 = time.time()
        for pos, tid in enumerate(prompt_ids):
            _ = forward_token(tid, pos, ssm, cvs, kvc)
        prefill_t = time.time() - t0
        # Greedy decode
        all_ids = list(prompt_ids)
        t1 = time.time()
        first_token = None
        for step in range(n_tokens):
            cur_pos = len(all_ids)
            last_token = all_ids[-1]
            logits = forward_token(last_token, cur_pos - 1, ssm, cvs, kvc)
            next_id = int(np.argmax(logits))
            all_ids.append(next_id)
            if first_token is None:
                first_token = next_id
        decode_t = time.time() - t1
        text = tok.decode(all_ids)
        return {
            'prompt': prompt,
            'first_token_id': first_token,
            'first_token_str': tok.decode([first_token]),
            'text': text,
            'prefill_sec': prefill_t,
            'decode_sec': decode_t,
            'prefill_tps': len(prompt_ids) / prefill_t,
            'decode_tps': n_tokens / decode_t,
        }

    # --------------------------------------------------------
    # Run prompts
    # --------------------------------------------------------
    print(f"\n[3/4] Running {len(args.prompts)} prompt(s) × {args.tokens} tokens each…")
    results = []
    for i, prompt in enumerate(args.prompts):
        print(f"\n--- Prompt {i+1}/{len(args.prompts)} ---")
        r = generate(prompt, args.tokens)
        results.append(r)
        print(f"  prefill: {r['prefill_sec']:.1f}s ({r['prefill_tps']:.2f} tok/s)")
        print(f"  decode:  {r['decode_sec']:.1f}s ({r['decode_tps']:.2f} tok/s, "
              f"{1000/r['decode_tps']:.0f} ms/tok)")
        print(f"  first token: {r['first_token_id']}  →  {r['first_token_str']!r}")
        print(f"\n  ┌──────────────────────────────")
        for line in r['text'].split("\n"):
            print(f"  │ {line}")
        print(f"  └──────────────────────────────")

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------
    print(f"\n[4/4] Branch III correctness sanity check…")
    paris_ok = None
    if not args.skip_sanity:
        sanity = next((r for r in results if r['prompt'] == SANITY_PROMPT), None)
        if sanity is None:
            print(f"  (canonical prompt {SANITY_PROMPT!r} not in this run; pass --prompts to include it)")
        else:
            paris_ok = (sanity['first_token_str'] == SANITY_FIRST_TOKEN)
            status = "✓ PASS" if paris_ok else "✗ FAIL"
            print(f"  {status}  Canonical Q: '{SANITY_PROMPT}' → first token "
                  f"{sanity['first_token_str']!r} (expected {SANITY_FIRST_TOKEN!r})")
            if not paris_ok:
                print(f"  ⚠ Branch III correctness regression — check git log and "
                      f"compare against research/branchIII_complete.md")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print(f"\n" + "=" * 64)
    print(f"Performance summary across {len(results)} prompt(s):")
    print(f"  avg prefill: {np.mean([r['prefill_tps'] for r in results]):.2f} tok/s")
    print(f"  avg decode:  {np.mean([r['decode_tps'] for r in results]):.2f} tok/s "
          f"({1000 / np.mean([r['decode_tps'] for r in results]):.0f} ms/tok)")
    print("=" * 64)

    ttnn.close_device(device)
    sys.exit(0 if paris_ok is not False else 1)


if __name__ == "__main__":
    main()
