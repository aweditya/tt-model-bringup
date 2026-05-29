#!/usr/bin/env python3
"""
Experiment 91h — Phase B′8: Qwen3.6-27B greedy text generation on Blackhole.

Wraps the validated B′7 full-model forward with:
  - tokenizer (Qwen3.6-27B's HF tokenizer)
  - input embedding lookup (host-side numpy, ~1 row × 5120 = tiny upload)
  - final RMSNorm + lm_head → logits
  - greedy argmax sampling
  - decode loop for N tokens

Memory:
  - Load all 64 layers of weights ONCE (~26.9 GB bf8) at startup
  - Keep weights on device throughout generation
  - DeltaNet state H + conv state per layer (already on device)
  - KV cache per attn layer (already on device)
  - lm_head on device

Gates:
  - First generated token is meaningful (greedy argmax of logits points to
    a real vocab token, not nonsense)
  - Generate ≥ 60 tokens (per feedback_generation_limits.md)
  - Text is coherent (subjective; we eyeball the output)

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91h_qwen36_27b_generate.py
"""
import os, sys, json, time, gc
import numpy as np
sys.path.insert(0, os.path.expanduser("~"))

import torch
import ttnn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# Reuse the validated kernels from B′6
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
MAX_POS = 256                  # KV cache size; 256 keeps memory tight
PROMPT_DEFAULT = "The capital of France is"
NUM_TOKENS_DEFAULT = 60

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def load_embed_lm_head_weights():
    """Load input embedding + lm_head + final norm. Returns dicts of numpy arrays."""
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
                    t = t.T  # [vocab, hidden] → [hidden, vocab] for x @ W
                weights[key] = t.copy()
    return weights


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', default=PROMPT_DEFAULT)
    p.add_argument('--tokens', type=int, default=NUM_TOKENS_DEFAULT)
    args = p.parse_args()

    print("=" * 64)
    print(f"Phase B′8 — Qwen3.6-27B greedy generation on Blackhole")
    print("=" * 64)
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Tokens: {args.tokens}")

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
    NUM_LAYERS = text_cfg['num_hidden_layers']
    HIDDEN = cfg['hidden']
    VOCAB = text_cfg['vocab_size']
    KEY_DIM = cfg['n_k_heads'] * cfg['k_dim']
    VAL_DIM = cfg['n_v_heads'] * cfg['v_dim']
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    # Tokenizer
    print(f"\n[1/5] Loading tokenizer + tokenizing prompt…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tok.encode(args.prompt)
    print(f"  {len(prompt_ids)} prompt tokens: {prompt_ids}")
    print(f"  decoded: {tok.decode(prompt_ids)!r}")

    # Load embedding + lm_head (kept on host; embed lookup is host-side)
    print(f"\n[2/5] Loading embedding + lm_head…")
    eweights = load_embed_lm_head_weights()
    embed_np = eweights['embed']                     # [vocab, hidden]
    final_norm_np = eweights['final_norm']            # [hidden]
    lm_head_np = eweights['lm_head']                  # [hidden, vocab]
    print(f"  embed: {embed_np.shape}, final_norm: {final_norm_np.shape}, "
          f"lm_head: {lm_head_np.shape}")

    # Device + initial states
    print(f"\n[3/5] Opening device + loading all {NUM_LAYERS} layers (bf8)…")
    device = ttnn.open_device(device_id=0)
    final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    t_load = time.time()
    layer_weights = []     # list of (layer_type, w_tt dict)
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
        if i % 8 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d} loaded ({time.time()-t_load:.1f}s elapsed)")
    print(f"  ✓ all {NUM_LAYERS} layers loaded in {time.time()-t_load:.1f}s")

    # Initial states
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

    # RoPE table cache (cosine/sine at each position)
    rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

    def rope_tables_for_pos(pos):
        angles = pos * freqs
        cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
        sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
        return (upload(cos_np, device, dtype=ttnn.bfloat16),
                upload(sin_np, device, dtype=ttnn.bfloat16))

    # ── Helper: one decode step (single token) ─────────────────
    def forward_one_token(token_id, cur_pos):
        """Embed token, forward through all 64 layers, return logits [vocab]."""
        # Host-side embed lookup (one row of embed table)
        x_np = embed_np[token_id]                       # [hidden]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.bfloat16)

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

        # Final RMSNorm + lm_head
        x_tt = ttnn.rms_norm(x_tt, weight=final_norm_tt, epsilon=EPS)
        logits_tt = ttnn.linear(x_tt, lm_head_tt, compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)

        # Read logits back, argmax on host (sampling is trivial)
        logits = ttnn.to_torch(logits_tt).float().numpy().flatten()[:VOCAB]
        return logits

    # ── Decode loop ─────────────────────────────────────────────
    print(f"\n[4/5] Generating {args.tokens} tokens (greedy)…")
    all_ids = list(prompt_ids)

    # Prefill: feed prompt tokens one at a time (no parallel prefill yet)
    t0 = time.time()
    for pos, tid in enumerate(prompt_ids):
        _ = forward_one_token(tid, pos)
    prefill_time = time.time() - t0
    print(f"  prefill ({len(prompt_ids)} tokens): {prefill_time:.1f}s "
          f"({prefill_time/len(prompt_ids)*1000:.0f} ms/tok)")

    # Decode: generate one token at a time
    t_decode = time.time()
    for step in range(args.tokens):
        cur_pos = len(all_ids)
        last_token = all_ids[-1]
        logits = forward_one_token(last_token, cur_pos - 1)  # feed prev token, predict next
        next_id = int(np.argmax(logits))
        all_ids.append(next_id)
        if step < 3 or step % 10 == 0:
            piece = tok.decode([next_id])
            elapsed = time.time() - t_decode
            print(f"    step {step:2d}: tok {next_id} → {piece!r}  "
                  f"[{elapsed/(step+1)*1000:.0f} ms/tok]")
        if next_id == tok.eos_token_id:
            print(f"    [EOS at step {step}]")
            break

    decode_time = time.time() - t_decode
    n_generated = len(all_ids) - len(prompt_ids)
    print(f"\n  decode: {n_generated} tokens in {decode_time:.1f}s "
          f"= {n_generated/decode_time:.2f} tok/s")

    print(f"\n[5/5] Generated text:")
    print(f"  ┌{'─'*60}")
    text = tok.decode(all_ids)
    for line in text.split('\n'):
        print(f"  │ {line}")
    print(f"  └{'─'*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
