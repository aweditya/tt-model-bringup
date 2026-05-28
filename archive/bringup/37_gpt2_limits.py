"""
Experiment 37: Pushing GPT-2 limits on Blackhole.

Four tests:
  1. Trace bucket at 512 — does it fit? Latency? Can we hold 5 buckets (32/64/128/256/512)?
  2. KV-cached decode at long sequences — 100/200/500 tokens, verify constant latency.
  3. Combined: traced prefill (fast) + KV-cached decode (unlimited).
  4. Peak throughput: fastest way to generate 100 tokens.

Summary table at the end.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch
import traceback

# ── Load GPT-2 ──────────────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 small...")
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    config = json.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

id_to_token = {v: k for k, v in vocab.items()}
n_heads = config['n_head']       # 12
d_model = config['n_embd']       # 768
head_dim = d_model // n_heads    # 64
n_layers = config['n_layer']     # 12
max_seq = 1024

wte = weights["wte.weight"]      # (50257, 768)
wpe = weights["wpe.weight"]      # (1024, 768)

def encode_simple(text):
    tokens, i = [], 0
    text_bytes = text.encode('utf-8')
    while i < len(text_bytes):
        best_len = 0
        for length in range(min(20, len(text_bytes) - i), 0, -1):
            candidate = text_bytes[i:i+length].decode('utf-8', errors='ignore')
            if i > 0 and text_bytes[i] == ord(' '):
                cand = '\u0120' + candidate[1:] if len(candidate) > 1 else '\u0120'
                if cand in vocab:
                    tokens.append(vocab[cand]); best_len = length; break
            if candidate in vocab:
                tokens.append(vocab[candidate]); best_len = length; break
        if best_len == 0:
            tokens.append(vocab.get(chr(text_bytes[i]), 0)); best_len = 1
        i += best_len
    return tokens

def decode_tokens(ids):
    return ''.join(id_to_token.get(int(i), '?').replace('\u0120', ' ') for i in ids)

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}")

# ── Device ───────────────────────────────────────────────────
import ttnn

device = ttnn.open_device(device_id=0)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── Upload weights (pre-split QKV) ──────────────────────────
print("Uploading weights (pre-split QKV)...")
t0 = time.perf_counter()

layer_w = []
for i in range(n_layers):
    p = f"h.{i}"
    w_attn = weights[f"{p}.attn.c_attn.weight"]
    b_attn = weights[f"{p}.attn.c_attn.bias"]
    layer_w.append({
        'ln1_g': to_dev(weights[f"{p}.ln_1.weight"]),
        'ln1_b': to_dev(weights[f"{p}.ln_1.bias"]),
        'w_q': to_dev(w_attn[:, :d_model]),
        'w_k': to_dev(w_attn[:, d_model:2*d_model]),
        'w_v': to_dev(w_attn[:, 2*d_model:]),
        'b_q': to_dev(b_attn[:d_model]),
        'b_k': to_dev(b_attn[d_model:2*d_model]),
        'b_v': to_dev(b_attn[2*d_model:]),
        'w_proj': to_dev(weights[f"{p}.attn.c_proj.weight"]),
        'b_proj': to_dev(weights[f"{p}.attn.c_proj.bias"]),
        'ln2_g': to_dev(weights[f"{p}.ln_2.weight"]),
        'ln2_b': to_dev(weights[f"{p}.ln_2.bias"]),
        'w_fc': to_dev(weights[f"{p}.mlp.c_fc.weight"]),
        'b_fc': to_dev(weights[f"{p}.mlp.c_fc.bias"]),
        'w_mlp': to_dev(weights[f"{p}.mlp.c_proj.weight"]),
        'b_mlp': to_dev(weights[f"{p}.mlp.c_proj.bias"]),
    })
ln_f_g = to_dev(weights["ln_f.weight"])
ln_f_b = to_dev(weights["ln_f.bias"])
print(f"  Weights uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")


# ══════════════════════════════════════════════════════════════
# Shared model functions
# ══════════════════════════════════════════════════════════════

def gpt2_layer(x, w, seq_len):
    """One transformer layer — all ops on device, zero CPU round-trips."""
    h = ttnn.layer_norm(x, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])
    q = ttnn.transpose(ttnn.reshape(q, [1, seq_len, n_heads, head_dim]), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, [1, seq_len, n_heads, head_dim]), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, [1, seq_len, n_heads, head_dim]), 1, 2)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    merged = ttnn.transformer.concatenate_heads(attn)
    x = ttnn.add(x, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))
    h2 = ttnn.layer_norm(x, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))

def forward_body(x, pad_len):
    """12-layer transformer + final LN. Fully on-device, traceable."""
    for i in range(n_layers):
        x = gpt2_layer(x, layer_w[i], pad_len)
    return ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)

def traced_forward(token_ids, pad_len, trace_info):
    """Run traced forward: write embeddings into input buffer, replay trace, read logits."""
    tid, x_in, x_out = trace_info
    seq_len = len(token_ids)
    ids = list(token_ids) + [50256] * (pad_len - seq_len)
    emb = (wte[ids] + wpe[:pad_len])[None, :, :]
    emb_t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
    while emb_t.dim() < 2:
        emb_t = emb_t.unsqueeze(0)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(emb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        x_in
    )
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    out = from_dev(x_out, (1, pad_len, d_model))
    return out[0, seq_len - 1, :] @ wte.T


# ══════════════════════════════════════════════════════════════
# TEST 1: Trace bucket at 512 + simultaneous 5-bucket test
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Trace bucket scaling (32, 64, 128, 256, 512)")
print("=" * 70)

PAD_LENS = [32, 64, 128, 256, 512]
trace_results = {}

# Part A: Individual trace per bucket (capture + release)
print("\n--- Part A: Individual trace capture (one at a time) ---")
for pad_len in PAD_LENS:
    print(f"\n  pad_len = {pad_len}:")
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)
    result = {'pad_len': pad_len, 'success': False}

    try:
        # Warmup
        x_warm = to_dev(dummy)
        x_warm = forward_body(x_warm, pad_len)
        _ = from_dev(x_warm, (1, pad_len, d_model))

        # Capture
        t_cap_start = time.perf_counter()
        x_in = to_dev(dummy)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        x_out = forward_body(x_in, pad_len)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        result['capture_ms'] = (time.perf_counter() - t_cap_start) * 1000

        # Warmup replays
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

        # Benchmark
        N_bench = max(5, int(500 / max(result['capture_ms'] / 10, 1)))
        t0 = time.perf_counter()
        for _ in range(N_bench):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        result['bench_ms'] = (time.perf_counter() - t0) / N_bench * 1000
        result['bench_n'] = N_bench
        result['success'] = True

        print(f"    Capture: {result['capture_ms']:.0f}ms")
        print(f"    Replay:  {result['bench_ms']:.2f}ms avg ({N_bench} iters, {1000/result['bench_ms']:.0f} fwd/sec)")

        # Correctness check
        real_tokens = list(range(min(pad_len, 50257)))[:pad_len]
        emb = (wte[real_tokens] + wpe[:pad_len])[None, :, :]
        emb_t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
        while emb_t.dim() < 2:
            emb_t = emb_t.unsqueeze(0)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(emb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            x_in
        )
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        out = from_dev(x_out, (1, pad_len, d_model))
        result['output_finite'] = bool(np.isfinite(out).all())
        print(f"    Output finite: {result['output_finite']}")

        ttnn.release_trace(device, tid)

    except Exception as e:
        result['error'] = str(e)
        print(f"    FAILED: {e}")
        traceback.print_exc()

    trace_results[pad_len] = result

# Part B: Simultaneous traces (all held at once)
print("\n--- Part B: Simultaneous trace capture (all held in memory) ---")
viable = [pl for pl in PAD_LENS if trace_results[pl]['success']]
print(f"  Attempting: {viable}")

sim_traces = {}
sim_success = {}

for pad_len in viable:
    print(f"\n  Capturing pad_len={pad_len} (keeping all previous)...")
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)

    try:
        # Warmup
        x_warm = to_dev(dummy)
        x_warm = forward_body(x_warm, pad_len)
        _ = from_dev(x_warm, (1, pad_len, d_model))

        # Capture (do NOT release)
        x_in = to_dev(dummy)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        x_out = forward_body(x_in, pad_len)
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # Verify
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

        sim_traces[pad_len] = (tid, x_in, x_out)
        sim_success[pad_len] = True
        print(f"    OK (total held: {len(sim_traces)})")

    except Exception as e:
        sim_success[pad_len] = False
        print(f"    FAILED: {e}")
        print(f"    Device memory exhausted. Held so far: {list(sim_traces.keys())}")
        break

# Verify all simultaneous traces still work
if len(sim_traces) > 1:
    print(f"\n  Verifying all {len(sim_traces)} traces still replay...")
    for pad_len, (tid, x_in, x_out) in sim_traces.items():
        try:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            out = from_dev(x_out, (1, pad_len, d_model))
            print(f"    pad_len={pad_len}: finite={np.isfinite(out).all()}")
        except Exception as e:
            print(f"    pad_len={pad_len}: REPLAY FAILED — {e}")

# Release all simultaneous traces
for pad_len, (tid, _, _) in sim_traces.items():
    ttnn.release_trace(device, tid)
print(f"  Released {len(sim_traces)} traces.")


# ══════════════════════════════════════════════════════════════
# TEST 2: KV-cached decode at long sequences
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: KV-cached decode — constant latency at long sequences")
print("=" * 70)

# Allocate KV caches
print("  Allocating KV caches...")
k_caches = []
v_caches = []
for i in range(n_layers):
    cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
    k_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
    v_caches.append(ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT))
cache_mb = n_layers * 2 * n_heads * max_seq * head_dim * 2 / 1024 / 1024
print(f"  {n_layers * 2} caches allocated ({cache_mb:.1f} MB)")


def reset_kv_caches():
    """Zero out all KV caches for a fresh run."""
    for i in range(n_layers):
        cache_np = np.zeros((1, n_heads, max_seq, head_dim), dtype=np.float32)
        k_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                       dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        v_caches[i] = ttnn.from_torch(torch.from_numpy(cache_np.copy()),
                                       dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)


def prefill_layer(x_tt, w, layer_idx, seq_len):
    """Prefill one layer: full-sequence SDPA, store K/V in cache."""
    h = ttnn.layer_norm(x_tt, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])
    q = ttnn.transpose(ttnn.reshape(q, [1, seq_len, n_heads, head_dim]), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, [1, seq_len, n_heads, head_dim]), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, [1, seq_len, n_heads, head_dim]), 1, 2)
    ttnn.kv_cache.fill_cache_for_user_(k_caches[layer_idx], k, batch_index=0)
    ttnn.kv_cache.fill_cache_for_user_(v_caches[layer_idx], v, batch_index=0)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    merged = ttnn.transformer.concatenate_heads(attn)
    x_tt = ttnn.add(x_tt, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))
    h2 = ttnn.layer_norm(x_tt, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x_tt, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))


def prefill(token_ids):
    """Run full prompt through all layers, fill KV caches, return logits."""
    seq_len = len(token_ids)
    pad_len = ((seq_len + 31) // 32) * 32
    ids = list(token_ids) + [50256] * (pad_len - seq_len)
    emb = (wte[ids] + wpe[:pad_len])[None, :, :]
    x = to_dev(emb)
    for i in range(n_layers):
        x = prefill_layer(x, layer_w[i], i, pad_len)
    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)
    out = from_dev(x, (1, pad_len, d_model))
    return out[0, seq_len - 1, :] @ wte.T


def decode_layer(x_tt, w, layer_idx, pos):
    """Decode one layer: single-token Q, Flash-Decode against KV cache."""
    h = ttnn.layer_norm(x_tt, weight=w['ln1_g'], bias=w['ln1_b'], epsilon=1e-5)
    q = ttnn.add(ttnn.matmul(h, w['w_q']), w['b_q'])
    k_new = ttnn.add(ttnn.matmul(h, w['w_k']), w['b_k'])
    v_new = ttnn.add(ttnn.matmul(h, w['w_v']), w['b_v'])
    k_new = ttnn.transpose(ttnn.reshape(k_new, [1, 1, n_heads, head_dim]), 1, 2)
    v_new = ttnn.transpose(ttnn.reshape(v_new, [1, 1, n_heads, head_dim]), 1, 2)
    ttnn.kv_cache.update_cache_for_token_(k_caches[layer_idx], k_new,
                                           update_index=pos, batch_offset=0)
    ttnn.kv_cache.update_cache_for_token_(v_caches[layer_idx], v_new,
                                           update_index=pos, batch_offset=0)
    q = ttnn.reshape(q, [1, 1, n_heads, head_dim])
    attn = ttnn.transformer.scaled_dot_product_attention_decode(
        q, k_caches[layer_idx], v_caches[layer_idx], cur_pos=[pos]
    )
    # Flash-Decode output [1,1,12,64] -> CPU reshape to [1,1,768]
    attn_np = ttnn.to_torch(attn).float().numpy()
    merged = to_dev(attn_np.reshape(1, 1, d_model))
    x_tt = ttnn.add(x_tt, ttnn.add(ttnn.matmul(merged, w['w_proj']), w['b_proj']))
    h2 = ttnn.layer_norm(x_tt, weight=w['ln2_g'], bias=w['ln2_b'], epsilon=1e-5)
    ff = ttnn.gelu(ttnn.add(ttnn.matmul(h2, w['w_fc']), w['b_fc']),
                   fast_and_approximate_mode=False)
    return ttnn.add(x_tt, ttnn.add(ttnn.matmul(ff, w['w_mlp']), w['b_mlp']))


def decode_step(token_id, pos):
    """Single decode step: one new token -> logits for next token."""
    emb = (wte[token_id] + wpe[pos])[None, None, :]  # (1, 1, 768)
    x = to_dev(emb)
    for i in range(n_layers):
        x = decode_layer(x, layer_w[i], i, pos)
    x = ttnn.layer_norm(x, weight=ln_f_g, bias=ln_f_b, epsilon=1e-5)
    out = from_dev(x, (1, 1, d_model))
    return out[0, 0, :] @ wte.T


def generate_kv(prompt_text, n_tokens):
    """Generate n_tokens using prefill + KV-cached decode. Returns (tokens, prefill_ms, decode_times)."""
    reset_kv_caches()
    token_ids = encode_simple(prompt_text)

    t0 = time.perf_counter()
    logits = prefill(token_ids)
    t_prefill = (time.perf_counter() - t0) * 1000

    pos = len(token_ids)
    generated = []
    decode_times = []

    for step in range(n_tokens):
        tok = int(np.argmax(logits))
        generated.append(tok)
        if tok == 50256:  # EOS
            break
        t0 = time.perf_counter()
        logits = decode_step(tok, pos)
        dt = (time.perf_counter() - t0) * 1000
        decode_times.append(dt)
        pos += 1

    return token_ids, generated, t_prefill, decode_times


# Run KV-cached decode at different generation lengths
prompt = "The meaning of life is"
kv_results = {}

for n_tokens in [100, 200, 500]:
    print(f"\n  Generating {n_tokens} tokens with KV cache...")
    try:
        token_ids, generated, t_prefill, decode_times = generate_kv(prompt, n_tokens)
        actual_generated = len(generated)

        # Analyze latency over windows
        if len(decode_times) >= 20:
            first_20 = np.mean(decode_times[:20])
            last_20 = np.mean(decode_times[-20:])
            mid_start = len(decode_times) // 2 - 10
            mid_20 = np.mean(decode_times[mid_start:mid_start+20])
        else:
            first_20 = last_20 = mid_20 = np.mean(decode_times) if decode_times else 0

        avg_all = np.mean(decode_times) if decode_times else 0
        kv_results[n_tokens] = {
            'success': True,
            'actual': actual_generated,
            'prefill_ms': t_prefill,
            'avg_ms': avg_all,
            'first_20_ms': first_20,
            'mid_20_ms': mid_20,
            'last_20_ms': last_20,
            'min_ms': min(decode_times) if decode_times else 0,
            'max_ms': max(decode_times) if decode_times else 0,
        }

        text_preview = decode_tokens(token_ids + generated[:30])
        print(f"    Prefill: {t_prefill:.0f}ms")
        print(f"    Generated: {actual_generated} tokens")
        print(f"    Avg decode: {avg_all:.1f}ms/tok ({1000/avg_all:.1f} tok/sec)")
        print(f"    First 20: {first_20:.1f}ms | Mid 20: {mid_20:.1f}ms | Last 20: {last_20:.1f}ms")
        print(f"    Min: {min(decode_times):.1f}ms, Max: {max(decode_times):.1f}ms")
        print(f"    Preview: '{text_preview}...'")

    except Exception as e:
        kv_results[n_tokens] = {'success': False, 'error': str(e)}
        print(f"    FAILED: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# TEST 3: Combined — traced prefill + KV-cached decode
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Combined — traced prefill (fast) + KV-cached decode (unlimited)")
print("=" * 70)

# For the combined approach, we use the traced forward for prefill speed,
# then switch to KV-cached decode for generation.
# The trick: after traced prefill, we need to populate the KV caches.
# So we do a KV-aware prefill (not traced) to fill caches, then decode.
# But we can measure: how much faster is traced prefill vs KV prefill?

# First, capture a trace for prefill at pad_len=32 (typical prompt size)
PREFILL_PAD = 32
print(f"\n  Capturing prefill trace (pad_len={PREFILL_PAD})...")

try:
    # Warmup
    dummy = np.zeros((1, PREFILL_PAD, d_model), dtype=np.float32)
    x_warm = to_dev(dummy)
    x_warm = forward_body(x_warm, PREFILL_PAD)
    _ = from_dev(x_warm, (1, PREFILL_PAD, d_model))

    # Capture
    x_in_pf = to_dev(dummy)
    tid_pf = ttnn.begin_trace_capture(device, cq_id=0)
    x_out_pf = forward_body(x_in_pf, PREFILL_PAD)
    ttnn.end_trace_capture(device, tid_pf, cq_id=0)

    # Verify
    ttnn.execute_trace(device, tid_pf, cq_id=0, blocking=True)
    print(f"    Trace captured successfully.")

    # Benchmark: traced prefill vs KV prefill for same prompt
    prompt_text = "The meaning of life is"
    token_ids = encode_simple(prompt_text)
    n_gen = 50

    # --- Method A: KV-only (prefill + decode) ---
    print(f"\n  Method A: KV-only (prefill + decode, {n_gen} tokens)...")
    reset_kv_caches()
    t0 = time.perf_counter()
    logits = prefill(token_ids)
    t_kv_prefill = (time.perf_counter() - t0) * 1000

    pos = len(token_ids)
    kv_only_times = []
    tok = int(np.argmax(logits))
    generated_a = [tok]
    for step in range(n_gen - 1):
        if tok == 50256:
            break
        t0 = time.perf_counter()
        logits = decode_step(tok, pos)
        kv_only_times.append((time.perf_counter() - t0) * 1000)
        tok = int(np.argmax(logits))
        generated_a.append(tok)
        pos += 1

    total_a = t_kv_prefill + sum(kv_only_times)
    print(f"    KV prefill: {t_kv_prefill:.0f}ms")
    print(f"    Decode avg: {np.mean(kv_only_times):.1f}ms/tok")
    print(f"    Total: {total_a:.0f}ms for {len(generated_a)} tokens")
    print(f"    Text: '{decode_tokens(token_ids + generated_a[:20])}...'")

    # --- Method B: Traced prefill + KV decode ---
    # Traced prefill gives us logits fast, but we still need KV caches filled.
    # Strategy: do traced prefill for fast logits, then do KV prefill in background
    # to fill caches, then switch to KV decode.
    # Actually — the honest approach is to just use KV prefill (it fills caches AND
    # gives logits). The traced prefill is only useful if we DON'T need KV caches
    # (i.e., pure traced generation where we re-run the full sequence each time).
    #
    # So the real combined approach is:
    #   - Use traced execution for short sequences (up to max bucket size)
    #   - Switch to KV-cached decode once we exceed the max traced bucket
    #
    # Let's implement that.

    print(f"\n  Method B: Traced generation (up to bucket limit) + KV decode (beyond)...")

    # Find largest available trace bucket
    max_bucket = max([pl for pl in PAD_LENS if trace_results.get(pl, {}).get('success', False)], default=0)
    print(f"    Max trace bucket: {max_bucket}")

    if max_bucket > 0:
        # Re-capture traces for the combined approach
        bucket_traces = {}
        for pad_len in sorted([pl for pl in PAD_LENS if pl <= max_bucket and trace_results[pl]['success']]):
            dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)
            x_warm = to_dev(dummy)
            x_warm = forward_body(x_warm, pad_len)
            _ = from_dev(x_warm, (1, pad_len, d_model))

            x_in = to_dev(dummy)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            x_out = forward_body(x_in, pad_len)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            bucket_traces[pad_len] = (tid, x_in, x_out)

        def get_bucket(seq_len):
            """Find smallest trace bucket that fits seq_len."""
            for pl in sorted(bucket_traces.keys()):
                if pl >= seq_len:
                    return pl
            return None

        # Generate with traced approach as long as we can, then switch to KV
        t0_total = time.perf_counter()
        all_tokens = list(token_ids)
        traced_count = 0

        # Phase 1: Traced generation
        while len(all_tokens) < len(token_ids) + n_gen:
            bucket = get_bucket(len(all_tokens))
            if bucket is None:
                break  # Switch to KV decode

            logits = traced_forward(all_tokens, bucket, bucket_traces[bucket])
            tok = int(np.argmax(logits))
            if tok == 50256:
                break
            all_tokens.append(tok)
            traced_count += 1

        t_traced_phase = (time.perf_counter() - t0_total) * 1000
        remaining = n_gen - traced_count

        # Phase 2: KV decode for remaining tokens
        kv_decode_times = []
        if remaining > 0 and all_tokens[-1] != 50256:
            # Need to do KV prefill for the full sequence so far
            reset_kv_caches()
            t0_kv = time.perf_counter()
            logits = prefill(all_tokens)
            t_kv_switch = (time.perf_counter() - t0_kv) * 1000

            pos = len(all_tokens)
            for step in range(remaining):
                tok = int(np.argmax(logits))
                if tok == 50256:
                    break
                all_tokens.append(tok)
                t0 = time.perf_counter()
                logits = decode_step(tok, pos)
                kv_decode_times.append((time.perf_counter() - t0) * 1000)
                pos += 1
        else:
            t_kv_switch = 0

        t_total_b = (time.perf_counter() - t0_total) * 1000
        generated_b = all_tokens[len(token_ids):]

        print(f"    Traced phase: {traced_count} tokens in {t_traced_phase:.0f}ms ({t_traced_phase/max(traced_count,1):.1f}ms/tok)")
        if kv_decode_times:
            print(f"    KV switch: {t_kv_switch:.0f}ms (prefill {len(all_tokens) - len(kv_decode_times) - len(token_ids)} tokens)")
            print(f"    KV decode: {len(kv_decode_times)} tokens, {np.mean(kv_decode_times):.1f}ms/tok avg")
        print(f"    Total: {t_total_b:.0f}ms for {len(generated_b)} tokens")
        print(f"    Text: '{decode_tokens(token_ids + generated_b[:20])}...'")

        # Compare
        print(f"\n  Comparison ({n_gen} tokens):")
        print(f"    Method A (KV-only):     {total_a:.0f}ms ({total_a/len(generated_a):.1f}ms/tok)")
        print(f"    Method B (trace+KV):    {t_total_b:.0f}ms ({t_total_b/max(len(generated_b),1):.1f}ms/tok)")

        # Release combined traces
        for pl, (tid, _, _) in bucket_traces.items():
            ttnn.release_trace(device, tid)
    else:
        print("    No trace buckets available, skipping combined approach.")

    ttnn.release_trace(device, tid_pf)

except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# TEST 4: Peak throughput — fastest 100 tokens
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Peak throughput — fastest way to generate 100 tokens")
print("=" * 70)

N_GEN = 100
prompt_text = "The future of artificial intelligence"
token_ids = encode_simple(prompt_text)
throughput_results = {}

# --- Approach A: Pure traced (re-run full sequence each step, up to max bucket) ---
max_traced_bucket = max([pl for pl in PAD_LENS if trace_results.get(pl, {}).get('success', False)], default=0)
if max_traced_bucket >= len(token_ids) + N_GEN:
    print(f"\n  Approach A: Pure traced (full-recompute each step, bucket={max_traced_bucket})...")
    # Need a trace for the right bucket
    pad_len = max_traced_bucket
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)
    x_warm = to_dev(dummy)
    x_warm = forward_body(x_warm, pad_len)
    _ = from_dev(x_warm, (1, pad_len, d_model))

    x_in_a = to_dev(dummy)
    tid_a = ttnn.begin_trace_capture(device, cq_id=0)
    x_out_a = forward_body(x_in_a, pad_len)
    ttnn.end_trace_capture(device, tid_a, cq_id=0)

    # Warmup
    traced_forward(token_ids, pad_len, (tid_a, x_in_a, x_out_a))

    t0 = time.perf_counter()
    all_tokens = list(token_ids)
    for step in range(N_GEN):
        logits = traced_forward(all_tokens, pad_len, (tid_a, x_in_a, x_out_a))
        tok = int(np.argmax(logits))
        if tok == 50256:
            break
        all_tokens.append(tok)
    t_a = (time.perf_counter() - t0) * 1000
    gen_a = len(all_tokens) - len(token_ids)

    throughput_results['pure_traced'] = {
        'time_ms': t_a, 'tokens': gen_a, 'ms_per_tok': t_a / max(gen_a, 1)
    }
    print(f"    {gen_a} tokens in {t_a:.0f}ms ({t_a/max(gen_a,1):.1f}ms/tok, {1000*gen_a/t_a:.0f} tok/sec)")
    print(f"    Text: '{decode_tokens(all_tokens[:len(token_ids)+20])}...'")
    ttnn.release_trace(device, tid_a)
else:
    print(f"\n  Approach A: Pure traced — SKIPPED (max bucket {max_traced_bucket} < {len(token_ids) + N_GEN} needed)")
    throughput_results['pure_traced'] = {'time_ms': 0, 'tokens': 0, 'ms_per_tok': 0, 'note': 'bucket too small'}

# --- Approach B: Pure KV-cached decode ---
print(f"\n  Approach B: Pure KV-cached decode...")
reset_kv_caches()

t0 = time.perf_counter()
logits = prefill(token_ids)
t_pf = (time.perf_counter() - t0) * 1000

pos = len(token_ids)
all_tokens = list(token_ids)
decode_times_b = []
for step in range(N_GEN):
    tok = int(np.argmax(logits))
    if tok == 50256:
        break
    all_tokens.append(tok)
    t0 = time.perf_counter()
    logits = decode_step(tok, pos)
    decode_times_b.append((time.perf_counter() - t0) * 1000)
    pos += 1

t_b = t_pf + sum(decode_times_b)
gen_b = len(all_tokens) - len(token_ids)
throughput_results['kv_cached'] = {
    'time_ms': t_b, 'tokens': gen_b, 'ms_per_tok': t_b / max(gen_b, 1),
    'prefill_ms': t_pf, 'decode_avg_ms': np.mean(decode_times_b) if decode_times_b else 0,
}
print(f"    Prefill: {t_pf:.0f}ms")
print(f"    {gen_b} tokens in {t_b:.0f}ms ({t_b/max(gen_b,1):.1f}ms/tok, {1000*gen_b/t_b:.0f} tok/sec)")
print(f"    Decode avg: {np.mean(decode_times_b):.1f}ms/tok")
print(f"    Text: '{decode_tokens(all_tokens[:len(token_ids)+20])}...'")

# --- Approach C: Traced expanding (use bucket traces, grow sequence) ---
# Re-run the traced approach with growing buckets
viable_buckets = sorted([pl for pl in PAD_LENS if trace_results.get(pl, {}).get('success', False)])
if viable_buckets:
    print(f"\n  Approach C: Traced with bucket progression {viable_buckets}...")

    # Capture fresh traces
    bucket_traces_c = {}
    for pad_len in viable_buckets:
        dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)
        x_warm = to_dev(dummy)
        x_warm = forward_body(x_warm, pad_len)
        _ = from_dev(x_warm, (1, pad_len, d_model))

        x_in = to_dev(dummy)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        x_out = forward_body(x_in, pad_len)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        bucket_traces_c[pad_len] = (tid, x_in, x_out)

    # Warmup
    for pl in viable_buckets:
        traced_forward(token_ids[:min(len(token_ids), pl)], pl, bucket_traces_c[pl])

    t0 = time.perf_counter()
    all_tokens = list(token_ids)
    traced_steps = 0
    for step in range(N_GEN):
        # Find bucket
        bucket = None
        for pl in viable_buckets:
            if pl >= len(all_tokens):
                bucket = pl
                break
        if bucket is None:
            break  # Exceeded all buckets

        logits = traced_forward(all_tokens, bucket, bucket_traces_c[bucket])
        tok = int(np.argmax(logits))
        if tok == 50256:
            break
        all_tokens.append(tok)
        traced_steps += 1

    t_c_traced = (time.perf_counter() - t0) * 1000
    gen_c_traced = traced_steps

    # If we ran out of buckets, switch to KV
    remaining = N_GEN - gen_c_traced
    t_c_kv = 0
    kv_steps = 0
    if remaining > 0 and (len(all_tokens) == 0 or all_tokens[-1] != 50256):
        reset_kv_caches()
        t0_kv = time.perf_counter()
        logits = prefill(all_tokens)
        t_kv_pf = (time.perf_counter() - t0_kv) * 1000

        pos = len(all_tokens)
        kv_times = []
        for step in range(remaining):
            tok = int(np.argmax(logits))
            if tok == 50256:
                break
            all_tokens.append(tok)
            t0 = time.perf_counter()
            logits = decode_step(tok, pos)
            kv_times.append((time.perf_counter() - t0) * 1000)
            pos += 1
        t_c_kv = t_kv_pf + sum(kv_times)
        kv_steps = len(kv_times)

    t_c = t_c_traced + t_c_kv
    gen_c = len(all_tokens) - len(token_ids)

    throughput_results['traced_plus_kv'] = {
        'time_ms': t_c, 'tokens': gen_c, 'ms_per_tok': t_c / max(gen_c, 1),
        'traced_tokens': gen_c_traced, 'traced_ms': t_c_traced,
        'kv_tokens': kv_steps, 'kv_ms': t_c_kv,
    }
    print(f"    Traced phase: {gen_c_traced} tokens in {t_c_traced:.0f}ms ({t_c_traced/max(gen_c_traced,1):.1f}ms/tok)")
    if kv_steps > 0:
        print(f"    KV phase: {kv_steps} tokens in {t_c_kv:.0f}ms ({t_c_kv/max(kv_steps,1):.1f}ms/tok)")
    print(f"    Total: {gen_c} tokens in {t_c:.0f}ms ({t_c/max(gen_c,1):.1f}ms/tok, {1000*gen_c/t_c:.0f} tok/sec)")
    print(f"    Text: '{decode_tokens(all_tokens[:len(token_ids)+20])}...'")

    # Release traces
    for pl, (tid, _, _) in bucket_traces_c.items():
        ttnn.release_trace(device, tid)


# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Test 1: Trace buckets
print("\n--- Trace Bucket Scaling ---")
print(f"  {'Bucket':>8} {'Capture':>10} {'Replay':>10} {'fwd/sec':>10} {'Status':>10}")
print("  " + "-" * 50)
for pl in PAD_LENS:
    r = trace_results.get(pl, {})
    if r.get('success'):
        print(f"  {pl:>8} {r['capture_ms']:>9.0f}ms {r['bench_ms']:>9.2f}ms "
              f"{1000/r['bench_ms']:>9.0f} {'OK':>10}")
    else:
        err = r.get('error', 'unknown')[:20]
        print(f"  {pl:>8} {'--':>10} {'--':>10} {'--':>10} {'FAIL':>10}")

n_sim = len(sim_traces)  # note: already released, but count was captured
print(f"\n  Simultaneous traces held: {list(sim_success.keys())}")
print(f"  Max simultaneous: {n_sim} buckets ({sorted([pl for pl, ok in sim_success.items() if ok])})")

# Test 2: KV decode latency
print("\n--- KV-Cached Decode Latency ---")
print(f"  {'Gen Len':>8} {'Prefill':>10} {'Avg':>10} {'First20':>10} {'Mid20':>10} {'Last20':>10} {'Status':>10}")
print("  " + "-" * 62)
for n_tok in [100, 200, 500]:
    r = kv_results.get(n_tok, {})
    if r.get('success'):
        print(f"  {n_tok:>8} {r['prefill_ms']:>9.0f}ms {r['avg_ms']:>9.1f}ms "
              f"{r['first_20_ms']:>9.1f}ms {r['mid_20_ms']:>9.1f}ms {r['last_20_ms']:>9.1f}ms {'OK':>10}")
    else:
        print(f"  {n_tok:>8} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {'FAIL':>10}")

# Test 4: Peak throughput
print("\n--- Peak Throughput (100 tokens) ---")
print(f"  {'Approach':>20} {'Time':>10} {'Tokens':>8} {'ms/tok':>10} {'tok/sec':>10}")
print("  " + "-" * 60)
for name, label in [('pure_traced', 'Pure Traced'), ('kv_cached', 'KV-Cached'), ('traced_plus_kv', 'Traced + KV')]:
    r = throughput_results.get(name, {})
    if r.get('tokens', 0) > 0:
        print(f"  {label:>20} {r['time_ms']:>9.0f}ms {r['tokens']:>8} "
              f"{r['ms_per_tok']:>9.1f}ms {1000*r['tokens']/r['time_ms']:>9.0f}")
    else:
        note = r.get('note', 'N/A')
        print(f"  {label:>20} {'--':>10} {'--':>8} {'--':>10} {note:>10}")

# Key insights
print("\n--- Key Findings ---")
if trace_results.get(512, {}).get('success'):
    print(f"  [*] 512-token trace bucket WORKS: {trace_results[512]['bench_ms']:.1f}ms/fwd")
else:
    print(f"  [*] 512-token trace bucket FAILED")

if kv_results.get(500, {}).get('success'):
    r = kv_results[500]
    drift = abs(r['last_20_ms'] - r['first_20_ms'])
    pct = drift / r['first_20_ms'] * 100 if r['first_20_ms'] > 0 else 0
    if pct < 10:
        print(f"  [*] KV decode latency is CONSTANT: {r['avg_ms']:.1f}ms/tok (drift: {pct:.1f}%)")
    else:
        print(f"  [*] KV decode latency DRIFTS: first={r['first_20_ms']:.1f}ms, last={r['last_20_ms']:.1f}ms ({pct:.1f}%)")

best = min(throughput_results.values(), key=lambda r: r.get('ms_per_tok', float('inf')) if r.get('tokens', 0) > 0 else float('inf'))
best_name = [k for k, v in throughput_results.items() if v is best][0]
if best.get('tokens', 0) > 0:
    print(f"  [*] Fastest 100-token approach: {best_name} at {best['ms_per_tok']:.1f}ms/tok ({1000/best['ms_per_tok']:.0f} tok/sec)")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("\nDone!")
