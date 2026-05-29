"""
Experiment 34: Trace bucket scaling — how far can we push it?

We currently capture traces for pad_len=32 and pad_len=64, giving fast
inference up to 64 tokens. Before investing in KV caching, let's see if
we can just capture more trace buckets (128, 256, 512) as a quick win.

Questions:
  1. How long does warmup + capture take for each pad_len?
  2. How long does a single trace replay take at each size?
  3. Can we hold ALL traces (32, 64, 128, 256) in device memory simultaneously?
  4. Where does device memory become the bottleneck?

Approach: same GPT-2 forward as demo.py with pre-split QKV weights.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch
import traceback

# ── Load GPT-2 weights ──────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 small (124M params)...")
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")

with open(config_path) as f:
    cfg = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

n_heads, d_model, n_layers = cfg['n_head'], cfg['n_embd'], cfg['n_layer']
head_dim = d_model // n_heads
wte, wpe = weights["wte.weight"], weights["wpe.weight"]

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}, head_dim={head_dim}")

# ── Open device ──────────────────────────────────────────────
import ttnn

print(f"Opening Blackhole device 0...")
device = ttnn.open_device(device_id=0)

# ── Helpers ──────────────────────────────────────────────────

def to_dev(arr):
    """Upload numpy array to Blackhole as bfloat16 tile-layout tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    """Download tensor from Blackhole."""
    t = ttnn.to_torch(tensor).float()
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)

# ── Upload weights (pre-split QKV) ──────────────────────────
print("Uploading 124M parameters to device (pre-splitting QKV)...")
t0 = time.perf_counter()

layer_w = []
for i in range(n_layers):
    p = f"h.{i}"
    w_attn = weights[f"{p}.attn.c_attn.weight"]  # (768, 2304)
    b_attn = weights[f"{p}.attn.c_attn.bias"]     # (2304,)
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
t_upload = (time.perf_counter() - t0) * 1000
print(f"  Weight upload: {t_upload:.0f}ms")

# ── GPT-2 forward (zero CPU round-trips) ────────────────────

def gpt2_layer(x, w, seq_len):
    """One transformer layer. All ops on device."""
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

# ══════════════════════════════════════════════════════════════
# Part 1: Individual trace capture (one at a time, release after)
# ══════════════════════════════════════════════════════════════
PAD_LENS = [32, 64, 128, 256]

print("\n" + "=" * 70)
print("PART 1: Individual trace capture (one at a time, release after each)")
print("=" * 70)

results = {}

for pad_len in PAD_LENS:
    print(f"\n--- pad_len = {pad_len} ---")
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)
    result = {'pad_len': pad_len, 'success': False}

    try:
        # Warmup: establishes buffer sizes for trace capture
        t_warmup_start = time.perf_counter()
        x_warm = to_dev(dummy)
        x_warm = forward_body(x_warm, pad_len)
        # Force sync by reading back a value
        _ = from_dev(x_warm, (1, pad_len, d_model))
        t_warmup = (time.perf_counter() - t_warmup_start) * 1000
        result['warmup_ms'] = t_warmup
        print(f"  Warmup:  {t_warmup:.0f}ms")

        # Trace capture
        t_capture_start = time.perf_counter()
        x_in = to_dev(dummy)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        x_out = forward_body(x_in, pad_len)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        t_capture = (time.perf_counter() - t_capture_start) * 1000
        result['capture_ms'] = t_capture
        print(f"  Capture: {t_capture:.0f}ms")

        # Single replay (verify it works + measure latency)
        t_replay_start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t_replay = (time.perf_counter() - t_replay_start) * 1000
        result['replay_ms'] = t_replay
        print(f"  Replay:  {t_replay:.1f}ms")

        # Warmup replays then benchmark
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        N_bench = max(5, int(500 / max(t_replay, 1)))  # aim for ~500ms total
        t_bench_start = time.perf_counter()
        for _ in range(N_bench):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        t_bench_avg = (time.perf_counter() - t_bench_start) / N_bench * 1000
        result['bench_ms'] = t_bench_avg
        result['bench_n'] = N_bench
        print(f"  Bench:   {t_bench_avg:.2f}ms avg ({N_bench} iters, {1000/t_bench_avg:.0f} fwd/sec)")

        # Correctness check: run a real input and verify output is finite
        real_tokens = list(range(pad_len))  # arbitrary token ids
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
        logits = out[0, -1, :] @ wte.T
        top_tok = int(np.argmax(logits))
        finite = np.isfinite(out).all()
        result['output_finite'] = bool(finite)
        result['top_token'] = top_tok
        print(f"  Output:  finite={finite}, top_token={top_tok}")

        result['success'] = True

        # Release this trace before next pad_len
        ttnn.release_trace(device, tid)
        print(f"  Trace released.")

    except Exception as e:
        result['error'] = str(e)
        print(f"  FAILED: {e}")
        traceback.print_exc()

    results[pad_len] = result

# ══════════════════════════════════════════════════════════════
# Part 2: Simultaneous trace capture (all at once, no releasing)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: Simultaneous traces (all held in device memory at once)")
print("=" * 70)

# Only attempt pad_lens that succeeded individually
viable = [pl for pl in PAD_LENS if results[pl]['success']]
print(f"Attempting simultaneous capture for: {viable}")

sim_traces = {}
sim_results = {}

for pad_len in viable:
    print(f"\n--- Capturing pad_len = {pad_len} (keeping all previous) ---")
    dummy = np.zeros((1, pad_len, d_model), dtype=np.float32)

    try:
        # Warmup
        x_warm = to_dev(dummy)
        x_warm = forward_body(x_warm, pad_len)
        _ = from_dev(x_warm, (1, pad_len, d_model))

        # Capture (do NOT release)
        t0 = time.perf_counter()
        x_in = to_dev(dummy)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        x_out = forward_body(x_in, pad_len)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        t_cap = (time.perf_counter() - t0) * 1000

        # Verify replay
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

        sim_traces[pad_len] = (tid, x_in, x_out)
        sim_results[pad_len] = {'success': True, 'capture_ms': t_cap}
        print(f"  OK — captured in {t_cap:.0f}ms (total traces held: {len(sim_traces)})")

    except Exception as e:
        sim_results[pad_len] = {'success': False, 'error': str(e)}
        print(f"  FAILED: {e}")
        traceback.print_exc()
        print(f"  Device memory likely exhausted at pad_len={pad_len}.")
        print(f"  Traces held so far: {list(sim_traces.keys())}")
        break

# Verify all simultaneous traces still work
if len(sim_traces) > 1:
    print(f"\n--- Verifying all {len(sim_traces)} traces still replay correctly ---")
    for pad_len, (tid, x_in, x_out) in sim_traces.items():
        try:
            # Write real data and replay
            real_tokens = list(range(pad_len))
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
            finite = np.isfinite(out).all()
            print(f"  pad_len={pad_len}: finite={finite}")
        except Exception as e:
            print(f"  pad_len={pad_len}: REPLAY FAILED — {e}")

# Release all simultaneous traces
for pad_len, (tid, _, _) in sim_traces.items():
    ttnn.release_trace(device, tid)

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'pad_len':>8} {'warmup':>10} {'capture':>10} {'replay':>10} {'bench':>10} {'fwd/sec':>10} {'status':>10}")
print("-" * 70)
for pad_len in PAD_LENS:
    r = results[pad_len]
    if r['success']:
        print(f"{pad_len:>8} {r['warmup_ms']:>9.0f}ms {r['capture_ms']:>9.0f}ms "
              f"{r['replay_ms']:>9.1f}ms {r['bench_ms']:>9.2f}ms "
              f"{1000/r['bench_ms']:>9.0f} {'OK':>10}")
    else:
        print(f"{pad_len:>8} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'FAIL':>10}")

print(f"\nSimultaneous traces:")
for pad_len in viable:
    if pad_len in sim_results:
        sr = sim_results[pad_len]
        status = f"OK ({sr['capture_ms']:.0f}ms)" if sr['success'] else f"FAIL: {sr.get('error', '?')[:40]}"
        print(f"  pad_len={pad_len}: {status}")

max_sim = max(sim_traces.keys()) if sim_traces else 0
print(f"\nMax simultaneous trace pad_len: {max_sim}")
print(f"Total simultaneous buckets held: {len(sim_traces)}")

if max_sim >= 256:
    print("\nVerdict: 4 buckets (32-256) fit! Can generate up to 256 tokens with traces.")
elif max_sim >= 128:
    print("\nVerdict: 3 buckets (32-128) fit. 128 tokens max with full trace coverage.")
elif max_sim >= 64:
    print("\nVerdict: Only 2 buckets (32-64) fit. Same as current demo.py.")
else:
    print("\nVerdict: Memory too tight even for 2 buckets.")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("\nDone!")
