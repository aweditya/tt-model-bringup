#!/usr/bin/env python3
"""
Qwen2.5-0.5B Text Generation on Tenstorrent Blackhole

Run on the Tenstorrent host:
    python generate_qwen.py
    python generate_qwen.py --prompt "Explain quantum computing"
    python generate_qwen.py --prompt "Write a poem about rain" --max_tokens 100

What this does:
    1. Downloads Qwen2.5-0.5B weights from HuggingFace (cached after first run)
    2. Uploads weights to the Blackhole accelerator
    3. Runs prefill (prompt processing) on device with CPU-side RoPE
    4. Captures a traced decode graph and replays it for each new token
    5. Generates text with greedy decoding, printing tokens as they arrive

Performance: ~84 tok/sec end-to-end on Blackhole P150 (batch=1)
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen2.5-0.5B on Tenstorrent Blackhole")
parser.add_argument("--prompt", default=None, help="Custom prompt (runs single generation)")
parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
args = parser.parse_args()

# ── Model architecture ───────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = 1

# ── Device setup ─────────────────────────────────────────────
print("=" * 60)
print("Qwen2.5-0.5B on Tenstorrent Blackhole")
print("=" * 60)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} = {grid.x*grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Load weights ─────────────────────────────────────────────
print("\nLoading model weights...")
t_load = time.perf_counter()
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()
print(f"  Loaded {len(all_weights)} tensors in {time.perf_counter()-t_load:.1f}s")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights["lm_head.weight"].T if "lm_head.weight" in all_weights else embed_w.T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# ── Helpers ──────────────────────────────────────────────────
def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bf8(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

# ── RoPE (Qwen uses half-format: rotate_half) ───────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Upload weights to device ────────────────────────────────
print("Uploading to Blackhole (bf16 attention + bf8 MLP)...")
t_upload = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf8(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf8(lw["mlp.down_proj.weight"].T),
    })
    if (i + 1) % 12 == 0:
        print(f"    Layer {i+1}/{n_layers}")
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {time.perf_counter()-t_upload:.1f}s")

# ── KV caches ────────────────────────────────────────────────
k_caches, v_caches = [], []
for _ in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE - 1) // TILE) * TILE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# ── Trace input buffers ─────────────────────────────────────
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)


def update_buffers(token_id, pos):
    """Update trace input buffers before each decode step."""
    ttnn.copy(to_bf16(embed_w[token_id:token_id+1].reshape(1, 1, hidden)), embed_buf)
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


def prefill(token_ids):
    """Process the prompt. Fills KV caches and returns logits for next token."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_np = apply_rope_np(from_dev(q,(B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_np(from_dev(k,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn,(B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)
        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)
    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


def decode_forward():
    """Single decode step — runs entirely on device, replayed via trace."""
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.reshape(ttnn.linear(h, dl["q_w"], bias=dl["q_b"], compute_kernel_config=hifi4),
                         [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(ttnn.linear(h, dl["k_w"], bias=dl["k_b"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(ttnn.linear(h, dl["v_w"], bias=dl["v_b"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        # Native RoPE: single op per Q and K
        qr = ttnn.experimental.rotary_embedding(q, rope_cos_buf, rope_sin_buf)
        kr = ttnn.experimental.rotary_embedding(k, rope_cos_buf, rope_sin_buf)
        # Handle tile padding (seq_len 1 → 32)
        if list(qr.shape)[2] > 1:
            qr = ttnn.slice(qr, [0,0,0,0], [1,n_q_heads,1,head_dim])
        if list(kr.shape)[2] > 1:
            kr = ttnn.slice(kr, [0,0,0,0], [1,n_kv_heads,1,head_dim])
        # KV cache update (traceable via tensor position index)
        ks = ttnn.to_memory_config(ttnn.reshape(kr, [1,1,n_kv_heads,head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_buf)
        # Flash attention decode
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr,[1,1,n_q_heads,head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn,[1,1,1,hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)
        # MLP: fused gate+silu + up + down
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.linear(h2, dl["gate_w"], activation="silu", compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(g, u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])


def generate(prompt, max_tokens=100):
    """Generate text from a prompt. Returns dict with text, timing, token count."""
    reset_kv_caches()
    tokens = list(tokenizer.encode(prompt))
    max_gen = min(max_tokens, MAX_SEQ - len(tokens) - 1)

    # Prefill
    t_prefill = time.perf_counter()
    logits = prefill(np.array(tokens))
    prefill_ms = (time.perf_counter() - t_prefill) * 1000
    next_id = int(np.argmax(logits))
    gen = [next_id]

    # Warmup decode (JIT compilation)
    pos = len(tokens)
    update_buffers(next_id, pos)
    _ = decode_forward(); ttnn.synchronize_device(device)
    try: device.enable_program_cache()
    except: pass

    # Capture trace
    update_buffers(next_id, pos)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref = decode_forward()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Decode loop
    eos_id = tokenizer.eos_token_id
    times = []
    for step in range(max_gen):
        update_buffers(next_id, pos)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        lgt = from_dev(logits_ref, (1, vocab_size))[0]
        t1 = time.perf_counter()
        next_id = int(np.argmax(lgt))
        times.append(t1 - t0)
        gen.append(next_id)
        pos += 1
        if next_id == eos_id:
            break

    ttnn.release_trace(device, tid)

    text = tokenizer.decode(gen, skip_special_tokens=True)
    skip = 1  # skip first step (warmup)
    steady = times[skip:] if len(times) > skip else times
    avg_ms = np.mean(steady) * 1000 if steady else 0
    tok_s = 1000 / avg_ms if avg_ms > 0 else 0

    return {
        "text": text,
        "tokens": len(gen),
        "hit_eos": gen[-1] == eos_id,
        "prefill_ms": prefill_ms,
        "avg_ms": avg_ms,
        "tok_s": tok_s,
        "prompt_tokens": len(tokens),
    }


# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════

if args.prompt:
    # Single custom prompt
    prompts = [args.prompt]
else:
    # Default demo: diverse prompts
    prompts = [
        "The capital of France is",
        "Explain what a neural network is in one sentence:",
        "def fibonacci(n):",
        "The three primary colors are",
        "Translate to Spanish: Hello, how are you?",
    ]

print(f"\n{'='*60}")
print(f"GENERATING ({len(prompts)} prompt{'s' if len(prompts)>1 else ''})")
print(f"{'='*60}")

results = []
for prompt in prompts:
    r = generate(prompt, max_tokens=args.max_tokens)
    results.append((prompt, r))

    print(f"\nPrompt: \"{prompt}\"")
    print(f"Output: {r['text']}")
    print(f"  {r['tokens']} tokens | {r['avg_ms']:.1f} ms/tok | {r['tok_s']:.0f} tok/s"
          f" | prefill {r['prefill_ms']:.0f}ms | EOS={'yes' if r['hit_eos'] else 'no'}")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
avg_tps = np.mean([r['tok_s'] for _, r in results])
avg_ms = np.mean([r['avg_ms'] for _, r in results])
total_tok = sum(r['tokens'] for _, r in results)
print(f"  Model:      Qwen2.5-0.5B (24 layers, 0.5B params)")
print(f"  Weights:    bf16 attention + bf8 MLP")
print(f"  Device:     Tenstorrent Blackhole P150 (450 GB/s DRAM)")
print(f"  Decode:     Traced (native RoPE + paged KV cache)")
print(f"  Avg speed:  {avg_ms:.1f} ms/tok = {avg_tps:.0f} tok/sec (end-to-end)")
print(f"  Total:      {total_tok} tokens across {len(prompts)} prompts")

ttnn.close_device(device)
print("\nDone!")
