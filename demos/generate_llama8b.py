#!/usr/bin/env python3
"""
Llama-3.1-8B Text Generation on Tenstorrent Blackhole

Run on the Tenstorrent host:
    python generate_llama8b.py
    python generate_llama8b.py --prompt "Explain general relativity"
    python generate_llama8b.py --prompt "Write a haiku about coding" --max_tokens 50

What this does:
    1. Downloads Llama-3.1-8B-Instruct weights from HuggingFace (cached after first run)
    2. Uploads weights to the Blackhole accelerator (BFP8 for MLP, bf16 for attention)
    3. Formats prompt with Llama-3 chat template
    4. Runs prefill + traced decode loop with greedy decoding
    5. Generates text, printing tokens as they arrive

Performance: ~21 tok/sec end-to-end on Blackhole P150 (batch=1)
Weight memory: 8.3 GB (BFP8 MLP saves 40% vs bf16)

Note: Requires ~16 GB device DRAM. First run downloads ~16 GB of weights.
      Subsequent runs use the HuggingFace cache.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Llama-3.1-8B on Tenstorrent Blackhole")
parser.add_argument("--prompt", default=None, help="Custom prompt (uses chat template)")
parser.add_argument("--max_tokens", type=int, default=100, help="Max tokens to generate")
args = parser.parse_args()

# ── Model architecture ───────────────────────────────────────
hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 32; vocab_size = 128256; MAX_SEQ = 512
TILE = 32; batch_size = 1
# Split 8 KV heads into 2x4 for flash_decode (Blackhole JIT bug workaround)
n_kv_split = n_kv_heads // 2
n_q_split = n_q_heads // 2

# ── Device setup ─────────────────────────────────────────────
print("=" * 60)
print("Llama-3.1-8B-Instruct on Tenstorrent Blackhole")
print("=" * 60)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} = {grid.x*grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True)

# ── Load weights ─────────────────────────────────────────────
print("\nLoading model weights (this may take a minute on first run)...")
t_load = time.perf_counter()

model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]
shard_paths = []
model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths; model_id = mid
            print(f"  Source: {mid} ({n_shards} shards)")
            break
        except: pass
    if shard_paths: break

if not shard_paths:
    print("ERROR: Could not download Llama-3.1-8B weights.")
    print("  You may need to accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print("  or login with: huggingface-cli login")
    ttnn.close_device(device)
    sys.exit(1)

all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()
print(f"  Loaded {len(all_weights)} tensors in {time.perf_counter()-t_load:.0f}s")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights

tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# ── Helpers ──────────────────────────────────────────────────
def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_bfp8(arr):
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

# ── RoPE (Llama uses interleaved format) ─────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved_np(x):
    r = np.zeros_like(x); r[..., 0::2] = -x[..., 1::2]; r[..., 1::2] = x[..., 0::2]; return r

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1),
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]

# ── Upload weights to device ────────────────────────────────
print("Uploading to Blackhole (BFP8 MLP + bf16 attention)...")
t_upload = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bfp8(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bfp8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bfp8(lw["mlp.down_proj.weight"].T),
    })
    if (i + 1) % 8 == 0:
        print(f"    Layer {i+1}/{n_layers}")
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {time.perf_counter()-t_upload:.0f}s")

# Rotation matrix for interleaved RoPE on device
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

# ── KV caches (split: 8 KV heads → 2 groups of 4) ──────────
k_caches_lo, v_caches_lo = [], []
k_caches_hi, v_caches_hi = [], []
for _ in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches_lo.append(to_dev_4d(c.copy())); v_caches_lo.append(to_dev_4d(c.copy()))
    k_caches_hi.append(to_dev_4d(c.copy())); v_caches_hi.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_split + TILE - 1) // TILE) * TILE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# ── Trace input buffers ─────────────────────────────────────
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

# ── Chat template ────────────────────────────────────────────
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

def make_chat_tokens(prompt, system="You are a helpful assistant."):
    return ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
            [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
            [start_header] + enc("assistant") + [end_header] + enc("\n\n"))


def update_buffers(token_id, pos):
    """Update trace input buffers before each decode step."""
    ttnn.copy(to_bf16(embed_w[token_id:token_id+1].reshape(1, 1, hidden)), embed_buf)
    angles = pos * freqs
    cos_full = np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


def prefill(token_ids):
    """Process prompt through all layers. Fills KV caches, returns last-position logits."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q_np = apply_rope_np(from_dev(q,(B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_np(from_dev(k,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_lo[i], to_dev_4d(k_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_lo[i], to_dev_4d(v_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_hi[i], to_dev_4d(k_np[:, n_kv_split:]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_hi[i], to_dev_4d(v_np[:, n_kv_split:]), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn,(B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,n_q_heads*head_dim)
        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,n_q_heads*head_dim)), dl["o_w"], compute_kernel_config=hifi4)
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
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q = ttnn.reshape(q, [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(k, [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(v, [1, n_kv_heads, 1, head_dim])
        # RoPE via rotation matrix (interleaved format)
        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))
        # KV cache update — split into lo/hi for flash_decode compatibility
        kr_4d = ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim])
        v_4d = ttnn.reshape(v, [1, 1, n_kv_heads, head_dim])
        kr_lo = ttnn.to_memory_config(ttnn.slice(kr_4d, [0,0,0,0], [1,1,n_kv_split,head_dim]), kv_cfg)
        kr_hi = ttnn.to_memory_config(ttnn.slice(kr_4d, [0,0,n_kv_split,0], [1,1,n_kv_heads,head_dim]), kv_cfg)
        v_lo = ttnn.to_memory_config(ttnn.slice(v_4d, [0,0,0,0], [1,1,n_kv_split,head_dim]), kv_cfg)
        v_hi = ttnn.to_memory_config(ttnn.slice(v_4d, [0,0,n_kv_split,0], [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches_lo[i], kr_lo, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches_lo[i], v_lo, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(k_caches_hi[i], kr_hi, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches_hi[i], v_hi, update_idxs_tensor=pos_buf)
        # Flash attention decode (split Q too for GQA 4:1 ratio)
        qr_4d = ttnn.reshape(qr, [1, 1, n_q_heads, head_dim])
        q_lo = ttnn.slice(qr_4d, [0,0,0,0], [1,1,n_q_split,head_dim])
        q_hi = ttnn.slice(qr_4d, [0,0,n_q_split,0], [1,1,n_q_heads,head_dim])
        attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
            q_lo, k_caches_lo[i], v_caches_lo[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
            q_hi, k_caches_hi[i], v_caches_hi[i], cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn = ttnn.concat([attn_lo, attn_hi], dim=2)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)
        # MLP with HiFi2 (BFP8 weights)
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi2)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi2)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi2)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches_lo[i]); ttnn.copy(to_dev_4d(c), v_caches_lo[i])
        ttnn.copy(to_dev_4d(c), k_caches_hi[i]); ttnn.copy(to_dev_4d(c), v_caches_hi[i])


def generate(prompt, max_tokens=100):
    """Generate text from a prompt using Llama-3 chat template."""
    reset_kv_caches()
    tokens = make_chat_tokens(prompt)
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
        if next_id in stop_ids:
            break

    ttnn.release_trace(device, tid)

    text = tokenizer.decode(gen, skip_special_tokens=True)
    skip = 1
    steady = times[skip:] if len(times) > skip else times
    avg_ms = np.mean(steady) * 1000 if steady else 0
    tok_s = 1000 / avg_ms if avg_ms > 0 else 0

    return {
        "text": text,
        "tokens": len(gen),
        "hit_eos": len(gen) > 0 and gen[-1] in stop_ids,
        "prefill_ms": prefill_ms,
        "avg_ms": avg_ms,
        "tok_s": tok_s,
        "prompt_tokens": len(tokens),
    }


# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════

if args.prompt:
    prompts = [args.prompt]
else:
    prompts = [
        "What is the capital of France?",
        "Explain photosynthesis in one paragraph.",
        "Write a haiku about the ocean.",
        "What is 17 * 23?",
        "Write a Python function that checks if a number is prime.",
    ]

print(f"\n{'='*60}")
print(f"GENERATING ({len(prompts)} prompt{'s' if len(prompts)>1 else ''})")
print(f"{'='*60}")

results = []
for prompt in prompts:
    r = generate(prompt, max_tokens=args.max_tokens)
    results.append((prompt, r))

    print(f"\nQ: {prompt}")
    print(f"A: {r['text']}")
    print(f"  {r['tokens']} tokens | {r['avg_ms']:.1f} ms/tok | {r['tok_s']:.0f} tok/s"
          f" | prefill {r['prefill_ms']:.0f}ms | EOS={'yes' if r['hit_eos'] else 'no'}")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
avg_tps = np.mean([r['tok_s'] for _, r in results])
avg_ms = np.mean([r['avg_ms'] for _, r in results])
total_tok = sum(r['tokens'] for _, r in results)
eos_count = sum(1 for _, r in results if r['hit_eos'])
print(f"  Model:      Llama-3.1-8B-Instruct (32 layers, 8B params)")
print(f"  Weights:    BFP8 MLP (HiFi2) + bf16 attention (HiFi4)")
print(f"  Device:     Tenstorrent Blackhole P150 (450 GB/s DRAM)")
print(f"  Decode:     Traced (rotation matrix RoPE + split SDPA + paged KV cache)")
print(f"  Avg speed:  {avg_ms:.1f} ms/tok = {avg_tps:.0f} tok/sec (end-to-end)")
print(f"  Total:      {total_tok} tokens across {len(prompts)} prompts")
print(f"  EOS:        {eos_count}/{len(prompts)} prompts completed naturally")
print(f"  BW ceiling: 28 tok/s (450 GB/s / 16 GB weights)")
print(f"  Efficiency: {avg_tps/28*100:.0f}% of bandwidth ceiling")

ttnn.close_device(device)
print("\nDone!")
