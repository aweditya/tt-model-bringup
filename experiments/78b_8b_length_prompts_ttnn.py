#!/usr/bin/env python3
"""
Experiment 78b: Length-prompted generation via TT-NN (fast version of exp 78).

Exp 77 proved TT-NN produces coherent greedy text (same quality as numpy float32).
Exp 78 (numpy-only) was too slow. This runs the same length prompts on TT-NN
at 19 tok/sec — instant results.

Question: Do explicit length instructions ("3 paragraphs, 3 sentences each")
get the 8B model to produce longer coherent output?
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
import ttnn

np.random.seed(42)

hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 32; vocab_size = 128256; MAX_SEQ = 512
intermediate_size = 14336; TILE_SIZE = 32; batch_size = 1
n_kv_split = n_kv_heads // 2; n_q_split = n_q_heads // 2

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150")

# Load model
print("Loading Llama-3.1-8B-Instruct...")
shard_paths = [hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct",
               f"model-{i+1:05d}-of-00004.safetensors") for i in range(4)]
all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()
print(f"  Loaded {len(all_weights)} tensors")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

tok_path = hf_hub_download("unsloth/Meta-Llama-3.1-8B-Instruct", "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# Device helpers
def to_bf16(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved(x):
    r = np.zeros_like(x); r[..., 0::2] = -x[..., 1::2]; r[..., 1::2] = x[..., 0::2]; return r

def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return np.repeat(np.cos(angles), 2, axis=-1), np.repeat(np.sin(angles), 2, axis=-1)

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved(x_4d) * sin_t[None, None]

# Upload weights
print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    })
    if (i+1) % 8 == 0: print(f"    Layer {i+1}/{n_layers}")
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del all_weights
print(f"  Uploaded in {time.perf_counter()-t0:.0f}s")

R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0; R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

k_caches_lo, v_caches_lo, k_caches_hi, v_caches_hi = [], [], [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches_lo.append(to_dev_4d(c.copy())); v_caches_lo.append(to_dev_4d(c.copy()))
    k_caches_hi.append(to_dev_4d(c.copy())); v_caches_hi.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_split + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def ttnn_prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_interleaved(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)
        q_np = apply_rope_interleaved_np(from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_interleaved_np(from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_lo[i], to_dev_4d(k_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_lo[i], to_dev_4d(v_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_hi[i], to_dev_4d(k_np[:, n_kv_split:]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_hi[i], to_dev_4d(v_np[:, n_kv_split:]), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,n_q_heads*head_dim)
        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,n_q_heads*head_dim)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)
    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]

def update_buffers(token_id, pos):
    ttnn.copy(to_bf16(embed_w[token_id:token_id+1].reshape(1, 1, hidden)), embed_buf)
    angles = pos * freqs
    ttnn.copy(to_dev_4d(np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_cos_buf)
    ttnn.copy(to_dev_4d(np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)

def decode_forward():
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
        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))
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
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)

# Chat template
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}

def make_chat_tokens(prompt, system="You are a helpful assistant."):
    return ([bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system) + [eot] +
            [start_header] + enc("user") + [end_header] + enc("\n\n" + prompt) + [eot] +
            [start_header] + enc("assistant") + [end_header] + enc("\n\n"))

def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches_lo[i]); ttnn.copy(to_dev_4d(c), v_caches_lo[i])
        ttnn.copy(to_dev_4d(c), k_caches_hi[i]); ttnn.copy(to_dev_4d(c), v_caches_hi[i])

def generate(prompt, max_tokens=300):
    reset_kv_caches()
    tokens = make_chat_tokens(prompt)
    logits = ttnn_prefill(np.array(tokens))
    next_id = int(np.argmax(logits))
    gen = [next_id]
    pos = len(tokens)

    update_buffers(next_id, pos)
    _ = decode_forward(); ttnn.synchronize_device(device)
    try: device.enable_program_cache()
    except: pass

    update_buffers(next_id, pos)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref = decode_forward()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    for step in range(max_tokens - 1):
        update_buffers(next_id, pos)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        logits = from_dev(logits_ref, (1, vocab_size))[0]
        next_id = int(np.argmax(logits))
        gen.append(next_id)
        pos += 1
        if next_id in stop_ids:
            break

    ttnn.release_trace(device, trace_id)
    return gen

# Test prompts
prompts = [
    ("Original short", "Write a short story about a robot that learns to paint."),
    ("Explicit 3-para", "Write a 3-paragraph story about a robot that learns to paint. Each paragraph must be at least 3 sentences long. Include dialogue between the robot and its creator."),
    ("Explicit 5-item list", "List exactly 5 benefits of exercise. For each benefit, write exactly 2 sentences explaining it."),
    ("Explicit 3-para factual", "Explain quantum computing in exactly 3 paragraphs. Paragraph 1: what it is. Paragraph 2: how qubits work. Paragraph 3: why it matters."),
    ("Detailed instruction", "You are writing a technical blog post. Write about how neural networks learn, covering: (1) forward propagation, (2) loss calculation, (3) backpropagation, and (4) gradient descent. Use clear examples."),
]

for name, prompt in prompts:
    print(f"\n{'='*70}")
    print(f"[{name}]")
    print(f"Prompt: {prompt[:70]}{'...' if len(prompt)>70 else ''}")
    print(f"{'='*70}")

    gen = generate(prompt, max_tokens=400)
    text = tokenizer.decode(gen, skip_special_tokens=True)
    words = text.split()
    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    print(f"  Tokens: {len(gen)}, Words: {len(words)}, Paragraphs: {len(paragraphs)}")
    print(f"  Hit EOS: {gen[-1] in stop_ids}")
    print(f"  ---")
    for line in text.split("\n")[:30]:
        print(f"  {line}")
    if len(text.split("\n")) > 30:
        print(f"  ... ({len(text.split(chr(10)))} lines total)")
    print(f"  ---")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

ttnn.close_device(device)
print("\nDone!")
