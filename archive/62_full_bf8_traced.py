#!/usr/bin/env python3
"""
Experiment 62: Full bf8 weights in traced decode with native RoPE.

From exp 61: ALL 24 layers safe for bf8 (min cosine 0.999644, 100% top-1 match).
From exp 60: native RoPE gives 7.1ms/tok = 140 tok/sec.
From exp 57c: bf8 MLP only gives 7.4ms/tok = 134 tok/sec.

Hypothesis: Full bf8 (ALL weights including attention) + native RoPE
gives additional speedup from halving attention weight bandwidth.
Current: bf8 MLP + bf16 attention = 7.1ms
Expected: bf8 ALL + native RoPE = ~6.8-7.0ms (bandwidth-bound ops benefit)

Generate 100 tokens to verify no quality degradation with full bf8.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=100)
args = parser.parse_args()

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print("Device: Blackhole P150")

# Load
print("Loading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

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

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))
def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# Upload: ALL bf8 weights (except biases/norms/lm_head which stay bf16)
print("Uploading weights (ALL bf8 + bf16 biases/norms/lm_head)...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf8(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf8(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf8(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf8(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf8(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf8(lw["mlp.down_proj.weight"].T),
    })
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# KV caches
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# Buffers
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)

# Prefill
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)
    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_np = apply_rope_half_np(from_dev(q,(B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_np = apply_rope_half_np(from_dev(k,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_np = from_dev(v,(B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np), is_causal=True, compute_kernel_config=hifi4)
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

# Decode with native RoPE
def decode_forward():
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"]),
                         [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"]),
                         [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"]),
                         [1, n_kv_heads, 1, head_dim])

        # Native RoPE
        qr = ttnn.experimental.rotary_embedding(q, rope_cos_buf, rope_sin_buf)
        kr = ttnn.experimental.rotary_embedding(k, rope_cos_buf, rope_sin_buf)

        # Handle seq padding
        qr_shape = list(qr.shape)
        if qr_shape[2] > 1:
            qr = ttnn.slice(qr, [0, 0, 0, 0], [1, n_q_heads, 1, head_dim])
        kr_shape = list(kr.shape)
        if kr_shape[2] > 1:
            kr = ttnn.slice(kr, [0, 0, 0, 0], [1, n_kv_heads, 1, head_dim])

        ks = ttnn.to_memory_config(ttnn.reshape(kr, [1,1,n_kv_heads,head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_buf)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr,[1,1,n_q_heads,head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn,[1,1,1,hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)

        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)

# Run
tokens_list = list(tokenizer.encode(args.prompt))
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))
print(f'Prompt: "{args.prompt}" ({len(tokens_list)} tokens), generating {max_gen}')

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
print(f"Prefill: {(time.perf_counter()-t0)*1000:.0f}ms")
next_id = int(np.argmax(logits))
tokens_list.append(next_id)

# Warmup + trace
update_buffers(next_id, len(tokens_list)-1)
_ = decode_forward(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

update_buffers(next_id, len(tokens_list)-1)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
print("Trace captured")

# Generate
trace_times = []
for step in range(max_gen - 1):
    update_buffers(next_id, len(tokens_list)-1)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    trace_times.append(time.perf_counter() - t0)
    logits = from_dev(logits_ref, (1,1,vocab_size))[0,0]
    next_id = int(np.argmax(logits))
    tokens_list.append(next_id)
    if next_id == tokenizer.eos_token_id:
        break

# Results
text = tokenizer.decode(tokens_list)
sustained = trace_times[1:] if len(trace_times) > 1 else trace_times
avg = np.mean(sustained) * 1000

print(f"\n{'='*60}")
print(f"RESULTS: Full bf8 + native RoPE (traced decode)")
print(f"{'='*60}")
print(f"  Trace exec: {avg:.1f}ms/tok ({1000/avg:.1f} tok/sec)")
print(f"  Tokens generated: {len(trace_times)}")
print(f"  Text: {text}")
print(f"\n  Comparison:")
print(f"    53e  all-bf16+rotmat:  7.6ms/tok (132 tok/sec)")
print(f"    57c  bf8-MLP+rotmat:   7.4ms/tok (134 tok/sec)")
print(f"    60   bf8-MLP+native:   7.1ms/tok (140 tok/sec)")
print(f"    62   all-bf8+native:   {avg:.1f}ms/tok ({1000/avg:.0f} tok/sec)")

ttnn.close_device(device)
print("\nDone!")
