#!/usr/bin/env python3
"""
Experiment 68: SmolLM3-3B port to Blackhole.

Architecturally interesting: NoPE (No Positional Embedding) every 4th layer.
36 layers, 2048 hidden, 16Q/4KV heads, head_dim=128, intermediate=11008.
4 KV heads = power of 2, so NO split SDPA needed (unlike Llama/Qwen3).

NoPE pattern: [1,1,1,0,1,1,1,0,...] where 1=skip RoPE, 0=apply RoPE.
Only 9 of 36 layers get positional encoding. The rest are position-agnostic.

Hypothesis: NoPE layers are faster (no RoPE matmul) but same SDPA cost.
~25-35 tok/sec expected (similar param count to Llama-3.2-3B but more layers).
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="The capital of France is")
parser.add_argument("--tokens", type=int, default=100)
args = parser.parse_args()

# SmolLM3-3B architecture
hidden = 2048; n_q_heads = 16; n_kv_heads = 4; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 5000000.0
n_layers = 36; vocab_size = 128256; MAX_SEQ = 256
intermediate_size = 11008
TILE_SIZE = 32; batch_size = 1

# NoPE: 1=skip RoPE, 0=apply RoPE
no_rope_layers = [1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0]
rope_layers = [i for i in range(n_layers) if no_rope_layers[i] == 0]
nope_layers = [i for i in range(n_layers) if no_rope_layers[i] == 1]
print(f"RoPE layers ({len(rope_layers)}): {rope_layers}")
print(f"NoPE layers ({len(nope_layers)}): {nope_layers[:10]}...")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model ──
print("Loading SmolLM3-3B...")
model_id = "HuggingFaceTB/SmolLM3-3B"
shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
shard_paths = [hf_hub_download(model_id, s) for s in shard_names]

all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            all_weights[key] = f.get_tensor(key).float().numpy()

print(f"  Loaded {len(all_weights)} tensors")

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)

print(f"  Q weight: {layer_weights_np[0]['self_attn.q_proj.weight'].shape}")
print(f"  K weight: {layer_weights_np[0]['self_attn.k_proj.weight'].shape}")
print(f"  gate_proj: {layer_weights_np[0]['mlp.gate_proj.weight'].shape}")
del all_weights

# Tokenizer
from transformers import PreTrainedTokenizerFast
tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

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


# ── RoPE: SmolLM3 uses Llama-style (interleaved, based on Llama architecture) ──
# Actually, SmolLM3 uses the standard Llama RoPE format
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved_np(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1),
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]

# Rotation matrix for on-device RoPE
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[2*i+1, 2*i] = -1.0
    R[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R)


# ── Upload weights ──
print("Uploading weights to device...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
dt_upload = time.perf_counter() - t0
print(f"  Uploaded in {dt_upload*1000:.0f}ms")

# ── KV caches ──
# 4 KV heads = power of 2, so NO split needed!
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

# ── Buffers ──
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


# ── Prefill ──
def prefill(token_ids):
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

        q_np = from_dev(q, (B*T, n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3)
        k_np = from_dev(k, (B*T, n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        v_np = from_dev(v, (B*T, n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

        # Apply RoPE only for RoPE-enabled layers
        if no_rope_layers[i] == 0:
            q_np = apply_rope_np(q_np, cos_t, sin_t)
            k_np = apply_rope_np(k_np, cos_t, sin_t)

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)

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


# ── Traced decode ──
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

        # Apply RoPE only for RoPE-enabled layers (static in trace graph)
        if no_rope_layers[i] == 0:
            q = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
            k = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))

        ks = ttnn.to_memory_config(ttnn.reshape(k, [1,1,n_kv_heads,head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_buf)

        # SDPA decode — 4 KV heads works directly, no split needed!
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(q, [1,1,n_q_heads,head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)

        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,n_q_heads*head_dim]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)

        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)

    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
tokens_list = list(tokenizer.encode(args.prompt))
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))
print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens\n")

t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
dt_prefill = time.perf_counter() - t0
next_id = int(np.argmax(logits))
tokens_list.append(next_id)
pos = len(tokens_list) - 1
print(f"Prefill: {dt_prefill*1000:.0f}ms")
print(f"First token: {next_id} ({tokenizer.decode([next_id])})")

# Warmup
update_buffers(next_id, pos)
_ = decode_forward()
ttnn.synchronize_device(device)

try:
    device.enable_program_cache()
except:
    pass

update_buffers(next_id, pos)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
print("Trace captured")

times = []
for step in range(max_gen - 1):
    update_buffers(next_id, pos)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    times.append(dt)

    logits = from_dev(logits_ref, (1, vocab_size))
    next_id = int(np.argmax(logits[0]))
    tokens_list.append(next_id)
    pos += 1
    if next_id == tokenizer.eos_token_id:
        break

text = tokenizer.decode(tokens_list)
sustained = times[1:] if len(times) > 1 else times
avg_ms = np.mean(sustained) * 1000
tps = 1000 / avg_ms

print(f"\n{'='*60}")
print(f"RESULTS: SmolLM3-3B on Blackhole P150")
print(f"{'='*60}")
print(f"  Architecture: {n_layers} layers, {hidden} hidden, {n_q_heads}Q/{n_kv_heads}KV heads, head_dim={head_dim}")
print(f"  NoPE: {len(nope_layers)}/{n_layers} layers skip RoPE, {len(rope_layers)} apply RoPE")
print(f"  Parameters: ~3B")
print(f"  Upload: {dt_upload*1000:.0f}ms")
print(f"  Prefill: {dt_prefill*1000:.0f}ms")
print(f"  Traced decode: {avg_ms:.1f}ms/tok ({tps:.1f} tok/sec)")
print(f"  Tokens generated: {len(tokens_list) - len(tokenizer.encode(args.prompt))}")
print(f"  Text: {text}")
print(f"\n  Comparison:")
print(f"    Qwen2.5-0.5B:   7.1ms/tok (140 tok/sec) — head_dim=64, 2 KV heads")
print(f"    Qwen3-0.6B:    13.2ms/tok  (76 tok/sec) — head_dim=128, 8 KV (split)")
print(f"    Llama-3.2-1B:  12.8ms/tok  (78 tok/sec) — head_dim=64, 8 KV (split)")
print(f"    Llama-3.2-3B:  29.7ms/tok  (34 tok/sec) — head_dim=128, 8 KV (split)")
print(f"    SmolLM3-3B:    {avg_ms:.1f}ms/tok ({tps:.0f} tok/sec) — head_dim=128, 4 KV (no split!)")

ttnn.close_device(device)
print("\nDone!")
