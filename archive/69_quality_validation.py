#!/usr/bin/env python3
"""
Experiment 69: Quality validation — correctness + sampling.

Two goals:
  1. Validate cosine similarity of TT-NN vs numpy float32 reference
  2. Add temperature + top-k sampling for coherent text generation

Uses Qwen2.5-0.5B (our most validated model) and Llama-3.2-1B-Instruct
(instruction-tuned for coherent responses).
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
parser.add_argument("--tokens", type=int, default=100)
args = parser.parse_args()

# Qwen2.5-0.5B architecture
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

# ── Load model ──
print("Loading Qwen2.5-0.5B-Instruct...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model_path = hf_hub_download(model_id, "model.safetensors")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)
del all_weights


# ══════════════════════════════════════════════════════════════
# PART 1: Numpy float32 reference forward pass
# ══════════════════════════════════════════════════════════════

def numpy_rms_norm(x, g, eps):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * g

def numpy_silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def numpy_forward(token_ids, layer_weights):
    """Pure numpy float32 forward pass — the ground truth."""
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)

    freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

    for i in range(n_layers):
        lw = layer_weights[i]
        h = numpy_rms_norm(x, lw["input_layernorm.weight"], rms_eps)

        q = h.reshape(B*T, hidden) @ lw["self_attn.q_proj.weight"].T + lw["self_attn.q_proj.bias"]
        k = h.reshape(B*T, hidden) @ lw["self_attn.k_proj.weight"].T + lw["self_attn.k_proj.bias"]
        v = h.reshape(B*T, hidden) @ lw["self_attn.v_proj.weight"].T + lw["self_attn.v_proj.bias"]

        q = q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE (half format)
        q_rot = np.concatenate([-q[..., half_dim:], q[..., :half_dim]], axis=-1)
        q = q * cos_t[None, None] + q_rot * sin_t[None, None]
        k_rot = np.concatenate([-k[..., half_dim:], k[..., :half_dim]], axis=-1)
        k = k * cos_t[None, None] + k_rot * sin_t[None, None]

        # GQA: expand KV heads
        gqa_ratio = n_q_heads // n_kv_heads
        k_exp = np.repeat(k, gqa_ratio, axis=1)
        v_exp = np.repeat(v, gqa_ratio, axis=1)

        # Attention
        scale = 1.0 / np.sqrt(head_dim)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * scale
        # Causal mask
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
        scores = scores + mask
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-9)
        attn_out = (attn_weights @ v_exp).transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o = attn_out.reshape(B*T, hidden) @ lw["self_attn.o_proj.weight"].T
        x2 = x + o.reshape(B, T, hidden)

        h2 = numpy_rms_norm(x2, lw["post_attention_layernorm.weight"], rms_eps)
        gate = h2.reshape(B*T, hidden) @ lw["mlp.gate_proj.weight"].T
        up = h2.reshape(B*T, hidden) @ lw["mlp.up_proj.weight"].T
        down = (numpy_silu(gate) * up) @ lw["mlp.down_proj.weight"].T
        x = x2 + down.reshape(B, T, hidden)

    x_final = numpy_rms_norm(x, final_norm_g, rms_eps)
    logits = x_final.reshape(B*T, hidden) @ lm_head_w
    return logits[-1]


# ══════════════════════════════════════════════════════════════
# PART 2: TT-NN forward pass (prefill for comparison)
# ══════════════════════════════════════════════════════════════

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

# Upload weights
print("Uploading weights...")
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf16(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf16(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf16(lw["mlp.down_proj.weight"].T),
    }
    dev_layers.append(dl)
final_norm_g_tt = to_bf16(final_norm_g)
lm_head_w_tt = to_bf16(lm_head_w)
print("  Done")

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
R_half = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_half[i, i + half_dim] = -1.0
    R_half[i + half_dim, i] = 1.0
R_tt = to_bf16(R_half)

# KV caches
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((1, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
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

    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q, (B*T, n_q_heads*head_dim)).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k_np = from_dev(k, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_np = from_dev(v, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q_rot = np.concatenate([-q_np[..., half_dim:], q_np[..., :half_dim]], axis=-1)
        q_np = q_np * cos_t[None, None] + q_rot * sin_t[None, None]
        k_rot = np.concatenate([-k_np[..., half_dim:], k_np[..., :half_dim]], axis=-1)
        k_np = k_np * cos_t[None, None] + k_rot * sin_t[None, None]

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)

        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o = ttnn.matmul(to_bf16(a_np.reshape(B*T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T, hidden)).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T, hidden)), weight=final_norm_g_tt, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


def decode_forward():
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q = ttnn.reshape(q, [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(k, [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(v, [1, n_kv_heads, 1, head_dim])

        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))

        ks = ttnn.to_memory_config(ttnn.reshape(kr, [1,1,n_kv_heads,head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1,1,n_kv_heads,head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_buf)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr, [1,1,n_q_heads,head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)

        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)

    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g_tt, epsilon=rms_eps), lm_head_w_tt, compute_kernel_config=hifi4)

final_g_tt = final_norm_g_tt


# ══════════════════════════════════════════════════════════════
# PART 3: Sampling functions
# ══════════════════════════════════════════════════════════════

def sample_top_k(logits, temperature=0.7, top_k=50):
    """Temperature + top-k sampling."""
    logits = logits / temperature
    # Top-k: keep only top_k logits
    indices = np.argsort(logits)[-top_k:]
    mask = np.full_like(logits, -float('inf'))
    mask[indices] = logits[indices]
    # Softmax
    probs = np.exp(mask - np.max(mask))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ══════════════════════════════════════════════════════════════
# Run validation
# ══════════════════════════════════════════════════════════════

# Format as chat for instruct model (manual template to avoid jinja2 version issue)
chat_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer.encode(chat_text)
print(f"Chat prompt ({len(tokens)} tokens): {chat_text[:100]}...")

# 1. Numpy reference
print("\n--- Numpy float32 reference ---")
t0 = time.perf_counter()
ref_logits = numpy_forward(np.array(tokens), layer_weights_np)
dt_np = time.perf_counter() - t0
ref_top5 = np.argsort(ref_logits)[-5:][::-1]
print(f"  Time: {dt_np*1000:.0f}ms")
print(f"  Top-5 tokens: {[tokenizer.decode([t]) for t in ref_top5]}")

# 2. TT-NN prefill
print("\n--- TT-NN bf16 prefill ---")
t0 = time.perf_counter()
tt_logits = ttnn_prefill(np.array(tokens))
dt_tt = time.perf_counter() - t0
tt_top5 = np.argsort(tt_logits)[-5:][::-1]
print(f"  Time: {dt_tt*1000:.0f}ms")
print(f"  Top-5 tokens: {[tokenizer.decode([t]) for t in tt_top5]}")

# 3. Compare
cosine = np.dot(ref_logits, tt_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(tt_logits) + 1e-9)
top1_match = ref_top5[0] == tt_top5[0]
top5_match = sum(1 for t in tt_top5 if t in ref_top5)

print(f"\n--- Correctness ---")
print(f"  Cosine similarity: {cosine:.6f}")
print(f"  Top-1 match: {top1_match} (numpy={tokenizer.decode([ref_top5[0]])}, ttnn={tokenizer.decode([tt_top5[0]])})")
print(f"  Top-5 overlap: {top5_match}/5")

if cosine < 0.99:
    print(f"  ⚠️  WARNING: Cosine below 0.99 — precision issue!")
else:
    print(f"  ✓ Cosine > 0.99 — precision validated")


# 4. Generate with sampling (instruction-tuned)
print(f"\n--- Generation with sampling (temp=0.7, top_k=50) ---")

# Prefill already done, first token from greedy
next_id = int(np.argmax(tt_logits))
tokens_list = list(tokens) + [next_id]
pos = len(tokens_list) - 1

# Warmup + trace
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

times = []
for step in range(args.tokens - 1):
    update_buffers(next_id, pos)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    times.append(dt)

    logits = from_dev(logits_ref, (1, vocab_size))[0]
    next_id = sample_top_k(logits, temperature=0.7, top_k=50)
    tokens_list.append(next_id)
    pos += 1
    if next_id == tokenizer.eos_token_id:
        break

# Also generate with greedy for comparison
greedy_tokens = list(tokens)
next_id_g = int(np.argmax(tt_logits))
greedy_tokens.append(next_id_g)
pos_g = len(greedy_tokens) - 1

for step in range(min(50, args.tokens - 1)):
    update_buffers(next_id_g, pos_g)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    logits_g = from_dev(logits_ref, (1, vocab_size))[0]
    next_id_g = int(np.argmax(logits_g))
    greedy_tokens.append(next_id_g)
    pos_g += 1
    if next_id_g == tokenizer.eos_token_id:
        break


text_sampled = tokenizer.decode(tokens_list, skip_special_tokens=True)
text_greedy = tokenizer.decode(greedy_tokens, skip_special_tokens=True)

sustained = times[1:] if len(times) > 1 else times
avg_ms = np.mean(sustained) * 1000

print(f"\n{'='*60}")
print(f"RESULTS: Qwen2.5-0.5B-Instruct Quality Validation")
print(f"{'='*60}")
print(f"\n  Correctness:")
print(f"    Cosine (numpy vs TT-NN): {cosine:.6f}")
print(f"    Top-1 token match: {top1_match}")
print(f"    Top-5 overlap: {top5_match}/5")
print(f"\n  Decode speed: {avg_ms:.1f}ms/tok ({1000/avg_ms:.0f} tok/sec)")
print(f"\n  GREEDY output ({len(greedy_tokens)-len(tokens)} tokens):")
# Extract just the assistant response
greedy_response = text_greedy.split("assistant\n")[-1] if "assistant" in text_greedy else text_greedy
print(f"    {greedy_response[:300]}")
print(f"\n  SAMPLED output ({len(tokens_list)-len(tokens)} tokens, temp=0.7, top_k=50):")
sampled_response = text_sampled.split("assistant\n")[-1] if "assistant" in text_sampled else text_sampled
print(f"    {sampled_response[:300]}")

ttnn.close_device(device)
print("\nDone!")
