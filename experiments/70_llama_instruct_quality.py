#!/usr/bin/env python3
"""
Experiment 70: Llama-3.2-1B-Instruct quality validation.

Tests whether a larger instruction-tuned model produces coherent text.
Exp 69 showed Qwen2.5-0.5B-Instruct has correct precision (cosine 0.999)
but poor text quality — likely because 0.5B is too small.

This experiment:
  1. Validates cosine similarity against numpy float32 reference
  2. Tests greedy + sampled generation quality
  3. Uses proper Llama-3 chat template for instruction following
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

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
parser.add_argument("--tokens", type=int, default=150)
args = parser.parse_args()

# Llama-3.2-1B architecture
hidden = 2048; n_q_heads = 32; n_kv_heads = 8; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 16; vocab_size = 128256; MAX_SEQ = 512
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()

# ── Load Llama-3.2-1B-Instruct ──
print("Loading Llama-3.2-1B-Instruct...")
model_ids = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
]

model_path = None
model_id = None
for mid in model_ids:
    try:
        model_path = hf_hub_download(mid, "model.safetensors")
        model_id = mid
        print(f"  Loaded from {mid}")
        break
    except Exception as e:
        print(f"  {mid}: {str(e)[:80]}")

if model_path is None:
    print("ERROR: Could not load model.")
    ttnn.close_device(device)
    sys.exit(1)

all_weights = {}
with safe_open(model_path, framework="pt") as f:
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
del all_weights

# Tokenizer
tok_path = hf_hub_download(model_id, "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

# ══════════════════════════════════════════════════════════════
# PART 1: Numpy float32 reference
# ══════════════════════════════════════════════════════════════

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def numpy_rms_norm(x, g, eps):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * g

def numpy_silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def rotate_interleaved_np(x):
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def numpy_forward(token_ids, layer_weights):
    """Pure numpy float32 forward pass — ground truth."""
    B, T = 1, len(token_ids)
    x = embed_w[token_ids].reshape(B, T, hidden)

    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.repeat(np.cos(angles), 2, axis=-1)  # interleaved
    sin_t = np.repeat(np.sin(angles), 2, axis=-1)

    for i in range(n_layers):
        lw = layer_weights[i]
        h = numpy_rms_norm(x, lw["input_layernorm.weight"], rms_eps)

        q = h.reshape(B*T, hidden) @ lw["self_attn.q_proj.weight"].T
        k = h.reshape(B*T, hidden) @ lw["self_attn.k_proj.weight"].T
        v = h.reshape(B*T, hidden) @ lw["self_attn.v_proj.weight"].T

        q = q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # RoPE (interleaved)
        q = q * cos_t[None, None] + rotate_interleaved_np(q) * sin_t[None, None]
        k = k * cos_t[None, None] + rotate_interleaved_np(k) * sin_t[None, None]

        # GQA: expand KV heads
        gqa_ratio = n_q_heads // n_kv_heads
        k_exp = np.repeat(k, gqa_ratio, axis=1)
        v_exp = np.repeat(v, gqa_ratio, axis=1)

        # Attention
        scale = 1.0 / np.sqrt(head_dim)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * scale
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
# PART 2: TT-NN forward pass
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
dt_upload = time.perf_counter() - t0
print(f"  Uploaded in {dt_upload*1000:.0f}ms")

# RoPE rotation matrix (interleaved format)
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0
    R_interleaved[2*i, 2*i+1] = 1.0
R_tt = to_bf16(R_interleaved)

# KV caches — split into 2 groups of 4 KV heads (sdpa_flash_decode power-of-2 bug)
n_kv_split = n_kv_heads // 2  # 4 heads per group
k_caches_lo, v_caches_lo = [], []
k_caches_hi, v_caches_hi = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches_lo.append(to_dev_4d(c.copy()))
    v_caches_lo.append(to_dev_4d(c.copy()))
    k_caches_hi.append(to_dev_4d(c.copy()))
    v_caches_hi.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_split + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# Buffers
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)


def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1),
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]


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

        q_np = apply_rope_interleaved_np(
            from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3),
            cos_t, sin_t)
        k_np = apply_rope_interleaved_np(
            from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3),
            cos_t, sin_t)
        v_np = from_dev(v, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

        # Split KV caches
        ttnn.kv_cache.fill_cache_for_user_(k_caches_lo[i], to_dev_4d(k_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_lo[i], to_dev_4d(v_np[:, :n_kv_split]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(k_caches_hi[i], to_dev_4d(k_np[:, n_kv_split:]), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches_hi[i], to_dev_4d(v_np[:, n_kv_split:]), batch_index=0)

        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)

        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    cos_full = np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
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

        # Split KV into 2 groups of 4 heads
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

        n_q_split = n_q_heads // 2
        qr_4d = ttnn.reshape(qr, [1, 1, n_q_heads, head_dim])
        q_lo = ttnn.slice(qr_4d, [0,0,0,0], [1,1,n_q_split,head_dim])
        q_hi = ttnn.slice(qr_4d, [0,0,n_q_split,0], [1,1,n_q_heads,head_dim])
        attn_lo = ttnn.transformer.scaled_dot_product_attention_decode(
            q_lo, k_caches_lo[i], v_caches_lo[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn_hi = ttnn.transformer.scaled_dot_product_attention_decode(
            q_hi, k_caches_hi[i], v_caches_hi[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        attn = ttnn.concat([attn_lo, attn_hi], dim=2)
        o = ttnn.matmul(ttnn.reshape(attn, [1,1,1,hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, o)

        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x = ttnn.add(x, d)

    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════

def sample_top_k(logits, temperature=0.6, top_k=50):
    logits = logits / temperature
    indices = np.argsort(logits)[-top_k:]
    mask = np.full_like(logits, -float('inf'))
    mask[indices] = logits[indices]
    probs = np.exp(mask - np.max(mask))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════

# Llama-3 Instruct chat template
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
# <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>\n\n

# Find special token IDs
special_tokens = {}
for tok_name in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
                  "<|eot_id|>", "<|end_of_text|>"]:
    tid = tokenizer.convert_tokens_to_ids(tok_name)
    if tid is not None:
        special_tokens[tok_name] = tid
        print(f"  {tok_name} = {tid}")

# Build chat tokens manually
bos = special_tokens.get("<|begin_of_text|>", 128000)
start_header = special_tokens.get("<|start_header_id|>", 128006)
end_header = special_tokens.get("<|end_header_id|>", 128007)
eot = special_tokens.get("<|eot_id|>", 128009)

system_msg = "You are a helpful assistant."
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
system_tokens = [bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system_msg) + [eot]
user_tokens = [start_header] + enc("user") + [end_header] + enc("\n\n" + args.prompt) + [eot]
assistant_tokens = [start_header] + enc("assistant") + [end_header] + enc("\n\n")

tokens = system_tokens + user_tokens + assistant_tokens
print(f"\nChat prompt ({len(tokens)} tokens)")
print(f"  Decoded: {tokenizer.decode(tokens)[:200]}...")

# Stop tokens for Llama-3
stop_ids = {eot, special_tokens.get("<|end_of_text|>", 128001)}

# 1. Numpy reference
print("\n--- Numpy float32 reference ---")
t0 = time.perf_counter()
ref_logits = numpy_forward(np.array(tokens), layer_weights_np)
dt_np = time.perf_counter() - t0
ref_top5 = np.argsort(ref_logits)[-5:][::-1]
print(f"  Time: {dt_np*1000:.0f}ms")
print(f"  Top-5: {[tokenizer.decode([t]) for t in ref_top5]}")

# 2. TT-NN prefill
print("\n--- TT-NN bf16 prefill ---")
t0 = time.perf_counter()
tt_logits = ttnn_prefill(np.array(tokens))
dt_tt = time.perf_counter() - t0
tt_top5 = np.argsort(tt_logits)[-5:][::-1]
print(f"  Time: {dt_tt*1000:.0f}ms")
print(f"  Top-5: {[tokenizer.decode([t]) for t in tt_top5]}")

# 3. Compare
cosine = np.dot(ref_logits, tt_logits) / (np.linalg.norm(ref_logits) * np.linalg.norm(tt_logits) + 1e-9)
top1_match = ref_top5[0] == tt_top5[0]
top5_match = sum(1 for t in tt_top5 if t in ref_top5)

print(f"\n--- Correctness ---")
print(f"  Cosine similarity: {cosine:.6f}")
print(f"  Top-1 match: {top1_match} (numpy={tokenizer.decode([ref_top5[0]])}, ttnn={tokenizer.decode([tt_top5[0]])})")
print(f"  Top-5 overlap: {top5_match}/5")

if cosine < 0.99:
    print(f"  WARNING: Cosine below 0.99!")
else:
    print(f"  Cosine > 0.99 — precision validated")

# 4. Numpy reference greedy decode — first 20 tokens as ground truth
print(f"\n--- Numpy reference greedy decode (20 tokens) ---")
np_tokens = list(tokens)
np_logits = ref_logits  # already computed from prefill
for step in range(20):
    next_np = int(np.argmax(np_logits))
    np_tokens.append(next_np)
    # Full numpy forward for next-token prediction
    np_logits = numpy_forward(np.array(np_tokens), layer_weights_np)
    sys.stdout.write(tokenizer.decode([next_np]))
    sys.stdout.flush()
np_response = tokenizer.decode(np_tokens[len(tokens):], skip_special_tokens=True)
print(f"\n  Numpy greedy (20 tok): {np_response}")

# 5. Generate with GREEDY on TT-NN
print(f"\n--- TT-NN greedy generation ({args.tokens} tokens) ---")
next_id = int(np.argmax(tt_logits))
greedy_list = list(tokens) + [next_id]
pos = len(greedy_list) - 1

# Warmup + trace
update_buffers(next_id, pos)
_ = decode_forward(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

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

    logits = from_dev(logits_ref, (1,1,vocab_size))[0,0]
    next_id = int(np.argmax(logits))
    greedy_list.append(next_id)
    pos += 1
    if next_id in stop_ids:
        print(f"  Hit stop token at step {step+1}")
        break

greedy_text = tokenizer.decode(greedy_list[len(tokens):], skip_special_tokens=True)

# Compare TT-NN greedy vs numpy greedy token-by-token
n_compare = min(len(np_tokens) - len(tokens), len(greedy_list) - len(tokens))
match_count = 0
print(f"\n--- Token-by-token comparison (first {n_compare} tokens) ---")
for j in range(n_compare):
    np_tok = np_tokens[len(tokens) + j]
    tt_tok = greedy_list[len(tokens) + j]
    match = "OK" if np_tok == tt_tok else "MISMATCH"
    if np_tok != tt_tok:
        print(f"  [{j}] numpy={tokenizer.decode([np_tok])!r} ({np_tok}) vs ttnn={tokenizer.decode([tt_tok])!r} ({tt_tok}) — {match}")
    else:
        match_count += 1
print(f"  Matched: {match_count}/{n_compare} tokens")

# 5. Generate with SAMPLING (need fresh KV cache)
print(f"\n--- Sampled generation (temp=0.6, top_k=50, {args.tokens} tokens) ---")
print("  Re-prefilling with fresh KV cache...")

# Reset KV caches
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
    ttnn.copy(to_dev_4d(c), k_caches_lo[i])
    ttnn.copy(to_dev_4d(c), v_caches_lo[i])
    ttnn.copy(to_dev_4d(c), k_caches_hi[i])
    ttnn.copy(to_dev_4d(c), v_caches_hi[i])

tt_logits2 = ttnn_prefill(np.array(tokens))
next_id = sample_top_k(tt_logits2, temperature=0.6, top_k=50)
sampled_list = list(tokens) + [next_id]
pos = len(sampled_list) - 1

# Re-capture trace with fresh state
update_buffers(next_id, pos)
_ = decode_forward(); ttnn.synchronize_device(device)

update_buffers(next_id, pos)
trace_id2 = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref2 = decode_forward()
ttnn.end_trace_capture(device, trace_id2, cq_id=0)

for step in range(args.tokens - 1):
    update_buffers(next_id, pos)
    ttnn.execute_trace(device, trace_id2, cq_id=0, blocking=True)

    logits = from_dev(logits_ref2, (1,1,vocab_size))[0,0]
    next_id = sample_top_k(logits, temperature=0.6, top_k=50)
    sampled_list.append(next_id)
    pos += 1
    if next_id in stop_ids:
        print(f"  Hit stop token at step {step+1}")
        break

sampled_text = tokenizer.decode(sampled_list[len(tokens):], skip_special_tokens=True)

# Results
sustained = times[1:] if len(times) > 1 else times
avg_ms = np.mean(sustained) * 1000

print(f"\n{'='*60}")
print(f"RESULTS: Llama-3.2-1B-Instruct Quality Validation")
print(f"{'='*60}")
print(f"\n  Correctness:")
print(f"    Cosine (numpy vs TT-NN): {cosine:.6f}")
print(f"    Top-1 match: {top1_match}")
print(f"    Top-5 overlap: {top5_match}/5")
print(f"\n  Decode speed: {avg_ms:.1f}ms/tok ({1000/avg_ms:.0f} tok/sec)")
print(f"\n  GREEDY output ({len(greedy_list)-len(tokens)} tokens):")
print(f"    {greedy_text[:500]}")
print(f"\n  SAMPLED output ({len(sampled_list)-len(tokens)} tokens, temp=0.6, top_k=50):")
print(f"    {sampled_text[:500]}")

ttnn.close_device(device)
print("\nDone!")
