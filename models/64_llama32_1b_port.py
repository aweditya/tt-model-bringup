#!/usr/bin/env python3
"""
Experiment 64: Llama-3.2-1B port to Blackhole.

Proves our infrastructure generalizes beyond Qwen. Key differences:
  - 16 layers (vs 24), 2048 hidden (vs 896), 32 Q heads (vs 14), 8 KV heads (vs 2)
  - intermediate_size = 8192 (vs 4864)
  - NO biases on Q/K/V/O projections (Qwen has biases)
  - Interleaved RoPE format (vs Qwen's half format)
  - rope_theta = 500,000 (vs 1,000,000)
  - vocab_size = 128,256 (vs 151,936)
  - Same ops: matmul, SDPA, RMSNorm, SiLU, RoPE — zero new primitives

Architecture: fewer, wider layers (2048→8192 FFN expansion).
GQA ratio: 4:1 (32 Q heads / 8 KV heads) vs Qwen's 7:1.

This experiment:
  1. Loads weights from HuggingFace (tries unsloth mirror if auth required)
  2. Full recompute forward for correctness verification
  3. Traced decode for speed measurement
  4. Generate 100 tokens to verify quality
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

# Prevent transformers from importing torchvision (crashes on remote host)
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

# Llama-3.2-1B architecture
hidden = 2048; n_q_heads = 32; n_kv_heads = 8; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 16; vocab_size = 128256; MAX_SEQ = 256
intermediate_size = 8192
TILE_SIZE = 32; batch_size = 1

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model ──
print("Loading Llama-3.2-1B...")
# Try multiple sources
model_ids = [
    "meta-llama/Llama-3.2-1B",
    "unsloth/Llama-3.2-1B",
]

model_path = None
for model_id in model_ids:
    try:
        model_path = hf_hub_download(model_id, "model.safetensors")
        print(f"  Loaded from {model_id}")
        break
    except Exception as e:
        print(f"  {model_id}: {str(e)[:80]}")

if model_path is None:
    print("ERROR: Could not load model. Try setting HF_TOKEN environment variable.")
    ttnn.close_device(device)
    sys.exit(1)

all_weights = {}
with safe_open(model_path, framework="pt") as f:
    for key in f.keys():
        all_weights[key] = f.get_tensor(key).float().numpy()

print(f"  Loaded {len(all_weights)} tensors")
# Print some weight names to verify structure
for k in sorted(all_weights.keys())[:10]:
    print(f"    {k}: {all_weights[k].shape}")

# Extract weights
embed_w = all_weights["model.embed_tokens.weight"]
final_norm_g = all_weights["model.norm.weight"]
# Llama often ties embeddings with lm_head
lm_head_w = all_weights.get("lm_head.weight", embed_w).T.copy()

print(f"  embed_w: {embed_w.shape}")
print(f"  lm_head_w: {lm_head_w.shape}")
print(f"  vocab_size: {embed_w.shape[0]}, hidden: {embed_w.shape[1]}")

# Verify architecture matches
assert embed_w.shape[1] == hidden, f"Hidden mismatch: {embed_w.shape[1]} vs {hidden}"

layer_weights_np = []
for i in range(n_layers):
    prefix = f"model.layers.{i}."
    lw = {k[len(prefix):]: v for k, v in all_weights.items() if k.startswith(prefix)}
    layer_weights_np.append(lw)

# Verify: Llama has NO biases on Q/K/V/O
has_bias = "self_attn.q_proj.bias" in layer_weights_np[0]
print(f"  Has attention biases: {has_bias}")
print(f"  Q weight shape: {layer_weights_np[0]['self_attn.q_proj.weight'].shape}")
print(f"  K weight shape: {layer_weights_np[0]['self_attn.k_proj.weight'].shape}")
print(f"  gate_proj shape: {layer_weights_np[0]['mlp.gate_proj.weight'].shape}")

del all_weights

# Tokenizer — load directly to avoid AutoTokenizer -> LlamaConfig -> torchvision chain
from transformers import PreTrainedTokenizerFast
tokenizer_id = model_id
tok_path = hf_hub_download(tokenizer_id, "tokenizer.json")
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


# ── RoPE: Llama uses INTERLEAVED format ──
# Interleaved: pairs (x[0],x[1]), (x[2],x[3]), ... get rotated together
# cos/sin: [c0,c0,c1,c1,...] via np.repeat
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved_np(x):
    """Llama's interleaved rotation: swap adjacent pairs with negation."""
    result = np.zeros_like(x)
    result[..., 0::2] = -x[..., 1::2]
    result[..., 1::2] = x[..., 0::2]
    return result

def get_rope_tables_interleaved(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.repeat(np.cos(angles), 2, axis=-1),  # [c0,c0,c1,c1,...]
            np.repeat(np.sin(angles), 2, axis=-1))

def apply_rope_interleaved_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]

# Rotation matrix for interleaved format (on-device decode)
R_interleaved = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_interleaved[2*i+1, 2*i] = -1.0    # result[2i] = -x[2i+1]
    R_interleaved[2*i, 2*i+1] = 1.0     # result[2i+1] = x[2i]
R_tt = to_bf16(R_interleaved)


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
    # Add biases only if present
    if has_bias:
        dl["q_b"] = to_bf16(lw["self_attn.q_proj.bias"])
        dl["k_b"] = to_bf16(lw["self_attn.k_proj.bias"])
        dl["v_b"] = to_bf16(lw["self_attn.v_proj.bias"])
    dev_layers.append(dl)

final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
dt_upload = time.perf_counter() - t0
print(f"  Uploaded in {dt_upload*1000:.0f}ms")

# ── KV caches ──
# Split into 2 groups of 4 KV heads each (sdpa_flash_decode only works with power-of-2 KV heads)
n_kv_split = n_kv_heads // 2  # 4 heads per group
k_caches_lo, v_caches_lo = [], []  # first 4 KV heads
k_caches_hi, v_caches_hi = [], []  # last 4 KV heads
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

# ── Buffers ──
embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

def update_buffers(token_id, pos):
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    angles = pos * freqs
    # Interleaved cos/sin
    cos_full = np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    sin_full = np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


# ── Prefill (CPU RoPE) ──
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_interleaved(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        # Q/K/V — no biases for Llama
        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)

        if has_bias:
            q = ttnn.add(q, dl["q_b"])
            k = ttnn.add(k, dl["k_b"])
            v = ttnn.add(v, dl["v_b"])

        q_np = apply_rope_interleaved_np(
            from_dev(q, (B,T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3),
            cos_t, sin_t)
        k_np = apply_rope_interleaved_np(
            from_dev(k, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3),
            cos_t, sin_t)
        v_np = from_dev(v, (B,T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

        # Split KV into two groups of 4 heads each
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


# ── Traced decode ──
def decode_forward():
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)

        q = ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4)
        k = ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4)
        v = ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4)

        if has_bias:
            q = ttnn.add(q, dl["q_b"])
            k = ttnn.add(k, dl["k_b"])
            v = ttnn.add(v, dl["v_b"])

        q = ttnn.reshape(q, [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(k, [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(v, [1, n_kv_heads, 1, head_dim])

        # On-device interleaved RoPE via rotation matrix
        qr = ttnn.add(ttnn.mul(q, rope_cos_buf), ttnn.mul(ttnn.matmul(q, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k, rope_cos_buf), ttnn.mul(ttnn.matmul(k, R_tt), rope_sin_buf))

        # Split KV into 2 groups of 4 heads (sdpa_flash_decode bug: only power-of-2 KV heads work)
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

        # Split SDPA: 2 groups of (16Q, 4KV) → concat
        n_q_split = n_q_heads // 2  # 16
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
# Run
# ══════════════════════════════════════════════════════════════
tokens_list = list(tokenizer.encode(args.prompt))
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list))
print(f'\nPrompt: "{args.prompt}" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
t_prefill = time.perf_counter() - t0
print(f"Prefill: {t_prefill*1000:.0f}ms")

next_id = int(np.argmax(logits))
tokens_list.append(next_id)
print(f"First token: {next_id} ({tokenizer.decode([next_id])})")

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
print("RESULTS: Llama-3.2-1B on Blackhole P150")
print(f"{'='*60}")
print(f"  Architecture: {n_layers} layers, {hidden} hidden, {n_q_heads} Q heads, {n_kv_heads} KV heads")
print("  Parameters: ~1.24B")
print(f"  Upload: {dt_upload*1000:.0f}ms")
print(f"  Prefill: {t_prefill*1000:.0f}ms")
print(f"  Traced decode: {avg:.1f}ms/tok ({1000/avg:.1f} tok/sec)")
print(f"  Tokens generated: {len(trace_times)}")
print(f"  Text: {text}")
print("\n  Comparison:")
print("    Qwen2.5-0.5B: 7.1ms/tok (140 tok/sec) — 24 layers, 896 hidden")
print(f"    Llama-3.2-1B: {avg:.1f}ms/tok ({1000/avg:.0f} tok/sec) — 16 layers, 2048 hidden")

ttnn.close_device(device)
print("\nDone!")
