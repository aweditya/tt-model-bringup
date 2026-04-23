#!/usr/bin/env python3
"""
Qwen1.5-MoE-A2.7B-Chat Text Generation on Tenstorrent Blackhole

Run on the Tenstorrent host:
    python generate_moe.py
    python generate_moe.py --prompt "Explain quantum computing"
    python generate_moe.py --prompt "Write a haiku about AI" --max_tokens 50
    python generate_moe.py --temperature 0.7 --max_tokens 200

What this does:
    1. Downloads Qwen1.5-MoE-A2.7B-Chat weights from HuggingFace (cached after first run)
    2. Uploads weights to the Blackhole accelerator (BFP8 experts, bf16 attention)
    3. Wraps prompt in ChatML template
    4. Runs prefill + eager decode with CPU routing and device expert execution
    5. Generates text, printing tokens as they arrive (streaming)

Architecture: 14.3B total params, 2.7B active per token
    - 24 layers, 60 experts (top-4 routing), shared expert with gating
    - MHA: 16 Q heads, 16 KV heads, head_dim=128
    - Eager decode: CPU routing selects top-4 experts, device runs matmuls

Performance: ~20 tok/s (optimized eager decode with on-device accumulation)
Weight memory: ~14.1 GB on device (BFP8 experts save ~40% vs bf16)

Note: First run downloads ~28 GB of weights (8 shards). Subsequent runs use cache.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from collections import defaultdict
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Qwen1.5-MoE-A2.7B-Chat on Tenstorrent Blackhole",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Examples:\n"
           "  python generate_moe.py --prompt 'What is machine learning?'\n"
           "  python generate_moe.py --prompt 'Write a poem' --max_tokens 200\n"
           "  python generate_moe.py --temperature 0.7 --max_tokens 150\n")
parser.add_argument("--prompt", default="What is the capital of France?",
                    help="Input prompt (default: 'What is the capital of France?')")
parser.add_argument("--max_tokens", type=int, default=100,
                    help="Maximum tokens to generate (default: 100)")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature; 0 = greedy (default: 0)")
args = parser.parse_args()

# ── Model architecture ───────────────────────────────────────
hidden = 2048; n_q_heads = 16; n_kv_heads = 16; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1e6
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = 1
n_experts = 60; top_k = 4
moe_intermediate = 1408; shared_intermediate = 5632

# ── Device setup ─────────────────────────────────────────────
print("=" * 60)
print("Qwen1.5-MoE-A2.7B-Chat on Tenstorrent Blackhole")
print("=" * 60)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} = {grid.x * grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Download model ───────────────────────────────────────────
model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
n_shards = 8
print(f"\nDownloading {model_id} ({n_shards} shards)...")
shard_paths = [hf_hub_download(model_id, f"model-{i+1:05d}-of-{n_shards:05d}.safetensors")
               for i in range(n_shards)]

# Build key-to-path index (avoids loading all weights into memory at once)
key_to_path = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            key_to_path[key] = path
print(f"  {len(key_to_path)} weight tensors across {n_shards} shards")

tokenizer = AutoTokenizer.from_pretrained(model_id)

# ── Helpers ──────────────────────────────────────────────────
def load_np(key):
    """Load a single weight tensor from the correct shard."""
    with safe_open(key_to_path[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()

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

# ── RoPE (half-format for Qwen) ─────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]

# ── Load embeddings + lm_head ───────────────────────────────
print("\nLoading embeddings and language model head...")
embed_w = load_np("model.embed_tokens.weight")   # [151936, 2048]
final_norm_g = load_np("model.norm.weight")
lm_head_w = load_np("lm_head.weight").T if "lm_head.weight" in key_to_path else embed_w.T.copy()
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)

# ── Upload weights to device ────────────────────────────────
print(f"\nUploading {n_layers} layers to Blackhole (BFP8 experts, bf16 attention)...")
t0_upload = time.perf_counter()
dev_layers = []
seg_w_np_cache = []  # Pre-cached numpy gate weights (avoids repeated from_dev)

for L in range(n_layers):
    p = f"model.layers.{L}."
    layer_keys = [k for k in key_to_path if k.startswith(p)]
    by_path = defaultdict(list)
    for k in layer_keys:
        by_path[key_to_path[k]].append(k)
    lw = {}
    for path, keys in by_path.items():
        with safe_open(path, framework="pt") as f:
            for k in keys:
                lw[k[len(p):]] = f.get_tensor(k).float().numpy()

    has_o_bias = "self_attn.o_proj.bias" in lw
    dl = {
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "o_b": to_bf16(lw["self_attn.o_proj.bias"]) if has_o_bias else None,
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "router_w": to_bf16(lw["mlp.gate.weight"].T),  # [2048, 60]
    }

    # Shared expert (BFP8 — large intermediate: 5632)
    dl["s_gate_w"] = to_bfp8(lw["mlp.shared_expert.gate_proj.weight"].T)
    dl["s_up_w"] = to_bfp8(lw["mlp.shared_expert.up_proj.weight"].T)
    dl["s_down_w"] = to_bfp8(lw["mlp.shared_expert.down_proj.weight"].T)

    # Shared expert gate: [1, 2048] weight stored transposed as [2048, 1] on device
    # CRITICAL: This is a LINEAR PROJECTION, not a scalar. Produces per-token sigmoid gating.
    seg_key = "mlp.shared_expert_gate.weight"
    if seg_key in lw:
        dl["seg_w"] = to_bf16(lw[seg_key].T)  # [2048, 1]
        seg_w_np_cache.append(lw[seg_key].T.copy())  # numpy cache for decode
    else:
        dl["seg_w"] = None
        seg_w_np_cache.append(None)

    # 60 routed experts (BFP8 — intermediate: 1408)
    experts = []
    for e in range(n_experts):
        experts.append({
            "g": to_bfp8(lw[f"mlp.experts.{e}.gate_proj.weight"].T),
            "u": to_bfp8(lw[f"mlp.experts.{e}.up_proj.weight"].T),
            "d": to_bfp8(lw[f"mlp.experts.{e}.down_proj.weight"].T),
        })
    dl["experts"] = experts
    dev_layers.append(dl)
    del lw

    elapsed = time.perf_counter() - t0_upload
    eta = elapsed / (L + 1) * (n_layers - L - 1)
    pct = (L + 1) / n_layers * 100
    print(f"  [{pct:5.1f}%] Layer {L+1}/{n_layers} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"  All layers uploaded in {time.perf_counter() - t0_upload:.0f}s")

# ── KV caches ────────────────────────────────────────────────
print("Allocating KV caches...")
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


# ── ChatML template ─────────────────────────────────────────
def make_chat_prompt(user_prompt):
    """Wrap user prompt in ChatML format for Qwen chat models."""
    return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


# ── Prefill ──────────────────────────────────────────────────
def prefill(token_ids):
    """Process prompt through all 24 layers. Fills KV caches, returns last-position logits."""
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B * T, hidden))

        # Attention
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_np = apply_rope_np(
            from_dev(q, (B, T, n_q_heads * head_dim)).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        k_np = apply_rope_np(
            from_dev(k, (B, T, n_kv_heads * head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3),
            cos_t, sin_t)
        v_np = from_dev(v, (B, T, n_kv_heads * head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, hidden)
        o = ttnn.matmul(to_bf16(a_np.reshape(B * T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        if dl["o_b"] is not None:
            o = ttnn.add(o, dl["o_b"])
        x2 = ttnn.add(x_tt, o)

        # MoE: device matmuls + CPU routing
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        h2_np = from_dev(h2, (B * T, hidden))

        # Router: device matmul, CPU softmax + top-4
        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)
        rl_np = from_dev(rl, (B * T, n_experts))
        rl_np = rl_np - rl_np.max(axis=-1, keepdims=True)
        probs = np.exp(rl_np) / np.exp(rl_np).sum(axis=-1, keepdims=True)

        # Find active experts across all tokens
        active = set()
        token_top4 = []
        for t in range(B * T):
            t4 = np.argsort(probs[t])[-top_k:]
            active.update(t4)
            token_top4.append(set(t4))

        # Run active experts on device, accumulate on CPU
        moe_np = np.zeros((B * T, hidden), dtype=np.float32)
        h2_tt = to_bf16(h2_np)
        for e in active:
            w_e = np.zeros((B * T, 1), dtype=np.float32)
            for t in range(B * T):
                if e in token_top4[t]:
                    w_e[t, 0] = probs[t, e]
            ew = dl["experts"][e]
            g = ttnn.matmul(h2_tt, ew["g"], compute_kernel_config=hifi4)
            u = ttnn.matmul(h2_tt, ew["u"], compute_kernel_config=hifi4)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["d"], compute_kernel_config=hifi4)
            moe_np += w_e * from_dev(d, (B * T, hidden))

        # Shared expert with per-token sigmoid gating
        sg = ttnn.matmul(h2_tt, dl["s_gate_w"], compute_kernel_config=hifi4)
        su = ttnn.matmul(h2_tt, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), dl["s_down_w"], compute_kernel_config=hifi4)
        sd_np = from_dev(sd, (B * T, hidden))
        if dl["seg_w"] is not None:
            # shared_expert_gate.weight is [1, 2048] stored as [2048, 1] on device
            # Per-token gating: sigmoid(hidden_states @ weight.T) -> [B*T, 1]
            seg_logit = h2_np @ from_dev(dl["seg_w"], (hidden, 1))
            seg_val = 1.0 / (1.0 + np.exp(-seg_logit))
            moe_np += seg_val * sd_np
        else:
            moe_np += sd_np

        x2_np = from_dev(x2, (B * T, hidden))
        x_np = (x2_np + moe_np).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B * T, hidden)), weight=final_g, epsilon=rms_eps)
    logits = from_dev(ttnn.matmul(x_tt, lm_h, compute_kernel_config=hifi4), (B * T, vocab_size))
    return logits[-1]


# ── Optimized decode step ────────────────────────────────────
def decode_step(token_id, pos):
    """
    Single-token decode with on-device accumulation.
    CPU only reads router logits (60 floats) + computes gate scalar per layer.
    All expert outputs, residuals, and weighting stay on device.
    """
    x = to_bf16(embed_w[token_id:token_id + 1].reshape(1, 1, hidden))

    # RoPE tables for this position
    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    cos_tt = to_dev_4d(cos_np)
    sin_tt = to_dev_4d(sin_np)
    pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)

    for i in range(n_layers):
        dl = dev_layers[i]

        # Attention with bias
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"]),
                         [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"]),
                         [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"]),
                         [1, n_kv_heads, 1, head_dim])

        # Native RoPE on device
        qr = ttnn.experimental.rotary_embedding(q, cos_tt, sin_tt)
        kr = ttnn.experimental.rotary_embedding(k, cos_tt, sin_tt)
        if list(qr.shape)[2] > 1:
            qr = ttnn.slice(qr, [0, 0, 0, 0], [1, n_q_heads, 1, head_dim])
        if list(kr.shape)[2] > 1:
            kr = ttnn.slice(kr, [0, 0, 0, 0], [1, n_kv_heads, 1, head_dim])

        # Paged KV cache update
        ks = ttnn.to_memory_config(ttnn.reshape(kr, [1, 1, n_kv_heads, head_dim]), kv_cfg)
        vs = ttnn.to_memory_config(ttnn.reshape(v, [1, 1, n_kv_heads, head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], ks, update_idxs_tensor=pos_tt)
        ttnn.experimental.paged_update_cache(v_caches[i], vs, update_idxs_tensor=pos_tt)

        # Flash decode
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr, [1, 1, n_q_heads, head_dim]), k_caches[i], v_caches[i],
            cur_pos_tensor=pos_tt, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn, [1, 1, 1, hidden]), dl["o_w"], compute_kernel_config=hifi4)
        if dl["o_b"] is not None:
            o = ttnn.add(o, dl["o_b"])
        x2 = ttnn.add(x, o)  # Post-attention residual stays ON DEVICE

        # MoE routing: single sync per layer
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)

        # CPU: read 60 floats, compute top-4 + gate scalar
        rl_np = from_dev(rl, (1, n_experts))[0]
        rl_np = rl_np - rl_np.max()
        probs = np.exp(rl_np) / np.exp(rl_np).sum()
        top4 = np.argsort(probs)[-top_k:]

        if seg_w_np_cache[i] is not None:
            h2_np = from_dev(h2, (1, hidden))
            seg_val = float(1.0 / (1.0 + np.exp(-(h2_np @ seg_w_np_cache[i]).item())))
        else:
            seg_val = 1.0

        # Top-4 experts: execute + accumulate ON DEVICE
        moe_acc = None
        for e in top4:
            ew = dl["experts"][e]
            g = ttnn.matmul(h2, ew["g"], compute_kernel_config=hifi4)
            u = ttnn.matmul(h2, ew["u"], compute_kernel_config=hifi4)
            d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["d"], compute_kernel_config=hifi4)
            weighted = ttnn.multiply(d, float(probs[e]))
            if moe_acc is None:
                moe_acc = weighted
            else:
                moe_acc = ttnn.add(moe_acc, weighted)

        # Shared expert: fully on device, gate applied as scalar multiply
        sg = ttnn.matmul(h2, dl["s_gate_w"], compute_kernel_config=hifi4)
        su = ttnn.matmul(h2, dl["s_up_w"], compute_kernel_config=hifi4)
        sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), dl["s_down_w"], compute_kernel_config=hifi4)
        moe_acc = ttnn.add(moe_acc, ttnn.multiply(sd, seg_val))

        # Residual ON DEVICE
        x = ttnn.add(x2, moe_acc)

    logits = ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)
    ttnn.synchronize_device(device)
    return from_dev(logits, (1, vocab_size))[0]


# ── Reset KV caches ──────────────────────────────────────────
def reset_kv():
    """Zero out all KV caches for a fresh generation."""
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches[i])
        ttnn.copy(to_dev_4d(c), v_caches[i])


# ── Sampling ─────────────────────────────────────────────────
def sample_token(logits, temperature):
    """Sample next token: greedy if temperature=0, otherwise softmax sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    return int(np.random.choice(len(probs), p=probs))


# ══════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════

# Enable program cache for faster dispatch
try:
    device.enable_program_cache()
except Exception:
    pass

# Multi-token EOS set for Qwen ChatML: eos, <|im_end|>, <|endoftext|>, <|im_start|>
eos_ids = {tokenizer.eos_token_id, 151643, 151644, 151645}

prompt_text = args.prompt
chat_prompt = make_chat_prompt(prompt_text)
tokens = list(tokenizer.encode(chat_prompt))
max_gen = min(args.max_tokens, MAX_SEQ - len(tokens) - 1)

print(f"\nPrompt: \"{prompt_text}\"")
print(f"Tokens: {len(tokens)} prompt + up to {max_gen} generated")
if args.temperature > 0:
    print(f"Sampling: temperature={args.temperature}")
else:
    print(f"Sampling: greedy (argmax)")
print(f"\n{'=' * 60}")
print()

# Prefill
reset_kv()
t_prefill = time.perf_counter()
logits = prefill(np.array(tokens))
prefill_ms = (time.perf_counter() - t_prefill) * 1000

next_id = sample_token(logits, args.temperature)
gen = [next_id]

# Print first token immediately
sys.stdout.write(tokenizer.decode([next_id]))
sys.stdout.flush()

# Decode loop
pos = len(tokens)
times = []

for step in range(max_gen - 1):
    if next_id in eos_ids:
        break

    t0 = time.perf_counter()
    logits = decode_step(next_id, pos)
    dt = time.perf_counter() - t0
    times.append(dt)

    next_id = sample_token(logits, args.temperature)
    gen.append(next_id)
    pos += 1

    # Stream token to stdout
    tok_text = tokenizer.decode([next_id])
    sys.stdout.write(tok_text)
    sys.stdout.flush()

    if next_id in eos_ids:
        break

# ── Final stats ──────────────────────────────────────────────
print(f"\n\n{'=' * 60}")
print("STATS")
print(f"{'=' * 60}")

n_generated = len(gen)
hit_eos = n_generated > 0 and gen[-1] in eos_ids
avg_ms = np.mean(times) * 1000 if times else 0
tok_s = 1000 / avg_ms if avg_ms > 0 else 0
total_time = prefill_ms / 1000 + sum(times)

print(f"  Model:       Qwen1.5-MoE-A2.7B-Chat (14.3B total, 2.7B active)")
print(f"  Weights:     BFP8 experts + bf16 attention (HiFi4)")
print(f"  Decode:      Optimized eager (on-device accumulation, CPU routing)")
print(f"  Tokens:      {n_generated} generated in {total_time:.1f}s")
print(f"  Prefill:     {prefill_ms:.0f}ms ({len(tokens)} tokens)")
print(f"  Decode:      {avg_ms:.0f} ms/tok = {tok_s:.1f} tok/s")
print(f"  EOS:         {'yes' if hit_eos else 'no'}")

ttnn.close_device(device)
print("\nDone!")
