#!/usr/bin/env python3
"""
Interactive Chat with Llama-3.1-8B on Tenstorrent Blackhole

Run on the Tenstorrent host:
    python3 chat.py
    python3 chat.py --system "You are a pirate. Respond in pirate speak."

Features:
    - Multi-turn conversation with history
    - Streaming token output (tokens appear as they're generated)
    - Llama-3 chat template (system + user + assistant roles)
    - /clear to reset conversation, /quit to exit
    - ~21 tok/sec end-to-end on Blackhole P150

Uses BFP8 MLP weights + HiFi2 for optimal speed/quality tradeoff.
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
parser = argparse.ArgumentParser(description="Chat with Llama-3.1-8B on Blackhole")
parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens per response")
args = parser.parse_args()

# ── Model architecture ───────────────────────────────────────
hidden = 4096; n_q_heads = 32; n_kv_heads = 8; head_dim = 128
half_dim = head_dim // 2; rms_eps = 1e-5; rope_theta = 500000.0
n_layers = 32; vocab_size = 128256; MAX_SEQ = 512
TILE = 32; batch_size = 1
n_kv_split = n_kv_heads // 2
n_q_split = n_q_heads // 2

# ── Device setup ─────────────────────────────────────────────
print("Loading Llama-3.1-8B-Instruct...")
device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=True)

# ── Load weights ─────────────────────────────────────────────
model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"]
shard_paths = []; model_id = None
for mid in model_ids:
    for n_shards in [4, 2]:
        try:
            names = [f"model-{i+1:05d}-of-{n_shards:05d}.safetensors" for i in range(n_shards)]
            paths = [hf_hub_download(mid, s) for s in names]
            shard_paths = paths; model_id = mid; break
        except: pass
    if shard_paths: break

if not shard_paths:
    print("ERROR: Could not download Llama-3.1-8B weights.")
    ttnn.close_device(device); sys.exit(1)

all_weights = {}
for path in shard_paths:
    with safe_open(path, framework="pt") as f:
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

# ── RoPE ─────────────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_interleaved_np(x):
    r = np.zeros_like(x); r[..., 0::2] = -x[..., 1::2]; r[..., 1::2] = x[..., 0::2]; return r

def get_rope_tables(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return np.repeat(np.cos(angles), 2, axis=-1), np.repeat(np.sin(angles), 2, axis=-1)

def apply_rope_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_interleaved_np(x_4d) * sin_t[None, None]

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights to Blackhole...")
t0 = time.perf_counter()
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
    if (i + 1) % 8 == 0: print(f"  Layer {i+1}/{n_layers}")
final_g = to_bf16(final_norm_g)
lm_h = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Ready in {time.perf_counter()-t0:.0f}s")

# ── KV caches ────────────────────────────────────────────────
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

embed_buf = to_bf16(np.zeros((1, 1, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)

# ── Chat template ────────────────────────────────────────────
enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
bos = 128000; start_header = 128006; end_header = 128007; eot = 128009
stop_ids = {eot, 128001}


def make_chat_tokens(history, system_msg):
    """Build full token sequence from conversation history."""
    tokens = [bos, start_header] + enc("system") + [end_header] + enc("\n\n" + system_msg) + [eot]
    for role, text in history:
        tokens += [start_header] + enc(role) + [end_header] + enc("\n\n" + text) + [eot]
    tokens += [start_header] + enc("assistant") + [end_header] + enc("\n\n")
    return tokens


def update_buffers(token_id, pos):
    ttnn.copy(to_bf16(embed_w[token_id:token_id+1].reshape(1, 1, hidden)), embed_buf)
    angles = pos * freqs
    ttnn.copy(to_dev_4d(np.repeat(np.cos(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_cos_buf)
    ttnn.copy(to_dev_4d(np.repeat(np.sin(angles), 2).reshape(1,1,1,head_dim).astype(np.float32)), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device), pos_buf)


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
    x = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.reshape(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4),
                         [1, n_q_heads, 1, head_dim])
        k = ttnn.reshape(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        v = ttnn.reshape(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4),
                         [1, n_kv_heads, 1, head_dim])
        # Native RoPE (interleaved format)
        qr = ttnn.experimental.rotary_embedding(q, rope_cos_buf, rope_sin_buf)
        kr = ttnn.experimental.rotary_embedding(k, rope_cos_buf, rope_sin_buf)
        if list(qr.shape)[2] > 1:
            qr = ttnn.slice(qr, [0,0,0,0], [1,n_q_heads,1,head_dim])
        if list(kr.shape)[2] > 1:
            kr = ttnn.slice(kr, [0,0,0,0], [1,n_kv_heads,1,head_dim])
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
        # MLP with HiFi2 (BFP8 weights) — fused gate+silu
        h2 = ttnn.rms_norm(x, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.linear(h2, dl["gate_w"], activation="silu", compute_kernel_config=hifi2)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi2)
        d = ttnn.matmul(ttnn.mul(g, u), dl["down_w"], compute_kernel_config=hifi2)
        x = ttnn.add(x, d)
    return ttnn.matmul(ttnn.rms_norm(x, weight=final_g, epsilon=rms_eps), lm_h, compute_kernel_config=hifi4)


def reset_kv_caches():
    for i in range(n_layers):
        c = np.zeros((batch_size, n_kv_split, MAX_SEQ, head_dim), dtype=np.float32)
        ttnn.copy(to_dev_4d(c), k_caches_lo[i]); ttnn.copy(to_dev_4d(c), v_caches_lo[i])
        ttnn.copy(to_dev_4d(c), k_caches_hi[i]); ttnn.copy(to_dev_4d(c), v_caches_hi[i])


# ══════════════════════════════════════════════════════════════
# INTERACTIVE CHAT
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Llama-3.1-8B Chat on Tenstorrent Blackhole P150")
print(f"  {grid.x}x{grid.y} = {grid.x*grid.y} cores | ~22 tok/sec")
print(f"  Type /clear to reset, /quit to exit")
print(f"{'='*60}\n")

history = []  # [(role, text), ...]

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue
    if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
        print("Goodbye!")
        break
    if user_input == "/clear":
        history = []
        print("(conversation cleared)\n")
        continue

    # Build token sequence with full history
    history.append(("user", user_input))
    tokens = make_chat_tokens(history, args.system)

    if len(tokens) > MAX_SEQ - 10:
        print("(context too long, clearing history)")
        history = [("user", user_input)]
        tokens = make_chat_tokens(history, args.system)

    # Reset KV caches and prefill full conversation
    reset_kv_caches()
    t0 = time.perf_counter()
    logits = prefill(np.array(tokens))
    next_id = int(np.argmax(logits))
    gen = [next_id]
    pos = len(tokens)

    # Warmup + trace capture
    update_buffers(next_id, pos)
    _ = decode_forward(); ttnn.synchronize_device(device)
    try: device.enable_program_cache()
    except: pass

    update_buffers(next_id, pos)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    logits_ref = decode_forward()
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Stream tokens
    sys.stdout.write("\nAssistant: ")
    sys.stdout.flush()

    # Print first token
    first_text = tokenizer.decode([next_id], skip_special_tokens=True)
    sys.stdout.write(first_text)
    sys.stdout.flush()

    for step in range(args.max_tokens - 1):
        update_buffers(next_id, pos)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        lgt = from_dev(logits_ref, (1, vocab_size))[0]
        next_id = int(np.argmax(lgt))

        if next_id in stop_ids:
            gen.append(next_id)
            break

        # Stream this token
        token_text = tokenizer.decode([next_id], skip_special_tokens=True)
        sys.stdout.write(token_text)
        sys.stdout.flush()

        gen.append(next_id)
        pos += 1

    ttnn.release_trace(device, tid)

    elapsed = time.perf_counter() - t0
    n_decode = len(gen) - 1
    tok_s = n_decode / elapsed if elapsed > 0 else 0

    sys.stdout.write(f"\n({len(gen)} tokens, {tok_s:.0f} tok/s)\n\n")
    sys.stdout.flush()

    # Add assistant response to history
    assistant_text = tokenizer.decode(gen, skip_special_tokens=True)
    history.append(("assistant", assistant_text))

    # Trim history if getting long (keep last 10 turns)
    if len(history) > 20:
        history = history[-20:]

ttnn.close_device(device)
