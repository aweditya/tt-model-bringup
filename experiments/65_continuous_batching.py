#!/usr/bin/env python3
"""
Experiment 65: Continuous batching prototype on Qwen2.5-0.5B.

Foundation for production serving: sequences enter and leave mid-generation.
Uses batch=8 traced decode as base (from exp 56).

Key mechanisms tested:
  1. Different prompts per batch slot (different lengths → different positions)
  2. Sequence completion → slot reuse with new prompt
  3. Position=-1 skips compute in SDPA decode (per tt-nn docs)
  4. Throughput measurement under continuous load

Hypothesis: Max-batch trace with slot masking achieves ≥90% of full-batch
throughput, even with sequences starting/finishing at different times.
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
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--max-tokens", type=int, default=60)
parser.add_argument("--total-requests", type=int, default=24)
args = parser.parse_args()

# Qwen2.5-0.5B architecture
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE_SIZE = 32
batch_size = args.batch

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# ── Load model (same as exp 56) ──
model_id = "Qwen/Qwen2.5-0.5B"
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

# RoPE
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def get_rope_single(pos):
    angles = pos * freqs
    cos_h = np.cos(angles)
    sin_h = np.sin(angles)
    return cos_h, sin_h

# Half-format rotation matrix
R_half = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_half[i, i + half_dim] = -1.0
    R_half[i + half_dim, i] = 1.0
R_tt = to_bf16(R_half)

# ── Upload weights ──
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
del layer_weights_np
dt_upload = time.perf_counter() - t0
print(f"  Uploaded in {dt_upload*1000:.0f}ms")

# ── KV caches ──
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_sh = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_cg = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_sh, head_dim), core_grid=kv_cg,
    strategy=ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=True)

# ── Buffers ──
embed_buf = to_bf16(np.zeros((1, 1, batch_size, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, batch_size, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, batch_size, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.zeros(batch_size, dtype=torch.int32), device=device)


def update_buffers_batch(token_ids, positions):
    """Update embeddings and RoPE for batch of sequences."""
    x_np = embed_w[token_ids].reshape(1, 1, batch_size, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)

    cos_all = np.zeros((1, batch_size, 1, head_dim), dtype=np.float32)
    sin_all = np.zeros((1, batch_size, 1, head_dim), dtype=np.float32)
    for b in range(batch_size):
        if positions[b] >= 0:  # Active slot
            cos_h, sin_h = get_rope_single(positions[b])
            cos_all[0, b, 0, :half_dim] = cos_h
            cos_all[0, b, 0, half_dim:] = cos_h
            sin_all[0, b, 0, :half_dim] = sin_h
            sin_all[0, b, 0, half_dim:] = sin_h
    ttnn.copy(to_dev_4d(cos_all), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_all), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device), pos_buf)


def prefill_single(tokens_np, batch_idx):
    """Prefill a single sequence into batch slot batch_idx (CPU path)."""
    B, T = 1, len(tokens_np)
    x_np = embed_w[tokens_np].reshape(B, T, hidden)

    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)

    for i in range(n_layers):
        lw = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B * T, hidden))
        h = ttnn.rms_norm(x_tt, weight=lw["ln1_g"], epsilon=rms_eps)

        q = ttnn.add(ttnn.matmul(h, lw["q_w"], compute_kernel_config=hifi4), lw["q_b"])
        k = ttnn.add(ttnn.matmul(h, lw["k_w"], compute_kernel_config=hifi4), lw["k_b"])
        v = ttnn.add(ttnn.matmul(h, lw["v_w"], compute_kernel_config=hifi4), lw["v_b"])

        q_np = from_dev(q, (B*T, n_q_heads*head_dim)).reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
        k_np = from_dev(k, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v_np = from_dev(v, (B*T, n_kv_heads*head_dim)).reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE on CPU
        q_np = q_np * cos_t[None, None] + np.concatenate([-q_np[..., half_dim:], q_np[..., :half_dim]], axis=-1) * sin_t[None, None]
        k_np = k_np * cos_t[None, None] + np.concatenate([-k_np[..., half_dim:], k_np[..., :half_dim]], axis=-1) * sin_t[None, None]

        # Fill THIS batch slot's KV cache
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=batch_idx)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=batch_idx)

        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn, (B, n_q_heads, T, head_dim)).transpose(0, 2, 1, 3).reshape(B, T, hidden)

        o = ttnn.matmul(to_bf16(a_np.reshape(B*T, hidden)), lw["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=lw["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, lw["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, lw["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), lw["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T, hidden)).reshape(B, T, hidden)

    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T, hidden)), weight=final_norm_g_tt, epsilon=rms_eps)
    logits = from_dev(ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4), (B*T, vocab_size))
    return logits[-1]


# ── Traced decode ──
def decode_forward_batch():
    x_tt = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_4d = ttnn.reshape(q_tt, [1, batch_size, n_q_heads, head_dim])
        k_4d = ttnn.reshape(k_tt, [1, batch_size, n_kv_heads, head_dim])
        v_4d = ttnn.reshape(v_tt, [1, batch_size, n_kv_heads, head_dim])

        q_rotated = ttnn.matmul(q_4d, R_tt)
        q_roped = ttnn.add(ttnn.mul(q_4d, rope_cos_buf), ttnn.mul(q_rotated, rope_sin_buf))
        k_rotated = ttnn.matmul(k_4d, R_tt)
        k_roped = ttnn.add(ttnn.mul(k_4d, rope_cos_buf), ttnn.mul(k_rotated, rope_sin_buf))

        k_for_cache = ttnn.reshape(k_roped, [1, batch_size, n_kv_heads, head_dim])
        v_for_cache = ttnn.reshape(v_4d, [1, batch_size, n_kv_heads, head_dim])
        k_sharded = ttnn.to_memory_config(k_for_cache, kv_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_for_cache, kv_mem_cfg)

        ttnn.experimental.paged_update_cache(k_caches[i], k_sharded, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_sharded, update_idxs_tensor=pos_buf)

        q_decode = ttnn.reshape(q_roped, [1, batch_size, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)

        merged = ttnn.reshape(attn, [1, 1, batch_size, hidden])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)

        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    return ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# Continuous batching simulation
# ══════════════════════════════════════════════════════════════

# Request queue — diverse prompts to stress-test independence
prompts = [
    "The capital of France is",
    "Once upon a time, there was a",
    "In the year 2030, artificial intelligence",
    "The quick brown fox jumps over",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "To make a good cup of coffee, you need",
    "The theory of relativity states that",
    "In machine learning, a neural network",
    "The speed of light in vacuum is",
    "Python is a programming language that",
    "Mount Everest is the highest mountain",
    "The human brain contains approximately",
    "Photosynthesis is the process by which",
    "The Declaration of Independence was signed in",
    "Quantum computing differs from classical computing",
    "The Great Wall of China was built to",
    "DNA stands for deoxyribonucleic acid and",
    "The Mona Lisa was painted by",
    "The periodic table organizes elements by",
    "Black holes are regions of spacetime where",
    "The Amazon rainforest is often called",
    "Shakespeare wrote many famous plays including",
    "The Internet was originally developed as",
]

total_requests = min(args.total_requests, len(prompts))
max_tokens = args.max_tokens

print(f"\n{'='*60}")
print(f"CONTINUOUS BATCHING: {total_requests} requests, batch={batch_size}, max_tokens={max_tokens}")
print(f"{'='*60}")

# ── Phase 1: Initial fill ──
# Fill all batch slots with initial prompts
class Slot:
    def __init__(self, idx):
        self.idx = idx          # Batch index
        self.active = False
        self.request_id = -1
        self.tokens = []
        self.position = -1      # -1 = inactive
        self.tokens_generated = 0
        self.next_token_id = 0
        self.start_time = 0.0

slots = [Slot(i) for i in range(batch_size)]
request_queue = list(range(total_requests))
completed_requests = []
next_request_id = 0

def admit_request(slot, request_id):
    """Prefill a new request into a batch slot."""
    prompt = prompts[request_id % len(prompts)]
    token_ids = tokenizer.encode(prompt)

    # Prefill
    logits = prefill_single(np.array(token_ids), slot.idx)
    first_token = int(np.argmax(logits))

    slot.active = True
    slot.request_id = request_id
    slot.tokens = list(token_ids) + [first_token]
    slot.position = len(token_ids)  # Next decode position
    slot.tokens_generated = 1
    slot.next_token_id = first_token
    slot.start_time = time.perf_counter()

def complete_request(slot):
    """Mark a request as complete and free the slot."""
    elapsed = time.perf_counter() - slot.start_time
    text = tokenizer.decode(slot.tokens)
    completed_requests.append({
        "request_id": slot.request_id,
        "tokens_generated": slot.tokens_generated,
        "elapsed": elapsed,
        "text": text[:200],
    })
    slot.active = False
    slot.position = -1
    slot.tokens_generated = 0


# Phase 1: Fill initial batch
print("\nPhase 1: Initial prefill...")
t_start = time.perf_counter()
for i in range(min(batch_size, total_requests)):
    req_id = request_queue.pop(0)
    admit_request(slots[i], req_id)
    print(f"  Slot {i}: req {req_id} — \"{prompts[req_id % len(prompts)][:40]}...\"")

# Warmup decode
token_ids_batch = [s.next_token_id if s.active else 0 for s in slots]
positions_batch = [s.position if s.active else -1 for s in slots]
update_buffers_batch(token_ids_batch, positions_batch)
_ = decode_forward_batch()
ttnn.synchronize_device(device)

# Enable program cache + capture trace
try:
    device.enable_program_cache()
except:
    pass

print("\nCapturing trace...")
update_buffers_batch(token_ids_batch, positions_batch)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward_batch()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
print("  Trace captured!")

# Phase 2: Continuous generation
print("\nPhase 2: Continuous generation...")
total_decode_steps = 0
total_tokens_generated = 0
decode_times = []
t_gen_start = time.perf_counter()

while any(s.active for s in slots) or request_queue:
    # Prepare batch
    token_ids_batch = []
    positions_batch = []
    for s in slots:
        if s.active:
            token_ids_batch.append(s.next_token_id)
            positions_batch.append(s.position)
        else:
            token_ids_batch.append(0)  # Padding
            positions_batch.append(-1)  # Skip in SDPA

    update_buffers_batch(token_ids_batch, positions_batch)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    decode_times.append(dt)
    total_decode_steps += 1

    # Read logits and advance each active slot
    logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
    active_this_step = 0
    for s in slots:
        if not s.active:
            continue
        active_this_step += 1
        next_id = int(np.argmax(logits[0, 0, s.idx, :]))
        s.tokens.append(next_id)
        s.position += 1
        s.tokens_generated += 1
        s.next_token_id = next_id
        total_tokens_generated += 1

        # Check completion: EOS or max tokens
        if next_id == tokenizer.eos_token_id or s.tokens_generated >= max_tokens or s.position >= MAX_SEQ - 1:
            complete_request(s)
            # Admit next request if available
            if request_queue:
                new_req_id = request_queue.pop(0)
                admit_request(s, new_req_id)
                print(f"  Step {total_decode_steps}: Slot {s.idx} → req {new_req_id} ({len(completed_requests)}/{total_requests} done)")

    if total_decode_steps % 50 == 0:
        active = sum(1 for s in slots if s.active)
        print(f"  Step {total_decode_steps}: {active}/{batch_size} active, {len(completed_requests)}/{total_requests} completed, {total_tokens_generated} total tokens")

t_gen_total = time.perf_counter() - t_gen_start

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("CONTINUOUS BATCHING RESULTS")
print(f"{'='*60}")

print(f"\nConfiguration:")
print(f"  Model: Qwen2.5-0.5B")
print(f"  Batch size: {batch_size}")
print(f"  Total requests: {total_requests}")
print(f"  Max tokens/request: {max_tokens}")

print(f"\nThroughput:")
sustained = decode_times[1:] if len(decode_times) > 1 else decode_times
avg_step = np.mean(sustained) * 1000
print(f"  Total tokens generated: {total_tokens_generated}")
print(f"  Total decode steps: {total_decode_steps}")
print(f"  Total time: {t_gen_total:.1f}s")
print(f"  Average step time: {avg_step:.1f}ms")
print(f"  Aggregate throughput: {total_tokens_generated / t_gen_total:.0f} tok/sec (including prefills)")
print(f"  Decode-only throughput: {total_tokens_generated / (sum(sustained)):.0f} tok/sec")

print(f"\nPer-request stats:")
if completed_requests:
    gen_counts = [r["tokens_generated"] for r in completed_requests]
    latencies = [r["elapsed"] for r in completed_requests]
    print(f"  Avg tokens/request: {np.mean(gen_counts):.1f}")
    print(f"  Avg latency/request: {np.mean(latencies)*1000:.0f}ms")
    print(f"  Min/Max latency: {min(latencies)*1000:.0f}ms / {max(latencies)*1000:.0f}ms")

print(f"\nSample outputs:")
for r in completed_requests[:5]:
    print(f"  Req {r['request_id']} ({r['tokens_generated']} tok, {r['elapsed']*1000:.0f}ms): {r['text'][:120]}...")

print(f"\nComparison:")
print(f"  Exp 56 batch={batch_size} (static):     {batch_size * 1000 / avg_step:.0f} tok/sec")
print(f"  Exp 65 continuous batching: {total_tokens_generated / t_gen_total:.0f} tok/sec (with prefill pauses)")

ttnn.close_device(device)
print("\nDone!")
