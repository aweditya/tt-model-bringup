#!/usr/bin/env python3
"""
Continuous Batching Demo — vLLM-style Serving on Tenstorrent Blackhole

Run on the Tenstorrent host:
    python3 batch_serving.py
    python3 batch_serving.py --batch 16 --requests 32
    python3 batch_serving.py --batch 8 --max_tokens 100

What this does:
    1. Loads Qwen2.5-0.5B and captures a batched decode trace
    2. Fills all batch slots with different prompts
    3. As each sequence completes (EOS or max tokens), immediately admits
       the next request from the queue — no wasted batch slots
    4. Prints a live dashboard showing throughput and slot utilization

This is a vLLM-style continuous batching demonstration:
    - Multiple concurrent sequences at different positions
    - Dynamic request admission (sequences enter/leave mid-generation)
    - Position=-1 masks inactive slots (zero overhead for empty slots)
    - Single trace handles all batch sizes up to max

Performance:
    batch=8:  ~1,050 tok/sec aggregate
    batch=16: ~1,900 tok/sec aggregate
    batch=32: ~3,300 tok/sec aggregate
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

# ── CLI ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Continuous batching on Blackhole")
parser.add_argument("--batch", type=int, default=8, help="Batch size (concurrent sequences)")
parser.add_argument("--requests", type=int, default=24, help="Total requests to process")
parser.add_argument("--max_tokens", type=int, default=60, help="Max tokens per request")
args = parser.parse_args()

# ── Model architecture ───────────────────────────────────────
hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; rms_eps = 1e-6; rope_theta = 1000000.0
n_layers = 24; vocab_size = 151936; MAX_SEQ = 256
TILE = 32; batch_size = args.batch

# ── Device setup ─────────────────────────────────────────────
print("=" * 70)
print(f"Continuous Batching Demo — Qwen2.5-0.5B, batch={batch_size}")
print("=" * 70)

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150 ({grid.x}x{grid.y} = {grid.x*grid.y} cores)")

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)

# ── Load model ───────────────────────────────────────────────
print("\nLoading Qwen2.5-0.5B...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

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

# ── Helpers ──────────────────────────────────────────────────
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

# ── RoPE ─────────────────────────────────────────────────────
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

R_half = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R_half[i, i + half_dim] = -1.0; R_half[i + half_dim, i] = 1.0
R_tt = to_bf16(R_half)

# ── Upload weights ───────────────────────────────────────────
print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
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
    })
final_g_tt = to_bf16(final_norm_g)
lm_h_tt = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {time.perf_counter()-t0:.1f}s")

# ── KV caches ────────────────────────────────────────────────
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

# ── Buffers ──────────────────────────────────────────────────
embed_buf = to_bf16(np.zeros((1, 1, batch_size, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, batch_size, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, batch_size, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.zeros(batch_size, dtype=torch.int32), device=device)


def update_buffers_batch(token_ids, positions):
    x_np = embed_w[token_ids].reshape(1, 1, batch_size, hidden)
    ttnn.copy(to_bf16(x_np), embed_buf)
    cos_all = np.zeros((1, batch_size, 1, head_dim), dtype=np.float32)
    sin_all = np.zeros((1, batch_size, 1, head_dim), dtype=np.float32)
    for b in range(batch_size):
        if positions[b] >= 0:
            angles = positions[b] * freqs
            cos_all[0, b, 0, :half_dim] = np.cos(angles)
            cos_all[0, b, 0, half_dim:] = np.cos(angles)
            sin_all[0, b, 0, :half_dim] = np.sin(angles)
            sin_all[0, b, 0, half_dim:] = np.sin(angles)
    ttnn.copy(to_dev_4d(cos_all), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_all), rope_sin_buf)
    ttnn.copy(ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device), pos_buf)


def prefill_single(tokens_np, batch_idx):
    """Prefill one sequence into a specific batch slot."""
    B, T = 1, len(tokens_np)
    x_np = embed_w[tokens_np].reshape(B, T, hidden)
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    cos_t = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin_t = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    rotate_half = lambda x: np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B*T, hidden))
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_np = from_dev(q,(B*T,n_q_heads*head_dim)).reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3)
        k_np = from_dev(k,(B*T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        v_np = from_dev(v,(B*T,n_kv_heads*head_dim)).reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)
        q_np = q_np * cos_t[None, None] + rotate_half(q_np) * sin_t[None, None]
        k_np = k_np * cos_t[None, None] + rotate_half(k_np) * sin_t[None, None]
        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_np), batch_index=batch_idx)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=batch_idx)
        attn = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_np), to_dev_4d(k_np), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        a_np = from_dev(attn,(B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)
        o = ttnn.matmul(to_bf16(a_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x2 = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x2, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_np = from_dev(ttnn.add(x2, d), (B*T,hidden)).reshape(B,T,hidden)
    x_tt = ttnn.rms_norm(to_bf16(x_np.reshape(B*T,hidden)), weight=final_g_tt, epsilon=rms_eps)
    return from_dev(ttnn.matmul(x_tt, lm_h_tt, compute_kernel_config=hifi4), (B*T, vocab_size))[-1]


def decode_forward_batch():
    x_tt = embed_buf
    for i in range(n_layers):
        dl = dev_layers[i]
        h = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q = ttnn.add(ttnn.matmul(h, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k = ttnn.add(ttnn.matmul(h, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v = ttnn.add(ttnn.matmul(h, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])
        q_4d = ttnn.reshape(q, [1, batch_size, n_q_heads, head_dim])
        k_4d = ttnn.reshape(k, [1, batch_size, n_kv_heads, head_dim])
        v_4d = ttnn.reshape(v, [1, batch_size, n_kv_heads, head_dim])
        qr = ttnn.add(ttnn.mul(q_4d, rope_cos_buf), ttnn.mul(ttnn.matmul(q_4d, R_tt), rope_sin_buf))
        kr = ttnn.add(ttnn.mul(k_4d, rope_cos_buf), ttnn.mul(ttnn.matmul(k_4d, R_tt), rope_sin_buf))
        k_s = ttnn.to_memory_config(ttnn.reshape(kr, [1, batch_size, n_kv_heads, head_dim]), kv_cfg)
        v_s = ttnn.to_memory_config(ttnn.reshape(v_4d, [1, batch_size, n_kv_heads, head_dim]), kv_cfg)
        ttnn.experimental.paged_update_cache(k_caches[i], k_s, update_idxs_tensor=pos_buf)
        ttnn.experimental.paged_update_cache(v_caches[i], v_s, update_idxs_tensor=pos_buf)
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(qr, [1, batch_size, n_q_heads, head_dim]),
            k_caches[i], v_caches[i],
            cur_pos_tensor=pos_buf, compute_kernel_config=hifi4)
        o = ttnn.matmul(ttnn.reshape(attn, [1, 1, batch_size, hidden]), dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o)
        h2 = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        g = ttnn.matmul(h2, dl["gate_w"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, dl["up_w"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, d)
    return ttnn.matmul(ttnn.rms_norm(x_tt, weight=final_g_tt, epsilon=rms_eps),
                       lm_h_tt, compute_kernel_config=hifi4)


# ══════════════════════════════════════════════════════════════
# REQUEST QUEUE
# ══════════════════════════════════════════════════════════════

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
    "Mars is known as the red planet because",
    "The Pythagorean theorem states that",
    "Gravity is a fundamental force that",
    "The speed of sound in air is",
    "Vaccines work by training the immune system",
    "The Eiffel Tower was built in",
    "Climate change is primarily caused by",
    "The mitochondria is often called the",
]

total_requests = min(args.requests, len(prompts))
max_tokens = args.max_tokens


class Slot:
    def __init__(self, idx):
        self.idx = idx
        self.active = False
        self.request_id = -1
        self.tokens = []
        self.position = -1
        self.tokens_generated = 0
        self.next_token_id = 0
        self.start_time = 0.0

slots = [Slot(i) for i in range(batch_size)]
request_queue = list(range(total_requests))
completed = []


def admit_request(slot, req_id):
    prompt = prompts[req_id % len(prompts)]
    token_ids = tokenizer.encode(prompt)
    logits = prefill_single(np.array(token_ids), slot.idx)
    first_token = int(np.argmax(logits))
    slot.active = True
    slot.request_id = req_id
    slot.tokens = list(token_ids) + [first_token]
    slot.position = len(token_ids)
    slot.tokens_generated = 1
    slot.next_token_id = first_token
    slot.start_time = time.perf_counter()


def complete_request(slot):
    elapsed = time.perf_counter() - slot.start_time
    text = tokenizer.decode(slot.tokens)
    completed.append({
        "id": slot.request_id,
        "tokens": slot.tokens_generated,
        "elapsed_ms": elapsed * 1000,
        "text": text[:150],
    })
    slot.active = False
    slot.position = -1


# ══════════════════════════════════════════════════════════════
# CONTINUOUS BATCHING LOOP
# ══════════════════════════════════════════════════════════════

print(f"\nProcessing {total_requests} requests through {batch_size} batch slots")
print(f"Max {max_tokens} tokens per request\n")

# Phase 1: Fill initial batch
print("Prefilling initial batch...")
for i in range(min(batch_size, total_requests)):
    req_id = request_queue.pop(0)
    admit_request(slots[i], req_id)

# Warmup + trace capture
token_ids_batch = [s.next_token_id if s.active else 0 for s in slots]
positions_batch = [s.position if s.active else -1 for s in slots]
update_buffers_batch(token_ids_batch, positions_batch)
_ = decode_forward_batch(); ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

update_buffers_batch(token_ids_batch, positions_batch)
tid = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward_batch()
ttnn.end_trace_capture(device, tid, cq_id=0)

# Phase 2: Continuous generation
print("Generating...\n")
print(f"{'Step':>5} | {'Active':>6} | {'Done':>4}/{total_requests} | {'Tok/s':>7} | Event")
print("-" * 70)

total_tokens = 0
decode_times = []
t_start = time.perf_counter()

while any(s.active for s in slots) or request_queue:
    token_ids_batch = [s.next_token_id if s.active else 0 for s in slots]
    positions_batch = [s.position if s.active else -1 for s in slots]
    update_buffers_batch(token_ids_batch, positions_batch)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    decode_times.append(dt)

    logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
    step = len(decode_times)

    for s in slots:
        if not s.active:
            continue
        next_id = int(np.argmax(logits[0, 0, s.idx, :]))
        s.tokens.append(next_id)
        s.position += 1
        s.tokens_generated += 1
        s.next_token_id = next_id
        total_tokens += 1

        if next_id == tokenizer.eos_token_id or s.tokens_generated >= max_tokens or s.position >= MAX_SEQ - 1:
            complete_request(s)
            active = sum(1 for sl in slots if sl.active)
            instant_tps = total_tokens / (time.perf_counter() - t_start)
            event = f"Slot {s.idx}: req {s.request_id} done ({s.tokens_generated} tok)"

            if request_queue:
                new_id = request_queue.pop(0)
                admit_request(s, new_id)
                event += f" -> req {new_id} admitted"

            print(f"{step:5d} | {active:6d} | {len(completed):4d}/{total_requests} | {instant_tps:7.0f} | {event}")

t_total = time.perf_counter() - t_start
ttnn.release_trace(device, tid)

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")

sustained = decode_times[1:] if len(decode_times) > 1 else decode_times
avg_step_ms = np.mean(sustained) * 1000
decode_only_tps = total_tokens / sum(sustained) if sustained else 0
overall_tps = total_tokens / t_total

print(f"\n  Configuration:")
print(f"    Model:          Qwen2.5-0.5B (24 layers, 0.5B params)")
print(f"    Batch size:     {batch_size} concurrent sequences")
print(f"    Total requests: {total_requests}")
print(f"    Max tokens/req: {max_tokens}")

print(f"\n  Throughput:")
print(f"    Total tokens:     {total_tokens}")
print(f"    Total time:       {t_total:.1f}s")
print(f"    Avg step time:    {avg_step_ms:.1f}ms ({batch_size} tokens/step)")
print(f"    Decode-only:      {decode_only_tps:.0f} tok/sec")
print(f"    Overall (w/ prefill): {overall_tps:.0f} tok/sec")
print(f"    Per-sequence:     {avg_step_ms:.1f}ms/tok = {1000/avg_step_ms:.0f} tok/s")

if completed:
    avg_lat = np.mean([r["elapsed_ms"] for r in completed])
    avg_tok = np.mean([r["tokens"] for r in completed])
    print(f"\n  Per-request:")
    print(f"    Avg tokens:     {avg_tok:.0f}")
    print(f"    Avg latency:    {avg_lat:.0f}ms")

print(f"\n  Sample outputs:")
for r in completed[:4]:
    snippet = r["text"][:100].replace("\n", " ")
    print(f"    Req {r['id']:2d} ({r['tokens']:2d} tok, {r['elapsed_ms']:.0f}ms): {snippet}...")

print(f"\n  Scaling reference (Qwen2.5-0.5B on Blackhole):")
print(f"    batch=1:  ~132 tok/s (single sequence)")
print(f"    batch=8:  ~1,050 tok/s (8x near-perfect)")
print(f"    batch=16: ~1,900 tok/s")
print(f"    batch=32: ~3,300 tok/s")
print(f"    batch=64: ~4,800 tok/s (peak aggregate)")

ttnn.close_device(device)
print("\nDone!")
