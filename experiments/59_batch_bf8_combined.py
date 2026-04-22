#!/usr/bin/env python3
"""
Experiment 59: Batch decode + bf8 MLP mixed precision combined.

Two proven optimizations:
  - exp 56 batch=8: 7.6ms/step = 1,050 tok/sec (perfect 8x scaling)
  - exp 57c bf8 MLP: 7.4ms/tok = 134.3 tok/sec (2% faster than all-bf16)

Hypothesis: Combining these gives ~1,070+ tok/sec at batch=8.
bf8 MLP reduces memory bandwidth (half the bytes for 75% of weights)
which should help more at higher batch where bandwidth starts to matter.

Expected at batch=32: current 9.6ms/step. If bf8 helps more at higher
batch (more bandwidth-bound), could see 8-9ms/step = 3,500-4,000 tok/sec.
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
parser.add_argument("--tokens", type=int, default=100)
parser.add_argument("--prompt", default="The capital of France is")
args = parser.parse_args()

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
print(f"Batch size: {batch_size}")

# ── Load model ────────────────────────────────────────────────
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


# ── Rotation matrix for on-device RoPE ──
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_bf16(R)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


# ── Upload weights: bf16 attention + bf8 MLP ──
print("Uploading weights (bf16 attn + bf8 MLP)...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    dev_layers.append({
        "ln1_g": to_bf16(lw["input_layernorm.weight"]),
        "q_w": to_bf16(lw["self_attn.q_proj.weight"].T),
        "q_b": to_bf16(lw["self_attn.q_proj.bias"]),
        "k_w": to_bf16(lw["self_attn.k_proj.weight"].T),
        "k_b": to_bf16(lw["self_attn.k_proj.bias"]),
        "v_w": to_bf16(lw["self_attn.v_proj.weight"].T),
        "v_b": to_bf16(lw["self_attn.v_proj.bias"]),
        "o_w": to_bf16(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_bf16(lw["post_attention_layernorm.weight"]),
        "gate_w": to_bf8(lw["mlp.gate_proj.weight"].T),
        "up_w": to_bf8(lw["mlp.up_proj.weight"].T),
        "down_w": to_bf8(lw["mlp.down_proj.weight"].T),
    })
final_norm_g_tt = to_bf16(final_norm_g)
lm_head_w_tt = to_bf16(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")


# ── KV caches (batched) ──
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE  # 32
kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)


# ── Input buffers for trace ──
embed_buf = to_dev_4d(np.zeros((1, 1, batch_size, hidden), dtype=np.float32))
rope_cos_buf = to_dev_4d(np.ones((1, 1, 1, head_dim), dtype=np.float32))
rope_sin_buf = to_dev_4d(np.zeros((1, 1, 1, head_dim), dtype=np.float32))
pos_buf = ttnn.from_torch(torch.zeros(batch_size, dtype=torch.int32), device=device)


def update_buffers_batch(token_ids, positions):
    x_np = embed_w[token_ids].reshape(1, 1, batch_size, hidden)
    ttnn.copy(to_dev_4d(x_np), embed_buf)

    pos = positions[0]
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    ttnn.copy(to_dev_4d(cos_full), rope_cos_buf)
    ttnn.copy(to_dev_4d(sin_full), rope_sin_buf)

    ttnn.copy(ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), device=device), pos_buf)


# ── Prefill (per-sequence, batch=1 style) ──
def prefill_single(token_ids, batch_idx):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_bf16(x_np.reshape(B * T, hidden))
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)
        q_tt = ttnn.add(ttnn.matmul(h_tt, dl["q_w"], compute_kernel_config=hifi4), dl["q_b"])
        k_tt = ttnn.add(ttnn.matmul(h_tt, dl["k_w"], compute_kernel_config=hifi4), dl["k_b"])
        v_tt = ttnn.add(ttnn.matmul(h_tt, dl["v_w"], compute_kernel_config=hifi4), dl["v_b"])

        q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
        k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
        v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

        q_4d = apply_rope_half_np(q_np.reshape(B,T,n_q_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        k_4d = apply_rope_half_np(k_np.reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3), cos_t, sin_t)
        v_4d = v_np.reshape(B,T,n_kv_heads,head_dim).transpose(0,2,1,3)

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_4d), batch_index=batch_idx)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_4d), batch_index=batch_idx)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B,n_q_heads,T,head_dim)).transpose(0,2,1,3).reshape(B,T,hidden)

        o_tt = ttnn.matmul(to_bf16(attn_np.reshape(B*T,hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B * T, hidden)).reshape(B, T, hidden)

    x_tt = to_bf16(x_np.reshape(B * T, hidden))
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B * T, vocab_size))[-1]


# ── Batch decode step (traceable) ──
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
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return logits_tt


# ══════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════
tokens_list_base = tokenizer.encode(args.prompt)
max_gen = min(args.tokens, MAX_SEQ - len(tokens_list_base))

print(f'\nPrompt: "{args.prompt}" ({len(tokens_list_base)} tokens)')
print(f"Generating {max_gen} tokens x {batch_size} sequences\n")

# Prefill each batch element
print("Prefilling...")
t0 = time.perf_counter()
first_logits = []
for b in range(batch_size):
    logits = prefill_single(np.array(tokens_list_base), b)
    first_logits.append(logits)
t_prefill = time.perf_counter() - t0
print(f"  Prefill {batch_size} sequences: {t_prefill*1000:.0f}ms")

next_ids = [int(np.argmax(first_logits[0]))] * batch_size
positions = [len(tokens_list_base)] * batch_size
tokens_per_seq = [[t for t in tokens_list_base] + [next_ids[0]] for _ in range(batch_size)]


# ── Non-traced batch decode (correctness check) ──
print("\n--- Non-traced batch decode (correctness) ---")
nontrace_times = []
for step in range(min(10, max_gen - 1)):
    update_buffers_batch(next_ids, positions)
    t0 = time.perf_counter()
    logits_tt = decode_forward_batch()
    ttnn.synchronize_device(device)
    dt = time.perf_counter() - t0
    nontrace_times.append(dt)

    logits = from_dev(logits_tt, (1, 1, batch_size, vocab_size))
    for b in range(batch_size):
        next_ids[b] = int(np.argmax(logits[0, 0, b, :]))
        positions[b] += 1
        tokens_per_seq[b].append(next_ids[b])

avg_nt = np.mean(nontrace_times[1:]) * 1000 if len(nontrace_times) > 1 else nontrace_times[0] * 1000
print(f"  Non-traced: {avg_nt:.1f}ms/step ({batch_size * 1000 / avg_nt:.1f} tok/sec)")
text_so_far = tokenizer.decode(tokens_per_seq[0])
print(f"  Seq[0]: {text_so_far}")


# ── Traced batch decode ──
print("\n--- Traced batch decode ---")
update_buffers_batch(next_ids, positions)
_ = decode_forward_batch()
ttnn.synchronize_device(device)
try: device.enable_program_cache()
except: pass

print("Capturing trace...")
update_buffers_batch(next_ids, positions)
t_cap0 = time.perf_counter()
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_forward_batch()
ttnn.end_trace_capture(device, trace_id, cq_id=0)
t_cap = time.perf_counter() - t_cap0
print(f"  Trace captured in {t_cap*1000:.0f}ms")

trace_times = []
for step in range(max_gen - 11):
    update_buffers_batch(next_ids, positions)

    t0 = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    dt = time.perf_counter() - t0
    trace_times.append(dt)

    logits = from_dev(logits_ref, (1, 1, batch_size, vocab_size))
    for b in range(batch_size):
        next_ids[b] = int(np.argmax(logits[0, 0, b, :]))
        positions[b] += 1
        tokens_per_seq[b].append(next_ids[b])
        if next_ids[b] == tokenizer.eos_token_id:
            next_ids[b] = tokenizer.eos_token_id


# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: batch={batch_size} + bf8 MLP mixed precision")
print("=" * 60)

if nontrace_times:
    avg_nt = np.mean(nontrace_times[1:]) * 1000 if len(nontrace_times) > 1 else 0
    nt_tps = batch_size * 1000 / avg_nt if avg_nt > 0 else 0
    print(f"\nNon-traced batch={batch_size}:")
    print(f"  {avg_nt:.1f}ms/step = {nt_tps:.1f} tok/sec aggregate")

if trace_times:
    sustained = trace_times[1:] if len(trace_times) > 1 else trace_times
    tr_avg = np.mean(sustained) * 1000
    tr_tps = batch_size * 1000 / tr_avg
    print(f"\nTraced batch={batch_size} + bf8 MLP:")
    print(f"  {tr_avg:.1f}ms/step = {tr_tps:.1f} tok/sec aggregate ({tr_avg/batch_size:.2f}ms/tok)")
    print(f"  First: {trace_times[0]*1000:.1f}ms")
    print(f"  Times: {[f'{t*1000:.1f}' for t in trace_times[:10]]}")

print(f"\n  Comparison (traced):")
print(f"    53e  b=1  bf16:     7.6ms/step =   132 tok/sec")
print(f"    57c  b=1  bf8-MLP:  7.4ms/tok  =   134 tok/sec")
print(f"    56   b={batch_size}  bf16:     {'7.6' if batch_size==8 else '?'}ms/step = {'1,050' if batch_size==8 else '?'} tok/sec")
if trace_times:
    print(f"    59   b={batch_size}  bf8-MLP:  {tr_avg:.1f}ms/step = {tr_tps:,.0f} tok/sec")

print(f"\n  Generated text (first 3 sequences):")
for b in range(min(3, batch_size)):
    text = tokenizer.decode(tokens_per_seq[b])
    print(f"    Seq[{b}]: {text[:200]}...")

all_same = all(tokens_per_seq[b] == tokens_per_seq[0] for b in range(batch_size))
print(f"\n  All sequences identical (greedy from same prompt): {all_same}")

ttnn.close_device(device)
print("\nDone!")
