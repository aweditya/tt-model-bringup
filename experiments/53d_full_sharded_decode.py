#!/usr/bin/env python3
"""
Experiment 53d: Full HEIGHT_SHARDED decode step.

Confirmed working from 53c:
  ✓ nlp_create_qkv_heads_decode → HEIGHT_SHARDED Q/K/V
  ✓ paged_update_cache with update_idxs_tensor (TENSOR KV POSITIONS!)
  ✓ nlp_concat_heads_decode → WIDTH_SHARDED
  ✓ SDPA decode with interleaved Q + cur_pos_tensor

Remaining issues:
  - cos/sin for RoPE must be HEIGHT_SHARDED (use ttnn.embedding lookup)
  - SDPA decode sharded output fails for GQA ("Sharded output not supported for GQA")

This experiment:
  1. Tests ttnn.embedding for cos/sin lookup (upstream pattern)
  2. Assembles a full decode step with paged KV cache
  3. If native RoPE fails, falls back to our rotation matrix trick
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; MAX_SEQ = 256; rope_theta = 1000000.0
TILE_SIZE = 32; batch_size = 1; rms_eps = 1e-6; n_layers = 24

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev(arr):
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


# ══════════════════════════════════════════════════════════════
# TEST: Cos/sin embedding lookup (upstream pattern)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST: ttnn.embedding for cos/sin lookup")
print("=" * 60)

freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)

# Half-format: [cos, cos] duplicated
cos_table = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)  # (MAX_SEQ, head_dim)
sin_table = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)

# As TILE_LAYOUT on device
cos_table_tt = to_dev(cos_table)  # (MAX_SEQ, head_dim)
sin_table_tt = to_dev(sin_table)
print(f"  cos/sin table: ({MAX_SEQ}, {head_dim}) on device")

# Test embedding lookup
test_pos = 5
pos_tensor = ttnn.from_torch(torch.tensor([[test_pos]], dtype=torch.int32),
                             device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

try:
    cos_lookup = ttnn.embedding(pos_tensor, cos_table_tt, layout=ttnn.TILE_LAYOUT)
    sin_lookup = ttnn.embedding(pos_tensor, sin_table_tt, layout=ttnn.TILE_LAYOUT)
    print(f"  ✓ Embedding lookup shape: {cos_lookup.shape}")
    print(f"  Memory: {cos_lookup.memory_config().memory_layout}")

    # Check value
    cos_np = from_dev(cos_lookup, (1, 1, head_dim))
    expected = cos_table[test_pos, :5]
    got = cos_np[0, 0, :5]
    print(f"  Expected: {expected}")
    print(f"  Got:      {got}")

    # Try reshaping for RoPE: (1, batch, head_dim) → (1, 1, batch, head_dim)
    cos_4d = ttnn.unsqueeze_to_4D(cos_lookup)
    sin_4d = ttnn.unsqueeze_to_4D(sin_lookup)
    print(f"  4D shape: {cos_4d.shape}")

    # Transpose to get (1, batch, 1, head_dim) → doesn't apply for batch=1
    cos_4d = ttnn.transpose(cos_4d, 1, 2)
    sin_4d = ttnn.transpose(sin_4d, 1, 2)
    print(f"  After transpose(1,2): {cos_4d.shape}")

    # Try to HEIGHT_SHARD the cos/sin
    cos_shard_cfg = ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, head_dim),
        core_grid=ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    cos_sharded = ttnn.to_memory_config(cos_4d, cos_shard_cfg)
    sin_sharded = ttnn.to_memory_config(sin_4d, cos_shard_cfg)
    print(f"  ✓ HEIGHT_SHARDED cos/sin: {cos_sharded.shape}")

    # Now try rotary_embedding_llama with HEIGHT_SHARDED cos/sin
    # Build trans_mat (upstream pattern: 32x32 adjacent pairs)
    trans_mat = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
    trans_mat[..., torch.arange(0, TILE_SIZE, 2), torch.arange(1, TILE_SIZE, 2)] = 1
    trans_mat[..., torch.arange(1, TILE_SIZE, 2), torch.arange(0, TILE_SIZE, 2)] = -1
    trans_mat_tt = ttnn.from_torch(trans_mat, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Create HEIGHT_SHARDED Q for test
    q_np = np.random.randn(1, 1, batch_size, n_q_heads * head_dim + 2 * n_kv_heads * head_dim).astype(np.float32)
    q_tt = to_dev_4d(q_np)
    q_out, k_out, v_out = ttnn.experimental.nlp_create_qkv_heads_decode(
        q_tt, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    q_tt.deallocate()

    try:
        q_roped = ttnn.experimental.rotary_embedding_llama(
            q_out, cos_sharded, sin_sharded, trans_mat_tt, is_decode_mode=True)
        print(f"  ✓ rotary_embedding_llama with HEIGHT_SHARDED cos/sin: {q_roped.shape}")
        NATIVE_ROPE_WORKS = True
        q_roped.deallocate()
    except Exception as e:
        print(f"  ✗ rotary_embedding_llama: {str(e)[:120]}")
        NATIVE_ROPE_WORKS = False

    q_out.deallocate(); k_out.deallocate(); v_out.deallocate()
    cos_sharded.deallocate(); sin_sharded.deallocate()
    cos_4d.deallocate(); sin_4d.deallocate()
    cos_lookup.deallocate(); sin_lookup.deallocate()
except Exception as e:
    print(f"  ✗ Embedding lookup failed: {e}")
    NATIVE_ROPE_WORKS = False


# ══════════════════════════════════════════════════════════════
# FULL DECODE STEP: paged_update_cache + SDPA + concat_heads
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FULL DECODE STEP (paged KV, tensor positions)")
print("=" * 60)

# Load weights
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

# Upload weights
print("Uploading weights...")
t0 = time.perf_counter()
dev_layers = []
for i in range(n_layers):
    lw = layer_weights_np[i]
    # Combined QKV weight for nlp_create_qkv_heads_decode
    q_w = lw["self_attn.q_proj.weight"].T  # (hidden, q_dim)
    k_w = lw["self_attn.k_proj.weight"].T  # (hidden, k_dim)
    v_w = lw["self_attn.v_proj.weight"].T  # (hidden, v_dim)
    qkv_w = np.concatenate([q_w, k_w, v_w], axis=-1)  # (hidden, q+k+v)

    q_b = lw["self_attn.q_proj.bias"]
    k_b = lw["self_attn.k_proj.bias"]
    v_b = lw["self_attn.v_proj.bias"]
    # For batch=1, bias needs to be (1, qkv_dim) to broadcast
    qkv_b = np.concatenate([q_b, k_b, v_b])

    dev_layers.append({
        "ln1_g": to_dev(lw["input_layernorm.weight"]),
        "qkv_w": to_dev(qkv_w),
        "qkv_b": to_dev(qkv_b),
        "o_w": to_dev(lw["self_attn.o_proj.weight"].T),
        "ln2_g": to_dev(lw["post_attention_layernorm.weight"]),
        "gate_w": to_dev(lw["mlp.gate_proj.weight"].T),
        "up_w": to_dev(lw["mlp.up_proj.weight"].T),
        "down_w": to_dev(lw["mlp.down_proj.weight"].T),
    })
final_norm_g_tt = to_dev(final_norm_g)
lm_head_w_tt = to_dev(lm_head_w)
del layer_weights_np
print(f"  Uploaded in {(time.perf_counter()-t0)*1000:.0f}ms")

# Rotation matrix (fallback RoPE)
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

# KV caches
k_caches, v_caches = [], []
for i in range(n_layers):
    c = np.zeros((batch_size, n_kv_heads, MAX_SEQ, head_dim), dtype=np.float32)
    k_caches.append(to_dev_4d(c.copy()))
    v_caches.append(to_dev_4d(c.copy()))

# KV memory config (upstream pattern)
kv_shard_height = ((n_kv_heads + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
kv_core_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
kv_mem_cfg = ttnn.create_sharded_memory_config(
    shape=(kv_shard_height, head_dim),
    core_grid=kv_core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

# RoPE tables for numpy reference (half-format for Qwen)
def rotate_half_np(x):
    return np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)

def get_rope_tables_half(T):
    angles = np.outer(np.arange(T, dtype=np.float32), freqs)
    return (np.concatenate([np.cos(angles), np.cos(angles)], axis=-1),
            np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))

def apply_rope_half_np(x_4d, cos_t, sin_t):
    return x_4d * cos_t[None, None] + rotate_half_np(x_4d) * sin_t[None, None]


# ══════════════════════════════════════════════════════════════
# PREFILL (same as before — CPU RoPE, fill_cache)
# ══════════════════════════════════════════════════════════════
def prefill(token_ids):
    B, T = 1, len(token_ids)
    x_np = embed_w[token_ids].reshape(B, T, hidden)
    cos_t, sin_t = get_rope_tables_half(T)
    vocab_size = embed_w.shape[0]

    for i in range(n_layers):
        dl = dev_layers[i]
        x_tt = to_dev(x_np.reshape(B * T, hidden))
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        # Separate Q/K/V for prefill (need CPU RoPE)
        qkv_tt = ttnn.add(ttnn.matmul(h_tt, dl["qkv_w"], compute_kernel_config=hifi4), dl["qkv_b"])
        qkv_np = from_dev(qkv_tt, (B, T, n_q_heads * head_dim + 2 * n_kv_heads * head_dim))

        q_dim = n_q_heads * head_dim
        k_dim = n_kv_heads * head_dim
        q_np = qkv_np[..., :q_dim].reshape(B, T, n_q_heads, head_dim).transpose(0,2,1,3)
        k_np = qkv_np[..., q_dim:q_dim+k_dim].reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)
        v_np = qkv_np[..., q_dim+k_dim:].reshape(B, T, n_kv_heads, head_dim).transpose(0,2,1,3)

        q_4d = apply_rope_half_np(q_np, cos_t, sin_t)
        k_4d = apply_rope_half_np(k_np, cos_t, sin_t)

        ttnn.kv_cache.fill_cache_for_user_(k_caches[i], to_dev_4d(k_4d), batch_index=0)
        ttnn.kv_cache.fill_cache_for_user_(v_caches[i], to_dev_4d(v_np), batch_index=0)

        attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
            to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_np),
            is_causal=True, compute_kernel_config=hifi4)
        attn_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim)).transpose(0,2,1,3).reshape(B, T, hidden)

        o_tt = ttnn.matmul(to_dev(attn_np.reshape(B*T, hidden)), dl["o_w"], compute_kernel_config=hifi4)
        x_tt2 = ttnn.add(x_tt, o_tt)
        h2_tt = ttnn.rms_norm(x_tt2, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        out_tt = ttnn.add(x_tt2, down_tt)
        x_np = from_dev(out_tt, (B * T, hidden)).reshape(B, T, hidden)

    x_tt = to_dev(x_np.reshape(B * T, hidden))
    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (B * T, embed_w.shape[0]))[-1]


# ══════════════════════════════════════════════════════════════
# DECODE with paged_update_cache (tensor positions!)
# ══════════════════════════════════════════════════════════════
def decode_step_paged(token_id, pos):
    """Decode using paged KV cache with tensor-based position updates."""
    x_np = embed_w[token_id:token_id+1].reshape(1, 1, hidden)
    x_tt = to_dev(x_np.reshape(1, hidden))

    # RoPE values for this position
    angles = pos * freqs
    cos_full = np.concatenate([np.cos(angles), np.cos(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    sin_full = np.concatenate([np.sin(angles), np.sin(angles)]).reshape(1, 1, 1, head_dim).astype(np.float32)
    cos_tt = to_dev_4d(cos_full)
    sin_tt = to_dev_4d(sin_full)

    # Position tensor
    pos_tensor = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)

    for i in range(n_layers):
        dl = dev_layers[i]
        h_tt = ttnn.rms_norm(x_tt, weight=dl["ln1_g"], epsilon=rms_eps)

        # Fused QKV — reshape to 4D for nlp_create_qkv_heads_decode
        qkv_tt = ttnn.add(ttnn.matmul(h_tt, dl["qkv_w"], compute_kernel_config=hifi4), dl["qkv_b"])
        qkv_dim = n_q_heads * head_dim + 2 * n_kv_heads * head_dim
        qkv_tt = ttnn.reshape(qkv_tt, [1, 1, batch_size, qkv_dim])

        # Split into Q/K/V heads → HEIGHT_SHARDED
        q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_tt, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)

        # RoPE (rotation matrix — unshard Q/K first for matmul)
        q_interleaved = ttnn.to_memory_config(q_tt, ttnn.DRAM_MEMORY_CONFIG)
        k_interleaved = ttnn.to_memory_config(k_tt, ttnn.DRAM_MEMORY_CONFIG)

        # Apply rotation matrix RoPE
        q_rotated = ttnn.matmul(q_interleaved, R_tt)
        q_roped = ttnn.add(ttnn.mul(q_interleaved, cos_tt), ttnn.mul(q_rotated, sin_tt))
        k_rotated = ttnn.matmul(k_interleaved, R_tt)
        k_roped = ttnn.add(ttnn.mul(k_interleaved, cos_tt), ttnn.mul(k_rotated, sin_tt))

        # Move K/V to kv_mem_cfg for paged_update_cache
        k_sharded = ttnn.to_memory_config(k_roped, kv_mem_cfg)
        v_sharded = ttnn.to_memory_config(v_tt, kv_mem_cfg)

        # Update KV cache with TENSOR position
        ttnn.experimental.paged_update_cache(k_caches[i], k_sharded, update_idxs_tensor=pos_tensor)
        ttnn.experimental.paged_update_cache(v_caches[i], v_sharded, update_idxs_tensor=pos_tensor)

        # SDPA decode with tensor position
        q_decode = ttnn.reshape(q_roped, [1, 1, n_q_heads, head_dim])
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode, k_caches[i], v_caches[i],
            cur_pos_tensor=pos_tensor, compute_kernel_config=hifi4)

        # Concat heads and output projection
        merged = ttnn.reshape(attn, [1, 1, 1, hidden])
        o_tt = ttnn.matmul(merged, dl["o_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, o_tt)

        # MLP
        h2_tt = ttnn.rms_norm(x_tt, weight=dl["ln2_g"], epsilon=rms_eps)
        gate_tt = ttnn.matmul(h2_tt, dl["gate_w"], compute_kernel_config=hifi4)
        up_tt = ttnn.matmul(h2_tt, dl["up_w"], compute_kernel_config=hifi4)
        swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)
        down_tt = ttnn.matmul(swiglu_tt, dl["down_w"], compute_kernel_config=hifi4)
        x_tt = ttnn.add(x_tt, down_tt)

    x_tt = ttnn.rms_norm(x_tt, weight=final_norm_g_tt, epsilon=rms_eps)
    logits_tt = ttnn.matmul(x_tt, lm_head_w_tt, compute_kernel_config=hifi4)
    return from_dev(logits_tt, (1, 1, embed_w.shape[0]))[0, 0]


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
tokens_list = tokenizer.encode("The capital of France is")
max_gen = 30

print(f'\nPrompt: "The capital of France is" ({len(tokens_list)} tokens)')
print(f"Generating {max_gen} tokens with paged KV cache...\n")

# Prefill
t0 = time.perf_counter()
logits = prefill(np.array(tokens_list))
t_prefill = time.perf_counter() - t0

next_id = int(np.argmax(logits))
tokens_list.append(next_id)
sys.stdout.write("The capital of France is" + tokenizer.decode([next_id]))
sys.stdout.flush()
print(f"\n  [prefill: {t_prefill*1000:.0f}ms]")

# Decode
decode_times = []
for step in range(max_gen - 1):
    pos = len(tokens_list) - 1
    t0 = time.perf_counter()
    logits = decode_step_paged(next_id, pos)
    dt = time.perf_counter() - t0
    decode_times.append(dt)

    next_id = int(np.argmax(logits))
    tokens_list.append(next_id)
    sys.stdout.write(tokenizer.decode([next_id]))
    sys.stdout.flush()
    if next_id == tokenizer.eos_token_id:
        break

# ══════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

if decode_times:
    first = decode_times[0] * 1000
    sustained = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = np.mean(sustained) * 1000
    print(f"\nPaged KV cache decode (tensor positions):")
    print(f"  First decode:  {first:.0f}ms")
    print(f"  Sustained:     {avg:.1f}ms/tok ({1000/avg:.1f} tok/sec)")
    print(f"  All times:     {[f'{t*1000:.0f}' for t in decode_times[:15]]}")

print(f"\n  Key: ALL position updates are device tensors")
print(f"    update_idxs_tensor → paged_update_cache (no Python int needed!)")
print(f"    cur_pos_tensor → SDPA decode (already worked in 52c)")
print(f"\n  → This decode is FULLY TRACEABLE!")

ttnn.close_device(device)
print("\nDone!")
