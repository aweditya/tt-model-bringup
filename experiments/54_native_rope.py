#!/usr/bin/env python3
"""
Experiment 54: Definitively determine rotary_embedding_llama's rotation format
and benchmark native RoPE vs rotation matrix approach.

Key questions:
  1. Does rotary_embedding_llama implement INTERLEAVED (adjacent-pair) or HALF (midpoint split)?
  2. How fast is native RoPE vs our rotation matrix approach?

Background:
  - Qwen2.5 uses HALF format: rotate_half splits at midpoint
  - The upstream trans_mat (32x32) does adjacent-pair swaps -> INTERLEAVED
  - From 53c: the op works with HEIGHT_SHARDED Q from nlp_create_qkv_heads_decode
  - From 53d: ttnn.embedding works for cos/sin lookup, cos/sin must be HEIGHT_SHARDED
  - rotary_embedding_llama in decode mode requires HEIGHT_SHARDED cos/sin
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

hidden = 896; n_q_heads = 14; n_kv_heads = 2; head_dim = 64
half_dim = head_dim // 2; MAX_SEQ = 256; rope_theta = 1000000.0
TILE_SIZE = 32; batch_size = 1

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
# Numpy reference implementations
# ══════════════════════════════════════════════════════════════
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

def apply_rope_interleaved_np(x, pos):
    """Interleaved rotation: adjacent pairs (x0,x1), (x2,x3), ..."""
    a = pos * freqs
    cos_full = np.repeat(np.cos(a), 2)
    sin_full = np.repeat(np.sin(a), 2)
    x_rot = np.empty_like(x)
    x_rot[..., 0::2] = -x[..., 1::2]
    x_rot[..., 1::2] = x[..., 0::2]
    return x * cos_full + x_rot * sin_full

def apply_rope_half_np(x, pos):
    """Half rotation: split at midpoint (Qwen format)."""
    a = pos * freqs
    cos_full = np.concatenate([np.cos(a), np.cos(a)])
    sin_full = np.concatenate([np.sin(a), np.sin(a)])
    rotated = np.concatenate([-x[..., half_dim:], x[..., :half_dim]], axis=-1)
    return x * cos_full + rotated * sin_full


# ══════════════════════════════════════════════════════════════
# Build cos/sin tables in BOTH formats
# ══════════════════════════════════════════════════════════════
angles = np.outer(np.arange(MAX_SEQ, dtype=np.float32), freqs)  # (MAX_SEQ, half_dim)

# INTERLEAVED: [c0, c0, c1, c1, ...] — matches adjacent-pair trans_mat
cos_interleaved_table = np.repeat(np.cos(angles), 2, axis=-1).astype(np.float32)  # (MAX_SEQ, head_dim)
sin_interleaved_table = np.repeat(np.sin(angles), 2, axis=-1).astype(np.float32)

# HALF: [c0..c31, c0..c31] — matches Qwen rotate_half
cos_half_table = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
sin_half_table = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)

# Upload tables for embedding lookup: (MAX_SEQ, head_dim) as 2D TILE_LAYOUT
cos_interleaved_tt = to_dev(cos_interleaved_table)
sin_interleaved_tt = to_dev(sin_interleaved_table)
cos_half_tt = to_dev(cos_half_table)
sin_half_tt = to_dev(sin_half_table)

# Build trans_mat (upstream 32x32 adjacent-pair swaps)
trans_mat = torch.zeros(1, 1, TILE_SIZE, TILE_SIZE)
trans_mat[..., torch.arange(0, TILE_SIZE, 2), torch.arange(1, TILE_SIZE, 2)] = 1
trans_mat[..., torch.arange(1, TILE_SIZE, 2), torch.arange(0, TILE_SIZE, 2)] = -1
trans_mat_interleaved = ttnn.from_torch(trans_mat, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

# trans_mat must be HEIGHT_SHARDED for rotary_embedding_llama in decode mode
# Shard on 1 core: (TILE_SIZE, TILE_SIZE) = (32, 32)
trans_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, TILE_SIZE),
    core_grid=ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
trans_mat_tt = ttnn.to_memory_config(trans_mat_interleaved, trans_shard_cfg)
print(f"trans_mat: {trans_mat_tt.shape}, memory: {trans_mat_tt.memory_config().memory_layout}")

# HEIGHT_SHARD config for cos/sin (single core for batch=1)
cos_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, head_dim),
    core_grid=ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(grid.x, grid.y), row_wise=True),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


def lookup_cos_sin(pos, cos_table_tt, sin_table_tt):
    """Embedding lookup for cos/sin at position, returns HEIGHT_SHARDED 4D tensors."""
    pos_t = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.int32),
                            dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos_out = ttnn.embedding(pos_t, cos_table_tt, layout=ttnn.TILE_LAYOUT)
    sin_out = ttnn.embedding(pos_t, sin_table_tt, layout=ttnn.TILE_LAYOUT)

    # (1, 1, head_dim) -> (1, 1, 1, head_dim) -> transpose -> HEIGHT_SHARD
    cos_4d = ttnn.transpose(ttnn.unsqueeze_to_4D(cos_out), 1, 2)
    sin_4d = ttnn.transpose(ttnn.unsqueeze_to_4D(sin_out), 1, 2)
    cos_sh = ttnn.to_memory_config(cos_4d, cos_shard_cfg)
    sin_sh = ttnn.to_memory_config(sin_4d, cos_shard_cfg)
    return cos_sh, sin_sh


def make_q_sharded(q_input_np):
    """Create HEIGHT_SHARDED Q from QKV input via nlp_create_qkv_heads_decode."""
    qkv_tt = to_dev_4d(q_input_np)
    q_sh, k_sh, v_sh = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv_tt, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
    qkv_tt.deallocate()
    return q_sh, k_sh, v_sh


# ══════════════════════════════════════════════════════════════
# TEST 1: Embedding lookup — verify it works
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: ttnn.embedding cos/sin lookup")
print("=" * 60)

test_pos = 5

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
    cos_np = from_dev(cos_sh, (1, 1, 1, head_dim))
    expected = cos_interleaved_table[test_pos, :5]
    got = cos_np[0, 0, 0, :5]
    print(f"  Embedding lookup OK")
    print(f"    Expected: {expected}")
    print(f"    Got:      {got}")
    print(f"    Shape: {cos_sh.shape}, Memory: {cos_sh.memory_config().memory_layout}")
    cos_sh.deallocate(); sin_sh.deallocate()
    EMBEDDING_OK = True
except Exception as e:
    print(f"  Embedding lookup FAILED: {e}")
    EMBEDDING_OK = False


# ══════════════════════════════════════════════════════════════
# TEST 2: rotary_embedding_llama with INTERLEAVED cos/sin
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: rotary_embedding_llama format detection")
print("=" * 60)

np.random.seed(42)
qkv_dim = n_q_heads * head_dim + 2 * n_kv_heads * head_dim
q_input_np = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)

# IMPORTANT: nlp_create_qkv_heads_decode may reorder elements.
# We must get the ACTUAL Q values after the split, then apply numpy RoPE to those.
qkv_tmp = to_dev_4d(q_input_np)
q_tmp, k_tmp, v_tmp = ttnn.experimental.nlp_create_qkv_heads_decode(
    qkv_tmp, num_heads=n_q_heads, num_kv_heads=n_kv_heads,
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
q_after_split = from_dev(q_tmp, (1, 1, n_q_heads, head_dim))
print(f"  Q after split shape: {q_after_split.shape}")
print(f"  Q[0,:5] raw input:   {q_input_np[0,0,0,:5]}")
print(f"  Q[0,:5] after split: {q_after_split[0,0,0,:5]}")
q_tmp.deallocate(); k_tmp.deallocate(); v_tmp.deallocate(); qkv_tmp.deallocate()

# Numpy references computed on the actual post-split Q values
ref_interleaved = np.array([apply_rope_interleaved_np(q_after_split[0,0,h:h+1], test_pos) for h in range(n_q_heads)])
ref_half = np.array([apply_rope_half_np(q_after_split[0,0,h:h+1], test_pos) for h in range(n_q_heads)])
ref_interleaved = ref_interleaved.reshape(1, 1, n_q_heads, head_dim)
ref_half = ref_half.reshape(1, 1, n_q_heads, head_dim)

cos_cross = np.dot(ref_interleaved.flatten(), ref_half.flatten()) / (
    np.linalg.norm(ref_interleaved) * np.linalg.norm(ref_half) + 1e-8)
print(f"  Cross-format cosine (interleaved vs half): {cos_cross:.6f}")

# Test A: INTERLEAVED cos/sin + trans_mat (HEIGHT_SHARDED)
print("\n  Test A: interleaved cos/sin (HEIGHT_SHARDED) + trans_mat")
cos_vs_interleaved_A = -1; cos_vs_half_A = -1

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
    q_sh, k_sh, v_sh = make_q_sharded(q_input_np)

    q_roped = ttnn.experimental.rotary_embedding_llama(
        q_sh, cos_sh, sin_sh, trans_mat_tt, is_decode_mode=True)
    q_np = from_dev(q_roped, (1, 1, n_q_heads, head_dim))

    cos_vs_interleaved_A = np.dot(q_np.flatten(), ref_interleaved.flatten()) / (
        np.linalg.norm(q_np) * np.linalg.norm(ref_interleaved) + 1e-8)
    cos_vs_half_A = np.dot(q_np.flatten(), ref_half.flatten()) / (
        np.linalg.norm(q_np) * np.linalg.norm(ref_half) + 1e-8)

    print(f"    Cosine vs interleaved ref: {cos_vs_interleaved_A:.6f}")
    print(f"    Cosine vs half ref:        {cos_vs_half_A:.6f}")
    print(f"    head0[:8]: {q_np[0,0,0,:8]}")
    print(f"    ref_inter: {ref_interleaved[0,0,0,:8]}")
    print(f"    ref_half:  {ref_half[0,0,0,:8]}")

    if cos_vs_interleaved_A > 0.99:
        print(f"    --> MATCHES INTERLEAVED")
    elif cos_vs_half_A > 0.99:
        print(f"    --> MATCHES HALF")
    else:
        print(f"    --> NEITHER matches well")

    q_roped.deallocate(); q_sh.deallocate(); k_sh.deallocate(); v_sh.deallocate()
    cos_sh.deallocate(); sin_sh.deallocate()
    TEST_A_OK = True
except Exception as e:
    print(f"    FAILED: {str(e)[:200]}")
    TEST_A_OK = False

# Test B: HALF cos/sin + trans_mat (HEIGHT_SHARDED)
print("\n  Test B: half cos/sin (HEIGHT_SHARDED) + trans_mat")
cos_vs_interleaved_B = -1; cos_vs_half_B = -1

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_half_tt, sin_half_tt)
    q_sh, k_sh, v_sh = make_q_sharded(q_input_np)

    q_roped = ttnn.experimental.rotary_embedding_llama(
        q_sh, cos_sh, sin_sh, trans_mat_tt, is_decode_mode=True)
    q_np = from_dev(q_roped, (1, 1, n_q_heads, head_dim))

    cos_vs_interleaved_B = np.dot(q_np.flatten(), ref_interleaved.flatten()) / (
        np.linalg.norm(q_np) * np.linalg.norm(ref_interleaved) + 1e-8)
    cos_vs_half_B = np.dot(q_np.flatten(), ref_half.flatten()) / (
        np.linalg.norm(q_np) * np.linalg.norm(ref_half) + 1e-8)

    print(f"    Cosine vs interleaved ref: {cos_vs_interleaved_B:.6f}")
    print(f"    Cosine vs half ref:        {cos_vs_half_B:.6f}")
    print(f"    head0[:8]: {q_np[0,0,0,:8]}")

    if cos_vs_interleaved_B > 0.99:
        print(f"    --> MATCHES INTERLEAVED (with half tables!)")
    elif cos_vs_half_B > 0.99:
        print(f"    --> MATCHES HALF")
    else:
        print(f"    --> NEITHER matches well")

    q_roped.deallocate(); q_sh.deallocate(); k_sh.deallocate(); v_sh.deallocate()
    cos_sh.deallocate(); sin_sh.deallocate()
    TEST_B_OK = True
except Exception as e:
    print(f"    FAILED: {str(e)[:200]}")
    TEST_B_OK = False


# ══════════════════════════════════════════════════════════════
# TEST 3: Probe with identity inputs to trace rotation pattern
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Identity probe — trace which elements map where")
print("=" * 60)

# Send unit vectors through the op to see the exact rotation
# Head 0: e0 = [1, 0, 0, ..., 0]
# Head 1: e1 = [0, 1, 0, ..., 0]
# Head 2: e2 = [0, 0, 1, ..., 0]
# etc.
probe_input = np.zeros((1, 1, batch_size, qkv_dim), dtype=np.float32)
for h in range(min(4, n_q_heads)):
    probe_input[0, 0, 0, h * head_dim + h] = 1.0

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
    q_sh, k_sh, v_sh = make_q_sharded(probe_input)

    q_probe = ttnn.experimental.rotary_embedding_llama(
        q_sh, cos_sh, sin_sh, trans_mat_tt, is_decode_mode=True)
    q_p = from_dev(q_probe, (1, 1, n_q_heads, head_dim))

    a = test_pos * freqs
    cos_vals = np.cos(a)
    sin_vals = np.sin(a)

    for h in range(4):
        nz = np.nonzero(np.abs(q_p[0, 0, h]) > 0.001)[0]
        vals = [(int(i), float(q_p[0, 0, h, i])) for i in nz]
        print(f"  Head {h} (e{h} input): nonzero at {vals}")

    # Interpretation for INTERLEAVED:
    #   e0 -> out[0] = cos[0], out[1] = sin[0]    (pair 0,1)
    #   e1 -> out[0] = -sin[0], out[1] = cos[0]   (pair 0,1)
    #   e2 -> out[2] = cos[1], out[3] = sin[1]    (pair 2,3)
    #   e3 -> out[2] = -sin[1], out[3] = cos[1]   (pair 2,3)
    print(f"\n  Expected for INTERLEAVED:")
    print(f"    e0 -> [cos[0]={cos_vals[0]:.4f}, sin[0]={sin_vals[0]:.4f}, 0, ...]")
    print(f"    e1 -> [-sin[0]={-sin_vals[0]:.4f}, cos[0]={cos_vals[0]:.4f}, 0, ...]")
    print(f"    e2 -> [0, 0, cos[1]={cos_vals[1]:.4f}, sin[1]={sin_vals[1]:.4f}, ...]")
    print(f"    e3 -> [0, 0, -sin[1]={-sin_vals[1]:.4f}, cos[1]={cos_vals[1]:.4f}, ...]")

    # Interpretation for HALF:
    #   e0 -> out[0] = cos[0], out[32] = sin[0]
    #   e1 -> out[1] = cos[1], out[33] = sin[1]
    print(f"  Expected for HALF:")
    print(f"    e0 -> out[0]={cos_vals[0]:.4f}, out[32]={sin_vals[0]:.4f}")
    print(f"    e1 -> out[1]={cos_vals[1]:.4f}, out[33]={sin_vals[1]:.4f}")

    q_probe.deallocate(); q_sh.deallocate(); k_sh.deallocate(); v_sh.deallocate()
    cos_sh.deallocate(); sin_sh.deallocate()
except Exception as e:
    print(f"  FAILED: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# TEST 4: K-head RoPE (n_kv_heads=2, HEIGHT_SHARDED)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: K-head RoPE (n_kv_heads=2)")
print("=" * 60)

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
    np.random.seed(42)
    qkv_k = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)
    q_k, k_k, v_k = make_q_sharded(qkv_k)
    print(f"  K shape: {k_k.shape}, memory: {k_k.memory_config().memory_layout}")

    k_roped = ttnn.experimental.rotary_embedding_llama(
        k_k, cos_sh, sin_sh, trans_mat_tt, is_decode_mode=True)
    k_np = from_dev(k_roped, (1, 1, n_kv_heads, head_dim))
    print(f"  K RoPE OK: {k_roped.shape}")
    print(f"  K head0[:5]: {k_np[0,0,0,:5]}")

    k_roped.deallocate(); q_k.deallocate(); k_k.deallocate(); v_k.deallocate()
    cos_sh.deallocate(); sin_sh.deallocate()
except Exception as e:
    print(f"  K RoPE FAILED: {str(e)[:200]}")


# ══════════════════════════════════════════════════════════════
# TEST 5: Fused QK RoPE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 5: Fused QK RoPE")
print("=" * 60)

try:
    cos_sh, sin_sh = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
    np.random.seed(42)
    qkv_fused = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)
    q_f, k_f, v_f = make_q_sharded(qkv_fused)

    q_r, k_r = ttnn.experimental.rotary_embedding_llama_fused_qk(
        q_f, k_f, cos_sh, sin_sh, trans_mat_tt)
    print(f"  Fused QK RoPE OK: Q={q_r.shape}, K={k_r.shape}")
    q_r_np = from_dev(q_r, (1, 1, n_q_heads, head_dim))
    k_r_np = from_dev(k_r, (1, 1, n_kv_heads, head_dim))
    print(f"  Q head0[:5]: {q_r_np[0,0,0,:5]}")
    print(f"  K head0[:5]: {k_r_np[0,0,0,:5]}")

    q_r.deallocate(); k_r.deallocate()
    q_f.deallocate(); k_f.deallocate(); v_f.deallocate()
    cos_sh.deallocate(); sin_sh.deallocate()
    FUSED_OK = True
except Exception as e:
    print(f"  Fused QK FAILED: {str(e)[:200]}")
    FUSED_OK = False


# ══════════════════════════════════════════════════════════════
# TEST 6: Speed — native RoPE vs rotation matrix
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 6: Speed comparison")
print("=" * 60)

N_ITERS = 200

# --- Setup rotation matrix approach ---
R = np.zeros((head_dim, head_dim), dtype=np.float32)
for i in range(half_dim):
    R[i + half_dim, i] = -1.0
    R[i, i + half_dim] = 1.0
R_tt = to_dev(R)

a = test_pos * freqs
cos_full = np.concatenate([np.cos(a), np.cos(a)]).reshape(1, 1, 1, head_dim).astype(np.float32)
sin_full = np.concatenate([np.sin(a), np.sin(a)]).reshape(1, 1, 1, head_dim).astype(np.float32)
rope_cos_tt = to_dev_4d(cos_full)
rope_sin_tt = to_dev_4d(sin_full)

np.random.seed(42)
q_4d_np = np.random.randn(1, n_q_heads, 1, head_dim).astype(np.float32)
q_4d_tt = to_dev_4d(q_4d_np)

# Warmup rotation matrix
for _ in range(10):
    q_rot = ttnn.matmul(q_4d_tt, R_tt)
    q_roped_mat = ttnn.add(ttnn.mul(q_4d_tt, rope_cos_tt), ttnn.mul(q_rot, rope_sin_tt))
ttnn.synchronize_device(device)

t0 = time.perf_counter()
for _ in range(N_ITERS):
    q_rot = ttnn.matmul(q_4d_tt, R_tt)
    q_roped_mat = ttnn.add(ttnn.mul(q_4d_tt, rope_cos_tt), ttnn.mul(q_rot, rope_sin_tt))
ttnn.synchronize_device(device)
dt_matrix = (time.perf_counter() - t0) / N_ITERS * 1000
print(f"  Rotation matrix (Q only):  {dt_matrix:.3f}ms/iter")

# --- Setup native RoPE ---
# Pre-create HEIGHT_SHARDED cos/sin for benchmark
try:
    cos_bench, sin_bench = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)

    # Need to re-create Q each time since rotary_embedding_llama might consume it
    # But for benchmarking, let's see if we can reuse the input
    np.random.seed(42)
    qkv_bench = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)
    q_bench, k_bench, v_bench = make_q_sharded(qkv_bench)

    # Warmup
    for _ in range(10):
        q_r = ttnn.experimental.rotary_embedding_llama(
            q_bench, cos_bench, sin_bench, trans_mat_tt, is_decode_mode=True)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        q_r = ttnn.experimental.rotary_embedding_llama(
            q_bench, cos_bench, sin_bench, trans_mat_tt, is_decode_mode=True)
    ttnn.synchronize_device(device)
    dt_native = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"  Native RoPE (Q only):      {dt_native:.3f}ms/iter")
    print(f"  Speedup:                   {dt_matrix/dt_native:.2f}x")

    q_bench.deallocate(); k_bench.deallocate(); v_bench.deallocate()
    cos_bench.deallocate(); sin_bench.deallocate()
    NATIVE_SPEED_OK = True
except Exception as e:
    print(f"  Native RoPE benchmark FAILED: {str(e)[:150]}")
    dt_native = -1
    NATIVE_SPEED_OK = False

# --- Fused QK benchmark ---
if FUSED_OK:
    try:
        cos_bench2, sin_bench2 = lookup_cos_sin(test_pos, cos_interleaved_tt, sin_interleaved_tt)
        np.random.seed(42)
        qkv_b2 = np.random.randn(1, 1, batch_size, qkv_dim).astype(np.float32)
        q_b2, k_b2, v_b2 = make_q_sharded(qkv_b2)

        for _ in range(10):
            qr, kr = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_b2, k_b2, cos_bench2, sin_bench2, trans_mat_tt)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            qr, kr = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_b2, k_b2, cos_bench2, sin_bench2, trans_mat_tt)
        ttnn.synchronize_device(device)
        dt_fused = (time.perf_counter() - t0) / N_ITERS * 1000
        print(f"  Fused QK RoPE (Q+K):       {dt_fused:.3f}ms/iter")
        print(f"  Speedup vs 2x matrix:      {(dt_matrix*2)/dt_fused:.2f}x")

        q_b2.deallocate(); k_b2.deallocate(); v_b2.deallocate()
        cos_bench2.deallocate(); sin_bench2.deallocate()
    except Exception as e:
        print(f"  Fused QK benchmark FAILED: {str(e)[:150]}")


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nFormat detection:")
if TEST_A_OK:
    print(f"  Test A (interleaved cos/sin): cos_inter={cos_vs_interleaved_A:.4f}, cos_half={cos_vs_half_A:.4f}")
if TEST_B_OK:
    print(f"  Test B (half cos/sin):        cos_inter={cos_vs_interleaved_B:.4f}, cos_half={cos_vs_half_B:.4f}")

print(f"\nEmbedding lookup: {'OK' if EMBEDDING_OK else 'FAILED'}")
print(f"Fused QK RoPE: {'OK' if FUSED_OK else 'FAILED'}")

if NATIVE_SPEED_OK:
    print(f"\nSpeed ({N_ITERS} iters):")
    print(f"  Rotation matrix: {dt_matrix:.3f}ms")
    print(f"  Native RoPE:     {dt_native:.3f}ms ({dt_matrix/dt_native:.2f}x)")

print(f"\nConclusion:")
if TEST_A_OK and cos_vs_interleaved_A > 0.99:
    print(f"  rotary_embedding_llama implements INTERLEAVED rotation (adjacent pairs).")
    print(f"  For Qwen: use interleaved cos/sin tables with this op.")
    print(f"  The rotation format is baked into trans_mat, not the model weights.")
elif TEST_A_OK and cos_vs_half_A > 0.99:
    print(f"  rotary_embedding_llama implements HALF rotation (midpoint split).")
    print(f"  This matches Qwen natively.")
elif TEST_B_OK and cos_vs_half_B > 0.99:
    print(f"  rotary_embedding_llama + half cos/sin = HALF rotation.")
else:
    print(f"  Format unclear — check probe results above.")

ttnn.close_device(device)
print("\nDone!")
