"""
Experiment 37: Single Qwen2.5-0.5B transformer layer on Blackhole.

Loads real weights from HuggingFace, runs a full layer (attention + MLP)
on TT-NN, compares against a NumPy reference implementation.

Layer structure:
  residual = x
  x = RMSNorm(x) -> Q/K/V projections -> RoPE(Q,K) -> GQA SDPA -> O proj
  x = residual + x
  residual = x
  x = RMSNorm(x) -> gate_proj, up_proj -> SwiGLU -> down_proj
  x = residual + x
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import time
import torch

from safetensors import safe_open
from huggingface_hub import hf_hub_download

import ttnn
from tt_jax import tensors

# ── Model config ────────────────────────────────────────────
hidden = 896
intermediate = 4864
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
rms_eps = 1e-6
rope_theta = 1000000.0
seq_len = 32
batch = 1
layer_idx = 0

# ── Load weights ────────────────────────────────────────────
print("Downloading Qwen2.5-0.5B weights...")
model_path = hf_hub_download("Qwen/Qwen2.5-0.5B", "model.safetensors")
print(f"  Model path: {model_path}")

weights = {}
with safe_open(model_path, framework="numpy") as f:
    prefix = f"model.layers.{layer_idx}."
    for key in f.keys():
        if key.startswith(prefix):
            short = key[len(prefix):]
            weights[short] = f.get_tensor(key)

print(f"  Loaded {len(weights)} tensors for layer {layer_idx}:")
for k, v in sorted(weights.items()):
    print(f"    {k}: {v.shape} {v.dtype}")

# ── Extract weight arrays ───────────────────────────────────
# HuggingFace stores (out_features, in_features), we need (in, out) for x @ W
ln1_g = weights["input_layernorm.weight"].astype(np.float32)         # (896,)
q_w = weights["self_attn.q_proj.weight"].astype(np.float32).T        # (896, 896)
q_b = weights["self_attn.q_proj.bias"].astype(np.float32)            # (896,)
k_w = weights["self_attn.k_proj.weight"].astype(np.float32).T        # (896, 128)
k_b = weights["self_attn.k_proj.bias"].astype(np.float32)            # (128,)
v_w = weights["self_attn.v_proj.weight"].astype(np.float32).T        # (896, 128)
v_b = weights["self_attn.v_proj.bias"].astype(np.float32)            # (128,)
o_w = weights["self_attn.o_proj.weight"].astype(np.float32).T        # (896, 896)
ln2_g = weights["post_attention_layernorm.weight"].astype(np.float32) # (896,)
gate_w = weights["mlp.gate_proj.weight"].astype(np.float32).T        # (896, 4864)
up_w = weights["mlp.up_proj.weight"].astype(np.float32).T            # (896, 4864)
down_w = weights["mlp.down_proj.weight"].astype(np.float32).T        # (4864, 896)

print("\nWeight shapes after transpose:")
print(f"  q_w: {q_w.shape}, k_w: {k_w.shape}, v_w: {v_w.shape}, o_w: {o_w.shape}")
print(f"  gate_w: {gate_w.shape}, up_w: {up_w.shape}, down_w: {down_w.shape}")

# ── NumPy reference ─────────────────────────────────────────

def rms_norm_np(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma

def apply_rope_np(x, seq_len, head_dim, base=1000000.0):
    """x: (1, n_heads, seq_len, head_dim)"""
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    angles = np.outer(positions, freqs)
    cos_t = np.cos(angles)[None, None, :, :]
    sin_t = np.sin(angles)[None, None, :, :]
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = np.zeros_like(x)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out

def sdpa_np(q, k, v, causal=True):
    """Scaled dot-product attention. q: (B,Hq,T,D), k/v: (B,Hkv,T,D)"""
    # Expand KV heads for GQA
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=1)
        v = np.repeat(v, n_rep, axis=1)
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    if causal:
        T = scores.shape[-1]
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
        scores = scores + mask
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return np.matmul(attn, v)

def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def qwen_layer_np(x, weights_dict):
    """Full Qwen2.5 layer in NumPy. x: (1, seq_len, hidden)"""
    B, T, D = x.shape

    # --- Self-attention block ---
    residual = x.copy()

    # RMSNorm
    h = rms_norm_np(x, ln1_g, rms_eps)

    # Q/K/V projections
    q = h @ q_w + q_b  # (1, T, 896)
    k = h @ k_w + k_b  # (1, T, 128)
    v = h @ v_w + v_b  # (1, T, 128)

    # Reshape to head format
    q = q.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)   # (1, 14, T, 64)
    k = k.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)  # (1, 2, T, 64)
    v = v.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)  # (1, 2, T, 64)

    # RoPE on Q and K
    q = apply_rope_np(q, T, head_dim, rope_theta)
    k = apply_rope_np(k, T, head_dim, rope_theta)

    # GQA SDPA
    attn_out = sdpa_np(q, k, v, causal=True)  # (1, 14, T, 64)

    # Concat heads and output projection
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)  # (1, T, 896)
    attn_out = attn_out @ o_w  # (1, T, 896)

    x = residual + attn_out

    # --- MLP block ---
    residual = x.copy()

    h = rms_norm_np(x, ln2_g, rms_eps)

    gate = h @ gate_w  # (1, T, 4864)
    up = h @ up_w      # (1, T, 4864)
    mlp_out = silu_np(gate) * up
    mlp_out = mlp_out @ down_w  # (1, T, 896)

    x = residual + mlp_out
    return x


# ── Run NumPy reference ─────────────────────────────────────
rng = np.random.RandomState(42)
x_np = rng.randn(batch, seq_len, hidden).astype(np.float32) * 0.02

print("\n" + "=" * 60)
print("Running NumPy reference...")
print("=" * 60)
t0 = time.perf_counter()
ref_out = qwen_layer_np(x_np, weights)
t_np = time.perf_counter() - t0
print(f"  NumPy time: {t_np*1000:.1f}ms")
print(f"  Output shape: {ref_out.shape}, norm: {np.linalg.norm(ref_out):.4f}")


# ── TT-NN implementation ───────────────────────────────────
print("\n" + "=" * 60)
print("Running TT-NN implementation...")
print("=" * 60)

device = ttnn.open_device(device_id=0)

def to_dev(arr):
    """Send numpy array to device as bfloat16 tile-layout tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    """Send 4D numpy array to device."""
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    """Retrieve tensor from device as numpy."""
    t = ttnn.to_torch(tensor).float()
    try:
        return t.reshape(shape).numpy()
    except RuntimeError:
        return t.squeeze().numpy().reshape(shape)

def cosine(a, b):
    return np.dot(a.flatten(), b.flatten()) / (
        np.linalg.norm(a.flatten()) * np.linalg.norm(b.flatten()) + 1e-8)


# Preload all weights to device
print("  Loading weights to device...")
ln1_g_tt = to_dev(ln1_g)
q_w_tt = to_dev(q_w)
q_b_tt = to_dev(q_b)
k_w_tt = to_dev(k_w)
k_b_tt = to_dev(k_b)
v_w_tt = to_dev(v_w)
v_b_tt = to_dev(v_b)
o_w_tt = to_dev(o_w)
ln2_g_tt = to_dev(ln2_g)
gate_w_tt = to_dev(gate_w)
up_w_tt = to_dev(up_w)
down_w_tt = to_dev(down_w)

# Precompute RoPE cos/sin tables
freqs = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
positions = np.arange(seq_len, dtype=np.float32)
angles = np.outer(positions, freqs)  # (seq_len, head_dim/2)
cos_table = np.cos(angles).astype(np.float32)  # (seq_len, 32)
sin_table = np.sin(angles).astype(np.float32)  # (seq_len, 32)

print("  All weights on device.")


def apply_rope_ttnn(x_4d_np, n_heads):
    """Apply RoPE via CPU (decomposed). x: (1, n_heads, seq_len, head_dim) numpy."""
    # Even/odd decomposition
    x_even = x_4d_np[..., 0::2]  # (1, H, T, 32)
    x_odd = x_4d_np[..., 1::2]

    cos_t = cos_table[None, None, :, :]  # (1, 1, T, 32)
    sin_t = sin_table[None, None, :, :]

    out = np.zeros_like(x_4d_np)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out


def qwen_layer_ttnn(x_np):
    """Full Qwen2.5 layer on TT-NN. x_np: (1, seq_len, hidden) numpy input."""
    B, T, D = x_np.shape

    x_tt = to_dev(x_np)

    # --- Self-attention block ---
    # RMSNorm (on device)
    h_tt = ttnn.rms_norm(x_tt, weight=ln1_g_tt, epsilon=rms_eps)

    # Q/K/V projections (on device)
    q_tt = ttnn.add(ttnn.matmul(h_tt, q_w_tt), q_b_tt)   # (1, T, 896)
    k_tt = ttnn.add(ttnn.matmul(h_tt, k_w_tt), k_b_tt)   # (1, T, 128)
    v_tt = ttnn.add(ttnn.matmul(h_tt, v_w_tt), v_b_tt)   # (1, T, 128)

    # Pull Q/K to CPU for reshape + RoPE, V for reshape
    q_np = from_dev(q_tt, (B, T, n_q_heads * head_dim))
    k_np = from_dev(k_tt, (B, T, n_kv_heads * head_dim))
    v_np = from_dev(v_tt, (B, T, n_kv_heads * head_dim))

    # Reshape to head format
    q_4d = q_np.reshape(B, T, n_q_heads, head_dim).transpose(0, 2, 1, 3)
    k_4d = k_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_4d = v_np.reshape(B, T, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # RoPE on Q and K (CPU — decomposed)
    q_4d = apply_rope_ttnn(q_4d, n_q_heads)
    k_4d = apply_rope_ttnn(k_4d, n_kv_heads)

    # Send back to device for SDPA
    q_dev = to_dev_4d(q_4d)
    k_dev = to_dev_4d(k_4d)
    v_dev = to_dev_4d(v_4d)

    # GQA SDPA (on device — native GQA support: 14 Q heads, 2 KV heads)
    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_dev, k_dev, v_dev, is_causal=True
    )

    # Pull back, concat heads, output projection
    attn_out_np = from_dev(attn_out_tt, (B, n_q_heads, T, head_dim))
    attn_out_np = attn_out_np.transpose(0, 2, 1, 3).reshape(B, T, D)

    attn_out_tt = to_dev(attn_out_np)
    o_tt = ttnn.matmul(attn_out_tt, o_w_tt)  # (1, T, 896)

    # Residual add (on device)
    x_tt2 = ttnn.add(to_dev(x_np), o_tt)

    # --- MLP block ---
    # RMSNorm
    h2_tt = ttnn.rms_norm(x_tt2, weight=ln2_g_tt, epsilon=rms_eps)

    # Gate and up projections
    gate_tt = ttnn.matmul(h2_tt, gate_w_tt)  # (1, T, 4864)
    up_tt = ttnn.matmul(h2_tt, up_w_tt)      # (1, T, 4864)

    # SwiGLU: silu(gate) * up (on device)
    swiglu_tt = ttnn.mul(ttnn.silu(gate_tt), up_tt)

    # Down projection
    down_tt = ttnn.matmul(swiglu_tt, down_w_tt)  # (1, T, 896)

    # Residual add
    out_tt = ttnn.add(x_tt2, down_tt)

    return from_dev(out_tt, (B, T, D))


# ── Run TT-NN ──────────────────────────────────────────────
print("\n  Running TT-NN forward pass...")
t0 = time.perf_counter()
tt_out = qwen_layer_ttnn(x_np)
t_tt = time.perf_counter() - t0
print(f"  TT-NN time: {t_tt*1000:.1f}ms (includes CPU round-trips for RoPE/reshape)")
print(f"  Output shape: {tt_out.shape}, norm: {np.linalg.norm(tt_out):.4f}")

# ── Compare ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Comparison: TT-NN vs NumPy reference")
print("=" * 60)

cos_sim = cosine(tt_out, ref_out)
max_err = np.abs(tt_out - ref_out).max()
mean_err = np.abs(tt_out - ref_out).mean()
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max absolute error: {max_err:.6f}")
print(f"  Mean absolute error: {mean_err:.6f}")

# Per-token cosine similarity
print("\n  Per-token cosine similarity (first 8 tokens):")
for t in range(min(8, seq_len)):
    tok_cos = cosine(tt_out[0, t], ref_out[0, t])
    print(f"    Token {t:2d}: {tok_cos:.6f}")

# ── Attention block vs MLP block breakdown ──────────────────
print("\n" + "=" * 60)
print("Breakdown: attention vs MLP accuracy")
print("=" * 60)

# Run attention block only (numpy)
residual = x_np.copy()
h = rms_norm_np(x_np, ln1_g, rms_eps)
q = (h @ q_w + q_b).reshape(1, seq_len, n_q_heads, head_dim).transpose(0, 2, 1, 3)
k = (h @ k_w + k_b).reshape(1, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
v = (h @ v_w + v_b).reshape(1, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
q = apply_rope_np(q, seq_len, head_dim, rope_theta)
k = apply_rope_np(k, seq_len, head_dim, rope_theta)
attn = sdpa_np(q, k, v, causal=True)
attn = attn.transpose(0, 2, 1, 3).reshape(1, seq_len, hidden) @ o_w
attn_ref = residual + attn

# Run attention block only (TT-NN) — reuse the existing path
h_tt = ttnn.rms_norm(to_dev(x_np), weight=ln1_g_tt, epsilon=rms_eps)
q_tt = ttnn.add(ttnn.matmul(h_tt, q_w_tt), q_b_tt)
k_tt_out = ttnn.add(ttnn.matmul(h_tt, k_w_tt), k_b_tt)
v_tt_out = ttnn.add(ttnn.matmul(h_tt, v_w_tt), v_b_tt)

q_np2 = from_dev(q_tt, (1, seq_len, n_q_heads * head_dim))
k_np2 = from_dev(k_tt_out, (1, seq_len, n_kv_heads * head_dim))
v_np2 = from_dev(v_tt_out, (1, seq_len, n_kv_heads * head_dim))

q_4d = apply_rope_ttnn(q_np2.reshape(1, seq_len, n_q_heads, head_dim).transpose(0, 2, 1, 3), n_q_heads)
k_4d = apply_rope_ttnn(k_np2.reshape(1, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3), n_kv_heads)
v_4d = v_np2.reshape(1, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

attn_tt = ttnn.transformer.scaled_dot_product_attention(
    to_dev_4d(q_4d), to_dev_4d(k_4d), to_dev_4d(v_4d), is_causal=True
)
attn_np = from_dev(attn_tt, (1, n_q_heads, seq_len, head_dim))
attn_np = attn_np.transpose(0, 2, 1, 3).reshape(1, seq_len, hidden)
o_tt = ttnn.matmul(to_dev(attn_np), o_w_tt)
attn_block_out = from_dev(ttnn.add(to_dev(x_np), o_tt), (1, seq_len, hidden))

attn_cos = cosine(attn_block_out, attn_ref)
print(f"  Attention block cosine: {attn_cos:.6f}")

# MLP block accuracy (feed attn_ref through MLP)
residual2 = attn_ref.copy()
h2 = rms_norm_np(attn_ref, ln2_g, rms_eps)
mlp_ref = residual2 + silu_np(h2 @ gate_w) * (h2 @ up_w) @ down_w

h2_tt = ttnn.rms_norm(to_dev(attn_ref), weight=ln2_g_tt, epsilon=rms_eps)
gate_tt2 = ttnn.matmul(h2_tt, gate_w_tt)
up_tt2 = ttnn.matmul(h2_tt, up_w_tt)
swiglu_tt2 = ttnn.mul(ttnn.silu(gate_tt2), up_tt2)
down_tt2 = ttnn.matmul(swiglu_tt2, down_w_tt)
mlp_block_out = from_dev(ttnn.add(to_dev(attn_ref), down_tt2), (1, seq_len, hidden))

mlp_cos = cosine(mlp_block_out, mlp_ref)
print(f"  MLP block cosine: {mlp_cos:.6f}")

# ── Timing benchmark ────────────────────────────────────────
print("\n" + "=" * 60)
print("Timing benchmark (10 iterations)")
print("=" * 60)

times = []
for i in range(10):
    t0 = time.perf_counter()
    _ = qwen_layer_ttnn(x_np)
    times.append(time.perf_counter() - t0)

times_ms = [t * 1000 for t in times]
print(f"  Mean: {np.mean(times_ms):.1f}ms")
print(f"  Min:  {np.min(times_ms):.1f}ms")
print(f"  Max:  {np.max(times_ms):.1f}ms")
print(f"  Throughput: {1000.0 / np.mean(times_ms):.0f} layer fwd/sec")

# ── Summary ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
  Model:           Qwen2.5-0.5B layer 0
  Input shape:     ({batch}, {seq_len}, {hidden})
  Cosine sim:      {cos_sim:.6f}
  Max error:       {max_err:.6f}
  Attn block cos:  {attn_cos:.6f}
  MLP block cos:   {mlp_cos:.6f}
  Mean latency:    {np.mean(times_ms):.1f}ms

  Status: {'PASS' if cos_sim > 0.99 else 'NEEDS INVESTIGATION'} (threshold: 0.99)
""")

ttnn.close_device(device)
print("Done!")
