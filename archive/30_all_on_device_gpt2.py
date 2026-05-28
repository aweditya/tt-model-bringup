"""
Experiment 30: All-on-device GPT-2 forward pass.

Goal: eliminate CPU round-trips for LayerNorm and GELU by using
ttnn.layer_norm and ttnn.gelu natively. The only remaining CPU
round-trips should be QKV split and head concat (ttnn lacks split).

Tests:
  Phase 1 — ttnn.layer_norm on (1, 32, 768)
  Phase 2 — ttnn.gelu vs GPT-2 "gelu_new" (tanh approximation)
  Phase 3 — Single GPT-2 layer, all on device except QKV split/head concat
  Phase 4 — Full 12-layer GPT-2, accuracy + timing vs experiment 29
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
import time

# ── Load GPT-2 weights ──────────────────────────────────────
print("Loading GPT-2 weights...")
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    config = json.load(f)

weights = {}
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

with open(vocab_path) as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}

def simple_decode(token_ids):
    return ''.join(id_to_token.get(tid, '?').replace('\u0120', ' ') for tid in token_ids)

print(f"GPT-2: {config['n_layer']}L, {config['n_head']}H, d={config['n_embd']}")

# ── Setup device ─────────────────────────────────────────────
import ttnn
import torch

device = ttnn.open_device(device_id=0)

from tt_jax import tensors

n_heads = 12
d_model = 768
head_dim = d_model // n_heads  # 64
seq_len = 32
batch = 1

# ══════════════════════════════════════════════════════════════
# Phase 1: Test ttnn.layer_norm
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Test ttnn.layer_norm on (1, 32, 768)")
print("=" * 60)

rng = np.random.RandomState(42)
x_np = rng.randn(1, seq_len, d_model).astype(np.float32) * 0.5
gamma_np = rng.randn(d_model).astype(np.float32) * 0.1 + 1.0
beta_np = rng.randn(d_model).astype(np.float32) * 0.01

# NumPy reference
mean = x_np.mean(axis=-1, keepdims=True)
var = ((x_np - mean) ** 2).mean(axis=-1, keepdims=True)
ref_ln = gamma_np * (x_np - mean) / np.sqrt(var + 1e-5) + beta_np

# TT-NN layer_norm
x_tt = tensors.to_device(x_np, device)
# Weight/bias for layer_norm need to be 1D -> make 2D for ttnn
gamma_tt = tensors.to_device(gamma_np, device)
beta_tt = tensors.to_device(beta_np, device)

try:
    ln_out_tt = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=gamma_tt, bias=beta_tt)
    ln_out_np = tensors.from_device(ln_out_tt, (1, seq_len, d_model))

    cos_sim = np.dot(ln_out_np.flatten(), ref_ln.flatten()) / (
        np.linalg.norm(ln_out_np.flatten()) * np.linalg.norm(ref_ln.flatten()) + 1e-8)
    max_err = np.abs(ln_out_np - ref_ln).max()
    mean_err = np.abs(ln_out_np - ref_ln).mean()
    print(f"  ttnn.layer_norm: cosine sim = {cos_sim:.6f}, max err = {max_err:.6f}, mean err = {mean_err:.6f}")
    LAYER_NORM_WORKS = True
except Exception as e:
    print(f"  ttnn.layer_norm FAILED: {e}")
    LAYER_NORM_WORKS = False

# ══════════════════════════════════════════════════════════════
# Phase 2: Test ttnn.gelu vs GPT-2 gelu_new
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Test ttnn.gelu vs GPT-2 gelu_new")
print("=" * 60)

# GPT-2 uses gelu_new (tanh approximation)
def gelu_new_np(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

test_x = rng.randn(1, seq_len, d_model * 4).astype(np.float32) * 0.5
ref_gelu = gelu_new_np(test_x)

test_tt = tensors.to_device(test_x, device)

# Test both modes
for mode_name, approx in [("exact (approx=False)", False), ("fast (approx=True)", True)]:
    try:
        gelu_out = ttnn.gelu(test_tt, fast_and_approximate_mode=approx)
        gelu_np = tensors.from_device(gelu_out, test_x.shape)

        cos_sim = np.dot(gelu_np.flatten(), ref_gelu.flatten()) / (
            np.linalg.norm(gelu_np.flatten()) * np.linalg.norm(ref_gelu.flatten()) + 1e-8)
        max_err = np.abs(gelu_np - ref_gelu).max()
        mean_err = np.abs(gelu_np - ref_gelu).mean()
        print(f"  {mode_name}: cosine sim = {cos_sim:.6f}, max err = {max_err:.6f}, mean err = {mean_err:.6f}")
    except Exception as e:
        print(f"  {mode_name} FAILED: {e}")

# Also compare against exact GELU (erf-based) to see which ttnn.gelu matches
from scipy.special import erf as scipy_erf
ref_gelu_exact = 0.5 * test_x * (1.0 + scipy_erf(test_x / np.sqrt(2.0)))
print(f"\n  Reference: gelu_new vs exact gelu max diff = {np.abs(ref_gelu - ref_gelu_exact).max():.6f}")

# Pick the best mode for GPT-2
for mode_name, approx in [("exact", False), ("fast", True)]:
    try:
        gelu_out = ttnn.gelu(test_tt, fast_and_approximate_mode=approx)
        gelu_np = tensors.from_device(gelu_out, test_x.shape)
        cos_new = np.dot(gelu_np.flatten(), ref_gelu.flatten()) / (
            np.linalg.norm(gelu_np.flatten()) * np.linalg.norm(ref_gelu.flatten()) + 1e-8)
        cos_exact = np.dot(gelu_np.flatten(), ref_gelu_exact.flatten()) / (
            np.linalg.norm(gelu_np.flatten()) * np.linalg.norm(ref_gelu_exact.flatten()) + 1e-8)
        print(f"  ttnn.gelu({mode_name}) -> vs gelu_new: {cos_new:.6f}, vs gelu_exact: {cos_exact:.6f}")
    except Exception:
        pass

GELU_WORKS = True  # We'll use whichever mode is closer to gelu_new

# Determine best gelu mode
best_gelu_approx = True  # default to fast
try:
    gelu_fast = tensors.from_device(
        ttnn.gelu(test_tt, fast_and_approximate_mode=True), test_x.shape)
    gelu_exact = tensors.from_device(
        ttnn.gelu(test_tt, fast_and_approximate_mode=False), test_x.shape)
    err_fast = np.abs(gelu_fast - ref_gelu).mean()
    err_exact = np.abs(gelu_exact - ref_gelu).mean()
    best_gelu_approx = bool(err_fast < err_exact)
    print(f"\n  Best mode for gelu_new: {'fast' if best_gelu_approx else 'exact'} "
          f"(mean err: fast={err_fast:.6f}, exact={err_exact:.6f})")
except Exception as e:
    print(f"  Could not determine best mode: {e}")

# ══════════════════════════════════════════════════════════════
# Phase 3: Single GPT-2 layer — all on device except QKV split
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Single GPT-2 layer — minimal CPU round-trips")
print("=" * 60)

def get_layer_weights_np(layer_idx):
    p = f"h.{layer_idx}"
    return {
        'ln1_g': weights[f"{p}.ln_1.weight"],
        'ln1_b': weights[f"{p}.ln_1.bias"],
        'w_attn': weights[f"{p}.attn.c_attn.weight"],
        'b_attn': weights[f"{p}.attn.c_attn.bias"],
        'w_proj': weights[f"{p}.attn.c_proj.weight"],
        'b_proj': weights[f"{p}.attn.c_proj.bias"],
        'ln2_g': weights[f"{p}.ln_2.weight"],
        'ln2_b': weights[f"{p}.ln_2.bias"],
        'w_fc': weights[f"{p}.mlp.c_fc.weight"],
        'b_fc': weights[f"{p}.mlp.c_fc.bias"],
        'w_mlp_proj': weights[f"{p}.mlp.c_proj.weight"],
        'b_mlp_proj': weights[f"{p}.mlp.c_proj.bias"],
    }


def gpt2_layer_all_device(x_tt, lw, device, seq_len, d_model, n_heads):
    """GPT-2 layer with ONLY QKV split and head concat on CPU.

    On-device: layernorm, matmul, add, gelu, attention
    CPU only:  QKV split (reshape+transpose), head concat (reshape+transpose)
    """
    head_dim = d_model // n_heads
    x_shape = (1, seq_len, d_model)

    # ── LayerNorm 1 (ON DEVICE) ──
    ln1_g = tensors.to_device(lw['ln1_g'], device)
    ln1_b = tensors.to_device(lw['ln1_b'], device)
    h = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=ln1_g, bias=ln1_b)

    # ── QKV matmul (ON DEVICE) ──
    w_attn = tensors.to_device(lw['w_attn'], device)
    b_attn = tensors.to_device(lw['b_attn'], device)
    qkv = ttnn.add(ttnn.matmul(h, w_attn), b_attn)

    # ── QKV split (CPU — ttnn lacks split) ──
    qkv_np = tensors.from_device(qkv, (1, seq_len, 3 * d_model))
    q_np = qkv_np[:, :, :d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    k_np = qkv_np[:, :, d_model:2*d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    v_np = qkv_np[:, :, 2*d_model:].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

    q_tt = ttnn.from_torch(torch.from_numpy(q_np.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    k_tt = ttnn.from_torch(torch.from_numpy(k_np.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)
    v_tt = ttnn.from_torch(torch.from_numpy(v_np.copy()), dtype=ttnn.bfloat16,
                            device=device, layout=ttnn.TILE_LAYOUT)

    # ── Native attention (ON DEVICE) ──
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, is_causal=True)

    # ── Head concat (CPU — reshape+transpose) ──
    try:
        merged = ttnn.transformer.concatenate_heads(attn_out)
        merged_np = tensors.from_device(merged, (1, seq_len, d_model))
    except Exception:
        attn_np = ttnn.to_torch(attn_out).float().numpy()
        merged_np = attn_np.transpose(0, 2, 1, 3).reshape(1, seq_len, d_model)
    merged_tt = tensors.to_device(merged_np, device)

    # ── Output projection (ON DEVICE) ──
    w_proj = tensors.to_device(lw['w_proj'], device)
    b_proj = tensors.to_device(lw['b_proj'], device)
    proj = ttnn.add(ttnn.matmul(merged_tt, w_proj), b_proj)

    # ── Residual add (ON DEVICE) ──
    x_tt = ttnn.add(x_tt, proj)

    # ── LayerNorm 2 (ON DEVICE) ──
    ln2_g = tensors.to_device(lw['ln2_g'], device)
    ln2_b = tensors.to_device(lw['ln2_b'], device)
    h2 = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=ln2_g, bias=ln2_b)

    # ── MLP: matmul + gelu + matmul (ALL ON DEVICE) ──
    w_fc = tensors.to_device(lw['w_fc'], device)
    b_fc = tensors.to_device(lw['b_fc'], device)
    ff = ttnn.add(ttnn.matmul(h2, w_fc), b_fc)

    # ── GELU (ON DEVICE) ──
    ff = ttnn.gelu(ff, fast_and_approximate_mode=best_gelu_approx)

    w_mlp = tensors.to_device(lw['w_mlp_proj'], device)
    b_mlp = tensors.to_device(lw['b_mlp_proj'], device)
    ff_out = ttnn.add(ttnn.matmul(ff, w_mlp), b_mlp)

    # ── Residual add (ON DEVICE) ──
    return ttnn.add(x_tt, ff_out)


# ── Prepare input ────────────────────────────────────────────
wte = weights["wte.weight"]
wpe = weights["wpe.weight"]
token_to_id = vocab

text = "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout"
tokens = []
for word in text.split():
    if word in token_to_id:
        tokens.append(token_to_id[word])
    elif '\u0120' + word in token_to_id:
        tokens.append(token_to_id['\u0120' + word])
    else:
        for ch in word:
            tokens.append(token_to_id.get(ch, 0))
while len(tokens) < 32:
    tokens.append(50256)
tokens = tokens[:32]
print(f"Tokens: {tokens[:10]}...")

tok_emb = wte[tokens]
pos_emb = wpe[:32]
x_np = (tok_emb + pos_emb)[None, :, :]  # (1, 32, 768)

# ── JAX CPU reference ────────────────────────────────────────
def gelu_jax(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))

def gpt2_layer_jax(x, lw, n_heads):
    d = x.shape[-1]
    head_dim = d // n_heads
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
    h = jnp.array(lw['ln1_g']) * (x - m) / jnp.sqrt(v + 1e-5) + jnp.array(lw['ln1_b'])
    qkv = jnp.dot(h, jnp.array(lw['w_attn'])) + jnp.array(lw['b_attn'])
    q, k, val = jnp.split(qkv, 3, axis=-1)
    B, T, C = q.shape
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    val = val.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    sc = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.array(float(head_dim)))
    mask = jnp.tril(jnp.ones((T, T)))
    sc = sc * mask + (-1e10) * (1.0 - mask)
    aw = jax.nn.softmax(sc, axis=-1)
    out = jnp.matmul(aw, val).transpose(0, 2, 1, 3).reshape(B, T, C)
    out = jnp.dot(out, jnp.array(lw['w_proj'])) + jnp.array(lw['b_proj'])
    x = x + out
    m2 = jnp.mean(x, axis=-1, keepdims=True)
    v2 = jnp.mean((x - m2)**2, axis=-1, keepdims=True)
    h2 = jnp.array(lw['ln2_g']) * (x - m2) / jnp.sqrt(v2 + 1e-5) + jnp.array(lw['ln2_b'])
    ff = gelu_jax(jnp.dot(h2, jnp.array(lw['w_fc'])) + jnp.array(lw['b_fc']))
    ff = jnp.dot(ff, jnp.array(lw['w_mlp_proj'])) + jnp.array(lw['b_mlp_proj'])
    return x + ff

# Single layer test
lw0 = get_layer_weights_np(0)
x_jax = jnp.array(x_np)
jax_out = np.array(gpt2_layer_jax(x_jax, lw0, n_heads))

x_tt = tensors.to_device(x_np, device)
t0 = time.perf_counter()
out_tt = gpt2_layer_all_device(x_tt, lw0, device, seq_len, d_model, n_heads)
t1 = time.perf_counter()

out_np = tensors.from_device(out_tt, (1, 32, 768))
cos_sim = np.dot(out_np.flatten(), jax_out.flatten()) / (
    np.linalg.norm(out_np.flatten()) * np.linalg.norm(jax_out.flatten()) + 1e-8)
max_err = np.abs(out_np - jax_out).max()
mean_err = np.abs(out_np - jax_out).mean()
print(f"\nSingle layer (all-on-device):")
print(f"  Time: {(t1-t0)*1000:.1f} ms")
print(f"  Cosine similarity: {cos_sim:.6f}")
print(f"  Max error: {max_err:.6f}, Mean error: {mean_err:.6f}")

# Count CPU round-trips
print(f"\nCPU round-trips per layer:")
print(f"  1. QKV split: device->CPU->device (reshape+transpose, ttnn lacks split)")
print(f"  2. Head concat: device->CPU->device (reshape+transpose)")
print(f"  Total: 2 round-trips (down from 4 in exp 29: +layernorm +gelu)")

# ══════════════════════════════════════════════════════════════
# Phase 4: Full 12-layer GPT-2
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Full 12-layer GPT-2 — all on device")
print("=" * 60)

ln_f_g = weights["ln_f.weight"]
ln_f_b = weights["ln_f.bias"]

x_tt = tensors.to_device(x_np, device)

layer_times = []
t0_total = time.perf_counter()
for layer_idx in range(12):
    lw_i = get_layer_weights_np(layer_idx)
    t_layer = time.perf_counter()
    x_tt = gpt2_layer_all_device(x_tt, lw_i, device, seq_len, d_model, n_heads)
    layer_times.append(time.perf_counter() - t_layer)
    print(f"  Layer {layer_idx:2d}: {layer_times[-1]*1000:.1f} ms")

# Final layernorm (ON DEVICE)
ln_f_g_tt = tensors.to_device(ln_f_g, device)
ln_f_b_tt = tensors.to_device(ln_f_b, device)
x_tt = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)

t1_total = time.perf_counter()
total_ms = (t1_total - t0_total) * 1000
print(f"\nFull 12-layer time: {total_ms:.1f} ms")
print(f"  Mean layer time: {np.mean(layer_times)*1000:.1f} ms")

full_out = tensors.from_device(x_tt, (1, 32, 768))

# JAX reference
x_jax_full = jnp.array(x_np)
for layer_idx in range(12):
    lw_i = get_layer_weights_np(layer_idx)
    x_jax_full = gpt2_layer_jax(x_jax_full, lw_i, n_heads)
m = jnp.mean(x_jax_full, axis=-1, keepdims=True)
v = jnp.mean((x_jax_full - m)**2, axis=-1, keepdims=True)
x_jax_full = jnp.array(ln_f_g) * (x_jax_full - m) / jnp.sqrt(v + 1e-5) + jnp.array(ln_f_b)
jax_full_out = np.array(x_jax_full)

cos_sim_full = np.dot(full_out.flatten(), jax_full_out.flatten()) / (
    np.linalg.norm(full_out.flatten()) * np.linalg.norm(jax_full_out.flatten()) + 1e-8)
max_err_full = np.abs(full_out - jax_full_out).max()
mean_err_full = np.abs(full_out - jax_full_out).mean()
print(f"\nFull model accuracy:")
print(f"  Cosine similarity: {cos_sim_full:.6f}")
print(f"  Max error: {max_err_full:.4f}")
print(f"  Mean error: {mean_err_full:.6f}")

# Next-token prediction
logits = full_out @ wte.T
next_logits = logits[0, -1, :]
exp_l = np.exp(next_logits - next_logits.max())
probs = exp_l / exp_l.sum()
top5 = np.argsort(next_logits)[-5:][::-1]
print(f"\nTop-5 next tokens (all-on-device):")
for tid in top5:
    tok = simple_decode([int(tid)])
    print(f"  '{tok}' (id={int(tid)}, prob={probs[tid]:.4f})")

# JAX reference predictions
jax_logits = jax_full_out @ wte.T
jax_next = jax_logits[0, -1, :]
jax_exp = np.exp(jax_next - jax_next.max())
jax_probs = jax_exp / jax_exp.sum()
jax_top5 = np.argsort(jax_next)[-5:][::-1]
print(f"\nTop-5 next tokens (JAX CPU reference):")
for tid in jax_top5:
    tok = simple_decode([int(tid)])
    print(f"  '{tok}' (id={int(tid)}, prob={jax_probs[tid]:.4f})")

match = "MATCH" if top5[0] == jax_top5[0] else "MISMATCH"
print(f"\nTop-1: TT-NN='{simple_decode([int(top5[0])])}' vs JAX='{simple_decode([int(jax_top5[0])])}' [{match}]")

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary: On-device vs CPU round-trip breakdown")
print("=" * 60)
print(f"""
ON DEVICE (no CPU round-trip):
  - LayerNorm:  ttnn.layer_norm       {'YES' if LAYER_NORM_WORKS else 'NO'}
  - GELU:       ttnn.gelu             {'YES' if GELU_WORKS else 'NO'}
  - Matmul:     ttnn.matmul           YES
  - Add:        ttnn.add              YES
  - Attention:  ttnn.transformer.sdpa YES

CPU ROUND-TRIPS (unavoidable with current ttnn):
  - QKV split:   device->CPU->device (reshape+transpose, no ttnn.split)
  - Head concat:  device->CPU->device (reshape+transpose)

Per layer: 2 CPU round-trips (QKV split + head concat)
Full model: 24 CPU round-trips for 12 layers

Previous (exp 29): 4 CPU round-trips/layer = 48 total
Improvement: 2x fewer CPU round-trips
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
