"""
Experiment 29: GPT-2 with ttnn.transformer native attention.

Instead of interpreting 4D Jaxpr ops (which fail on TT-NN),
use ttnn.transformer.scaled_dot_product_attention directly.

This tests whether TT-NN's native FlashAttention-2 can replace our
hand-rolled attention and give both better accuracy and speed.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import time
from collections import Counter

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

# ── Test: Can we call ttnn.transformer.scaled_dot_product_attention? ──
print("\n=== Phase 1: Test native attention ===")

n_heads = 12
d_model = 768
head_dim = d_model // n_heads  # 64
seq_len = 32
batch = 1

# Create random Q, K, V in the expected format: [batch, n_heads, seq, head_dim]
rng = np.random.RandomState(42)
q_np = rng.randn(batch, n_heads, seq_len, head_dim).astype(np.float32) * 0.1
k_np = rng.randn(batch, n_heads, seq_len, head_dim).astype(np.float32) * 0.1
v_np = rng.randn(batch, n_heads, seq_len, head_dim).astype(np.float32) * 0.1

q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)
k_tt = ttnn.from_torch(torch.from_numpy(k_np), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)
v_tt = ttnn.from_torch(torch.from_numpy(v_np), dtype=ttnn.bfloat16,
                        device=device, layout=ttnn.TILE_LAYOUT)

print(f"Q shape: {q_tt.shape}, K shape: {k_tt.shape}, V shape: {v_tt.shape}")

try:
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, is_causal=True)
    print(f"Native attention output shape: {attn_out.shape}")
    attn_np = ttnn.to_torch(attn_out).float().numpy()
    print(f"Output stats: mean={attn_np.mean():.4f}, std={attn_np.std():.4f}")

    # Compare with JAX reference
    q_jax = jnp.array(q_np)
    k_jax = jnp.array(k_np)
    v_jax = jnp.array(v_np)
    scale = 1.0 / np.sqrt(head_dim)
    scores = jnp.matmul(q_jax, jnp.swapaxes(k_jax, -2, -1)) * scale
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    scores = scores * mask + (-1e10) * (1.0 - mask)
    attn_w = jax.nn.softmax(scores, axis=-1)
    ref_out = np.array(jnp.matmul(attn_w, v_jax))

    cos_sim = np.dot(attn_np.flatten(), ref_out.flatten()) / (
        np.linalg.norm(attn_np.flatten()) * np.linalg.norm(ref_out.flatten()) + 1e-8)
    max_err = np.abs(attn_np.reshape(ref_out.shape) - ref_out).max()
    print(f"vs JAX ref: cosine sim = {cos_sim:.6f}, max err = {max_err:.6f}")

    NATIVE_ATTN_WORKS = True
except Exception as e:
    print(f"Native attention FAILED: {e}")
    NATIVE_ATTN_WORKS = False


# ── Phase 2: Full GPT-2 layer with native attention ─────────
if NATIVE_ATTN_WORKS:
    print("\n=== Phase 2: GPT-2 layer with native attention ===")

    from tt_jax import tensors

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

    def layernorm_tt(x_tt, gamma_tt, beta_tt, device, x_shape):
        """LayerNorm on device using our interpreter's ops."""
        # For now, do layernorm via CPU — it's fast and correct
        x_np = tensors.from_device(x_tt, x_shape)
        g_np = tensors.from_device(gamma_tt, (x_shape[-1],))
        b_np = tensors.from_device(beta_tt, (x_shape[-1],))

        mean = x_np.mean(axis=-1, keepdims=True)
        var = ((x_np - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = g_np * (x_np - mean) / np.sqrt(var + 1e-5) + b_np
        return tensors.to_device(x_norm, device)

    def gelu_np(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def gpt2_layer_native(x_tt, lw, device, seq_len, d_model, n_heads):
        """One GPT-2 layer using ttnn native ops where possible."""
        head_dim = d_model // n_heads
        x_shape = (1, seq_len, d_model)

        # LayerNorm 1
        ln1_g = tensors.to_device(lw['ln1_g'], device)
        ln1_b = tensors.to_device(lw['ln1_b'], device)
        h = layernorm_tt(x_tt, ln1_g, ln1_b, device, x_shape)

        # QKV projection: h @ w_attn + b_attn -> (1, T, 3*d_model)
        w_attn = tensors.to_device(lw['w_attn'], device)
        b_attn = tensors.to_device(lw['b_attn'], device)
        qkv = ttnn.add(ttnn.matmul(h, w_attn), b_attn)

        # Split QKV and reshape to 4D for native attention
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

        # Native attention!
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=True)

        # Concatenate heads back: (1, n_heads, T, head_dim) -> (1, T, d_model)
        try:
            merged = ttnn.transformer.concatenate_heads(attn_out)
            merged_np = tensors.from_device(merged, (1, seq_len, d_model))
        except Exception:
            attn_np = ttnn.to_torch(attn_out).float().numpy()
            merged_np = attn_np.transpose(0, 2, 1, 3).reshape(1, seq_len, d_model)
        merged_tt = tensors.to_device(merged_np, device)

        # Output projection
        w_proj = tensors.to_device(lw['w_proj'], device)
        b_proj = tensors.to_device(lw['b_proj'], device)
        proj = ttnn.add(ttnn.matmul(merged_tt, w_proj), b_proj)

        # Residual connection
        x_tt = ttnn.add(x_tt, proj)

        # LayerNorm 2
        ln2_g = tensors.to_device(lw['ln2_g'], device)
        ln2_b = tensors.to_device(lw['ln2_b'], device)
        h2 = layernorm_tt(x_tt, ln2_g, ln2_b, device, x_shape)

        # MLP: gelu(h @ w_fc + b_fc) @ w_mlp_proj + b_mlp_proj
        w_fc = tensors.to_device(lw['w_fc'], device)
        b_fc = tensors.to_device(lw['b_fc'], device)
        ff = ttnn.add(ttnn.matmul(h2, w_fc), b_fc)

        # GELU — use CPU for now (ttnn.gelu might not match GPT-2's gelu_new)
        ff_np = tensors.from_device(ff, (1, seq_len, d_model * 4))
        ff_np = gelu_np(ff_np)
        ff = tensors.to_device(ff_np, device)

        w_mlp = tensors.to_device(lw['w_mlp_proj'], device)
        b_mlp = tensors.to_device(lw['b_mlp_proj'], device)
        ff_out = ttnn.add(ttnn.matmul(ff, w_mlp), b_mlp)

        # Residual
        return ttnn.add(x_tt, ff_out)

    # Prepare input
    wte = weights["wte.weight"]
    wpe = weights["wpe.weight"]

    # Simple tokenize
    token_to_id = vocab
    text = "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout"
    # Crude tokenizer
    tokens = []
    for word in text.split():
        if word in token_to_id:
            tokens.append(token_to_id[word])
        elif '\u0120' + word in token_to_id:
            tokens.append(token_to_id['\u0120' + word])
        else:
            for ch in word:
                tokens.append(token_to_id.get(ch, 0))

    # Pad to 32
    while len(tokens) < 32:
        tokens.append(50256)
    tokens = tokens[:32]
    print(f"Tokens: {tokens[:10]}...")

    # Embeddings
    tok_emb = wte[tokens]  # (32, 768)
    pos_emb = wpe[:32]     # (32, 768)
    x_np = (tok_emb + pos_emb)[None, :, :]  # (1, 32, 768)
    print(f"Input shape: {x_np.shape}")

    # JAX CPU reference (single layer)
    def gelu_jax(x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))

    def gpt2_layer_jax(x, lw, n_heads):
        d = x.shape[-1]
        head_dim = d // n_heads
        # LN1
        m = jnp.mean(x, axis=-1, keepdims=True)
        v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
        h = jnp.array(lw['ln1_g']) * (x - m) / jnp.sqrt(v + 1e-5) + jnp.array(lw['ln1_b'])
        # Attention
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
        # LN2
        m2 = jnp.mean(x, axis=-1, keepdims=True)
        v2 = jnp.mean((x - m2)**2, axis=-1, keepdims=True)
        h2 = jnp.array(lw['ln2_g']) * (x - m2) / jnp.sqrt(v2 + 1e-5) + jnp.array(lw['ln2_b'])
        # MLP
        ff = gelu_jax(jnp.dot(h2, jnp.array(lw['w_fc'])) + jnp.array(lw['b_fc']))
        ff = jnp.dot(ff, jnp.array(lw['w_mlp_proj'])) + jnp.array(lw['b_mlp_proj'])
        return x + ff

    lw0 = get_layer_weights_np(0)
    x_jax = jnp.array(x_np)
    jax_out = np.array(gpt2_layer_jax(x_jax, lw0, n_heads))
    print(f"JAX reference: shape={jax_out.shape}, mean={jax_out.mean():.4f}")

    # Run on Blackhole
    x_tt = tensors.to_device(x_np, device)
    t0 = time.perf_counter()
    out_tt = gpt2_layer_native(x_tt, lw0, device, seq_len, d_model, n_heads)
    t1 = time.perf_counter()

    out_np = tensors.from_device(out_tt, (1, 32, 768))
    print(f"\nBlackhole layer time: {(t1-t0)*1000:.1f} ms")

    cos_sim = np.dot(out_np.flatten(), jax_out.flatten()) / (
        np.linalg.norm(out_np.flatten()) * np.linalg.norm(jax_out.flatten()) + 1e-8)
    max_err = np.abs(out_np - jax_out).max()
    mean_err = np.abs(out_np - jax_out).mean()
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Max error: {max_err:.6f}")
    print(f"Mean error: {mean_err:.6f}")

    # ── Phase 3: Full 12-layer GPT-2 with native attention ───
    print("\n=== Phase 3: Full 12-layer GPT-2 ===")

    ln_f_g = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]

    x_tt = tensors.to_device(x_np, device)

    t0 = time.perf_counter()
    for layer_idx in range(12):
        lw_i = get_layer_weights_np(layer_idx)
        x_tt = gpt2_layer_native(x_tt, lw_i, device, seq_len, d_model, n_heads)
    # Final layernorm
    ln_f_g_tt = tensors.to_device(ln_f_g, device)
    ln_f_b_tt = tensors.to_device(ln_f_b, device)
    x_tt = layernorm_tt(x_tt, ln_f_g_tt, ln_f_b_tt, device, (1, 32, 768))
    t1 = time.perf_counter()
    print(f"Full 12-layer time: {(t1-t0)*1000:.1f} ms")

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
    print(f"Full model cosine sim: {cos_sim_full:.6f}")
    print(f"Full model max error: {max_err_full:.4f}")

    # Next-token prediction
    logits = full_out @ wte.T  # (1, 32, 50257)
    next_logits = logits[0, -1, :]
    exp_l = np.exp(next_logits - next_logits.max())
    probs = exp_l / exp_l.sum()
    top5 = np.argsort(next_logits)[-5:][::-1]
    print(f"\nTop-5 next tokens (TT-NN native attention):")
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

# ── Cleanup ─────��────────────────────────────────────────────
ttnn.close_device(device)
print("\nDone!")
