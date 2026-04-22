"""
Experiment 31: GPT-2 text generation on Blackhole.

Greedy autoregressive decoding: generate text token-by-token
using real GPT-2 weights on Tenstorrent Blackhole.

This is the ultimate demo — the hardware produces coherent English text.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax.numpy as jnp
import jax
import time
import torch

# ── Load GPT-2 weights ──────────────────────────────────────
from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

print("Loading GPT-2 small...")
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
token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

def decode_tokens(ids):
    return ''.join(id_to_token.get(int(i), '?').replace('\u0120', ' ') for i in ids)

def encode_simple(text):
    """Greedy longest-match tokenizer."""
    tokens = []
    i = 0
    text_bytes = text.encode('utf-8')
    while i < len(text_bytes):
        best_len = 0
        for length in range(min(20, len(text_bytes) - i), 0, -1):
            candidate = text_bytes[i:i+length].decode('utf-8', errors='ignore')
            if i > 0 and text_bytes[i] == ord(' '):
                cand_space = '\u0120' + candidate[1:] if len(candidate) > 1 else '\u0120'
                if cand_space in token_to_id:
                    tokens.append(token_to_id[cand_space])
                    best_len = length
                    break
            if candidate in token_to_id:
                tokens.append(token_to_id[candidate])
                best_len = length
                break
        if best_len == 0:
            tokens.append(token_to_id.get(chr(text_bytes[i]), 0))
            best_len = 1
        i += best_len
    return tokens

n_heads = config['n_head']
d_model = config['n_embd']
head_dim = d_model // n_heads
n_layers = config['n_layer']

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}")

# ── Device setup ─────────────────────────────────────────────
import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)

# ── Helper functions ─────────────────────────────────────────

wte = weights["wte.weight"]  # (50257, 768)
wpe = weights["wpe.weight"]  # (1024, 768)
ln_f_g = weights["ln_f.weight"]
ln_f_b = weights["ln_f.bias"]

def get_layer_weights(layer_idx):
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

# Pre-upload all weights to device
print("Uploading weights to device...")
t0 = time.perf_counter()
layer_weights_tt = []
for i in range(n_layers):
    lw = get_layer_weights(i)
    lw_tt = {}
    for k, v in lw.items():
        lw_tt[k] = tensors.to_device(v, device)
    layer_weights_tt.append(lw_tt)
ln_f_g_tt = tensors.to_device(ln_f_g, device)
ln_f_b_tt = tensors.to_device(ln_f_b, device)
t_upload = time.perf_counter() - t0
print(f"Weight upload: {t_upload*1000:.0f} ms")


def layernorm_device(x_tt, gamma_tt, beta_tt):
    """LayerNorm on device via ttnn.layer_norm."""
    return ttnn.layer_norm(x_tt, weight=gamma_tt, bias=beta_tt, epsilon=1e-5)


def gelu_device(x_tt):
    """GELU on device — matches GPT-2's gelu_new at bfloat16 precision."""
    return ttnn.gelu(x_tt, fast_and_approximate_mode=False)


def gpt2_layer(x_tt, lw_tt, seq_len):
    """One GPT-2 layer: LN → Attn → Add → LN → MLP → Add.

    Only 2 CPU round-trips: QKV split and head concat.
    Everything else runs on Blackhole.
    """
    # LayerNorm 1 (on device)
    h = layernorm_device(x_tt, lw_tt['ln1_g'], lw_tt['ln1_b'])

    # QKV projection (on device)
    qkv = ttnn.add(ttnn.matmul(h, lw_tt['w_attn']), lw_tt['b_attn'])

    # Split QKV and reshape for native attention (CPU round-trip #1)
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

    # Native FlashAttention-2 (on device)
    attn_out = ttnn.transformer.scaled_dot_product_attention(q_tt, k_tt, v_tt, is_causal=True)

    # Concat heads (CPU round-trip #2)
    try:
        merged = ttnn.transformer.concatenate_heads(attn_out)
        merged_np = tensors.from_device(merged, (1, seq_len, d_model))
    except Exception:
        attn_np = ttnn.to_torch(attn_out).float().numpy()
        merged_np = attn_np.transpose(0, 2, 1, 3).reshape(1, seq_len, d_model)
    merged_tt = tensors.to_device(merged_np, device)

    # Output projection + residual (on device)
    proj = ttnn.add(ttnn.matmul(merged_tt, lw_tt['w_proj']), lw_tt['b_proj'])
    x_tt = ttnn.add(x_tt, proj)

    # LayerNorm 2 (on device)
    h2 = layernorm_device(x_tt, lw_tt['ln2_g'], lw_tt['ln2_b'])

    # MLP (on device)
    ff = ttnn.add(ttnn.matmul(h2, lw_tt['w_fc']), lw_tt['b_fc'])
    ff = gelu_device(ff)
    ff_out = ttnn.add(ttnn.matmul(ff, lw_tt['w_mlp_proj']), lw_tt['b_mlp_proj'])

    return ttnn.add(x_tt, ff_out)


def gpt2_forward(token_ids):
    """Full GPT-2 forward pass. Returns logits for next token."""
    seq_len = len(token_ids)
    # Pad to tile-aligned length (multiple of 32)
    padded_len = ((seq_len + 31) // 32) * 32
    padded_ids = list(token_ids) + [50256] * (padded_len - seq_len)

    # Embeddings
    tok_emb = wte[padded_ids]
    pos_emb = wpe[:padded_len]
    x_np = (tok_emb + pos_emb)[None, :, :]  # (1, padded_len, 768)

    x_tt = tensors.to_device(x_np, device)

    # 12 transformer layers
    for i in range(n_layers):
        x_tt = gpt2_layer(x_tt, layer_weights_tt[i], padded_len)

    # Final layernorm (on device)
    x_tt = layernorm_device(x_tt, ln_f_g_tt, ln_f_b_tt)

    # Get hidden state for last real token
    x_np = tensors.from_device(x_tt, (1, padded_len, d_model))
    last_hidden = x_np[0, seq_len - 1, :]  # (768,)

    # Project to vocab
    logits = last_hidden @ wte.T  # (50257,)
    return logits


def generate(prompt, max_new_tokens=30, temperature=0.0):
    """Greedy autoregressive generation."""
    token_ids = encode_simple(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {len(token_ids)}")
    print(f"Generating {max_new_tokens} tokens...")

    generated = []
    for i in range(max_new_tokens):
        t0 = time.perf_counter()
        logits = gpt2_forward(token_ids)
        t1 = time.perf_counter()

        if temperature == 0.0:
            next_token = int(np.argmax(logits))
        else:
            logits = logits / temperature
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))

        token_ids.append(next_token)
        generated.append(next_token)
        tok_str = decode_tokens([next_token])

        if i < 5 or i == max_new_tokens - 1:
            print(f"  [{i+1:2d}] '{tok_str}' (id={next_token}, {(t1-t0)*1000:.0f}ms)")
        elif i == 5:
            print(f"  ...")

        # Stop on EOS
        if next_token == 50256:
            break

    full_text = decode_tokens(token_ids)
    return full_text, generated


# ── JAX CPU reference ────────────────────────────────────────
print("\n=== JAX CPU Reference ===")

def gpt2_forward_jax(token_ids):
    """JAX CPU reference forward pass."""
    seq_len = len(token_ids)
    padded_len = ((seq_len + 31) // 32) * 32
    padded_ids = list(token_ids) + [50256] * (padded_len - seq_len)

    tok_emb = jnp.array(wte[padded_ids])
    pos_emb = jnp.array(wpe[:padded_len])
    x = (tok_emb + pos_emb)[None, :, :]

    for layer_idx in range(n_layers):
        lw = get_layer_weights(layer_idx)
        # LN1
        m = jnp.mean(x, axis=-1, keepdims=True)
        v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
        h = jnp.array(lw['ln1_g']) * (x - m) / jnp.sqrt(v + 1e-5) + jnp.array(lw['ln1_b'])
        # Attention
        qkv = jnp.dot(h, jnp.array(lw['w_attn'])) + jnp.array(lw['b_attn'])
        q, k, val = jnp.split(qkv, 3, axis=-1)
        B, T, C = q.shape
        hd = C // n_heads
        q = q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        val = val.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        sc = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.array(float(hd)))
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
        def gelu_new(x):
            return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))
        ff = gelu_new(jnp.dot(h2, jnp.array(lw['w_fc'])) + jnp.array(lw['b_fc']))
        ff = jnp.dot(ff, jnp.array(lw['w_mlp_proj'])) + jnp.array(lw['b_mlp_proj'])
        x = x + ff

    # Final LN
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
    x = jnp.array(ln_f_g) * (x - m) / jnp.sqrt(v + 1e-5) + jnp.array(ln_f_b)
    return np.array(x[0, seq_len - 1, :] @ jnp.array(wte).T)

prompt = "The meaning of life is"
tokens = encode_simple(prompt)
t0 = time.perf_counter()
jax_logits = gpt2_forward_jax(tokens)
t1 = time.perf_counter()
jax_top5 = np.argsort(jax_logits)[-5:][::-1]
print(f"JAX CPU forward: {(t1-t0)*1000:.0f}ms")
print(f"Top-5: {[decode_tokens([t]) for t in jax_top5]}")

# ── Generate on Blackhole ────────────────────────────────────
print("\n=== Blackhole Generation ===")

prompts = [
    "The meaning of life is",
    "Once upon a time, in a land far away,",
    "Artificial intelligence will",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    text, gen_tokens = generate(prompt, max_new_tokens=20)
    print(f"\nFull output: '{text}'")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("\nDone!")
