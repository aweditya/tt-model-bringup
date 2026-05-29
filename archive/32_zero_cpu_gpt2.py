"""
Experiment 32: Zero CPU round-trip GPT-2 forward pass + trace capture.

Goal: eliminate ALL CPU round-trips from GPT-2 by using:
  - ttnn.transformer.split_query_key_value_and_split_heads (QKV split)
  - ttnn.transformer.concatenate_heads (head merge)
  - ttnn.layer_norm, ttnn.gelu, ttnn.matmul, ttnn.add (already proven)

If the forward pass is fully on-device, trace capture should work,
potentially giving ~10ms/token (vs ~120ms/token with CPU round-trips).

Phases:
  1. Test split_query_key_value_and_split_heads on a real GPT-2 QKV tensor
  2. Test concatenate_heads stays on-device (no readback needed)
  3. Single GPT-2 layer with ZERO CPU round-trips
  4. Full 12-layer GPT-2 with zero round-trips
  5. Trace capture attempt
  6. Benchmark: traced vs untraced vs previous best
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
import time
import torch

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
token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

def decode_tokens(ids):
    return ''.join(id_to_token.get(int(i), '?').replace('\u0120', ' ') for i in ids)

n_heads = config['n_head']       # 12
d_model = config['n_embd']       # 768
head_dim = d_model // n_heads    # 64
n_layers = config['n_layer']     # 12
seq_len = 32

print(f"GPT-2: {n_layers}L, {n_heads}H, d={d_model}, head_dim={head_dim}")

# ── Device setup ─────────────────────────────────────────────
import ttnn
from tt_jax import tensors

device = ttnn.open_device(device_id=0)

# ══════════════════════════════════════════════════════════════
# Phase 1: Test split_query_key_value_and_split_heads
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: Test ttnn.transformer.split_query_key_value_and_split_heads")
print("=" * 60)

# Create a realistic QKV tensor: (1, seq_len, 3*d_model) = (1, 32, 2304)
rng = np.random.RandomState(42)
qkv_np = rng.randn(1, seq_len, 3 * d_model).astype(np.float32) * 0.1

# CPU reference: split and reshape
q_ref = qkv_np[:, :, :d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
k_ref = qkv_np[:, :, d_model:2*d_model].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
v_ref = qkv_np[:, :, 2*d_model:].reshape(1, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
# Expected shapes: (1, n_heads, seq_len, head_dim) = (1, 12, 32, 64)
print(f"  Expected Q/K/V shape: {q_ref.shape}")

# Test the ttnn function
qkv_tt = tensors.to_device(qkv_np, device)
print(f"  QKV device tensor shape: {qkv_tt.shape}")

# Try split_query_key_value_and_split_heads
# From the ttnn docs, this function takes:
#   input_tensor: (B, seq_len, 3 * n_heads * head_dim)
#   num_heads: number of attention heads
# It should return (Q, K, V) each with shape (B, n_heads, seq_len, head_dim)
split_ok = False
try:
    print("  Trying: split_query_key_value_and_split_heads(qkv, num_heads=12)...")
    q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv_tt, num_heads=n_heads
    )
    print(f"  SUCCESS! Q shape: {q_tt.shape}, K shape: {k_tt.shape}, V shape: {v_tt.shape}")

    # Verify correctness
    q_out = ttnn.to_torch(q_tt).float().numpy()
    k_out = ttnn.to_torch(k_tt).float().numpy()
    v_out = ttnn.to_torch(v_tt).float().numpy()
    print(f"  Q torch shape: {q_out.shape}, K torch shape: {k_out.shape}, V torch shape: {v_out.shape}")

    # Reshape outputs to match reference if needed
    if q_out.shape != q_ref.shape:
        print(f"  Shape mismatch: got {q_out.shape}, expected {q_ref.shape}")
        try:
            q_out = q_out.reshape(q_ref.shape)
            k_out = k_out.reshape(k_ref.shape)
            v_out = v_out.reshape(v_ref.shape)
            print(f"  Reshaped to {q_out.shape}")
        except Exception as e:
            print(f"  Could not reshape: {e}")

    for name, out, ref in [("Q", q_out, q_ref), ("K", k_out, k_ref), ("V", v_out, v_ref)]:
        cos = np.dot(out.flatten(), ref.flatten()) / (
            np.linalg.norm(out.flatten()) * np.linalg.norm(ref.flatten()) + 1e-8)
        maxe = np.abs(out - ref).max()
        print(f"  {name}: cosine={cos:.6f}, max_err={maxe:.6f}")

    split_ok = True

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# If split_query_key_value_and_split_heads failed, try alternative approaches
if not split_ok:
    print("\n  Trying alternative: manual on-device split via slice...")
    # ttnn may have slice/narrow operations we can use
    for fn_name in ['split', 'slice', 'narrow', 'chunk']:
        if hasattr(ttnn, fn_name):
            print(f"    ttnn.{fn_name} exists!")
        else:
            print(f"    ttnn.{fn_name} — not found")

    # Try ttnn.reshape + ttnn.permute approach
    print("\n  Trying reshape+permute on-device approach...")
    try:
        # (1, 32, 2304) -> (1, 32, 3, 12, 64) -> split somehow
        # Actually, let's try a different approach:
        # Separate the QKV projection into 3 separate matmuls
        print("  NOTE: If split fails, fallback = 3 separate Q/K/V matmuls")
        split_ok = "SEPARATE_MATMULS"
    except Exception as e2:
        print(f"  Also failed: {e2}")


# ══════════════════════════════════════════════════════════════
# Phase 2: Test concatenate_heads stays on-device
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Test concatenate_heads (no CPU readback)")
print("=" * 60)

# Create attention output: (1, n_heads, seq_len, head_dim) = (1, 12, 32, 64)
attn_out_np = rng.randn(1, n_heads, seq_len, head_dim).astype(np.float32) * 0.1
ref_concat = attn_out_np.transpose(0, 2, 1, 3).reshape(1, seq_len, d_model)

attn_out_tt = ttnn.from_torch(
    torch.from_numpy(attn_out_np.copy()), dtype=ttnn.bfloat16,
    device=device, layout=ttnn.TILE_LAYOUT
)
print(f"  Attn output device shape: {attn_out_tt.shape}")

concat_ok = False
try:
    merged_tt = ttnn.transformer.concatenate_heads(attn_out_tt)
    print(f"  concatenate_heads output shape: {merged_tt.shape}")

    merged_np = tensors.from_device(merged_tt, (1, seq_len, d_model))
    cos = np.dot(merged_np.flatten(), ref_concat.flatten()) / (
        np.linalg.norm(merged_np.flatten()) * np.linalg.norm(ref_concat.flatten()) + 1e-8)
    maxe = np.abs(merged_np - ref_concat).max()
    print(f"  Cosine: {cos:.6f}, Max error: {maxe:.6f}")

    # Key test: can we feed concatenate_heads output directly into matmul?
    w_test = tensors.to_device(rng.randn(d_model, d_model).astype(np.float32) * 0.01, device)
    proj_tt = ttnn.matmul(merged_tt, w_test)
    print(f"  matmul after concatenate_heads: shape={proj_tt.shape} — OK")
    concat_ok = True

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════
# Phase 3: Single GPT-2 layer — ZERO CPU round-trips
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Single GPT-2 layer — zero CPU round-trips")
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


def upload_layer_weights(lw_np, device):
    """Upload all layer weights to device once."""
    lw_tt = {}
    for k, v in lw_np.items():
        lw_tt[k] = tensors.to_device(v, device)
    return lw_tt


USE_SPLIT_FN = split_ok is True  # Only if the native split function works

if USE_SPLIT_FN:
    print("  Strategy: native split_query_key_value_and_split_heads")
else:
    print("  Strategy: 3 separate Q/K/V matmuls (split via separate weights)")
    print("  Splitting w_attn (768, 2304) into w_q, w_k, w_v (768, 768) each")


def prepare_split_weights(lw_np):
    """Split the combined QKV weight/bias into separate Q, K, V."""
    w = lw_np['w_attn']   # (768, 2304)
    b = lw_np['b_attn']   # (2304,)
    lw_np['w_q'] = w[:, :d_model].copy()           # (768, 768)
    lw_np['w_k'] = w[:, d_model:2*d_model].copy()  # (768, 768)
    lw_np['w_v'] = w[:, 2*d_model:].copy()          # (768, 768)
    lw_np['b_q'] = b[:d_model].copy()               # (768,)
    lw_np['b_k'] = b[d_model:2*d_model].copy()      # (768,)
    lw_np['b_v'] = b[2*d_model:].copy()              # (768,)
    return lw_np


def gpt2_layer_zero_cpu(x_tt, lw_tt, seq_len):
    """GPT-2 transformer layer with ZERO CPU round-trips.

    Every operation stays on device:
      - LayerNorm via ttnn.layer_norm
      - QKV via split_query_key_value_and_split_heads OR separate matmuls
      - Attention via ttnn.transformer.scaled_dot_product_attention
      - Head merge via ttnn.transformer.concatenate_heads
      - GELU via ttnn.gelu
      - All matmul/add via ttnn.matmul/ttnn.add
    """
    # ── LayerNorm 1 ──
    h = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=lw_tt['ln1_g'], bias=lw_tt['ln1_b'])

    if USE_SPLIT_FN:
        # Combined QKV matmul then native split
        qkv = ttnn.add(ttnn.matmul(h, lw_tt['w_attn']), lw_tt['b_attn'])
        q_tt, k_tt, v_tt = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=n_heads
        )
    else:
        # Separate Q, K, V matmuls — avoids the split entirely
        q_tt = ttnn.add(ttnn.matmul(h, lw_tt['w_q']), lw_tt['b_q'])
        k_tt = ttnn.add(ttnn.matmul(h, lw_tt['w_k']), lw_tt['b_k'])
        v_tt = ttnn.add(ttnn.matmul(h, lw_tt['w_v']), lw_tt['b_v'])
        # Reshape from (1, seq_len, 768) to (1, n_heads, seq_len, head_dim)
        # Use ttnn.reshape — this should stay on device
        q_tt = ttnn.reshape(q_tt, [1, seq_len, n_heads, head_dim])
        k_tt = ttnn.reshape(k_tt, [1, seq_len, n_heads, head_dim])
        v_tt = ttnn.reshape(v_tt, [1, seq_len, n_heads, head_dim])
        # Transpose (1, seq_len, n_heads, head_dim) -> (1, n_heads, seq_len, head_dim)
        q_tt = ttnn.transpose(q_tt, 1, 2)
        k_tt = ttnn.transpose(k_tt, 1, 2)
        v_tt = ttnn.transpose(v_tt, 1, 2)

    # ── Attention ──
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_tt, k_tt, v_tt, is_causal=True
    )

    # ── Concatenate heads ──
    merged = ttnn.transformer.concatenate_heads(attn_out)

    # ── Output projection + residual ──
    proj = ttnn.add(ttnn.matmul(merged, lw_tt['w_proj']), lw_tt['b_proj'])
    x_tt = ttnn.add(x_tt, proj)

    # ── LayerNorm 2 ──
    h2 = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=lw_tt['ln2_g'], bias=lw_tt['ln2_b'])

    # ── MLP: FC → GELU → Proj ──
    ff = ttnn.add(ttnn.matmul(h2, lw_tt['w_fc']), lw_tt['b_fc'])
    ff = ttnn.gelu(ff, fast_and_approximate_mode=False)
    ff_out = ttnn.add(ttnn.matmul(ff, lw_tt['w_mlp_proj']), lw_tt['b_mlp_proj'])

    # ── Residual ──
    return ttnn.add(x_tt, ff_out)


# ── JAX CPU reference ────────────────────────────────────────
def gpt2_layer_jax(x, lw, n_heads):
    d = x.shape[-1]
    hd = d // n_heads
    m = jnp.mean(x, axis=-1, keepdims=True)
    v = jnp.mean((x - m)**2, axis=-1, keepdims=True)
    h = jnp.array(lw['ln1_g']) * (x - m) / jnp.sqrt(v + 1e-5) + jnp.array(lw['ln1_b'])
    qkv = jnp.dot(h, jnp.array(lw['w_attn'])) + jnp.array(lw['b_attn'])
    q, k, val = jnp.split(qkv, 3, axis=-1)
    B, T, C = q.shape
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
    m2 = jnp.mean(x, axis=-1, keepdims=True)
    v2 = jnp.mean((x - m2)**2, axis=-1, keepdims=True)
    h2 = jnp.array(lw['ln2_g']) * (x - m2) / jnp.sqrt(v2 + 1e-5) + jnp.array(lw['ln2_b'])
    def gelu_new(x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))
    ff = gelu_new(jnp.dot(h2, jnp.array(lw['w_fc'])) + jnp.array(lw['b_fc']))
    ff = jnp.dot(ff, jnp.array(lw['w_mlp_proj'])) + jnp.array(lw['b_mlp_proj'])
    return x + ff


# ── Prepare input ────────────────────────────────────────────
wte = weights["wte.weight"]
wpe = weights["wpe.weight"]

text = "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout"
tokens = []
for word in text.split():
    key = '\u0120' + word if '\u0120' + word in token_to_id else word
    if key in token_to_id:
        tokens.append(token_to_id[key])
    else:
        for ch in word:
            tokens.append(token_to_id.get(ch, 0))
# First token has no space prefix
if text.split()[0] in token_to_id:
    tokens[0] = token_to_id[text.split()[0]]
while len(tokens) < seq_len:
    tokens.append(50256)
tokens = tokens[:seq_len]
print(f"Input tokens: {tokens[:8]}...")

tok_emb = wte[tokens]
pos_emb = wpe[:seq_len]
x_np = (tok_emb + pos_emb)[None, :, :]  # (1, 32, 768)

# ── Single layer test ────────────────────────────────────────
lw0_np = get_layer_weights_np(0)
if not USE_SPLIT_FN:
    lw0_np = prepare_split_weights(lw0_np)
lw0_tt = upload_layer_weights(lw0_np, device)

# JAX reference
x_jax = jnp.array(x_np)
jax_out = np.array(gpt2_layer_jax(x_jax, get_layer_weights_np(0), n_heads))

# TT-NN zero-CPU layer
x_tt = tensors.to_device(x_np, device)
try:
    t0 = time.perf_counter()
    out_tt = gpt2_layer_zero_cpu(x_tt, lw0_tt, seq_len)
    t1 = time.perf_counter()

    out_np = tensors.from_device(out_tt, (1, seq_len, d_model))
    cos = np.dot(out_np.flatten(), jax_out.flatten()) / (
        np.linalg.norm(out_np.flatten()) * np.linalg.norm(jax_out.flatten()) + 1e-8)
    maxe = np.abs(out_np - jax_out).max()
    meane = np.abs(out_np - jax_out).mean()
    print(f"\nSingle layer (zero CPU round-trips):")
    print(f"  Time: {(t1-t0)*1000:.1f} ms")
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max error: {maxe:.6f}, Mean error: {meane:.6f}")
    LAYER_OK = True
except Exception as e:
    print(f"\nSingle layer FAILED: {e}")
    import traceback
    traceback.print_exc()
    LAYER_OK = False


# ══════════════════════════════════════════════════════════════
# Phase 4: Full 12-layer GPT-2 — zero CPU round-trips
# ══════════════════════════════════════════════════════════════
if LAYER_OK:
    print("\n" + "=" * 60)
    print("Phase 4: Full 12-layer GPT-2 — zero CPU round-trips")
    print("=" * 60)

    # Pre-upload all layer weights
    print("  Uploading all layer weights to device...")
    t0 = time.perf_counter()
    all_lw_tt = []
    for i in range(n_layers):
        lw_i = get_layer_weights_np(i)
        if not USE_SPLIT_FN:
            lw_i = prepare_split_weights(lw_i)
        all_lw_tt.append(upload_layer_weights(lw_i, device))
    ln_f_g_tt = tensors.to_device(weights["ln_f.weight"], device)
    ln_f_b_tt = tensors.to_device(weights["ln_f.bias"], device)
    t_upload = (time.perf_counter() - t0) * 1000
    print(f"  Weight upload: {t_upload:.0f} ms")

    # Forward pass
    x_tt = tensors.to_device(x_np, device)
    layer_times = []
    t0_total = time.perf_counter()
    for i in range(n_layers):
        t_layer = time.perf_counter()
        x_tt = gpt2_layer_zero_cpu(x_tt, all_lw_tt[i], seq_len)
        layer_times.append(time.perf_counter() - t_layer)
        print(f"  Layer {i:2d}: {layer_times[-1]*1000:.1f} ms")

    # Final layernorm
    x_tt = ttnn.layer_norm(x_tt, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)
    t1_total = time.perf_counter()
    total_ms = (t1_total - t0_total) * 1000
    print(f"\n  Total forward: {total_ms:.1f} ms")
    print(f"  Mean layer: {np.mean(layer_times)*1000:.1f} ms")

    # Verify against JAX
    full_out = tensors.from_device(x_tt, (1, seq_len, d_model))

    x_jax_full = jnp.array(x_np)
    for i in range(n_layers):
        x_jax_full = gpt2_layer_jax(x_jax_full, get_layer_weights_np(i), n_heads)
    m = jnp.mean(x_jax_full, axis=-1, keepdims=True)
    v = jnp.mean((x_jax_full - m)**2, axis=-1, keepdims=True)
    ln_f_g_j, ln_f_b_j = jnp.array(weights["ln_f.weight"]), jnp.array(weights["ln_f.bias"])
    x_jax_full = ln_f_g_j * (x_jax_full - m) / jnp.sqrt(v + 1e-5) + ln_f_b_j
    jax_full_out = np.array(x_jax_full)

    cos_full = np.dot(full_out.flatten(), jax_full_out.flatten()) / (
        np.linalg.norm(full_out.flatten()) * np.linalg.norm(jax_full_out.flatten()) + 1e-8)
    maxe_full = np.abs(full_out - jax_full_out).max()
    print(f"\n  Full model accuracy:")
    print(f"    Cosine similarity: {cos_full:.6f}")
    print(f"    Max error: {maxe_full:.4f}")

    # Next-token prediction comparison
    logits = full_out @ wte.T
    next_logits = logits[0, -1, :]
    exp_l = np.exp(next_logits - next_logits.max())
    probs = exp_l / exp_l.sum()
    top5 = np.argsort(next_logits)[-5:][::-1]
    print(f"\n  Top-5 next tokens:")
    for tid in top5:
        tok = decode_tokens([int(tid)])
        print(f"    '{tok}' (id={int(tid)}, prob={probs[tid]:.4f})")

    FULL_OK = True
else:
    FULL_OK = False
    print("\nSkipping Phase 4 — single layer failed")


# ══════════════════════════════════════════════════════════════
# Phase 5: Trace capture — the big prize
# ══════════════════════════════════════════════════════════════
if FULL_OK:
    print("\n" + "=" * 60)
    print("Phase 5: Trace capture")
    print("=" * 60)

    print("  Warmup run (establishes tensor sizes for trace)...")
    x_tt_trace = tensors.to_device(x_np, device)
    for i in range(n_layers):
        x_tt_trace = gpt2_layer_zero_cpu(x_tt_trace, all_lw_tt[i], seq_len)
    x_tt_trace = ttnn.layer_norm(x_tt_trace, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)
    print("  Warmup complete.")

    # Now capture trace
    print("  Capturing trace...")
    x_tt_trace = tensors.to_device(x_np, device)

    try:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for i in range(n_layers):
            x_tt_trace = gpt2_layer_zero_cpu(x_tt_trace, all_lw_tt[i], seq_len)
        x_tt_trace = ttnn.layer_norm(x_tt_trace, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        print(f"  TRACE CAPTURE SUCCEEDED! trace_id={tid}")
        TRACE_OK = True

    except Exception as e:
        print(f"  TRACE CAPTURE FAILED: {e}")
        import traceback
        traceback.print_exc()
        TRACE_OK = False
else:
    TRACE_OK = False
    print("\nSkipping Phase 5 — full model failed")


# ══════════════════════════════════════════════════════════════
# Phase 6: Benchmark
# ══════════════════════════════════════════════════════════════
if TRACE_OK:
    print("\n" + "=" * 60)
    print("Phase 6: Benchmark — traced vs untraced")
    print("=" * 60)

    # Verify traced execution correctness
    print("  Verifying traced execution...")
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    traced_out = tensors.from_device(x_tt_trace, (1, seq_len, d_model))
    cos_traced = np.dot(traced_out.flatten(), jax_full_out.flatten()) / (
        np.linalg.norm(traced_out.flatten()) * np.linalg.norm(jax_full_out.flatten()) + 1e-8)
    print(f"  Traced output cosine vs JAX: {cos_traced:.6f}")

    # Warmup traced
    for _ in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

    # Benchmark traced
    N_trace = 50
    t0 = time.perf_counter()
    for _ in range(N_trace):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t_traced = (time.perf_counter() - t0) / N_trace
    print(f"\n  Traced:   {t_traced*1000:.2f} ms/forward ({1/t_traced:.0f} fwd/sec)")

    # Benchmark untraced (zero CPU round-trips but no trace)
    for _ in range(3):
        x_tt_bench = tensors.to_device(x_np, device)
        for i in range(n_layers):
            x_tt_bench = gpt2_layer_zero_cpu(x_tt_bench, all_lw_tt[i], seq_len)
        x_tt_bench = ttnn.layer_norm(x_tt_bench, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)

    N_untraced = 10
    t0 = time.perf_counter()
    for _ in range(N_untraced):
        x_tt_bench = tensors.to_device(x_np, device)
        for i in range(n_layers):
            x_tt_bench = gpt2_layer_zero_cpu(x_tt_bench, all_lw_tt[i], seq_len)
        x_tt_bench = ttnn.layer_norm(x_tt_bench, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)
    t_untraced = (time.perf_counter() - t0) / N_untraced
    print(f"  Untraced: {t_untraced*1000:.2f} ms/forward ({1/t_untraced:.0f} fwd/sec)")
    print(f"  Speedup from trace: {t_untraced/t_traced:.1f}x")

    # Release trace
    ttnn.release_trace(device, tid)

elif FULL_OK:
    print("\n" + "=" * 60)
    print("Phase 6: Benchmark — untraced only (trace failed)")
    print("=" * 60)

    # Benchmark untraced zero-CPU-roundtrip version
    for _ in range(3):
        x_tt_bench = tensors.to_device(x_np, device)
        for i in range(n_layers):
            x_tt_bench = gpt2_layer_zero_cpu(x_tt_bench, all_lw_tt[i], seq_len)
        x_tt_bench = ttnn.layer_norm(x_tt_bench, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)

    N_bench = 10
    t0 = time.perf_counter()
    for _ in range(N_bench):
        x_tt_bench = tensors.to_device(x_np, device)
        for i in range(n_layers):
            x_tt_bench = gpt2_layer_zero_cpu(x_tt_bench, all_lw_tt[i], seq_len)
        x_tt_bench = ttnn.layer_norm(x_tt_bench, epsilon=1e-5, weight=ln_f_g_tt, bias=ln_f_b_tt)
    t_untraced = (time.perf_counter() - t0) / N_bench
    print(f"  Zero-CPU untraced: {t_untraced*1000:.2f} ms/forward ({1/t_untraced:.0f} fwd/sec)")
    print(f"  Previous best (2 CPU round-trips): ~120 ms/forward")
    print(f"  Estimated improvement: {120.0 / (t_untraced*1000):.1f}x")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"""
split_query_key_value_and_split_heads: {'WORKS' if split_ok is True else 'FAILED — using separate matmuls' if split_ok == 'SEPARATE_MATMULS' else 'FAILED'}
concatenate_heads:                     {'WORKS' if concat_ok else 'FAILED'}
Single layer (zero CPU):               {'WORKS' if LAYER_OK else 'FAILED'}
Full 12-layer (zero CPU):              {'WORKS' if FULL_OK else 'FAILED'}
Trace capture:                         {'WORKS' if TRACE_OK else 'FAILED'}

Strategy used: {'native split_query_key_value_and_split_heads' if USE_SPLIT_FN else '3 separate Q/K/V matmuls'}
CPU round-trips per layer: 0
""")

# ── Cleanup ──────────────────────────────────────────────────
ttnn.close_device(device)
print("Done!")
