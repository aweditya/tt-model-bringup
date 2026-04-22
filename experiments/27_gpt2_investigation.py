"""
Experiment 27: Can we run a REAL pretrained GPT-2 on Blackhole?

Until now, all experiments used random weights. This experiment loads
actual GPT-2-small (124M params) from HuggingFace and investigates:

1. What Jaxpr primitives does a real GPT-2 forward pass need?
2. How many of our 20 supported ops cover those needs?
3. Can we run at least a single layer with real weights?
4. Can we run the full model?

GPT-2 small: d_model=768, n_heads=12, n_layers=12, vocab=50257
Memory: ~248MB in bfloat16
"""

import sys, os
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import time
from collections import Counter

# ============================================================
# Phase 1: Load GPT-2 weights from safetensors (no transformers import needed)
# ============================================================

print("=" * 70)
print("PHASE 1: Loading GPT-2 small weights")
print("=" * 70)

from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

# Download model files
model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
tokenizer_path = hf_hub_download("gpt2", "tokenizer.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    config = json.load(f)
print(f"Config: n_layer={config['n_layer']}, n_head={config['n_head']}, "
      f"n_embd={config['n_embd']}, vocab_size={config['vocab_size']}")

# Load weights
weights = {}
total_params = 0
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
        total_params += weights[key].size

print(f"Total parameters: {total_params:,} ({total_params * 2 / 1024**2:.1f} MB in bfloat16)")
print(f"Number of weight tensors: {len(weights)}")
print("\nKey weight shapes:")
for name in sorted(weights.keys()):
    w = weights[name]
    if 'h.0.' in name or 'wte' in name or 'wpe' in name or 'ln_f' in name:
        print(f"  {name:45s} {str(w.shape):>20s}  ({w.size:>10,} params)")

# Simple tokenizer using vocab.json
with open(vocab_path) as f:
    vocab = json.load(f)
# Invert: token_str -> id
token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

def simple_encode(text):
    """Crude byte-level encoding (matches GPT-2 BPE for simple ASCII)."""
    # For a proper test, use the tokenizer.json. But for simple ASCII text,
    # we can use the vocab directly for individual characters/words.
    # Better approach: use the tokenizer.json merges
    with open(tokenizer_path) as f:
        tok_config = json.load(f)

    # Use the pre_tokenizer + model from tokenizer.json
    # For simplicity, just encode known tokens
    # GPT-2 encodes "The" as a single token, spaces as "Ġ" prefix
    tokens = []
    i = 0
    text_bytes = text.encode('utf-8')
    # Try greedy longest-match from vocab
    while i < len(text_bytes):
        best_token = None
        best_len = 0
        for length in range(min(20, len(text_bytes) - i), 0, -1):
            candidate = text_bytes[i:i+length].decode('utf-8', errors='ignore')
            # GPT-2 uses "Ġ" (byte 0xC4 0xA0) for space prefix
            if i > 0 and text_bytes[i] == ord(' '):
                candidate_with_space = '\u0120' + candidate[1:] if len(candidate) > 1 else '\u0120'
                if candidate_with_space in token_to_id:
                    tokens.append(token_to_id[candidate_with_space])
                    best_len = length
                    break
            if candidate in token_to_id:
                tokens.append(token_to_id[candidate])
                best_len = length
                break
        if best_len == 0:
            # Fallback: encode byte by byte
            byte_val = text_bytes[i]
            # GPT-2 byte fallback tokens
            tokens.append(token_to_id.get(chr(byte_val), 0))
            best_len = 1
        i += best_len
    return tokens

def simple_decode(token_ids):
    """Decode token ids back to text."""
    parts = []
    for tid in token_ids:
        tok = id_to_token.get(tid, '?')
        tok = tok.replace('\u0120', ' ')  # GPT-2 space prefix
        parts.append(tok)
    return ''.join(parts)


# Test tokenizer
test_text = "The meaning of life is"
test_tokens = simple_encode(test_text)
print(f"\nTokenizer test: '{test_text}' -> {test_tokens}")
print(f"Decoded: '{simple_decode(test_tokens)}'")


# ============================================================
# Phase 2: Write a pure-JAX GPT-2 forward pass and trace it
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: Tracing GPT-2 forward pass through make_jaxpr")
print("=" * 70)


def gelu(x):
    """GPT-2 uses the 'gelu_new' approximation."""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    """Standard layer norm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def attention_hf(x, w_attn, b_attn, w_proj, b_proj, n_heads):
    """Multi-head causal self-attention (HuggingFace weight convention).

    HF GPT-2 Conv1D: output = input @ weight + bias (weight is NOT transposed)
    c_attn.weight shape: (768, 2304), c_proj.weight shape: (768, 768)
    """
    B, T, C = x.shape
    head_dim = C // n_heads

    # QKV projection: x @ w_attn + b_attn
    qkv = jnp.dot(x, w_attn) + b_attn  # (B, T, 3*C)
    q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, C)

    # Reshape to (B, n_heads, T, head_dim)
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product attention with causal mask
    scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

    # Causal mask
    mask = jnp.tril(jnp.ones((T, T)))
    scores = scores * mask + (-1e10) * (1.0 - mask)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn_weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)

    return jnp.dot(out, w_proj) + b_proj


def mlp_hf(x, w_fc, b_fc, w_proj, b_proj):
    """GPT-2 MLP block (HuggingFace convention: x @ weight + bias)."""
    h = gelu(jnp.dot(x, w_fc) + b_fc)
    return jnp.dot(h, w_proj) + b_proj


def gpt2_block_hf(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                  ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj, n_heads):
    """One GPT-2 transformer block (HF weight convention)."""
    h = layer_norm(x, ln1_g, ln1_b)
    h = attention_hf(h, w_attn, b_attn, w_proj, b_proj, n_heads)
    x = x + h
    h = layer_norm(x, ln2_g, ln2_b)
    h = mlp_hf(h, w_fc, b_fc, w_mlp_proj, b_mlp_proj)
    x = x + h
    return x


def gpt2_single_layer_hf(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                          ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj,
                          ln_f_g, ln_f_b):
    """Single GPT-2 layer + final layernorm (HF convention)."""
    x = gpt2_block_hf(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                      ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj, 12)
    return layer_norm(x, ln_f_g, ln_f_b)


# ============================================================
# Phase 2a: Trace a SINGLE LAYER and analyze ops
# ============================================================

print("\n--- Tracing single GPT-2 layer ---")

# Prepare weights for layer 0
def get_layer_weights(layer_idx):
    """Extract weights for one GPT-2 layer as JAX arrays."""
    p = f"h.{layer_idx}"
    return {
        'ln1_g': jnp.array(weights[f"{p}.ln_1.weight"]),
        'ln1_b': jnp.array(weights[f"{p}.ln_1.bias"]),
        'w_attn': jnp.array(weights[f"{p}.attn.c_attn.weight"]),
        'b_attn': jnp.array(weights[f"{p}.attn.c_attn.bias"]),
        'w_proj': jnp.array(weights[f"{p}.attn.c_proj.weight"]),
        'b_proj': jnp.array(weights[f"{p}.attn.c_proj.bias"]),
        'ln2_g': jnp.array(weights[f"{p}.ln_2.weight"]),
        'ln2_b': jnp.array(weights[f"{p}.ln_2.bias"]),
        'w_fc': jnp.array(weights[f"{p}.mlp.c_fc.weight"]),
        'b_fc': jnp.array(weights[f"{p}.mlp.c_fc.bias"]),
        'w_mlp_proj': jnp.array(weights[f"{p}.mlp.c_proj.weight"]),
        'b_mlp_proj': jnp.array(weights[f"{p}.mlp.c_proj.bias"]),
    }

# GPT-2 small config
n_heads = 12
d_model = 768
seq_len = 16  # Start small for tracing
batch_size = 1

# Dummy input (embedded tokens, not raw token ids)
x_dummy = jnp.ones((batch_size, seq_len, d_model), dtype=jnp.float32)

# Get layer 0 weights
lw = get_layer_weights(0)
wpe = jnp.array(weights["wpe.weight"])  # (1024, 768)
ln_f_g = jnp.array(weights["ln_f.weight"])
ln_f_b = jnp.array(weights["ln_f.bias"])

print(f"\nWeight shapes for layer 0:")
for k, v in sorted(lw.items()):
    print(f"  {k:20s}: {v.shape}")
print(f"  wpe: {wpe.shape}")

# Trace single layer
jaxpr_1layer = make_jaxpr(gpt2_single_layer_hf)(
    x_dummy,
    lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
    lw['w_proj'], lw['b_proj'],
    lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
    lw['w_mlp_proj'], lw['b_mlp_proj'],
    ln_f_g, ln_f_b
)

# Count primitives
op_counts = Counter()
def count_ops(jaxpr):
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        op_counts[name] += 1
        # Recurse into sub-jaxprs
        for p_name, p_val in eqn.params.items():
            if hasattr(p_val, 'jaxpr'):
                sub = p_val.jaxpr if hasattr(p_val, 'jaxpr') else p_val
                count_ops(sub)
            elif hasattr(p_val, 'eqns'):
                count_ops(p_val)

count_ops(jaxpr_1layer.jaxpr)

print(f"\nJaxpr primitives in single GPT-2 layer ({sum(op_counts.values())} total ops):")
from tt_jax.ops import REGISTRY
supported_ops = set(REGISTRY.keys()) | {'custom_jvp_call', 'pjit'}

missing_ops = set()
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    status = "OK" if op in supported_ops else "MISSING"
    if status == "MISSING":
        missing_ops.add(op)
    print(f"  {op:30s}: {count:4d}  [{status}]")

if missing_ops:
    print(f"\n*** MISSING OPS: {sorted(missing_ops)} ***")
    print("These need to be implemented before we can run GPT-2.")
else:
    print(f"\n*** ALL OPS SUPPORTED! ***")


# ============================================================
# Phase 3: Validate with JAX reference (CPU)
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: JAX CPU reference output")
print("=" * 70)

# Tokenize a real sentence — pad to tile-aligned length (multiple of 32)
# TT-NN requires tile-aligned dimensions for most operations.
text = "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout"
tokens = simple_encode(text)
# Pad or truncate to exactly 32 tokens for tile alignment
if len(tokens) < 32:
    # Pad with EOS token (50256) — GPT-2's padding convention
    tokens = tokens + [50256] * (32 - len(tokens))
elif len(tokens) > 32:
    tokens = tokens[:32]
print(f"Input: '{text}'")
print(f"Tokens ({len(tokens)}): {tokens}")
print(f"Decoded back: '{simple_decode(tokens)}'")
print(f"Sequence length: {len(tokens)}")

# Get token embeddings from the real weights
wte = jnp.array(weights["wte.weight"])  # (50257, 768)
wpe_full = jnp.array(weights["wpe.weight"])  # (1024, 768)

# Look up embeddings
token_embeds = wte[jnp.array(tokens)]  # (T, 768)
seq_len_real = len(tokens)
pos_embeds = wpe_full[:seq_len_real, :]  # (T, 768)
x_input = (token_embeds + pos_embeds)[None, :, :]  # (1, T, 768)
print(f"Input tensor shape: {x_input.shape}")

# Run single layer on JAX CPU
t0 = time.time()
jax_out_1layer = gpt2_single_layer_hf(
    x_input,
    lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
    lw['w_proj'], lw['b_proj'],
    lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
    lw['w_mlp_proj'], lw['b_mlp_proj'],
    ln_f_g, ln_f_b
)
t1 = time.time()
print(f"\nJAX CPU single layer: {(t1-t0)*1000:.1f}ms")
print(f"Output shape: {jax_out_1layer.shape}")
print(f"Output stats: mean={float(jax_out_1layer.mean()):.4f}, "
      f"std={float(jax_out_1layer.std()):.4f}, "
      f"min={float(jax_out_1layer.min()):.4f}, "
      f"max={float(jax_out_1layer.max()):.4f}")


# ============================================================
# Phase 4: Run on Blackhole (only if all ops are supported)
# ============================================================

if missing_ops:
    # ============================================================
    # Phase 4-ALT: Analyze missing ops and estimate effort
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Missing ops analysis")
    print("=" * 70)

    print(f"\nMissing ops: {sorted(missing_ops)}")
    print("\nAnalysis of each missing op:")

    # Print the Jaxpr to see how missing ops are used
    for eqn in jaxpr_1layer.jaxpr.eqns:
        if eqn.primitive.name in missing_ops:
            print(f"\n  {eqn.primitive.name}:")
            print(f"    params: {eqn.params}")
            in_shapes = [v.aval.shape if hasattr(v, 'aval') else '(literal)' for v in eqn.invars]
            out_shapes = [v.aval.shape for v in eqn.outvars]
            print(f"    input shapes: {in_shapes}")
            print(f"    output shapes: {out_shapes}")

    print(f"\nCannot run on Blackhole — missing ops: {sorted(missing_ops)}")
    print("Need to implement these before proceeding.")

else:
    print("\n" + "=" * 70)
    print("PHASE 4: Running GPT-2 layer on Blackhole via Jaxpr interpreter")
    print("=" * 70)

    import ttnn

    # Reset device
    os.system("tt-smi -r 0 2>/dev/null")
    time.sleep(2)
    device = ttnn.open_device(device_id=0)

    try:
        from tt_jax.interpret import Interpreter

        interp = Interpreter(device)

        # Re-trace with the actual input shape
        jaxpr_real = make_jaxpr(gpt2_single_layer_hf)(
            x_input,
            lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
            lw['w_proj'], lw['b_proj'],
            lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
            lw['w_mlp_proj'], lw['b_mlp_proj'],
            ln_f_g, ln_f_b
        )

        args = [
            np.array(x_input),
            np.array(lw['ln1_g']), np.array(lw['ln1_b']),
            np.array(lw['w_attn']), np.array(lw['b_attn']),
            np.array(lw['w_proj']), np.array(lw['b_proj']),
            np.array(lw['ln2_g']), np.array(lw['ln2_b']),
            np.array(lw['w_fc']), np.array(lw['b_fc']),
            np.array(lw['w_mlp_proj']), np.array(lw['b_mlp_proj']),
            np.array(ln_f_g), np.array(ln_f_b),
        ]

        print(f"\nRunning GPT-2 layer 0 on Blackhole (seq_len={seq_len_real})...")
        t0 = time.time()
        tt_out = interp.run(jaxpr_real, args)
        t1 = time.time()
        print(f"TT-NN execution: {(t1-t0)*1000:.1f}ms")

        # Compare with JAX reference
        jax_ref = np.array(jax_out_1layer)
        tt_np = np.array(tt_out) if not isinstance(tt_out, np.ndarray) else tt_out

        print(f"\nTT-NN output shape: {tt_np.shape}")
        print(f"TT-NN output stats: mean={tt_np.mean():.4f}, std={tt_np.std():.4f}")
        print(f"JAX   output stats: mean={jax_ref.mean():.4f}, std={jax_ref.std():.4f}")

        # Accuracy check
        if tt_np.shape == jax_ref.shape:
            abs_err = np.abs(tt_np - jax_ref)
            rel_err = abs_err / (np.abs(jax_ref) + 1e-6)
            cos_sim = np.dot(tt_np.flatten(), jax_ref.flatten()) / (
                np.linalg.norm(tt_np.flatten()) * np.linalg.norm(jax_ref.flatten()) + 1e-8)
            print(f"\nAccuracy vs JAX CPU reference:")
            print(f"  Max absolute error: {abs_err.max():.6f}")
            print(f"  Mean absolute error: {abs_err.mean():.6f}")
            print(f"  Max relative error: {rel_err.max():.6f}")
            print(f"  Mean relative error: {rel_err.mean():.6f}")
            print(f"  Cosine similarity: {cos_sim:.6f}")
        else:
            print(f"  Shape mismatch: TT-NN={tt_np.shape} vs JAX={jax_ref.shape}")

        # ============================================================
        # Phase 5: Full 12-layer GPT-2
        # ============================================================

        print("\n" + "=" * 70)
        print("PHASE 5: Full 12-layer GPT-2 on Blackhole")
        print("=" * 70)

        def gpt2_full_hf(x, *layer_args):
            """Full GPT-2: 12 layers + final layernorm.

            layer_args: 12 * (ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                               ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj)
                        + ln_f_g, ln_f_b
            """
            n_per_layer = 12
            n_layers = (len(layer_args) - 2) // n_per_layer

            for i in range(n_layers):
                off = i * n_per_layer
                x = gpt2_block_hf(
                    x,
                    layer_args[off+0], layer_args[off+1],
                    layer_args[off+2], layer_args[off+3],
                    layer_args[off+4], layer_args[off+5],
                    layer_args[off+6], layer_args[off+7],
                    layer_args[off+8], layer_args[off+9],
                    layer_args[off+10], layer_args[off+11],
                    12
                )
            x = layer_norm(x, layer_args[-2], layer_args[-1])
            return x

        # Collect all layer weights
        all_layer_args_jax = []
        for layer_idx in range(12):
            lw_i = get_layer_weights(layer_idx)
            all_layer_args_jax.extend([
                lw_i['ln1_g'], lw_i['ln1_b'],
                lw_i['w_attn'], lw_i['b_attn'],
                lw_i['w_proj'], lw_i['b_proj'],
                lw_i['ln2_g'], lw_i['ln2_b'],
                lw_i['w_fc'], lw_i['b_fc'],
                lw_i['w_mlp_proj'], lw_i['b_mlp_proj'],
            ])
        all_layer_args_jax.extend([ln_f_g, ln_f_b])

        print(f"Total weight tensors: {len(all_layer_args_jax)}")

        # JAX CPU reference for full model
        print("\nRunning full 12-layer GPT-2 on JAX CPU...")
        t0 = time.time()
        jax_full_out = gpt2_full_hf(x_input, *all_layer_args_jax)
        t1 = time.time()
        print(f"JAX CPU full model: {(t1-t0)*1000:.1f}ms")
        print(f"Output shape: {jax_full_out.shape}")

        # Project to vocab (logits) — tied embeddings
        logits = jnp.dot(jax_full_out, wte.T)  # (1, T, 50257)
        next_token_logits = logits[0, -1, :]
        top5_ids = jnp.argsort(next_token_logits)[-5:][::-1]
        print(f"\nJAX CPU top-5 next tokens for '{text}':")
        for tid in top5_ids:
            tok = simple_decode([int(tid)])
            prob = float(jax.nn.softmax(next_token_logits)[int(tid)])
            print(f"  '{tok}' (id={int(tid)}, prob={prob:.4f})")

        # Trace full model
        print("\nTracing full 12-layer GPT-2...")
        t0 = time.time()
        jaxpr_full = make_jaxpr(gpt2_full_hf)(x_input, *all_layer_args_jax)
        t1 = time.time()
        print(f"Trace time: {(t1-t0)*1000:.1f}ms")

        op_counts_full = Counter()
        def count_ops_full(jaxpr):
            for eqn in jaxpr.eqns:
                op_counts_full[eqn.primitive.name] += 1
                for p_name, p_val in eqn.params.items():
                    if hasattr(p_val, 'jaxpr'):
                        count_ops_full(p_val.jaxpr if hasattr(p_val, 'jaxpr') else p_val)
                    elif hasattr(p_val, 'eqns'):
                        count_ops_full(p_val)
        count_ops_full(jaxpr_full.jaxpr)
        print(f"Total Jaxpr ops in full model: {sum(op_counts_full.values())}")

        # Run full model on Blackhole
        all_args_np = [np.array(x_input)] + [np.array(a) for a in all_layer_args_jax]

        print(f"\nRunning full 12-layer GPT-2 on Blackhole...")
        interp2 = Interpreter(device)
        t0 = time.time()
        tt_full_out = interp2.run(jaxpr_full, all_args_np)
        t1 = time.time()
        print(f"TT-NN full model execution: {(t1-t0)*1000:.1f}ms")

        # Compare
        jax_full_ref = np.array(jax_full_out)
        tt_full_np = np.array(tt_full_out) if not isinstance(tt_full_out, np.ndarray) else tt_full_out

        print(f"\nFull model output comparison:")
        print(f"  TT-NN shape: {tt_full_np.shape}, JAX shape: {jax_full_ref.shape}")

        if tt_full_np.shape == jax_full_ref.shape:
            abs_err_full = np.abs(tt_full_np - jax_full_ref)
            cos_sim_full = np.dot(tt_full_np.flatten(), jax_full_ref.flatten()) / (
                np.linalg.norm(tt_full_np.flatten()) * np.linalg.norm(jax_full_ref.flatten()) + 1e-8)
            print(f"  Max absolute error: {abs_err_full.max():.4f}")
            print(f"  Mean absolute error: {abs_err_full.mean():.4f}")
            print(f"  Cosine similarity: {cos_sim_full:.6f}")

            # TT-NN next token predictions
            tt_logits = tt_full_np @ np.array(wte).T
            tt_next_logits = tt_logits[0, -1, :]
            tt_exp = np.exp(tt_next_logits - tt_next_logits.max())
            tt_probs = tt_exp / tt_exp.sum()
            top5_tt = np.argsort(tt_next_logits)[-5:][::-1]
            print(f"\nTT-NN Blackhole top-5 next tokens for '{text}':")
            for tid in top5_tt:
                tok = simple_decode([int(tid)])
                print(f"  '{tok}' (id={int(tid)}, prob={tt_probs[tid]:.4f})")

            # Check if top-1 matches
            jax_top1 = int(top5_ids[0])
            tt_top1 = int(top5_tt[0])
            match_str = "MATCH" if jax_top1 == tt_top1 else "MISMATCH"
            print(f"\nTop-1 prediction: JAX='{simple_decode([jax_top1])}' vs "
                  f"TT-NN='{simple_decode([tt_top1])}' [{match_str}]")

    except Exception as e:
        import traceback
        print(f"\n*** ERROR: {e}")
        traceback.print_exc()

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)
        print("Device closed.")

print("\n" + "=" * 70)
print("EXPERIMENT 27 COMPLETE")
print("=" * 70)
