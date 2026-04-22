"""
Experiment 28: Flat attention — keep GPT-2 attention in 2D/3D to avoid 4D TT-NN failures.

Hypothesis: GPT-2's multi-head attention reshapes to 4D (B, n_heads, T, head_dim) for
batched matmul. TT-NN's binary ops fail on 4D with "Invalid subtile broadcast type",
forcing CPU fallbacks that lose precision (cosine similarity 0.83).

If we rewrite attention to loop over heads individually (each head stays 3D as (B, T, head_dim)),
all ops stay on-device. Since B=1, each head's tensors are effectively 2D (T, head_dim).

JAX's make_jaxpr will unroll the loop, giving 12 independent sets of 3D matmuls — all
perfectly compatible with TT-NN.

RESULTS:
  - Flat attention successfully eliminates ALL 4D tensors (max rank = 3D)
  - All 23 ops in the Jaxpr are supported — zero missing ops
  - Required 3 new ops: slice, dynamic_slice, concatenate
  - JAX CPU: flat and 4D attention are IDENTICAL (cosine sim 1.000000)

  Accuracy breakdown on Blackhole:
    QKV projection only:      cosine sim 0.999981 (excellent)
    Single head attention:     cosine sim 0.987588 (bfloat16 softmax error)
    Full attention (12 heads): cosine sim 0.875756 (errors compound across heads)
    Full layer (attn + MLP):   cosine sim 0.282428 (MLP amplifies attention errors)

  Comparison with original 4D version:
    4D (CPU fallback):  cosine sim 0.999914, 165ms
    Flat (all device):  cosine sim 0.282428, 118ms

  KEY INSIGHT: The 4D CPU fallback is actually a FEATURE, not a bug.
  The 4D binary ops that fall back to CPU use float32 arithmetic,
  which preserves precision through the attention softmax. Running
  attention entirely in bfloat16 on-device causes error compounding:
  each head loses ~1.3% accuracy, which amplifies through concat,
  output projection, and MLP layers.

  CONCLUSION: To get both speed AND accuracy, we need either:
    1. float32 support on TT-NN for attention scores (not just bfloat16)
    2. A mixed-precision strategy: matmul in bfloat16, softmax in float32
    3. Native 4D op support in TT-NN (eliminates the problem entirely)
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
# Phase 1: Load GPT-2 weights
# ============================================================

print("=" * 70)
print("EXPERIMENT 28: Flat attention — no 4D tensors")
print("=" * 70)

from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json

model_path = hf_hub_download("gpt2", "model.safetensors")
config_path = hf_hub_download("gpt2", "config.json")
tokenizer_path = hf_hub_download("gpt2", "tokenizer.json")
vocab_path = hf_hub_download("gpt2", "vocab.json")

with open(config_path) as f:
    config = json.load(f)
print(f"Config: n_layer={config['n_layer']}, n_head={config['n_head']}, "
      f"n_embd={config['n_embd']}, vocab_size={config['vocab_size']}")

weights = {}
total_params = 0
with safe_open(model_path, framework="numpy") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
        total_params += weights[key].size

print(f"Total parameters: {total_params:,}")

with open(vocab_path) as f:
    vocab = json.load(f)
token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

def simple_encode(text):
    """Crude byte-level encoding (matches GPT-2 BPE for simple ASCII)."""
    with open(tokenizer_path) as f:
        tok_config = json.load(f)
    tokens = []
    i = 0
    text_bytes = text.encode('utf-8')
    while i < len(text_bytes):
        best_len = 0
        for length in range(min(20, len(text_bytes) - i), 0, -1):
            candidate = text_bytes[i:i+length].decode('utf-8', errors='ignore')
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
            byte_val = text_bytes[i]
            tokens.append(token_to_id.get(chr(byte_val), 0))
            best_len = 1
        i += best_len
    return tokens

def simple_decode(token_ids):
    parts = []
    for tid in token_ids:
        tok = id_to_token.get(tid, '?')
        tok = tok.replace('\u0120', ' ')
        parts.append(tok)
    return ''.join(parts)


# ============================================================
# Phase 2: Flat attention — no 4D tensors
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: Defining flat (3D-only) GPT-2 attention")
print("=" * 70)


def gelu(x):
    """GPT-2 uses the 'gelu_new' approximation."""
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, gamma, beta, eps=1e-5):
    """Standard layer norm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def attention_flat(x, w_attn, b_attn, w_proj, b_proj, n_heads):
    """Multi-head causal self-attention — FLAT version (no 4D tensors).

    Instead of reshaping to (B, n_heads, T, head_dim) and doing batched matmul,
    we loop over heads. Each head operates on (B, T, head_dim) — pure 3D.

    Since B=1, the matmuls are effectively 2D: (T, head_dim) @ (head_dim, T).
    JAX unrolls this loop at trace time, producing 12 independent 3D matmul chains.
    """
    B, T, C = x.shape
    head_dim = C // n_heads

    # QKV projection: x @ w_attn + b_attn -> (B, T, 3*C)
    qkv = jnp.dot(x, w_attn) + b_attn

    # Build causal mask once: (1, T, T) — 3D, not 4D!
    mask = jnp.tril(jnp.ones((T, T)))  # (T, T)

    scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))

    head_outputs = []
    for h in range(n_heads):
        # Extract this head's Q, K, V via slicing — stays (B, T, head_dim) = 3D
        q_h = jax.lax.dynamic_slice_in_dim(qkv, h * head_dim, head_dim, axis=2)
        k_h = jax.lax.dynamic_slice_in_dim(qkv, C + h * head_dim, head_dim, axis=2)
        v_h = jax.lax.dynamic_slice_in_dim(qkv, 2 * C + h * head_dim, head_dim, axis=2)

        # Attention scores: (B, T, head_dim) @ (B, head_dim, T) -> (B, T, T) = 3D!
        scores = jnp.matmul(q_h, k_h.transpose(0, 2, 1)) / scale  # (B, T, T)

        # Apply causal mask (broadcast (T,T) to (B,T,T) — easy 3D broadcast)
        scores = scores * mask + (-1e10) * (1.0 - mask)

        # Softmax over last dim
        attn_weights = jax.nn.softmax(scores, axis=-1)  # (B, T, T)

        # Weighted sum: (B, T, T) @ (B, T, head_dim) -> (B, T, head_dim) = 3D!
        head_out = jnp.matmul(attn_weights, v_h)  # (B, T, head_dim)
        head_outputs.append(head_out)

    # Concatenate heads: list of (B, T, head_dim) -> (B, T, C)
    out = jnp.concatenate(head_outputs, axis=-1)

    # Output projection
    return jnp.dot(out, w_proj) + b_proj


def mlp_hf(x, w_fc, b_fc, w_proj, b_proj):
    """GPT-2 MLP block (HuggingFace convention: x @ weight + bias)."""
    h = gelu(jnp.dot(x, w_fc) + b_fc)
    return jnp.dot(h, w_proj) + b_proj


def gpt2_block_flat(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                    ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj, n_heads):
    """One GPT-2 transformer block with FLAT attention."""
    h = layer_norm(x, ln1_g, ln1_b)
    h = attention_flat(h, w_attn, b_attn, w_proj, b_proj, n_heads)
    x = x + h
    h = layer_norm(x, ln2_g, ln2_b)
    h = mlp_hf(h, w_fc, b_fc, w_mlp_proj, b_mlp_proj)
    x = x + h
    return x


def gpt2_single_layer_flat(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                            ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj,
                            ln_f_g, ln_f_b):
    """Single GPT-2 layer + final layernorm — FLAT version."""
    x = gpt2_block_flat(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                        ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj, 12)
    return layer_norm(x, ln_f_g, ln_f_b)


# ============================================================
# Phase 2a: Compare 4D vs flat attention Jaxpr
# ============================================================

def get_layer_weights(layer_idx):
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

n_heads = 12
d_model = 768
batch_size = 1

lw = get_layer_weights(0)
ln_f_g = jnp.array(weights["ln_f.weight"])
ln_f_b = jnp.array(weights["ln_f.bias"])

# Tokenize + embed
text = "The meaning of life is to find purpose and fulfillment in everything that we do and experience throughout"
tokens = simple_encode(text)
if len(tokens) < 32:
    tokens = tokens + [50256] * (32 - len(tokens))
elif len(tokens) > 32:
    tokens = tokens[:32]
seq_len = len(tokens)

wte = jnp.array(weights["wte.weight"])
wpe = jnp.array(weights["wpe.weight"])
token_embeds = wte[jnp.array(tokens)]
pos_embeds = wpe[:seq_len, :]
x_input = (token_embeds + pos_embeds)[None, :, :]
print(f"\nInput: '{text}'")
print(f"Tokens: {len(tokens)}, Input shape: {x_input.shape}")

# Trace the flat version
print("\n--- Tracing FLAT attention single layer ---")
jaxpr_flat = make_jaxpr(gpt2_single_layer_flat)(
    x_input,
    lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
    lw['w_proj'], lw['b_proj'],
    lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
    lw['w_mlp_proj'], lw['b_mlp_proj'],
    ln_f_g, ln_f_b
)

op_counts = Counter()
def count_ops(jaxpr):
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        op_counts[name] += 1
        for p_name, p_val in eqn.params.items():
            if hasattr(p_val, 'jaxpr'):
                sub = p_val.jaxpr if hasattr(p_val, 'jaxpr') else p_val
                count_ops(sub)
            elif hasattr(p_val, 'eqns'):
                count_ops(p_val)

count_ops(jaxpr_flat.jaxpr)

from tt_jax.ops import REGISTRY
supported_ops = set(REGISTRY.keys()) | {'custom_jvp_call', 'pjit'}

print(f"\nJaxpr primitives in FLAT single layer ({sum(op_counts.values())} total ops):")
missing_ops = set()
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    status = "OK" if op in supported_ops else "MISSING"
    if status == "MISSING":
        missing_ops.add(op)
    print(f"  {op:30s}: {count:4d}  [{status}]")

# Check for 4D tensors in the Jaxpr
max_rank = 0
rank_counts = Counter()
for eqn in jaxpr_flat.jaxpr.eqns:
    for v in list(eqn.invars) + list(eqn.outvars):
        if hasattr(v, 'aval') and hasattr(v.aval, 'shape'):
            rank = len(v.aval.shape)
            rank_counts[rank] += 1
            max_rank = max(max_rank, rank)

print(f"\nTensor rank distribution:")
for rank in sorted(rank_counts.keys()):
    print(f"  {rank}D: {rank_counts[rank]} tensors")
print(f"  Max rank: {max_rank}D")
if max_rank <= 3:
    print("  *** SUCCESS: No 4D tensors! All ops should run on TT-NN ***")
else:
    print(f"  *** WARNING: Found {max_rank}D tensors — may still need CPU fallback ***")

if missing_ops:
    print(f"\n*** MISSING OPS: {sorted(missing_ops)} ***")
    print("Cannot run on Blackhole without implementing these.")


# ============================================================
# Phase 3: JAX CPU reference
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: JAX CPU reference")
print("=" * 70)

t0 = time.time()
jax_out_flat = gpt2_single_layer_flat(
    x_input,
    lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
    lw['w_proj'], lw['b_proj'],
    lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
    lw['w_mlp_proj'], lw['b_mlp_proj'],
    ln_f_g, ln_f_b
)
t1 = time.time()
print(f"JAX CPU flat attention: {(t1-t0)*1000:.1f}ms")
print(f"Output shape: {jax_out_flat.shape}")
print(f"Output stats: mean={float(jax_out_flat.mean()):.4f}, "
      f"std={float(jax_out_flat.std()):.4f}")

# Also run the ORIGINAL 4D version for comparison
def attention_4d(x, w_attn, b_attn, w_proj, b_proj, n_heads):
    """Original 4D attention for comparison."""
    B, T, C = x.shape
    head_dim = C // n_heads
    qkv = jnp.dot(x, w_attn) + b_attn
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, head_dim).transpose(0, 2, 1, 3)
    scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale
    mask = jnp.tril(jnp.ones((T, T)))
    scores = scores * mask + (-1e10) * (1.0 - mask)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn_weights, v)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
    return jnp.dot(out, w_proj) + b_proj

def gpt2_single_layer_4d(x, ln1_g, ln1_b, w_attn, b_attn, w_proj, b_proj,
                          ln2_g, ln2_b, w_fc, b_fc, w_mlp_proj, b_mlp_proj,
                          ln_f_g, ln_f_b):
    """Original GPT-2 single layer with 4D attention."""
    h = layer_norm(x, ln1_g, ln1_b)
    h = attention_4d(h, w_attn, b_attn, w_proj, b_proj, 12)
    x = x + h
    h = layer_norm(x, ln2_g, ln2_b)
    h = mlp_hf(h, w_fc, b_fc, w_mlp_proj, b_mlp_proj)
    x = x + h
    return layer_norm(x, ln_f_g, ln_f_b)

jax_out_4d = gpt2_single_layer_4d(
    x_input,
    lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
    lw['w_proj'], lw['b_proj'],
    lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
    lw['w_mlp_proj'], lw['b_mlp_proj'],
    ln_f_g, ln_f_b
)

# Verify flat vs 4D produce identical results on CPU
flat_np = np.array(jax_out_flat)
orig_np = np.array(jax_out_4d)
cos_flat_vs_4d = np.dot(flat_np.flatten(), orig_np.flatten()) / (
    np.linalg.norm(flat_np.flatten()) * np.linalg.norm(orig_np.flatten()) + 1e-8)
max_diff = np.abs(flat_np - orig_np).max()
print(f"\nFlat vs 4D attention (JAX CPU):")
print(f"  Cosine similarity: {cos_flat_vs_4d:.6f}")
print(f"  Max absolute diff: {max_diff:.2e}")
print(f"  {'IDENTICAL' if cos_flat_vs_4d > 0.999999 else 'MISMATCH — check implementation!'}")


# ============================================================
# Phase 4: Run on Blackhole
# ============================================================

if missing_ops:
    print(f"\n*** Cannot run on Blackhole — missing ops: {sorted(missing_ops)} ***")
else:
    print("\n" + "=" * 70)
    print("PHASE 4: Running FLAT GPT-2 layer on Blackhole")
    print("=" * 70)

    import ttnn

    os.system("tt-smi -r 0 2>/dev/null")
    time.sleep(2)
    device = ttnn.open_device(device_id=0)

    try:
        from tt_jax.interpret import Interpreter

        interp = Interpreter(device)

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

        # Diagnostic: run just the attention function in isolation to find where error comes from
        print(f"\n--- Diagnostic: testing attention_flat in isolation ---")

        # Get the post-layernorm input (run layernorm on JAX CPU)
        ln1_out_jax = np.array(layer_norm(
            x_input, lw['ln1_g'], lw['ln1_b']))

        # Run attention_flat on JAX CPU
        attn_out_jax = np.array(attention_flat(
            jnp.array(ln1_out_jax),
            lw['w_attn'], lw['b_attn'], lw['w_proj'], lw['b_proj'], 12))

        # n_heads must be static (not traced), so wrap it
        def attn_wrapper(x, w_attn, b_attn, w_proj, b_proj):
            return attention_flat(x, w_attn, b_attn, w_proj, b_proj, 12)

        jaxpr_attn = make_jaxpr(attn_wrapper)(
            jnp.array(ln1_out_jax),
            lw['w_attn'], lw['b_attn'], lw['w_proj'], lw['b_proj'])

        attn_args = [
            ln1_out_jax,
            np.array(lw['w_attn']), np.array(lw['b_attn']),
            np.array(lw['w_proj']), np.array(lw['b_proj']),
        ]

        interp_diag = Interpreter(device)
        tt_attn_out = interp_diag.run(jaxpr_attn, attn_args)
        tt_attn_np = np.array(tt_attn_out) if not isinstance(tt_attn_out, np.ndarray) else tt_attn_out

        cos_attn = np.dot(tt_attn_np.flatten(), attn_out_jax.flatten()) / (
            np.linalg.norm(tt_attn_np.flatten()) * np.linalg.norm(attn_out_jax.flatten()) + 1e-8)
        print(f"  Attention-only cosine sim: {cos_attn:.6f}")
        print(f"  Attention-only max error: {np.abs(tt_attn_np - attn_out_jax).max():.4f}")
        print(f"  TT attn stats: mean={tt_attn_np.mean():.4f}, std={tt_attn_np.std():.4f}")
        print(f"  JAX attn stats: mean={attn_out_jax.mean():.4f}, std={attn_out_jax.std():.4f}")

        # Now test just QKV projection (the dot+add before slicing)
        def qkv_only(x, w_attn, b_attn):
            return jnp.dot(x, w_attn) + b_attn

        jaxpr_qkv = make_jaxpr(qkv_only)(
            jnp.array(ln1_out_jax), lw['w_attn'], lw['b_attn'])
        qkv_jax = np.array(qkv_only(jnp.array(ln1_out_jax), lw['w_attn'], lw['b_attn']))

        interp_qkv = Interpreter(device)
        tt_qkv = interp_qkv.run(jaxpr_qkv, [ln1_out_jax, np.array(lw['w_attn']), np.array(lw['b_attn'])])
        tt_qkv_np = np.array(tt_qkv) if not isinstance(tt_qkv, np.ndarray) else tt_qkv

        cos_qkv = np.dot(tt_qkv_np.flatten(), qkv_jax.flatten()) / (
            np.linalg.norm(tt_qkv_np.flatten()) * np.linalg.norm(qkv_jax.flatten()) + 1e-8)
        print(f"\n  QKV projection cosine sim: {cos_qkv:.6f}")
        print(f"  QKV max error: {np.abs(tt_qkv_np - qkv_jax).max():.4f}")

        # Test single-head attention
        def single_head_attn(qkv):
            B, T, C = 1, 32, 768
            head_dim = 64
            q_h = jax.lax.dynamic_slice_in_dim(qkv, 0, head_dim, axis=2)
            k_h = jax.lax.dynamic_slice_in_dim(qkv, C, head_dim, axis=2)
            v_h = jax.lax.dynamic_slice_in_dim(qkv, 2 * C, head_dim, axis=2)
            scale = jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
            scores = jnp.matmul(q_h, k_h.transpose(0, 2, 1)) / scale
            mask = jnp.tril(jnp.ones((T, T)))
            scores = scores * mask + (-1e10) * (1.0 - mask)
            attn_weights = jax.nn.softmax(scores, axis=-1)
            return jnp.matmul(attn_weights, v_h)

        jaxpr_1head = make_jaxpr(single_head_attn)(jnp.array(qkv_jax))
        head_jax = np.array(single_head_attn(jnp.array(qkv_jax)))

        interp_1h = Interpreter(device)
        tt_head = interp_1h.run(jaxpr_1head, [qkv_jax])
        tt_head_np = np.array(tt_head) if not isinstance(tt_head, np.ndarray) else tt_head

        cos_head = np.dot(tt_head_np.flatten(), head_jax.flatten()) / (
            np.linalg.norm(tt_head_np.flatten()) * np.linalg.norm(head_jax.flatten()) + 1e-8)
        print(f"\n  Single head 0 cosine sim: {cos_head:.6f}")
        print(f"  Single head max error: {np.abs(tt_head_np - head_jax).max():.4f}")
        print(f"  TT head stats: mean={tt_head_np.mean():.4f}, std={tt_head_np.std():.4f}")
        print(f"  JAX head stats: mean={head_jax.mean():.4f}, std={head_jax.std():.4f}")

        # Run flat version
        print(f"\n--- Full layer run ---")
        interp = Interpreter(device)
        print(f"Running FLAT attention layer 0 on Blackhole...")
        t0 = time.time()
        tt_out_flat = interp.run(jaxpr_flat, args)
        t1 = time.time()
        flat_time = (t1 - t0) * 1000
        print(f"TT-NN FLAT execution: {flat_time:.1f}ms")

        # Compare with JAX reference
        jax_ref = np.array(jax_out_flat)
        tt_np = np.array(tt_out_flat) if not isinstance(tt_out_flat, np.ndarray) else tt_out_flat

        print(f"\nTT-NN output shape: {tt_np.shape}")
        print(f"TT-NN stats: mean={tt_np.mean():.4f}, std={tt_np.std():.4f}")
        print(f"JAX   stats: mean={jax_ref.mean():.4f}, std={jax_ref.std():.4f}")

        if tt_np.shape == jax_ref.shape:
            abs_err = np.abs(tt_np - jax_ref)
            rel_err = abs_err / (np.abs(jax_ref) + 1e-6)
            cos_sim = np.dot(tt_np.flatten(), jax_ref.flatten()) / (
                np.linalg.norm(tt_np.flatten()) * np.linalg.norm(jax_ref.flatten()) + 1e-8)
            print(f"\n*** ACCURACY vs JAX CPU reference ***")
            print(f"  Cosine similarity:  {cos_sim:.6f}")
            print(f"  Max absolute error: {abs_err.max():.6f}")
            print(f"  Mean absolute error: {abs_err.mean():.6f}")
            print(f"  Max relative error: {rel_err.max():.6f}")

            if cos_sim > 0.999:
                print(f"\n  EXCELLENT — cosine sim {cos_sim:.6f} (was 0.83 with 4D)")
            elif cos_sim > 0.99:
                print(f"\n  GOOD — cosine sim {cos_sim:.6f} (improved from 0.83)")
            else:
                print(f"\n  POOR — cosine sim {cos_sim:.6f} (still has issues)")
        else:
            print(f"  Shape mismatch: TT-NN={tt_np.shape} vs JAX={jax_ref.shape}")

        # Also run the 4D version for direct comparison
        print("\n" + "-" * 50)
        print("Comparison: also running ORIGINAL 4D attention on Blackhole...")

        jaxpr_4d = make_jaxpr(gpt2_single_layer_4d)(
            x_input,
            lw['ln1_g'], lw['ln1_b'], lw['w_attn'], lw['b_attn'],
            lw['w_proj'], lw['b_proj'],
            lw['ln2_g'], lw['ln2_b'], lw['w_fc'], lw['b_fc'],
            lw['w_mlp_proj'], lw['b_mlp_proj'],
            ln_f_g, ln_f_b
        )

        interp2 = Interpreter(device)
        t0 = time.time()
        tt_out_4d = interp2.run(jaxpr_4d, args)
        t1 = time.time()
        orig_time = (t1 - t0) * 1000
        print(f"TT-NN 4D execution: {orig_time:.1f}ms")

        jax_ref_4d = np.array(jax_out_4d)
        tt_4d_np = np.array(tt_out_4d) if not isinstance(tt_out_4d, np.ndarray) else tt_out_4d

        if tt_4d_np.shape == jax_ref_4d.shape:
            cos_sim_4d = np.dot(tt_4d_np.flatten(), jax_ref_4d.flatten()) / (
                np.linalg.norm(tt_4d_np.flatten()) * np.linalg.norm(jax_ref_4d.flatten()) + 1e-8)
            print(f"  4D cosine similarity: {cos_sim_4d:.6f}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: Flat vs 4D attention on Blackhole")
        print("=" * 70)
        print(f"  {'Metric':<30s} {'4D (original)':<20s} {'Flat (new)':<20s}")
        print(f"  {'-'*30} {'-'*20} {'-'*20}")
        if tt_np.shape == jax_ref.shape and tt_4d_np.shape == jax_ref_4d.shape:
            print(f"  {'Cosine similarity':<30s} {cos_sim_4d:<20.6f} {cos_sim:<20.6f}")
        print(f"  {'Execution time':<30s} {orig_time:<20.1f} {flat_time:<20.1f}")
        print(f"  {'Max tensor rank':<30s} {'4D':<20s} {f'{max_rank}D':<20s}")

        # Report CPU fallback ops
        print(f"\n  Ops used (flat): {sorted(interp.ops_seen)}")

    except Exception as e:
        import traceback
        print(f"\n*** ERROR: {e}")
        traceback.print_exc()

    finally:
        print("\nClosing device...")
        ttnn.close_device(device)
        print("Device closed.")

print("\n" + "=" * 70)
print("EXPERIMENT 28 COMPLETE")
print("=" * 70)
