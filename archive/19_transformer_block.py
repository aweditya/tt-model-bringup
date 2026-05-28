"""
Experiment 19: Full Transformer Encoder Block on Blackhole
==========================================================
Build a complete transformer encoder block using direct TT-NN ops:
  - Multi-head self-attention (4 heads, d_model=256, d_k=64)
  - Add & LayerNorm (residual + normalization)
  - Feed-forward network (256 -> 1024 -> 256)
  - Add & LayerNorm again

Tests:
  1. Correctness vs PyTorch reference (same weights)
  2. Eager vs trace-captured performance
  3. Sequence length scaling (64, 128, 256, 512)
  4. Explore ttnn.transformer module utilities
"""

import ttnn
import torch
import torch.nn as nn
import time
import math

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
N_CORES = grid.x * grid.y
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {N_CORES} cores")
print()

# ============================================================
# Configuration
# ============================================================
BATCH = 1
SEQ_LEN = 128
D_MODEL = 256
N_HEADS = 4
D_K = D_MODEL // N_HEADS  # 64
D_FF = 1024

print(f"Config: batch={BATCH}, seq_len={SEQ_LEN}, d_model={D_MODEL}, "
      f"n_heads={N_HEADS}, d_k={D_K}, d_ff={D_FF}")
print()


# ============================================================
# Helpers
# ============================================================
def to_ttnn(t, dev=device):
    """Convert torch tensor to TT-NN on device, tile-aligned."""
    if isinstance(t, (int, float)):
        t = torch.tensor([[t]], dtype=torch.float32)
    while t.dim() < 2:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)


def from_ttnn(t, shape):
    """Read back from TT-NN and crop to original shape."""
    out = ttnn.to_torch(t).float()
    # Crop away padding
    slices = [slice(0, s) for s in shape]
    while out.dim() > len(shape):
        out = out.squeeze(0)
    while out.dim() < len(shape):
        out = out.unsqueeze(0)
    return out[tuple(slices)]


def dealloc(*tensors):
    for t in tensors:
        try:
            t.deallocate()
        except:
            pass


# ============================================================
# TEST 0: Discover available ops
# ============================================================
print("=" * 60)
print("TEST 0: Discover available ops for transformer block")
print("=" * 60)
print()

# Check layer_norm
norm_ops = [x for x in dir(ttnn) if 'norm' in x.lower()]
print(f"  Norm-related in ttnn: {norm_ops}")

# Check gelu
gelu_ops = [x for x in dir(ttnn) if 'gelu' in x.lower()]
print(f"  GELU-related in ttnn: {gelu_ops}")

# Check ttnn.transformer module
if hasattr(ttnn, 'transformer'):
    transformer_ops = [x for x in dir(ttnn.transformer) if not x.startswith('_')]
    print(f"  ttnn.transformer contents: {transformer_ops}")
print()

# Test layer_norm
print("  Testing ttnn.layer_norm...")
try:
    x_test = torch.randn(1, 32, 256)
    x_tt = to_ttnn(x_test)
    # Try with weight and bias
    gamma = torch.ones(256)
    beta = torch.zeros(256)
    gamma_tt = to_ttnn(gamma)
    beta_tt = to_ttnn(beta)
    out = ttnn.layer_norm(x_tt, weight=gamma_tt, bias=beta_tt)
    ttnn.synchronize_device(device)
    out_torch = ttnn.to_torch(out).float().squeeze()[:32, :256]
    ref = torch.nn.functional.layer_norm(x_test.squeeze(), [256])
    err = (out_torch - ref).abs()
    print(f"    ttnn.layer_norm: WORKS, max_err={err.max().item():.6f}, mean_err={err.mean().item():.6f}")
    HAS_LAYER_NORM = True
    dealloc(x_tt, gamma_tt, beta_tt, out)
except Exception as e:
    print(f"    ttnn.layer_norm: FAILED - {e}")
    HAS_LAYER_NORM = False

# Test gelu
print("  Testing ttnn.gelu...")
try:
    x_test = torch.randn(32, 256)
    x_tt = to_ttnn(x_test)
    out = ttnn.gelu(x_tt)
    ttnn.synchronize_device(device)
    out_torch = ttnn.to_torch(out).float().squeeze()[:32, :256]
    ref = torch.nn.functional.gelu(x_test)
    err = (out_torch - ref).abs()
    print(f"    ttnn.gelu: WORKS, max_err={err.max().item():.4f}, mean_err={err.mean().item():.4f}")
    HAS_GELU = True
    dealloc(x_tt, out)
except Exception as e:
    print(f"    ttnn.gelu: FAILED - {e}")
    HAS_GELU = False

# Test concatenate_heads
print("  Testing ttnn.transformer.concatenate_heads...")
try:
    # concatenate_heads expects [batch, heads, seq, d_k] -> [batch, 1, seq, d_model]
    x_test = torch.randn(1, N_HEADS, 32, D_K)
    x_tt = to_ttnn(x_test)
    out = ttnn.transformer.concatenate_heads(x_tt)
    ttnn.synchronize_device(device)
    print(f"    concatenate_heads: WORKS, input {x_test.shape} -> output {out.shape}")
    HAS_CONCAT_HEADS = True
    dealloc(x_tt, out)
except Exception as e:
    print(f"    concatenate_heads: FAILED - {e}")
    HAS_CONCAT_HEADS = False

print()
activation_name = "gelu" if HAS_GELU else "relu"
print(f"  Will use: layer_norm={'ttnn.layer_norm' if HAS_LAYER_NORM else 'manual'}, "
      f"activation={activation_name}, "
      f"concat_heads={'ttnn.transformer.concatenate_heads' if HAS_CONCAT_HEADS else 'reshape'}")
print()


# ============================================================
# Initialize weights (shared between PyTorch and TT-NN)
# ============================================================
torch.manual_seed(42)

# Attention weights: Q, K, V, and output projections
Wq = torch.randn(D_MODEL, D_MODEL) * (1.0 / D_MODEL ** 0.5)
Wk = torch.randn(D_MODEL, D_MODEL) * (1.0 / D_MODEL ** 0.5)
Wv = torch.randn(D_MODEL, D_MODEL) * (1.0 / D_MODEL ** 0.5)
Wo = torch.randn(D_MODEL, D_MODEL) * (1.0 / D_MODEL ** 0.5)

# Layer norm 1 parameters
ln1_gamma = torch.ones(D_MODEL)
ln1_beta = torch.zeros(D_MODEL)

# FFN weights (no biases — avoids TT-NN broadcast issues, common in modern transformers)
W1 = torch.randn(D_MODEL, D_FF) * (1.0 / D_MODEL ** 0.5)
W2 = torch.randn(D_FF, D_MODEL) * (1.0 / D_FF ** 0.5)

# Layer norm 2 parameters
ln2_gamma = torch.ones(D_MODEL)
ln2_beta = torch.zeros(D_MODEL)


# ============================================================
# PyTorch reference implementation
# ============================================================
def transformer_block_pytorch(x, Wq, Wk, Wv, Wo, ln1_g, ln1_b, W1, W2, ln2_g, ln2_b):
    """Manual transformer encoder block in PyTorch for reference."""
    B, S, D = x.shape

    # Multi-head self-attention
    Q = x @ Wq  # [B, S, D]
    K = x @ Wk
    V = x @ Wv

    # Reshape to [B, N_HEADS, S, D_K]
    Q = Q.view(B, S, N_HEADS, D_K).transpose(1, 2)
    K = K.view(B, S, N_HEADS, D_K).transpose(1, 2)
    V = V.view(B, S, N_HEADS, D_K).transpose(1, 2)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(D_K)
    scores = Q @ K.transpose(-2, -1) * scale  # [B, N_HEADS, S, S]
    weights = torch.softmax(scores, dim=-1)
    attn_out = weights @ V  # [B, N_HEADS, S, D_K]

    # Concatenate heads
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

    # Output projection
    attn_out = attn_out @ Wo

    # Add & Norm 1
    x = torch.nn.functional.layer_norm(x + attn_out, [D], ln1_g, ln1_b)

    # FFN (no biases)
    ff = x @ W1
    if HAS_GELU:
        ff = torch.nn.functional.gelu(ff)
    else:
        ff = torch.relu(ff)
    ff = ff @ W2

    # Add & Norm 2
    x = torch.nn.functional.layer_norm(x + ff, [D], ln2_g, ln2_b)

    return x


# ============================================================
# TT-NN implementation
# ============================================================
# Upload weights to device
Wq_tt = to_ttnn(Wq)
Wk_tt = to_ttnn(Wk)
Wv_tt = to_ttnn(Wv)
Wo_tt = to_ttnn(Wo)
ln1_g_tt = to_ttnn(ln1_gamma)
ln1_b_tt = to_ttnn(ln1_beta)
W1_tt = to_ttnn(W1)
W2_tt = to_ttnn(W2)
ln2_g_tt = to_ttnn(ln2_gamma)
ln2_b_tt = to_ttnn(ln2_beta)


def manual_layer_norm_ttnn(x_tt, gamma_tt, beta_tt):
    """Manual layer norm on TT-NN: (x - mean) / sqrt(var + eps) * gamma + beta."""
    mean = ttnn.mean(x_tt, dim=-1, keepdim=True)
    x_centered = ttnn.subtract(x_tt, mean)
    var = ttnn.mean(ttnn.multiply(x_centered, x_centered), dim=-1, keepdim=True)
    eps_tt = to_ttnn(torch.tensor([[1e-5]]))
    var_eps = ttnn.add(var, eps_tt)
    inv_std = ttnn.rsqrt(var_eps)
    normed = ttnn.multiply(x_centered, inv_std)
    scaled = ttnn.multiply(normed, gamma_tt)
    result = ttnn.add(scaled, beta_tt)
    dealloc(mean, x_centered, var, eps_tt, var_eps, inv_std, normed, scaled)
    return result


def transformer_block_ttnn(x_tt, seq_len):
    """Full transformer encoder block on TT-NN."""
    # x_tt shape: [1, seq_len, D_MODEL] (tile-padded)

    # --- Multi-head self-attention ---
    # Q, K, V projections: [1, seq, D_MODEL] @ [D_MODEL, D_MODEL] -> [1, seq, D_MODEL]
    Q = ttnn.matmul(x_tt, Wq_tt)
    K = ttnn.matmul(x_tt, Wk_tt)
    V = ttnn.matmul(x_tt, Wv_tt)

    # Reshape to [BATCH, N_HEADS, seq, D_K] for SDPA
    # From [1, seq, D_MODEL] -> [1, seq, N_HEADS, D_K] -> [1, N_HEADS, seq, D_K]
    Q = ttnn.reshape(Q, (BATCH, seq_len, N_HEADS, D_K))
    Q = ttnn.transpose(Q, 1, 2)
    K = ttnn.reshape(K, (BATCH, seq_len, N_HEADS, D_K))
    K = ttnn.transpose(K, 1, 2)
    V = ttnn.reshape(V, (BATCH, seq_len, N_HEADS, D_K))
    V = ttnn.transpose(V, 1, 2)

    # Scaled dot-product attention (FlashAttention-2)
    try:
        attn_out = ttnn.transformer.scaled_dot_product_attention(Q, K, V)
    except Exception as e:
        # Fallback: manual attention
        print(f"    [WARN] SDPA failed ({e}), falling back to manual attention")
        scale = 1.0 / math.sqrt(D_K)
        K_T = ttnn.transpose(K, -2, -1)
        scores = ttnn.matmul(Q, K_T)
        scores = ttnn.multiply(scores, scale)
        weights = ttnn.softmax(scores, dim=-1)
        attn_out = ttnn.matmul(weights, V)
        dealloc(K_T, scores, weights)

    dealloc(Q, K, V)

    # Concatenate heads: [1, N_HEADS, seq, D_K] -> [1, seq, D_MODEL]
    if HAS_CONCAT_HEADS:
        try:
            attn_out = ttnn.transformer.concatenate_heads(attn_out)
            # Should give [1, 1, seq, D_MODEL] -- reshape to [1, seq, D_MODEL]
            attn_out = ttnn.reshape(attn_out, (BATCH, seq_len, D_MODEL))
        except Exception:
            attn_out = ttnn.transpose(attn_out, 1, 2)
            attn_out = ttnn.reshape(attn_out, (BATCH, seq_len, D_MODEL))
    else:
        attn_out = ttnn.transpose(attn_out, 1, 2)
        attn_out = ttnn.reshape(attn_out, (BATCH, seq_len, D_MODEL))

    # Output projection
    attn_out = ttnn.matmul(attn_out, Wo_tt)

    # --- Add & Norm 1 ---
    residual1 = ttnn.add(x_tt, attn_out)
    dealloc(attn_out)

    if HAS_LAYER_NORM:
        normed1 = ttnn.layer_norm(residual1, weight=ln1_g_tt, bias=ln1_b_tt)
    else:
        normed1 = manual_layer_norm_ttnn(residual1, ln1_g_tt, ln1_b_tt)
    dealloc(residual1)

    # --- Feed-Forward Network (no biases) ---
    ff = ttnn.matmul(normed1, W1_tt)
    if HAS_GELU:
        ff = ttnn.gelu(ff)
    else:
        ff = ttnn.relu(ff)
    ff = ttnn.matmul(ff, W2_tt)

    # --- Add & Norm 2 ---
    residual2 = ttnn.add(normed1, ff)
    dealloc(normed1, ff)

    if HAS_LAYER_NORM:
        output = ttnn.layer_norm(residual2, weight=ln2_g_tt, bias=ln2_b_tt)
    else:
        output = manual_layer_norm_ttnn(residual2, ln2_g_tt, ln2_b_tt)
    dealloc(residual2)

    return output


# ============================================================
# TEST 1: Correctness — TT-NN vs PyTorch
# ============================================================
print("=" * 60)
print("TEST 1: Correctness — TT-NN vs PyTorch reference")
print("=" * 60)
print()

torch.manual_seed(123)
x_input = torch.randn(BATCH, SEQ_LEN, D_MODEL)

# PyTorch reference
ref_output = transformer_block_pytorch(
    x_input, Wq, Wk, Wv, Wo, ln1_gamma, ln1_beta,
    W1, W2, ln2_gamma, ln2_beta
)
print(f"  PyTorch reference output shape: {ref_output.shape}")
print(f"  PyTorch output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")

# TT-NN
x_tt = to_ttnn(x_input)
try:
    out_tt = transformer_block_ttnn(x_tt, SEQ_LEN)
    ttnn.synchronize_device(device)
    out_torch = from_ttnn(out_tt, (BATCH, SEQ_LEN, D_MODEL))
    dealloc(out_tt)

    abs_err = (out_torch - ref_output).abs()
    print(f"  TT-NN output shape: {out_torch.shape}")
    print(f"  TT-NN output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")
    print(f"  Max absolute error:  {abs_err.max().item():.4f}")
    print(f"  Mean absolute error: {abs_err.mean().item():.4f}")
    print(f"  Relative error (vs output range): {abs_err.max().item() / (ref_output.max() - ref_output.min()).item() * 100:.2f}%")

    TTNN_WORKS = True
except Exception as e:
    print(f"  TT-NN FAILED: {e}")
    import traceback
    traceback.print_exc()
    TTNN_WORKS = False

dealloc(x_tt)
print()


# ============================================================
# TEST 2: Eager vs Trace-captured
# ============================================================
if TTNN_WORKS:
    print("=" * 60)
    print("TEST 2: Eager vs Trace-captured performance")
    print("=" * 60)
    print()

    x_data = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    x_tt = to_ttnn(x_data)

    # --- Eager benchmark ---
    # Warmup
    for _ in range(3):
        out = transformer_block_ttnn(x_tt, SEQ_LEN)
        ttnn.synchronize_device(device)
        dealloc(out)

    REPS = 20
    times_eager = []
    for _ in range(REPS):
        start = time.perf_counter()
        out = transformer_block_ttnn(x_tt, SEQ_LEN)
        ttnn.synchronize_device(device)
        times_eager.append(time.perf_counter() - start)
        dealloc(out)

    avg_eager = sum(times_eager) / len(times_eager) * 1000
    print(f"  Eager:  {avg_eager:.3f} ms  (avg of {REPS} runs)")

    # --- Trace capture ---
    try:
        # Dry run for shape
        out_dry = transformer_block_ttnn(x_tt, SEQ_LEN)
        ttnn.synchronize_device(device)
        dealloc(out_dry)

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        out_traced = transformer_block_ttnn(x_tt, SEQ_LEN)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # Warmup trace
        for _ in range(3):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

        times_trace = []
        for _ in range(REPS):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            times_trace.append(time.perf_counter() - start)

        avg_trace = sum(times_trace) / len(times_trace) * 1000
        print(f"  Traced: {avg_trace:.3f} ms  (avg of {REPS} runs)")
        print(f"  Speedup: {avg_eager / avg_trace:.2f}x")

        ttnn.release_trace(device, trace_id)
        dealloc(out_traced)
    except Exception as e:
        print(f"  Trace capture FAILED: {e}")
        import traceback
        traceback.print_exc()

    dealloc(x_tt)
    print()


# ============================================================
# TEST 3: Sequence length scaling
# ============================================================
if TTNN_WORKS:
    print("=" * 60)
    print("TEST 3: Sequence length scaling")
    print("=" * 60)
    print()

    SEQ_LENS = [64, 128, 256, 512]
    REPS = 10

    print(f"  {'Seq Len':<10} {'Eager (ms)':<15} {'Traced (ms)':<15} {'Speedup':<10}")
    print(f"  {'-'*50}")

    for sl in SEQ_LENS:
        x_data = torch.randn(BATCH, sl, D_MODEL)
        x_tt = to_ttnn(x_data)

        # Warmup
        try:
            for _ in range(2):
                out = transformer_block_ttnn(x_tt, sl)
                ttnn.synchronize_device(device)
                dealloc(out)
        except Exception as e:
            print(f"  {sl:<10} FAILED: {e}")
            dealloc(x_tt)
            continue

        # Eager
        times_e = []
        for _ in range(REPS):
            start = time.perf_counter()
            out = transformer_block_ttnn(x_tt, sl)
            ttnn.synchronize_device(device)
            times_e.append(time.perf_counter() - start)
            dealloc(out)
        avg_e = sum(times_e) / len(times_e) * 1000

        # Trace
        avg_t_str = "N/A"
        speedup_str = ""
        try:
            out_dry = transformer_block_ttnn(x_tt, sl)
            ttnn.synchronize_device(device)
            dealloc(out_dry)

            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out_tr = transformer_block_ttnn(x_tt, sl)
            ttnn.end_trace_capture(device, tid, cq_id=0)

            for _ in range(2):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            times_t = []
            for _ in range(REPS):
                start = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                times_t.append(time.perf_counter() - start)
            avg_t = sum(times_t) / len(times_t) * 1000
            avg_t_str = f"{avg_t:.3f}"
            speedup_str = f"{avg_e / avg_t:.2f}x"

            ttnn.release_trace(device, tid)
            dealloc(out_tr)
        except Exception as e:
            avg_t_str = f"FAIL"
            speedup_str = f"({e})"[:30]

        dealloc(x_tt)
        print(f"  {sl:<10} {avg_e:<15.3f} {avg_t_str:<15} {speedup_str}")

    print()


# ============================================================
# TEST 4: What TT-NN transformer utilities exist?
# ============================================================
print("=" * 60)
print("TEST 4: TT-NN transformer module exploration")
print("=" * 60)
print()

if hasattr(ttnn, 'transformer'):
    all_attrs = [x for x in dir(ttnn.transformer) if not x.startswith('_')]
    print(f"  ttnn.transformer has {len(all_attrs)} public attributes:")
    for attr in sorted(all_attrs):
        obj = getattr(ttnn.transformer, attr)
        doc = ""
        if hasattr(obj, '__doc__') and obj.__doc__:
            doc = obj.__doc__.split('\n')[0][:80]
        print(f"    {attr:<50} {doc}")
    print()

    # Categorize
    attn_ops = [x for x in all_attrs if 'attention' in x.lower() or 'sdpa' in x.lower()]
    head_ops = [x for x in all_attrs if 'head' in x.lower() or 'split' in x.lower()]
    norm_ops = [x for x in all_attrs if 'norm' in x.lower() or 'soft' in x.lower()]
    other_ops = [x for x in all_attrs if x not in attn_ops + head_ops + norm_ops]

    print(f"  Attention ops ({len(attn_ops)}): {attn_ops}")
    print(f"  Head manipulation ({len(head_ops)}): {head_ops}")
    print(f"  Normalization/softmax ({len(norm_ops)}): {norm_ops}")
    print(f"  Other ({len(other_ops)}): {other_ops}")
else:
    print("  ttnn.transformer module NOT FOUND")

print()


# ============================================================
# Op count summary
# ============================================================
print("=" * 60)
print("Summary: Transformer Block Op Breakdown")
print("=" * 60)
print(f"""
  Architecture: Transformer encoder block
    batch={BATCH}, seq_len={SEQ_LEN}, d_model={D_MODEL}, heads={N_HEADS}, d_ff={D_FF}

  Attention sub-block:
    - 3x matmul (Q, K, V projections)
    - 3x reshape + transpose (split heads)
    - 1x SDPA (FlashAttention-2) or 4 ops (matmul, scale, softmax, matmul)
    - 1x concatenate_heads (or transpose + reshape)
    - 1x matmul (output projection)
    - 1x add (residual)
    - 1x layer_norm

  FFN sub-block:
    - 2x matmul (linear layers, no biases)
    - 1x gelu/relu
    - 1x add (residual)
    - 1x layer_norm

  Total: ~15-20 TT-NN ops per forward pass
  Parameters: {(D_MODEL*D_MODEL*4 + D_MODEL*D_FF*2 + D_MODEL*2*2):,} (~{(D_MODEL*D_MODEL*4 + D_MODEL*D_FF*2)*2/1024/1024:.1f} MB in bf16)
""")

# Cleanup weights
dealloc(Wq_tt, Wk_tt, Wv_tt, Wo_tt, ln1_g_tt, ln1_b_tt,
        W1_tt, W2_tt, ln2_g_tt, ln2_b_tt)

ttnn.close_device(device)
print("Done!")
