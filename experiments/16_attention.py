"""
Experiment 16: Self-Attention on Blackhole
==========================================
Hypothesis: We can implement the core transformer self-attention mechanism
using TT-NN primitives on Blackhole. This combines matmul, transpose,
scaling, and softmax — the complete attention pipeline.

Tests:
  1. Manual single-head self-attention (Q @ K^T / sqrt(d_k), softmax, @ V)
  2. Multi-head attention (4 heads, d_model=256, d_k=64)
  3. Benchmark attention at different sequence lengths
  4. Trace-captured attention vs eager
"""

import ttnn
import torch
import time
import math

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {grid.x*grid.y} cores")
print()

# ============================================================
# Helper: create tile-aligned TT-NN tensor
# ============================================================
def to_ttnn(t, dev=device):
    """Convert a torch tensor to TT-NN on device, padding to tile alignment."""
    while t.dim() < 2:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)


# ============================================================
# TEST 1: Manual single-head self-attention
# ============================================================
print("=" * 60)
print("TEST 1: Manual single-head self-attention")
print("=" * 60)
print("  batch=1, seq_len=128, d_model=256, d_k=64")
print()

BATCH = 1
SEQ_LEN = 128
D_MODEL = 256
D_K = 64

torch.manual_seed(42)

# Input: (batch, seq_len, d_model) but we work with 2D on TT-NN: (seq_len, d_model)
x_torch = torch.randn(SEQ_LEN, D_MODEL)

# Weight matrices
Wq_torch = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
Wk_torch = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
Wv_torch = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)

# --- PyTorch reference ---
Q_ref = x_torch @ Wq_torch            # (128, 64)
K_ref = x_torch @ Wk_torch            # (128, 64)
V_ref = x_torch @ Wv_torch            # (128, 64)
scores_ref = Q_ref @ K_ref.T / math.sqrt(D_K)  # (128, 128)
weights_ref = torch.softmax(scores_ref, dim=-1)  # (128, 128)
output_ref = weights_ref @ V_ref       # (128, 64)

print(f"  PyTorch ref output shape: {output_ref.shape}")
print(f"  PyTorch ref output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")

# --- TT-NN implementation ---
print("\n  Running on Blackhole...")

# Upload to device
x_tt = to_ttnn(x_torch)
Wq_tt = to_ttnn(Wq_torch)
Wk_tt = to_ttnn(Wk_torch)
Wv_tt = to_ttnn(Wv_torch)

# Scale factor as a Python float (ttnn.multiply supports scalar)
scale_val = 1.0 / math.sqrt(D_K)  # 0.125

# Q, K, V projections
Q_tt = ttnn.matmul(x_tt, Wq_tt)    # (128, 64)
K_tt = ttnn.matmul(x_tt, Wk_tt)    # (128, 64)
V_tt = ttnn.matmul(x_tt, Wv_tt)    # (128, 64)
ttnn.synchronize_device(device)
print(f"  Q shape: {Q_tt.shape}, K shape: {K_tt.shape}, V shape: {V_tt.shape}")

# K^T: transpose last two dims
K_T_tt = ttnn.transpose(K_tt, -2, -1)  # (64, 128)
ttnn.synchronize_device(device)
print(f"  K^T shape: {K_T_tt.shape}")

# scores = Q @ K^T
scores_tt = ttnn.matmul(Q_tt, K_T_tt)  # (128, 128)
ttnn.synchronize_device(device)
print(f"  scores shape: {scores_tt.shape}")

# Scale by 1/sqrt(d_k) using scalar multiply
scores_tt = ttnn.multiply(scores_tt, scale_val)
ttnn.synchronize_device(device)

# Softmax
weights_tt = ttnn.softmax(scores_tt, dim=-1)  # (128, 128)
ttnn.synchronize_device(device)

# Output = weights @ V
output_tt = ttnn.matmul(weights_tt, V_tt)  # (128, 64)
ttnn.synchronize_device(device)
print(f"  output shape: {output_tt.shape}")

# Compare
output_torch = ttnn.to_torch(output_tt).squeeze().float()[:SEQ_LEN, :D_K]
abs_err = (output_torch - output_ref).abs()
print(f"\n  Correctness vs PyTorch:")
print(f"    Max abs error:  {abs_err.max().item():.4f}")
print(f"    Mean abs error: {abs_err.mean().item():.4f}")

# Also check intermediate: attention weights should sum to ~1
weights_torch = ttnn.to_torch(weights_tt).squeeze().float()[:SEQ_LEN, :SEQ_LEN]
row_sums = weights_torch.sum(dim=-1)
print(f"    Attention weight row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f} (expect ~1.0)")

# Cleanup test 1 intermediates
for t in [x_tt, Wq_tt, Wk_tt, Wv_tt, Q_tt, K_tt, V_tt, K_T_tt,
          scores_tt, weights_tt, output_tt]:
    try:
        t.deallocate()
    except:
        pass


# ============================================================
# TEST 2: Multi-head attention
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Multi-head attention")
print("=" * 60)
print("  4 heads, d_model=256, d_k=64 per head")
print()

N_HEADS = 4

torch.manual_seed(42)
x_torch = torch.randn(SEQ_LEN, D_MODEL)

# Per-head weight matrices
Wq_heads = [torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5) for _ in range(N_HEADS)]
Wk_heads = [torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5) for _ in range(N_HEADS)]
Wv_heads = [torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5) for _ in range(N_HEADS)]

# Output projection: concat of heads (N_HEADS * D_K) -> D_MODEL
Wo_torch = torch.randn(N_HEADS * D_K, D_MODEL) * (1.0 / (N_HEADS * D_K) ** 0.5)

# --- PyTorch reference ---
head_outputs_ref = []
for h in range(N_HEADS):
    Q = x_torch @ Wq_heads[h]
    K = x_torch @ Wk_heads[h]
    V = x_torch @ Wv_heads[h]
    scores = Q @ K.T / math.sqrt(D_K)
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V
    head_outputs_ref.append(out)

concat_ref = torch.cat(head_outputs_ref, dim=-1)  # (128, 256)
mha_output_ref = concat_ref @ Wo_torch  # (128, 256)

print(f"  PyTorch ref: concat shape {concat_ref.shape}, output shape {mha_output_ref.shape}")

# --- TT-NN implementation ---
# Strategy: compute each head separately (avoid complex 4D reshaping),
# then concatenate using torch and re-upload for output projection.
print("  Running on Blackhole (per-head loop)...")

x_tt = to_ttnn(x_torch)

head_outputs_tt = []
for h in range(N_HEADS):
    Wq_tt = to_ttnn(Wq_heads[h])
    Wk_tt = to_ttnn(Wk_heads[h])
    Wv_tt = to_ttnn(Wv_heads[h])

    Q_tt = ttnn.matmul(x_tt, Wq_tt)
    K_tt = ttnn.matmul(x_tt, Wk_tt)
    V_tt = ttnn.matmul(x_tt, Wv_tt)

    K_T_tt = ttnn.transpose(K_tt, -2, -1)
    scores_tt = ttnn.matmul(Q_tt, K_T_tt)
    scores_tt = ttnn.multiply(scores_tt, scale_val)
    weights_tt = ttnn.softmax(scores_tt, dim=-1)
    out_tt = ttnn.matmul(weights_tt, V_tt)
    ttnn.synchronize_device(device)

    # Read back this head's output
    head_out_torch = ttnn.to_torch(out_tt).squeeze().float()[:SEQ_LEN, :D_K]
    head_outputs_tt.append(head_out_torch)

    # Cleanup head tensors
    for t in [Wq_tt, Wk_tt, Wv_tt, Q_tt, K_tt, V_tt, K_T_tt,
              scores_tt, weights_tt, out_tt]:
        try:
            t.deallocate()
        except:
            pass

# Concatenate heads and do output projection on device
concat_tt_torch = torch.cat(head_outputs_tt, dim=-1)  # (128, 256)
concat_tt = to_ttnn(concat_tt_torch)
Wo_tt = to_ttnn(Wo_torch)

mha_output_tt = ttnn.matmul(concat_tt, Wo_tt)
ttnn.synchronize_device(device)

mha_out_torch = ttnn.to_torch(mha_output_tt).squeeze().float()[:SEQ_LEN, :D_MODEL]

# Compare
abs_err_mha = (mha_out_torch - mha_output_ref).abs()
print(f"\n  Multi-head attention correctness:")
print(f"    Max abs error:  {abs_err_mha.max().item():.4f}")
print(f"    Mean abs error: {abs_err_mha.mean().item():.4f}")

# Per-head correctness
for h in range(N_HEADS):
    head_err = (head_outputs_tt[h] - head_outputs_ref[h]).abs()
    print(f"    Head {h}: max err = {head_err.max().item():.4f}")

# Cleanup
for t in [x_tt, concat_tt, Wo_tt, mha_output_tt]:
    try:
        t.deallocate()
    except:
        pass


# ============================================================
# TEST 3: Benchmark attention at different sequence lengths
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Attention latency vs sequence length")
print("=" * 60)
print(f"  d_model=256, d_k=64, single-head attention")
print()

SEQ_LENS = [64, 128, 256, 512, 1024]
REPS = 20

def attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val):
    """Single-head attention forward pass."""
    Q = ttnn.matmul(x_tt, Wq_tt)
    K = ttnn.matmul(x_tt, Wk_tt)
    V = ttnn.matmul(x_tt, Wv_tt)
    K_T = ttnn.transpose(K, -2, -1)
    scores = ttnn.matmul(Q, K_T)
    scores = ttnn.multiply(scores, scale_val)
    weights = ttnn.softmax(scores, dim=-1)
    output = ttnn.matmul(weights, V)
    return output

print(f"  {'Seq Len':<10} {'Latency (ms)':<15} {'TFLOPS':<10}")
print(f"  {'-'*40}")

torch.manual_seed(42)

for seq_len in SEQ_LENS:
    # Generate data
    x_data = torch.randn(seq_len, D_MODEL)
    Wq_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
    Wk_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
    Wv_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)

    x_tt = to_ttnn(x_data)
    Wq_tt = to_ttnn(Wq_data)
    Wk_tt = to_ttnn(Wk_data)
    Wv_tt = to_ttnn(Wv_data)

    # Warmup
    for _ in range(5):
        out = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
        ttnn.synchronize_device(device)
        out.deallocate()

    # Benchmark
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        out = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - start)
        out.deallocate()

    avg_ms = sum(times) / len(times) * 1000

    # FLOPS calculation for attention:
    # 3x projection matmuls: 3 * (2 * seq_len * d_model * d_k)
    # scores matmul: 2 * seq_len * d_k * seq_len
    # output matmul: 2 * seq_len * seq_len * d_k
    flops_proj = 3 * (2 * seq_len * D_MODEL * D_K)
    flops_scores = 2 * seq_len * D_K * seq_len
    flops_output = 2 * seq_len * seq_len * D_K
    total_flops = flops_proj + flops_scores + flops_output
    tflops = total_flops / (avg_ms / 1000) / 1e12

    print(f"  {seq_len:<10} {avg_ms:<15.3f} {tflops:<10.4f}")

    for t in [x_tt, Wq_tt, Wk_tt, Wv_tt]:
        try:
            t.deallocate()
        except:
            pass


# ============================================================
# TEST 4: Trace-captured attention vs eager
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Trace-captured attention vs eager")
print("=" * 60)
print(f"  seq_len=128, d_model=256, d_k=64")
print()

torch.manual_seed(42)
SEQ = 128

x_data = torch.randn(SEQ, D_MODEL)
Wq_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
Wk_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
Wv_data = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)

x_tt = to_ttnn(x_data)
Wq_tt = to_ttnn(Wq_data)
Wk_tt = to_ttnn(Wk_data)
Wv_tt = to_ttnn(Wv_data)

# --- Eager baseline ---
# Warmup
for _ in range(5):
    out = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
    ttnn.synchronize_device(device)
    out.deallocate()

REPS = 50
times_eager = []
for _ in range(REPS):
    start = time.perf_counter()
    out = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
    ttnn.synchronize_device(device)
    times_eager.append(time.perf_counter() - start)
    out.deallocate()

avg_eager = sum(times_eager) / len(times_eager)

# --- Trace capture ---
# Dry run to establish shapes
out_dry = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
ttnn.synchronize_device(device)
out_dry.deallocate()

# Capture trace
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
out_traced = attention_forward(x_tt, Wq_tt, Wk_tt, Wv_tt, scale_val)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Warmup trace
for _ in range(5):
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

times_trace = []
for _ in range(REPS):
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    times_trace.append(time.perf_counter() - start)

avg_trace = sum(times_trace) / len(times_trace)
speedup = avg_eager / avg_trace

print(f"  Eager:   {avg_eager*1000:.3f} ms")
print(f"  Traced:  {avg_trace*1000:.3f} ms")
print(f"  Speedup: {speedup:.2f}x")
print()

# Attention has 8 ops: 3 matmul + 1 transpose + 1 matmul + 1 multiply + 1 softmax + 1 matmul
# At ~21us dispatch per op, expect ~0.168ms dispatch overhead
dispatch_saved = (avg_eager - avg_trace) * 1000
print(f"  Dispatch overhead removed: {dispatch_saved:.3f} ms")
print(f"  Attention ops: ~8, estimated dispatch @ 21us/op: ~0.168 ms")

ttnn.release_trace(device, trace_id)

# Cleanup
for t in [x_tt, Wq_tt, Wk_tt, Wv_tt, out_traced]:
    try:
        t.deallocate()
    except:
        pass


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)
print("""
  Self-attention works on Blackhole using TT-NN primitives:
    - matmul for Q/K/V projections and score/output computation
    - transpose for K^T
    - multiply for scaling by 1/sqrt(d_k)
    - softmax for attention weights

  Multi-head attention works via per-head loop (computing each
  head independently), avoiding complex 4D reshape issues.

  Key ops in the attention pipeline:
    3x matmul (projections) + transpose + matmul (scores) +
    multiply (scale) + softmax + matmul (output) = 8 ops total
""")

ttnn.close_device(device)
print("Done!")
