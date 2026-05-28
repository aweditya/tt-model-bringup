"""
Experiment 17: Flash Attention Investigation on Blackhole
=========================================================
Flash attention's key insight: instead of materializing the full NxN attention
matrix in memory (O(N^2) memory), compute attention in blocks/tiles, keeping
running statistics (online softmax) in fast local memory.

On GPUs this means SRAM; on Blackhole this means L1 SRAM (1.5 MB per core,
~165 MB total across 110 cores).

Tests:
  1. Standard attention memory usage at various sequence lengths
  2. Does TT-NN have built-in flash/fused attention ops?
  3. Manual tiled attention (flash attention concept)
  4. Memory comparison: standard vs flash attention
"""

import ttnn
import torch
import time
import math

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
N_CORES = grid.x * grid.y
print(f"Device: Blackhole p150a, {grid.x}x{grid.y} = {N_CORES} cores")

# Blackhole L1 specs
L1_PER_CORE_BYTES = 1_572_864  # 1.5 MB
L1_TOTAL_BYTES = L1_PER_CORE_BYTES * N_CORES
print(f"L1 per core: {L1_PER_CORE_BYTES / 1024:.0f} KB")
print(f"L1 total:    {L1_TOTAL_BYTES / (1024**2):.0f} MB")
print()


def to_ttnn(t, dev=device):
    """Convert a torch tensor to TT-NN on device, padding to tile alignment."""
    while t.dim() < 2:
        t = t.unsqueeze(0)
    h, w = t.shape[-2], t.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)


# ============================================================
# TEST 1: Standard attention memory usage
# ============================================================
print("=" * 60)
print("TEST 1: Standard attention memory usage vs sequence length")
print("=" * 60)
print(f"  d_k=64, single-head attention, bf16 (2 bytes per element)")
print()

D_K = 64
D_MODEL = 256
SEQ_LENS = [128, 256, 512, 1024, 2048]
BF16_BYTES = 2

print(f"  {'Seq Len':<10} {'Attn Matrix':<15} {'Fits L1/core?':<15} {'Fits L1 total?':<16} {'Time (ms)':<12}")
print(f"  {'-'*68}")

torch.manual_seed(42)

for seq_len in SEQ_LENS:
    attn_matrix_bytes = seq_len * seq_len * BF16_BYTES
    fits_per_core = "YES" if attn_matrix_bytes <= L1_PER_CORE_BYTES else "NO"
    fits_total = "YES" if attn_matrix_bytes <= L1_TOTAL_BYTES else "NO"

    # Try to actually run standard attention
    try:
        x_torch = torch.randn(seq_len, D_MODEL)
        Wq = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
        Wk = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)
        Wv = torch.randn(D_MODEL, D_K) * (1.0 / D_MODEL ** 0.5)

        x_tt = to_ttnn(x_torch)
        Wq_tt = to_ttnn(Wq)
        Wk_tt = to_ttnn(Wk)
        Wv_tt = to_ttnn(Wv)

        scale_val = 1.0 / math.sqrt(D_K)

        # Warmup
        Q = ttnn.matmul(x_tt, Wq_tt)
        K = ttnn.matmul(x_tt, Wk_tt)
        V = ttnn.matmul(x_tt, Wv_tt)
        K_T = ttnn.transpose(K, -2, -1)
        scores = ttnn.matmul(Q, K_T)
        scores = ttnn.multiply(scores, scale_val)
        weights = ttnn.softmax(scores, dim=-1)
        output = ttnn.matmul(weights, V)
        ttnn.synchronize_device(device)
        for t in [Q, K, V, K_T, scores, weights, output]:
            try:
                t.deallocate()
            except:
                pass

        # Timed run (average of 10)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            Q = ttnn.matmul(x_tt, Wq_tt)
            K = ttnn.matmul(x_tt, Wk_tt)
            V = ttnn.matmul(x_tt, Wv_tt)
            K_T = ttnn.transpose(K, -2, -1)
            scores = ttnn.matmul(Q, K_T)
            scores = ttnn.multiply(scores, scale_val)
            weights = ttnn.softmax(scores, dim=-1)
            output = ttnn.matmul(weights, V)
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - start)
            for t in [Q, K, V, K_T, scores, weights, output]:
                try:
                    t.deallocate()
                except:
                    pass

        avg_ms = sum(times) / len(times) * 1000
        time_str = f"{avg_ms:.3f}"

        for t in [x_tt, Wq_tt, Wk_tt, Wv_tt]:
            try:
                t.deallocate()
            except:
                pass

    except Exception as e:
        time_str = f"FAIL: {e}"

    attn_size_str = f"{attn_matrix_bytes / 1024:.0f} KB" if attn_matrix_bytes < 1024**2 else f"{attn_matrix_bytes / (1024**2):.1f} MB"
    print(f"  {seq_len:<10} {attn_size_str:<15} {fits_per_core:<15} {fits_total:<16} {time_str:<12}")


# ============================================================
# TEST 2: Does TT-NN have built-in flash/fused attention?
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 2: Built-in flash/fused attention ops in TT-NN?")
print("=" * 60)
print()

# Search for attention-related ops
attention_ops = [x for x in dir(ttnn) if 'attention' in x.lower() or 'sdpa' in x.lower() or 'flash' in x.lower()]
print(f"  Attention-related in ttnn: {attention_ops}")

# Check ttnn.transformer module
print()
if hasattr(ttnn, 'transformer'):
    transformer_ops = [x for x in dir(ttnn.transformer) if not x.startswith('_')]
    print(f"  ttnn.transformer module exists!")
    print(f"  Contents: {transformer_ops}")
    print()

    # Look for attention-specific ops
    attn_related = [x for x in transformer_ops if 'attention' in x.lower() or 'sdpa' in x.lower() or 'flash' in x.lower() or 'softmax' in x.lower()]
    print(f"  Attention-related ops: {attn_related}")
    print()

    # Try to use scaled_dot_product_attention if it exists
    if hasattr(ttnn.transformer, 'scaled_dot_product_attention'):
        print("  Found ttnn.transformer.scaled_dot_product_attention!")
        print(f"  Docstring: {ttnn.transformer.scaled_dot_product_attention.__doc__[:500] if ttnn.transformer.scaled_dot_product_attention.__doc__ else 'None'}")
        print()

        # Try calling it
        print("  Attempting to call scaled_dot_product_attention...")
        try:
            torch.manual_seed(42)
            seq_len = 128
            d_k = 64
            Q_torch = torch.randn(seq_len, d_k)
            K_torch = torch.randn(seq_len, d_k)
            V_torch = torch.randn(seq_len, d_k)

            Q_tt = to_ttnn(Q_torch)
            K_tt = to_ttnn(K_torch)
            V_tt = to_ttnn(V_torch)

            out = ttnn.transformer.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
            ttnn.synchronize_device(device)
            print(f"  SUCCESS! Output shape: {out.shape}")

            # Compare with manual
            scale = 1.0 / math.sqrt(d_k)
            scores_ref = Q_torch @ K_torch.T * scale
            weights_ref = torch.softmax(scores_ref, dim=-1)
            output_ref = weights_ref @ V_torch

            out_torch = ttnn.to_torch(out).squeeze().float()[:seq_len, :d_k]
            err = (out_torch - output_ref).abs()
            print(f"  Max error vs manual: {err.max().item():.4f}")
            print(f"  Mean error vs manual: {err.mean().item():.4f}")

            for t in [Q_tt, K_tt, V_tt, out]:
                try:
                    t.deallocate()
                except:
                    pass
        except Exception as e:
            print(f"  FAILED: {e}")
            print()

            # Try with 4D tensors (batch, heads, seq, d_k) as SDPA usually expects
            print("  Retrying with 4D tensors (1, 1, seq, d_k)...")
            try:
                Q_4d = torch.randn(1, 1, seq_len, d_k)
                K_4d = torch.randn(1, 1, seq_len, d_k)
                V_4d = torch.randn(1, 1, seq_len, d_k)

                Q_tt = to_ttnn(Q_4d)
                K_tt = to_ttnn(K_4d)
                V_tt = to_ttnn(V_4d)

                out = ttnn.transformer.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
                ttnn.synchronize_device(device)
                print(f"  SUCCESS with 4D! Output shape: {out.shape}")

                for t in [Q_tt, K_tt, V_tt, out]:
                    try:
                        t.deallocate()
                    except:
                        pass
            except Exception as e:
                print(f"  FAILED with 4D: {e}")

    # Try scaled_dot_product_attention_decode if it exists
    if hasattr(ttnn.transformer, 'scaled_dot_product_attention_decode'):
        print(f"\n  Also found: ttnn.transformer.scaled_dot_product_attention_decode")
        print(f"  Docstring: {ttnn.transformer.scaled_dot_product_attention_decode.__doc__[:500] if ttnn.transformer.scaled_dot_product_attention_decode.__doc__ else 'None'}")

    # Check for concatenate_heads, split_query_key_value_and_split_heads
    for op_name in ['concatenate_heads', 'split_query_key_value_and_split_heads', 'attention_softmax', 'attention_softmax_']:
        if hasattr(ttnn.transformer, op_name):
            print(f"\n  Found: ttnn.transformer.{op_name}")

    # Benchmark built-in SDPA vs manual if SDPA worked
    print()
    print("  --- Benchmarking built-in SDPA vs manual attention ---")
    try:
        torch.manual_seed(42)
        seq_len = 128
        d_k = 64

        # Try various tensor shapes for SDPA
        for shape_desc, q_shape, k_shape, v_shape in [
            ("2D (seq, d_k)", (seq_len, d_k), (seq_len, d_k), (seq_len, d_k)),
            ("4D (1,1,seq,d_k)", (1, 1, seq_len, d_k), (1, 1, seq_len, d_k), (1, 1, seq_len, d_k)),
            ("4D (1,8,seq,d_k)", (1, 8, seq_len, d_k), (1, 8, seq_len, d_k), (1, 8, seq_len, d_k)),
        ]:
            try:
                Q_t = torch.randn(*q_shape)
                K_t = torch.randn(*k_shape)
                V_t = torch.randn(*v_shape)
                Q_tt = to_ttnn(Q_t)
                K_tt = to_ttnn(K_t)
                V_tt = to_ttnn(V_t)

                out = ttnn.transformer.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
                ttnn.synchronize_device(device)
                print(f"  SDPA works with {shape_desc}: output shape {out.shape}")

                # Benchmark
                for t in [Q_tt, K_tt, V_tt, out]:
                    try:
                        t.deallocate()
                    except:
                        pass

                Q_tt = to_ttnn(Q_t)
                K_tt = to_ttnn(K_t)
                V_tt = to_ttnn(V_t)

                # Warmup
                for _ in range(5):
                    out = ttnn.transformer.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
                    ttnn.synchronize_device(device)
                    out.deallocate()

                times = []
                for _ in range(20):
                    start = time.perf_counter()
                    out = ttnn.transformer.scaled_dot_product_attention(Q_tt, K_tt, V_tt)
                    ttnn.synchronize_device(device)
                    times.append(time.perf_counter() - start)
                    out.deallocate()

                avg_ms = sum(times) / len(times) * 1000
                print(f"    SDPA latency: {avg_ms:.3f} ms")

                for t in [Q_tt, K_tt, V_tt]:
                    try:
                        t.deallocate()
                    except:
                        pass
                break  # If one shape works, we found it

            except Exception as e:
                print(f"  SDPA with {shape_desc}: FAILED - {str(e)[:120]}")
                for t_name in ['Q_tt', 'K_tt', 'V_tt']:
                    try:
                        eval(t_name).deallocate()
                    except:
                        pass

    except Exception as e:
        print(f"  Benchmark failed: {e}")

else:
    print("  ttnn.transformer module NOT found")

# Also check for any other relevant modules
for mod_name in ['experimental', 'operations']:
    if hasattr(ttnn, mod_name):
        mod = getattr(ttnn, mod_name)
        attn_ops = [x for x in dir(mod) if 'attention' in x.lower() or 'flash' in x.lower()]
        if attn_ops:
            print(f"\n  Attention-related in ttnn.{mod_name}: {attn_ops}")


# ============================================================
# TEST 3: Manual tiled attention (flash attention concept)
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 3: Manual tiled attention (flash attention concept)")
print("=" * 60)
print()

"""
Flash attention algorithm (simplified):
  For each block of K, V:
    1. Compute partial scores: S_block = Q @ K_block^T / sqrt(d_k)
    2. Find block max: m_block = max(S_block, dim=-1)
    3. Update running max: m_new = max(m_old, m_block)
    4. Compute correction factor for previous accumulator
    5. Compute exp(S_block - m_new) for current block
    6. Update running sum and output accumulator

This requires elementwise ops (exp, max, subtract) and careful bookkeeping.
Let's see how far we can get with TT-NN.
"""

SEQ_LEN = 256
D_K = 64
BLOCK_SIZE = 64  # Process K,V in blocks of 64 tokens
N_BLOCKS = SEQ_LEN // BLOCK_SIZE

torch.manual_seed(42)
Q_torch = torch.randn(SEQ_LEN, D_K)
K_torch = torch.randn(SEQ_LEN, D_K)
V_torch = torch.randn(SEQ_LEN, D_K)

scale = 1.0 / math.sqrt(D_K)

# --- PyTorch reference (standard attention) ---
scores_ref = Q_torch @ K_torch.T * scale
weights_ref = torch.softmax(scores_ref, dim=-1)
output_ref = weights_ref @ V_torch
print(f"  Reference output shape: {output_ref.shape}")
print(f"  Reference output range: [{output_ref.min():.4f}, {output_ref.max():.4f}]")
print()

# --- PyTorch tiled flash attention (to validate algorithm before TT-NN) ---
print("  Step 1: Validate tiled attention in PyTorch first...")
print()

# Online softmax tiled attention in pure PyTorch
O_acc = torch.zeros(SEQ_LEN, D_K)
l_acc = torch.zeros(SEQ_LEN, 1)  # running sum of exp
m_acc = torch.full((SEQ_LEN, 1), -1e9)  # running max

for b in range(N_BLOCKS):
    start_idx = b * BLOCK_SIZE
    end_idx = start_idx + BLOCK_SIZE

    K_block = K_torch[start_idx:end_idx, :]  # (BLOCK_SIZE, D_K)
    V_block = V_torch[start_idx:end_idx, :]  # (BLOCK_SIZE, D_K)

    # Partial scores: (SEQ_LEN, BLOCK_SIZE)
    S_block = Q_torch @ K_block.T * scale

    # Block-wise max: (SEQ_LEN, 1)
    m_block = S_block.max(dim=-1, keepdim=True).values

    # New running max
    m_new = torch.maximum(m_acc, m_block)

    # Correction for previous accumulator
    correction = torch.exp(m_acc - m_new)

    # Exp of current block scores
    P_block = torch.exp(S_block - m_new)

    # Update running sum
    l_new = correction * l_acc + P_block.sum(dim=-1, keepdim=True)

    # Update output accumulator
    O_acc = correction * O_acc + P_block @ V_block

    m_acc = m_new
    l_acc = l_new

# Final normalization
O_tiled = O_acc / l_acc

err_tiled_pytorch = (O_tiled - output_ref).abs()
print(f"  PyTorch tiled vs standard attention:")
print(f"    Max error:  {err_tiled_pytorch.max().item():.6f}")
print(f"    Mean error: {err_tiled_pytorch.mean().item():.6f}")
print(f"    Algorithm is {'CORRECT' if err_tiled_pytorch.max().item() < 1e-4 else 'INCORRECT'}!")
print()

# --- Now try the same in TT-NN ---
print("  Step 2: Attempt tiled attention in TT-NN...")
print()

try:
    Q_tt = to_ttnn(Q_torch)

    # We need: exp, max-reduce, subtraction, and careful accumulation
    # Check what ops are available
    has_exp = hasattr(ttnn, 'exp')
    has_max = hasattr(ttnn, 'max')
    has_sub = hasattr(ttnn, 'subtract') or hasattr(ttnn, 'sub')
    has_where = hasattr(ttnn, 'where')
    has_recip = hasattr(ttnn, 'reciprocal')

    print(f"    Available ops: exp={has_exp}, max={has_max}, subtract={has_sub}, where={has_where}, reciprocal={has_recip}")

    # The challenge: online softmax requires keeping running statistics
    # and updating the accumulator with correction factors.
    # This needs: exp, max (reduction), subtract, multiply, add -- all elementwise or reduction.

    # Attempt the tiled computation
    O_acc_tt = to_ttnn(torch.zeros(SEQ_LEN, D_K))
    l_acc_tt = to_ttnn(torch.zeros(SEQ_LEN, 32))  # padded to tile width
    m_acc_tt = to_ttnn(torch.full((SEQ_LEN, 32), -1e9))  # padded

    success = True
    block_times = []

    for b in range(N_BLOCKS):
        block_start = time.perf_counter()
        start_idx = b * BLOCK_SIZE
        end_idx = start_idx + BLOCK_SIZE

        K_block = K_torch[start_idx:end_idx, :]
        V_block = V_torch[start_idx:end_idx, :]
        K_block_tt = to_ttnn(K_block)
        V_block_tt = to_ttnn(V_block)

        # Partial scores: Q @ K_block^T -> (SEQ_LEN, BLOCK_SIZE)
        K_block_T_tt = ttnn.transpose(K_block_tt, -2, -1)
        S_block_tt = ttnn.matmul(Q_tt, K_block_T_tt)
        S_block_tt = ttnn.multiply(S_block_tt, scale)
        ttnn.synchronize_device(device)

        # Read back to CPU for the online softmax bookkeeping
        # (The tricky part: max reduction, exp, correction factor updates)
        S_block_cpu = ttnn.to_torch(S_block_tt).squeeze().float()[:SEQ_LEN, :BLOCK_SIZE]

        # Do the online softmax statistics on CPU
        m_acc_cpu = ttnn.to_torch(m_acc_tt).squeeze().float()[:SEQ_LEN, :1]
        l_acc_cpu = ttnn.to_torch(l_acc_tt).squeeze().float()[:SEQ_LEN, :1]
        O_acc_cpu = ttnn.to_torch(O_acc_tt).squeeze().float()[:SEQ_LEN, :D_K]

        m_block = S_block_cpu.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_acc_cpu, m_block)
        correction = torch.exp(m_acc_cpu - m_new)
        P_block = torch.exp(S_block_cpu - m_new)
        l_new = correction * l_acc_cpu + P_block.sum(dim=-1, keepdim=True)
        O_new = correction * O_acc_cpu + P_block @ V_block

        # Upload updated accumulators back
        for t in [O_acc_tt, l_acc_tt, m_acc_tt, S_block_tt, K_block_tt, V_block_tt, K_block_T_tt]:
            try:
                t.deallocate()
            except:
                pass

        O_acc_tt = to_ttnn(O_new)
        l_acc_tt = to_ttnn(torch.nn.functional.pad(l_new, (0, 31)))  # pad to 32 wide
        m_acc_tt = to_ttnn(torch.nn.functional.pad(m_new, (0, 31)))

        ttnn.synchronize_device(device)
        block_times.append(time.perf_counter() - block_start)

    # Final normalization on CPU
    O_final_cpu = ttnn.to_torch(O_acc_tt).squeeze().float()[:SEQ_LEN, :D_K]
    l_final_cpu = ttnn.to_torch(l_acc_tt).squeeze().float()[:SEQ_LEN, :1]
    O_result = O_final_cpu / l_final_cpu

    err_ttnn_tiled = (O_result - output_ref).abs()
    print(f"    TT-NN tiled attention (hybrid CPU/device):")
    print(f"      Max error vs standard:  {err_ttnn_tiled.max().item():.4f}")
    print(f"      Mean error vs standard: {err_ttnn_tiled.mean().item():.4f}")
    print(f"      Per-block avg time:     {sum(block_times)/len(block_times)*1000:.3f} ms")
    print(f"      Total time:             {sum(block_times)*1000:.3f} ms")
    print()
    print(f"    NOTE: This is a HYBRID implementation -- matmul on device,")
    print(f"    online softmax bookkeeping on CPU. A pure device implementation")
    print(f"    would need TT-NN reduce-max along rows and elementwise exp,")
    print(f"    which we attempt next.")

    for t in [Q_tt, O_acc_tt, l_acc_tt, m_acc_tt]:
        try:
            t.deallocate()
        except:
            pass

except Exception as e:
    print(f"    FAILED: {e}")
    import traceback
    traceback.print_exc()

# --- Attempt pure device tiled attention ---
print()
print("  Step 3: Attempt PURE DEVICE tiled attention (no CPU roundtrip)...")
print()

try:
    Q_tt = to_ttnn(Q_torch)

    # Initialize accumulators on device
    O_acc_tt = to_ttnn(torch.zeros(SEQ_LEN, D_K))
    # For l and m, we need (SEQ_LEN, 1) but tile-aligned means (SEQ_LEN, 32)
    l_acc_tt = to_ttnn(torch.ones(SEQ_LEN, 32) * 1e-10)  # small positive to avoid div by 0
    m_acc_tt = to_ttnn(torch.full((SEQ_LEN, 32), -1e9))

    pure_device_success = True
    pure_device_error = None

    for b in range(N_BLOCKS):
        start_idx = b * BLOCK_SIZE
        end_idx = start_idx + BLOCK_SIZE

        K_block_tt = to_ttnn(K_torch[start_idx:end_idx, :])
        V_block_tt = to_ttnn(V_torch[start_idx:end_idx, :])

        # S_block = Q @ K_block^T * scale
        K_block_T_tt = ttnn.transpose(K_block_tt, -2, -1)
        S_block_tt = ttnn.matmul(Q_tt, K_block_T_tt)
        S_block_tt = ttnn.multiply(S_block_tt, scale)

        # Now we need: max(S_block, dim=-1) -- this is the hard part
        try:
            m_block_tt = ttnn.max(S_block_tt, dim=-1)
            ttnn.synchronize_device(device)
            print(f"    Block {b}: ttnn.max(dim=-1) returned shape {m_block_tt.shape}")
        except Exception as e:
            print(f"    Block {b}: ttnn.max(dim=-1) FAILED: {str(e)[:100]}")
            pure_device_success = False
            pure_device_error = str(e)
            for t in [K_block_tt, V_block_tt, K_block_T_tt, S_block_tt]:
                try:
                    t.deallocate()
                except:
                    pass
            break

        # If max worked, try exp(S_block - m_block)
        try:
            # Subtract max for numerical stability
            S_shifted = ttnn.subtract(S_block_tt, m_block_tt)
            P_block_tt = ttnn.exp(S_shifted)
            ttnn.synchronize_device(device)
            print(f"    Block {b}: exp(S - max) worked, shape {P_block_tt.shape}")
        except Exception as e:
            print(f"    Block {b}: exp/subtract FAILED: {str(e)[:100]}")
            pure_device_success = False
            pure_device_error = str(e)
            break

        # Cleanup block tensors
        for t in [K_block_tt, V_block_tt, K_block_T_tt, S_block_tt]:
            try:
                t.deallocate()
            except:
                pass
        try:
            m_block_tt.deallocate()
            S_shifted.deallocate()
            P_block_tt.deallocate()
        except:
            pass

    if pure_device_success:
        print(f"\n    Pure device tiled attention: all blocks succeeded!")
    else:
        print(f"\n    Pure device tiled attention: BLOCKED by op failure")
        print(f"    Limiting factor: {pure_device_error}")

    for t in [Q_tt, O_acc_tt, l_acc_tt, m_acc_tt]:
        try:
            t.deallocate()
        except:
            pass

except Exception as e:
    print(f"    Pure device attempt FAILED: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# TEST 4: Memory comparison
# ============================================================
print(f"\n{'=' * 60}")
print("TEST 4: Memory comparison -- standard vs flash attention")
print("=" * 60)
print()

SEQ_LEN = 1024
D_K = 256
BLOCK_SIZE = 64

print(f"  Parameters: seq_len={SEQ_LEN}, d_model={D_K}")
print()

# Standard attention memory
attn_matrix_bytes = SEQ_LEN * SEQ_LEN * BF16_BYTES
q_bytes = SEQ_LEN * D_K * BF16_BYTES
k_bytes = SEQ_LEN * D_K * BF16_BYTES
v_bytes = SEQ_LEN * D_K * BF16_BYTES
output_bytes = SEQ_LEN * D_K * BF16_BYTES
total_standard = attn_matrix_bytes + q_bytes + k_bytes + v_bytes + output_bytes

# Flash attention memory (per block)
# At any point we hold: Q (full), one K block, one V block, partial scores, accumulators
q_flash = SEQ_LEN * D_K * BF16_BYTES
k_block_bytes = BLOCK_SIZE * D_K * BF16_BYTES
v_block_bytes = BLOCK_SIZE * D_K * BF16_BYTES
scores_block_bytes = SEQ_LEN * BLOCK_SIZE * BF16_BYTES
acc_output_bytes = SEQ_LEN * D_K * BF16_BYTES
acc_stats_bytes = SEQ_LEN * 2 * BF16_BYTES  # m and l vectors (ideally just seq_len wide)
total_flash = q_flash + k_block_bytes + v_block_bytes + scores_block_bytes + acc_output_bytes + acc_stats_bytes

def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    else:
        return f"{b/(1024**2):.2f} MB"

print(f"  Standard Attention:")
print(f"    Q:                {fmt_bytes(q_bytes)}")
print(f"    K:                {fmt_bytes(k_bytes)}")
print(f"    V:                {fmt_bytes(v_bytes)}")
print(f"    Attention matrix: {fmt_bytes(attn_matrix_bytes)}  <-- THIS IS THE O(N^2) COST")
print(f"    Output:           {fmt_bytes(output_bytes)}")
print(f"    TOTAL:            {fmt_bytes(total_standard)}")
print(f"    Fits in L1/core:  {'YES' if total_standard <= L1_PER_CORE_BYTES else 'NO'} ({fmt_bytes(L1_PER_CORE_BYTES)} per core)")
print()

print(f"  Flash Attention (block_size={BLOCK_SIZE}):")
print(f"    Q (full):         {fmt_bytes(q_flash)}")
print(f"    K block:          {fmt_bytes(k_block_bytes)}")
print(f"    V block:          {fmt_bytes(v_block_bytes)}")
print(f"    Score block:      {fmt_bytes(scores_block_bytes)}  <-- O(N*B) not O(N^2)")
print(f"    Output accum:     {fmt_bytes(acc_output_bytes)}")
print(f"    Running stats:    {fmt_bytes(acc_stats_bytes)}")
print(f"    TOTAL:            {fmt_bytes(total_flash)}")
print(f"    Fits in L1/core:  {'YES' if total_flash <= L1_PER_CORE_BYTES else 'NO'} ({fmt_bytes(L1_PER_CORE_BYTES)} per core)")
print()

savings = 1.0 - total_flash / total_standard
print(f"  Memory savings: {savings*100:.1f}%")
print(f"  Ratio: standard uses {total_standard / total_flash:.1f}x more memory")
print()

# At what sequence length does standard attention overflow L1 per core?
print(f"  At what seq_len does the attention matrix alone overflow L1/core?")
max_seq_l1 = int(math.sqrt(L1_PER_CORE_BYTES / BF16_BYTES))
print(f"    L1/core = {fmt_bytes(L1_PER_CORE_BYTES)}")
print(f"    Max seq_len for attn matrix in L1/core: {max_seq_l1}")
print(f"    (seq_len^2 * 2 bytes <= {L1_PER_CORE_BYTES} bytes)")
print()

# Total L1
max_seq_total = int(math.sqrt(L1_TOTAL_BYTES / BF16_BYTES))
print(f"    Max seq_len for attn matrix in total L1: {max_seq_total}")
print(f"    (That's {max_seq_total} tokens -- beyond most practical uses)")


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("Summary")
print("=" * 60)
print("""
  1. STANDARD ATTENTION MEMORY: The NxN attention matrix grows quadratically.
     At seq_len=1024 with d=256, the attention matrix alone is 2 MB.
     It fits in total L1 (165 MB) easily but exceeds per-core L1 (1.5 MB)
     once seq_len > ~886.

  2. BUILT-IN OPS: TT-NN's ttnn.transformer module contains attention-related
     ops. Whether scaled_dot_product_attention works determines if Tenstorrent
     already has an optimized (possibly flash-like) attention kernel.

  3. TILED ATTENTION: The flash attention algorithm can be implemented as a
     hybrid (matmul on device, softmax bookkeeping on CPU). Pure device
     implementation depends on ttnn.max (reduction) and ttnn.exp working
     correctly with the right tensor shapes.

  4. MEMORY SAVINGS: Flash attention reduces peak memory from O(N^2) to O(N*B)
     where B is the block size. For seq_len=1024, block_size=64:
     standard uses ~3x more memory than flash.
     The savings grow quadratically with sequence length.
""")

ttnn.close_device(device)
print("Done!")
