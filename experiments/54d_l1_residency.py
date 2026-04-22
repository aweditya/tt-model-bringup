#!/usr/bin/env python3
"""
Experiment 54d: L1 SRAM residency for decode intermediates on Blackhole P150.

Key question: Can we keep intermediate tensors in L1 between ops by specifying
memory_config on the output, eliminating DRAM round-trips?

Tests:
  A: matmul with output memory_config (L1 sharded vs L1 interleaved vs DRAM)
  B: Chain of ops (matmul -> silu -> matmul) with L1 intermediates vs DRAM
  C: rms_norm with HEIGHT_SHARDED input
  D: Simulated transformer layer op chain with L1 intermediates
  E: Latency comparison: DRAM vs L1 intermediates

Model dims: Qwen 0.5B decode (batch=1, seq_len=1 -> tile-padded to 32)
  hidden=896, head_dim=64, n_q_heads=14, intermediate=4864
"""

import sys, os, time
sys.path.insert(0, os.path.expanduser("~"))

import numpy as np
import torch
import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Device: Blackhole P150, {grid.x}x{grid.y} = {grid.x*grid.y} cores")

# Model dimensions
hidden = 896
n_q_heads = 14
n_kv_heads = 2
head_dim = 64
intermediate = 4864  # Qwen MLP intermediate size

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)

def to_dev(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    while t.dim() < 2: t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def to_dev_4d(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

def from_dev(tensor, shape):
    t = ttnn.to_torch(tensor).float()
    try: return t.reshape(shape).numpy()
    except RuntimeError: return t.squeeze().numpy().reshape(shape)

def make_height_shard_config(num_cores, shard_h, shard_w):
    """Create HEIGHT_SHARDED memory config with explicit ShardSpec."""
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

def mem_layout_str(tensor):
    """Get memory layout string from tensor."""
    try:
        mc = tensor.memory_config()
        return f"{mc.memory_layout} in {mc.buffer_type}"
    except:
        return "unknown"

# Pre-allocate weight tensors
print("\nAllocating weight tensors...")
w_qkv_np = np.random.randn(hidden, (n_q_heads + 2 * n_kv_heads) * head_dim).astype(np.float32)  # 896 -> 1152
w_o_np = np.random.randn(n_q_heads * head_dim, hidden).astype(np.float32)  # 896 -> 896
w_gate_np = np.random.randn(hidden, intermediate).astype(np.float32)  # 896 -> 4864
w_up_np = np.random.randn(hidden, intermediate).astype(np.float32)    # 896 -> 4864
w_down_np = np.random.randn(intermediate, hidden).astype(np.float32)  # 4864 -> 896

# ================================================================
# TEST A: matmul with output memory_config parameter
# ================================================================
print("\n" + "=" * 60)
print("TEST A: matmul with output memory_config")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
x_tt = to_dev_4d(x_np)
w_tt = to_dev(np.random.randn(hidden, hidden).astype(np.float32))

l1_shard_cfg = make_height_shard_config(1, 32, hidden)

# A1: matmul with L1 HEIGHT_SHARDED output
print("\n  A1: matmul(..., memory_config=L1_HEIGHT_SHARDED)")
try:
    r = ttnn.matmul(x_tt, w_tt, memory_config=l1_shard_cfg, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# A2: matmul with L1_MEMORY_CONFIG (interleaved L1)
print("\n  A2: matmul(..., memory_config=ttnn.L1_MEMORY_CONFIG)")
try:
    r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# A3: matmul with DRAM (baseline)
print("\n  A3: matmul(..., memory_config=ttnn.DRAM_MEMORY_CONFIG)")
try:
    r = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# A4: matmul with sharded INPUT and L1 sharded output
print("\n  A4: matmul(sharded_input, w, memory_config=L1_HEIGHT_SHARDED)")
try:
    x_sharded = ttnn.to_memory_config(x_tt, l1_shard_cfg)
    r = ttnn.matmul(x_sharded, w_tt, memory_config=l1_shard_cfg, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
    x_sharded.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# A5: matmul with sharded INPUT and L1 interleaved output
print("\n  A5: matmul(sharded_input, w, memory_config=L1_MEMORY_CONFIG)")
try:
    x_sharded = ttnn.to_memory_config(x_tt, l1_shard_cfg)
    r = ttnn.matmul(x_sharded, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
    x_sharded.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# A6: matmul with non-square weight (hidden -> intermediate for MLP)
print("\n  A6: matmul (896 -> 4864) with L1 output")
w_wide = to_dev(np.random.randn(hidden, intermediate).astype(np.float32))
l1_wide_cfg = make_height_shard_config(1, 32, intermediate)
try:
    r = ttnn.matmul(x_tt, w_wide, memory_config=l1_wide_cfg, compute_kernel_config=hifi4)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")
    # Fallback: try L1 interleaved
    try:
        r = ttnn.matmul(x_tt, w_wide, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
        print(f"    -> L1_INTERLEAVED fallback: {r.shape}, memory: {mem_layout_str(r)}")
        r.deallocate()
    except Exception as e2:
        print(f"    X L1_INTERLEAVED fallback: {str(e2).split(chr(10))[0][:120]}")
w_wide.deallocate()

w_tt.deallocate()
x_tt.deallocate()


# ================================================================
# TEST B: Op chain matmul -> silu -> matmul with L1 intermediates
# ================================================================
print("\n" + "=" * 60)
print("TEST B: Op chain (matmul -> silu -> matmul) L1 vs DRAM")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
# Use smaller intermediate for this test to fit in L1
small_inter = 896  # same as hidden for simplicity
w1_np = np.random.randn(hidden, small_inter).astype(np.float32)
w2_np = np.random.randn(small_inter, hidden).astype(np.float32)

x_tt = to_dev_4d(x_np)
w1_tt = to_dev(w1_np)
w2_tt = to_dev(w2_np)

l1_cfg = make_height_shard_config(1, 32, hidden)

# B1: All DRAM intermediates
print("\n  B1: All DRAM intermediates")
try:
    h1 = ttnn.matmul(x_tt, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    matmul1 -> {mem_layout_str(h1)}")
    h2 = ttnn.silu(h1)
    print(f"    silu    -> {mem_layout_str(h2)}")
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    matmul2 -> {mem_layout_str(h3)}")
    ref = from_dev(h3, (1, 1, 32, hidden))
    h1.deallocate(); h2.deallocate(); h3.deallocate()
    print(f"    OK, output norm: {np.linalg.norm(ref):.4f}")
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# B2: L1 interleaved intermediates
print("\n  B2: L1 interleaved intermediates")
try:
    h1 = ttnn.matmul(x_tt, w1_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    matmul1 -> {mem_layout_str(h1)}")
    h2 = ttnn.silu(h1)
    print(f"    silu    -> {mem_layout_str(h2)}")
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    print(f"    matmul2 -> {mem_layout_str(h3)}")
    out = from_dev(h3, (1, 1, 32, hidden))
    cos = np.dot(out.flatten(), ref.flatten()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-8)
    h1.deallocate(); h2.deallocate(); h3.deallocate()
    print(f"    OK, cosine vs DRAM: {cos:.6f}")
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# B3: L1 HEIGHT_SHARDED intermediates (need sharded input for sharded matmul output)
print("\n  B3: L1 HEIGHT_SHARDED intermediates (shard input first)")
try:
    x_sharded = ttnn.to_memory_config(x_tt, l1_cfg)
    h1 = ttnn.matmul(x_sharded, w1_tt, memory_config=l1_cfg, compute_kernel_config=hifi4)
    print(f"    matmul1 -> {mem_layout_str(h1)}")
    h2 = ttnn.silu(h1)
    print(f"    silu    -> {mem_layout_str(h2)}")
    # silu output preserves sharding; feed directly to matmul2
    h3 = ttnn.matmul(h2, w2_tt, memory_config=l1_cfg, compute_kernel_config=hifi4)
    print(f"    matmul2 -> {mem_layout_str(h3)}")
    out = from_dev(h3, (1, 1, 32, hidden))
    cos = np.dot(out.flatten(), ref.flatten()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-8)
    h1.deallocate(); h2.deallocate(); h3.deallocate(); x_sharded.deallocate()
    print(f"    OK, cosine vs DRAM: {cos:.6f}")
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

w1_tt.deallocate(); w2_tt.deallocate(); x_tt.deallocate()


# ================================================================
# TEST C: rms_norm with HEIGHT_SHARDED input
# ================================================================
print("\n" + "=" * 60)
print("TEST C: rms_norm with sharded input")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
gamma_np = np.ones((1, 1, 1, hidden), dtype=np.float32)
x_tt = to_dev_4d(x_np)
gamma_tt = to_dev_4d(gamma_np)
l1_cfg = make_height_shard_config(1, 32, hidden)

# C1: rms_norm on DRAM input
print("\n  C1: rms_norm(DRAM input)")
try:
    r = ttnn.rms_norm(x_tt, weight=gamma_tt)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    ref_rms = from_dev(r, (1, 1, 32, hidden))
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# C2: rms_norm on HEIGHT_SHARDED input
print("\n  C2: rms_norm(HEIGHT_SHARDED input)")
try:
    x_sharded = ttnn.to_memory_config(x_tt, l1_cfg)
    r = ttnn.rms_norm(x_sharded, weight=gamma_tt)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    out_rms = from_dev(r, (1, 1, 32, hidden))
    cos = np.dot(out_rms.flatten(), ref_rms.flatten()) / (np.linalg.norm(out_rms) * np.linalg.norm(ref_rms) + 1e-8)
    print(f"    cosine vs DRAM version: {cos:.6f}")
    r.deallocate(); x_sharded.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# C3: rms_norm with output memory_config
print("\n  C3: rms_norm(..., memory_config=L1_HEIGHT_SHARDED)")
try:
    r = ttnn.rms_norm(x_tt, weight=gamma_tt, memory_config=l1_cfg)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

# C4: rms_norm with L1 interleaved output
print("\n  C4: rms_norm(..., memory_config=L1_MEMORY_CONFIG)")
try:
    r = ttnn.rms_norm(x_tt, weight=gamma_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"    -> output: {r.shape}, memory: {mem_layout_str(r)}")
    r.deallocate()
except Exception as e:
    print(f"    X {str(e).split(chr(10))[0][:120]}")

gamma_tt.deallocate(); x_tt.deallocate()


# ================================================================
# TEST D: Simulated transformer layer with L1 intermediates
# ================================================================
print("\n" + "=" * 60)
print("TEST D: Simulated transformer layer op chain")
print("=" * 60)
print(f"  Shapes: hidden={hidden}, intermediate={intermediate}")
print(f"  heads: q={n_q_heads}, kv={n_kv_heads}, head_dim={head_dim}")

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
gamma_np = np.ones((1, 1, 1, hidden), dtype=np.float32)

# Weight tensors on device
w_qkv_tt = to_dev(w_qkv_np)
w_o_tt = to_dev(w_o_np)
w_gate_tt = to_dev(w_gate_np)
w_up_tt = to_dev(w_up_np)
w_down_tt = to_dev(w_down_np)
gamma_tt = to_dev_4d(gamma_np)

l1_hidden = make_height_shard_config(1, 32, hidden)

def run_transformer_layer(x_tt, mode, label):
    """Run one transformer layer, tracking memory configs.
    mode: 'dram', 'l1_interleaved', or 'l1_mixed' (sharded where possible)
    """
    print(f"\n  {label}:")
    ops_log = []

    if mode == 'dram':
        mem_out = ttnn.DRAM_MEMORY_CONFIG
    elif mode == 'l1_interleaved':
        mem_out = ttnn.L1_MEMORY_CONFIG
    else:
        # l1_mixed: use L1 interleaved everywhere (sharded output needs sharded input)
        mem_out = ttnn.L1_MEMORY_CONFIG

    residual = x_tt

    # 1. RMS Norm — supports memory_config output (even L1 sharded from DRAM input)
    try:
        if mode == 'l1_mixed':
            normed = ttnn.rms_norm(x_tt, weight=gamma_tt, memory_config=l1_hidden)
        else:
            normed = ttnn.rms_norm(x_tt, weight=gamma_tt, memory_config=mem_out)
        ops_log.append(f"    rms_norm -> {mem_layout_str(normed)}")
    except Exception as e:
        ops_log.append(f"    rms_norm X {str(e).split(chr(10))[0][:80]}")
        normed = ttnn.rms_norm(x_tt, weight=gamma_tt, memory_config=mem_out)
        ops_log.append(f"    rms_norm (fallback) -> {mem_layout_str(normed)}")

    # 2. QKV projection (896 -> 1152) — matmul with sharded input can output sharded
    qkv_dim = (n_q_heads + 2 * n_kv_heads) * head_dim  # 1152
    try:
        if mode == 'l1_mixed':
            l1_qkv = make_height_shard_config(1, 32, qkv_dim)
            qkv = ttnn.matmul(normed, w_qkv_tt, memory_config=l1_qkv, compute_kernel_config=hifi4)
        else:
            qkv = ttnn.matmul(normed, w_qkv_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    qkv_proj -> {mem_layout_str(qkv)}")
    except Exception as e:
        ops_log.append(f"    qkv_proj X {str(e).split(chr(10))[0][:80]}")
        qkv = ttnn.matmul(normed, w_qkv_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    qkv_proj (fallback) -> {mem_layout_str(qkv)}")
    normed.deallocate()

    # 3. Simulate attention output (identity matmul to get hidden-size tensor)
    # Need to unshard qkv first if it's sharded (matmul input needs matching)
    try:
        qkv_interl = ttnn.to_memory_config(qkv, mem_out)
        qkv.deallocate()
    except:
        qkv_interl = qkv

    # Simulate: create hidden-sized tensor for o_proj
    try:
        attn_out = ttnn.matmul(x_tt, w_eye_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    attn_sim -> {mem_layout_str(attn_out)}")
    except Exception as e:
        ops_log.append(f"    attn_sim X {str(e).split(chr(10))[0][:80]}")
        attn_out = x_tt
    qkv_interl.deallocate()

    # 4. O projection — if l1_mixed, shard input first then output sharded
    try:
        if mode == 'l1_mixed':
            attn_sharded = ttnn.to_memory_config(attn_out, l1_hidden)
            o_out = ttnn.matmul(attn_sharded, w_o_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
            attn_sharded.deallocate()
        else:
            o_out = ttnn.matmul(attn_out, w_o_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    o_proj   -> {mem_layout_str(o_out)}")
    except Exception as e:
        ops_log.append(f"    o_proj X {str(e).split(chr(10))[0][:80]}")
        o_out = ttnn.matmul(attn_out, w_o_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    o_proj (fallback) -> {mem_layout_str(o_out)}")
    attn_out.deallocate()

    # 5. Residual add — elementwise supports sharded output from interleaved inputs
    try:
        if mode == 'l1_mixed':
            # o_out is sharded, residual is interleaved — need both same layout
            o_interl = ttnn.to_memory_config(o_out, mem_out)
            h = ttnn.add(residual, o_interl, memory_config=l1_hidden)
            o_interl.deallocate()
        else:
            h = ttnn.add(residual, o_out, memory_config=mem_out)
        ops_log.append(f"    res_add  -> {mem_layout_str(h)}")
    except Exception as e:
        ops_log.append(f"    res_add X {str(e).split(chr(10))[0][:80]}")
        h = ttnn.add(residual, o_out)
        ops_log.append(f"    res_add (fallback) -> {mem_layout_str(h)}")
    o_out.deallocate()

    # 6. RMS Norm 2 — can output to sharded from interleaved input
    try:
        if mode == 'l1_mixed':
            # h might be sharded already; rms_norm needs interleaved input
            h_interl = ttnn.to_memory_config(h, mem_out)
            normed2 = ttnn.rms_norm(h_interl, weight=gamma_tt, memory_config=l1_hidden)
            h_interl.deallocate()
        else:
            normed2 = ttnn.rms_norm(h, weight=gamma_tt, memory_config=mem_out)
        ops_log.append(f"    rms_norm2 -> {mem_layout_str(normed2)}")
    except Exception as e:
        ops_log.append(f"    rms_norm2 X {str(e).split(chr(10))[0][:80]}")
        normed2 = ttnn.rms_norm(h, weight=gamma_tt, memory_config=mem_out)
        ops_log.append(f"    rms_norm2 (fallback) -> {mem_layout_str(normed2)}")

    # 7. MLP gate projection (896 -> 4864) — sharded input -> sharded output
    l1_inter = make_height_shard_config(1, 32, intermediate)
    try:
        if mode == 'l1_mixed':
            gate = ttnn.matmul(normed2, w_gate_tt, memory_config=l1_inter, compute_kernel_config=hifi4)
        else:
            gate = ttnn.matmul(normed2, w_gate_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    gate_proj -> {mem_layout_str(gate)}")
    except Exception as e:
        ops_log.append(f"    gate_proj X {str(e).split(chr(10))[0][:80]}")
        gate = ttnn.matmul(normed2, w_gate_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    gate_proj (fallback) -> {mem_layout_str(gate)}")

    # 8. SiLU on gate — preserves sharding
    try:
        gate_act = ttnn.silu(gate)
        ops_log.append(f"    silu     -> {mem_layout_str(gate_act)}")
    except Exception as e:
        ops_log.append(f"    silu X {str(e).split(chr(10))[0][:80]}")
        gate_act = gate
    gate.deallocate()

    # 9. Up projection (896 -> 4864)
    try:
        if mode == 'l1_mixed':
            up = ttnn.matmul(normed2, w_up_tt, memory_config=l1_inter, compute_kernel_config=hifi4)
        else:
            up = ttnn.matmul(normed2, w_up_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    up_proj  -> {mem_layout_str(up)}")
    except Exception as e:
        ops_log.append(f"    up_proj X {str(e).split(chr(10))[0][:80]}")
        up = ttnn.matmul(normed2, w_up_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    up_proj (fallback) -> {mem_layout_str(up)}")
    normed2.deallocate()

    # 10. gate * up (elementwise multiply) — should preserve sharding
    try:
        mlp_h = ttnn.mul(gate_act, up)
        ops_log.append(f"    gate*up  -> {mem_layout_str(mlp_h)}")
    except Exception as e:
        ops_log.append(f"    gate*up X {str(e).split(chr(10))[0][:80]}")
        # unshard both and multiply
        ga_i = ttnn.to_memory_config(gate_act, mem_out)
        up_i = ttnn.to_memory_config(up, mem_out)
        mlp_h = ttnn.mul(ga_i, up_i)
        ga_i.deallocate(); up_i.deallocate()
        ops_log.append(f"    gate*up (fallback) -> {mem_layout_str(mlp_h)}")
    gate_act.deallocate(); up.deallocate()

    # 11. Down projection (4864 -> 896) — sharded input -> sharded output
    try:
        if mode == 'l1_mixed':
            down = ttnn.matmul(mlp_h, w_down_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
        else:
            down = ttnn.matmul(mlp_h, w_down_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    down_proj -> {mem_layout_str(down)}")
    except Exception as e:
        ops_log.append(f"    down_proj X {str(e).split(chr(10))[0][:80]}")
        down = ttnn.matmul(mlp_h, w_down_tt, memory_config=mem_out, compute_kernel_config=hifi4)
        ops_log.append(f"    down_proj (fallback) -> {mem_layout_str(down)}")
    mlp_h.deallocate()

    # 12. Residual add
    try:
        if mode == 'l1_mixed':
            # h may be sharded, down may be sharded — need compatible layouts
            h_i = ttnn.to_memory_config(h, mem_out)
            d_i = ttnn.to_memory_config(down, mem_out)
            out = ttnn.add(h_i, d_i, memory_config=l1_hidden)
            h_i.deallocate(); d_i.deallocate()
        else:
            out = ttnn.add(h, down, memory_config=mem_out)
        ops_log.append(f"    res_add2 -> {mem_layout_str(out)}")
    except Exception as e:
        ops_log.append(f"    res_add2 X {str(e).split(chr(10))[0][:80]}")
        out = ttnn.add(h, down)
        ops_log.append(f"    res_add2 (fallback) -> {mem_layout_str(out)}")
    h.deallocate(); down.deallocate()

    for line in ops_log:
        print(line)

    return out


# Identity weight for attention simulation
w_eye_tt = to_dev(np.eye(hidden, dtype=np.float32))

# Run DRAM version
x_tt = to_dev_4d(x_np)
try:
    out_dram = run_transformer_layer(x_tt, mode='dram', label="DRAM intermediates")
    out_dram.deallocate()
except Exception as e:
    print(f"  DRAM path failed: {str(e).split(chr(10))[0][:120]}")

# Run L1 interleaved version
x_tt = to_dev_4d(x_np)
try:
    out_l1i = run_transformer_layer(x_tt, mode='l1_interleaved', label="L1 INTERLEAVED intermediates")
    out_l1i.deallocate()
except Exception as e:
    print(f"  L1 interleaved path failed: {str(e).split(chr(10))[0][:120]}")

# Run L1 mixed (sharded where possible) version
x_tt = to_dev_4d(x_np)
try:
    out_l1m = run_transformer_layer(x_tt, mode='l1_mixed', label="L1 MIXED (sharded where possible)")
    out_l1m.deallocate()
except Exception as e:
    print(f"  L1 mixed path failed: {str(e).split(chr(10))[0][:120]}")


# ================================================================
# TEST E: Latency comparison
# ================================================================
print("\n" + "=" * 60)
print("TEST E: Latency comparison — DRAM vs L1")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
w1_np = np.random.randn(hidden, hidden).astype(np.float32)
w2_np = np.random.randn(hidden, hidden).astype(np.float32)

x_tt = to_dev_4d(x_np)
w1_tt = to_dev(w1_np)
w2_tt = to_dev(w2_np)
gamma_tt2 = to_dev_4d(np.ones((1, 1, 1, hidden), dtype=np.float32))

N_WARMUP = 10
N_ITERS = 100

configs = {
    "DRAM": ttnn.DRAM_MEMORY_CONFIG,
    "L1_INTERLEAVED": ttnn.L1_MEMORY_CONFIG,
}

# Check if HEIGHT_SHARDED works for the full chain (need sharded input)
try:
    _xs = ttnn.to_memory_config(x_tt, l1_hidden)
    _test = ttnn.matmul(_xs, w1_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    _test2 = ttnn.silu(_test)
    _test3 = ttnn.matmul(_test2, w2_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    _xs.deallocate(); _test.deallocate(); _test2.deallocate(); _test3.deallocate()
    has_sharded = True
    print("  HEIGHT_SHARDED chain works, including in benchmark")
except Exception as e:
    has_sharded = False
    print(f"  HEIGHT_SHARDED chain failed: {str(e).split(chr(10))[0][:100]}")

# Benchmark helper
def bench_chain(name, run_fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        run_fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run_fn()
    ttnn.synchronize_device(device)
    elapsed = time.perf_counter() - t0
    us_per = elapsed / n_iters * 1e6
    print(f"  {name:25s}: {us_per:8.1f} us/iter  ({n_iters/elapsed:.0f} iter/s)")

def run_dram():
    h1 = ttnn.matmul(x_tt, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h1.deallocate(); h2.deallocate(); h3.deallocate()

def run_l1_interleaved():
    h1 = ttnn.matmul(x_tt, w1_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h1.deallocate(); h2.deallocate(); h3.deallocate()

def run_l1_sharded():
    xs = ttnn.to_memory_config(x_tt, l1_hidden)
    h1 = ttnn.matmul(xs, w1_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    xs.deallocate(); h1.deallocate(); h2.deallocate(); h3.deallocate()

print("\n  Chain: matmul -> silu -> matmul")
bench_chain("DRAM", run_dram, N_WARMUP, N_ITERS)
bench_chain("L1_INTERLEAVED", run_l1_interleaved, N_WARMUP, N_ITERS)
if has_sharded:
    bench_chain("L1_HEIGHT_SHARDED", run_l1_sharded, N_WARMUP, N_ITERS)

# Full chain: rms_norm -> matmul -> silu -> matmul
print("\n  Full chain: rms_norm -> matmul -> silu -> matmul")

def run_full_dram():
    n = ttnn.rms_norm(x_tt, weight=gamma_tt2)
    h1 = ttnn.matmul(n, w1_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=hifi4)
    n.deallocate(); h1.deallocate(); h2.deallocate(); h3.deallocate()

def run_full_l1i():
    n = ttnn.rms_norm(x_tt, weight=gamma_tt2, memory_config=ttnn.L1_MEMORY_CONFIG)
    h1 = ttnn.matmul(n, w1_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
    n.deallocate(); h1.deallocate(); h2.deallocate(); h3.deallocate()

def run_full_l1_sharded():
    # rms_norm: DRAM input -> HEIGHT_SHARDED output (supported!)
    n = ttnn.rms_norm(x_tt, weight=gamma_tt2, memory_config=l1_hidden)
    # matmul: sharded input -> sharded output
    h1 = ttnn.matmul(n, w1_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    h2 = ttnn.silu(h1)
    h3 = ttnn.matmul(h2, w2_tt, memory_config=l1_hidden, compute_kernel_config=hifi4)
    n.deallocate(); h1.deallocate(); h2.deallocate(); h3.deallocate()

bench_chain("DRAM", run_full_dram, N_WARMUP, N_ITERS)
bench_chain("L1_INTERLEAVED", run_full_l1i, N_WARMUP, N_ITERS)

# Check if full sharded chain works
try:
    run_full_l1_sharded()
    has_full_sharded = True
    print("  Full sharded chain verified, benchmarking...")
except Exception as e:
    has_full_sharded = False
    print(f"  Full sharded chain failed: {str(e).split(chr(10))[0][:100]}")

if has_full_sharded:
    bench_chain("L1_HEIGHT_SHARDED", run_full_l1_sharded, N_WARMUP, N_ITERS)

x_tt.deallocate(); w1_tt.deallocate(); w2_tt.deallocate(); gamma_tt2.deallocate()


# ================================================================
# TEST F: Summary of op memory_config support
# ================================================================
print("\n" + "=" * 60)
print("TEST F: Op-by-op memory_config support probe")
print("=" * 60)

x_np = np.random.randn(1, 1, 32, hidden).astype(np.float32)
x_tt = to_dev_4d(x_np)
w_tt = to_dev(np.random.randn(hidden, hidden).astype(np.float32))
gamma_tt3 = to_dev_4d(np.ones((1, 1, 1, hidden), dtype=np.float32))

l1_cfg = make_height_shard_config(1, 32, hidden)

ops_to_test = [
    ("matmul",   lambda: ttnn.matmul(x_tt, w_tt, memory_config=l1_cfg, compute_kernel_config=hifi4)),
    ("add",      lambda: ttnn.add(x_tt, x_tt, memory_config=l1_cfg)),
    ("mul",      lambda: ttnn.mul(x_tt, x_tt, memory_config=l1_cfg)),
    ("silu",     lambda: ttnn.silu(x_tt, memory_config=l1_cfg)),
    ("relu",     lambda: ttnn.relu(x_tt, memory_config=l1_cfg)),
    ("neg",      lambda: ttnn.neg(x_tt, memory_config=l1_cfg)),
    ("rms_norm", lambda: ttnn.rms_norm(x_tt, weight=gamma_tt3, memory_config=l1_cfg)),
]

for op_name, op_fn in ops_to_test:
    try:
        r = op_fn()
        print(f"  {op_name:12s} memory_config=L1_SHARDED: OK -> {mem_layout_str(r)}")
        r.deallocate()
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  {op_name:12s} memory_config=L1_SHARDED: FAIL ({err})")
        # Try L1 interleaved
        try:
            # Re-run with L1 interleaved
            r2 = None
            if op_name == "matmul":
                r2 = ttnn.matmul(x_tt, w_tt, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=hifi4)
            elif op_name == "add":
                r2 = ttnn.add(x_tt, x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif op_name == "mul":
                r2 = ttnn.mul(x_tt, x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif op_name == "silu":
                r2 = ttnn.silu(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif op_name == "relu":
                r2 = ttnn.relu(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif op_name == "neg":
                r2 = ttnn.neg(x_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif op_name == "rms_norm":
                r2 = ttnn.rms_norm(x_tt, weight=gamma_tt3, memory_config=ttnn.L1_MEMORY_CONFIG)
            if r2 is not None:
                print(f"  {op_name:12s} memory_config=L1_INTERL:  OK -> {mem_layout_str(r2)}")
                r2.deallocate()
        except Exception as e2:
            print(f"  {op_name:12s} memory_config=L1_INTERL:  FAIL ({str(e2).split(chr(10))[0][:80]})")

x_tt.deallocate(); w_tt.deallocate(); gamma_tt3.deallocate()


# ================================================================
# CLEANUP
# ================================================================
w_qkv_tt.deallocate(); w_o_tt.deallocate()
w_gate_tt.deallocate(); w_up_tt.deallocate(); w_down_tt.deallocate()
gamma_tt.deallocate()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

ttnn.close_device(device)
