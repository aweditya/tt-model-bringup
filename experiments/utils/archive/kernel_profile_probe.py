#!/usr/bin/env python3
"""
Kernel profile probe for Qwen3.6-27B decode kernels.

Runs each kernel category at production shapes for the 27B model. For each:
  - median latency (50 iter, warmup 5)
  - bytes moved (tensor sizes touched)
  - effective DRAM bandwidth = bytes / latency
  - efficiency vs theoretical 200 GB/s P150 ceiling
  - classification: bw-bound (efficiency > 60%) vs compute-bound (efficiency < 30%)

Dumps JSON to ~/tt-xla/.cache/kernel_profile_<host>_<timestamp>.json.

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/utils/kernel_profile_probe.py

No weights downloaded — config only.
"""
import os, sys, json, time, socket
from datetime import datetime
import numpy as np
import torch
import ttnn
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(line_buffering=True)

MODEL_ID = "Qwen/Qwen3.6-27B"
N_WARMUP = 5
N_MEASURE = 50
P150_DRAM_BW_GB = 200.0   # theoretical ceiling per chip
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def _bytes_for(dtype):
    return {
        ttnn.bfloat16: 2,
        ttnn.float32: 4,
        ttnn.bfloat8_b: 1,
        ttnn.int32: 4,
    }[dtype]


def _alloc(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, value=0.1):
    arr = np.full(shape, value, dtype=np.float32)
    return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype, device=device, layout=layout)


def _time(fn, device):
    # warmup
    for _ in range(N_WARMUP):
        out = fn()
        ttnn.synchronize_device(device)
    # measure
    times = []
    for _ in range(N_MEASURE):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def profile_linear(device, name, in_dim, out_dim, w_dtype=ttnn.bfloat8_b):
    """Single-row matmul: [1, in_dim] @ [in_dim, out_dim] -> [1, out_dim]"""
    x = _alloc([1, in_dim], device, dtype=ttnn.bfloat16)
    w = _alloc([in_dim, out_dim], device, dtype=w_dtype)
    median, p95 = _time(lambda: ttnn.linear(x, w, compute_kernel_config=hifi4), device)
    # bytes moved: weight (dominant) + activation in + activation out
    w_bytes = in_dim * out_dim * _bytes_for(w_dtype)
    act_in = in_dim * 2     # bf16
    act_out = out_dim * 2
    total_bytes = w_bytes + act_in + act_out
    bw_gb = total_bytes / 1e9 / (median / 1000.0)
    eff = bw_gb / P150_DRAM_BW_GB * 100
    # FLOP for matmul: 2 * in * out (one multiply + one add per output element)
    gflops = 2 * in_dim * out_dim / 1e9 / (median / 1000.0)
    return {
        'op': 'ttnn.linear',
        'name': name,
        'shape': f'[1,{in_dim}] @ [{in_dim},{out_dim}]',
        'w_dtype': str(w_dtype),
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': total_bytes / 1e6,
        'eff_bw_gb_s': bw_gb,
        'eff_pct_of_ceiling': eff,
        'gflops': gflops,
        'bw_bound': eff > 60,
    }


def profile_sdpa_decode(device, max_pos, n_q=32, n_kv=4, head_dim=256):
    q = _alloc([1, 1, n_q, head_dim], device, dtype=ttnn.bfloat16)
    k = _alloc([1, n_kv, max_pos, head_dim], device, dtype=ttnn.bfloat16)
    v = _alloc([1, n_kv, max_pos, head_dim], device, dtype=ttnn.bfloat16)
    cur_pos = ttnn.from_torch(torch.tensor([max_pos - 1], dtype=torch.int32), device=device)
    fn = lambda: ttnn.transformer.scaled_dot_product_attention_decode(
        q, k, v, cur_pos_tensor=cur_pos, compute_kernel_config=hifi4)
    median, p95 = _time(fn, device)
    kv_bytes = 2 * n_kv * max_pos * head_dim * 2  # K + V, bf16
    q_bytes = n_q * head_dim * 2
    total_bytes = kv_bytes + q_bytes
    bw_gb = total_bytes / 1e9 / (median / 1000.0)
    return {
        'op': 'ttnn.transformer.scaled_dot_product_attention_decode',
        'name': f'sdpa_decode_max_pos_{max_pos}',
        'shape': f'q=[1,1,{n_q},{head_dim}] k/v=[1,{n_kv},{max_pos},{head_dim}]',
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': total_bytes / 1e6,
        'eff_bw_gb_s': bw_gb,
        'eff_pct_of_ceiling': bw_gb / P150_DRAM_BW_GB * 100,
        'bw_bound': (bw_gb / P150_DRAM_BW_GB * 100) > 60,
    }


def profile_scatter_kv(device, max_pos=256, n_kv=4, head_dim=256):
    cache = _alloc([1, n_kv, max_pos, head_dim], device, dtype=ttnn.bfloat16)
    src = _alloc([1, n_kv, 1, head_dim], device, dtype=ttnn.bfloat16)
    idx = ttnn.from_torch(
        torch.from_numpy(np.zeros((1, n_kv, 1, head_dim), dtype=np.int32)),
        dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)
    fn = lambda: ttnn.scatter(cache, dim=2, index=idx, src=src)
    median, p95 = _time(fn, device)
    cache_bytes = n_kv * max_pos * head_dim * 2
    src_bytes = n_kv * head_dim * 2
    return {
        'op': 'ttnn.scatter',
        'name': f'scatter_kv_max_pos_{max_pos}',
        'shape': f'cache=[1,{n_kv},{max_pos},{head_dim}] src=[1,{n_kv},1,{head_dim}]',
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': (cache_bytes + src_bytes) / 1e6,
        'eff_bw_gb_s': (cache_bytes + src_bytes) / 1e9 / (median / 1000.0),
        'eff_pct_of_ceiling': (cache_bytes + src_bytes) / 1e9 / (median / 1000.0) / P150_DRAM_BW_GB * 100,
    }


def profile_copy(device, name, shape, dtype=ttnn.bfloat16):
    src = _alloc(shape, device, dtype=dtype)
    dst = _alloc(shape, device, dtype=dtype)
    fn = lambda: ttnn.copy(src, dst)
    median, p95 = _time(fn, device)
    total_bytes = int(np.prod(shape)) * _bytes_for(dtype) * 2  # read + write
    bw_gb = total_bytes / 1e9 / (median / 1000.0)
    return {
        'op': 'ttnn.copy',
        'name': name,
        'shape': str(shape),
        'dtype': str(dtype),
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': total_bytes / 1e6,
        'eff_bw_gb_s': bw_gb,
        'eff_pct_of_ceiling': bw_gb / P150_DRAM_BW_GB * 100,
    }


def profile_elementwise(device, op_name, op_fn, name, shape):
    a = _alloc(shape, device, dtype=ttnn.bfloat16)
    b = _alloc(shape, device, dtype=ttnn.bfloat16)
    if op_name == 'unary':
        fn = lambda: op_fn(a)
        n_reads = 1
    else:
        fn = lambda: op_fn(a, b)
        n_reads = 2
    median, p95 = _time(fn, device)
    total_bytes = int(np.prod(shape)) * 2 * (n_reads + 1)  # reads + 1 write
    bw_gb = total_bytes / 1e9 / (median / 1000.0)
    return {
        'op': f'ttnn.{name}',
        'name': name,
        'shape': str(shape),
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': total_bytes / 1e6,
        'eff_bw_gb_s': bw_gb,
        'eff_pct_of_ceiling': bw_gb / P150_DRAM_BW_GB * 100,
    }


def profile_rms_norm(device, shape):
    x = _alloc(shape, device, dtype=ttnn.bfloat16)
    w = _alloc([shape[-1]], device, dtype=ttnn.bfloat16)
    fn = lambda: ttnn.rms_norm(x, weight=w, epsilon=1e-6)
    median, p95 = _time(fn, device)
    total_bytes = (int(np.prod(shape)) + shape[-1]) * 2
    bw_gb = total_bytes / 1e9 / (median / 1000.0)
    return {
        'op': 'ttnn.rms_norm',
        'name': f'rms_norm_{shape}',
        'shape': str(shape),
        'median_ms': median,
        'p95_ms': p95,
        'bytes_moved_mb': total_bytes / 1e6,
        'eff_bw_gb_s': bw_gb,
        'eff_pct_of_ceiling': bw_gb / P150_DRAM_BW_GB * 100,
    }


def main():
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    HIDDEN = text_cfg['hidden_size']
    N_Q = text_cfg['num_attention_heads']
    N_KV = text_cfg['num_key_value_heads']
    HEAD_DIM = text_cfg['head_dim']
    N_K = text_cfg['linear_num_key_heads']
    N_V = text_cfg['linear_num_value_heads']
    K_DIM = text_cfg['linear_key_head_dim']
    V_DIM = text_cfg['linear_value_head_dim']
    INTERMEDIATE = text_cfg['intermediate_size']
    KEY_DIM = N_K * K_DIM
    VAL_DIM = N_V * V_DIM
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    CONV_KERNEL = text_cfg['linear_conv_kernel_dim']

    print(f"Model: {MODEL_ID}")
    print(f"  HIDDEN={HIDDEN}  N_Q={N_Q}  N_KV={N_KV}  HEAD_DIM={HEAD_DIM}")
    print(f"  N_K={N_K}  N_V={N_V}  K_DIM={K_DIM}  V_DIM={V_DIM}")
    print(f"  INTERMEDIATE={INTERMEDIATE}  KEY_DIM={KEY_DIM}  VAL_DIM={VAL_DIM}")
    print(f"  CONV_DIM={CONV_DIM}  CONV_KERNEL={CONV_KERNEL}")

    device_id = int(os.environ.get('TT_DEVICE_ID', '0'))
    print(f"  device_id = {device_id}")
    device = ttnn.open_device(device_id=device_id)
    try:
        results = []

        print("\n[1/7] ttnn.linear at production shapes (bf8 weights)...")
        results.append(profile_linear(device, 'attn_q_qk',     HIDDEN, 2 * N_Q * HEAD_DIM))
        results.append(profile_linear(device, 'attn_k',        HIDDEN, N_KV * HEAD_DIM))
        results.append(profile_linear(device, 'attn_v',        HIDDEN, N_KV * HEAD_DIM))
        results.append(profile_linear(device, 'attn_o',        N_Q * HEAD_DIM, HIDDEN))
        results.append(profile_linear(device, 'dn_in_proj_all', HIDDEN, CONV_DIM + VAL_DIM + 2 * N_V))
        results.append(profile_linear(device, 'dn_out_proj',   VAL_DIM, HIDDEN))
        results.append(profile_linear(device, 'mlp_gate_up',   HIDDEN, 2 * INTERMEDIATE))
        results.append(profile_linear(device, 'mlp_down',      INTERMEDIATE, HIDDEN))

        print("[2/7] SDPA decode at MAX_POS = 256, 1024, 8192...")
        for mp in [256, 1024, 8192]:
            try:
                results.append(profile_sdpa_decode(device, mp, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM))
            except Exception as e:
                results.append({'op': 'ttnn.transformer.scaled_dot_product_attention_decode',
                                'name': f'sdpa_decode_max_pos_{mp}', 'error': str(e)[:200]})

        print("[3/7] scatter on KV cache (MAX_POS=256)...")
        try:
            results.append(profile_scatter_kv(device, max_pos=256, n_kv=N_KV, head_dim=HEAD_DIM))
        except Exception as e:
            results.append({'op': 'ttnn.scatter', 'error': str(e)[:200]})

        print("[4/7] ttnn.copy at state-buffer shapes...")
        results.append(profile_copy(device, 'copy_kv',   [1, N_KV, 256, HEAD_DIM], dtype=ttnn.bfloat16))
        results.append(profile_copy(device, 'copy_ssm',  [N_V, K_DIM, V_DIM],      dtype=ttnn.float32))
        results.append(profile_copy(device, 'copy_conv', [CONV_DIM, CONV_KERNEL - 1], dtype=ttnn.float32))

        print("[5/7] rms_norm at residual / per-head shapes...")
        results.append(profile_rms_norm(device, [1, HIDDEN]))
        results.append(profile_rms_norm(device, [N_V, V_DIM]))

        print("[6/7] elementwise (mul, add, silu) at residual shape...")
        results.append(profile_elementwise(device, 'binary', ttnn.mul, 'mul', [1, HIDDEN]))
        results.append(profile_elementwise(device, 'binary', ttnn.add, 'add', [1, HIDDEN]))
        results.append(profile_elementwise(device, 'unary', ttnn.silu, 'silu', [1, INTERMEDIATE]))

        print("[7/7] experimental.rotary_embedding...")
        try:
            t = _alloc([1, 1, N_Q, HEAD_DIM], device, dtype=ttnn.bfloat16)
            cos = _alloc([1, 1, 1, HEAD_DIM], device, dtype=ttnn.bfloat16)
            sin = _alloc([1, 1, 1, HEAD_DIM], device, dtype=ttnn.bfloat16)
            fn = lambda: ttnn.experimental.rotary_embedding(t, cos, sin, token_idx=0)
            median, p95 = _time(fn, device)
            bytes_moved = (N_Q * HEAD_DIM * 2) * 2 + HEAD_DIM * 2 * 2
            bw = bytes_moved / 1e9 / (median / 1000.0)
            results.append({
                'op': 'ttnn.experimental.rotary_embedding',
                'name': 'rope_native',
                'shape': f'[1,1,{N_Q},{HEAD_DIM}]',
                'median_ms': median, 'p95_ms': p95,
                'bytes_moved_mb': bytes_moved / 1e6,
                'eff_bw_gb_s': bw,
                'eff_pct_of_ceiling': bw / P150_DRAM_BW_GB * 100,
            })
        except Exception as e:
            results.append({'op': 'ttnn.experimental.rotary_embedding', 'error': str(e)[:200]})

        host = socket.gethostname()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.expanduser(f"~/tt-xla/.cache/kernel_profile_{host}_{ts}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({
                'host': host,
                'timestamp': ts,
                'model_id': MODEL_ID,
                'config': {
                    'hidden': HIDDEN, 'n_q': N_Q, 'n_kv': N_KV, 'head_dim': HEAD_DIM,
                    'n_k': N_K, 'n_v': N_V, 'k_dim': K_DIM, 'v_dim': V_DIM,
                    'intermediate': INTERMEDIATE, 'conv_dim': CONV_DIM,
                    'conv_kernel': CONV_KERNEL,
                },
                'p150_dram_bw_gb_s_ceiling': P150_DRAM_BW_GB,
                'results': results,
            }, f, indent=2)
        print(f"\n  results -> {out_path}")

        print("\n" + "=" * 78)
        print(f"{'KERNEL':<40} {'SHAPE/SIZE':<30} {'ms':<8} {'GB/s':<8} {'%CEIL':<6}")
        print("=" * 78)
        for r in results:
            if 'error' in r:
                print(f"{r.get('op','?')[:40]:<40} ERROR: {r['error'][:60]}")
            else:
                eff = r.get('eff_pct_of_ceiling', 0)
                tag = ' bw' if eff > 60 else ('  compute' if eff < 30 else '   mid')
                shape = r.get('shape', r.get('name', ''))[:30]
                print(f"{r['op'][:40]:<40} {shape:<30} "
                      f"{r['median_ms']:6.2f}  {r.get('eff_bw_gb_s', 0):6.1f}  "
                      f"{eff:5.1f}{tag}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
