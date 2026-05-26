#!/usr/bin/env python3
"""
Probe #2: paged_scaled_dot_product_attention_decode read smoke test.

Does the op even RUN at our shapes without crashing? Numerical correctness
is probe #3 (separate); this just checks it doesn't immediately blow up.

Shapes (Qwen3.6-27B target):
  - B=1, N_Q=32, N_KV=4 (GQA ratio 8), HD=256
  - block_size P=64
  - max_num_blocks=8 (= 512 max position)
  - cur_pos_tensor with one valid value

Run on qb1 (qb2 busy):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_sdpa_read_smoke_probe.py
"""
import sys
import time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

B = 1
N_Q = 32
N_KV = 4
P = 64
HD = 256
MAX_BLOCKS = 8


def main():
    print("=" * 64)
    print(f"Probe #2: paged SDPA decode read smoke")
    print(f"  shapes: B={B}, N_Q={N_Q}, N_KV={N_KV}, P={P}, HD={HD}, max_blocks={MAX_BLOCKS}")
    print(f"  GQA ratio: N_Q/N_KV = {N_Q // N_KV} (op shape-compatibility test)")
    print("=" * 64)

    device = ttnn.open_device(device_id=0)
    try:
        # Build random paged cache
        rng = np.random.default_rng(42)
        cache_shape = (MAX_BLOCKS, N_KV, P, HD)
        k_np = rng.standard_normal(cache_shape).astype(np.float32) * 0.1
        v_np = rng.standard_normal(cache_shape).astype(np.float32) * 0.1
        keys_tt = ttnn.from_torch(torch.from_numpy(k_np), dtype=ttnn.bfloat16,
                                   device=device, layout=ttnn.TILE_LAYOUT)
        values_tt = ttnn.from_torch(torch.from_numpy(v_np), dtype=ttnn.bfloat16,
                                     device=device, layout=ttnn.TILE_LAYOUT)

        # Q tensor: [1, B, N_Q, HD]
        q_np = rng.standard_normal((1, B, N_Q, HD)).astype(np.float32) * 0.1
        q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        print(f"  q shape: {tuple(q_tt.shape)}")
        print(f"  k/v shape: {tuple(keys_tt.shape)}")

        # Page table
        page_table_np = np.arange(MAX_BLOCKS, dtype=np.int32).reshape(B, MAX_BLOCKS)
        page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                         dtype=ttnn.int32, device=device,
                                         layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"  page_table shape: {tuple(page_table_tt.shape)}, values={page_table_np[0, :].tolist()}")

        # cur_pos: at position 100 (well inside block 1, plenty of valid prefix)
        cur_pos_np = np.array([100], dtype=np.uint32)
        cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos_np.astype(np.int32)),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)
        print(f"  cur_pos: {cur_pos_np.tolist()}")

        hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, math_approx_mode=False,
        )

        # Try the call
        print("\nCalling paged_scaled_dot_product_attention_decode...")
        try:
            ttnn.synchronize_device(device)
            t0 = time.time()
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_tt, keys_tt, values_tt, page_table_tt,
                cur_pos_tensor=cur_pos_tt,
                compute_kernel_config=hifi4,
            )
            ttnn.synchronize_device(device)
            t1 = time.time()
            print(f"  ✓ output shape: {tuple(attn.shape)}")
            print(f"  ✓ wall time: {(t1-t0)*1000:.2f} ms")
            attn_np = ttnn.to_torch(attn).float().cpu().numpy()
            print(f"  ✓ readback shape: {attn_np.shape}")
            print(f"  ✓ sample output[0, 0, 0, 0:5]: {attn_np.flatten()[:5]}")
            print(f"  ✓ output stats: mean={attn_np.mean():.4f}, std={attn_np.std():.4f}, max|·|={np.abs(attn_np).max():.4f}")
            print()
            print("  ✓ Op runs at our GQA shape (N_Q=32, N_KV=4). Proceeds to probe #3 (numerical correctness).")
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  ✗ FAILED: {type(e).__name__}: {msg[:160]}")
            print(f"     Full error: {str(e)[:500]}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
