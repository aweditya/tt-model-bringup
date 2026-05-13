#!/usr/bin/env python3
"""
Probe #3 (early): does paged SDPA decode actually break the MAX_POS=256
cliff? This is THE big question.

Sweep effective context = max_num_blocks * P, with P=64. Test:
  - 256 (matches non-paged ceiling)
  - 512 (where non-paged dies)
  - 4096, 16384, 32768 (long context for daily-driver)

If paged decode runs cleanly at 32k, C'0.5 has its kernel and the
daily-driver target is in sight. If it cliffs somewhere, we find out
the new ceiling.

Run on qb1 (qb2 busy):
    cd ~/tt-xla && .venv/bin/python experiments/utils/paged_sdpa_scaling_probe.py
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
P = 64        # block size — agent's sweet spot
HD = 256


def main():
    print("=" * 64)
    print(f"Probe: paged SDPA decode scaling to long context")
    print(f"  shapes: B={B}, N_Q={N_Q}, N_KV={N_KV}, P={P}, HD={HD}")
    print("=" * 64)

    # Sweep effective context = max_num_blocks * P
    test_contexts = [
        (4, 256),       # = non-paged ceiling
        (8, 512),       # = where non-paged dies
        (32, 2048),
        (64, 4096),
        (128, 8192),
        (256, 16384),
        (512, 32768),   # = daily-driver target
    ]

    device = ttnn.open_device(device_id=0)
    rng = np.random.default_rng(42)
    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False,
    )

    try:
        print(f"\n{'max_blocks':>12} {'max_pos':>9} {'cache_MB':>10} {'wall_ms':>10}  result")
        print("-" * 65)

        for max_blocks, max_pos in test_contexts:
            # Cache shape [max_blocks, N_KV, P, HD]
            cache_bytes = max_blocks * N_KV * P * HD * 2
            cache_mb = cache_bytes / (1024 ** 2)
            try:
                cache_np = rng.standard_normal((max_blocks, N_KV, P, HD)).astype(np.float32) * 0.05
                keys_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                           device=device, layout=ttnn.TILE_LAYOUT)
                vals_tt = ttnn.from_torch(torch.from_numpy(cache_np), dtype=ttnn.bfloat16,
                                           device=device, layout=ttnn.TILE_LAYOUT)

                q_np = rng.standard_normal((1, B, N_Q, HD)).astype(np.float32) * 0.1
                q_tt = ttnn.from_torch(torch.from_numpy(q_np), dtype=ttnn.bfloat16,
                                        device=device, layout=ttnn.TILE_LAYOUT)

                page_table_np = np.arange(max_blocks, dtype=np.int32).reshape(B, max_blocks)
                page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                                 dtype=ttnn.int32, device=device,
                                                 layout=ttnn.ROW_MAJOR_LAYOUT)

                cur_pos_np = np.array([max_pos - 1], dtype=np.int32)
                cur_pos_tt = ttnn.from_torch(torch.from_numpy(cur_pos_np),
                                              dtype=ttnn.int32, device=device,
                                              layout=ttnn.ROW_MAJOR_LAYOUT)

                ttnn.synchronize_device(device)
                t0 = time.time()
                _ = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_tt, keys_tt, vals_tt, page_table_tt,
                    cur_pos_tensor=cur_pos_tt,
                    compute_kernel_config=hifi4,
                )
                ttnn.synchronize_device(device)
                t1 = time.time()
                wall_ms = (t1 - t0) * 1000
                print(f"{max_blocks:>12} {max_pos:>9} {cache_mb:>10.1f} {wall_ms:>10.2f}  ✓ OK")

                del keys_tt, vals_tt, q_tt, page_table_tt, cur_pos_tt
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                # Pull out L1 numbers if present
                full = str(e)
                if "circular buffers" in full and "beyond max L1" in full:
                    import re
                    m = re.search(r'grow to (\d+) B.*max L1 size of (\d+) B', full)
                    if m:
                        wanted, limit = int(m.group(1)), int(m.group(2))
                        print(f"{max_blocks:>12} {max_pos:>9} {cache_mb:>10.1f}  ✗ L1 overflow: "
                              f"needs {wanted/1024:.1f} KB, limit {limit/1024:.1f} KB")
                        continue
                print(f"{max_blocks:>12} {max_pos:>9} {cache_mb:>10.1f}  ✗ {msg[:60]}")

        print()
        print("Interpretation:")
        print("  - If 32k passes: C'0.5 has its kernel. Paged SDPA decode breaks the cliff.")
        print("  - If only smaller sizes pass: there's a new ceiling — find it precisely.")
        print("  - If all sizes fail: paged also hits the cliff. Need program_config tuning.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
