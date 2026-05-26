#!/usr/bin/env python3
"""
Find the actual MAX_POS ceiling for stock ttnn SDPA-decode on Blackhole.

The first SDPA probe (sdpa_long_pos_probe.py) found that MAX_POS=32768
overflows L1: "circular buffers grow to 1901120 B beyond max L1 size of
1572864 B". The op fails at ANY cur_pos because the failure is
allocation-time, not execution-time.

This probe sweeps MAX_POS to find the largest cache size that doesn't
overflow. cur_pos=128 throughout (small, doesn't matter — the failure
is in cache-shape-driven buffer planning).

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/utils/sdpa_max_pos_ceiling_probe.py
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)

N_KV = 4
N_Q = 32
HEAD_DIM = 256


def main():
    print("=" * 64)
    print("Probe: find max MAX_POS that fits SDPA-decode in L1")
    print("=" * 64)

    # Sweep: tile-aligned candidates from 128 (known-working) to 32768 (known-broken)
    candidates = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 10240, 12288,
                  14336, 16384, 18432, 20480, 24576, 32768]

    device = ttnn.open_device(device_id=0)
    rng = np.random.default_rng(42)
    cur_pos = 128

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)

    try:
        print(f"\n{'MAX_POS':>10} {'cache_MB':>10}  result")
        print("-" * 55)

        last_pass = None
        first_fail = None

        for max_pos in candidates:
            cache_bytes = 1 * N_KV * max_pos * HEAD_DIM * 2  # bf16, single side
            cache_mb = cache_bytes / 1024 / 1024
            try:
                # Small caches — ok to reupload each iter
                k_np = rng.standard_normal((1, N_KV, max_pos, HEAD_DIM)).astype(np.float32) * 0.1
                v_np = rng.standard_normal((1, N_KV, max_pos, HEAD_DIM)).astype(np.float32) * 0.1
                kv_k_tt = ttnn.from_torch(torch.from_numpy(k_np), dtype=ttnn.bfloat16,
                                          device=device, layout=ttnn.TILE_LAYOUT)
                kv_v_tt = ttnn.from_torch(torch.from_numpy(v_np), dtype=ttnn.bfloat16,
                                          device=device, layout=ttnn.TILE_LAYOUT)

                q_np = rng.standard_normal((N_Q, HEAD_DIM)).astype(np.float32) * 0.1
                q_tt = ttnn.from_torch(
                    torch.from_numpy(q_np.reshape(1, 1, N_Q, HEAD_DIM)),
                    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
                cur_pos_tt = ttnn.from_torch(
                    torch.tensor([cur_pos], dtype=torch.int32), device=device)

                _ = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_tt, kv_k_tt, kv_v_tt,
                    cur_pos_tensor=cur_pos_tt,
                    compute_kernel_config=hifi4)
                ttnn.synchronize_device(device)
                print(f"{max_pos:>10} {cache_mb:>10.1f}  ✓ OK")
                last_pass = max_pos
                # Cleanup (gc on next iter)
                del kv_k_tt, kv_v_tt, q_tt, cur_pos_tt
            except Exception as e:
                msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                # Extract the L1 size requested vs limit if it's that error
                full_msg = str(e)
                if "circular buffers" in full_msg and "beyond max L1" in full_msg:
                    # Try to pull the numbers
                    import re
                    m = re.search(r'grow to (\d+) B.*max L1 size of (\d+) B', full_msg)
                    if m:
                        wanted, limit = int(m.group(1)), int(m.group(2))
                        over = (wanted - limit) / 1024
                        print(f"{max_pos:>10} {cache_mb:>10.1f}  ✗ L1 overflow: needs "
                              f"{wanted/1024:.1f} KB, limit {limit/1024:.1f} KB "
                              f"(over by {over:.1f} KB)")
                    else:
                        print(f"{max_pos:>10} {cache_mb:>10.1f}  ✗ L1 overflow (unparseable)")
                else:
                    print(f"{max_pos:>10} {cache_mb:>10.1f}  ✗ {msg[:60]}")
                if first_fail is None:
                    first_fail = max_pos

        print()
        if last_pass is not None and first_fail is not None:
            print(f"Verdict: SDPA-decode handles MAX_POS up to {last_pass}.")
            print(f"  First failure at MAX_POS={first_fail}.")
            print(f"  Ceiling for daily-driver: {last_pass} tokens.")
        elif last_pass is not None:
            print(f"Verdict: SDPA-decode worked at all tested sizes up to {last_pass}.")
        else:
            print(f"Verdict: even MAX_POS={candidates[0]} fails — investigate.")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
