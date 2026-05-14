#!/usr/bin/env python3
"""
ttnn.conv1d depthwise replacement for DeltaNet conv1d block (qb2).

Research finding: ttnn.conv1d supports `groups=in_channels` for depthwise
convolution. Used in Mamba 2.8B production (models/demos/wormhole/mamba).
Question: is it faster than our manual `reshape + concat + mul + sum + silu + slice`
sequence (B4, 1.03 ms/layer = 21.5% of DeltaNet)?

Current decomposition (from conv1d_fusion_probe.py):
  V1 full path: 1.030 ms
  - sum (reduce dim=-1): 0.674 ms (65.5%)
  - state mgmt (concat + slice): 0.218 ms (21.1%)
  - mul: 0.133 ms (13.0%)
  - silu: 0.005 ms (0.5%)

If ttnn.conv1d does the same depthwise math in one fused call, we'd expect
~10× speedup (similar to QK normalize's 88% reduction with rms_norm).

Caveats:
- ttnn.conv1d may have shape/layout constraints (TILE vs ROW_MAJOR, batch dim, etc.)
- Activations need to be in conv1d's expected layout (may need reshape ops)
- Need silu fused or as separate op
- State management still needs to happen externally (conv state buffer)

Test: build a 3-tap depthwise conv equivalent to our DeltaNet conv1d:
  in_channels = CONV_DIM = 8192
  out_channels = CONV_DIM (depthwise, groups=CONV_DIM)
  kernel_size = 3
  padding = 0 (we manage state externally to provide 2 prior tokens)
  Compare to current manual sequence at same input shape.

Run:
    ssh qb2 'cd ~/tt-xla && pkill -9 -f serve.server; .venv/bin/python experiments/utils/conv1d_depthwise_probe.py'
"""
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


CONV_DIM = 8192
KERNEL = 3
DTYPE = ttnn.bfloat16


def sync_time(device, fn, N=50, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / N


def main():
    print("=" * 78)
    print("ttnn.conv1d depthwise vs manual conv1d (qb2)")
    print("=" * 78)
    print(f"CONV_DIM={CONV_DIM}  KERNEL={KERNEL}  (DeltaNet depthwise 3-tap)")

    print("\n[1] Open device...")
    device = ttnn.open_device(device_id=0)
    print("  ✓ device open")

    try:
        print("\n[2] Build state...")
        rng = np.random.default_rng(42)
        # Manual path inputs (current shape)
        mixed_qkv = rng.standard_normal((1, CONV_DIM)).astype(np.float32) * 0.3
        conv_state = rng.standard_normal((CONV_DIM, KERNEL - 1)).astype(np.float32) * 0.5
        conv_weight = rng.standard_normal((CONV_DIM, KERNEL)).astype(np.float32) * 0.3

        # ttnn.conv1d expected layout: torch convention [N, C_in, L]
        # For our 1-token decode: N=1, C_in=CONV_DIM, L=KERNEL (provide 3 timesteps:
        # 2 from state + 1 current). conv1d returns [N, C_out, 1] = [1, CONV_DIM, 1].
        # Weight in conv layout: [C_out, C_in/groups, kernel_size] = [CONV_DIM, 1, KERNEL]
        # (depthwise: C_in/groups = 1 because groups = CONV_DIM = C_in).
        conv_weight_depthwise = conv_weight.reshape(CONV_DIM, 1, KERNEL)

        # Build [1, CONV_DIM, KERNEL] input for ttnn.conv1d:
        # concat(state, current_token) per channel.
        conv_input_full = np.concatenate(
            [conv_state, mixed_qkv.reshape(CONV_DIM, 1)], axis=-1
        )  # [CONV_DIM, KERNEL]
        # Reshape to [1, CONV_DIM, KERNEL]
        conv_input_torch = conv_input_full.reshape(1, CONV_DIM, KERNEL)

        def up(arr, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(torch.from_numpy(arr), dtype=DTYPE,
                                    device=device, layout=layout)

        mq = up(mixed_qkv)
        cs = up(conv_state)
        cw = up(conv_weight)
        cw_dw = up(conv_weight_depthwise, layout=ttnn.ROW_MAJOR_LAYOUT)
        ci_dw_torch = torch.from_numpy(conv_input_torch).to(torch.bfloat16)

        # === V1: current manual path ===
        def v1():
            mixed_col = ttnn.reshape(mq, [CONV_DIM, 1])
            conv_input = ttnn.concat([cs, mixed_col], dim=-1)
            conv_prod = ttnn.mul(conv_input, cw)
            conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
            return conv_out

        # === V2: ttnn.conv1d depthwise ===
        # NOTE: ttnn.conv1d may require various layout/sharding params
        # Will discover signature errors at runtime.
        def v2():
            ci_dw_tt = ttnn.from_torch(
                ci_dw_torch.unsqueeze(0) if ci_dw_torch.dim() == 3 else ci_dw_torch,
                dtype=DTYPE, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            # Try ttnn.conv1d directly. Will iterate on errors.
            try:
                out = ttnn.conv1d(
                    input_tensor=ci_dw_tt,
                    weight_tensor=cw_dw,
                    in_channels=CONV_DIM,
                    out_channels=CONV_DIM,
                    kernel_size=KERNEL,
                    stride=1,
                    padding=0,
                    groups=CONV_DIM,
                    batch_size=1,
                    device=device,
                )
            except Exception as e:
                raise RuntimeError(f"ttnn.conv1d failed: {type(e).__name__}: {e}") from e
            return ttnn.silu(out)

        print("\n[3] Math sanity for V1...")
        out_v1 = ttnn.to_torch(v1()).float().cpu().numpy().flatten()[:CONV_DIM]
        gold = (conv_input_full * conv_weight).sum(axis=-1)
        gold_silu = gold * (1.0 / (1.0 + np.exp(-gold)))
        cos_v1 = float(out_v1 @ gold_silu /
                       (np.linalg.norm(out_v1) * np.linalg.norm(gold_silu) + 1e-12))
        print(f"  cos(V1 ttnn, numpy gold) = {cos_v1:.6f}")

        # === Try V2 ===
        print("\n[4] Math sanity for V2 (ttnn.conv1d depthwise)...")
        try:
            out_v2_raw = v2()
            out_v2 = ttnn.to_torch(out_v2_raw).float().cpu().numpy().flatten()[:CONV_DIM]
            cos_v2 = float(out_v2 @ gold_silu /
                           (np.linalg.norm(out_v2) * np.linalg.norm(gold_silu) + 1e-12))
            print(f"  cos(V2 ttnn.conv1d, numpy gold) = {cos_v2:.6f}")
            v2_works = cos_v2 >= 0.99
        except Exception as e:
            print(f"  ✗ V2 ttnn.conv1d FAILED: {type(e).__name__}: {str(e)[:300]}")
            v2_works = False

        # === Latency ===
        print("\n[5] Latency benchmark (N=50, warmup=5)...")
        ms_v1 = sync_time(device, v1)
        print(f"  V1 manual (concat+mul+sum+silu):  {ms_v1:.4f} ms")
        if v2_works:
            try:
                ms_v2 = sync_time(device, v2)
                print(f"  V2 ttnn.conv1d depthwise+silu:    {ms_v2:.4f} ms  "
                      f"({(1 - ms_v2 / ms_v1) * 100:+.1f}% vs V1)")
            except Exception as e:
                print(f"  V2 latency benchmark failed: {type(e).__name__}: {str(e)[:160]}")
        else:
            print(f"  V2 skipped (math/setup failed)")

    finally:
        try:
            ttnn.close_device(device)
            print("\n  ✓ device closed")
        except Exception as e:
            print(f"\n  ✗ close error: {e}")


if __name__ == "__main__":
    main()
