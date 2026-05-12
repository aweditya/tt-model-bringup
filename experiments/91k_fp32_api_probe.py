#!/usr/bin/env python3
"""
Experiment 91k — Phase B′9 prep: ttnn fp32 API surface probe.

Before implementing fp32 residual stream, we need to know which ttnn ops
accept fp32 inputs natively. Specifically:
  (1) fp32 tensor on device with TILE_LAYOUT
  (2) fp32 + fp32 → ?  (residual add)
  (3) rms_norm(fp32 input, bf16 weight) → ?  (does it consume fp32?)
  (4) rms_norm(fp32 input, fp32 weight) → ?
  (5) linear(fp32 activation, bf16 weight) → ?  (does engine demote silently?)
  (6) typecast bf16 → fp32 → bf16 (roundtrip cost?)

Each test prints success/failure and the output dtype. Run in <60s on qb2.
"""
import os, sys, time
import numpy as np
import torch
import ttnn

HIDDEN = 5120

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)


def case(label, fn):
    print(f"\n── {label}")
    try:
        result = fn()
        print(f"   PASS  {result}")
    except Exception as e:
        print(f"   FAIL  {type(e).__name__}: {e}")


def to_dev(arr, dtype, device, layout=ttnn.TILE_LAYOUT):
    t = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))
    while t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=layout)


def main():
    print("=" * 64)
    print("Phase B′9 prep — ttnn fp32 API surface probe")
    print("=" * 64)
    t0 = time.time()
    device = ttnn.open_device(device_id=0)

    # (1) fp32 tensor on device
    case("fp32 tensor creation [1, 5120] TILE_LAYOUT",
         lambda: f"dtype={to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device).dtype}")

    # (2) fp32 + fp32 add
    def test_add_fp32():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        y = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        z = ttnn.add(x, y)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.add(fp32, fp32)", test_add_fp32)

    # (2b) bf16 + fp32 (cross-dtype add)
    def test_add_mixed():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.bfloat16, device)
        y = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        z = ttnn.add(x, y)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.add(bf16, fp32)", test_add_mixed)

    # (3) rms_norm with fp32 input, bf16 weight
    def test_rms_fp32_in_bf16_w():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        w = to_dev(np.ones(HIDDEN), ttnn.bfloat16, device)
        z = ttnn.rms_norm(x, weight=w, epsilon=1e-6)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.rms_norm(fp32 input, bf16 weight)", test_rms_fp32_in_bf16_w)

    # (4) rms_norm with fp32 input, fp32 weight
    def test_rms_fp32_in_fp32_w():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        w = to_dev(np.ones(HIDDEN), ttnn.float32, device)
        z = ttnn.rms_norm(x, weight=w, epsilon=1e-6)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.rms_norm(fp32 input, fp32 weight)", test_rms_fp32_in_fp32_w)

    # (5) linear with fp32 activation, bf16 weight
    def test_linear_fp32_act_bf16_w():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        w = to_dev(np.random.randn(HIDDEN, HIDDEN) * 0.02, ttnn.bfloat16, device)
        z = ttnn.linear(x, w, compute_kernel_config=hifi4)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.linear(fp32 act, bf16 weight, hifi4)", test_linear_fp32_act_bf16_w)

    # (5b) linear with fp32 activation, bf8 weight
    def test_linear_fp32_act_bf8_w():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.float32, device)
        w = to_dev(np.random.randn(HIDDEN, HIDDEN) * 0.02, ttnn.bfloat8_b, device)
        z = ttnn.linear(x, w, compute_kernel_config=hifi4)
        return f"output dtype={z.dtype}  shape={tuple(z.shape)}"
    case("ttnn.linear(fp32 act, bf8 weight, hifi4)", test_linear_fp32_act_bf8_w)

    # (6) typecast roundtrip
    def test_typecast_roundtrip():
        x_bf16 = to_dev(np.random.randn(1, HIDDEN), ttnn.bfloat16, device)
        x_fp32 = ttnn.typecast(x_bf16, ttnn.float32)
        x_back = ttnn.typecast(x_fp32, ttnn.bfloat16)
        # Verify it survives
        a = ttnn.to_torch(x_bf16).float().numpy()
        b = ttnn.to_torch(x_back).float().numpy()
        max_diff = float(np.max(np.abs(a - b)))
        return f"bf16→fp32 dtype={x_fp32.dtype}  fp32→bf16 dtype={x_back.dtype}  max|Δ|={max_diff}"
    case("ttnn.typecast bf16↔fp32 roundtrip", test_typecast_roundtrip)

    # (7) output dtype override on add: can we force fp32 output?
    def test_add_with_output_dtype():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.bfloat16, device)
        y = to_dev(np.random.randn(1, HIDDEN), ttnn.bfloat16, device)
        z = ttnn.add(x, y, dtype=ttnn.float32)
        return f"output dtype={z.dtype}"
    case("ttnn.add(bf16, bf16, dtype=fp32)", test_add_with_output_dtype)

    # (8) linear with output_dtype override to fp32
    def test_linear_output_fp32():
        x = to_dev(np.random.randn(1, HIDDEN), ttnn.bfloat16, device)
        w = to_dev(np.random.randn(HIDDEN, HIDDEN) * 0.02, ttnn.bfloat8_b, device)
        z = ttnn.linear(x, w, compute_kernel_config=hifi4, dtype=ttnn.float32)
        return f"output dtype={z.dtype}"
    case("ttnn.linear(... dtype=fp32)", test_linear_output_fp32)

    ttnn.close_device(device)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
