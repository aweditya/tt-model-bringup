#!/usr/bin/env python3
"""
Probe: which ttnn op patterns trigger host-to-device writes inside a
begin_trace_capture / end_trace_capture window?

Background: our C'4 v4 trace capture fails with:
  TT_FATAL: Writes are not supported during trace capture. trace id: 0
  ... convert_python_tensor_to_tt_tensor -> to_device -> enqueue_write_tensor

Hypothesis: scalar broadcasts in ttnn.mul / ttnn.add / ttnn.sub etc. internally
wrap the scalar in a torch.tensor and ttnn.from_torch(device=...), which is a
host write. The trace capture rejects host writes mid-capture.

Tests (each independently captures + ends a trace; if a test passes, the op
pattern is trace-safe):
  1. ttnn.add(tensor, tensor)   — should work (pure device op)
  2. ttnn.add(tensor, 1.0)      — scalar broadcast as float
  3. ttnn.add(tensor, 1)        — scalar broadcast as int
  4. ttnn.mul(tensor, 1.0)      — scalar mul
  5. ttnn.add(tensor, scalar_tt) where scalar_tt is pre-alloc'd [1,1] — should work
  6. ttnn.full / ttnn.zeros inside trace
  7. ttnn.rms_norm with epsilon=1e-6 kwarg

Run on qb2 (device 1, qb2 server already on device 0):
    cd ~/tt-xla && TT_DEVICE_ID=1 .venv/bin/python experiments/utils/scalar_in_trace_probe.py
"""
import os, sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def alloc(shape, dtype=ttnn.bfloat16, device=None, value=0.1):
    arr = np.full(shape, value, dtype=np.float32)
    return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype, device=device,
                            layout=ttnn.TILE_LAYOUT)


def try_op(label, fn, device):
    print(f"  [{label}]", end=" ")
    try:
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _ = fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        ttnn.release_trace(device, tid)
        print("PASS")
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:120]
        print(f"FAIL: {msg}")
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass
        return False


def main():
    device_id = int(os.environ.get("TT_DEVICE_ID", "0"))
    print(f"Probe: scalar/op patterns in trace capture  (device_id={device_id})")
    print("=" * 72)
    device = ttnn.open_device(device_id=device_id)
    try:
        # Pre-allocate operands OUTSIDE the trace (no allocation during trace is allowed
        # — separate constraint from the "no host writes" we're studying).
        a = alloc([32, 64], device=device)
        b = alloc([32, 64], device=device, value=0.2)
        scalar_tt = alloc([1, 1], device=device, value=2.0)

        try_op("tensor+tensor",     lambda: ttnn.add(a, b), device)
        try_op("tensor+float",      lambda: ttnn.add(a, 1.0), device)
        try_op("tensor+int",        lambda: ttnn.add(a, 1), device)
        try_op("tensor*float",      lambda: ttnn.mul(a, 1.0), device)
        try_op("tensor+scalar_tt",  lambda: ttnn.add(a, scalar_tt), device)
        try_op("tensor*scalar_tt",  lambda: ttnn.mul(a, scalar_tt), device)
        try_op("rms_norm float eps",
               lambda: ttnn.rms_norm(a, weight=alloc([64], device=device, value=1.0), epsilon=1e-6),
               device)

        # Compound expression mimicking deltanet's RMSNorm-via-rsqrt pattern
        # (without using ttnn.rms_norm): q = q * rsqrt(sum(q*q) + EPS)
        def deltanet_norm():
            qq = ttnn.mul(a, a)
            s  = ttnn.sum(qq, dim=-1, keepdim=True)
            return ttnn.mul(a, ttnn.rsqrt(ttnn.add(s, 1e-6)))   # ← scalar
        try_op("deltanet-norm float eps", deltanet_norm, device)

        # Compound expression mimicking softplus(a+bias): log(exp(a+bias) + 1)
        def softplus_like():
            return ttnn.log(ttnn.add(ttnn.exp(a), 1.0))
        try_op("softplus 1.0 scalar", softplus_like, device)

        # Per-tensor scale: q = q * (1.0 / sqrt(K))
        try_op("tensor*literal_scale", lambda: ttnn.mul(a, 1.0 / (128.0 ** 0.5)), device)

        # ttnn.exp/silu/sigmoid — unary, no scalar
        try_op("ttnn.exp",     lambda: ttnn.exp(a), device)
        try_op("ttnn.silu",    lambda: ttnn.silu(a), device)
        try_op("ttnn.sigmoid", lambda: ttnn.sigmoid(a), device)

        # rsqrt with pre-allocated tensor (no scalar)
        try_op("rsqrt(t+scalar_tt)", lambda: ttnn.rsqrt(ttnn.add(a, scalar_tt)), device)

        print("=" * 72)
        print("VERDICT")
        print("=" * 72)
        print("Any FAIL row above is a pattern we must avoid inside the trace.")
        print("PASS rows are safe to use as-is.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
