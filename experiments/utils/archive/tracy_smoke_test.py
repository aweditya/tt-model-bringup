#!/usr/bin/env python3
"""
Smoke test for ttnn Tracy + device profiler APIs.

Answers:
  1. Do tracy zones work without any env var setup?
  2. Does get_latest_programs_perf_data() return anything by default?
  3. What does ReadDeviceProfiler(device) return, and in what format?
  4. Is TT_METAL_DEVICE_PROFILER=1 required to populate per-op data?

Run on qb2:
    cd ~/tt-xla && .venv/bin/python experiments/utils/tracy_smoke_test.py
    # Then try with profiler env var:
    TT_METAL_DEVICE_PROFILER=1 .venv/bin/python experiments/utils/tracy_smoke_test.py
"""
import os, sys, time
import numpy as np
import torch
import ttnn


def main():
    print("=" * 64)
    print("Tracy / device profiler smoke test")
    print("=" * 64)
    print(f"TT_METAL_DEVICE_PROFILER = {os.environ.get('TT_METAL_DEVICE_PROFILER', '<unset>')}")
    print(f"TT_METAL_ENABLE_TRACY    = {os.environ.get('TT_METAL_ENABLE_TRACY', '<unset>')}")
    print(f"TRACY_ENABLE             = {os.environ.get('TRACY_ENABLE', '<unset>')}")

    # 1. Try a tracy zone WITHOUT opening device — just the Python API
    print("\n[1] Calling ttnn.start_tracy_zone / stop_tracy_zone (Python-only)…")
    try:
        ttnn.start_tracy_zone("smoke_test.py", "main", 30, 0xff0000)
        time.sleep(0.001)
        ttnn.stop_tracy_zone("python_only_zone", 0xff0000)
        print("  OK — tracy zone calls did not error")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # 2. Open a device and run a tiny op
    print("\n[2] Opening device + running a tiny op (1x32x32 matmul)…")
    try:
        device = ttnn.open_device(device_id=0)
    except Exception as e:
        print(f"  device open failed: {e}")
        sys.exit(1)

    try:
        a = ttnn.from_torch(torch.randn(1, 32, 32).float(),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(torch.randn(1, 32, 32).float(),
                            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ttnn.start_tracy_zone("smoke_test.py", "matmul_region", 50)
        c = ttnn.matmul(a, b)
        ttnn.synchronize_device(device)
        ttnn.stop_tracy_zone("matmul_region", 0)
        print(f"  OK — matmul completed; output shape: {tuple(c.shape)}")
    except Exception as e:
        print(f"  matmul failed: {type(e).__name__}: {e}")

    # 3. Try ReadDeviceProfiler
    print("\n[3] Calling ttnn.ReadDeviceProfiler(device)…")
    try:
        result = ttnn.ReadDeviceProfiler(device)
        print(f"  OK — return type: {type(result).__name__}")
        if result is not None:
            print(f"  value preview: {repr(result)[:200]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # 4. Try get_latest_programs_perf_data
    print("\n[4] Calling ttnn.get_latest_programs_perf_data()…")
    try:
        data = ttnn.get_latest_programs_perf_data()
        print(f"  return type: {type(data).__name__}")
        if data is None:
            print("  data is None — likely TT_METAL_DEVICE_PROFILER not enabled")
        elif hasattr(data, '__len__'):
            print(f"  data length: {len(data)}")
            if len(data) > 0:
                print(f"  first entry type: {type(data[0]).__name__}")
                print(f"  first entry preview: {repr(data[0])[:400]}")
        else:
            print(f"  value preview: {repr(data)[:400]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # 5. Try get_all_programs_perf_data
    print("\n[5] Calling ttnn.get_all_programs_perf_data()…")
    try:
        data = ttnn.get_all_programs_perf_data()
        print(f"  return type: {type(data).__name__}")
        if hasattr(data, '__len__'):
            print(f"  data length: {len(data)}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
