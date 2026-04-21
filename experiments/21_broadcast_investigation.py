"""
Experiment 21: On-device broadcast investigation.

Our #1 performance bottleneck and trace-capture blocker is that TT-NN
TILE_LAYOUT can't implicitly broadcast mismatched shapes, forcing CPU
round-trips via broadcast_to_match().

This experiment systematically tests every TT-NN broadcast mechanism
to find on-device alternatives:
  1. ttnn.repeat / ttnn.repeat_interleave
  2. ttnn.concat with repeated tensors
  3. Direct ttnn binary ops with mismatched shapes (does it ever work?)
  4. ROW_MAJOR layout broadcast (maybe it works there?)
  5. ttnn.expand (if it exists)

Run: python3 21_broadcast_investigation.py
"""

import numpy as np
import torch
import ttnn
import time


def test_broadcast(name, fn, device):
    """Test a broadcast approach and report success/failure."""
    try:
        result = fn(device)
        print(f"  [OK]    {name}")
        return True, result
    except Exception as e:
        err = str(e)[:120]
        print(f"  [FAIL]  {name}: {err}")
        return False, None


def main():
    device = ttnn.open_device(device_id=0)
    print("=" * 70)
    print("Experiment 21: On-device broadcast investigation")
    print("=" * 70)

    # Reference data
    a_np = np.random.randn(32, 64).astype(np.float32)
    b_row = np.random.randn(1, 64).astype(np.float32)   # broadcast over rows
    b_col = np.random.randn(32, 1).astype(np.float32)   # broadcast over cols
    b_scalar = np.array([[3.14]], dtype=np.float32)       # scalar broadcast

    # Create device tensors in TILE_LAYOUT
    a_tt = ttnn.from_torch(torch.from_numpy(a_np).float().unsqueeze(0).unsqueeze(0),
                           dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # ============================================================
    # Test 1: Can ttnn binary ops handle mismatched shapes at all?
    # ============================================================
    print("\n--- Test 1: Direct binary ops with shape mismatch (TILE_LAYOUT) ---")

    def test_add_row_broadcast(dev):
        b = ttnn.from_torch(
            torch.from_numpy(np.broadcast_to(b_row, (32, 64)).copy()).float().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(a_tt, b)

    def test_add_col_broadcast(dev):
        # (32, 64) + (32, 1) — pad (32,1) to (32,32)
        b = ttnn.from_torch(
            torch.nn.functional.pad(torch.from_numpy(b_col).float(), (0, 31)).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(a_tt, b)

    def test_add_pre_expanded(dev):
        # Pre-expand b_row to (32, 64) before sending to device
        expanded = np.broadcast_to(b_row, (32, 64)).copy()
        b = ttnn.from_torch(
            torch.from_numpy(expanded).float().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(a_tt, b)

    test_broadcast("add pre-expanded (32,64)+(32,64)", test_add_pre_expanded, device)
    test_broadcast("add row broadcast (32,64)+(32,64 from 1,64)", test_add_row_broadcast, device)

    # ============================================================
    # Test 2: ttnn.repeat
    # ============================================================
    print("\n--- Test 2: ttnn.repeat ---")

    def test_repeat(dev):
        b = ttnn.from_torch(
            torch.from_numpy(b_row).float().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        # Try to repeat along dim 2 (rows) to get (1, 1, 32, 64)
        return ttnn.repeat(b, ttnn.Shape([1, 1, 32, 1]))

    test_broadcast("ttnn.repeat (1,1,1,64) -> (1,1,32,64)", test_repeat, device)

    # ============================================================
    # Test 3: ttnn.repeat_interleave
    # ============================================================
    print("\n--- Test 3: ttnn.repeat_interleave ---")

    def test_repeat_interleave(dev):
        b = ttnn.from_torch(
            torch.nn.functional.pad(torch.from_numpy(b_col).float(), (0, 31)).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        return ttnn.repeat_interleave(b, 64, dim=3)

    test_broadcast("ttnn.repeat_interleave col->matrix", test_repeat_interleave, device)

    # ============================================================
    # Test 4: ROW_MAJOR layout (maybe broadcast works there?)
    # ============================================================
    print("\n--- Test 4: ROW_MAJOR layout binary ops ---")

    def test_row_major_add(dev):
        a_rm = ttnn.from_torch(
            torch.from_numpy(a_np).float(),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_rm = ttnn.from_torch(
            torch.from_numpy(b_row).float(),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.add(a_rm, b_rm)

    def test_row_major_mul(dev):
        a_rm = ttnn.from_torch(
            torch.from_numpy(a_np).float(),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
        b_rm = ttnn.from_torch(
            torch.from_numpy(b_row).float(),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.mul(a_rm, b_rm)

    test_broadcast("ROW_MAJOR add (32,64)+(1,64)", test_row_major_add, device)
    test_broadcast("ROW_MAJOR mul (32,64)*(1,64)", test_row_major_mul, device)

    # ============================================================
    # Test 5: ttnn.expand / ttnn.broadcast_to (if they exist)
    # ============================================================
    print("\n--- Test 5: ttnn.expand / ttnn.broadcast_to ---")

    for name in ['expand', 'broadcast_to', 'tile', 'broadcast']:
        fn = getattr(ttnn, name, None)
        if fn is not None:
            print(f"  [EXISTS] ttnn.{name}")
        else:
            print(f"  [NONE]   ttnn.{name}")

    # ============================================================
    # Test 6: Scalar operations (ttnn.add with float)
    # ============================================================
    print("\n--- Test 6: Scalar broadcast (tensor + float) ---")

    def test_scalar_add(dev):
        return ttnn.add(a_tt, 3.14)

    def test_scalar_mul(dev):
        return ttnn.multiply(a_tt, 2.5)

    test_broadcast("ttnn.add(tensor, 3.14)", test_scalar_add, device)
    test_broadcast("ttnn.multiply(tensor, 2.5)", test_scalar_mul, device)

    # ============================================================
    # Test 7: ttnn.concat to simulate broadcast
    # ============================================================
    print("\n--- Test 7: ttnn.concat for row broadcast ---")

    def test_concat_broadcast(dev):
        # Create 32 copies of b_row via concat
        b = ttnn.from_torch(
            torch.nn.functional.pad(torch.from_numpy(b_row).float(), (0, 0, 0, 31)).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        # This should give us (1, 1, 32, 64)
        return b

    test_broadcast("pre-pad row to (32,64)", test_concat_broadcast, device)

    # ============================================================
    # Test 8: Check what TT-NN ops are in the transformer namespace
    # ============================================================
    print("\n--- Test 8: Available ttnn operations for broadcast ---")
    broadcast_related = []
    for name in sorted(dir(ttnn)):
        if any(kw in name.lower() for kw in ['broad', 'repeat', 'expand', 'tile', 'bcast']):
            broadcast_related.append(name)
    print(f"  Broadcast-related ops: {broadcast_related}")

    # ============================================================
    # Test 9: Benchmark pre-expansion vs CPU round-trip
    # ============================================================
    print("\n--- Test 9: Performance comparison ---")

    # Method A: CPU round-trip (current approach)
    def cpu_roundtrip():
        t = ttnn.to_torch(a_tt).float()
        b_expanded = np.broadcast_to(b_row, (32, 64)).copy()
        b_dev = ttnn.from_torch(
            torch.from_numpy(b_expanded).float().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(a_tt, b_dev)

    # Method B: Pre-expand on host (no device read needed)
    def host_preexpand():
        b_expanded = np.broadcast_to(b_row, (32, 64)).copy()
        b_dev = ttnn.from_torch(
            torch.from_numpy(b_expanded).float().unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        return ttnn.add(a_tt, b_dev)

    # Warmup
    for _ in range(3):
        cpu_roundtrip()
        host_preexpand()

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        cpu_roundtrip()
    t_cpu = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        host_preexpand()
    t_host = (time.perf_counter() - t0) / N

    print(f"  CPU round-trip: {t_cpu*1000:.3f} ms")
    print(f"  Host pre-expand: {t_host*1000:.3f} ms")
    print(f"  Speedup: {t_cpu/t_host:.2f}x")

    ttnn.close_device(device)
    print("\n" + "=" * 70)
    print("Experiment 21 complete!")


if __name__ == '__main__':
    main()
