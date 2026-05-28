#!/usr/bin/env python3
"""
Experiment 87 — Multi-chip primitives (Phase A7).

Validates ttnn 0.69's multi-chip API on qb1's 4-chip Blackhole quietbox.
We target 2 chips for Phase B (Qwen3.6-35B-A3B). Tests opened with
unknown-exact API — script tries multiple variants and reports which one
ttnn 0.69 accepts.

Tests in order:
  A7.1  Open 2-chip mesh, allocate test tensor
  A7.2  all_gather across 2 chips
  A7.3  all_reduce (sum) across 2 chips
  A7.4  reduce_scatter for TP-linear pattern
  A7.5  all_to_all_dispatch for MoE pattern

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/87_multichip_primitives.py
"""
import os, sys, time, traceback
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn


def try_open_2chip_mesh():
    """Probe several plausible mesh-open APIs. Return the working device + name."""
    attempts = [
        ("ttnn.open_mesh_device(mesh_shape=(2,1))",
         lambda: ttnn.open_mesh_device(mesh_shape=(2, 1))),
        ("ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2,1))",
         lambda: ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))),
        ("ttnn.distributed.open_mesh_device(mesh_shape=(2,1))",
         lambda: ttnn.distributed.open_mesh_device(mesh_shape=(2, 1))),
        ("ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(2,1))",
         lambda: ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))),
        ("ttnn.MeshDevice(mesh_shape=(2,1))",
         lambda: ttnn.MeshDevice(mesh_shape=(2, 1))),
    ]
    for label, fn in attempts:
        try:
            d = fn()
            print(f"  OK  {label}")
            return d, label
        except Exception as e:
            msg = str(e).split('\n')[0][:80]
            print(f"  FAIL  {label}: {msg}")
    return None, None


def discover_mesh_api():
    """List functions/classes in ttnn that look mesh-related."""
    print("\nDiscovered mesh-related symbols:")
    for name in dir(ttnn):
        if any(k in name.lower() for k in ['mesh', 'distrib', 'multi_device']):
            print(f"  ttnn.{name}")
    if hasattr(ttnn, 'distributed'):
        for name in dir(ttnn.distributed):
            if not name.startswith('_'):
                print(f"  ttnn.distributed.{name}")


def main():
    print("=" * 64)
    print("Phase A7 — Multi-chip primitives probe")
    print("=" * 64)

    print("\n[A7.0] Discovering mesh API...")
    discover_mesh_api()

    print("\n[A7.1] Opening 2-chip mesh...")
    mesh, api_name = try_open_2chip_mesh()
    if mesh is None:
        print("\nNo mesh API worked. Cannot proceed with A7.")
        sys.exit(1)
    print(f"\nUsing: {api_name}")
    print(f"Mesh: {mesh}")
    print(f"Num devices: {mesh.get_num_devices() if hasattr(mesh, 'get_num_devices') else 'unknown'}")

    # --- A7.2: all_gather smoke test ---
    print("\n[A7.2] all_gather across 2 chips")
    try:
        # Each chip's chunk: [1, 1024]; all_gather along dim 0 -> [2, 1024]
        chunk = np.random.randn(1, 1024).astype(np.float32) * 0.1
        t = torch.from_numpy(chunk)
        chip_tensor = ttnn.from_torch(t, dtype=ttnn.bfloat16, device=mesh,
                                       layout=ttnn.TILE_LAYOUT)
        gathered = ttnn.all_gather(chip_tensor, dim=0)
        out = ttnn.to_torch(gathered).float().numpy()
        print(f"  OK  output shape: {out.shape}")
    except Exception as e:
        print(f"  FAIL: {str(e)[:120]}")
        traceback.print_exc()

    # --- A7.3: all_reduce sum ---
    print("\n[A7.3] all_reduce sum across 2 chips")
    try:
        chunk = np.full((1, 1024), 0.5, dtype=np.float32)
        chip_tensor = ttnn.from_torch(torch.from_numpy(chunk), dtype=ttnn.bfloat16,
                                       device=mesh, layout=ttnn.TILE_LAYOUT)
        # If each chip has 0.5, sum should be 1.0 everywhere
        reduced = ttnn.all_reduce(chip_tensor)
        out = ttnn.to_torch(reduced).float().numpy()
        expected = 1.0
        actual = float(out.mean())
        passed = abs(actual - expected) < 0.05
        print(f"  expected ~{expected}, got {actual:.4f}  ->  {'OK' if passed else 'FAIL'}")
    except Exception as e:
        print(f"  FAIL: {str(e)[:120]}")
        traceback.print_exc()

    # --- A7.4: reduce_scatter ---
    print("\n[A7.4] reduce_scatter (TP-linear pattern)")
    try:
        chunk = np.full((2, 1024), 1.0, dtype=np.float32)
        chip_tensor = ttnn.from_torch(torch.from_numpy(chunk), dtype=ttnn.bfloat16,
                                       device=mesh, layout=ttnn.TILE_LAYOUT)
        scattered = ttnn.reduce_scatter(chip_tensor, dim=0)
        out = ttnn.to_torch(scattered).float().numpy()
        # After reduce-scatter sum across 2 chips, each chip gets half the sum
        # Each chip sees [1, 1024] of value 2.0 (sum across 2 chips)
        print(f"  shape: {out.shape}  mean: {float(out.mean()):.4f} (expected ~2.0)")
    except Exception as e:
        print(f"  FAIL: {str(e)[:120]}")

    # --- A7.5: all_to_all_dispatch for MoE (smoke) ---
    print("\n[A7.5] all_to_all_dispatch (MoE pattern, smoke)")
    if hasattr(ttnn, 'all_to_all_dispatch'):
        print("  ttnn.all_to_all_dispatch exists. Smoke test pending until "
              "we know the exact signature (it has metadata + token tensors).")
    else:
        print("  ttnn.all_to_all_dispatch NOT present in this build")

    print("\n=== A7 complete ===")
    if hasattr(ttnn, 'close_mesh_device'):
        ttnn.close_mesh_device(mesh)
    elif hasattr(ttnn, 'close_device'):
        ttnn.close_device(mesh)


if __name__ == "__main__":
    main()
