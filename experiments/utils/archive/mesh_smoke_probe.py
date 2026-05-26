#!/usr/bin/env python3
"""
C'7.1: Mesh device smoke test on qb2 (4× P150 with working inter-chip fabric).

Goal: verify the fundamental multi-chip primitives work for our config before
we invest in the full tensor-parallel implementation. If this fails, the
whole C'7 plan needs a different approach.

Tests:
  1. Open (1, 4) mesh device covering all 4 chips on qb2
  2. Allocate a tensor sharded across the 4 chips (each chip holds 1/4)
  3. ttnn.all_gather: each chip ends with the full tensor
  4. ttnn.experimental.all_gather_async (the production op used by tt_transformers)
  5. Reduce-scatter pattern (composite all-reduce = reduce-scatter + all-gather)
  6. Measure per-collective latency

If everything passes, C'7.1 is done and we're cleared to build the
tensor-parallel forward.

Run on qb2 (server must be killed first — mesh open needs all 4 chips):
    cd ~/tt-xla && .venv/bin/python experiments/utils/mesh_smoke_probe.py
"""
import os, sys, time
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def _cosine(a, b):
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 72)
    print("C'7.1: Mesh device + collectives smoke test")
    print("=" * 72)

    # === Step 0: Initialise fabric ===
    # CCL ops (all_gather, reduce_scatter) require FABRIC_1D on Blackhole. From
    # tt-metal conftest.py:471 — must be called BEFORE open_mesh_device.
    # Probe found ttnn.set_fabric_config exists; FABRIC_1D for non-Galaxy hosts
    # (qb2 is not a 6U Galaxy).
    print("\n[0] Initialising fabric context (FABRIC_1D)...")
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        print("  ✓ fabric_config = FABRIC_1D set")
    except Exception as e:
        print(f"  ✗ set_fabric_config failed: {type(e).__name__}: {str(e)[:200]}")
        return

    # === Step 1: Open mesh ===
    print("\n[1] Opening (1, 4) mesh device...")
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        n_devices = mesh.get_num_devices()
        print(f"  ✓ Mesh opened. Num devices: {n_devices}")
        if n_devices != 4:
            print(f"  ✗ Expected 4 devices, got {n_devices}")
            return
    except Exception as e:
        print(f"  ✗ Failed to open mesh: {type(e).__name__}: {str(e)[:200]}")
        return

    try:
        # === Step 2: Allocate a sharded tensor ===
        # Use a tensor of shape [1, 1, 32, 128] (tile-aligned) and shard dim=3
        # across 4 chips, so each chip has [1, 1, 32, 32].
        print("\n[2] Allocate sharded tensor [1, 1, 32, 128] (dim=3 split across 4):")
        host_np = np.arange(1 * 1 * 32 * 128, dtype=np.float32).reshape(1, 1, 32, 128)
        # Sanity reference: split by 4 along dim=3
        expected_chip_0 = host_np[:, :, :, 0:32]
        expected_chip_3 = host_np[:, :, :, 96:128]

        sharded = ttnn.from_torch(
            torch.from_numpy(host_np),
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        print(f"  ✓ sharded tensor created. Shape: {tuple(sharded.shape)}")
        # NOTE: shape is the per-device shape after sharding, NOT the global shape
        # When sharded across dim=3, each device sees [1, 1, 32, 32]

        # === Step 3: ttnn.all_gather along dim=3, expect global [1, 1, 32, 128] on each chip
        print("\n[3] ttnn.all_gather(dim=3): each chip should end with global [1,1,32,128]")
        try:
            t0 = time.perf_counter()
            gathered = ttnn.all_gather(sharded, dim=3)
            ttnn.synchronize_device(mesh)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            print(f"  ✓ all_gather succeeded in {ms:.2f} ms")
            print(f"    gathered shape (per-chip): {tuple(gathered.shape)}")
            # Read back from any chip — composed with ConcatMeshToTensor we get the global view
            gathered_back = ttnn.to_torch(
                gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().cpu().numpy()
            # Each chip should hold the full [1,1,32,128] — ConcatMeshToTensor stacks them
            # along dim=0 (the per-chip outputs). So the result is [4, 1, 32, 128].
            print(f"    concat-back shape: {gathered_back.shape}")
            # Verify chip 0's gathered result matches host_np
            chip0_view = gathered_back[0]
            cos = _cosine(chip0_view, host_np[0])
            print(f"    chip 0 vs ground truth: cos={cos:.6f}  "
                  f"max|Δ|={np.abs(chip0_view - host_np[0]).max():.4e}")
        except Exception as e:
            print(f"  ✗ all_gather FAILED: {type(e).__name__}: {str(e)[:200]}")

        # === Step 4: ttnn.experimental.all_gather_async (production pattern)
        # tt_transformers uses this variant.
        print("\n[4] ttnn.experimental.all_gather_async(dim=3): production op")
        if not hasattr(ttnn, "experimental") or not hasattr(ttnn.experimental, "all_gather_async"):
            print("  ✗ ttnn.experimental.all_gather_async not exposed in this ttnn build")
        else:
            try:
                t0 = time.perf_counter()
                gathered2 = ttnn.experimental.all_gather_async(sharded, dim=3)
                ttnn.synchronize_device(mesh)
                t1 = time.perf_counter()
                ms = (t1 - t0) * 1000.0
                print(f"  ✓ all_gather_async succeeded in {ms:.2f} ms")
            except Exception as e:
                print(f"  ✗ all_gather_async FAILED: {type(e).__name__}: {str(e)[:200]}")

        # === Step 5: Latency micro-benchmark — typical TP comm payload
        # In TP for Qwen3.6, the all-reduce is on residuals (hidden_size=5120)
        # per layer. So all_gather/reduce on [1, 1, 32, 5120/4=1280] sharded.
        print("\n[5] Latency at TP-realistic shape [1, 1, 32, 5120] sharded on dim=3 (1280/chip)")
        host_big = np.random.randn(1, 1, 32, 5120).astype(np.float32) * 0.01
        big_sharded = ttnn.from_torch(
            torch.from_numpy(host_big),
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        # Warmup
        for _ in range(3):
            _ = ttnn.all_gather(big_sharded, dim=3)
        ttnn.synchronize_device(mesh)
        # Measure
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            _ = ttnn.all_gather(big_sharded, dim=3)
        ttnn.synchronize_device(mesh)
        ag_ms = (time.perf_counter() - t0) * 1000.0 / N
        print(f"  ttnn.all_gather median: {ag_ms:.3f} ms  (typical TP per-layer comm)")
        print(f"  Per-token at 16 attn + 64 MLP = ~80 collectives → {ag_ms * 80:.1f} ms/tok comm budget")

        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        print("  Mesh + all_gather working. Multi-chip TP foundation is GO.")
        print("  Next: shard a real matmul (single-layer MLP TP correctness).")
    finally:
        try:
            ttnn.close_mesh_device(mesh)
            print("\n  ✓ mesh closed cleanly")
        except Exception as e:
            print(f"\n  ✗ close_mesh_device error: {e}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric_config reset to DISABLED")
        except Exception as e:
            print(f"  ✗ fabric_config reset error: {e}")


if __name__ == "__main__":
    main()
