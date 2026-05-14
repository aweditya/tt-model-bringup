#!/usr/bin/env python3
"""Sanity-test argmax after the all_gather → slice → untilize path.

Vocab probe failed: argmax returned bit-patterns of max values, not indices.
This isolates whether all_gather or slice or large-vocab is the cause.
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity #3 — all_gather → slice → untilize → argmax")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    NCHIPS = mesh.get_num_devices()
    print(f"mesh chips: {NCHIPS}")

    try:
        # ---------- Test A: argmax post-all_gather (no slice) -------------
        # Build a sharded [1, 128] with each chip holding [1, 32]
        # Peak at global idx 73 (which is chip 2, local idx 9)
        # When gathered, peak should still be at global idx 73.
        print("\n=== Test A: sharded [1, 128] → all_gather → argmax (no slice) ===")
        x_np = np.linspace(0.0, 1.0, 128, dtype=np.float32).reshape(1, 128)
        x_np[0, 73] = 5.0

        x_sh_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        print(f"  sharded shape: {tuple(x_sh_tt.shape)}  (expected (1,32) per chip)")
        ttnn.synchronize_device(mesh)

        x_gathered = ttnn.all_gather(x_sh_tt, dim=-1)
        ttnn.synchronize_device(mesh)
        print(f"  gathered shape: {tuple(x_gathered.shape)}  layout={x_gathered.layout}")

        x_rm = ttnn.untilize(x_gathered, use_multicore=True)
        ttnn.synchronize_device(mesh)
        print(f"  untilized shape: {tuple(x_rm.shape)}  layout={x_rm.layout}")

        idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        print(f"  argmax shape: {tuple(idx_tt.shape)}  dtype={idx_tt.dtype}")
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print(f"  argmax vals (per chip): {idx_concat.cpu().numpy().reshape(-1).tolist()}  expected=73")

        # ---------- Test B: with slice (mimic full probe path) -----------
        # Sharded [1, 256] → AG → [1, 1024] → slice [1, 128] → argmax
        print("\n=== Test B: sharded [1, 256] → AG → slice [1, 128] → argmax ===")
        x_np_padded = np.linspace(0.0, 1.0, 1024, dtype=np.float32).reshape(1, 1024)
        x_np_padded[0, 73] = 5.0  # peak inside slice region
        # Add a fake peak in PADDING region to confirm slice works
        x_np_padded[0, 500] = 10.0  # peak in padding (should be sliced away)

        x_sh_tt = ttnn.from_torch(
            torch.from_numpy(x_np_padded),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        ttnn.synchronize_device(mesh)
        print(f"  sharded shape: {tuple(x_sh_tt.shape)}")

        x_gathered = ttnn.all_gather(x_sh_tt, dim=-1)
        ttnn.synchronize_device(mesh)
        print(f"  gathered shape: {tuple(x_gathered.shape)}")

        x_sliced = ttnn.slice(x_gathered, [0, 0], [1, 128])
        ttnn.synchronize_device(mesh)
        print(f"  sliced shape: {tuple(x_sliced.shape)}  layout={x_sliced.layout}")

        x_rm = ttnn.untilize(x_sliced, use_multicore=True)
        ttnn.synchronize_device(mesh)
        print(f"  untilized shape: {tuple(x_rm.shape)}")

        idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print(f"  argmax vals: {idx_concat.cpu().numpy().reshape(-1).tolist()}  expected=73")

        # ---------- Test C: large vocab (mimic production size 152064) -----
        print("\n=== Test C: sharded [1, 62080] → AG → slice [1, 38016] → argmax (mini-prod size) ===")
        # 248320 / 4 = 62080 per chip; slice to 152064 (5×32×950)... use easier numbers
        # NCHIPS=4. Per-chip 62080 = 5×32×388. Use simpler [1, 4096] sharded → [1, 16384] gathered → slice [1, 8192]
        N_PER = 4096
        N_TOTAL = N_PER * NCHIPS  # 16384
        N_SLICE = 8192
        target = 7500  # inside slice region
        x_np_big = (np.random.RandomState(7).standard_normal((1, N_TOTAL)).astype(np.float32) * 0.1)
        x_np_big[0, target] = 99.0
        # Also a peak above the slice region (should NOT win)
        x_np_big[0, 12000] = 999.0

        x_sh_tt = ttnn.from_torch(
            torch.from_numpy(x_np_big),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        x_gathered = ttnn.all_gather(x_sh_tt, dim=-1)
        ttnn.synchronize_device(mesh)
        print(f"  gathered shape: {tuple(x_gathered.shape)}")

        x_sliced = ttnn.slice(x_gathered, [0, 0], [1, N_SLICE])
        ttnn.synchronize_device(mesh)
        print(f"  sliced shape: {tuple(x_sliced.shape)}  layout={x_sliced.layout}")

        x_rm = ttnn.untilize(x_sliced, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print(f"  argmax vals: {idx_concat.cpu().numpy().reshape(-1).tolist()}  expected={target}")

        # ---------- Test D: WITHOUT slice — gather + untilize + argmax over full padded -----
        print("\n=== Test D: argmax over FULL gathered (no slice), find peak in padding ===")
        # Reuse x_np_big — peak at 12000 (in 16384) should win when no slice
        x_sh_tt2 = ttnn.from_torch(
            torch.from_numpy(x_np_big),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        x_g2 = ttnn.all_gather(x_sh_tt2, dim=-1)
        x_rm2 = ttnn.untilize(x_g2, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_tt2 = ttnn.argmax(x_rm2, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_concat2 = ttnn.to_torch(idx_tt2, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        print(f"  argmax vals: {idx_concat2.cpu().numpy().reshape(-1).tolist()}  expected=12000 (peak in padding)")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
