#!/usr/bin/env python3
"""argmax sanity #4 — production vocab size [1, 152064]."""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity #4 — production vocab [1, 152064] on (1,4) mesh")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh chips: {mesh.get_num_devices()}")

    VOCAB = 152064
    VOCAB_PADDED = 248320

    try:
        # Sharded [1, 248320] → AG → [1, 248320 * 4] is wrong...
        # Real path: per-chip slab is 248320/4 = 62080. After AG: [1, 248320].
        # Slice to 152064. We don't want extra factor of 4 — replicate input,
        # since the goal is just to test ttnn.argmax on a [1, 152064] tensor.

        # Pattern 1: directly replicated [1, 152064] — does argmax handle this size?
        print("\n=== Pattern 1: replicated [1, 152064], peak at idx 100000 ===")
        x_np = (np.random.RandomState(7).standard_normal((1, VOCAB)).astype(np.float32) * 0.1)
        x_np[0, 100000] = 99.0
        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        x_rm = ttnn.untilize(x_tt, use_multicore=True)
        ttnn.synchronize_device(mesh)
        # Try both keepdim values
        for kd in [True, False]:
            idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=kd, use_multicore=True)
            ttnn.synchronize_device(mesh)
            idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            v = idx_concat.cpu().numpy().reshape(-1)
            print(f"  keepdim={kd}: shape={tuple(idx_tt.shape)} dtype={idx_tt.dtype}  vals={v[:4].tolist()}  expected=100000")
            ttnn.deallocate(idx_tt)

        # Pattern 2: sharded [1, 248320] from real-shape probe, gather, slice, untilize, argmax
        print("\n=== Pattern 2: sharded [1, 248320] → AG → slice [1, 152064] → argmax ===")
        x_pad = np.zeros((1, VOCAB_PADDED), dtype=np.float32)
        x_pad[:, :VOCAB] = (np.random.RandomState(7).standard_normal((1, VOCAB)).astype(np.float32) * 0.1)
        x_pad[0, 100000] = 99.0
        x_sh_tt = ttnn.from_torch(
            torch.from_numpy(x_pad),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        print(f"  per-chip shape: {tuple(x_sh_tt.shape)}")
        x_g = ttnn.all_gather(x_sh_tt, dim=-1)
        ttnn.synchronize_device(mesh)
        print(f"  gathered shape: {tuple(x_g.shape)}")
        x_s = ttnn.slice(x_g, [0, 0], [1, VOCAB])
        ttnn.synchronize_device(mesh)
        print(f"  sliced shape: {tuple(x_s.shape)} layout={x_s.layout}")
        x_rm = ttnn.untilize(x_s, use_multicore=True)
        ttnn.synchronize_device(mesh)
        for kd in [True, False]:
            idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=kd, use_multicore=True)
            ttnn.synchronize_device(mesh)
            idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            v = idx_concat.cpu().numpy().reshape(-1)
            print(f"  keepdim={kd}: vals={v[:4].tolist()}  expected=100000")
            ttnn.deallocate(idx_tt)

        # Pattern 3: sharded [1, 248320] → AG → untilize → argmax (no slice — find peak in padding)
        print("\n=== Pattern 3: sharded [1, 248320] → AG → untilize → argmax (no slice) ===")
        x_pad[0, 200000] = 999.0  # peak in PADDING region
        x_sh_tt2 = ttnn.from_torch(
            torch.from_numpy(x_pad),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        x_g2 = ttnn.all_gather(x_sh_tt2, dim=-1)
        ttnn.synchronize_device(mesh)
        x_rm2 = ttnn.untilize(x_g2, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_tt = ttnn.argmax(x_rm2, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        v = idx_concat.cpu().numpy().reshape(-1)
        print(f"  full-padded argmax vals: {v[:4].tolist()}  expected=200000")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
