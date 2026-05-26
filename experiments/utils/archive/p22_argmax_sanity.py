#!/usr/bin/env python3
"""Sanity-test ttnn.argmax in isolation on (1,4) mesh.

The vocab-sharded probe found argmax returning fp32-bit-pattern-like values
instead of indices. This probe isolates the issue.
"""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity — small known input")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh chips: {mesh.get_num_devices()}")

    try:
        # Build a known [1, 128] vector with max at index 73 — replicated.
        # Range small to easily distinguish bit-patterns.
        x_np = np.linspace(0.0, 1.0, 128, dtype=np.float32).reshape(1, 128)
        x_np[0, 73] = 5.0  # peak at idx 73
        print(f"x[0,73]={x_np[0,73]}  expected argmax=73")

        x_tt = ttnn.from_torch(
            torch.from_numpy(x_np),
            dtype=ttnn.bfloat16, device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)

        # Try several call patterns
        for label, kwargs in [
            ("dim=-1 keepdim=False  use_multicore=True", dict(dim=-1, keepdim=False, use_multicore=True)),
            ("dim=-1 keepdim=True   use_multicore=True", dict(dim=-1, keepdim=True, use_multicore=True)),
            ("dim=-1 keepdim=False  use_multicore=False", dict(dim=-1, keepdim=False, use_multicore=False)),
        ]:
            print(f"\n--- {label} ---")
            x_rm = ttnn.untilize(x_tt, use_multicore=True)
            try:
                idx_tt = ttnn.argmax(x_rm, **kwargs)
                ttnn.synchronize_device(mesh)
                print(f"  idx_tt shape={tuple(idx_tt.shape)} dtype={idx_tt.dtype} layout={idx_tt.layout}")
                idx_host = ttnn.from_device(idx_tt)
                idx_t = ttnn.to_torch(idx_host)
                print(f"  to_torch shape={tuple(idx_t.shape)} dtype={idx_t.dtype}")
                vals = idx_t.cpu().numpy().reshape(-1)
                print(f"  values [first 8]: {vals[:8]}")
                # try mesh composer
                idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                print(f"  concat-shape: {tuple(idx_concat.shape)} dtype={idx_concat.dtype}")
                print(f"  concat-vals: {idx_concat.cpu().numpy().reshape(-1)[:16]}")
                ttnn.deallocate(idx_tt)
            except Exception as e:
                print(f"  EXCEPTION: {type(e).__name__}: {str(e)[:200]}")
            ttnn.deallocate(x_rm)

        # Also try with single device (no mesh composer)
        print("\n--- using to_torch on raw mesh tensor (no composer) ---")
        x_rm = ttnn.untilize(x_tt, use_multicore=True)
        idx_tt = ttnn.argmax(x_rm, dim=-1, keepdim=True, use_multicore=True)
        ttnn.synchronize_device(mesh)
        print(f"  idx_tt shape={tuple(idx_tt.shape)} dtype={idx_tt.dtype}")
        # Try just .cpu():
        try:
            idx_t = ttnn.to_torch(ttnn.from_device(idx_tt))
            print(f"  from_device → to_torch shape={tuple(idx_t.shape)} dtype={idx_t.dtype}")
            print(f"  values: {idx_t.cpu().numpy().reshape(-1)[:8]}")
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {str(e)[:200]}")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
