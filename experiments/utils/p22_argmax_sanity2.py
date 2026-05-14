#!/usr/bin/env python3
"""Sanity-test ttnn.argmax with ConcatMeshToTensor readback on replicated mesh tensor."""
import sys
import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


def main():
    print("=" * 78)
    print("argmax sanity #2 — read via ConcatMeshToTensor(dim=0)")
    print("=" * 78)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"mesh chips: {mesh.get_num_devices()}")

    try:
        # Replicated input, peak at idx 73
        x_np = np.linspace(0.0, 1.0, 128, dtype=np.float32).reshape(1, 128)
        x_np[0, 73] = 5.0
        print(f"expected argmax = 73")

        # Try different starting layouts
        for layout_name, layout in [("ROW_MAJOR", ttnn.ROW_MAJOR_LAYOUT), ("TILE", ttnn.TILE_LAYOUT)]:
            print(f"\n=== input layout: {layout_name} ===")
            x_tt = ttnn.from_torch(
                torch.from_numpy(x_np),
                dtype=ttnn.bfloat16, device=mesh,
                layout=layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.synchronize_device(mesh)

            if layout == ttnn.TILE_LAYOUT:
                # need row-major for argmax
                x_rm = ttnn.untilize(x_tt, use_multicore=True)
                ttnn.synchronize_device(mesh)
            else:
                x_rm = x_tt

            # Try argmax with use_multicore=True + keepdim=True (matches Galaxy)
            for kwargs in [
                dict(dim=-1, keepdim=True, use_multicore=True),
                dict(dim=-1, keepdim=False, use_multicore=True),
                dict(dim=-1, keepdim=True, use_multicore=False),
                dict(dim=-1, keepdim=False, use_multicore=False),
            ]:
                try:
                    idx_tt = ttnn.argmax(x_rm, **kwargs)
                    ttnn.synchronize_device(mesh)
                    idx_concat = ttnn.to_torch(idx_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                    v = idx_concat.cpu().numpy().reshape(-1)
                    print(f"  {kwargs}: shape={tuple(idx_tt.shape)} dtype={idx_tt.dtype} "
                          f"vals[:4]={v[:4]} expected=73")
                    ttnn.deallocate(idx_tt)
                except Exception as e:
                    print(f"  {kwargs}: EXCEPTION {type(e).__name__}: {str(e)[:120]}")

    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
