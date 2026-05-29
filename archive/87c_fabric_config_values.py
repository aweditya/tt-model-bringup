#!/usr/bin/env python3
"""Probe valid values for ttnn.FabricConfig and set_fabric_config signature."""
import os, sys
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn

print("FabricConfig enum values:")
for name in dir(ttnn.FabricConfig):
    if not name.startswith('_'):
        val = getattr(ttnn.FabricConfig, name)
        print(f"  ttnn.FabricConfig.{name} = {val}")

print("\nset_fabric_config signature/doc:")
print(f"  doc: {ttnn.set_fabric_config.__doc__}")

print("\nTrying each FabricConfig value before mesh open + all_gather:")
for name in dir(ttnn.FabricConfig):
    if name.startswith('_') or 'DISABLED' in name.upper(): continue
    val = getattr(ttnn.FabricConfig, name)
    try:
        ttnn.set_fabric_config(val)
        print(f"  set_fabric_config({name}) OK -> opening mesh...")
        mesh = ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))
        chunk = np.full((1, 1024), 0.5, dtype=np.float32)
        t = ttnn.from_torch(torch.from_numpy(chunk), dtype=ttnn.bfloat16,
                             device=mesh, layout=ttnn.TILE_LAYOUT)
        gathered = ttnn.all_gather(t, dim=0)
        out = ttnn.to_torch(gathered).float().numpy()
        print(f"  >>> SUCCESS with {name}: gathered shape={out.shape}")
        ttnn.distributed.close_mesh_device(mesh)
        # Once we find one that works, stop
        break
    except Exception as e:
        print(f"  {name}: {str(e).split(chr(10))[0][:120]}")
        try:
            ttnn.distributed.close_mesh_device(mesh)
        except Exception:
            pass
