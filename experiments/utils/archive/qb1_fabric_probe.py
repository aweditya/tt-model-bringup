#!/usr/bin/env python3
"""Probe whether qb1 has inter-chip fabric (CLAUDE.md says no; verifying live)."""
import ttnn

print("[1] try set_fabric_config(FABRIC_1D)…")
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("  set_fabric_config returned without error")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    raise SystemExit(1)

print("[2] try open_mesh_device((1,4))…")
try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  mesh opened: {mesh}")
    print("  ✓ qb1 HAS fabric — can do TP work here")
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")
    raise SystemExit(2)
