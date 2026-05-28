#!/usr/bin/env python3
"""
Experiment 87b — Probe fabric init for multi-chip collectives.

A7 found the mesh-open API but collectives fail with:
  TT_FATAL @ tt_metal/fabric/control_plane.cpp: fabric_context_ != nullptr

We need to init the fabric. This script probes several plausible APIs.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python experiments/87b_fabric_init_probe.py
"""
import os, sys, traceback
sys.path.insert(0, os.path.expanduser("~"))
import numpy as np
import torch
import ttnn


def discover_fabric():
    """List anything fabric-related in ttnn."""
    print("Fabric-related symbols:")
    for ns_name, ns in [('ttnn', ttnn),
                         ('ttnn.distributed', getattr(ttnn, 'distributed', None)),
                         ('ttnn.experimental', getattr(ttnn, 'experimental', None))]:
        if ns is None: continue
        for name in dir(ns):
            if 'fabric' in name.lower() or 'topology' in name.lower():
                print(f"  {ns_name}.{name}")


def try_open_with_options():
    """Probe variants of mesh open with fabric-related options."""
    attempts = []

    # Variant 1: open_mesh_device with all defaults
    attempts.append(("open_mesh_device(MeshShape(2,1))",
                     lambda: ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))))

    # Variant 2: try create_mesh_device (different API)
    if hasattr(ttnn.distributed, 'create_mesh_device'):
        attempts.append(("create_mesh_device(MeshShape(2,1))",
                         lambda: ttnn.distributed.create_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))))

    # Variant 3: with fabric_config kwarg if available
    # We don't know the exact signature; just try.

    print("\nVariants:")
    devices = []
    for label, fn in attempts:
        try:
            d = fn()
            print(f"  OK  {label}  -> {d}")
            devices.append((label, d))
        except TypeError as e:
            print(f"  TypeError  {label}: {str(e)[:120]}")
        except Exception as e:
            print(f"  FAIL  {label}: {str(e).split(chr(10))[0][:120]}")
    return devices


def try_fabric_init(mesh):
    """Probe ways to init the fabric."""
    candidates = []
    for sym in ['init_fabric', 'fabric_init', 'set_fabric_config',
                'set_default_fabric_config', 'configure_fabric']:
        for ns_name, ns in [('ttnn', ttnn),
                             ('ttnn.distributed', getattr(ttnn, 'distributed', None)),
                             ('ttnn.experimental', getattr(ttnn, 'experimental', None))]:
            if ns is None: continue
            if hasattr(ns, sym):
                candidates.append((f"{ns_name}.{sym}", getattr(ns, sym)))
    print(f"\nFabric-init candidates: {[c[0] for c in candidates]}")
    return candidates


def test_collective(mesh):
    """Try a simple all_gather to verify fabric works."""
    try:
        chunk = np.full((1, 1024), 0.5, dtype=np.float32)
        t = ttnn.from_torch(torch.from_numpy(chunk), dtype=ttnn.bfloat16,
                             device=mesh, layout=ttnn.TILE_LAYOUT)
        gathered = ttnn.all_gather(t, dim=0)
        out = ttnn.to_torch(gathered).float().numpy()
        print(f"  all_gather OK, shape={out.shape}")
        return True
    except Exception as e:
        print(f"  all_gather FAIL: {str(e).split(chr(10))[0][:120]}")
        return False


def main():
    discover_fabric()

    print("\n=== Try env var ===")
    # Sometimes setting an env var before opening helps
    os.environ['TT_METAL_FABRIC_ENABLED'] = '1'
    print("set TT_METAL_FABRIC_ENABLED=1")

    print("\n=== Variant openings ===")
    devices = try_open_with_options()

    for label, mesh in devices:
        print(f"\n=== Testing collectives on {label} ===")
        ok = test_collective(mesh)
        if ok:
            print(f"\n  SUCCESS: {label} supports collectives")

        # Probe fabric init APIs on this mesh
        candidates = try_fabric_init(mesh)
        for cand_name, cand_fn in candidates:
            try:
                cand_fn(mesh)
                print(f"  Called {cand_name}(mesh) without error")
                ok = test_collective(mesh)
                if ok:
                    print(f"  >>> AFTER {cand_name}: collectives WORK!")
                    break
            except TypeError as e:
                # Try without mesh arg
                try:
                    cand_fn()
                    print(f"  Called {cand_name}() — no-arg form")
                    ok = test_collective(mesh)
                    if ok:
                        print(f"  >>> AFTER {cand_name}(): collectives WORK!")
                        break
                except Exception as e2:
                    print(f"  {cand_name}: TypeError on both arity")
            except Exception as e:
                print(f"  {cand_name} raised: {str(e)[:100]}")

        # Close the mesh
        try:
            ttnn.distributed.close_mesh_device(mesh)
        except Exception:
            pass


if __name__ == "__main__":
    main()
