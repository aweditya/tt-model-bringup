#!/usr/bin/env python3
"""
P7.7 — Layer 0 DN → Layer 1 DN, NO MLP between.

P7.5 ruled out state copies. P7.6 proved layer 1 DN works alone.
This isolates: does layer 0 DN ALONE wedge layer 1, or does it take MLP?

If layer 1 PASSES → MLP's all_reduce is the killer.
If layer 1 HANGS → layer 0 DN itself is the killer.
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.environ['TP_MAX_LAYERS'] = '4'

PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from experiments.serve.server_tp import bootstrap, MeshServerState
from experiments.utils.p7_dn_layer1_bisect import deltanet_unrolled


def main():
    print("=" * 78, flush=True)
    print("P7.7: layer 0 DN → layer 1 DN, NO MLP between", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
    try:
        t_boot = time.time()
        bootstrap(state)
        print(f"[bootstrap] returned in {time.time() - t_boot:.1f}s", flush=True)

        import ttnn
        import torch
        import numpy as np

        cfg = state.cfg
        HIDDEN = cfg['hidden']
        token_id = 128
        x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                 device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        ttnn.synchronize_device(state.mesh)
        print("\n[setup] x_tt fresh embedding ready", flush=True)

        # --- Layer 0 DN unrolled ---
        print("\n[layer 0] DN UNROLLED (full, with all_reduce + state copies)", flush=True)
        assert state.layers[0]['type'] == 'linear_attention'
        t0 = time.time()
        x_tt = deltanet_unrolled(state.mesh, x_tt, state.layers[0]['dn'], cfg, t0)
        print(f"\n[layer 0] DN done in {(time.time()-t0)*1000:.0f} ms", flush=True)

        # --- NO MLP ---
        print("\n[skip] No MLP between layers", flush=True)

        # --- Layer 1 DN unrolled ---
        print("\n[layer 1] DN UNROLLED after layer 0 DN (no MLP)", flush=True)
        assert state.layers[1]['type'] == 'linear_attention'
        t0 = time.time()
        x_tt = deltanet_unrolled(state.mesh, x_tt, state.layers[1]['dn'], cfg, t0)
        print(f"\n[layer 1] DN done in {(time.time()-t0)*1000:.0f} ms", flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P7.7 PASSED — DN→DN chain works without MLP", flush=True)
        print("    → MLP's collective is the culprit (H1 confirmed direction)", flush=True)
        print("=" * 78, flush=True)

    finally:
        try:
            import ttnn
            if state.mesh is not None:
                ttnn.close_mesh_device(state.mesh)
                print("\n  ✓ mesh closed cleanly", flush=True)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED", flush=True)
        except Exception as e:
            print(f"  ✗ cleanup error: {e}", flush=True)


if __name__ == "__main__":
    main()
