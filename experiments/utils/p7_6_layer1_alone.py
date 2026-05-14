#!/usr/bin/env python3
"""
P7.6 — does layer 1 DN hang ON ITS OWN, with no prior layer-0 execution?

P7 found layer 1 DN hangs at `k_col = reshape(k)`.
P7.5 ruled out H2 (state-copy mutation): hang persists when layer 0
skips ttnn.copy on ssm/conv_st.

This probe: SKIP layer 0 entirely. Feed a fresh replicated embedding
straight into layer 1 DN unrolled. Two outcomes:

  PASS → bug is in layer 0's output (probably H1: all_reduce fallback
         leaves residual in an unusable memory_config that breaks the
         next DN's recurrence).

  HANG → bug is layer-1-specific (the layer-1 weights themselves cause
         k_col reshape to deadlock — much harder to fix).
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
    print("P7.6: layer 1 DN ALONE — no prior layer 0 execution", flush=True)
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
        token_id, cur_pos = 128, 0
        x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                 device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        ttnn.synchronize_device(state.mesh)
        print("\n[setup] x_tt fresh embedding ready", flush=True)

        # --- SKIP layer 0 entirely. Feed x_tt directly into layer 1 DN. ---
        print("\n[layer 1 ALONE] DN UNROLLED on fresh embedding", flush=True)
        assert state.layers[1]['type'] == 'linear_attention'
        t0 = time.time()
        x_tt = deltanet_unrolled(state.mesh, x_tt, state.layers[1]['dn'], cfg, t0)
        print(f"\n[layer 1 ALONE] DN COMPLETED in {(time.time()-t0)*1000:.0f} ms",
              flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P7.6 PASSED — layer 1 DN works in isolation", flush=True)
        print("    → bug is in layer 0's OUTPUT (likely H1: all_reduce layout)", flush=True)
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
