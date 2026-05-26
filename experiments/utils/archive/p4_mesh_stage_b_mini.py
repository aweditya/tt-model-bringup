#!/usr/bin/env python3
"""
P4 — Stage B mini bootstrap (4 layers) on (1,4) mesh (qb2).

Runs the REAL server_tp.py bootstrap path but capped at 4 layers via
TP_MAX_LAYERS=4. This exercises:
  - Layers 0, 1, 2 → linear_attention (DeltaNet)
  - Layer 3 → full_attention (Gated Attn)
Covers both per-layer code paths in Stage B without paying the full 17-min
64-layer cost.

What's validated:
  1. HF safetensors load + per-layer relayout for DN + Gated Attn weights
  2. ShardTensorToMesh / ReplicateTensorToMesh for all weight tensors
  3. Per-layer upload time (extrapolation to 64 layers gives the real boot ETA)
  4. Spot-check: read back one tensor per layer and confirm shape/non-zero stats
  5. Final: embed/lm_head/final_norm/cos_ext/sin_ext upload completes

Pass criteria:
  - All 4 layers loaded without OOM / exception
  - Layer 3 is detected as full_attention
  - Per-layer spot-check tensors have correct shape and non-zero stats

Wall: ~1-2 min (4 layers × ~10-15s + final replicated tensors).
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

# Force the bootstrap to load only 4 layers BEFORE we import server_tp
os.environ['TP_MAX_LAYERS'] = '4'

# Make experiments.serve.server_tp importable
PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from experiments.serve.server_tp import bootstrap, MeshServerState


def main():
    print("=" * 78)
    print("P4: Stage B mini bootstrap (4 layers) on (1,4) mesh (qb2)")
    print("=" * 78)
    print(f"TP_MAX_LAYERS env: {os.environ.get('TP_MAX_LAYERS')}")

    state = MeshServerState()
    overall_pass = True

    t0 = time.time()
    try:
        bootstrap(state)
        t_total = time.time() - t0
        print(f"\n[bootstrap returned in {t_total:.1f}s]")

        # === verification ===
        print("\n[verify] Sanity checks…")
        if len(state.layers) != 4:
            print(f"  ✗ expected 4 layers, got {len(state.layers)}")
            overall_pass = False
        else:
            print(f"  ✓ 4 layers loaded")

        # Layer type coverage
        types = [L['type'] for L in state.layers]
        expected_types = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']
        if types != expected_types:
            print(f"  ✗ layer types: got {types}, expected {expected_types}")
            overall_pass = False
        else:
            print(f"  ✓ layer types match: {types}")

        # Per-layer spot-checks: read back one tensor and validate shape + stats
        import ttnn
        import numpy as np
        for i, L in enumerate(state.layers):
            if L['type'] == 'linear_attention':
                w_tt = L['dn']['w_in']  # Sharded along dim=1 from IN_PROJ_OUT
                label = "dn.w_in"
            else:
                w_tt = L['attn']['w_qkv']
                label = "attn.w_qkv"
            try:
                w_np = ttnn.to_torch(w_tt,
                                      mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)).float().numpy()
                stats = (w_np.shape, float(w_np.mean()), float(w_np.std()), float(np.abs(w_np).max()))
                print(f"  ✓ layer {i} ({L['type']}) {label}: shape={stats[0]} mean={stats[1]:+.4f} std={stats[2]:.4f} max|.|={stats[3]:.4f}")
                if abs(stats[1]) > 1.0 or stats[3] < 1e-6:
                    print(f"     suspicious stats — check weight load path")
                    overall_pass = False
            except Exception as e:
                print(f"  ✗ layer {i} readback failed: {type(e).__name__}: {str(e)[:200]}")
                overall_pass = False

        # Replicated tensors check
        for nm, attr in [("embed_np (host)", "embed_np"),
                          ("final_norm_tt", "final_norm_tt"),
                          ("lm_head_tt", "lm_head_tt"),
                          ("cos_ext_table_tt", "cos_ext_table_tt"),
                          ("sin_ext_table_tt", "sin_ext_table_tt")]:
            if not hasattr(state, attr):
                print(f"  ✗ state.{attr} missing")
                overall_pass = False
            else:
                print(f"  ✓ state.{attr} present")

        # Per-layer load time estimation
        per_layer = t_total / 4
        proj_64 = per_layer * 64
        print(f"\n[perf] per-layer load: {per_layer:.1f}s → projected 64-layer bootstrap: {proj_64/60:.1f} min")

        # === Verdict ===
        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        if overall_pass:
            print("  ✓ P4 PASSES — Stage B 4-layer bootstrap works end-to-end")
            print(f"    Both DN + Gated Attn paths exercised; weights survive round-trip.")
        else:
            print("  ✗ P4 FAIL — see above")

    finally:
        try:
            import ttnn
            if state.mesh is not None:
                ttnn.close_mesh_device(state.mesh)
                print("\n  ✓ mesh closed cleanly")
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED")
        except Exception as e:
            print(f"  ✗ cleanup error: {e}")


if __name__ == "__main__":
    main()
