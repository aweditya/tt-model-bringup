#!/usr/bin/env python3
"""
P6 — multi-step decode with REAL Qwen3.6-27B weights on (1,4) mesh (qb2).

Bootstraps server_tp.py at TP_MAX_LAYERS=4 (3× DeltaNet + 1× Gated Attn).
Then runs `forward_token_tp` for 5 sequential decode steps and verifies:

  1. State threading: SSM buffer for DN layers CHANGES across steps
     (proves ttnn.copy in deltanet_step_tp actually mutates state on mesh)
  2. KV cache: each step's cur_pos row in the Gated Attn cache is populated
     (proves update_cache_for_token_ works in production multi-step path)
  3. Numerical sanity: all outputs finite, reasonable magnitude
  4. Output replication: all 4 chips agree on lm_head logits

Why this matters:
  - This is the FIRST end-to-end mesh decode loop with REAL weights.
  - If P6 passes, the only remaining unknown is scale (4 → 64 layers) and
    full-vocab logit quality. Both are plumbing.
  - If P6 fails, the failure mode tells us exactly which integration broke
    (state threading, KV cache, output replication, or numerical drift).

Pass: 5 sequential decode steps complete; SSM changes monotonically per DN
layer; KV row at each cur_pos populated; logits finite + chips agree.

Wall: ~70s (50s bootstrap + 5 forward steps + readbacks).
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

os.environ['TP_MAX_LAYERS'] = '4'

PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from experiments.serve.server_tp import bootstrap, forward_token_tp, MeshServerState


def main():
    print("=" * 78)
    print("P6: multi-step decode with REAL weights on (1,4) mesh (qb2)")
    print("=" * 78)
    print(f"TP_MAX_LAYERS={os.environ.get('TP_MAX_LAYERS')}")

    state = MeshServerState()
    overall_pass = True

    t0 = time.time()
    try:
        bootstrap(state)
        print(f"[bootstrap returned in {time.time() - t0:.1f}s]")

        import ttnn
        import numpy as np

        # Helper: read a sharded tensor → concatenated numpy
        def read_sharded(t, dim=0):
            return ttnn.to_torch(
                t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=dim)
            ).float().numpy()

        # Pre-state snapshots (before step 0)
        # layers 0/1/2 are DN → snapshot ssm
        # layer 3 is Gated Attn → snapshot kc/vc
        dn_indices = [i for i, L in enumerate(state.layers) if L['type'] == 'linear_attention']
        attn_indices = [i for i, L in enumerate(state.layers) if L['type'] == 'full_attention']
        print(f"\n[layout] DN layers: {dn_indices}  Gated Attn layers: {attn_indices}")

        print(f"\n[init snapshots] zero-state check:")
        ssm_prev = {}
        for i in dn_indices:
            arr = read_sharded(state.layers[i]['dn']['ssm'], dim=0)
            ssm_prev[i] = arr
            print(f"  layer {i} ssm shape={arr.shape}  mean={arr.mean():+.6f}  "
                  f"abs.max={float(np.abs(arr).max()):.6f}  (expect ~0 init)")

        # Token sequence: pick arbitrary token ids in vocab range
        # First 5 tokens — they don't need to be a real prompt; we're testing the
        # mechanics, not semantics.
        token_ids = [128, 256, 512, 1024, 2048]

        print(f"\n[forward] running {len(token_ids)} decode steps…")
        for step_idx, token_id in enumerate(token_ids):
            cur_pos = step_idx
            t_step = time.time()
            logits_tt = forward_token_tp(state, token_id, cur_pos)
            ttnn.synchronize_device(state.mesh)
            step_ms = (time.time() - t_step) * 1000

            # Verify logits replication + finiteness
            logits_np = ttnn.to_torch(
                logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().numpy()
            NCHIPS = state.mesh.get_num_devices()
            n_finite = int(np.isfinite(logits_np).sum())
            n_total = logits_np.size
            chips_diff = float(np.abs(logits_np[0] - logits_np[1]).max())
            # all-chip pairwise max diff:
            chips_max_diff = 0.0
            for j in range(1, NCHIPS):
                d = float(np.abs(logits_np[0] - logits_np[j]).max())
                chips_max_diff = max(chips_max_diff, d)
            argmax = int(logits_np[0].argmax())
            mag = float(np.abs(logits_np[0]).max())
            print(f"  step {step_idx} (token={token_id}, cur_pos={cur_pos}):  "
                  f"{step_ms:.0f}ms  finite={n_finite}/{n_total}  "
                  f"max|abs|={mag:.2f}  argmax={argmax}  chip_disagree={chips_max_diff:.4f}")

            if n_finite != n_total:
                print(f"     ✗ NON-FINITE values in logits")
                overall_pass = False
            if chips_max_diff > 1e-2:
                print(f"     ✗ chips disagree on logits ({chips_max_diff:.4f})")
                overall_pass = False
            if mag > 1e4:
                print(f"     ✗ logits magnitude exploded ({mag})")
                overall_pass = False

            # State-change check on DN layers
            for i in dn_indices:
                arr = read_sharded(state.layers[i]['dn']['ssm'], dim=0)
                diff_from_prev = float(np.abs(arr - ssm_prev[i]).max())
                changed = diff_from_prev > 1e-7
                marker = "✓" if changed else "✗"
                print(f"     {marker} layer {i} DN ssm advanced from prev: max|Δ|={diff_from_prev:.6f}")
                if not changed:
                    overall_pass = False
                ssm_prev[i] = arr

            # KV cache check for attention layer
            for i in attn_indices:
                # cache was rebuilt as 4D [B=1, N_KV=4, MAX_POS, HEAD_DIM] sharded dim=1
                # → per-chip [1, 1, MAX_POS, HEAD_DIM] → concat dim=0 → [4, 1, MAX_POS, HEAD_DIM]
                kc = read_sharded(state.layers[i]['attn']['kc'], dim=0)
                row_at_cp = kc[:, 0, cur_pos, :]
                row_mag = float(np.abs(row_at_cp).max())
                future = kc[:, 0, cur_pos + 1:, :]
                others_mag = float(np.abs(future).max()) if future.size > 0 else 0.0
                populated = row_mag > 1e-6
                marker = "✓" if populated else "✗"
                print(f"     {marker} layer {i} attn kc[cur_pos={cur_pos}]: max|.|={row_mag:.4f}  "
                      f"vs future-rows max|.|={others_mag:.4f}")
                if not populated:
                    overall_pass = False

        print("\n" + "=" * 78)
        print("VERDICT")
        print("=" * 78)
        if overall_pass:
            print("  ✓ P6 PASSES — multi-step decode with real weights works end-to-end")
            print("    State threading verified across DN layers; KV cache populated")
            print("    correctly; logits replicated + finite. Path to full 64-layer bootstrap")
            print("    + actual multi-chip generate_tp is unblocked.")
        else:
            print("  ✗ P6 PARTIAL/FAIL — see above for failing assertions")

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
