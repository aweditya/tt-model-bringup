#!/usr/bin/env python3
"""
P6.5 — single forward step with per-sub-block instrumentation (qb2).

P6 (multi-step) ran for 30 min without printing past "[forward] running 5
decode steps…" and timed out. P6.5 isolates ONE forward step and adds:
  - timing + sync after each per-layer block (DN/attn + MLP)
  - timing + sync after the final norm + lm_head

If JIT is the issue, we'll see slow first-block times but progress. If a
specific op deadlocks, we'll see exactly which block stops printing.

Bootstrap is the same TP_MAX_LAYERS=4 path used by P4/P6 (proven). The
divergence is just in the forward path.
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

os.environ['TP_MAX_LAYERS'] = '4'

PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from experiments.serve.server_tp import (
    bootstrap, MeshServerState,
    deltanet_step_tp, gated_attn_step_tp, mlp_step_tp,
    MAX_POS,
)


def main():
    print("=" * 78, flush=True)
    print("P6.5: single forward step with per-block instrumentation (qb2)", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
    overall_pass = True

    try:
        t0 = time.time()
        bootstrap(state)
        print(f"[bootstrap] returned in {time.time() - t0:.1f}s", flush=True)

        import ttnn
        import torch
        import numpy as np

        cfg = state.cfg
        HIDDEN = cfg['hidden']
        HEAD_DIM = cfg['head_dim']
        ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])

        print(f"\n[setup] preparing inputs for ONE decode step (token=128, cur_pos=0)…", flush=True)

        token_id = 128
        cur_pos = 0

        x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np),
                                 dtype=ttnn.bfloat16, device=state.mesh,
                                 layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([cur_pos], dtype=torch.int32),
            device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        ttnn.synchronize_device(state.mesh)
        print(f"  ✓ inputs ready", flush=True)

        print(f"\n[forward] walking layers (sync+print after each block)…", flush=True)

        for i, layer in enumerate(state.layers):
            t_block = time.time()
            print(f"  layer {i} [{layer['type']}]: attn/dn block START", flush=True)
            try:
                if layer['type'] == 'linear_attention':
                    x_tt = deltanet_step_tp(state, x_tt, layer['dn'], cfg)
                else:
                    x_tt = gated_attn_step_tp(state, x_tt, layer['attn'],
                                                cur_pos_tt, cur_pos, cos_tt, sin_tt, cfg)
                ttnn.synchronize_device(state.mesh)
                print(f"  layer {i} [{layer['type']}]: attn/dn block DONE in "
                      f"{(time.time()-t_block)*1000:.0f}ms", flush=True)
            except Exception as e:
                print(f"  layer {i} [{layer['type']}]: attn/dn block FAILED: "
                      f"{type(e).__name__}: {str(e)[:400]}", flush=True)
                overall_pass = False
                raise

            t_mlp = time.time()
            print(f"  layer {i} mlp block START", flush=True)
            try:
                x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
                ttnn.synchronize_device(state.mesh)
                print(f"  layer {i} mlp block DONE in {(time.time()-t_mlp)*1000:.0f}ms", flush=True)
            except Exception as e:
                print(f"  layer {i} mlp block FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
                overall_pass = False
                raise

        # Final
        t_final = time.time()
        print(f"\n  final norm START", flush=True)
        x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
        ttnn.synchronize_device(state.mesh)
        print(f"  final norm DONE in {(time.time()-t_final)*1000:.0f}ms", flush=True)

        t_lm = time.time()
        print(f"  lm_head START", flush=True)
        logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
        ttnn.synchronize_device(state.mesh)
        print(f"  lm_head DONE in {(time.time()-t_lm)*1000:.0f}ms", flush=True)

        logits_np = ttnn.to_torch(
            logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().numpy()
        print(f"\n  logits shape={logits_np.shape}  finite={int(np.isfinite(logits_np).sum())}/{logits_np.size}  "
              f"max|.|={float(np.abs(logits_np[0]).max()):.2f}  argmax={int(logits_np[0].argmax())}", flush=True)

        chips_max_diff = 0.0
        for j in range(1, state.mesh.get_num_devices()):
            d = float(np.abs(logits_np[0] - logits_np[j]).max())
            chips_max_diff = max(chips_max_diff, d)
        print(f"  chip disagreement: max|Δ|={chips_max_diff:.4f}", flush=True)

        print("\n" + "=" * 78, flush=True)
        if overall_pass:
            print("  ✓ P6.5 PASSES — single forward step works end-to-end on mesh", flush=True)
        else:
            print("  ✗ P6.5 FAIL", flush=True)
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
