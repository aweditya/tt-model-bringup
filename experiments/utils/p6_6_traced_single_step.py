#!/usr/bin/env python3
"""
P6.6 — single forward step wrapped in begin/end_trace_capture on (1,4) mesh (qb2).

P6.5 showed eager layer-1 DN hangs. Two questions this probe answers:

  Q1: Does trace capture itself complete past layer 1 DN?
      (Capture runs ops on device same as eager — but maybe a different CQ
      avoids whatever fabric state is deadlocking.)

  Q2: If capture completes, does execute_trace also complete?

Three milestones with prints:
  (a) Before trace capture: log "before begin_trace_capture"
  (b) Inside trace capture (per-block prints, same instrumentation as P6.5)
  (c) After end_trace_capture: log "trace captured (id=...)"
  (d) execute_trace × 1: log "executed" or where it hangs

If trace bypasses the deadlock — we have the path to multi-chip inference.
If trace also hangs at layer 1 DN — bisection (P7) is the next step.

Wall: TBD. Eager hung indefinitely at this point. Timeout 25 min.
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


def run_forward(state, x_tt, cur_pos_tt, cur_pos, cos_tt, sin_tt, label):
    """One full forward pass through state.layers; prints per-block."""
    import ttnn
    cfg = state.cfg
    for i, layer in enumerate(state.layers):
        print(f"  [{label}] layer {i} [{layer['type']}]: attn/dn START", flush=True)
        t0 = time.time()
        try:
            if layer['type'] == 'linear_attention':
                x_tt = deltanet_step_tp(state, x_tt, layer['dn'], cfg)
            else:
                x_tt = gated_attn_step_tp(state, x_tt, layer['attn'],
                                            cur_pos_tt, cur_pos, cos_tt, sin_tt, cfg)
            print(f"  [{label}] layer {i} [{layer['type']}]: attn/dn DONE in {(time.time()-t0)*1000:.0f}ms", flush=True)
        except Exception as e:
            print(f"  [{label}] layer {i} attn/dn FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise

        t1 = time.time()
        try:
            x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
            print(f"  [{label}] layer {i} mlp DONE in {(time.time()-t1)*1000:.0f}ms", flush=True)
        except Exception as e:
            print(f"  [{label}] layer {i} mlp FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise

    print(f"  [{label}] final norm START", flush=True)
    x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
    logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
    print(f"  [{label}] final norm+lm_head DONE", flush=True)
    return logits_tt


def main():
    print("=" * 78, flush=True)
    print("P6.6: traced single forward step (qb2)", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
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

        cur_pos = 0
        token_id = 128

        # Inputs (same as P6.5)
        print(f"\n[setup] inputs for token={token_id}, cur_pos={cur_pos}…", flush=True)
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

        # === Q1: try begin_trace_capture WITHOUT warmup ===
        # Per C'4v4 single-chip finding, JIT runs during capture. That's fine
        # for one-shot; we just need to see if it COMPLETES.
        print(f"\n[trace] calling begin_trace_capture…", flush=True)
        t_cap = time.time()
        trace_id = None
        try:
            trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
            print(f"  ✓ begin_trace_capture returned (id={trace_id})", flush=True)
        except Exception as e:
            print(f"  ✗ begin_trace_capture FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise

        try:
            logits_tt = run_forward(state, x_tt, cur_pos_tt, cur_pos, cos_tt, sin_tt, "capture")
        except Exception as e:
            print(f"\n[trace] forward inside capture FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            try:
                ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
            except Exception:
                pass
            raise

        try:
            ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
            print(f"\n  ✓ end_trace_capture returned ({(time.time()-t_cap)*1000:.0f}ms total capture)", flush=True)
        except Exception as e:
            print(f"  ✗ end_trace_capture FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise

        # === Q2: execute_trace ===
        print(f"\n[trace] execute_trace…", flush=True)
        t_exec = time.time()
        try:
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(state.mesh)
            print(f"  ✓ execute_trace DONE in {(time.time()-t_exec)*1000:.0f}ms", flush=True)
        except Exception as e:
            print(f"  ✗ execute_trace FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
            raise

        # Read logits
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

        try:
            ttnn.release_trace(state.mesh, trace_id)
            print(f"  ✓ trace released", flush=True)
        except Exception as e:
            print(f"  ✗ release_trace error: {e}", flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P6.6 PASSES — traced forward works on mesh", flush=True)
        print("    Multi-chip TP path via trace is unblocked.", flush=True)
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
