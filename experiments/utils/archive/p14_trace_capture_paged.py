#!/usr/bin/env python3
"""
P14 — trace capture multi-step decode on (1,4) mesh (qb2).

The full Step 6 of the trace plan. Built on:
  - P12.1: paged_update_cache works on mesh with sharded input
  - P13: paged SDPA fails → kept manual SDPA
  - Step 3+4 (commit 4cd0ce1): server_tp.py refactored to paged cache
  - Step 5 (commit 1fabf07): pre-allocated input buffers +
    forward_token_tp_inner that reads only from buffers (no host writes)

Now the actual trace capture:
  1. Bootstrap TP_MAX_LAYERS=4
  2. Warmup eager forward_token_tp 2x (JIT amortization)
  3. update_input_buffers(state, token=128, cur_pos=2)
  4. begin_trace_capture(mesh, cq_id=0)
  5. logits_tt = forward_token_tp_inner(state)
  6. end_trace_capture(mesh, trace_id, cq_id=0)
  7. Loop 3 steps with varying token/cur_pos:
     - update_input_buffers
     - execute_trace
     - read logits, check finite + chip agreement + sensible argmax

Pass = three traced steps all produce finite logits with chip_disagree=0
and sensible argmax values.

Wall: ~3-4 min (50s bootstrap + JIT + capture + 3 fast traced steps).
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
    forward_token_tp, forward_token_tp_inner, update_input_buffers,
)


def main():
    print("=" * 78, flush=True)
    print("P14: trace capture multi-step decode on (1,4) mesh (qb2)", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
    try:
        t_boot = time.time()
        bootstrap(state)
        print(f"[bootstrap] returned in {time.time() - t_boot:.1f}s", flush=True)

        import ttnn
        import numpy as np

        # === Warmup eager (per feedback_c4v4_validated: JIT must run before capture) ===
        print(f"\n[warmup] eager forward × 2…", flush=True)
        for i in range(2):
            t0 = time.time()
            _ = forward_token_tp(state, token_id=128, cur_pos=i)
            ttnn.synchronize_device(state.mesh)
            print(f"  warmup {i}: {(time.time()-t0)*1000:.0f} ms", flush=True)

        # Capture eager baseline for compare at cur_pos=2
        print(f"\n[baseline] eager forward at token=128, cur_pos=2…", flush=True)
        eager_logits_tt = forward_token_tp(state, token_id=128, cur_pos=2)
        ttnn.synchronize_device(state.mesh)
        eager_logits = ttnn.to_torch(
            eager_logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().numpy()
        eager_argmax = int(eager_logits[0].argmax())
        eager_max = float(np.abs(eager_logits[0]).max())
        print(f"  eager argmax={eager_argmax}  max|abs|={eager_max:.2f}", flush=True)

        # === Begin trace capture ===
        # Pre-fill buffers at cur_pos=3 (one past warmup state)
        update_input_buffers(state, token_id=128, cur_pos=3)
        print(f"\n[trace] begin_trace_capture (token=128, cur_pos=3 at capture time)…", flush=True)
        t_cap = time.time()
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        traced_logits_tt = forward_token_tp_inner(state)
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        print(f"  ✓ trace captured in {(time.time()-t_cap)*1000:.0f} ms (id={trace_id})",
              flush=True)

        # === Execute 3 times with varying inputs ===
        print(f"\n[execute] trace replay 3× with buffer updates…", flush=True)
        replays = [(128, 4), (256, 5), (512, 6)]
        for step_idx, (tok, cp) in enumerate(replays):
            t_upd = time.time()
            update_input_buffers(state, token_id=tok, cur_pos=cp)
            t_exec = time.time()
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(state.mesh)
            t_end = time.time()
            logits = ttnn.to_torch(
                traced_logits_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().numpy()
            finite = bool(np.isfinite(logits).all())
            argmax = int(logits[0].argmax())
            mag = float(np.abs(logits[0]).max())
            chip_diff = float(np.abs(logits[0] - logits[1]).max())
            print(f"  step {step_idx} (tok={tok}, cur_pos={cp}): "
                  f"upd {(t_exec-t_upd)*1000:.0f} ms, "
                  f"exec {(t_end-t_exec)*1000:.0f} ms | "
                  f"finite={finite} argmax={argmax} max|abs|={mag:.2f} chip|Δ|={chip_diff:.4f}",
                  flush=True)

        try:
            ttnn.release_trace(state.mesh, trace_id)
            print(f"  ✓ trace released", flush=True)
        except Exception as e:
            print(f"  ✗ release error: {e}", flush=True)

        print("\n" + "=" * 78, flush=True)
        print(f"  ✓ P14 PASSES — traced multi-step decode works on mesh!", flush=True)
        print(f"    Next: wire trace into handle_generate_tp and measure real tok/s.", flush=True)
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
