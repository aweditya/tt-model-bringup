#!/usr/bin/env python3
"""
P9 — 3 forward_token_tp calls IN A ROW, no readbacks between (qb2).

Hypothesis test from feedback_p6_step2_hangs.md:
  - Slow JIT recompile at step 2? REFUTED by P6-extended (15+ min still hung).
  - Real deadlock at step 2?
    P6 includes per-step readbacks (read_sharded on ssm/kc) between forwards.
    Those readbacks force sync + transfer per chip. They may interact with
    the next forward in unexpected ways.

P9 removes ALL between-step work. Three forward_token_tp calls:
  call 0 (token=128, cur_pos=0): expect ~500 ms (cold JIT first DN/MLP/Attn)
  call 1 (token=256, cur_pos=1): expect ~70 ms  (warm)
  call 2 (token=512, cur_pos=2): ???

If call 2 hangs → deadlock is in forward_token_tp itself; readbacks were
not the cause. Move to next hypothesis (sync after ttnn.copy state mutation
or deallocate intermediates).

If call 2 completes → P6's readback code was triggering the hang; fix the
readback pattern.

Wall budget: 60s bootstrap + 3 forward calls. 5 min timeout is plenty.
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
    print("=" * 78, flush=True)
    print("P9: three forward_token_tp calls, NO readbacks between", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
    try:
        t_boot = time.time()
        bootstrap(state)
        print(f"[bootstrap] returned in {time.time() - t_boot:.1f}s", flush=True)

        import ttnn

        steps = [(128, 0), (256, 1), (512, 2)]
        for i, (tok, cp) in enumerate(steps):
            print(f"\n[call {i}] forward_token_tp(token={tok}, cur_pos={cp}) START", flush=True)
            t0 = time.time()
            _ = forward_token_tp(state, tok, cp)
            ttnn.synchronize_device(state.mesh)
            print(f"[call {i}] DONE in {(time.time()-t0)*1000:.0f} ms", flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P9 PASSED — three forward calls complete without readbacks", flush=True)
        print("    → P6's hang at step 2 came from readback code, not forward_token_tp itself.", flush=True)
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
