#!/usr/bin/env python3
"""Floor microbench — split the ~63 ms B-independent floor into its parts.

Block profile (B=1, owned_gdn+shiftacc): MLP 29.4 ms + DN 25.1 + ATT 6.6 +
rest 2.2. The floor is weight-streaming matmuls + per-layer overhead. This
isolates (traced, B=1) the two suspects:
  - one all-reduce on [1, HIDDEN]  (×128/step: 48 DN + 16 ATT + 64 MLP out-projs)
  - the MLP matmuls gate/up/down    (bf8 weight-streaming, ×64 layers)
so we know whether to attack the collectives or the matmul BW.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_floor_microbench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_tp as base       # noqa: E402
import server_tp_cb as cb      # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import ttnn, torch
    STEPS, WARM = 100, 5
    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    HIDDEN = state.cfg['hidden']
    mesh = state.mesh

    def repl(shape):
        return ttnn.from_torch(torch.randn(*shape) * 0.1, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=mesh,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    def time_traced(build_fn):
        for _ in range(2):
            o = build_fn(); ttnn.synchronize_device(mesh)
            try: ttnn.deallocate(o)
            except Exception: pass
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        o = build_fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(WARM):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(STEPS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / STEPS * 1000.0
        ttnn.release_trace(mesh, tid)
        return ms

    mlp = next(l['mlp'] for l in state.layers)
    wg, wu, wd = mlp['w_gate'], mlp['w_up'], mlp['w_down']
    log(f"HIDDEN={HIDDEN}  w_gate={list(wg.shape)} w_up={list(wu.shape)} w_down={list(wd.shape)}")

    h = repl((1, HIDDEN))
    interm_chip = wg.shape[-1]
    h2 = repl((1, interm_chip))
    ar_in = repl((1, HIDDEN))  # PRE-ALLOCATED — never from_torch inside a trace-
    # capture region (host→device transfer can't be captured → hangs).

    log("=== per-op traced (B=1) — logged incrementally (matmuls first) ===")
    gate_ms = time_traced(lambda: ttnn.linear(h, wg, activation="silu"))
    log(f"  MLP gate matmul        : {gate_ms*1000:7.2f} us   (×64 = {gate_ms*64:6.2f} ms)")
    up_ms = time_traced(lambda: ttnn.linear(h, wu))
    log(f"  MLP up matmul          : {up_ms*1000:7.2f} us   (×64 = {up_ms*64:6.2f} ms)")
    down_ms = time_traced(lambda: ttnn.linear(h2, wd))
    log(f"  MLP down matmul        : {down_ms*1000:7.2f} us   (×64 = {down_ms*64:6.2f} ms)")
    mlp_mm = (gate_ms + up_ms + down_ms) * 64
    log(f"  MLP matmuls total      : {mlp_mm:6.2f} ms   (block profile MLP = 29.4 ms)")
    ar_ms = time_traced(lambda: cb._tp_all_reduce(state, ar_in))
    log(f"  all_reduce[1,{HIDDEN}] : {ar_ms*1000:7.2f} us   (×128/step = {ar_ms*128:6.2f} ms)")
    log(f"  collectives total      : {ar_ms*128:6.2f} ms   of the ~63 ms floor")
    log("  → if collectives are large → fuse all_reduce into out-proj (rs_matmul);")
    log("    if MLP matmuls ≫ their bf8 BW floor → core_grid tuning (A004 lever).")


if __name__ == "__main__":
    main()
