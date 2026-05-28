#!/usr/bin/env python3
"""DNK-G3b — within-DN vector-op attribution (traced, B=32).

DNK-G3 showed the DN matmuls are flat with B; the 3.61 ms/seq DN slope is the
per-slot VECTOR ops. This ranks them: isolate the DN layers (cb_skip_blocks=
{attn,mlp}) and skip one DN sub-op at a time (cb_dn_skip), timing execute_trace.
DN-only-full − skip_X = X's cost across the 48 DN layers at B. Tells us which
vector op to fuse next (after owned_gdn fused the recurrence).

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_profile_dn.py --batch 32 --steps 50 --owned-gdn
"""
from __future__ import annotations

import argparse
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


def time_dn(state, B, dn_skip, steps, warmup, blocks_per_seq):
    import ttnn, time as _t
    state.cb_skip_blocks = {'attn', 'mlp'}   # DN-only forward
    state.cb_dn_skip = set(dn_skip)
    cb.setup_cb_state(state, B, blocks_per_seq=blocks_per_seq)
    cb.cb_reset_states(state)
    for i in range(2):
        cb.update_input_buffers_batched(state, [760] * B, [i] * B)
        am = cb.forward_batch_tp_inner(state); ttnn.deallocate(am)
    ttnn.synchronize_device(state.mesh)
    cb.update_input_buffers_batched(state, [760] * B, [2] * B)
    tid = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    am = cb.forward_batch_tp_inner(state)
    ttnn.end_trace_capture(state.mesh, tid, cq_id=0)
    for _ in range(warmup):
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    t0 = _t.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(state.mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(state.mesh)
    ms = (_t.perf_counter() - t0) / steps * 1000.0
    ttnn.release_trace(state.mesh, tid)
    ttnn.deallocate(am)
    for li in list(state.cb_kv.keys()):
        for k in ('kc', 'vc'):
            try: ttnn.deallocate(state.cb_kv[li][k])
            except Exception: pass
    state.cb_dn_skip = set(); state.cb_skip_blocks = set()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--blocks-per-seq", type=int, default=8)
    ap.add_argument("--owned-gdn", action="store_true")
    args = ap.parse_args()

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    state.cb_dn_recurrence_mode = "owned_gdn" if args.owned_gdn else "manual"
    log(f"DN recurrence: {state.cb_dn_recurrence_mode}")

    B = args.batch
    variants = [("full", []), ("-conv", ['conv']), ("-qknorm", ['qknorm']),
                ("-recur", ['recur']), ("-outgate", ['outgate'])]
    row = {}
    log(f"=== DN-only (48 layers) sub-op timing @ B={B} (steps={args.steps}) ===")
    for name, skip in variants:
        ms = time_dn(state, B, skip, args.steps, args.warmup, args.blocks_per_seq)
        row[name] = ms
        log(f"  {name:9s}: execute {ms:7.2f} ms")
    full = row['full']
    log(f"=== attribution (full − skip_X = X cost across 48 DN layers @ B={B}) ===")
    for name in ('-conv', '-qknorm', '-recur', '-outgate'):
        cost = full - row[name]
        log(f"  {name[1:]:8s}: {cost:7.2f} ms  ({100*cost/full:4.1f}% of DN-only full {full:.1f})")
    log("  the largest is the next vector-op fusion target (conv1d expected).")


if __name__ == "__main__":
    main()
