#!/usr/bin/env python3
"""CB throughput bench — eager ms/step + aggregate tok/s at B=1,8,32.

The whole point of continuous batching: decode at batch=1 is memory-bound (the
per-token matmul [1,K]x[K,N] is dominated by streaming the weight from DRAM).
At batch=B the SAME weight bytes serve B tokens, so cost grows sub-linearly in
B and aggregate throughput scales until the matmul becomes compute-bound. This
bench measures that scaling directly on the batched forward (server_tp_cb).

EAGER numbers (per-op Python dispatch, not traced) — so absolute tok/s is below
the traced production figure, but the SCALING across B is dispatch-invariant
(dispatch is ~constant per step regardless of B) and is the signal we want
before investing in a B=32 trace (CB4).

Sync-bounded timing (feedback_sync_bounded_timing): synchronize before and
after the timed loop; warm up first to amortize kernel compile.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_bench_throughput.py --batches 1,8,32 --steps 30
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


def bench_one(state, B, steps, warmup, blocks_per_seq):
    import ttnn
    cb.setup_cb_state(state, B, blocks_per_seq=blocks_per_seq)
    cb.cb_reset_states(state)
    tok = 760  # arbitrary fixed token; throughput is shape-bound, not value-bound

    def step(pos):
        cb.update_input_buffers_batched(state, [tok] * B, [pos] * B)
        am = cb.forward_batch_tp_inner(state)
        ttnn.deallocate(am)

    for i in range(warmup):
        step(i)
    ttnn.synchronize_device(state.mesh)

    t0 = time.perf_counter()
    for i in range(steps):
        step(warmup + i)
    ttnn.synchronize_device(state.mesh)
    t1 = time.perf_counter()

    ms_step = (t1 - t0) / steps * 1000.0
    agg_tps = B * 1000.0 / ms_step
    # free the big per-B KV pool before the next B (avoid fragmentation/OOM)
    for li in list(state.cb_kv.keys()):
        for k in ('kc', 'vc'):
            try: ttnn.deallocate(state.cb_kv[li][k])
            except Exception: pass
    return ms_step, agg_tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--blocks-per-seq", type=int, default=8,
                    help="KV blocks/slot (8*32=256 tokens ctx). Small keeps the "
                         "B=32 pool modest; throughput is matmul-bound, not ctx-bound here.")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    # match the proven manual DN math (owned_gdn is B=1-only)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"

    log(f"=== eager batched decode throughput (steps={args.steps}, warmup={args.warmup}) ===")
    base_ms = None
    rows = []
    for B in batches:
        ms, tps = bench_one(state, B, args.steps, args.warmup, args.blocks_per_seq)
        if base_ms is None and B == 1:
            base_ms = ms
        eff = (base_ms / ms) if base_ms else float('nan')  # ms_B1/ms_B → per-step slowdown factor
        rows.append((B, ms, tps))
        log(f"  B={B:3d}: {ms:8.2f} ms/step   agg {tps:8.2f} tok/s"
            + (f"   (step is {ms/base_ms:.2f}x B=1; {tps/(1000.0/base_ms):.1f}x B=1 throughput)"
               if base_ms else ""))
    log("=== summary (eager; trace will raise absolute tok/s, scaling holds) ===")
    for B, ms, tps in rows:
        log(f"  B={B:3d}  {ms:8.2f} ms/step  {tps:8.2f} tok/s")


if __name__ == "__main__":
    main()
