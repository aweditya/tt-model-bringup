#!/usr/bin/env python3
"""Sync-bounded total-time bench for production dn_forward_ttnn on (1,4).

Calls server_35b_ttnn.dn_forward_ttnn directly (the production function)
with use_owned_gdn=True, use_owned_decay_gate=True, in_proj_combined
already fused. Reports mean / median / std over N iters after warmup.

This is the baseline budget for the workflow's Step 1: how much DN time
is there to optimize per call, and what's the ceiling impact on the
143.6 ms/tok number?

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/utils/bench_dn_total.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--layer-idx", type=int, default=0)
    args = ap.parse_args()

    log("bootstrap (production state)…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    layer_idx = args.layer_idx
    assert state.layer_types[layer_idx] == "linear_attention", \
        f"expected linear_attention at layer {layer_idx}; got {state.layer_types[layer_idx]}"

    log("synthesizing h…")
    rng = np.random.default_rng(0)
    h_np = rng.normal(0, 5.0, size=(1, srv.HIDDEN)).astype(np.float32)
    h_tt = srv.np_to_replicated(h_np, state.mesh)

    w = state.per_layer_tt[layer_idx]
    dn_state = state.dn_caches_tt[layer_idx]

    import ttnn

    log(f"warmup x{args.n_warmup}…")
    for _ in range(args.n_warmup):
        out, _, _ = srv.dn_forward_ttnn(
            h_tt, w, state.mesh, dn_state,
            use_owned_gdn=True, use_owned_decay_gate=True,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(state.mesh)

    log(f"timed x{args.n_iters} (eager, sync-bounded)…")
    ts = []
    for _ in range(args.n_iters):
        ttnn.synchronize_device(state.mesh)
        t0 = time.perf_counter()
        out, _, _ = srv.dn_forward_ttnn(
            h_tt, w, state.mesh, dn_state,
            use_owned_gdn=True, use_owned_decay_gate=True,
        )
        ttnn.synchronize_device(state.mesh)
        ts.append((time.perf_counter() - t0) * 1000.0)
        ttnn.deallocate(out)

    ts = np.array(ts)
    log("")
    log(f"=== DN total time, eager, layer {layer_idx} ===")
    log(f"  mean   {ts.mean():7.3f} ms")
    log(f"  median {np.median(ts):7.3f} ms")
    log(f"  min    {ts.min():7.3f} ms")
    log(f"  max    {ts.max():7.3f} ms")
    log(f"  std    {ts.std():7.3f} ms")
    log(f"  per-token (x30 DN layers) -> {ts.mean() * 30:.1f} ms/tok (eager)")
    log(f"  baseline 143.6 ms/tok is trace-mode; this is the eager upper bound")

    ttnn.deallocate(h_tt)
    ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    main()
