#!/usr/bin/env python3
"""CB5 — per-block compute attribution of the traced batched decode.

CB4 measured traced step_ms ≈ 73 + 4.3·B (memory-bound 73 ms floor + 4.3 ms/seq
compute → ~232 tok/s aggregate ceiling). To raise the ceiling we must cut the
4.3 ms/seq, but FIRST we attribute it (profile-driven non-negotiable): which
block — 48 DeltaNet layers (manual recurrence), 16 attention layers, or the 64
MLPs — dominates the per-token compute, and which scales most with B?

Method: capture the SAME traced forward with one block type no-op'd
(cb_skip_blocks) and time execute_trace. full − skip_X = block X's traced cost.
Done at B=1 and B=32; the block whose cost grows most B=1→B=32 owns the 4.3 ms/seq
slope and is the kernel target.

Skipping a block makes the residual stream wrong (we don't read outputs) — this
is a TIMING ablation only.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb_profile_blocks.py --batches 1,32 --steps 50
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


def time_trace(state, B, skip, steps, warmup, blocks_per_seq):
    """Capture forward with `skip` blocks no-op'd; return mean execute_trace ms."""
    import ttnn
    import time as _t
    state.cb_skip_blocks = set(skip)
    cb.setup_cb_state(state, B, blocks_per_seq=blocks_per_seq)
    cb.cb_reset_states(state)
    for i in range(2):  # JIT warmup
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
    state.cb_skip_blocks = set()
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--blocks-per-seq", type=int, default=8)
    ap.add_argument("--owned-gdn", action="store_true",
                    help="use the batched owned_gdn DN recurrence kernel")
    args = ap.parse_args()
    batches = [int(x) for x in args.batches.split(",")]

    log("bootstrap production 27B server (server_tp)…")
    state = base.MeshServerState() if hasattr(base, "MeshServerState") else base.State()
    base.bootstrap(state)
    state.deltanet_recurrence_mode = "manual"
    state.deltanet_decay_gate_mode = "manual"
    state.deltanet_decay_mode = "native_softplus"
    state.cb_dn_recurrence_mode = "owned_gdn" if args.owned_gdn else "manual"
    log(f"DN recurrence: {state.cb_dn_recurrence_mode}")

    variants = [("full", []), ("-DN", ['dn']), ("-ATT", ['attn']), ("-MLP", ['mlp'])]
    results = {}
    for B in batches:
        log(f"=== B={B} per-block trace timing (steps={args.steps}) ===")
        row = {}
        for name, skip in variants:
            ms = time_trace(state, B, skip, args.steps, args.warmup, args.blocks_per_seq)
            row[name] = ms
            log(f"  {name:5s}: execute {ms:7.2f} ms")
        full = row['full']
        dn = full - row['-DN']; attn = full - row['-ATT']; mlp = full - row['-MLP']
        rest = full - dn - attn - mlp  # embed + final norm + lm_head + collectives
        log(f"  attribution @ B={B}: DN={dn:6.2f}  ATT={attn:6.2f}  MLP={mlp:6.2f}  "
            f"rest={rest:6.2f}  (sum {dn+attn+mlp+rest:.2f} vs full {full:.2f})")
        results[B] = dict(full=full, DN=dn, ATT=attn, MLP=mlp, rest=rest)

    if len(batches) >= 2:
        b0, b1 = batches[0], batches[-1]
        log(f"=== scaling {b0}→{b1} (Δms, and Δms/seq = compute slope) ===")
        for k in ('full', 'DN', 'ATT', 'MLP', 'rest'):
            d = results[b1][k] - results[b0][k]
            per = d / (b1 - b0)
            log(f"  {k:5s}: Δ={d:7.2f} ms  → {per:5.2f} ms/seq")
        log("  the block with the largest ms/seq owns the throughput ceiling → kernel target")


if __name__ == "__main__":
    main()
