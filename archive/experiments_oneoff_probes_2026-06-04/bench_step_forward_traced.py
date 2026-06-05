#!/usr/bin/env python3
"""Trace-mode bench of step_forward_ttnn for paired A/B comparisons.

Captures the production step_forward_inner in a ttnn trace, then times
execute_trace across N iterations. Reports mean/median ms/tok and the
sequence of generated tokens.

Designed for paired A/B: invoke twice with --fused-qk-norm true and false,
diff the two outputs.

Run on qb1:
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/bench_step_forward_traced.py --fused-qk-norm true
  (then again with --fused-qk-norm false)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_bool(s):
    s = s.lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"bool flag: {s!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused-qk-norm", type=parse_bool, required=True,
                    help="Override state.dn_fused_qk_norm (true/false).")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--n-warmup", type=int, default=3,
                    help="Pre-trace warmup decode steps (JIT cache populate).")
    args = ap.parse_args()

    log(f"=== trace bench fused_qk_norm={args.fused_qk_norm} ===")
    log("bootstrap…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    state.dn_fused_qk_norm = args.fused_qk_norm
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    # If we forced fused_qk_norm=False AFTER bootstrap allocated the
    # weight, the layer_forward dispatcher won't pass the weight — that's
    # the intended A/B semantics.

    log(f"prompt: {args.prompt!r}")
    prompt_ids = state.tokenizer.encode(args.prompt)
    log(f"prompt_ids = {prompt_ids}")

    import ttnn

    # Prefill the prompt eagerly (teacher-force) so the DN/KV caches are at
    # the production state when we enter decode.
    log("prefill prompt eagerly…")
    last_argmax = None
    for p, tid in enumerate(prompt_ids):
        last_argmax = srv.step_forward_ttnn(state, tid, p)
    log(f"  prefill done. first decoded next_id={last_argmax} text={state.tokenizer.decode([last_argmax])!r}")

    # Warmup eager decode (so JIT cache for decode path is populated).
    log(f"warmup eager decode x{args.n_warmup}…")
    pos = len(prompt_ids)
    cur = last_argmax
    for _ in range(args.n_warmup):
        cur = srv.step_forward_ttnn(state, cur, pos)
        pos += 1
    ttnn.synchronize_device(state.mesh)

    # Capture trace of step_forward_inner. Trace requires reading from
    # pre-allocated buffers only (no Python-side branching). state.tok_buf
    # and state.rot_idxs_buf are written OUTSIDE the trace via
    # update_input_buffers each step.
    log("capture trace…")
    # Seed input buffers for capture.
    srv.update_input_buffers(state, cur, pos)
    trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    argmax_trace_tt = srv.step_forward_inner(state)
    ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
    log("  trace captured.")

    # Time N execute_trace iterations + autoregressive decode.
    log(f"timed decode x{args.max_new} (execute_trace + readback)…")
    generated = [cur]
    ts = []
    for step in range(args.max_new):
        srv.update_input_buffers(state, generated[-1], pos)
        ttnn.synchronize_device(state.mesh)
        t0 = time.perf_counter()
        ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(state.mesh)
        ts.append((time.perf_counter() - t0) * 1000.0)
        # Readback OUTSIDE trace.
        next_id_t = ttnn.to_torch(
            argmax_trace_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )
        next_id = int(next_id_t.flatten()[0].item())
        generated.append(next_id)
        pos += 1

    ts = np.array(ts)
    text = state.tokenizer.decode(prompt_ids + generated)

    log("")
    log(f"=== results fused_qk_norm={args.fused_qk_norm} ===")
    log(f"  execute_trace ms: mean {ts.mean():7.3f}  median {np.median(ts):7.3f}  "
        f"min {ts.min():.3f}  max {ts.max():.3f}  std {ts.std():.3f}")
    log(f"  per-token: {ts.mean():.2f} ms/tok TRACE")
    log("  generated text:")
    log(f"  {text!r}")
    log(f"  token sequence (first {len(generated)}):")
    log(f"  {generated}")

    ttnn.release_trace(state.mesh, trace_id)
    ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    main()
