#!/usr/bin/env python3
"""Tracy-instrumented profile of the 35B-A3B traced decode step.

Goal: get per-op timing breakdown (device us, op-to-op gap, DRAM GB/s,
"Bound" classification) so we can tell whether the 5120 small matmuls
per token are dispatch-bound or BW-bound. Decides whether the batched
expert matmul is the right next optimization.

Setup:
  - Bootstrap pattern_a_batched (the production trace-clean MoE path)
  - Capture trace once
  - Run a few warmup execute_trace iterations (unsignposted)
  - tracy.signpost("Performance pass start")
  - Run N=10 steady-state execute_trace iterations (signposted region)
  - tracy.signpost("Performance pass end")
  - The signposted region is what tt-perf-report will analyze

Run on qb1 (Tracy capture wraps the python invocation):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  .venv/bin/python -m tracy -r -p -v \\
      -o .cache/perf_logs/tracy_traced_decode \\
      experiments/utils/tracy_profile_traced_decode.py

After it completes look for the CSV at
.cache/perf_logs/tracy_traced_decode/ops_perf_results_*.csv and feed to
tt-perf-report:
  pipx install tt-perf-report
  tt-perf-report .cache/perf_logs/tracy_traced_decode/ops_perf_results_*.csv \\
                 --csv .cache/perf_logs/tt_perf_report_35b.csv
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402
import tracy  # noqa: E402 — provided by tt-metal


N_WARMUP_EAGER = 3     # eager forwards so JIT cache is populated
N_PERF_ITERS = 1       # ONE signposted eager forward — gives per-op times.
                       # Trace REPLAY doesn't issue ttnn ops (pre-recorded),
                       # so signposting trace execute gives Tracy 0 ops to see.


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap (moe_mode=pattern_a_batched, the production trace path)…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]

    # Eager warmup so JIT kernels are compiled, then reset state.
    log(f"eager warmup {N_WARMUP_EAGER} forwards…")
    for i in range(N_WARMUP_EAGER):
        srv.step_forward_ttnn(state, tok_id, i)
    state.reset_caches_ttnn()
    ttnn.synchronize_device(state.mesh)

    # Signposted eager forward — Tracy captures every ttnn op dispatched
    # during this region, giving tt-perf-report per-op times.
    log(f"signposted eager forward (1 full decode step)…")
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    next_id = srv.step_forward_ttnn(state, tok_id, 0)
    ttnn.synchronize_device(state.mesh)
    iter_ms = [(time.time() - t0) * 1000.0]
    tracy.signpost("Performance pass end")
    log(f"  eager forward: {iter_ms[0]:.1f} ms, next_id={next_id}")
    log("Done — Tracy CSV should be written under the -o output folder.")


if __name__ == "__main__":
    main()
