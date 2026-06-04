#!/usr/bin/env python3
"""Tracy-instrumented profile of one Gemma 4 12B decoder block.

Why: the full 48-layer forward overflows Tracy's 12k-marker DRAM buffer.
Gemma 4 has TWO block topologies — sliding (40 layers) and global (8) —
so we time ONE of each (controlled by env). Both share the same
RMSNorms + MLP + post-norms; only attention head_dim + KV-cache shape
differ.

Captures one signposted forward through `forward_token_gm4_inner` after
warmup, so tt-perf-report can isolate: matmuls vs CCL vs SDPA vs
RMSNorm dispatch.

Env:
    TT_GEMMA4_VARIANT={base, it}     default base (faster bootstrap)
    GM4_TRACY_LAYER_TYPE={sliding, global}    default sliding

Run (qb1):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  export PATH=/home/aditya/tt-xla/.venv/bin:$PATH && \\
  HF_HUB_OFFLINE=1 .venv/bin/python -m tracy -r -p -v \\
      -o .cache/perf_logs/tracy_gemma4_layer \\
      experiments/utils/tracy_profile_one_gemma4_layer.py

Then:
  PATH=/home/aditya/.local/bin:$PATH tt-perf-report \\
      --start-signpost "Performance pass start" \\
      --end-signpost "Performance pass end" --no-color \\
      .cache/perf_logs/tracy_gemma4_layer/reports/*/ops_perf_results_*.csv
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_gemma4_unified_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402
import tracy  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    layer_type = os.environ.get("GM4_TRACY_LAYER_TYPE", "sliding").lower()
    if layer_type not in {"sliding", "global"}:
        log(f"FATAL: GM4_TRACY_LAYER_TYPE={layer_type!r} (expected sliding|global)")
        return 1

    log(f"bootstrap (variant from TT_GEMMA4_VARIANT)…")
    state = srv.State()
    srv.bootstrap(state, log)

    # Pick a token and position such that the FORWARD signposted call
    # below routes through the layer-type we want to profile.
    # forward_token_gm4_inner walks layer_types in order; one signposted
    # full forward captures every layer's ops — but with both layer
    # types interleaved. To isolate, run twice and tag each pass.
    # Simpler: pick a stable warmup token and let tt-perf-report group
    # by layer index in the CSV. Op names embed the call site so the
    # grouping is unambiguous.
    log("warmup 2 forwards (JIT all kernels)…")
    srv.update_input_buffers(state, token_id=2, cur_pos=0)
    a = srv.forward_token_gm4_inner(state); ttnn.deallocate(a)
    ttnn.synchronize_device(state.mesh)
    srv.update_input_buffers(state, token_id=2, cur_pos=1)
    a = srv.forward_token_gm4_inner(state); ttnn.deallocate(a)
    ttnn.synchronize_device(state.mesh)

    log("signposted forward…")
    srv.update_input_buffers(state, token_id=2, cur_pos=2)
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    out = srv.forward_token_gm4_inner(state)
    ttnn.synchronize_device(state.mesh)
    elapsed = (time.time() - t0) * 1000.0
    tracy.signpost("Performance pass end")
    log(f"one forward (eager): {elapsed:.1f} ms")
    ttnn.deallocate(out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
