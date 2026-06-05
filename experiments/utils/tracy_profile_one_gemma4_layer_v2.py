#!/usr/bin/env python3
"""v2 Tracy probe for Gemma 4: single forward, no warmup.

Why: the v1 probe (`tracy_profile_one_gemma4_layer.py`) ran 2 warmup
forwards before the signposted forward to fill kernel JIT caches. That
overflowed the 12k-marker DRAM buffer mid-run so per-op DEVICE KERNEL
DURATION came back zero (the host timestamps were still intact but
unusable for tt-perf-report).

This v2 skips warmup entirely. The first forward is slow due to JIT
compile (~30-60s) but each device kernel still has correct cycle counts
in the markers. Total ops captured stays well below 12k for one forward
(~3.8k device ops per forward × tile/device-id duplication < 12k after
the per-core multiplication).

If marker overflow still happens, drop to 1 sliding layer + lm_head by
truncating NUM_LAYERS via env GM4_NUM_LAYERS_OVERRIDE (not implemented
here — would need server code knob).

Env:
    TT_GEMMA4_VARIANT={base, it}     default base (faster bootstrap)

Run (qb2):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  export PATH=/home/aditya/tt-xla/.venv/bin:$PATH && \\
  HF_HUB_OFFLINE=1 TT_GEMMA4_VARIANT=it .venv/bin/python -m tracy -r -p -v \\
      -o .cache/perf_logs/tracy_gemma4_v2 \\
      experiments/utils/tracy_profile_one_gemma4_layer_v2.py

Then:
  PATH=/home/aditya/.local/bin:$PATH tt-perf-report \\
      --start-signpost "Performance pass start" \\
      --end-signpost "Performance pass end" --no-color \\
      .cache/perf_logs/tracy_gemma4_v2/reports/*/ops_perf_results_*.csv

Forks `tracy_profile_one_gemma4_layer.py:36-93` (same bootstrap +
signpost pattern); removed warmup loop, added explicit comment.
"""
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
    log("bootstrap (variant from TT_GEMMA4_VARIANT)…")
    state = srv.State()
    srv.bootstrap(state, log)

    # No warmup — first signposted forward will JIT-compile every kernel.
    # That's OK because tt-perf-report reads DEVICE KERNEL DURATION
    # markers (per-op cycle counts), not wall-clock; cold compile only
    # affects host time. The signposts bracket exactly ONE forward so
    # we capture all device kernel times without marker-buffer overflow.
    log("signposted cold forward (JIT-compiles ~30-60s)…")
    srv.update_input_buffers(state, token_id=2, cur_pos=0)
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    out = srv.forward_token_gm4_inner(state)
    ttnn.synchronize_device(state.mesh)
    elapsed = (time.time() - t0) * 1000.0
    tracy.signpost("Performance pass end")
    log(f"one forward (cold/JIT): {elapsed:.1f} ms")
    ttnn.deallocate(out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
