#!/usr/bin/env python3
"""Tracy-instrumented profile of one DN forward call (post task 64 fusion).

Mirrors experiments/utils/tracy_profile_one_moe.py — same harness, but
profiles dn_forward_ttnn instead of moe_forward_ttnn_pattern_a_batched.

DN runs on 30 of 40 layers in 35B-A3B → it's the biggest pool by op-count.
We need per-op timing on the *current* DN forward (with the task 64
in_proj fusion + owned_gdn + owned_decay_gate already on) to decide
what to fuse next.

Two probes:
  - moe — already exists (tracy_profile_one_moe.py)
  - dn  — this file

Profiling the full traced step overflows Tracy's 12000-marker DRAM
buffer; one block at a time fits and the shape repeats across all
layers of the same type, so per-block stats are representative.

Run on qb1 (Tracy capture wrapper):
  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
  export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \\
  export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \\
  export ARCH_NAME=blackhole && \\
  export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \\
  export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \\
  export PATH=/home/aditya/tt-xla/.venv/bin:$PATH && \\
  .venv/bin/python -m tracy -r -p -v -o .cache/perf_logs/tracy_one_dn \\
      experiments/utils/tracy_profile_one_dn.py

After completion the CSV is at
.cache/perf_logs/tracy_one_dn/reports/*/ops_perf_results_*.csv —
feed to tt-perf-report or analyze with experiments/utils/analyze_ops_perf_results.py.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402
import tracy  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap (moe_mode=pattern_a_batched, dn_owned_gdn=True, dn_owned_decay_gate=True)…")
    state = srv.State()
    state.moe_mode = "pattern_a_batched"
    # Owned kernels are on by default (set in State.__init__) — verify.
    assert state.dn_owned_gdn, "dn_owned_gdn must be True for the production profile"
    assert state.dn_owned_decay_gate, "dn_owned_decay_gate must be True for the production profile"
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    # Synthesize input h: shape [1, HIDDEN] replicated. Use a post-rms-norm
    # scale (~5.0) so the recurrence + matmul magnitudes are realistic.
    rng = np.random.default_rng(0)
    h_np = rng.normal(0, 5.0, size=(1, srv.HIDDEN)).astype(np.float32)
    h_tt = srv.np_to_replicated(h_np, state.mesh)
    log(f"h_tt shape={list(h_tt.shape)} dtype={h_tt.dtype}")

    # Pick layer 0 (a linear_attention layer per 35B-A3B's mix). dn_state is
    # already initialized by reset_caches_ttnn().
    layer_idx = 0
    assert state.layer_types[layer_idx] == "linear_attention", \
        f"expected linear_attention at layer 0; got {state.layer_types[layer_idx]}"
    dn_state = state.dn_caches_tt[layer_idx]  # (conv_state_tt, recurrent_state_tt)
    log(f"layer_idx={layer_idx} type=linear_attention dn_state set")

    # Warmup to populate JIT cache. dn_forward_ttnn mutates state; we use
    # the same state across warmup + the signposted call (same as eager).
    log("warmup 2 DN calls…")
    for _ in range(2):
        out, new_conv, new_rec = srv.dn_forward_ttnn(
            h_tt, state.per_layer_tt[layer_idx], state.mesh, dn_state,
            use_owned_gdn=True, use_owned_decay_gate=True,
        )
        ttnn.deallocate(out)
        # Thread the new state forward.
        ttnn.deallocate(dn_state[0]); ttnn.deallocate(dn_state[1])
        dn_state = (new_conv, new_rec)
    ttnn.synchronize_device(state.mesh)

    log("signposted DN call…")
    ttnn.synchronize_device(state.mesh)
    tracy.signpost("Performance pass start")
    t0 = time.time()
    out, new_conv, new_rec = srv.dn_forward_ttnn(
        h_tt, state.per_layer_tt[layer_idx], state.mesh, dn_state,
        use_owned_gdn=True, use_owned_decay_gate=True,
    )
    ttnn.synchronize_device(state.mesh)
    elapsed = (time.time() - t0) * 1000.0
    tracy.signpost("Performance pass end")
    log(f"one DN call: {elapsed:.2f} ms")

    ttnn.deallocate(out); ttnn.deallocate(new_conv); ttnn.deallocate(new_rec)
    ttnn.deallocate(dn_state[0]); ttnn.deallocate(dn_state[1])
    ttnn.deallocate(h_tt)


if __name__ == "__main__":
    main()
