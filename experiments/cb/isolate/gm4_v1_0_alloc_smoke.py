#!/usr/bin/env python3
"""v1.0 smoke — setup_cb_state allocates without OOM at B=2 and B=4.

Quickest CB validation: just exercise the allocator and the
host→device buffer plumbing. Doesn't run a forward yet (v1.1+ does).

Run via harness:
  ssh qb1 'touch tt-xla/.cache/gm4_runtime/trig/v1_0_alloc_smoke'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as base  # noqa: E402
import server_gemma4_unified_cb as cb      # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    owned_state = state is None
    if owned_state:
        log("bootstrapping Gemma 4 12B (~80s)…")
        state = base.State()
        base.bootstrap(state, log=log)
    else:
        log("using pre-bootstrapped state from harness")

    for B in [2, 4]:
        log(f"--- setup_cb_state(B={B}) ---")
        t0 = time.time()
        cb.setup_cb_state(state, B=B)
        log(f"  alloc: {time.time()-t0:.1f}s")
        log(f"  cb_B={state.cb_B}, blocks/seq={state.cb_blocks_per_seq}, "
            f"total_blocks={state.cb_total_blocks}")
        log(f"  cb_kv_caches_tt: {len(state.cb_kv_caches_tt)} layers "
            f"(sliding caches: {sum(len(c) for c in state.cb_kv_caches_tt)})")
        cb.update_input_buffers_batched(
            state, token_ids=[2] * B, cur_positions=[0] * B)
        log(f"  ✓ update_input_buffers_batched(B={B}) OK")
        # Free the caches to avoid double-allocation when the loop iterates.
        # (The harness keeps state alive; if we don't free, the next B will
        # OOM on top of the previous.)
        for layer_caches in state.cb_kv_caches_tt:
            for (kc, vc) in layer_caches:
                import ttnn
                ttnn.deallocate(kc)
                ttnn.deallocate(vc)
        state.cb_kv_caches_tt = []

    log("VERDICT: PASS")

    if owned_state:
        import ttnn
        ttnn.close_device(state.mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
