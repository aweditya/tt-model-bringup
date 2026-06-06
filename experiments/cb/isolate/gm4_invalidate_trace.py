#!/usr/bin/env python3
"""Clear `state.trace_id` so the next `ensure_decode_trace(state)` call
recaptures with the current code. Use after editing+reloading the inner
forward via the dev harness.

Why this exists: round-3 perf chase needed to A/B kernel-time deltas
across rapid `_compute_rope_for_forward` and other inner-forward edits.
The harness `_reload` swaps the module dict in place but leaves
`state.trace_id` set, so `ensure_decode_trace` short-circuits and the
v04 validator measures the STALE trace. Forks the "single-purpose
isolation probe" pattern at experiments/cb/isolate/gm4_*.py:1.

Trigger via the dev harness:
  ssh qb2 'touch tt-xla/.cache/gm4_runtime/trig/invalidate_trace'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None):
    if state is None:
        log("ERROR: need to be triggered through the dev harness so state is alive")
        return 1
    import ttnn  # noqa
    old_id = getattr(state, "trace_id", None)
    if old_id is None:
        log("state.trace_id already None — nothing to release")
        return 0
    log(f"releasing trace id {old_id}…")
    try:
        ttnn.release_trace(state.mesh, old_id)
    except Exception as e:
        log(f"  ttnn.release_trace raised: {e!r} (continuing — may have already been freed)")
    state.trace_id = None
    # Also drop the captured output handle so we recapture cleanly.
    if getattr(state, "traced_argmax_tt", None) is not None:
        try:
            ttnn.deallocate(state.traced_argmax_tt)
        except Exception as e:
            log(f"  deallocate(traced_argmax_tt) raised: {e!r}")
        state.traced_argmax_tt = None
    log("invalidated — next ensure_decode_trace will recapture")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
