#!/usr/bin/env python3
"""Tracy-instrumented profile of one CB engine step in sampling mode.

Bootstraps server_tp + CBEngine(sampling=True), captures the logits trace,
warms up, signposts `cb_perf_start`, replays N steady-state steps, signposts
`cb_perf_end`. The signposted region is what tt-perf-report analyses for
per-op device times + DRAM GB/s + utilisation vs the P150 ceiling.

Two key answers we want from this:
  1. At B=N, what's the host-loop share vs device share of the step?
     (Settles whether on-device top-k is worth shipping first.)
  2. Which kernels dominate the step? What % of the 512 GB/s P150 ceiling do
     they hit? (Points at the next perf increment.)

Run on qb1 — the wrapper script handles the env block + Tracy launch:

  cd ~/tt-xla && tt-smi -r 0,1,2,3 && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$HOME/tenstorrent/tt-metal/build_tracy \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/lib:$TT_BUILD_DIR/ttnn \\
    .venv/bin/python -m tracy -r -p -v \\
      -o .cache/perf_logs/tracy_cb_sampling \\
      experiments/cb/profile/tracy_cb_step.py --slots 4 --steps 50

After it lands, analyse:
  ~/.local/bin/tt-perf-report .cache/perf_logs/tracy_cb_sampling/ops_perf_results_*.csv \\
                              --csv .cache/perf_logs/cb_perf_report.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "experiments" / "cb").is_dir())
sys.path.insert(0, str(_PROJECT / "experiments" / "cb"))
sys.path.insert(0, str(_PROJECT / "experiments" / "serve"))

import tracy  # noqa: E402  (must import before server_tp pulls in ttnn)

from _runner import bootstrap_27b_cb, log  # noqa: E402
from cb_engine import CBEngine               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=4, help="non-signposted warmup steps")
    ap.add_argument("--steps", type=int, default=50, help="signposted steady-state steps")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt", default="Tell me a fun fact about silicon.")
    args = ap.parse_args()

    log("bootstrap server_tp…")
    state, _ = bootstrap_27b_cb()

    eos_id = getattr(state.tok, "eos_token_id", None)
    eos_id = int(eos_id) if eos_id is not None else -1
    prompt_ids = state.tok.encode(args.prompt)

    log(f"start CBEngine: slots={args.slots}, sampling=True")
    engine = CBEngine(state, slots=args.slots, max_new_cap=args.steps + args.warmup + 8,
                      eos_id=eos_id, sampling=True).start()

    # Submit `slots` requests so every slot is active each step.
    sampling = {"temperature": args.temperature, "top_p": 0.95, "top_k": 0, "seed": 0}
    handles = [engine.submit(prompt_ids, max_new=args.steps + args.warmup + 4,
                              sampling=sampling) for _ in range(args.slots)]
    log(f"submitted {args.slots} requests; warming up {args.warmup} steps…")

    consumed = [0] * args.slots
    def drain(n_target_per_slot: int):
        """Block until every slot has emitted at least n_target_per_slot tokens."""
        while True:
            ready = True
            for i, h in enumerate(handles):
                while consumed[i] < n_target_per_slot:
                    try:
                        kind, payload = h._q.get(timeout=30.0)
                    except Exception:
                        ready = False
                        break
                    if kind == "tok":
                        consumed[i] += 1
                    else:
                        return  # request terminated early
                if not ready:
                    break
            if ready:
                return

    drain(args.warmup)
    log(f"warmup done; signposting `cb_perf_start` and running {args.steps} steady-state steps…")
    tracy.signpost("cb_perf_start")
    drain(args.warmup + args.steps)
    tracy.signpost("cb_perf_end")
    log(f"signposted region done ({args.steps} steps × B={args.slots})")

    engine.stop()
    log("engine stopped")


if __name__ == "__main__":
    main()
