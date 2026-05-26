#!/usr/bin/env python3
"""B16-profile — measure prefill and decode latency for server_35b_ttnn.

Method:
  - bootstrap (one-time, ~107s)
  - warmup: run 3 throwaway forward steps (compiles JIT kernels)
  - prefill: time each token of the prompt sequentially; collect per-token ms
  - decode: greedy generate N tokens; collect per-token ms
  - report: min, median, p95, max ms, and tok/s for both

Establishes the "no trace, no owned kernels" baseline. Comparison points
in memory:
  - numpy server_35b.py (B14): 3.44 tok/s
  - 27B production server_tp.py: 12.93 tok/s (with trace + owned kernels)
  - 27B pre-trace baseline (memory note): ~200 ms/tok (~5 tok/s)

This tells us: (a) where 35B baseline sits, (b) headroom from B17 trace
capture, (c) headroom from B18 owned kernels.

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \
    export ARCH_NAME=blackhole && \
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \
    .venv/bin/python -u experiments/utils/profile_35b_ttnn.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

PROMPT = "The capital of France is"
WARMUP_STEPS = 3
DECODE_STEPS_DEFAULT = 24


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(label, times_ms):
    arr = np.array(times_ms)
    log(f"  {label}: n={len(arr)} "
        f"min={arr.min():.1f} med={np.median(arr):.1f} "
        f"p95={np.percentile(arr, 95):.1f} max={arr.max():.1f} ms/tok  "
        f"throughput={1000.0 / np.median(arr):.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moe-mode", choices=["topk", "pattern_a_batched"], default="topk",
                    help="topk = host-readback A/B reference (trace-incompatible); "
                         "pattern_a_batched = batched on-device MoE (trace-clean, production).")
    ap.add_argument("--decode-steps", type=int, default=DECODE_STEPS_DEFAULT)
    args = ap.parse_args()

    log(f"bootstrap (moe_mode={args.moe_mode})…")
    t0 = time.time()
    state = srv.State()
    state.moe_mode = args.moe_mode  # set BEFORE bootstrap — controls upload layout
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()
    bootstrap_s = time.time() - t0
    log(f"bootstrap took {bootstrap_s:.1f} s")

    prompt_ids = state.tokenizer.encode(PROMPT)
    log(f"\nprompt: {PROMPT!r} → ids {prompt_ids} (len {len(prompt_ids)})")

    # Warmup: throwaway forward steps (compiles JIT, fills caches with prompt)
    log(f"\nwarmup {WARMUP_STEPS} forwards (uses first prompt tokens; not timed)…")
    for p in range(WARMUP_STEPS):
        tid = int(prompt_ids[p % len(prompt_ids)])
        srv.step_forward_ttnn(state, tid, p)
    log("warmup done.")

    # Reset state caches AFTER warmup so prefill timing starts clean
    state.reset_caches_ttnn()

    # ── PREFILL ──────────────────────────────────────────────────────
    log(f"\nprefill: teacher-force {len(prompt_ids)} prompt tokens sequentially…")
    prefill_ms = []
    last_argmax = None
    for p, tid in enumerate(prompt_ids):
        t = time.time()
        last_argmax = srv.step_forward_ttnn(state, int(tid), p)
        prefill_ms.append((time.time() - t) * 1000.0)
        log(f"  pos {p} tok {int(tid):>6d} → next id {last_argmax:>6d} "
            f"({prefill_ms[-1]:.1f} ms)")
    stats("prefill (per-token)", prefill_ms)

    # ── DECODE ────────────────────────────────────────────────────────
    log(f"\ndecode: greedy generate {args.decode_steps} tokens autoregressively…")
    decode_ms = []
    pos = len(prompt_ids)
    cur = last_argmax
    generated = [cur]
    for step in range(args.decode_steps - 1):
        t = time.time()
        next_id = srv.step_forward_ttnn(state, cur, pos)
        decode_ms.append((time.time() - t) * 1000.0)
        generated.append(next_id)
        cur = next_id
        pos += 1
    stats("decode (per-token)", decode_ms)

    log(f"\ngenerated text: {state.tokenizer.decode(prompt_ids + generated)!r}")

    # Combined summary
    log("\n=== summary ===")
    log(f"  bootstrap:       {bootstrap_s:.1f} s (one-time)")
    log(f"  prefill median:  {np.median(prefill_ms):.1f} ms/tok  "
        f"({1000.0 / np.median(prefill_ms):.2f} tok/s)")
    log(f"  decode  median:  {np.median(decode_ms):.1f} ms/tok  "
        f"({1000.0 / np.median(decode_ms):.2f} tok/s)")
    log(f"  comparison: numpy server_35b.py 3.44 tok/s | "
        f"27B production 12.93 tok/s (traced + owned kernels)")


if __name__ == "__main__":
    main()
