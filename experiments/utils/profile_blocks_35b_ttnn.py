#!/usr/bin/env python3
"""B16-profile-blocks — per-block timing breakdown for server_35b_ttnn.

Wraps server_35b_ttnn.dn_forward_ttnn / attn_forward_ttnn / moe_forward_ttnn
with a timed shim, then runs a few decode steps. Aggregates total wall-time
per block type (DN vs attn vs MoE vs everything-else) and prints per-tok
breakdown so we know where the 484 ms/tok baseline is spent.

ttnn dispatches are asynchronous on the host side. To make the per-block
times meaningful we call `ttnn.synchronize_device(state.mesh)` before reading
the wall clock for each block's output. This trades a bit of throughput for
attribution accuracy — small price for a one-shot profile.

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=$HOME/tenstorrent/tt-metal && \
    export TT_BUILD_DIR=$TT_METAL_HOME/build_Release && \
    export ARCH_NAME=blackhole && \
    export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH && \
    export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH && \
    .venv/bin/python -u experiments/utils/profile_blocks_35b_ttnn.py
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402

PROMPT = "The capital of France is"
WARMUP_STEPS = 3
DECODE_STEPS = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Per-step accumulator (reset per token)
_step_times_ms = defaultdict(list)


def _sync(mesh):
    """Force any pending ttnn dispatches to complete for accurate wall-time."""
    ttnn.synchronize_device(mesh)


def install_block_timers(mesh):
    """Monkey-patch the 3 block forwards with timed wrappers."""
    orig_dn = srv.dn_forward_ttnn
    orig_attn = srv.attn_forward_ttnn
    orig_moe = srv.moe_forward_ttnn

    # Pass through **kwargs so additions like sub_capture / state /
    # kv_cache don't break the wrapper when block signatures evolve.
    def timed_dn(*args, **kwargs):
        _sync(mesh)
        t0 = time.time()
        out = orig_dn(*args, **kwargs)
        _sync(mesh)
        _step_times_ms["dn"].append((time.time() - t0) * 1000.0)
        return out

    def timed_attn(*args, **kwargs):
        _sync(mesh)
        t0 = time.time()
        out = orig_attn(*args, **kwargs)
        _sync(mesh)
        _step_times_ms["attn"].append((time.time() - t0) * 1000.0)
        return out

    def timed_moe(*args, **kwargs):
        _sync(mesh)
        t0 = time.time()
        out = orig_moe(*args, **kwargs)
        _sync(mesh)
        _step_times_ms["moe"].append((time.time() - t0) * 1000.0)
        return out

    srv.dn_forward_ttnn = timed_dn
    srv.attn_forward_ttnn = timed_attn
    srv.moe_forward_ttnn = timed_moe


def main():
    log("bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode(PROMPT)
    log(f"prompt ids: {prompt_ids}")

    log(f"warmup {WARMUP_STEPS} forwards (not timed)…")
    for p in range(WARMUP_STEPS):
        srv.step_forward_ttnn(state, int(prompt_ids[p % len(prompt_ids)]), p)
    state.reset_caches_ttnn()

    log("install block timers")
    install_block_timers(state.mesh)

    log("prefill 5 + decode 8 with sync-bounded per-block timing…")
    cur = None
    for p, tid in enumerate(prompt_ids):
        _step_times_ms.clear()
        t0 = time.time()
        cur = srv.step_forward_ttnn(state, int(tid), p)
        wall_ms = (time.time() - t0) * 1000.0
        dn_total = sum(_step_times_ms["dn"])
        attn_total = sum(_step_times_ms["attn"])
        moe_total = sum(_step_times_ms["moe"])
        other = wall_ms - (dn_total + attn_total + moe_total)
        log(f"  prefill pos {p}: total {wall_ms:.0f} ms  "
            f"dn={dn_total:.0f} ({len(_step_times_ms['dn'])}× = {dn_total / max(1, len(_step_times_ms['dn'])):.1f} ms/L)  "
            f"attn={attn_total:.0f} ({len(_step_times_ms['attn'])}× = {attn_total / max(1, len(_step_times_ms['attn'])):.1f} ms/L)  "
            f"moe={moe_total:.0f} ({len(_step_times_ms['moe'])}× = {moe_total / max(1, len(_step_times_ms['moe'])):.1f} ms/L)  "
            f"other={other:.0f}")

    pos = len(prompt_ids)
    for step in range(DECODE_STEPS):
        _step_times_ms.clear()
        t0 = time.time()
        cur = srv.step_forward_ttnn(state, cur, pos)
        wall_ms = (time.time() - t0) * 1000.0
        dn_total = sum(_step_times_ms["dn"])
        attn_total = sum(_step_times_ms["attn"])
        moe_total = sum(_step_times_ms["moe"])
        other = wall_ms - (dn_total + attn_total + moe_total)
        log(f"  decode  pos {pos}: total {wall_ms:.0f} ms  "
            f"dn={dn_total:.0f}  attn={attn_total:.0f}  moe={moe_total:.0f}  other={other:.0f}")
        pos += 1

    # Aggregate over the last few decode steps (post-prefill, post-warmup)
    log("\n=== aggregate (final decode step) ===")
    n_dn = len(_step_times_ms["dn"])
    n_attn = len(_step_times_ms["attn"])
    n_moe = len(_step_times_ms["moe"])
    dn_total = sum(_step_times_ms["dn"])
    attn_total = sum(_step_times_ms["attn"])
    moe_total = sum(_step_times_ms["moe"])
    log(f"  Per-token totals on final decode step:")
    log(f"    DN   block: {dn_total:6.1f} ms  ({n_dn} layers, {dn_total / n_dn:.2f} ms/layer)")
    log(f"    ATTN block: {attn_total:6.1f} ms  ({n_attn} layers, {attn_total / max(1,n_attn):.2f} ms/layer)")
    log(f"    MOE  block: {moe_total:6.1f} ms  ({n_moe} layers, {moe_total / max(1,n_moe):.2f} ms/layer)")
    total_blocks = dn_total + attn_total + moe_total
    log(f"    SUM blocks: {total_blocks:.1f} ms")
    if total_blocks > 0:
        log(f"  Share:  DN {dn_total/total_blocks*100:.1f}%  "
            f"attn {attn_total/total_blocks*100:.1f}%  "
            f"moe {moe_total/total_blocks*100:.1f}%")


if __name__ == "__main__":
    main()
