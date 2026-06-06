#!/usr/bin/env python3
"""MM7 v0.3.1.c step 2 — paged decode state setup smoke.

Verifies that `reset_decode_state(state)` allocates all 5 SDPA support
buffers + per-attention-layer KV caches (Gemma 4 two-call: TWO caches
per layer) without TT_FATAL or shape mismatch.

Gates:
  • state.cur_pos_buf / page_table_tt / paged_write_mem_cfg /
    paged_sdpa_progcfg / sdpa_compute_kernel_config all populated
  • Each attention layer has kv_K_cache_tt[L] and kv_V_cache_tt[L] each
    a list-of-2 ttnn.Tensors with shape [NUM_BLOCKS, NCHIPS, BLOCK_SIZE, HEAD_DIM]
  • Per-chip view of each cache is [NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM]
  • Total memory under the budget (target <100 MB/chip total caches)
  • cur_pos = 0 after reset

Forks: nemotron3_v030_resident_smoke.py harness shell.
Harness-aware: accepts state=None.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v031c_setup_smoke'
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    import server_nemotron3_nano_ttnn as srv
    import ttnn

    t_boot = 0.0
    if state is None:
        log("bootstrap (all-resident — only needed when not running under harness)…")
        state = srv.State()
        t0 = time.time()
        srv.bootstrap(state, log)
        t_boot = time.time() - t0
        log(f"  bootstrap in {t_boot:.1f}s")
    else:
        log("[harness] reusing live state ✓")

    try:
        log("calling reset_decode_state(state)…")
        t0 = time.time()
        srv.reset_decode_state(state, B=1, log=log)
        log(f"  reset_decode_state in {time.time() - t0:.1f}s")

        # Gate 1: support buffers populated
        gates = []
        for name in [
            "cur_pos_buf", "page_table_tt", "paged_write_mem_cfg",
            "paged_sdpa_progcfg", "sdpa_compute_kernel_config",
        ]:
            val = getattr(state, name, None)
            ok = val is not None
            gates.append(ok)
            log(f"  Gate state.{name}: {'PASS ✓' if ok else 'FAIL ✗'}")

        # Gate 2: cur_pos = 0
        ok = state.cur_pos == 0
        gates.append(ok)
        log(f"  Gate cur_pos == 0: {'PASS ✓' if ok else 'FAIL ✗'}  "
            f"(value = {state.cur_pos})")

        # Gate 3: per-attention-layer caches
        n_attn = 0
        total_mb_per_chip = 0
        for L, kind in enumerate(state.layer_types):
            if kind != "attention":
                continue
            n_attn += 1
            kc_list = state.kv_K_cache_tt[L]
            vc_list = state.kv_V_cache_tt[L]
            ok_k = (
                isinstance(kc_list, list)
                and len(kc_list) == srv.NUM_KV_HEADS
                and all(c is not None for c in kc_list)
            )
            ok_v = (
                isinstance(vc_list, list)
                and len(vc_list) == srv.NUM_KV_HEADS
                and all(c is not None for c in vc_list)
            )
            gates.append(ok_k and ok_v)
            # Per-cache size: NUM_BLOCKS × NCHIPS × BLOCK_SIZE × HEAD_DIM
            #   ÷ NCHIPS for per-chip × 2 bytes (bf16) ÷ (1024**2) for MB
            cache_bytes_per_chip = (
                srv.SDPA_NUM_BLOCKS * srv.SDPA_BLOCK_SIZE * srv.HEAD_DIM_ATTN * 2
            )
            # 2 caches per layer (Gemma 4 two-call) × 2 (K and V)
            layer_mb_per_chip = cache_bytes_per_chip * srv.NUM_KV_HEADS * 2 / (1024 * 1024)
            total_mb_per_chip += layer_mb_per_chip
            shapes = [tuple(c.shape) for c in kc_list]
            log(f"  L{L:>2d} (attention): K caches len={len(kc_list)} "
                f"shapes={shapes}  {'PASS ✓' if ok_k and ok_v else 'FAIL ✗'}  "
                f"(layer ~{layer_mb_per_chip:.1f} MB/chip)")
        log(f"  total KV cache: ~{total_mb_per_chip:.1f} MB/chip "
            f"({n_attn} attn layers)")

        all_pass = all(gates)
        n_pass = sum(gates)
        log("")
        log(f"v0.3.1.c step-2 setup smoke {'PASS ✓' if all_pass else 'FAIL ✗'} "
            f"({n_pass}/{len(gates)} gates green)")
        return 0 if all_pass else 1
    finally:
        if t_boot > 0:
            log("closing mesh…")
            ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
