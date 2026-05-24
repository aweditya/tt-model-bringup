#!/usr/bin/env python3
"""
P150 memory hierarchy probe (qb1, single device).

Measures (no model weights downloaded — pure synthetic tensors):

  (1) DRAM write  bandwidth (host -> device) via ttnn.from_torch
  (2) DRAM read   bandwidth (device -> host) via ttnn.to_torch
  (3) DRAM streaming on-device bandwidth via ttnn.clone (DRAM->DRAM copy)
  (4) DRAM capacity (largest allocatable bf16 tensor; bracketed, NOT to system-OOM)
  (5) L1 capacity per Tensix core (queried + literal allocation test)
  (6) Compute-with-storage grid size (active Tensix grid)
  (7) Effective BW vs published spec (512 GB/s peak DRAM, ~32 GB DRAM capacity).

Compares (3) to the 400.7 GB/s "78% of peak" MLP number reported in
`feedback_qb1_mlp_at_78pct_peak.md`.

All timing regions are wrapped:
   ttnn.synchronize_device(device)  BEFORE t0
   ttnn.synchronize_device(device)  AFTER  t1
per `feedback_sync_bounded_timing.md` — ttnn dispatch is async.

Output:
  - JSON results to .cache/p150_memory_bandwidth/results.json
  - Short summary printed to stdout

Run on qb1 (single P150, no model server running):
    cd ~/tt-xla \\
      && export TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
      && export TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
      && export ARCH_NAME=blackhole \\
      && export PYTHONPATH=$TT_METAL_HOME/ttnn:$PYTHONPATH \\
      && export LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib:$LD_LIBRARY_PATH \\
      && .venv/bin/python -u experiments/utils/p150_memory_bandwidth_probe.py
"""
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# ----- Published P150 spec constants (from Tenstorrent docs + memory notes) ---
P150_DRAM_PEAK_GB_S = 512.0     # docs.tenstorrent.com/aibs/blackhole/specifications.html
P150_DRAM_NOMINAL_GB = 32.0     # 8x4GB GDDR6
TENSIX_FULL_GRID_COLS = 14      # logical full grid (Blackhole)
TENSIX_FULL_GRID_ROWS = 10
TENSIX_FULL_CORES_PRE_FW19_5 = 140  # before fw v19.5.0
TENSIX_FULL_CORES_POST_FW19_5 = 120 # after fw v19.5.0 silently disabled 20

# Sweep sizes for read/write/copy (bytes worth of bf16 elements)
# bf16 = 2 bytes/elem -> elements = SIZE_BYTES // 2
SIZE_BYTES_LIST = [
    1   * 1024 * 1024,    # 1 MiB
    16  * 1024 * 1024,    # 16 MiB
    256 * 1024 * 1024,    # 256 MiB
    1024 * 1024 * 1024,   # 1 GiB
]

# ---------------------------------------------------------------------------


def _flat_shape_for_bytes(nbytes, dtype_bytes=2):
    """Pick a 2D tile-aligned shape (32-aligned in both dims) holding nbytes."""
    elems = nbytes // dtype_bytes
    # Aim for a "tall thin" matrix with row dim a multiple of 32.
    # Width = 32*N rows, columns = 32*M ; choose columns = 4096 (128 tiles wide)
    cols = 4096
    rows = elems // cols
    # round rows down to multiple of 32
    rows = (rows // 32) * 32
    if rows == 0:
        rows = 32
    return (rows, cols)


def _sync(device):
    ttnn.synchronize_device(device)


def _time_block(fn, device, n_warmup=2, n_measure=5):
    for _ in range(n_warmup):
        fn()
        _sync(device)
    samples = []
    for _ in range(n_measure):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    samples_ms = [s * 1000.0 for s in samples]
    return {
        "median_ms": float(np.median(samples_ms)),
        "min_ms": float(np.min(samples_ms)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "n_measure": n_measure,
        "samples_ms": samples_ms,
    }


def measure_write_bandwidth(device, size_bytes):
    """Host -> device DRAM. ttnn.from_torch writes the tensor to device DRAM."""
    shape = _flat_shape_for_bytes(size_bytes, dtype_bytes=2)
    actual_bytes = shape[0] * shape[1] * 2
    host = torch.randn(shape, dtype=torch.bfloat16)

    # Warmup + measure separately because ttnn.from_torch ALLOCATES each call.
    # We cannot reuse the same target buffer with ttnn.from_torch.
    # The op cost dominates and includes PCIe + DRAM write; both contribute.
    # We deallocate between calls to avoid OOM at large sizes.
    holder = {}

    def _do():
        if "t" in holder:
            ttnn.deallocate(holder["t"])
            del holder["t"]
        holder["t"] = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    stats = _time_block(_do, device, n_warmup=1, n_measure=3)
    if "t" in holder:
        ttnn.deallocate(holder["t"])
    bw_gb_s = (actual_bytes / 1e9) / (stats["median_ms"] / 1000.0)
    return {
        "bytes": actual_bytes,
        "shape": list(shape),
        **stats,
        "bandwidth_gb_s": bw_gb_s,
        "pct_of_dram_peak": bw_gb_s / P150_DRAM_PEAK_GB_S * 100.0,
    }


def measure_read_bandwidth(device, size_bytes):
    """Device DRAM -> host via ttnn.to_torch. Tensor pre-allocated, reused across iters."""
    shape = _flat_shape_for_bytes(size_bytes, dtype_bytes=2)
    actual_bytes = shape[0] * shape[1] * 2
    host = torch.randn(shape, dtype=torch.bfloat16)
    dev_t = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _do():
        # ttnn.to_torch -> CPU tensor. The cost includes device-side untilize (small)
        # and DRAM read + PCIe DMA; for large tensors PCIe dominates.
        _ = ttnn.to_torch(dev_t)

    try:
        stats = _time_block(_do, device, n_warmup=1, n_measure=3)
    finally:
        ttnn.deallocate(dev_t)
    bw_gb_s = (actual_bytes / 1e9) / (stats["median_ms"] / 1000.0)
    return {
        "bytes": actual_bytes,
        "shape": list(shape),
        **stats,
        "bandwidth_gb_s": bw_gb_s,
        "pct_of_dram_peak": bw_gb_s / P150_DRAM_PEAK_GB_S * 100.0,
    }


def measure_on_device_copy_bandwidth(device, size_bytes):
    """
    On-device DRAM -> DRAM copy. ttnn.clone allocates a new tensor and copies.
    This is the canonical "DRAM streaming" measurement.

    Bandwidth counted as 2*size (one read of source + one write of destination),
    which is the convention used in HBM/GDDR streaming benchmarks (and matches
    `feedback_qb1_mlp_at_78pct_peak.md`'s weight-read accounting if you double
    it for a "copy" op).

    We also report the one-way figure for comparison.
    """
    shape = _flat_shape_for_bytes(size_bytes, dtype_bytes=2)
    actual_bytes = shape[0] * shape[1] * 2
    host = torch.randn(shape, dtype=torch.bfloat16)
    src = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ttnn.clone creates a new tensor in DRAM; old one is unaffected.
    # Each call returns a new tensor, so we deallocate within the timed
    # closure so n_measure iters don't OOM at 1 GiB.
    holder = {}

    def _do():
        if "t" in holder:
            ttnn.deallocate(holder["t"])
            del holder["t"]
        holder["t"] = ttnn.clone(src, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    try:
        stats = _time_block(_do, device, n_warmup=1, n_measure=3)
    finally:
        if "t" in holder:
            ttnn.deallocate(holder["t"])
        ttnn.deallocate(src)

    bytes_moved_streaming = 2 * actual_bytes  # 1 read + 1 write
    bw_gb_s_streaming = (bytes_moved_streaming / 1e9) / (stats["median_ms"] / 1000.0)
    bw_gb_s_oneway = (actual_bytes / 1e9) / (stats["median_ms"] / 1000.0)
    return {
        "bytes_logical": actual_bytes,
        "bytes_moved_streaming": bytes_moved_streaming,
        "shape": list(shape),
        **stats,
        "bandwidth_gb_s_streaming": bw_gb_s_streaming,   # canonical (R+W)
        "bandwidth_gb_s_oneway": bw_gb_s_oneway,         # alt accounting
        "pct_of_dram_peak_streaming": bw_gb_s_streaming / P150_DRAM_PEAK_GB_S * 100.0,
    }


def measure_dram_capacity(device, start_gb=4.0, max_gb=30.0, step_gb=2.0):
    """
    Bracket-bisect-ish capacity probe: allocate a single DRAM bf16 tensor of
    progressively larger size until one fails, then bisect between the last
    success and the first failure to within 0.25 GB.

    Each attempt deallocates immediately, so a clean run leaves DRAM idle.
    """
    print(f"\n[capacity] Bracketing DRAM capacity from {start_gb} GB up to {max_gb} GB...")
    results = []
    last_success_gb = 0.0
    first_fail_gb = None

    size_gb = start_gb
    while size_gb <= max_gb + 1e-9:
        nbytes = int(size_gb * (1024 ** 3))
        shape = _flat_shape_for_bytes(nbytes, dtype_bytes=2)
        actual_gb = shape[0] * shape[1] * 2 / (1024 ** 3)
        print(f"  trying {size_gb:.2f} GB (actual {actual_gb:.3f} GB, shape {shape})...", end="")
        try:
            host = torch.zeros(shape, dtype=torch.bfloat16)
            t = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            print(" OK")
            ttnn.deallocate(t)
            ttnn.synchronize_device(device)
            results.append({"size_gb": actual_gb, "outcome": "ok"})
            last_success_gb = actual_gb
            size_gb += step_gb
        except Exception as e:
            print(f" FAIL: {type(e).__name__}: {str(e)[:140]}")
            results.append({"size_gb": actual_gb, "outcome": "fail", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            first_fail_gb = actual_gb
            break

    # bisect between last_success_gb and first_fail_gb to 0.25 GB resolution
    if first_fail_gb is not None and last_success_gb > 0:
        lo = last_success_gb
        hi = first_fail_gb
        while hi - lo > 0.25:
            mid = (lo + hi) / 2.0
            nbytes = int(mid * (1024 ** 3))
            shape = _flat_shape_for_bytes(nbytes, dtype_bytes=2)
            actual_gb = shape[0] * shape[1] * 2 / (1024 ** 3)
            print(f"  bisect: {mid:.3f} GB (actual {actual_gb:.3f}) ...", end="")
            try:
                host = torch.zeros(shape, dtype=torch.bfloat16)
                t = ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(device)
                print(" OK")
                ttnn.deallocate(t)
                ttnn.synchronize_device(device)
                results.append({"size_gb": actual_gb, "outcome": "ok"})
                lo = actual_gb
                last_success_gb = max(last_success_gb, actual_gb)
            except Exception as e:
                print(f" FAIL: {type(e).__name__}: {str(e)[:100]}")
                results.append({"size_gb": actual_gb, "outcome": "fail", "error": f"{type(e).__name__}: {str(e)[:200]}"})
                hi = actual_gb

    return {
        "largest_alloc_gb": last_success_gb,
        "first_fail_gb": first_fail_gb,
        "trace": results,
    }


def measure_l1_capacity_per_core(device):
    """
    Probe L1 SRAM size per Tensix core.

    Strategy:
      1. Ask the device object directly via any L1-size method that exists.
      2. Get the compute-with-storage grid size (active cores).
    """
    out = {}
    try:
        gs = device.compute_with_storage_grid_size()
        out["compute_with_storage_grid_x"] = int(gs.x)
        out["compute_with_storage_grid_y"] = int(gs.y)
        out["compute_with_storage_cores"] = int(gs.x) * int(gs.y)
    except Exception as e:
        out["compute_with_storage_grid_error"] = f"{type(e).__name__}: {e}"

    for meth in ("l1_size_per_core", "worker_l1_size", "l1_size"):
        if hasattr(device, meth):
            try:
                v = getattr(device, meth)()
                out[meth] = int(v)
            except Exception as e:
                out[f"{meth}_error"] = f"{type(e).__name__}: {e}"

    for meth in ("dram_size_per_channel", "num_dram_channels", "dram_grid_size"):
        if hasattr(device, meth):
            try:
                v = getattr(device, meth)()
                out[meth] = repr(v) if not isinstance(v, (int, float)) else v
            except Exception as e:
                out[f"{meth}_error"] = f"{type(e).__name__}: {e}"

    if hasattr(device, "arch"):
        try:
            out["arch"] = repr(device.arch())
        except Exception as e:
            out["arch_error"] = f"{type(e).__name__}: {e}"

    return out


def main():
    host = socket.gethostname()
    ts = datetime.now(timezone.utc).isoformat()
    print("=" * 72)
    print(f"P150 memory bandwidth probe (host={host}, time={ts})")
    print("=" * 72)

    device = ttnn.open_device(device_id=0)
    print(f"opened device_id=0")

    results = {
        "host": host,
        "timestamp_utc": ts,
        "published_specs": {
            "dram_peak_gb_s": P150_DRAM_PEAK_GB_S,
            "dram_capacity_gb_nominal": P150_DRAM_NOMINAL_GB,
            "tensix_cores_pre_fw19_5": TENSIX_FULL_CORES_PRE_FW19_5,
            "tensix_cores_post_fw19_5": TENSIX_FULL_CORES_POST_FW19_5,
        },
        "size_sweep_bytes": SIZE_BYTES_LIST,
    }

    try:
        # ---- (6) device + L1 info ----
        print("\n--- L1 / grid / device info ---")
        l1info = measure_l1_capacity_per_core(device)
        for k, v in l1info.items():
            print(f"  {k}: {v}")
        results["device_info"] = l1info

        # ---- (1) DRAM write ----
        print("\n--- DRAM WRITE (host -> device) ---")
        write_results = []
        for nb in SIZE_BYTES_LIST:
            print(f"  size = {nb/1024/1024:.0f} MiB ({nb/1e9:.3f} GB):", end=" ")
            try:
                r = measure_write_bandwidth(device, nb)
                print(f"median={r['median_ms']:.2f} ms  BW={r['bandwidth_gb_s']:.1f} GB/s  ({r['pct_of_dram_peak']:.1f}% of 512 peak)")
                write_results.append({"requested_bytes": nb, **r})
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"FAIL: {err}")
                write_results.append({"requested_bytes": nb, "error": err})
        results["dram_write"] = write_results

        # ---- (2) DRAM read ----
        print("\n--- DRAM READ (device -> host) ---")
        read_results = []
        for nb in SIZE_BYTES_LIST:
            print(f"  size = {nb/1024/1024:.0f} MiB ({nb/1e9:.3f} GB):", end=" ")
            try:
                r = measure_read_bandwidth(device, nb)
                print(f"median={r['median_ms']:.2f} ms  BW={r['bandwidth_gb_s']:.1f} GB/s  ({r['pct_of_dram_peak']:.1f}% of 512 peak)")
                read_results.append({"requested_bytes": nb, **r})
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"FAIL: {err}")
                read_results.append({"requested_bytes": nb, "error": err})
        results["dram_read"] = read_results

        # ---- (3) On-device copy (DRAM streaming) ----
        print("\n--- DRAM STREAMING (on-device DRAM -> DRAM via ttnn.clone) ---")
        copy_results = []
        for nb in SIZE_BYTES_LIST:
            print(f"  size = {nb/1024/1024:.0f} MiB ({nb/1e9:.3f} GB):", end=" ")
            try:
                r = measure_on_device_copy_bandwidth(device, nb)
                print(f"median={r['median_ms']:.2f} ms  streaming BW (R+W)={r['bandwidth_gb_s_streaming']:.1f} GB/s  ({r['pct_of_dram_peak_streaming']:.1f}% of 512 peak)  one-way={r['bandwidth_gb_s_oneway']:.1f} GB/s")
                copy_results.append({"requested_bytes": nb, **r})
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"FAIL: {err}")
                copy_results.append({"requested_bytes": nb, "error": err})
        results["dram_copy"] = copy_results

        # ---- (4) DRAM capacity ----
        # Coarse 8-30 GB sweep here (we know <=30 works). Run
        # `p150_capacity_probe.py` for the finer bracket above 30 GB.
        print("\n--- DRAM CAPACITY (bracketed) ---")
        cap = measure_dram_capacity(device, start_gb=8.0, max_gb=30.0, step_gb=2.0)
        results["dram_capacity"] = cap
        print(f"  largest allocatable: {cap['largest_alloc_gb']:.3f} GB")
        if cap["first_fail_gb"] is not None:
            print(f"  first failure at:    {cap['first_fail_gb']:.3f} GB")

    finally:
        try:
            ttnn.close_device(device)
            print("\nclosed device cleanly")
        except Exception as e:
            print(f"close_device error: {e}")

    # ---- write results ----
    out_dir = Path.home() / "tt-xla" / ".cache" / "p150_memory_bandwidth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults JSON: {out_path}")

    # ---- summary table ----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    def _row(label, val_str, ref_str=""):
        print(f"  {label:32s} {val_str:24s} {ref_str}")

    if "dram_write" in results and results["dram_write"]:
        best = max((r for r in results["dram_write"] if "bandwidth_gb_s" in r), key=lambda x: x["bandwidth_gb_s"], default=None)
        if best:
            _row("DRAM write (host->dev) best", f"{best['bandwidth_gb_s']:.1f} GB/s @ {best['bytes']/1e9:.2f} GB", "(PCIe-bound)")
    if "dram_read" in results and results["dram_read"]:
        best = max((r for r in results["dram_read"] if "bandwidth_gb_s" in r), key=lambda x: x["bandwidth_gb_s"], default=None)
        if best:
            _row("DRAM read  (dev->host) best", f"{best['bandwidth_gb_s']:.1f} GB/s @ {best['bytes']/1e9:.2f} GB", "(PCIe-bound)")
    if "dram_copy" in results and results["dram_copy"]:
        best = max((r for r in results["dram_copy"] if "bandwidth_gb_s_streaming" in r), key=lambda x: x["bandwidth_gb_s_streaming"], default=None)
        if best:
            _row(
                "DRAM streaming (R+W)",
                f"{best['bandwidth_gb_s_streaming']:.1f} GB/s @ {best['bytes_logical']/1e9:.2f} GB",
                f"({best['pct_of_dram_peak_streaming']:.1f}% of 512 peak)",
            )
            _row(
                "DRAM streaming (one-way)",
                f"{best['bandwidth_gb_s_oneway']:.1f} GB/s",
                "(comparable to MLP weight-read accounting)",
            )
    if "dram_capacity" in results:
        cap = results["dram_capacity"]
        _row("DRAM capacity (largest alloc)", f"{cap['largest_alloc_gb']:.2f} GB", f"(nominal {P150_DRAM_NOMINAL_GB} GB)")
    if "device_info" in results:
        di = results["device_info"]
        if "compute_with_storage_cores" in di:
            _row(
                "Compute-with-storage cores",
                f"{di['compute_with_storage_cores']} ({di['compute_with_storage_grid_x']}x{di['compute_with_storage_grid_y']})",
                f"(theoretical full grid {TENSIX_FULL_CORES_PRE_FW19_5} pre-fw19.5 / {TENSIX_FULL_CORES_POST_FW19_5} post)",
            )
        for k in ("l1_size_per_core", "worker_l1_size", "l1_size"):
            if k in di:
                _row(f"L1 size ({k})", f"{di[k]} B = {di[k]/1024:.0f} KiB", "")
        for k in ("dram_size_per_channel", "num_dram_channels"):
            if k in di:
                _row(k, f"{di[k]}", "")
    print("=" * 72)


if __name__ == "__main__":
    main()
