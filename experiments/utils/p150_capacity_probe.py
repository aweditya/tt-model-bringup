#!/usr/bin/env python3
"""
P150 capacity probe — DRAM and L1 limits (qb1, single device).

Companion to p150_memory_bandwidth_probe.py. Runs only the slow capacity
probes (separated so we don't re-run the BW sweep when extending the
capacity search range). Faster startup; reuses the same single device.

(1) DRAM capacity: bracket from <start_gb> upward until first failure,
    then bisect to within 0.25 GB. We expect ~30-32 GB (8 channels x 4 GB
    nominal; some reserved for fabric / dispatch / firmware).

(2) L1 per-Tensix-core capacity: allocate INTERLEAVED L1 bf16 tensors
    sized such that bytes/110_cores grows past the per-core L1 limit.
    Published Blackhole L1: 1.5 MB (1536 KB) per Tensix core.

Output JSON appended to .cache/p150_memory_bandwidth/capacity_results.json.

Run on qb1:
    cd ~/tt-xla && .venv/bin/python -u experiments/utils/p150_capacity_probe.py
"""
import json
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


def _flat_shape_for_bytes(nbytes, dtype_bytes=2, cols=4096):
    """Tile-aligned 2D shape (rows multiple of 32, cols fixed) holding nbytes."""
    elems = nbytes // dtype_bytes
    rows = elems // cols
    rows = (rows // 32) * 32
    if rows == 0:
        rows = 32
    return (rows, cols)


def bracket_dram_capacity(device, start_gb=28.0, max_gb=36.0, step_gb=1.0):
    print(f"\n[dram_capacity] bracket from {start_gb} to {max_gb} GB, step {step_gb} GB")
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
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f" FAIL: {err}")
            results.append({"size_gb": actual_gb, "outcome": "fail", "error": err})
            first_fail_gb = actual_gb
            break

    # bisect to 0.1 GB resolution
    if first_fail_gb is not None and last_success_gb > 0:
        lo, hi = last_success_gb, first_fail_gb
        while hi - lo > 0.1:
            mid = (lo + hi) / 2.0
            nbytes = int(mid * (1024 ** 3))
            shape = _flat_shape_for_bytes(nbytes, dtype_bytes=2)
            actual_gb = shape[0] * shape[1] * 2 / (1024 ** 3)
            print(f"  bisect: {mid:.3f} GB (actual {actual_gb:.3f})...", end="")
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
                err = f"{type(e).__name__}: {str(e)[:200]}"
                print(f" FAIL: {err}")
                results.append({"size_gb": actual_gb, "outcome": "fail", "error": err})
                hi = actual_gb

    return {
        "largest_alloc_gb": last_success_gb,
        "first_fail_gb": first_fail_gb,
        "trace": results,
    }


def probe_l1_capacity(device):
    """
    L1 per-core capacity probe.

    Strategy: allocate INTERLEAVED L1 bf16 tensors of size = 110_cores * P_per_core.
    Sweep P_per_core upward until allocation fails. The largest successful
    P_per_core is the per-core L1 budget after ttnn's reservations.

    Published Blackhole L1 per Tensix = 1.5 MB (1536 KB).
    """
    print("\n[l1_capacity] sweeping interleaved L1 allocation size")
    n_cores = 110  # from compute_with_storage_grid_size (11x10)
    cols = 32      # one tile-width column to keep simple distribution

    # Sweep sizes from 64 KB/core up to 2.5 MB/core, in 64 KB steps
    results = []
    last_ok = 0
    first_fail = None
    sweep_kb = list(range(64, 2560, 64))
    for kb_per_core in sweep_kb:
        # rows are split across cores; pick rows such that total bytes ≈ kb_per_core * n_cores * 1024
        total_bytes = kb_per_core * 1024 * n_cores
        elems = total_bytes // 2
        rows = (elems // cols // 32) * 32
        if rows < 32:
            continue
        shape = (rows, cols)
        actual_kb_per_core = rows * cols * 2 / 1024.0 / n_cores
        try:
            host = torch.zeros(shape, dtype=torch.bfloat16)
            t = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(t)
            ttnn.synchronize_device(device)
            results.append({"kb_per_core": actual_kb_per_core, "shape": list(shape), "outcome": "ok"})
            last_ok = actual_kb_per_core
            if kb_per_core % 256 == 0:
                print(f"  {kb_per_core} KB/core (actual {actual_kb_per_core:.2f}): OK")
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"  {kb_per_core} KB/core (actual {actual_kb_per_core:.2f}): FAIL ({err})")
            results.append({"kb_per_core": actual_kb_per_core, "shape": list(shape), "outcome": "fail", "error": err})
            first_fail = actual_kb_per_core
            break

    return {
        "largest_ok_kb_per_core": last_ok,
        "first_fail_kb_per_core": first_fail,
        "compute_with_storage_cores": n_cores,
        "trace": results,
    }


def main():
    host = socket.gethostname()
    ts = datetime.now(timezone.utc).isoformat()
    print(f"P150 capacity probe (host={host}, time={ts})")

    device = ttnn.open_device(device_id=0)
    print("opened device_id=0")

    results = {
        "host": host,
        "timestamp_utc": ts,
    }

    try:
        results["dram_capacity"] = bracket_dram_capacity(device, start_gb=28.0, max_gb=36.0, step_gb=1.0)
        results["l1_capacity"] = probe_l1_capacity(device)
    finally:
        try:
            ttnn.close_device(device)
            print("\nclosed device cleanly")
        except Exception as e:
            print(f"close_device error: {e}")

    out_dir = Path.home() / "tt-xla" / ".cache" / "p150_memory_bandwidth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "capacity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults JSON: {out_path}")

    print("\n" + "=" * 64)
    print("CAPACITY SUMMARY")
    print("=" * 64)
    dc = results["dram_capacity"]
    print(f"  DRAM: largest allocation = {dc['largest_alloc_gb']:.3f} GB")
    if dc["first_fail_gb"]:
        print(f"        first failure at      = {dc['first_fail_gb']:.3f} GB")
    lc = results["l1_capacity"]
    print(f"  L1 (interleaved over 110 cores):")
    print(f"        largest OK = {lc['largest_ok_kb_per_core']:.1f} KB/core")
    if lc["first_fail_kb_per_core"]:
        print(f"        first fail = {lc['first_fail_kb_per_core']:.1f} KB/core")
    print(f"        total L1 (sum over cores) ≈ {lc['largest_ok_kb_per_core'] * 110 / 1024:.1f} MB")
    print("=" * 64)


if __name__ == "__main__":
    main()
