#!/usr/bin/env python3
"""Isolate the TRISC compile failure seen in v0.2 forward smoke.

The drafter v0.2 smoke hit `ckernel_sfpu_binary_pow.h:195: sfpi::copysgn
returns vSMag but expected vInt` when JIT-compiling `layernorm` trisc1
for shape `[1, 1, HIDDEN=1024]` (rms_norm of the pre_projection output).

This probe tests THREE rms_norm shapes:
  (A) HIDDEN=1024 rank-3 [1, 1, 1024] — what drafter sees (failed)
  (B) HIDDEN=1024 rank-2 [1, 1024]     — same width, simpler rank
  (C) HIDDEN=3840 rank-3 [1, 1, 3840]  — target Gemma 4 12B's width
                                          (whether the target server's
                                           layernorm cold-compile also fails)

If ALL three fail → tt-metal build on qb2 is broken at any rms_norm. Escalate.
If only (A) fails → the kernel-compile-time-args combo for HIDDEN=1024 is
  the new path that the SFPI bug fires on.

Open mesh once; deallocate test tensors between cases. Captures the FIRST
exception per case; reports a 3-row summary at the end.

Run on qb2:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \
        experiments/cb/isolate/gemma4_assistant_rms_norm_isolate.py'
"""
from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import torch

import ttnn


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_replicated(np_arr, mesh, layout=ttnn.TILE_LAYOUT,
                     dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.from_numpy(np_arr.astype(np.float32)),
        dtype=dtype, layout=layout, device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def try_rms_norm(mesh, shape, hidden, name):
    log(f"--- case {name}: shape={shape}, hidden={hidden} ---")
    try:
        x = make_replicated(
            np.random.randn(*shape).astype(np.float32), mesh)
        w = make_replicated(
            np.ones(hidden, dtype=np.float32), mesh)
        y = ttnn.rms_norm(x, weight=w, epsilon=1e-6)
        log(f"  rms_norm OK; y.shape={list(y.shape)}")
        ttnn.deallocate(x); ttnn.deallocate(w); ttnn.deallocate(y)
        return True, None
    except Exception as e:
        log(f"  rms_norm FAILED: {type(e).__name__}: {str(e)[:200]}")
        return False, str(e)


def main() -> int:
    log("opening 1x4 mesh…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    log(f"  mesh: {mesh}")

    cases = [
        ("A_hidden1024_rank3", (1, 1, 1024), 1024),
        ("B_hidden1024_rank2", (1, 1024),    1024),
        ("C_hidden3840_rank3", (1, 1, 3840), 3840),
        ("D_hidden1024_rank3_tile_pad", (1, 32, 1024), 1024),
    ]

    summary = []
    for name, shape, hidden in cases:
        ok, err = try_rms_norm(mesh, shape, hidden, name)
        summary.append((name, ok, err))

    log("")
    log("=" * 64)
    log("SUMMARY")
    log("=" * 64)
    for name, ok, err in summary:
        flag = "OK" if ok else "FAIL"
        log(f"  {name}: {flag}" + (f"  ({err[:80]}…)" if err else ""))

    ttnn.close_mesh_device(mesh)
    log("mesh closed.")
    all_ok = all(s[1] for s in summary)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
