#!/usr/bin/env python3
"""MM7 v0.2.a — L0..L5 chained forward smoke.

Validates per-layer block composition by feeding TT's chained output
from layer N as the input to layer N+1, and comparing to HF
hidden_states[N+1]. Covers all 3 block kinds:
  L0 = mamba2   (DN with SSD via owned kernel)
  L1 = moe      (Expert Parallel — v0.1.4 path)
  L2 = mamba2
  L3 = moe
  L4 = mamba2
  L5 = attention (GQA, no RoPE)

Gate: per-layer cos(TT block_out, HF hidden_states[L+1]) >= 0.999.
This validates the full per-block dispatch + the chain composition
(no drift accumulation problems in bf16 across heterogeneous blocks).

REUSE: each block forward (attn/mamba2/moe_ep) already passes its
single-layer smoke; this is JUST the chain test.

Run on qb1 (default EP mode for MoE):
    NEMOTRON3_UPLOAD_LAYERS=0,1,2,3,4,5 \\
        TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
        TT_BUILD_DIR=$TT_METAL_HOME/build_Release ARCH_NAME=blackhole \\
        PYTHONPATH=$TT_METAL_HOME/ttnn \\
        LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
        .venv/bin/python -u experiments/cb/isolate/nemotron3_v02a_l0_l5_chain_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
COS_GATE = 0.999
LAYER_RANGE = list(range(6))  # L0..L5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_and_mad(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mad = float(np.mean(np.abs(a - b)))
    return cos, mad


def main() -> int:
    os.environ.setdefault(
        "NEMOTRON3_UPLOAD_LAYERS",
        ",".join(str(L) for L in LAYER_RANGE),
    )
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")

    log("loading HF oracle hidden_states…")
    hidden_states = np.load(ORACLE_DIR / "hidden_states.npy")
    log(f"  hidden_states shape: {hidden_states.shape}  "
        f"(L0..L{hidden_states.shape[0]-1})")

    log("bootstrapping (L0..L5)…")
    import server_nemotron3_nano_ttnn as srv
    import ttnn
    state = srv.State()
    t0 = time.time()
    srv.bootstrap(state, log)
    log(f"  bootstrap in {time.time() - t0:.1f}s")

    try:
        # Start with hidden_states[0] = post-embed input to L0.
        h = hidden_states[0].copy()  # [S, HIDDEN]
        log(f"start: h shape {h.shape}, "
            f"L0..L{LAYER_RANGE[-1]} on layer_types "
            f"{[state.layer_types[L] for L in LAYER_RANGE]}")

        gates = []
        for L in LAYER_RANGE:
            kind = state.layer_types[L]
            t_fwd = time.time()
            if kind == "attention":
                res = srv.attn_block_eager(state, h, L)
            elif kind == "mamba2":
                res = srv.mamba2_block_eager(state, h, L)
            elif kind == "moe":
                res = srv.moe_block_eager_ep(state, h, L)
            else:
                raise NotImplementedError(f"L{L} kind={kind!r}")
            h_next = res["block_out"]
            if h_next.ndim == 3:
                h_next = h_next[0]
            elapsed = time.time() - t_fwd
            cos, mad = cos_and_mad(h_next, hidden_states[L + 1])
            ok = cos >= COS_GATE
            gates.append(ok)
            log(f"L{L} ({kind:>9s}) cos={cos:.6f} mad={mad:.4e} "
                f"{'PASS ✓' if ok else 'FAIL ✗'} ({elapsed:.1f}s)")
            h = h_next

        all_pass = all(gates)
        n_pass = sum(gates)
        log("")
        log(f"v0.2.a L0..L5 chain "
            f"{'PASS ✓' if all_pass else 'FAIL ✗'} "
            f"({n_pass}/{len(LAYER_RANGE)} gates green)")
        return 0 if all_pass else 1
    finally:
        log("closing mesh…")
        ttnn.close_mesh_device(state.mesh)


if __name__ == "__main__":
    sys.exit(main())
