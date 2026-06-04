#!/usr/bin/env python3
"""Probe: does ttnn.embedding(uint32-[1], cos_table) actually return the
indexed row, or does it silently return row 0 regardless?

The full-forward probe shows pos 1 attention contributing ~0 (TT echoes
input). The primitive paged write/read PASSES (gm4_sliding_write_read.py).
The next-most-likely culprit: rot_idxs_buf lookup not advancing → RoPE
applied with cos=1, sin=0 at all positions → no positional encoding.

Test: build a small cos/sin table where row i = [i, i, i, …]. Look up
row 1, 5, 17 via ttnn.embedding with the same buffer dtype/layout as the
server (uint32 [1] ROW_MAJOR). Verify returned row contains the expected
constant. If it returns row 0 — bug confirmed.

Run on qb1:
  cd ~/tt-xla && \\
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \\
    TT_BUILD_DIR=$TT_METAL_HOME/build_Release \\
    ARCH_NAME=blackhole \\
    PYTHONPATH=$TT_METAL_HOME/ttnn \\
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \\
    .venv/bin/python -u experiments/cb/isolate/gm4_rope_lookup.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TABLE_ROWS = 64
TABLE_COLS = 256  # matches HEAD_DIM_SLIDING


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import ttnn
    log("opening device 0")
    device = ttnn.open_device(device_id=0)
    try:
        # Table: row i has all entries = float(i). So lookup of row k → all k.
        table_np = np.tile(np.arange(TABLE_ROWS, dtype=np.float32)[:, None],
                            (1, TABLE_COLS))  # [TABLE_ROWS, TABLE_COLS]
        table_tt = ttnn.from_torch(torch.from_numpy(table_np),
                                   dtype=ttnn.bfloat16,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        all_ok = True
        for query_row in [0, 1, 5, 17, 33]:
            # Same dtype/layout as server's rot_idxs_buf.
            idx_buf = ttnn.from_torch(
                torch.tensor([query_row], dtype=torch.int32),
                dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
            )
            out = ttnn.embedding(idx_buf, table_tt)
            out_np = ttnn.to_torch(out).float().numpy().reshape(-1)
            # Expect all entries == query_row.
            expected = float(query_row)
            mean_val = float(out_np.mean())
            same = np.allclose(out_np, expected, atol=0.05)
            log(f"  query_row={query_row:3d}  mean(out)={mean_val:7.3f}  "
                f"expected={expected:7.3f}  {'PASS' if same else 'FAIL'}")
            all_ok = all_ok and same
            ttnn.deallocate(out); ttnn.deallocate(idx_buf)

        ttnn.deallocate(table_tt)
        if not all_ok:
            log("FAIL: ttnn.embedding lookup is not advancing past row 0 (or "
                "is returning wrong rows). This explains pos > 0 attention "
                "failure: RoPE always applies row 0 (cos=1, sin=0 = identity).")
            raise SystemExit(1)
        log("PASS: ttnn.embedding works correctly. RoPE plumbing is "
            "exonerated; bug is elsewhere.")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
