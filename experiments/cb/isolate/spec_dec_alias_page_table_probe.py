#!/usr/bin/env python3
"""Phase 2.B.0 — host-only probe for `build_verify_alias_page_table_host`.

No device required. Validates the alias-table semantics that the B=K+1
verify trace depends on: rows [verify_offset, verify_offset+K+1) all
point at row 0's KV blocks; other rows unchanged.

Run on qb1 (or anywhere; pure host):
  ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
    experiments/cb/isolate/spec_dec_alias_page_table_probe.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

from spec_dec_scheduler import build_verify_alias_page_table_host  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(state=None) -> int:  # state= accepted for dev-harness compat
    log("== build_verify_alias_page_table_host smoke ==")
    # Realistic Gemma 4 page-table shape: NUM_BLOCKS=256, pages_per_seq=8
    # (Phase 2.A used these; see server_gemma4_unified_ttnn.py setup_paged_decode_state)
    NUM_ROWS = 256
    PAGES_PER_SEQ = 8

    # Synthesize a deterministic base page-table (unique per-row identity).
    base = torch.arange(
        NUM_ROWS * PAGES_PER_SEQ, dtype=torch.int32,
    ).reshape(NUM_ROWS, PAGES_PER_SEQ)

    cases = [
        (5, 1),     # K=5, verify_offset=1 — single-stream default
        (3, 1),     # K=3, smaller lookahead
        (7, 1),     # K=7, larger lookahead
        (5, 8),     # K=5, verify_offset=8 — non-default offset
    ]

    passed = 0
    failed = 0
    for K, verify_offset in cases:
        log("")
        log(f"--- K={K}, verify_offset={verify_offset} ---")
        alias = build_verify_alias_page_table_host(base, K, verify_offset)

        # Gate 1: shape preserved + dtype int32
        assert alias.shape == base.shape, \
            f"shape changed: {tuple(alias.shape)} vs {tuple(base.shape)}"
        assert alias.dtype == torch.int32, f"dtype not int32: {alias.dtype}"
        log(f"  shape={tuple(alias.shape)} dtype={alias.dtype} ✓")

        # Gate 2: row 0 unchanged
        assert torch.equal(alias[0], base[0]), \
            f"row 0 changed unexpectedly: {alias[0]} vs {base[0]}"
        log(f"  row 0 unchanged ✓")

        # Gate 3: K+1 verify rows all alias to row 0
        for i in range(K + 1):
            row = verify_offset + i
            assert torch.equal(alias[row], base[0]), (
                f"alias row {row} != base row 0: "
                f"alias={alias[row].tolist()} base[0]={base[0].tolist()}"
            )
        log(f"  K+1={K+1} alias rows [{verify_offset}..{verify_offset+K})"
            f" all == row 0 ✓")

        # Gate 4: other rows (not row 0, not in alias range) unchanged
        alias_range = set(range(verify_offset, verify_offset + K + 1))
        for r in range(NUM_ROWS):
            if r == 0 or r in alias_range:
                continue
            assert torch.equal(alias[r], base[r]), (
                f"row {r} changed unexpectedly: "
                f"alias={alias[r][:3].tolist()}... "
                f"base={base[r][:3].tolist()}..."
            )
        log(f"  {NUM_ROWS - 1 - (K+1)} unrelated rows unchanged ✓")

        # Gate 5: alias-table input wasn't mutated (clone-correctness)
        # base[0][0] should still be 0 (its original value)
        assert base[0, 0] == 0, "base page-table was mutated by helper!"
        log(f"  base page-table not mutated ✓")

        passed += 1

    # Gate 6: shape assertion fires when num_rows < verify_offset+K+1
    log("")
    log("--- num_rows too small (should raise AssertionError) ---")
    small = torch.zeros((4, 2), dtype=torch.int32)
    try:
        build_verify_alias_page_table_host(small, K=5, verify_offset=1)
        log("  ✗ expected AssertionError; got none")
        failed += 1
    except AssertionError as e:
        log(f"  AssertionError raised ✓ ({e!s:80}...)")
        passed += 1

    log("")
    log("=" * 60)
    log(f"PASS {passed}/{passed + failed}")
    log("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
