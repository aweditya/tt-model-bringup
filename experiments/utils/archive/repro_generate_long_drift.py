"""Minimal reproducer for the `generate_long --max-pos 512` slice crash.

Bug: server.py builds `cos_ext_table_tt` / `sin_ext_table_tt` at hardcoded
MAX_POS=256 during bootstrap (server.py:167, 178-179). When `generate_long`
is called with --max-pos 512, the per-token forward at cur_pos=256 calls
`ttnn.slice(state.cos_ext_table_tt, [256, 0], [257, ROTARY_DIM])` which
fails because the table only has 256 rows.

This reproducer faithfully replays the slice pattern (no model weights needed)
to confirm the failure point and validate the fix.

Run on qb1:
  ssh qb1 'cd tt-xla && .venv/bin/python -m experiments.utils.repro_generate_long_drift'
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import torch
import ttnn

MAX_POS = 256       # hardcoded constant in server.py
HEAD_DIM = 256      # from Qwen3.6-27B config
PARTIAL_ROTARY = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY)  # 64


def build_cos_sin_ext_table(table_max_pos: int):
    """Mirrors server.bootstrap lines 163-183 — extended (HEAD_DIM-wide) table."""
    half_rot = ROTARY_DIM // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(table_max_pos).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    pad = HEAD_DIM - ROTARY_DIM
    cos_pad = np.ones((table_max_pos, pad), dtype=np.float32)
    sin_pad = np.zeros((table_max_pos, pad), dtype=np.float32)
    cos_ext = np.concatenate([cos_all, cos_pad], axis=-1).astype(np.float32)
    sin_ext = np.concatenate([sin_all, sin_pad], axis=-1).astype(np.float32)
    return cos_ext, sin_ext


def upload(arr_np, device):
    t = torch.from_numpy(arr_np)
    return ttnn.from_torch(t, dtype=ttnn.float32, device=device,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)


def try_slice(cos_tt, sin_tt, cur_pos: int):
    """Mirror of handle_generate_paged forward_token's per-step slice."""
    cos = ttnn.slice(cos_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
    sin = ttnn.slice(sin_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
    return cos, sin


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # qb1 has 4 chips; the persistent server is on device 0. Use 1 by default
    # so this can run alongside the server without conflict.
    ap.add_argument("--device-id", type=int, default=1)
    cli = ap.parse_args()
    device = ttnn.open_device(device_id=cli.device_id)
    try:
        # === REPRO PHASE: build table at MAX_POS=256, slice at cur_pos=256 ===
        print(f"[repro] building cos/sin ext table at MAX_POS={MAX_POS}")
        cos_np, sin_np = build_cos_sin_ext_table(MAX_POS)
        print(f"[repro]   table shape: {cos_np.shape}")
        cos_tt = upload(cos_np, device)
        sin_tt = upload(sin_np, device)

        print("[repro] slicing at cur_pos=0 ... ", end="")
        try_slice(cos_tt, sin_tt, 0)
        print("OK")

        print("[repro] slicing at cur_pos=255 ... ", end="")
        try_slice(cos_tt, sin_tt, 255)
        print("OK")

        print("[repro] slicing at cur_pos=256 (should fail) ... ", end="")
        crashed = False
        try:
            try_slice(cos_tt, sin_tt, 256)
            print("DID NOT CRASH (unexpected!)")
        except RuntimeError as e:
            crashed = True
            msg = str(e)
            print("CRASHED (expected)")
            short = msg.split("\n")[0:3]
            for ln in short:
                print(f"    {ln}")
        if not crashed:
            print("[repro] BUG NOT REPRODUCED — slice unexpectedly succeeded")
            return 1

        # === FIX VALIDATION PHASE: build at 512, slice at 256+ ===
        print("\n[fix] rebuilding cos/sin ext table at MAX_POS=512")
        cos_np2, sin_np2 = build_cos_sin_ext_table(512)
        print(f"[fix]   table shape: {cos_np2.shape}")
        cos_tt2 = upload(cos_np2, device)
        sin_tt2 = upload(sin_np2, device)

        for cur in (0, 255, 256, 400, 511):
            print(f"[fix] slicing at cur_pos={cur} ... ", end="")
            try:
                try_slice(cos_tt2, sin_tt2, cur)
                print("OK")
            except RuntimeError as e:
                print("FAILED")
                print(f"    {str(e).splitlines()[0]}")
                return 1

        # Show that values at cur_pos 100 are exactly the same in 256-table and
        # 512-table (no wraparound, no different freq). This rules out a wrong-
        # angle drift theory for the < 256 positions.
        cos_a = cos_np[100]
        cos_b = cos_np2[100]
        diff = np.max(np.abs(cos_a - cos_b))
        print(f"\n[fix] cos[100] diff between 256-table and 512-table: max|delta|={diff:.2e}")
        assert diff == 0.0, "tables should match in overlap region"
        print("[fix] FIX VALIDATED: rebuilding table at requested max_pos avoids the crash.")
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
