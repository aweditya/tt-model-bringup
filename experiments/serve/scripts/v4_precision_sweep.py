#!/usr/bin/env python3
"""v4 chunked DN precision sweep — REAL ENGLISH TEXT.

First version used synthetic IDs (100..132) which produced degenerate
model activations (cos 0.4-0.9 even with default compute). Real English
text gives meaningful activations and a fair baseline.

IDS_* below come from tokenizing a passage about computing history via
Qwen/Qwen3.6-27B tokenizer (see ssh one-off in conversation history).
First 32/64/128 tokens of that passage.
"""
import os
import socket
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(THIS, "..")))
import protocol as P

SOCK = os.path.expanduser("~/tt-xla/.cache/server_tp.sock")

IDS_32 = [760, 3712, 314, 23470, 42824, 1599, 22892, 494, 279, 650, 92016, 310, 6278, 48889, 22858, 13, 22044, 21461, 5631, 2878, 1040, 279, 55438, 472, 6348, 1560, 310, 51807, 4584, 43389, 12282, 321]
IDS_64 = [760, 3712, 314, 23470, 42824, 1599, 22892, 494, 279, 650, 92016, 310, 6278, 48889, 22858, 13, 22044, 21461, 5631, 2878, 1040, 279, 55438, 472, 6348, 1560, 310, 51807, 4584, 43389, 12282, 321, 9297, 310, 6999, 13934, 17943, 13, 561, 94562, 13395, 1452, 279, 2002, 303, 279, 3200, 220, 16, 24, 19, 15, 82, 26477, 1691, 8882, 321, 10281, 7370, 13, 47758, 43517, 18788, 8765]
IDS_128 = [760, 3712, 314, 23470, 42824, 1599, 22892, 494, 279, 650, 92016, 310, 6278, 48889, 22858, 13, 22044, 21461, 5631, 2878, 1040, 279, 55438, 472, 6348, 1560, 310, 51807, 4584, 43389, 12282, 321, 9297, 310, 6999, 13934, 17943, 13, 561, 94562, 13395, 1452, 279, 2002, 303, 279, 3200, 220, 16, 24, 19, 15, 82, 26477, 1691, 8882, 321, 10281, 7370, 13, 47758, 43517, 18788, 8765, 1179, 11391, 314, 1308, 375, 1052, 8366, 264, 3074, 15911, 13, 10875, 6278, 35378, 6435, 30995, 314, 1308, 375, 1052, 321, 8754, 10895, 303, 14835, 3808, 1599, 34523, 13, 9493, 1452, 13771, 2878, 1040, 68022, 321, 28121, 3440, 2279, 279, 3110, 21110, 303, 19820, 10903, 321, 5484, 6618, 13, 42500, 5757, 15707, 944, 3808, 11168, 314, 12282, 11, 7984, 279, 5281, 2483, 364, 6278]

IDS_BY_LEN = {32: IDS_32, 64: IDS_64, 128: IDS_128}


def _send(cmd, args, timeout=900.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    try:
        s.sendall(P.pack_request(cmd, args))
        raw = P.read_line(s, max_bytes=64 << 20)
    finally:
        s.close()
    if not raw:
        raise RuntimeError("server returned no data (likely died)")
    resp = P.parse_response(raw)
    if resp.type == "error":
        raise RuntimeError(f"server error: {resp.msg}")
    return resp.data or {}


def run_one(seq_len, use_chunked=True):
    payload = {
        "prompt_ids": IDS_BY_LEN[seq_len],
        "mode": "parallel_attn",
    }
    if use_chunked:
        payload["use_chunked_dn"] = True
    data = _send("probe_prefill_vs_decode_loop_tp", payload, timeout=900.0)
    pc = data["per_position_cosine"]
    tag = "CHUNKED" if use_chunked else "v3     "
    print(f"  {tag} seq={seq_len:3d}  ref={data['reference_ms']:5.0f}ms  test={data['test_ms']:5.0f}ms  "
          f"cos_med={pc['median']:.4f}  max_abs={data['max_abs_diff']:.2e}  "
          f"top1={data['top1_agreement']}", flush=True)
    return data


def main():
    print("=== v4 chunked DN precision sweep (REAL English text) ===\n", flush=True)
    print("Warmup (JIT compile, ignored):", flush=True)
    run_one(32, use_chunked=True)
    print()
    print("Measurements (chunked-DN vs v3 baseline at each seq):", flush=True)
    for sl in (32, 64, 128):
        run_one(sl, use_chunked=False)
        run_one(sl, use_chunked=True)


if __name__ == "__main__":
    main()
