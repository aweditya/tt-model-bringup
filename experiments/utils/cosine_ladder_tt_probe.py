#!/usr/bin/env python3
"""
Cosine-ladder TT probe — drives the qb1 server's `cosine_ladder` handler
to teacher-force a fixed token sequence (from the HF reference) and dump
per-position TT logits to disk. Then computes the cosine ladder vs HF.

Two stages:
  1. Send `cosine_ladder` request to server with prompt_ids + generated_ids.
     Server (which already holds device 0) runs the paged forward, captures
     per-step logits to ~/tt-xla/.cache/cosine_ladder_tt_logits.npz, returns
     the path.
  2. Load TT logits + HF reference logits; compute per-position cosine +
     top-1 match. Print + save the ladder.

This avoids the chip-lock contention that prevents independent processes
from opening a second device while the server holds the cluster.

Requires:
  - cosine_ladder_hf_ref.py has been run → ~/tt-xla/.cache/cosine_ladder_hf_ref.npz
  - server.py is running with the `cosine_ladder` handler installed

Run:
    cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/cosine_ladder_tt_probe.py \
        --positions "1,5,10,25,50,75,100"
"""
import argparse
import json
import os
import socket
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/tt-xla"))
from experiments.serve import protocol as P  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

HF_REF_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_hf_ref.npz")
TT_LOGITS_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_tt_logits.npz")
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/cosine_ladder_tt_results.json")


def send_rpc(cmd: str, args: dict, timeout: float = 3600.0) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(P.SOCKET_PATH)
    try:
        sock.sendall(P.pack_request(cmd, args))
        raw = P.read_line(sock, max_bytes=64 << 20)
    finally:
        sock.close()
    if not raw:
        return {"error": "server returned no data"}
    resp = P.parse_response(raw)
    if resp.type == "error":
        return {"error": resp.msg}
    return resp.data or {}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default=HF_REF_PATH)
    p.add_argument("--tt-logits", default=TT_LOGITS_PATH)
    p.add_argument("--max-pos", type=int, default=512)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--positions",
                   default="1,5,10,25,50,75,100,150,200,300,400")
    p.add_argument("--skip-tt", action="store_true",
                   help="Skip running the server probe; reuse existing TT logits.")
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()

    print("=" * 72, flush=True)
    print(f"Cosine-ladder TT probe (via server)", flush=True)
    print(f"  HF ref: {args.ref}", flush=True)
    print("=" * 72, flush=True)

    if not os.path.exists(args.ref):
        print(f"ERROR: HF reference missing at {args.ref}", flush=True)
        sys.exit(2)
    ref = np.load(args.ref, allow_pickle=True)
    prompt_ids = ref["prompt_ids"].astype(int).tolist()
    generated_ids = ref["generated_ids"].astype(int).tolist()
    logits_ref = ref["logits_at_step"]
    P_len = len(prompt_ids)
    M = len(generated_ids)
    VOCAB = logits_ref.shape[1]
    ref_dtype = str(ref["dtype"][0])
    prompt = str(ref["prompt"][0])
    print(f"HF ref:  prompt={prompt!r}  P={P_len}  M={M}  vocab={VOCAB}  "
          f"dtype={ref_dtype}", flush=True)

    # Step 1: drive server's cosine_ladder handler (unless --skip-tt).
    if not args.skip_tt:
        print(f"\nCalling server 'cosine_ladder' handler "
              f"(P+M = {P_len + M} positions, max_pos = {args.max_pos})…",
              flush=True)
        resp = send_rpc("cosine_ladder", {
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "out_path": args.tt_logits,
            "max_pos": args.max_pos,
            "block_size": args.block_size,
        })
        if "error" in resp:
            print(f"server error: {resp['error']}", flush=True)
            sys.exit(3)
        print(f"server result:", flush=True)
        print(json.dumps(resp, indent=2), flush=True)
    else:
        print("--skip-tt: reusing existing TT logits at "
              f"{args.tt_logits}", flush=True)

    # Step 2: load TT logits + compute ladder
    if not os.path.exists(args.tt_logits):
        print(f"ERROR: TT logits missing at {args.tt_logits}", flush=True)
        sys.exit(4)
    tt = np.load(args.tt_logits)
    logits_tt = tt["logits"]
    assert logits_tt.shape == (M, VOCAB), (
        f"shape mismatch: TT={logits_tt.shape}, expected=({M}, {VOCAB})")

    print(f"\n--- COSINE LADDER ---", flush=True)
    print(f"{'pos':>5s}  {'hf_top1':>8s}  {'tt_top1':>8s}  {'mtch':>4s}  "
          f"{'cos_full':>10s}  {'cos_top128':>10s}  "
          f"{'hf_margin':>10s}  {'tt_margin':>10s}", flush=True)
    print("-" * 90, flush=True)

    requested = [int(x.strip()) for x in args.positions.split(",")
                 if x.strip() and 1 <= int(x.strip()) <= M]

    per_pos_cos = np.empty(M, dtype=np.float64)
    per_pos_match = np.zeros(M, dtype=bool)
    for i in range(M):
        per_pos_cos[i] = cosine(logits_tt[i], logits_ref[i])
        per_pos_match[i] = (int(logits_tt[i].argmax()) ==
                             int(logits_ref[i].argmax()))

    ladder_records = []
    for pos in requested:
        i = pos - 1
        hf_top1 = int(logits_ref[i].argmax())
        tt_top1 = int(logits_tt[i].argmax())
        c = per_pos_cos[i]
        idx_top = np.argsort(-np.abs(logits_ref[i]))[:128]
        c128 = cosine(logits_tt[i][idx_top], logits_ref[i][idx_top])
        sorted_hf = np.sort(logits_ref[i])[::-1]
        margin_hf = float(sorted_hf[0] - sorted_hf[1])
        sorted_tt = np.sort(logits_tt[i])[::-1]
        margin_tt = float(sorted_tt[0] - sorted_tt[1])
        mark = "Y" if hf_top1 == tt_top1 else "N"
        print(f"{pos:5d}  {hf_top1:8d}  {tt_top1:8d}  {mark:>4s}  "
              f"{c:10.7f}  {c128:10.7f}  "
              f"{margin_hf:10.4f}  {margin_tt:10.4f}", flush=True)
        ladder_records.append({
            "position": pos,
            "hf_top1_id": hf_top1,
            "tt_top1_id": tt_top1,
            "top1_match": bool(hf_top1 == tt_top1),
            "cos_full_vocab": float(c),
            "cos_top128_by_hf_abs": float(c128),
            "hf_top1_margin": margin_hf,
            "tt_top1_margin": margin_tt,
        })

    thresholds = [0.999, 0.99, 0.9, 0.5]
    first_break = {}
    for thr in thresholds:
        below = np.where(per_pos_cos < thr)[0]
        first_break[str(thr)] = int(below[0] + 1) if len(below) else None
    print(f"\nFirst position where cos drops below threshold:", flush=True)
    for thr, pos in first_break.items():
        print(f"  cos < {thr}: position = {pos}", flush=True)

    print(f"\nTop-1 match rate cumulative:", flush=True)
    for pos in requested:
        rate = float(per_pos_match[:pos].mean())
        print(f"  positions 1..{pos}: {int(per_pos_match[:pos].sum())}/{pos} = "
              f"{rate*100:.1f}% match", flush=True)

    print(f"\nCosine percentiles across all {M} positions:", flush=True)
    for q in [5, 25, 50, 75, 95]:
        v = float(np.percentile(per_pos_cos, q))
        print(f"  p{q}: {v:.7f}", flush=True)
    print(f"  min: {float(per_pos_cos.min()):.7f}  "
          f"max: {float(per_pos_cos.max()):.7f}", flush=True)

    out_obj = {
        "ref_dtype": ref_dtype,
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "M": M,
        "vocab": VOCAB,
        "ladder": ladder_records,
        "first_below_threshold": first_break,
        "per_pos_cos": per_pos_cos.tolist(),
        "per_pos_top1_match": per_pos_match.tolist(),
        "generated_ids_hf": generated_ids,
        "generated_ids_tt": [int(logits_tt[i].argmax()) for i in range(M)],
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
