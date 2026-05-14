#!/usr/bin/env python3
"""
P21 — fp32 SDPA cliff probe.

Tests whether running attention with higher-precision compute_kernel_config
flags pushes the bf16 prefill drift cliff past position 129. The cliff was
mechanistically pinned by Agent Q in feedback_bf16_prefill_drift_cliff.md:
teacher-forced cosine vs HF bf16 oracle falls off a cliff at pos 129
(cos 0.8517 vs 0.9997 at pos 125).

The hypothesis under test: enabling more aggressive fp32 accumulation flags
inside the SDPA decode kernel — specifically `packer_l1_acc=True` plus
candidate variants from llama3_70b_galaxy and deepseek_v3 references —
will push the cliff later.

NOTE: per feedback_fp32_kv_cache.md, fp32 KV storage with bf16 typecast at
read-time gives ZERO benefit (paged SDPA decode hard-rejects fp32 input).
The remaining lever in pure Python is the SDPA kernel's compute_kernel_config.

Variants (matched to 91f._p21_make_sdpa_cfg):
  A   : production default — HiFi4 + fp32_dest_acc_en=True
  B   : A + packer_l1_acc=True            (70b galaxy general-matmul hifi4)
  B2  : A explicit                        (70b galaxy compute_kernel_config_sdpa)
  B3  : HiFi2 + no fp32_dest_acc          (70b galaxy SDPA_DECODE_COMPUTE_PROGCFG; counter-test)
  B4  : B + dst_full_sync_en=True

Run (on qb1, server up):
    cd ~/tt-xla && .venv/bin/python -u \
        experiments/utils/p21_fp32_sdpa_cliff_probe.py

Outputs (under ~/tt-xla/.cache/p21_fp32_sdpa_probe/):
  - logits_<variant>.npz   : per-position TT logits, one file per variant
  - results.json           : combined ladder summary across variants
  - probe.log              : stdout transcript
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/tt-xla"))

# Make stdout line-buffered so SSH-piped output streams cleanly.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from experiments.serve import protocol as P  # noqa: E402

HF_REF_PATH = os.path.expanduser(
    "~/tt-xla/.cache/cosine_ladder_v2/cosine_ladder_hf_ref_500.npz"
)
PROBE_DIR = os.path.expanduser("~/tt-xla/.cache/p21_fp32_sdpa_probe")
SENTINEL_PATH = os.path.expanduser("~/tt-xla/.cache/p21_sdpa_variant.txt")

# Variants we exercise. Order matters: A first (so log shows current baseline).
VARIANTS = ["A", "B", "B2", "B3", "B4"]

# Positions we report in the ladder. These bracket Agent Q's known cliff (129).
REPORT_POSITIONS = [1, 10, 50, 100, 125, 128, 129, 130, 140, 150, 200, 250, 300, 400, 500]


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


def write_sentinel(variant: str) -> None:
    os.makedirs(os.path.dirname(SENTINEL_PATH), exist_ok=True)
    with open(SENTINEL_PATH, "w") as f:
        f.write(variant.strip().upper() + "\n")


def cliff_position(per_pos_cos: np.ndarray, thr: float) -> int:
    """Return the first 1-indexed position where cos < thr (or 0 if never)."""
    below = np.where(per_pos_cos < thr)[0]
    return int(below[0] + 1) if len(below) else 0


def run_variant(variant: str, prompt_ids, generated_ids, logits_ref,
                max_pos: int, block_size: int) -> dict:
    out_path = os.path.join(PROBE_DIR, f"logits_{variant}.npz")
    print(f"\n{'='*72}", flush=True)
    print(f"VARIANT {variant}", flush=True)
    print(f"{'='*72}", flush=True)

    # 1) Flip sentinel so the reloaded _91f picks up the variant.
    write_sentinel(variant)
    print(f"  sentinel = {variant} ({SENTINEL_PATH})", flush=True)

    # 2) Trigger reload of _91f on the server. This re-imports the module
    #    from disk, which re-evaluates `sdpa_kcfg = _p21_resolve_sdpa_cfg()`.
    t0 = time.time()
    resp = send_rpc("reload_kernels", {})
    print(f"  reload_kernels → {resp}  ({(time.time()-t0)*1000:.1f} ms)",
          flush=True)
    if "error" in resp:
        return {"variant": variant, "error": resp["error"]}

    # 3) Run cosine_ladder on the server. The handler writes TT logits to
    #    out_path and returns timings.
    t0 = time.time()
    resp = send_rpc("cosine_ladder", {
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "out_path": out_path,
        "max_pos": max_pos,
        "block_size": block_size,
    })
    wall_s = time.time() - t0
    print(f"  cosine_ladder → wall {wall_s:.1f}s", flush=True)
    if "error" in resp:
        return {"variant": variant, "error": resp["error"], "wall_s": wall_s}
    print(f"  server response:", flush=True)
    print(f"  {json.dumps(resp, indent=2)}", flush=True)

    # 4) Compute cliff. Compare TT logits to HF bf16 ref.
    tt = np.load(out_path)
    logits_tt = tt["logits"]
    assert logits_tt.shape == logits_ref.shape, (
        f"shape mismatch tt={logits_tt.shape} ref={logits_ref.shape}")
    M = logits_tt.shape[0]
    per_pos_cos = np.empty(M, dtype=np.float64)
    per_pos_match = np.zeros(M, dtype=bool)
    for i in range(M):
        per_pos_cos[i] = cosine(logits_tt[i], logits_ref[i])
        per_pos_match[i] = (int(logits_tt[i].argmax()) ==
                             int(logits_ref[i].argmax()))

    # 5) Print the ladder.
    print(f"\n  --- ladder vs HF bf16 ---", flush=True)
    print(f"  {'pos':>5s}  {'cos':>10s}  {'top1?':>6s}", flush=True)
    print(f"  " + "-" * 30, flush=True)
    rec_rows = []
    for pos in REPORT_POSITIONS:
        if pos > M:
            continue
        i = pos - 1
        c = per_pos_cos[i]
        m = "Y" if per_pos_match[i] else "N"
        print(f"  {pos:5d}  {c:10.7f}  {m:>6s}", flush=True)
        rec_rows.append({
            "pos": pos,
            "cos": float(c),
            "top1_match": bool(per_pos_match[i]),
            "hf_top1": int(logits_ref[i].argmax()),
            "tt_top1": int(logits_tt[i].argmax()),
        })

    cliff_099 = cliff_position(per_pos_cos, 0.99)
    cliff_09 = cliff_position(per_pos_cos, 0.9)
    cliff_05 = cliff_position(per_pos_cos, 0.5)
    print(f"\n  cliff @ cos < 0.99 : pos {cliff_099 or 'never'}", flush=True)
    print(f"  cliff @ cos < 0.9  : pos {cliff_09 or 'never'}", flush=True)
    print(f"  cliff @ cos < 0.5  : pos {cliff_05 or 'never'}", flush=True)

    cum_match_150 = float(per_pos_match[:min(150, M)].mean())
    cum_match_200 = float(per_pos_match[:min(200, M)].mean())
    cum_match_500 = float(per_pos_match[:M].mean())
    print(f"  top-1 match 1..150 : {cum_match_150*100:.1f}%", flush=True)
    print(f"  top-1 match 1..200 : {cum_match_200*100:.1f}%", flush=True)
    print(f"  top-1 match 1..{M:d} : {cum_match_500*100:.1f}%", flush=True)

    return {
        "variant": variant,
        "wall_s": wall_s,
        "server_reply": resp,
        "out_path": out_path,
        "cliff": {
            "cos_lt_0.99": cliff_099,
            "cos_lt_0.9": cliff_09,
            "cos_lt_0.5": cliff_05,
        },
        "top1_match_rate": {
            "1..150": cum_match_150,
            "1..200": cum_match_200,
            f"1..{M}": cum_match_500,
        },
        "ladder": rec_rows,
        "per_pos_cos": per_pos_cos.tolist(),
        "per_pos_match": per_pos_match.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="Comma-separated subset of variants to run "
                         "(default: all)")
    ap.add_argument("--ref", default=HF_REF_PATH)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--restore-A-at-end", action="store_true", default=True,
                    help="Reset the sentinel back to A after the probe so "
                         "the persistent server returns to production config")
    args = ap.parse_args()

    os.makedirs(PROBE_DIR, exist_ok=True)

    if not os.path.exists(args.ref):
        print(f"ERROR: HF reference missing at {args.ref}", flush=True)
        sys.exit(2)
    ref = np.load(args.ref, allow_pickle=True)
    prompt_ids = ref["prompt_ids"].astype(int).tolist()
    generated_ids = ref["generated_ids"].astype(int).tolist()
    logits_ref = ref["logits_at_step"]
    P_len = len(prompt_ids)
    M = len(generated_ids)
    print(f"HF ref: P={P_len}  M={M}  vocab={logits_ref.shape[1]}  "
          f"prompt={str(ref['prompt'][0])!r}", flush=True)
    print(f"Sentinel: {SENTINEL_PATH}", flush=True)
    print(f"Output dir: {PROBE_DIR}", flush=True)

    requested = [v.strip().upper() for v in args.variants.split(",") if v.strip()]
    all_results = []
    t_total = time.time()
    for v in requested:
        try:
            r = run_variant(v, prompt_ids, generated_ids, logits_ref,
                            args.max_pos, args.block_size)
        except Exception as e:
            r = {"variant": v, "exception": repr(e)}
            print(f"  EXCEPTION: {e!r}", flush=True)
        all_results.append(r)

    total_wall = time.time() - t_total
    print(f"\n{'='*72}", flush=True)
    print(f"SUMMARY (total wall {total_wall:.1f}s)", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"\n{'variant':>8s}  {'cliff<0.99':>12s}  {'cliff<0.9':>12s}  "
          f"{'cliff<0.5':>12s}  {'top1@200':>10s}  {'wall_s':>8s}", flush=True)
    print(f"{'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}", flush=True)
    def _fmt(v):
        return "never" if not v else str(v)

    for r in all_results:
        v = r.get("variant", "?")
        if "error" in r or "exception" in r:
            print(f"{v:>8s}  ERROR/EXCEPTION  {r.get('error') or r.get('exception')}",
                  flush=True)
            continue
        c = r["cliff"]
        m200 = r["top1_match_rate"]["1..200"]
        ws = r.get("wall_s", 0.0)
        print(f"{v:>8s}  {_fmt(c['cos_lt_0.99']):>12s}  "
              f"{_fmt(c['cos_lt_0.9']):>12s}  "
              f"{_fmt(c['cos_lt_0.5']):>12s}  "
              f"{m200*100:9.1f}%  {ws:8.1f}", flush=True)

    # Save combined results.
    summary_path = os.path.join(PROBE_DIR, "results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "ref": args.ref,
            "max_pos": args.max_pos,
            "block_size": args.block_size,
            "M": M,
            "P": P_len,
            "variants_requested": requested,
            "results": all_results,
            "total_wall_s": total_wall,
        }, f, indent=2)
    print(f"\nSaved → {summary_path}", flush=True)

    if args.restore_A_at_end:
        write_sentinel("A")
        print(f"Sentinel restored to A (production default).", flush=True)
        # And reload so the live server picks up the production default too.
        resp = send_rpc("reload_kernels", {})
        print(f"Final reload_kernels → {resp}", flush=True)


if __name__ == "__main__":
    main()
