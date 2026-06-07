#!/usr/bin/env python3
"""Drafter v0.2 — full 4-layer forward smoke vs HF oracle.

Forks `experiments/cb/isolate/gemma4_assistant_embed_smoke.py` (v0.1 probe)
+ `experiments/cb/isolate/gm4_v01_L0_cos.py` (target's per-layer validator).

For each of the 5 oracle prompts (`hf_oracle_gemma4_12b_assistant/prompt_*`):
  1. Load drafter_inputs_embeds.npy + shared_kv_{sliding,full}_{K,V}.npy.
  2. Run `srv.drafter_forward(state, inputs_embeds, shared_kv_states)`.
  3. Gate:
     - logits cos vs drafter_logits.npy >= 0.999
     - hidden cos vs drafter_hidden.npy >= 0.999
     - argmax exact match vs drafter_argmax.npy on >= 4/5 prompts
       (5th may flip if it's a bf16 near-tie — top-K is recorded in
       drafter_topk_ids.npy for diagnosis).

Run on qb2:
    ssh qb2 'cd ~/tt-xla && .venv/bin/python -u \
        experiments/cb/isolate/gemma4_assistant_forward_smoke.py'
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_12b_assistant_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"
COS_THRESH = 0.999
NUM_PROMPTS = 5
MIN_ARGMAX_MATCHES = 4  # 4/5 prompts must argmax-match exactly.


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def mad(a, b):
    return float(np.abs(
        a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)
    ).max())


def load_prompt(i: int):
    base = ORACLE_DIR / f"prompt_{i}"
    return {
        "input_ids": np.load(base / "input_ids.npy"),
        "drafter_inputs_embeds": np.load(base / "drafter_inputs_embeds.npy"),
        "shared_kv_sliding_K": np.load(base / "shared_kv_sliding_K.npy"),
        "shared_kv_sliding_V": np.load(base / "shared_kv_sliding_V.npy"),
        "shared_kv_full_K": np.load(base / "shared_kv_full_K.npy"),
        "shared_kv_full_V": np.load(base / "shared_kv_full_V.npy"),
        "drafter_hidden": np.load(base / "drafter_hidden.npy"),
        "drafter_logits": np.load(base / "drafter_logits.npy"),
        "drafter_argmax": np.load(base / "drafter_argmax.npy"),
        "drafter_topk_ids": np.load(base / "drafter_topk_ids.npy"),
    }


def main() -> int:
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        return 1

    log("bootstrapping drafter (~30 sec)…")
    t0 = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t0:.1f}s")

    results = []
    for i in range(NUM_PROMPTS):
        log("=" * 64)
        log(f"prompt {i}")
        log("=" * 64)
        d = load_prompt(i)
        L_kv = d["shared_kv_sliding_K"].shape[2]
        log(f"  input_ids L={d['input_ids'].shape[1]}  L_kv={L_kv}")
        log(f"  drafter_inputs_embeds shape={d['drafter_inputs_embeds'].shape}")
        log(f"  shared_kv_sliding K shape={d['shared_kv_sliding_K'].shape}")
        log(f"  shared_kv_full K shape={d['shared_kv_full_K'].shape}")
        log(f"  HF argmax={int(d['drafter_argmax'].flatten().item())}, "
            f"top4={d['drafter_topk_ids'].flatten().tolist()[:4]}")

        shared_kv_states = {
            "sliding_attention": (d["shared_kv_sliding_K"],
                                   d["shared_kv_sliding_V"]),
            "full_attention":    (d["shared_kv_full_K"],
                                   d["shared_kv_full_V"]),
        }
        log("  running drafter_forward…")
        t_fw = time.time()
        out = srv.drafter_forward(state, d["drafter_inputs_embeds"],
                                   shared_kv_states)
        fw_s = time.time() - t_fw
        log(f"  drafter_forward wall: {fw_s*1000:.1f} ms")

        c_hidden = cos(out["hidden"], d["drafter_hidden"])
        m_hidden = mad(out["hidden"], d["drafter_hidden"])
        c_logits = cos(out["logits"], d["drafter_logits"])
        m_logits = mad(out["logits"], d["drafter_logits"])
        tt_argmax = int(out["argmax"].flatten().item())
        hf_argmax = int(d["drafter_argmax"].flatten().item())
        argmax_match = tt_argmax == hf_argmax
        topk = d["drafter_topk_ids"].flatten().tolist()
        in_topk = tt_argmax in topk
        log(f"  hidden cos={c_hidden:.6f} mad={m_hidden:.6f}")
        log(f"  logits cos={c_logits:.6f} mad={m_logits:.6f}")
        log(f"  TT argmax={tt_argmax}  HF argmax={hf_argmax}  "
            f"exact={'YES' if argmax_match else 'NO'}  in_top8={'YES' if in_topk else 'NO'}")
        results.append({
            "i": i, "cos_hidden": c_hidden, "mad_hidden": m_hidden,
            "cos_logits": c_logits, "mad_logits": m_logits,
            "tt_argmax": tt_argmax, "hf_argmax": hf_argmax,
            "argmax_match": argmax_match, "in_top8": in_topk,
            "wall_ms": fw_s * 1000.0,
        })

    # Summary
    log("")
    log("=" * 64)
    log("SUMMARY")
    log("=" * 64)
    cos_h_ok = sum(1 for r in results if r["cos_hidden"] >= COS_THRESH)
    cos_l_ok = sum(1 for r in results if r["cos_logits"] >= COS_THRESH)
    argmax_ok = sum(1 for r in results if r["argmax_match"])
    in_topk_ok = sum(1 for r in results if r["in_top8"])
    mean_wall = float(np.mean([r["wall_ms"] for r in results]))
    log(f"  hidden cos >= {COS_THRESH}: {cos_h_ok}/{NUM_PROMPTS}")
    log(f"  logits cos >= {COS_THRESH}: {cos_l_ok}/{NUM_PROMPTS}")
    log(f"  argmax exact match:        {argmax_ok}/{NUM_PROMPTS}")
    log(f"  argmax in HF top-8:        {in_topk_ok}/{NUM_PROMPTS}")
    log(f"  mean forward wall: {mean_wall:.1f} ms")
    for r in results:
        log(f"    p{r['i']}: cos_h={r['cos_hidden']:.4f} cos_l={r['cos_logits']:.4f} "
            f"tt={r['tt_argmax']} hf={r['hf_argmax']} "
            f"{'OK' if r['argmax_match'] else ('topK' if r['in_top8'] else 'MISS')}")

    overall = (cos_h_ok == NUM_PROMPTS
               and cos_l_ok == NUM_PROMPTS
               and argmax_ok >= MIN_ARGMAX_MATCHES)
    log(f"  overall: {'PASS' if overall else 'FAIL'}")

    import ttnn
    ttnn.close_mesh_device(state.mesh)
    log("mesh closed.")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
