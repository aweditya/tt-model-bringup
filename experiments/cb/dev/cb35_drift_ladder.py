"""Harness-callable cosine ladder for 35B drift A/B.

Mirrors `experiments/utils/cosine_ladder_35b.py` but takes a
pre-bootstrapped `state` (the dev harness contract) so we iterate in
~30 sec instead of paying the 14-min bootstrap per A/B variant.

Config via env vars (set before `touch trig/drift_ladder` on qb1):

    CB35_DN_DTYPE         bf16 | fp32     default bf16
    CB35_OWNED_GDN        on   | off      default on  (forced off if fp32)
    CB35_OWNED_DECAY_GATE on   | off      default on  (forced off if fp32)
    CB35_LADDER_POSITIONS comma-sep ints  default "0,1,2,5,10"
    CB35_ORACLE_DIR       path             default .cache/hf_oracle_35b_100tok
    CB35_OUT_JSON         path             default .cache/cb35_runtime/drift_ladder_<dtype>_<gdn>_<gate>.json

Critical output: `cos@L32 pos1` printed prominently. That's the
drift-origin metric per `feedback_35b_a3b_l32_dn_decode_drift.md`
(0.9311 for the bf16 baseline; H1 ships if this climbs ≥ 0.99).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ttnn

import experiments.serve.server_35b_ttnn as srv


def _log(msg: str) -> None:
    print(f"[drift_ladder] {msg}", flush=True)


def _cos(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_positions(env_val: str | None) -> list[int]:
    if not env_val:
        return [0, 1, 2, 5, 10]
    out = []
    for x in env_val.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main(state):
    """Harness entry point. `state` is a fully-bootstrapped State."""
    dn_dtype_str = os.environ.get("CB35_DN_DTYPE", "bf16").lower()
    owned_gdn_str = os.environ.get("CB35_OWNED_GDN", "on").lower()
    owned_dg_str = os.environ.get("CB35_OWNED_DECAY_GATE", "on").lower()
    positions = _parse_positions(os.environ.get("CB35_LADDER_POSITIONS"))
    oracle_dir = Path(os.environ.get("CB35_ORACLE_DIR",
                                      ".cache/hf_oracle_35b_100tok"))

    fp32 = (dn_dtype_str == "fp32")
    use_owned_gdn = (owned_gdn_str == "on") and not fp32
    use_owned_dg = (owned_dg_str == "on") and not fp32

    out_default = (f".cache/cb35_runtime/drift_ladder_"
                   f"{dn_dtype_str}_gdn{owned_gdn_str if not fp32 else 'off'}_"
                   f"dg{owned_dg_str if not fp32 else 'off'}.json")
    out_json = Path(os.environ.get("CB35_OUT_JSON", out_default))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    _log(f"config: dn_dtype={dn_dtype_str} owned_gdn={use_owned_gdn} "
         f"owned_decay_gate={use_owned_dg} positions={positions}")
    _log(f"oracle:  {oracle_dir}")
    _log(f"out:     {out_json}")

    if not oracle_dir.exists():
        _log(f"FATAL: oracle dir missing — run experiments/utils/hf_reference_35b.py "
             f"--output-dir {oracle_dir}")
        return 1

    hf_hidden = np.load(oracle_dir / "hidden_states.npy")
    hf_logits = np.load(oracle_dir / "logits.npy")
    hf_argmax = np.load(oracle_dir / "argmax.npy")
    hf_prompt = np.load(oracle_dir / "prompt_ids.npy")
    n_layers = state.text_cfg.num_hidden_layers
    seq_len = hf_prompt.shape[0]
    positions = [p for p in positions if 0 <= p < seq_len]
    _log(f"oracle: seq_len={seq_len} n_layers={n_layers}")

    state.dn_owned_gdn = use_owned_gdn
    state.dn_owned_decay_gate = use_owned_dg
    state.dn_state_dtype = ttnn.float32 if fp32 else ttnn.bfloat16

    _log("reset_caches_ttnn (allocates DN caches in chosen dtype)…")
    state.reset_caches_ttnn()

    # JIT warmup: one forward + sync, then reset before the real ladder.
    _log("warmup forward + sync…")
    _ = srv.step_forward_ttnn(state, int(hf_prompt[0]), 0)
    ttnn.synchronize_device(state.mesh)
    state.reset_caches_ttnn()

    per_pos = []
    L32_pos1_cos = None
    for pos in positions:
        cap = {}
        t0 = time.time()
        tt_next = srv.step_forward_ttnn(state, int(hf_prompt[pos]), pos, capture=cap)
        step_ms = (time.time() - t0) * 1e3

        cos_per_layer = [_cos(cap["embed"], hf_hidden[0, pos])]
        for L in range(n_layers):
            cos_per_layer.append(_cos(cap[f"layer_{L}"], hf_hidden[L + 1, pos]))
        cos_final = _cos(cap["final_norm"], hf_hidden[-1, pos])
        cos_logits = _cos(cap["logits"], hf_logits[pos])
        hf_arg = int(hf_argmax[pos])
        top1 = (tt_next == hf_arg)

        # Index of L=N in cos_per_layer is N+1 (idx 0 is embed).
        cos_L32 = cos_per_layer[32 + 1]
        if pos == 1:
            L32_pos1_cos = cos_L32

        per_pos.append({
            "pos": pos,
            "tt_argmax": int(tt_next),
            "hf_argmax": hf_arg,
            "top1_match": bool(top1),
            "cos_per_layer": [round(c, 6) for c in cos_per_layer],
            "cos_L32": round(cos_L32, 6),
            "cos_final_norm": round(cos_final, 6),
            "cos_logits": round(cos_logits, 6),
            "step_ms": round(step_ms, 1),
        })
        _log(f"  pos={pos:3d} tt={tt_next:6d} hf={hf_arg:6d} match={'Y' if top1 else 'N'} "
             f"cos_L32={cos_L32:.4f} cos_final={cos_final:.4f} step={step_ms:.0f}ms")

    summary = {
        "config": {
            "dn_dtype": dn_dtype_str,
            "owned_gdn": owned_gdn_str,
            "owned_decay_gate": owned_dg_str,
            "effective": {
                "dn_state_dtype": "fp32" if fp32 else "bf16",
                "use_owned_gdn": use_owned_gdn,
                "use_owned_decay_gate": use_owned_dg,
            },
        },
        "oracle_dir": str(oracle_dir),
        "positions": positions,
        "L32_pos1_cos": L32_pos1_cos,
        "per_pos": per_pos,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {out_json}")

    # The headline number — bf16 baseline is 0.9311; H1 ships if ≥ 0.99.
    if L32_pos1_cos is not None:
        verdict = "PASS" if L32_pos1_cos >= 0.99 else \
                  "PARTIAL" if L32_pos1_cos > 0.95 else "NO-MOVE"
        _log("=" * 60)
        _log(f"HEADLINE: cos@L32 pos 1 = {L32_pos1_cos:.4f}  [{verdict}]")
        _log(f"          (bf16 baseline = 0.9311; H1 target ≥ 0.99)")
        _log("=" * 60)
    return 0
