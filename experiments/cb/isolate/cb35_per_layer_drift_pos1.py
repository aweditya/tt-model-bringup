"""Per-layer h cosine ladder at the 35B pos1→pos5 cliff — bisect L_locus.

Step 3 of `research/35b_drift_briefing_2026-06-04.md`. The cliff is
`cos_L32: 0.99 @ pos 1 → 0.32 @ pos 5` with `owned_gdn=ON`. This probe
captures the per-layer hidden state at pos 0 (PASS reference), pos 1
(just-before-cliff), pos 3 (mid-cliff), pos 5 (post-cliff) and compares
each to HF `hidden_states[L+1, pos, :]` to find the smallest L where
TT diverges from HF.

Fork of `experiments/cb/isolate/gm4_per_layer_drift_pos1.py` (the
working Gemma 4 pattern) targeting the 35B server + long oracle.

Differences from the Gemma 4 fork:
- `server_35b_ttnn.step_forward_ttnn` already writes
  `capture[f"layer_{L}"]` per layer (no server diff required).
- Walk positions SEQUENTIALLY (0..max) because the DN recurrence at
  pos N depends on state from pos 0..N-1. Only RECORD at the positions
  of interest.
- Reset DN+KV caches up front + after JIT warmup (matches
  `cb35_drift_ladder.py`).
- Force `owned_gdn=ON` and `dn_state_dtype=bf16` — the manual
  recurrence path is broken (cos@L32 pos 0 = 0.08) per
  `[[feedback-35b-manual-recurrence-path-broken]]`.

Harness usage (qb1):
    1. Deploy: `bash scripts/deploy.sh experiments/cb/isolate/cb35_per_layer_drift_pos1.py`
    2. Trigger: `touch ~/tt-xla/.cache/cb35_runtime/trig/per_layer_drift_pos1`
    3. Log: `tail -f ~/tt-xla/.cache/cb35_runtime/harness.log`

Env (all optional):
    CB35_ORACLE_DIR        default .cache/hf_oracle_35b_long
    CB35_PROBE_POSITIONS   default "0,1,3,5"  (must include 0; max bounds walk)
    CB35_CLIFF_THRESHOLD   default 0.95       (cos below this = cliff)
    CB35_OUT_JSON          default .cache/cb35_runtime/per_layer_drift_pos1.json

Expected outcomes (per the briefing):
1. pos 0 stays cos > 0.99 across all 40 layers (this is the known
   bit-clean reference); pos 1 also clean (cos_L32 ≈ 0.99).
2. pos 5 shows a sharp cliff at some L_locus; first L where
   cos_per_layer[L+1] < CB35_CLIFF_THRESHOLD.
3. pos 3 lands somewhere on the slope — pinpoints whether the cliff
   is positional (sharp) or accumulating (gradual).

If `L_locus` is reported, the next investigation is a sub-op probe at
`(L_locus, P_cliff)` using `step_forward_ttnn`'s `sub_capture_layers`
hook (see briefing §3 Step 3).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import ttnn

# Harness adds <PROJECT_ROOT>/experiments/serve to sys.path; bare module
# name matches the convention used by `cb35_drift_ladder.py`.
import server_35b_ttnn as srv


def _log(msg: str) -> None:
    print(f"[per_layer_drift] {msg}", flush=True)


def _cos(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_positions(env_val: str | None) -> list[int]:
    if not env_val:
        return [0, 1, 3, 5]
    out = []
    for x in env_val.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    out = sorted(set(out))
    if 0 not in out:
        out = [0] + out
    return out


def main(state):
    """Harness entry point. `state` is a pre-bootstrapped 35B State."""
    oracle_dir = Path(os.environ.get("CB35_ORACLE_DIR",
                                      ".cache/hf_oracle_35b_long"))
    positions = _parse_positions(os.environ.get("CB35_PROBE_POSITIONS"))
    cliff_thr = float(os.environ.get("CB35_CLIFF_THRESHOLD", "0.95"))
    out_json = Path(os.environ.get("CB35_OUT_JSON",
                                    ".cache/cb35_runtime/per_layer_drift_pos1.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not oracle_dir.exists():
        _log(f"FATAL: oracle dir missing — run "
             f"experiments/utils/hf_reference_35b.py --output-dir {oracle_dir}")
        return 1

    hf_hidden = np.load(oracle_dir / "hidden_states.npy")     # [N+1, seq, HIDDEN]
    hf_logits = np.load(oracle_dir / "logits.npy")             # [seq, VOCAB]
    hf_argmax = np.load(oracle_dir / "argmax.npy")             # [seq]
    hf_prompt = np.load(oracle_dir / "prompt_ids.npy")         # [seq]
    n_layers = state.text_cfg.num_hidden_layers
    seq_len = hf_prompt.shape[0]
    positions = [p for p in positions if 0 <= p < seq_len]

    _log(f"config: positions={positions} cliff_thr={cliff_thr}")
    _log(f"oracle: {oracle_dir} seq_len={seq_len} n_layers={n_layers} "
         f"hidden_states={hf_hidden.shape}")
    _log(f"out:    {out_json}")

    # Force the known-good kernel path. The manual recurrence else-branch
    # is broken (cos@L32 pos 0 = 0.08); fp32 H_t auto-disables owned_gdn
    # → routes through the broken path. Pin to bf16 + owned kernels.
    state.dn_owned_gdn = True
    state.dn_owned_decay_gate = True
    state.dn_state_dtype = ttnn.bfloat16

    _log("reset_caches_ttnn (allocates DN caches in bf16)…")
    state.reset_caches_ttnn()

    # JIT warmup: one forward to compile, then reset to clean state.
    _log("warmup forward + sync…")
    _ = srv.step_forward_ttnn(state, int(hf_prompt[0]), 0)
    ttnn.synchronize_device(state.mesh)
    state.reset_caches_ttnn()

    # Walk 0..max(positions) SEQUENTIALLY — DN recurrence at pos N needs
    # state produced by walking pos 0..N-1. Only RECORD at the targets.
    record_set = set(positions)
    walk_end = max(positions) + 1
    captures: dict[int, dict] = {}
    for pos in range(walk_end):
        cap: dict | None = {} if pos in record_set else None
        t0 = time.time()
        tt_next = srv.step_forward_ttnn(state, int(hf_prompt[pos]), pos, capture=cap)
        step_ms = (time.time() - t0) * 1e3
        if cap is not None:
            cap["_tt_next"] = int(tt_next)
            cap["_step_ms"] = step_ms
            captures[pos] = cap
            _log(f"  recorded pos={pos:3d} tt_next={tt_next:6d} "
                 f"hf={int(hf_argmax[pos]):6d} step={step_ms:.0f}ms")

    # ── Per-layer cos ladder. For each captured position, walk layers,
    #     compute cos(TT[L], HF[L+1, pos]), tag the first L below threshold.
    _log("=" * 96)
    _log("per-layer h cosine ladder — TT[layer_L] vs HF[L+1, pos, :]")
    header = "L  | type |" + " | ".join(f" pos{p} cos  pass" for p in positions) + " |"
    _log(header)
    _log("-" * len(header))

    # cliff_per_pos[pos] = first L where cos < cliff_thr (None if never).
    cliff_per_pos: dict[int, int | None] = {p: None for p in positions}
    per_layer_rows: list[dict] = []

    # Anchor row: embed (HF index 0). Useful for visual continuity, not for
    # cliff detection.
    embed_row = {"L": -1, "type": "EMB"}
    for p in positions:
        c = _cos(captures[p]["embed"], hf_hidden[0, p])
        embed_row[f"cos_pos{p}"] = c
    per_layer_rows.append(embed_row)
    line = "E  | EMB  |"
    for p in positions:
        c = embed_row[f"cos_pos{p}"]
        line += f" {c:.5f} {'PASS' if c >= cliff_thr else 'FAIL'} |"
    _log(line)

    for L in range(n_layers):
        lt = state.layer_types[L]
        lt_short = "DN" if lt == "linear_attention" else "AT"
        row = {"L": L, "type": lt_short}
        line = f"L{L:2d} | {lt_short}   |"
        for p in positions:
            c = _cos(captures[p][f"layer_{L}"], hf_hidden[L + 1, p])
            row[f"cos_pos{p}"] = c
            passed = c >= cliff_thr
            tag = "PASS" if passed else "FAIL"
            line += f" {c:.5f} {tag} |"
            if (not passed) and cliff_per_pos[p] is None:
                cliff_per_pos[p] = L
        # delta vs pos 0 for the LAST recorded position — quick visual.
        if 0 in row and positions[-1] != 0:
            pass
        per_layer_rows.append(row)
        # Mark the row that contains the FIRST cliff for any position.
        first_cliffs_here = [p for p in positions
                             if cliff_per_pos[p] == L]
        if first_cliffs_here:
            line += f"  ← FIRST CLIFF pos{first_cliffs_here}"
        _log(line)

    # Final norm + logits row.
    fn_row = {"L": n_layers, "type": "FN"}
    line = "FN | FN   |"
    for p in positions:
        c = _cos(captures[p]["final_norm"], hf_hidden[-1, p])
        fn_row[f"cos_pos{p}"] = c
        line += f" {c:.5f} {'PASS' if c >= cliff_thr else 'FAIL'} |"
    per_layer_rows.append(fn_row)
    _log(line)

    lg_row = {"L": n_layers + 1, "type": "LG"}
    line = "LG | LG   |"
    for p in positions:
        c = _cos(captures[p]["logits"], hf_logits[p])
        lg_row[f"cos_pos{p}"] = c
        line += f" {c:.5f} {'PASS' if c >= cliff_thr else 'FAIL'} |"
    per_layer_rows.append(lg_row)
    _log(line)
    _log("=" * 96)

    # ── Verdict.
    headline_pos = next((p for p in positions if p != 0), positions[-1])
    L_locus = cliff_per_pos.get(headline_pos)
    delta_pos0_pos_headline = None
    if headline_pos != 0:
        last_layer_row = per_layer_rows[-3]   # last layer (-2 = FN, -1 = LG)
        delta_pos0_pos_headline = (
            last_layer_row.get("cos_pos0", float("nan")) -
            last_layer_row.get(f"cos_pos{headline_pos}", float("nan"))
        )

    _log(f"VERDICT @ cliff_thr={cliff_thr}:")
    for p in positions:
        Lp = cliff_per_pos[p]
        if Lp is None:
            _log(f"  pos {p}: NO CLIFF — all 40 layers cos >= {cliff_thr}")
        else:
            lt = state.layer_types[Lp]
            _log(f"  pos {p}: FIRST CLIFF at L{Lp} ({lt})")
    if delta_pos0_pos_headline is not None:
        _log(f"  Δ(pos0 - pos{headline_pos}) at final layer = "
             f"{delta_pos0_pos_headline:+.4f}")

    if L_locus is not None:
        lt = state.layer_types[L_locus]
        _log("")
        _log(f"NEXT STEP: sub-op probe at (L={L_locus}, pos={headline_pos}). "
             f"Layer type = {lt}. Use step_forward_ttnn capture with "
             f"sub_capture_layers=[{L_locus}] — populates "
             f"capture['layer_{L_locus}_sub'] with attn/MoE/DN sub-step "
             f"arrays. Compare to HF intra-layer hooks "
             f"(hf_reference_35b.py --hook-attn-layer {L_locus} or "
             f"--hook-dn-layer {L_locus}).")

    # ── Write JSON.
    summary = {
        "config": {
            "oracle_dir": str(oracle_dir),
            "positions": positions,
            "cliff_threshold": cliff_thr,
            "owned_gdn": True,
            "owned_decay_gate": True,
            "dn_state_dtype": "bf16",
        },
        "n_layers": n_layers,
        "layer_types": list(state.layer_types),
        "tt_next_per_pos": {p: int(captures[p]["_tt_next"]) for p in positions},
        "hf_argmax_per_pos": {p: int(hf_argmax[p]) for p in positions},
        "cliff_per_pos": {p: cliff_per_pos[p] for p in positions},
        "per_layer_rows": [
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in r.items()}
            for r in per_layer_rows
        ],
    }
    out_json.write_text(json.dumps(summary, indent=2))
    _log(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    # Allows local syntax-check; real run is via the dev harness.
    _log("ERROR: must be invoked via the cb35 dev harness "
         "(needs a pre-bootstrapped State). "
         "See module docstring for usage.")
    sys.exit(2)
