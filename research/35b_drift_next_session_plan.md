# 35B drift — next session plan

**Session-handoff scaffolding for task #163.**

We spent ~4 hours today getting to this picture:

- **H1** (DN H_t bf16 round-trip per step, Ollama precedent `ollama#15865`):
  fix shipped via `TT_DN_STATE_DTYPE=fp32` env hook in commit `92b442f`.
  Mechanically correct (server boots, generates tokens), but **drift
  symptom UNCHANGED**. H1 alone is not the lever.
- **fp32 residual stream** extension on top of H1 (commits
  `35ea58f` + `7c3ede6` + `1c650b7`) — hung in `engine.start()` warmup
  on two consecutive runs. **Reverted** in `5c5228c`.
- **Critical lesson** (memory: `feedback_use_dev_harness_for_iteration.md`):
  we burned 4× 14-min full-server bootstraps when the dev harness
  (`scripts/run_harness_tmux.sh qb1`) iterates in ~30 sec on the
  resident model. Non-negotiable #1 ("think first") was violated.

## State to pick up — ALL READY

- ✅ Server-side: `TT_DN_STATE_DTYPE=fp32` env hook works for opt-in
  fp32 H_t with manual recurrence path. Default bf16 unchanged.
- ✅ HF oracles BOTH generated and on qb1:
  - `.cache/hf_oracle_35b_100tok/` — 5 positions ("The capital of
    France is"), enough for the cos@L32 pos 1 headline question.
  - `.cache/hf_oracle_35b_long/` — 85 positions (math ladder
    prompt), covers the long-context drift regime.
- ✅ Dev harness scaffold: `scripts/run_harness_tmux.sh qb1` running
  in tmux session `cb35`; trigger files in
  `~/tt-xla/.cache/cb35_runtime/trig/` map to
  `experiments/cb/{validate,isolate,bench,dev}/cb35_<name>.py` and
  iterate in ~30 sec on resident state.
- ✅ **6 probe wrappers DEPLOYED** to qb1:
  - `cb35_drift_bf16` / `cb35_drift_fp32_h` / `cb35_drift_fp32_h_no_dg`
    (5-position oracle — fast headline)
  - `cb35_drift_long_bf16` / `cb35_drift_long_fp32_h` /
    `cb35_drift_long_fp32_h_no_dg` (85-position oracle, full ladder)
  - All wrap `cb35_drift_ladder.main(state)` with env config.
  - Each prints `HEADLINE: cos@L32 pos 1 = X.XXXX [PASS|PARTIAL|NO-MOVE]`
    and writes a JSON to `.cache/cb35_runtime/drift_ladder_*.json`.

## GO-button — once `cb35` harness is ready

Check harness status:
```bash
ssh qb1 'tail -3 ~/tt-xla/.cache/cb35_runtime/harness.log'
# Expect "[harness] ready. Drop trigger files into …"
```

When ready, fire the 3 short probes back-to-back (each ~30 sec):
```bash
# H0 baseline (memory expectation: cos@L32 pos 1 = 0.9311)
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_bf16'
sleep 35
ssh qb1 'cat tt-xla/.cache/cb35_runtime/trig/last.log | tail -20'

# H1: fp32 H_t  (headline question — does cos climb ≥ 0.99?)
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_fp32_h'
sleep 35
ssh qb1 'cat tt-xla/.cache/cb35_runtime/trig/last.log | tail -20'

# H1+: fp32 H_t + owned_decay_gate off
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_fp32_h_no_dg'
sleep 35
ssh qb1 'cat tt-xla/.cache/cb35_runtime/trig/last.log | tail -20'
```

Then the 3 long-context variants:
```bash
for v in drift_long_bf16 drift_long_fp32_h drift_long_fp32_h_no_dg; do
  ssh qb1 "touch tt-xla/.cache/cb35_runtime/trig/$v"
  sleep 60
  ssh qb1 "cat tt-xla/.cache/cb35_runtime/trig/last.log | tail -20"
done
```

The JSON files left behind under `.cache/cb35_runtime/`:
```
drift_ladder_bf16_gdnon_dgon.json     ← H0 short
drift_ladder_fp32_gdnoff_dgon.json    ← H1 short
drift_ladder_fp32_gdnoff_dgoff.json   ← H1+ short
drift_ladder_*_long.*.json (similar)  ← full ladder runs
```

Each has per-position `cos_per_layer`, `cos_L32`, `cos_final_norm`,
`cos_logits`, and a top-level `L32_pos1_cos` for the headline.

## Decision tree after the headline

| `H0 cos@L32 pos1` | `H1 cos@L32 pos1` | Verdict |
|---|---|---|
| ≈ 0.9311 | ≥ 0.99 | **H1 confirmed.** Long-context probe should show top-1 lift too. Plumb to server, ship. |
| ≈ 0.9311 | 0.93–0.97 | **H1 partial.** Decay+state mixed. Try `_no_dg` variant; consider H3 (decay-only fp32). |
| ≈ 0.9311 | ≈ 0.9311 (unchanged) | **H1 dead.** Move to H3 (decay alone), H4 (RMSNorm), H5 (conv1d). New probe per hypothesis. |
| Doesn't match memory | — | Suspect oracle mismatch (chat-template vs raw, wrong prompt). Verify oracle path. |

## Hypotheses to test (in priority order, all via the harness)

| H | What | Probe |
|---|---|---|
| **H1 alone insufficient** | Confirmed already, but get the cos@L32 number to *quantify* | drift_ladder default vs `--dn-state-dtype fp32 --owned-gdn off` |
| **H3 (decay fp32)** | Cast ONLY g_decay to fp32, keep H_t bf16. Tests whether decay quantization (not state) is the dominant noise | drift_ladder + new variant flag |
| **H4 (RMSNorm precision)** | Cast h_norm input to fp32 before rms_norm at L32 specifically | drift_ladder + new variant flag |
| **H5 (conv1d state)** | fp32 conv state. Memory says it's "less drift-prone" but never measured directly | drift_ladder + conv-fp32 variant |

## Key memory entries (don't re-discover)

- `feedback_35b_a3b_l32_dn_decode_drift.md` — drift origin pinpointed
- `feedback_35b_dn_h_state_drift_lever.md` — original H1 identification
- `feedback_35b_a3b_diagnosis_invalidated.md` — SDPA isn't the cause
- `feedback_35b_a3b_sdpa_swap_result.md` — what SDPA swap did help
- `feedback_use_dev_harness_for_iteration.md` — don't restart serve_cb
- `feedback_numpy_reference.md` — never use AutoModel
- `reference_hf_oracle_pattern.md` — canonical oracle pattern

## DO NOT

- DO NOT restart `serve_cb.sh start` for an experiment — use the harness.
- DO NOT re-attempt fp32 residual stream without a ladder-confirmed
  cos@L32 improvement first.
- DO NOT validate fp32 H_t via "did it boot and generate tokens" —
  use the cosine number, not subjective text quality.
