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

## State to pick up

- ✅ Server-side: `TT_DN_STATE_DTYPE=fp32` env hook works for opt-in
  fp32 H_t with manual recurrence path. Disabled `dn_owned_gdn` and
  `dn_owned_decay_gate` when fp32 is requested. Default bf16 path
  unchanged.
- ✅ Probes: `experiments/utils/cosine_ladder_35b.py` has CLI hooks
  `--dn-state-dtype {bf16,fp32}`, `--owned-gdn {on,off}`,
  `--owned-decay-gate {on,off}` for exactly the A/B we need.
- ✅ HF reference generator: `experiments/utils/hf_reference_35b.py`
  produces the oracle that `cosine_ladder_35b.py` consumes
  (under `.cache/hf_oracle_35b*/`).
- ✅ Dev harness scaffold: `scripts/run_harness_tmux.sh qb1` launches
  long-lived python with `state` resident; trigger files in
  `~/tt-xla/.cache/cb35_runtime/trig/` map to
  `experiments/cb/{validate,isolate,bench,dev}/cb35_<name>.py` and
  iterate in ~30 sec.
- ❌ **HF oracle not yet generated on qb1.** Need to run
  `experiments/utils/hf_reference_35b.py` (one-time, ~30-60 min on
  HF CPU) before the cosine ladder is useful.
- ❌ **Dev-harness wrapper around cosine_ladder** not yet written.
  cosine_ladder_35b.py is a standalone script (does its own
  bootstrap). Need a thin harness-callable probe that takes
  `state=<harness state>` and runs the ladder logic without
  re-bootstrapping.

## First 30 minutes of next session

1. **Verify harness is up**
   ```bash
   ssh qb1 'tmux ls'                          # expect: cb35: 1 windows
   ssh qb1 'tail -20 ~/tt-xla/.cache/cb35_runtime/harness.log'
   ```
   If not running, restart with `bash scripts/run_harness_tmux.sh qb1`.

2. **Generate HF oracle (one-time)**
   - Local (CPU): `python3 experiments/utils/hf_reference_35b.py
     --out .cache/hf_oracle_35b_100tok --max-new 100`
     (~30-60 min on a modern CPU)
   - Then sync to qb1: `bash scripts/deploy.sh .cache/hf_oracle_35b_100tok/`
   - OR: run on qb1 directly via the harness (faster — has more RAM)

3. **Write thin harness wrapper for cosine ladder**
   `experiments/cb/dev/cb35_drift_ladder.py`:
   - Signature: `def main(state)`
   - Args via env: `CB35_LADDER_DN_DTYPE=fp32 CB35_LADDER_OWNED_GDN=off`
   - Loads HF oracle from `.cache/hf_oracle_35b_100tok/`
   - For pos in [0, 1, 2, 5, 10]: run `step_forward_ttnn` with
     `capture={}` dict, compare per-layer hidden states to HF,
     write cos numbers to JSON.
   - Print cos@L32 pos1 prominently — that's the drift origin
     per memory `feedback_35b_a3b_l32_dn_decode_drift.md`.

4. **First numerical answer**: A/B between bf16 default and
   `TT_DN_STATE_DTYPE=fp32` H_t-only via the harness:
   ```bash
   # baseline
   ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_ladder'

   # fp32 H_t variant
   ssh qb1 'env CB35_LADDER_DN_DTYPE=fp32 CB35_LADDER_OWNED_GDN=off \
     touch tt-xla/.cache/cb35_runtime/trig/drift_ladder'
   ```
   (Note: env vars need to be passed to harness, not trigger. Check
   how the existing probes handle config — may need to write a
   variant file per config.)

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
