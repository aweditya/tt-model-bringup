# 35B drift — next session plan

> **PARKED 2026-06-03**. Pivoted to Gemma 4 12B bringup (#165) because
> 14-min 35B harness bootstrap was rate-limiting AND the harness hung
> silently mid-investigation (#166 will harden it). All probe
> infrastructure on qb1 was killed when we released the mesh. Pickup:
> (a) ship #166 (harden harness ~12 LOC), (b) re-start tmux `cb35`
> harness via `bash scripts/run_harness_tmux.sh qb1`, (c) re-deploy
> `experiments/cb/dev/cb35_drift_cliff_search.py` and trigger it for
> Step 1. The §REAL findings table is the load-bearing data.
>
> Cross-pollination opportunity: Gemma 4 v0.3 exercises sliding+global
> positional-state paths in isolation. If a positional-state bug
> surfaces there, it likely shares mechanism with the cliff here —
> resume #163 with that information in hand.

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

## GO-button — SUPERSEDED 2026-06-03

> The 6 probes below already ran. Their results overturn the working
> model; do NOT re-run them blindly. See **§REAL findings** + **§What
> to do next** below for the current step. Section preserved for
> historical / commit-archaeology context only.

<details>
<summary>(click to expand — stale)</summary>

```bash
ssh qb1 'tail -3 ~/tt-xla/.cache/cb35_runtime/harness.log'
# Expect "[harness] ready. Drop trigger files into …"

# H0 baseline (memory expectation cos@L32 pos 1 = 0.9311 — STALE; real owned_gdn = 0.99)
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_bf16'
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_fp32_h'
ssh qb1 'touch tt-xla/.cache/cb35_runtime/trig/drift_fp32_h_no_dg'
for v in drift_long_bf16 drift_long_fp32_h drift_long_fp32_h_no_dg; do
  ssh qb1 "touch tt-xla/.cache/cb35_runtime/trig/$v"
done
```

JSON outputs under `.cache/cb35_runtime/drift_ladder_*.json`.

</details>

## Decision tree after the headline — SUPERSEDED 2026-06-03

> Predicated on the stale 0.9311 baseline; all three rows are
> falsified by the §REAL findings. Kept for archaeology only.

<details>
<summary>(click to expand — stale)</summary>

| `H0 cos@L32 pos1` | `H1 cos@L32 pos1` | Verdict |
|---|---|---|
| ≈ 0.9311 | ≥ 0.99 | **H1 confirmed.** Long-context probe should show top-1 lift too. Plumb to server, ship. |
| ≈ 0.9311 | 0.93–0.97 | **H1 partial.** Decay+state mixed. Try `_no_dg` variant; consider H3 (decay-only fp32). |
| ≈ 0.9311 | ≈ 0.9311 (unchanged) | **H1 dead.** Move to H3 (decay alone), H4 (RMSNorm), H5 (conv1d). New probe per hypothesis. |

</details>

## REAL findings (2026-06-03 — read this first, supersedes both sections above)

The probes ran. **All 3 hypotheses in the table above were
disproved in a different way than expected.** The actual picture:

### Finding 1: Memory's "0.9311 baseline" was stale

`feedback_35b_a3b_l32_dn_decode_drift.md` cited `cos@L32 pos 1 =
0.9311` as the drift origin. That number was from an OLDER run on
the broken manual path. The current owned_gdn baseline at pos 1 is
**0.99**. Pos 1 is not the drift origin anymore.

### Finding 2: Real drift is a CLIFF between pos 1 and pos 5

Measured 2026-06-03 on the 85-token ladder prompt with owned_gdn=ON
(default config), via `cb35_drift_long_bf16`:

| pos | cos_L32 | cos_final | top1 |
|---|---|---|---|
| 0  | 0.9896 | 0.9950 | ✓ |
| 1  | 0.9864 | 0.9925 | ✓ |
| **5**  | **0.3172** | **0.5210** | ❌ |
| 10 | 0.3910 | 0.2974 | ❌ |
| 25 | 0.3781 | 0.1922 | ❌ |
| 40 | 0.4647 | 0.5505 | ❌ |
| 60 | 0.4652 | 0.6254 | ✓ |
| 80 | 0.4364 | 0.6351 | ❌ |

The cliff is BETWEEN pos 1 and pos 5 — sharp, not gradual. Short
prompts that don't cross this cliff produce coherent output (Hello
→ "How can I help"). Longer prompts collapse. Memory:
`feedback_35b_drift_cliff_pos1_to_pos5`.

### Finding 3: Manual recurrence path is structurally broken

Same probe with `owned_gdn=OFF` (forces manual recurrence):

| pos | cos_L32 |
|---|---|
| 0 | **0.0771** (essentially random) |
| 1 | 0.0891 |
| 5 | 0.0412 |

Pos 0 has no recurrence state — both paths see zero. So this
0.08 vs 0.99 is the manual path itself doing wrong math, not a
precision issue. The fp32 H_t fix (commit `92b442f`) auto-disables
owned_gdn → routes through this broken path → that's why it was
"mechanically valid but drift unchanged". The math was wrong before
fp32 storage ever mattered. Memory:
`feedback_35b_manual_recurrence_path_broken`.

### What to do next — sequential decision tree

> Stale H1/H3/H4/H5 ladder (DN H_t / decay / RMSNorm / conv1d
> precision) is **rejected** — predicated on the stale 0.9311
> baseline. The new investigation localizes a positional-state bug.

**Prerequisites — already staged on qb1:**

- Dev harness in tmux `cb35` (resident model, ~30 sec iter).
- HF oracles at `.cache/hf_oracle_35b_100tok/` (5 pos),
  `.cache/hf_oracle_35b_long/` (85 pos).
- Probe core `experiments/cb/dev/cb35_drift_ladder.py` + 6 thin
  env-config wrappers deployed.
- Trigger directory `~/tt-xla/.cache/cb35_runtime/trig/`.

**Step 1 — Localize the cliff between pos 1 and pos 5 (one probe, ~30 sec):**

```bash
# Edit experiments/cb/dev/cb35_drift_long_bf16.py to set:
#   os.environ["CB35_LADDER_POSITIONS"] = "0,1,2,3,4,5,6,7"
# (the wrapper is a thin env-setter calling cb35_drift_ladder.main)

bash scripts/deploy.sh experiments/cb/dev/cb35_drift_long_bf16.py
ssh qb1 'touch ~/tt-xla/.cache/cb35_runtime/trig/drift_long_bf16'
sleep 35
ssh qb1 'cat ~/tt-xla/.cache/cb35_runtime/trig/last.log | tail -25'
```

Goal: find whether the cliff lands at pos 2, 3, 4, or 5. Headline
metric: position P where `cos_L32` first drops below 0.95 (from
0.99 at pos 1). Call this `P_cliff`.

**Step 2 — Per-layer cos at `P_cliff` (which layer first drifts?):**

`cb35_drift_ladder.py` already records `cos_per_layer` for every
position in the JSON output. After Step 1, inspect:

```bash
ssh qb1 'python3 -c "import json; d=json.load(open(\"~/tt-xla/.cache/cb35_runtime/drift_ladder_bf16_gdnon_dgon.json\")); pos=str(<P_cliff>); print([(L, d[pos][\"cos_per_layer\"][L]) for L in range(40)])"'
```

(or write a 10-line script `experiments/cb/dev/inspect_cliff_layer.py`
in the project — do NOT inline.) Find the smallest layer index `L`
where `cos_per_layer[L] < 0.95`. Call this `L_locus`. Memory's
"L32 is the locus" was on stale data and may not hold.

**Step 3 — Sub-op probe at `(L_locus, P_cliff)`:**

`step_forward_ttnn` accepts `sub_capture_layers=[L_locus]` and
fills `capture["layer_<L>_sub"]` with attn/MoE/DN sub-step arrays.
Add a new wrapper `experiments/cb/dev/cb35_drift_subop.py` that:

1. Reuses `cb35_drift_ladder` plumbing.
2. Passes `sub_capture_layers=[L_locus]` and `CB35_LADDER_POSITIONS=str(P_cliff)`.
3. Compares each sub-op output cos vs HF oracle's matching sub-op
   activation (`hf_reference_35b` already captures these — verify
   by reading the oracle script).

Headline: which sub-op (RoPE? KV-cache read? conv1d state shift?
DN gate? MoE routing logits?) first drops below 0.95.

**Working hypothesis flavor**: sharp cliff between pos 1-5 looks
like a positional-state bug (RoPE pos lookup, KV cache write/read
at pos > 0, conv1d 4-tap window shift, sliding KV mask), NOT a
precision decay (which would be gradual). The sub-op probe pinpoints
which positional-state op silently breaks at `P_cliff`.

**Step 4 — Manual recurrence path repair (deferred, task #164):**

Structural bug (cos 0.08 @ pos 0 with owned_gdn=OFF) is real but
orthogonal to the user-facing cliff. Track separately. Don't ship
fp32 H_t fixes until manual is repaired.

## Stale hypothesis ladder — SUPERSEDED 2026-06-03

> H1/H3/H4/H5 (DN precision) ladder predicated on the 0.9311 baseline.
> Listed here so it isn't re-derived in a future session.

<details>
<summary>(click to expand — stale)</summary>

| H | What | Probe |
|---|---|---|
| **H1 alone insufficient** | Confirmed already, but get the cos@L32 number to *quantify* | drift_ladder default vs `--dn-state-dtype fp32 --owned-gdn off` |
| **H3 (decay fp32)** | Cast ONLY g_decay to fp32, keep H_t bf16. Tests whether decay quantization (not state) is the dominant noise | drift_ladder + new variant flag |
| **H4 (RMSNorm precision)** | Cast h_norm input to fp32 before rms_norm at L32 specifically | drift_ladder + new variant flag |
| **H5 (conv1d state)** | fp32 conv state. Memory says it's "less drift-prone" but never measured directly | drift_ladder + conv-fp32 variant |

</details>

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
