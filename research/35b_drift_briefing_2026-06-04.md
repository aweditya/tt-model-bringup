# 35B-A3B drift — fresh-investigator briefing (2026-06-04)

Produced by background agent before resuming the parked drift bug
(tasks #163, #164, #170). Use this as the entry point for the
overnight investigation. Reference plan:
[`research/35b_drift_next_session_plan.md`](35b_drift_next_session_plan.md).

## 1. What is already known

- **Symptom**: short prompts ("Hello") generate coherent text; long
  prompts collapse to repetitive token loops after ~20 tokens.
- **Mechanism (current model)**: a *sharp cliff between pos 1 and
  pos 5* in the teacher-forced cosine ladder with `owned_gdn=ON`
  (default). Cliff goes `cos_L32: 0.99 → 0.32` in 4 positions —
  NOT a gradual per-step decay.
- Measured 2026-06-03 on qb1 via
  `archive/cb35_drift_wrappers_2026-06-04/cb35_drift_long_bf16.py`
  (archived 2026-06-04 after the cliff resolved per
  `feedback_35b_drift_resolved_2026-06-04`) against
  `.cache/hf_oracle_35b_long` (85-tok math ladder prompt). See
  `feedback_35b_drift_cliff_pos1_to_pos5.md` and §REAL findings in
  the next-session plan.
- **Pos 1 IS clean** under the current owned-kernel path
  (`cos_L32 = 0.9864`, `cos_final = 0.9925`). The legacy memory claim
  *"L32 pos 1 = 0.9311 drift origin"*
  (`feedback_35b_a3b_l32_dn_decode_drift.md`) is **STALE** — that
  number came from the broken manual recurrence path.
- L32 IS a DeltaNet layer, but *"L32 is THE locus"* may also be stale.
  Per-layer cos at the cliff position has not yet been bisected with
  the current owned-kernel-on baseline.

## 2. Hypotheses tried + rejected, what's still open

**Rejected** (with evidence):
- "bf16 SDPA HiFi4 is the bug" — 27B retrieves at L=500 with the same
  kernel (`feedback_35b_a3b_diagnosis_invalidated.md`).
- "fp32 softmax differentiator" — HF retrieves with bf16-softmax
  monkey-patch.
- "RoPE-broadcast workaround introduces noise" — bit-inert.
- "fp32 H_t storage alone fixes it" — commit `92b442f` shipped the
  hook, drift unchanged. Worse: setting `TT_DN_STATE_DTYPE=fp32`
  auto-disables `owned_gdn`, routing through the broken manual path
  (`feedback_35b_manual_recurrence_path_broken.md`). The H1/H3/H4/H5
  (DN precision) ladder is rejected because it was predicated on the
  stale 0.9311 baseline.
- "MoE router top-k symptom" — H3 rejected
  (`feedback_35b_a3b_h3_router_rejected.md`): router faithfully
  top-k's an already-drifted input.

**Open / unverified**: the sharp cliff between pos 1-5 smells like
a *positional-state bug* — RoPE pos lookup, KV cache write/read at
pos > 0, conv1d 4-tap window shift, sliding KV mask — NOT a precision
decay (which would be gradual).

## 3. Next concrete investigation step

**Step 1 — Bisect the cliff position (pos 2/3/4/5).** Edit the
archived `archive/cb35_drift_wrappers_2026-06-04/cb35_drift_long_bf16.py`
(fork it back into `experiments/cb/dev/` if reviving the probe) to set
`CB35_LADDER_POSITIONS = "0,1,2,3,4,5,6,7"`, deploy, trigger. Find
`P_cliff` = position where `cos_L32` first drops below 0.95.

**Step 2 — Per-layer cos at `P_cliff`.** `cb35_drift_ladder.py`
already records `cos_per_layer`. Find smallest `L` where
`cos_per_layer[L] < 0.95` — call this `L_locus`. Memory's "L32 is
the locus" may not hold under `owned_gdn=ON`.

**Step 3 — Sub-op probe at `(L_locus, P_cliff)`.** `step_forward_ttnn`
already accepts `sub_capture_layers=[L_locus]` and populates
`capture["layer_<L>_sub"]` with attn/MoE/DN sub-step arrays.
`hf_reference_35b.py` has matching hook plumbing (see
`--hook-attn-layer N`). Fork the Gemma 4 pattern
`experiments/cb/isolate/gm4_per_layer_drift_pos1.py` (it's clean:
HF `hidden_states[L+1, pos, :]` vs TT `cap["layer_h"][L]`, prints
PASS/FAIL ladder + "FIRST CLIFF" tag) — but target the 35B server
(`server_35b_ttnn`) and oracle at `.cache/hf_oracle_35b_long`.

## 4. Other relevant probes / kernels

- `experiments/cb/isolate/` 35B-relevant: `paged_sdpa.py`,
  `chunked_sdpa.py`, `paged_update_cache.py`.
  `gm4_per_layer_drift_pos1.py` is the forkable bisection pattern.
  The owned_gdn / dn_recurrence / conv_reform probes were archived
  2026-06-04 to `archive/cb_engine_scaffolding_2026-06-04/`.
- `experiments/cb/dev/`: `cb35_drift_ladder.py` (core) +
  `cb35_drift_cliff_search.py` (may already cover Step 1 — check
  before forking). The 6 env-wrapper variants
  (cb35_drift_bf16 / cb35_drift_fp32_h{,_no_dg} /
  cb35_drift_long_bf16{,_manual} / cb35_drift_long_fp32_h{,_no_dg})
  were archived 2026-06-04 to `archive/cb35_drift_wrappers_2026-06-04/`
  after the cliff resolved.
- HF oracle: `experiments/utils/hf_reference_35b.py`. Outputs at
  `.cache/hf_oracle_35b_long/` (85 pos) and
  `.cache/hf_oracle_35b_100tok/` (5 pos).
- **CRITICAL BLOCKER**: `feedback_35b_manual_recurrence_path_broken.md`
  — manual path gives `cos@L32 pos 0 = 0.08` (random). All bisections
  MUST run with `owned_gdn=ON`. Do NOT compare against manual path.

## 5. Harness setup

`bash scripts/run_harness_tmux.sh cb35 qb1` (defaults are
`cb35 qb1`, so bare `bash scripts/run_harness_tmux.sh` also works).
Bootstraps `experiments/cb/dev/cb35_dev_harness.py` (~14 min, fine
overnight) in detached tmux session `cb35`. Resident model iterates
in ~30 sec per probe. Triggers:
`touch ~/tt-xla/.cache/cb35_runtime/trig/<probe_name>`. Log:
`~/tt-xla/.cache/cb35_runtime/harness.log`. Harness was hardened
2026-06-03 (commit `84efe50`) against the silent-hang issue.

## 6. Things to AVOID

- **DO NOT use the manual recurrence path** for ANY bisection — it's
  broken at pos 0 (`cos_L32 = 0.08`).
- **DO NOT set `TT_DN_STATE_DTYPE=fp32`** — auto-disables `owned_gdn`,
  routes through the broken manual path.
- **DO NOT trust the "L32 is the drift origin" / "cos@L32 pos 1 =
  0.9311" memories** — both stale, from the broken manual path. Real
  owned_gdn baseline at pos 1 is 0.99.
- **DO NOT restart `serve_cb.sh`** — use the resident tmux dev harness
  (30 sec/iter vs 14 min restart).
- **DO NOT chase precision/SDPA hypotheses again** — 27B is the free
  control and works with the same SDPA kernel + B3 + bf16 KV. The bug
  is 35B-specific structure, not generic precision.
- **DO NOT validate fixes by "did it boot and generate tokens"** — use
  the cosine number.
- **DO NOT write inline python** or use `/tmp`; permanent files only.
- **REUSE MANDATE**: grep `experiments/cb/dev/cb35_*` and
  `experiments/cb/isolate/` before writing new files.
  `cb35_drift_cliff_search.py` already exists — check whether it
  covers Step 1.

## Key files

- `research/35b_drift_next_session_plan.md` — live plan, read
  §REAL findings.
- `research/35b_a3b_correctness_plan.md` — historical context, mostly
  superseded.
- `experiments/cb/isolate/gm4_per_layer_drift_pos1.py` — forkable
  bisection pattern.
- `experiments/cb/dev/cb35_drift_ladder.py` + wrappers — deployed
  probes.
- `experiments/utils/hf_reference_35b.py` — oracle generator.
- `scripts/run_harness_tmux.sh` — harness launcher.
