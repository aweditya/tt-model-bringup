# Model bringup recipe (Tenstorrent Blackhole, Qwen3.6/Gemma 4 lineage)

What makes a new-model bringup go from weeks to days. Follow it for every
new model. Keep it concise (~200 lines); update it whenever a new lesson
lands.

> **The one rule above all rules**: every new file or function must cite
> the existing pattern it forks. Three models (27B, 35B-A3B, Gemma 4 12B)
> were brought up faster each time because nothing was reinvented.

## Table of contents

1. [Before you write any code](#1-before-you-write-any-code)
2. [The staging ladder](#2-the-staging-ladder-mandatory)
3. [Infrastructure that pays for itself](#3-infrastructure-that-pays-for-itself-many-times-over)
4. [Diagnostic recipe](#4-diagnostic-recipe-when-a-forward-is-broken)
5. [Why bringup got faster](#5-why-bringup-got-faster-the-meta-lesson)
6. [When to deviate](#6-when-to-deviate)

------------------------------------------------------------------------

## 1. Before you write any code

1. **Plan-of-action in `research/<model>_bringup_plan.md`** (mandatory).
   The 27B plan, 35B plan, and Gemma 4 plan are all the same shape: a
   sub-stage table (v0.1 → v0.2 → v0.3 → v0.4 → v1 → v2) with explicit
   *gates* per row and a *fork base* per file. **No code lands without
   a plan row.**
2. **REUSE map in the plan** — table of `(existing file, role for this
   model)`. Forces you to look before writing. Reference list:
   - `experiments/serve/server_tp.py`, `server_tp_cb.py`, `cb_api.py`,
     `cb_scheduler.py`, `cb_engine.py` (production CB stack)
   - `experiments/serve/server_35b_ttnn.py`, `server_35b_cb.py` (MoE
     reference)
   - `experiments/cb/dev/cb35_dev_harness.py` → fork for fast iteration
   - `experiments/cb/isolate/paged_sdpa.py`,
     `paged_update_cache.py`, `dn_recurrence.py` (primitive probes)
   - `experiments/utils/hf_reference_35b.py`,
     `cosine_ladder_*.py`, `needle_haystack_35b_*.py`
3. **Non-negotiables** (`CLAUDE.md`):
   - Remote-only execution (`ssh qb1` / `ssh qb2`); single device first
   - No `python -c`; no `/tmp`; permanent files under `experiments/`
   - Frequent commits
   - **Plan, then act. No hand-wavy claims.** Every drift / perf claim
     has a probe behind it.
4. **Common bugs (read once, expect to hit at least three)**:
   - `[[ttnn-slice-view-decay]]` — `ttnn.slice`/`reshape` return VIEWS;
     don't deallocate the source while a view is live. *Masked at decode
     pos 0; surfaces at pos > 0.*
   - `[[ttnn-list-rebinding-leaks]]` — recreating buffers per step leaks;
     use `copy_host_to_device_tensor` for in-place updates.
   - `[[feedback-harness-state-version-skew]]` — dev harness keeps
     `State` alive across `importlib.reload(base)`; new fields need
     `getattr(state, "X", None)` defensive lazy-init.
   - `[[ttnn-rms-norm-shape-drift]]` — rank-3 rms_norm input; folding to
     rank-2 drifts in bf16.
   - `[[ttnn-shard-1d-vs-2d]]` — `ShardTensorToMesh(dim=0)` not
     `ShardTensor2dMesh` for the "stack per-chip" pattern.
   - `[[feedback-gemma4-sdpa-scale-1]]` — model-specific SDPA `scale`;
     read HF source, don't assume `1/sqrt(d_k)`.
   - `[[paged-update-cache-nkv-per-chip]]` — `page_table.dim(0) ==
     input.dim(1)`. NKV>1 per chip splits into N calls.
   - `[[read-kernel-source-first]]` — fork into a new shape regime →
     read `TT_FATAL` assertions before writing the call. Cheap
     insurance.

------------------------------------------------------------------------

## 2. The staging ladder (mandatory)

| Stage | Adds | Gate |
|---|---|---|
| **v0.0** | HF oracle: scripts/utils/`hf_reference_<model>.py` generates `prompt_ids.npy`, `hidden_states.npy` (per layer, per position), `argmax.npy`, optional L0 sub-captures. Use the `.venv-gemma4`-style venv if the model needs a newer `transformers`. | Oracle dir exists; layer count + HIDDEN match expected |
| **v0.1.0** | Bootstrap on (1,4): mesh, fabric, weights upload, embed lookup, sqrt(HIDDEN) scale, L0 input_layernorm | cos ≥ 0.999 on `embed_scaled` + `in_norm` vs HF oracle |
| **v0.1.1-3** | L0 sub-ops one at a time (q/k/v_proj → q/k/v_norm → RoPE → SDPA → o_proj → MLP) | cos ≥ 0.999 at each sub-op vs HF |
| **v0.2** | All N layers + final_norm + lm_head + softcap | argmax matches HF at pos 0 |
| **v0.3** | KV cache + paged SDPA + multi-step decode | argmax matches HF for tokens 0..5 |
| **v0.3.2** | **Per-step per-layer DECODE cosine ladder vs HF (N≥50 steps)** — fork `experiments/utils/cosine_ladder_hf_<model>.py` + `experiments/cb/isolate/<model>_long_decode_vs_hf_ladder.py`. Teacher-force HF's decode_ids through our forward, capture per-layer hidden via `step_forward_v0X(capture=...)`, cosine per (step, layer) | **min per-layer cos ≥ 0.99 across all N steps**, argmax-match ≥ 95%, cos_logits ≥ 0.99. **DO NOT SKIP** — argmax-match alone hides catastrophic hidden-state drift ([[feedback-argmax-hides-hidden-state-drift]]) |
| **v0.3.3** | Long INPUT context (cos ladder at L=200+, needle haystack) | argmax match ≥ 90%; median cos ≥ 0.99 |
| **v0.3.4** | **300-token generation smoke** — `python3 scripts/chat.py` or `chat_curl.py` with `--max 300 --temp 0.4 --top-p 0.9`; structural coherence check (`longest_run < 50 chars`, `tail_100_uniq > 5`). | No `####` / `***` / repetition collapse in 300 tokens |
| **v0.4** | Trace capture (two-phase warmup; `forward_token_inner` reads only state buffers) | 100 traced steps == 100 eager token-for-token |
| **v1.0-v1.6** | Continuous batching: setup → batched embed/RoPE → batched attention → end-to-end → 3a/3b/3c gates at B=2 → B=4 | All 3a/3b/3c PASS at B=4 |
| **v2** | HTTP wire-up: register in `cb_api.BACKENDS` + `cb_scheduler._BACKEND_MODULES`, tokenizer + chat template | `curl /v1/chat/completions` returns sensible text |

Each stage gates the next. Don't move on until cos ≥ 0.99 on the relevant
slice. Track the gate verbatim in the plan row.

------------------------------------------------------------------------

## 3. Infrastructure that pays for itself many times over

- **HF oracle pattern** — `experiments/utils/hf_reference_<model>.py` runs
  HF on CPU with `output_hidden_states=True` + forward hooks on L0
  sub-modules. Run once; reuse for every subsequent gate. Avoid
  `AutoModel.from_pretrained` issues — see `[[numpy-reference]]`.
- **Isolation probes** in `experiments/cb/isolate/` — one probe per
  primitive (`paged_sdpa.py`, `paged_update_cache.py`,
  `<model>_<feature>_<scenario>.py`). Bisect *before* fixing.
- **Cosine ladder** — per-position cos against HF
  `hidden_states[-1, pos, :]`. Three gates: argmax match rate ≥ 90%
  (primary, production-relevant); median cos ≥ 0.99; 5th-pct cos ≥ 0.95
  (absorbs single-position bf16 outliers). Strict MIN ≥ X is too strict
  for bf16 at L > 100.
- **Per-layer drift ladder** — when a forward is broken at pos > 0,
  capture h after every layer and compare to HF
  `hidden_states[L+1, pos, :]`. Finds the cliff layer in one run.
- **Needle-haystack** — fork `needle_haystack_35b_ttnn.py`. 8-char
  random password at frac=0.5 of a distractor haystack. Gates:
  L=100 ALL Y; L=256 ≥75%; L=512 ≥50% Y+P. Catches long-context drift
  that cosine ladders miss.
- **Dev harness** — fork `cb35_dev_harness.py` /
  `gm4_dev_harness.py`. Bootstrap once into a tmux'd Python; trigger
  tests via `touch trig/<name>`. Bootstraps 80s-14m once; iterations
  are seconds. **Always use this for iteration. NEVER restart
  `serve_cb.sh` per fix.**
- **Two-phase warmup before trace capture** —
  `[[ttnn-multi-trace-two-phase-warmup]]`. 2 eager forwards
  JIT-compile every kernel, then `begin_trace_capture` →
  forward → `end_trace_capture`. Trace capture cannot tolerate JIT
  (hangs Blackhole).
- **trace_region_size** — default 50 MB OOMs ≥40-layer decode
  traces. Use 400 MB for decode-only, 800 MB if also tracing prefill.
- **Memory entries** under `~/.claude/.../memory/feedback_*.md` for
  every non-obvious bug. Updated index in `MEMORY.md`. These persist
  across sessions and stop you re-hitting the same trap.

------------------------------------------------------------------------

## 4. Diagnostic recipe when a forward is broken

1. **Identify pass/fail boundary first** — is pos 0 PASS? Then the bug
   is positional or state-related. Is L=N PASS but L=N+1 FAIL? Then
   isolate to that layer.
2. **Bisect with isolation probes** — if SDPA in the full forward looks
   wrong, run `<model>_<shape>_write_read.py` on the primitive with
   random K/V. If primitive PASSES, the bug is in the composition (q/k/v
   projections + norms + RoPE + GQA mapping). If FAILS, kernel-level.
3. **Per-layer drift ladder at pos N** — capture h after every layer at
   the failing position, compare to HF. The first L where TT drops
   below cos 0.99 is your bisection target.
4. **Sub-op capture inside that layer** — extend the layer function with
   `capture` dict writes after every sub-op. Compare each to HF L0_*.npy
   captures or hf-reference L=N hooks.
5. **Read kernel source for `TT_FATAL` assertions** before writing a new
   kernel call. Especially for "moving to a new shape regime"
   (different NKV / head_dim / B).
6. **When you fix it, write a memory entry** with `Why:` + `How to apply:`.

------------------------------------------------------------------------

## 5. Why bringup got faster (the meta-lesson)

Three things compound:
- **Forking, not writing**. Each new model copies 80-90% of the
  previous one's CB module / dev harness / probes / oracle script. The
  remaining 10-20% is the model-specific composition (different
  attention shape, norm style, activation, RoPE variant). The cost is
  understanding the diff, not building from scratch.
- **The bug catalog is finite**. After bringing up 3 models, the same
  ~12 footguns reappear (view-decay, harness skew, scale=1.0, NKV
  contract, etc.). The first model paid for them; subsequent models
  recognize them in seconds.
- **The validation infrastructure is generic**. HF oracle, cosine
  ladder, needle haystack, dev harness, per-layer drift ladder — none
  of these care which model you're bringing up. Reuse them
  immediately and the gates write themselves.

**Concretely**: the 27B took ~3 weeks. The 35B-A3B (MoE + DN, novel
shapes) took ~2 weeks. **Gemma 4 12B (sliding + global hybrid, p-RoPE,
v_norm, layer_scalar, attention_k_eq_v) took ~36 hours from oracle to
HTTP chat** — bootstrap → 48 layers → traced decode → CB at B=4 → chat.
The recipe above is *why*.

------------------------------------------------------------------------

## 6. When to deviate

- **A new arch primitive (e.g. DeltaNet for 35B, sliding window for
  Gemma 4)**: write the probe FIRST, before integrating. The probe
  pays for itself the moment integration breaks.
- **A new venv requirement** (e.g. `transformers` not on
  default `.venv`): set up `.venv-<model>` and document the
  invocation in the bringup plan.
- **A new chat-template format**: tokenizer install + chat template
  load is part of v2 wire-up. If the base model has no template,
  install a minimal one in `bootstrap` (Gemma 4 example in
  `server_gemma4_unified_ttnn.py:bootstrap`).
- **A model the kernel doesn't natively support** (e.g. owned MoE
  expert FFN): you write a custom kernel from scratch (G0..G4
  staging). Plan a 1-2 week buffer.
