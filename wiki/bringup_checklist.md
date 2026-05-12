# Model Bringup Checklist

The playbook for porting any new model to Tenstorrent ttnn. Born from the Qwen3.6-27B
bringup which surfaced **7 distinct bugs** — 5 of which would have been caught on day 1
by following this checklist.

This list isn't exhaustive but each item maps to a real bug from session 2026-05-12.

---

## Step 0 — Architecture due diligence

Before writing any code:

- [ ] Locate the HF transformers modeling source for this model family. Path is usually
      `site-packages/transformers/models/<family>/modeling_<family>.py`.
- [ ] Read the `*DecoderLayer` class top-to-bottom. Note every operation, every cast,
      every helper called.
- [ ] Read every helper class referenced (RMSNorm, RotaryEmbedding, attention impl).
      Norms especially: many model families use non-standard formulas.
- [ ] If the model has a recurrent component (DeltaNet, Mamba, etc.), read the
      `torch_*_rule` reference implementation (`torch_recurrent_gated_delta_rule`,
      `torch_chunk_gated_delta_rule`, etc.). These are pure-torch reference impls used
      when the fast path isn't available.

---

## Step 1 — Enumerate every weight key (catches missing-weight bugs)

Before touching the loader, run a weight audit:

```bash
python experiments/utils/weight_audit.py \
    --model <MODEL_ID> --list-layer-keys 0
python experiments/utils/weight_audit.py \
    --model <MODEL_ID> --list-layer-keys <FIRST_DIFFERENT_LAYER_TYPE>
```

- [ ] For every distinct layer type, list ALL safetensors keys. Cross-check vs the keys
      your loader requests.
- [ ] If you find a key in safetensors that your loader skips, **either load it or
      explicitly document why you're not loading it**. Skipping silently is how bugs
      #1, #2, #3 hid for 5 phases of B' work.
- [ ] Use `--diff-loader` mode once your loader is ready, with a JSON of the loader's
      expected keys, to assert exact match.

---

## Step 2 — Audit every normalization (catches parameterization bugs)

For each `*Norm` class the model uses:

- [ ] Read the `forward` method. What's the **exact formula**?
- [ ] Common variants:
  - Standard RMSNorm:   `output = x / sqrt(mean(x²) + eps) * weight`
  - Qwen3.5 RMSNorm:    `output = x / sqrt(mean(x²) + eps) * (1.0 + weight)`  ← **adds 1!**
  - LayerNorm:          `output = (x - mean) / sqrt(var + eps) * weight + bias`
  - RMSNormGated:       `output = (x / sqrt(mean(x²) + eps) * weight) * silu(gate)`

- [ ] Check the init: `weight = zeros(dim)` typically means `(1+w)` parameterization
      because the trained scale needs to shift away from 0. `weight = ones(dim)` is
      the standard parameterization.
- [ ] Verify weight stats: load a few norm weights via `experiments/utils/weight_audit.py
      --stats <keys>`. Mean ≈ 0 → (1+w) form. Mean ≈ 1 → standard form.

This step would have caught bug #4 (we used standard `w` for weights that needed `1+w`).

---

## Step 3 — Audit every gating operation

- [ ] sigmoid? silu? tanh? Read the source for each gate.
- [ ] Order: does the model normalize BEFORE the gate or AFTER? (Qwen3.5 DeltaNet:
      RMSNorm output, then silu-gate. Qwen3.5 Gated Attention: sigmoid-gate the SDPA output.)
- [ ] Is the gate per-head or per-element flattened? (HF reshapes z to per-head before
      passing to RMSNormGated; we initially didn't, and the result was correct only
      because the gate-mul math is dim-agnostic — but it's brittle.)

---

## Step 4 — Probe library semantics (catches op-behavior bugs)

ttnn ops can have non-obvious behavior. Probe before assuming:

- [ ] `ttnn.repeat`: tile-style or interleave? Run
      `python experiments/utils/repeat_semantics_probe.py` once per ttnn version
      to be sure. (Bug #5 was here.)
- [ ] `ttnn.rms_norm`: scale-invariant when input magnitudes are tiny? It is NOT when
      `eps` dominates `variance`. This isn't a ttnn-specific bug but interacts with
      the choice of which scaling factors to apply (bug #7).

---

## Step 5 — Audit every scaling factor (catches numerical-regime bugs)

This is the lesson from bug #7. Even constants that "shouldn't matter" can matter:

- [ ] Apply EVERY constant scaling op from the reference, even if you can argue it's
      "cosine-invariant" or "absorbed by downstream norm." Examples we hit:
  - `query = query * (1/sqrt(head_k_dim))` before recurrence output (bug #7)
  - `attn = attn * (1/sqrt(d_k))` before softmax (standard attention scale)
- [ ] If you choose to skip a scaling, document the empirical evidence (a probe showing
      the output is identical with or without it across the production magnitudes).
- [ ] See `feedback_dont_dismiss_audit_lines.md` for the cautionary tale.

---

## Step 6 — Establish HF ground truth (validation oracle)

Don't trust your own numpy reference. Build the HF oracle:

```bash
python experiments/utils/hf_full_model_oracle.py --dump-hidden-states
```

This produces:
- Top-100 predictions for a canonical prompt → `hf_oracle_topk.json`
- All 64+ per-layer hidden states → `hf_per_layer_hidden_states.npz`

The hidden states are the inputs for per-layer isolation testing (next step).

- [ ] Verify the top-1 logit margin is large (≥ 1.0 in logit space). If it's small,
      pick a better prompt — you want clear ground truth.

---

## Step 7 — Per-layer validation (the gold standard)

For each layer type:

```bash
python experiments/91r_per_layer_diff.py --layers <list>
```

- [ ] Sample a representative set: layers 0, 1, 2 (catch position-accumulation bugs),
      one of each layer type (catch layer-type-specific bugs), one deep layer
      (verify depth doesn't compound).
- [ ] Gate at per-layer cosine ≥ 0.999 (last position).
- [ ] If a specific layer drops, write a substep dump (`91s` pattern: capture every
      intermediate, save deferred to host) and a substep comparator (`91t` pattern).

---

## Step 8 — End-to-end generation (final eyeball)

- [ ] Run the full forward + lm_head with 60-100+ tokens. Greedy decode first, then sampling.
- [ ] Verify top-1 matches HF's ground truth target. For "The capital of France is",
      target is ` Paris` with high probability.
- [ ] Stress-test with longer prompts and different domains before declaring done.

---

## Common pitfalls (anti-checklist)

Things that LOOK right but aren't:

- ❌ **Comparing ttnn-to-our-numpy and seeing high cosine.** Both might share the same bug.
      Always use an external oracle.
- ❌ **Isolation test passes with embed-scale inputs.** Real production inputs are larger
      and may exercise different numerical regimes.
- ❌ **Cosine 0.999 globally must mean the tensor is right.** Per-row magnitudes can
      differ by constants that the global cosine averages away.
- ❌ **"This scaling factor is cosine-invariant so it doesn't matter."** It might at
      production magnitudes; it might NOT at edge-case magnitudes.
- ❌ **Inline `python -c` for one-off audits.** Promote everything to a permanent utility
      script the first time you use it; you'll need it again.

---

## Reference utilities

After session 2026-05-12, the bringup toolkit lives in `experiments/utils/`:

- `weight_audit.py` — list keys, weight stats, loader vs safetensors diff
- `repeat_semantics_probe.py` — verify ttnn.repeat tile vs interleave
- `patch_ttnn_llk_roundmode.py` — patch the upstream LLK int→RoundMode bug (one-time per ttnn install)
- `hf_full_model_oracle.py` — HF CPU forward, top-K + per-layer hidden states
- `hf_layer0_substep_dump.py` — HF substep capture via PyTorch hooks
- `ttnn_rms_norm_probe.py` — verify ttnn.rms_norm at a specific shape vs numpy reference
- `norm_in_per_row_probe.py` — per-row cosine + magnitude diagnostic (extends to any captured pair)
- `softplus_stability_probe.py` — verify softplus numerical stability for your model's `a + dt_bias` distribution
- `gated_formula_probe.py` — mix-and-match `(our_input, hf_input) × (our_gate, hf_gate)` to isolate which input is wrong
- `substep_compare.py` — compare two npz dumps with layout-aware mapping
