# Debugging Methodology — when generation is wrong

The workflow that cracked bug #7 of the Qwen3.6-27B port. Each step narrows the search space.

---

## When to start this

You've run the full forward + lm_head and the output is wrong:
- Fixed point on a single token (`'FRFRFR...'` or `'00000...'`)
- Top-5 contains no sensible tokens
- Top-1 logit margin is tiny (e.g., < 0.5 in logit space — HF healthy is ≥ 1.0)

If the cosine vs HF on the FINAL hidden state is < 0.95, this methodology is for you.

---

## Step 1 — Establish HF ground truth

If you don't already have it:

```bash
python experiments/utils/hf_full_model_oracle.py --dump-hidden-states
```

Produces:
- `~/tt-xla/.cache/hf_oracle_topk.json` — what HF predicts for your prompt
- `~/tt-xla/.cache/hf_per_layer_hidden_states.npz` — all 65 hidden states (embed + 64 layer outputs)

Verify HF top-1 is sensible and confident. If you don't have a clear target, no amount of debugging will tell you where to aim.

---

## Step 2 — Per-layer isolation cosine

Your full forward has compounded drift. Decompose it:

```bash
python experiments/91r_per_layer_diff.py --layers 0,1,2,3,7,11,15,31,47,63
```

This script, for each layer N: feeds HF's `hidden_{N}` (the real input to layer N from a full HF forward) into your ttnn layer N, compares the output to HF's `hidden_{N+1}`. Each layer is tested in **isolation** with **production-magnitude inputs** and **fresh state**.

Pattern recognition:
- All layers cosine ≥ 0.999 → bug is in lm_head or final norm. Inspect lm_head ranks for sensible tokens (`experiments/91n_lm_head_inspection.py`).
- One specific layer type (DeltaNet, Gated Attn, MoE) has worse cosine → bug is in that layer's path
- One specific layer has catastrophic cosine while neighbors are fine → bug is in layer-specific weights (e.g., a corrupt safetensors key)
- Cosine degrades with position within a layer → state accumulation bug (conv1d state, SSM state, KV cache)
- Cosine degrades with depth → drift compounding (try fp32 residual stream as a band-aid)

---

## Step 3 — Substep capture for the worst layer

If a specific layer is bad, dump every intermediate.

**HF side** (~30 sec):
```bash
python experiments/utils/hf_layer0_substep_dump.py --layer <N> --input-from-hidden
```

**ttnn side**: copy/extend `experiments/91s_layer2_full_substep_dump.py` for your target layer.

**Compare**: copy/extend `experiments/91t_layer2_substep_compare.py`. The LAYOUT mapping is the tricky part — HF captures via PyTorch hooks (input/output of each named submodule), ttnn captures whatever you put `to_np` on. Pay attention to:
- HF's hook captures function I/O, NOT internal intermediates. RMSNormGated.out includes the gate.
- HF's tensors have batched shape `[1, seq, ...]`; ttnn captures are per-position
- Beware: feeding embed to a constructed layer (HF substep default) does NOT match feeding `hidden_N` (production). Use `--input-from-hidden` so the inputs match.

Walk the comparison output top-to-bottom. The FIRST substep where cosine drops below 0.999 localizes the bug to ONE operation.

---

## Step 4 — Substep boundary isolation

You found that, e.g., the RMSNormGated step drops cosine from 0.9999 to 0.81. Now narrow further.

**The mix-and-match probe** (`experiments/utils/gated_formula_probe.py` is the template):

For the suspect operation `output = f(input_A, input_B, ...)`:

| input_A | input_B | result_cosine_vs_hf |
|---|---|---|
| ours | ours | low (the broken case) |
| ours | HF | ? |
| HF | ours | ? |
| HF | HF | should be 1.0 (sanity) |

Whichever cell first becomes high tells you which input is fine. The remaining inputs are where the bug is.

For our bug #7: `OUR norm_in × OUR silu_z` = 0.81, `OUR norm_in × HF silu_z` = 0.81, `HF norm_in × OUR silu_z` = 0.9999. Therefore `silu_z` was fine; `norm_in` (recurrence output) was the problem.

---

## Step 5 — Per-row diagnostic

If a tensor differs but global cosine looks OK (≥ 0.99) and the downstream output is wrong, the global cosine is averaging over per-row catastrophic mismatches.

Pattern: `experiments/utils/norm_in_per_row_probe.py`

For each row, compute:
- per-row cosine (ours vs reference)
- per-row magnitude ratio (‖ours‖ / ‖HF‖)

Smoking guns:
- A small subset of rows with catastrophic cosine → look for what's special about those rows (specific heads, specific positions)
- Constant magnitude ratio across all rows → MISSING SCALING FACTOR. Hunt for it in the reference. (This is bug #7.)
- Magnitude ratio = √(some_dim) → likely a missing scale-by-1/sqrt(dim) somewhere

---

## Step 6 — Apply the reference line, don't argue

When you find a candidate missing line in the reference:
- **Just add it.** Don't argue about whether it should matter analytically.
- If you're sure it's wrong, ablate it after empirical confirmation.
- See `feedback_dont_dismiss_audit_lines.md` for the canonical cautionary tale.

---

## Anti-patterns (things that wasted hours)

- ❌ Reading 100 lines of HF source looking for a bug that's on line 1 of a helper
- ❌ Hypothesizing "drift compounding" when the cosine drop is at a SINGLE step
- ❌ Blaming bf16 quantization for cosine 0.81 (bf16 alone gives ~0.99)
- ❌ Trusting your own numpy reference (it shares the same bugs as your ttnn)
- ❌ Believing global cosine ≥ 0.999 means a tensor is correct
- ❌ Reasoning "this scaling factor doesn't change cosine" without checking edge regimes

---

## Toolkit reference

| Tool | Purpose |
|---|---|
| `experiments/utils/weight_audit.py` | Enumerate safetensors keys, stats, loader diff |
| `experiments/utils/hf_full_model_oracle.py` | HF CPU forward, top-K + per-layer hidden states |
| `experiments/utils/hf_layer0_substep_dump.py` | HF substep dump via PyTorch hooks |
| `experiments/utils/repeat_semantics_probe.py` | Verify ttnn.repeat behavior |
| `experiments/utils/patch_ttnn_llk_roundmode.py` | Patch upstream ttnn LLK bug |
| `experiments/utils/ttnn_rms_norm_probe.py` | Verify ttnn.rms_norm matches numpy at a specific shape |
| `experiments/utils/norm_in_per_row_probe.py` | Per-row cosine + magnitude diff |
| `experiments/utils/softplus_stability_probe.py` | Verify softplus numerics on real `a+dt_bias` distribution |
| `experiments/utils/gated_formula_probe.py` | Mix-and-match input isolation |
| `experiments/utils/substep_compare.py` | Compare two npz substep dumps |
| `experiments/91r_per_layer_diff.py` | Per-layer cosine vs HF |
| `experiments/91s_layer2_full_substep_dump.py` | Template: ttnn substep dump for a layer |
| `experiments/91t_layer2_substep_compare.py` | Template: substep comparison |
