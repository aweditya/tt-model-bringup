# Reusable Diagnostic Utilities

These are NOT one-shot experiments. They are reusable tools that every model bringup is likely to need. When you reach for a `python -c`, write a utility instead — it goes here.

Each utility has CLI args, a docstring explaining when to use it, and is idempotent.

## What's in here

| Utility | When to use it |
|---|---|
| **`weight_audit.py`** | First thing during any new model bringup. Lists safetensors weight keys, computes stats, diffs your loader against safetensors. |
| **`hf_full_model_oracle.py`** | Establish ground truth: what does HF transformers predict for your prompt? Top-K logits + (optional) per-layer hidden states for downstream isolation testing. |
| **`hf_layer0_substep_dump.py`** | HF substep dump for one specific layer via PyTorch forward hooks. Supports `--input-from-hidden` to use the real production input (= hidden_N from a full HF forward), not just embed-of-prompt. |
| **`repeat_semantics_probe.py`** | Verify whether a library op like `ttnn.repeat` is tile-style or interleave-style. Cheap, run once per ttnn version. |
| **`patch_ttnn_llk_roundmode.py`** | Patch the upstream ttnn LLK `int → sfpi::RoundMode` bug. Walks the ttnn install, backs up affected files, replaces literal `0` with `sfpi::RoundMode::NearestEven`. Restore mode included. |
| **`ttnn_rms_norm_probe.py`** | Verify ttnn.rms_norm at a specific shape matches numpy. Useful when you suspect a ttnn op of misbehaving. |
| **`softplus_stability_probe.py`** | Verify softplus (`log(exp(x)+1)`) is numerically stable for your model's `a + dt_bias` distribution. Loads layer N's relevant weights + a real input and checks per-position. |
| **`gated_formula_probe.py`** | Mix-and-match `(our_input, hf_input) × (our_gate, hf_gate)` to isolate WHICH input is responsible for a downstream divergence. Pattern: if `HF/HF` matches but `OUR/OUR` doesn't, the bug is in your inputs. |
| **`norm_in_per_row_probe.py`** | When global cosine looks good but downstream is wrong, run this to check per-row cosines + per-row magnitudes. Catches the "constant magnitude ratio" signature of a missing scaling factor. |
| **`substep_compare.py`** | Compare two npz dumps (HF substep + your ttnn substep) with layout-aware key mapping. Reports per-substep cosines, sorted worst-first. |

## How to use these together (the workflow)

```bash
# 1. Establish ground truth
python experiments/utils/hf_full_model_oracle.py --dump-hidden-states

# 2. Run per-layer cosine diff (uses HF hidden states as inputs)
python experiments/91r_per_layer_diff.py --layers 0,1,2,3,7,...

# 3. If a specific layer drops cosine, substep-dump that layer (both sides)
python experiments/utils/hf_layer0_substep_dump.py --layer 2 --input-from-hidden
python experiments/91s_layer2_full_substep_dump.py   # template for any layer

# 4. Diff the substeps to find the first divergence
python experiments/91t_layer2_substep_compare.py     # template

# 5. If a substep diverges, mix-and-match to find which input is wrong
python experiments/utils/gated_formula_probe.py      # template for the gated-op case

# 6. If a tensor diverges with global cosine ≥ 0.99, per-row analysis
python experiments/utils/norm_in_per_row_probe.py    # template
```

See `wiki/debugging_methodology.md` for the full reasoning behind this workflow.

## How to add a new utility

If you've reached for a one-off `python -c` more than twice, promote it:

1. Drop the script in this directory with a clear name (`*_probe.py` for empirical investigations, `*_audit.py` for inspection-only, `*_oracle.py` for reference-data generation).
2. Add a docstring explaining the symptom that motivated it.
3. Use `argparse` so it's CLI-driven, not hardcoded.
4. Save outputs to `~/tt-xla/.cache/`, never `/tmp`.
5. Add it to the table above.
