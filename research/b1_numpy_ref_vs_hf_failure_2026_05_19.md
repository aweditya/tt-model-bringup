# B.1 numpy ref FAILS HF validation — 2026-05-19

## What we set out to do

Phase B.1 of the prefill-kernel build:
- Write `91w` numpy prefill reference (seq_len=128) — done
- Validate `91w` against HF Qwen3.6-27B prefill via `91x` HF oracle — **THIS FAILED**

## Result

| Layer | HF per-pos norm | Our per-pos norm | Min cosine | Median cosine | Verdict |
|---|---|---|---|---|---|
| 0 (DeltaNet) | 79.31 | 5.91 | -0.43 | 0.75 | **Uncorrelated** |
| 3 (Gated Attn) | 90.98 | 19.63 | 0.65 | 0.86 | Major bug |

## Critical diagnostic

Per-position cosine for layer 0 at position 0 = **0.72**. Position 0 of a
prefill is just the single-token decode (no prior state). So the bug is in
the single-token math (`91b._91b.deltanet_layer`), NOT in the prefill loop
wrapper.

Our 91b numpy ref has been wrong all along. Decode production uses
`owned_gdn` (custom kernel), validated separately — that's why this stayed
hidden. The 91b "validation" earlier (91o, 91p) probably only matched the
specific prompt "The capital of France is" by accident; random input
exposes the drift.

## Why this happened

1. **Model has revised**: `text_cfg.model_type = qwen3_5_text` — HF
   transformers ships the official Qwen3.6 modeling code at
   `transformers/models/qwen3_5/modeling_qwen3_5.py`. Our 91b was
   developed early in the bringup against possibly-different HF code.
2. **Production decode = owned_gdn**: our kernel is closer to correct
   than 91b, so 91b's drift didn't propagate to user-visible output.
3. **No round-trip validation in B′3**: original validation gate was
   "ttnn matches numpy ref", not "numpy ref matches HF". A drifting
   numpy ref + drifting ttnn impl that diverged together would have
   passed.

## What this means for Phase B (prefill kernel build)

- **GOOD news**: we caught the bug BEFORE writing 400+ lines of ttnn
  prefill code. This is exactly why the validation gate exists.
- **BAD news**: we need to fix 91b's `deltanet_layer` math before
  proceeding. The HF Qwen3.6-27B implementation is the source of truth.
- **Possible follow-on**: if 91b is wrong about DeltaNet math, our
  `owned_gdn` kernel might also be subtly wrong vs HF. Worth a separate
  validation pass after fixing 91b. (User-visible output is coherent so
  the drift is small, but we should know precisely.)

## Debug plan (Phase B.1.5)

1. **Read HF `Qwen3_5` modeling source** at
   `transformers/models/qwen3_5/modeling_qwen3_5.py`. Specifically find:
   - The Qwen3.5 / Qwen3-Next "linear_attention" class
   - Its `forward` method (slow torch fallback, since fla is not installed)
   - Compare math step-by-step to `91b.deltanet_layer`

2. **Identify divergence(s)**. Hypotheses:
   - Activation function mismatch (silu vs swish vs something else)
   - Decay formula sign / normalization
   - L2-norm placement
   - Output gate placement (pre vs post out_proj)
   - Missing scale factor
   - Wrong residual placement

3. **Fix `91b.deltanet_layer`**. Maybe make a `91b_v2_qwen36_27b_numpy_ref.py`
   to preserve the old one for archeological reference.

4. **Re-run `91x`**. Gate: per-position cos ≥ 0.999.

5. **Validate `owned_gdn` against fixed `91b_v2`**. If owned_gdn was based
   on 91b's incorrect math, it may need a fix too — but production output
   is coherent so any drift is small. Document precisely.

6. **Resume Phase B.2 (ttnn prefill on qb1)** only after numpy ref
   validation passes.

## Estimated extra time

- Reading HF source + finding bug: 2-4 hours
- Fixing 91b: 1-2 hours
- Re-validating: 1 bootstrap (HF oracle takes ~5 min)
- **Total: ~half-day delay** on Phase B timeline. Phase B was 12 days;
  now estimated 12.5 days.

## Why this is worth documenting

User asked to understand "the challenges of doing something like this".
This is one of the textbook challenges: a numpy reference can drift from
ground truth without ever being caught if downstream uses a different
codepath. The validation gate (compare math reference to authoritative
HF impl) is the single most important defensive practice. Always do this
BEFORE building the kernel, not after.

## Artifacts

- `experiments/91w_qwen36_27b_prefill_numpy_ref.py` — our numpy prefill (current, has bug)
- `experiments/91x_hf_prefill_oracle_seq128.py` — HF oracle (works correctly)
- `.cache/qwen36_27b_prefill_numpy_ref_seq128.npz` (qb1) — our numpy output
- `.cache/qwen36_27b_hf_prefill_oracle_seq128.npz` (local) — HF ground truth + per-position cosines
