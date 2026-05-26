# DeltaNet Bug Localized — RMSNormGated on `[48, 128]` shape

**Date**: 2026-05-12
**Status**: Bug location pinpointed. Cause hypothesized but not yet verified.

## The data

`experiments/91t_layer2_substep_compare.py` after fixing layout mappings:

```
PERFECT (cosine = 1.0):
                  layer.in    1.00000
       input_layernorm.out    1.00000

EXCELLENT (≥ 0.9999):
              in_proj_*.out    0.99985 – 0.99996
      conv1d.out (pre-silu)    0.99997
       linear_attn.norm.in    0.99996   ← recurrence output is CORRECT

BUG STARTS HERE:
   linear_attn.norm.out (=gated)   0.80938 worst   ← problem step
          out_proj.out             0.74508 worst
   post_deltanet                   0.94055 worst
   post_attn_layernorm.out         0.76003 worst
   mlp.gate_proj_silu              0.69375 worst
   mlp.up_proj.out                 0.70890 worst
   mlp.down_proj.out               −0.067 to 0.93
   layer.out                       0.50816 worst
```

## What this tells us

The bug is in one of:
- `ttnn.rms_norm(out_per_head, weight=linear_attn_norm)` where `out_per_head` is shape `[48, 128]` and weight is `[128]`
- `ttnn.silu(z_tt)` (less likely — silu is simple element-wise)
- `ttnn.mul(out_normed, silu_z)` between shapes `[1, 6144]` and `[1, 6144]` (very unlikely)

Most suspicious: **`ttnn.rms_norm` at shape `[48, 128]`**. Per-row RMSNorm is mathematically scale-invariant — input cosine 0.99996 MUST give output cosine ≥ 0.99996 if the formula is correct. We see 0.81. Therefore the op isn't computing what we expect.

## Top hypotheses

1. **Tile-padding contaminating variance** — TILE_LAYOUT pads `[48, 128]` to `[64, 128]` internally (next multiple of 32). If the variance reduction includes the padded rows or columns, the per-row normalization is off.
2. **Weight broadcast against the leading dim** — weight `[128]` should be applied per-column (broadcast across rows). Maybe ttnn applies it differently for this 2D shape.
3. **Reshape from `[1, 6144]` → `[48, 128]` doesn't reorder data in TILE_LAYOUT** — the reshape may be a view-only operation that doesn't actually move data, so subsequent ops see the original layout.
4. **fp32 input + bf16 weight produces unexpected output dtype** — but our 91k probe showed `rms_norm(fp32, bf16) → fp32`. Unless there's a different path at the `[48, 128]` shape.

## Why earlier validation missed this

- B'7 (full 64-layer forward) was validated by cosine to numpy ref, but the numpy ref ALSO used per-head RMSNorm (the same way), so they agreed even when both were buggy compared to HF.
- 91p (layer 0 substep) had cosine 0.997 — the RMSNorm issue was small enough at layer 0's input magnitudes to look like "minor drift."

## Test in next session

Write `experiments/utils/ttnn_rms_norm_probe.py`:
- Construct an exact copy of layer 2's `out_per_head` (load from `~/tt-xla/.cache/ttnn_layer2_substeps_full.npz`)
- Construct an exact copy of the `linear_attn.norm.weight`
- Apply ttnn.rms_norm via the same call we use
- Apply RMSNorm by hand in numpy using the same formula
- Compare. If they differ → ttnn.rms_norm is the bug at this shape.

## Fix candidate

If ttnn.rms_norm is broken at `[48, 128]`, implement RMSNorm by hand:
```python
# Replace ttnn.rms_norm(out_per_head, weight, eps) with:
sq = ttnn.mul(out_per_head, out_per_head)
mean_sq = ttnn.div(ttnn.sum(sq, dim=-1, keepdim=True), V_DIM)
inv_rms = ttnn.rsqrt(ttnn.add(mean_sq, EPS))
out_normed = ttnn.mul(ttnn.mul(out_per_head, inv_rms), weight)
```

We already use these primitives elsewhere in DeltaNet (L2 norm Q/K is essentially the same math).

## Session summary

This was the productive yield of the patient line-by-line substep approach. Even when the recurrence audit didn't find anything, the substep capture pinpointed the bug to a single ttnn op call. From cosine 0 (random) at start of session → bug isolated to one operation at end.

Total session bugs found: 6 in our impl + 1 upstream ttnn JIT bug + this 7th (suspected ttnn op behavior). Per-layer cosine for HF-verified parts (full attention) is 0.9999+ across the entire model.

Next session goal: verify hypothesis 1-4 with the rms_norm probe, then patch.
