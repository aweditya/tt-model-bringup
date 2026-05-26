# DeltaNet 4-linear input-projection fusion — plan

**Date**: 2026-05-12
**Target file**: `experiments/91f_qwen36_27b_full_ondevice.py` lines 135-138
**Phase tag**: "C'-DN-fusion" (independent of the numbered C'-phase order)
**Estimated win**: ~3-5 ms/tok decode (-2% to -2.5%)

## The opportunity

`deltanet_step_ondevice` currently runs **four separate `ttnn.linear` calls** on the same input `h_tt`:

```python
h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
mixed_qkv = ttnn.linear(h_tt, w_tt['in_proj_qkv'], compute_kernel_config=hifi4)  # [1, CONV_DIM]
z_tt      = ttnn.linear(h_tt, w_tt['in_proj_z'],   compute_kernel_config=hifi4)  # [1, VAL_DIM]
a_tt      = ttnn.linear(h_tt, w_tt['in_proj_a'],   compute_kernel_config=hifi4)  # [1, N_V_HEADS]
b_tt      = ttnn.linear(h_tt, w_tt['in_proj_b'],   compute_kernel_config=hifi4)  # [1, N_V_HEADS]
```

Each call:
1. Reads `h_tt` from L1
2. Reads its weight from DRAM
3. Issues a dispatch (~30 µs Python → ttnn → device)
4. Writes its output to L1

All four read the SAME `h_tt`. We can concatenate the four weight matrices into one and do a single `ttnn.linear`, then slice the output.

## Shape arithmetic for Qwen3.6-27B

- `HIDDEN = 5120`
- `N_K_HEADS = 16`, `K_DIM = 128` → `KEY_DIM = 2048`
- `N_V_HEADS = 32`, `V_DIM = 128` → `VAL_DIM = 4096`
- `CONV_DIM = 2 × KEY_DIM + VAL_DIM = 8192`

| Linear | Weight shape | Output shape |
|---|---|---|
| in_proj_qkv | [5120, 8192]   | [1, 8192] |
| in_proj_z   | [5120, 4096]   | [1, 4096] |
| in_proj_a   | [5120, 32]     | [1, 32]   |
| in_proj_b   | [5120, 32]     | [1, 32]   |
| **fused**   | **[5120, 12352]** | **[1, 12352]** |

Total weight params: 5120 × 12352 ≈ 63M (identical sum of the four, just laid out contiguously).

## Why this is essentially free correctness-wise

The math is identical. `[h] · concat(W1, W2, W3, W4) = concat([h]·W1, [h]·W2, [h]·W3, [h]·W4)`. We just slice the result back into 4 pieces. The slice positions are all 32-tile-aligned:

| Piece | Slice range | Tiles |
|---|---|---|
| mixed_qkv | [0, 0]      to [1, 8192]   | 256 |
| z_tt      | [0, 8192]   to [1, 12288]  | 128 |
| a_tt      | [0, 12288]  to [1, 12320]  | 1   |
| b_tt      | [0, 12320]  to [1, 12352]  | 1   |

Every boundary is a multiple of 32 (the tile width). `ttnn.slice` will be a view-only op (no data movement) on bf16/fp32 outputs.

## Implementation steps

### Step 1: Weight loader — produce fused weight

Modify `load_layer_weights_all` in `91f` (around line 95-101). When `layer_type == 'linear_attention'`:

```python
# After loading the four individual weights:
w_qkv = weights.pop('in_proj_qkv')   # [HIDDEN, CONV_DIM]
w_z   = weights.pop('in_proj_z')     # [HIDDEN, VAL_DIM]
w_a   = weights.pop('in_proj_a')     # [HIDDEN, N_V_HEADS]
w_b   = weights.pop('in_proj_b')     # [HIDDEN, N_V_HEADS]
weights['in_proj_all'] = np.concatenate([w_qkv, w_z, w_a, w_b], axis=1)
```

Note the `.pop()` — the OLD keys are removed so we don't waste DRAM uploading both fused and unfused. The fused weight is bf8 by default (matches the prior `'proj' in k` heuristic if we rename or add 'proj' to the new key — it does contain 'proj'). Verified.

### Step 2: Forward path — single linear + 4 slices

In `deltanet_step_ondevice`, replace lines 135-138 with:

```python
h_tt = ttnn.rms_norm(x_tt, weight=w_tt['input_layernorm'], epsilon=EPS)
all_tt = ttnn.linear(h_tt, w_tt['in_proj_all'], compute_kernel_config=hifi4)
mixed_qkv = ttnn.slice(all_tt, [0, 0],        [1, CONV_DIM])
z_tt      = ttnn.slice(all_tt, [0, CONV_DIM], [1, CONV_DIM + VAL_DIM])
a_tt      = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM],            [1, CONV_DIM + VAL_DIM + N_V_HEADS])
b_tt      = ttnn.slice(all_tt, [0, CONV_DIM + VAL_DIM + N_V_HEADS], [1, CONV_DIM + VAL_DIM + 2 * N_V_HEADS])
```

The rest of `deltanet_step_ondevice` (lines 140+) is unchanged.

### Step 3: Correctness gate

- Add a probe `experiments/utils/deltanet_fusion_probe.py` that runs deltanet_step with old and new weight layout side-by-side on the same input, asserts cosine ≥ 0.9999
- Run `91r_per_layer_diff.py` for DeltaNet layers (the default sample 0, 1, 2 already covers them) — gate ≥ 0.9997
- Run Paris demo sanity

### Step 4: Perf measurement

Run `perf_baseline.py --phase DN-fusion` and `perf_diff.py`. Expected delta:
- single_deltanet_step: 3.41 → ~3.30 ms (about -0.1 ms)
- decode step: 210 → ~206 ms (about -4 ms)

## Where the win comes from (and where it doesn't)

**Wins:**
- 4 dispatches → 1 dispatch. Save ~3 × 30 µs Python overhead per layer × 48 layers = **~4.3 ms/tok decode** in eager mode.
- Same total weight bytes read — bandwidth is unchanged. (This is NOT a bandwidth optimization.)
- After C'4 trace capture: Python overhead is gone anyway, so this fusion's contribution disappears. **The win is eager-mode only.**

**Why we still want it:**
- Smaller eager-mode dispatch count helps the dispatcher pipeline (same insight as C'1)
- Makes the deltanet step more compact and audit-friendly
- Lays groundwork for chunked prefill (C'5), where the in_proj is applied to a chunk of tokens — fusion may matter more there because the matmul itself becomes substantial work

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Slice on `[1, 12352]` output has unexpected shape constraint | Low | probe verifies; all boundaries are 32-aligned |
| Fused weight matrix exceeds some ttnn matmul size limit | Low | 5120 × 12352 is well within range; existing in_proj_qkv at 5120×8192 already works |
| bf8 weight quantization differs slightly when fused vs separate (different per-channel scaling) | Medium | check on the probe; if so, gate on quantization-agnostic test |
| `np.concatenate` along axis=1 fails for non-contiguous loads | Low | force `.copy()` after concat |

## When NOT to do this

If C'4 (trace capture) lands first, this fusion's win disappears. Decision: if C'5 (chunked prefill) goes ahead of C'4, do this fusion AS PART of C'5 because the math reuse is direct.

## Non-negotiables this plan honors

- Plan first (this doc) ✓
- One variable at a time — pure dispatch-count reduction, no math change ✓
- Permanent files (probe, analysis) ✓
- Correctness gate before perf measurement ✓
- Remote-only execution (probe + 91r + demo + perf on qb2) ✓
- No /tmp (logs/JSONs in `~/tt-xla/.cache/`) ✓

## Estimated implementation effort

~30 LOC change across `91f` (weight loader + forward path). Probe ~80 LOC. Total ~2 hours of careful work, dominated by the validation cycle.

**Verdict:** small but clean win. Best done either right before C'5 (so they share the implementation discipline) or as a quick interlude between bigger phases.
