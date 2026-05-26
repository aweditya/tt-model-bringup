# C'0.6 — RoPE table precompute (eliminate per-step host RoPE recompute)

**Date**: 2026-05-12
**Estimated win**: small in absolute ms but cleans the host-compute path; **hard prereq for C'4 trace capture**

## What changes

Currently `91l` recomputes one row of cos/sin per token at decode/prefill time via `rope_tables_for_pos(pos)` (lines 200-206), then uploads it (~100 µs PCIe per token). At 32k context decode (1000+ tokens), that's seconds of accumulated overhead.

Precompute the full `[MAX_POS, rotary_dim]` table at startup and slice on-device per step.

## Exact edits

### `experiments/91l_fp32_residual_generate.py`

Replace lines 196-206:

```python
# BEFORE:
rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
half_rot = rotary_dim // 2
freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

def rope_tables_for_pos(pos):
    angles = pos * freqs
    cos_np = np.concatenate([np.cos(angles), np.cos(angles)]).astype(np.float32)
    sin_np = np.concatenate([np.sin(angles), np.sin(angles)]).astype(np.float32)
    return (upload(cos_np, device, dtype=ttnn.float32),
            upload(sin_np, device, dtype=ttnn.float32))

# AFTER:
rotary_dim = int(cfg['head_dim'] * cfg['partial_rotary_factor'])
half_rot = rotary_dim // 2
freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))

# C'0.6: precompute the full RoPE table once. Per step we slice one row
# on-device — zero host compute, zero PCIe upload per token.
# Memory cost: 2 × MAX_POS × rotary_dim × 4 B. At MAX_POS=32k, rotary_dim=64,
# that's 16 MB total. Trivial on a 32 GB chip with ~5 GB headroom.
positions = np.arange(MAX_POS).astype(np.float32)
all_angles = positions[:, None] * freqs[None, :]
cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)  # [MAX_POS, rotary_dim]
sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)
```

Replace line 214:

```python
# BEFORE:
cos_tt, sin_tt = rope_tables_for_pos(cur_pos)

# AFTER:
cos_tt = ttnn.slice(cos_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
sin_tt = ttnn.slice(sin_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
```

### `experiments/91f_qwen36_27b_full_ondevice.py` (test harness)

Lines 391-398 — same pattern. Replace per-step compute with precomputed table + slice.

### `experiments/utils/perf_baseline.py`

Same pattern. Find `rope_for_pos` or equivalent inside its forward harness.

### `experiments/demo_qwen36_27b.py`

Same pattern. Find `rope_for_pos` or equivalent.

## Correctness gate

This is **math-identical**. The cos/sin values produced at `pos=k` should match the per-position recompute bit-for-bit (modulo whatever fp accuracy difference, but both paths use the same intermediate values). Gate:

- 91r per-layer cosine: identical to C'1 baseline (within bf16 noise)
- Paris demo first token = " Paris"

## Open question: ttnn.slice on TILE_LAYOUT row-slice

`ttnn.slice(cos_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])` slices 1 row out of MAX_POS rows. Tiles are 32×32. When `cur_pos` is not a multiple of 32, the slice crosses tile boundaries and ttnn must actually move data (not just adjust metadata). Cost should still be << the 100 µs PCIe upload it replaces, but if `ttnn.slice` complains about non-tile-aligned bounds, fallbacks:

1. **`ttnn.embedding(table, indices)`** — designed exactly for this. Indices=[cur_pos], table=cos_table_tt, output=one row.
2. **Keep table in ROW_MAJOR layout**, slice, then cast to TILE. Adds overhead but avoids the boundary issue.
3. **`ttnn.experimental.rotary_embedding`** with `token_index=pos` — this is C'3 proper, swap for the full native op. Larger change but folds C'0.6 + C'3 together.

Decision rule: if slice works at non-tile-aligned bounds (verify in the 91r run), ship as-is. Otherwise pick option 1.

## Effort

~30 LOC total across 4 files. The most time-consuming part is correctness validation (91r per-layer gate + Paris demo).

## When to apply

**Wait until the C'2 perf_baseline lands** so we don't tangle this change with the bf16 residual ablation. Currently C'2's edits sit uncommitted in the working tree. After perf_baseline resolves:
- If C'2 commits → apply C'0.6 on top
- If C'2 reverts → apply C'0.6 on clean tree

## Why this matters beyond the small ms win

C'4 trace capture **cannot** do per-step host compute. The trace replays exactly the device program captured at trace time. So our current `rope_tables_for_pos(pos)` — which builds a Python float, calls `np.cos`, calls `ttnn.from_torch` — would break trace capture. RoPE precompute is therefore a **hard prereq for C'4**, not just a nice-to-have.

Pair this with C'3 (native `ttnn.experimental.rotary_embedding`) once we want the partial-rotary slice/concat dance to also be trace-friendly.
