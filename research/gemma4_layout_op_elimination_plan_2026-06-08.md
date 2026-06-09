# Gemma 4 layout-op elimination — plan of action (2026-06-08)

User pivot: pause spec-dec work, attack the 33% layout overhead in
vanilla Gemma 4 forward. Tracy capture
(`~/tt-xla/.cache/perf_logs/tracy_gemma4_v2_132855/.logs/` on qb1)
shows actual matmul = 8%, layout = 33% of host wall.

## Top targets

| Op | % host | est calls/fwd | likely source | first-pass fix |
|---|---|---|---|---|
| SliceDeviceOperation | **10%** | ~560 | RoPE row lookups + Q/KV-head splits in attention | hoist once per forward; persistent shape buffers |
| TypecastDeviceOperation | **9%** | ~1050 | bf16 ↔ fp32 transitions (somewhere upcasting) | audit `from_torch` dtype, `to_layout` dtype |
| TilizeWithValPadding | **8%** | ~600 | inputs to `paged_fused_update_cache` need TILE+pad | pre-stage cache K/V as TILE in update path |
| TilizeDeviceOperation | **5%** | ~660 | generic ROW_MAJOR → TILE before matmul | pre-stage RoPE tables as TILE? cos/sin currently ROW_MAJOR |
| UntilizeWithUnpadding | **5%** | ~260 | paged_sdpa output → unpacked | argmax path? readback path? |
| InterleavedToSharded | **4%** | ~230 | between matmuls switching shard pattern | persistent sharded outputs across layers |
| ReshapeView | **4%** | ~230 | per-head reshape, per-NKV split | usually free (view); count includes program build dispatch |
| ConcatDeviceOperation | **3%** | ~175 | RoPE rotate-half (when not roll-fused), QKV gather | already roll-fused on Gemma 4 — investigate residue |

**Sum**: ~48% of host time is layout-or-precision shuffling.

## Pre-work

The Gemma 4 v2 tracy capture **crashed post-process before populating
per-op DEVICE kernel duration** (we have HOST wall only, with op
names). Need clean device timing to confirm which ops are
host-bound vs kernel-bound. Options:

### Option A: smaller forward → smaller marker set, no overflow
Add `GM4_NUM_LAYERS_OVERRIDE=2` env knob to `server_gemma4_unified_ttnn`
that truncates the forward to N layers. v2 already comments on this as
the right fix:

> If marker overflow still happens, drop to 1 sliding layer + lm_head
> by truncating NUM_LAYERS via env GM4_NUM_LAYERS_OVERRIDE
> (not implemented here — would need server code knob).

Probe: `tracy_profile_one_gemma4_subset.py` — bootstrap with
`GM4_NUM_LAYERS_OVERRIDE=4` (1 sliding + 1 global + lm_head), capture
one forward. Marker count ÷ 12 → 615 ops, well under 12k limit.

### Option B: process the existing capture manually
The existing `cpp_device_perf_report.csv` has GLOBAL CALL COUNT
(unique per device op call). `tracy_ops_data.csv` has MessageName with
op name. Join on GLOBAL CALL COUNT to recover op names. Add helper
`experiments/utils/tracy_join_device_host.py`. Skips re-capture cost.

**Decision**: try **Option B first** (10-min effort, reuses existing
capture). If join recovers > 80% of device rows, ship the analysis
and move on. If not, fall back to Option A.

## Audit (work order)

### Step 1: name-join the existing capture (Option B)
- Write `experiments/utils/tracy_join_device_host.py`
- Output: top-20 device ops with KERNEL DURATION (not just host wall)
- Identify which ops are kernel-bound vs dispatch-bound

### Step 2: source-code grep — where does each top op come from
- Slice: grep `ttnn.slice` + `ttnn.embedding` + `ttnn.split`
- Typecast: grep `ttnn.typecast` + `dtype=` in matmul output configs +
  `from_torch(dtype=...)`
- TilizeWithValPadding: grep `paged_fused_update_cache` call sites +
  callers of `update_input_buffers`
- InterleavedToSharded: grep `interleaved_to_sharded` + per-matmul
  `output_mem_config`

### Step 3: one-line fixes
Likely candidates based on prior 27B / Gemma 4 experience:

1. **Persistent TILE cos/sin tables** — currently ROW_MAJOR (per
   target's pattern). Each `_lookup_rope` does
   `embedding (ROW_MAJOR) → to_layout(TILE)`. If we store TILE upfront,
   the per-call to_layout disappears. ~80-100 Tilize ops/fwd saved.

2. **`paged_fused_update_cache` input pre-tiled** — task #198
   already pending. K/V going into the cache currently need
   TilizeWithValPadding because the projection output is in a
   sharded layout the cache expects differently.

3. **bf16 throughout** — find the typecast spots. Most likely
   candidates: lm_head final, argmax input, RoPE rotation.

### Step 4: re-capture + diff
After each fix, re-run tracy_profile_one_gemma4_layer_v2 + diff against
baseline. Target: layout-op share drops 33% → < 20%.

## Constraints

- Each fix must NOT regress numerical correctness. Pre-existing tests:
  - decode tok argmax matches HF (#165 v0.2 gate)
  - per-layer cosine ladder for sanity
- Re-run multi-step chat after each push.
- Probes go in `experiments/cb/isolate/` per non-negotiables.
- Helper tools (CSV join, plot) in `experiments/utils/`.

## Estimated impact

If we cut layout from 33% → 15% (achievable per the 27B precedent):
- Per-step host time drops ~18% → ~9 ms/tok saved at 47 ms baseline
- Decode goes 47 → 38 ms/tok = **1.24× speedup**
- Prefill (= L × decode in our impl) gets the same per-token win
- Real prefill win comes from #290 (chunked prefill, 5-10× TTFT)
