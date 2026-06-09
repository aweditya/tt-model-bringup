# Gemma 4 layout-op elimination — plan of action (2026-06-08)

User pivot: pause spec-dec work, attack the 33% layout overhead in
vanilla Gemma 4 forward. Tracy capture
(`~/tt-xla/.cache/perf_logs/tracy_gemma4_v2_132855/.logs/` on qb1)
shows actual matmul = 8%, layout = 33% of host wall.

## Top targets — UPDATED 2026-06-08 PM with clean N=4 device CSV

The original ranking was HOST WALL — turns out dispatch overhead and
device-kernel time are completely decoupled. Re-ran tracy with
`GM4_NUM_LAYERS_OVERRIDE=4` to avoid marker overflow → clean
`ops_perf_results_*.csv` with populated `OP CODE` + `DEVICE KERNEL
DURATION`. The true ranking:

| Op | **device kernel %** | host wall % | calls (N=4 fwd) | priority |
|---|---|---|---|---|
| **TypecastDeviceOperation** | **38%** | 9% | 308 | **HIGHEST** |
| **TilizeDeviceOperation** | **22%** | 5% | 200 | **HIGH** |
| MatmulDeviceOperation | 16% | 8% | 116 | actual compute |
| BinaryNgDeviceOperation | 13% | 5% | 92 | mul/add (RoPE, residuals) |
| TernaryDeviceOperation | 4.5% | 2% | 32 | addcmul (RoPE fused) |
| UntilizeDeviceOperation | 3.4% | <1% | 24 | small |
| UnaryNgDeviceOperation | 2.8% | <1% | 20 | small |
| AllGather | 0.6% | 2% | 36 | small |
| **LayerNormDeviceOperation** | **0.0%** (28us total!) | 8% | 116 | NOT a target |
| **SliceDeviceOperation** | **0.0%** | 10% | 160 | NOT a target |
| ReshapeView | 0.0% | 4% | 64 | NOT a target |
| TilizeWithValPadding | 0.0% | 8% | 192 | NOT a target |
| InterleavedToSharded | 0.0% | 4% | 64 | NOT a target |
| ConcatDeviceOperation | 0.0% | 3% | 48 | NOT a target |
| PagedFusedUpdateCache | 0.0% | 2% | 32 | already fast |

### Key reversal

The plan's original priorities (sharded RMSNorm, slice reduction,
TileWithValPadding pre-stage) attack **near-zero-cost device ops**.
Host wall = (kernel + dispatch + sync) was misleading on small ops.

**Real targets**: Typecast and Tilize, which together are **60% of
device kernel time**. Eliminating Typecast (~38%) could drop the
forward by ~18ms at the 47ms baseline = **1.6× speedup**.

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

### Step 3: REVISED fix priorities (post device-CSV finding)

The new device-time ranking puts Typecast (38%) and Tilize (22%) far
ahead of everything else. Hypothesis (needs confirmation):

1. **Typecast tax = fp32_dest_acc + bf16 downstream**.
   Our matmul config has `fp32_dest_acc_en=True` (line 106) — the
   matmul accumulator stays in fp32 for precision. The packer then
   converts the result back to bf16 for the downstream rms_norm /
   binary / next matmul. THAT conversion shows up as
   TypecastDeviceOperation. We can't simply turn off fp32_dest_acc
   (we know from `[[bf16-chain-drift-at-B-gt-1]]` that we need it).

   **Real fix options**:
   - (a) Specify `output_dtype=bfloat16` explicitly on matmul calls —
     might be no-op if packer already does this, but worth measuring.
   - (b) Reduce matmul COUNT — every fused matmul kills a typecast.
     `paged_fused_update_cache` (already done), QKV concat-fuse
     (worth exploring), gate+up fuse for SwiGLU (Tenstorrent does
     this).
   - (c) Keep certain chains in fp32 (norm → matmul → norm) so the
     typecast happens ONCE per layer, not per-matmul.

2. **Tilize 22%** — fewer but bigger. Likely sources:
   - matmul output ROW_MAJOR → next op TILE
   - rms_norm output dtype/layout mismatches
   - cos/sin table ROW_MAJOR → embedding lookup TILE conversion
     (every layer's RoPE chain)

   **Real fix**: persistent TILE rep for cos/sin tables; specify
   `output_mem_config` for matmul + layernorm to skip the implicit
   conversion.

### REMOVED priorities (do NOT pursue based on data)
- ~~Sharded decode rms_norm (#194)~~: LayerNorm is **0.0% device
  time, 28us total over 116 calls**. Already negligible — chasing
  this is misallocated effort. (The audit doc claimed "claimed ~10×
  speedup" but at 28us total it doesn't matter.)
- ~~Slice elimination~~: 160 slices, 0.0% device time. Pure dispatch
  overhead — already free.
- ~~Reshape/Concat reduction~~: same — host wall ≠ device cost.

### Final strategy after #292 research (research/precision_long_context_2026-06-08.md)

The research note nails down the precision tradeoff. Key facts:

- **`fp32_dest_acc` is real, but K-dependent**. The mantissa-rounding
  error in bf16 accumulation grows as O(sqrt(K)·eps). For K ≤ 4096
  (Q/K/V/gate/up projections) it's invisible to cosine; for o_proj,
  down_proj, lm_head, and **S·V where K = sequence length**, it
  matters and gets worse with context.
- **Typecast as a separate op is Tenstorrent-specific**. NVIDIA folds
  the FP32→bf16 cast into the MMA epilogue at ~0% cost. Tensix issues
  it as a packer instruction — what Tracy reports.
- **No production stack runs fp32 KV**. All default to model dtype.
  Even DeepSeek-V3's aggressive FP8 deployment **explicitly keeps
  attention + norms + lm_head in BF16/FP32**.

### Concrete fix order (replaces earlier guesses)

**Step 0: BLOCKER GATE — #291 long-context argmax baseline captured**
(in flight at the time of this revision). No Step 1+ ships until that
baseline is committed AND the change keeps it green.

**Step 1: (b) Reduce matmul COUNT via fusion — DO FIRST**
- QKV concat-fuse: one matmul instead of three for Q, K, V projections
- gate+up SwiGLU fuse: one matmul instead of two for MLP gate, up
- Halves the matmul→typecast pairs in attention + MLP
- Pure win: no precision change, no long-context risk
- Standard fusion every production stack already does
- Per layer: ~5 matmuls → ~3 matmuls = ~40% matmul-typecast pair reduction

**Step 2: (d) Selectively disable `fp32_dest_acc` on small-K matmuls — GATED**
- SAFE to disable: Q/K/V/gate/up where K ≤ 4096
- KEEP enabled: o_proj (K = num_heads × head_dim), down_proj (K =
  intermediate_size = 15360), lm_head (K = hidden, feeds argmax),
  attention S·V (K = sequence length — the long-context blast radius)
- Per-op probe: flip ONE matmul, run cosine ladder + needle, expand
- Gate sequence: (i) #291 argmax baseline still matches at L=2048,
  (ii) #294 needle haystack passes at L=4k AND L=32k, (iii) cosine
  ladder ≥ 0.999 at every layer

**Step 3: (a) Verify matmul output is already bf16 — sanity check**
- Grep `ttnn.matmul` for any `output_dtype=fp32`; should be none
- If found, fix (low-risk)

**Step 4: (c) fp32 chain handling — DEFERRED**
- Doubles L1 footprint; conflicts with trace budget
- 35B precedent of fp32 RMSNorm TRISC hang
- Only revisit if Steps 1-3 don't get us under 30 ms/tok

### Projected impact
- Step 1 alone: ~40% Typecast reduction (5 matmul/layer → 3) → ~15% wall
  saved → 47 ms → ~40 ms/tok
- Step 1 + Step 2: 38% Typecast → ~10-15% Typecast → 47 ms → ~32 ms/tok
- 1.5× total speedup, all without touching long-context accuracy

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
