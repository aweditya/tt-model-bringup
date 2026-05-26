# reshape-view + dealloc-source audit — 2026-05-20

## Bug class
`ttnn.reshape` returns a VIEW into the source tensor's buffer. Deallocating
the source frees that backing memory. The next allocator request may reuse
the freed buffer and silently overwrite the view's data. Result: code that
reads the view after a few subsequent allocations gets garbage / zeros.

Caught at B.2.2 root-cause (research/B22_OVERNIGHT_LOG_2026_05_20.md):
v3 prefill's `x_seq` view of `dn_out_4d_tile` got zeroed mid-MLP because
`rms_norm` / `linear` allocations after the dealloc reused the freed
buffer.

## Audit scope
`experiments/serve/server_tp.py` only — the active production / probe code.
Tool: Python script matching `target = ttnn.reshape(source, ...)` followed
within 15 lines by `ttnn.deallocate(source)`.

Total matches: **20 sites**.

## Per-site triage

### Active production / probe path (v3 prefill)

| Site | Status | Notes |
|---|---|---|
| L1445 — `_x_seq_view = reshape(embed_raw)` | ✅ **FIXED** (clone) | x_seq is fresh allocation |
| L1632 — `_x_view = reshape(dn_out_4d_tile)` | ✅ **FIXED** (clone) | x_seq is fresh allocation |
| L1470 — `cos_seq_tt = reshape(cos_seq_raw)` | ✅ **FIXED** (clone, defensive) | held until first attn layer for RoPE; small tensor so was likely safe, but applied clone for consistency |
| L1471 — `sin_seq_tt = reshape(sin_seq_raw)` | ✅ **FIXED** (clone, defensive) | same as cos |
| L1578 — `x_pos_4d = reshape(x_pos_out)` in DN inner loop | ⚪ Safe | x_pos_4d consumed by `to_layout` (copy) before `deallocate(x_pos_out)`; no read of view after dealloc |

### Internal to DN / Attn step functions (lines 680-1382)

These reshapes produce short-lived intermediates that are consumed within
2-3 lines, often via `to_layout` (which copies) or `mul`/`add` etc., before
the source is deallocated. Low risk class. **No fix needed unless symptoms appear.**

Sites:
- L680 (custom AG+sum reshape — consumed by `ttnn.sum` before dealloc)
- L750/759 (mixed_col in DN — passed to conv1d kernel before dealloc)
- L756 (conv_out — consumed by next op chain)
- L816 (decay reshape — consumed by gating immediately)
- L1294-1300 (qg/k_tt/v_tt in attn — passed to RoPE/SDPA)
- L1382 (attn_flat — passed to w_o matmul)

### Other probe handlers (lines 3046+, 3409+, 3528+, 4042+)

These are in non-production probe handlers (`probe_*` functions invoked
only by the debug client). Not in `handle_generate_tp` call path.
**Same trap class but no production exposure.** Could be future bugs if
those probes are extended/used; not blocking ship.

Sites:
- L3046, L3083 (probe_multirow_construct_vs_per_position)
- L3409, L3441 (some prefill probe)
- L3528 (another prefill probe)
- L4042 (handle_probe_explicit_all_reduce_tp)

## Fix pattern (canonical)

```python
view = ttnn.reshape(source, target_shape)
result = ttnn.clone(view)         # fresh allocation, severed from source
ttnn.deallocate(view)
ttnn.deallocate(source)           # now safe — result doesn't depend on source
# ... use `result` freely across many ops ...
```

The `clone` materializes data into a fresh DRAM allocation that the
allocator marks as live. `deallocate(source)` then only frees the
original buffer, which the allocator can recycle without affecting `result`.

## Detection heuristic

Risk of latent corruption is proportional to:
- **Size of the freed buffer** (large buffers more likely to be reused)
- **Number of allocations between dealloc and final read of view** (more
  allocations = more chances of reuse)
- **Lifetime of the view** (held across function boundaries vs immediate use)

The B.2.2 v3 case scored high on all three: x_seq is [5, 5120] bf16
= 51 KB, MLP does ~10 allocations after the dealloc before the residual
add, and x_seq was held from layer entry to layer end (~15 ops). Perfect storm.

cos_seq/sin_seq are [5, 128] = 1.3 KB and used within 1-2 attention layers.
Smaller buffer = less likely to be reused, fewer intervening allocations.
Empirically not corrupted (cos=1.0 at seq=5 pre-fix), but fixed defensively.

## Recommendation
Adopt as a lint rule for future tt-metal code: any `ttnn.reshape` whose
result is held across multiple subsequent allocations should be followed by
`ttnn.clone`. Cost: one memcpy per long-lived reshape. Pays for itself by
preventing silent corruption that's brutal to debug (took 10 tests + 8
bootstraps in this session to localize).
