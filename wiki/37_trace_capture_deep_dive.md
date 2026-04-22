# Wiki 37: Trace Capture Deep Dive — From 49 to 136 tok/sec

## Q: How does trace capture work on Blackhole?

**A:** `ttnn.begin_trace_capture` records the device-side operation graph. `ttnn.execute_trace` replays it without Python dispatch overhead. The trace records the GRAPH, not the VALUES — device tensor contents can be updated between replays via `ttnn.copy()`.

```python
# Capture
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# ... run ops (recorded, not truly executed) ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Replay (updates input buffers, then replays)
ttnn.copy(new_input, input_buffer)
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
output = from_dev(output_ref, shape)
```

Result: **7.4ms/tok (135.6 tok/sec)** — 2.83x faster than non-traced (20ms/tok).

## Q: What are the requirements for trace capture?

**A:** From upstream docs + our experiments:

1. **Fixed tensor shapes** — all tensors must have the same shape at every replay. Single-token decode (1×1×896) qualifies perfectly.
2. **No host-side control flow** — no Python if/else based on tensor values during the traced section.
3. **Pre-allocated buffers** — input/output tensors must exist before trace capture. Update contents with `ttnn.copy()`.
4. **Program cache should be enabled** — `ttnn.device.enable_program_cache(device)` before capture. Without it, each trace captures uncompiled kernels.
5. **Warmup first** — run one non-traced iteration to compile all kernels before capturing.

## Q: What can't be traced?

**A:** Python scalar parameters that change per step get **baked into the trace**:

| Parameter | Type | Traceable? |
|-----------|------|-----------|
| Embedding input | Device tensor (via `ttnn.copy`) | ✅ Yes |
| RoPE cos/sin | Device tensor (via `ttnn.copy`) | ✅ Yes |
| `cur_pos` in SDPA decode | Python `[int]` | ❌ Frozen |
| `cur_pos_tensor` in SDPA decode | Device tensor (int32) | ✅ Yes |
| `update_index` in `update_cache_for_token_` | Python `int` | ❌ Frozen |

The `cur_pos_tensor` parameter exists and works (verified in exp 52c). But `update_cache_for_token_` only accepts integer `update_index` — this is the blocking issue.

## Q: What approaches were tried for correct traced decode?

**A:**

### Approach 1: Single trace with stale positions (exp 52)
Bake one position into the trace. Replay with updated embeddings and RoPE, but position stays fixed. Result: 7ms/tok but text quality degrades ("the city, , is a city of the").

### Approach 2: Per-position traces (exp 52b)
Capture a separate trace for each decode position (5, 6, 7, ...). Problem: each trace capture takes ~29ms (same as non-traced), and multiple active traces corrupt each other's buffers. The warning "Allocating device buffers is unsafe due to the existence of an active trace" explains the garbled output.

### Approach 3: Hybrid (proposed but not yet implemented)
Use `cur_pos_tensor` for SDPA (traced) and handle KV cache update outside the trace. Requires splitting the trace or using `paged_update_cache` with `update_idxs_tensor` (which needs HEIGHT_SHARDED tensors).

## Q: What's the path to correct 136+ tok/sec?

**A:** The official tt-metal models use:
- `paged_update_cache` with `update_idxs_tensor` — accepts device tensor for position
- `scaled_dot_product_attention_decode` with `cur_pos_tensor` — already works
- Both require **HEIGHT_SHARDED** tensor layouts

HEIGHT_SHARDED is the same refactor needed for:
1. Native `ttnn.experimental.rotary_embedding` (eliminates rotation matrix overhead)
2. L1 SRAM residency (eliminates DRAM round-trips between ops)
3. Correct traced decode (tensor-based positions)

This single refactor unlocks all three optimizations simultaneously.

## Q: What upstream issues are relevant?

**A:** From our research:
- **No Blackhole-specific SDPA decode blockers** — #30362 (paged SDPA PCC) was fixed
- **Single-device trace is stable** — #30762 (event sync during trace) was resolved
- **First-iteration latency is expected** — #24744 documented the RISC-V warmup cost (~5451 vs ~191 cycles)
- **HEIGHT_SHARDED RoPE exists** — #14540 (fused_rotary_embedding) merged Nov 2025, supports batch ≤ 32
- **nanoGPT training hangs on BH** — #36318 is non-deterministic, worth monitoring

---

*Experiments 52-52c. Trace capture: 7ms/tok proof-of-concept, 20ms/tok correct. April 2026.*
