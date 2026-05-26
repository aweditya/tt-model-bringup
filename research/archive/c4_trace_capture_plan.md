# C'4 — Trace capture for decode step

**Goal**: capture one full Qwen3.6-27B decode step as a single device program; execute via `ttnn.execute_trace` instead of dispatching ops one at a time. This eliminates Python-in-the-loop entirely on the hot path.

**Expected win**: -20 to -50 ms/tok decode. After C'1's pipelining surprise (host-blocking ops were hiding pipelining), the ceiling is higher than original estimates. With Python-in-the-loop gone entirely, we expect the device pipeline to compact further.

## Reference pattern (from `experiments/85_8b_full_bfp8.py:303-334`)

```python
# Warmup — pay JIT once
_ = decode_fn()
ttnn.synchronize_device(device)
device.enable_program_cache()

# Capture
update_buffers(next_id, pos)
tid = ttnn.begin_trace_capture(device, cq_id=0)
logits_ref = decode_fn()
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute (fast path)
for step in range(N):
    update_buffers(next_id, pos)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    logits = from_dev(logits_ref, (1, vocab_size))[0]
    next_id = int(np.argmax(logits))
```

`update_buffers` writes the new (token_id, cur_pos) into PRE-ALLOCATED device tensors via `ttnn.copy_host_to_device_tensor`. `decode_fn` reads from those pre-allocated tensors. The trace stays identical; only the input contents change per step.

## Trace-hostile patterns to fix in current code

| Pattern | Where | Fix |
|---|---|---|
| `cur_pos` Python int in `ttnn.slice` for RoPE lookup | C'0.6 added `cos_tt = ttnn.slice(cos_table_tt, [cur_pos, 0], ...)` in `91l` | Replace with `ttnn.embedding(cos_table_tt, cur_pos_tt)` — dynamic-index op that reads from device tensor |
| `cur_pos` in scatter index for KV cache write | C'1 `ttnn.scatter(... index=index_tt with cur_pos)` | Same fix — index tensor populated from `cur_pos_tt` via `ttnn.full` or pre-allocated index tensor updated per step |
| Host-side `embed_np[token_id]` lookup | `91l:209` | Pre-allocate `embed_tensor [1, HIDDEN]`; per step: write `embed_np[token_id]` to it via `copy_host_to_device_tensor` |
| Python-int `cur_pos` everywhere it appears | search-replace audit | Centralize: ONE `cur_pos_tt` device tensor; all sites read it |

## Implementation order

1. **Audit**: grep `cur_pos` (Python int) in `91l` and `91f`. Categorize each use:
   - Pure scalar (for if/control flow): fine; resolved at trace time
   - Used as op argument: needs to become a device tensor read
2. **Refactor `91l` forward path**:
   - Pre-allocate `embed_tt`, `cur_pos_tt`, `cos_row_tt`, `sin_row_tt` ONCE
   - `update_buffers(token_id, cur_pos)` writes to these via `copy_host_to_device_tensor`
   - `forward_token` reads only from these device tensors
3. **Refactor `91f.gated_attn_step_ondevice`**:
   - Scatter index must be a device tensor (built from `cur_pos_tt` via `ttnn.full_like` or pre-allocated `index_tt` updated per step)
   - SDPA already takes `cur_pos_tensor` ✓ no change
4. **Add trace path** to `91l`:
   - Warmup: one eager decode step to flush JIT
   - Capture: begin/end trace around `forward_token`
   - Execute: per-step `update_buffers` + `execute_trace`
5. **Gate via server**:
   - rsync 91l, 91f changes
   - `reload_kernels`
   - `run_91r` (validates eager path still works)
   - New endpoint `run_traced_decode` (Phase 2 server) OR add an env flag to `run_91r` that exercises the trace
6. **Perf measurement**: traced vs eager decode_step times

## Memory budget for trace

- Trace itself: stored in L1 dispatch buffers; small (~hundreds of KB based on prior exps)
- Pre-allocated input tensors: `embed_tt [1, HIDDEN]` = 10 KB, `cur_pos_tt [1] int32` = 4 B, `cos/sin_row_tt [1, ROTARY_DIM]` = 256 B each
- KV caches: same as today
- Logits output: pre-allocated `logits_tt [1, vocab_size]` = ~1 MB
- Total: negligible

## Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Some op inside the forward path captures Python value implicitly | Trace fails at end_trace_capture — error gives the offending op. Search-replace audit before capture |
| 2 | Trace pinned in L1 conflicts with weight residency | We have ~5 GB headroom after 27 GB weights; trace is small. |
| 3 | KV cache GROWS each step (different shape per cur_pos when sliced for SDPA)? | SDPA-decode handles this internally; cache is static-allocated, cur_pos drives logical length |
| 4 | Hot-reload via server breaks the captured trace | Restart server after C'4 capture; trace is part of server state. Phase 2: persist trace across reloads |
| 5 | First few execute_trace calls slow due to L1 warm-up | Skip first 3 steps in perf measurement |

## What this delivers

- Decode step: ~210 ms → ~160-180 ms (-50 ms estimated)
- Variance: nearly zero (no Python dispatch jitter)
- Foundation for C'7 multi-chip TP (trace replays cleanly across mesh)
- One less variable when debugging future perf phases

## Effort

~3-4 hours including the cur_pos audit, refactor, and validation. With the server in place, iteration cycles are 5 sec — this should be doable in one focused session.

## Why this is the right time

- C'0.6 (RoPE precompute) just shipped — eliminates per-step host RoPE compute, a hard prereq for trace
- C'1 ttnn.scatter for cache write — works, but uses cur_pos Python int; need device-tensor variant
- Persistent server works — iteration is fast
- Long-context blocked on writer hang anyway; short-context is where the wins are
