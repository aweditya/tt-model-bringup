# Nemotron-3 Nano 30B-A3B — Trace Capture Plan-of-Action (v0.4.1)

**Owner**: Aditya Sriram • **Status**: planning • **Created**: 2026-06-05

## TL;DR

Current state (post v0.4.0h.a): warm decode at **0.26s/step ≈ 3.8 tok/s
warm / 5.0 steady-state** (~60× cumulative since v0.4.0d baseline).
n-step chain 7/7 PASS. Mamba2 is fully pure-ttnn. MoE combine on-device.

**v0.4.1.a diagnostic probe (commit `bfb04bb`)** confirms the trace
system hard-rejects host bridges with TT_FATAL "Writes are not allowed
inside a captured trace". Definitive blocker list captured (below).

Goal: trace capture (`v0.4.1`) to compress per-step dispatch on the
remaining ~14 ops × 23 mamba2 layers + similar for MoE + attn = ~1000
dispatches per step. Realistic outcome 2-3× further speedup → 0.2-0.3s
per step (3-5 tok/s), still 6-10× shy of the 30 tok/s v0.5 target.
The big additional wins are pre-trace (vocab-shard lm_head, RMSNorm
fusion, on-device MoE topk) but trace UNLOCKS those wins.

## Reuse mandate — what we already shipped, fork verbatim

- 27B traced decode at `server_tp.py:1820-1900` (B=32 traced, 232 tok/s
  aggregate, P5 cb5_traced)
- 35B traced decode at `server_35b_ttnn.py:2200-2280` (B=8 traced, B=1
  HTTP demo 3.13 tok/s)
- Gemma 4 traced decode at `server_gemma4_unified_ttnn.py:1850-1940`
  (47.5 ms/tok traced after vocab-shard lm_head, P22)
- Memory `[[ttnn-multi-trace-two-phase-warmup]]` — compile ALL paths
  before capturing ANY trace; without this, decode JIT compilation
  between captures corrupts prefill trace memory (CPU 99% hang seen
  on 27B/35B/Gemma 4 the first time we tried single-phase)
- Memory `[[feedback-two-phase-warmup]]` — same lesson, durable
- Memory `[[feedback-ttnn-trace-region-size]]` — default 50 MB
  suffices for non-chunked-prefill; bump to 400 MB for prefill+decode
  combined (we already set 400 MB in the dev harness)

**Read first**: pages 1840-1910 of `experiments/serve/server_tp.py`
(27B fast-path) for the literal pattern: state buffers, host inputs
through `cur_pos_buf`+`tok_id_buf`+`page_table_tt`, trace capture
boundary, replay loop.

## Prereqs we already have

✅ Bootstrap `state.mesh` resident (v0.3.0 all-layers-resident)
✅ Per-layer KV cache on device (v0.3.1.c step 2, `setup_paged_decode_state`)
✅ `cur_pos_buf` for device-side position advance (v0.3.1.c step 3b)
✅ `tok_id_buf` (or equivalent) for device-side token feed — verify
✅ Mamba2 conv_state ON-DEVICE (v0.4.0c persistent `conv_state_tt`)
✅ Mamba2 conv1d replaced by matmul-fold (v0.4.0e — eliminates the
   `groups=6144` conv1d kernel's heavy program-factory setup cost)

## Definitive trace-blocker list (audited via v0.4.1.a probe 2026-06-05)

In strict elimination order (each unblocks more of the decode path):

| # | Op | Location | Trace fix |
|---|---|---|---|
| 1 | `ttnn.from_torch(ids)` for embed | `embed_lookup` server.py:893-907 | Pre-allocate `tok_buf` (uint32 [1,1] device). Host writes via `ttnn.copy_host_to_device_tensor` OUTSIDE trace. Forward reads buffer → `ttnn.embedding` → ttnn.Tensor. Forks 27B `update_input_buffers` at `server_tp.py:2058-2079`. |
| 2 | `ttnn.to_torch(scores_tt)` + host argpartition + `ttnn.from_torch(topk_indices)` | `moe_block_eager_ep_tt` server.py:2618, 2658 | On-device router via `ttnn.topk` + `ttnn.embedding`-as-gather. Probe at `nemotron3_v040hb_ondevice_router_probe.py` shows cos=0.9997 but 6/8 tie-breaking mismatch — risky without long-context check. Defer until trace lets us iterate fast. |
| 3 | `ttnn.to_torch(h_input_tt)` + `ttnn.from_torch(sharded)` | `moe_block_eager_ep_tt` server.py:2640-2651 | **Investigated 2026-06-05.** No ttnn primitive exists for on-device replicate→shard (4 candidates failed in probe `nemotron3_v041c_reshard_probe.py`). `all_to_all_dispatch` rejects replicated h (shape-mismatch with sharded topk indices — probe `nemotron3_v041d_replicated_dispatch_probe.py`). Production fix per research: **dual-resident layer outputs** — each layer outputs `(block_repl, block_shard)` so MoE consumes the shard directly. Significant refactor scope. Alternative: Megatron-TP shard-throughout (heaviest). For now: **trace integration PARKED**; current 0.26s warm eager ≈ 3.8 tok/s steady is usable for long-context correctness work. |
| 4 | `ttnn.to_torch(h_final_tt)` + `apply_final_norm`/`apply_lm_head_and_argmax` numpy | server.py:920-960 | EITHER make pure-ttnn (logits/argmax stay on device) OR leave OUTSIDE the captured trace (27B's `forward_token_tp_inner` returns `traced_argmax_tt` on device; host-side readback happens AFTER `execute_trace`). 27B path is simpler. |

**Strategy**: tackle in order. Each item, validate via the v0.4.1.a
probe (which will progress further before hitting the next blocker).
When all four are clear, v0.4.1.b multi-trace + correctness gate.

## The OLD trace blocker (was eliminated by v0.4.0g)

**The Mamba2 SSD wrapper takes/returns NUMPY arrays.**

`mamba2_block_eager_tt` (server_nemotron3_nano_ttnn.py:1655-1980) does:
```python
y_list = []
for p in range(S):
    new_state, y_p = _step_mod.mamba2_decode_step_ttnn(
        x=x_inner_np[:, p, :, :],       # ← NUMPY input
        ...
        ssm_state=ssm_state,            # ← NUMPY in/out
        device=state.mesh,
    )
    ssm_state = new_state
    y_list.append(y_p)
y_post_ssd = np.stack(y_list, axis=1)   # ← NUMPY accumulator
y_flat = y_post_ssd.reshape(B, S, NH * HD)
y_tt = ttnn.from_torch(...)             # ← re-upload to device
```

Every layer does: device → numpy → device. This:
1. Breaks the trace (numpy ops aren't capturable).
2. Costs the readback+upload itself (probably ~5-10ms × 23 layers =
   115-230 ms of the current 0.7s step).

**Task #223 (v0.4.0) was incorrectly marked completed**. Re-opening.

## Roadmap

### v0.4.0g (prereq for v0.4.1) — Mamba2 SSD wrapper takes ttnn.Tensor I/O

Refactor `nemotron3_mamba2_step.mamba2_decode_step_ttnn` (and the host
of helpers around it) so:

- Input: `x_tt, B_tt, C_tt, dt_tt, ssm_state_tt` (all ttnn.Tensor on `state.mesh`)
- Output: `new_ssm_state_tt, y_tt` (both ttnn.Tensor)
- ssm_state stays resident across decode steps (already lifted in
  state.ssm_state_tt during reset_decode_state, but the SSD wrapper
  still receives/returns numpy)

The owned kernel (`nemotron3_mamba2_decode_owned`) already operates on
device tensors — the WRAPPER does the numpy bridging. We need a
ttnn-pure wrapper. There's likely already a `_kernel_call` path that
takes tensors; the wrapper just hides it behind numpy for legacy
probes.

**Sub-tasks**:
- v0.4.0g.1: read the kernel's existing tensor-mode entry point at
  `experiments/serve/nemotron3_mamba2_step.py:mamba2_decode_step_ttnn`
  (~lines 200-400 probably)
- v0.4.0g.2: add `mamba2_decode_step_ttnn_pure` variant that takes
  ttnn.Tensor in and returns ttnn.Tensor out. Reuse same kernel call.
- v0.4.0g.3: update `mamba2_block_eager_tt` to use the new pure
  variant; remove the numpy roundtrip on the decode path (gate S==1,
  prefill keeps numpy if needed).
- v0.4.0g.4: ssm_state stays as a TT tensor between layers — wire to
  `state.ssm_state_tt[L]` instead of `state.ssm_state_np[L]`.
- Gate: n-step chain ≥ 6/7 PASS, identical TT sequence to v0.4.0e.

**Likely perf** by itself (no trace yet): 115-230 ms saved →
0.5-0.6s step.

### v0.4.0h — on-device MoE host paths

Audit `moe_block_eager_ep_tt` for residual numpy roundtrips:
- Router topk (currently host-side `np.argpartition` over [B,128] logits)
- Combine weighted-sum (currently host-side `topk_weights × combine_out`)
- Shared expert + residual add (currently host-side `+`)

These were intentionally host-side at v0.1.4 EP because the dominant
cost was the all_to_all dispatch, not the topk. Now that conv1d is
gone, they may matter.

**Likely perf** by itself: 30-80 ms saved → 0.4-0.5s step.

### v0.4.1.a — single decode step trace

Same pattern as 27B/35B/Gemma 4:
1. Build host-input buffers (`tok_id_buf`, `cur_pos_buf`, `page_table_tt`)
   on device.
2. Define `step_forward_traced(state) → logits_tt` that reads buffers
   instead of accepting args.
3. Compile via 1 eager call (no trace) to warm the JIT for ALL paths
   present in the step.
4. `ttnn.begin_trace_capture(state.mesh, cq_id=0)`
5. Call `step_forward_traced(state)` (this captures the trace)
6. `ttnn.end_trace_capture(state.mesh, ...)`
7. Replay 100 steps via `ttnn.execute_trace(trace_id)`, comparing
   token-for-token vs eager.
**Gate**: 100/100 token-for-token match vs eager (commit `425acad`
state).

### v0.4.1.b — prefill+decode multi-trace (two-phase warmup)

This is where 27B/35B got bit by single-phase warmup (`[[ttnn-multi-trace-two-phase-warmup]]`).

Protocol (forks 27B `_capture_both_traces` at `server_tp.py:1870`):
1. **Phase 1 — compile ALL traces eagerly first**:
   - `_eager_prefill(state)` — runs the prefill path WITHOUT trace
   - `_eager_decode(state)` — runs the decode path WITHOUT trace
   - This pre-warms BOTH JIT caches.
2. **Phase 2 — capture both traces back-to-back**:
   - `begin_trace_capture` → run prefill → `end_trace_capture` → save
     `prefill_trace_id`
   - `begin_trace_capture` → run decode → `end_trace_capture` → save
     `decode_trace_id`

DO NOT JIT-compile anything BETWEEN the two capture calls; otherwise
the prefill trace memory gets stomped by decode JIT (this is the
99% CPU hang we hit on 27B initially).

**Gate**: 8-step traced chain == v0.3.3 6/7 baseline.

### v0.4.1.c — trace_region_size + long-context smoke

- `trace_region_size = 400 MB` (already in dev harness). Verify
  prefill + decode together fit. If not, bump to 800 MB.
- Long-context: 256-step, 1024-step decode traced. Watch for trace
  memory exhaustion + position-buffer wraparound.

### v0.4.1.d — perf measurement + Tracy A/B

- tt-perf-report on traced step. Identify next bottleneck.
- Compare ms/tok eager vs traced. Expected 2-3× decode speedup.

## The Mamba2 fp32 ssm_state risk (LEARNED FROM 35B)

Mamba2 requires fp32 ssm_state for numerical stability (~10x smaller
clamp epsilon error than bf16). Our owned kernel ALREADY runs cb_outer
in fp32 ([[feedback-mm-init-prime-required]] 4-ingredient recipe).

**Risk**: 35B's dn_state in fp32 caused a TRISC hang under trace
(memory `[[35b-dn-h-state-drift-lever]]`). The fp32 path was the only
correct one but it wouldn't trace cleanly. The workaround was
bf16-state + a drift penalty (3% at 64 tok was acceptable).

**Mitigation plan**:
1. Trace with fp32 ssm_state first (the correct config).
2. If trace fails with TRISC hang at iter ~4 (the canonical signature),
   capture the lkkrace and check whether it matches 35B's signature.
3. If yes, fall back to bf16 ssm_state, measure drift at 64/256/1024
   tokens, ship if <5% at 256 tokens.
4. Track this risk explicitly — the v0.4.1 task description gets a
   "RISK: fp32 ssm trace may TRISC-hang like 35B" line.

## Estimated calendar

- v0.4.0g (SSD wrapper refactor): 0.5-1 day
- v0.4.0h (MoE host paths): 0.25 day
- v0.4.1.a (single-step trace): 0.5 day
- v0.4.1.b (multi-trace): 0.5 day
- v0.4.1.c (long-context): 0.5 day
- v0.4.1.d (perf measure): 0.25 day

**Total**: 2.5-3 days to v0.4.1 traced+correctness gated. Then v0.4.2
(prefill+decode both traced for cb_engine path) and v0.5 (the demo-
ready perf pass with vocab-shard, RMSNorm fusion, etc.).

## Non-blocking parallel work

- Gemma 4 perf rounds continue on qb2 (background agent).
- Tracing prereq work (v0.4.0g) is INDEPENDENT of v0.4.0f profile —
  both can be done; the profile just shows whether 0.7s is dominated
  by Mamba2 SSD numpy bridges (likely yes → v0.4.0g is the biggest
  win) or something else.

## Open questions

1. Does the owned kernel wrapper already have a ttnn-pure entrypoint?
   Need to check `experiments/serve/nemotron3_mamba2_step.py`.
2. How does the kernel handle state.ssm_state_tt sharding across the
   (1,4) mesh? The 35B `qwen36_gdn_decode_owned` has a specific
   shard pattern; ours likely mirrors it.
3. The cur_pos_buf protocol is already in place (`attn_decode_step_tt`
   reads it). The matmul-fold uses host `S=1` literal — does that
   trace cleanly, or do we need a device-side S buffer too? (Probably
   fine since S is a compile-time constant for the decode trace.)
