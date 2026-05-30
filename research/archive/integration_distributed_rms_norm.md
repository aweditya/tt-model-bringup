# Integration outline: distributed RMSNorm (candidate #1) + residual fusion

Status: code-ready outline. No device runs. Cite paths assume project root.

## Summary

- Pre-state: `_rms_norm_manual` (`experiments/serve/server_tp.py:361-376`) calls `ttnn.rms_norm` on a **replicated** activation. 305 calls/token × 4 chips do **redundant full-vector** reductions.
- Post-state (Step 1): swap to `ttnn.rms_norm_pre_all_gather` + tiny stats `all_gather` + `ttnn.rms_norm_post_all_gather` on a **fractured** activation. Each chip reduces only HIDDEN/4. The stats payload is exactly 1 tile wide (`ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp:6-9`) so the AG cost is dominated by per-link latency, not bandwidth.
- Post-state (Step 2 — residual fusion): unclear win. See "Residual fusion: O2 was half-right" below — the API exists (`residual_input_tensor=`) but its semantics in the **plain** pre/post pair do NOT carry `(x+r)` through to post-AG output, so we cannot use it as a drop-in residual-add fuser without also taking on a `fused_rms_minimal` op. That op is in `experimental/transformer/fused_distributed_rmsnorm` which Agent N's audit flagged as **NOT in our build** (`feedback_ttnn_fused_ops_gap_analysis.md` line 72-74).
- Estimated win: **15-20 ms/tok base** (the 18 ms Agent N projected assumes pure dispatch savings; bandwidth savings from 4× less compute likely small at our shapes). **+0 ms/tok from residual fusion until we either land `fused_rms_minimal` or get explicit confirmation that the pre-AG-with-residual semantics meet our use case.**
- Risk: **medium → high**. The Step 1 path is NOT a drop-in. It requires switching the residual stream from replicated to fractured, which means every `ttnn.all_reduce` exit becomes `ttnn.reduce_scatter` and the residual add becomes a per-chip fractured add. Galaxy's `llama_decoder.py:142` confirms the invariant: "Norms take fractured inputs and output replicated across devices." Going halfway (replicated input → distributed norm) buys nothing because each chip still reduces the full vector.

## Distribution context (critical correction)

`server_tp.py:632` uploads `x_buf` with `ttnn.ReplicateTensorToMesh(mesh)` — the residual stream is REPLICATED. Each `all_reduce` exit at lines 462, 499, 607 also returns REPLICATED. Today the norm sees a replicated input and `ttnn.rms_norm` on a replicated tensor is correct but 4× redundant — exactly what the audit (`feedback_ttnn_fused_ops_gap_analysis.md`) and `reference_multi_chip_opt_menu.md` note.

To benefit from `rms_norm_pre_all_gather`, the input to the norm must be **width-fractured** (each chip holds HIDDEN/4). This means we cannot just swap `_rms_norm_manual`; we have to change the residual stream topology:

| Today | Distributed |
|---|---|
| `partial = ttnn.linear(h, w_down)` (partial sums, dim=1 sharded) | same |
| `reduced = ttnn.all_reduce(partial)` → REPLICATED | `reduced = ttnn.reduce_scatter(partial, dim=1)` → FRACTURED |
| `x_out = ttnn.add(x_tt, reduced)` (replicated + replicated) | `x_out = ttnn.add(x_tt, reduced)` (fractured + fractured) — requires x_tt also fractured |
| `h_tt = _rms_norm_manual(x_tt, ...)` (replicated norm, 4× redundant) | `stats = ttnn.rms_norm_pre_all_gather(x_tt)`; `stats_g = ttnn.all_gather(stats, dim=-1)`; `out = ttnn.rms_norm_post_all_gather(x_tt, stats_g, ...)` |

So the residual stream needs to be **width-sharded** end-to-end. This means at bootstrap:
- `state.x_buf` (line 632) must change `ReplicateTensorToMesh` → `ShardTensorToMesh(mesh, dim=-1)` and the host fill in `update_input_buffers` (`server_tp.py:629-633`) must shard the embedding row.
- Every norm-weight upload (`input_norm_tt` line 185, `post_norm_tt` line 184, `final_norm_tt` line 276) needs to be sharded along dim=-1 too, since post-AG applies gamma per-tile and gamma must match the input shape — see `rmsnorm_distributed_nanobind.cpp:155` ("the last padded dim of stats must be a multiple of TILE_WIDTH"). Note Galaxy uses `self.norm.weight_distributed` (`distributed_norm.py:67,80`).
- `update_input_buffers`'s `cur_pos`/`cos`/`sin` stay replicated (they aren't on the residual stream).
- Final lm_head: today `state.lm_head_tt` is uploaded for `x` REPLICATED × `lm_head` REPLICATED. With fractured x, lm_head becomes a `linear` where the input is sharded along reduction-dim → needs `all_reduce` (or use distributed lm_head per candidate #6).

That last point ties #1 and #6 together: **shipping #1 without #6 means inserting an `all_gather(x, dim=-1)` immediately before the final lm_head**, which costs back some of the savings. Cleanest order is #6 first (or both together).

## Where to apply / where NOT to apply

Five distinct norm call types in `server_tp.py`. Decision per type:

| Type | Sites | Shape | Apply distributed RMSNorm? |
|---|---|---|---|
| Pre-norm (DN) | line 394 | `[1, HIDDEN=5120]` | **YES.** 48 calls/tok. Biggest single win. |
| Pre-norm (gated attn) | line 529 | `[1, HIDDEN=5120]` | **YES.** 16 calls/tok. |
| Pre-norm (MLP, both layer types) | line 489 | `[1, HIDDEN=5120]` | **YES.** 64 calls/tok. |
| Final norm | line 672 | `[1, HIDDEN=5120]` | **YES** (cheap, one call, but only after the residual stream is already fractured — otherwise it's a no-op). |
| q_norm/k_norm (gated attn) | lines 542, 543 | `[NQ_PER_CHIP=6, HEAD_DIM=128]` / `[NKV_PER_CHIP=1, HEAD_DIM=128]` | **NO.** Already per-head on per-chip data; HEAD_DIM=128 = 4 tiles per row only — the AG fixed cost (2 ops + sync) likely exceeds the 4× redundancy savings. Keep `_rms_norm_manual` as-is. |
| QK l2 scale (DN) | lines 433, 434 | `[NV_PER_CHIP, K_DIM=128]` | **NO.** Same reasoning as q_norm/k_norm. These already operate on per-chip head subsets. |
| linear_attn_norm (DN, per-head) | line 454 | `[NV_PER_CHIP, V_DIM=128]` | **NO.** Same. |

Net change: **128 of 305 calls/tok converted** (DN pre-norm 48 + DN MLP-pre-norm 48 + attn-pre-norm 16 + attn-MLP-pre-norm 16 + final 1 = 129). The remaining 177 stay manual.

Recovered dispatches (if 128 calls × (7-op → 3-op = 4 ops saved/call): 512 dispatches/tok ≈ 7.7 ms at 15 µs/dispatch. **Plus** the 4× redundant compute removed per converted call — at HIDDEN=5120, the per-chip sum-of-squares drops from 5120 elements to 1280; not the bottleneck at decode batch=1 but real on Tracy time.

Agent N's 18.3 ms/tok projection assumed all 305 norms convert. Realistic Step 1 win: **8-12 ms/tok**. Combined with #6 (vocab-sharded lm_head) saving its own 4-10 ms, the joint refactor is more attractive.

## Recipe quote from Galaxy

`experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390` (plain decode prefill / non-sharded):

```python
def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config, tt_ccl=None):
    use_2d_grid = False
    tt_stats = ttnn.rms_norm_pre_all_gather(
        inp, compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16, use_2d_core_grid=use_2d_grid)
    tt_stats_gathered = tt_ccl.line_all_gather(
        tt_stats, dim=3, cluster_axis=1, num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="LAYERNORM")
    tt_stats.deallocate(True)
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, epsilon=epsilon, weight=gamma,
        compute_kernel_config=compute_kernel_config, use_2d_core_grid=use_2d_grid)
    return tt_out, None
```

Usage site: `llama_decoder.py:148` (`attn_in_sharded, _ = self.attention_norm(x, None, mode)`).

`llama_decoder.py:142` invariant comment: *"Norms take fractured inputs and output replicated across devices. attn_in_sharded=norm(x+h), h = x+h happens implicitly"*. The "implicitly" here refers to the **sharded** decode path (`tt_sharded_distributed_rmsnorm`, `llama_ccl.py:1413`), NOT to the plain pre/post pair quoted above. The plain pair does NOT carry residual.

## Residual fusion: O2 was half-right

`reference_multi_chip_opt_menu_v2.md:19-20` claims `residual_input_tensor=` on `rms_norm_pre_all_gather` "fuses the residual add INTO the norm kernel." The kwarg exists (`ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp:18`, and the nanobind binding `rmsnorm_distributed_nanobind.cpp:82`). The pre-AG compute kernel **does** support FUSE_PRE_ADD (`rmsnorm_pre_allgather.cpp:47-55`) which makes the kernel set `cb_inp = cb_in0 + cb_res` and then compute stats over `(x+r)`.

**But**: the pre-AG op outputs ONLY the stats tensor — see `layernorm_pre_all_gather_device_operation.hpp:46` (`tensor_return_value_t = Tensor`, singular) and `rmsnorm_pre_allgather.cpp:82` (the fused-input CB is `cb_pop_front`'d after the reduction, never written out). The `(x+r)` is discarded.

Meanwhile, `rms_norm_post_all_gather` (`rmsnorm_post_all_gather.cpp:34`) accepts **no residual** and re-reads the original `input_tensor`. Result: if you pass `residual_input_tensor=r` to pre-AG and `input_tensor=x` to post-AG, you get `norm(x) / RMS(x+r)` — wrong semantics for both `norm(x+r)` (the LLM block residual we want) and `norm(x) + r` (untouched residual).

To use `residual_input_tensor=` correctly, you'd need to either:
1. Compute `x + r` yourself with `ttnn.add` and pass it as `input` to BOTH pre-AG and post-AG. This adds back the explicit `ttnn.add` we were trying to fuse out → no net win.
2. Use the experimental `ttnn.fused_rms_minimal` op (Galaxy `llama_ccl.py:1413`), which is end-to-end norm+AG+residual in one op. **This op is in `experimental/transformer/fused_distributed_rmsnorm`** which Agent N's audit (`feedback_ttnn_fused_ops_gap_analysis.md` line 73) lists as **MISSING in our build** (newer naming).

**Recommendation:** Skip residual fusion in Step 1. Verify `fused_rms_minimal` actually exists in our ttnn after a build update before committing to Step 2. Until then, the `ttnn.add(x_tt, reduced)` at `server_tp.py:468, 504, 611` stay as separate ops (acceptable cost — 64 adds/tok × ~0.06 ms = ~4 ms/tok). The remaining +0 win means Step 2 is gated on a build-tool change, not just code.

## Code changes (Step 1 only)

### Change 1: bootstrap residual stream to width-sharded

`server_tp.py:629-633`:

```python
# BEFORE
x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
x_host = ttnn.from_torch(torch.from_numpy(x_np),
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

# AFTER
x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
x_host = ttnn.from_torch(torch.from_numpy(x_np),
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1))
```

Symmetric change required to `state.x_buf` allocator (allocator currently uses ReplicateTensorToMesh; trace was captured against that buffer spec). After this, `state.x_buf` is `[1, HIDDEN/4]` per chip.

### Change 2: bootstrap norm weights to width-sharded

`server_tp.py:184, 185, 204, 245, 246, 276` (`upload_replicated` for *_norm tensors) → switch to `upload_sharded(..., dim=-1)` for the four norm types that get distributed (input_norm, post_norm, final_norm). Keep q_norm/k_norm/linear_attn_norm/q_l2_scale/k_l2_scale REPLICATED.

### Change 3: helper rewrite

Replace `_rms_norm_manual` at `server_tp.py:361-376` with:

```python
def _rms_norm_distributed(x_tt, weight_tt, eps):
    """Width-distributed RMS norm: each chip holds HIDDEN/4 of x and weight.
    Recipe from llama_ccl.py:1358-1390. Stats AG is ~1 tile, cheap.
    """
    import ttnn
    stats = ttnn.rms_norm_pre_all_gather(x_tt, dtype=ttnn.bfloat16)
    stats_g = ttnn.all_gather(stats, dim=-1)  # tiny payload
    ttnn.deallocate(stats)
    out = ttnn.rms_norm_post_all_gather(
        x_tt, stats_g, epsilon=eps, weight=weight_tt)
    ttnn.deallocate(stats_g)
    return out
```

Keep `_rms_norm_manual` callable (for debug fallback + the 5 norm sites we are NOT converting). Do not remove.

### Change 4: convert the 5 norm sites

`server_tp.py:394` (DN pre-norm):
```python
h_tt = _rms_norm_distributed(x_tt, dn['input_norm'], EPS)
```

`server_tp.py:489` (MLP pre-norm):
```python
h_tt = _rms_norm_distributed(x_tt, mlp['post_norm'], EPS)
```

`server_tp.py:529` (attn pre-norm):
```python
h_tt = _rms_norm_distributed(x_tt, attn['input_norm'], EPS)
```

`server_tp.py:672` (final norm):
```python
x_tt = _rms_norm_distributed(x_tt, state.final_norm_tt, 1e-6)
```

### Change 5: switch all_reduce → reduce_scatter at every residual exit

`server_tp.py:461-466` (DN out_proj exit), `499-503` (MLP exit), `606-610` (attn exit):

```python
# BEFORE (one of three identical patterns)
try:
    reduced = ttnn.all_reduce(partial)
except Exception:
    scattered = ttnn.reduce_scatter(partial, dim=1)
    reduced = ttnn.all_gather(scattered, dim=1)

# AFTER
reduced = ttnn.reduce_scatter(partial, dim=1)  # per-chip [1, HIDDEN/4]
```

This is what makes the next norm's input fractured. The `ttnn.add(x_tt, reduced)` at line 468 is now fractured+fractured (both per-chip `[1, HIDDEN/4]`).

### Change 6: insert all_gather before lm_head (until #6 ships)

`server_tp.py:672-673`:

```python
x_tt = _rms_norm_distributed(x_tt, state.final_norm_tt, 1e-6)
x_tt = ttnn.all_gather(x_tt, dim=-1)  # gather to full HIDDEN for replicated lm_head
logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
```

This single AG at the end costs back ~0.5-1 ms/tok. It goes away once candidate #6 (vocab-sharded lm_head) lands.

## What probably does NOT change

- `_rms_norm_manual` stays as the implementation for q_norm, k_norm, linear_attn_norm, and the QK l2_scale norms (lines 433, 434, 454, 542, 543). Their shapes (`[..., HEAD_DIM=128]` or `[..., K_DIM=128]`) are too narrow to amortize the AG fixed cost.
- All `ttnn.add` residual adds (`server_tp.py:468, 504, 611`) — stay as-is. Step 2 (fusion) gated on `fused_rms_minimal` shipping.
- Trace-captured paths: `forward_token_tp_inner` (`server_tp.py:651-674`) is captured into a trace by `_ensure_decode_trace`. The trace recapture must include the new ops AND see the new buffer shapes. Re-warmup with `reset_state` → execute a synthetic forward → re-capture is already the protocol; no new machinery needed.

## Validation plan

1. **Probe first.** Build `experiments/utils/p21_distributed_rms_norm_probe.py` modeled on `feedback_p7_mlp_wedges_next_dn.md` style:
   - Open (1,4) mesh + fabric (set_fabric_config(FABRIC_1D) before open_mesh_device — see `feedback_c71_mesh_smoke_pass.md`).
   - Upload synthetic `[1, HIDDEN]` activation, ONCE replicated and ONCE width-sharded.
   - Compute `_rms_norm_manual` on replicated, `_rms_norm_distributed` on fractured.
   - Use `ttnn.all_gather(out, dim=-1)` on the fractured result, compare per-chip cosine to a numpy oracle.
   - **Gate:** cos ≥ 0.999 on all 4 chips; max chip-to-chip drift ≤ 1e-6.
   - Repeat with the **q_norm shape** (`[6, 128]`) to confirm our "do not convert" decision: measure ms and verify it's slower than manual.
2. **Step 1 wiring (on a feature branch).**
   - Change residual stream to width-sharded.
   - Re-bootstrap. Verify weights uploaded ok.
   - Run `run_91r` (per-layer cosine sanity) — gate cos ≥ 0.997 per layer.
   - Run `bench_decode_tp` — gate ≥ 7.02 tok/s baseline (no regression) plus aim for 7.5+ tok/s.
   - Run `cosine_ladder` — gate matches the bf16-vs-fp32 ladder we have today.
3. **Mesh recovery:** if the probe wedges (mesh hang), `ssh qb2 'tt-smi -r 0,1,2,3'` (`feedback_mesh_recovery_after_kill.md`).

## Risk register

1. **Residual stream sharding cascades through 7+ unrelated sites.** Embedding host write, paged KV update sharded write (the `_shard_for_paged_write` helper at `server_tp.py:563-569` assumes a particular per-chip residual layout), the lm_head call site, the trace recapture, the reset_state buffers (`server_tp.py:702+`). High probability of a "everything compiles but the third token is garbage" debug spiral.
2. **AG-on-stats payload may NOT be cheaper than redundant compute at small batch.** Tracy data not gathered yet. Agent N's 18 ms/tok used a 15 µs/dispatch model; if dispatch is already pipelined (`feedback_pipelining_already_wins.md`) actual savings could be < 5 ms/tok.
3. **`rms_norm_post_all_gather` requires weight (gamma) in TILE layout matching the sharded input** (`rmsnorm_distributed_nanobind.cpp:155-157`). Need to verify gamma uploaded as TILE_LAYOUT (it is in our `upload_replicated`) AND ShardTensorToMesh dim=-1 doesn't produce a sub-tile-padded shard. At HIDDEN=5120 → 5120/4=1280 → 1280/32=40 tiles per chip → OK.
4. **`ttnn.all_reduce` is currently in a try/except fallback to reduce_scatter+all_gather.** If the try-branch was succeeding silently and we swap to a hard reduce_scatter, we lose the safety net. Add a `try`/`raise` with a clearer error.
5. **No bias support gotcha.** The nanobind docs (`rmsnorm_distributed_nanobind.cpp:156`) say "If weight is provided, bias must also be provided. Gamma and beta must have the same layout." For RMS norm we have no bias. The actual cpp implementation accepts `bias=None` (`rmsnorm_post_all_gather.cpp:49` passes std::nullopt through), so the docstring is over-restrictive vs the binding. Verify in probe.
6. **`use_2d_core_grid`** is set to False in Galaxy's plain decode recipe (`llama_ccl.py:1366`). Mirror that; the 2D path is gated on different program configs.
7. **Trace recapture must succeed** with new tensor shapes (`x_buf` now `[1, 1280]` per chip not `[1, 5120]`). Trace was historically fragile to buffer-spec changes (`feedback_c4_trace_cache_threading.md`).
8. **Numpy oracle drift across 64 layers.** With 4× fewer reduction lanes per chip but a stats-AG roundtrip, the bf16 numerics of `RMS(x[0:1280]) + RMS(x[1280:2560]) + ...` after AG may diverge from `RMS(x[0:5120])` due to different summation order. Probe must explicitly measure this and accept up to ~1e-3 per-layer cosine if the cumulative end-of-model cosine stays ≥ 0.997.

## Effort estimate

- Helper rewrite + Step 1 site replacements (5 norm sites + 3 all_reduce → reduce_scatter + 1 final AG): ~80 LOC
- Bootstrap changes (residual stream sharding + 4 norm-weight uploads): ~20 LOC
- Trace recapture wiring fix-ups: ~20 LOC
- Probe `p21_distributed_rms_norm_probe.py`: ~150 LOC (mesh open + 2 shapes + numpy oracle + per-chip cosines + perf timing)
- Total: **270 LOC** code + **~5-7 hours** implementation + **2-3 hours** validation (probe → run_91r → bench_decode_tp → cosine_ladder)

Step 2 (residual fusion) deferred until `fused_rms_minimal` shipping confirmed in our build (separate ~2h ttnn rebuild + 1h re-audit). Realistic ROI for Step 2 once unlocked: 3-5 ms/tok additional savings.

## Reference paths

- Production recipe (plain): `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390`
- Production recipe (sharded/residual-fused): `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1393-1429` — **uses `ttnn.fused_rms_minimal` not pre/post-AG**
- Galaxy DistributedNorm wrapper: `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/distributed_norm.py:60-84`
- Galaxy usage sites: `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_decoder.py:148, 155, 157, 175, 179, 181`
- Op signatures: `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_pre_all_gather.hpp:15-22`, `rmsnorm_post_all_gather.hpp:15-25`
- Nanobind docstring + constraints: `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/rmsnorm_distributed_nanobind.cpp:23-86, 90-170`
- Pre-AG compute kernel (residual semantics): `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp:38-82`
- Existing audit: `feedback_ttnn_fused_ops_gap_analysis.md` line 24 (18.3 ms/tok projection), line 47-58 (suggested integration order), line 72-74 (`fused_distributed_rmsnorm` MISSING in our build)
- O2 addendum: `reference_multi_chip_opt_menu_v2.md:19-20` (residual_input_tensor= claim — partial; see "O2 was half-right" above)
- Replicated residual stream confirm: `experiments/serve/server_tp.py:632` + `184, 185, 204, 245, 246, 276`
- All-reduce-returns-replicated invariant: `experiments/serve/server_tp.py:461-466, 499-503, 606-610`
- Galaxy "Norms take fractured inputs" invariant: `experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/llama_decoder.py:142`
- Mesh fabric prereq: `feedback_c71_mesh_smoke_pass.md`
- Mesh recovery: `feedback_mesh_recovery_after_kill.md`
- Trace fragility precedent: `feedback_c4_trace_cache_threading.md`
- Dispatch-vs-pipelining cost model: `feedback_pipelining_already_wins.md`
