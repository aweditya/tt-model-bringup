# Kernel work — implementation readiness synthesis

After ~2,244 lines of research across docs 01-11, this is the executive
summary: what we now know with evidence, what to build first, what to defer,
and the risk register.

## What we know (with evidence)

### Architecture (docs 01, 08)

- **Blackhole P150**: 1.5 MB L1 per Tensix core (same as Wormhole — no enlargement), 8 GDDR6 DRAM controllers × 4 GB = 32 GB chip-wide. **No L2.**
- **Per core**: 5 RISC-V CPUs — 2 dataflow (BRISC + NCRISC) + 3 compute (TRISC0/1/2) — running DIFFERENT programs concurrently. Critical mental shift from CUDA's SIMT.
- **Compute kernel is ONE source compiled three times** into unpack/math/pack binaries; you don't hand-write 3 source files. Dst is a GPR-like register file passed via `tile_regs_*` (Marty calls this "abuse Dst as scratch").
- **Tile = atomic compute unit** (32×32 elements, four 16×16 faces). Lane→tile mapping is load-bearing for non-element-wise ops (RoPE's gotcha — Marty's M-1 technique).

### Memory + dataflow (docs 05, 09)

- **MemoryConfig matters**: `ttnn.DRAM_MEMORY_CONFIG` (INTERLEAVED) is our default. `HEIGHT_SHARDED` / `WIDTH_SHARDED` / `BLOCK_SHARDED` exist for specific kernel kinds.
- **#16674 (Blackhole writer hang)** correlates with **sharded paged_update_cache input**. Workaround: use INTERLEAVED. Root cause not in public sources.
- **Circular buffers are the only sync** between reader/compute/writer. No atomics, no fences, just `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`.
- **Reader/writer monopolize NoC; compute never calls `noc_async_*`** — strict role separation.
- **Granularity ≥ 2 in CBs** is the universal pipeline trick (double-buffering).

### How to ship a custom op (doc 02)

- 5 required files per op: `op.{hpp,cpp}` + `op_nanobind.{hpp,cpp}` + `device/*_device_operation.{hpp,cpp}` + ≥1 `program_factory.cpp` + kernels under `device/kernels/`.
- Python namespace controlled by `ttnn::bind_function<"name", "ttnn.category.">`.
- **Experimental vs production is JUST a directory + namespace + bind-prefix convention** — no stability machinery.
- No sideload story documented; in-tree contribution is the realistic flow. We'd either fork tt-metal or build inside our own tree against ttnn headers.

### Trace capture rules (doc 06)

- **The blocker we hit** (JIT binary upload during capture) is structural: `EnqueueMeshWorkload` unconditionally calls `compile → load_binaries → generate_dispatch` every enqueue. `load_binaries` skips work only after `program_binary_status_` records the mesh device, which only happens on the **first non-trace enqueue**.
- **Solution**: warmup pass before `begin_trace_capture` (now in our server). The `tt_dit.Tracer` class is the canonical reference pattern — uses `prep_run=True + clone_prep_inputs=True`.
- **What IS allowed inside trace**: device-to-device copies (`ttnn.copy(src_dev, dst_dev)`), new allocations (overlap-checked at end-capture), trace_id is per-CQ.
- **What is NOT allowed**: any host-to-device write (`ttnn.from_torch(device=...)`, `copy_host_to_device_tensor`), event sync.

### SDPA decode (doc 07)

- **L1 budget formula**: `Sk_chunk_t = min(512, largest_pow2_divisor(s)) / 32`. Total L1 ≈ K_cb + V_cb + qk_im + stats + output + overhead.
- For our shape (`N_KV=4, HD=256, bf16`): MAX_POS=256 fits at ~1.0 MB; MAX_POS≥1024 needs ~1.9 MB and overflows 1.5 MB.
- **Paged variant** defaults to `Sk_chunk_t = dst_size = 8` regardless of logical seq length. Page table indirection: logical S scales via page-table columns. **L1 per core is constant** — no overflow.
- Paged and non-paged share the **identical compute binary**; paging is ~40 lines in the reader.

### update_cache pattern (docs 04, 09)

- `ttnn.kv_cache.update_cache_for_token_(cache, input, update_index)` does in-place writes.
- Cache layout: `[1, B, P, D]` (heads in dim 1, positions in dim 2 — matches our `[1, N_KV=4, MAX_POS, HEAD_DIM]`).
- The **writer kernel** does the actual data move (single-row in-L1 scatter then re-tilize); compute is a glorified tilize/untilize driver.
- **Implication for any custom in-place op we write**: we don't need a real compute kernel — just a reader (no-op) and a writer that does the row update.

### GitHub design + tech reports (doc 10)

- **NoC flush ordering on Blackhole** is the canonical kernel-writer gotcha — explicit flushes needed because data + semaphore signals get reordered. (BlackholeBringUpProgrammingGuide.md)
- `aligned_size_per_bank()` ≠ `buffer->size()` for padded sharded tensors — possible bug in our `phase_b1_pass` weight skeleton's memory accounting.
- **Issue #14540** is the contribution PR template — pre-impl review plan with shapes, CB-to-core mapping, producer/consumer integration, batch limits BEFORE any C++.
- 214 PRs match `sdpa decode` — a high-value periodic skim.

### External wisdom (docs 08, 11)

- **Marty (clehaxze.tw)**:
  - "Shape it like the hardware" (M-1): every non-trivial op must respect the 32×32 tile + 4-face layout.
  - "Dst is a GPR" (M-2): reuse the Dst register as scratch, not just an output sink.
  - "Reader does indirection" (M-3): all data-layout-aware logic (page tables, scatter indices, rotation pattern dispatch) lives in the reader, NOT the compute.
- **SFPSTORE lane skipping** (0→0, 1→2, 2→4...) is a correctness hazard not in official docs.
- **No fp32 subtract on SFPU** — synthesized as `SFPMAD` with VB=-1.0, doubling subtract cost.
- **MOP_CFG/REPLAY macro recorders** hide loop overhead — relevant if we hand-roll tight loops.
- **`exp_24f` over `exp_21f`** for position values > 256 (Marty's RoPE post) — accumulates ~18 absolute error at large args.

## What to build first (concrete plan)

### Order of operations, weighted by leverage and effort

| # | Action | Effort | Expected gain | Status |
|---|---|---|---|---|
| 1 | `update_cache_probe.py` on remote | 20 min | confirms ttnn.kv_cache.update_cache works for our shape, gives baseline perf | IN FLIGHT |
| 2 | Swap `ttnn.scatter` → `ttnn.kv_cache.update_cache_for_token_` in `gated_attn_step_ondevice` + `_traced` | 2-3 h | -10 ms/tok (5%); eliminates 1.6 ms/tok of in-trace `ttnn.copy` overhead | gated on #1 |
| 3 | Paged SDPA migration to unlock MAX_POS > 256 | 1-2 days | unlocks 32k context (the daily-driver requirement) | independent of #1, #2 |
| 4 | Async input-buffer updates + readback in bench_decode_traced | 4-6 h | recovers ~30-46 ms/tok of host overhead | independent |
| 5 | bf8 KV cache for long context | 2-3 h prototype + validate at MAX_POS=8k | halves cache bandwidth for long context | follows #3 |
| 6 | Custom in-place scatter kernel (if update_cache underperforms) | 1-2 weeks | best case 30-60× scatter speedup; effort is non-trivial | gated on #1 outcome |
| 7 | Multi-chip TP (C'7) | 2-3 weeks | 3-4× decode at 4 chips on qb2 | the scoreboard event |

### #2 specifics (the immediate win after probe)

In `experiments/91f_qwen36_27b_full_ondevice.py`:

**Eager `gated_attn_step_ondevice` (around line 467)** — currently:
```python
kv_cache_k_tt = ttnn.scatter(kv_cache_k_tt, dim=2, index=index_tt, src=k_for_cache)
kv_cache_v_tt = ttnn.scatter(kv_cache_v_tt, dim=2, index=index_tt, src=v_for_cache)
```

Should become:
```python
ttnn.kv_cache.update_cache_for_token_(kv_cache_k_tt, k_for_cache, cur_pos)
ttnn.kv_cache.update_cache_for_token_(kv_cache_v_tt, v_for_cache, cur_pos)
```

Note: `cur_pos` is a Python int here (passed alongside `cur_pos_tt` for SDPA). The `_for_token_` variant takes int directly.

**Traced `gated_attn_step_ondevice_traced` (around line 487)** — same swap. But:
- `cur_pos` would need to be a runtime arg, not Python int, for the trace to be position-independent
- The current `index_tt` buffer pattern wouldn't apply — `update_cache_for_token_` takes a scalar position
- The in-trace `ttnn.copy(scatter_out, cache_in)` lines become unnecessary (update_cache mutates in place natively)

**Server impact**:
- `_setup_traced_decode` no longer needs `index_buf` (delete that allocation + `index_tt` writes from `_update_input_buffers`)
- One less host-write op per step (faster `_update_input_buffers`)

**Correctness gate**: same 5-step validation we ran for C'4 v4 — cosine ≥ 0.9999, top-1 match all 5.

### #3 specifics (paged SDPA migration)

Per doc 07:
- Replace `ttnn.transformer.scaled_dot_product_attention_decode(q, k, v, cur_pos_tensor=...)`
- With `ttnn.transformer.paged_scaled_dot_product_attention_decode(q, k_paged, v_paged, page_table, cur_pos_tensor=...)`
- New tensors needed:
  - `k_paged, v_paged`: `[max_num_blocks, N_KV, block_size, HEAD_DIM]` where `block_size` is typically 32
  - `page_table`: `[batch=1, max_blocks_per_seq]` int32 — maps logical position chunks → physical blocks
- For our `MAX_POS=32k, block_size=32`: page_table is `[1, 1024]` int32
- The reader does the page-table indirection; compute is the same kernel as non-paged

This is a bigger surgery than #2 but unlocks the daily-driver use case.

## What to defer

- **Custom in-place scatter kernel** (we drafted the design in `research/c_scatter_kernel_design.md`). Defer **unless `update_cache_for_token_` underperforms** — which is unlikely per the kernel structure we just learned (writer-only data movement).
- **Multi-chip TP (C'7)**. Big lever but 2-3 weeks. Not the next-30-day project.
- **chunked prefill via Neumann series (C'5)**. Primitives green; full kernel is ~1 week of focused work; gated on having `update_cache` and paged SDPA landed first.
- **Custom kernels for SSM/conv state updates in DeltaNet**. Same pattern as scatter, similar complexity. Defer until we've shipped the simpler `update_cache` swap.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `update_cache_for_token_` shape requirements don't match our cache layout | LOW (Grok uses identical layout) | HIGH (need rework or custom kernel) | probe first |
| `update_cache_for_token_` hangs on Blackhole via #16674 with INTERLEAVED | LOW (per doc 05 analysis) | HIGH | probe at MAX_POS=8k; have custom-kernel fallback |
| Paged SDPA introduces correctness regression at our N_KV=4 | LOW (validated in memory at 32k) | HIGH | re-validate with multi-step cosine gate when integrating |
| Async I/O changes break the trace replay determinism | MED | MED | strict warmup + sync pattern from `tt_dit.Tracer` |
| Network outages slow remote testing | OBSERVED | LOW per-event, HIGH cumulative | run probes/benchmarks via persistent server so cold restart costs are paid once |

## What the team should read first

If a new contributor joins the project tomorrow with 1 hour:

1. **`08_tensix_vs_cuda_programming_model.md`** — the foundational mental shift (30 min)
2. **`12_implementation_readiness.md`** — this doc, for the concrete next actions (15 min)
3. **`04_update_cache_reference_op.md`** — the closest reference op for what we're doing (15 min)

If they have 3 more hours:
- `09_production_kernel_dataflow_survey.md` (the pattern catalog)
- `07_sdpa_decode_and_paged_variant.md` (the L1 budget formula + paged migration)
- Marty's "Programming Tenstorrent Processors" (in `tt_docs_corpus/blogs/clehaxze/`)

## Open empirical questions (probes to run)

| Question | Probe | Status |
|---|---|---|
| Does `update_cache_for_token_` work in-place at our shape with INTERLEAVED? | `experiments/utils/update_cache_probe.py` | IN FLIGHT |
| Does it work inside a trace? | same probe, has trace test | IN FLIGHT |
| What's the actual perf delta vs ttnn.scatter? | same probe | IN FLIGHT |
| Does paged SDPA work at our N_KV=4 + MAX_POS=32k? | need `paged_sdpa_probe.py` | NOT WRITTEN |
| Does the page table need any specific construction? | doc 07 has the recipe; needs validation | NOT VALIDATED |
| Can we capture a paged-SDPA trace? | combine probe with trace test | NOT WRITTEN |

## Status

- Research phase: COMPLETE (11 docs, 17 blogs, 6 reference repos, all tech reports locally)
- Probe phase: 1/4 in flight, 3 not written
- Implementation phase: pending probe results

The team is **ready to start implementation work** as soon as the network stabilizes enough to run the first probe and validate the `update_cache` swap.
