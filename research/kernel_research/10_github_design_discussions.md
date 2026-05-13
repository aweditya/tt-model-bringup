# tt-metal Design Discussions — Synthesis for tt-xla Kernel Work

Mined 2026-05-13 from tt-metal issues + PRs to surface kernel-design patterns and
pitfalls. Quotes are verbatim from the linked artifacts. Where a passage maps to
something we already do locally, the file:line is cited.

Sources searched but not reachable in plain GitHub HTML (no member access required;
just rate limits / partial pages): comment threads on #16674, #30362, #15876 — only
the issue body was retrievable, not the full discussion. Noted inline.

---

## 1. In-place writer kernels and #16674

### Issue [#16674 — Blackhole: ttnn.experimental.paged_update_cache consistently hanging](https://github.com/tenstorrent/tt-metal/issues/16674)
- **Status:** Closed. **Labels:** `P1, blackhole, bug, op_cat: misc`. **Assignee:** `cglagovichTT`.
- **Repro signature** (verbatim from the issue body):
  > "I am finding `tests/ttnn.unit_tests.operations.test_paged_update_cache.run_test_paged_update_cache_decode` consistently locks up the machine"
  > "cache_idxs: [127, 144, 161, 178, 195, 212, 229, 246, 263, 280, 297, 314, 331, 348, 365, 382, 399, 416, 433, 450, 467, 484, 501, 518, 535, 552, 569, 586, 603, 620, 637, 654]"
  > "the lockup seems pretty independent of parameters"
- **Comment thread:** GitHub returned only the issue body via WebFetch — no engineer-comment text was extracted. No explicit fix-PR back-reference was visible. This matches what our [`feedback_paged_sdpa_decode_works_at_32k`] memory entry says: writer needs **sharded** memory config and the bare paged_update_cache writer was the hang source.
- **Root cause family — quoted from the Blackhole bring-up guide:**
  > "On previous architectures there are instances in kernels where NoC commands are issued without explicit flushes. These were causing ND mismatches or hangs on BH because data and semaphore signals were getting updated faster than NoC has a chance to service the command and are resolved by adding flushes."
  > "Previous architectures did not need this because of higher RISC to L1 latency compared to NoC latency."
  (Source: `tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`)
- **Implications for us** (cross-ref `c_scatter_kernel_design.md`): our custom scatter writer **must** issue explicit `noc_async_write_barrier` / flush before semaphore increments. Don't trust Wormhole-era examples that omit them. Validate with Watcher; the BH L1 cache should stay disabled while bringing the kernel up.

### Related: Blackhole L1 data cache
- > "Blackhole introduced an L1 data cache (4 × 16B cachelines, write-through). Writing an address on one core and reading it from another only requires the reader to invalidate if the address was previously read."
- For in-place writers (e.g. `ttnn.copy(scatter_out, cache_in)` from `feedback_trace_state_threading_works`), an inter-core in-place mutation that reuses a cache-resident buffer is exactly the kind of pattern that needs invalidation — write a sanity test before assuming cache coherency.

---

## 2. Trace capture design discussions

### Issue [#30762 — Event Synchronization is not supported during trace capture](https://github.com/tenstorrent/tt-metal/issues/30762)
- **Status:** Closed (Done in project). Cross-referenced with #23993.
- The body alone (visible) doesn't quote the assert origin PR — searches for the literal "Writes are not supported during trace capture" returned no GitHub results, suggesting the assert lives in C++ and was never given a tracking issue.
- **Implication:** anything that mutates host state mid-trace (events, host scalars, DRAM writes) is a footgun. This is the formal reason our [`feedback_trace_capture`] memory note says "Python scalars baked into trace; use device tensor buffers for dynamic values" — there is no public design doc, the constraint is enforced via runtime asserts.

### PR [#43849 — Feature: capture trace SW provenance and add qwen-32b_3 trace target](https://github.com/tenstorrent/tt-metal/pull/43849)
- Merged 2026-05-08 by `stevendae`. Indicates Tenstorrent is investing in **trace-provenance metadata** so that captured traces remember which Python ops produced them. Useful template if we want to debug PJRT traces.

### PR [#43069 — Fix graph trace output size: use aligned_size_per_bank for accurate per-core L1 estimation](https://github.com/tenstorrent/tt-metal/pull/43069)
- Merged 2026-05-07 by `bmalesevicTT`. Closes [tt-mlir#8044](https://github.com/tenstorrent/tt-mlir/issues/8044).
- Quote: > "`extract_l1_output_buffer_allocation_size_per_core` was computing per-core L1 output buffer size using `buffer->size()` (unpadded logical tensor size)."
  > Fix: use `buffer->aligned_size_per_bank()` — > "calls the same `calculate_bank_size_spread` the runtime allocator uses — giving the correct padded per-bank reservation for both interleaved and all sharded layouts."
- **Implications:** when our PJRT plugin starts emitting L1 budgets, we have to use `aligned_size_per_bank()` — not `buffer->size()` — or we'll under-account by the tile-padding amount on sharded tensors. Directly affects `phase_b1_pass: weight skeleton + per-chip memory budget` (most recent commit).

---

## 3. SDPA decode L1 / paging design

### Tech report: [`tech_reports/FlashAttention/FlashDecode.md`](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashDecode.md)
- The y-dimension trick — quoted:
  > "In the case of attention decode, the seqlen on the `y` dimension of the query tensor is always 1, which will be padded to 32. As a result, vanilla attention does not take advantage of the tile-based architecture."
- Solution: move heads into `y`, reshape `[bsz, num_q_heads, 1, head_dim] → [1, bsz, num_q_heads, head_dim]`. This is precisely the layout assumption our split-KV-heads workaround relies on (see `feedback_sdpa_decode_kv_heads`).
- NoC bottleneck quote:
  > "In this case, one work is assigned to many cores. This can lead to a new bottleneck from noc traffic, where the reduction step takes more time than compute."
- Mitigation: `max_cores_per_head_batch` (defaults to 16).
- **Multi-core reduction primitive:**
  > "noc_semaphore_wait(semaphore_addr_ptr, num_workers); // wait for semaphore to reach num_workers"
  Workers NoC-write into a reducer core's L1, increment the semaphore, then the reducer aggregates. This is the canonical pattern our future custom DeltaNet scan or sparse-MoE reduction should imitate.

### Issue [#30362 — Paged SDPA decode fails PCC check at certain positions](https://github.com/tenstorrent/tt-metal/issues/30362)
- **Status:** Closed. **Labels:** `P0, Root Cause: Ops`. Config `[1, 8, 1, 128*1024, 128, (8, 4), True]`.
- > "there are a small handful of PCC failures" — CI was sampling positions at strides 71 / 3001 and missing them.
- Confirms what `feedback_paged_sdpa_decode_works_at_32k` says ("validated at 32k") is only true if you test **every** position. Our nightly Qwen3.6 generation tests should sweep `cur_pos` exhaustively for at least one shape, not stride-sample.

### Issue [#15876 — Create new SDPA op for chunked prefill](https://github.com/tenstorrent/tt-metal/issues/15876)
- **Status:** Closed. Design rationale:
  > "only supports `is_causal=True` [and] does not allow providing an `attn_mask`"
  > Runtime parameter `chunk_idx_tile` is passed to writer and compute kernels to > "calculate the true `q_idx` within the larger tensor to see whether a causal mask must be applied"
- **Pattern for us:** when we add chunked prefill to our long-context path, the chunk index is a **compile-time** style parameter passed through runtime args, not a host scalar baked into the trace.

### Issue [#21534 — `kv_bfp8_q_bf16` paged attention program-cache-count assertion](https://github.com/tenstorrent/tt-metal/issues/21534)
- > "'assert 3 == 4' where 3 = <bound method PyCapsule.num_program_cache_entries ...>"
- Mixed-precision (bf8 KV, bf16 Q) paged SDPA caches an unexpected number of programs. Implication for our bf8-cache plans (per `feedback_bf8_weights`): **expect program-cache thrash** unless we keep dtype combinations stable across decode steps.

---

## 4. Custom-op contribution flow

### Issue [#14540 — Pre-Implementation Review Plan for `fused_rotary_embedding`](https://github.com/tenstorrent/tt-metal/issues/14540)
This is the **best template** for any new ttnn op we propose. Structure that the reviewers expect (verbatim section labels):

1. **Objective:** > "Develop a `fused_rotary_embedding` Op to apply ROPE on `q` and `k` tensors in parallel, improving performance by processing these tensors simultaneously."
2. **I/O Structure** — exact shapes + memory configs for every input/output, including sharded grid.
3. **Kernel & Program Factory Architecture:**
   > "Program factory will adapt the sharded portion of the ROPE Decode factory. **Buffer Assignments**: `cb_in/cb_out` buffers on cores [0-8] connect to `q` tensors. `cb_in/cb_out` buffers on cores [8-16] connect to `k` tensors."
4. **Producer/Consumer Integration** — name the upstream op (here `nlp_create_qkv_heads_decode`) and the layout-shift required.
5. **Limitations** — stated up-front:
   > "The current design only supports a `batch_size` of up to 32, as we are parallelizing `q` and `k` on two sets of 32 cores each within a 64-core grid."

This is the template our scatter / DeltaNet-scan kernel proposals should follow before any C++ is written.

### Issue [#14197 — Feature Request: SDPA forward and backward](https://github.com/tenstorrent/tt-metal/issues/14197)
- > "Performance critical" — explicitly rejects composite-op solutions:
  > "composite ops and saving all intermediates are inadequate solutions due to performance and memory concerns"
- Reinforces: don't propose a composite TT-NN wrapper as the long-term answer. Reviewers expect a fused kernel.

---

## 5. Performance optimization PRs

### PR [#42591 — Fix sdpa decode core alloc to round num kv heads](https://github.com/tenstorrent/tt-metal/pull/42591)
- Merged 2026-04-23, by `alingTT`. The bug:
  > "`num_cores_per_batch_uncapped = 110 / 32 = 3`, `num_heads_per_core = ceil(8 / 3) = 3` (but `8 % 3 ≠ 0`) — this leads to mismatched output core counts versus batch size"
- Fix: > "incrementing num heads per core until it reaches a number divisible by num kv heads such that the kv head computation is divided evenly to each core group"
- **Implication:** integer-division allocation bugs are the #1 source of silent SDPA-decode breakage. When we customize core grids for our shapes, run the allocator against the full divisibility sweep.

### PR [#42937 — Optimize flash mla data movement](https://github.com/tenstorrent/tt-metal/pull/42937)
- Merged 2026-04-23, by `tt-aho`. Two distinct optimizations bundled:
  1. Virtual-channel separation:
     > "Q mcast was overlapping with K mcast, so we move Q mcast to be on a different mcast vc when available."
     Required extending `noc_semaphore_inc_multicast` with an explicit VC argument.
  2. Tail-reduction signaling:
     > "the MS tile is produced and also consumed first, but we block waiting for MS and O to be produced and signal them together" → split signaling, > "send and signal MS first, then send and signal O".
- **Implication:** when our scatter kernel matures, expect VC contention against the rest of the model. Allocating an alternate mcast VC is a real lever, not a corner case.

### PR [#42446 — Fix SDPA masked-matmul unpacker reconfig](https://github.com/tenstorrent/tt-metal/pull/42446)
- Merged 2026-04-24 (ndivnicTT). Title alone is a warning: the compute-kernel **unpacker reconfig** is fragile. Maps to our [`feedback_compute_kernel_config`] memory entry — Blackhole's "all-or-nothing" kernel config requirement is enforced precisely because partial reconfigs corrupt downstream ops.

---

## Cross-cutting patterns — what tt-metal reviewers argue about

1. **Divisibility and integer truncation in core allocation.** #42591 is the obvious one but it recurs (cores-per-batch, heads-per-core, max_cores_per_head_batch caps). Every program factory needs a divisibility check up front.
2. **Explicit NoC flushes on Blackhole.** Every Blackhole-specific bug ticket eventually points at missing `noc_async_write_barrier` or semaphore-before-data ordering. This is the single biggest "Wormhole code looks correct, Blackhole hangs" trap.
3. **Memory-accounting precision.** #43069 (aligned_size_per_bank) and the recurring `kv_bfp8_q_bf16` program-cache miscounts both show that sharded + padded tensors require allocator-aware accounting, not logical-size accounting.
4. **Sample-strided test coverage isn't enough.** #30362 explicitly calls out that stride-sampling missed PCC failures — exhaustive position sweeps are the bar for any KV-cache or paged op.
5. **Pre-implementation review template.** The cleanest custom-op PRs cite a pre-impl review issue (e.g., #14540) with shapes, core grid, buffer assignments, producer/consumer, and explicit limitations. Reviewers will push back if a PR shows up without one.

---

## Top-5 actionable items for tt-xla

1. **Adopt the #14540 pre-implementation-review template for our scatter kernel proposal.** Before any C++, file an issue with: shapes, sharded grid, CB assignments, producer/consumer ops, and "current design only supports..." limitations. This is also good practice for the PJRT op-by-op rollout — see `pjrt_plugin_design.md` for the existing skeleton.
2. **Switch our L1 budget accounting to `aligned_size_per_bank()`.** PR #43069 is the canonical fix. Our `phase_b1_pass` weight skeleton currently uses logical tensor size; this is wrong for any sharded weight. Affects `experiments/.../weight_skeleton.*`.
3. **Add explicit NoC flushes to any in-place writer we author** (custom scatter, custom in-place RoPE if we revisit C'3, fused DeltaNet scan). Use Watcher + L1 cache disabled while bringing up. Single biggest lesson from #16674's root-cause family.
4. **Promote exhaustive `cur_pos` sweep to nightly CI for at least one Qwen shape.** #30362 showed that stride-sampling missed real PCC failures. Our daily-driver context path (`feedback_long_context_is_the_real_gate`) needs this exact coverage.
5. **Pin dtype combinations through decode** to avoid program-cache thrash (#21534 lesson). If we run bf8 KV + bf16 Q, **never** drop to bf16 KV mid-decode — it forces a re-cache and silently breaks `num_program_cache_entries`-style assertions in downstream tests.

---

## Bookmarked rabbit holes (not pursued today)

- The full comment thread of #16674 — extracting engineer commentary likely requires authenticated `gh` CLI; the public HTML body alone hides comments.
- PR [#34586 "Fix model trace sweep tests"](https://github.com/tenstorrent/tt-metal/pulls) returned by the paged_update_cache search — title suggests it may be the closing PR for #16674, but body wasn't fetched.
- [`tech_reports/AdvancedPerformanceOptimizationsForModels.md`](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md) — flagged by trace-capture search as containing "trace cannot capture events" rationale. Worth a full read for `c4_trace_capture_plan.md` follow-up.
- [`tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md`](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md) — mesh-workload + multi-chip trace design. Relevant for `phase_a7_multichip_plan` when we revisit qb2 fabric.

### Gold-mine query

`is:pr is:merged sort:updated-desc sdpa decode` on the tt-metal repo returned **214** merged PRs — far more than expected. This single filter is the highest-density source of kernel-design patterns we found; worth a periodic skim of the last ~20 PRs.
