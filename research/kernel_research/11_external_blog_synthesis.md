# External Blog Synthesis: Tenstorrent Kernel Wisdom Curated for tt-model-bringup

**Corpus path:** `tt_docs_corpus/blogs/` (10 Marty posts + 7 Corsix Wormhole-series posts; see per-author `INDEX.md`).
**Audience:** anyone working on Qwen3.6-27B decode optimization, in-place KV scatter, multi-chip TP (QuietBox, 4 P150s), and custom Tensix kernels.
**Scope:** what these external authors got right that the official docs don't say loudly enough.

---

## 1. Data-Parallelism Principles (Marty's Pattern Library)

Marty's writing converges on one mental model: **the chip is a 2D grid of independent dataflow threads, not a SIMT array**. Three quotes carry the load:

- *"Programming LLKs is like controlling a massive processor by coding as if it is an embedded system."* — Mandelbrot post, "Mindset" section.
- *"There is no cache hierarchy on Tenstorrent chips - this is a deliberate design choice."* — Programming Tenstorrent, Memory section.
- *"The Tenstorrent chips themselves have enough NoC bandwidth and computing power to act as a switch on its own. While compute is not affected at all."* — Programming Tenstorrent, Multi-Chip section.

Practical distribution patterns Marty actually used:

| Pattern | Where | What to steal |
|---------|-------|---------------|
| SPMD by row-chunk | Mandelbrot multi-core | `split_work_to_cores` + per-core `start_row`/`end_row` runtime args. Stays bandwidth-flat to 16384x16384. |
| Phase split (compute-bound vs memory-bound) | RoPE post, "Parallelization Across Cores" | Two independent `split_work_to_cores` calls (active phase, passive phase) then `merge()`. Reader/writer kernels handle each separately. |
| Both-NoC trick | Programming Tenstorrent, "Insights NOT in Official Docs" #1 | Use NoC0 and NoC1 simultaneously for **doubled read bandwidth**; cost is losing the ability to overlap reads with writes. |
| Same-address multicast | Memory post | Because lock-step allocation guarantees identical addresses across cores, `noc_async_write_multicast` with one address fans out to every core. |
| Ring/all-reduce primitives | Programming Tenstorrent, "CCL" | `ttnn.all_reduce`, `ttnn.broadcast` — Tenstorrent's NCCL equivalent. |

Hard latency numbers we should design around come from **Corsix Part 3, "Key Findings"**: 9 cycles per NoC hop, same-row round-trip 90 cycles, same-column 108, off-diagonal 198. That's the floor for any data-parallel reduce we sketch for C'7 on qb2.

---

## 2. Kernel-Writing Tactical Advice (the unwritten manual)

The blog posts beat the official docs in three areas:

### Debugging tactics
- **Visual output inspection over print debugging** — Mandelbrot post detected `SFPSTORE` lane skipping by *seeing stripes* in the rendered image. Equivalent for us: dump intermediate tile state to fp32 row-major and `numpy.imshow` from CPU. Cheaper than Tracy for correctness bugs.
- **Tracy is real but gated** — RoPE post, "Tracy Profiling": `BUILD_TRACY` CMake flag + `DeviceZoneScopedN("ROPE-TILE")` markers + `TT_METAL_DEVICE_PROFILER=1`. The catch we already hit (`reference_tracy_build_qb1.md`): a non-Tracy ttnn binary aborts under `TT_METAL_DEVICE_PROFILER=1` — Marty confirms the rebuild is required, not a configuration mistake.
- **Code-review reference impls** — Mandelbrot, "Code Review as Learning Tool": when docs are sparse, the canonical move is reading `_calculate_sfpu_binary_` in `tt-llk-wh-b0/`. Pattern `ITERATIONS = 8` is convention for overlapping unpack + math.

### Common pitfalls
- **`SFPSTORE` lane skipping** (Mandelbrot): writes from `lreg` to `dst` skip lanes 0->0, 1->2, 2->4, ... You must process even and odd lanes separately. `vConstTileId` holds `lane_id * 2`.
- **No fp32 subtract instruction** (Corsix Part 6): always synthesized as `SFPMAD` with VB = -1.0. If you wonder why a "trivial" subtract burns a multiplier slot, this is why.
- **`v_if`/`v_elseif` execute both sides** (FOSDEM 2026 draft, RoPE post): no early branch termination. *"Be wary about the performance drag from having 2 large possible paths."* Restructure to keep predicated branches small.
- **Hardware-bug list** (Corsix Part 6): PRNG poor (adjacent lanes share 30/32 bits); `ShiftLanesRight` broken on every 8th lane; denormals flushed to zero; 2-cycle latency needs `SFPNOP` between dependents. The team should keep this checklist near the kernel code.
- **TTNN view ops are not free** (Programming Tenstorrent, "View Operations"): `transpose`/`slice` on the last two (tilized) dims actually copy memory. Permute on non-tilized dims is fast. Aligns with our existing `feedback_ttnn_slice_row_aligned.md` finding.

### Performance heuristics
- **Hoist trig/exp outside the inner loop** — RoPE post, "Optimizations Timeline": fp division hoisting alone went 44us -> 39us; reusing exp per column went -17us. Dispatch wins are amortized when CB depth is enough.
- **Dst registers double as a scratchpad** (FOSDEM 2026 draft + RoPE post optimization #6): *"if you ever have expensive computation, it is totally safe to just put that in one of the Dst registers, twice"* — i.e., write the precomputed value twice (so both halves of the double-buffered Dst see it) and skip recomputation on subsequent passes. Marty got 32us -> 22.6us per 4 tile pairs from this on RoPE.
- **Magic number 8** — Marty observed that LLKs use `ITERATIONS = 8` to overlap unpack and math via the Dst double-buffering. Match this in custom kernels.

---

## 3. Specific to Our Project

### RoPE on Tensix (Marty's RoPE post -> our partial-RoPE level-1 plan)
Marty's structure for our team to compare against:
1. Split input into **active** (rotate) and **passive** (passthrough) regions; route active via `cb_in0`, passive via a *separate* `cb_bypass` CB (CBIndex c_17) that goes reader->writer without touching compute. Saves an entire compute-kernel cycle per passive tile.
2. SFPU per-face loop (4 faces of 16x16) — uses `TTI_SETRWC` with `CR_D, 8` to advance Dst by 8 rows between faces.
3. **Numerical accuracy** required `exp_24f` (Moroz et al. 2022) — built-in `exp_21f` had max error ~18 after `*10000`. Our `feedback_partial_rope_level1_trick.md` and `feedback_c3_native_rope_abandoned.md` should retain this — manual rotate-half is the right answer, but if we *do* write our own SFPU exp, this is the citation.
4. **Per-batch positions, not per-row** — Marty's hardest blunder. If we add a custom RoPE kernel, the position-index tensor is `[B, N]` row-major and read **once per batch** outside the inner per-tile loops.
5. **Phase-split scheduling** — `split_work_to_cores` separately on active and passive tile counts, then `merge()`. Map directly: if our future kernel has compute-heavy + memory-heavy phases, do not co-schedule them on the same core.

### Custom Tensix Kernel for In-Place KV Scatter (our `c_scatter_kernel_design.md`)
Patterns from the blogs that we should fold into the design doc:
- **Lock-step allocation** (Memory post): the same address on every core means we can write the scatter result by `noc_async_write_multicast` if multiple chips share the same paged cache layout — relevant for C'7 multi-chip.
- **Mutex / semaphore primitives** (Corsix Part 5): `ATGETM`/`SEMWAIT` are the right tools for in-place mutation when paged_update_cache needs ordered writes across N reader cores into one writer's cache tile.
- **Reader/Writer can be cooperating, not just sequential** (Programming Tenstorrent): the writer can `noc_async_write` partial tiles back to the source's L1 — which is exactly the "in-place" pattern we want. CB c_17 in Marty's RoPE post is the template (reader-to-writer bypass).
- **`paged_update_cache` corollary** — Corsix Part 4 details how cross-chip routing works (queue + 6D address). Multi-chip in-place scatter is non-trivial because we need ordered writes across the Ethernet fabric — submission queue + `CMD_ORDERED` flag matter.

### Multi-Chip TP for Qwen3.6-27B (phase C'7) on qb2
- **Corsix Part 4** is the definitive reference for what we'll actually be talking to on qb2's working fabric. The `eth_queue_t` / `routing_cmd_t` structures and three payload modes (inline 4B / 1KB / 3.75GB DMA) frame the bandwidth/latency trade-offs of all_reduce vs gather designs.
- **Marty's CCL section** + **lock-step allocation** make `ttnn.all_reduce` and `ttnn.broadcast` the first-cut primitives — only drop to custom CCL if profiling shows the canned op is wrong shape.
- **NoC propagation** numbers (Corsix Part 3): for a 4-chip ring, every all-reduce step pays at least one Ethernet RTT + a same-row NoC trip per leg. The 9 cycles/hop floor sets our perf ceiling math.

---

## 4. Reading Priority (30-minute developer)

| Rank | Post | Why it beats the alternatives |
|------|------|--------------------------------|
| 1 | **Marty — Programming Tenstorrent Processors (2025-04-21)** | The single best concept-to-code map. Reader/compute/writer + CB + TTNN gotchas + multi-chip in one place. If you read nothing else, read this. |
| 2 | **Marty — The Real Tensix Programming Model (FOSDEM 2026)** | Crystallizes the 5-baby-RISC-V mental model and the Dst register lifecycle (`tile_regs_acquire/commit/wait/release`). Fixes the misconception that "compute kernel" is one kernel — it's UNPACK+MATH+PACK threaded via macro magic. |
| 3 | **Corsix — Wormhole Part 5: Taking Apart T Tiles** | Maps every Metalium API call to what the silicon is actually doing. After this, "what does `acquire_dst()` do?" is no longer a mystery. Pair with Corsix Part 6 if you have an extra 15 minutes for SFPU ISA. |

Honourable mention if the developer is specifically writing a kernel: **Marty — RoPE post (2025-10-26)**. End-to-end op write, including the Dst-scratchpad trick and the phase-split scheduling pattern. Worth a second 30 minutes when ready.

---

## 5. What Surprised Us

- **The `SFPSTORE` lane-skipping pattern** (Mandelbrot post) — this is a *correctness* hazard, not a perf one, and is not in any official doc we've found. Anyone writing an SFPU op for the first time will get garbage and not know why. We should bake this into our internal SFPU checklist.
- **"No fp32 subtract"** (Corsix Part 6) — explains why anything subtractive on SFPU has 2x the throughput cost we'd otherwise expect. Useful when reasoning about softmax-style ops.
- **PRNG quality "adjacent lanes share 30/32 bits"** (Corsix Part 6) — if we ever add stochastic rounding or dropout, the SFPU PRNG is not a real RNG.
- **Marty's "RISC-V backend compiler crashes when broadcasting lane-ID calcs"** (Mandelbrot) — the compiler limitation forces manual duplication. We should expect to hit similar walls and have a manual-duplication template ready.
- **Tenstorrent supports `MOP_CFG` + `REPLAY` instruction expanders** (Corsix Part 5) — the chip has built-in macro recording, used by SFPU and (per FOSDEM draft) actively exploited to "hide" loop body cost while the baby core does control flow. If we ever measure dispatch-bound kernels, this is the lever.
