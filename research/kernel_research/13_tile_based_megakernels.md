# 13 — Tile-Based DSLs and Megakernels: A Tensix Mapping

**Audience**: anyone on tt-model-bringup who's been writing owned TT-Metal kernels by hand (qwen36_gdn_decode_owned, qwen36_topk_owned, qwen36_moe_ffn_decode_owned, nemotron3_mamba2_decode_owned, MM7) and is wondering whether the tile-ai stack (TileLang / TileRT / TileOps) and the megakernel idea from recent GPU inference work apply to us. Companion to entry 08 (`08_tensix_vs_cuda_programming_model.md`).

**Stance**: empirical, not aspirational. Two of the three TileLang components are CUDA-first; the third (TileLoom, the arxiv paper) actually targets Tenstorrent. Most of what's interesting to us is the *idea pattern*, not the runtime — and the idea pattern overlaps with what we already do, with a few concrete refinements.

**Sources**:
- `tile-ai/tilelang` README (github.com), confirmed via `gh repo view tile-ai/tilelang`: DSL on TVM, GEMM/FlashAttention/MLA examples, backends include NVIDIA, AMD MI300X, Apple Metal, Ascend NPU, WebGPU. **No Tenstorrent backend listed.**
- `tile-ai/TileRT` README: ultra-low-latency LLM inference runtime; v0.1.4 ships pre-built wheels for 8× NVIDIA B200 only. CUDA-only by construction.
- `tile-ai/TileOps` README: GPU operator library for LLMs built on TileLang; explicitly NVIDIA SM\_90 (Hopper) only; spec-driven kernels generated/evaluated via AI agents.
- TileLoom paper, arxiv:2512.22168v2 — "TileLoom: Automatic Dataflow Planning for Tile-Based Languages on Spatial Dataflow Accelerators." Evaluated on Tenstorrent Wormhole (8×8) and Blackhole (12×10). Frontends: Triton (via `triton-shared`) and Helion. Backend: TT-Metalium.
- Internal: `experiments/owned_ops/qwen36_gdn_decode_owned/README.md`, `experiments/owned_ops/qwen36_moe_ffn_decode_owned/README.md`, `experiments/owned_ops/qwen36_topk_owned/README.md`, `experiments/cb/isolate/mamba2_*.py`. Memory: `[[feedback-mm-init-prime-required]]`, `[[reference-tt-llk-frozen-in-tt-metal]]`, `[[feedback-gdn-vs-mamba2-kernel-delta]]`, `[[feedback-sdpa-transpose-b-flag-escape-hatch]]`.

**Disclaimer up front.** Yossi's email pointed at "megakernel concepts" alongside the arxiv link. The arxiv paper at that link is TileLoom, which is *not* a megakernel paper — it's a tile-distribution compiler for spatial accelerators. Megakernel as a term originates from Hazy Research / Stanford work in 2025 (single-CUDA-kernel-per-transformer-step). Both ideas are relevant to us but they're different things. This entry treats them as two threads.

---

## 1. What is TileLang / TileRT / TileOps?

The three are stacked layers from one team:

| Layer | What it is | Hardware | Where it bites us |
| --- | --- | --- | --- |
| **TileLang** | Pythonic DSL on TVM; users write block-level kernels (matmul, FlashAttention, LinearAttention) at "tile" granularity, the compiler emits CUDA/HIP/Metal/Ascend/WebGPU | NVIDIA H100/A100/4090, AMD MI250/MI300X, Apple Metal (PR #799, 2025-10), Ascend NPU (2025-09), WebGPU. **No Tenstorrent.** | The abstraction we'd port from, not the runtime |
| **TileOps** | High-level operator library on top of TileLang; "spec-driven" agent-generated kernels | NVIDIA Hopper SM\_90 only | Style reference for a kernel registry, not portable |
| **TileRT** | LLM inference runtime that schedules tile-level tasks across multiple devices with overlapped execution | NVIDIA B200, 8-GPU only, prebuilt wheels | A scheduling pattern (fine-grained tile tasks dispatched dynamically) — not a runtime we can install |

**TileLang's central abstraction** (paraphrased from the README and what gets sketched in their MLA/FlashAttention examples) is *block-level* programming: instead of writing one CUDA thread's work and then trusting MMA intrinsics + cooperative groups, you describe what a thread block does to a 2D tile of the input. The compiler decides warp/MMA mapping, async copy schedules, and shared-memory layouts. A few lines of Python become a competitive FlashMLA on H100; their May-2025 post claimed 80 lines for MLA decode at par with hand-tuned `FlashMLA`. Backends are added as new TVM lowering passes (e.g. CuTeDSL backend in PR #1421, Dec 2025; the Z3-arith-analyzer integration in PR #1367).

**The mental model**, the part that's worth porting: *"a kernel's body operates on tiles; mapping tiles to execution units (warps / cores / chips) is the compiler's job, not the programmer's."* This is genuinely different from CUDA (per-thread, warps as a side-effect) and Triton (per-program, tile-by-tile but still warp-aware). TileLang sits closer to CUTLASS-CuTe in altitude but with a Python frontend and a TVM IR.

### What TileLoom adds

The arxiv paper (2512.22168v2) is what makes this Tenstorrent-relevant. TileLoom is an MLIR-based compiler that takes a tile-language program (Triton or Helion) and lowers it to TT-Metalium, deciding:

- **Spatiotemporal mapping** (§2.2, §2.4): which tiles run on which cores (`affine.parallel` over the core array), and which run in which temporal wave (`affine.for`).
- **Dataflow planning** (§2.3): spatial reuse (which operands multicast across the NoC), temporal reuse (which stay in L1 across loop iterations), buffer placement (L1 vs DRAM).
- **Lifetime analysis** (§3.1) over the block-level compute graph to pick CB allocations and the synchronization between dataflow/compute kernels.

Reported numbers, all relative to vendor TT-Metal ops:

| Workload | Wormhole | Blackhole |
| --- | --- | --- |
| FlashAttention | 1.94× | 1.98× |
| GEMM | 0.95× | 1.10× |
| Flash Decode | 0.84× | 0.87× |
| Mamba Chunk Scan (vs unfused TTNN) | 27.23× | 16.27× |

Two honest things about those numbers: (a) GEMM doesn't beat vendor because vendor is already hand-tuned; (b) the Mamba 27× number is against the unfused composition, which is the trap our `qwen36_gdn_decode_owned` work also exists to avoid — it's not 27× against a hand-fused TT-Metalium kernel. Limitations the paper itself raises: Flash Decode underperforms because there isn't much dataflow choice to make on that op, and the performance model carries ±17% error which an optional top-k profiling step has to close (§3.3.3).

**What TileLoom does *not* do**: synthesize the kernel from scratch. It takes an already-tile-level kernel (a Triton or Helion program) and assigns tiles to cores. It's the *mapping* problem, not the *kernel-writing* problem.

---

## 2. What is a megakernel?

I want to be careful here because the term has two distinct lineages and the arxiv link the team sent us is *not* an example of either.

**Lineage A — single-CUDA-kernel transformers (Hazy Research, 2025).** The idea: instead of launching ~150 separate CUDA kernels per transformer step (attention QKV-projection, RMSNorm, SDPA, AllReduce, FFN-up, GeLU, FFN-down, sampling, …), launch *one* persistent kernel that uses a state machine + on-device task queue to walk the whole step. The wins they target:
- **Launch overhead**: a single kernel launch instead of ~150 means ~tens of μs saved per step. At 200 tok/s the budget is ~5 ms/tok; launch overhead is a meaningful fraction.
- **No host round-trip between ops** ⇒ no `cudaStreamSynchronize` storms; control flow stays on-device.
- **L2 / shared-memory reuse across ops** that would have been flushed across kernel boundaries.
- **Persistent thread blocks** that hold state (KV cache pointers, allreduce buffers) without re-launching.

The Hazy implementation is essentially a giant CUDA kernel whose threads execute a tiny on-device interpreter that pops "tasks" (matmul tile, RMS row, sample) off a queue. The win is on the order of 1.2–1.5× over Triton-fused baselines, more vs naive PyTorch.

**Lineage B — fused-tile kernels in tile-DSLs (TileLoom's Mamba evaluation).** Here "megakernel" is more modest: instead of three or four back-to-back TT-Metalium ops, a single TT-Metalium kernel computes a chunk of Mamba in L1 without round-tripping to DRAM. That's exactly what our `qwen36_gdn_decode_owned` and `qwen36_moe_ffn_decode_owned` are. The win is **DRAM round-trip elimination**, not launch overhead. On Tenstorrent today, where one trace replay is ~50–80 ms/tok at 35B, kernel-dispatch overhead is dwarfed by NoC + DRAM traffic — fewer kernel boundaries means fewer L1↔DRAM round-trips, and that *is* the dominant lever (the qwen36_gdn_decode_owned microbench shows ~15 μs gain over component-chain, which is dispatch, but the *kernel-time* gain comes from skipping intermediate DRAM stores).

**Translation to Tensix.** Lineage A's launch-overhead motivation doesn't move the needle for us: a TT-Metal trace replay is one host launch per token already (see `06_trace_capture_internals.md`); we don't pay per-op launch. What we *do* pay is **NoC + DRAM traffic between ops** and **CB-pipeline startup costs** (`mm_init` priming, transpose state setup — see `[[feedback-mm-init-prime-required]]`). Lineage B's "keep more state in L1 across what used to be op boundaries" is the lever that *is* moving for us in `qwen36_gdn_decode_owned`'s native-IO path and in MM7's full Mamba2 SSD step.

So: megakernels-in-the-Hazy-sense are a CUDA-specific answer to a CUDA-specific bottleneck. The underlying idea — *fewer kernel boundaries means keeping more state hot across what used to be a sync point* — is what we already practice and should keep pushing on. That's the connection point.

---

## 3. The Tensix programming model in tile-DSL language

Mapping the TileLang vocabulary onto what we already wrote in entry 08:

| TileLang concept | Tensix mapping | Notes |
| --- | --- | --- |
| **Tile** (the unit of work) | 32×32 BFloat16 block stored as 4×(16×16) faces in TILE_LAYOUT (`tiles.md:25-39`) | Already the native unit. The size matches exactly. |
| **Block-level program** (one thread block per output tile, compiler picks warps) | Reader/compute/writer trinity on **one Tensix core**, compiler picks RISC-V split (TRISC0/1/2) | Tensix's "block" *is* a core; the analogy of "compiler picks warps" maps to "compiler picks which TRISC unpacks/maths/packs." We currently do this picking by hand. |
| **Thread-block-level scheduling at tile granularity** | Per-core runtime args; host loop assigns `(slot, value_tile)` blocks to cores | This is `split_work_to_cores` (`08:138`). Static, host-determined. |
| **Shared memory (CUTLASS shmem ↔ TileLang `T.alloc_shared`)** | L1 SRAM, 1.5 MB/core, addressed via CBs and explicit `noc_async_*` | No HW-managed cache — programmer holds the staging. |
| **Async copy (`cp.async` / TMA)** | `noc_async_read_tile` + `noc_async_read_barrier` + CB push | NoC packets are *the* async-copy primitive on Tensix. |
| **Persistent kernel / persistent block** | Trace replay (one program slot per core, no re-launch) | We get persistence for free via `06_trace_capture_internals.md`. |
| **MMA / WGMMA / matrix-core intrinsic** | `matmul_tiles(cb_a, cb_b, …, dst_idx)` on the FPU | One instruction per 32×32 tile pair. |
| **Block-reduce (warp shuffle + shmem)** | SFPU `reduce_tile` within one core; multicast + semaphore across cores | Multi-core reduce is explicit, not a primitive. |
| **Tile autotuning** (TileLang's autotuner picks `block_M, block_N, block_K`) | Host-side picking of `CoreRangeSet` shape + per-core `(rows, cols)` split | We currently do this by hand based on the kernel; an autotuner would explore `(num_cores, tiles_per_core, cb_depth)`. |

The structural mapping is **clean for kernels whose work is regular tile×tile multiply-or-reduce** — GEMM, attention, the Mamba2 SSD reduction. It is **awkward for kernels with face-layout-sensitive math** (RoPE, anything where SFPU lane position matters; see `08:142-159`) because TileLang's tile abstraction is layout-opaque.

**The cleanest TileLang→Tensix translation**, on paper:

```python
# Hypothetical TileLang for one decoder layer's GDN recurrence
@T.prim_func
def gdn_step(Q: T.tensor((1, S, D), "bf16"), K: ..., V: ...,
             alpha: T.tensor((1, S, 1), "bf16"), beta: ...,
             State: T.tensor((1, S, D, D), "bf16"),
             Out:   T.tensor((1, S, D), "bf16")):
    with T.Kernel(S, T.ceildiv(D, 32)) as (slot, vt):     # one (slot, value_tile) per core
        st = T.alloc_shared((D, 32), "bf16")              # L1 staging
        T.copy(State[0, slot, :, vt*32:(vt+1)*32], st)    # CB push from BRISC
        # ... fused arithmetic in L1 ...
        T.copy(st, State[0, slot, :, vt*32:(vt+1)*32])
        T.copy(Out_tile, Out[0, slot, vt*32:(vt+1)*32])
```

The `T.Kernel(S, T.ceildiv(D, 32))` line is **exactly** the `block = (slot, value_tile)` decomposition we already wrote into `qwen36_gdn_decode_owned`'s host code (README §"SPMD work unit", line 41). The `T.alloc_shared` + `T.copy` are CBs + `noc_async_read_tile` + `cb_push_back`. The arithmetic body is what's currently spread across `kernels/compute/qwen36_gdn_decode_owned.cpp`.

What that 20-line TileLang snippet would buy us, if a TileLoom-style backend existed for Tensix that handled it:
- Auto-`mm_init` priming insertion (the `[[feedback-mm-init-prime-required]]` 4-ingredient recipe).
- Auto-CB sizing (today we hand-tune to 2-tile depth for double-buffering).
- Auto-multicore split (today: hand-written host loop in the program factory).
- Auto-`transpose_wh_tile` insertion when an operand needs column-major (today: `k_col` keyword arg toggles this).

What it wouldn't buy us, until a Tensix-aware compiler with face-layout awareness exists:
- Anything where SFPU lane mapping matters (RoPE, GDN's per-slot reductions when the slot dim straddles faces).
- The `pack_reconfig` + `mm_init_short` discipline for fused multi-op kernels — this requires the compiler to understand the FPU/SFPU mode-switch cost, which is hardware-specific in a way that TileLang's TVM backend layer doesn't currently know about.

---

## 4. Where this could land on our work

Concrete, in priority order.

### 4a. Megakernel candidates we should look at hard

**Already done (one-op-fused):** `qwen36_gdn_decode_owned` (5 sub-ops fused), `qwen36_moe_ffn_decode_owned` G1b (gate_up + silu + up + eo + cross-expert reduce, 5 ops fused in one Tensix core), `qwen36_decay_gate_decode_owned`, `nemotron3_mamba2_decode_owned` (MM7 SSD step). These are the model-specific recurrences where the "intermediate stays in L1" argument is sharpest.

**Worth scoping next:**

| Candidate | Sub-ops fused | Why it's a fit |
| --- | --- | --- |
| **Post-attention residual stack** | `paged_sdpa_output` + `o_proj_matmul` + `add(residual)` + `rms_norm` | The output tile pattern is identical for all four (per-slot, per-hidden-tile); SDPA already writes to L1; o_proj's K-axis is small enough to keep one head's K in L1; the rest is single-tile arithmetic. Production today round-trips to DRAM three times. |
| **Pre-MoE-router stack** | `rms_norm` + `router_logits_matmul` + `softmax_topk` (using owned topk) | All three are per-token, hidden-sized. The router logits matmul is small (`HIDDEN×NUM_EXPERTS`), and the owned topk already exists (qwen36_topk_owned). One core, one fused kernel could subsume them per slot. |
| **KV-cache scatter + RoPE + paged update** | already fused on the CUDA side in vLLM | We currently do these as three TTNN ops with intermediate DRAM tensors. RoPE is the painful one because of face-layout (Marty's post + entry 08 §7c) — but precisely because it's lane-aware, doing it adjacent to the K writeback saves a DRAM round-trip. |

**Not a fit:**

- **The full decoder layer.** On CUDA, "one kernel per transformer step" is feasible because the L2 cache is the bottleneck. On Tensix there is no L2; every cross-core data movement is an explicit NoC packet (`08:60-72`). A "one kernel" for a full layer would require multicasting activations across 130 cores for every all-reduce, semaphored against TP collectives, and would explode L1 budgets per core. The natural unit on Tensix is one core's worth of work per kernel, not one whole layer. Lineage-A megakernels don't translate.
- **MoE expert routing across cores.** Each expert's weights are on a different chip (mesh dim 1). A megakernel can't span chips; cross-chip is `ttnn.all_to_all_dispatch` (`[[reference-all-to-all-dispatch-shape-contract]]`) and lives between kernels.

### 4b. The 4-ingredient mm_init recipe as a TileLang annotation

`[[feedback-mm-init-prime-required]]`: Blackhole TRISC hangs at ~4 iters of transpose+matmul+binary chains unless we (1) prime with a real matmul, (2) pre-transpose, (3) `mm_init_short` per iter, (4) `pack_reconfig` per iter. This is **not in any TT-Metal doc** (`[[reference-tt-llk-frozen-in-tt-metal]]`); we discovered it by hanging the chip four times.

A TileLang-style DSL is exactly the right place to encode this kind of folklore as a compiler invariant: the IR pass that lowers a `T.fused_matmul_transpose_binary(...)` macro inserts the four ingredients automatically. *In principle*, that's the kind of constraint a TileLoom-style Tensix backend would absorb. In practice, our `qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp` carries those four ingredients by hand, and we re-paste them into MM7's Mamba2 compute kernel and into qwen36_moe_ffn_decode_owned. This is the strongest argument I can think of *for* adopting a DSL on Tensix — not for performance, but for **encoding the hardware's undocumented invariants in one place**.

### 4c. Tile-distribution autotuning for owned kernels

TileLoom's central contribution is automatic spatiotemporal mapping. Our hand-tuned analog: `split_work_to_cores(grid, num_work)` for the regular cases, plus careful per-program-factory work assignment for MM7 G2 (64 (batch, head) blocks across the Tensix grid, one head per core — see `feedback_mm_init_prime_required.md` and the G2 commit `bcc2ce9`).

The case where automation would help: **MM7 G3** (B=2 at 64 heads, blocks_per_core>1 — currently parked because the existing factory wedges). TileLoom's dataflow planner explicitly handles "more tile instances than cores → temporal waves" (§2.2). If we ever go after multi-batch decode at full head count, the mapping problem grows combinatorially and hand-tuning each new shape is sunk cost.

### 4d. What we shouldn't import

- **TileRT itself** — Linux x86_64, CUDA 13.2, B200 wheels. Inapplicable.
- **TileOps** — Hopper-only. The API style ("spec-driven, agent-generated kernels") is plausible to fork as a methodology — we could imagine a `tt-ops` registry where every owned kernel ships with a numpy oracle, a correctness ladder, and a microbench — but that's just naming what we already do across `experiments/owned_ops/*/test_*.py` + `benchmark_*.py`.
- **TileLang as a Python library on Tensix** — there is no Tenstorrent backend in the TileLang TVM tree. Adding one is the *exact* contribution TileLoom makes, and TileLoom is a research prototype, not a shipped backend.

The pragmatic adoption is: **read TileLoom's MLIR passes for ideas, keep writing TT-Metal kernels by hand, but lift the 4-ingredient recipes and the megakernel-candidate list into a small Python codegen layer of our own.** Already partly true: `experiments/owned_ops/*/integrate_into_ttmetal.py` plus `_fork_from_upstream.py` (qwen36_topk_owned) are codegen scripts in spirit. Generalizing those into "given a fused-op spec, emit the four kernels + the program factory + the host registration" is a finite, sub-week project that captures most of the TileLang value at our scale.

---

## 5. Open questions

1. **Is `qwen36_moe_ffn_decode_owned` actually faster end-to-end after G2/G3?** G1b proved single-core correctness, but full-decode benchmark hasn't shipped (see its README §Stages). The megakernel hypothesis (DRAM-traffic reduction dominates) predicts a real win; we should measure before assuming.
2. **Can we hoist the `mm_init` priming + `pack_reconfig` discipline into a header-level macro shared across all owned kernels?** Today each kernel re-pastes it. A `tt_metal_mega_recipe.hpp` of ours, included from every owned compute kernel, would make the invariant grep-able and patchable.
3. **Is the post-attention-residual fusion (4a row 1) worth scoping?** It would need a probe analogous to `experiments/cb/isolate/paged_sdpa_*.py` that times the full residual stack with and without round-trip to DRAM. Cheap to do; do it.
4. **Does TileLoom's source ship anywhere?** The paper doesn't link a repo. If the team open-sources their MLIR passes, we should fork and read, because that's the lowest-effort path to a TileLang→Tensix bridge. Currently: unknown.
5. **Does the Hazy-style "task queue inside the kernel" idea have any Tensix analog?** Trace replay is already a "one launch per token" runtime; the only thing missing relative to a CUDA megakernel is on-device control-flow over which op runs next. Tensix RISC-Vs are full general-purpose cores — they can run a state machine. Whether there is a meaningful win over trace replay (which is already device-resident) is unclear and probably worth zero engineering effort until trace replay is no longer dominant.
6. **Face-layout-sensitive ops in a tile DSL.** RoPE and any per-slot reduction where the slot dim crosses face boundaries (`08:150`, Marty 2025-10-26) need face-aware codegen that TileLang doesn't currently expose. If we were to build a Tensix backend, this is the first abstraction leak we'd hit. Open: is face awareness an MLIR pass or a library-of-blessed-primitives like SFPU verbs?

---

## 6. The one-line takeaway

The tile-DSL stack we care about is one paper deep (TileLoom) and even there the contribution is *mapping*, not *kernel writing* — which is good news, because mapping is the part we're already doing by hand in `split_work_to_cores` and would happily automate. The megakernel idea, in the Tensix-relevant form (fewer kernel boundaries → more L1 reuse), is what `qwen36_gdn_decode_owned`, `qwen36_moe_ffn_decode_owned`, and MM7 already implement; the next concrete adoption is **post-attention residual fusion** and **a shared header that bakes the `mm_init`+`pack_reconfig` recipe into one place**.
