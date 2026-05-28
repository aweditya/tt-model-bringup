# A dataflow → Tensix mapping representation (TDG: Tensix Dataflow Graph)

**Status:** living design doc. Research + design only — nothing here has run on
hardware. Every hardware claim cites a source (repo doc, tt-metal doc, or paper).
**Goal:** replace "write a kernel by intuition" with "compose a small set of
visual primitives that make the SPMD core split, the MIMD intra-tile pipeline,
the NoC traffic, and the L1/CB buffering all *explicit and checkable* before we
build."

---

## 1. Problem statement

We write custom Tensix kernels (`owned_gdn`, `owned_decay_gate`, the
`qwen36_moe_ffn_decode_owned` G0→G4 build) and they *work*, but they leave the
P150 badly under-utilised. The MoE FFN scoping doc measured the production
batched MoE matmul at **104 GB/s of 404 GB/s peak (26%) and ~150 GFLOPS of
9.25 TFLOPS (1.6%)** — "neither BW-bound nor compute-bound … it's tile-overhead
and core-utilisation bound" (`research/35b_moe_ffn_kernel_scoping.md`). The owned
kernels themselves are single-core or sparse: the GDN kernel runs "8 slots × 4
value_tiles = 32 blocks across 110 cores (each core gets ≤1 block; ~78 cores
idle)"; the MoE G1a kernel is "single-core loop over all E experts. 1 of 110
cores used. ~99% compute idle" (`research/35b_moe_ffn_kernel_perf_deferrals.md`,
rows D-G0-01, D-G1a-02).

The decisions that *cause* under-utilisation are made implicitly, scattered
across three files (program factory + reader + compute), and only become visible
after a Tracy run or a deadlock at a larger shape. Two of the three deadlocks we
hit in the MoE kernel
(`research/35b_moe_ffn_kernel_perf_deferrals.md`, "Deadlocks caught at
large-shape") were classic dataflow-graph errors — a CB depth smaller than a
`cb_wait_front(cb, N)` count, and a producer-ordering inversion — that a
*representation with rate/depth annotations would have flagged statically.*

So the thesis: **the artifact we keep re-deriving in our heads is a dataflow
graph annotated with hardware bindings.** If we draw it explicitly, kernel
design becomes composition, and a handful of cheap static checks catch the
under-utilisation and the deadlocks before we touch the device.

### What "principled" has to mean here (the 4 levers)

A P150 kernel's performance is determined by four decisions. The representation
must make each one a first-class, annotatable thing:

1. **SPMD partition** — how the logical iteration space is split across the
   ~110 Tensix cores. (Are 78 of them idle? `35b_moe_ffn_kernel_scoping.md`.)
2. **MIMD intra-tile pipeline** — how the work inside one Tensix is split across
   the 5 baby RISC-Vs: 2 data-movement (NCRISC/BRISC reader+writer) and the
   3-stage unpack→math→pack compute pipe (TRISC). Are reader and compute
   overlapped, or serialised? (`research/01_tenstorrent_hardware.md`.)
3. **NoC traffic** — what moves DRAM→L1, L1→L1 (mcast/gather), L1→DRAM, and at
   what volume. (The MoE Candidate-A cross-core reduction is a NoC pattern we
   *don't yet have*; `35b_moe_ffn_kernel_scoping.md`.)
4. **L1 / CB residency** — what stays resident vs streams, and the *depth* of
   each circular buffer (the FIFO between the baby cores). Depth-too-small is a
   deadlock; depth-too-large wastes the 1.39 MiB L1.

A representation that doesn't surface all four is just a flowchart.

---

## 2. The hardware, distilled to what the representation must model

Sourced from `research/01_tenstorrent_hardware.md`,
`feedback_p150_memory_bandwidth_measured` (MEMORY.md), the Corsix tt-wh series,
and clehaxze's "Programming Tenstorrent processors". Numbers are P150 / Blackhole
on this project's qb1.

### 2.1 Two levels of parallelism (the user's framing)

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  CHIP  =  SPMD across ~110 worker Tensix cores (2D NoC torus)          │
  │                                                                        │
  │   each core runs the SAME program on DIFFERENT data, talks over NoC    │
  │                                                                        │
  │   ┌──────────┐   NoC0 (E,S) ─────────►   ┌──────────┐                  │
  │   │ Tensix   │                            │ Tensix   │   ...            │
  │   │ (x0,y0)  │   ◄───────── NoC1 (W,N)    │ (x1,y0)  │                  │
  │   └────┬─────┘                            └──────────┘                  │
  │        │  ZOOM IN: one Tensix = MIMD across 5 baby RISC-V cores         │
  │        ▼                                                                │
  │   ┌──────────────────────────────────────────────────────────────┐    │
  │   │  DATA-MOVEMENT (MIMD)        COMPUTE pipe (MIMD, 3 TRISC)      │    │
  │   │                                                               │    │
  │   │  NCRISC ──reader──►  [CB_in] ──► UNPACK ─► MATH ─► PACK ─► [CB_out]  │
  │   │  BRISC  ──writer──◄──────────────────────────────────────────┘    │
  │   │                                       │        │        │          │    │
  │   │                                  SrcA/SrcB   FPU/SFPU   Dst regs    │    │
  │   │   FPU = 32x32 matrix engine    SFPU = 32-lane vector engine         │    │
  │   └──────────────────────────────────────────────────────────────┘    │
  │                                                                        │
  │   L1 SRAM per core: 1.39 MiB (1408 KB user-allocatable). NO cache.     │
  └──────────────────────────────────────────────────────────────────────┘
   DRAM: 31.81 GB, 8 banks, ~404 GB/s measured (79% of 512 peak).
```

Key facts the representation must respect:

| Fact | Value | Source |
|---|---|---|
| Worker Tensix cores | ~110 (140 total, harvested) | `feedback_p150_memory_bandwidth_measured` |
| Baby RISC-Vs per Tensix | 5: 1 BRISC, 3 TRISC, 1 NCRISC | `research/01_tenstorrent_hardware.md` |
| Kernels per op | reader (NCRISC) + compute (TRISC×3) + writer (BRISC) | tt-metal METALIUM_GUIDE |
| Compute sub-pipeline | UNPACK → MATH → PACK (3 stages, the 3 TRISC) | `research/01`; clehaxze |
| Tile granularity | 32×32, contiguous in memory ("tilized") | tt-metal docs; `research/01` |
| L1 per core | 1.39 MiB, no cache, explicit DMA | `feedback_p150_memory_bandwidth_measured` |
| CB abstraction | FIFO between kernels, HW-mutex backed, double-buffer to overlap | tt-metal METALIUM_GUIDE; `research/01` |
| NoC | 2 unidirectional NoCs, 2D torus; ~9 cyc/hop; mcast supported | `research/01`; Corsix tt-wh |
| DRAM BW | ~404 GB/s measured | `feedback_p150_memory_bandwidth_measured` |
| Matmul fidelity | LoFi/HiFi2/HiFi4 multi-pass; bf16 = 4 passes | `research/01` |

### 2.2 The CB is the load-bearing abstraction

A circular buffer is the FIFO the user already programs against:
`cb_reserve_back` / `cb_push_back` (producer) and `cb_wait_front` /
`cb_pop_front` (consumer). It is *exactly* an SDF channel: a producer writes a
fixed number of tokens (tiles) per firing, a consumer reads a fixed number per
firing, and the buffer has a finite depth. This is the single most important
observation in this doc — see §3.4 and §4.

The reader→compute→writer trio per core is a **3-actor pipeline** (NCRISC,
TRISC-pipe, BRISC) connected by CBs. The chip is **~110 copies of that pipeline**
connected by the NoC. So the natural representation is a **two-level dataflow
graph**: an inner per-core actor graph, and an outer per-chip SPMD graph, with
the NoC as the edges of the outer graph.

---

## 3. Prior art: what transfers, what's missing

| Representation | Core idea | Transfers to Tensix? | What's missing |
|---|---|---|---|
| **Eyeriss row-stationary / dataflow taxonomy** (weight/output/row-stationary) | classify *what data stays put* in PEs to maximise reuse | **Yes, as vocabulary.** "h resident across experts" (MoE) is weight/input-stationary; GDN "state resident" is output-stationary. Naming the stationarity makes the L1-residency choice explicit. | Eyeriss is a *fixed-dataflow* taxonomy for a systolic PE array. Tensix cores are programmable MIMD pipelines, not PEs; no notion of CB depth, of the 3-stage TRISC pipe, or of two NoCs. |
| **Halide** (algorithm ⊥ schedule; `tile/parallel/vectorize/compute_at/store_at`) | separate *what* from *where/when*; schedule is a small DSL | **Yes, conceptually — this is the spine of the proposal.** Our "dataflow" = Halide algorithm; our "mapping annotations" = Halide schedule. `parallel`≈SPMD core split, `compute_at`≈CB residency/streaming boundary, `store_at`≈which CB. | Halide's machine model is CPU/GPU loop nests with caches. It has no first-class NoC, no explicit FIFO-depth/deadlock notion, no unpack/math/pack split. `tensorize` is the closest hook to a tile-matmul intrinsic. |
| **Timeloop + Accelergy** | loop-nest mapping + analytic energy/latency + bandwidth/buffer model; *mapper searches mappings* | **Partly.** The per-level buffer-size + bandwidth + utilisation model is exactly the roofline math we do by hand in `35b_moe_ffn_kernel_scoping.md`. Worth borrowing as the *cost annotation* on edges/nodes. | Buffer hierarchy is a tree of memories, not a 2D-torus of programmable cores; no mcast/gather over a NoC; no FIFO-deadlock model. |
| **Maestro** | data-centric directives; spatial vs temporal loops; reports reuse, NoC traversals, roofline | **Yes for the analysis output.** Maestro's spatial-loop (unrolled across PEs) vs temporal-loop (over time) split *is* our SPMD-across-cores vs sequential-within-core split. "NoC traversals" is a stat we want. | Same gap: PE-array model, fixed interconnect, no programmable per-core pipeline, no FIFO depth. |
| **SDF / Kahn Process Networks** (Lee & Messerschmitt 1987) | actors fire on fixed token rates; *rate-consistency* + *initial tokens* give static deadlock-freedom + finite buffer sizing | **Strongly — and under-used.** A CB *is* an SDF channel. Our two large-shape deadlocks were a rate/depth violation and a missing-initial-token / producer-order inversion — *exactly* the two things SDF analysis decides statically. | SDF alone has no spatial/hardware notion (no cores, no L1 budget, no roofline). It's the *correctness/scheduling layer*, not the perf layer. |
| **tt-mlir TTIR / TTNN / TTKernel dialects** (Tenstorrent's own compiler) | MLIR dialects: TTIR adds tiling/layout/fusion, lowers to Metalium | Already exists and is the *production automatic* path. But it's a compiler IR, not a **human-facing visual design medium** — you don't sketch a TTIR module to reason about a hand-written kernel. | Not designed for the "I am hand-writing a custom LLK kernel and want to see my SPMD/MIMD/NoC/CB decisions" workflow. No deliberate visual form. |
| **TL** (arXiv 2512.22168, "Tile-based languages for spatial dataflow", lowers Triton→TT-Metalium) | a *hardware representation* capturing interconnect topology + memory hierarchy + compute, and a mapper that "distributes tile instances across spatially distributed cores and exploits the NoC and distributed memories for reuse" | **This is the closest existing thing and it targets Tenstorrent directly.** It validates our problem framing almost verbatim ("distribute tile instances across cores, exploit NoC + distributed memory for reuse"). | It is an *automatic compiler* (Triton in, executable out). Our need is the opposite end: a *human design notation* for kernels we write by hand in LLK, where the whole point is to keep manual control. TL is the destination; TDG is the napkin. |

### 3.4 Verdict

No existing artifact is a **human-facing, visual, compositional design notation
for hand-written Tensix kernels** that simultaneously models the SPMD core split,
the MIMD intra-tile pipeline, the NoC, and CB depth/deadlock. The pieces all
exist:

- **Halide's algorithm⊥schedule split** → the overall shape.
- **SDF channel rates + depth + initial tokens** → the CB correctness layer
  (deadlock-free + buffer sizing, which is precisely where we got burned).
- **Maestro/Timeloop cost model** → the per-node/edge perf annotations (bytes,
  BW%, core-utilisation, roofline).
- **Eyeriss stationarity vocabulary** → naming the L1-residency choice.

The contribution proposed below is the **synthesis and the visual binding**, not
a new theory. We are honest about that in §6.

---

## 4. Proposed representation: the **Tensix Dataflow Graph (TDG)**

A TDG has **two layers** that mirror the SPMD/MIMD split, plus a third "binding"
overlay. You can draw it on paper; a rendered version is a layered node-link
diagram (see §4.5).

### 4.1 Layer 0 — the algorithm graph (hardware-free)

Nodes = **tile-ops** (the math the user actually writes: `matmul_tiles`,
`silu_tile`, `add_tiles`, `transpose_wh_tile`, …). Edges = **tensor
dependencies** carrying a *tile-rate* (how many 32×32 tiles flow per firing).
This is the Halide "algorithm" — no cores, no CBs yet. For the decay/gate kernel
this is literally the 5-op chain in the header comment of
`qwen36_decay_gate_decode_owned.cpp`.

### 4.2 Layer 1 — the per-core actor graph (the MIMD pipeline)

This is one Tensix's reader/compute/writer trio, drawn as **3 actors connected
by CB channels**. This is where MIMD-within-tile becomes visual.

```
   ── per-core actor graph (Tensix-local) ────────────────────────────────
                                                                         
   [NCRISC: reader]                                       [BRISC: writer]
        │  produces tiles                                      ▲ consumes
        ▼                                                      │
   ╔════════╗  rate / depth   ┌─────────── TRISC compute ─────────┐  ╔════════╗
   ║ CB_in  ║════════════════►│ UNPACK → MATH(FPU|SFPU) → PACK     │═►║ CB_out ║
   ╚════════╝   r=1 / d=2     └───────────────────────────────────┘  ╚════════╝
```

**Annotations on each element:**

- **Actor (RISC binding):** which baby core runs it — `NCRISC` (reader),
  `BRISC` (writer), or the `TRISC` compute pipe. The compute actor internally
  shows `UNPACK→MATH→PACK` and *which engine* MATH uses: `FPU` (matrix, for
  `matmul_tiles`) or `SFPU` (vector, for `silu`/`exp`/`sigmoid`/`softplus`). This
  is what makes the data-movement-vs-compute overlap (or lack of it) visible.
- **CB channel:** annotated `r=<tiles/firing>` (the `cb_push_back`/
  `cb_pop_front` count) and `d=<depth>` (the `num_tiles` in the
  `CircularBufferConfig`). This is the SDF channel triple.
- **Initial tokens:** a black dot on a channel = tiles that must be present
  before the consumer's first firing (the SDF "initial token" — see the MoE
  producer-ordering deadlock in §5).

### 4.3 Layer 2 — the SPMD overlay (the core grid + NoC)

The same Layer-1 actor graph is **replicated across a CoreRangeSet**, and the
overlay says *how*. Drawn as a grid of cores with NoC edges.

```
   ── SPMD overlay ───────────────────────────────────────────────────────
                                                                         
   work_units = <iteration space>        e.g. MoE: E_LOCAL=64 experts
   split:  split_work_to_cores(grid, units)   ← the actual tt-metal helper
   cores:  64 of 110 active   ◄── UTILISATION ANNOTATION (the lever!)
                                                                         
        DRAM ──(stream)──►  ┌────┬────┬────┬────┐                          
                            │ c0 │ c1 │ c2 │ .. │   each cell = a full      
        DRAM ──(mcast h)──► ├────┼────┼────┼────┤   Layer-1 actor graph     
            broadcast       │ .. │    │    │    │                          
                            └────┴────┴────┴────┘                          
                                  │ NoC reduce (tree / ring)  ◄── NoC EDGE  
                                  ▼                                         
                            [routed_acc]  →  DRAM                           
```

**Annotations:**

- **Partition node:** the iteration space and the split policy
  (`split_work_to_cores`, as in the GDN program factory). Carries the
  **utilisation number** — active cores / 110 — which is the headline lever.
- **NoC edges** come in three flavours, each drawn distinctly:
  - `DRAM→L1 stream` (per-core independent reads),
  - `mcast` (one core / DRAM broadcasts to many — e.g. broadcasting `h`),
  - `gather / reduce` (many cores → one; the cross-core sum the MoE kernel
    needs but doesn't have yet).
  Each NoC edge is annotated with **bytes moved** so the roofline is on the
  picture.

### 4.4 The binding overlay = the "schedule" (Halide-style)

Layer 0 is invariant (the math). The *mapping* is the set of choices that turn
Layer 0 into Layers 1+2 — this is the Halide schedule made concrete for Tensix:

| Schedule decision | TDG annotation | tt-metal mechanism |
|---|---|---|
| Which loop is SPMD | partition node on Layer 2 | `CoreRangeSet` + `split_work_to_cores` |
| What stays resident in L1 | "stationary" tag on a CB (Eyeriss vocab) | `cb_wait_front(cb_h, …)` once, never pop until end (MoE `h`) |
| What streams | CB with `d=2` double-buffer, looped fill | reader loop + double-buffered CB |
| Reader/compute overlap | parallel actors + CB depth ≥ 2 | NCRISC vs TRISC, `d≥2` |
| FPU vs SFPU per op | engine tag on the MATH stage | `matmul_tiles` vs `*_tile` SFPU calls |
| Inter-core comm | NoC edge type (stream/mcast/reduce) | `noc_async_read` vs mcast vs CCL/ring |

**Compositionality.** Because Layer 0 is a plain tile-op graph and the binding is
a separate overlay, kernels compose two ways:

1. **Horizontally (fusion):** concatenate two Layer-0 graphs and *delete the DRAM
   round-trip between them* — the edge that was "PACK→CB_out→DRAM→CB_in→UNPACK"
   collapses to "PACK→CB→UNPACK", staying in L1. This is literally what the MoE
   kernel does to the gate_up→silu→down chain, and what owned_decay_gate did to
   its 10-op chain. The representation makes the deletable edges obvious.
2. **Vertically (re-binding):** keep Layer 0 fixed and swap the binding overlay —
   e.g. take the single-core MoE G1a graph and re-partition Layer 2 from
   "1 core, loop over E" to "64 cores, 1 expert each" without touching the math.
   That is exactly the G1a→G2 step in our build plan, expressed as an overlay
   diff.

### 4.5 What the rendered (richer) version looks like

On paper / ASCII it's the diagrams above. A tool-rendered version would be a
**layered node-link diagram** (think tt-explorer / Netron, but annotated):

- Layer-1 actor graph as a horizontal swimlane per RISC (NCRISC / TRISC /
  BRISC), so reader/compute/writer overlap is a glance.
- CB channels as pipes with a **fill gauge** = depth, coloured red if
  `depth < max(cb_wait_front count)` (a static deadlock flag).
- Layer-2 as a heatmap over the physical 2D core grid: green = active, grey =
  idle (so the "78 idle cores" jumps out), with NoC edges as arrows whose
  thickness = bytes.
- Each node carries a tooltip with the Maestro/Timeloop-style cost: tiles in/out,
  bytes, BW% of 404 GB/s, FLOP% of 9.25 TFLOPS.

---

### 4.6 Worked example A — tiled matmul (the canonical primitive)

Algorithm (Layer 0): `C[m,n] = Σ_k A[m,k]·B[k,n]`, tile-rate = `Kt` tiles per
output tile. Binding for the multi-core reuse matmul (per tt-metal
`matmul_multi_core_optimizations/data_reuse`):

```
  LAYER 0 (algorithm)            LAYER 1 (per-core actor graph)
  ┌────────────┐                 NCRISC: read A-row block, B-col block
  │ matmul_tiles│                   │ r=in0_block_w  d=2 (double-buffer)
  │  (FPU)      │                   ▼
  │  acc over Kt│            CB_A ══════╗
  └─────┬──────┘            CB_B ══════╣═► UNPACK→[FPU acc Kt]→PACK ═► CB_interm (d, c_24)
        │                                                              │ partials reused
        ▼                                                              ▼  in-L1 (no DRAM)
     C tile                                              BRISC: write C subblock → DRAM

  LAYER 2 (SPMD overlay)
   work = Mt×Nt output tiles ;  split_work_to_cores(grid, Mt*Nt)
   reuse: A-row mcast across a core column, B-col mcast across a core row (NoC mcast edges)
   utilisation: ideally all 110 cores; annotate actual from get_large_matmul_params
```

What the TDG exposes that intuition hides: (a) the **`CB_interm` (c_24) is the
"output-stationary" choice** — partials never hit DRAM (Eyeriss vocab, tt-metal
data-reuse doc); (b) the **two mcast NoC edges** (A across a column, B across a
row) are the reuse mechanism and their byte cost is on the picture; (c) **decode
matmuls have M=1**, i.e. `Mt=1`, so the Mt×Nt work space collapses and naive
core-splitting starves cores — *this is the root cause* the scoping doc found by
hand ("the matmul is … core-utilisation bound", `35b_moe_ffn_kernel_scoping.md`).
The representation would have shown a 1-row-tall Layer-2 grid immediately.

### 4.7 Worked example B — fused decay/gate (the user's shipped kernel)

This is `owned_decay_gate` (`…/compute/qwen36_decay_gate_decode_owned.cpp`),
verbatim structure. It's the cleanest demonstration of **MIMD-pipeline +
horizontal fusion**.

```
  LAYER 0 (the 5-op DeltaNet decay/gate chain — from the kernel header comment)

   a ──┐                                          A_log ──► exp ──► neg ──┐
   dt_bias─► add ──► softplus(SFPU) ──► [softplus_a]                       │
                                            │                              ▼
                                            └────────────► mul(SFPU) ──► g ──► exp(SFPU) ──► decay_out
   b ───────────────────────────► sigmoid(SFPU) ───────────────────────────────────────► beta_out

  LAYER 1 (per-core actor graph — ALL on ONE core, single tile per CB)

   NCRISC reader: 4 reads → CB_A, CB_B, CB_DT_BIAS, CB_A_LOG   (r=1, d=2 each)
        │
        ▼   TRISC compute pipe, all SFPU except the add/mul (eltwise):
   CB_A,CB_DT_BIAS ═► UNPACK→[add]→PACK ═► CB_SOFTPLUS ═► …(softplus)… ═╗
   CB_A_LOG        ═► UNPACK→[exp,neg]→PACK ═► CB_NEG_EXP_A ═══════════╣═► mul → CB_G → exp → CB_DECAY_OUT
   CB_B            ═► UNPACK→[sigmoid]→PACK ═══════════════════════════════════════► CB_BETA_OUT
        │
        ▼   BRISC writer: CB_DECAY_OUT, CB_BETA_OUT → DRAM

  LAYER 2 (SPMD overlay)
   work = 1 block ;  CoreRangeSet = {(0,0)}  ←  1 of 110 cores.  UTILISATION = 0.9% (!!)
```

What the TDG exposes:

- **The fusion win is visual:** Layer 0 is a 5-node graph; the production path was
  10 ttnn ops each with its own DRAM round-trip. Every intermediate
  (`softplus`, `neg_exp_A`, `g`) is an *internal CB edge*, never DRAM. The
  deleted DRAM edges *are* the +2.5% tok/s win (`feedback_owned_decay_gate_shipped`).
- **It's all SFPU, no FPU.** The MATH-engine tags show the matrix engine is idle
  the whole kernel — fine here (this is vector work), but the representation makes
  "am I using the right engine" a glance.
- **Layer 2 screams 0.9% utilisation.** For a 1-tile-per-call decode op that may
  be unavoidable per-call, but the picture invites the obvious question the
  intuition path skips: *could 12 DeltaNet heads' decay/gates be one SPMD launch
  across 12 cores instead of 12 serial single-core launches?*

---

## 5. Worked example C — `qwen36_moe_ffn_decode_owned` (where it pays off most)

This is the live kernel and the live deadlock-bug story. Layer 0 / Layer 1 from
`…/compute/qwen36_moe_ffn_decode_owned.cpp` and the reader; Layer 2 choices from
`35b_moe_ffn_kernel_scoping.md`.

```
  LAYER 0 (per expert e, the MoE FFN math)

   h ─┬─► matmul(W1[e]) ─► gate_up ─┬─slice gate─► silu(SFPU) ─┐
      │     (FPU, Kt=hidden_tiles)   └─slice up──────────────► mul ─► mid ─► matmul(W2[e]) ─► eo
   h stays resident ───────────────────────────────────────────────(FPU)        │
                                                          rw[e] ─► scale ─► +acc ─► CB_PARTIAL
                                            after all E experts: CB_PARTIAL ─► out

  LAYER 1 (per-core actor graph — current G1a, ONE core)

   NCRISC reader (note the ORDER, it's load-bearing):
        phase1: stream h → CB_H  (resident: cb_wait_front once, pop at very end)
        phase2 per e:  rw[e] → CB_RW   ◄── MUST be pushed BEFORE W2 (initial-token!)
                       W1[e] → CB_W1   (r=hidden_tiles)
                       W2[e] → CB_W2   (r=mid_tiles)
   TRISC compute: matmul(FPU) → SFPU silu → mul → matmul(FPU) → scale → add-accumulate
   BRISC writer: CB_OUT → DRAM

  LAYER 2 (SPMD overlay — current vs target)
   CURRENT (G1a):  work=1 block, 1 core.        UTILISATION = 0.9%.  26.4 ms/call @ prod shape.
   TARGET  (G2):   work=E_LOCAL=64 experts, split_work_to_cores → 64 cores.  UTIL = 58%.
                   + NoC REDUCE edge: 64 per-expert [1,HIDDEN] partials → tree-sum → routed.
   TARGET  (G3):   mcast h to all cores once (NoC mcast edge) + partition (expert × out_col).
```

What the representation exposes — and would have caught:

1. **The two deadlocks were TDG-checkable static errors.** From
   `35b_moe_ffn_kernel_perf_deferrals.md`:
   - *Bug 1 (CB depth):* `cb_wait_front(cb_w1, hidden_tiles)` needs CB_W1
     `d ≥ hidden_tiles`, but `d=2`. In TDG terms: the consumer's per-firing rate
     `r=hidden_tiles` **exceeds the channel depth** `d=2` — the red-pipe static
     flag in §4.5. Pure SDF: depth must be ≥ the consume rate. A glance at the
     annotated channel catches it; a toy shape where `hidden_tiles=2` hid it.
   - *Bug 2 (producer ordering):* reader pushed `W1→W2→rw`, but compute waits on
     `CB_RW` *before* draining `CB_W2`. In TDG terms: `CB_RW` needs an **initial
     token before the consumer's first firing** in the eo block; drawing the
     black-dot initial-token annotation forces you to place `rw` first — which is
     exactly the fix the reader comment now documents ("Reader must produce rw
     *before* W2"). This is the textbook SDF initial-token / KPN ordering result.
2. **The utilisation lever is the headline.** Layer 2 says 0.9% now → 58% at G2.
   The whole G0→G4 ladder is a sequence of **Layer-2 overlay diffs on a fixed
   Layer-0 graph** — which is precisely the vertical-composition story of §4.4,
   and explains why we can stage it without re-deriving the math.
3. **The missing primitive is a NoC edge type.** Candidates A/B/C in the scoping
   doc differ *only* in their Layer-2 NoC-reduce pattern (per-output-tile reduce
   vs one ring-reduce vs mcast+tree-reduce). The scoping doc calls the cross-core
   sum "the hard part … owned-GDN never does cross-core sum." In TDG that's
   "we have stream and mcast edges in our library but not a reduce edge" — a
   concrete, nameable gap, not a vibe.
4. **It also predicts the modest payoff honestly.** The roofline annotations on
   the W1/W2 stream edges (256 MB + 128 MB/call, 26% of 404 GB/s) say the op is
   *neither* BW- nor FLOP-bound, so the win must come from the utilisation lever
   (Layer 2), not from precision (confirming the empirically-rejected HiFi2
   experiment, HANDOFF.md). The representation routes you to the right lever.

---

## 6. Honest assessment, open questions, what to prototype

### Is this novel?

**Partly, and only at the synthesis/UX level — not theoretically.** Every
ingredient is borrowed: algorithm⊥schedule (Halide), channel rate/depth/initial-
token + deadlock-freedom (SDF/KPN, Lee & Messerschmitt 1987), per-level
bytes/BW/util cost (Timeloop/Maestro), stationarity vocabulary (Eyeriss). The
problem statement is essentially the TL paper's
(arXiv 2512.22168) verbatim — *distribute tile instances across cores, exploit
NoC + distributed memory for reuse* — and TL even lowers to TT-Metalium. tt-mlir
already has TTIR/TTKernel dialects doing this automatically.

The genuinely missing thing is a **human-facing, visual design notation for
hand-written LLK kernels** that puts the four levers (SPMD util, MIMD pipeline,
NoC edge type, CB depth/initial-tokens) on one diagram, where you *retain manual
control* (TL/tt-mlir take it away). So TDG is best positioned as **a design and
review notation that sits one level above LLK and one level below tt-mlir** — the
napkin you draw before writing the program factory, and the artifact you diff in
code review. Not a compiler. That framing is what keeps it from being a re-skin.

### Concrete things to prototype next (cheap, no device)

1. **A TDG-from-source linter (static, host-only).** Parse a program factory +
   compute kernel and emit the annotated graph + the two static checks that would
   have caught both MoE deadlocks: (a) `depth(cb) ≥ max(cb_wait_front count)`;
   (b) initial-token ordering — every CB that a consumer waits on before its
   first push must be produced first by the reader. These are mechanical AST
   checks over `cb_*` calls and the `CircularBufferConfig` depths. **Highest ROI:
   it directly attacks the failure mode we already hit twice.**
2. **A utilisation annotator.** From the `CoreRangeSet` / `split_work_to_cores`
   args, compute active-cores/110 and the per-edge byte/BW% roofline (we already
   do this math by hand in the scoping docs) and render Layer 2 as the core-grid
   heatmap. Turns the "78 idle cores" finding into an automatic output.
3. **One re-binding case study on paper:** express owned_decay_gate's
   single-core launch and a hypothetical 12-head SPMD launch as two Layer-2
   overlays on the same Layer-0 graph, and predict the trace delta with the
   Maestro-style cost model — *then* (later, on hardware) check the prediction.
   This tests whether the representation actually predicts, per the project's
   no-handwaving rule.

### Open questions (flagged, not resolved)

- **Granularity of a "tile-op" node.** Is `matmul_reduce` (a Kt-step accumulation
  loop) one node or Kt nodes? Probably one node with a `rate=Kt` self-loop, but
  this needs to be pinned for the linter.
- **Does the SDF abstraction hold for data-dependent control?** The MoE router's
  top-k makes *which* experts fire data-dependent; SDF is static-rate. We likely
  need the "boolean/dynamic dataflow" (BDF) extension or to treat routing as a
  fixed E_LOCAL with masking (which is what the batched path already does).
- **How does Layer 2 model the 2-NoC torus precisely?** ~9 cyc/hop and two
  unidirectional NoCs (`research/01`) mean mcast/reduce costs depend on physical
  core placement. v1 can use a placement-agnostic byte model; a v2 could fold in
  hop counts à la Maestro NoC-traversal stats.
- **Relationship to tt-mlir.** Could the linter consume TTKernel-dialect output
  instead of parsing C++? That would make TDG a *view* over the existing IR
  rather than a separate parser — worth checking before building a parser.

---

## Sources

Repo / project (paths relative to working dir):
- `research/01_tenstorrent_hardware.md` — Tensix 5-RISC split, NoC torus, L1, fidelity.
- `research/35b_moe_ffn_kernel_scoping.md` — MoE roofline (26% BW / 1.6% FLOP), candidates A–D, M=1 utilisation.
- `research/35b_moe_ffn_kernel_perf_deferrals.md` — the two large-shape deadlocks (CB depth; producer ordering).
- `experiments/owned_ops/qwen36_decay_gate_decode_owned/device/kernels/{compute,dataflow}/…` — worked example B.
- `experiments/owned_ops/qwen36_moe_ffn_decode_owned/device/kernels/{compute,dataflow}/…` — worked example C.
- `experiments/owned_ops/qwen36_gdn_decode_owned/device/qwen36_gdn_decode_owned_program_factory.cpp` — `split_work_to_cores` SPMD pattern.
- MEMORY: `feedback_p150_memory_bandwidth_measured` (404 GB/s, 110 cores, 1.39 MiB L1); `feedback_owned_decay_gate_shipped`; HANDOFF.md (HiFi2 rejected = not math-bound).

External:
- tt-metal METALIUM_GUIDE + multi-core matmul data-reuse doc: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md , https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_multi_core_optimizations/data_reuse.html
- Corsix "Tenstorrent Wormhole" series: https://www.corsix.org/content/tt-wh-part1
- clehaxze "Programming Tenstorrent processors": https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi
- Halide (algorithm⊥schedule): https://halide-lang.org/ , https://halide-lang.org/tutorials/tutorial_lesson_08_scheduling_2.html
- Maestro (data-centric dataflow, spatial/temporal loops): https://arxiv.org/abs/1805.02566 , https://maestro.ece.gatech.edu/
- Timeloop + Accelergy: https://accelergy.mit.edu/timeloop.pdf , https://accelergy.mit.edu/tutorial.html
- SDF / Kahn Process Networks: Lee & Messerschmitt 1987; https://ptolemy.berkeley.edu/projects/embedded/eecsx44/lectures/Spring2013/dataflow.pdf
- Eyeriss row-stationary dataflow taxonomy (Chen et al.) — DNN dataflow classification.
- tt-mlir (TTIR/TTNN/TTKernel dialects): https://github.com/tenstorrent/tt-mlir , https://docs.tenstorrent.com/tt-mlir/
- TL (Triton→TT-Metalium tile-language compiler, closest prior art): https://arxiv.org/abs/2512.22168
