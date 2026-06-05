# 66. Anatomy of a Blackhole kernel — the dataflow / hardware map

**Audience**: you've watched the 35B owned-GDN kernel get built, and
you're about to fork it for Mamba2 SSD. You want a sharper mental
model of *how a Tensix tile actually runs* before you stare at the
LLK source.

> **Source files cited**: `experiments/owned_ops/qwen36_gdn_decode_owned/`
> (our shipped GDN), `experiments/owned_ops/qwen36_conv1d_decode_owned/`
> (our shipped per-step Conv1d), the
> [tt-metal docs](https://docs.tenstorrent.com/tt-metalium/),
> wiki §[06](06_blackhole_first_contact.md),
> wiki §[08](08_memory_hierarchy.md),
> wiki §[09](09_sharded_memory_and_dispatch.md).

------------------------------------------------------------------------

## 1. The Blackhole physical layout (one paragraph refresh)

A P150 has a grid of **Tensix cores** (110 user-allocatable after the
firmware-19.5.0 silent downgrade — see `feedback_p150_firmware_core_check`).
Each Tensix is a small heterogeneous compute island:

- **3 RISC-V cores** for orchestration: `BRISC` (handles the first NoC),
  `NCRISC` (handles the second NoC), and **3 TRISC cores** (TRISC0/1/2)
  that drive the **vector-matrix compute engine** ("the math unit").
- **1.39 MiB L1 SRAM**, fast (~1 ns access), local to this Tensix.
- **NoC ports** (two, named NOC_0 / NOC_1) connecting to all other
  Tensix and to DRAM controllers.

Off-chip: **8 × 4 GB GDDR6 channels** = ~32 GB DRAM with **404 GB/s
measured stream bandwidth** (see `feedback_p150_memory_bandwidth_measured`).

So when we say "the kernel runs on Tensix N", we mean five RISC-V
cores + 1.39 MiB L1 + two NoC links + the math unit, all on one tile.

------------------------------------------------------------------------

## 2. The kernel = three programs, three roles

A custom ttnn op like `qwen36_gdn_decode_owned` ships
[3 LLK kernel files](../experiments/owned_ops/qwen36_gdn_decode_owned/device/kernels/):

```
device/kernels/
├── compute/qwen36_gdn_decode_owned.cpp      → runs on TRISC{0,1,2}
└── dataflow/
    ├── reader_qwen36_gdn_decode_owned.cpp   → runs on BRISC (or NCRISC)
    └── writer_qwen36_gdn_decode_owned.cpp   → runs on the other NoC core
```

These three programs run **concurrently on the same Tensix tile**,
producing/consuming **circular buffers** (CBs) that live in that
tile's L1.

```
                    +---------------------- L1 SRAM (1.39 MiB) ----------------------+
DRAM ─NoC→ [READER] ┤  CB_IN_0   CB_IN_1  …  CB_INTER_0  CB_INTER_1  …  CB_OUT_0  ├─[WRITER] →NoC→ DRAM
                    │     ↓        ↓                                        ↑       │
                    │     └─→ [COMPUTE on TRISCs: tile math, multiply, exp, etc.] ──┘
                    +-----------------------------------------------------------------+
```

The **reader** fetches input tiles from DRAM into L1 CBs. The
**compute** kernel consumes input CBs, runs the math on the
vector-matrix engine, produces output CBs. The **writer** drains
output CBs back to DRAM. All three execute as a producer-consumer
pipeline against the CBs, fully overlapped — that's how a Tensix
hides DRAM latency.

(There's an SPMD aspect to this — N copies of the same kernel run on
N different Tensix simultaneously, each handling its slice of work.
See §6.)

------------------------------------------------------------------------

## 3. The tile (32×32) is the unit of everything

The math unit operates on **fixed 32×32 tiles**. Always. Every value
in L1 lives in a tile; every compute op takes tiles in and produces
tiles out.

Sizes in bytes:
- `bf16 tile`: 32 × 32 × 2 = **2 KiB**
- `fp32 tile`: 32 × 32 × 4 = **4 KiB**

This is why every `[B, N, K]` tensor in our code is implicitly
**padded to multiples of 32 along each axis**, then "tiled" into a
`[B, ceil(N/32), ceil(K/32)]` sequence of 2-KiB or 4-KiB blobs. The
**circular buffer** then holds a stream of these blobs.

CBs are allocated with a fixed page size = 1 tile and a fixed
capacity (1, 2, 4, … tiles). A CB with capacity = 2 tiles supports
double-buffering: while the writer fills tile N, the consumer is
processing tile N-1.

For Mamba2 SSD, an interesting consequence: `ssm_state` per head is
`[head_dim, ssm_state] = [64, 128]`. That's exactly **2 tiles × 4
tiles = 8 tiles per head**. We can't subdivide a tile, so the SPMD
unit can't be finer than "one tile of state".

------------------------------------------------------------------------

## 4. Where the recurrent state actually lives

This is the question you asked: *"I guess you want to keep the state
resident in L1 and stream inputs?"* Mostly yes, with one important
nuance.

### Within ONE kernel invocation (= ONE decode step):

- The state tile is **read from DRAM into a CB** by the reader.
- It stays in L1 the entire kernel — compute reads from the input
  CB, produces an updated state in an output CB.
- The writer streams the updated state from the output CB back to
  DRAM.

So the state **does** travel DRAM ↔ L1 once per decode step. It's
NOT "resident in L1 forever"; the CBs are freed at kernel end.

### Between decode steps:

- The state lives in DRAM.
- The next decode step's kernel invocation re-reads the same
  DRAM address into a fresh CB.
- This is **fine** because Mamba2's per-step compute (~16K FMAs per
  head) is bigger than the state DMA (8 tiles × 4 KiB × 2-way RW =
  64 KiB / head / step). At 404 GB/s DRAM, that's 0.16 µs / head /
  step — a rounding error compared to the ~100 µs / token / layer
  budget.

### Why not keep it in L1 across steps?

Two reasons:
1. **L1 is shared** across kernels. Other ttnn ops in the forward
   (norm, matmul, attention) reuse the same Tensix cores; their CBs
   would clobber persistent L1.
2. **Persistent L1 state** is a feature you'd opt into with explicit
   "sharded tensor" layouts — see `[[ttnn-shard-1d-vs-2d]]`. Possible
   but complicates the program factory; the GDN kernel didn't need
   it, and Mamba2 won't either at v0.

### Implication for owned kernel design

The CB layout in `qwen36_gdn_decode_owned/device/qwen36_gdn_decode_owned_program_factory.cpp`
allocates `CB_STATE_IN`, `CB_STATE_OUT`, etc. with capacities sized
to **the full state for one Tensix's assigned heads**. The state is
allocated, used, deallocated within one kernel invocation. The
**DRAM tensor** is what survives across steps.

------------------------------------------------------------------------

## 5. Tensor sizes for Mamba2 — does it all fit?

For Nemotron-3 Nano's Mamba2 shapes (per the architecture brief §4.3):

```
num_heads = 64           # split across cores
head_dim  = 64           # within a head: 2 tiles wide
ssm_state = 128          # within a head: 4 tiles wide
n_groups  = 8            # B / C broadcast factor
```

Per-(batch, head) L1 footprint, fp32 state:

| Tensor | Shape | Tiles | Bytes |
|---|---|---|---|
| `x` | `[head_dim=64]` | 2 (cols, 1 row of broadcasted scalar) | 4 KiB |
| `z` | `[head_dim=64]` | 2 | 4 KiB |
| `dt`, `dt_bias` | `[1]` | 1 | 4 KiB |
| `A_log`, `D` | `[1]` | 1 each | 4 KiB each |
| `B`, `C` | `[ssm_state=128]` | 4 each | 16 KiB each |
| `ssm_state` (fp32 in/out) | `[64, 128]` | 8 × 2 (double-buffer) | 64 KiB |
| Compute intermediates (decay, dt_B, input_contrib) | ~8 tiles | 32 KiB |
| **Total per (batch, head)** | | | **~160 KiB** |

L1 budget per Tensix is ~1408 KiB (1.39 MiB minus the kernel code
~4-8 KiB and some reserved areas). So one Tensix can fit **~8 heads
in L1 simultaneously** — but each head's work is independent so we
typically assign **1 head per core, 64 cores, 46 idle** at B=1.

At B=8 (Phase 1 CB target), that's 8 batches × 64 heads = 512 work
units, packed into ~110 cores → roughly 5 (batch, head) pairs per
core. Per-core L1: 5 × 160 KiB = 800 KiB. Still fits, with ~600 KiB
slack for the compute kernel + double-buffering headroom.

### Why "1 head per core" beats "8 heads per core" at B=1

Per-Tensix math throughput is ~16 GFLOPs/s (rough order). Per
(batch, head) work: 16K FMAs. So one head's math takes ~1 µs of pure
compute. The L1 ↔ DRAM transfer is ~0.2 µs. If you pack 8 heads onto
one core, that's 8 µs of compute serial — vs running 8 cores in
parallel for 1 µs. Sharding wins as long as you have idle cores.

The "sweep over configurations" question (your phrasing) IS the
right framing here. In G2 we'll try (1 head/core, 2 heads/core, 4
heads/core) and pick the one that gives the best
clock-cycles-per-token. The 35B benchmark file
(`experiments/owned_ops/qwen36_gdn_decode_owned/benchmark_qwen36_gdn_decode_owned.py`)
is the template — fork it for Mamba2 in G2.

------------------------------------------------------------------------

## 6. SPMD partition — block = (batch, head)

Multiple Tensix cores run the **same** compute kernel program in
parallel, each on a different **block** of work. The block scheme
for our owned kernels:

| Kernel | Block | Cores at B=1 |
|---|---|---|
| `qwen36_gdn_decode_owned` (35B) | `(slot, value_tile)` — slots from batch, value_tile = head_dim chunk | 32 slots × 4 value_tiles = 128 blocks (more than cores at B=1, packed) |
| `qwen36_conv1d_decode_owned` (35B) | `(D_chunk)` — channel-axis chunk | D=4096, ~32 channels/core × ~128 cores |
| `nemotron3_mamba2_decode_owned` (proposed, G1+) | `(batch, head)` | 1 × 64 = 64 blocks at B=1 → 1 head/core, 46 idle |

The program factory's job is to (a) decide how many blocks total, (b)
assign blocks to cores, (c) pass per-block runtime args ("you're
block 23 → batch 0, head 23 → DRAM address such-and-such").

The 35B `qwen36_gdn_decode_owned_program_factory.cpp:create()`
function shows the pattern: it calls `split_work_to_cores` (a tt-metal
helper) which returns the (cores × blocks_per_core) partitioning,
then iterates and emits per-core runtime args via
`SetRuntimeArgs(program, kernel, core, args)`.

------------------------------------------------------------------------

## 7. Cross-tile communication — is there any?

For the SSD recursion proper, **no**: each (batch, head) is fully
independent. The state update reads only the head's own state, the
output reduce sums only over the head's own ssm_state axis. Pure
embarrassingly-parallel.

The only inter-core data movement is:

1. **Inputs from DRAM** — every core reads the head-specific slice of
   `x`, `dt`, etc. from DRAM. These are independent DRAM reads;
   they don't go core-to-core via NoC.
2. **Per-group B/C broadcast** — `B[g]` is shared by 8 heads (one
   group). Two strategies:
   - **Replicate B in DRAM**: cheap host-side cost, each core reads
     its head's B from a per-head DRAM offset. **Wastes DRAM
     bandwidth** (8× the B traffic for the same content).
   - **Multicast B via NoC**: one Tensix reads B from DRAM, multicasts
     it to the 7 other cores in the group via NoC. Saves DRAM, adds
     NoC traffic. ~Free latency-wise because NoC bandwidth is much
     higher than DRAM.
   For v0 G1 ship the **replicate** strategy (simpler). At G2+ measure
   and revisit. Multicast is an easy optimisation if needed.

3. **Output gather** — the kernel returns `y[B, num_heads, head_dim]`.
   Each core writes its `(batch, head)` slice to a different DRAM
   address. **No cross-core gather** — DRAM does the assembly.

So **for Mamba2 SSD the compute kernel is core-local; only DRAM ↔
core dataflow exists, no core ↔ core except optional B/C
multicast**.

This is in stark contrast to e.g. **all-reduce** in our TP path,
where every core's residual stream gets summed across all chips and
broadcast back. That uses NoC + inter-chip fabric heavily. The CCL
ops (`ttnn.all_reduce`, `ttnn.all_gather`) implement those
patterns.

### Lessons we have on cross-tile / cross-chip communication

From the project memory + research:
- `[[feedback-p1-num-links-2-shipped]]` — using 2 NoC links for
  `all_reduce` gives +1.65% perf vs default 1 link on (1,4) mesh.
  Cheap tunable.
- `[[feedback-async-ccl-negative]]` — `all_reduce_async` adds +4%
  setup; on serial residual streams there's no overlap budget to
  amortise, so async LOSES. Use sync.
- `[[reference-multi-chip-opt-menu]]` + v2 — 14 + 11 candidates for
  the TP fabric. `all_gather_concat` and `llama_rs_matmul` are the
  top new candidates not yet shipped.
- `[[reference-multi-chip-web-research]]` — tt-metal issue #26252 /
  #33147 have ongoing CCL kernel work; ASPLOS TT-bench paper
  reports a 70% overlap ceiling on inter-chip work.

For Mamba2 SSD specifically, we don't need cross-chip — each chip
handles its own slice of (heads, batches) and there's no
cross-chip reduce within the SSD step. The cross-chip work happens
in the attention layers (KV all-gather) and lm_head (vocab-shard
gather).

------------------------------------------------------------------------

## 8. DRAM ↔ L1 dataflow: stream vs sharded

ttnn supports two main tensor layouts for DRAM:

### Interleaved (default)

Tiles are striped across **all 8 DRAM channels** round-robin. A
tile's address is `(tile_id % 8, tile_id // 8 × tile_bytes)`. The
reader kernel uses a "tile accessor" that knows this striping and
issues per-tile DRAM reads. Each tile read is ~250 ns latency,
hidden by double-buffering across many tiles in flight.

Strengths: load-balances all 8 channels automatically; works for
any shape.

Weaknesses: each tile read is its own NoC transaction → higher
per-tile overhead. For very small tensors (<8 tiles) you don't
saturate the 404 GB/s.

### Sharded

The tensor is explicitly partitioned across a chosen set of L1 (or
DRAM) shards. Each shard holds a contiguous slice; one core reads
from a specific shard with explicit address arithmetic. Often used
for **persistent L1 tensors** (weights repeated every step).

Strengths: deterministic placement, low overhead for hot data.

Weaknesses: shape constraints (must divide evenly), more host-side
glue, easy to mis-shard (see `[[feedback-ttnn-shard-1d-vs-2d]]`).

### Mamba2 SSD: which layout?

- **Per-step inputs (x, dt, B, C)**: interleaved DRAM is fine —
  they're tiny (~tens of tiles per layer) and read once per token.
- **`ssm_state` (fp32, the persistent recurrent state)**:
  **interleaved DRAM**. Each (batch, head) reads/writes its own
  8-tile state slice. Sharding the state to L1 would persist across
  kernel calls (so faster) BUT cap our per-slot batch B at ~110
  cores × 8 heads × 1 batch / 8 heads = 110 / 8 = ~14 simultaneous
  slots. We want B=32 in CB. So DRAM-resident state, accept the
  64 KiB / token / head transfer cost (negligible at 404 GB/s).

------------------------------------------------------------------------

## 9. Putting it together: the per-decode-step lifecycle

For one Mamba2 layer, one decode step, B=1, one Tensix handling
head h:

```
t = 0 (kernel launch on Tensix h)
   READER:  fetch x[h], dt[h], B[h//8], C[h//8], A_log[h],
            dt_bias[h], D[h], ssm_state[h] from DRAM into L1 CBs
   COMPUTE: dt_eff = clamp(softplus(dt + dt_bias), …)
            A      = -exp(A_log)
            decay  = exp(dt_eff * A)
            dt_B   = dt_eff * B[h//8]           (broadcast)
            for d_tile in [0, 1]:               (head_dim = 2 tiles)
                state_new = decay * state_old + dt_B * x[d_tile]
                y[d_tile] = sum_s(C * state_new) + D * x[d_tile]
                → write state_new to CB_STATE_OUT
                → write y[d_tile] to CB_Y
   WRITER:  drain CB_STATE_OUT to DRAM ssm_state[h] (overwrite)
            drain CB_Y to DRAM y[batch, h]
t = ~5 µs (kernel done; CBs deallocated; L1 freed for next op)
```

At step t+1, the same Tensix re-reads `ssm_state[h]` from DRAM (now
updated) and runs again with new inputs. Across 23 Mamba layers per
token, this kernel runs 23 times (with 23 different state tensors).

------------------------------------------------------------------------

## 10. Why GDN and Mamba2 are similar at the kernel level

Both are **recurrent linear-attention-style mixers**: per-step, read
a fixed-size recurrent state, apply input-dependent updates, write
state back, emit a per-token output. The kernel pattern is:

| Stage | GDN | Mamba2 SSD |
|---|---|---|
| Read inputs | q, k, v, state, alpha, beta | x, z, dt, B, C, state, A_log, dt_bias, D |
| Per-(slot/head) prep | state_scaled = α·state, pred = k @ state_scaled | dt_eff = clamp(softplus(dt+dt_bias)); A = -exp(A_log); decay = exp(dt_eff·A); dt_B = dt_eff·B |
| State update | delta = β·(v - pred); state_new = state_scaled + outer(k, delta) | state_new = decay·state + outer(dt_B, x) |
| Output | out = q @ state_new | y = C @ state_new + D·x |
| Write back | state, out | state, y |

The **stage shapes are identical**: 5 phases per kernel invocation,
~5 input tensors, in/out state, output. The math differs in stages
2 and 3 (selectivity mechanism). But the **CB layout, the LLK call
pattern, the SPMD partition, the DRAM access patterns** all port
1:1 from GDN to Mamba2. That's why §3a of the Nemotron bringup plan
estimates only ~500 net-new LOC for the compute kernel — most of
the 1.4k LOC GDN scaffolding ports verbatim.

------------------------------------------------------------------------

## 11. Tracy + tt-perf-report — how to actually measure

Once the kernel runs, the empirical questions ("did sharding help?
is the C-reduce or the state update the bottleneck?") have
authoritative answers via Tracy profiling on Blackhole:

```
bash experiments/utils/run_tracy_probe.sh <op-name>
# Output: per-op kernel time, host dispatch time, NoC traffic, DRAM bytes
```

`feedback_tracy_tp_breakdown` and
`feedback_p150_memory_bandwidth_measured` documents the harness. Use
it after every kernel change at G1..G4 to keep an honest perf log;
don't extrapolate from the math (`feedback_perf_no_handwaving` rule).

------------------------------------------------------------------------

## TL;DR for your specific questions

- **"Dataflow → hardware mapping for GDN and Mamba2 — similar?"**
  *Yes, identical pattern*: 3 LLK programs (reader / compute / writer)
  on each Tensix, CBs as the substrate, block = (batch_or_slot, head_or_value_tile).
- **"State resident in L1, stream inputs?"**
  *Resident **within** one kernel invocation* (i.e. one decode step
  per layer); evicted to DRAM at kernel end; re-streamed from DRAM at
  the next step. The compute → state DMA ratio is comfortably
  compute-bound, so this is fine.
- **"How do tensor sizes work out?"**
  Per (batch, head): ~160 KiB L1 footprint including double-buffer.
  Per-Tensix 1.39 MiB budget → fits ~8 (batch, head) pairs but we
  shard 1 per core to use parallelism.
- **"Cross-tile communication?"**
  *None* in the SSD recursion proper. Optional per-group B/C
  multicast at G2+; not needed for v0. Inter-chip CCL is a different
  story (attention + lm_head only, not Mamba2).
- **"SPMD partition over Tensix?"**
  Block = (batch, head). At B=1: 64 heads → 64 cores, 46 idle.
  At B=32: 32·64 = 2048 blocks packed across ~110 cores.
- **"Do we sweep configurations?"**
  Yes — fork `qwen36_gdn_decode_owned/benchmark_*.py` at G2 to try
  (1, 2, 4, 8) heads/core × (DRAM, sharded) × (B=1, B=8, B=32).
  Pick the Tracy winner.
- **"Cross-tile comm lessons?"**
  Inter-chip: `num_links=2` for `all_reduce` (+1.65%); avoid
  `all_reduce_async` on serial streams (LOSES). Intra-chip: prefer
  replicate-from-DRAM for v0 simplicity; revisit multicast at G2.
- **"DRAM streaming vs NoC?"**
  Per-step state IS streamed from DRAM via NoC each step. The CBs
  in L1 are the working set; DRAM is the long-lived store.

------------------------------------------------------------------------

## Related

- Architecture brief: `research/nemotron3_nano_architecture_brief.md`
- G1 kernel design: `research/mm7_g1_mamba2_kernel_design.md`
- Mamba math primer: `wiki/65_mamba_state_space_models.md`
- Memory hierarchy: `wiki/08_memory_hierarchy.md`
- Sharded memory: `wiki/09_sharded_memory_and_dispatch.md`
- Multi-trace warmup discipline: `wiki/62_metal_trace_blackhole.md`
