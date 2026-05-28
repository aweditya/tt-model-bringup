# Tenstorrent kernel design worksheet

A fill-before-you-code checklist for mapping an op onto a P150. Companion to
`kernel_dataflow_representation.md` (the TDG notation) — that doc is the *what*
(a visual dataflow graph); this is the *process* (the questions that catch idle
cores, CB deadlocks, producer-ordering bugs, and globally-starved kernels).

**Core principle:** design the per-core *streaming dataflow* (reader → compute →
writer + circular buffers) first; optimize the tile/SFPU math last. A TT kernel
is a 3-kernel pipeline, not "one tile function × N cores." The four levers are
SPMD core partition, intra-core MIMD pipeline, NoC traffic, and L1/CB residency
— fill all four below before writing a line of kernel code.

Empirical backing (this repo, 2026-05): every win/loss on the 27B CB path was a
dataflow decision, not tile math — the owned_gdn bug was a CB dual-consumer
race; the conv 5× came from killing tile padding; the matmuls were already fine
because weights amortize across the batch.

---

## STEP 0 — Do you even need a custom kernel?  (the cheapest win is no kernel)

Before any of the below, answer: **can existing ttnn ops express this without a
DRAM round-trip or a tile-padding tax?** If yes, reformulate at the op level and
stop — you may get the whole win for zero kernel code.

- [ ] Is a hot tensor padded? (last dim or a small inner dim < 32 → padded to a
      full tile = wasted traffic ∝ pad factor). **Reformulate to keep all hot
      tensors whole-tile.**
- [ ] Does the op re-stream a small persistent state from DRAM every call?
- [ ] Can a different *layout* (split a `[.,.,K]` into K columns; transpose a
      weight once) remove the waste using only existing ops?

WORKED EXAMPLE (conv1d, DNK-G4): the depthwise 3-tap conv built `conv_input
[B, C, K=4]`; in TILE layout K=4 padded to 32 → 8× waste, and conv was 72% of
the DeltaNet cost. **No kernel was needed** — reformulating as shift-accumulate
(`out = silu(Σ_k w_k·s_k)`, muls+adds on `[B,C]` tiles, state = K-1 separate
`[B,C]` columns) was 28.76× faster isolated and cut the CB step-time slope from
3.6 → 0.71 ms/seq (B=64: 208 → 593 tok/s). FAILURE MODE that proves the rule:
a *contained* version that kept the `[B,C,3]` state and sliced columns from it
REGRESSED — slicing/concat-ing a tile-padded `[.,.,K]` tensor reintroduces the
exact tax. **The persistent state itself must be padding-free, not just the
compute.**

Only if STEP 0 fails → proceed to a custom kernel below.

---

## STEP 1 — Classify the op by state/reuse (not by formula)

Pick the dominant class; it dictates the mapping:

| class | hard problem | example here |
|---|---|---|
| pure elementwise | none — trivially SPMD over tiles | silu, add, sigmoid |
| elementwise + small shift state | keep the tiny state padding-free | **conv1d (3-tap)** |
| reduction | reduce-tree / partial accumulation | rms_norm, softmax denom |
| block matmul | weight residency + core grid | in_proj/out_proj (weight-bound) |
| recurrent / linear-attention state | **state placement + scan partition** | **GatedDeltaNet recurrence** |
| gather/scatter | irregular NoC + page tables | paged KV update/read |

- [ ] Class: ____   Hard problem: ____

NOTE: classify **per sub-op**, not per layer — one DeltaNet layer spans three
classes (recurrence, conv, matmul), each with a different mapping. Use a
within-block profile (`cb_profile_dn.py`) to rank sub-ops and spend effort where
the time is (conv 72%, recurrence 9%, norms <1% here).

---

## STEP 2 — Unit of ownership (what does ONE core own for the whole op?)

Not "one tile." Start from the algorithm. Candidates: one (head × channel_tile),
one (expert × token_block), one channel-block, one recurrent-state shard, one
(batch, head) slot.

- [ ] One core owns: ____
- [ ] Active cores = ____ / 110. (If ≪ 110, ownership is too coarse/sparse —
      fix this BEFORE micro-optimizing a tile. Idle-core kernels are the #1
      failure here.)
- [ ] Tiles of useful work per core: ____

WORKED EXAMPLE (owned_gdn recurrence): SPMD unit = `(slot, value_tile)`. For
continuous batching, FOLD the batch into the slot dim (`[B,NV,K,V]→[1,B·NV,K,V]`)
so the kernel parallelises B·NV·value_tiles blocks across cores — each slot's
recurrence is independent. At B=32 that's 384·4 = 1536 blocks over 110 cores
(~14/core), good utilisation, no kernel change.

---

## STEP 3 — Make L1 residency explicit

- [ ] Persistent in L1 (don't re-stream): ____ (recurrent state, gates, decay,
      expert weights, partials)
- [ ] Streamed from DRAM each call: ____
- [ ] DRAM round-trips per call: ____ (minimise — most TT wins are avoided
      round-trips, not prettier SFPU loops)

CAVEAT for traced decode: L1 does NOT persist across kernel invocations, so a
per-step recurrent state must live in DRAM between steps and be re-streamed —
that round-trip is partly inherent. Quantify it (it sets a floor on the win)
before assuming a custom kernel beats an op-level reformulation.

---

## STEP 4 — CB schedule (steady-state rates)

For each core, write the pipeline as rates:
- [ ] reader produces N tiles of: ____ ; depth = ____
- [ ] compute consumes N, reloads intermediate/state tiles: ____ ; CBs: ____
- [ ] writer consumes M tiles of: ____ ; depth = ____
- [ ] **CB depths ≥ consume-rate?** (depth < tiles-consumed-per-firing → deadlock)
- [ ] **Single consumer per CB?** (two consumers on one CB = race — the
      owned_gdn high-slot bug: the output matmul AND the writer both popped
      `cb_state_out`; fix = a compute-owned intermediate CB for the output)
- [ ] **Initial tokens / producer order correct?** (missing initial token or
      producer-after-consumer = deadlock)

NoC edges (annotate bytes for the roofline):
- [ ] stream (DRAM↔L1): ____   mcast: ____   reduce/all-reduce: ____

---

## STEP 5 — Tile/SFPU math (write it LAST; keep it boring)

`cb_wait_front → unpack into Dst → FPU/SFPU → pack → cb_push_back`. The compute
kernel is the least interesting part of a fast TT kernel.

- [ ] math fidelity: ____ (HiFi4 default; HiFi2 only if measured equal)
- [ ] fp32_dest_acc: ____ (halves Dst space; needed for precision-sensitive
      accumulation — but watch the kernel-config size limit, below)

---

## SMELL TEST (compute before committing to code)

```
active_cores  ×  useful_tiles/core  ×  reuse_before_DRAM_spill  ×  CB_pipeline_overlap
```
If ANY term is near zero, tile-level optimization will NOT save the kernel.

- kdim conv failed on `useful_tiles/core ≈ 1/8` (4 real taps in a 32-row tile).
- a 27B kernel can fail on `active_cores` if ownership leaves most of 110 idle.

---

## Hard constraints learned on this hardware (P150, our build)

- **Kernel-config size limit (70656 B on TENSIX):** a compute kernel with too
  many code branches overflows it (adding a duplicate mode branch pushed
  owned_gdn to 77024 B → fatal). Prefer in-loop conditionals over duplicated
  branches; remove dead debug modes.
- **Device kernels JIT-compile from the `.cpp` at runtime** → editing a
  `device/kernels/.../*.cpp` takes effect on the next op call with NO ttnn
  rebuild. Editing the device-op / program-factory (`.cpp` compiled into the
  `.so`) DOES need a rebuild (and `cmake --install` does NOT update the venv
  `.so` — copy `_ttnn*.so` into `.venv/.../ttnn/`).
- **paged SDPA decode requires `num_cores_available ≥ B`** — size the SDPA
  program-config grid to cover B (B ≤ ~110).
- **ttnn.slice / reshape return VIEWS** — never `deallocate` a source while a
  view of it is live (view-decay); whether a slice is a view vs copy depends on
  tile alignment (tile-aligned offsets → view; sub-tile → materialised copy).
- **In-order command queue** lets you shift state in place with sequential
  `ttnn.copy`s (read-before-write within a step is safe) — used by the conv
  shift register and the DN ssm commit.

---

## How to use this

Fill STEP 0 first. If it passes, reformulate and you're done (the conv case).
If not, fill STEPS 1–5 + the smell test BEFORE coding; the answers become the
TDG graph (`kernel_dataflow_representation.md`). Re-profile after, and if a sub-op
is still hot, re-classify it (STEP 1) rather than micro-optimizing the current map.
