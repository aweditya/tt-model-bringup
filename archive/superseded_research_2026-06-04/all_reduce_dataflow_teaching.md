# ttnn.all_reduce on Blackhole P150 — Physical Dataflow Walkthrough

Teaching companion to `research/all_reduce_kernel_audit.md`. The audit answered
*"what is the algorithm?"*. This doc answers *"where does each byte physically
sit, and how does it move?"* — i.e., the hardware mapping.

Setup we are tracing:
```
mesh shape (1, 4)                  partial : [seq_len, HIDDEN=5120] bf16
device   : 4× Blackhole P150        layout  : TILE_LAYOUT, INTERLEAVED, DRAM
call     : ttnn.all_reduce(partial, cluster_axis=1, num_links=2,
                           topology=ttnn.Topology.Linear)
```

---

## A. The Blackhole P150 chip in one diagram

Per Tenstorrent's P150 specs page: **120 Tensix cores per chip**, **512 GB/s
GDDR6** DRAM, **4× QSFP-DD 800 Gbps** ethernet ports for inter-chip fabric
([docs.tenstorrent.com/aibs/blackhole/specifications.html](https://docs.tenstorrent.com/aibs/blackhole/specifications.html)).
The P150 inherits the Wormhole tile architecture; the public reverse-engineering
work for Wormhole (Corsix parts 1–8) is the right mental model for what each
tile is, and the tt-isa reference manual is the source of truth (per Corsix part 8).

ASCII layout (schematic, not exact silicon coordinates):

```
                  one Blackhole P150 chip (logical view)

           +-------------------------------------------------+
           |   ETH cores  (E)  — 800 Gb QSFP-DD ×4           |
           |   . . . each E tile = 1 ERISC + NoC routers     |
           +---+-----------------------------------------+---+
           | D |                                         | D |
           | R |     Tensix grid — 120 T tiles total     | R |
           | A |     each T tile:                        | A |
           | M |       • BRISC (RV  – reader/dispatch)   | M |
           |   |       • NCRISC (RV – writer/NoC1)       |   |
           | b |       • TRISC0  – UNPACK                | b |
           | a |       • TRISC1  – MATH (add_tiles)      | a |
           | n |       • TRISC2  – PACK                  | n |
           | k |       • 1.5 MB L1 SRAM (CBs live here)  | k |
           | s |       • Matrix unit (Dst register)      | s |
           |   |       • SFPU vector unit                |   |
           +---+-----------------------------------------+---+
           |   2D NoC mesh: NoC0 (E/S), NoC1 (W/N),          |
           |   32-byte links, wraps at edges (Corsix part 1) |
           +-------------------------------------------------+
```

Per-T-tile structure (5 baby RISC-Vs, L1, matrix/vector units, NoC routers) is
from Corsix part 1. The two NoCs and their direction split (NoC0 = E/S,
NoC1 = W/N) is also part 1. The 9-cycle-per-hop NoC propagation number is from
part 3. Tensix Sync provides 8 hardware semaphores per tile with `SEMINIT/POST/
GET/WAIT` instructions (Corsix part 5) — these are the metal-level building
blocks for the higher-level `noc_semaphore_set` we see in kernel source.

**qb2 host: 4 chips wired as a (1, 4) line.** Each chip uses 2 of its 4 QSFP-DD
links for the cluster-axis-1 direction (validated 2026-05-19 — see
`feedback_p1_num_links_2_shipped.md`). The friend's `link_dict["P150x4"] =
(2, 2)` codifies that 2 links per axis is the architected pair count.

```
        (1, 4) mesh — Linear topology on qb2

        +---+        +---+        +---+        +---+
        |C0 |========|C1 |========|C2 |========|C3 |
        +---+   2    +---+   2    +---+   2    +---+
                 links per hop (num_links=2)
        end          middle        middle          end
```

Linear vs Ring: in Linear there's no wrap from C3 back to C0; in Ring there is
(see audit §3 — "end chips (0 and 3) have only one usable neighbor link;
intermediate chips have two"). qb2's cabling is wired as a line.

---

## B. Where the input tensor lives before the call

`partial` is a bf16, TILE_LAYOUT, INTERLEAVED tensor in DRAM on each chip.
"INTERLEAVED" means tiles are striped across **all DRAM banks** in row-major
tile order, not packed into one bank — this keeps every memory controller busy
during a streaming read. P150 has multiple DRAM banks per chip; total bandwidth
is 512 GB/s (Tenstorrent BH specs page).

A TILE in tt-metal is a **32×32 element block**. For `[seq_len, 5120]` bf16:

```
tile size : 32 × 32 × 2 bytes = 2048 bytes = 2 KB
tiles per row (HIDDEN dim) : 5120 / 32 = 160 tiles
tiles per col (seq dim)    : ceil(seq_len/32)
```

At call time the kernels look like:

1. **Reader kernel** runs on BRISC of N worker Tensix cores
   (`worker_reader.cpp:20–57` per the audit). Each worker is assigned a tile
   range. The reader issues NoC reads into a circular buffer (CB) in its own L1
   SRAM.
2. **CBs live in per-core L1** (1.5 MB total per T tile, per Corsix part 1).
   The reader uses `cb_reserve_back / cb_push_back`; the consumer (compute or
   writer) uses `cb_wait_front / cb_pop_front`. This is the standard tt-metal
   producer/consumer handshake.
3. **NoC routing**: BRISC issues `noc_async_read` over **NoC0 (E/S)** for one
   direction of traffic; NCRISC uses **NoC1 (W/N)** for the return-path writes
   (Corsix part 1: each tile has 4 outbound + 4 inbound 32-byte links to its
   neighbors; the two NoCs split direction). This is why having two NoCs
   matters — read and write can stream in parallel without contention.

```
   DRAM bank K        L1 of worker core (x,y)
   +---------+        +-----------+
   |  tile   | -NoC-> | CB0 input |  reader (BRISC)
   |  ...    |        | CB1 input |
   +---------+        | CB2 outp. |  consumer (compute)
                      +-----------+
```

---

## C. Reduce-scatter — the physical flow

`ttnn.all_reduce` decomposes into **reduce-scatter then all-gather** (audit §1,
algorithm). For Linear topology with `num_links=2`, here's what happens
physically across the 4 chips. Each chip's full `[seq_len, 5120]` partial gets
chopped into 4 chunks along HIDDEN (each chunk = `[seq_len, 1280]`). After
reduce-scatter, chip `i` owns the fully-summed chunk `i`.

Linear reduce-scatter is implemented as a pipelined line-pass. The key idea is
**accumulate-as-you-pass**: a chunk leaves chip 0 with partial sum `p0`;
chip 1 receives it, adds its own `p1`, forwards `p0+p1`; chip 2 adds `p2`,
forwards `p0+p1+p2`; chip 3 adds `p3` and **keeps** the final sum locally.

```
chunk 3 (the "stays at chip 3" chunk):

   C0.chunk3 ─ ETH ─► C1 ─(+C1.chunk3)─► C2 ─(+C2.chunk3)─► C3 ─(+C3.chunk3)
                                                                    │
                                                                    ▼
                                                           final reduced chunk 3
                                                           lives in C3 L1/DRAM
```

All four "destination" chunks happen in parallel, each flowing toward the chip
that will own it. The two end chips (C0, C3) only forward one direction; the
middle chips (C1, C2) forward both ways. This is the asymmetry the audit calls
out — end chips have one useful neighbor link per axis, middles have two.

**Per-link work split.** The 4 chunks are partitioned by HIDDEN-tile index
across the `num_links=2` ethernet links, round-robin across worker cores
(audit §3 cites `all_reduce_async_program_factory.cpp:268–294`). Concretely:
some worker Tensix cores are bound to link-A, others to link-B; each writer
opens a `FabricConnectionManager` to one link (`worker_writer.cpp:79–81`).

**What rides on each link.** The writer composes packets containing
(payload tiles + a fabric routing header). The header is prebuilt **once** with
`fabric_set_line_multicast_route()` for forward and backward directions
(`worker_writer.cpp:93–98`) — this avoids the cost of building headers per
packet (audit §3 calls this out).

---

## D. The compute kernel — what `add_tiles` actually does

When a chunk arrives at chip K from a neighbor, the writer on the sender side
NoC-writes the payload **directly into chip K's worker CBs** via the fabric.
The local compute kernel (`reduction.cpp`, only 64 lines) then sums it with the
local partial.

The hot inner loop (`reduction.cpp:42–47`):

```cpp
add_tiles(cb_in0, cb_in1,
          block * block_num_tiles + p * max_dst_tiles + i,   // operand A index in CB
          (block + 1) * block_num_tiles + p * max_dst_tiles + i,  // operand B index
          i);                                                 // destination Dst slot
```

What this maps to at the silicon level (per Corsix part 5 + part 6 +
tt-metal compute API):

1. `tile_regs_acquire()` reserves the **Dst register** — a large 2D SRAM that
   the Matrix unit writes its output to (Corsix part 6: "Dst is the large 2D
   piece of memory that the Tensix Matrix unit writes results to"; 512 or 1024
   rows × 16 lanes per row). Dst is the rendezvous between the MATH thread and
   the PACK thread.
2. `add_tiles(cb_in0, cb_in1, a_idx, b_idx, dst_idx)` is a high-level LLK call
   that internally dispatches three coordinated micro-ops across **three of the
   five baby RISC-Vs**:
   - **TRISC0 (UNPACK)** reads two 32×32 tiles out of L1 CBs `cb_in0[a_idx]`
     and `cb_in1[b_idx]` into the SrcA/SrcB matrix registers.
   - **TRISC1 (MATH)** runs the Matrix unit's element-wise add accumulating
     into `Dst[dst_idx]`.
   - These three threads use **Tensix Sync semaphores** (8 hardware semaphores
     per tile, Corsix part 5) to handshake: UNPACK signals MATH when SrcA/SrcB
     are loaded; MATH signals PACK when Dst is ready.
3. `pack_tile(i, cb_out0, ...)` runs on **TRISC2 (PACK)** and moves Dst tile
   `i` into the output CB.
4. `tile_regs_release()` returns Dst to the free pool.

The interesting bit: `cb_wait_front(cb_in0, num_blocks * block_num_tiles)` at
the top (`reduction.cpp:25`) **blocks until every chunk from every neighbor
has been delivered**. That semaphore is incremented by the remote chip's
fabric writer. If a single chip fails to signal, this `cb_wait_front` spins
forever — the wedge described in audit §4.

```
   per-T-tile compute pipeline for one add_tiles call

   L1 CB cb_in0  ──┐
                   ├─►[UNPACK/TRISC0]──► SrcA ──┐
   L1 CB cb_in1  ──┘                            ├─►[MATH/TRISC1]──► Dst[i] ──►[PACK/TRISC2]──► L1 CB cb_out0
                                       SrcB ──┘
```

---

## E. All-gather phase

After reduce-scatter, each chip has its own `[seq_len, HIDDEN/4]` reduced
chunk. All-gather then circulates those chunks so every chip ends up with all
4 chunks concatenated back to `[seq_len, HIDDEN]`.

Same physical machinery, different recipe:

| Step | Reduce-Scatter | All-Gather |
|---|---|---|
| Reader | reads local partial from DRAM into CB | reads local chunk from CB |
| Writer | sends chunk via fabric to next chip | sends chunk via fabric (multicast forward+backward) |
| Compute | `add_tiles` to sum incoming + local | **none** — the data is final, just pass-through |
| Output | local reduced chunk in L1 | full reconstructed tensor written back to DRAM |

In Linear topology, all-gather still walks chunks chip-by-chip, but it does so
in **both directions simultaneously** (the forward and backward packet headers
both pre-set in `worker_writer.cpp:97–98`). Middle chips receive a chunk from
one side, write it to their local output buffer, then immediately forward it
to the other side. The fabric `line_multicast_route` mechanism makes the chunk
visible to all chips on a path in one go.

---

## F. Synchronization at the metal

Three layers of synchronization, each at a different physical location.

**1. Intra-tile semaphores (UNPACK↔MATH↔PACK).** Live in the Tensix Sync unit
inside each T tile — 8 hardware semaphores with 4-bit counters, accessed via
`SEMINIT/SEMPOST/SEMGET/SEMWAIT` (Corsix part 5). These never leave the tile;
zero NoC traffic. Used inside `add_tiles`/`pack_tile` LLK calls.

**2. Per-link reduction semaphores (chip-local L1).** Audit §4 cites
`all_reduce_async_program_factory.cpp:350–354`. One semaphore per active link
sits in **L1 of a designated worker core**. Remote chip's writer issues
`noc_semaphore_set(...)` which is a NoC write to that L1 address with the
atomic-increment flag set (Corsix part 1: writes can carry an "atomic
increment at destination" attribute; Wormhole atomic ops include `ATINCGET`).
Local compute kernel's `cb_wait_front` is a busy-poll on that L1 word.

**3. Global "all-chips-done" semaphore.** One word per program, in L1 of a
known worker. `out_ready_sem_wait_value = ring_size = 4` for our (1,4) mesh
(audit §4, citing `worker_writer.cpp:51`). Every chip's writer increments it
once when its phase is done. Until all 4 increments arrive, downstream
consumers block. This is the metal-level "barrier" for the whole all-reduce.

```
   semaphore physical placement

   ┌─────────────────────────────────────────────────────┐
   │                       Chip K                        │
   │   ┌──────────────┐         ┌──────────────────┐     │
   │   │   T tile     │         │  designated      │     │
   │   │ (per-tile    │         │   "barrier" core │     │
   │   │  Tensix Sync │         │   L1 sem word    │◄────┼── NoC write +
   │   │  hw sems)    │         │  (out_ready_sem) │     │   atomic inc
   │   └──────────────┘         └──────────────────┘     │   from remote
   │   ┌──────────────────────────────────────────┐      │   writer
   │   │  worker T tile L1                        │      │
   │   │   ┌──── reduction sem (link 0) ◄─────────┼──────┼── NoC+atomic
   │   │   └──── reduction sem (link 1) ◄─────────┼──────┼── from remote
   │   └──────────────────────────────────────────┘      │   writer
   └─────────────────────────────────────────────────────┘
```

The ERISC cores themselves use a separate fabric flow-control mechanism:
**fabric stream registers** on the ETH cores manage `sender_free_slots`,
`receiver_free_slots`, `sender_acks`, `sender_completions` (per
`tt_metal/fabric/debug/README.md`). Healthy state = high free-slot values, low
ack/completion values. The `fabric_erisc_dumper.py` tool can dump these
registers in real time to see fabric-level backpressure — useful for diagnosing
the wedge described in the audit.

---

## G. What we use vs. what's available

| Parameter | Our choice | Hardware max on P150 | Headroom |
|---|---|---|---|
| Topology | Linear | Ring (if cabled) | end-chip asymmetry; ~1 extra hop on average |
| num_links per axis | 2 | 4 QSFP-DD per chip total | another factor on aggregate link BW if axis-2 isn't used elsewhere |
| Eth link rate | 800 Gbps per QSFP-DD | same | n/a |
| NoC | 2 NoCs, 32 B/link | same | already maxed by tt-metal |

Per-link probe (`feedback_p1_num_links_2_shipped.md`, 2026-05-19): going from
`num_links=1` to `num_links=2` saved ~11% on all_reduce at `[1, 5120]` bf16
and translated to +1.65% end-to-end tok/s (12.72 → 12.93). At this shape
`all_gather` is bandwidth-flat (single 0.193 ms whether L=1 or L=2), meaning
the gather phase is launch-dominated, not bandwidth-dominated. The remaining
slack is **NOT** in raw link bandwidth; it's in collective scheduling.

Note on cores: the qb2 chips may be running firmware v19.5.0+ which
silently disabled 20 Tensix cores per chip (140 → 120) per
`feedback_p150_firmware_core_check.md`. The 120-core figure in section A is
the post-v19.5.0 number from the official BH specs page; if the host is on
older firmware the count could be higher. (Unverified for qb2 specifically.)

---

## H. Common failure modes and Python-visible symptoms

| Mechanism | Where it lives | What Python sees |
|---|---|---|
| Reduction semaphore never signaled (audit §7.1) | L1 of worker core | `ttnn.all_reduce(...)` hangs with **99% CPU**, no exception, no timeout. Process must be killed. Matches our B.2.2 wedge symptom. |
| Global `out_ready_sem` undercount (audit §7.2) | L1 of barrier core | Same silent hang; one chip's writer crashed/preempted before its `noc_semaphore_inc`. |
| NoC congestion on a single eth link (audit §7.3) | Eth core fabric stream registers | Latency spike (not hang); visible as `receiver_free_slots` near 0 in `fabric_erisc_dumper.py --fabric-streams`. |
| Eth link physical failure | QSFP-DD port | Driver error at `open_mesh_device` time, or fabric init failure on `set_fabric_config(FABRIC_1D)` (per `feedback_c71_mesh_smoke_pass.md`). Not a silent hang — usually a clear error. |
| L1 CB oversubscription | Worker T tile L1 (1.5 MB cap) | Compile error (page size × num pages > L1 budget) or runtime memory-allocator error. Not a silent hang. |
| Stale program-cache key with wrong link count | Host runtime | Wrong perf (silent), but math correct. Caught by re-baselining after any CCL config change. |

The key tool for diagnosing the silent-hang class: poll the ERISC fabric
stream registers with `fabric_erisc_dumper.py --fabric-streams --poll`. If
free slots are stuck at zero on one link while acks/completions on the peer
chip climb, you've localized backpressure to a specific eth pair. If free
slots are healthy but the local compute kernel is still spinning, the wedge
is at the per-link reduction semaphore in L1 — read it directly with the same
tool by `--addresses 0x<sem_addr>`.

---

## Sources

- **Audit**: `research/all_reduce_kernel_audit.md` (algorithm, kernel files, semaphores)
- **Kernel source**: `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/{compute/reduction.cpp,dataflow/worker_writer.cpp}`
- **Fabric debug**: `experiments/.refs/tt-metal/tt_metal/fabric/debug/README.md`
- **BH P150 specs**: https://docs.tenstorrent.com/aibs/blackhole/specifications.html (120 Tensix, 512 GB/s GDDR6, 4× QSFP-DD 800 Gbps)
- **Corsix Wormhole series**: parts 1, 3, 5, 6, 7 (https://www.corsix.org/content/tt-wh-part1 ... part8). Wormhole shares the T-tile / NoC / ETH architecture pattern that Blackhole inherits; for chip-level dataflow the Wormhole reverse-engineering is the best public source.
  - Part 1: 10×12 tile grid, 5 baby RISC-Vs per T tile, 2 NoCs (NoC0 E/S, NoC1 W/N), 32-byte links
  - Part 3: NoC propagation = 9 cycles/hop
  - Part 4: ETH tile = 100 Gb/s per direction (WH; BH P150 uses faster QSFP-DD 800 Gbps)
  - Part 5: Tensix Sync = 8 hw semaphores per tile; SEMINIT/POST/GET/WAIT
  - Part 6: Dst register (matrix unit output), Tensix Vector (SFPU)
  - Part 7: Matrix unit does `Dst += SrcB @ SrcA` (used by `matmul_tiles`; `add_tiles` is the eltwise sibling)
  - Part 8: pointer to https://github.com/tenstorrent/tt-isa-documentation for the authoritative reference manual
- **Memory notes**: `feedback_p1_num_links_2_shipped.md` (num_links=2 ships), `feedback_p150_firmware_core_check.md` (120-vs-140 core caveat), `feedback_c71_mesh_smoke_pass.md` (fabric init prerequisite), `reference_p150_roofline_priority.md` (512 GB/s DRAM peak)
