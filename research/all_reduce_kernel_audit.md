# ttnn.all_reduce Kernel Audit — 2026-05-20

Research-only writeup of how `ttnn.all_reduce` works internally on Tenstorrent
Blackhole P150 (1, 4) mesh with `cluster_axis=1, num_links=2, topology=Linear`.

## Executive summary

`ttnn.all_reduce` is **internally a two-phase composite**: it calls
`reduce_scatter_minimal_async` then `all_gather_async`. There's also a
fallback alternative path (`all_gather + moreh_sum`) for edge-case shapes.
The work is split across **dataflow kernels** (Reader/Writer on Tensix
cores), **compute kernels** (`add_tiles`, on Tensix compute cores), and
**fabric/EDM kernels** (on ethernet cores). Synchronization is per-link
**semaphores** plus a **global "all-chips-done" semaphore**.

## 1. Algorithm

`all_reduce.cpp:16-57` → `all_reduce_async.cpp:287-445`:

```
Phase 1 (Reduce-Scatter via reduce_scatter_minimal_async):
  - Each chip ends up with [seq_len, HIDDEN/4] containing the partial-summed
    slice for its index along cluster_axis.
  - Output size per chip: S/N bytes (where S = input bytes, N = num_devices).

Phase 2 (All-Gather via all_gather_async):
  - Each chip gathers all 4 scattered chunks → reconstructs full [seq_len, HIDDEN].
  - Output: S bytes per chip; identical data on every chip.
```

**Implication:** our `force_composite_ccl` flag (which calls
`ttnn.reduce_scatter + ttnn.all_gather` directly) uses the **same underlying
primitives** as `ttnn.all_reduce` internally. The composite path is mostly
a different *entry point*, not a fundamentally different algorithm. Output
tensors might differ in kernel-cache state but math/data path is similar.

## 2. Kernel architecture

Per chip during reduce-scatter phase:

| Kernel | Core type | Job |
|---|---|---|
| Reader (`ring_reduce_scatter_minimal_async_reader.cpp`) | Tensix RISCV | Reads input from DRAM via NOC, stages into circular buffers |
| Reduction Compute (`reduction.cpp:8-64`) | Tensix compute | Runs `add_tiles` to sum CBs, packs into output CB |
| Writer (`ring_reduce_scatter_minimal_async_writer.cpp`) | Tensix RISCV | Sends partial-sum chunks via fabric to neighbors |
| EDM (Ethernet Data Mover) | Ethernet core | Manages actual packet transmission over eth links |

All-gather phase: similar kernels but no reduction — Reader/Writer/EDM pass
chunks around without summing.

## 3. Fabric usage on BH P150x4 with num_links=2, Linear topology

- BH P150 has 4 eth links per axis; `num_links=2` uses two of them.
- Per-link work division: each link gets a subset of output cores (round-robin,
  `all_reduce_async_program_factory.cpp:268-294`).
- **Linear topology asymmetry**: end chips (0 and 3) have only one usable
  neighbor link; intermediate chips (1, 2) have two. End chips are bottlenecks.
- Packet headers pre-computed (`fabric_set_line_multicast_route()`,
  `worker_writer.cpp:93-98`) to avoid per-packet header construction.

## 4. Synchronization model

**Per-link reduction semaphores** (`all_reduce_async_program_factory.cpp:350-354`):
- One semaphore per active link, initialized to 0.
- Remote chip's Writer kernel issues `noc_semaphore_set()` after sending a chunk.
- Local Compute kernel waits via `cb_wait_front()` before reducing the chunk.

**Global "all-chips-done" semaphore** (`worker_writer.cpp:51`,
`all_reduce_async_program_factory.cpp:530-533`):
- `out_ready_sem_wait_value = ring_size` (= 4 in our case).
- Writer increments this semaphore for each chip that's submitted its
  contribution.
- Output buffer is not "released" for downstream use until all N chips have
  signaled.

**Critical observation for our wedge:**
> "If the reduction semaphore on the worker is not properly signaled by the
> fabric writer on the sender chip, the compute kernel blocks forever in
> `cb_wait_front()` (reduction.cpp:25). **No timeout → wedge.**"

This matches our 99% CPU silent hang symptom exactly. The kernel is spinning
in a barrier wait that never resolves.

## 5. Output tensor handling

`all_reduce_async_device_operation.cpp:85-88`:
```cpp
auto output_spec = compute_output_specs(args, tensor_args);
return create_device_tensor(output_spec, tensor_args.input_tensor.device());
```

Each call produces a **fresh DeviceStorage** via `create_device_tensor`. No
explicit aliasing with intermediate buffers, BUT:

- The reduce-scatter intermediate buffer is explicitly `.deallocate()`'d
  before all-gather begins (`all_reduce_async.cpp:254-256, 279`). If that
  hook silently fails, the intermediate's storage could persist and shadow
  the all-gather output.
- The global semaphore for "all-chips-done" lives in a buffer (line 533).
  If that buffer is deallocated or reused between launch and execution,
  writes go to garbage and downstream signals never arrive.

## 6. Composite path comparison

When triggered (`all_reduce_async.cpp:203-238, 331-366`):
- Used for edge-case shapes (2D mesh + FABRIC_2D, certain dim/shard mismatches).
- Replaces the two-phase RS+AG with: **all_gather + local moreh_sum**.
- Reshape input → all-gather along a new leading dim → moreh_sum on dim 0 →
  reshape back.
- Different kernel set: generic all_gather + `moreh_sum` (compute-only, no
  fabric for the reduction).
- Lifecycle: only one fabric phase instead of two; one global barrier.

**Our `force_composite_ccl` flag** does `ttnn.reduce_scatter + ttnn.all_gather`
directly. This is NOT the composite-fallback path described above (that one is
all_gather + moreh_sum). Our flag uses the same primitives as `ttnn.all_reduce`
internally calls — just exposed at a different API surface. May still
produce a different output tensor lineage (different program-cache key,
different semaphore set).

## 7. Likely wedge mechanisms (matched against our symptom)

The agent identified six concrete failure mechanisms. The ones matching
"99% CPU silent hang, no error" are:

1. **Synchronization stall in reduce-scatter** (mechanism #1):
   - Worker compute kernel blocks in `cb_wait_front()` because the sender's
     fabric writer never signals the reduction semaphore.
   - Root cause: EDM (eth core) fails to deliver packet, semaphore address
     stale, or `FabricConnectionManager.open_finish()` doesn't actually
     ensure connectivity.
   - **No timeout → infinite spin → 99% CPU forever** ← matches our symptom

2. **Global semaphore undercount** (mechanism #2):
   - Writer waits for `out_ready_sem_wait_value = ring_size`. If one chip's
     writer crashes/preempts before signaling, others spin waiting.
   - Same symptom: infinite spin in writer kernel.

3. **NOC congestion on single link** (mechanism #4):
   - Linear topology with `num_links=2` means end chips bottleneck.
   - If a link has high latency or fails, no rebalancing.
   - Writer's `cb_wait_front()` after line 113 never resolves.

The wedge in our case isn't necessarily *in* `all_reduce` — but the all_reduce
**chain of synchronization** may leave the mesh in a state where a SUBSEQUENT
op's `cb_wait_front()` similarly hangs. This could happen if all_reduce's
semaphore wasn't fully cleared (e.g., the global semaphore value persists
above its expected reset), so the next op that waits on it sees stale state.

## 8. Implications for the B.2.2 wedge

- The agent's identified failure mechanisms (semaphore stall, global
  undercount, NOC congestion) all produce **exactly our symptom** (silent
  spin, no error).
- The wedge happens **inside** `deltanet_step_tp`, specifically at the
  first `rms_norm` call after all_reduce. If rms_norm's kernel uses any
  shared semaphore that all_reduce's prior call left in a bad state, this
  would explain the silent hang.
- The hypothesis "all_reduce dirties shared mesh-level synchronization
  state, and the next op's `cb_wait_front()` inherits the dirty state"
  is consistent with all observed evidence:
  - All single ttnn ops pass on the contaminated slice (in isolation, each
    creates fresh semaphores)
  - The full DN sequence wedges (some op inside hits the contaminated state)
  - Per_position_list works because each DN call gets fresh semaphores
    in a clean per-position context

## 9. What this suggests for fixes

- **Force-fresh tensors don't help** (already confirmed) — the wedge isn't
  about the output tensor's storage state.
- **Composite CCL flag** (`force_composite_ccl`) might help if it uses a
  different semaphore set. Test in flight.
- **Custom all_reduce as all_gather + sum** (user's idea): use a completely
  different op composition. The agent's audit confirms this is what the
  fallback "composite path" already does inside `ttnn.all_reduce` — but
  only triggered for edge cases. Calling it explicitly might bypass any
  shared-semaphore contamination.
- **Bug worth filing upstream**: the agent's analysis suggests there's a
  potential class of bugs around semaphore lifecycle after a successful
  `ttnn.all_reduce`. Whether ours is one of those, only further testing
  can confirm.

## Citations

- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp:16-57` — user entry
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:287-445` — dispatcher
- `.../all_reduce_async/device/all_reduce_async_program_factory.cpp:268-294` (link partitioning), `:349-354` (semaphores), `:530-533` (global semaphore)
- `.../all_reduce_async/device/kernels/dataflow/worker_reader.cpp:20-57` — reader
- `.../all_reduce_async/device/kernels/dataflow/worker_writer.cpp:33-120` — writer + fabric
- `.../all_reduce_async/device/kernels/compute/reduction.cpp:8-64` — compute (add_tiles)
- `.../all_reduce_async/device/all_reduce_async_device_operation.cpp:85-88` — output tensor creation
- `.../reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:15-23` — RS dispatch
- `.../all_reduce_async/all_reduce_async.cpp:203-238, 331-366` — composite fallback decision
