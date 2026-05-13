# 06 — Trace Capture Internals (and the JIT-upload trap)

Companion to `feedback_c4v4_validated.md`. We bled a session on a "Writes are not
supported during trace capture" assert; the stack trace pinned it on
`load_binaries -> enqueue_write_shard` from inside `ttnn.rms_norm`. This page is
the post-mortem grounded in actual `tt-metal` sources, not folklore.

All file paths below are inside `experiments/.refs/tt-metal/`.

## 1. What a trace IS

A trace is a **pre-serialised byte stream of fast-dispatch commands** stored in
a DRAM `MeshBuffer`, plus host-side bookkeeping describing how many workers
each replay completes.

The buffer is built by toggling each device's `SystemMemoryManager` into
"bypass mode" during capture: instead of sending dispatch commands down the
prefetcher to the chip, the manager appends them to a `bypass_data` vector.
`record_end()` concatenates the bypass data and finally allocates the trace
`MeshBuffer` via `MeshTrace::populate_mesh_buffer` (`mesh_trace.cpp:49`).

Concretely the trace persists as:

- `MeshTraceDescriptor::data` / `ordered_trace_data` — the dispatch command bytes
  (`tt_metal/impl/trace/trace_buffer.hpp:35-42`)
- `MeshTraceDescriptor::descriptors` — per-sub-device `TraceWorkerDescriptor`s
  recording how many worker cores complete and how many GO signals get sent
- `MeshTraceBuffer::mesh_buffer` — the DRAM buffer holding `data` after end-capture
  (`trace_buffer.hpp:44-52`)
- Per program: a `TraceNode` carrying the `ProgramImpl`, RTA blobs, CB configs, and
  `TraceDispatchMetadata` (`tt_metal/impl/trace/trace_node.hpp:14-47`).

Replay (`execute_trace`) just issues a `prefetch_exec_buf` command pointing at
that DRAM region. No Python, no dispatch-command generation, no kernel rebuild.

## 2. Capture-time rules (with the asserts that enforce them)

`FDMeshCommandQueue::record_begin` sets `trace_id_` and flips `set_bypass_mode(true)`
on every device's `SystemMemoryManager` (`fd_mesh_command_queue.cpp:1063-1081`).
While `trace_id_.has_value()` is true, any API that would issue host->device
traffic asserts:

| Operation | Assert | File:line |
|---|---|---|
| `enqueue_write_shard_to_core` | `Writes are not supported during trace capture` | `fd_mesh_command_queue.cpp:456` |
| `enqueue_read_shard_from_core` | `Reads are not supported during trace capture` | `fd_mesh_command_queue.cpp:497` |
| `write_shard_to_device` (host->device tensor) | `Writes are not supported during trace capture. trace id: {}` | `fd_mesh_command_queue.cpp:590` |
| `read_shard_from_device` | `Reads are not supported during trace capture` | `fd_mesh_command_queue.cpp:624` |
| `enqueue_record_event_to_host` | `Event Synchronization is not supported during trace capture` | `fd_mesh_command_queue.cpp:717` |
| `enqueue_wait_for_event` (host-side) | same | `fd_mesh_command_queue.cpp:784` |

So the **enumeration of forbidden ops during capture**, by source evidence, is:

1. **Host->device tensor uploads** — `ttnn.from_torch(..., device=...)`,
   `ttnn.to_device`, `ttnn.copy_host_to_device_tensor`. Trips assert at line 590.
2. **JIT kernel-binary uploads** — `MeshWorkloadImpl::load_binaries`
   (`distributed/mesh_workload.cpp:129`) calls
   `mesh_cq.enqueue_write_shard_to_sub_grid(...)` (line 175) on the kernel-binary
   `MeshBuffer`. This is a write_shard, so it trips the same assert at line 590.
   **This was our blocker.**
3. **Device->host reads** — any sync/readback, line 624/497.
4. **Host-side event sync** — lines 717, 784.

What does NOT trip the assert (by absence in source):

- **Device->device copies** (`ttnn.copy(src_dev, dst_dev)`) — these enqueue a
  program, not a host-buffer write. They get recorded into the trace. Confirmed
  by `feedback_trace_state_threading_works.md` (in-trace `ttnn.copy(scatter_out,
  cache_in)` works).
- **New device allocations during capture** — `Buffer::create` is not gated by
  `trace_id_`. The allocator runs, the address shows up in trace commands. There
  IS a soft constraint: in dynamic trace_region_size mode the trace buffer is
  allocated top-down at end-capture, and `populate_mesh_buffer`
  (`mesh_trace.cpp:96-131`) asserts the trace buffer doesn't overlap with any
  buffer allocated during capture. So allocations are *permitted but cost DRAM
  high-water*. Pre-allocate before capture when possible.

## 3. The JIT / program-cache interaction (THE thing that burned us)

When you call any `ttnn.<op>`, on the first invocation for a given (op, shape,
dtype, layout, memory-config, ...) tuple, ttnn:

1. Compiles the kernel sources -> SPIR/FW objects (cached on disk under the build dir).
2. Builds a `Program` + `ProgramImpl` and inserts it in the program cache.
3. On `EnqueueMeshWorkload`, `distributed/distributed.cpp:24-30` runs three steps
   unconditionally:
   ```
   mesh_workload.impl().compile(mesh_cq.device());
   mesh_workload.impl().load_binaries(mesh_cq);
   mesh_workload.impl().generate_dispatch_commands(mesh_cq);
   ```
4. **`load_binaries` writes the kernel binaries to a per-program DRAM
   `MeshBuffer`** the first time the workload is enqueued on a device. The
   check at `mesh_workload.cpp:134-141` only short-circuits when
   `program_binary_status_` already contains an entry for the mesh device. On
   first call it falls through to lines 162-179 and calls
   `enqueue_write_shard_to_sub_grid`.
5. Subsequent enqueues on the same device hit the early-return branch (status =
   Committed), no host write.

That step 4 was happening *inside* `begin_trace_capture / end_trace_capture`
when the traced forward contained an op pattern (shape × dtype × layout) that
hadn't been seen before. `enqueue_mesh_workload` itself asserts that binaries
were sent (`fd_mesh_command_queue.cpp:287-289`: `program_binary_status != NotSent`),
so the system *requires* `load_binaries` to run before workload enqueue — and
that requirement is incompatible with bypass mode.

`generate_dispatch_commands` itself (`program.cpp:1839-1879`) does not write to
the device — it just builds command sequences and caches them keyed by
`active_sub_device_manager_id`. Once cached, future calls skip the work
(line 1857: `if (!cached_program_command_sequences.contains(command_hash))`).

**The fix is to run the same forward eagerly once before capture.** That
populates both the disk kernel cache AND `program_binary_status_` on the mesh,
so all `load_binaries` calls inside the captured forward hit the early return.
We saw exactly this — `feedback_c4v4_validated.md` reports the trace capture
dropped to 0.7s after re-adding the warmup pass.

## 4. Lifecycle

```
BeginTraceCapture(device, cq_id)
  -> MeshTrace::next_id()                          # atomic counter
  -> mesh_device.begin_mesh_trace(cq_id, trace_id) # mesh_device.cpp:1230
       -> mark_allocations_safe()
       -> create empty MeshTraceBuffer in sub-device manager
       -> FDMeshCommandQueue::record_begin
            -> reset host dispatch state
            -> trace_id_ = trace_id
            -> sysmem_manager.set_bypass_mode(true) on all devices

<user runs ttnn ops; each enqueue_mesh_workload short-circuits through
 the bypass branch at fd_mesh_command_queue.cpp:314-328, pushing
 a MeshTraceNode instead of issuing dispatch commands>

EndTraceCapture(device, trace_id, cq_id)
  -> FDMeshCommandQueue::record_end (line 1102)
       -> reconcile device-range program list
       -> serialize bypass_data per device range into trace_ctx_->ordered_trace_data
       -> trace_id_ = nullopt                      # capture window closes
       -> sysmem_manager.set_bypass_mode(false)
  -> MeshTrace::populate_mesh_buffer (mesh_trace.cpp:49)
       -> allocate DRAM MeshBuffer of total_trace_size, write bypass_data into it
  -> mark_allocations_unsafe                       # new allocations now would
                                                    # collide with trace buffer

ExecuteTrace(device, trace_id, cq_id, blocking)
  -> FDMeshCommandQueue::enqueue_trace (line 1020)
       -> issue trace_dispatch::issue_trace_commands per device
          (a prefetch_exec_buf pointing at trace_buffer->mesh_buffer)
       -> update_worker_state_post_trace_execution

ReleaseTrace(device, trace_id)
  -> MeshDeviceImpl::release_mesh_trace (mesh_device.cpp:1205)
       -> sub_device_manager.release_trace(trace_id)
       -> if trace_buffers_size_ == 0: mark_allocations_safe()
```

## 5. Mutability between replays

Replay is fully owned by the trace: the dispatch byte stream embeds every
buffer address that was live at capture time. Anything the trace touches must
remain alive at that address.

What you can still mutate **between** `execute_trace` calls:

- **Input/state buffers**, via `ttnn.copy_host_to_device_tensor(src_host, dst_dev)`.
  This is an `enqueue_write_shard` on `cq_id` (or another CQ) and is legal
  outside the capture window. The `tt_dit` `Tracer` uses exactly this
  (`tt_dit/utils/tracing.py:232-233`).
- **Output buffers** are read back via reads; the docstring warns they get
  overwritten on the next call: "The tracer returns the same output tensor
  objects every time; a subsequent call overwrites previous results in place"
  (`tracing.py:37-38`).

In-trace state threading (our scatter -> copy pattern) works because the copy
is a *device program* recorded into the trace, not a host write. See
`feedback_trace_state_threading_works.md`.

## 6. Multi-CQ behaviour

`trace_id_` lives on a single `FDMeshCommandQueue` instance — it is per-CQ
state (`fd_mesh_command_queue.cpp:1074`). The assert at
`mesh_device.cpp:1232-1236` only blocks re-entrant capture on the *same* CQ:
"CQ {} is already being used for tracing tid {}". So yes — you can capture on
`cq_id=1` while running ops on `cq_id=0`, and production demos do (e.g. the
ViT `..._2cq_trace.py` perf test). What you cannot do is start a second
capture on a CQ that's already capturing.

## 7. Size / kernel-count limits

There is no hard "op count" cap visible in source — the limit is **DRAM space
for the serialised dispatch stream**. Two configurations:

- **Static**: if `trace_region_size` is set at device open, all traces share a
  dedicated reservation; `populate_mesh_buffer` asserts cumulative size
  fits (`mesh_trace.cpp:73-78`).
- **Dynamic** (`trace_region_size = 0`): the trace buffer is allocated top-down
  in regular DRAM at end-capture. The DRAM high-water-mark tracking
  (`mesh_trace.cpp:96-131`) checks the trace buffer didn't collide with
  buffers allocated during capture.

Our 64-layer forward is ~thousands of programs but each program contributes
~tens to hundreds of bytes of dispatch commands plus an `exec_buf_end`. A few
MB of dispatch stream is typical; well under the trace region used by stock
demos.

## 8. The right pattern

What production code (`models/demos/gpt_oss/tt/model.py:765-795`,
`models/tt_dit/utils/tracing.py`) does, and what we now do too:

```python
# 1. Allocate persistent device-side state buffers OUTSIDE the trace.
#    These are the inputs/outputs/state that will get mutated between replays.
inp = ttnn.from_torch(..., device=device)        # host->device, OK
cache = ttnn.from_torch(..., device=device)      # host->device, OK

# 2. WARMUP: run the function eagerly once. Synchronize. Clear any in-place
#    state mutations you don't want carried into the trace (clear_kv_caches in
#    gpt_oss, _clone_tensor in tt_dit Tracer).
out = run_model(inp, cache)
ttnn.synchronize_device(device)
# (optional) reset any state buffer the warmup mutated

# 3. Capture.
tid = ttnn.begin_trace_capture(device, cq_id=0)
out = run_model(inp, cache)                      # all programs cached now
ttnn.end_trace_capture(device, tid, cq_id=0)

# 4. Replay loop. Update inputs via copy_host_to_device_tensor or device-side
#    copies, then execute.
for step in range(N):
    ttnn.copy_host_to_device_tensor(next_input_host, inp)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    result_host = ttnn.from_device(out)
```

Key non-obvious points:

- The warmup pass must touch every (op, shape, dtype, layout, memcfg) tuple the
  capture will see. A second forward through the same Python code is the
  simplest sufficient condition.
- If you swap kernel modules (`reload_kernels`), the program cache may still
  hold compiled programs that no longer match the new ttnn code, but the binary
  upload status persists. Force a clean recapture (`recapture=True` in our
  server pattern, see `feedback_c4v4_validated.md`).
- Don't allocate device buffers inside `run_model` that you only want to live
  for one step — they'll persist past the trace and inflate the DRAM
  high-water-mark check.

## 9. Pointers if revisiting

- `tt_metal/distributed/fd_mesh_command_queue.cpp` — `record_begin` (1063),
  `record_end` (1102), `enqueue_mesh_workload` bypass branch (314), `enqueue_trace` (1020).
- `tt_metal/distributed/mesh_device.cpp:1230-1301` — public `begin/end/replay_mesh_trace`.
- `tt_metal/distributed/mesh_workload.cpp:129` — `load_binaries` (the trap).
- `tt_metal/distributed/distributed.cpp:19-30` — order of compile / load_binaries /
  generate_dispatch_commands on every enqueue.
- `tt_metal/distributed/mesh_trace.cpp:49` — `populate_mesh_buffer`, trace-region
  sizing.
- `tt_metal/impl/program/program.cpp:1822-1879` — `allocate_kernel_bin_buf_on_device`,
  `generate_dispatch_commands` caching.
- `ttnn/cpp/ttnn/operations/trace.cpp` — thin ttnn wrappers over the mesh APIs.
- `models/tt_dit/utils/tracing.py` — reference `Tracer` class with `prep_run`
  (warmup) and `clone_prep_inputs`. Real-world pattern.
- `models/demos/gpt_oss/tt/model.py:765-795` — concrete "Compile / Capture trace"
  comment-pair you can grep for.
