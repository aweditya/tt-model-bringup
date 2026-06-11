# PJRT Plugin Interface: What Would a TT Backend Need?

## Q: What must a minimal PJRT plugin implement?

**A:** A PJRT plugin is a shared library (`.so`) that exports `GetPjrtApi()` returning a function pointer table. JAX loads it via:
```python
jax.xla_bridge.register_plugin('blackhole', priority=500, library_path='pjrt_plugin_tt.so')
```

The minimum required functions are:

| Function | Purpose |
|----------|---------|
| `PJRT_Plugin_Initialize` | Return plugin name/version |
| `PJRT_Client_Create` | Create the central coordinator |
| `PJRT_Client_Devices` | Enumerate available devices |
| `PJRT_Buffer_FromHostBuffer` | Host → device transfer |
| `PJRT_Buffer_ToHostBuffer` | Device → host transfer |
| `PJRT_Client_Compile` | Compile StableHLO → executable |
| `PJRT_LoadedExecutable_Execute` | Run compiled code |
| Destructors for all objects | Cleanup |

## Q: How does compilation work in a PJRT plugin?

**A:** JAX sends a StableHLO module (MLIR text or binary) to `PJRT_Client_Compile()`. The plugin must:

1. **Parse the StableHLO** — ~100 possible ops (matmul, add, relu, reshape, etc.)
2. **Lower to device code** — however the plugin wants
3. **Return a PjRtLoadedExecutable** — an opaque handle JAX can call Execute() on

The key insight from our experiments: **lowering doesn't have to mean MLIR passes.** A viable approach:

```
StableHLO ops → TT-NN op calls → TT-NN trace capture → replayable trace
```

## Q: How does our trace capture discovery map to PJRT?

**A:** Perfectly. Here's the mapping:

| PJRT Concept | Our Implementation |
|---|---|
| `Client_Compile(stablehlo)` | Walk StableHLO, map to TT-NN ops, wrap in `begin_trace_capture` / `end_trace_capture` |
| `Executable_Execute(buffers)` | `copy_host_to_device_tensor` → `execute_trace` → `to_torch` |
| `Buffer_FromHostBuffer` | `ttnn.from_torch(..., device=device)` |
| `Buffer_ToHostBuffer` | `ttnn.to_torch(tensor)` |
| `Client_Devices` | `ttnn.open_device(device_id=0)` |

This is what we called the **"Level 1" backend** in wiki/12:
- **No kernel fusion** — each StableHLO op maps 1:1 to a TT-NN op
- **No MLIR lowering** — we skip TTIR/TTNN dialect entirely
- **Just trace wrapping** — the 3.23x speedup from experiment 12

## Q: What's the execution flow end-to-end?

```
User writes:    y = jax.jit(model)(x)

JAX traces:     model(tracer) → Jaxpr → StableHLO module

Plugin compiles:
  1. Parse StableHLO: [matmul(%0, %1), add(%2, %3), relu(%4), ...]
  2. Allocate device buffers for intermediates
  3. begin_trace_capture(device)
  4. For each op: call ttnn equivalent (ttnn.matmul, ttnn.add, ttnn.relu...)
  5. end_trace_capture(device) → trace_id
  6. Return PjRtExecutable wrapping trace_id

Plugin executes:
  1. copy_host_to_device_tensor(input, pre-allocated buffer)
  2. execute_trace(device, trace_id) — replays all ops, no Python dispatch
  3. to_torch(output_buffer) → return to JAX
```

## Q: What are the hard parts?

1. **Op coverage**: StableHLO has ~100 ops. TT-NN doesn't have 1:1 mappings for all of them. Scatter, gather, dynamic slicing, and control flow are tricky.

2. **Shape/layout handling**: StableHLO uses row-major dense tensors. TT-NN uses 32×32 tile layout. The plugin must handle padding, tiling, and layout conversion.

3. **Memory management**: Traces pin device memory. Large models may not fit in L1 or even DRAM if intermediates aren't freed properly.

4. **Dynamic shapes**: Traces are fixed-shape. If JAX sends different input shapes, we need to re-capture the trace (like CUDA graph re-capture on shape change).

## Q: Is this approach viable as a real backend?

**A:** For inference, yes. The trace-based approach gives:
- 3.23x speedup over eager (experiment 12)
- Correct results with new data (verified)
- No compiler infrastructure needed

For training, it's harder — gradients introduce dynamic control flow, and the trace must be re-captured for each new graph structure.

The existing tt-xla takes the harder path (full MLIR: StableHLO → TTIR → TTNN → Metalium) because it aims for Level 2 optimizations (fusion, sharding). But a Level 1 trace-based backend could ship faster and provide a useful baseline.

## Sources
- Research notes (research/03_jax_xla_pjrt.md)
- Experiment 12 results (trace capture)
- JAX PJRT documentation
- OpenXLA PJRT C API headers
