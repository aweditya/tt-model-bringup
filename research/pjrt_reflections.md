# PJRT Plugin Reflections Log

Design decisions, trade-offs, and things we'd do differently.

---

## 2026-04-23: Initial Design Decisions

### Why interpretation over compilation?

We chose the applejax-style interpretation approach (walk StableHLO ops, dispatch to ttnn) over building a full compiler (StableHLO → TTIR → TTNN flatbuffer). Reasoning:

1. **Time**: A compiler is 3-6 months. Interpretation is 3-5 weeks.
2. **We already have it in Python**: experiments/tt_jax/ops.py maps 28 Jaxpr ops to ttnn. The C++ plugin is essentially a port.
3. **ttnn trace capture gives us "free compilation"**: First execution interprets ops eagerly. Second execution replays the recorded trace with zero dispatch overhead. This is the key insight — we get compilation-level performance without writing a compiler.
4. **Risk**: tt-mlir exists but the official tt-xla plugin segfaults on our Blackhole. Building on someone else's broken compiler is worse than building our own simple interpreter.

**Regret potential**: If StableHLO ops don't map cleanly to ttnn (e.g., complex scatter/gather patterns), we'll wish we had a compiler's pattern-matching power. But for transformer inference, the ops are simple and well-understood.

### Why not just improve the Python interpreter?

Our Python tt_jax interpreter (exp 14-20) already works. Why rewrite in C++?

1. **JAX integration**: A PJRT plugin integrates natively with JAX. Users write normal JAX code, it "just works" on TT devices. No custom interpreter API.
2. **Dispatch overhead**: Python interpreter adds ~50μs per op on top of ttnn's ~30μs. C++ eliminates this.
3. **Trace capture**: ttnn C++ API supports trace capture natively. From Python, we proved trace capture works (exp 95). In C++, we can capture the entire model as one trace.
4. **Ecosystem**: With a PJRT plugin, all JAX libraries (Flax, Orbax, etc.) work automatically.

### Phase 1 scope: absolute minimum

Phase 1 is just "jax.devices() shows a TT device." This means:
- PJRT_Client_Create: open ttnn device
- PJRT_Client_Devices: return device list
- PJRT_Client_PlatformName: return "tenstorrent"
- Everything else: UNIMPLEMENTED stubs

This is intentionally minimal. We want to validate the build system, linking, and plugin loading before writing any op implementations.

### Build system concerns

The biggest risk is MLIR/StableHLO build on the remote host. If it fails, we're blocked. Mitigation:
- Start with header-only StableHLO parsing (just include the protobuf definitions)
- Defer full MLIR dependency until Phase 3 when we need to actually parse StableHLO programs
- For Phase 1-2, we only need pjrt_c_api.h (a single header from XLA)

---

## MoE Optimization Reflections (Exps 90-99)

### What worked well

1. **Profiling first (exp 94)**: Without profiling, we would have guessed wrong about bottlenecks. The data showed expert dispatch (32%) was the biggest target, not attention.
2. **Partial tracing (exp 95)**: The insight that attention is a static graph (traceable) while MoE routing is dynamic (must be eager) led to the best optimization: 22.7 tok/s.
3. **Device-side routing (exp 92)**: Moving softmax/topk/sigmoid to device saved host round-trips. Small gain (2%) but validated that these ops work on Blackhole.

### What didn't work

1. **Multi-CQ (exp 96)**: Event sync works but `synchronize_device()` drains both queues. The optimization requires restructuring the decode loop, not just adding events. Lesson: async primitives only help if the code structure allows actual overlap.

### What we wish we'd done differently

1. **Should have profiled earlier**: We did 4 experiments (90-93) before profiling. Each one was useful but somewhat blind. Profile at exp 91, optimize from data.
2. **Expert weight layout**: We haven't tried DRAM-sharded or different memory configs for expert weights. The 60-expert model reads 518 MB/layer — memory layout matters enormously here.
3. **Batch size > 1**: All experiments are batch=1. MoE models amortize expert dispatch over batch elements. Batch=4 or batch=8 could change the performance picture dramatically.

### The 7ms theoretical floor

From exp 94 profiling: 1059 ops × 30μs dispatch = ~32ms overhead. Even with perfect trace capture of everything, the minimum is bounded by weight reads: 4 experts × 3 matmuls × (2048×1408×1 byte) = 34.5 MB/token → 34.5/450 = 0.077ms. Plus shared expert: 3 × (2048×5632×1) = 34.5 MB → 0.077ms. Plus attention weight reads. Total compute floor is ~7ms → 143 tok/s theoretical maximum.

We're at 44ms. The gap is almost entirely dispatch overhead (30μs × 1059 ops). This is why the PJRT plugin with trace capture is so important — it could eliminate dispatch overhead entirely.

---

## Phase 1 Implementation: Skeleton (April 2026)

### PJRT_Api struct layout problem

The biggest risk in Phase 1 is ABI compatibility. The `PJRT_Api` struct is a flat C struct with ~100+ function pointers. The field ORDER determines which function JAX calls for which operation. If our struct layout doesn't match what jaxlib expects, function pointers get misaligned and we get silent corruption or segfaults.

**Decision**: Vendor a minimal pjrt_c_api.h for development, create `fetch_pjrt_header.sh` to download the real one before any actual testing. This lets us develop the C++ structure without fighting the full XLA header dependency chain.

### Memory ownership model

PJRT uses opaque pointer types. JAX holds `PJRT_Client*`, `PJRT_Device*`, etc. and passes them back to us. Our ownership model:

- **Client owns device and memory**: Embedded members in PJRT_Client, not heap-allocated. Valid for the client's lifetime.
- **Buffers and executables**: Heap-allocated, created/destroyed independently by JAX.
- **Events**: Heap-allocated, always immediately ready (synchronous Phase 1).

This avoids use-after-free: as long as client is alive, all device/memory pointers are valid.

### ttnn headers only in .cc files

Store ttnn handles as `void*` in headers, cast to concrete types in .cc files. Only .cc files include `<ttnn/ttnn.hpp>`. Same pattern used by opaque pointer APIs everywhere. Trades type safety for compile-time isolation.

### What Phase 1 implements

- Error API: create, destroy, message, code
- Plugin API: initialize, attributes
- Client API: create (opens ttnn device 0), destroy, platform name/version, device enumeration
- Device/DeviceDescription API: id, kind, debug string, process index, memory spaces
- Memory API: id, kind, debug string, addressable devices
- Event API: destroy, is_ready (always true), error, await (no-op), on_ready (immediate callback)
- Buffer metadata: element type, dimensions, size, device, memory, is_deleted
- Stubs for: Compile, Execute, BufferFromHostBuffer, ToHostBuffer

### Files created

```
pjrt_plugin/
  CMakeLists.txt          - Build system (finds ttnn, builds .so)
  src/
    plugin.cc             - PJRT_Api function table + GetPjrtApi()
    client.h/cc           - TtClient: device lifecycle, metadata
    buffer.h/cc           - TtBuffer: tensor wrapper (stub)
    executable.h/cc       - TtExecutable: StableHLO wrapper (stub)
    plugin.lds            - Linker script to export only GetPjrtApi
    ops/
      arithmetic.h/cc     - (Phase 3 placeholder)
      matmul.h/cc         - (Phase 3 placeholder)
      elementwise.h/cc    - (Phase 4 placeholder)
  jax_plugins/tt/
    __init__.py           - Plugin registration with JAX
  tests/
    conftest.py           - Shared fixtures
    test_device_discovery.py  - Phase 1 test
    test_buffer.py        - Phase 2 tests (skipped)
    test_basic_ops.py     - Phase 3 tests (skipped)
    test_matmul.py        - Phase 3 tests (skipped)
  scripts/
    fetch_pjrt_header.sh  - Download matching PJRT header from XLA
    build.sh              - Build on remote host
  third_party/pjrt/
    pjrt_c_api.h          - Vendored PJRT C API (simplified for dev)
```

### Next steps

1. SSH to remote host, run `fetch_pjrt_header.sh` to get the real PJRT header
2. Run `build.sh` to compile the plugin
3. Run `test_device_discovery.py` to verify `jax.devices()` shows our device
4. Begin Phase 2: BufferFromHostBuffer + ToHostBuffer

---

## Experiments 97-99 Results (April 23, 2026)

### The dispatch wall is real

All three optimization attempts failed to beat exp 95's 22.7 tok/s:

| Exp | What we tried | Result | Why it failed |
|-----|--------------|--------|---------------|
| 97 | swiglu fusion | 20.8 (-8%) | `ttnn.swiglu()` crashes on Blackhole; fallback adds overhead |
| 98 | Multi-CQ pipelining | 20.9 (-8%) | CQ overhead > routing sync savings (0.28ms/layer) |
| 99 | HiFi2 + DRAM-sharded | 22.7 (0%) | Not bandwidth or compute bound at batch=1 |

### What we learned

1. **ttnn.swiglu() is broken on Blackhole**: `ShapeBase[] index out of range. 3 not in [-4, 2)`. The op exists but doesn't handle 4D tensor shapes. Would need to file a bug with Tenstorrent.

2. **Multi-CQ has fixed overhead**: Opening a device with 2 command queues adds baseline overhead. The routing readback we're trying to hide is only 0.28ms/layer = 6.7ms total. But multi-CQ's overhead eats all of that. The first token was 587ms vs ~80ms normally — CQ initialization is expensive.

3. **Batch=1 MoE is dispatch-bound, period**: We confirmed this definitively. DRAM sharding, compute fidelity, L1 memory config — none of these matter because the bottleneck is 1000+ Python→C++ dispatch round-trips at 30μs each. The ONLY way to attack this is:
   - Full trace capture (impossible for dynamic MoE routing)
   - Batch > 1 (amortize dispatch over batch elements)
   - C++ dispatch (PJRT plugin eliminates Python overhead)
   - Fewer ops (already fused everything fusible)

### Regrets

1. **Should have predicted exp 97 failure**: swiglu is a relatively new ttnn op. We should have tested it in isolation before building a full experiment around it. The pre-test was there but the 4D decode shape wasn't tested.

2. **Multi-CQ overhead was predictable**: Exp 96 already showed no speedup from multi-CQ. We should have analyzed WHY before trying a more complex pipelining scheme. The answer was: `synchronize_device()` drains both queues, and opening 2 CQs adds overhead.

3. **The right experiment was batch > 1**: With 4 tokens per step, we'd dispatch the same 1000 ops but get 4x the useful work. This is the classic "if you can't reduce overhead, amortize it" approach.

### What's next

The optimization path for batch=1 MoE is essentially exhausted at the ttnn Python API level. 22.7 tok/s is our ceiling without:
1. **Batching** (exp 100?): batch=4 or batch=8 decode
2. **PJRT plugin with trace**: C++ dispatch + trace capture could push toward the 143 tok/s theoretical ceiling
3. **Full-model trace with static expert selection**: If we pick experts based on a pattern (e.g., always same top-4), we can trace the entire model. Loses some accuracy but would be dramatically faster.

This is a pivotal moment: we've exhausted incremental optimizations. The next leap requires architectural change (batching or PJRT).

---

## Phase 1 Completion: What Actually Happened (April 23, 2026)

### The nullptr segfault lesson

The hardest bug in Phase 1 was a segfault on `jax.devices()`. Our plugin loaded fine, `GetPjrtApi()` returned a valid struct, but JAX crashed immediately.

**Root cause**: JAX's C++ wrapper (`PjRtCApiClient`) calls function pointers from the PJRT_Api struct WITHOUT nullptr checks. If even one function pointer is null, JAX segfaults — not on that function, but on whatever code path touches it during initialization. This includes functions you'd think are optional like `PJRT_Client_TopologyDescription` or `PJRT_Executable_Serialize`.

**Fix**: Fill ALL 115 function pointers. Real implementations where we can, `Unimplemented()` stubs everywhere else. The stub returns `PJRT_Error_Code_UNIMPLEMENTED` instead of segfaulting.

**Lesson**: Never leave function pointers null in a C API vtable. Even "unused" ones will be called.

### ABI debugging technique

We validated struct layout by loading both our .so and the official tt-xla .so with ctypes, reading raw bytes at known offsets, and comparing function pointer positions. This caught zero bugs (our layout was correct) but gave us confidence to debug the segfault as a nullptr issue rather than a layout issue.

**Pattern**: `ctypes.c_void_p.from_buffer_copy(raw_bytes, offset).value` to read function pointers from a PJRT_Api struct.

### struct_size matters

PJRT_Api v0.70 has `struct_size = 944` (118 × 8 bytes). The official tt-xla plugin ships v0.68 with `struct_size = 936`. JAX uses struct_size to know which fields are valid. Getting this wrong means JAX reads past the struct or ignores valid fields.

### Consolidation was the right call

We started with separate buffer.h/cc, executable.h/cc, ops/*.h/cc files. Consolidated everything into client.h (struct definitions) + plugin.cc (all callbacks). This eliminated circular dependency issues and made the code much easier to navigate. At 580 lines, plugin.cc is manageable. If it grows past ~1000 lines in Phase 3, we can split again with clear boundaries.

### Phase 1 final state

- 26KB .so, single exported symbol (`GetPjrtApi`)
- `jax.devices()` → `[TT Blackhole (device 0)]`
- `jax.default_backend()` → `tt` (priority 500)
- Build: cmake + g++ on remote host, no ttnn dependency yet
- All 115 function pointers populated (23 real, 92 stubs)
