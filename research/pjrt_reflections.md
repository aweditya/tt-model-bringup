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

---

## Phase 4: Reduce + Composite Ops (April 26, 2026)

### The "inspect first" approach paid off

Before implementing any new ops, we wrote `inspect_stablehlo.py` to lower softmax, layer norm, RMS norm, MLP, SiLU MLP, and attention to StableHLO and print the IR. Key finding: **only 13 unique ops across all transformer-relevant functions**, and we already supported 12. The only missing op was `stablehlo.reduce`.

This saved us from implementing ops we didn't need (e.g., slice, gather, iota are not needed for these core blocks). The design doc listed ~20 ops for Phase 4 but the actual IR showed only 1 was missing.

### `bytecode_to_text` format differs from `as_text`

We wrote a second inspection script (`inspect_bytecode_format.py`) to check what the engine's actual `bytecode_to_text` path produces vs JAX's `as_text()`. Key differences:
- Functions use `"func.func"() <...> ({` generic syntax (quoted)
- Returns use `"func.return"(...)` (quoted)
- Transpose uses `dims = [...]` not `permutation = [...]`
- Function calls use `"func.call"(...)` without callee name in text

This was critical — parsing the wrong format would have caused silent failures.

### stablehlo.reduce: simpler than expected

We feared body region parsing (the design doc flagged it as Medium risk). But `bytecode_to_text` produces the compact `applies stablehlo.XXX across dimensions = [N]` shorthand, not the full body region. This made parsing trivial: just regex for the function name.

### Hex float constants

StableHLO uses hex IEEE 754 for special values: `0xFF800000` = -infinity (used as reduce-max init value). We added `struct.unpack` handling for these. This is the kind of edge case that only shows up with real JAX programs, not toy examples.

### func.call: positional dispatch

The bytecode format doesn't include callee names in `"func.call"`. Functions appear in module order: first is main, subsequent are private. Calls map to private functions by order of appearance. This works for current JAX output but is fragile — if JAX reorders functions, we'll need to parse the callee attribute from the `<...>` dict.

### Phase 4 partial state (reduce + composite ops)

- 29 engine tests pass (was 20 before)
- New ops: reduce (sum, max, min, prod), func.call
- New composite tests: softmax, layer norm, RMS norm, attention, MLP+relu, SiLU MLP
- Also fixed: hex float constants, transpose `dims` format
- Engine still runs on numpy (CPU). No ttnn linked yet.

### What's still missing for a full transformer

Looking at real transformer decode (KV cache updates, multi-head attention with reshapes):
- `stablehlo.slice` / `stablehlo.dynamic_slice` — head splitting
- `stablehlo.concatenate` — head reassembly
- `stablehlo.gather` — embedding lookup
- `stablehlo.dynamic_update_slice` — KV cache update
- `stablehlo.compare` + `stablehlo.select` — causal masking
- `stablehlo.iota` — index generation for masks

These are needed for Phase 4 completion but not for the core compute blocks which are now verified.

---

## Phase 4 continued: Transformer decode ops (April 26, 2026)

### Second round of "inspect first"

Wrote `inspect_transformer_decode.py` covering MHA with reshapes, KV cache (slice/scatter), causal masking (tril → iota+compare+select), embedding lookup (gather), argmax, and a full tiny decode step. Found 22 unique ops total, 14 already supported. 8 missing.

Wrote `inspect_bytecode_new_ops.py` to verify the **bytecode format** for each new op — critical since bytecode format can differ from `as_text()`. All new ops matched the `as_text()` format except the portable artifact round-trip failed for one case (MLIR attribute index error) but worked fine via the portable artifact deserializer path.

### Implemented 6 new ops

1. **slice** — `stablehlo.slice %arg0 [0:1, 0:4, 0:8, 0:16]` — static slicing with start:limit:stride per dimension. Maps to `a[tuple(slice(s, l, st))]`.
2. **compare** — `stablehlo.compare GT, %a, %b, FLOAT` — element-wise comparison (GT/LT/GE/LE/EQ/NE) returning boolean tensor.
3. **select** — `stablehlo.select %pred, %true, %false` — ternary element-wise selection. Maps to `np.where`.
4. **iota** — `stablehlo.iota dim = 0 : tensor<4x4xi32>` — sequential indices along a dimension. Used for causal mask generation (tril pattern).
5. **concatenate** — `stablehlo.concatenate %a, %b, dim = 1` — concatenation along a dimension.
6. **and/or** — boolean logic ops for compound predicates.

### Fixed batched dot_general

Multi-head attention uses `dot_general` with `batching_dims`, which previously called a non-existent `np.einsum_dot_general`. Fixed by building an einsum string dynamically: assign letters to batch dims (shared), contracting dims (shared), and free dims (unique), then call `np.einsum(subscripts, a, b)`. This generalized approach handles all dot_general patterns.

### Zero-argument functions

`jnp.arange(8)` produces a StableHLO module with a zero-argument function — no `^bb0(...)` entry block. The parser didn't handle this, silently producing no outputs. Fixed by detecting `"func.func"() ... ({` lines and starting a function context with empty args.

### Test suite unification

Removed `JAX_PLATFORMS=cpu` from test_engine.py (was poisoning the process for PJRT tests). Instead: (1) use `np.ones(...)` instead of `jnp.ones(...)` for example args (avoids device placement), (2) force CPU lowering in `get_bytecode` via `jax.default_device(cpu)`. All tests now run together: 72 pass, 2 skip.

### Current state: 72/74 tests pass

- Engine: 38 tests (was 29) — slice, compare, select, iota, concatenate, and, tril, MHA
- PJRT pipeline: 23 tests (was 18) — slice, where, tril, concatenate, multi-head attention
- Buffer: 6 tests
- Device: 5 tests
- Skipped: 2 (test_matmul.py stubs)

### What's still missing for full transformer decode

Two complex ops that use body regions or generic MLIR syntax:

1. **stablehlo.scatter** — KV cache update (`cache.at[pos].set(value)`). Uses `"stablehlo.scatter"(...)` generic format with a body region. Needed for in-place KV cache updates.

2. **stablehlo.gather** — Embedding lookup (`table[token_ids]`). Uses `"stablehlo.gather"(...)` generic format with dimension_numbers attribute. Needed for token → embedding mapping.

3. **Multi-output C++ support** — `plugin.cc` has `num_outputs` hardcoded to 1. Decode step returns `(out, k_cache, v_cache)` — 3 outputs. Need to parse output count from StableHLO return statement.

4. **argmax** — Greedy decoding uses a complex reduce with body region (not `applies` shorthand). Has 2 inputs, 2 init values, and a multi-op body (compare + select + and + or). This is significantly more complex than our current reduce parser.

### Priorities (now resolved)

All items completed:
1. ~~scatter + gather~~ — Done. Multi-line parser fix (move accumulator before ^bb0 detection).
2. ~~Multi-output support~~ — Done. Python engine handles arbitrary output counts.
3. ~~argmax~~ — Done. Multi-output reduce with reducer body → np.argmax.

---

## 2026-04-26: Phase 4 complete — all transformer decode ops working

### Scatter multi-line parser fix

The original parser used function attributes (`parse_stablehlo._pending`) to track multi-line accumulation. The inner `^bb0(%arg2, %arg3):` from scatter's body region was processed before the accumulator, overwriting `current_func` with wrong args. Fix: move accumulator check to be the FIRST thing in the loop, before `^bb0` detection. Also switched to local variable instead of function attributes.

### Brace counting upgrade

The accumulator originally tracked `({` and `})` for scatter's body regions. Multi-output reduce uses plain `{`/`}` without parentheses. Upgraded to count all `{`/`}` characters. Added `seen_open` flag to prevent premature termination when the body starts on the next line (reduce's first line has zero braces, body `{` is on line 2).

### Multi-output reduce (argmax)

JAX compiles `jnp.argmax` as a dual reduce: `%1:2 = stablehlo.reduce(values init: -inf), (indices init: 0)`. The `:2` suffix means 2 outputs. References use `%1#0` (max value) and `%1#1` (argmax index). Rather than implementing a general reducer body interpreter, we pattern-match this as argmax and use `np.argmax`.

Parser changes: (1) SSA regex handles `%name:N`, (2) return parser handles `%name#N`, (3) multi-output results stored as `name#0`, `name#1` in value dict.

### Full decode step validated

The test_decode_step test runs the complete transformer decode loop through the engine:
RMS norm → QKV projection → split heads → KV cache scatter update → slice active KV → batched dot_general attention → softmax → output projection → residual → MLP → residual.
Returns 3 outputs (hidden, new_k_cache, new_v_cache). All verified against JAX CPU reference.

### Current state: 44/44 engine tests pass

All 22 unique StableHLO ops needed for transformer inference are implemented.

### Remaining for end-to-end PJRT demo

1. ~~**Multi-output C++ support**~~ — Done. `count_outputs()` helper called from C++ Compile.
2. **Rebuild .so** on remote — scatter/gather/argmax only work in engine tests. Need to rebuild C++ plugin and verify through full PJRT pipeline (jax.jit → C++ → Python engine → result).
3. ~~**Phase 5: ttnn**~~ — In progress. See below.

---

## 2026-04-27: Phase 5 — Moving from numpy to ttnn (Blackhole)

### The key architectural decision: Python-side device management

We faced three approaches for getting computation onto the Blackhole:

**A. Change engine.py internally** — keep `execute_stablehlo(bytecode, numpy_inputs) → numpy_outputs` interface, but internally convert to ttnn tensors, run on device, convert back.

**B. Change C++ to manage ttnn buffers** — link ttnn in CMake, use `ttnn::from_torch()` in `BufferFromHostBuffer`, pass device tensors to engine.

**C. Full compiler** — StableHLO → TTIR → TTNN flatbuffer. The "right" approach, 3-6 months.

We chose **A**. Reasoning:
1. **Zero C++ changes**. The thin shell stays thin. All iteration happens in Python.
2. **Same interface**. C++ calls `execute_stablehlo(bytecode, inputs)` exactly as before. Returns numpy. All existing tests work unchanged.
3. **Python can import ttnn directly** — the remote host has ttnn installed. No CMake linking hell.
4. **Validated by applejax** — their Metal PJRT plugin uses the same pattern: C++ ABI adapter, Python/Swift does the real work.

The cost: extra host↔device copies per Execute call (numpy → ttnn → execute → ttnn → numpy). For Phase 6, we'd move buffer management to C++ to keep data on device between calls. But for correctness validation, the extra copies are fine.

### Dual-mode design

Added `TT_PJRT_USE_DEVICE=1` env var to toggle between numpy (CPU) and ttnn (device) execution. Default is numpy for backward compatibility. All 44 existing tests pass unchanged.

The dispatch is clean: `execute_op()` calls either `_execute_op_numpy()` or `_execute_op_device()`. The numpy path is identical to the old code. The device path maps each StableHLO op to its ttnn equivalent.

### Op migration tiers

Organized ops into tiers by complexity:

**Tier 1 (direct ttnn):** add, sub, mul, div, max, min, neg, exp, log, tanh, rsqrt, sqrt — all have 1:1 ttnn equivalents. `np.add(a,b)` → `ttnn.add(a,b)`.

**Tier 2 (shape ops):** reshape → `ttnn.reshape`, transpose → `ttnn.permute`, broadcast → `ttnn.repeat`. Convert is identity (bf16 throughout). Constants generate on CPU then `_to_device()`.

**Tier 3 (matmul):** Simple and batched matmuls → `ttnn.matmul`. Complex dot_general (non-standard contraction axes) → CPU fallback with einsum.

**Tier 4 (CPU roundtrip):** slice, scatter, gather, iota, argmax, and/or — no good ttnn equivalents. Pattern: `_from_device()` → numpy op → `_to_device()`. This is the same approach as `experiments/tt_jax/ops.py`.

**Key insight from ops.py:** divide maps to `ttnn.mul(a, ttnn.reciprocal(b))`, not a native divide. `max(x, 0)` should map to `ttnn.relu` — haven't added that pattern-match yet but it's a future optimization.

### What went right

1. **Reusing ops.py patterns** — every ttnn mapping was already proven in experiments/tt_jax/ops.py. No guessing.
2. **Clean separation** — `_execute_op_numpy` and `_execute_op_device` are parallel code paths. Easy to diff, easy to debug.
3. **All 44 numpy tests pass immediately** — the refactor from `execute_op` → `_execute_op_numpy` + `_execute_op_device` was mechanical.

### What I'm worried about

1. **bf16 precision**: ttnn runs in bf16. Our tests compare against float32 numpy. Softmax with large logits, layer norm variance, attention scores — all sensitive to precision. Need to widen tolerances carefully.
2. **Tile alignment**: ttnn requires 32x32 tiles. Small test tensors (e.g., `[4]`) become `[1, 32]` after padding. `_from_device` must handle unpadding correctly.
3. **Reduce shape semantics**: StableHLO reduce removes the reduced dimension. ttnn `sum(dim=, keepdim=True)` keeps it. Need reshape after reduce to match expected output shape.
4. **Device availability**: The tenstorrent kernel module doesn't auto-load after reboot. Blocked on `sudo modprobe tenstorrent`.

### Current state

- engine.py: dual-mode with all 22 ops migrated to ttnn paths
- test_engine.py: 44/44 pass (numpy mode, verified on remote host)
- test_engine_device.py: written, ready to run when device is available
- smoke_test_device.py: written, blocked on kernel module

### Next steps

1. Load kernel module → run smoke test → run device tests
2. Fix any device-specific failures (tile alignment, precision, shape mismatches)
3. Add trace capture (Step 6 in plan)
4. Benchmark: eager vs traced, compare to native ttnn experiments

---

## 2026-05-11: Phase 5 — First light on Blackhole (qb1)

### The bootstrap

The old `ssh tenstorrent` VM lost its Blackhole passthrough (friend disconnected it). New host `ssh qb1` — fresh Ubuntu 22.04 box with the tt-kmd driver loaded and 4 Blackhole chips visible. No Python deps, no tt-metal install.

Bootstrap (split into 5 scripts at `pjrt_plugin/scripts/qb1_phase_*.sh`):
- Phase A: filesystem prep, env vars in `.bashrc` (TTNN_CACHE_DIR=~/tt-xla/.cache/ttnn)
- Phase B: clone 17 Tenstorrent repos to `~/tenstorrent/`
- Phase C: `uv venv` + uv pip install jax==0.6.2 jaxlib==0.6.2 torch numpy pytest
- Phase D: `uv pip install ttnn==0.69.0` — major finding, ttnn ships a PyPI wheel that matches the source release tag exactly. Skipped the 30-60 min source build.
- Phase E: rsync project to qb1

Total bootstrap time: ~8 minutes (would have been ~50 minutes if we'd built tt-metal from source).

### First-light results

```
=== ALL SMOKE TESTS PASSED ===
23 passed in 17.93s
```

`test_engine_device.py` runs 23 tests covering:
- 11 elementwise ops (add, sub, mul, div, neg, exp, log, tanh, rsqrt, sqrt, max)
- 3 shape ops (reshape, transpose, broadcast)
- 2 matmul (simple + batched dot_general)
- 2 reduce (sum, max)
- slice, compare, concatenate
- 2 composite (softmax, linear layer)

All pass with `atol=0.05, rtol=0.05` (widened for bf16) on the Blackhole, end-to-end through our dispatch path: numpy → `_to_device` → ttnn op → `_from_device` → numpy.

### What surprised me

1. **JIT cache cold start is brutal.** First test takes ~3 seconds. The kernel JIT pipeline is `mean=489ms` per build, 91 builds total. Once warm, ops run in ms. This is exactly the dispatch-wall problem we ran into in experiments 90-99 — and exactly what trace capture is for.
2. **The PyPI wheel includes pre-compiled firmware.** `BuildKernels` log: `Using pre-compiled firmware from: /home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/tt_metal/pre-compiled/...`. We didn't need our `~/tenstorrent/tt-metal` source at all for this run — pure wheel install. The source clone is now reference material.
3. **ttnn opens all 4 chips for topology discovery** even when we only ask for device 0. `MeshDevice(1x1 grid, 1 devices)` confirms the mesh is single-device, but UMD still touches all 4 to map the cluster. Per the "use device 0 only" rule, this is fine — we're just one chip in a discovered cluster.

### What's broken / dirty

1. **ttnn config still has `tmp_dir=/tmp/ttnn`.** TTNN_CACHE_DIR redirects model_cache, but `tmp_dir` is a separate setting we haven't overridden yet. Need to find the env var or call `ttnn.CONFIG.tmp_dir = ...` in `engine.py`.
2. **bf16 precision drift visible but tolerable.** All tests pass with widened tolerances. Softmax in bf16 is within 2%, matmul within 5%. For decoder correctness we'll need to track this more carefully.
3. **Cold-start JIT.** First run pays 7s for kernel JIT. Subsequent runs use the kernel cache. We'll need to pre-warm or use trace capture to hide this.

### What this unlocks

- All future device work runs on qb1 — no more "blocked on hardware."
- The dual-mode engine works as designed: `TT_PJRT_USE_DEVICE=1` flips the switch.
- We can now move to **Phase 5 Step 6: trace capture** to attack the dispatch wall.
- We can also try the **full PJRT pipeline** (rebuild .so, run test_basic_ops.py with the device engine).

---

## 2026-05-11 (cont.): Full PJRT pipeline green on Blackhole

### End-to-end success

After rebuilding the .so on qb1, ran `test_basic_ops.py` in device mode (the canonical end-to-end PJRT test). Initial run: 12/27 pass, 15 fail — all bf16 precision. After fixing tolerances and four engine bugs (below): **27/27 pass**.

Three test suites all green on qb1:
- `test_engine.py` (numpy mode): 44/44 in 1.0s
- `test_engine_device.py` (engine direct → ttnn): 23/23 in 2.8s
- `test_basic_ops.py` (full PJRT pipeline → device): 27/27 in 3.4s

That's 94 tests covering: numpy CPU path, direct ttnn dispatch, and end-to-end `jax.jit(f)(x)` → C++ PJRT plugin → Python engine → ttnn → Blackhole → results back.

### The four engine bugs I had to fix

1. **`assert_close` infinite recursion.** My sed-replace `np.testing.assert_allclose` → `assert_close` caught the call inside the helper itself. RecursionError on every test. One-line fix.
2. **Plugin double-registration.** JAX auto-discovers `jax_plugins.tt` namespace AND our conftest fixture explicitly registers. The second registration throws `ALREADY_EXISTS`. Fix: tolerate it.
3. **broadcast_in_dim using ttnn shape, not StableHLO shape.** This was the deep one. `_to_device` unsqueezes 1D tensors to 2D minimum, so a StableHLO `tensor<4xf32>` (logical shape `(4,)`) becomes ttnn shape `(1, 4)`. Using `.shape` to compute the broadcast intermediate shape produces garbage. Fix: track LOGICAL shapes from the IR in a per-execution `_logical_shapes` dict, populated from func args, op result_types, and through `func.call` boundaries.
4. **`extract_result_type` chokes on multi-type result strings.** Ops like `stablehlo.select` emit `pred_type, val_type` after `:` (no `->`). My helper returned the whole blob; `parse_tensor_type` then raised ValueError. Fix: when there's no `->`, grab the LAST `tensor<...>` from the tail.

Bonus: indices for gather/scatter came back as floats from ttnn (we upload everything as bf16). numpy refuses to index with floats — cast to int64 in the device-mode gather/scatter wrappers.

### What I learned

1. **The impedance mismatch between StableHLO and ttnn is real.** ttnn assumes ≥2D tensors with 32×32 tile alignment. StableHLO has whatever rank/shape JAX produced. Bridging means tracking BOTH the logical shape (for IR semantics) and the device shape (for op dispatch). Doing one without the other gives subtle bugs.
2. **bf16 contaminates everywhere.** We upload weights, inputs, *and* integer indices as bf16. The indices have to be cast back at the CPU-roundtrip boundary. If we ever want fast on-device gather/scatter, we'll need to preserve integer types end-to-end, which means a dtype-aware upload path.
3. **Tolerance discipline.** Device mode hits the bf16 floor. A 128-deep matmul drifts ~5%. A scalar add doesn't drift at all. A blanket envelope hides real bugs (test_larger_matmul's `Max relative difference: 24.8` looks alarming but is the relative error on a single near-zero output entry). The right approach is mode-aware `max(test_tol, mode_floor)`.
4. **Dispatch wall confirmed at warm cache.** Engine-direct test_engine_device.py: 23 tests in 2.78s. Full PJRT pipeline (same compute through more C++ shims): 27 tests in 3.38s. The PJRT overhead is real but small — most of the time is in ttnn dispatch + host transfers. Trace capture is the next attack on this.

### What's still broken / known issues

1. `ttnn.CONFIG.tmp_dir` defaults to `/tmp/ttnn`. We override it in `engine.py` but the override fires only AFTER ttnn module init, which has already printed the default. Cosmetic — the override does take effect for actual writes.
2. The `Initial ttnn.CONFIG` debug line shows in every test run. Not actionable, just noise.
3. test_larger_matmul required atol=1.0 — bf16's 7-bit mantissa accumulates over a 128-deep contraction. Real Q/K matmuls in transformers are ~64-128 deep with similar drift expectations.

### Next: performance

We have correctness. Time to measure. Plan:
1. Microbenchmark the engine: time a single matmul end-to-end (eager). Compare to native ttnn.matmul without our engine.
2. Profile to find the dispatch/transfer/compute split.
3. Implement trace capture (Phase 5 Step 6) and re-measure.
4. Then op fusion (Phase 5 Step 7).
