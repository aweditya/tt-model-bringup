# PJRT Plugin Design for Tenstorrent Blackhole

## Date: April 2026

Research document for building a custom PJRT plugin that compiles JAX programs to run on Tenstorrent Blackhole via the ttnn C++ API. This is the interpretation approach (walk StableHLO, dispatch to ttnn), modeled after applejax.

---

## 1. PJRT Plugin Architecture Overview

### What PJRT Is

PJRT (Portable JAX Runtime) is the standard C API interface that connects ML frameworks (JAX, PyTorch/XLA) to hardware backends. It defines a function pointer table (`PJRT_Api` struct) with ~100 function pointers covering:

- Plugin lifecycle (initialize, attributes)
- Client management (create, destroy, platform info)
- Device discovery (enumerate, describe, memory spaces)
- Buffer management (host-to-device, device-to-host, delete)
- Compilation (StableHLO module in, executable out)
- Execution (run executable with buffer arguments)
- Event/async management

### How a Plugin Works End-to-End

```
1. JAX discovers plugin via jax_plugins/ namespace package
2. Plugin's initialize() calls xb.register_plugin("tt", library_path="pjrt_plugin_tt.so")
3. JAX calls GetPjrtApi() -> returns PJRT_Api* function pointer table
4. JAX calls PJRT_Client_Create -> plugin creates client, discovers devices
5. User writes: y = jax.jit(f)(x)
6. JAX lowers f to StableHLO MLIR bytecode
7. JAX calls PJRT_Client_Compile(stablehlo_module) -> plugin parses module, returns executable
8. JAX calls PJRT_Client_BufferFromHostBuffer(x) -> plugin copies x to device memory
9. JAX calls PJRT_LoadedExecutable_Execute(executable, [buffer]) -> plugin runs on device
10. JAX calls PJRT_Buffer_ToHostBuffer(result) -> plugin copies result back to host
```

### The Two Implementation Approaches

| Approach | Description | Used By |
|----------|-------------|---------|
| **C API directly** | Implement PJRT_Api function pointers as C functions | Tenstorrent tt-xla (via IREE-PJRT fork) |
| **C++ classes + wrapper** | Inherit PjRtClient/PjRtDevice/PjRtBuffer, use XLA's C-to-C++ wrapper | Intel, AMD (requires building against XLA repo) |

We should use the **C API directly**, same as applejax and tt-xla. This avoids linking against XLA's massive codebase -- we only need the PJRT C API header and MLIR/StableHLO libraries for parsing.

---

## 2. applejax Architecture Deep Dive

applejax (v0.9.7, March 2026) is our closest template. It implements a PJRT plugin for Apple Metal via interpretation of StableHLO.

### Source Structure

```
applejax/
  src/
    jax_plugins/mps/
      __init__.py              # Plugin registration (initialize(), library discovery)
    pjrt_plugin/
      pjrt_api.cc              # PJRT_Api struct with ~100 function pointers
      mps_client.h/mm          # Client: device management, buffer creation, compilation
      mps_executable.h/mm      # StableHLO parsing, execution plan, op dispatch
      ops/
        registry.h             # Op registration macros and dispatch table
        binary_ops.mm          # add, sub, mul, div, dot, dot_general, compare, select
        unary_ops.mm           # sin, cos, exp, log, tanh, sqrt, rsqrt, erf, ...
        shape_ops.mm           # gather, scatter, reshape, pad, transpose, concatenate
        reduction_ops.mm       # sum, prod, max, min, argmax, cumsum
        tensor_creation_ops.mm # constant, iota
        bitwise_ops.mm         # and, or, xor, shifts
        control_flow_ops.mm    # cond, while_loop, scan
        convolution_ops.mm     # 1D, 2D, depthwise, transposed
        fft_ops.mm             # fft, ifft, rfft
        sort_ops.mm            # sort, argsort, top_k
        linalg_ops.mm          # cholesky, triangular_solve, eig
  scripts/
    setup_deps.sh              # Builds LLVM/MLIR/StableHLO (~30 min)
  pyproject.toml               # scikit-build-core, jax_plugins entry point
  CMakeLists.txt               # Links MLIR, StableHLO, protobuf, abseil
```

### Key Design Patterns

**1. Op Registration via Static Initialization**

```cpp
// registry.h defines:
// OpHandler = function pointer returning ProcessResult
// OpRegistry = static unordered_map<string, OpHandler>

// In binary_ops.mm:
REGISTER_MLIR_BINARY_OP("stablehlo.add", addition, add);
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// The macro expands to:
static bool _reg_add = OpRegistry::Register("stablehlo.add", GraphOpHandler<HandleAdd>);
```

**Our equivalent would be:**
```cpp
REGISTER_TT_OP("stablehlo.add", HandleAdd);
// -> OpRegistry::Register("stablehlo.add", [](HandlerContext& ctx) {
//     auto a = ctx.GetInput(0); auto b = ctx.GetInput(1);
//     return ttnn::add(a, b);
// });
```

**2. Execution Plan (Lazy Compilation)**

On first `Execute()`, `MpsExecutable::BuildExecutionPlan()`:
1. Walks the StableHLO module's entry function
2. Partitions ops into segments (graph ops vs native/CPU ops)
3. For graph segments: builds an MPSGraph computation graph
4. For native segments: records handler + input/output slots
5. Caches the plan for subsequent executions

**Our equivalent**: Walk StableHLO ops, dispatch to ttnn. On first execution, optionally capture a ttnn trace for subsequent fast replay.

**3. Buffer Management**

- `BufferFromHostBuffer()`: allocates MTLBuffer, copies host data
- Buffers track dtype, shape, device assignment
- `ToHostBuffer()`: copies device data back to host
- Reference counting for lifecycle management

**Our equivalent**: `ttnn::Tensor` objects with `ttnn::from_torch()` for host-to-device and `tensor.cpu()` for device-to-host.

**4. Compilation = Parsing**

applejax does NOT compile StableHLO to an optimized representation. `Compile()` simply parses the MLIR bytecode into an `mlir::ModuleOp` and wraps it in an `MpsExecutable`. Actual op dispatch happens at execution time.

This is the interpretation approach: parse once, dispatch per-execution. With ttnn trace capture, we can get compilation-like performance: dispatch once, replay the trace on subsequent calls.

---

## 3. PJRT C API: Required Functions

### Minimum Viable Set (7 functions)

These are the functions JAX actually calls during basic usage:

| Function | Purpose | Complexity |
|----------|---------|------------|
| `PJRT_Client_Create` | Create client, discover devices | Medium -- initialize ttnn device |
| `PJRT_Client_BufferFromHostBuffer` | Copy host data to device | Medium -- ttnn::from_torch() |
| `PJRT_Client_Compile` | Parse StableHLO module | Medium -- MLIR parsing |
| `PJRT_LoadedExecutable_Execute` | Run on device | High -- op dispatch loop |
| `PJRT_Buffer_ToHostBuffer` | Copy device data to host | Low -- tensor.cpu() |
| `PJRT_Buffer_Delete` | Free device memory | Low -- deallocate tensor |
| `PJRT_LoadedExecutable_Delete` | Free executable | Low -- cleanup |

### Required Boilerplate (~30 functions)

These are simple metadata/lifecycle functions JAX queries during setup:

- **Error API** (3): `Error_Destroy`, `Error_Message`, `Error_GetCode`
- **Client info** (5): `PlatformName`, `PlatformVersion`, `ProcessIndex`, `Devices`, `AddressableDevices`
- **Device info** (8): `DeviceDescription_Id`, `_Kind`, `_DebugString`, `_ToString`, `_ProcessIndex`, `Device_IsAddressable`, `_LocalHardwareId`, `_MemorySpaces`
- **Memory info** (4): `Memory_Id`, `_Kind`, `_DebugString`, `_AddressableByDevices`
- **Executable metadata** (7): `Executable_Name`, `_NumOutputs`, `_SizeOfGeneratedCodeInBytes`, `_Destroy`, `LoadedExecutable_AddressableDevices`, `_Fingerprint`, `_Destroy`
- **Event API** (5): `Event_Destroy`, `_IsReady`, `_Error`, `_Await`, `_OnReady`

### Functions We Can Stub (set to nullptr)

- `PJRT_Client_CreateViewOfDeviceBuffer` -- DLPack interop
- `PJRT_Buffer_CopyToMemory` -- cross-device copy
- `PJRT_Buffer_Bitcast` -- reinterpret buffer type
- `PJRT_Executable_Serialize/Deserialize` -- executable caching
- `PJRT_TopologyDescription_*` -- multi-host topology
- `GetCompiledMemoryStats` -- memory profiling
- All DMA/async management functions

### The Full PJRT_Api Struct

applejax sets ~80 of ~100+ function pointers. Of those, ~50 are trivial metadata returns. The real implementation work is in the 7 core functions above.

---

## 4. StableHLO Ops: What JAX Programs Generate

### Ops for a Simple Linear Layer: `y = x @ W + b`

```
stablehlo.constant     # W, b values
stablehlo.dot_general  # matrix multiply
stablehlo.add          # bias addition
stablehlo.broadcast_in_dim  # broadcast b to match output shape
```

### Ops for a Full Transformer Decode Step

Based on our Jaxpr interpreter's 28 ops, the StableHLO equivalents are:

| Our Jaxpr Op | StableHLO Equivalent | ttnn Implementation |
|-------------|---------------------|---------------------|
| add | stablehlo.add | ttnn::add |
| sub | stablehlo.subtract | ttnn::subtract |
| mul | stablehlo.multiply | ttnn::multiply |
| div | stablehlo.divide | ttnn::divide |
| neg | stablehlo.negate | ttnn::neg |
| exp | stablehlo.exponential | ttnn::exp |
| log | stablehlo.log | ttnn::log |
| sqrt | stablehlo.sqrt | ttnn::sqrt |
| rsqrt | stablehlo.rsqrt | ttnn::rsqrt |
| reciprocal | stablehlo.custom_call | ttnn::reciprocal |
| max | stablehlo.maximum | ttnn::maximum |
| tanh | stablehlo.tanh | ttnn::tanh |
| ge | stablehlo.compare (GE) | ttnn::ge |
| select_n | stablehlo.select | ttnn::where |
| dot_general | stablehlo.dot_general | ttnn::matmul / ttnn::linear |
| reduce_max | stablehlo.reduce (max) | ttnn::max |
| reduce_sum | stablehlo.reduce (sum) | ttnn::sum |
| broadcast_in_dim | stablehlo.broadcast_in_dim | ttnn::repeat / reshape |
| reshape | stablehlo.reshape | ttnn::reshape |
| transpose | stablehlo.transpose | ttnn::permute |
| squeeze | stablehlo.reshape | ttnn::reshape |
| slice | stablehlo.slice | ttnn::slice |
| dynamic_slice | stablehlo.dynamic_slice | ttnn::slice (with computed indices) |
| concatenate | stablehlo.concatenate | ttnn::concat |
| split | stablehlo.slice (multiple) | ttnn::slice |
| gather | stablehlo.gather | ttnn::embedding |
| iota | stablehlo.iota | manual construction |
| convert_element_type | stablehlo.convert | ttnn::typecast |
| integer_pow | stablehlo.custom_call or power | ttnn::pow |
| stop_gradient | (no-op) | passthrough |

### Minimum Viable Op Set for `jax.jit(lambda x: x @ W + b)(x)`

5 ops: `constant`, `dot_general`, `add`, `broadcast_in_dim`, `convert`

### Minimum Viable Op Set for a Transformer Forward Pass

~20 ops: the 28 above minus some that XLA may optimize away, plus a few XLA may introduce:
- `stablehlo.reduce` (general form, needs body region parsing)
- `stablehlo.while` (autoregressive generation loop, if traced through jax.jit)
- `stablehlo.dynamic_update_slice` (KV cache updates)

### applejax's 71 StableHLO Ops

For reference, applejax implements 71 StableHLO ops + 12 CHLO ops across 2000+ tests. This covers the vast majority of JAX programs. We would start with ~20 and expand as needed.

---

## 5. Build System Requirements

### Dependencies

| Dependency | Purpose | How to Get |
|------------|---------|-----------|
| **LLVM/MLIR** | Parse StableHLO MLIR bytecode | Build from source (~30 min) via setup_deps.sh |
| **StableHLO** | StableHLO dialect definitions | Build from source (part of setup_deps.sh) |
| **Abseil** | String utilities, logging | Build from source (part of setup_deps.sh) |
| **Protobuf** | Serialization (used by PJRT) | Build from source (part of setup_deps.sh) |
| **XLA headers** | PJRT C API header (pjrt_c_api.h) | Download from XLA repo (header-only) |
| **tt-metal/ttnn** | Tenstorrent runtime + ops | Already installed on remote host |

### Version Pinning Strategy

applejax pins exact commits for LLVM, StableHLO, and XLA, derived from the jaxlib version it targets:

- LLVM: `f6d0a512972a74ef100723b9526a6a0ddb23f894`
- StableHLO: `127d2f238010589ac96f2f402a27afc9dccbb7ab`
- XLA: `bb760b047bdbfeff962f0366ad5cc782c98657e0`
- Abseil: `20250127.0`
- Protobuf: `29.3`

We must determine the correct commits for our target jaxlib version. The process:
1. Check which jaxlib is installed: `pip show jaxlib`
2. Find the XLA commit jaxlib was built from: `python -c "import jaxlib; print(jaxlib.__version__)"`
3. Look up that XLA commit's LLVM and StableHLO pins in `xla/third_party/llvm/workspace.bzl` and `xla/third_party/stablehlo/workspace.bzl`

### CMake Build Structure

```cmake
cmake_minimum_required(VERSION 3.20)
project(pjrt_plugin_tt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find MLIR/LLVM (built by setup_deps.sh)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Find StableHLO
find_path(STABLEHLO_INCLUDE_DIR stablehlo/dialect/StablehloOps.h
    PATHS ${CMAKE_PREFIX_PATH}/include)

# Find tt-metal/ttnn
# tt-metal is typically installed at /opt/tt-metal or via TT_METAL_HOME env var
set(TT_METAL_HOME $ENV{TT_METAL_HOME})
find_path(TTNN_INCLUDE_DIR ttnn/ttnn.hpp
    PATHS ${TT_METAL_HOME}/ttnn)

# Build the shared library
add_library(pjrt_plugin_tt SHARED
    pjrt_api.cc
    tt_client.cc
    tt_executable.cc
    tt_buffer.cc
    ops/registry.cc
    ops/binary_ops.cc
    ops/unary_ops.cc
    ops/shape_ops.cc
    ops/reduction_ops.cc
    ops/tensor_creation_ops.cc
)

target_link_libraries(pjrt_plugin_tt PRIVATE
    # MLIR/StableHLO (static libs)
    StablehloOps StablehloSerialization StablehloPasses
    MLIRParser MLIRBytecodeReader MLIRFuncDialect MLIRIR
    MLIRSupport MLIRTransforms
    LLVMSupport
    protobuf::libprotobuf
    absl::strings absl::log absl::synchronization
    # ttnn (shared lib)
    ttnn  # or link path to libttnn.so
)
```

### Build on Remote Host

The remote Tenstorrent host has:
- Ubuntu (likely 22.04 or 24.04)
- Python 3.x with tt-metal/ttnn installed
- GCC/Clang available

We need to:
1. Adapt applejax's `setup_deps.sh` to build MLIR/StableHLO on Linux (it currently targets macOS)
2. Remove all Apple/Metal framework references
3. Add ttnn include paths and link flags
4. Build with `cmake -B build && cmake --build build`

### Python Package Structure

```
jax_plugins/
  tt/
    __init__.py          # initialize() -> xb.register_plugin("tt", library_path=...)
    libpjrt_plugin_tt.so # compiled C++ plugin
```

```python
# jax_plugins/tt/__init__.py
import os
import jax._src.xla_bridge as xb

def initialize():
    path = os.path.join(os.path.dirname(__file__), 'libpjrt_plugin_tt.so')
    xb.register_plugin('tt', priority=500, library_path=path, options=None)
```

```toml
# pyproject.toml
[project.entry-points."jax_plugins"]
tt = "jax_plugins.tt"
```

---

## 6. Key Design Decisions and Trade-offs

### Decision 1: Interpretation vs Compilation

**Choice: Interpretation (walk StableHLO, dispatch to ttnn)**

- Compilation (extend XLA or use tt-mlir) requires a compiler team we don't have
- Interpretation is what applejax does and it works: 71 ops, 2000+ tests, 3x speedup
- We can add ttnn trace capture on top for near-compilation performance
- We already have 28 op handlers in Python; porting to C++ is mechanical

### Decision 2: C API vs C++ API

**Choice: C API directly**

- C++ API requires linking against XLA's codebase (massive)
- C API only needs the pjrt_c_api.h header
- applejax and tt-xla both use the C API approach
- We define our own C++ classes (TtClient, TtBuffer, etc.) and wire them to C function pointers

### Decision 3: Eager Dispatch vs Graph Caching

**Choice: Eager dispatch with trace capture**

- Phase 1: Parse StableHLO, walk ops, call ttnn immediately (like applejax)
- Phase 2: Add trace capture -- on first Execute(), record ttnn ops via `ttnn::begin_trace_capture()`. On subsequent calls with same shapes, replay the trace. This gives us compilation-level performance without a compiler.

This is a significant advantage over applejax. Apple's MPSGraph provides graph-level batching, but ttnn trace capture is even better: it records the exact sequence of kernel launches, memory allocations, and data movement, then replays them with zero dispatch overhead.

### Decision 4: Single-Device vs Multi-Device

**Choice: Single-device only (device 0)**

- Our hardware is one Blackhole P150
- Multi-device adds massive complexity (sharding, all-reduce, device placement)
- Start with hardcoded device 0, expand later if needed

### Decision 5: Supported Dtypes

**Choice: bfloat16 primary, float32 for host transfers**

- Blackhole hardware is optimized for bfloat16
- Host-to-device: accept float32, convert to bfloat16 on device
- Device-to-host: convert bfloat16 to float32 for host consumption
- Support float32 on device for accumulation where needed

### Decision 6: Memory Management

**Choice: Simple allocation, no memory pools initially**

- Allocate ttnn tensors on L1 or DRAM as ttnn decides
- Let ttnn handle memory placement (it has good heuristics)
- Track tensor lifecycle via PJRT buffer reference counting
- No custom allocator or memory pool in Phase 1

---

## 7. Implementation Plan: Phased Approach

### Phase 1: Skeleton (1 week)

**Goal**: `jax.devices()` shows a Tenstorrent device.

Files to create:
- `pjrt_api.cc` -- PJRT_Api struct with function pointers, GetPjrtApi()
- `tt_client.h/cc` -- TtClient class: device init, platform info
- `tt_device.h/cc` -- TtDevice class: metadata, memory space
- `tt_buffer.h/cc` -- TtBuffer class: stub
- `tt_executable.h/cc` -- TtExecutable class: stub
- `jax_plugins/tt/__init__.py` -- plugin registration

Implement:
- `PJRT_Client_Create` -> open ttnn device 0
- All metadata functions (platform name/version, device enumeration)
- `PJRT_Error_*` functions
- `PJRT_Event_*` functions (synchronous -- events are always ready)
- Stub everything else to return "unimplemented"

**Test**: `python -c "import jax; print(jax.devices())"`
Expected: `[TtDevice(id=0)]`

**Estimated lines**: ~800 C++, ~30 Python

### Phase 2: Buffer Transfer (3-5 days)

**Goal**: `jax.device_put(x, tt_device)` and `jax.device_get(buffer)` work.

Implement:
- `PJRT_Client_BufferFromHostBuffer`:
  - Accept host pointer + dtype + shape
  - Create numpy array from raw pointer
  - Convert to torch tensor
  - Call `ttnn.from_torch(tensor, device=device, dtype=ttnn.bfloat16)`
  - Wrap in TtBuffer
- `PJRT_Buffer_ToHostBuffer`:
  - Call `buffer.cpu()` to get host tensor
  - Convert to numpy
  - Copy to output pointer
- `PJRT_Buffer_Delete`: deallocate ttnn tensor
- Buffer metadata: dtype, shape, device

**Test**: Round-trip host data through device.
```python
x = jnp.ones((4, 4))
y = jax.device_put(x, jax.devices('tt')[0])
z = jax.device_get(y)
assert np.allclose(x, z, atol=1e-2)  # bfloat16 precision
```

**Estimated lines**: ~400 C++

### Phase 3: MVP Execution (1-2 weeks)

**Goal**: `jax.jit(lambda x: x @ W + b)(input)` runs on Tenstorrent.

Implement:
- `PJRT_Client_Compile`:
  - Receive StableHLO MLIR bytecode
  - Parse with `mlir::parseSourceString` or `mlir::readBytecodeFile`
  - Extract entry function
  - Wrap in TtExecutable (lazy -- don't walk ops yet)
- `PJRT_LoadedExecutable_Execute`:
  - Walk the StableHLO entry function's operations
  - For each op, look up handler in registry
  - Call handler with input ttnn tensors
  - Return output buffers
- Op handlers (5 minimum):
  - `stablehlo.constant` -> create ttnn tensor from literal data
  - `stablehlo.dot_general` -> ttnn::matmul
  - `stablehlo.add` -> ttnn::add
  - `stablehlo.broadcast_in_dim` -> ttnn::repeat / reshape
  - `stablehlo.convert` -> ttnn::typecast

**Test**:
```python
W = jnp.ones((4, 4))
b = jnp.ones((4,))
y = jax.jit(lambda x: x @ W + b)(jnp.ones((2, 4)))
# y should be [[5, 5, 5, 5], [5, 5, 5, 5]]
```

**Estimated lines**: ~1200 C++ (executable + 5 ops + registry)

### Phase 4: Transformer Ops (1 week)

**Goal**: Full set of ops for transformer inference.

Port remaining ops from our Python interpreter to C++:
- Elementwise: sub, mul, div, neg, exp, log, sqrt, rsqrt, reciprocal, max, tanh
- Comparison/selection: compare (ge/gt/le/lt/eq/ne), select
- Shape: reshape, transpose, slice, dynamic_slice, concatenate, gather
- Reduction: reduce_max, reduce_sum
- Other: iota, integer_pow (via repeated multiply)

Each op handler is 10-40 lines of C++. Total: ~20 ops x ~25 lines = ~500 lines.

**Test**: Run our Qwen-0.5B or Llama-8B model through `jax.jit`.

**Estimated lines**: ~500 C++

### Phase 5: Trace Capture (3-5 days)

**Goal**: Second execution of any jit'd function is fast (trace replay).

Implement:
- On first `Execute()` call for a given executable + input shapes:
  - Call `ttnn::begin_trace_capture(device, cq_id)`
  - Run the op dispatch loop
  - Call `ttnn::end_trace_capture(device, trace_id, cq_id)`
  - Cache the trace_id keyed by (executable_id, input_shapes)
- On subsequent `Execute()` calls:
  - Look up cached trace_id
  - Copy input data into the trace's input buffers
  - Call `ttnn::execute_trace(device, trace_id, cq_id, blocking=false)`
  - Return the trace's output buffers

This should give us performance comparable to our Python trace capture (132 tok/sec on Qwen-0.5B).

**Estimated lines**: ~200 C++

### Phase 6: Polish and Testing (1 week)

- Error handling: catch ttnn exceptions, return PJRT errors
- Memory leak audit: ensure all buffers are freed
- Edge cases: zero-sized tensors, scalar values, unusual dtypes
- Integration tests: run JAX test suite against our backend
- Performance benchmarking: compare to our Python interpreter

---

## 8. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **MLIR/StableHLO build fails on remote host** | Blocks everything -- can't parse StableHLO without MLIR libs | Test build early (Day 1). If it fails, explore extracting MLIR libs from jaxlib wheel as alternative. |
| **ttnn C++ API differs from Python API** | Our op handlers won't work | We know ttnn has a C++ API (tt-train uses it). May need to use `tt_metal` lower-level API for some ops. Test with simple matmul first. |
| **StableHLO ops have unexpected structure** | Op handlers break on real JAX programs | Use `jax.jit(f).lower(x).as_text()` to inspect actual StableHLO before implementing handlers. XLA may fuse/transform ops in unexpected ways. |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **PJRT version mismatch** | Plugin crashes on load | Pin jaxlib version. Match PJRT_Api struct size exactly. |
| **Trace capture incompatible with PJRT** | Can't cache traces across Execute() calls | Trace capture needs stable input buffer addresses. May need to pre-allocate fixed buffers and copy inputs into them. |
| **reduce ops have body regions** | Need to parse reduction body (e.g., max of two elements) | Can hardcode common patterns (sum, max, min, prod) and fail on custom reductions initially. |
| **Dynamic shapes** | JAX may send different shapes across calls | Cache executables/traces per shape signature. Rebuild execution plan on shape change. |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Missing ops** | JAX program fails with "unimplemented op" | Good error message + incremental addition. Our 28 Jaxpr ops cover transformers. |
| **bfloat16 precision** | Numerical differences vs CPU | Already validated: all models >0.999 cosine similarity in our experiments. |
| **Single device limitation** | Can't do data parallelism | Not needed for our use case (single P150). |

---

## 9. Comparison: Our Plugin vs Alternatives

| Factor | Our PJRT Plugin | Python Interpreter (current) | Official tt-xla |
|--------|-----------------|------------------------------|-----------------|
| jax.jit support | Yes | No | Yes |
| vmap/grad support | Yes | No | Yes |
| Flax/Optax compat | Yes | No | Yes |
| Build complexity | High (MLIR deps) | None | Very high (tt-mlir) |
| Op count needed | ~20 for transformers | 28 (done) | All StableHLO |
| Performance | Good (trace capture) | Good (trace capture) | Best (compiler opt) |
| Debugging | Hard (C++ on remote) | Easy (Python) | Very hard (MLIR) |
| Maintenance | Medium (jaxlib pins) | Low | High (upstream) |
| Time to build | 3-5 weeks | Done | N/A (theirs) |
| Stability | Our control | Our control | 863 open issues |

---

## 10. StableHLO Parsing: Technical Details

### How to Parse a StableHLO Module

JAX sends StableHLO to the plugin as either:
1. MLIR bytecode (binary, via `mlir::readBytecodeFile`)
2. MLIR text (human-readable, via `mlir::parseSourceString`)

The plugin receives this in `PJRT_Client_Compile` as a `PJRT_Program` struct containing:
- `format`: string identifying the format (e.g., "mlir" or "stablehlo")
- `code`: byte buffer containing the module

Parsing requires:
```cpp
#include "mlir/Parser/Parser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "stablehlo/dialect/StablehloOps.h"

// In Compile():
mlir::MLIRContext context;
context.loadDialect<mlir::stablehlo::StablehloDialect>();
context.loadDialect<mlir::func::FuncDialect>();

auto module = mlir::parseSourceString<mlir::ModuleOp>(code_str, &context);
// or for bytecode:
auto module = mlir::readBytecodeFile(buffer_ref, &context);

auto entry_fn = module->lookupSymbol<mlir::func::FuncOp>("main");
```

### Walking Operations

```cpp
entry_fn.walk([&](mlir::Operation* op) {
    std::string op_name = op->getName().getStringRef().str();
    // op_name is e.g. "stablehlo.add", "stablehlo.dot_general"

    auto* handler = OpRegistry::Find(op_name);
    if (!handler) {
        return error("Unimplemented op: " + op_name);
    }

    HandlerContext ctx{op, value_map, device};
    auto result = handler->invoke(ctx);
    // Map op results to ttnn tensors for downstream ops
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
        value_map[op->getResult(i)] = result[i];
    }
});
```

### Extracting Op Parameters

StableHLO ops carry their configuration as MLIR attributes:

```cpp
// For stablehlo.dot_general:
auto dot_attrs = op->getAttrOfType<mlir::stablehlo::DotDimensionNumbersAttr>(
    "dot_dimension_numbers");
auto lhs_batch_dims = dot_attrs.getLhsBatchingDimensions();
auto lhs_contract_dims = dot_attrs.getLhsContractingDimensions();

// For stablehlo.broadcast_in_dim:
auto broadcast_dims = op->getAttrOfType<mlir::DenseIntElementsAttr>(
    "broadcast_dimensions");

// For stablehlo.reduce:
// The reduction body is a nested region with a block of ops
auto& body = op->getRegion(0).front();
// Parse body to determine reduction type (sum, max, etc.)
```

### Constant Handling

```cpp
// stablehlo.constant has a 'value' attribute containing the tensor data
auto const_attr = op->getAttrOfType<mlir::DenseElementsAttr>("value");
auto type = const_attr.getType();
auto shape = type.getShape();  // e.g., {4, 4}
auto dtype = type.getElementType();  // e.g., f32, bf16

// Extract raw data and create ttnn tensor
if (dtype.isF32()) {
    auto values = const_attr.getValues<float>();
    // -> ttnn::from_torch(torch::from_blob(data, shape, torch::kFloat32))
}
```

---

## 11. ttnn C++ API: What We Know

### From tt-train (the reference C++ consumer of ttnn)

tt-train's `CMakeLists.txt` shows how to link against ttnn from C++:
- Include path: `${TT_METAL_HOME}/ttnn/cpp`
- Link: `libttnn.so` (shared library)
- Must align 3rdparty dependencies via CPM
- Must set include paths manually

### Key ttnn C++ Functions We Need

```cpp
#include <ttnn/ttnn.hpp>

// Device management
auto device = ttnn::open_device(0);
ttnn::close_device(device);

// Tensor creation
auto tensor = ttnn::from_torch(torch_tensor, dtype, device);
auto host_tensor = tensor.cpu();

// Operations
auto c = ttnn::add(a, b);
auto c = ttnn::matmul(a, b);
auto c = ttnn::multiply(a, b);
auto c = ttnn::exp(a);
// ... all ops we use from Python are available in C++

// Trace capture
auto trace_id = ttnn::begin_trace_capture(device, cq_id);
// ... run ops ...
ttnn::end_trace_capture(device, trace_id, cq_id);
ttnn::execute_trace(device, trace_id, cq_id, blocking);
```

### Uncertainty: Python-Only APIs

Some ttnn features we use from Python may not have direct C++ equivalents:
- `ttnn.transformer.scaled_dot_product_attention` -- likely available in C++
- `ttnn.transformer.paged_update_cache` -- uncertain
- Memory config specifications -- should be available

We should verify by inspecting ttnn C++ headers on the remote host early.

---

## 12. Concrete First Steps

### Day 1: Environment Validation

```bash
ssh tenstorrent

# 1. Check what's installed
pip show jax jaxlib
python -c "import jaxlib; print(jaxlib.__version__)"

# 2. Check ttnn C++ headers
ls $TT_METAL_HOME/ttnn/cpp/ttnn/
find $TT_METAL_HOME -name "ttnn.hpp" -o -name "device.hpp" | head -10

# 3. Check compiler toolchain
g++ --version
cmake --version
clang++ --version

# 4. Test minimal C++ ttnn program
cat > /tmp/test_ttnn.cpp << 'EOF'
#include <ttnn/ttnn.hpp>
#include <iostream>
int main() {
    auto device = ttnn::open_device(0);
    std::cout << "Device opened successfully" << std::endl;
    ttnn::close_device(device);
    return 0;
}
EOF
# Attempt to compile and link
```

### Day 2: MLIR/StableHLO Build

Fork applejax's `setup_deps.sh`, strip macOS-specific parts, build on Linux.

### Day 3-5: Phase 1 Skeleton

Implement the minimal PJRT_Api struct, get `jax.devices()` working.

---

## 13. File Count and Size Estimates

| Component | Files | Lines (est.) | Complexity |
|-----------|-------|-------------|------------|
| pjrt_api.cc (function pointer table) | 1 | 300 | Low -- mostly struct init |
| tt_client.h/cc (device management) | 2 | 400 | Medium |
| tt_device.h/cc (device metadata) | 2 | 200 | Low |
| tt_buffer.h/cc (tensor wrapper) | 2 | 400 | Medium |
| tt_executable.h/cc (StableHLO walker) | 2 | 600 | High |
| ops/registry.h/cc | 2 | 150 | Low |
| ops/binary_ops.cc | 1 | 200 | Medium |
| ops/unary_ops.cc | 1 | 150 | Low |
| ops/shape_ops.cc | 1 | 300 | Medium |
| ops/reduction_ops.cc | 1 | 200 | Medium |
| ops/tensor_creation_ops.cc | 1 | 100 | Low |
| CMakeLists.txt | 1 | 80 | Medium |
| setup_deps.sh | 1 | 200 | Medium |
| jax_plugins/tt/__init__.py | 1 | 30 | Low |
| pyproject.toml | 1 | 30 | Low |
| **Total** | **20** | **~3,300** | |

---

## 14. Open Questions

1. **Can we extract MLIR libs from the jaxlib wheel instead of building from source?** jaxlib ships with MLIR inside. If we can link against those, we skip the 30-minute MLIR build entirely. Need to check if jaxlib exposes the needed symbols.

2. **Does ttnn's C++ API require torch?** Our Python code uses `ttnn.from_torch()`. The C++ equivalent may require libtorch linkage. Alternative: use `ttnn::Tensor` constructors directly with raw data pointers.

3. **How does JAX serialize StableHLO to the plugin?** Is it MLIR bytecode, text, or a protobuf? This determines which MLIR parsing function we call. Need to check `PJRT_Program.format` field.

4. **Can we run the setup_deps.sh build on the remote host?** The host may have limited disk space or build tools. Need to verify GCC/Clang, CMake, and ~5GB disk space for LLVM build.

5. **What happens with `stablehlo.reduce` body regions?** These contain nested ops (e.g., `stablehlo.add` for sum reduction, `stablehlo.maximum` for max reduction). We need to parse the body to determine the reduction type, or implement a general region interpreter.

6. **Will JAX's XLA optimizations change the StableHLO we receive?** JAX may run optimization passes before sending StableHLO to the plugin. This could introduce ops we don't expect or fuse ops in unexpected ways. Need to inspect actual StableHLO output with `jax.jit(f).lower(x).as_text()`.

---

## Sources

- [applejax GitHub](https://github.com/danielpcox/applejax) -- v0.9.7, our primary template
- [PJRT Plugin Integration Guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [PJRT C++ API Overview](https://openxla.org/xla/pjrt/cpp_api_overview)
- [PJRT C API Header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
- [tt-xla GitHub](https://github.com/tenstorrent/tt-xla) -- Tenstorrent's official PJRT plugin
- [tt-metal/ttnn GitHub](https://github.com/tenstorrent/tt-metal)
- [StableHLO Spec](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
- Our existing research: `research/jax_backend_2026_update.md`, `research/jax_infrastructure_2026.md`
- Our Jaxpr interpreter: `experiments/tt_jax/ops.py` (28 ops, working on 4 models)
