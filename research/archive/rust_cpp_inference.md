# Research: Rust/C++ Inference on Tenstorrent (Eliminating Python Dispatch)

Date: 2026-04-22

## Context

Our Qwen2.5-0.5B inference on Blackhole P150 shows:
- ~39us Python dispatch overhead per op
- 24 layers x ~15 ops/layer = ~360 ops = ~14ms pure dispatch overhead
- Trace capture eliminates this (7.6ms traced vs 21ms non-traced)
- But trace has limitations: no dynamic control flow, argmax is 90ms inside trace

Question: Can we eliminate Python entirely and dispatch from C++ or Rust?

---

## 1. TT-NN C++ API: Fully Accessible, First-Class

### Key Finding: `_ttnncpp.so` is a standalone C++ library

The tt-metal build system explicitly separates two targets:
- **`_ttnncpp.so`** (50MB) -- Pure C++ shared library, no Python dependency
- **`_ttnn.so`** (13MB) -- Python bindings (nanobind) that wrap ttnncpp

From `ttnn/CMakeLists.txt`:
```
# 1) Without python bindings
#    ttnncpp will be a dynamic library that can be linked to any project
# 2) With python bindings
#    ttnn will be a dynamic library with ttnncpp statically linked to it
```

Build target: `TTNN::CPP` (aliased as `TTNN::TTNN`)

### C++ API Mirrors Python Exactly

Every Python `ttnn.*` call maps 1:1 to a C++ function in the `ttnn::` namespace. The headers are in `ttnn/api/ttnn/` and `ttnn/cpp/ttnn/operations/`.

**Device management:**
```cpp
#include <ttnn/device.hpp>
auto device = ttnn::open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);
```

**Tensor creation + ops (from ttnn/examples/add/add.cpp):**
```cpp
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

auto input = ttnn::zeros(shape, DataType::BFLOAT16, TILE_LAYOUT, *device);
auto output = input + 3.0f;  // operator overloading works
```

**Matmul:**
```cpp
#include <ttnn/operations/matmul/matmul.hpp>

ttnn::Tensor result = ttnn::matmul(
    input_a, input_b,
    /*transpose_a=*/false, /*transpose_b=*/false,
    memory_config, dtype, program_config, activation,
    compute_kernel_config, core_grid);
```

**RMS Norm:**
```cpp
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>

ttnn::Tensor normed = ttnn::rms_norm(input, epsilon, weight, bias,
    residual, memory_config, program_config, compute_kernel_config);
```

**Softmax (with fused scale+mask for attention):**
```cpp
#include <ttnn/operations/normalization/softmax/softmax.hpp>

ttnn::Tensor out = ttnn::softmax(input, dim, memory_config, compute_kernel_config);
// Or fused: ttnn::scale_mask_softmax_in_place(input, scale, mask, ...);
```

**Embedding:**
```cpp
#include <ttnn/operations/embedding/embedding.hpp>

ttnn::Tensor out = ttnn::embedding(input_ids, weight, pad_token, layout, type, dtype, mem_config);
```

**Trace capture/replay (the exact same API we use from Python):**
```cpp
#include <ttnn/operations/trace.hpp>

auto tid = ttnn::operations::trace::begin_trace_capture(device, std::nullopt);
// ... run ops ...
ttnn::operations::trace::end_trace_capture(device, tid, std::nullopt);
ttnn::operations::trace::execute_trace(device, tid, std::nullopt, /*blocking=*/false);
ttnn::operations::trace::release_trace(device, tid);
```

### How to Build a C++ Program Against TT-NN

From the examples CMakeLists.txt:
```cmake
add_executable(example_add)
target_sources(example_add PRIVATE add/add.cpp)
target_link_libraries(example_add PRIVATE TTNN::CPP)
```

That's it. Link against `TTNN::CPP` and you get the full TT-NN API.

Dependencies at runtime: `_ttnncpp.so` -> `libtt_metal.so` (or .a), `libtt-umd.so`, `libtt_stl.so`, `libtracy.so`

### Existing C++ Test/Benchmark Suite

The tt-metal repo contains extensive C++ tests and benchmarks:
- `tests/ttnn/unit_tests/gtests/test_add.cpp` -- Unit tests using gtest
- `tests/ttnn/benchmark/cpp/matmul/test_matmul_benchmark.cpp` -- Full matmul benchmark with trace support
- `tests/ttnn/unit_tests/gtests/test_async_runtime.cpp` -- Async multi-queue runtime tests
- `tt_metal/programming_examples/eltwise_binary/` -- Low-level Metalium examples

The matmul benchmark is particularly relevant: it demonstrates traced C++ matmul execution on a single device, using `begin_trace_capture`/`end_trace_capture`/`execute_trace` from pure C++.

---

## 2. Rust FFI to TT-NN

### No Existing Rust Bindings for TT-NN

There are no Rust crates that wrap TT-NN or TT-Metalium ops. The only Rust code in the Tenstorrent ecosystem is:

- **Luwen** (`github.com/tenstorrent/luwen`) -- 98.5% Rust, but it is a low-level system interface library (hardware discovery, diagnostics, PCIe communication). It does NOT expose neural network operations. Think of it as a Rust equivalent of `tt-smi`.

### Feasibility of Rust FFI

A Rust wrapper is entirely feasible because `_ttnncpp.so` exports all symbols with standard C++ mangling. The approach would be:

1. **Write a thin C wrapper** (`extern "C"`) around the C++ functions we need (device open/close, matmul, softmax, rms_norm, embedding, trace ops, tensor creation/destruction)
2. **Generate Rust bindings** with `bindgen` against the C header
3. **Link against** `_ttnncpp.so` + dependencies

Alternatively, use `cxx` crate for direct C++/Rust interop without a C intermediate layer.

**Estimated effort:** ~1 week for a minimal wrapper covering the ~15 ops needed for Qwen2.5 inference. The hard part is not the FFI itself but reproducing the exact tensor layout, memory config, and compute kernel config decisions currently embedded in our Python code.

### Key challenge: C++ object lifecycle

TT-NN's `Tensor` class manages device memory with reference counting and RAII. A Rust wrapper must carefully manage:
- `std::shared_ptr<MeshDevice>` lifetime
- `Tensor` ownership (move semantics, deallocate())
- `MemoryConfig`, `DeviceComputeKernelConfig` construction

---

## 3. Alternative Compilation Paths

### tt-mlir's `ttnn-standalone` Tool

The tt-mlir compiler can generate standalone C++ code from MLIR:

```bash
ttmlir-opt --ttir-to-emitc-pipeline model.mlir | \
ttmlir-translate --mlir-to-cpp > ttnn-standalone.cpp
```

This generates a C++ file that calls TT-NN operations directly, then compiles and links against `_ttnncpp.so`. This is the closest thing to "compile model to C++ binary" that exists today.

**Limitation:** Requires going through the tt-mlir compiler pipeline (TTIR -> EmitC -> C++). We would need to get our Qwen model through tt-forge or tt-torch first.

### TT-Forge (MLIR Compiler Stack)

TT-Forge is Tenstorrent's compiler that accepts PyTorch, JAX, ONNX models and compiles them to optimized TT-NN op sequences via MLIR. The pipeline:

1. Frontend (tt-torch, tt-xla, tt-forge-fe) ingests the model
2. tt-mlir lowers through TTIR -> TTNN -> TTKernel dialects
3. Generated code calls TT-NN C++ API

This is the "official" path for production inference, but it is a compiler approach (static graph optimization), not a runtime dispatch approach.

### TT-Forge-ONNX

Export Qwen to ONNX, then use tt-forge-onnx to compile and run. This gives us the compiler's optimization passes but removes our manual control over op fusion, memory placement, etc.

---

## 4. Python Dispatch Overhead Breakdown

### What Causes the ~39us/op?

The 39us per-op overhead measured in our Python inference path comes from several layers:

**Layer 1: Python interpreter (~5-10us)**
- Function call overhead in CPython
- Argument packing/unpacking
- Python object reference counting

**Layer 2: nanobind marshaling (~2-5us)**
- nanobind (NOT pybind11 -- TT uses nanobind since ~2025) converts Python objects to C++ types
- nanobind is ~1.3x faster than pybind11 for raw call overhead
- Type checking, optional argument handling, keyword argument parsing
- Per the nanobind benchmarks: a simple function call costs ~100-200ns, but complex argument patterns with optionals/defaults cost more

**Layer 3: C++ dispatch overhead (~20-30us)**
- This is the dominant cost and exists even in pure C++
- Operation validation (shape checks, dtype compatibility)
- Program cache lookup (hash computation on op params)
- Command queue submission to device
- Buffer allocation/management for output tensors

**Key insight:** Even in pure C++, dispatch overhead is not zero. The tt-metal matmul benchmark exists specifically to measure traced vs non-traced C++ performance. The host dispatch pipeline involves:
1. Validate inputs (shapes, dtypes, memory configs)
2. Look up or compile the device program (program cache)
3. Set runtime args for the program
4. Submit program to the command queue
5. Allocate output buffer

**What trace eliminates:** Steps 1-4 are recorded once and replayed from DRAM. The host just says "replay trace" which is a single command queue submission.

### Estimated Savings from C++ vs Python

If ~10-15us of the 39us is Python/nanobind overhead, and ~25us is C++ dispatch:
- **Pure C++ dispatch: ~25us/op** (saves ~14us/op = ~5ms for full model)
- **Traced C++ dispatch: ~0.1-1us/op** (same as traced Python, since trace bypasses both)

For non-traced inference, C++ gives a meaningful but not transformative speedup: from ~21ms to ~16ms (estimate). The real win remains trace capture.

---

## 5. Practical Approaches (Ranked by Effort/Impact)

### Option A: Hybrid C++ dispatch with Python setup (LOW effort, MEDIUM impact)
Write the forward pass loop in C++ (linking against `_ttnncpp.so`), but keep Python for model loading and weight preparation. The C++ binary receives pre-loaded device tensors and runs the transformer loop.

**Pros:** Eliminates ~5ms Python overhead per forward pass. Enables C++ control flow (argmax without trace penalty).
**Cons:** Still ~25us/op C++ dispatch. Two-language build complexity.

### Option B: C++ with trace for static parts + C++ dispatch for dynamic parts (MEDIUM effort, HIGH impact)
Trace the attention+FFN blocks (static computation graph) in C++, use normal C++ dispatch only for the dynamic parts (argmax, KV cache update).

**Pros:** Best of both worlds. Traced blocks run at ~7.6ms, argmax runs at C++ speed (no 90ms trace penalty), total could be ~8-9ms.
**Cons:** Requires careful trace boundary management.

### Option C: tt-mlir/tt-forge compilation (HIGH effort, HIGH impact)
Get Qwen through the tt-forge compiler pipeline. This gives compiler-level optimizations (op fusion, memory planning) that our hand-written Python code doesn't have.

**Pros:** Compiler optimizations could exceed hand-tuned performance. Official Tenstorrent path.
**Cons:** tt-forge may not support all Qwen ops. Less control. Compiler bugs.

### Option D: Rust FFI wrapper (MEDIUM effort, SAME impact as Option A)
Same as Option A but in Rust. No additional performance benefit over C++ -- the bottleneck is the C++ dispatch layer, not the calling language.

**Pros:** Memory safety, better tooling.
**Cons:** Extra FFI complexity for no perf gain over C++.

---

## 6. Key Files and References

### tt-metal C++ API headers (on remote at /home/aditya/old/tt-metal/):
- `ttnn/api/ttnn/device.hpp` -- Device management (open_mesh_device, etc.)
- `ttnn/api/ttnn/types.hpp` -- Core types (Tensor, MemoryConfig, DataType)
- `ttnn/cpp/ttnn/operations/matmul/matmul.hpp` -- Matmul
- `ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.hpp` -- RMS Norm
- `ttnn/cpp/ttnn/operations/normalization/softmax/softmax.hpp` -- Softmax
- `ttnn/cpp/ttnn/operations/embedding/embedding.hpp` -- Embedding
- `ttnn/cpp/ttnn/operations/trace.hpp` -- Trace capture/replay
- `ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp` -- Element-wise ops

### C++ examples and tests:
- `ttnn/examples/add/add.cpp` -- Minimal TT-NN C++ example (17 lines)
- `ttnn/examples/CMakeLists.txt` -- Shows how to link (just `TTNN::CPP`)
- `tests/ttnn/benchmark/cpp/matmul/test_matmul_benchmark.cpp` -- Full traced matmul benchmark
- `tests/ttnn/unit_tests/gtests/test_add.cpp` -- Unit test pattern
- `tests/ttnn/unit_tests/gtests/test_async_runtime.cpp` -- Multi-queue async tests
- `tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp` -- Low-level Metalium API

### Build artifacts:
- `build_Release/ttnn/_ttnncpp.so` -- Standalone C++ library (50MB)
- `build_Release/lib/_ttnn.so` -- Python bindings (13MB, wraps ttnncpp)
- `build_Release/lib/libtt_metal.a` -- Static metalium library
- `build_Release/lib/libtt-umd.so` -- User-mode driver
- `build_Release/lib/libtt_stl.so` -- TT standard library

### Python binding layer (nanobind, NOT pybind11):
- `ttnn/cpp/ttnn-nanobind/operations/trace.cpp` -- Example nanobind wrapper
- `scripts/block_ttnn_pybind_changes.py` -- Enforces nanobind-only policy

### External tools:
- `ttnn-standalone` (in tt-mlir repo) -- Generates standalone C++ from MLIR
- Luwen (`github.com/tenstorrent/luwen`) -- Rust system interface (NOT for ops)
- TT-Forge (`github.com/tenstorrent/tt-forge`) -- MLIR compiler stack

---

## 7. Conclusions

1. **The C++ API is production-ready and fully documented.** Every TT-NN Python op has an exact C++ equivalent. The build system explicitly supports standalone C++ programs via `TTNN::CPP`.

2. **Rust adds FFI complexity with no performance benefit.** The bottleneck is the C++ dispatch layer (~25us/op), not the Python->C++ transition (~10-15us/op). Rust would call the same C++ functions.

3. **The biggest win is still trace capture**, which works identically from C++ and Python. The advantage of C++ is handling the non-traceable parts (argmax, dynamic control flow) without the 90ms trace penalty.

4. **Recommended experiment:** Write a minimal C++ forward pass for one transformer layer, benchmark it against our Python version (both traced and non-traced), and measure the actual C++ dispatch overhead. The matmul benchmark in `tests/ttnn/benchmark/cpp/matmul/` is a perfect template.

5. **The tt-mlir/tt-forge compiler path is the long-term answer** for production inference, but requires the compiler to support our model. Worth investigating in parallel.
