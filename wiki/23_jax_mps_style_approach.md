# 23: Deep Research — The jax-mps-Style PJRT Plugin Approach for TT-NN

## Part 1: How jax-metal / jax-mps Works

### Q: What is jax-metal and how does Apple's official JAX Metal backend work?

A: There are actually TWO Apple Silicon JAX backends worth studying:

1. **jax-metal** (official Apple): A closed-source PJRT plugin (`pip install jax-metal`) that uses the OpenXLA compiler to lower StableHLO to Metal Performance Shaders (MPS). It compiles StableHLO into MPSGraph executables and dispatches to the GPU via Metal runtime APIs. Apple reports up to 28x speedup over CPU on M2 Max, with an average of 10x.

2. **jax-mps / applejax** (community): Open-source PJRT plugins that do NOT use the OpenXLA compiler. Instead, they interpret StableHLO directly by walking the MLIR ops and dispatching to MLX (Apple's open-source tensor library). This is the approach closest to what we'd build.

The community approach is our template because:
- Like us, it targets a non-CUDA accelerator with its own tensor library (MLX for them, TT-NN for us)
- It does NOT need an MLIR compiler pipeline -- it interprets StableHLO directly
- It maps ~71 StableHLO ops to the target library's equivalents
- It works with `jax.jit`, `vmap`, and `grad` automatically (because it's a real PJRT plugin)

### Q: What is the architecture of jax-mps/applejax?

A: Three-stage pipeline:

```
JAX Python --> jax.jit traces --> StableHLO bytecode --> PJRT plugin --> MLX GPU execution
```

The key C++ source files in the plugin:

| File | Purpose | Our equivalent |
|------|---------|---------------|
| `pjrt_api.cc` | Exports `GetPjrtApi()`, assembles ~80 function pointers | (we don't have this) |
| `mlx_client.h/cc` | Device management, CompileStableHLO() | `ttnn.open_device()` |
| `mlx_executable.h/cc` | Walk StableHLO, dispatch to MLX ops, Execute() | `interpret.py` |
| `mlx_buffer.h/cc` | Host-device transfers, memory management | `tensors.py` |
| `stablehlo_parser.h/cc` | Parse StableHLO bytecode/text via MLIR libs | (we skip this -- we read Jaxpr) |
| `ops/arithmetic.cc` | add, mul, div, exp, log, etc. --> MLX | `ops.py` elementwise |
| `ops/shape.cc` | broadcast_in_dim, reshape, transpose --> MLX | `ops.py` shape ops |
| `ops/reduction.cc` | reduce_sum, reduce_max --> MLX | `ops.py` reductions |
| `ops/linalg.cc` | dot_general, cholesky, triangular_solve --> MLX | `ops.py` dot_general |
| `ops/gather_scatter.cc` | gather, scatter --> MLX | (we don't have these yet) |
| `ops/slice.cc` | slice, dynamic_slice --> MLX | (we don't have these yet) |
| `ops/control_flow.cc` | if, while, case --> MLX | (we don't have these) |
| `ops/sort_fft_complex.cc` | sort, fft, complex ops --> MLX | (we don't have these) |
| `passes/` | Optimization passes (fuse_bias_add, fuse_softmax) | (we don't do fusion) |

Language composition: **53% C++, 41% Python, 3% shell, 2% CMake**. Approximately 24 C++ source files for the core plugin.

### Q: How does jax-mps register as a JAX plugin?

A: Two mechanisms, both standard for PJRT plugins:

**Mechanism 1: Namespace package** -- Place a module under `jax_plugins/` with an `initialize()` function:

```python
# jax_plugins/mps/__init__.py
import os
import jax._src.xla_bridge as xb

def initialize():
    path = os.path.join(os.path.dirname(__file__), 'pjrt_plugin_mps.so')
    xb.register_plugin('mps', priority=500, library_path=path)
```

**Mechanism 2: Entry point** -- Declare in `pyproject.toml`:

```toml
[project.entry-points.'jax_plugins']
mps = "jax_plugins.mps"
```

JAX discovers plugins via `importlib.metadata.entry_points(group='jax_plugins')` on import. The user selects the plugin via `JAX_PLATFORMS=mps` environment variable. Once registered, `jax.devices()` returns the plugin's devices and all JAX operations route through it.

For our TT-NN plugin, this would be:

```python
# jax_plugins/tt/__init__.py
def initialize():
    path = os.path.join(os.path.dirname(__file__), 'pjrt_plugin_tt.so')
    xb.register_plugin('tt', priority=500, library_path=path)
```

Then: `JAX_PLATFORMS=tt python -c "import jax; print(jax.devices())"` --> `[TtDevice(id=0)]`

### Q: How does jax-mps compile StableHLO?

A: The `PJRT_Client_Compile` function receives StableHLO bytecode from JAX, then:

1. **Parse**: `parseStableHLOBytecode()` uses MLIR libraries to parse the bytecode into an `mlir::ModuleOp` with an entry `mlir::func::FuncOp`. This requires linking against ~20 MLIR libraries and ~14 StableHLO static libraries.

2. **Walk**: `MlxExecutable::Create()` walks the MLIR operations in the entry function. For each op, it looks up a handler in a static registry (similar to our `REGISTRY` dict).

3. **Build callable**: Each handler maps one StableHLO op to one or more MLX calls. The result is a `std::function<vector<array>(vector<array>)>` -- a callable that transforms MLX arrays.

4. **Optional MLX compile**: On first `Execute()`, it tries `mlx::core::detail::compile()` to fuse the op graph. Falls back to direct interpretation if compilation fails.

This is conceptually identical to our Jaxpr interpreter -- walk an IR, dispatch to a target library -- except they walk StableHLO (MLIR) instead of Jaxpr, and dispatch to MLX instead of TT-NN.

### Q: What shortcuts did jax-mps take?

A: Several pragmatic ones:
- **No kernel fusion**: Each StableHLO op maps 1:1 to an MLX op (though applejax added optional fusion passes later)
- **Single device only**: No multi-device support
- **No quantization**: uniform_quantize/dequantize not implemented
- **No async ops**: Everything is synchronous
- **CPU fallback**: Some ops (eigendecomposition, SVD, Schur) fall back to Apple Accelerate on CPU
- **Disabled shardy partitioner**: Avoids StableHLO parsing issues with JAX's sharding
- **No training-specific optimization**: Works but not tuned for backward passes

These are exactly the shortcuts we'd take too.


## Part 2: StableHLO vs Jaxpr

### Q: What is StableHLO and how does it differ from Jaxpr?

A: They are different intermediate representations (IRs) at different levels of the JAX compilation pipeline:

```
Python function
    |
    v
Jaxpr (JAX's internal IR, Python data structure)
    |  jax.jit.lower()
    v
StableHLO (MLIR dialect, serializable bytecode)
    |  XLA compiler (or PJRT plugin)
    v
Device code (CUDA PTX, Metal shaders, TT-NN ops, etc.)
```

**Jaxpr** is:
- A Python data structure (`jax.core.Jaxpr`)
- Produced by `jax.make_jaxpr(f)(x)` -- tracing the function
- Contains ~50 primitive operations (JAX-specific naming: `add`, `mul`, `dot_general`, etc.)
- Includes JAX-specific concepts: `stop_gradient`, `custom_vjp`, `pjit`
- Not serializable across processes (Python objects with references)

**StableHLO** is:
- An MLIR dialect with a formal specification (https://openxla.org/stablehlo/spec)
- Has **98 defined ops** with formal verifiers and type inference
- Serializable as MLIR bytecode -- can cross process/language boundaries
- Backward and forward compatible (versioned serialization format)
- The standard interface between ML frameworks and compilers
- Used by JAX, PyTorch/XLA, and TensorFlow to communicate with hardware backends

### Q: Why does PJRT use StableHLO instead of Jaxpr?

A: Three reasons:

1. **Language independence**: StableHLO is an MLIR dialect with C++ APIs. PJRT plugins are C/C++ shared libraries. Jaxpr is a Python data structure -- you can't pass it to a C++ library without serialization.

2. **Framework independence**: StableHLO is the common IR for JAX, PyTorch/XLA, and TensorFlow. A PJRT plugin that accepts StableHLO works with all three frameworks automatically.

3. **Stability guarantee**: StableHLO has backward compatibility guarantees. Jaxpr's internal representation can change between JAX versions without notice.

### Q: Could we bypass StableHLO and use Jaxpr in a PJRT plugin?

A: Not within the PJRT protocol. The `PJRT_Client_Compile` function signature expects either StableHLO bytecode or HLO proto. There is no Jaxpr entry point in the PJRT C API.

However, we could build a **hybrid** approach:
- Register a PJRT plugin (for `jax.devices()` and buffer management)
- In `Compile()`, receive StableHLO but immediately convert it back to a Jaxpr-like representation
- Execute using our existing Python-style dispatch

This is essentially what jax-mps does -- it receives StableHLO but doesn't compile it to native code. It walks the ops and dispatches to library calls, which is interpretation, not compilation.

### Q: How do StableHLO ops map to our existing 20 Jaxpr primitives?

A: StableHLO has **98 ops** total. Here's how they relate to what we already support:

| Our Jaxpr primitive | StableHLO equivalent | Notes |
|---------------------|---------------------|-------|
| `add` | `stablehlo.add` | Direct 1:1 |
| `sub` | `stablehlo.subtract` | Direct 1:1 |
| `mul` | `stablehlo.multiply` | Direct 1:1 |
| `div` | `stablehlo.divide` | Direct 1:1 |
| `neg` | `stablehlo.negate` | Direct 1:1 |
| `exp` | `stablehlo.exponential` | Direct 1:1 |
| `log` | `stablehlo.log` | Direct 1:1 |
| `sqrt` | `stablehlo.sqrt` | Direct 1:1 |
| `rsqrt` | `stablehlo.rsqrt` | Direct 1:1 |
| `reciprocal` | (no direct equivalent) | StableHLO uses `divide(1, x)` |
| `max` (elementwise) | `stablehlo.maximum` | Direct 1:1 |
| `integer_pow` | `stablehlo.power` | StableHLO has general power |
| `dot_general` | `stablehlo.dot_general` | Direct 1:1, same semantics |
| `reduce_max` | `stablehlo.reduce` with max | StableHLO has generic reduce |
| `reduce_sum` | `stablehlo.reduce` with add | StableHLO has generic reduce |
| `broadcast_in_dim` | `stablehlo.broadcast_in_dim` | Exact same semantics |
| `reshape` | `stablehlo.reshape` | Direct 1:1 |
| `transpose` | `stablehlo.transpose` | Direct 1:1 |
| `squeeze` | (no direct equivalent) | Handled by reshape in StableHLO |
| `convert_element_type` | `stablehlo.convert` | Direct 1:1 |
| `stop_gradient` | (not in StableHLO) | Training-only, handled at JAX level |

**Summary: 18 of our 20 Jaxpr primitives map directly to StableHLO ops.** The two that don't (`reciprocal`, `squeeze`) are trivially expressed in StableHLO terms.

### Q: What is the `reduce` op difference between Jaxpr and StableHLO?

A: This is important. In Jaxpr, we have separate primitives `reduce_sum` and `reduce_max`. In StableHLO, there's a single generic `stablehlo.reduce` that takes a **computation body** -- an inner function that defines how elements are combined. For sum, the body is `add`; for max, the body is `maximum`. This means our PJRT plugin would need to parse the reduction body to determine which TT-NN reduction to call.

### Q: What additional StableHLO ops would we need for a useful backend?

A: Beyond our current 20, a minimal PJRT plugin for inference needs approximately **35-40 StableHLO ops**. applejax implements ~71 StableHLO ops + 12 CHLO (custom HLO) ops including erf, top_k, acos, sinh, erf_inv. The extra ops group into categories:

| Category | StableHLO ops | Purpose |
|----------|--------------|---------|
| **Essential for transformer** | `gather`, `slice`, `dynamic_slice`, `concatenate`, `select`, `pad` | Token embeddings, attention masks, KV cache |
| **Activation/math** | `tanh`, `cosine`, `sine`, `logistic`, `clamp`, `sign`, `floor`, `ceil` | GELU, positional encoding |
| **Comparison** | `compare` | Generates boolean masks |
| **Data generation** | `iota`, `constant` | Index arrays, constant tensors |
| **Convolution** | `convolution` | CNN support |
| **Control flow** | `while`, `if`, `case` | Dynamic-length generation |
| **Scatter** | `scatter`, `dynamic_update_slice` | KV cache update |
| **Communication** | `all_reduce`, `all_gather` | Multi-device (future) |
| **Advanced** | `fft`, `sort`, `cholesky`, `triangular_solve` | Scientific/linalg |


## Part 3: What Would a TT-NN PJRT Plugin Look Like?

### Q: What are the four key components of a TT-NN PJRT plugin?

A: Following the jax-mps architecture:

**Component 1: Plugin Entry Point** (`pjrt_api.cc`)
- Exports `GetPjrtApi()` via `dlsym()`
- Returns a `PJRT_Api` struct filled with ~80 function pointers
- The C++ wrapper (`pjrt_c_api_wrapper_impl.h`) auto-generates most of these from a `PjRtClient` subclass
- This is pure boilerplate -- copy from jax-mps and change the class names

**Component 2: Client & Device Management** (`tt_client.cc`, `tt_device.cc`)
- `PJRT_Client_Create` calls `ttnn::open_device(0)`
- Reports device capabilities (memory size, compute cores, device name "Blackhole")
- Handles device enumeration (we'd report 1 device, device_id=0)
- Manages device lifecycle (open on create, close on destroy)

**Component 3: Buffer Management** (`tt_buffer.cc`)
- `PJRT_Client_BufferFromHostBuffer`: host numpy array --> ttnn tensor with tile padding
- `PJRT_Buffer_ToHostBuffer`: ttnn tensor --> host numpy array with unpadding
- Must handle the TILE_LAYOUT 32-alignment requirement (pad shapes to multiples of 32)
- Must handle dtype conversion (float32 from JAX <--> bfloat16 on device)
- This maps directly to our `tensors.py` `to_device()` / `from_device()`

**Component 4: Compilation & Execution** (`tt_executable.cc`, `stablehlo_parser.cc`, `ops/*.cc`)
- `PJRT_Client_Compile`: parse StableHLO bytecode, walk ops, build a dispatch plan
- `PJRT_LoadedExecutable_Execute`: execute the dispatch plan on TT-NN
- The op handlers are isomorphic to our Python `REGISTRY` dict
- Optionally wrap execution in `ttnn::begin_trace_capture` / `ttnn::end_trace_capture`

### Q: What's the concrete file structure?

A:

```
pjrt-plugin-tt/
  CMakeLists.txt                    # Build system
  src/
    pjrt_api.cc                     # GetPjrtApi() entry point (~300 lines)
    tt_client.h/cc                  # Device open/close, CompileStableHLO() (~200 lines)
    tt_device.h/cc                  # Device description, memory spaces (~100 lines)
    tt_buffer.h/cc                  # to_device/from_device with tile padding (~250 lines)
    tt_executable.h/cc              # Walk StableHLO, dispatch to TT-NN (~400 lines)
    stablehlo_parser.h/cc           # Parse StableHLO bytecode via MLIR (~150 lines)
    ops/
      registry.h                    # Op handler type + registration (~50 lines)
      arithmetic.cc                 # add, sub, mul, div, exp, log, etc. (~300 lines)
      shape.cc                      # broadcast_in_dim, reshape, transpose (~250 lines)
      reduction.cc                  # reduce (sum, max, min, prod) (~150 lines)
      linalg.cc                     # dot_general, conv (~200 lines)
      data.cc                       # constant, iota, select, gather, slice (~300 lines)
    pjrt_stubs.cc                   # Stubbed PJRT functions (~200 lines)
  python/
    jax_plugins/tt/__init__.py      # Plugin registration (~40 lines)
  tests/
    test_basic.py                   # JAX-level tests (~200 lines)
  third_party/
    mlir/                           # MLIR/StableHLO headers + static libs
    tt-metal/                       # TT-NN headers + libs
```

**Estimated total: ~2,600 lines of C++, ~240 lines of Python.**

For reference, our current Python interpreter is ~585 lines across 4 files. The PJRT plugin would be roughly 4-5x more code, mostly due to C++ boilerplate, PJRT API wiring, and StableHLO parsing.


## Part 4: How Would Compile() Work?

### Q: What happens when JAX calls Compile() on our plugin?

A: The full pipeline from `jax.jit(f)(x)` to a compiled executable:

```
Step 1: JAX traces f(x) into Jaxpr (Python-side, automatic)
Step 2: JAX lowers Jaxpr to StableHLO (Python-side, automatic)
Step 3: JAX serializes StableHLO to bytecode
Step 4: JAX calls PJRT_Client_Compile(bytecode) on our plugin
Step 5: Our plugin parses the StableHLO bytecode back to MLIR ops
Step 6: Our plugin walks each op and builds a dispatch plan
Step 7: Returns a PjRtLoadedExecutable handle
```

Steps 1-4 are done by JAX automatically. We implement steps 5-7.

### Q: What does "build a dispatch plan" mean concretely?

A: It means creating a list of callable functions, each mapping one StableHLO op to TT-NN calls. In C++:

```cpp
// In tt_executable.cc
struct OpDispatch {
    std::function<ttnn::Tensor(std::vector<ttnn::Tensor>&)> handler;
    std::vector<int> input_indices;   // which intermediate tensors are inputs
    int output_index;                  // where to store the result
};

class TtExecutable {
    std::vector<OpDispatch> dispatch_plan_;

    static std::unique_ptr<TtExecutable> Create(mlir::ModuleOp module) {
        auto exec = std::make_unique<TtExecutable>();
        auto func = getEntryFunction(module);

        for (auto& op : func.getOps()) {
            if (auto add = dyn_cast<stablehlo::AddOp>(op)) {
                exec->dispatch_plan_.push_back({
                    .handler = [](auto& tensors) {
                        return ttnn::add(tensors[0], tensors[1]);
                    },
                    .input_indices = getInputIndices(add),
                    .output_index = getOutputIndex(add),
                });
            }
            // ... similar for mul, matmul, reshape, etc.
        }
        return exec;
    }
};
```

This is conceptually identical to our Python interpreter walking Jaxpr equations. The only difference is the source IR (StableHLO MLIR ops vs Jaxpr equations) and the language (C++ vs Python).

### Q: What MLIR libraries do we need for parsing?

A: This is the hardest part of the build. jax-mps links against:

**~20 MLIR libraries**: MLIRBytecodeReader, MLIRParser, MLIRIR, MLIRSupport, MLIRFuncDialect, etc.

**~14 StableHLO libraries**: StablehloOps, StablehloAssemblyFormat, StablehloBytecode, etc.

Plus their transitive dependencies: LLVM support libraries, protobuf, abseil.

Getting these pre-built for our Linux host (Ubuntu on the Tenstorrent machine) is non-trivial. Options:
1. Build MLIR + StableHLO from source (30+ minutes, complex CMake)
2. Extract pre-built libs from a matching JAX wheel (hacky but faster)
3. Use the JAX-bundled XLA compiler libraries (if they're accessible)


## Part 5: How Would Execute() Work?

### Q: What happens when the compiled executable runs?

A: Two modes:

**Mode 1: Direct dispatch** (simpler, like our current interpreter)
```cpp
void TtExecutable::Execute(std::vector<ttnn::Tensor>& inputs) {
    std::vector<ttnn::Tensor> intermediates(num_intermediates_);

    // Copy inputs into intermediate slots
    for (int i = 0; i < inputs.size(); i++)
        intermediates[i] = inputs[i];

    // Walk the dispatch plan
    for (auto& dispatch : dispatch_plan_) {
        std::vector<ttnn::Tensor> args;
        for (int idx : dispatch.input_indices)
            args.push_back(intermediates[idx]);
        intermediates[dispatch.output_index] = dispatch.handler(args);
    }

    output_ = intermediates.back();
}
```

**Mode 2: Trace-wrapped dispatch** (much faster, eliminates C++ dispatch overhead too)
```cpp
void TtExecutable::Execute(std::vector<ttnn::Tensor>& inputs) {
    if (!trace_captured_) {
        // First run: capture a trace
        for (int i = 0; i < inputs.size(); i++)
            input_buffers_[i] = inputs[i];

        uint32_t tid = ttnn::begin_trace_capture(device_, cq_id_);
        DirectDispatch(input_buffers_);  // run once to record
        ttnn::end_trace_capture(device_, cq_id_, tid);
        trace_id_ = tid;
        trace_captured_ = true;
    } else {
        // Subsequent runs: overwrite input buffers and replay
        for (int i = 0; i < inputs.size(); i++)
            ttnn::copy_host_to_device_tensor(inputs[i], input_buffers_[i]);
        ttnn::execute_trace(device_, cq_id_, trace_id_, false);
    }
}
```

### Q: Can we use ttnn trace capture in the PJRT context?

A: Yes, and this is a key advantage. The PJRT Execute() boundary is the perfect place for trace capture because:

1. **Fixed graph**: After Compile(), the op graph is frozen. Same ops, same shapes every time.
2. **Clear input/output boundary**: Execute() receives input buffers and returns output buffers. We know exactly what to overwrite.
3. **No Python in the loop**: The trace replay path has zero Python dispatch overhead AND zero C++ dispatch overhead -- it's pure hardware command replay.

The trace capture approach turns our PJRT plugin into something like CUDA graphs -- record once, replay many times. The transformer encoder's 50+ ops would replay as a single hardware command sequence.

### Q: What about dynamic shapes?

A: Traces are fixed-shape. If JAX sends different input shapes (e.g., different batch sizes), we'd need to either:
1. Re-capture the trace (detect shape change, discard old trace)
2. Maintain a cache of traces keyed by input shapes
3. Pad all inputs to a maximum shape (wasteful but simple)

jax-mps handles this by not using traces -- they re-dispatch every time. For our use case (inference with fixed batch size), trace capture is the right choice.


## Part 6: Broadcast Problem in the PJRT Context

### Q: How did jax-mps handle broadcast_in_dim?

A: Their handler in `ops/shape.cc`:
1. Build an intermediate shape filled with 1s
2. Set non-broadcast dimensions to their source sizes
3. `mlx::core::reshape()` to the intermediate shape
4. `mlx::core::broadcast_to()` to the final output shape

This works seamlessly because MLX supports implicit broadcasting on its GPU arrays. **We cannot do this with TT-NN TILE_LAYOUT implicit broadcasting** -- that was our #1 blocker.

### Q: Does on-device broadcast via ttnn.repeat solve this?

A: Yes. Our experiment 21 discovered that TT-NN provides several on-device broadcast operations:

| Operation | What it does | Works with TILE_LAYOUT? |
|-----------|-------------|------------------------|
| `ttnn.repeat(tensor, reps)` | Tile repetition along dims | YES -- confirmed working |
| `ttnn.repeat_interleave(tensor, n, dim)` | Repeat elements along a dim | YES |
| `ttnn.expand(tensor, shape)` | Expand singleton dims | Exists, needs testing |
| `ttnn.bcast(a, b, math_op, dim)` | Fused broadcast + binary op | BF16 only, H/W/HW dims |

The `ttnn.repeat` approach:
```python
# (1, 1, 1, 64) --> (1, 1, 32, 64) -- entirely on device
b_expanded = ttnn.repeat(b, ttnn.Shape([1, 1, 32, 1]))  # repeat 32x along dim 2
```

Performance impact:

| Method | Latency | Notes |
|--------|---------|-------|
| CPU round-trip (current) | 0.147 ms | Read + broadcast + write |
| Host pre-expand | 0.084 ms | No read, just write expanded |
| `ttnn.repeat` (on-device) | ~0 ms | No host transfers at all |

With 10 broadcasts per transformer forward pass, CPU round-trips cost 1.47 ms -- **26% of our 5.59 ms forward time**. Eliminating them is the single highest-value optimization.

### Q: How would broadcast work in a PJRT plugin specifically?

A: In a PJRT plugin, the `Compile()` step sees the entire graph. We could:

1. **Simple approach**: For each `stablehlo.broadcast_in_dim`, insert a `ttnn::repeat()` call in the dispatch plan. This keeps shapes matched for all downstream binary ops.

2. **Fused approach**: Pattern-match `broadcast_in_dim` --> `add` sequences and replace with `ttnn::bcast(a, b, BcastOpMath::ADD, dim)` -- a single fused on-device op.

3. **Compiler approach**: What the official tt-mlir does -- lower `stablehlo.broadcast_in_dim` through TTIR dialect --> TTNN dialect, inserting optimized repeat/bcast ops.

The key insight: **the broadcast fix is independent of the PJRT decision**. Whether we use our Python interpreter or build a PJRT plugin, `ttnn.repeat` eliminates the CPU round-trip. But in the PJRT context, we can also fuse broadcasts with consuming ops since we see the full graph at compile time.

### Q: How does the official tt-xla (via tt-mlir) handle broadcasts?

A: The tt-mlir compiler handles `stablehlo.broadcast_in_dim` as a first-class MLIR operation:
1. `stablehlo.broadcast_in_dim` is lowered through TTIR dialect --> TTNN dialect
2. The compiler inserts explicit `ttnn.repeat` or reshape operations as needed
3. The compiler may fuse the broadcast with the consuming op

The tt-torch compatibility table shows `stablehlo.broadcast_in_dim` passing 862+ test cases. This confirms TT-NN can handle broadcast patterns -- we just need to use the right API calls.


## Part 7: Pros and Cons vs Current Jaxpr Interpreter

### Q: What do we gain and lose with each approach?

**Current approach: Python Jaxpr Interpreter**

| Dimension | Assessment |
|-----------|-----------|
| **Development effort** | Done for 20 ops. ~585 lines Python. |
| **Performance** | 348 fwd/sec with on-device broadcast. Limited by Python dispatch (~21us/op). With trace: ~9us/op. |
| **jax.jit support** | NO. Must manually call `jax.make_jaxpr()` + `interpreter.run()`. |
| **vmap/grad** | NO. Would need separate implementation. |
| **Maintainability** | Excellent. Plain Python dict registry. Anyone can add an op in 5 minutes. |
| **Trace capture** | Works when all ops stay on-device (requires ttnn.repeat for broadcast). |
| **Ecosystem integration** | None. Libraries (Flax, Optax) can't use our backend transparently. |

**PJRT Plugin approach (jax-mps style)**

| Dimension | Assessment |
|-----------|-----------|
| **Development effort** | ~2,600 lines C++. 2-4 weeks for MVP. Build system complexity. |
| **Performance** | Higher ceiling. On-device broadcast. Trace-wrapped execution. <1us C++ dispatch. |
| **jax.jit support** | YES, automatic. This is the whole point. |
| **vmap/grad** | YES, automatic. JAX handles transformations before lowering to StableHLO. |
| **Maintainability** | Harder. C++ compilation, MLIR dependency, CMake. |
| **Trace capture** | Could wrap entire `Execute()` in trace for maximum speed. |
| **Ecosystem integration** | Full. `jax.devices()` shows "tt", Flax/Optax just work. |

### Q: What's the performance comparison in concrete terms?

Current Python interpreter path:
```
jax.make_jaxpr(model)(x)   -->  Jaxpr (Python IR)
interpreter.run(jaxpr, x)  -->  walks 50+ equations, ~21us each
                            -->  50 * 21us = 1.05ms Python dispatch
                            -->  plus actual TT-NN compute time
                            -->  plus CPU round-trips for broadcast (if not fixed)
```

PJRT plugin path:
```
jax.jit(model)(x)          -->  StableHLO (compiled once, cached)
plugin.Execute(x)          -->  C++ dispatch, <1us per op
                            -->  or wrapped in trace: near-zero dispatch
                            -->  no CPU round-trips (on-device broadcast)
```

The dispatch overhead gap: **21us/op (Python) vs <1us/op (C++) vs ~0us/op (traced)**. With trace capture in our Python interpreter, we already get close to 0us/op -- so the real win from PJRT is **ecosystem integration** (jax.jit, vmap, grad, Flax compatibility), not raw dispatch speed.

### Q: Is jax.jit worth 2-4 weeks of C++ work?

**Arguments for yes:**
- `jax.jit` enables XLA-level optimizations (constant folding, dead code elimination, CSE) before the graph reaches our plugin
- `vmap` and `grad` for free means batched inference and potentially training
- Library compatibility (Flax, Optax) means running real models without manual Jaxpr extraction
- Our op registry translates almost 1:1 to C++ handler functions

**Arguments for no:**
- We already have 21/21 tests passing and 348 fwd/sec on a transformer encoder
- The C++ build complexity is significant (MLIR dependencies alone are painful)
- For a course project, demonstrating the concept matters more than production integration
- The broadcast fix benefits our Python interpreter too


## Part 8: The Three Paths Forward

### Q: What are the three paths, and which should we choose?

**Path A: Full PJRT with StableHLO Compiler**

```
JAX --> StableHLO --> our PJRT plugin --> parse StableHLO --> dispatch to TT-NN
```

- Effort: 2-4 weeks, ~2,600 lines C++
- Gets: jax.jit, vmap, grad, full ecosystem
- Risk: MLIR build complexity, C++ debugging on remote host
- This is the jax-mps approach

**Path B: PJRT with Jaxpr Pass-Through**

```
JAX --> StableHLO --> our PJRT plugin --> convert back to Jaxpr-like --> dispatch to TT-NN
```

- Effort: 3-5 weeks (more complex because of the conversion layer)
- Gets: Same as Path A, but we reuse our Python op handlers
- Risk: Unnecessary complexity -- if we're parsing StableHLO, just dispatch from there
- Not recommended. If you're going to parse StableHLO anyway, dispatch directly.

**Path C: Keep Interpreter + Add Trace Capture (recommended)**

```
JAX --> jax.make_jaxpr --> our Python interpreter --> TT-NN with trace capture
```

With the key enhancement: replace CPU broadcast with `ttnn.repeat` so trace capture works for ALL ops.

- Effort: 1-2 days for broadcast fix, 1 week for polish
- Gets: ~2-3x speedup from trace, all ops on-device, potentially 500+ fwd/sec
- Loses: No jax.jit, no vmap, no Flax compatibility
- For CS440LX: this is the pragmatic choice

### Q: What's the progressive migration strategy?

A phased approach that starts with the highest-leverage change:

**Phase 1 (1-2 days)**: Fix broadcast with `ttnn.repeat` / `ttnn.expand`
```python
# In tensors.py broadcast_to_match():
# BEFORE (CPU round-trip):
a_np = from_device(a_tt, a_shape)
a_np = np.broadcast_to(a_np, out_shape).copy()
a_tt = to_device(a_np, device)

# AFTER (on-device):
a_tt = ttnn.repeat(a_tt, repeat_shape)
```
If this works, trace capture becomes fully functional. Benchmark the transformer.

**Phase 2 (1 week, optional)**: Start PJRT plugin skeleton
- Use the C++ wrapper to auto-generate PJRT function pointers
- Get `jax.devices()` showing a Tenstorrent device
- Implement buffer transfer (our `to_device` / `from_device` in C++)

**Phase 3 (1 week, optional)**: Port op handlers to C++
- Logic is identical to our Python handlers, just different syntax
- Wire up Compile() --> walk StableHLO --> dispatch to TT-NN

**Phase 4 (1 week, optional)**: Integration testing
- Run the transformer encoder through `jax.jit` on the TT device
- Compare performance with our Python interpreter


## Part 9: Minimum Viable PJRT Plugin

### Q: What would the absolute minimum viable PJRT plugin need?

A: To run `y = jax.jit(lambda x: x @ w + b)(input)`:

1. `PJRT_Client_Create` --> `ttnn::open_device(0)`
2. `PJRT_Client_BufferFromHostBuffer` --> pad to 32-aligned, `ttnn::from_torch(..., TILE_LAYOUT)`
3. `PJRT_Client_Compile` --> parse StableHLO, build op dispatch list
4. `PJRT_LoadedExecutable_Execute` --> walk ops calling `ttnn::matmul`, `ttnn::add`
5. `PJRT_Buffer_ToHostBuffer` --> `ttnn::to_torch()`, unpad
6. Destructors --> `ttnn::close_device()`

For this minimal case, we need handlers for: `constant`, `dot_general`, `add`, `broadcast_in_dim`, and `convert`. That's **5 StableHLO ops**.

### Q: How many lines of code, and what are the dependencies?

A: Estimated breakdown:

| Component | Lines (C++) | Difficulty |
|-----------|------------|------------|
| PJRT API wiring (pjrt_api.cc + stubs) | ~500 | Low (boilerplate) |
| Client + device management | ~300 | Low |
| Buffer management (with tile padding) | ~250 | Medium |
| StableHLO parser | ~150 | Medium (MLIR linking) |
| Op handlers (5 ops for MVP) | ~200 | Low (direct translation) |
| Python registration | ~40 | Low |
| **Total** | **~1,440** | |

Dependencies:
- TT-Metal / TT-NN: already installed on remote host
- MLIR libraries (~20 static libs): must be built or extracted
- StableHLO libraries (~14 static libs): must be built or extracted
- Protobuf, Abseil: transitive dependencies of MLIR
- CMake 3.20+: build system

The hardest part is NOT the code -- it's **getting MLIR/StableHLO libraries compiled for our Linux host**. Building MLIR from source takes 30+ minutes and requires significant disk space. jax-mps bundles pre-built MLIR in `third_party/`.

### Q: Are there other open-source PJRT plugins we can study besides jax-mps?

A: Yes, several:

| Plugin | Target | Approach | Open Source? |
|--------|--------|----------|-------------|
| **jax-mps / applejax** | Apple Silicon (MLX) | StableHLO --> MLX ops | Yes |
| **tt-xla** (official Tenstorrent) | Tenstorrent (tt-mlir) | StableHLO --> TTIR --> TTNN | Yes |
| **Intel XPU plugin** | Intel GPUs (SYCL) | StableHLO --> SYCL kernels | Yes |
| **jax-metal** (Apple official) | Apple Silicon (Metal) | StableHLO --> MPSGraph | No (closed source) |
| **CUDA PJRT** | NVIDIA GPUs | StableHLO --> CUDA via XLA | Yes (part of XLA) |
| **TPU PJRT** | Google TPUs | StableHLO --> TPU IR | Yes (part of XLA) |

The official tt-xla is particularly instructive: 84% Python / 14% C++, 19 C++ implementation files, delegates compilation to tt-mlir. It does NOT work on our Blackhole because tt-mlir lacks full Blackhole support in released packages.

### Q: What would convince us to go full PJRT?

Three criteria:
1. We've exhausted Python interpreter performance (broadcast fixed, trace working, still want more)
2. We need library compatibility (Flax, Optax) for a specific demo
3. We have 2+ weeks of development time remaining

If all three are true, the PJRT plugin is worth building. applejax proved it's feasible for a small team -- they went from zero to 91.5% JAX test compatibility with ~71 StableHLO ops.


## Key Takeaways

1. **jax-mps/applejax is our template**: ~24 C++ files, ~2,600 lines, walks StableHLO and dispatches to MLX. We'd do the same but dispatch to TT-NN.

2. **Our op registry translates directly**: 18/20 Jaxpr primitives have exact StableHLO equivalents. The code structure is isomorphic.

3. **The broadcast fix is independent of the PJRT decision**: Use `ttnn.repeat()` to eliminate CPU round-trips. Test this FIRST regardless of which path we take.

4. **PJRT's real value is ecosystem integration**: `jax.jit`, `vmap`, `grad`, and library compatibility. Raw execution speed can be matched by trace capture in our Python interpreter.

5. **The build system is the hardest part**: Linking against MLIR (~20 libs), StableHLO (~14 libs), and TT-Metal simultaneously requires careful CMake work.

6. **Progressive migration is possible and recommended**: Fix broadcast --> add trace --> (optionally) build PJRT shell --> port handlers to C++.

7. **StableHLO has 98 ops; we'd need ~35-40 for inference, ~71 for broad compatibility**. applejax shows this is achievable.


## Sources

- jax-mps: https://github.com/tillahoffmann/jax-mps
- applejax (enhanced fork): https://github.com/danielpcox/applejax
- applejax PyPI: https://pypi.org/project/applejax/
- StableHLO spec: https://openxla.org/stablehlo/spec
- StableHLO interpreter status: https://openxla.org/stablehlo/interpreter_status
- PJRT C++ API overview: https://openxla.org/xla/pjrt/cpp_api_overview
- PJRT plugin integration guide: https://openxla.org/xla/pjrt/pjrt_integration
- PJRT C API header: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h
- PJRT C API wrapper: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_wrapper_impl.h
- Apple Metal JAX: https://developer.apple.com/metal/jax/
- Official tt-xla: https://github.com/tenstorrent/tt-xla
- tt-xla docs: https://docs.tenstorrent.com/tt-xla/getting_started.html
- PJRT plugin blog post: https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html
- JAX FFI docs: https://docs.jax.dev/en/latest/ffi.html
- TT-NN API docs: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html
- TT-NN broadcast_in_dim support: https://docs.tenstorrent.com/tt-torch/ops/stablehlo/stablehlo.broadcast_in_dim.html
- JAX PJRT plugin discussion: https://github.com/jax-ml/jax/discussions/34648
