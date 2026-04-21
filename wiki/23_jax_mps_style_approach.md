# 23: The jax-mps Approach — Could We Build a PJRT Plugin for TT-NN?

## Part 1: How jax-mps Works

**Q: What is jax-mps and why should we study it?**

A: jax-mps (https://github.com/tillahoffmann/jax-mps) is a community-built JAX backend for Apple Silicon that maps StableHLO ops to MLX ops via a C++ PJRT plugin. It achieves ~3.7x speedup over CPU for ResNet18 training on an M4 MacBook Air, with 91.5% of JAX tests passing.

This is the closest analog to what we'd build. Like us, jax-mps:
- Targets a non-CUDA accelerator with its own tensor library (MLX for them, TT-NN for us)
- Does NOT use an MLIR compiler pipeline — it interprets StableHLO directly
- Maps ~70 StableHLO ops to the target library's equivalents
- Works with `jax.jit`, `vmap`, and `grad` automatically (because it's a real PJRT plugin)

There's also an enhanced fork called applejax (https://github.com/danielpcox/applejax) with 71 StableHLO graph ops, 12 CHLO ops, complex number support, full linalg, and 2000+ tests.

**Q: What's the architecture of jax-mps?**

A: Three-stage pipeline, nearly identical in spirit to our Jaxpr interpreter:

```
JAX Python → jax.jit traces → StableHLO bytecode → PJRT plugin → MLX GPU execution
```

The key files in `src/pjrt_plugin/`:

| File | Purpose | Our equivalent |
|------|---------|---------------|
| `pjrt_api.cc` | Exports `GetPjrtApi()`, assembles ~80 function pointers | (we don't have this) |
| `mlx_client.h/cc` | Device management, CompileStableHLO() | `ttnn.open_device()` |
| `mlx_executable.h/cc` | Walk StableHLO, dispatch to MLX ops, Execute() | `interpret.py` |
| `mlx_buffer.h/cc` | Host-device transfers, memory management | `tensors.py` |
| `stablehlo_parser.h/cc` | Parse StableHLO bytecode/text via MLIR libs | (we skip this — we read Jaxpr) |
| `ops/arithmetic.cc` | add, mul, div, exp, log, etc. → MLX | `ops.py` elementwise |
| `ops/shape.cc` | broadcast_in_dim, reshape, transpose → MLX | `ops.py` shape ops |
| `ops/reduction.cc` | reduce_sum, reduce_max → MLX | `ops.py` reductions |
| `ops/linalg.cc` | dot_general, cholesky, triangular_solve → MLX | `ops.py` dot_general |
| `ops/gather_scatter.cc` | gather, scatter → MLX | (we don't have these yet) |
| `ops/slice.cc` | slice, dynamic_slice → MLX | (we don't have these yet) |
| `ops/control_flow.cc` | if, while, case → MLX | (we don't have these) |
| `ops/sort_fft_complex.cc` | sort, fft, complex ops → MLX | (we don't have these) |
| `passes/` | Optimization passes (fuse_bias_add, fuse_softmax) | (we don't do fusion) |

Language composition: **53% C++, 41% Python, 3% shell, 2% CMake**. Approximately 24 C++ source files for the core plugin.

**Q: How does jax-mps compile StableHLO?**

A: The `PJRT_Client_Compile` function receives StableHLO bytecode from JAX, then:

1. **Parse**: `parseStableHLOBytecode()` uses MLIR libraries to parse the bytecode into an `mlir::ModuleOp` with an entry `mlir::func::FuncOp`. This requires linking against ~20 MLIR libraries and ~14 StableHLO static libraries.

2. **Walk**: `MlxExecutable::Create()` walks the MLIR operations in the entry function. For each op, it looks up a handler in a static registry (similar to our `REGISTRY` dict).

3. **Build callable**: Each handler maps one StableHLO op to one or more MLX calls. The result is a `std::function<vector<array>(vector<array>)>` — a callable that transforms MLX arrays.

4. **Optional MLX compile**: On first `Execute()`, it tries `mlx::core::detail::compile()` to fuse the op graph. Falls back to direct interpretation if compilation fails.

This is conceptually identical to our approach — walk an IR, dispatch to a target library — except they walk StableHLO (MLIR) instead of Jaxpr, and they dispatch to MLX instead of TT-NN.

**Q: How does their broadcast_in_dim work?**

A: From `ops/shape.cc`, their handler for `stablehlo.broadcast_in_dim`:

1. Build an intermediate shape filled with 1s
2. Set non-broadcast dimensions to their source sizes
3. `mlx::core::reshape()` to the intermediate shape
4. `mlx::core::broadcast_to()` to the final output shape

This works seamlessly because MLX supports implicit broadcasting on its GPU arrays. **We can't do this on TT-NN TILE_LAYOUT** — that's our #1 blocker. More on this in Part 4.

**Q: What shortcuts did jax-mps take?**

A: Several pragmatic ones:
- **No kernel fusion**: Each StableHLO op maps 1:1 to an MLX op (though they added optional fusion passes later)
- **Single device only**: No multi-device support
- **No quantization**: uniform_quantize/dequantize not implemented
- **No async ops**: Everything is synchronous
- **CPU fallback**: Some ops (eigendecomposition) fall back to Apple Accelerate on CPU
- **Disabled shardy partitioner**: Avoids StableHLO parsing issues with JAX's sharding
- **No training-specific optimization**: Works but not tuned for backward passes

These are exactly the shortcuts we'd take too.


## Part 2: StableHLO vs Jaxpr — The Op Mapping

**Q: How do StableHLO ops map to our existing 20 Jaxpr primitives?**

A: StableHLO has **107 ops** total. Here's how they relate to what we already support:

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

**Q: Is StableHLO a strict superset of Jaxpr primitives?**

A: Not exactly a superset in the formal sense — they're different IRs with different design goals — but **StableHLO covers everything Jaxpr does and much more**:

StableHLO ops we'd need to add beyond our current 20 for a useful backend:

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

A minimal PJRT plugin for inference needs approximately **35-40 StableHLO ops** — roughly double our current Jaxpr coverage. jax-mps implements ~71 StableHLO ops + 12 CHLO (custom HLO) ops.

**Q: What is the `reduce` op difference?**

A: This is important. In Jaxpr, we have separate primitives `reduce_sum` and `reduce_max`. In StableHLO, there's a single generic `stablehlo.reduce` that takes a **computation body** — an inner function that defines how elements are combined. For sum, the body is `add`; for max, the body is `maximum`. This means our PJRT plugin would need to parse the reduction body to determine which TT-NN reduction to call.


## Part 3: Concrete Implementation Plan

**Q: What would a minimal PJRT plugin for TT-NN look like?**

A: Here's the concrete file structure:

```
pjrt-plugin-tt/
├── CMakeLists.txt                    # Build system
├── src/
│   ├── pjrt_api.cc                   # GetPjrtApi() entry point (~300 lines)
│   ├── tt_client.h/cc                # Device open/close, CompileStableHLO() (~200 lines)
│   ├── tt_device.h/cc                # Device description, memory spaces (~100 lines)
│   ├── tt_buffer.h/cc                # to_device/from_device with tile padding (~250 lines)
│   ├── tt_executable.h/cc            # Walk StableHLO, dispatch to TT-NN (~400 lines)
│   ├── stablehlo_parser.h/cc         # Parse StableHLO bytecode via MLIR (~150 lines)
│   ├── ops/
│   │   ├── registry.h                # Op handler type + registration (~50 lines)
│   │   ├── arithmetic.cc             # add, sub, mul, div, exp, log, etc. (~300 lines)
│   │   ├── shape.cc                  # broadcast_in_dim, reshape, transpose (~250 lines)
│   │   ├── reduction.cc              # reduce (sum, max, min, prod) (~150 lines)
│   │   ├── linalg.cc                 # dot_general, conv (~200 lines)
│   │   └── data.cc                   # constant, iota, select, gather, slice (~300 lines)
│   └── pjrt_stubs.cc                 # Stubbed PJRT functions (~200 lines)
├── python/
│   └── jax_plugins/tt/__init__.py    # Plugin registration (~40 lines)
├── tests/
│   └── test_basic.py                 # JAX-level tests (~200 lines)
└── third_party/
    ├── mlir/                         # MLIR/StableHLO headers + static libs
    └── tt-metal/                     # TT-NN headers + libs
```

**Estimated total: ~2,600 lines of C++, ~240 lines of Python.**

For reference, our current Python interpreter is ~585 lines across 4 files. The PJRT plugin would be roughly 4-5x more code, mostly due to C++ boilerplate, PJRT API wiring, and StableHLO parsing.

**Q: What's the build system?**

A: **CMake**, following jax-mps's approach. Key dependencies:

```cmake
cmake_minimum_required(VERSION 3.20)
project(pjrt_plugin_tt)

# Find TT-Metal/TT-NN
find_package(tt-metal REQUIRED)  # or manual path to libtt_metal.so

# MLIR/StableHLO (pre-built static libraries)
# These are ~20 MLIR libs + ~14 StableHLO libs
set(MLIR_LIBS MLIRBytecodeReader MLIRParser MLIRIR MLIRSupport ...)
set(STABLEHLO_LIBS StablehloOps StablehloAssemblyFormat StablehloBytecode ...)

# Main plugin shared library
add_library(pjrt_plugin_tt SHARED
    src/pjrt_api.cc
    src/tt_client.cc
    src/tt_device.cc
    src/tt_buffer.cc
    src/tt_executable.cc
    src/stablehlo_parser.cc
    src/ops/arithmetic.cc
    src/ops/shape.cc
    src/ops/reduction.cc
    src/ops/linalg.cc
    src/ops/data.cc
    src/pjrt_stubs.cc
)

target_link_libraries(pjrt_plugin_tt PRIVATE
    tt_metal ttnn              # TT-NN
    ${MLIR_LIBS}               # MLIR bytecode parsing
    ${STABLEHLO_LIBS}          # StableHLO dialect
    protobuf abseil            # Dependencies
)
```

The hardest part of the build: **getting MLIR/StableHLO static libraries compiled for our Linux host**. jax-mps bundles pre-built MLIR in `third_party/`. We'd need to do the same, or build from source (which takes 30+ minutes).

**Q: How would we link against TT-NN?**

A: TT-NN is already installed on our remote host. We'd link against:

```
/path/to/tt-metal/build/lib/libtt_metal.so
/path/to/tt-metal/build/lib/libttnn.so
```

Plus headers from:
```
/path/to/tt-metal/tt_metal/include/
/path/to/tt-metal/ttnn/include/
```

The TT-NN C++ API mirrors the Python API closely — `ttnn::add()`, `ttnn::matmul()`, `ttnn::reshape()` are all available as C++ functions.

**Q: What's the minimum viable implementation?**

A: To run `y = jax.jit(lambda x: x @ w + b)(input)`:

1. `PJRT_Client_Create` → `ttnn::open_device(0)`
2. `PJRT_Client_BufferFromHostBuffer` → pad to 32-aligned, `ttnn::from_torch(..., TILE_LAYOUT)`
3. `PJRT_Client_Compile` → parse StableHLO, build op dispatch list
4. `PJRT_LoadedExecutable_Execute` → walk ops calling ttnn::matmul, ttnn::add
5. `PJRT_Buffer_ToHostBuffer` → `ttnn::to_torch()`, unpad
6. Destructors → `ttnn::close_device()`

For this minimal case, we need handlers for: `constant`, `dot_general`, `add`, `broadcast_in_dim`, and `convert`. That's 5 StableHLO ops.


## Part 4: The Broadcast Problem

**Q: What broadcast capabilities does TT-NN actually have?**

A: More than we thought! TT-NN provides several on-device broadcast operations:

| Operation | What it does | Layout support |
|-----------|-------------|----------------|
| `ttnn.bcast(a, b, math_op, dim)` | Binary op with broadcasting b over a | BF16 only, H/W/HW dims |
| `ttnn.expand(tensor, shape)` | Expand singleton dims (like torch.expand) | Copies data (not a view) |
| `ttnn.repeat(tensor, reps)` | Tile repetition along dims | General |
| `ttnn.repeat_interleave(tensor, n, dim)` | Repeat elements along a dim | Works with TILE_LAYOUT |

**The key discovery: `ttnn.bcast` supports ADD, SUB, MUL along H, W, or HW dimensions with specific shape constraints.** And `ttnn.expand` does explicit broadcasting of singleton dimensions.

Constraints for `ttnn.bcast`:
- Both inputs must be BF16 (we already use BF16)
- Broadcast dimension can be W (column broadcast), H (row broadcast), or HW (scalar broadcast)
- Shape rules: for W-broadcast, Y dims must match; for H-broadcast, X dims must match

**Q: Could we use ttnn.expand/repeat to eliminate CPU round-trips?**

A: Yes! Our current `broadcast_to_match()` in `tensors.py` does:
```
device → CPU (from_device) → np.broadcast_to → CPU → device (to_device)
```

We could replace this with:
```
ttnn.expand(tensor, target_shape)  # stays on device!
```

Or for cases where we need to replicate along a specific axis:
```
ttnn.repeat(tensor, repetition_vector)  # stays on device!
```

This would eliminate the CPU round-trip that currently breaks trace capture for any graph with broadcasting. The `broadcast_in_dim` handler would become:

```python
def op_broadcast_in_dim(interp, invars, params, eqn):
    a = interp.eval_var(invars[0])
    out_shape = eqn.outvars[0].aval.shape
    in_shape = eqn.invars[0].aval.shape
    if in_shape == out_shape:
        return a
    # Use ttnn.expand instead of CPU round-trip!
    return ttnn.expand(a, list(out_shape))
```

**This is potentially a bigger win than switching to PJRT.** If `ttnn.expand` works reliably with TILE_LAYOUT, we could fix the broadcast problem in our current Python interpreter AND make all ops trace-compatible.

**Q: How does the official tt-xla (via tt-mlir) handle broadcasts?**

A: The tt-mlir compiler handles `stablehlo.broadcast_in_dim` as a first-class MLIR operation. Their approach:

1. **MLIR lowering**: `stablehlo.broadcast_in_dim` is lowered through TTIR dialect → TTNN dialect
2. **Op insertion**: The compiler inserts explicit `ttnn.repeat` or reshape operations as needed
3. **Optimization**: The compiler may fuse the broadcast with the consuming op (e.g., fuse broadcast+add into a single bcast kernel)

The tt-torch compatibility table shows `stablehlo.broadcast_in_dim` is supported with 862+ test cases passing. This confirms that TT-NN can handle broadcast patterns — we just need to use the right API calls.

**Q: Could we solve this at the PJRT/compiler level?**

A: Yes, and this is a key insight. In a PJRT plugin, the `Compile()` step sees the entire graph. We could:

1. **Pattern match**: Detect `broadcast_in_dim` → `add` patterns
2. **Replace with**: `ttnn.bcast(a, b, BcastOpMath::ADD, dim)` — a single fused on-device op
3. **Or insert explicit expand**: Before any binary op with mismatched shapes, insert `ttnn.expand()` to pre-broadcast

This is exactly what jax-mps does — their `broadcast_in_dim` handler calls `mlx::core::broadcast_to()` which is an on-device operation. The only reason we do CPU round-trips is that we didn't know about `ttnn.expand`.


## Part 5: Pros and Cons Analysis

**Q: What do we gain and lose with each approach?**

### Current approach: Python Jaxpr Interpreter

| Dimension | Assessment |
|-----------|-----------|
| **Development effort** | Done for 20 ops. ~585 lines Python. |
| **Performance ceiling** | Limited. CPU round-trips for broadcast. No fusion. Python dispatch overhead (~21us/op, reduced to ~9us with trace). |
| **jax.jit support** | NO. Must manually call `jax.make_jaxpr()` + `interpreter.run()`. |
| **vmap/grad** | NO. Would need separate implementation. |
| **Maintainability** | Excellent. Plain Python dict registry. Anyone can add an op. |
| **Trace capture** | Works but broken by broadcast CPU round-trips. |
| **Ecosystem integration** | None. Libraries (Flax, Optax) can't use our backend transparently. |

### PJRT Plugin approach (jax-mps style)

| Dimension | Assessment |
|-----------|-----------|
| **Development effort** | ~2,600 lines C++. 2-4 weeks for MVP. Build system complexity. |
| **Performance ceiling** | Higher. On-device broadcast. Trace-wrapped execution. Optional MLX-style compilation. |
| **jax.jit support** | YES, automatic. This is the whole point. |
| **vmap/grad** | YES, automatic. JAX handles transformations before lowering to StableHLO. |
| **Maintainability** | Harder. C++ compilation, MLIR dependency, CMake. |
| **Trace capture** | Could wrap entire `Execute()` in trace capture for max speed. |
| **Ecosystem integration** | Full. `jax.devices()` shows "tt", Flax/Optax just work. |

### The critical question: is jax.jit worth 2-4 weeks of C++ work?

**Arguments for yes:**
- `jax.jit` is not just syntactic sugar — it enables XLA-level optimizations, constant folding, dead code elimination, and CSE before the graph even reaches our plugin
- `vmap` and `grad` for free means we could do batched inference and potentially training
- Library compatibility means we could run real models (Flax transformers, etc.) without manual Jaxpr extraction
- Our op registry (`REGISTRY` dict) translates almost 1:1 to C++ handler functions

**Arguments for no:**
- We already have 21/21 tests passing and 179 fwd/sec on a transformer encoder
- The C++ build complexity is significant (MLIR dependencies alone are painful)
- For a course project, demonstrating the concept matters more than production integration
- The broadcast fix (ttnn.expand) would benefit our Python interpreter too

**Q: What's the performance comparison in concrete terms?**

Current Python interpreter path:
```
jax.make_jaxpr(model)(x)   →  Jaxpr (Python IR)
interpreter.run(jaxpr, x)  →  walks 50+ equations, ~21us each
                            →  50 * 21us = 1.05ms Python dispatch
                            →  plus actual compute time
                            →  plus CPU round-trips for broadcast
```

PJRT plugin path:
```
jax.jit(model)(x)          →  StableHLO (compiled once, cached)
plugin.Execute(x)          →  C++ dispatch, <1us per op
                            →  or wrapped in trace: near-zero dispatch
                            →  no CPU round-trips (on-device broadcast)
```

The dispatch overhead gap: **21us/op (Python) vs <1us/op (C++) vs ~0us/op (traced)**. With trace capture in our Python interpreter, we already get close to 0us/op for the dispatch — so the real win from PJRT is broadcast elimination and ecosystem integration, not raw dispatch speed.


## Part 6: Hybrid Approaches

**Q: Is there a middle ground between Python interpreter and full PJRT plugin?**

A: Yes, several options:

### Option A: Fix broadcast first, keep Python interpreter

The single highest-value change: replace CPU round-trip broadcasts with `ttnn.expand()`:

```python
# In tensors.py broadcast_to_match():
# BEFORE (CPU round-trip):
a_np = from_device(a_tt, a_shape)
a_np = np.broadcast_to(a_np, out_shape).copy()
a_tt = to_device(a_np, device)

# AFTER (on-device):
a_tt = ttnn.expand(a_tt, list(out_shape))
```

If this works, our trace capture becomes fully functional for all ops. We'd get:
- All ops trace-compatible (no skip_eqns needed)
- 2-3x additional speedup from trace
- Still no jax.jit, but much faster execution

**Estimated effort: 1-2 hours to test, half a day to integrate.**

### Option B: JAX FFI custom calls

JAX's Foreign Function Interface (`jax.ffi`) lets you register C/C++ functions as JAX custom calls:

```python
# Register a C++ function
jax.ffi.register_ffi_target("tt_matmul", capsule_ptr)

# Call it from JAX (works with jax.jit!)
result = jax.ffi.ffi_call(
    "tt_matmul",
    result_shape_dtypes=jax.ShapeDtypeStruct(out_shape, jnp.float32),
    x, w
)
```

This works with `jax.jit` because it becomes a `custom_call` in the HLO graph. But:
- You'd need to write C++ wrappers for each TT-NN op
- No automatic vmap support (must provide `vmap_method`)
- No automatic grad support (must use `jax.custom_vjp`)
- Essentially building a PJRT plugin piecemeal, with more boilerplate

**Verdict: Not recommended. If you're writing C++ anyway, build the PJRT plugin properly.**

### Option C: Register custom JAX primitives (pure Python)

You can register new JAX primitives that dispatch to TT-NN:

```python
import jax
from jax import core

tt_matmul_p = core.Primitive('tt_matmul')

@tt_matmul_p.def_impl
def tt_matmul_impl(x, w):
    # Move to device, compute, move back
    x_tt = tensors.to_device(x, device)
    w_tt = tensors.to_device(w, device)
    return tensors.from_device(ttnn.matmul(x_tt, w_tt), out_shape)

@tt_matmul_p.def_abstract_eval
def tt_matmul_abstract(x, w):
    return core.ShapedArray((x.shape[0], w.shape[1]), x.dtype)
```

This gives you `jax.jit` compatibility (the primitive becomes an XLA custom_call). But:
- Each call does host→device→host round-trip (no persistent device buffers)
- No graph-level optimization
- Essentially eager mode with extra steps

**Verdict: Useful for testing individual ops, not viable for performance.**

### Option D: Progressive migration (recommended for our timeline)

A phased approach:

1. **Week 1**: Fix broadcast with `ttnn.expand`. Make all ops trace-compatible. Benchmark the full transformer with trace capture. This alone could double our 179 fwd/sec.

2. **Week 2**: If time permits, start the PJRT plugin skeleton. Use the C→C++ wrapper (`pjrt_c_api_wrapper_impl.h`) that auto-generates C function pointers from a `PjRtClient` subclass. Get `jax.devices()` showing a TT device.

3. **Week 3**: Port our op handlers from Python to C++ (the logic is identical, just different syntax). Wire up Compile → walk StableHLO → dispatch to TT-NN.

4. **Week 4**: Integration testing. Run the transformer encoder through `jax.jit` on the TT device.

**Q: What would Phase 2 look like concretely?**

The minimum skeleton to get `jax.devices()` to show a Tenstorrent device:

```cpp
// tt_client.h
class TtClient : public xla::PjRtClient {
 public:
  TtClient() {
    device_ = ttnn::open_device(0);
  }

  absl::string_view platform_name() const override { return "tt"; }
  int device_count() const override { return 1; }
  int addressable_device_count() const override { return 1; }

  // The key method — receives StableHLO, returns executable
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation,
      CompileOptions options) override;

  // Host → device
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type,
      absl::Span<const int64_t> dims, ...) override;

 private:
  ttnn::Device* device_;
};
```

```python
# jax_plugins/tt/__init__.py
def initialize():
    path = os.path.join(os.path.dirname(__file__), 'pjrt_plugin_tt.so')
    xb.register_plugin('tt', priority=500, library_path=path)
```

Then: `python -c "import jax; print(jax.devices())"` → `[TtDevice(id=0)]`


## Part 7: Decision Framework

**Q: Given where we are (21/21 tests, 179 fwd/sec transformer), what should we do next?**

Decision tree:

```
Is broadcast the #1 bottleneck?
├── YES → Fix broadcast with ttnn.expand (Option A, 1 day)
│         Then benchmark transformer with full trace capture
│         ├── If 300+ fwd/sec → Ship it. Write the wiki entry. Move on.
│         └── If <300 fwd/sec → Investigate other bottlenecks
│
└── Is jax.jit ecosystem integration the goal?
    ├── YES → Build PJRT plugin (Option D phases 2-4, 2-3 weeks)
    └── NO → Polish the Python interpreter, add more ops, write paper
```

**The broadcast fix is the highest-leverage change regardless of which path we choose.** If `ttnn.expand` works on-device with TILE_LAYOUT:
- Python interpreter: trace capture works for ALL ops, 2-3x speedup
- PJRT plugin: the #1 hard problem is already solved

**Q: What would convince us to go full PJRT?**

Three criteria:
1. We've exhausted Python interpreter performance (broadcast fixed, trace working, still want more)
2. We need library compatibility (Flax, Optax, etc.) for a specific demo
3. We have 2+ weeks of development time remaining

If all three are true, the PJRT plugin is worth building. jax-mps proved it's feasible for a small team — they went from zero to 91.5% JAX test compatibility.

## Key Takeaways

1. **jax-mps is our template**: ~24 C++ files, ~2,600 lines, walks StableHLO and dispatches to MLX. We'd do the same but dispatch to TT-NN.

2. **Our op registry translates directly**: 18/20 Jaxpr primitives have exact StableHLO equivalents. The code structure is isomorphic.

3. **The broadcast fix is independent of the PJRT decision**: Use `ttnn.expand()` or `ttnn.repeat()` to eliminate CPU round-trips. Test this FIRST.

4. **PJRT's real value is ecosystem integration**: `jax.jit`, `vmap`, `grad`, and library compatibility. Raw execution speed can be matched by trace capture.

5. **The build system is the hardest part**: Linking against MLIR (~20 libs), StableHLO (~14 libs), and TT-Metal simultaneously requires careful CMake work.

6. **Progressive migration is possible**: Fix broadcast → add trace → (optionally) build PJRT shell → port handlers to C++.

## Sources

- jax-mps: https://github.com/tillahoffmann/jax-mps
- applejax (enhanced fork): https://github.com/danielpcox/applejax
- StableHLO spec: https://openxla.org/stablehlo/spec
- PJRT C++ API overview: https://openxla.org/xla/pjrt/cpp_api_overview
- PJRT plugin integration guide: https://openxla.org/xla/pjrt/pjrt_integration
- Official tt-xla: https://github.com/tenstorrent/tt-xla
- JAX FFI docs: https://docs.jax.dev/en/latest/ffi.html
- TT-NN API docs: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html
- TT-NN broadcast_in_dim support: https://docs.tenstorrent.com/tt-torch/ops/stablehlo/stablehlo.broadcast_in_dim.html
