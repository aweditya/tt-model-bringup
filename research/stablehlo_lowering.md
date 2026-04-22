# StableHLO Lowering Research: From JAX to TT-NN via StableHLO

## 1. StableHLO Basics

### What is StableHLO?

StableHLO is an operation set for high-level operations (HLO) in machine learning. It serves as a **portability layer** between ML frameworks (JAX, PyTorch, TensorFlow) and ML compilers/hardware backends. It is an MLIR dialect with ~100 formally specified operations, serializable as MLIR bytecode, and designed with strict compatibility guarantees: **5 years backward compatibility, 2 years forward compatibility**.

### The HLO Family Tree: HLO -> MHLO -> StableHLO

Three dialects form a lineage:

```
HLO (XLA's original internal IR)
 |
 v
MHLO ("meta"-HLO, MLIR dialect version of HLO, adds dynamic shape support)
 |
 v
StableHLO (stabilized MHLO with formal spec, versioning, and compatibility guarantees)
```

**HLO** is XLA's original internal representation. It lives inside the XLA compiler and is tightly coupled to XLA's optimization passes. Not designed for external consumption -- its format can change between XLA releases without notice.

**MHLO** (now in `tensorflow/mlir-hlo`) was the first MLIR-ification of HLO. It added dynamic shape support and exposed HLO ops as proper MLIR operations. But it lacked stability guarantees -- ops could be added/removed/changed between releases.

**StableHLO** (in `openxla/stablehlo`) is the stabilized successor to MHLO. It uses MLIR bytecode as its serialization format and provides versioning via the VHLO (Versioned HLO) dialect. The key insight: StableHLO is what you build against if you want your compiler/backend to survive JAX version upgrades.

For our PJRT plugin, we only care about StableHLO. We never need to touch HLO or MHLO directly.

### There is also CHLO

CHLO ("client" HLO) is a higher-level dialect containing ops like `erf`, `top_k`, `acos`, `sinh`, `erf_inv` that decompose into StableHLO primitives. Some backends (applejax) implement CHLO ops directly for performance, but they can always be decomposed. We can ignore CHLO initially.

### The Complete StableHLO Op Set (107 ops)

Organized by category (from the official spec and interpreter status page):

**Elementwise (48 ops):**
abs, add, and, atan2, bitcast_convert, cbrt, ceil, clamp, compare, complex, convert, cosine, count_leading_zeros, divide, exponential, exponential_minus_one, floor, imag, is_finite, log, log_plus_one, logistic, map, maximum, minimum, multiply, negate, not, or, popcnt, power, real, reduce_precision, remainder, round_nearest_afz, round_nearest_even, rsqrt, select, shift_left, shift_right_arithmetic, shift_right_logical, sign, sine, sqrt, subtract, tan, tanh, xor

**Data Movement (12 ops):**
broadcast_in_dim, concatenate, dynamic_slice, dynamic_update_slice, gather, pad, reshape, reverse, scatter, slice, sort, transpose

**Reduction (5 ops):**
convolution, dot_general, reduce, reduce_window, select_and_scatter

**Control Flow (5 ops):**
after_all, case, if, optimization_barrier, while

**Distribution/Collective (11 ops):**
all_gather, all_reduce, all_to_all, collective_broadcast, collective_permute, infeed, outfeed, partition_id, recv, replica_id, send

**Dynamism (9 ops):**
dynamic_broadcast_in_dim, dynamic_conv, dynamic_gather, dynamic_iota, dynamic_pad, dynamic_reshape, get_dimension_size, real_dynamic_slice, set_dimension_size

**Miscellaneous (10 ops):**
batch_norm_grad, batch_norm_inference, batch_norm_training, cholesky, constant, fft, iota, rng, rng_bit_generator, triangular_solve

**Extensibility (3 ops):**
custom_call, get_tuple_element, tuple

**Quantization (2 ops):**
uniform_dequantize, uniform_quantize

**Modularity (4 ops -- structural, not really "ops"):**
call, func, module, return

### Key Observations About the Op Set

1. **48 elementwise ops are the bulk**, but most are trivial 1:1 mappings (add->ttnn.add, etc.). Many (complex, imag, real, bitwise ops) are irrelevant for inference.

2. **`reduce` is generic**: Unlike Jaxpr's separate `reduce_sum` and `reduce_max`, StableHLO has a single `reduce` op with an inner computation body. The body defines the reduction function (add for sum, maximum for max, etc.). Our PJRT plugin would need to inspect the body to determine which TT-NN reduction to call.

3. **`dot_general` is the same**: Identical semantics to Jaxpr's `dot_general` with `dimension_numbers` specifying contraction and batch dimensions.

4. **`custom_call` is the escape hatch**: Backends use this for fused ops (cuDNN attention, cuBLAS matmul). StableHLO recently added `composite` as a cleaner alternative for representing high-level ops like SDPA with their decompositions.

5. **Distribution ops are for multi-device**: `all_reduce`, `all_gather`, etc. are only needed for multi-device parallelism. Skip entirely for single-device.


## 2. JAX -> StableHLO Pipeline

### The Full Compilation Pipeline

```
Python function f(x)
    |  jax.jit(f) -- traces the function
    v
Jaxpr (JAX's internal Python IR, ~50 primitives)
    |  jax.jit(f).lower(x) -- lowers to StableHLO
    v
StableHLO (MLIR dialect, ~100 ops, serializable bytecode)
    |  jax.jit(f).lower(x).compile() -- sends to backend
    v
PJRT_Client_Compile() receives StableHLO bytecode
    |  Backend-specific compilation
    v
Device code (CUDA PTX, Metal shaders, TT-NN ops, etc.)
```

### How to Inspect StableHLO from JAX

Three methods, from simple to export-ready:

**Method 1: `jax.jit(f).lower(x).as_text()`**
```python
import jax
import jax.numpy as jnp

def f(x, w, b):
    return jnp.dot(x, w) + b

x = jnp.ones((4, 8))
w = jnp.ones((8, 16))
b = jnp.ones((16,))

lowered = jax.jit(f).lower(x, w, b)
print(lowered.as_text())
# Prints the StableHLO module as MLIR text
```

**Method 2: `lowered.compiler_ir(dialect="stablehlo")`**
Returns the MLIR module object for programmatic inspection.

**Method 3: `jax.export.export(jax.jit(f))(*shapes).mlir_module()`**
The export API for cross-framework interop:
```python
from jax import export
input_shapes = [jax.ShapeDtypeStruct(s.shape, s.dtype) for s in [x, w, b]]
exported = export.export(jax.jit(f))(*input_shapes)
stablehlo_module = exported.mlir_module()
```

### What Does StableHLO Look Like for a Simple Function?

For `jnp.add(x, y)` with scalar int32 inputs:

```mlir
module @jit_plus attributes {jax.uses_shape_polymorphism = false} {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}
```

Key features:
- `module @jit_plus` -- named after the jitted function
- `func.func public @main` -- single entry point with typed arguments
- `tensor<i32>` -- fully typed tensors (shape + dtype)
- `stablehlo.add` -- prefixed operation name
- SSA form -- `%0`, `%arg0` are SSA values

### What Does a Transformer Look Like in StableHLO?

Based on our Jaxpr analysis (Wiki 21), a single transformer block has 56 Jaxpr equations. In StableHLO, the same block would look roughly like:

```mlir
module @jit_transformer {
  func.func public @main(%input: tensor<1x32x64xf32>,
                          %wq: tensor<64x64xf32>,
                          %wk: tensor<64x64xf32>,
                          ... ) -> tensor<1x32x64xf32> {
    // Q/K/V projections
    %q = stablehlo.dot_general %input, %wq, ... : (tensor<1x32x64xf32>, tensor<64x64xf32>) -> tensor<1x32x64xf32>
    %k = stablehlo.dot_general %input, %wk, ... : ...
    %v = stablehlo.dot_general %input, %wv, ... : ...

    // Attention scores: Q @ K^T / sqrt(d)
    %kt = stablehlo.transpose %k, dims = [0, 2, 1] : ...
    %scores = stablehlo.dot_general %q, %kt, ... : ...
    %scale = stablehlo.constant dense<0.125> : tensor<f32>  // 1/sqrt(64)
    %scaled = stablehlo.divide %scores, %scale_broadcast : ...

    // Softmax (decomposed, not fused)
    %max = stablehlo.reduce(%scaled, %neg_inf) ... { stablehlo.maximum }
    %max_bc = stablehlo.broadcast_in_dim %max, ... : ...
    %shifted = stablehlo.subtract %scaled, %max_bc : ...
    %exps = stablehlo.exponential %shifted : ...
    %sum = stablehlo.reduce(%exps, %zero) ... { stablehlo.add }
    %sum_bc = stablehlo.broadcast_in_dim %sum, ... : ...
    %attn = stablehlo.divide %exps, %sum_bc : ...

    // Context: attn @ V
    %ctx = stablehlo.dot_general %attn, %v, ... : ...

    // Output projection + residual
    %proj = stablehlo.dot_general %ctx, %wo, ... : ...
    %res1 = stablehlo.add %proj, %input : ...

    // Layer norm (decomposed into ~10 ops each)
    // ... reduce, subtract, power, reduce, add, rsqrt, multiply, add ...

    // FFN
    %ff1 = stablehlo.dot_general %normed, %w1, ... : ...
    %relu = stablehlo.maximum %ff1, %zero_bc : ...  // ReLU = max(x, 0)
    %ff2 = stablehlo.dot_general %relu, %w2, ... : ...

    // Residual + layer norm
    // ...
    return %output : tensor<1x32x64xf32>
  }
}
```

The structure is almost identical to Jaxpr -- same ops, same decompositions, same count (~56 ops). The differences are syntactic (MLIR text format vs Python data structure) and naming (stablehlo.exponential vs exp, stablehlo.subtract vs sub).

### Where Does Our Jaxpr Interpreter Fit vs StableHLO Compiler?

```
                    Current approach (Jaxpr interpreter)
                    ====================================
JAX Python --> trace --> Jaxpr --> [our interpreter.py] --> TT-NN
                         ^
                         |
                         We intercept HERE (before StableHLO lowering)


                    Future approach (StableHLO via PJRT)
                    =====================================
JAX Python --> trace --> Jaxpr --> StableHLO bytecode --> [PJRT plugin] --> TT-NN
                                                          ^
                                                          |
                                                          We intercept HERE
```

The Jaxpr interpreter bypasses the standard compilation pipeline. It works, but it means:
- No `jax.jit` support (we use `jax.make_jaxpr` manually)
- No XLA optimizations (constant folding, CSE, dead code elimination)
- No vmap/grad (those transform Jaxpr before lowering)
- No framework compatibility (Flax, Optax expect a real backend)

A StableHLO compiler via PJRT would integrate properly into the standard pipeline, getting all of the above for free.

### Advantages of StableHLO Over Direct Jaxpr Interpretation

| Dimension | Jaxpr | StableHLO |
|-----------|-------|-----------|
| **Stability** | Internal, can change between JAX versions | Formal spec, 5yr backward compat |
| **Serialization** | Python objects, not serializable | MLIR bytecode, cross-process |
| **Language** | Python only | C++ MLIR APIs (for PJRT plugin) |
| **Framework support** | JAX only | JAX + PyTorch/XLA + TensorFlow |
| **Optimizations** | None (raw trace) | XLA pre-optimizes before sending |
| **jax.jit** | Not supported | Native integration |
| **vmap/grad** | Not supported | Automatic (transforms happen before lowering) |

The bottom line: Jaxpr interpretation is great for prototyping. StableHLO is what you build a production backend against.


## 3. StableHLO -> TT-NN Mapping

### Direct 1:1 Mappings (Already Implemented in Our Jaxpr Interpreter)

| StableHLO Op | TT-NN Equivalent | Notes |
|--------------|-------------------|-------|
| `stablehlo.add` | `ttnn.add` | Direct |
| `stablehlo.subtract` | `ttnn.sub` | Direct |
| `stablehlo.multiply` | `ttnn.mul` | Direct |
| `stablehlo.divide` | `ttnn.mul(a, ttnn.reciprocal(b))` | No native div in TT-NN |
| `stablehlo.negate` | `ttnn.neg` | Direct |
| `stablehlo.exponential` | `ttnn.exp` | Direct |
| `stablehlo.log` | `ttnn.log` | Direct |
| `stablehlo.sqrt` | `ttnn.sqrt` | Direct |
| `stablehlo.rsqrt` | `ttnn.rsqrt` | Direct |
| `stablehlo.maximum` | `ttnn.maximum` (or `ttnn.relu` for max(x,0)) | Pattern-match ReLU |
| `stablehlo.tanh` | `ttnn.tanh` | Direct |
| `stablehlo.dot_general` | `ttnn.matmul` | Same dimension_numbers semantics |
| `stablehlo.broadcast_in_dim` | `ttnn.repeat` (on-device) | Our key broadcast fix |
| `stablehlo.reshape` | `ttnn.reshape` | Direct |
| `stablehlo.transpose` | `ttnn.permute` | Direct |
| `stablehlo.convert` | Pass-through (all bfloat16 on device) | Dtype handling at boundaries |
| `stablehlo.constant` | `ttnn.from_torch(...)` | Materialize constant tensor |
| `stablehlo.concatenate` | `ttnn.concat` | Direct |
| `stablehlo.slice` | CPU fallback (ttnn lacks general slicing) | Needs work |
| `stablehlo.dynamic_slice` | CPU fallback | Needs work |

**Count: 20 ops already have TT-NN implementations** in our Jaxpr interpreter. These translate directly to StableHLO handler functions.

### The `reduce` Op: Key Difference from Jaxpr

In Jaxpr, we have separate `reduce_sum` and `reduce_max` primitives. In StableHLO, there is a single generic `reduce` op with an inner computation body:

```mlir
// StableHLO reduce for sum
%result = stablehlo.reduce(%input, %init) applies {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
    stablehlo.return %sum : tensor<f32>
} dimensions = [2] : (tensor<1x32x64xf32>, tensor<f32>) -> tensor<1x32xf32>
```

Our StableHLO handler would need to:
1. Inspect the inner computation body
2. If the body is a single `add` -> dispatch to `ttnn.sum`
3. If the body is a single `maximum` -> dispatch to `ttnn.max`
4. If the body is a single `minimum` -> dispatch to `ttnn.min`
5. If the body is something else -> CPU fallback

This is a parsing complexity that Jaxpr avoids, but it is straightforward to implement.

### Additional StableHLO Ops Needed for Full Transformer Inference

Beyond our current 20, a complete transformer inference path requires:

| StableHLO Op | TT-NN Mapping | Purpose |
|--------------|---------------|---------|
| `gather` | CPU fallback initially; ttnn.embedding for token lookup | Token embeddings, indexing |
| `scatter` | `ttnn.scatter` or CPU fallback | KV cache updates |
| `dynamic_update_slice` | `ttnn.paged_update_cache` or CPU | KV cache write |
| `pad` | `ttnn.pad` or CPU fallback | Padding for tile alignment |
| `select` | `ttnn.where` | Conditional selection (masking) |
| `compare` | `ttnn.ge`, `ttnn.eq`, etc. | Mask generation |
| `iota` | CPU-generated index tensor | Index arrays for masking |
| `cosine` | `ttnn.cos` | Positional encoding (RoPE) |
| `sine` | `ttnn.sin` | Positional encoding (RoPE) |
| `logistic` | `ttnn.sigmoid` | Sigmoid activation |
| `clamp` | `ttnn.clip` | Value clamping |
| `sign` | `ttnn.sign` | Sign function |
| `floor` | `ttnn.floor` | Floor rounding |
| `reduce_window` | Decompose to reduce + slice | Pooling operations |

**Estimate: ~35 ops for transformer inference, ~50 for broad model coverage.**

### SDPA and Fused Ops: `custom_call` and `composite`

StableHLO represents fused operations two ways:

**`custom_call`**: The traditional escape hatch. XLA uses this for cuDNN SDPA, cuBLAS GEMMs, etc. The call target is a string, and the semantics are opaque to the StableHLO level:
```mlir
%result = stablehlo.custom_call @cudnn_sdpa(%q, %k, %v) : ...
```

**`composite`** (newer): Encapsulates a high-level op with its decomposition. The compiler can choose to use the fused version or the decomposition:
```mlir
%result = stablehlo.composite "sdpa" (%q, %k, %v) {
  decomposition = @sdpa_decomposed
} : ...
```

For our TT-NN backend:
- We could pattern-match the softmax decomposition (reduce_max -> sub -> exp -> reduce_sum -> div) and replace it with `ttnn.softmax`
- We could pattern-match Q@K^T/sqrt(d) @ softmax(scores) @ V and replace with `ttnn.transformer.scaled_dot_product_attention`
- Or we could handle `composite` ops directly if JAX starts emitting them for SDPA

The safest initial approach: don't fuse anything. Handle each op individually. Add fusion as optimization later.

### StableHLO Ops with No TT-NN Equivalent

| StableHLO Op | Situation |
|--------------|-----------|
| `fft` | No TT-NN FFT. CPU fallback. |
| `triangular_solve` | No TT-NN triangular solve. CPU fallback. |
| `cholesky` | No TT-NN Cholesky. CPU fallback. |
| `sort` | Limited TT-NN sort support. CPU fallback likely. |
| `scatter` (general) | TT-NN scatter is limited. CPU fallback for complex cases. |
| `rng` / `rng_bit_generator` | TT-NN has limited RNG. May need host-side generation. |
| All collective ops | Single-device only. Not needed. |
| All dynamic shape ops | Static shapes only. Not needed for fixed-shape inference. |
| Complex number ops | Not needed for standard inference. |
| Bitwise/shift ops | Not needed for standard inference. |


## 4. Existing XLA Backends for Reference

### How the CUDA/GPU Backend Compiles StableHLO

The XLA:GPU pipeline is the gold standard:

```
StableHLO
  |  Legalize to HLO (stablehlo -> hlo conversion)
  v
HLO (XLA's internal representation)
  |  Optimization passes:
  |    - SPMD partitioner (multi-device)
  |    - Layout assignment (memory layout optimization)
  |    - Algebraic simplification
  |    - Constant folding
  |    - Dead code elimination
  |    - Common subexpression elimination
  v
Optimized HLO
  |  FUSION (XLA's most important optimization)
  |    - Groups elementwise chains into single kernels
  |    - Avoids materializing intermediates in HBM
  |    - Each fusion -> one GPU kernel
  v
Fused HLO
  |  Code generation (dual strategy):
  |    - Library calls: cuBLAS for matmul, cuDNN for convolutions/SDPA, NCCL for collectives
  |    - Triton emitters: for fused ops involving matmul + elementwise
  |    - LLVM -> PTX: for pure elementwise fusions, reductions, transposes
  v
CUDA PTX + library calls
  |  CUDA runtime
  v
GPU execution
```

Key insight: **XLA does heavy optimization on HLO before generating code**. Fusion alone can 2-5x performance by eliminating memory round-trips. This is what our Jaxpr interpreter misses entirely -- every op is a separate TT-NN dispatch with its own memory reads/writes.

However, TT-NN's trace capture mechanism partially compensates: by recording and replaying the entire op sequence, we eliminate dispatch overhead (though not the inter-op memory traffic).

### How the TPU Backend Compiles StableHLO

TPU compilation is similar to GPU but with TPU-specific concerns:

```
StableHLO -> HLO -> TPU-optimized HLO -> TPU IR -> hardware instructions
```

Key differences from GPU:
- TPU has a systolic array architecture (like Tenstorrent's matrix engine)
- Layout is critical: TPU needs specific tiling for the systolic array (similar to our TILE_LAYOUT 32-alignment requirement)
- XLA inserts explicit padding/layout-conversion ops for TPU alignment

The TPU backend is most architecturally similar to what a Tenstorrent backend would look like, because both have:
- Tile-based compute (32x32 tiles for TT, 128x128 for TPU)
- Mandatory tile alignment (pad to tile boundaries)
- Explicit memory hierarchy (L1/DRAM for TT, HBM/VMEM for TPU)

### The IREE Project

IREE (Intermediate Representation Execution Environment) is an alternative to XLA for compiling ML models:

```
StableHLO (or TOSA, or Linalg)
  |  IREE frontend
  v
Linalg on tensors
  |  Tiling and distribution
  v
Linalg on buffers
  |  Backend-specific lowering
  v
LLVM IR / SPIR-V / VMVX
  |  Target runtime
  v
CPU / Vulkan GPU / embedded device
```

IREE is relevant because:
1. It accepts StableHLO as input (same as what PJRT sends)
2. It progressively lowers through MLIR dialects (StableHLO -> Linalg -> Vector -> LLVM)
3. It demonstrates how to build a retargetable compiler using MLIR infrastructure
4. It does NOT have a Tenstorrent backend (yet), but the architecture shows how one could be added

### Tenstorrent's tt-mlir: The Official Approach

Tenstorrent's own MLIR compiler follows this pipeline:

```
StableHLO
  |  stablehlo-to-ttir pass
  v
TTIR (high-level tensor ops, hardware-agnostic)
  |  ttir-to-ttnn pass (adds layout, memory space, grid info)
  v
TTNN dialect (maps to tt-nn API, has explicit tile layout and memory)
  |  optimization passes (fusion, sharding, layout)
  v
Optimized TTNN
  |  lowering (optional, for custom kernels)
  v
TTKernel (circular buffers, tile registers, NoC transactions)
  |  or TTMetal (host-side buffer alloc, program enqueue)
  v
Flatbuffer -> tt-metalium runtime -> hardware execution
```

The tt-mlir compiler defines three key dialects:
- **TTIR**: Named ops on tensors, similar to StableHLO/TOSA. Hardware-agnostic.
- **TTNN**: Maps to the tt-nn API. Includes explicit layout info (TILE vs ROW_MAJOR), memory spaces (L1 vs DRAM), and compute kernel configs.
- **TTKernel**: Low-level. Exposes circular buffers, tile registers, NoC transactions, explicit synchronization. For hand-written kernels.

The `compile_stablehlo_to_flatbuffer()` function chains: `build_stablehlo_module` -> `_run_ttir_pipeline` -> `ttnn_to_flatbuffer_file`. The flatbuffer is then executed by the tt-metalium runtime.

**Why we can't use tt-mlir directly**: It lacks full Blackhole support in released packages. The tt-xla plugin (which depends on tt-mlir) doesn't work on our hardware. This is what motivated our Jaxpr interpreter approach in the first place.


## 5. PJRT Plugin Architecture

### What's the Minimal PJRT Plugin?

The PJRT C API requires a shared library (.so) exporting one function:

```c
const PJRT_Api* GetPjrtApi();
```

This returns a struct of ~80 function pointers. For a minimal inference-only plugin, we need 7 functions (see Wiki 22 for details):

1. `PJRT_Client_Create` -- open TT-NN device
2. `PJRT_Client_BufferFromHostBuffer` -- host -> device transfer (with tile padding)
3. `PJRT_Client_Compile` -- receive StableHLO, build dispatch plan
4. `PJRT_LoadedExecutable_Execute` -- run the dispatch plan
5. `PJRT_Buffer_ToHostBuffer` -- device -> host (with unpadding)
6. `PJRT_Buffer_Delete` -- free device buffer
7. `PJRT_LoadedExecutable_Delete` -- free executable

The C++ wrapper (`pjrt_c_api_wrapper_impl.h`) auto-generates all C function pointers from a C++ `PjRtClient` subclass, reducing boilerplate massively.

### How jax-mps / applejax Works (Our Template)

Architecture:
```
JAX Python -> jax.jit traces -> StableHLO bytecode -> PJRT plugin -> MLX GPU execution
```

The plugin receives StableHLO bytecode, parses it using MLIR libraries, walks the ops, and dispatches to MLX (Apple's tensor library). This is **interpretation, not compilation** -- exactly like our Jaxpr interpreter but at the StableHLO level instead of Jaxpr.

Key C++ files:
- `pjrt_api.cc`: Exports `GetPjrtApi()`, assembles function pointers (~300 lines boilerplate)
- `mlx_client.cc`: Device management, `CompileStableHLO()` (~200 lines)
- `mlx_executable.cc`: Walk StableHLO, dispatch to MLX, `Execute()` (~400 lines)
- `mlx_buffer.cc`: Host-device transfers (~250 lines)
- `stablehlo_parser.cc`: Parse StableHLO bytecode via MLIR (~150 lines)
- `ops/*.cc`: Op handlers grouped by category (~1,000 lines total)

Total: ~2,600 lines C++, ~240 lines Python.

applejax implements ~71 StableHLO ops + 12 CHLO ops and achieves 91.5% JAX test compatibility. This proves the approach is viable for a small team.

### How jax-metal (Apple Official) Works

Apple's official plugin uses the OpenXLA compiler (not interpretation):
```
StableHLO -> XLA compiler -> MPSGraph executables -> Metal runtime
```

This gets XLA's optimization passes (fusion, layout assignment, etc.) but requires linking the entire XLA compiler. Much heavier than the jax-mps approach.

### Plugin Registration

Two mechanisms for JAX to discover a plugin:

**Namespace package:**
```python
# jax_plugins/tt/__init__.py
import os, jax._src.xla_bridge as xb

def initialize():
    path = os.path.join(os.path.dirname(__file__), 'pjrt_plugin_tt.so')
    xb.register_plugin('tt', priority=500, library_path=path)
```

**Entry point in pyproject.toml:**
```toml
[project.entry-points.'jax_plugins']
tt = "jax_plugins.tt"
```

User selects the backend: `JAX_PLATFORMS=tt python my_model.py`

### Compilation vs Execution Path

**Compilation** (happens once per unique function+shapes):
1. JAX calls `PJRT_Client_Compile(stablehlo_bytecode)`
2. Plugin parses StableHLO bytecode into MLIR ops
3. Plugin walks ops and builds a dispatch plan (list of TT-NN calls)
4. Returns a handle to the executable

**Execution** (happens every call):
1. JAX calls `PJRT_LoadedExecutable_Execute(input_buffers)`
2. Plugin runs the dispatch plan: for each op, call the TT-NN function
3. Returns output buffers

With trace capture, execution becomes:
1. First call: run dispatch plan while recording a TT-NN trace
2. Subsequent calls: overwrite input buffers, replay the trace

### Buffer Management

The hardest part of the PJRT interface for Tenstorrent:

**Host -> Device** (`PJRT_Client_BufferFromHostBuffer`):
1. Receive raw host buffer (numpy array data pointer)
2. Pad shape to 32-aligned (TILE_LAYOUT requirement)
3. Convert dtype if needed (float32 -> bfloat16)
4. Call `ttnn.from_torch(padded_tensor, layout=TILE_LAYOUT, device=device)`

**Device -> Host** (`PJRT_Buffer_ToHostBuffer`):
1. Call `ttnn.to_torch(tt_tensor)`
2. Unpad to original shape (remove tile padding)
3. Convert dtype back (bfloat16 -> float32)
4. Copy to host buffer

This maps directly to our `tensors.py` `to_device()` / `from_device()` functions.


## 6. Our Path Forward

### Current State: Jaxpr Interpreter

What we have:
- 28 Jaxpr primitives implemented in `ops.py` (~460 lines)
- Python interpreter in `interpret.py` (~120 lines)
- Tensor utilities in `tensors.py` (tile padding, host/device transfer)
- Working transformer inference: 179 fwd/sec (experiment 20), up to 348 with on-device broadcast
- Working GPT-2 and Qwen-0.5B text generation
- Trace capture for fixed-shape workloads

What we lack:
- No `jax.jit` integration (manual `jax.make_jaxpr` + `interpreter.run`)
- No vmap/grad support
- No Flax/Optax compatibility
- Jaxpr format can change between JAX versions

### Future State: StableHLO via PJRT

What we would gain:
- `jax.jit` works natively: `y = jax.jit(f)(x)` dispatches to Tenstorrent
- `vmap` and `grad` for free (JAX handles these before StableHLO lowering)
- XLA pre-optimization (constant folding, CSE, DCE) before reaching our backend
- Framework compatibility (Flax, Optax, other JAX libraries)
- Stable interface that won't break with JAX upgrades
- Cross-framework support (PyTorch/XLA could also target our plugin)

### Could We Incrementally Add StableHLO Support?

Yes. The progression would be:

**Phase 0 (done)**: Jaxpr interpreter with 28 ops, working inference.

**Phase 1 (1-2 days)**: Ensure all ops are on-device (broadcast fix with `ttnn.repeat`). Enable full trace capture. This benefits us regardless of which path we take.

**Phase 2 (1 week)**: Build PJRT plugin skeleton in C++.
- `PJRT_Client_Create` -> `ttnn::open_device(0)`
- Buffer management (host/device transfer with tile padding)
- `jax.devices()` returns `[TtDevice(id=0)]`
- No compilation yet -- just device discovery and buffer management

**Phase 3 (1-2 weeks)**: Add StableHLO parsing and op dispatch.
- Link against MLIR + StableHLO libraries (the hardest build system task)
- Parse StableHLO bytecode in `Compile()`
- Walk ops and dispatch to TT-NN (translate our Python `REGISTRY` to C++)
- Start with 5 ops: `constant`, `dot_general`, `add`, `broadcast_in_dim`, `convert`
- Test: `y = jax.jit(lambda x: x @ w + b)(input)` runs on Tenstorrent

**Phase 4 (1 week)**: Port remaining ops and add trace capture.
- Translate all 28 op handlers from Python to C++
- Wrap `Execute()` in TT-NN trace capture for maximum speed
- Run the transformer through `jax.jit`

### What Would a Minimal StableHLO -> TT-NN Compiler Look Like?

Not a "compiler" in the traditional sense -- it's an **interpreter** (like jax-mps), walking StableHLO ops and dispatching to TT-NN library calls. The code structure mirrors our Python interpreter:

```cpp
// C++ equivalent of our Python REGISTRY
using OpHandler = std::function<ttnn::Tensor(
    std::vector<ttnn::Tensor>& intermediates,
    mlir::Operation& op)>;

std::unordered_map<std::string, OpHandler> REGISTRY = {
    {"stablehlo.add", [](auto& t, auto& op) {
        return ttnn::add(t[getInput(op, 0)], t[getInput(op, 1)]);
    }},
    {"stablehlo.multiply", [](auto& t, auto& op) {
        return ttnn::mul(t[getInput(op, 0)], t[getInput(op, 1)]);
    }},
    {"stablehlo.dot_general", [](auto& t, auto& op) {
        return ttnn::matmul(t[getInput(op, 0)], t[getInput(op, 1)]);
    }},
    // ... same pattern for all ops
};

// C++ equivalent of our Python interpreter.run()
std::vector<ttnn::Tensor> execute(
    mlir::func::FuncOp func,
    std::vector<ttnn::Tensor>& inputs) {

    std::vector<ttnn::Tensor> intermediates(numValues);

    // Bind inputs
    for (int i = 0; i < inputs.size(); i++)
        intermediates[i] = inputs[i];

    // Walk ops and dispatch
    for (auto& op : func.getOps()) {
        auto name = op.getName().getStringRef().str();
        auto handler = REGISTRY.at(name);
        intermediates[getOutput(op)] = handler(intermediates, op);
    }

    return getResults(intermediates, func);
}
```

This is structurally identical to `interpret.py`. The logic doesn't change -- only the language (C++ vs Python), the source IR (StableHLO MLIR vs Jaxpr), and the integration mechanism (PJRT vs manual invocation).

### The Build System Challenge

The hardest part of a PJRT plugin is NOT the code -- it's getting the dependencies to link:

| Dependency | What it provides | Challenge |
|------------|-----------------|-----------|
| MLIR (~20 static libs) | MLIR IR, bytecode reader, parser | Must build from source or extract from JAX wheel |
| StableHLO (~14 static libs) | StableHLO dialect, bytecode support | Must match the version JAX was built with |
| TT-Metal / TT-NN | Device runtime, tensor ops | Already installed on remote host |
| Protobuf, Abseil | Transitive deps of MLIR | Version conflicts are common |

jax-mps solves this by bundling pre-built MLIR/StableHLO in `third_party/`. Building MLIR from source takes 30+ minutes and significant disk space.

An alternative: extract the pre-built MLIR/StableHLO libraries from the installed `jaxlib` wheel, since JAX already ships with these built for the target platform.

### Realistic Assessment

For CS440LX, the pragmatic path is:
1. **Keep the Jaxpr interpreter** as our primary execution engine
2. **Fix broadcast** with `ttnn.repeat` to enable full trace capture (Phase 1)
3. **Document the StableHLO path** thoroughly (this document) so the next person can build it
4. **Optionally** build the PJRT skeleton (Phase 2) as a proof-of-concept

The Jaxpr interpreter has already proven the core thesis: TT-NN can execute transformer workloads dispatched from JAX IR. The PJRT plugin is an engineering project that wraps this same capability in the standard interface. The hard research is done; the remaining work is integration.


## Sources

- StableHLO specification: https://openxla.org/stablehlo/spec
- StableHLO overview: https://openxla.org/stablehlo
- StableHLO interpreter status: https://openxla.org/stablehlo/interpreter_status
- StableHLO GitHub: https://github.com/openxla/stablehlo
- JAX AOT lowering docs: https://docs.jax.dev/en/latest/aot.html
- JAX export tutorial: https://openxla.org/stablehlo/tutorials/jax-export
- XLA:GPU architecture: https://openxla.org/xla/gpu_architecture
- PJRT plugin integration: https://openxla.org/xla/pjrt/pjrt_integration
- PJRT plugin blog post: https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html
- jax-mps: https://github.com/tillahoffmann/jax-mps
- applejax: https://github.com/danielpcox/applejax
- tt-mlir: https://github.com/tenstorrent/tt-mlir
- tt-mlir docs: https://docs.tenstorrent.com/tt-mlir/
- tt-mlir StableHLO builder: https://docs.tenstorrent.com/tt-mlir/builder/stablehlo-builder.html
- IREE project: https://iree.dev/
- IREE GitHub: https://github.com/iree-org/iree
- jax-metal (Apple official): https://developer.apple.com/metal/jax/
- StableHLO composite for SDPA discussion: https://groups.google.com/a/openxla.org/g/openxla-discuss/c/kRe0B1mugZI
