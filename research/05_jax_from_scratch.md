# JAX From Scratch: What It Is and Why It Matters

## What is JAX?

JAX = "Just Another XLA" — a Python library for numerical computing by Google.

**Core idea**: Take NumPy-like code, trace it into a computation graph, then compile it with XLA for fast execution on accelerators (GPU, TPU, or custom hardware via PJRT).

## The Three Key Transformations

JAX is built around **function transformations** — you write a function, then transform it:

1. **`jax.jit`** — JIT compile a function with XLA. First call traces + compiles; subsequent calls reuse compiled code.
2. **`jax.grad`** — Automatic differentiation. Returns a new function that computes gradients.
3. **`jax.vmap`** — Automatic vectorization/batching. Turns a function that operates on single examples into one that operates on batches.

Plus `jax.pmap` for multi-device parallelism.

## JAX vs PyTorch — The Key Difference

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Paradigm | Imperative, OOP | Functional, transformations |
| Graphs | Dynamic (eager by default) | Static (traced by `jit`) |
| State | Mutable (in-place ops, model.parameters()) | Immutable (pure functions, explicit state) |
| Compilation | Optional (torch.compile) | Core feature (jit) |
| Ecosystem | Huge (torchvision, HuggingFace, etc.) | Smaller but growing (Flax, Orbax, etc.) |
| XLA | Via torch-xla (bolted on) | Native (built around XLA) |

**Key insight**: JAX's functional + compiled approach means XLA sees the **entire computation graph** at once, enabling whole-program optimizations that are impossible with eager execution.

## The Compilation Pipeline

```
Python function
    ↓ jax.jit() traces the function
Jaxpr (JAX's internal IR — a functional trace)
    �� lowered to
StableHLO (MLIR dialect — portable, versioned)
    ↓ consumed by
XLA Compiler (or alternative compiler via PJRT)
    ↓ HLO optimization passes (fusion, layout, etc.)
Device-specific code (PTX for GPU, HLO→TPU, etc.)
    ↓
Execution on hardware
```

## Why the solidsf.com rvLLM Result is Impressive

Source: https://docs.solidsf.com/docs/bench

**What they achieved**: 16,794 tok/s on Gemma 4 E4B using TPU v6e-4, in ~500 lines of JAX with zero custom kernels.

**Why JAX enabled this**:
1. `jax.jit` compiled the **entire forward pass into a single fused while loop** — no Python overhead, no kernel dispatch overhead between layers
2. XLA's fusion passes automatically merged operations that would be separate kernel launches on GPU
3. SPMD tensor parallelism across 4 TPU chips with minimal code (`jax.sharding`)
4. int8 quantization with bf16 compute — expressed declaratively, optimized by compiler

**Contrast with GPU approach**: Their H100 implementation needed ~1350 CUDA graph nodes, 14 kernel launches per layer, custom fused kernels for quant/RoPE/residual — way more engineering effort for lower performance.

**The lesson**: When the compiler can see everything, it can optimize everything. JAX gives the compiler maximum visibility.

## What is StableHLO?

StableHLO = Stable High-Level Operations. An MLIR dialect with ~100 operations for ML.

**Role**: The portability layer between frameworks (JAX, PyTorch, TensorFlow) and compilers (XLA, IREE, TT-MLIR). Think of it as the "LLVM IR of ML" — a stable interface that decouples frontends from backends.

**Key properties**:
- Backward compatible for 5 years, forward compatible for 2 years
- ~100 ops (matmul, conv, reduce, gather, scatter, etc.)
- Supports dynamism and quantization
- Based on MLIR infrastructure

## What is MLIR?

MLIR = Multi-Level Intermediate Representation. A compiler framework (part of LLVM project) that lets you:

1. Define **dialects** — domain-specific IR vocabularies (e.g., `stablehlo`, `linalg`, `arith`, `llvm`)
2. Write **passes** — transformations that optimize or lower one dialect to another
3. **Progressively lower** from high-level to low-level: `stablehlo` → `linalg` → `scf` → `llvm`

**Why it matters for TT-XLA**: Tenstorrent's `tt-mlir` compiler defines its own MLIR dialects (TTIR, TTNN) that represent operations at different abstraction levels for their hardware.

## The Full TT-XLA Pipeline

```
JAX/PyTorch model
    ↓ jit trace / torch.compile
VHLO (versioned HLO)
    ↓
StableHLO (MLIR dialect — framework-level tensor ops)
    ��� tt-mlir compiler
TTIR (TT Intermediate Representation — hardware-agnostic)
    ↓ tt-mlir lowering passes
TTNN (backend-specific IR modeling TT-NN API calls)
    ��� code generation
TT-NN C++/Python code → TT-Metalium → Blackhole hardware
```

Each `↓` is an MLIR lowering pass that transforms operations from one dialect to another. The IRs are preserved in an `irs/` directory for debugging.

**Codegen outputs**:
- `codegen_py`: Python calling TT-NN (needs tt-xla environment)
- `codegen_cpp`: Standalone C++ calling TT-NN directly (portable!)

Sources:
- rvLLM benchmarks: https://docs.solidsf.com/docs/bench
- StableHLO: https://openxla.org/stablehlo
- MLIR: https://mlir.llvm.org/
- TT-XLA codegen: https://docs.tenstorrent.com/tt-xla/getting_started_codegen.html
- Cornell MLIR overview: https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/mlir/
- Patrick Kidger's JAX guide: https://kidger.site/thoughts/torch2jax/
