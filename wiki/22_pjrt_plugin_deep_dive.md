# 22: PJRT Plugin Deep Dive — How JAX Talks to Hardware

## What is PJRT?

**Q: What is PJRT and why does it matter for our project?**

A: PJRT (Portable Runtime) is the **plugin interface** that JAX uses to talk to hardware backends. Every device that JAX supports — CUDA GPUs, TPUs, Apple Silicon (MPS), Intel SYCL — implements a PJRT plugin. It's the single integration point between JAX and any accelerator.

The flow: `JAX Python → XLA compiler → StableHLO IR → PJRT plugin → hardware`

## The C API

**Q: What does a PJRT plugin look like at the C level?**

The `PJRT_Api` struct (defined in `xla/pjrt/c/pjrt_c_api.h`) contains **~50+ function pointers**. Your shared library (.so) must export one function:

```c
const PJRT_Api* GetPjrtApi();
```

JAX finds this via `dlsym()` after `dlopen()`. The function pointers group into four categories:

| Category | Key Functions | Purpose |
|----------|-------------|---------|
| Client | `PJRT_Client_Create`, `Client_Devices`, `Client_Compile`, `Client_BufferFromHostBuffer` | Create device context, compile programs, allocate buffers |
| Device | `PJRT_Device_Id`, `Device_Description`, `Device_MemorySpaces` | Enumerate and describe hardware |
| Buffer | `PJRT_Buffer_ToHostBuffer`, `Buffer_Delete`, `Buffer_GetReadyFuture` | Host↔device transfers |
| Executable | `PJRT_LoadedExecutable_Execute`, `LoadedExecutable_Delete` | Run compiled programs |

**Q: Do I really need to implement all ~50 functions?**

No. The **C++ wrapper shortcut** (`pjrt_c_api_wrapper_impl.h`) auto-generates all C function pointers from a C++ `PjRtClient` subclass. You implement one class + `PJRT_Client_Create`; the wrapper handles the rest.

## The Minimal Path for Forward-Only Inference

**Q: What's the absolute minimum for running inference?**

Seven functions:
1. `PJRT_Client_Create` — create your TT-NN device client
2. `PJRT_Client_BufferFromHostBuffer` — host → device transfer
3. `PJRT_Client_Compile` — receive StableHLO, produce executable
4. `PJRT_LoadedExecutable_Execute` — run the compiled graph
5. `PJRT_Buffer_ToHostBuffer` — device → host transfer
6. `PJRT_Buffer_Delete` — cleanup
7. `PJRT_LoadedExecutable_Delete` — cleanup

## Plugin Registration with JAX

**Q: How does JAX discover a PJRT plugin?**

Two mechanisms:

1. **Namespace package**: Place a module under `jax_plugins/` with an `initialize()` function:
```python
import jax._src.xla_bridge as xb
def initialize():
    path = os.path.join(os.path.dirname(__file__), 'pjrt_plugin_tt.so')
    xb.register_plugin('tt', priority=500, library_path=path)
```

2. **Entry point**: Declare in `pyproject.toml`:
```toml
[project.entry-points.'jax_plugins']
tt = "jax_plugins.tt"
```

## How tt-xla (Official Tenstorrent Plugin) Works

**Q: What approach does the official tt-xla take?**

Their pipeline: **JAX → XLA HLO → TT-MLIR → TT-NN**

Key findings:
- **84% Python / 14% C++** — the C++ core implements the PJRT API; Python handles packaging and tests
- They do **not** go directly from HLO to TT-NN. They depend on **tt-mlir** (Tenstorrent's MLIR compiler) for the heavy lifting
- 19 C++ implementation files: `client_instance.cc`, `device_instance.cc`, `buffer_instance.cc`, `executable_instance.cc`, etc.
- Entry point: `dylib_entry_point.cc` → `api_bindings.cc` binds all PJRT functions
- **~48 supported ops** tested individually (unary math, binary arithmetic, comparisons, conv, dot_general, gather, scatter, reduce, reshape)
- Model-level tests: MLP, MNIST, ResNet v1.5
- Many PJRT functions remain stubbed (memory stats, cost analysis, async tracking)

**Q: Why doesn't tt-xla work on our Blackhole?**

The critical dependency is **tt-mlir**, not tt-xla itself. Since tt-xla delegates all compilation to `TTMLIRCompiler`/`TTMLIRRuntime`, Blackhole support depends entirely on whether tt-mlir supports the Blackhole architecture. Our installation failures likely stem from tt-mlir (or its transitive dependency tt-metalium) not yet having full Blackhole support in their released packages. The tt-xla repo itself is hardware-agnostic.

## Our Options

**Q: What paths could we take to build a real JAX backend?**

Three viable approaches, ordered by effort:

### Path 1: Jaxpr Interpreter (what we have now)
- **How**: Intercept at `jax.make_jaxpr` level, map primitives to TT-NN in Python
- **Pros**: Already working! 19/19 tests pass. No C++ needed. Fast iteration.
- **Cons**: No `jax.jit` support, no XLA optimizations, we handle scheduling ourselves
- **Effort**: Done for basic ops. ~1 week for full transformer.

### Path 2: StableHLO Interpreter via PJRT Plugin
- **How**: Implement a minimal PJRT plugin whose `Compile` step receives StableHLO, maps ops to TT-NN
- **Pros**: Gets `jax.jit` support, integrates with JAX ecosystem properly
- **Cons**: Must parse StableHLO (MLIR dialect), need C++ for the plugin shell
- **Effort**: ~2-4 weeks. This is what jax-mps (Apple Silicon) does.

### Path 3: Full tt-xla Integration
- **How**: Fix the tt-mlir dependency chain for Blackhole
- **Pros**: Production-quality, maintained by Tenstorrent
- **Cons**: Blocked on tt-mlir Blackhole support (external dependency)
- **Effort**: Unknown, depends on Tenstorrent's release schedule

### Recommendation

For CS440LX: **Path 1 is the win**. We've already demonstrated the core concept. Path 2 is the stretch goal if we want the full `jax.jit` experience. Path 3 is out of our control.

## Key Insight

The gap between our Jaxpr interpreter and a "real" PJRT plugin is not the op mapping (we've already done that) — it's the **compilation interface**. A PJRT plugin receives StableHLO (an MLIR dialect), not Jaxpr. But StableHLO ops are a superset of Jaxpr primitives, so our TT-NN op registry translates almost directly.
