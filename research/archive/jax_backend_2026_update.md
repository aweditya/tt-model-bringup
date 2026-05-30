# JAX Backend for Tenstorrent: 2026 Landscape Update

## Date: April 2026

This document updates our earlier research (stablehlo_lowering.md, jax_infrastructure_paths.md, wiki/23) with the current state of PJRT plugin development, vendor backends, and Tenstorrent's own compiler stack.


## 1. PJRT C API: Current State (April 2026)

### API Version and Stability

The PJRT C API is at **version 0.104**. It continues to evolve with incremental additions (roughly 50 minor versions since our last research). Recent additions focus on:

- **Memory statistics** (v0.104): `total_allocation_bytes`, `indefinite_allocations`, `peak_unpadded_heap_bytes` in `GetCompiledMemoryStats`
- **Error handling** (v0.103): `payload` and `enum_payload` on error buffers
- **Parameter memory kinds** (v0.102): `PJRT_Executable_ParameterMemoryKinds`
- **Buffer operations** (v0.98): `PJRT_Buffer_Bitcast`
- **DLPack support**: `PJRT_Client_CreateViewOfDeviceBuffer` and reference counting for zero-copy interop

### ABI Compatibility: Still Not Stable

The critical finding: **PJRT still does not have stable ABI compatibility**. The official documentation states "We will start supporting ABI compatibility soon" -- the same language from 2024. In practice, you must match your plugin build against the exact jaxlib version. The framework provides a 6-week forward compatibility window for minor version updates, but this is fragile.

This means any PJRT plugin we build would need to be rebuilt whenever jaxlib updates. This is the same constraint all third-party plugins face (Intel, Apple, Tenstorrent official).

### Extension Mechanism

PJRT now has a formal extension mechanism: plugins can expose optional or experimental features with their own compatibility guarantees, independent of the core API. This could be useful for TT-NN-specific features (trace capture, L1 memory management) that don't map to the standard PJRT abstraction.

### Plugin Registration: Unchanged

The two mechanisms remain the same:
1. Namespace packages under `jax_plugins/` with `initialize()` calling `xb.register_plugin()`
2. Entry points in `pyproject.toml`

No new registration mechanisms have been added.

### What This Means for Us

The PJRT C API is mature enough to build against. The lack of ABI stability is annoying but manageable -- all existing third-party plugins deal with it. The core set of ~7 functions we identified in our earlier research (Client_Create, BufferFromHostBuffer, Compile, Execute, BufferToHost, Buffer_Delete, Executable_Delete) remains the correct minimal surface.


## 2. What Other Hardware Vendors Have Done

### Apple: jax-metal (Official) and applejax (Community)

**jax-metal** (Apple official):
- Closed-source PJRT plugin, installable via `pip install jax-metal`
- Uses OpenXLA compiler to lower StableHLO -> MPSGraph executables
- Still marked "experimental" as of late 2025, with warnings that not all JAX functionality is correctly supported
- Reports up to 28x CPU speedup on M2 Max
- Ongoing compatibility issues with recent macOS versions (issue #34109, Dec 2025)

**applejax** (community, formerly jax-mps):
- Latest release **v0.9.7** (March 11, 2026), 303 commits, actively maintained
- Implements **71 StableHLO ops + 12 CHLO ops** across 2000+ tests
- Walks StableHLO and dispatches to MLX (interpretation, not compilation)
- Built against jaxlib 0.9.x; requires LLVM and StableHLO pinned to specific commits
- Build process: `setup_deps.sh` (~30 min to build MLIR/StableHLO), then `pip install -e .`
- Achieves modest 3x speedup over CPU for ResNet18 on M4

**Key takeaway**: applejax remains our closest template. It proves a small team can build a working PJRT plugin with ~71 ops and 2000+ passing tests. The 30-minute MLIR build is the main friction point.

### Intel: Extension for OpenXLA

- Active development, supports Intel Data Center GPU Max Series and Arc B-Series
- Uses the XLA compiler (not interpretation) with target-specific passes for Intel GPUs
- Leverages oneAPI performance libraries (oneDNN, oneCCL) for acceleration
- JAX upgraded to v0.5.0 in their latest release
- Supports Python 3.10-3.13
- Added SPMD/scale-up support via oneAPI Collective Communications Library

**Key takeaway**: Intel takes the heavyweight approach -- they fork/extend the XLA compiler. This is impractical for us because it requires deep XLA compiler expertise and massive build infrastructure.

### AMD: ROCm JAX Plugin

- `jax-rocm7-pjrt` and `jax-rocm7-plugin` packages available
- Supports JAX 0.8.2 with ROCm 7.12.0
- Built nightly via GitHub Actions
- Official Docker images released quarterly with each ROCm release
- Covers gfx950, gfx94X, gfx90a, gfx908 GPU architectures

**Key takeaway**: AMD also extends the XLA compiler (adding HIP/ROCm code generation). Like Intel, this is the "big company" approach that requires maintaining a fork of XLA.

### Summary: Two Approaches in the Wild

| Approach | Who uses it | Build complexity | Performance | Flexibility |
|----------|-------------|-----------------|-------------|-------------|
| **Interpretation** (walk StableHLO, dispatch to library) | applejax, jax-mps | Medium (MLIR deps) | Good (library-level) | High |
| **Compilation** (extend XLA compiler) | Intel, AMD, Apple official, Tenstorrent | Very high (XLA fork) | Best (fusion, optimization) | Low |

For us, interpretation is the only viable approach. We don't have the resources to maintain an XLA compiler fork.


## 3. Tenstorrent's Official Compiler Stack: TT-MLIR, TT-Forge, TT-XLA

### TT-MLIR (The Compiler)

- **3,562 commits**, 264 stars, 1.1k open issues, 238 release tags
- Compilation pipeline: **StableHLO -> TTIR -> TTNN -> Flatbuffer -> tt-metalium**
- Built on LLVM MLIR infrastructure
- Targets both Wormhole and Blackhole
- The `compile_stablehlo_to_flatbuffer()` function chains: `build_stablehlo_module` -> `_run_ttir_pipeline` -> `ttnn_to_flatbuffer_file`
- Three compilation targets: TTNN, TTMetal, EmitC
- Has golden tensor support for comparing device output against PyTorch references

### TT-Forge (The Ecosystem)

- **512 commits**, 800+ model variants tested in CI
- Explicitly supports **Blackhole** alongside Wormhole in architecture diagram
- Supported models include: Llama 3.1/3.2 (1B-70B), Qwen 2.5/3 (0.5B-32B), Falcon-3, Phi-1/2/3/3.5, Gemma 1.1/2, ResNet-50, ViT, Stable Diffusion XL
- Multi-chip support for larger models (N300+)
- Pipeline: **StableHLO-IR -> TT-IR -> Graph Passes -> (TTNN-IR/TT-Metal-IR/TTKernel-IR) -> TT-Metalium -> Hardware**

### TT-XLA (The PJRT Frontend)

- **2,251 commits**, latest dev release April 22, 2026
- Installable: `pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/`
- Also available as Docker image: `ghcr.io/tenstorrent/tt-xla-slim:latest`
- Supports both JAX (via PJRT) and PyTorch (via torch-xla/torch.compile)
- **863 open issues** -- significant active development
- Recent issues reveal:
  - vLLM integration work (continuous batching, CPU sampling)
  - `stablehlo.gather` lowered to `ttnn.embedding` corrupting integer vocab IDs (#4329)
  - Training test instability (abort signals)
  - Vision model accuracy issues
  - Multi-chip/distributed training focus
- Build from source requires Ubuntu 24.04, Python 3.12, Clang 20, GCC 13, plus tt-mlir toolchain

### Does TT-XLA Work on Blackhole Now?

This is the critical question. Evidence suggests **yes, with caveats**:

1. TT-Forge explicitly lists Blackhole as supported hardware
2. TT-XLA packages are shipping daily dev releases (the package we'd install)
3. The 863 open issues suggest active bringup, not production stability
4. The p150 tensor core downgrade (140 -> 120 cores, January 2026) indicates active firmware/software co-evolution
5. The example models in docs are MNIST and Tiny Llama -- not exactly pushing the envelope

**We should test this.** A concrete next step is to `pip install pjrt-plugin-tt` on our Blackhole machine and try the documented examples. If they work, our entire PJRT story changes -- we might be able to use the official plugin directly rather than building our own.

### Why We Originally Couldn't Use TT-XLA

Our earlier research (wiki/23) noted that tt-mlir lacked "full Blackhole support in released packages." That assessment was from mid-2025. Given the pace of development (3,562 commits in tt-mlir, daily tt-xla releases), this may have changed. The 800+ model variants in tt-forge CI strongly suggests Blackhole is now a first-class target.


## 4. JAX FFI: A Lighter Alternative?

JAX's Foreign Function Interface (FFI) is worth noting as a potential middle path:

- `jax.ffi.register_ffi_target()` registers a compiled C/C++ function
- `jax.ffi.ffi_call()` invokes it from JAX code
- Works with `jax.jit` -- the FFI call becomes a `custom_call` in the StableHLO graph
- Does NOT require a full PJRT plugin

This could let us wrap individual TT-NN ops (matmul, softmax, SDPA) as FFI targets callable from standard JAX code. The device memory management would still be manual, but we'd get `jax.jit` compilation for the CPU-side graph with FFI escapes to TT-NN for the heavy ops.

**Caveat**: The FFI custom-call API/ABI is still "experimental and can be broken at any time." Not recommended as a primary path, but worth watching.


## 5. Our Three Paths Forward (Updated Assessment)

### Path A: Jaxpr Interpreter (Current)

```
JAX -> jax.make_jaxpr -> our Python interpreter -> TT-NN ops
```

**Status**: Working. 28 ops, trace capture, 4 validated models (0.5B-8B), 132 tok/sec decode.

**Pros**:
- Done. Working now.
- Full control, easy to debug
- Fast iteration on new models

**Cons**:
- No `jax.jit`, no `vmap`, no `grad`
- No Flax/Optax compatibility
- Jaxpr format can change between JAX versions
- Requires manual `jax.make_jaxpr` + `interpreter.run()`

**Verdict**: Solid for research and demos. Not a path to production.

### Path B: Custom PJRT Plugin (applejax-style)

```
JAX -> jax.jit -> StableHLO -> our PJRT plugin (.so) -> TT-NN ops
```

**Updated effort estimate**:
- Phase 1: PJRT skeleton (device discovery, buffer management) -- 1 week
- Phase 2: StableHLO parsing + 5-op MVP (constant, dot_general, add, broadcast_in_dim, convert) -- 1-2 weeks
- Phase 3: Port remaining ~25 ops from Python to C++ -- 1 week
- Phase 4: Trace capture integration -- 3-5 days
- Total: **3-5 weeks, ~2,600 lines C++**

**Build dependency challenge**:
- MLIR libraries (~20 static libs) must be built from source (~30 min) or extracted from jaxlib wheel
- StableHLO libraries (~14 static libs) must match jaxlib version exactly
- Must rebuild when jaxlib updates (no ABI stability)
- applejax's `setup_deps.sh` shows this is manageable but non-trivial

**Pros**:
- Full `jax.jit`, `vmap`, `grad` support
- Ecosystem integration (Flax, Optax, etc.)
- XLA pre-optimization before reaching our backend
- Stable StableHLO interface (5yr backward compat)

**Cons**:
- Significant C++ build complexity
- Must maintain version alignment with jaxlib
- Debugging MLIR C++ on remote host is painful
- 3-5 weeks of engineering work

**Verdict**: The ambitious but feasible path. applejax proves it works with a small team.

### Path C: Use Official TT-XLA (NEW -- Not Previously Considered Seriously)

```
JAX -> jax.jit -> StableHLO -> pjrt-plugin-tt (.so) -> tt-mlir -> TT-NN
```

**What changed**: TT-XLA now ships daily dev releases, TT-Forge claims 800+ model variants on Blackhole, and `pip install pjrt-plugin-tt` is a one-liner.

**The experiment we should run**:
```bash
ssh tenstorrent
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
python -c "import jax; print(jax.devices())"  # Does it show a TT device?
# Then try MNIST or a simple matmul
```

**Pros**:
- Zero development effort if it works
- Maintained by Tenstorrent's compiler team
- Full compilation pipeline with optimization passes
- 800+ model variants in CI

**Cons**:
- 863 open issues suggests rough edges
- We lose control over op dispatch and performance tuning
- May conflict with our existing tt-metal/tt-nn installation
- If something breaks, debugging is through tt-mlir's MLIR pipeline (opaque to us)
- Unknown performance characteristics (does it use trace capture? How does it handle our specific models?)

**Verdict**: Must test before making any decision. If it works on our Blackhole with Qwen/Llama, it obsoletes both Path A and Path B for production use. We'd keep Path A for research/understanding.

### Path D: TT-MLIR Bridge (Hybrid)

```
JAX -> jax.make_jaxpr -> our Python code -> compile_stablehlo_to_flatbuffer() -> tt-metalium
```

**Concept**: Instead of building a full PJRT plugin, use tt-mlir's Python API (`compile_stablehlo_to_flatbuffer()`) to compile StableHLO modules we construct programmatically. This gets tt-mlir's optimization passes without needing a PJRT plugin.

**Pros**:
- Gets tt-mlir's TTIR->TTNN optimization passes (fusion, layout, sharding)
- Python-level integration (no C++ plugin needed)
- We can construct StableHLO from our Jaxpr trace or from `jax.jit(...).lower(...).as_text()`

**Cons**:
- Still no `jax.jit` integration (we'd call the compiler manually)
- Depends on tt-mlir Python bindings being available and working on Blackhole
- Unclear if the flatbuffer execution path is optimized for our use case

**Verdict**: Worth exploring as a stepping stone. Test `compile_stablehlo_to_flatbuffer()` with a simple model to see if it produces correct results on Blackhole.


## 6. Concrete Next Steps (Ordered by Information Value)

### Step 1: Test Official TT-XLA on Blackhole (1 day)

```bash
ssh tenstorrent
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
python -c "
import jax
print(jax.devices())  # Should show TT device

import jax.numpy as jnp
x = jnp.ones((4, 4))
y = jax.jit(lambda x: x @ x)(x)
print(y)  # Should be 4.0 everywhere
"
```

If this works, try Qwen-0.5B via HuggingFace + Flax. If THAT works, our whole strategy changes.

If it fails (likely given the 863 open issues and our specific hardware setup), document the failure mode. This tells us whether the official path is 1 month away or 1 year away from being usable for us.

**Estimated effort**: 2-4 hours including environment setup.

### Step 2: Test tt-mlir StableHLO Builder (1 day)

```python
# Can we use tt-mlir's Python API directly?
from tt_mlir import compile_stablehlo_to_flatbuffer
# Construct a simple StableHLO module
# Compile and execute on Blackhole
```

This tests Path D. If tt-mlir's Python API works, we can incrementally migrate our interpreter to use compiled flatbuffers instead of direct TT-NN calls.

**Estimated effort**: 2-4 hours.

### Step 3: Evaluate Results and Choose Path (1 day)

Based on Steps 1-2:
- If TT-XLA works -> Path C, contribute to upstream, file issues for our use cases
- If tt-mlir Python API works but TT-XLA doesn't -> Path D, use the compiler as a library
- If neither works -> Path A (keep interpreter) or Path B (build custom PJRT)

### Step 4: If Building Custom PJRT (Path B), Start with applejax Fork (3-5 weeks)

1. Fork applejax's build infrastructure
2. Replace MLX dispatch with TT-NN dispatch
3. Start with 5 ops: constant, dot_general, add, broadcast_in_dim, convert
4. Test: `y = jax.jit(lambda x: x @ w + b)(input)` on Tenstorrent
5. Incrementally port our 28 op handlers from Python to C++
6. Add trace capture in Execute()


## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| TT-XLA doesn't work on our Blackhole | Medium | Low (we already have a working path) | Test in Step 1, fall back to our interpreter |
| PJRT ABI breaks with jaxlib update | High | Medium (must rebuild plugin) | Pin jaxlib version, track upstream |
| MLIR build fails on remote host | Medium | High (blocks Path B entirely) | Use applejax's setup_deps.sh as template |
| tt-mlir Blackhole support is incomplete | Medium | Low (we have direct TT-NN fallback) | Test in Step 2 |
| Our interpreter becomes unmaintainable | Low | High | Document thoroughly, limit op count |

## 8. Key Findings Since Last Research

1. **TT-XLA is now installable via pip** with daily dev releases. This was not the case during our initial research. It should be tested immediately.

2. **TT-Forge claims 800+ model variants** including Qwen 2.5/3 (our exact models) on Blackhole. If true, the official stack may already solve our problem.

3. **PJRT ABI stability is still not guaranteed.** All third-party plugins deal with this. It's annoying but not a blocker.

4. **applejax has matured significantly** (v0.9.7, March 2026, 2000+ tests). It's the best template for a custom PJRT plugin.

5. **The Blackhole p150 tensor core count was reduced** from 140 to 120 (January 2026). Our existing experiments may need recalibration if firmware was updated.

6. **JAX FFI exists as a lighter alternative** to full PJRT for wrapping individual ops, but it's experimental and doesn't give full backend integration.

7. **The PJRT C API is at v0.104** with ~50 new minor versions since our last look, but the core plugin interface (the ~7 functions we need) is unchanged.


## Sources

- [PJRT Plugin Integration Guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [PJRT C API Changelog](https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md)
- [PJRT C++ API Overview](https://openxla.org/xla/pjrt/cpp_api_overview)
- [Google PJRT Plugin Blog Post (2024)](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html)
- [applejax GitHub](https://github.com/danielpcox/applejax)
- [jax-mps GitHub](https://github.com/tillahoffmann/jax-mps)
- [Apple Metal JAX](https://developer.apple.com/metal/jax/)
- [Intel Extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla)
- [AMD ROCm JAX](https://github.com/ROCm/rocm-jax)
- [tt-mlir GitHub](https://github.com/tenstorrent/tt-mlir)
- [tt-mlir StableHLO Builder Docs](https://docs.tenstorrent.com/tt-mlir/builder/stablehlo-builder.html)
- [tt-forge GitHub](https://github.com/tenstorrent/tt-forge)
- [tt-xla GitHub](https://github.com/tenstorrent/tt-xla)
- [tt-xla Getting Started](https://docs.tenstorrent.com/tt-xla/getting_started.html)
- [tt-xla Open Issues](https://github.com/tenstorrent/tt-xla/issues)
- [JAX FFI Documentation](https://docs.jax.dev/en/latest/ffi.html)
- [Blackhole p150 Core Downgrade](https://www.tomshardware.com/tech-industry/semiconductors/jim-kellers-tenstorrent-is-downgrading-blackhole-p150-cards-from-140-to-120-tensor-cores-via-firmware-update-will-ship-cards-with-120-tensor-cores-going-forward-company-claims-existing-users-should-expect-1-2-percent-performance-drop)
- [JAX GitHub Discussions on Metal/PJRT](https://github.com/jax-ml/jax/discussions/34648)
