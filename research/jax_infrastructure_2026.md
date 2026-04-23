# JAX/XLA Infrastructure Research for Tenstorrent Backend (April 2026)

## Context

This document surveys the bleeding-edge JAX/XLA infrastructure landscape as of April 2026, evaluated through the lens of building a JAX backend for Tenstorrent Blackhole devices. Our current approach uses TT-NN directly (28 ops, trace capture, 4 validated models, 132 tok/sec decode). The original goal was a JAX/XLA PJRT backend plugin, but direct TT-NN proved more productive for learning the hardware. This document asks: what has changed in the JAX ecosystem that might shift that calculus?

---

## 1. PJRT Plugin Status

### API Maturity

The PJRT C API is at **version 0.104** (up from ~0.54 when we started). Recent additions focus on memory statistics, error payloads, parameter memory kinds, buffer bitcasting, and DLPack zero-copy interop. The core plugin surface -- the ~7 functions needed for a minimal backend (Client_Create, BufferFromHostBuffer, Compile, Execute, BufferToHost, Buffer_Delete, Executable_Delete) -- has been stable for over a year.

The extension mechanism is now formalized: plugins can expose optional features with independent compatibility guarantees. This is directly relevant to TT-NN-specific features (trace capture, L1 memory management, NoC configuration) that have no analog in the standard PJRT abstraction.

### ABI Stability: Still Missing

The critical limitation remains unchanged: **PJRT does not have stable ABI compatibility**. The documentation still says "We will start supporting ABI compatibility soon" -- identical language from 2024. In practice, plugins must be rebuilt against each jaxlib version. A 6-week forward compatibility window exists for minor versions, but it is fragile. Every third-party plugin (Intel, Apple, AMD, Tenstorrent official) deals with this constraint.

### New Hardware Backends (2025-2026)

| Backend | Approach | Status |
|---------|----------|--------|
| **Tenstorrent tt-xla** | Compilation (StableHLO -> tt-mlir -> TTNN) | Daily dev releases, 863+ open issues, Nebula boards only (no Galaxy) |
| **Intel Extension for OpenXLA** | XLA compiler fork + oneAPI | Supports Data Center GPU Max + Arc B-Series, JAX 0.5.0+, SPMD via oneCCL |
| **AMD ROCm** | XLA compiler fork + HIP | jax-rocm7-pjrt packages, nightly builds, covers gfx950/gfx94X/gfx90a |
| **Apple jax-metal** | Closed-source, OpenXLA compiler -> MPSGraph | Still "experimental," up to 28x CPU speedup on M2 Max |
| **applejax** (community) | Interpretation (StableHLO -> MLX dispatch) | v0.9.7, 71 StableHLO ops + 12 CHLO, 2000+ tests, built against jaxlib 0.9.x |
| **jax-mps** (community) | Interpretation (StableHLO -> MLX) | PJRT plugin, ~3.7x speedup over CPU for ResNet18 on M4 |
| **NVIDIA multimesh-jax** | Custom PJRT for MPMD pipeline parallelism | v0.2, enables pipeline parallelism across GPU submeshes, patches JAX/jaxlib |

### What This Means for Us

Two viable approaches exist in the wild:

1. **Interpretation** (walk StableHLO, dispatch to a library): applejax, jax-mps. Medium build complexity, library-level performance, high flexibility. This is the only approach feasible for a small team.
2. **Compilation** (extend/fork XLA compiler): Intel, AMD, Apple official, Tenstorrent. Very high build complexity, best performance via fusion/optimization, low flexibility. Requires a dedicated compiler team.

applejax remains our closest template: a small team, 71 ops, 2000+ tests, interpretation-based, built against jaxlib 0.9.x with a 30-minute MLIR build. If we built a custom PJRT plugin, we would fork their architecture and replace MLX dispatch with TT-NN dispatch.

---

## 2. StableHLO

### Current State

StableHLO continues to mature as the universal ML compiler interface. Key developments since our last research:

- **MHLO deprecation is underway.** In Q4 2024, the project began migrating useful passes (canonicalization, folder patterns) from MHLO to StableHLO. An RFC for external MHLO deprecation timelines was planned. StableHLO is ready to fully supersede MHLO/HLO as the compiler interface.
- **Hardware-independent optimization consolidation.** StableHLO is absorbing graph simplification passes from Google AI Edge and JAX-Enzyme, so all PJRT plugins benefit from a shared optimization baseline before hardware-specific lowering.
- **Componentization.** The XLA project is creating dedicated components for HLO mirroring the StableHLO repo structure, and moving all OpenXLA backends behind PJRT plugins. This reduces coupling between the XLA compiler internals and backend implementations.
- **107 ops, formally specified.** The op set is stable with 5-year backward compatibility and 2-year forward compatibility via the VHLO (Versioned HLO) dialect.

### Can We Target StableHLO Instead of XLA HLO?

Yes, and we should. StableHLO is the correct interface for any new backend. The tradeoffs:

| Factor | StableHLO | XLA HLO |
|--------|-----------|---------|
| Stability | 5yr backward, 2yr forward | Can change between XLA releases |
| Serialization | MLIR bytecode, versioned | Internal, no external guarantees |
| Ecosystem | JAX, PyTorch/XLA, TensorFlow, IREE | XLA-internal only |
| Optimization | Absorbing hardware-independent passes | Full XLA pass pipeline |
| Documentation | Formal spec, 107 ops documented | Internal codebase knowledge |

For an interpretation-based backend, StableHLO is strictly better than HLO. We parse the StableHLO module, walk ops, and dispatch to TT-NN -- exactly what applejax does with MLX.

For a compilation-based backend, Tenstorrent's tt-mlir already implements the full StableHLO -> TTIR -> TTNN -> Flatbuffer pipeline. There is no reason to touch HLO directly.

### StableHLO and Cross-Framework Portability

A significant benefit: StableHLO is produced by JAX, PyTorch/XLA, and TensorFlow, and consumed by XLA, IREE, and third-party compilers. If we build a StableHLO-consuming backend, we automatically support models from multiple frameworks.

A recent research paper (April 2026, arxiv.org/html/2604.12090) demonstrates using StableHLO for cross-architecture performance modeling of distributed ML workloads, validating its role as the universal intermediate representation.

---

## 3. JAX Pallas

### What Pallas Is

Pallas is JAX's kernel authoring language. It lets you write custom kernels in Python using JAX primitives, with explicit control over memory hierarchy, parallelism, and tiling. Kernels are defined with:

- **Grid**: iteration space (e.g., `(4, 5)` = 20 iterations)
- **BlockSpec**: defines how to slice input/output arrays into blocks, with an `index_map` function mapping grid indices to block indices
- **Kernel body**: JAX operations executed per grid point on block-sized data

### Backend Lowering

Pallas lowers kernels to different representations per backend:
- **TPU**: Mosaic (Google's internal TPU compiler)
- **GPU**: Triton IR (OpenAI's GPU kernel compiler) or Mosaic GPU (newer, Google-developed)

The backend is selected via `compiler_params` (e.g., `compiler_params=pltriton.CompilerParams()`).

### Could Pallas Map to Tenstorrent's Metalium Kernels?

This is the most speculative but potentially highest-impact question.

**Tenstorrent's kernel model** (Metalium/TT-Metal):
- Each Tensix core has 5 RISC-V CPUs, FPU/SFPU compute units, 1.5MB SRAM
- Three kernel types per core: **reader** (data input), **compute** (FPU/SFPU ops), **writer** (data output)
- Kernels coordinate through circular buffers in SRAM
- Data movement via Network-on-Chip (NoC)
- Explicit tiling: data is processed in tiles (typically 32x32)

**Pallas's model**:
- Grid of iterations with blocked data access
- Explicit memory hierarchy (scratch buffers)
- Tiled computation on blocks of data

**The mapping is not clean, but not impossible:**

| Pallas Concept | Metalium Equivalent | Difficulty |
|---------------|---------------------|------------|
| Grid | Multi-core dispatch (one grid point per Tensix core) | Medium -- grid maps to core grid |
| BlockSpec | Tile specification (32x32 tiles in L1 SRAM) | Medium -- block_shape maps to tile dims |
| Kernel body | Compute kernel (FPU/SFPU operations) | Hard -- Pallas uses JAX ops, Metalium uses C++ |
| Scratch memory | L1 SRAM circular buffers | Hard -- different coordination model |
| Data movement | Reader/writer kernels + NoC | Very hard -- Pallas abstracts this away |

**The fundamental mismatch**: Pallas assumes the runtime handles data movement (like a GPU's memory hierarchy). Metalium requires explicit reader/writer kernels that program NoC data movement. A Pallas-to-Metalium compiler would need to synthesize reader/writer kernels from Pallas's implicit data movement patterns.

**Verdict**: A full Pallas backend for Tenstorrent would be a major compiler project (months of work). However, there is a lighter path: use Pallas's grid/BlockSpec abstraction as a Python-level description language for Metalium kernels, with manual lowering. This would give users a familiar API while generating the three-kernel-per-core structure that Metalium requires.

This is a "Phase 3" idea -- valuable only after a working PJRT plugin exists.

---

## 4. Shardy / GSPMD

### The Transition

Shardy replaced GSPMD as JAX's default partitioner in **JAX 0.7.1**. As of **March 2026**, Shardy is the only partitioner in JAX -- the ability to fall back to GSPMD has been removed. Shardy is an MLIR-based tensor partitioning system that emerged from the collaboration of both the GSPMD and PartIR teams.

Key properties:
- Automatically parallelizes programs across multiple devices/chips
- Inserts communication ops (AllGather, ReduceScatter, AllReduce) as needed
- Users annotate sharding via `jax.sharding.NamedSharding` and mesh definitions
- Equivalent or better performance than GSPMD across Google workloads

### Relevance for Multi-Chip Tenstorrent Systems

**Galaxy systems** are Tenstorrent's multi-chip configurations (e.g., N300 with 2 Wormhole chips, larger Galaxy racks). Shardy is directly relevant for these:

- **tt-xla already had to deal with this.** Issue #1481 in tt-xla tracks the impact of Shardy becoming default in JAX 0.7.1. Their multi-chip tests were originally designed for both GSPMD and Shardy.
- **tt-xla does not yet support Galaxy boards.** The documentation explicitly states "only Tenstorrent nebula boards are supported and galaxy boards are not yet supported."
- **Our Blackhole setup is single-chip.** Shardy is irrelevant for our current hardware (one p150 card). It becomes relevant if/when we target multi-chip configurations.

### What Shardy Means for a Custom PJRT Plugin

If we build a custom PJRT plugin, Shardy integration is mostly free: Shardy operates at the StableHLO level before the program reaches the PJRT plugin. The plugin receives already-sharded StableHLO modules. The plugin only needs to handle device-to-device communication primitives (AllReduce, etc.) if targeting multi-chip.

For single-chip (our case), Shardy adds zero complexity to the plugin implementation.

---

## 5. JAX Export / Serialization

### jax.export and the .vjxp Format

JAX now has a mature export system:

```python
exported = jax.export.export(my_function)(input_spec)
jax.export.save_model(exported, "model.vjxp")
# Later, on any system:
loaded = jax.export.load_model("model.vjxp")
result = loaded.call(inputs)
```

Key properties:
- Exports to StableHLO MLIR bytecode in a `.vjxp` (Versioned JAX Program) file
- Includes calling convention, pytree structure, sharding metadata
- **6-month backward compatibility**: an exported artifact can be loaded by a JAX runtime up to 6 months newer
- **3-week forward compatibility**: can be loaded by slightly older runtimes
- Supports symbolic shapes for dynamic dimensions
- Higher-order gradients work on exported programs

### Can We Export JAX Models to Run on Tenstorrent?

Yes, via two paths:

**Path 1: jax.export -> StableHLO -> tt-mlir**
Export a JAX model to StableHLO, then feed it to tt-mlir's `compile_stablehlo_to_flatbuffer()`. This is essentially what tt-xla does, but decoupled from the PJRT runtime. We could do this without a PJRT plugin.

**Path 2: jax.export -> StableHLO -> IREE -> Tenstorrent**
IREE (Intermediate Representation Execution Environment) consumes StableHLO and can target various backends (CPU, Vulkan, CUDA, Metal). An IREE backend for Tenstorrent would be another compilation path. AMD submitted an IREE-based SDXL implementation to MLPerf in April 2025, showing IREE's viability for real workloads.

**Path 3: jax.export -> StableHLO -> our interpreter**
We could parse the exported StableHLO module and walk it with our TT-NN dispatch, similar to how applejax works. This avoids both PJRT plugin complexity and tt-mlir dependency.

### Practical Value

jax.export is most valuable as a **decoupling layer**. It lets users develop and validate models in standard JAX (on CPU/GPU), then export for Tenstorrent deployment. This is a cleaner user experience than requiring users to install our PJRT plugin during development.

---

## 6. MLX Comparison

### MLX's Success Story

Apple's MLX has become the definitive ML framework for Apple Silicon:

- **WWDC 2025**: Three dedicated sessions establishing MLX as preferred LLM inference framework
- **Ollama adoption** (March 2026): Ollama switched to MLX as its inference engine on Apple Silicon
- **Hardware co-design**: Apple's M5 chip includes Neural Accelerators specifically designed for MLX compute patterns, yielding 4.06x TTFT improvement for Qwen3-14B-4bit vs M4
- **Ecosystem**: MLX 0.31.x with C++, C, Python, and Swift APIs

### Key Design Principles and Lessons

**1. Unified Memory as a Feature, Not a Constraint**
MLX was designed around Apple Silicon's unified memory. Arrays live in shared memory; ops run on CPU or GPU without data copies. Tenstorrent's architecture has a different memory model (per-core 1.5MB SRAM + shared DRAM), but the lesson applies: **design the framework around the hardware's actual memory hierarchy, not around an abstraction that fights it**.

Our direct TT-NN approach already does this -- we explicitly manage L1 SRAM allocation, DRAM placement, and tile layouts. A JAX PJRT plugin would hide this behind PJRT's memory abstraction, potentially losing performance.

**2. Lazy Evaluation Enables Global Optimization**
MLX's lazy execution builds a computation graph before executing, enabling the scheduler to optimize across operations. TT-NN's trace capture achieves something similar -- it records a sequence of ops, then replays them without Python overhead. The lesson: **deferred execution is essential for accelerator performance**.

**3. Familiar APIs Lower Adoption Barriers**
MLX mirrors NumPy/PyTorch APIs. This is why people adopt it. For Tenstorrent, the equivalent would be ensuring JAX/PyTorch code runs with minimal modification -- which is exactly what a PJRT plugin provides.

**4. Hardware-Software Co-Design Wins**
Apple optimized M5 hardware for MLX patterns. Tenstorrent has the same opportunity: the Metalium ISA and NoC topology could be co-optimized with TT-NN op implementations. This is a long-term hardware architecture consideration, not something we can address in a CS440LX project, but it validates the importance of understanding the hardware deeply.

**5. The "Two-Tier" Approach**
Apple ships both MLX (for researchers/developers who want control) and Foundation Models framework (for app developers who want simplicity). Tenstorrent could similarly offer TT-NN (direct control, our current approach) alongside a JAX/PyTorch plugin (ecosystem integration).

---

## 7. Other Hardware Backend Evolution

### Google TPU

The most mature JAX backend, and the original motivation for JAX's existence:
- JAX was designed TPU-first; the CPU/GPU backends came later
- Pallas/Mosaic enables custom TPU kernels in Python
- Ironwood TPU (2025) demonstrates the co-designed AI stack: JAX -> XLA -> Pallas -> Mosaic -> hardware
- Key lesson: **owning the full stack (framework + compiler + hardware) enables unmatched optimization**

### Intel

- Maintains `intel-extension-for-openxla` -- a fork/extension of the XLA compiler
- Targets Data Center GPU Max Series and Arc B-Series GPUs
- Added SPMD/scale-up support via oneAPI Collective Communications Library (oneCCL)
- Key lesson: **the heavyweight XLA fork approach works but requires a dedicated compiler team** and massive CI infrastructure. Intel has both; we don't.

### AMD

- Ships `jax-rocm7-pjrt` and `jax-rocm7-plugin` packages
- Built nightly via GitHub Actions
- Also extends the XLA compiler with HIP/ROCm code generation
- Key lesson: **same as Intel -- big company approach, impractical for a small team**

### Community (applejax, jax-mps)

- Interpretation-based: walk StableHLO, dispatch to a library (MLX)
- applejax: 71 StableHLO ops, 12 CHLO ops, 2000+ tests, 30-min MLIR build
- jax-mps: PJRT plugin using MLX, ~3.7x CPU speedup for ResNet18
- Key lesson: **a small team CAN build a working PJRT plugin via interpretation**. The op count is manageable (~71 ops for good coverage). The MLIR build dependency is the main friction.

### NVIDIA multimesh-jax

- Novel approach: custom PJRT plugin enabling MPMD (Multiple Program, Multiple Data) workflows
- Enables pipeline parallelism across GPU submeshes within a single `jax.jit`
- Currently requires patches to JAX/jaxlib (not upstream)
- Key lesson: **PJRT is flexible enough to support novel execution models**, not just standard single-device dispatch

### Tenstorrent Official (tt-xla)

- 2,251 commits, daily dev releases, installable via pip
- Pipeline: StableHLO -> tt-mlir -> TT-NN -> Metalium
- Supports JAX (PJRT) and PyTorch (torch-xla/torch.compile)
- **863+ open issues** indicate active but immature development
- Recent work: vLLM integration (continuous batching, sparse attention for DeepSeek v3.2), torch-xla 2.10 compatibility
- **Only supports Nebula boards; Galaxy (multi-chip) not yet supported**
- Blackhole support: actively expanding (MoE BH support issue opened April 2026)

---

## 8. Recommendation

### Where We Are

We have a working direct-TT-NN inference engine: 28 ops, trace capture, 4 validated models (Qwen 0.5B/1.5B, Llama 3B/8B), 132 tok/sec decode, full correctness validation. This is **good enough for research and demos** but lacks JAX ecosystem integration (no `jax.jit`, no `vmap`, no `grad`, no Flax/Optax).

### What's Changed Since We Started

1. **tt-xla ships daily dev releases and is pip-installable.** This was not true when we started. It may already solve our problem for production use.
2. **Shardy replaced GSPMD.** Multi-chip sharding is now default in JAX.
3. **StableHLO is consolidating.** MHLO deprecation is underway. StableHLO is the one interface to target.
4. **applejax matured to v0.9.7.** A proven template for small-team PJRT plugins via interpretation.
5. **jax.export produces portable .vjxp artifacts.** Decouples model development from deployment hardware.
6. **MLX became the Apple Silicon standard.** Validates the hardware-specific framework approach.

### Ranked Recommendations

**1. Test the official tt-xla plugin (1 day, highest information value)**

Before building anything, test whether `pip install pjrt-plugin-tt` works on our Blackhole machine. If it does, the entire PJRT discussion becomes moot for practical purposes -- we would use the official plugin for JAX integration and keep our direct TT-NN engine for research and custom optimization.

```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
python -c "import jax; print(jax.devices())"
```

**2. Experiment with jax.export -> StableHLO -> manual interpretation (2-3 days)**

Use `jax.export.export()` to produce StableHLO from a simple model, then parse and dispatch to TT-NN. This tests whether we can bridge JAX's export system to our existing op handlers without a full PJRT plugin. It is the lightest-weight integration path.

**3. Build a minimal PJRT plugin (3-5 weeks, if #1 fails and we want full JAX integration)**

Fork applejax's build infrastructure, replace MLX dispatch with TT-NN. Start with 5 ops (constant, dot_general, add, broadcast_in_dim, convert). This gives us `jax.jit`, `vmap`, `grad`, and ecosystem compatibility. The MLIR build dependency is the main risk.

**4. Pallas-to-Metalium kernel bridge (speculative, Phase 3)**

Only worth pursuing after a working PJRT plugin exists. Use Pallas's grid/BlockSpec abstraction as a description language for Metalium kernels. This would be a unique contribution to the ecosystem -- no one has mapped Pallas to a non-GPU, non-TPU architecture.

**5. Shardy/multi-chip support (not relevant for current hardware)**

Only relevant when targeting Galaxy or multi-chip configurations. Our single p150 card does not benefit from Shardy. File this for future reference.

### The Big Picture

The JAX/XLA ecosystem is converging on StableHLO as the universal interface. Every path forward -- whether official tt-xla, custom PJRT, jax.export, or even IREE -- goes through StableHLO. Our existing 28 TT-NN op handlers map naturally to StableHLO ops. The question is not *what* to target (StableHLO) but *how* to integrate (PJRT plugin vs. export pipeline vs. official tt-xla).

The highest-leverage move is testing the official tt-xla plugin. If it works on Blackhole with our models, we skip months of engineering. If it doesn't, we know exactly what gap to fill and can choose between the applejax-style interpretation approach (feasible for a small team) or the export-and-interpret approach (even lighter weight).

---

## Sources

- [PJRT Plugin Integration Guide](https://openxla.org/xla/pjrt/pjrt_integration)
- [PJRT C++ API Overview](https://openxla.org/xla/pjrt/cpp_api_overview)
- [PJRT Plugin Blog Post (2024)](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html)
- [StableHLO Project](https://openxla.org/stablehlo)
- [StableHLO Roadmap](https://openxla.org/stablehlo/roadmap)
- [StableHLO Spec](https://openxla.org/stablehlo/spec)
- [StableHLO Releases](https://github.com/openxla/stablehlo/releases)
- [Cross-Architecture Performance Modeling via StableHLO (arxiv, April 2026)](https://arxiv.org/html/2604.12090)
- [JAX Pallas Documentation](https://docs.jax.dev/en/latest/pallas/index.html)
- [Pallas Design](https://docs.jax.dev/en/latest/pallas/design/design.html)
- [Pallas Grid and BlockSpec](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)
- [Pallas GPU Reference (Mosaic GPU)](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)
- [Shardy Guide for JAX Users](https://openxla.org/shardy/getting_started_jax)
- [Shardy JAX Migration](https://docs.jax.dev/en/latest/shardy_jax_migration.html)
- [Shardy GitHub](https://github.com/openxla/shardy)
- [Default Shardy in JAX 0.7.1 (tt-xla Issue #1481)](https://github.com/tenstorrent/tt-xla/issues/1481)
- [JAX Export Documentation](https://docs.jax.dev/en/latest/export/export.html)
- [jax.export Module](https://docs.jax.dev/en/latest/jax.export.html)
- [Exporting StableHLO from JAX Tutorial](https://openxla.org/stablehlo/tutorials/jax-export)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX and M5 Neural Accelerators (Apple Research)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [MLX at WWDC 2025](https://developer.apple.com/videos/play/wwdc2025/315/)
- [Ollama Powered by MLX](https://ollama.com/blog/mlx)
- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [applejax GitHub](https://github.com/danielpcox/applejax)
- [jax-mps GitHub](https://github.com/tillahoffmann/jax-mps)
- [Intel Extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla)
- [NVIDIA multimesh-jax](https://github.com/nv-legate/multimesh-jax)
- [tt-xla GitHub](https://github.com/tenstorrent/tt-xla)
- [tt-xla Getting Started](https://docs.tenstorrent.com/tt-xla/getting_started.html)
- [tt-xla Issues](https://github.com/tenstorrent/tt-xla/issues)
- [tt-mlir GitHub](https://github.com/tenstorrent/tt-mlir)
- [tt-forge GitHub](https://github.com/tenstorrent/tt-forge)
- [TT-Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metalium Documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html)
- [IREE Project](https://iree.dev/)
- [IREE GitHub](https://github.com/iree-org/iree)
- [vLLM TPU Backend (JAX + PyTorch)](https://blog.vllm.ai/2025/10/16/vllm-tpu.html)
- [Asynchronous PjRT for JAX on GPU (JAX/OpenXLA DevLab Fall 2025)](https://www.youtube.com/watch?v=zoS4EOqFmew)
