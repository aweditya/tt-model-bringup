# JAX, XLA, and PJRT Backend Architecture

## The Big Picture

```
JAX Python code
    ↓ jax.jit()
Jaxpr (JAX's internal IR)
    ↓
StableHLO (standard ML IR)
    ↓ PJRT Plugin Interface
Hardware-specific compiler + runtime
    ↓
Device execution
```

## PJRT (Portable JAX Runtime)

PJRT is the standard interface for connecting ML frameworks (JAX, PyTorch/XLA) to hardware backends. It is:
- Hardware-independent
- Framework-independent
- Not coupled to XLA's compiler (can use proprietary compilers)

### Core C++ Classes

1. **PjRtClient**: Central coordinator. Manages devices, memory spaces, buffer creation, compilation, execution.
2. **PjRtDevice**: Represents a single hardware device. Has metadata, memory spaces.
3. **PjRtMemorySpace**: Unpinned (flexible location) or pinned (bound to device). Tracks resident buffers.
4. **PjRtBuffer**: On-device data. Handles host↔device transfers with configurable semantics (zero-copy, etc.).
5. **PjRtCompiler**: Optional XLA-specific compilation utilities.
6. **PjRtExecutable**: Serializable compiled artifact. PjRtLoadedExecutable is ready-to-run with Execute(), ExecuteSharded(), ExecutePortable().

### Implementation Options

**Option A: Direct C API**
Implement `PJRT_Api` struct with function pointers directly.

**Option B: C++ API + C Wrapper (recommended if building against XLA repo)**
Inherit from C++ PJRT base classes, use provided C→C++ wrapper.

### Minimum Required Implementations
- `PJRT_Client_Create` — creates client instance
- `GetPjrtApi()` — returns function pointer table
- Optional: `PJRT_TopologyDescription_Create`, `PJRT_Plugin_Initialize`, `PJRT_Plugin_Attributes`

### JAX Plugin Registration

```python
# jax_plugins/my_plugin/__init__.py
import os
import jax._src.xla_bridge as xb

def initialize():
    path = os.path.join(os.path.dirname(__file__), 'my_plugin.so')
    xb.register_plugin('my_plugin', priority=500, library_path=path, options=None)
```

### Plugin Discovery (two mechanisms)
1. **Namespace packages**: `jax_plugins/my_plugin/` directory
2. **Package metadata**: entry-points in pyproject.toml under `jax_plugins` group

### Backend Selection
```python
jax.config.update("jax_platforms", "my_plugin")
# or
# JAX_PLATFORMS=my_plugin environment variable
```

### Execution Flow
```
Load plugin → Create client → Attach memory spaces → Transfer host data to buffers
→ Compile StableHLO modules → Execute with buffer arguments → Extract results via futures → Cleanup
```

## Existing Third-Party PJRT Backends

| Backend | Hardware | Notes |
|---------|----------|-------|
| CUDA | NVIDIA GPUs | Reference implementation, part of XLA |
| Intel GPU (SYCL) | Intel GPUs | Via intel-extension-for-tensorflow |
| Apple Metal | Apple Silicon | jax-metal package |
| **Tenstorrent** | Blackhole/Wormhole | tt-xla, uses TT-MLIR |

## TT-XLA: Tenstorrent's PJRT Implementation

**THIS IS DIRECTLY RELEVANT — Tenstorrent already has a PJRT plugin!**

### Architecture
```
JAX/PyTorch → PJRT Plugin (pjrt_plugin_tt.so) → StableHLO → TT-MLIR → TT-Metal → Hardware
```

### Components
- `pjrt_implementation/` — Core PJRT device (C++)
- `python_package/` — Python bindings
  - `jax_plugin_tt` — JAX wrapper
  - `torch_plugin_tt` — PyTorch/XLA wrapper
- Fork of iree-pjrt as starting point

### Installation
```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
tt-forge-install
```

### Build from Source
- Requires: Ubuntu 24.04, Python 3.12, Clang 20, CMake
- Dependencies: TT-MLIR toolchain (built separately)
- Build: cmake + ninja

### Key Stats
- 2,233 commits, 853 open issues, Apache-2.0 license
- Python 83.9%, C++ 13.8%
- Active development with bounty program

## XLA Compilation Pipeline

```
StableHLO (input)
    ↓
HLO (XLA's High-Level Operations)
    ↓ Optimization passes
Optimized HLO
    ↓ Backend-specific lowering
Target IR (LLVM IR for CPU/GPU, or custom)
    ↓
Machine code / device binary
```

## Key Question for Our Project

Since Tenstorrent already has tt-xla, our project could focus on:
1. **Understanding** how tt-xla works end-to-end (the compilation pipeline)
2. **Contributing** to tt-xla (853 open issues!)
3. **Building something complementary** (e.g., a simpler/educational PJRT plugin)
4. **Benchmarking and analyzing** the existing implementation
5. **Extending** tt-xla with new features or optimizations

Sources:
- PJRT integration guide: https://openxla.org/xla/pjrt/pjrt_integration
- PJRT C++ API: https://openxla.org/xla/pjrt/cpp_api_overview
- tt-xla repo: https://github.com/tenstorrent/tt-xla
- tt-xla docs: https://docs.tenstorrent.com/tt-xla/getting_started.html
- Intel PJRT blog: https://opensource.googleblog.com/2023/06/accelerate-jax-models-on-intel-gpus-via-pjrt.html
- PJRT overview blog: https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html
