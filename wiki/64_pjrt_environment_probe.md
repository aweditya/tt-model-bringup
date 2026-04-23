# Wiki 64: PJRT Plugin Environment Probe

## Q: What's the development environment on our Blackhole host?

| Component | Version/Status |
|-----------|---------------|
| OS | Ubuntu 22.04.5 LTS |
| GCC | 11.4 (C++17 OK) |
| Clang | 17.0.6 |
| CMake | 4.2.3 |
| Python | 3.10.12 |
| JAX | 0.6.2 |
| jaxlib | 0.6.2 |
| ttnn | 0.68.0 (pip-installed) |
| Free disk | **24 GB only** (94% full) |

## Q: Can we compile C++ against the ttnn API?

**No, not currently.** The ttnn pip package doesn't include the main C++ API headers (`ttnn.hpp`). The official tt-xla plugin bundles 3096 headers in `jax_plugins/pjrt_plugin_tt/tt-mlir/install/tt-metal/`, but they have deep dependency chains (`tt_stl/span.hpp`, `hostdevcommon/`, etc.) that require a full tt-metal source build to resolve.

## Q: Does jaxlib ship MLIR libraries we can use?

**Python bindings yes, C++ no.** jaxlib's `_mlir.so` is a 16KB Python extension stub. It provides `jaxlib.mlir.ir.Context()` and `jaxlib.mlir.dialects.stablehlo` (works!), but no C++ headers or linkable MLIR symbols.

## Q: What StableHLO does JAX generate for common ops?

**Simple add:** Just `stablehlo.add %arg0, %arg1 : tensor<2x3xf32>` — one op.

**Matmul:** `stablehlo.dot_general` with `contracting_dims = [1] x [0]` — one op.

**Softmax:** Decomposes to ~10 ops: `reduce(max)`, `subtract`, `exp`, `reduce(add)`, `divide`, `broadcast_in_dim`.

**SwiGLU MLP:** ~12 ops including `silu` as a separate function with `negate`, `exp`, `add`, `divide`, `multiply`.

The ops are clean and predictable. For a basic transformer: `constant`, `dot_general`, `add`, `subtract`, `multiply`, `divide`, `negate`, `exponential`, `broadcast_in_dim`, `reduce`, `reshape`, `transpose`, `convert` — about 13 core ops.

## Q: What's the official tt-xla plugin situation?

Installed at `jax_plugins/pjrt_plugin_tt/`. It:
- Registers as platform "tt" at priority 500
- Bundles its own `libtt_metal.so` + `libTTMLIRRuntime.so` + `pjrt_plugin_tt.so`
- **Segfaults** on initialization at `SystemMesh::instance()` → `convert_1d_mesh_adjacency_to_row_major_vector` (2-device mesh topology bug)
- Auto-loads whenever JAX imports — **must be disabled** for our work

## Q: What are the constraints for building a custom PJRT plugin?

1. **Can't compile against ttnn C++ headers** — deep dependency chain
2. **Can't easily build MLIR from source** — 24GB disk probably insufficient
3. **CAN parse StableHLO from Python** — jaxlib MLIR bindings work
4. **CAN use ttnn from Python** — proven across 99 experiments
5. **Must disable official tt-xla** — it auto-loads and crashes
