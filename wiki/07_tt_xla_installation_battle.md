# TT-XLA Installation: What We Learned the Hard Way

## Q: Can you just `pip install` tt-xla and run JAX on Tenstorrent?

**A: Not yet on our system.** The pip-installable PJRT plugin crashes during device initialization. Here's the full story.

## What We Tried

### Environment
- Host: Ubuntu 22.04, Python 3.10
- Hardware: 2× Blackhole p150a cards (device 0 and 1)
- Driver: TT-KMD 2.6.0, firmware 19.6.0.0

### Attempt 1: pjrt-plugin-tt 0.3.0
```
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```
**Result**: Segfault in `convert_1d_mesh_adjacency_to_row_major_vector` during `populateDevices()`. The fabric/mesh code dereferences a null pointer.

### Attempt 2: pjrt-plugin-tt 0.2.0
**Result**: `TT_FATAL: topology_info.corners.size() == 2` — the fabric code expects a specific mesh topology that doesn't match our 2-device setup.

### Attempt 3: pjrt-plugin-tt 0.1.0
**Result**: `TT_FATAL: physical_chip_ids.size() == mesh_ew_size` — similar mesh topology mismatch.

### Environment variables tried (all failed)
- `ARCH_NAME=blackhole`
- `TT_METAL_MESH_SHAPE=1,1` and `2,1`
- `TT_METAL_SKIP_FABRIC=1`
- `TT_METAL_MESH_DEVICE_IDS=0`
- `TT_METAL_DEVICE_IDS=[0]`
- `JAX_PLATFORMS=tt`

### Why it fails

The PJRT plugin bundles its own copy of `libtt_metal.so` (inside `tt-mlir/install/lib/`). This bundled tt-metal version expects specific mesh/fabric topology configurations that don't match what our hardware + driver + firmware combination provides.

The crash happens in `tt::pjrt::ClientInstance::populateDevices()` which calls into `tt::tt_metal::distributed::MeshDevice::create()` which calls into the fabric control plane initialization — all before any user code runs.

### What would likely fix it

1. **Docker container** (`ghcr.io/tenstorrent/tt-xla-slim:latest`) — bundles a matching set of drivers + libraries + firmware. We couldn't try this because we lack passwordless sudo for Docker.
2. **Building from source** on Ubuntu 24.04 + Python 3.12 — the officially supported configuration.
3. **Matching firmware** — updating the device firmware to match what the PJRT plugin expects.
4. **Filing an issue** on tenstorrent/tt-xla about multi-device Blackhole initialization with the pip wheel.

## What this tells us about the TT-XLA ecosystem

1. **The software stack is tightly coupled**: driver version, firmware version, tt-metal version, and PJRT plugin version must all match. This is normal for accelerator software (similar to CUDA toolkit version matching).

2. **Multi-device setups are harder**: The fabric/mesh initialization code is relatively new and assumes specific topologies. Single-device setups may work fine.

3. **The recommended path is Docker or build-from-source**: The pip wheel is a convenience package that may not work on all system configurations.

4. **Active development**: 855 open issues and frequent releases indicate rapid iteration. The failure modes we hit are in the mesh/fabric code, which is one of the newer parts of the stack.

## What we CAN do

Despite tt-xla not working yet, we have:
- **Working TT-NN access** (direct Metalium/TT-NN Python API) — we already ran matmuls at 222 TFLOPS
- **Working JAX on CPU** — we can study the full compilation pipeline, inspect StableHLO/HLO, benchmark
- **The codegen path** — if we get tt-xla working later, it can generate standalone C++ that calls TT-NN directly

## Experiment

The failed attempts are documented in the experiment logs. Key commands:
```bash
ssh tenstorrent
source ~/tt-xla-env/bin/activate
export ARCH_NAME=blackhole
python3 -c "import jax; print(jax.devices())"  # crashes
```

## Sources
- tt-xla docs: https://docs.tenstorrent.com/tt-xla/getting_started.html
- tt-xla issues: https://github.com/tenstorrent/tt-xla/issues
- Experiment attempts (run 2026-04-21)
