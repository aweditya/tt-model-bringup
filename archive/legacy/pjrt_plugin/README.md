# PJRT Plugin for Tenstorrent Blackhole

Custom PJRT plugin that connects JAX to Tenstorrent Blackhole devices via ttnn.
Uses the interpretation approach: walks StableHLO ops and dispatches to ttnn at execution time.

## Status

- **Phase 1 (current)**: Skeleton. `jax.devices()` shows TT device.
- Phase 2: Buffer transfer (host <-> device)
- Phase 3: Compile + Execute (StableHLO interpretation)
- Phase 4: Full transformer op set
- Phase 5: Trace capture for fast replay

## Building

```bash
# On the remote Tenstorrent host:
ssh tenstorrent
cd tt-xla/pjrt_plugin

# 1. Fetch the exact PJRT header for your jaxlib version
./scripts/fetch_pjrt_header.sh

# 2. Build
export TT_METAL_HOME=/path/to/tt-metal
./scripts/build.sh

# 3. Test
python -m pytest tests/test_device_discovery.py -v
```

## Architecture

```
JAX program
  |
  v
jax.jit(f) -> StableHLO MLIR
  |
  v
PJRT Plugin (this code)
  |-- Compile: parse StableHLO module
  |-- Execute: walk ops, dispatch to ttnn
  |-- Buffer: host <-> device via ttnn tensors
  v
ttnn C++ API -> Tenstorrent Blackhole
```

## Directory Structure

```
pjrt_plugin/
  CMakeLists.txt              # Build system
  src/
    plugin.cc                 # PJRT_Api entry point + function table
    client.h/cc               # TT client (device lifecycle)
    buffer.h/cc               # Buffer management (stub in Phase 1)
    executable.h/cc           # StableHLO execution (stub in Phase 1)
    ops/
      arithmetic.h/cc         # add, subtract, multiply, divide
      matmul.h/cc             # dot_general
      elementwise.h/cc        # exp, log, tanh, etc.
  jax_plugins/tt/
    __init__.py               # Plugin registration
  tests/
    conftest.py               # Shared fixtures
    test_device_discovery.py  # Phase 1 test
    test_buffer.py            # Phase 2 tests
    test_basic_ops.py         # Phase 3 tests
    test_matmul.py            # Phase 3 tests
  scripts/
    fetch_pjrt_header.sh      # Download matching PJRT header
    build.sh                  # Build on remote host
  third_party/pjrt/
    pjrt_c_api.h              # Vendored PJRT C API header
```

## Design Decisions

See `research/pjrt_reflections.md` for detailed design rationale.

Key choices:
- **C API directly** (not C++ wrapper): avoids XLA codebase dependency
- **Interpretation** (not compilation): walk StableHLO, dispatch to ttnn
- **Single device** (device 0 only): simplicity for our P150 hardware
- **ttnn headers in .cc only**: keeps compile times fast, avoids leaking ttnn types
