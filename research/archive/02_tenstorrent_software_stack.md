# Tenstorrent Software Stack

## Stack Overview (Bottom to Top)

```
Hardware (Blackhole/Wormhole)
    ↓
TT-Metalium (low-level kernel programming, C++)
    ↓
TT-NN (neural network operator library, C++/Python, PyTorch-like API)
    ↓
TT-MLIR (MLIR-based compiler, StableHLO → Flatbuffer)
    ↓
TT-Forge (high-level compiler frontend, PyTorch/ONNX ingestion)
TT-XLA (PJRT plugin for JAX/PyTorch-XLA, outputs StableHLO to TT-MLIR)
```

## TT-Metalium (Low-Level)

OpenCL-like C++ API for direct hardware programming.

### Workflow
1. Open device, get command queue
2. Create buffers (DRAM or L1)
3. Write input data
4. Compile kernels (reader, compute, writer)
5. Allocate circular buffers for inter-kernel communication
6. Set runtime arguments
7. Execute and read results

### Key APIs
```cpp
IDevice* device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
auto buf = CreateBuffer({.device=device, .size=sz, .page_size=pg, .buffer_type=BufferType::DRAM});
EnqueueWriteBuffer(cq, buf, data, false);
```

### Fast Dispatch
- Sacrifices one RISC-V core per command queue for async operations
- Two command queues enable overlapping compute + data transfer
- Disable with `TT_METAL_SLOW_DISPATCH_MODE=1` for debugging

### Low-Level Kernels (LLKs)
- Hardware-generation-agnostic API
- Wormhole: 64-wide vector unit; Blackhole: 32-wide (transparent to programmer)
- Functions like `sin_tile(0)` dispatch to optimized implementations per generation

## TT-NN (Neural Network Library)

PyTorch-like Python/C++ API built on Metalium.

```python
import ttnn
device = ttnn.open_device(device_id=0)
x = ttnn.from_torch(a, dtype=ttnn.BFLOAT16, device=device, layout=ttnn.TILE)
y = ttnn.from_torch(b, dtype=ttnn.BFLOAT16, device=device, layout=ttnn.TILE)
z = ttnn.add(x, y)
result = ttnn.to_torch(z, dtype=torch.float32)
```

### Leaky Abstractions (Important!)
- **View ops on tiled dims are expensive**: transpose/permute on last 2 dims requires actual memory copy
- **Manual deallocation needed**: `tensor.deallocate()` to prevent L1 exhaustion
- **Tile layout mandatory**: 32×32 tiles, must pad tensors accordingly

### CCL (Collective Communications Library)
```python
reduced = ttnn.all_reduce(z)
broadcasted = ttnn.broadcast(z)
```

## TT-MLIR (Compiler)

MLIR-based compiler that takes StableHLO and produces optimized code for TT hardware.
- GitHub: https://github.com/tenstorrent/tt-mlir

## TT-Forge (High-Level Compiler Frontend)

MLIR-based compiler frontend for PyTorch/ONNX models.
- GitHub: https://github.com/tenstorrent/tt-forge

## Key GitHub Repositories

| Repo | Purpose | URL |
|------|---------|-----|
| tt-metal | TT-NN + TT-Metalium | https://github.com/tenstorrent/tt-metal |
| tt-mlir | MLIR compiler | https://github.com/tenstorrent/tt-mlir |
| tt-forge | High-level compiler | https://github.com/tenstorrent/tt-forge |
| tt-xla | PJRT plugin (JAX/PyTorch-XLA) | https://github.com/tenstorrent/tt-xla |
| tt-forge-onnx | ONNX graph compiler | https://github.com/tenstorrent/tt-forge-onnx |
| ttsim | Full-system simulator | https://github.com/tenstorrent/ttsim |
| polaris | High-level simulator | https://github.com/tenstorrent/polaris |
| riscv-ocelot | RISC-V OoO core (Berkeley BOOM fork) | https://github.com/tenstorrent/riscv-ocelot |
| tt-bh-linux | Blackhole Linux demo | https://github.com/tenstorrent/tt-bh-linux |

(111 total repos in the org — these are the most relevant)

## Key Contrast with GPU Programming

| Aspect | GPU (CUDA) | Tenstorrent (Metalium) |
|--------|-----------|----------------------|
| Parallelism | SIMT, thousands of threads | Single thread per RISC-V core, 5 cores per Tensix |
| Memory | Cache hierarchy, unified address space | Explicit DMA, NoC-based tuple addressing |
| Latency hiding | Warp switching | None — deterministic, predictable latency |
| Programming | Kernel = single function | Kernel = 3 cooperating programs (reader/compute/writer) |
| Data format | Any alignment | 32×32 tiles mandatory for compute |

Sources:
- clehaxze: https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi
- TT-Metalium guide: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- TT-NN README: https://github.com/tenstorrent/tt-metal/blob/main/ttnn/README.md
