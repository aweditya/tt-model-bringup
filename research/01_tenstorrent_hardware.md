# Tenstorrent Hardware Architecture

## Chip Generations

### Wormhole (Previous Gen)
- **Tile grid**: 10x12 grid of tiles
- **Tile types**: ARC (1), D/DRAM (18), E/Ethernet (16), PCIe (1), T/Tensix (80 theoretical, 64-72 usable after harvesting)
- **Products**: n150s (1 ASIC, 12GB GDDR6), n300s (2 ASICs, 24GB total)
- **Memory per Tensix**: 1.5MB SRAM (1464 KiB usable + specialized buffers)

### Blackhole (Current Gen, what we have)
- **Process**: 6nm
- **Tensix cores**: 140 total, 120 usable after harvesting
- **Big RISC-V cores**: 16 SiFive x280 cores (can run Linux!)
- **SRAM**: 180MB total (1.5MB per Tensix core)
- **DRAM**: 28GB (p100a) or 32GB (p150a/p150b) GDDR6
- **Memory bandwidth**: 448-512 GB/s
- **Peak compute**: 664 TFLOPS (BlockFP8), ~372 TFLOPS (FP16)
- **Interconnect**: 4x QSFP-DD 800G passive ports
- **PCIe**: Gen 5 x16
- **TDP**: Up to 300W
- **Supported precisions**: FP8, FP16, BF16, INT8, block floating-point; Big RISC-V cores support up to FP64

## Tensix Core Architecture (Common to Wormhole & Blackhole)

Each Tensix core contains:
- **5 Baby RISC-V cores** (RV32IM): 1 Brisc, 3 Trisc (T0/T1/T2), 1 NC (Network Controller)
- **Matrix engine (FPU)**: 8x16 @ 16x16 primitive; builds 32x32 from 16 primitives
- **Vector engine (SFPU)**: 32-wide SIMD, 8 vector registers, programmable shader-like
- **Pack/Unpack units**: Data format conversion between memory and compute
- **1.5MB L1 SRAM**: Directly addressable, no cache hierarchy
- **Two NoC routers**: For the 2D torus interconnect

### Data Flow Pipeline
```
DRAM → NoC0 → Unpack → SrcA/SrcB → Matrix/Vector → Dst → Pack → NoC1 → DRAM
```

### Kernel Structure (3 cooperating kernels per operation)
1. **Reader kernel** (data movement in): DMA from DRAM/other tiles to L1
2. **Compute kernel**: Drives Matrix/Vector units via Tensix instructions
3. **Writer kernel** (data movement out): DMA from L1 to DRAM/other tiles

Synchronized via circular buffers backed by hardware mutexes.

### Tensix Instruction Pipeline
- 8-bit opcode + 24-bit operands (distinct from RISC-V encoding)
- **Macro-Op Expander**: Single MOP instruction → sequence of Tensix ops
- **Replay Expander**: Record/playback instruction sequences
- **8 mutexes + 8 semaphores** for synchronization
- **8 execution resources**: Scalar (ThCon), Matrix (FPU), Unpack, Pack, Vector (SFPU), ThCfg, TDMA, Xmov

### Matrix Multiplication Fidelity Modes
Multi-pass approach using 7-bit × 5-bit multipliers:
- **LoFi only**: FP8, BFP2, BFP4 (1 pass, highest throughput)
- **LoFi + HiFi2**: BFP8 (2 passes)
- **LoFi + HiFi2 + HiFi3 + HiFi4**: BF16, FP16 (4 passes)

### Vector (SFPU) Unit
- 32 SIMD lanes, 8 vector registers (L0-L7)
- Per-lane conditional execution via flags
- LUT-based approximations for transcendental functions
- Cross-lane operations: rotation, shifting, 4x4 transpose
- Automatic data type conversion during load/store

## Memory Architecture

### Hierarchy
1. **Per-core registers**: Tensix scalar GPRs (64 per pipe × 3 pipes = 192 total > 160 RISC-V GPRs)
2. **L1 SRAM** (1.5MB/core): No cache, explicit management, holds kernels + circular buffers
3. **DRAM**: Accessed via NoC DMA, not load/store

### Addressing
- NoC tuple-based: `(x, y, addr)` identifies any memory location on chip
- No unified address space — everything is explicit DMA
- Lock-step allocation: all DRAM controllers allocate same-size blocks → single pointer works

### Data Layout Modes
- **Interleaved** (default): Round-robin across DRAM controllers by page
- **Sharded**: Data chunks mapped to specific core L1s for locality
- **Tile layout**: 32×32 element tiles (padded), matches hardware compute units

### Key Implications
- Manual deallocation required (L1 too scarce for GC)
- No multitenancy (L1 scarcity + address conflicts)
- Blackhole's 32GB exceeds 32-bit addressing → future 64-bit API needed

## Interconnect

### NoC (Network on Chip)
- Two unidirectional NoCs forming a 2D torus
- NoC #0: East + South bound
- NoC #1: West + North bound
- 32-byte channels per direction per tile
- Per-hop latency: ~9 clock cycles (~9ns at 1GHz)
- Edges wrap around (interleaved physical layout minimizes worst-case distance)

### Chip-to-Chip (Ethernet)
- Each E tile: 100Gb/s bidirectional
- Standard Ethernet (not proprietary)
- 6D addressing: (NoC X, NoC Y, Shelf X, Shelf Y, Rack X, Rack Y)
- Scales: n300 (2 chips) → QuietBox (8 chips) → Galaxy (32+ chips)

## Manufacturing: Harvesting
- Defective Tensix rows are disabled entirely
- Even defect-free chips disable rows for consistency
- Wormhole: 1-2 rows disabled; Blackhole: 140→120 usable cores

Sources:
- Corsix blog series (Parts 1-8): https://www.corsix.org/content/tt-wh-part1
- clehaxze: https://clehaxze.tw/gemlog/2025/04-21-programming-tensotrrent-processors.gmi
- clehaxze memory: https://clehaxze.tw/gemlog/2025/03-17-memory-on-tenstorrent.gmi
- Blackhole specs: https://docs.tenstorrent.com/aibs/blackhole/specifications.html
- Hot Chips 2024 presentation: https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf
- Corsix tt-isa-documentation repo (comprehensive ISA reference)
