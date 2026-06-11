# TT-ISA Documentation Repository

**Repo**: https://github.com/tenstorrent/tt-isa-documentation
**Status**: Living document, still being written (275 commits, 95 stars)
**Author**: Corsix (same person who wrote the blog series)
**License**: Apache 2.0 + CC BY-ND 4.0

This is the **comprehensive ISA reference manual** for Tenstorrent hardware. Written by the Corsix blog author as a follow-up to the 8-part blog series.

## Covered Architectures

### BlackholeA0 (our target!)
```
BlackholeA0/
├── README.md              — ASIC overview (140 Tensix tiles, 2D torus layout, p100/p150 configs)
├── TensixTile/
│   ├── README.md          — Tile overview (1536 KiB L1, 5 RISC-V cores, coprocessor)
│   ├── BabyRISCV/         — RV32IM core documentation
│   ├── TensixCoprocessor/ — Matrix, Vector, Unpack, Pack, Scalar unit docs
│   ├── DebugTimestamper.md
│   ├── PIC.md             — Interrupt controller
│   ├── SoftReset.md
│   └── TileControlDebugStatus.md
├── NoC/
│   ├── README.md          — NoC architecture
│   ├── Coordinates.md     — Coordinate system
│   ├── RoutingPaths.md    — Routing and congestion
│   ├── MemoryMap.md       — MMIO registers
│   ├── Atomics.md
│   ├── Counters.md
│   └── Interrupts.md
├── EthernetTile/
├── L2CPUTile/             — Big RISC-V (SiFive x280) cores
├── PCIExpressTile/
```

### WormholeB0
```
WormholeB0/
├── README.md
├── TensixTile/
├── ARCTile/               — Chip/board management (not in Blackhole)
├── DRAMTile/              — GDDR6 tile docs
├── EthernetTile/
├── NoC/
├── PCIExpressTile/
```

### Other
```
Diagrams/                  — Visual documentation
Miscellaneous/FMA/         — Fused Multiply-Add details
Glossary.md               — Terminology definitions
```

## Key Differences: Blackhole vs Wormhole

From what we can infer:
- Blackhole has L2CPUTile (Big RISC-V cores) — Wormhole does not
- Blackhole has no ARCTile or DRAMTile directories (different memory/management architecture)
- Blackhole: 140 Tensix tiles; Wormhole: up to 80

## Priority Reading Order for Our Project

1. `BlackholeA0/README.md` — overall chip layout
2. `BlackholeA0/TensixTile/README.md` — what's in a compute tile
3. `BlackholeA0/TensixTile/TensixCoprocessor/` — the compute units we need to target
4. `BlackholeA0/NoC/` — how data moves around the chip
5. `BlackholeA0/TensixTile/BabyRISCV/` — the control cores
6. `Glossary.md` — terminology

## Why This Matters

If we're building or understanding a JAX→Tenstorrent compilation pipeline, we need to understand:
- What operations the hardware can execute natively (ISA)
- How data is laid out and moved (NoC, memory map)
- What constraints the compiler must respect (tile sizes, data formats, synchronization)

This repo is the ground truth for all of that.
