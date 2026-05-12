# Phase A6 + A7 Results

## A6 v1 — Chunked-serial DeltaNet prefill scan

`experiments/85_deltanet_scan_v1.py` on qb1.

| T | cosine(H_final) | cosine(out[T-1]) | total time | µs/tok | prefill tok/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.999812 | 0.999825 | 219 ms | 3422 | 292 |
| 256 | 0.999691 | 0.999718 | 321 ms | 1253 | 798 |
| 1024 | 0.999715 | 0.999687 | 1287 ms | 1257 | **796** |

**PASS** — cosine well above 0.99 gate at all lengths. Steady-state ~1.25 ms/token = **~800 tok/s prefill** on one chip.

For a 4096-token user prompt (e.g., paste a code repo): ~5 seconds prefill. Tolerable for a chat interface, not great.

Memory ceiling: state I/O of 4 MB at 450 GB/s = ~9 µs/tok floor on 1 chip. We're at 1250 µs = ~0.7% of ceiling. Same dispatch-bound territory as A3. Full Blelloch v2 with on-device parallel scan would push toward this floor.

## A7 — Multi-chip primitives (PARTIAL)

`experiments/87_multichip_primitives.py` + 2 follow-up probes.

### Mesh opening — WORKS

```python
mesh = ttnn.distributed.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 1))
# → MeshDevice(2x1 grid, 2 devices)
```

### Tensor placement on mesh — WORKS

`ttnn.from_torch(t, device=mesh, ...)` happily uploads to both chips.

### Collectives — BLOCKED on fabric init

All three of `all_gather`, `all_reduce`, `reduce_scatter` fail with:
```
TT_FATAL @ tt_metal/fabric/control_plane.cpp: fabric_context_ != nullptr
"Trying to get un-initialized fabric context"
```

Discovery turned up `ttnn.set_fabric_config(config, ...)` with these enum values:
- `FabricConfig.FABRIC_1D`
- `FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE`
- `FabricConfig.FABRIC_1D_RING`
- `FabricConfig.FABRIC_2D`
- `FabricConfig.FABRIC_2D_TORUS_X / Y / XY`
- `FabricConfig.CUSTOM`

I tried all of them. `CUSTOM` advances past the "uninit" error but collectives still fail. The non-CUSTOM values all hit `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp:220: tt::exception` — Blackhole firmware refuses to load the fabric config.

### Hypotheses for why fabric init fails on qb1

1. **Firmware mismatch**: Phase 5 reflections noted FW 19.6.0 on qb1 vs the latest fully-tested 19.5.0. Multi-chip fabric may not work on the newer FW with this ttnn release.
2. **Hardware topology**: qb1's 4 P150s may not be wired with the inter-chip fabric links the configs expect. The QB chassis has PCIe but possibly no fabric mesh.
3. **Missing init step**: The tt-metal `conftest.py` shows a `set_fabric(...)` wrapper that takes more args than just `config`. Maybe we need a specific `reliability_mode` or `manager_mode` for Blackhole.

### Confirmed for Phase B planning

- `ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` exist (Phase A7 plan flagged these). They're the MoE-specific collectives. **Probably also blocked on the same fabric init** until we sort it out.
- We CAN open a multi-chip mesh and put weights on it. So we have *memory parallelism* (the model can fit) but not *compute parallelism* (no all-reduce yet).

### What this means for Branch III

Phase B can't be full 2-chip TP from day 1. Options:

- **B-α: skip fabric for now**. Run Qwen3.6-35B-A3B on one chip with aggressive bf8 + 1K context. The math is tight (37 GB at 4K) but ~30 GB at 1K context might just fit. Realistic.
- **B-β: investigate fabric more**. Read `~/tenstorrent/tt-metal` source for Blackhole-specific fabric setup. May find the magic incantation; may discover the hardware doesn't support fabric.
- **B-γ: smaller model first**. Run a smaller Qwen3.6-class model (do they ship a 7B variant?) on one chip to validate the arch end-to-end, then revisit fabric.

I'm going with **B-α**: implement single-chip B with aggressive memory budgeting, document fabric as a known follow-up. Multi-chip TP becomes a Phase C work item rather than Phase B prerequisite.

## Updated Phase B plan

Phase B (revised):
- B1: Weight loading + bf8 quant + cur-chip placement (single chip first; mesh-aware code so we can extend later)
- B2-B6: Single chip integration + correctness + perf
- B7 (new): Investigate fabric for multi-chip when we hit the memory wall
