# Fabric Diagnosis — qb1 is Single-Chip-Only

Definitive finding from `tt-smi -ls`:

```
ETH core heartbeat check failed on device ASIC ID: 143238000415488, ETH core (1, 1, ETH, NOC0)
ETH core heartbeat check failed on device ASIC ID: 143238000415488, ETH core (16, 1, ETH, NOC0)
ETH core heartbeat check failed on device ASIC ID: 143238000415488, ETH core (2, 1, ETH, NOC0)
ETH core heartbeat check failed on device ASIC ID: 143238000415488, ETH core (15, 1, ETH, NOC0)
... same on all 4 chips (ASIC IDs 143238000415488 / 417280 / 418592 / 418816) ...
```

## What this means

Every ETH core on every chip times out waiting for its peer. The heartbeat doesn't advance because **no remote chip responds**. The 4 P150s in qb1's chassis are:

- ✅ Detected via PCIe (4 separate `/dev/tenstorrent/N`)
- ✅ Individually functional (single-chip workloads work)
- ❌ **NOT physically wired together** via the inter-chip ethernet

This is consistent with qb1 being a standard 4×PCIe-card server, not a Quietbox / Galaxy / T3K with internal fabric cabling. Each chip is an island.

## Implications

| Capability | Status |
|---|---|
| Single-chip workloads | ✅ Full support |
| `ttnn.MeshDevice` software mesh | ✅ Opens fine |
| `ttnn.all_gather` / `all_reduce` / `reduce_scatter` | ❌ Requires fabric, broken |
| `ttnn.all_to_all_dispatch` (MoE) | ❌ Same — broken |
| Multi-chip TP via fabric | ❌ Impossible |
| Multi-chip TP via host RAM shuttle | ⚠️ Possible but ~5-10× slower |
| 4 independent workloads, one per chip | ✅ Works (4 separate `ttnn.open_device`) |

## Branch III pivot

The original Phase B plan called for 2-chip TP because Qwen3.6-35B-A3B is 34.7 GB bf8 and doesn't fit one chip's ~30 GB DRAM. Multi-chip is blocked here, so we pivot:

**New target: Qwen3.6-27B (dense, ~27 GB bf8)** — fits one chip with headroom for KV/scratch.

| | Qwen3.6-35B-A3B | Qwen3.6-27B |
|---|---|---|
| Params | 35B (3B active) | 27B (all active) |
| Architecture | DeltaNet + Attn + MoE | Standard transformer |
| Released | 2026-04-24 | 2026-04-24 (same week) |
| Single-chip bf8 fit | ❌ 34.7 GB | ✅ ~27 GB |
| Throughput target | 15-25 tok/s on 2 chips | ~10 tok/s on 1 chip (no MoE acceleration) |
| Architecturally novel | Yes (DeltaNet) | No (dense GQA) |

We **keep** all of Phase A's work — DeltaNet, Gated Attention, MoE block validation, parallel scan v1, multi-chip primitives probe. They're still relevant for the day someone wires the fabric. For now we put them on the shelf.

We **start** a new Phase B' on Qwen3.6-27B, which is a much simpler bringup: standard transformer (GQA 32/8 head_dim 128 per typical Qwen3 shape), no DeltaNet, no MoE expert-parallel. Probably 4-6 hours of work given everything Phase A taught us.

## What could change this verdict

1. **Cabling** — if Tenstorrent or whoever owns the box can run inter-card ETH cables, fabric becomes live and we go back to the 35B plan.
2. **Better firmware** — qb1 has FW 19.6.0 vs 19.5.0 tested. Unlikely but possible that a future ttnn build configures fabric over PCIe somehow. We don't see evidence of this in the current API.
3. **Different host** — if there's a real Quietbox or Galaxy accessible later, multi-chip work resumes.

## Status

This file documents the constraint. Branch III pivots to Qwen3.6-27B in a fresh Phase B'. Original Phase A artifacts remain valid and committed.
