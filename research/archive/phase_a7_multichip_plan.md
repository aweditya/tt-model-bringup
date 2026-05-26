# Phase A7 — Multi-Chip Primitives Plan

## Why we need this

Per Branch III memory math: Qwen3.6-35B-A3B in bf8 is ~37 GB at 4K context, exceeds one P150's ~30 GB usable DRAM. Phase B targets 2-chip TP. A7 validates the multi-chip primitives in isolation before integration.

## What ttnn 0.69 actually exposes (from API docs)

```
Mesh / sharding:
  ttnn.MeshDevice              — multi-chip device handle (single-device opens as 1×1 mesh)
  ttnn.mesh_partition          — partition tensor across mesh devices along a dim
  ttnn.create_sharded_memory_config  — distributed-tensor memory config
  ttnn.interleaved_to_sharded / sharded_to_interleaved  — layout conversion
  ttnn.reshard                 — between sharded layouts

Collectives:
  ttnn.all_gather              — gather tensors from all devices along a dim
  ttnn.all_reduce              — sum across all devices (other ops not visible in docs)
  ttnn.all_broadcast           — broadcast from one device to all
  ttnn.broadcast               — single-sender broadcast
  ttnn.reduce_scatter          — reduce-then-scatter across dim
  ttnn.reduce_to_root          — tree reduction with output on root only
  ttnn.point_to_point          — direct two-device send/receive

MoE-specific (this is HUGE):
  ttnn.all_to_all_dispatch     — dispatch tokens to devices owning selected experts
  ttnn.all_to_all_combine      — gather per-expert outputs back to source tokens
```

The MoE-specific `all_to_all_dispatch` and `all_to_all_combine` are exactly what expert parallelism needs. We don't have to write the routing manually — ttnn provides it. **This significantly de-risks Phase C** (Qwen3-Coder-Next across 4 chips).

## Exact mesh-device opening API — TBD

Docs are truncated on the precise multi-chip open call. Need to introspect on qb1:
- `ttnn.distributed.open_mesh_device(mesh_shape=(2, 1), device_ids=[0, 1])` (likely)
- Or `ttnn.MeshDevice(shape=(2, 1), device_ids=[0, 1])`
- Or `ttnn.open_device(device_id=...)` with a list

We confirmed from Phase 4 / Phase 5 work that single-chip `ttnn.open_device(device_id=0)` returns `MeshDevice(1×1 grid, 1 devices)` — the mesh abstraction is the canonical handle.

## What A7 validates (in isolation)

| Sub-step | Test | Pass criterion |
|---|---|---|
| A7.1 | Open 2-chip mesh, allocate test tensor on each | mesh shape correct, both chips report distinct DRAM addresses |
| A7.2 | `all_gather` of [B=1, H=2048] across 2 chips | gathered tensor shape correct, math correct |
| A7.3 | `all_reduce` (sum) across 2 chips | result on each chip matches numpy sum |
| A7.4 | `reduce_scatter` for TP linear: each chip computes partial matmul, reduce-scatter | correctness check vs single-chip matmul |
| A7.5 | `all_to_all_dispatch` MoE pattern | tokens correctly routed to their expert's chip |
| A7.6 | Round-trip latency of each primitive | timing data for Phase B/C planning |

## File layout

```
experiments/87_multichip_open.py        — A7.1
experiments/87b_multichip_collectives.py — A7.2/3/4
experiments/87c_multichip_moe.py         — A7.5
```

Or one file `experiments/87_multichip_primitives.py` with sub-functions per sub-step. Probably the latter.

## Performance ceiling notes for multi-chip

Each Blackhole-to-Blackhole link bandwidth: TBD (need to measure or look up). Probably ~50-100 GB/s on PCIe / on-chip link. Compared to DRAM 450 GB/s, **inter-chip is the new bottleneck for any all-reduce-heavy workload**.

For TP linear layers: each layer needs an all-reduce of the partial outputs. If the all-reduce is bandwidth-bound at 50 GB/s for an [B, hidden]=[1, 2048] tensor (4 KB at bf16), latency is ~80 ns — negligible vs the matmul time. So small per-step transfers are fine.

For MoE all-to-all-dispatch: tokens (small) move; weights (large) stay home. Bandwidth doesn't dominate.

The real concern is **synchronization overhead** — every collective is a barrier. 40 layers × 2 collectives each (attn TP all-reduce + MoE all-to-all) = 80 barriers per token. If each barrier costs 10 µs, that's 0.8 ms/tok overhead.

## What I'm NOT doing in A7

- Implementing the model — that's Phase B
- Running the existing Qwen1.5-MoE on 2 chips for comparison (would be nice but A0 already covered MoE perf)
- 4-chip primitives — same APIs but bigger mesh. Test 2-chip first.

## Status

⏸ Ready to write when qb1 stabilizes. Plan is concrete; A7 is mostly "validate the primitives exist and work as documented, capture timing."
