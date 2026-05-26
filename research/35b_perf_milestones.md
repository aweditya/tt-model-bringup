# 35B-A3B perf milestones — single source of truth

Per-token decode latency on Qwen3.6-35B-A3B, qb1 (1,4) P150 mesh. Every number
sync-bounded (`ttnn.synchronize_device` before and after), 10+ steady-state
iterations, prefill+decode bit-identical to topk baseline.

| Date | Mode | ms/tok | tok/s | Speedup | Commit |
|---|---|---|---|---|---|
| 2026-05-24 | topk eager (baseline) | 480 | 2.08 | 1.0× | `fd4367f` |
| 2026-05-24 | looped Pattern A traced | 308 | 3.24 | 1.56× | `4cac36a` |
| 2026-05-25 | batched Pattern A eager | 267 | 3.74 | 1.80× | `961ce7f` |
| 2026-05-25 | **batched Pattern A TRACED** | **146** | **6.85** | **3.29×** | (this row) |
| — | 27B production (reference) | 77 | 12.93 | 6.23× target | — |

## Where the wins came from

1. **Correctness foundation** (`fd4367f`): q_norm / k_norm `+1` zero-centered
   offset. Not a perf change but baseline doesn't exist without correctness.
2. **Pattern A MoE** (Mixtral-style): on-device top-k mask × all-experts
   compute eliminates the host readback that was blocking trace.
3. **Trace capture** of the looped path: kept the same op count but amortized
   the 5120-op-per-token dispatch overhead.
4. **Batched expert matmul**: 5120 expert matmuls per token → 80 (40 layers
   × 2 stacked-expert matmuls). The biggest single win.
5. **Fused mul+sum reduction** as a single matmul: `mul(expert_out, rw) +
   sum(dim=0)` ≡ `matmul(rw_1xK, expert_out_2d)`. Sidesteps view-decay on
   the broadcast op.

## What's left between us and 27B parity (146 → 77 ms/tok ~ 1.9× more)

Per tt-perf-report's earlier per-op advice on the looped path:
- **HiFi2** instead of HiFi4 on expert matmuls — 2× kernel-time speedup. The
  kernel time is no longer dominated by dispatch, so this saving is real now.
- **L1-placed input** for matmuls — cuts DRAM BW pressure on the per-step
  input read.
- **bf8 expert weights** — halves the per-layer weight read (~7 MB → ~3.5 MB).
  No correctness risk per our earlier 27B work (bf8 weights are prod there).

Re-profile the traced batched path first to see which is the new largest hot
spot before applying optimizations one at a time (each gets a real
measurement, not a projection).
