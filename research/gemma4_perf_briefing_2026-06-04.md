# Gemma 4 12B perf-opt briefing (2026-06-04)

From the background research agent. Baseline **51.3 ms/tok traced, 19.5
tok/s** on (1,4) P150. Roofline floor: bf16 6 GB/chip × 404 GB/s ⇒
14.85 ms/tok = 67.3 tok/s. **We're at 29% of BW ceiling — 3.4× headroom**.

## Capture + analyse

Fork `experiments/utils/tracy_profile_one_moe.py` →
`tracy_profile_one_gemma4_layer.py` (the full forward overflows
Tracy's 12k-marker DRAM buffer; one block at a time).

```
ssh qb1 'cd ~/tt-xla && bash experiments/utils/run_tracy_probe.sh \
    experiments/utils/tracy_profile_one_gemma4_layer.py tracy_gemma4_layer'

ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/analyze_ops_perf_results.py \
    .cache/perf_logs/tracy_gemma4_layer/reports/*/ops_perf_results_*.csv'

ssh qb1 'PATH=/home/aditya/.local/bin:$PATH tt-perf-report --start-signpost "Performance pass start" \
    --end-signpost "Performance pass end" --no-color \
    .cache/perf_logs/tracy_gemma4_layer/reports/*/ops_perf_results_*.csv'
```

**Trace replay dispatches no ttnn ops** — profile the eager equivalent
or the trace-capture phase.

## TOP 3 OPTIMIZATIONS (in order)

### #1: Vocab-sharded lm_head + on-device argmax (~4h, +5-8%)
- Forks `feedback_vocab_sharded_lm_head_result.md` (P22 commit `ef3f336`).
- 27B: 11.43 → 12.02 tok/s (+5.1% e2e, **14.6× isolated**).
- **Likely BIGGER on Gemma 4**: vocab=**262144** (vs 27B's 248320).
  262144 % 4 = 65536 (clean shard), % 32 = 8192 (tile-aligned).
- Tied embed = lm_head.T → same shard layout is consistent.
- Gotcha: `argmax(keepdim=False)` returns garbage at large N — keep True.

### #2: Distributed RMSNorm across mesh (~6h, +12-15 ms/tok projected)
- Forks `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:1358-1390`.
- 27B projected 18.3 ms/tok at 305 norm calls/tok.
- **Gemma 4 has 4 RMSNorms/layer × 48 layers + 1 final = 193 calls/tok**
  (slightly fewer than 27B but every one is a candidate).
- Uses Llama-style `w` (no `+1.0` Qwen offset) — matches the
  reference exactly, no patch needed.

### #3: paged_scaled_dot_product_attention_decode + B3 HiFi2 on globals (~6h, +20-40% on attn)
- Forks `feedback_paged_sdpa_shipped_tp.md` (commit `4741253`).
- 27B: 7.02 → 11.43 tok/s on (1,4) mesh (+62%).
- **Already shipped on Gemma 4 sliding layers**. The 8 global layers
  also benefit; sliding pattern already validates the kernel contract.
- Double-duty: B3 HiFi2 also fixes the long-context drift cliff
  ([[fp32-sdpa-cliff-probe]]).

## Already RULED OUT

- **DRAM-sharded MLP matmul** ([[dram-sharded-mlp-probe]]): 2.1× SLOWER
  on P150 at batch=1. Interleaved bf8 already at 79% of 404 GB/s.
  Gemma 4's MLP=15360 is even smaller than 27B's 17408 — same regime.
- **Async all_reduce** ([[async-ccl-negative]]): +4% setup tax, no
  overlap window on serial residual.
- **HiFi2 on activation-bound matmuls** ([[35b_perf_milestones]]):
  no-op at MoE shape. Reserve HiFi2 for SDPA and lm_head.
- **bf8 weights as a perf lever** ([[bf8-mlp-weights]]): kernel-time
  IDENTICAL at bf16 (930 μs both). Only for memory pressure.

## Roofline

12B bf16 = 24 GB / 6 GB/chip / 404 GB/s = **14.85 ms/tok = 67 tok/s**.
Current 51.3 ms = 29%. Headroom **3.4×**. bf8 floor 7.4 ms = 135 tok/s
(only worth chasing after we're inside 2× of bf16 floor).

## "If Tracy shows..."

- **lm_head matmul dominates** → ship #1 (vocab-shard).
- **CCL kernels top** → ship #2 (distributed RMSNorm + reduce_scatter).
- **manual SDPA chain dominates global-attn** → ship #3 (paged SDPA + B3).
- **matmuls <50% of 404 GB/s** → audit kernel_config (HiFi2,
  packer_l1_acc, `out_subblock_h*w ≥ 2`) before any layout change.
- **paged_update_cache twice per layer** → ship
  `paged_fused_update_cache` (~1.6 ms/tok, 1 hr).
- **`to_memory_config` between layers** → keep residual in L1.
- **dispatch fraction >0.5 in trace** → trace not actually working;
  debug before perf.
