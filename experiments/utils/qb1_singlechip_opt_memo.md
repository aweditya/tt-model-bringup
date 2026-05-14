# qb1 Single-Chip Qwen3.6-27B Optimization Memo

Scope: qb1 only, one P150, current `experiments/91f_qwen36_27b_full_ondevice.py`
through `experiments/serve/server.py`. Do not stop/start the server or run
standalone device probes without coordination.

## Baseline facts

- Current measured single-chip production point is `192.81 ms/tok = 5.19 tok/s`
  after QK rms_norm fusion (`feedback_qk_rms_norm_shipped.md`).
- Full-decode timing is authoritative; per-block projections are ceilings only
  (`feedback_real_vs_projected.md`, `feedback_benchmark_methodology.md`).
- Current 91f already ships fused DN input projection, attn QKV projection
  fusion, QK `ttnn.rms_norm`, rotate-only RoPE, paged KV, and C'4 traced
  state threading.
- Tech-report anchors:
  - `experiments/.refs/tt-metal/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md`
    says trace removes host dispatch gaps but fixed tensor addresses are
    required; multiple command queues can overlap I/O with model execution.
  - `experiments/.refs/tt-metal/tech_reports/FlashAttention/FlashDecode.md`
    says decode SDPA uses `cur_pos_tensor` for causal masking and that
    over-assigning cores per KV head can create NOC/reduction overhead.
  - `experiments/.refs/tt-metal/tech_reports/ttnn/operation-tracing.md`
    documents operation metadata tracing, useful for op-count deltas when
    device profiling is unavailable.

## Hypothesis 1: traced decode is still paying avoidable host/I/O time

In `server.py:handle_bench_decode_traced`, each token still updates embedding,
RoPE, position, and scatter-index buffers from host and then reads back full
logits for host argmax. TT-Metal examples show a traced on-device argmax
feedback loop in `models/demos/audio/whisper/tt/whisper_generator.py`.

Validation helper: `experiments/utils/qb1_traced_overhead_probe.py`.

Gate: if `median_ms - median_exec_ms >= 5 ms/tok`, prototype a separate
on-device argmax/token-feedback trace before editing 91f/server production.
This is a measurement of opportunity size, not a speedup claim.

2026-05-14 result: resident-server run on qb1 device 0 measured
`median_full_ms=241.56`, `median_execute_trace_ms=195.74`, gap
`45.82 ms/tok` across 3×32-token runs with validation cosine
`0.9999997887209274`. Artifact:
`research/probe_logs/qb1_traced_overhead_2026-05-14.json`.

Next validation should split that 45.82 ms/tok into input-buffer update,
`execute_trace`, and logits readback, still through the persistent server. If
readback dominates, prototype on-device argmax/token feedback. If input updates
dominate, prototype command-queue overlap or fewer host buffer updates.

## Hypothesis 2: if host/I/O is small, remaining single-chip work is kernel-body

Memory notes point at DeltaNet small ops, not MLP bandwidth:
`feedback_deltanet_perop_findings.md`, `feedback_conv1d_diagnosis.md`,
`feedback_qb1_mlp_at_78pct_peak.md`, and the negative
`feedback_dram_sharded_mlp_probe.md`. If Hypothesis 1 measures <5 ms/tok
overhead, the next qb1 probe should use Tracy (`reference_tracy_build_qb1.md`)
or operation tracing to refresh the C' post-QK-fusion breakdown before any
new fusion work.

Candidate validation path: run the existing
`experiments/utils/tracy_traced_decode_probe.py` under the qb1 Tracy wrapper,
or add a no-device operation-trace diff helper for 91f variants. Do not infer
tok/s from isolated component timing; require `bench_decode_traced` afterward.
