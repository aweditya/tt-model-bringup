# qb2 Multi-Chip Performance Plan — 2026-05-14

Goal: optimize Qwen3.6-27B TP4 on qb2 toward hardware-ceiling proximity. Use
measured full-decode performance as the source of truth. Do not claim an
optimization win until it passes correctness gates and improves measured
production-path ms/tok.

## Current measured baseline

Server: qb2 persistent `experiments.serve.server_tp`, P25 path active
(`embed_tt`, `cos_table_tt`, `sin_table_tt` in server log).

Command:

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/qb2_tp_generate_bench.py \
  --runs 5 --warmup 1 --max-tokens 30 --chunk-size 30'
```

Result JSON:

```text
~/tt-xla/.cache/qb2_tp_generate_bench/results_20260514_220947.json
~/tt-xla/.cache/qb2_tp_generate_bench/results_20260514_222053.json  # after component endpoint restart
```

Measured, excluding one warmup:

| Metric | Value |
|---|---:|
| median ms/tok | 82.809 |
| mean ms/tok | 82.807 |
| median tok/s | 12.076 |
| min/max tok/s | 12.075 / 12.078 |

Prompt output is coherent ("Paris" continuation). This is the benchmark
baseline for subsequent qb2 changes.

Post-endpoint restart recheck (`results_20260514_222053.json`) matched the
baseline: median 82.808 ms/tok, median 12.076 tok/s. The instrumentation did
not perturb the production generate path.

## Current component decomposition

Server-resident endpoint:

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
  bench_decode_tp_components --iters 20 --warmup 3'
```

Result JSON:

```text
~/tt-xla/.cache/qb2_tp_components/results_20260514_222202.json
```

Measured medians:

| Component | Median ms |
|---|---:|
| `update_input_buffers` only | 0.718 |
| `execute_trace` only | 82.183 |
| `update_input_buffers + execute_trace` | 82.693 |
| tiny argmax readback | 1.419 |

Interpretation:

- The production timed region is almost entirely trace replay. The remaining
  host update path is less than 1 ms/tok in this sync-bounded probe.
- On-device `plus_one` for position state may still be worth doing for
  cleanliness, but this measurement bounds the immediate opportunity: the
  whole measured update region is ~0.7 ms/tok.
- The next multi-chip optimization should target trace-body work or collective
  layout inside the trace, not host input-buffer overhead.

## Rules for this track

- Main thread owns qb2 multi-chip work.
- Use the persistent server unless a raw ttnn probe is explicitly coordinated.
  `ttnn` opens all chips; concurrent device processes can SIGBUS or hang
  (`feedback_mtp_head_probe.md`).
- Before claiming a win, run the same full-decode benchmark and record JSON.
- Any projected improvement must state the measured component cost it removes
  or the roofline calculation behind it.
- If cosine/top-1 correctness drops below the current path, stop and ablate
  before optimizing (`feedback_correctness_first.md`).

## Immediate hypotheses

### H0: Refresh the P25 breakdown — done

Hypothesis: post-P25 bottlenecks differ enough from the old 7.02 tok/s
breakdown that we need a new decomposition before choosing code changes.

Evidence: P22 removed full-logits readback; P25 moved embedding/cos/sin lookup
on device. Old `feedback_tracy_tp_breakdown.md` still includes 9.4 ms logits
readback and 1.9 ms update_input_buffers from the pre-P22/P25 path.

Validation:

- Implemented server-resident `bench_decode_tp_components` in
  `experiments/serve/server_tp.py` and client command in
  `experiments/serve/client_tp.py`.
- Gate passed: post-endpoint generate benchmark still reproduces ~82.8 ms/tok.
- Result: trace replay is the dominant measured region; update/input overhead
  is small.

### H1: Move remaining position updates on device

Hypothesis: P25 still writes `tok_buf`, `cur_pos_buf`, and `rot_idxs_buf` from
host each step. Moving `cur_pos` / rotary index increment into the trace may
reduce the remaining host update cost, but the measured opportunity is bounded
by the full update bucket.

Evidence:

- P25 benchmark showed net +0.4% after replacing host embedding/cos/sin writes
  with on-device lookup.
- Current code still writes `cur_pos_buf` and `rot_idxs_buf` in
  `server_tp.py:update_input_buffers`.

Validation:

- H0 measured update at ~0.718 ms/tok. Treat this as a cleanup/latency-tail
  candidate, not the main path to hardware-ceiling proximity.
- Correctness gate: generated token sequence matches current top-1 for a short
  deterministic prompt; paged cache positions advance exactly.

### H2: CCL/layout cleanup around attention and MLP exits

Hypothesis: current row-parallel exits may pay avoidable collective/layout cost.
Potential mechanisms include `all_gather_concat`, persistent-buffer async CCL,
or fused matmul+collective APIs.

Evidence:

- Candidate APIs and production references are cataloged in
  `research/multi_chip_optimizations_menu_v2_addendum.md`.
- Current server uses `ttnn.all_reduce` exits in DeltaNet, MLP, and gated
  attention.

Validation:

- Do not start with a broad refactor. Pick one exit path after H0 identifies a
  measurable collective/layout bucket.
- Implement behind a flag or separate helper.
- Correctness gate: inter-chip outputs agree and sequence top-1 matches current
  path.
- Perf gate: same full-decode benchmark improves over 82.809 ms/tok median.

### H3: Reduce manual RoPE dispatches

Hypothesis: current manual partial-RoPE sequence still dispatches many small ops
per full-attention layer. A native/fused RoPE path may reduce that cost if the
current TT-NN API supports Qwen's partial rotary shape.

Evidence:

- Manual rotate-only remains in `server_tp.py:gated_attn_step_tp`.
- Prior native RoPE attempts had shape/padding constraints
  (`feedback_c3_native_rope_abandoned.md`,
  `feedback_native_rope_api_shape.md`), so this is an API-compatibility
  question first.

Validation:

- Read current tt-metal docs/reference before coding.
- Probe Q/K RoPE equivalence at production shapes.
- Only run full decode after the tensor-level gate passes.

## Deprioritized unless new evidence appears

- Distributed RMSNorm Step 1: correct but slower in P24
  (`feedback_distributed_rms_norm_failed.md`).
- DRAM-sharded MLP on single P150/current shape: 2.1x slower in P23
  (`feedback_dram_sharded_mlp_probe.md`).
- Speculative decoding D'3: below break-even acceptance at current evidence
  (`feedback_d3_dont_ship_yet.md`).
