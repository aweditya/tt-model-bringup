# Tracy + tt-perf-report profiling cheatsheet (qb1)

## One-line summary
Per-op kernel time + op-to-op gap + tt-perf-report's "Bound" classification for one MoE / DN / attn call inside a signposted ttnn region.

## One-time setup (qb1)
```
pipx install tt-perf-report                                    # gives ~/.local/bin/tt-perf-report
python3 experiments/utils/_patch_tracy_assertion.py            # idempotent; safe to re-run
```
The patch turns Tracy's `assert candidates` (process_ops_logs.py ~L561) into a `continue` so the host/device merge step doesn't abort on trace-replay ops that don't match host op_ids.

## Capture one MoE call
Working scope = ONE MoE call inside `tracy.signpost(...)`. The probe (`tracy_profile_one_moe.py`) bootstraps Pattern A, warms JIT, then signposts exactly one `moe_forward_ttnn_pattern_a`.

```
ssh qb1 'cd ~/tt-xla && bash experiments/utils/run_tracy_probe.sh \
    experiments/utils/tracy_profile_one_moe.py tracy_one_moe'
```
The wrapper does `tt-smi -r 0,1,2,3`, exports ttnn env + venv-first PATH (Tracy subprocess uses `/usr/bin/python3` otherwise), runs `python -m tracy -r -p -v -o .cache/perf_logs/<dir> <probe>`, and prints the CSV path.

## Read the report
Headline numbers (median kernel, median op2op, dispatch fraction) — the wrapper already runs this, or invoke manually:
```
ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/analyze_ops_perf_results.py \
    .cache/perf_logs/tracy_one_moe/reports/*/ops_perf_results_*.csv'
```
Per-op detail table with "Bound" classification + advice:
```
ssh qb1 'PATH=/home/aditya/.local/bin:$PATH tt-perf-report \
    .cache/perf_logs/tracy_one_moe/reports/*/ops_perf_results_*.csv'
```

## When NOT to use
Don't profile the full forward — overflows Tracy's 12000-marker-per-RISCV DRAM buffer ("markers were dropped!"). Capture ONE MoE / DN / attn block at a time; the structure repeats across all 40 layers.

## Headline reference numbers (Pattern A MoE, eager, 2026-05-25)
- Median matmul kernel: **~30 μs** (tight p25..p95)
- Median op-to-op gap: **~3.4 ms**
- Dispatch fraction: **0.997**

Future runs should land near these for eager Pattern A; batched matmul or trace should drop dispatch fraction substantially.
