# Profiling quick reference — Tracy + tt-perf-report (qb1)

## 1. What it's for

`tt-perf-report` consumes a Tracy `ops_perf_results_*.csv` and emits a per-op table with **kernel device time**, **op-to-op gap**, **cores used**, **DRAM %**, **FLOPs %**, **math fidelity**, plus a **Bound** classifier (`SLOW` / `HOST` / etc.) and a one-line piece of advice per SLOW op (e.g. "place input 0 in L1", "HiFi2 gives 2× kernel speedup"). It auto-detects CSV v2.1 + `blackhole` arch. Use it when you can't tell whether an op is dispatch-bound, BW-bound, or compute-bound — the classifier tells you which.

## 2. One-time setup (qb1)

```
ssh qb1 'pipx install tt-perf-report'                                # → ~/.local/bin/tt-perf-report
ssh qb1 'cd ~/tt-xla && python3 experiments/utils/_patch_tracy_assertion.py'
```

The patch turns Tracy's `assert candidates` (`tracy/process_ops_logs.py:561`) into `continue`, so the host/device merge doesn't abort on traced-replay ops whose host op_ids don't match. Idempotent. Stock tt-metal at `~/tenstorrent/tt-metal` is already Tracy-enabled (default build) — no rebuild needed.

## 3. Capture

```
ssh qb1 'cd ~/tt-xla && bash experiments/utils/run_tracy_probe.sh \
    experiments/utils/tracy_profile_one_moe.py tracy_one_moe'
```

The wrapper does `tt-smi -r 0,1,2,3`, exports ttnn env + venv-first `PATH` (Tracy's subprocess calls `python3 -m tracy <script>` — needs the venv `python3` so the `tracy` module resolves), runs `python -m tracy -r -p -v -o .cache/perf_logs/<dir> <probe>`, then auto-runs the analyzer on the resulting CSV.

**Scope rule: profile ONE block at a time** (one MoE / DN / attn call inside `tracy.signpost("Performance pass start")` … `tracy.signpost("Performance pass end")`). The full forward overflows Tracy's per-RISCV **12000-marker DRAM buffer** ("markers were dropped!"). `tracy_profile_one_moe.py` is the canonical probe — warms JIT twice, signposts one `moe_forward_ttnn_pattern_a`.

## 4. Read the report

Per-op table with Bound + advice, signpost-filtered:
```
ssh qb1 'PATH=/home/aditya/.local/bin:$PATH tt-perf-report \
    --start-signpost "Performance pass start" \
    --end-signpost   "Performance pass end" \
    --no-color \
    .cache/perf_logs/tracy_one_moe/reports/*/ops_perf_results_*.csv'
```
Add `--csv <out>` to dump the report, `--stacked-csv` / `--no-stacked-report` for the alternate views.

Headline aggregates (median kernel μs, median op2op μs, **dispatch fraction**, per-op-code counts) — no pandas dep:
```
ssh qb1 'cd ~/tt-xla && .venv/bin/python experiments/utils/analyze_ops_perf_results.py \
    .cache/perf_logs/tracy_one_moe/reports/*/ops_perf_results_*.csv'
```

## 5. Reference numbers (Pattern A MoE on qb1, eager, 2026-05-25)

- Median matmul kernel: **~30 μs** (p25..p95 all within 29.6..30.0 μs — kernel itself is tight)
- Median op-to-op gap: **~3.4 ms** (eager)
- **Dispatch fraction: ~0.997** (eager). Anything <0.5 in a future run = trace is doing its job.
- tt-perf-report's advice on every Pattern A matmul (shapes `32×2048×1024`, `32×512×2048`, `32×2048×128`):
  1. "Output subblock 1×1 is small — try `out_subblock_h * out_subblock_w >= 2`"
  2. "HiFi2 gives 2× kernel speedup vs HiFi4 (discards lowest activation bit)"
  3. "Input 0 in `DEV_0_DRAM_INTERLEAVED` — consider L1"
- Platform: Blackhole, **110 worker cores** (post fw 19.5.0/19.6.0 silent downgrade), **404 GB/s measured DRAM BW** (79% of 512 peak). tt-perf-report reads `blackhole` from the CSV.

## 6. Gotchas

- **DRAM marker buffer overflow** on full forward → profile one MoE/DN/attn block at a time. The block repeats across all 40 layers, so per-op insight is identical.
- **Trace replay dispatches no new ttnn ops.** Signposting `execute_trace` gives Tracy 0 ops to see. To profile what's inside a trace, profile the **eager** equivalent, or profile the trace-**capture** phase.
- **Merged CSV needs `_patch_tracy_assertion.py`** — without it `tt-perf-report` aborts on traced replays where host op_ids don't match.
- **PATH must put the venv first** so Tracy's subprocess `python3` resolves to the venv binary (where `tracy` lives), not `/usr/bin/python3`. `run_tracy_probe.sh` handles this.
- **Always warm JIT before the signposted call** — capturing during JIT hangs.

## 7. Files

- `experiments/utils/run_tracy_probe.sh` — one-command capture wrapper
- `experiments/utils/tracy_profile_one_moe.py` — canonical one-block probe
- `experiments/utils/tracy_profile_traced_decode.py` — fuller probe; overflows buffer (kept as cautionary example)
- `experiments/utils/_patch_tracy_assertion.py` — idempotent one-time patch
- `experiments/utils/analyze_ops_perf_results.py` — pandas-free aggregate analyzer
- `research/35b_tt_perf_report_findings.md` — full empirical writeup (matmul stats table, per-op counts, action items)
