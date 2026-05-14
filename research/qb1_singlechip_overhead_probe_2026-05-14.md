# qb1 Single-Chip Host/I/O Overhead Probe — 2026-05-14

Scope: qb1, one P150, resident `experiments/serve/server.py` only. This
experiment must not stop/start the server or open a standalone device handle.

## Hypothesis

Traced decode is still paying avoidable per-token host/I/O time outside the
captured graph. In `bench_decode_traced`, the authoritative full-token timing
includes input-buffer updates, `execute_trace`, synchronization, and logits
readback. The server also reports the `execute_trace`-only median. Their gap is
the opportunity size for on-device argmax/token feedback or command-queue I/O
overlap.

## Validation

Helper:

```bash
ssh qb1 'cd ~/tt-xla && .venv/bin/python -u experiments/utils/qb1_traced_overhead_probe.py \
  --tokens 32 --warmup 5 --runs 3 --validate-steps 5 \
  --output-json research/probe_logs/qb1_traced_overhead_2026-05-14.json'
```

The helper only calls server RPCs over the Unix socket. It refuses to infer any
speedup; it records full median, `execute_trace` median, and their difference.

## Decision Gate

Pursue on-device argmax/token feedback or multi-CQ I/O overlap only if the
median gap is at least 5 ms/token across repeated resident-server runs.

If the median gap is below 5 ms/token, Hypothesis 1 is invalidated for now and
the next low-risk qb1 work should refresh traced kernel-body attribution via
Tracy or operation metadata tracing.

## Result

Run artifact:
`research/probe_logs/qb1_traced_overhead_2026-05-14.json`

Measured on the resident qb1 server (`loaded=true`, `mock=false`, device 0):

| Runs | Tokens/run | Median full | Median execute_trace | Median gap | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| 3 | 32 | 241.56 ms/tok | 195.74 ms/tok | 45.82 ms/tok | pursue token feedback / I/O overlap |

Validation cosine stayed high (`min_cos=0.9999997887209274` on all runs).
This is an opportunity-size measurement only; it is not a speedup claim.

Next low-risk experiment: add a server-resident decomposition probe that times
input-buffer update, trace execution, and logits readback separately, still via
the persistent server. If readback dominates, prototype on-device argmax/token
feedback. If input updates dominate, prototype command-queue overlap or fewer
host buffer updates.

## Invalidation Criteria

- `median_overhead_ms < 5.0` across repeated resident-server runs.
- Server status reports `mock=true`, `loaded=false`, or validation cosine fails.
- The probe requires trace recapture or standalone device ownership while
  another process owns the device.
- Full-token timing regresses but `execute_trace` timing does not reproduce on
  rerun, indicating server load/noise rather than a stable optimization target.
