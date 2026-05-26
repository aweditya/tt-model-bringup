# qb2 Decode Profile — 2026-05-15

Resident-server fallback profile for the current P25 qb2 TP decode body.

Artifacts:

- `.cache/qb2_tp_profile/results_decode_op_counts_20260515_0129.json`
- `.cache/qb2_tp_profile/results_decode_op_timed_20260515_0130.json`

Important limitation: this is not true Tracy timing inside `execute_trace`.
qb2 is not currently running a Tracy/profiler-enabled server build, and the
non-negotiable server rule prevents opening a competing raw TTNN process. The
endpoint profiles the same production forward function used to capture the
decode trace, but executes it eagerly inside the resident server. The timed
mode sync-bounds every recorded TTNN op, so the totals do not equal the
`82.2 ms` trace replay time.

## Count Breakdown

Total recorded TTNN calls in one decode body: `4268`.

| Category | Count |
| --- | ---: |
| DeltaNet recurrence | 816 |
| DeltaNet decay/gate | 480 |
| DeltaNet other | 384 |
| DeltaNet QKV repeat | 336 |
| Matmul | 321 |
| Attention other | 320 |
| RoPE | 320 |
| RMSNorm | 305 |
| DeltaNet conv | 288 |
| DeltaNet output gate | 240 |
| DeltaNet state update | 144 |
| Collectives | 129 |
| MLP other | 128 |
| Cache update | 32 |
| SDPA | 16 |
| LM head / IO | 5 |

Top operation counts:

| Op | Count |
| --- | ---: |
| `ttnn.reshape` | 1043 |
| `ttnn.slice` | 593 |
| `ttnn.mul` | 528 |
| `ttnn.linear` | 321 |
| `ttnn.rms_norm` | 305 |
| `ttnn.add` | 304 |
| `ttnn.sum` | 144 |
| `ttnn.exp` | 144 |
| `ttnn.all_reduce` | 128 |

## Timed Eager Proxy

Timed mode reported `672.098 ms` total profiled op time and `960.573 ms`
whole eager forward time. These numbers are only a proxy for relative
category weight; do not compare them directly to trace replay.

| Category | Count | Sync-bounded ms | Share |
| --- | ---: | ---: | ---: |
| DeltaNet recurrence | 816 | 117.053 | 17.42% |
| Matmul | 321 | 77.583 | 11.54% |
| DeltaNet decay/gate | 480 | 73.958 | 11.00% |
| DeltaNet conv | 288 | 63.987 | 9.52% |
| DeltaNet other | 384 | 61.822 | 9.20% |
| RoPE | 320 | 44.845 | 6.67% |
| DeltaNet QKV repeat | 336 | 44.593 | 6.63% |
| Attention other | 320 | 37.012 | 5.51% |
| RMSNorm | 305 | 36.737 | 5.47% |
| Collectives | 129 | 35.197 | 5.24% |
| DeltaNet output gate | 240 | 31.685 | 4.71% |
| MLP other | 128 | 27.706 | 4.12% |
| DeltaNet state update | 144 | 13.021 | 1.94% |
| Cache update | 32 | 3.641 | 0.54% |
| SDPA | 16 | 2.152 | 0.32% |
| LM head / IO | 5 | 0.861 | 0.13% |

## Interpretation

The profile supports the current working model: the 82 ms trace is a large
unfused batch-1 decode graph. The dominant evidence is not SDPA, cache update,
or LM-head readback. The largest safe-proxy category is DeltaNet recurrence,
but the wider DeltaNet small-op envelope is larger than recurrence alone:
recurrence, decay/gate, conv, QKV repeat, output gate, state update, and
miscellaneous DeltaNet ops together account for most recorded calls and most
sync-bounded proxy time.

Practical next target: a guarded DeltaNet recurrence/body fusion experiment.
The validation gate should be tensor-level recurrence equivalence first, then
temporary trace timing, then full decode timing. No speedup should be claimed
until the full trace/full decode comparison moves.

## Tracy Trace Replay Pass

Artifacts:

- `research/probe_logs/qb2_tp_tracy_p25_sync_20260515_0446/.logs/`
- `.cache/qb2_tp_tracy/p25_manual_sync_summary_20260515_0446.json`
- Parser: `experiments/utils/analyze_tracy_overlap.py`

Command shape:

```bash
ssh qb2 'cd ~/tt-xla && PATH=$HOME/tt-xla/.venv/bin:$PATH \
  TT_METAL_HOME=$HOME/tenstorrent/tt-metal ARCH_NAME=blackhole \
  PYTHONPATH=$HOME/tenstorrent/tt-metal/ttnn:$PYTHONPATH \
  LD_LIBRARY_PATH=$HOME/tenstorrent/tt-metal/build_tracy_gcc12_nodist/lib:$LD_LIBRARY_PATH \
  python3 -m tracy -r -v -p --sync-host-device --device-trace-profiler \
    --op-support-count 20000 \
    --tracy-tools-folder $HOME/tenstorrent/tt-metal/build_tracy_gcc12_nodist/tools/profiler/bin \
    -o research/probe_logs/qb2_tp_tracy_p25_sync_20260515_0446 \
    experiments/utils/qb2_tp_tracy_profile_probe.py \
    --iters 2 --warmup 1 --mode manual \
    --output-json .cache/qb2_tp_tracy/p25_manual_sync_summary_20260515_0446.json'
```

Measured replay stayed on the P25 baseline:

| Component | Median ms |
| --- | ---: |
| `execute_trace` | 82.293 |
| `update_input_buffers + execute_trace` | 82.799 |
| argmax readback | 1.214 |

The synced run produced `sync_device_info.csv`, `profile_log_device.csv`,
`tracy_profile_log_host.tracy`, `tracy_ops_times.csv`, and
`tracy_ops_data.csv`. The final Tenstorrent op/device join failed:

```text
AssertionError: Device data missing: Op 1026 not present in cpp_device_perf_report.csv for device 2 (trace_id=None)
```

Manual parsing with `analyze_tracy_overlap.py` applies the sync scale/shift and
shows coarse chip-level overlap. Across six trace runs, max median start skew
between the four devices was about `0.051 ms`; max median end skew was about
`0.057 ms`. Per-device `TRACE-FW` spans were about `82.18-82.22 ms`.

Interpretation: all four chips are active over the same replay window. This is
useful evidence that the TP replay is not serialized chip-by-chip. It does not
prove communication is overlapped with compute inside a token. The current CSVs
do not label collective intervals versus matmul intervals inside the replay,
and host `TT_DNN_DEVICE_OP` metadata coverage was incomplete (`540` known op
IDs, `71136` timing rows, `70596` rows without expanded metadata).

Next profiler step, if overlap remains the key question: rerun with NOC/fabric
event collection or add explicit annotations around collective-heavy regions,
then require device-timeline evidence before claiming comm/compute overlap.
