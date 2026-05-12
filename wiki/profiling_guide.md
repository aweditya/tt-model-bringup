# Profiling Guide — three views of the same workload

Profiling ttnn workloads correctly requires understanding three different measurement strategies. Each gives a different view of the same thing. Use the right one for the question you're asking.

## TL;DR

| Question you're asking | Tool |
|---|---|
| "Did C'1 make the full decode step faster?" | **Sync-bounded host timing** (default) |
| "How is wall time spread across the regions of my code?" | **Sync-bounded host timing** + the layer-type compounding estimate |
| "Where on the timeline are my warmups vs my measured runs?" | **Tracy zones** |
| "What's the pure kernel time of this specific op (no host noise)?" | **Device profiler** |
| "Which Tensix cores are busy and when?" | **Device profiler** + Tracy viewer |
| "Is the device idle / pipeline-stalled?" | **Tracy viewer** showing gaps |

## 1. Sync-bounded host timing (the default in `perf_baseline.py`)

**Pattern:**
```python
ttnn.synchronize_device(device)
t0 = time.time()
region(...)
ttnn.synchronize_device(device)
t1 = time.time()
# t1 - t0 = end-to-end wall time
```

**Correct for:** comparing the same workload across changes ("did C'1 reduce the full decode step time?")

**Watch out for:**
- The sync itself takes time (~50 µs). Negligible at our op scales (multi-ms).
- Sync DESTROYS async pipelining within the region. If you measure individual ops in isolation, the sum overstates the real cost (production would pipeline). Our harness explicitly handles this by also reporting `compounding estimate` vs `measured full decode` so you can see the pipelining savings.

**No setup required.** Always available.

## 2. Tracy zones (`--enable-tracy`)

**Pattern:**
```python
ttnn.start_tracy_zone(source_file, function_name, line_number, color)
# ...region runs...
ttnn.stop_tracy_zone(zone_name, color)
```

**Correct for:**
- Visualizing the timeline of where time goes
- Correlating host-side regions with device-side kernel events (ttnn emits both)
- Spotting pipeline stalls / idle gaps

**Setup required:**
- ttnn must be built with Tracy enabled (verified for qb2 — `experiments/utils/tracy_availability_probe.py` confirmed the API exists)
- A **Tracy server** must be running on the host or reachable over network for live capture
- OR start the workload first, then start Tracy capture before the interesting region

**To view Tracy data:**
1. Download Tracy from https://github.com/wolfpld/tracy/releases (Linux/Mac/Windows binary)
2. Run `tracy-server` on a network-accessible host, OR record to .tracy file
3. Open the .tracy file in `tracy-profiler` UI

The Tracy UI gives you:
- Timeline of host zones (your `start/stop_tracy_zone` markers)
- Per-op device timeline (kernel start/end, what each Tensix core was doing)
- Flame graph, statistics, frame views

**Overhead:** small (~1-5%) when actively recording; negligible when not connected.

## 3. Device profiler (`--enable-device-profiler`, env: `TT_METAL_DEVICE_PROFILER=1`)

**Status: NOT available on our installed ttnn wheels (qb1 + qb2 verified).**

The ttnn wheel ships with Tracy API hooks (so `start_tracy_zone` / `stop_tracy_zone` calls succeed as no-ops) BUT the underlying tt-metal C++ binary is not Tracy-linked. Setting `TT_METAL_DEVICE_PROFILER=1` triggers a fatal assert at device open:

```
TT_FATAL: TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build of tt-metal.
```

To enable device profiling we'd need to **rebuild ttnn from source with Tracy enabled** (or wait for a Tracy-enabled wheel from upstream). This is a separate ~2-4 hour engagement.

For now, **rely on sync-bounded host timing** (mode 1) for cross-phase comparison. The diagnostic signature "device-time << host-time" that would tell us we're dispatch-bound isn't available without device profiler, but we have a proxy: if the layer-type compounding estimate is much LARGER than the measured full-decode time, dispatch overhead within the full step is large (the saving comes from pipelining across ops). Conversely, if compounding estimate ≈ measured full-decode, dispatch is small and trace capture would give modest gains.

**Pattern (for when Tracy-enabled build is available):**
```python
# After running some ops
ttnn.ReadDeviceProfiler(device)         # flush device-side timestamps to host
data = ttnn.get_latest_programs_perf_data()   # per-op records
# data is a sequence of per-program (op) records with on-device cycle counts
```

**Correct for (when available):**
- Pure kernel execution time per op (no host/dispatch noise)
- Identifying the slowest single op in a region
- Understanding why a "fast-looking" host timing might hide a slow kernel

**Setup required:**
- `TT_METAL_DEVICE_PROFILER=1` set BEFORE process start
- **ttnn build must be Tracy-enabled** ← currently not the case for us

**Caveat:** the per-op records measure on-device cycle counts. They DON'T include host-side dispatch or sync time. So if your host timing says 100 ms and device profiler says 30 ms, the missing 70 ms is dispatch + sync (or pipeline gaps).

That 70 ms-gap signature is itself diagnostic — if device-time is much smaller than host-time, you're dispatch-bound and trace capture (C'4) will help massively.

## How to use these together (the canonical workflow)

```bash
# Step 1: baseline with sync-bounded host timing
.venv/bin/python experiments/utils/perf_baseline.py --phase C0

# Step 2: same workload with device profiler to get pure kernel breakdown
TT_METAL_DEVICE_PROFILER=1 .venv/bin/python \
    experiments/utils/perf_baseline.py --phase C0 \
    --enable-device-profiler

# Step 3: if there's a specific region you want to understand visually,
# Tracy zone it and view in the Tracy UI
.venv/bin/python experiments/utils/perf_baseline.py --phase C0 \
    --enable-tracy
```

The output JSON from each run contains:
- `raw_times_ms[label]` — list of per-run times for each region
- `stats[label]` — median/min/max/stdev
- `derived` — compounding estimate vs measured
- `_device_profiler[label]` (if enabled) — per-op device data

Diff two phases' JSON files to attribute exactly how much each change moved each region.

## The mistake we want to avoid

The B'9 mistake — hypothesizing about a bug without measurement, applying a fix, and seeing no improvement — translates directly to perf:

- **Don't:** "I think the KV roundtrip is slow. Let me implement `paged_update_cache`."
- **Do:** "The baseline shows `prefill_plus_one_decode` median = 1.2 s. Let me capture device-profiler data, see what's actually heavy, then attack the biggest single thing."

Every perf phase commit message should include the BEFORE and AFTER medians for the targeted region. No medians → no claim.
