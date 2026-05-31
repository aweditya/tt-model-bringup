# CB device sampling — perf split + plan (2026-05-30)

## Measurement

`CBEngine` instrumented with two new histograms (`cb_step_device_seconds`,
`cb_step_sample_seconds`) that split each `_step_sampled` call into:

- **device** — `execute_trace` + `to_torch` + `.float().numpy()` (includes the
  device sync that `to_torch` triggers, plus the `[B, vocab]` readback).
- **sample** — the Python `for s in range(B)` loop that calls
  `base._sample_from_logits(logits[s], …)` (per-slot temp/top-p/top-k via
  numpy sort + softmax + `rng.choice`).

Run: 32 concurrent SSE clients × 30 s × `max_tokens=24` × sampling
(temp=0.8, top_p=0.95) against `serve_cb.sh` (TT_CB_SLOTS=32, sampling=True)
on qb1.

```
cb_step_seconds_sum            = 36.59 s   (47 steps)
cb_step_seconds mean           = 778 ms / step
cb_step_device_seconds_sum     =  5.83 s
cb_step_device_seconds mean    = 124 ms / step    (16% of step)
cb_step_sample_seconds_sum     = 30.42 s
cb_step_sample_seconds mean    = 647 ms / step    (83% of step)
per-slot sample cost (B=32)    = 647 / 32 ≈ 20 ms / slot / step
```

## What this overturns

The earlier "host loop is a small O(B) cost, device dominates" back-of-envelope
(after the N=32 P5.1 result) was wrong. At B=32 with the production
`_sample_from_logits` numpy path:

- Each slot does `astype(float64) → top_k partition → softmax → top_p sort +
  cumsum + mask + renorm → rng.choice(vocab)`. With vocab=248320 the
  `np.argsort(p)[::-1]` alone is ~10 ms; total ~20 ms / slot — consistent with
  Python+numpy on a 1-D 248k-element array. There is no batching across slots.
- The device side (replayed logits trace + readback) is 124 ms / step at B=32 —
  in line with the earlier 125 ms baseline at B=4 plus mild O(B) growth in
  matmul / SDPA / `[B,vocab]` readback.

Net: the host sample loop dominates **5:1** at B=32. A B=4 measurement is
queued (the engine_sampling validator).

## Three concrete wins, in priority order

### W1 — vectorised greedy host argmax (cheapest, biggest single fix for greedy traffic)

`_step_sampled` falls through to `logits[s].argmax()` whenever
`sampling is None`. Replace the per-slot loop with one numpy call:

```python
greedy_mask = np.array([self.reqs[self.slots[s]]['sampling'] is None
                        for s in range(self.B) if self.slots[s] is not None])
amx = logits.argmax(axis=-1)                       # one [B] call
# fill greedy slots from amx, sampled slots from the per-slot loop
```

For all-greedy batches (the historical default, plus the chat-API's `temp=0`
requests) this turns 20 ms × B into ~10 ms total — already a >50× per-slot
win.

### W2 — on-device top-k (settles sampling too)

Capture a second trace ending in `ttnn.topk(rm, k=K, dim=-1)` (K ≥ effective
top-p, e.g. K=128). Readback shrinks `[B, vocab=248320]` → `[B, K=128]` —
roughly 2000× less data. Per-slot sampling becomes a K-element argsort which
is ~10 µs instead of ~20 ms — **a ~2000× per-slot win**.

Risks: capturing two traces (argmax + topk-logits) is the gamble we ducked in
P1; needs validation. `ttnn.topk` exists and was used in the older 35B-A3B
work — check the existing call sites for the right config.

Predicted full step at B=32 after W2: 124 ms (device) + ~5 ms (top-k readback
+ per-slot work) = **~130 ms / step**, vs current 778 ms — a **~6× total
step speedup** at B=32. Aggregate throughput at B=32 would go from
~40 tok/s → ~250 tok/s.

### W3 — vectorise the sample math across slots

Even without on-device top-k, the `_sample_from_logits` arithmetic
(`astype(float64) / temperature`, `exp`, `/sum`, `cumsum`, `argsort`) can be
done over `[B, vocab]` with numpy broadcasting. The only per-slot work is
`rng.choice(vocab, p=row)` — that one is per-slot because each request has
its own rng/seed. Cuts the host loop ~5–10× without device changes.

## Sequencing

1. **W1** ships in 30 LOC; correctness is identity (numpy argmax equals
   per-slot argmax on the same logits). Land + validate against the existing
   `engine_sampling.py` gate.
2. **W2** is the real prize but needs trace capture + validation cycle.
   Worth it; the on-device top-k path also unlocks lower memory pressure for
   the future per-request sampling-with-large-K case.
3. **W3** is a fallback if W2's two-trace capture turns out to be unreliable.

Once any of these lands, re-run the same instrumented load and confirm
`cb_step_sample_seconds / cb_step_seconds` dropped.

## Tracy aside (negative result)

`experiments/cb/profile/tracy_cb_step.py` runs against the Tracy-enabled
`build_tracy/` and the C++ post-processing step (`tt-perf-report`'s
`_enrich_ops_from_perf_csv`) asserts on missing op data inside trace replays
and crashes — even with `--device-trace-profiler`. The captured CSVs are
present (`tracy_ops_data.csv`, `cpp_device_perf_report.csv`,
`profile_log_device.csv`) but **trace-replayed ops are not emitted as
`TT_DNN_DEVICE_OP` host records**, so they can't be joined to op names. This
is a tt-metal-side limitation, not a setup bug. The instrumented histogram
approach above is the practical answer until tt-metal grows trace-aware
per-op profiling.
