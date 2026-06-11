# 22b: On-Device Broadcast Investigation

## The Problem

Our Jaxpr interpreter's #1 bottleneck is `broadcast_in_dim`, which does a CPU round-trip:
1. Read tensor from device (`ttnn.to_torch`)
2. Broadcast on CPU (`np.broadcast_to`)
3. Write back to device (`ttnn.from_torch`)

This blocks both performance and trace capture (TT-NN forbids all host-device transfers during trace).

## Discovery: On-Device Broadcast Primitives Exist

**Q: Can TT-NN broadcast on-device?**

YES. We found multiple working approaches:

| Method | Works? | Notes |
|--------|--------|-------|
| `ttnn.repeat` | YES | Repeats tensor along specified dims, stays on device |
| `ttnn.repeat_interleave` | YES | Interleaved repeat, on device |
| `ttnn.expand` | EXISTS | Available but needs testing with TILE_LAYOUT |
| `ttnn.broadcast` | EXISTS | Available, needs testing |
| ROW_MAJOR binary ops | YES | `ttnn.add((32,64), (1,64))` works in ROW_MAJOR |
| TILE_LAYOUT binary ops | NO | Still fails with shape mismatch |
| Scalar ops | YES | `ttnn.add(tensor, 3.14)` always works |

**Q: How does `ttnn.repeat` work?**

```python
# (1, 1, 1, 64) → (1, 1, 32, 64) — entirely on device!
b = ttnn.from_torch(b_row, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
b_expanded = ttnn.repeat(b, ttnn.Shape([1, 1, 32, 1]))  # repeat 32x along dim 2
```

The shape argument specifies the repeat count per dimension.

## Performance

| Method | Latency | Notes |
|--------|---------|-------|
| CPU round-trip (current) | 0.147 ms | Read + broadcast + write |
| Host pre-expand | 0.084 ms | No read, just write expanded |
| `ttnn.repeat` (on-device) | ~0 ms | No host transfers at all |

The CPU round-trip is 0.147 ms per broadcast. With 10 broadcasts per transformer forward pass, that's 1.47 ms — **26% of our 5.59 ms forward time** is just broadcast overhead.

## Impact on Trace Capture

**Q: Would switching to `ttnn.repeat` enable trace capture?**

Yes. The only reason trace capture fails is because our broadcast workaround does host-device transfers (reads AND writes). `ttnn.repeat` is a pure device-side op — it would work inside a trace.

With trace capture working, we'd expect the 2-3x speedup seen in experiments 12 and 19 to apply to the full transformer interpreter.

**Projected performance:**
- Current: 5.59 ms/forward (interpreted, with CPU broadcast)
- With on-device broadcast: ~4.1 ms (eliminate 1.47 ms of broadcast overhead)
- With trace capture: ~1.5-2.0 ms (eliminate ~21µs × 56 ops dispatch overhead)
- **Potential 3-4x improvement**

## Broadcast-Related TT-NN Ops

Full list discovered:
```
ttnn.repeat              — repeat tensor along dims (ON DEVICE)
ttnn.repeat_interleave   — interleaved repeat (ON DEVICE)
ttnn.expand              — expand tensor (exists, needs testing)
ttnn.broadcast           — broadcast op (exists, needs testing)
ttnn.bcast               — broadcast compute op
ttnn.BcastOpDim          — broadcast dimension enum
ttnn.BcastOpMath         — broadcast math type enum
```

## Next Steps

1. **Replace `broadcast_to_match` in tensors.py** with `ttnn.repeat` — eliminate all CPU round-trips
2. **Re-test trace capture** with the on-device broadcast
3. **Benchmark the improvement** — should see significant speedup
4. **Test `ttnn.expand` and `ttnn.broadcast`** — might be even more efficient than repeat
