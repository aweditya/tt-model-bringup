# Wiki 52: Benchmark Audit — Our Reported Numbers Were Wrong

## The Finding

**Experiments 60-73 timed only `ttnn.execute_trace()`, excluding host-device data transfer.** The actual end-to-end decode speed is significantly slower than reported.

## The Methodology Bug

All experiments used this timing pattern:

```python
update_buffers(next_id, pos)          # NOT timed (3.3% of real time)
t0 = time.perf_counter()
ttnn.execute_trace(device, ..., blocking=True)  # ONLY this timed
times.append(time.perf_counter() - t0)
logits = from_dev(logits_ref, ...)    # NOT timed (32.9% of real time!)
next_id = int(np.argmax(logits))      # NOT timed (negligible)
```

## Experiment 81: Component Breakdown (Qwen2.5-0.5B)

| Component | Time (ms) | % of Total | What It Does |
|-----------|-----------|-----------|--------------|
| `update_buffers` | 0.39 | 3.3% | 3× `ttnn.copy` for embed, RoPE, pos |
| `execute_trace` | 7.58 | 63.6% | Full transformer forward pass on device |
| `from_dev` | 3.91 | 32.9% | `ttnn.to_torch()` logits readback |
| `np.argmax` | 0.03 | 0.2% | Token selection |
| **TOTAL** | **11.91** | 100% | |

## Corrected Performance Numbers

| Model | Reported tok/s | Corrected tok/s* | Overhead % |
|-------|---------------|-----------------|------------|
| Qwen2.5-0.5B | 140 | ~84 | 57% |
| Llama-3.2-1B | 78 | ~60** | ~30% |
| Llama-3.2-3B | 34 | ~29** | ~17% |
| Llama-3.1-8B | 19 | ~18 | ~6% |

*Estimated based on exp 81 measurement + scaling analysis
**Need to run exp 81 on each model to get exact numbers

### Why Larger Models Are Less Affected

The `from_dev` overhead is roughly constant (~4ms for any vocab size readback). As trace execution time grows with model size, the constant overhead becomes a smaller fraction:

- 0.5B: 4ms / 7.6ms = 53% overhead
- 8B: 4ms / 52ms = 8% overhead

## The `from_dev` Bottleneck

Reading back 151,936 × 4 bytes = 608 KB takes 3.91ms = **155 MB/s effective throughput**.

This is far below PCIe 4.0 bandwidth (~25 GB/s). The bottleneck is NOT PCIe transfer — it's the `ttnn.to_torch()` conversion (tensor format conversion, memory copies, Python overhead).

## Implications

1. **Our small-model numbers were inflated by ~40-57%.** The 0.5B model is really 84 tok/s, not 140.
2. **Our large-model numbers were roughly correct.** The 8B at 18 tok/s (exp 80, which timed correctly) is the honest number.
3. **The biggest optimization opportunity is `from_dev`.** If we could eliminate the 4ms readback:
   - 0.5B would go from 84 → 132 tok/s (1.6x)
   - 8B would go from 18 → 19 tok/s (negligible)
4. **On-device argmax would eliminate `from_dev` entirely** — we'd only need to read back a single int32.

## Action Items

- [ ] Re-run exp 81 on all models to get exact corrected numbers
- [ ] Investigate on-device argmax (we tried before, 90ms in trace — was this a bug?)
- [ ] Profile `ttnn.to_torch()` to understand the 155 MB/s bottleneck
- [ ] Consider reading back only top-k logits instead of full vocab
- [ ] Update PLAN.md with corrected numbers once verified
