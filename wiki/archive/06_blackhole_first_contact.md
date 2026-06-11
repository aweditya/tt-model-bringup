# First Contact with Blackhole

## Q: What hardware do we have?

**A (tt-smi snapshot + Experiment 06):**

| Property | Value |
|----------|-------|
| Board type | **p150a** (×2 boards, we use device 0) |
| Compute grid | 11 × 10 = **110 Tensix cores** |
| DRAM | 32 GB GDDR6 at 16 GT/s |
| AI clock | 800 MHz |
| ASIC temperature | ~50°C idle |
| Firmware | Bundle 19.6.0.0, DM App 0.22.0.0 |
| Driver | TT-KMD 2.6.0 |
| Harvesting | 0x0 (no cores harvested!) |
| Enabled Tensix columns | 0xFFF (all 12 columns) |
| Enabled GDDR | 0xFF (all 8 channels) |
| Power (idle) | ~35W (TDP limit 150W) |

Note: 140 total Tensix on Blackhole, but 110 usable (11×10 compute grid). The remaining 30 are likely used for ethernet, dispatch, etc.

## Q: Can we run computation on it?

**A (Experiment 06):** Yes! All four operations worked:

| Operation | Result |
|-----------|--------|
| Tensor add (32×32) | Correct (max error 0.014 — bf16 rounding) |
| Matmul (64×256 @ 256×128) | Correct (max error 0.71 — expected for bf16) |
| MLP layer (matmul + relu) | Correct (max error 0.96 — bf16 accumulation) |
| Matmul benchmark (512×1024 @ 1024×512) | **29.9x faster than CPU** |

## Q: Why are the errors "large"?

**A:** bfloat16 has only **7 bits of mantissa** (vs 23 for float32). For a matmul with 256 or 1024 accumulations, the rounding errors accumulate. Max error of 0.7 on a matmul with values in [-1, 1] range is completely normal for bf16. This is why ML uses bf16 — the reduced precision is acceptable for neural network training/inference.

## Q: How fast is the Blackhole?

**A (Experiment 06):**

```
Matmul (512, 1024) @ (1024, 512):
  Blackhole: 0.021 ms  →  26 TFLOPS (bf16)
  CPU:       0.616 ms
  Speedup:   29.9x
```

**But** 26 TFLOPS is only **7% of peak** (372 TFLOPS bf16). Why?

The matmul is **too small** to saturate 110 Tensix cores. Each core can do an 8×16 @ 16×16 multiply per cycle at 800 MHz. With 110 cores, we need much larger matrices to keep them all busy. This is the same phenomenon as GPU utilization — you need enough parallelism to fill the hardware.

L1 vs DRAM made almost no difference (1.01x) because at this size, the data fits easily and the bottleneck is core utilization, not memory bandwidth.

## Q: What does TT-NN code look like?

**A:** Very similar to PyTorch, but with explicit device management and tile layout:

```python
import ttnn, torch

device = ttnn.open_device(device_id=0)

# Data must be tiled (multiples of 32) and explicitly placed on device
x = ttnn.from_torch(torch.randn(1,1,64,256), dtype=ttnn.bfloat16,
                     device=device, layout=ttnn.TILE_LAYOUT)
w = ttnn.from_torch(torch.randn(1,1,256,128), dtype=ttnn.bfloat16,
                     device=device, layout=ttnn.TILE_LAYOUT)

result = ttnn.matmul(x, w)               # runs on Blackhole
out = ttnn.to_torch(result)               # move back to CPU

result.deallocate()                        # manual memory management!
ttnn.close_device(device)
```

Key differences from PyTorch:
- **Explicit tile layout**: dimensions must be multiples of 32
- **Explicit device placement**: `device=device`
- **Manual deallocation**: L1 SRAM is scarce (1.5MB per core)
- **4D tensors**: TT-NN often wants (batch, channel, height, width)

## Experiment

`experiments/06_blackhole_first_contact.py` — run on the remote host with `ARCH_NAME=blackhole`.

## Sources
- Experiment 06 results (run 2026-04-21 on Blackhole p150a device 0)
- tt-smi snapshot data
- Blackhole specs: https://docs.tenstorrent.com/aibs/blackhole/specifications.html
