# Phase 5 Benchmark Results

Run with `bench_device.py`. Each section is one run.

## Run 2026-05-11 17:14:06 (sha=, baseline-eager)

### Surface 1 — Raw ttnn (tensors on device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| ttnn.add 1x32 | 162.2 | 187.5 |
| ttnn.add 32x32 | 155.1 | 184.6 |
| ttnn.exp 1x32 | 117.2 | 141.8 |
| ttnn.matmul 64x64 | 59.4 | 74.1 |
| ttnn.matmul 256x256 | 59.3 | 72.6 |

### Surface 2 — Engine eager (_execute_op_device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| engine.add 1x32 | 132.5 | 167.3 |
| engine.exp 1x32 | 76.5 | 90.5 |
| engine.matmul 64x64 | 56.8 | 86.8 |
| engine.matmul 256x256 | 55.3 | 68.9 |

### Surface (parse) — bytecode_to_text + parse_stablehlo

| op | mean (us) | p99 (us) |
|---|---:|---:|
| parse: x + 1 | 1386.5 | 1419.1 |
| parse: softmax | 1664.5 | 1701.1 |

### Surface 3 — Engine end-to-end execute_stablehlo

| op | mean (us) | p99 (us) |
|---|---:|---:|
| e2e: x + 1 (1-op) | 1986.7 | 2028.5 |
| e2e: exp(x) (1-op) | 1700.8 | 1733.8 |
| e2e: a @ b 64x64 | 1798.4 | 1876.6 |
| e2e: linear (a@w+b) | 2413.0 | 2457.3 |
| e2e: softmax (1x64) | 3360.4 | 3437.2 |

### Surface 4 — jax.jit (full PJRT pipeline)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| _error | 0.0 | 0.0 |
| _msg: INTERNAL: execute_stablehlo failed (check stderr) | 0.0 | 0.0 |

## Run 2026-05-11 17:24:20 (sha=, baseline-eager)

### Surface 1 — Raw ttnn (tensors on device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| ttnn.add 1x32 | 91.3 | 106.8 |
| ttnn.add 32x32 | 91.7 | 106.9 |
| ttnn.exp 1x32 | 73.8 | 86.4 |
| ttnn.matmul 64x64 | 42.6 | 55.1 |
| ttnn.matmul 256x256 | 44.7 | 55.1 |

### Surface 2 — Engine eager (_execute_op_device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| engine.add 1x32 | 95.4 | 111.6 |
| engine.exp 1x32 | 75.7 | 89.1 |
| engine.matmul 64x64 | 57.0 | 70.9 |
| engine.matmul 256x256 | 56.7 | 69.3 |

### Surface (parse) — bytecode_to_text + parse_stablehlo

| op | mean (us) | p99 (us) |
|---|---:|---:|
| parse: x + 1 | 1379.1 | 1404.4 |
| parse: softmax | 1656.3 | 1686.7 |

### Surface 3 — Engine end-to-end execute_stablehlo

| op | mean (us) | p99 (us) |
|---|---:|---:|
| e2e: x + 1 (1-op) | 1993.8 | 2026.7 |
| e2e: exp(x) (1-op) | 1696.5 | 1722.2 |
| e2e: a @ b 64x64 | 1786.8 | 1818.5 |
| e2e: linear (a@w+b) | 2424.2 | 2466.4 |
| e2e: softmax (1x64) | 3372.6 | 3424.7 |

### Surface 4 — jax.jit (full PJRT pipeline)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| jit: x + 1 | 2086.2 | 2124.0 |
| jit: exp(x) | 1737.2 | 1765.2 |
| jit: a @ b 64x64 | 1828.5 | 1857.5 |

## Run 2026-05-11 17:38:17 (sha=, step6-trace)

### Surface 1 — Raw ttnn (tensors on device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| ttnn.add 1x32 | 166.9 | 194.5 |
| ttnn.add 32x32 | 144.9 | 193.1 |
| ttnn.exp 1x32 | 104.5 | 126.2 |
| ttnn.matmul 64x64 | 58.2 | 72.5 |
| ttnn.matmul 256x256 | 61.4 | 80.0 |

### Surface 2 — Engine eager (_execute_op_device)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| engine.add 1x32 | 128.4 | 165.4 |
| engine.exp 1x32 | 77.1 | 89.5 |
| engine.matmul 64x64 | 57.8 | 68.5 |
| engine.matmul 256x256 | 57.1 | 66.3 |

### Surface (parse) — bytecode_to_text + parse_stablehlo

| op | mean (us) | p99 (us) |
|---|---:|---:|
| parse: x + 1 | 1372.7 | 1395.4 |
| parse: softmax | 1654.0 | 1683.7 |

### Surface 3 — Engine end-to-end (eager, no trace)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| e2e: x + 1 (1-op) | 1999.1 | 2030.6 |
| e2e: exp(x) (1-op) | 1706.5 | 1737.8 |
| e2e: a @ b 64x64 | 1794.8 | 1829.1 |
| e2e: linear (a@w+b) | 2429.9 | 2475.8 |
| e2e: softmax (1x64) | 3385.5 | 3449.9 |

### Surface 5 — Engine traced (begin/end_trace_capture)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| traced: x + 1 (1-op) | 156.0 | 168.5 |
| traced: exp(x) (1-op) | 154.8 | 168.9 |
| traced: a @ b 64x64 | 199.4 | 211.7 |
| traced: linear (a@w+b) [no-trace] | 748.5 | 770.0 |
| traced: softmax (1x64) [no-trace] | 1491.7 | 1521.6 |

### Surface 4 — jax.jit (full PJRT pipeline)

| op | mean (us) | p99 (us) |
|---|---:|---:|
| jit: x + 1 | 256.3 | 269.7 |
| jit: exp(x) | 254.7 | 271.8 |
| jit: a @ b 64x64 | 312.1 | 327.2 |
