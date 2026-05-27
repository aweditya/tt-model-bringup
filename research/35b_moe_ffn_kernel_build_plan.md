# MoE FFN kernel — build plan (G0 → G4)

Companion to `research/35b_moe_ffn_kernel_scoping.md` (architecture mapping).
This doc says HOW we build, not WHAT we build.

## Source tree

```
experiments/owned_ops/qwen36_moe_ffn_decode_owned/
├── README.md                                       (design intent + status log)
├── INTEGRATION.md                                  (how to copy into tt-metal)
├── integrate_into_ttmetal.py                       (script — see owned-GDN for template)
├── qwen36_moe_ffn_decode_owned.{hpp,cpp}           (public API)
├── qwen36_moe_ffn_decode_owned_nanobind.{hpp,cpp}  (python binding)
├── device/
│   ├── qwen36_moe_ffn_decode_owned_device_operation.{hpp,cpp}
│   ├── qwen36_moe_ffn_decode_owned_device_operation_types.hpp
│   ├── qwen36_moe_ffn_decode_owned_program_factory.{hpp,cpp}
│   └── kernels/
│       ├── compute/qwen36_moe_ffn_decode_owned.cpp
│       └── dataflow/{reader,writer}_qwen36_moe_ffn_decode_owned.cpp
├── test_qwen36_moe_ffn_decode_owned.py             (correctness gate)
└── benchmark_qwen36_moe_ffn_decode_owned.py        (microbench)

experiments/utils/
└── moe_ffn_kernel_oracle.py                         (numpy oracle)
```

## Build cadence — borrow from owned-GDN G0..G4

Each stage:
1. Implement.
2. `integrate_into_ttmetal.py` to drop into qb1's tt-metal source.
3. Rebuild `_ttnn.so` (`cmake --build build_Release --target ttnn`).
4. Run isolation test → pcc gate.
5. Run microbench → record traced ms/call.
6. Commit before moving to next stage.

### G0 — Scaffold (no math)

Just enough to register the op and return a zero output of the right shape.

API:
```python
out = ttnn.experimental.qwen36_moe_ffn_decode_owned(
    h, W1, W2, routing_weight,
    output_memory_config=None, output_tensor=None,
)
# Returns: Tensor of shape [1, HIDDEN] bf16, all zeros.
```

Source plumbing only — reader/writer/compute kernels are trivial. Verifies
build, nanobind binding, program factory dispatch. ~250 LOC across the file
tree, with all of the kernel-side `.cpp` files being ~10 LOC stubs.

Gate: `hasattr(ttnn.experimental, 'qwen36_moe_ffn_decode_owned')` and the
op runs without exception, returns zeros.

### G0a — Numpy oracle + isolation harness

In parallel with G0 since it doesn't need device builds. Lives in
`experiments/utils/moe_ffn_kernel_oracle.py` so it can be reused by both
the isolation test and any future bench. Pattern: mirror
`experiments/utils/gdn_kernel_oracle.py`.

Oracle math:
```python
def moe_ffn_numpy(h, W1, W2, routing_weight):
    # h: [HIDDEN], W1: [E, HIDDEN, 2*MOE_INTER], W2: [E, MOE_INTER, HIDDEN]
    # routing_weight: [E]; returns [HIDDEN]
    gate_up = h @ W1                          # [E, 2*MOE_INTER]
    gate = gate_up[:, :MOE_INTER]              # [E, MOE_INTER]
    up   = gate_up[:, MOE_INTER:]              # [E, MOE_INTER]
    mid  = silu(gate) * up                     # [E, MOE_INTER]
    expert_out = mid @ W2                       # [E, HIDDEN] via per-e matmul
    routed = (routing_weight[:, None] * expert_out).sum(axis=0)  # [HIDDEN]
    return routed
```

Harness: builds bf16 fixtures, uploads, calls kernel, reads back, computes
pcc + max_abs_diff against oracle. Two fixtures:
- Toy: E=2, HIDDEN=4, MOE_INTER=2 — verifies plumbing
- Prod: E=64, HIDDEN=2048, MOE_INTER=512 — measures realistic behavior

### G1 — Single-core full chain

ONE core does everything for all experts sequentially. No cross-core
reduction. Compute kernel iterates expert e in [0, E):
1. Stream W1[e] tile-row-block at a time from DRAM, do matmul into DEST,
   pack to CB_GATE_UP (resident, 2*MOE_INTER tiles).
2. In-CB: split CB_GATE_UP into gate/up halves; compute silu(gate)*up
   into CB_MID (resident, MOE_INTER tiles).
3. Stream W2[e] from DRAM, matmul mid @ W2[e] into DEST tile-by-tile,
   pack each output tile to CB_EXPERT_OUT (resident).
4. Multiply CB_EXPERT_OUT by routing_weight[e] (scalar broadcast).
5. Accumulate into CB_ROUTED_ACC.

At end: writer packs CB_ROUTED_ACC to output tensor.

L1 budget per core (in production shapes):
- CB_H: 64 tiles × ~2 KB = 4 KB
- CB_W1: 2 tiles (streamed) = ~4 KB
- CB_GATE_UP: 32 tiles = ~64 KB (full row resident)
- CB_MID: 16 tiles = ~32 KB
- CB_W2: 2 tiles (streamed) = ~4 KB
- CB_EXPERT_OUT: 64 tiles = ~128 KB
- CB_ROUTED_ACC: 64 tiles = ~128 KB
- Total ~364 KB << 1408 KB available.

Gate: pcc > 0.9999 vs oracle at toy + prod shapes. Microbench: expect SLOWER
than the Python chain because we use one core; goal is correctness, not
speed.

### G2 — Multi-core + cross-core reduce

Per-expert work split. `total_work = E_LOCAL = 64`. `split_work_to_cores`
gives 64 cores doing 1 expert each (110-core grid, 46 idle).

Each core runs the SAME logic as G1's expert loop, but for only ONE
expert. Then a cross-core reduce sums the 64 per-expert partials into
the final routed.

Cross-core reduce primitive — two options:
- **Tree reduce**: log2(64) = 6 levels. Each level halves active cores;
  losing-half cores write their CB_ROUTED to a neighbor's L1 via NoC,
  neighbor sums. Synchronize between levels via global semaphores.
- **Linear reduce on core 0**: cores 1..63 write to core 0's L1 sequentially;
  core 0 sums. Simpler, slower.

Start linear (simpler), upgrade to tree if it's the bottleneck. Memory
note `feedback_async_ccl_negative` is relevant — but this is intra-chip
NoC, not inter-chip fabric, so should be cheaper.

Gate: pcc > 0.9999 vs oracle + vs G1. Microbench: expect 64x speedup over
G1 modulo reduce overhead.

### G3 — Hybrid: mcast h + (expert, output_col) partition

Two optimizations on top of G2:
1. **Mcast h**: one reader core reads h from DRAM, multicasts to all
   compute cores' L1. Saves 64x DRAM reads of h (small but free).
2. **Finer work split**: `total_work = E_LOCAL × num_output_col_groups`.
   E.g., 64 × 2 = 128 work units (each handles half of HIDDEN output
   columns for one expert), uses all 110 cores. Each core does:
   - Stream W1[e] (same 4 MB)
   - Compute gate_up_partial (full row, since gate/up need full inner dim)
   - silu*up
   - Compute expert_out only for assigned output_cols (half = 1 MB W2 stream)
   - Scale by rw[e]
   - Cross-core reduce only over the experts assigned to this column group

Gate: pcc > 0.9999. Microbench: hope for sub-G2 ms/call. If not, G3 is
wasted effort and we stop here.

### G4 — Integration

Add `state.moe_owned_ffn = False` (default), and a branch in
`moe_forward_ttnn_pattern_a_batched` that uses the kernel when toggle is
True. The kernel replaces these lines (post the routing-weight construction):

```python
gate_up_batched = ttnn.matmul(h_3d_repeat, w["experts_gate_up_local"], ...)
gate_batched = ttnn.slice(...)
up_batched   = ttnn.slice(...)
mid_batched  = ttnn.mul(gate_batched, up_batched, [SILU])
expert_out_batched = ttnn.matmul(mid_batched, w["experts_down_local"], ...)
expert_out_2d = ttnn.reshape(expert_out_batched, [E_LOCAL, HIDDEN])
rw_1xK = ttnn.reshape(routing_weight_3d, [1, E_LOCAL])
routed_local = ttnn.matmul(rw_1xK, expert_out_2d, ...)
```

Becomes:

```python
routed_local = ttnn.experimental.qwen36_moe_ffn_decode_owned(
    h_tt,
    w["experts_gate_up_local"],
    w["experts_down_local"],
    routing_weight,
)
```

Gates: 
- Cos via `test_owned_gdn_greedy_generation.py` (Paris coherent)
- Long context via needle-haystack L=100 (`N4Y2BWLS` retrieved)
- Traced ms/tok measured via `trace_demo_full_step.py`

Promote to default if traced gain > 2 ms/tok AND both correctness gates pass.

## Tripwires — when to stop

- G1 doesn't pcc-pass the toy fixture after a day of work → kernel-writing
  approach is wrong for these shapes; pivot to lighter optimization (e.g.,
  in_proj fusion or just accept 143.6 ms/tok).
- G2 lands but kernel time is HIGHER than the current Python chain in
  trace → kernel overhead exceeds what fusion saves; ship G1 results and
  stop.
- G3 doesn't beat G2 → utilization wasn't the bottleneck; ship G2.
- G4 promotion gate (≥2 ms/tok traced) fails → keep toggle off, document
  the kernel as available-but-not-default.

## Why this scope

The owned-GDN kernel took ~5 days of work end-to-end (G0 → integration)
per the commit history. MoE FFN is more involved (two matmuls + cross-core
reduce vs GDN's one recurrence + no cross-core). Estimate: 7-10 days of
focused work for G0 → G3. The G1 tripwire keeps the risk bounded — if
single-core correctness doesn't come together quickly, we know early.
