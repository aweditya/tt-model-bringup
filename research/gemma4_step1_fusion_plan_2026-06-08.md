# Gemma 4 #289 Step 1: matmul concat-fusion plan (2026-06-08)

After #292 research locked the strategy, Step 1 is the no-precision-risk
warmup: concat-fuse the matmul pairs that share the same input (Q/K/V
share `in_norm`; gate/up share `pre_ff_norm`). Forks the established
35B `in_proj_combined` pattern (`server_35b_ttnn.py:400`).

## Reuse precedent

`experiments/serve/server_35b_ttnn.py:400`:
```python
fused = ttnn.matmul(h_tt, w["in_proj_combined"], compute_kernel_config=HIFI4)
# slices for mixed_qkv / z / a / b
```

35B numbers (per comment at line 398): `0.130 → 0.093 ms (bit-exact)` per
DN layer. About 28% reduction on that block.

## Sub-tasks

### #293 — Step 1a: sliding QKV concat-fuse (40 layers)

**Per-chip weight shapes** (sliding, NCHIPS=4):
| Weight | HF shape (post-transpose) | Per-chip after shard | Output dim |
|---|---|---|---|
| q_proj | [3840, 4096] | [3840, 1024] | NQ_PER_CHIP × HEAD_DIM_SLIDING = 4 × 256 |
| k_proj | [3840, 2048] | [3840, 512] | NKV_PER_CHIP × HEAD_DIM_SLIDING = 2 × 256 |
| v_proj | [3840, 2048] | [3840, 512] | NKV_PER_CHIP × HEAD_DIM_SLIDING = 2 × 256 |
| **qkv_combined** | — | **[3840, 2048]** | sum = 1024 + 512 + 512 |

All three are sharded along the SAME axis (output, axis=1) with the
SAME per-chip pattern. Concat is straight `np.concatenate(..., axis=1)`
on the host before sharding.

**Upload (`upload_attn_layer_sliding`)**:
```python
qkv_w = np.concatenate([q_w, k_w, v_w], axis=1)  # [3840, 2048] per chip
w["qkv_proj_combined"] = np_stacked_to_sharded(
    shard_along(qkv_w, axis=1), mesh)
# Keep q_proj/k_proj/v_proj for now (parallel-paths during smoke); env
# gate TT_GM4_FUSE_QKV=1 picks the combined path. Drop the unfused
# weights once #291 --verify is green at both paths.
```

**Forward (`_drafter_attn_sliding` and `_layer_pos0_sliding_paged`)**:
```python
if FUSE_QKV:
    qkv = ttnn.matmul(in_norm, w["qkv_proj_combined"],
                       compute_kernel_config=HIFI4)
    q = ttnn.slice(qkv, [0, 0, 0], [1, 1, NQ_PER_CHIP * HEAD_DIM_SLIDING])
    k = ttnn.slice(qkv, [0, 0, NQ_PER_CHIP * HEAD_DIM_SLIDING],
                       [1, 1, (NQ_PER_CHIP + NKV_PER_CHIP_SLIDING) * HEAD_DIM_SLIDING])
    v = ttnn.slice(qkv, [0, 0, (NQ_PER_CHIP + NKV_PER_CHIP_SLIDING) * HEAD_DIM_SLIDING],
                       [1, 1, (NQ_PER_CHIP + 2 * NKV_PER_CHIP_SLIDING) * HEAD_DIM_SLIDING])
    ttnn.deallocate(qkv)
else:
    q = ttnn.matmul(in_norm, w["q_proj"], ...)
    k = ttnn.matmul(in_norm, w["k_proj"], ...)
    v = ttnn.matmul(in_norm, w["v_proj"], ...)
```

Slicing is **0.0% device kernel time** per #289 tracy. Free.

**Global layers (8 of 48): SKIP this round.** Global has K/V REPLICATED
(NKV=1, replicated across mesh) and Q SHARDED. Different mesh
distribution per weight prevents a clean single-matmul fuse. Revisit
in v2 if needed.

### #294 — Step 1b: gate+up SwiGLU concat-fuse (all 48 layers)

**Per-chip weight shapes** (MLP, NCHIPS=4):
| Weight | HF shape (post-transpose) | Per-chip after shard | Output dim |
|---|---|---|---|
| gate_proj | [3840, 15360] | [3840, 3840] | INTERMEDIATE_PER_CHIP |
| up_proj | [3840, 15360] | [3840, 3840] | INTERMEDIATE_PER_CHIP |
| **gate_up_combined** | — | **[3840, 7680]** | 2 × INTERMEDIATE_PER_CHIP |

Both sharded the same way (axis=1). Concat is clean.

**Upload (`upload_mlp_layer`)**:
```python
gate_up_w = np.concatenate([gate_w, up_w], axis=1)  # [3840, 7680] per chip
w["gate_up_proj_combined"] = np_stacked_to_sharded(
    shard_along(gate_up_w, axis=1), mesh)
```

**Forward**:
```python
if FUSE_GATE_UP:
    gate_up = ttnn.matmul(pre_ff, w["gate_up_proj_combined"],
                            compute_kernel_config=HIFI4)
    gate = ttnn.slice(gate_up, [0, 0, 0], [1, 1, INTERMEDIATE_PER_CHIP])
    up   = ttnn.slice(gate_up, [0, 0, INTERMEDIATE_PER_CHIP],
                                [1, 1, 2 * INTERMEDIATE_PER_CHIP])
    # gelu applied as a separate unary on gate (was previously activation=gelu
    # on the gate matmul; keep behaviour equivalent by adding a ttnn.gelu after
    # the slice).
    gate = ttnn.gelu(gate)
    mid = ttnn.mul(gate, up)
    ttnn.deallocate(gate_up); ttnn.deallocate(gate); ttnn.deallocate(up)
```

**Caveat**: existing code uses `ttnn.matmul(..., activation="gelu")`
fused on gate. With the fuse, we'd run gelu as a separate op (cheap).
The activation move is a known pattern from 35B (#247) — neutral on
perf, simpler code.

## Validation gates

Each sub-task must pass BEFORE merging:

1. **#291 long-context argmax gate**:
   ```
   ssh qb1 'cd ~/tt-xla && bash scripts/run_remote.sh \
     experiments/cb/isolate/gemma4_long_context_argmax_gate.py --verify'
   ```
   ALL FOUR L (128/512/1024/2048) must reproduce baseline byte-for-byte.

2. **Tracy device-CSV delta**:
   - Pre-fuse: `MatmulDeviceOperation` count from baseline tracy
   - Post-fuse: re-run subset tracy at GM4_NUM_LAYERS_OVERRIDE=4
   - Expected matmul count delta: -2 per sliding layer (1a) and/or -1
     per layer (1b). Typecast count should drop proportionally.

3. **No regression in existing smokes**:
   - 27B v0.4 multi-step chat smoke (BASE prompt)
   - Existing IT smoke if cached

4. **Env-gated rollout**:
   - `TT_GM4_FUSE_QKV=1` for 1a (default off until --verify green)
   - `TT_GM4_FUSE_GATE_UP=1` for 1b
   - Once verified, flip defaults to ON; drop unfused weight upload
     after a deprecation grace period.

## Estimated impact

Per #289 device CSV (4-layer subset):
- Total matmul calls: 116
- Matmul fraction of device time: 16%
- Typecast device time: 38%

Each matmul matches ~3 typecasts (308/116) — likely input + output + intermediate. Fusing 2 → 1 saves ~3 typecasts. Per-layer counts:
- Sliding 1a: saves 2 matmuls × 3 typecasts = 6 typecasts per sliding layer × 40 layers = 240 typecasts saved.
- 1b: saves 1 matmul × 3 typecasts = 3 typecasts per layer × 48 layers = 144 typecasts saved.

Total per forward: ~384 typecasts saved out of ~3700 estimated total (extrapolating 308 × 12) = **~10% typecast reduction = ~3-4% wall time saved**.

Lower than the research note's "halves matmul-typecast pairs" claim
because not every typecast comes from matmul output. Still a real
win, and **the safer setup for Step 2** (selective fp32_dest_acc
disable on small-K projections) which delivers the big chunk.

## Implementation order

1. #294 (gate+up MLP fuse) FIRST — simpler (all 48 layers, no global vs sliding split)
2. #293 (sliding QKV fuse) SECOND — sliding-only, requires picking the per-layer branch
3. Run --verify after each
4. Combine both env gates default-on
5. Re-tracy to confirm matmul + typecast count drop

## Risks + mitigations

- **Activation move (gelu off matmul → separate)** can be slower if the fused matmul activation is more efficient than a separate gelu unary. Mitigate: time the smoke after #294; if slower, leave activation="gelu" on the FUSED matmul (it only acts on the gate slice — would need careful kernel review; might not be straightforward). Backup: skip 1b if it regresses.
- **bf16 vs fp32_dest_acc accumulator**: unchanged in Step 1. Same precision contract; long-context gate should reproduce byte-for-byte. If it doesn't, that's an info leak from one of the unfused matmuls' output dtype path that we need to investigate before Step 2.
- **DRAM-sharded MLP (#239 in flight)**: Step 1b changes the upload path. Coordinate with #239 — pick whichever lands first as the base.
