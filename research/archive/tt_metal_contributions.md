# TT-Metal Open Source Contribution Opportunities

## Novel Findings (Not Reported Upstream)

### 1. Kernel Config State Leak on Blackhole (HIGH PRIORITY)
**Bug:** Applying `WormholeComputeKernelConfig(HiFi4, fp32_dest_acc_en=True)` to one op (e.g., SDPA) but leaving subsequent ops at default causes corruption. Cosine drops from 0.999 to 0.873 at layer 3. The leak is directional: HiFi4→default corrupts; default→HiFi4 does not.

**Evidence:** Experiments 46b-d with clear repro. Wiki 35 documents this in detail.

**Workaround:** Apply same config to ALL ops (ALL-or-nothing).

**Status:** No existing issue or PR. Should file.

### 2. ttnn.split Fails with Tile-Padded Tensors
**Bug:** `ttnn.split(tensor_of_shape_1_14_1_64, 2, dim=-1)` fails because tile padding inflates to `(1, 14, 32, 64)` and reshape to `(1, 14, 1, 32)` can't handle the element count mismatch.

**Workaround:** Rotation matrix trick for RoPE (exp 51b).

**Status:** No existing issue. Should file.

## Related Existing Issues

- **#12330** — Flash Attention GQA sharded output not supported. We hit this with Qwen (14Q/2KV).
- **#28807** — `ttnn.to_memory_config` crashes with nd shard spec. Same class as our L1_HEIGHT_SHARDED_MEMORY_CONFIG crash.
- **#25503** — Linear defaults to small core grid on Blackhole P150 (22-24 cores instead of 88+).
- **#41827** — MoE Blackhole support being actively worked on.

## Recent Relevant PRs

- **#41806** (Apr 18, 2026) — Removed architecture asserts from compute kernel config. Validates our WormholeComputeKernelConfig usage.
- **#41790** (Apr 18, 2026) — Fixed SDPA write barriers and multicast destination count. Check if firmware 19.6.0 includes this.

## Recommended Actions

1. File kernel config state leak bug (novel, high-impact)
2. File ttnn.split tile padding bug (clear repro)
3. Comment on #12330 and #28807 with Blackhole findings
4. Check if CB config dispatch corruption fix resolves our kernel config leak
