# MoE FFN kernel — perf deferrals (track-as-we-build)

Running list of correctness-first shortcuts in the qwen36_moe_ffn_decode_owned
kernel. Each row is a place where we picked the simpler/safer option for
faster bring-up; the right column is the perf-window for revisiting once
the kernel is correct.

| ID | Stage | Shortcut | Perf cost (est.) | Revisit lever |
|---|---|---|---|---|
| D-G0-01 | G0 | Single core only (1 of 110 Tensix cores) | ~99% of compute capacity idle | Move to G2 multi-core split; one core per expert. |
| D-G0-02 | G0 | bf16 only (no fp32 path) | Locks downstream to bf16 KV/state; no fallback for long-context fp32 experimentation | Add fp32 branch in validate + compute kernel once correctness is stable. |
| D-G0-03 | G0 | Single device (no mesh sharding) | Can't run on (1,4) production mesh | G4 mesh adapter + ShardTensorToMesh-aware accessors. |
| D-G0-04 | G0 | Reader streams h but compute discards the data | Wasted DRAM read of h (4 KB per call) | Trivial cleanup in G1 once we actually USE h. |
| D-G0-05 | G0 | Output emitted as zero tiles via `tile_regs_acquire` + immediate pack (DST register assumed zero on acquire) | Correctness risk if DST isn't zero on acquire on this build | Replace with explicit zero-init or copy_tile from a zero-fill CB in G1. |
| D-G0-06 | G0 | No CB sizing tuning — all CBs are double-buffered (depth 2), no analysis of which want depth=2 vs key_tiles*2 | Possibly some bubble cycles in the pipeline | Profile via tracy after G1 lands; tune deep buffers for the matmul streams. |
| D-G0-07 | G0 | HiFi4 + fp32_dest_acc_en hardcoded in program_factory | Matches production matmul fidelity (correct for production); can't experiment with HiFi2 without rebuilds | Plumb compute_kernel_config through the op as the friend-repo `qwen36_gdn_decode` does. |
| D-G0-08 | G0 | Compute kernel runs the same loop regardless of `debug_fill` | No actual debug-fill behavior yet | Wire debug_fill = copy h's first tile to output once G1 reads h for real. |

(Rows will be added as G1, G2, G3 land. Use this doc as input to a future
"perf cleanup pass" session once the kernel is correct.)

## Why we deferred each

- **D-G0-01 / D-G0-03 / D-G0-04**: G0 is a build/plumbing check. Real
  multi-core + mesh work needs the device op's validate logic to handle
  sharded tensors and the program factory to call `split_work_to_cores`.
  Adding both at G0 obscures whether the basic dispatch path works.
- **D-G0-02**: bf16-only matches every existing owned kernel (decay_gate,
  conv1d_owned, gdn_owned). Adding fp32 doubles the test matrix.
- **D-G0-05**: This is the only correctness risk in G0. If DST isn't zero
  on acquire, our G0 output is garbage and we get false smoke-test pass
  if the harness happens to compare to zeros and the garbage is also zeros
  due to allocator state. Mitigation: G0 smoke test should fill the output
  buffer with NON-ZERO sentinel before the kernel call, then assert all
  zeros after.
- **D-G0-06**: CB depth tuning is profile-driven; pointless before the
  pipeline does meaningful work.
- **D-G0-07**: Production correctness is the priority; matching 91f's
  HiFi4 + fp32_dest_acc keeps us on the validated numerics path.
- **D-G0-08**: G0 doesn't read h's data so debug_fill has nothing to do.

## Format for new entries

```
| D-Gn-NN | Gn | <one-sentence shortcut> | <perf-cost estimate> | <revisit plan> |
```
