# Custom (owned) compute kernels

Two kinds of custom kernel live in this repo. They install differently:

- **Owned TT-NN ops** (`experiments/owned_ops/<name>/`) — full C++ ops added to a
  tt-metal checkout. **Require a tt-metal rebuild.** Each dir is self-contained
  (`.cpp/.hpp` + nanobind, `sources.cmake`, `integrate_into_ttmetal.py`,
  `INTEGRATION.md` with the validation log, `test_*.py`). Install with the
  orchestrator below; they then appear under `ttnn.experimental.<name>`.
- **Device-kernel patches** (`experiments/kernel_patches/<name>/`) — a modified
  compute-kernel `.cpp` for an *already-installed* op. Device kernels
  **JIT-compile at runtime → NO ttnn rebuild**; you copy the patched `.cpp` over
  the one in the tt-metal source tree and it takes effect on next run. See that
  dir's README. (Today: `qwen36_gdn_decode_owned`'s `debug_mode=10` batched
  safe-output path, used by continuous batching.)

## Install (run on the TT host, after building tt-metal)

```bash
scripts/build_owned_ops.sh            # the 2 production 27B ops + rebuild ttnn
scripts/build_owned_ops.sh --all      # every op below
scripts/build_owned_ops.sh --dry-run  # show what integrate would touch
```
Env: `TT_METAL` (default `~/tenstorrent/tt-metal`), `BUILD_DIR` (default
`$TT_METAL/build_Release`). Per-op manual steps + validation gates are in each
op's `INTEGRATION.md`.

## The ops

| Op | Role |
|---|---|
| `qwen36_gdn_decode_owned` | **Production.** Fused GatedDeltaNet decode recurrence; the 27B serving path calls it (`deltanet_recurrence_mode="owned_gdn"`). |
| `qwen36_decay_gate_decode_owned` | **Production.** Fused decay/gate (+2.5% tok/s). |
| `qwen36_gdn_delta`, `qwen36_gdn_prediction`, `qwen36_gdn_decay_state`, `qwen36_gdn_outer_update`, `qwen36_gdn_output` | GDN sub-ops from the decomposed bring-up; fused into `gdn_decode_owned`. Not on the runtime path — kept for reference/microbench. |
| `qwen36_conv1d_decode_owned` | Experimental conv1d decode op (the serving path uses the shift-accumulate reformulation instead). |
| `qwen36_moe_ffn_decode_owned` | **In progress** (35B MoE FFN kernel, G0–G4). Not built into prod. |

`scripts/build_owned_ops.sh` (no args) installs only the two production ops,
since those are what the 27B server calls at runtime; `--all` matches a
fully-populated dev QuietBox (where the complete `qwen36_gdn_*` set is built).
