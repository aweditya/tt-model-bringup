# owned_gdn batched-output patch (debug_mode == 10)

`compute_qwen36_gdn_decode_owned.cpp` is the patched version of the device
compute kernel at (on qb1/qb2):

```
$TT_METAL_HOME/ttnn/cpp/ttnn/operations/experimental/transformer/
  qwen36_gdn_decode_owned/device/kernels/compute/qwen36_gdn_decode_owned.cpp
```

It lives in the tt-metal tree (not this repo), so this copy is the
version-controlled source of the change. Device compute kernels are
**JIT-compiled from the .cpp at runtime**, so updating this file on qb1 takes
effect on the next op call — **no ttnn rebuild needed**.

## What changed and why

The owned-GDN recurrence is per-slot independent and the kernel parallelizes
over `slots = state.shape[1]`. So continuous batching folds the batch into the
slots dim: reshape `[B, NV, K, V] → [1, B·NV, K, V]` and call the unmodified
op. The RECURRENCE (`H_new`) is then bit-correct at any B.

But the original production path (`debug_mode == 0`) had a latent bug at high
slot counts: it computes `out = q @ state_next` by reading **`cb_state_out`**,
which the writer ALSO pops — a dual-consumer race. Once a core processes >1
block (slots > ~24, e.g. B≥3 with NV=12), the writer pops a block's state_out
tiles before the output matmul reads them, corrupting the OUTPUT of the early
slots (the recurrence is unaffected — it never reads `q`/output). The kernel was
only ever exercised at slots=12 (B=1), so this never surfaced in production.

The patch adds a `safe_out` path (selected by `debug_mode == 10`) that routes
the output matmul through **`cb_state_next_internal`** (produced AND consumed by
the compute kernel; `add_state_to_two` packs to both internal + state_out), so
`cb_state_out` has a single consumer (the writer). This is the "duplicate
internal state_next CB" the README notes was removed for B=1 perf.

Done as an **in-loop conditional** inside the existing `else` (mode 0) branch,
NOT a duplicate mode branch — a duplicate branch pushed the kernel binary to
77024 B, over the 70656 B TENSIX kernel-config limit. The conditional adds only
a few branch instructions (`add_state_to_two` is already in the binary via modes
3/5/7/8) and fits.

**`debug_mode == 0` is byte-identical to the original** (`safe_out=false` →
`add_state_to_out` + output from `cb_state_out`), so the B=1 production path
(server_tp `deltanet_recurrence_mode="owned_gdn"`) is UNTOUCHED.

## Validation

`experiments/cb_owned_gdn_batch_isolation.py --debug-mode 10`: fold-into-slots,
per-slot cos vs numpy GatedDeltaNet, B ∈ {1,2,4,8,16,32}. All PASS
(cos(H_new) and cos(out) ≈ 0.99998). With `--debug-mode 0`: B=1,2 pass, B≥3
out fails (the original race) — confirms the diagnosis and that mode 0 is the
original behavior.

## Usage from continuous batching

CB calls owned_gdn with the folded shapes + `debug_mode=10`:
`[1, B·NV, K, V]` state, `[1, B·NV, 1, K]` q/k, `[1, B·NV, 1, V]` v,
`[1, B·NV, 1, 1]` alpha/beta, `native_io=True`, `debug_mode=10`; output
`[1, B·NV·V]` → reshape `[B, NV, V]`.
