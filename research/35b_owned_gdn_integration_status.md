# 35B GDN kernel integration — current state (2026-05-26)

## TL;DR

`ttnn.experimental.qwen36_gdn_decode_owned` is built into qb1's tt-metal, the
kernel passes its own single-chip correctness gate at 35B shape (slots=8,
key_dim=value_dim=128), and the python wrapper exists in `dn_forward_ttnn`
behind `state.dn_owned_gdn = True`. **The toggle is default False — production
path is the manual recurrence (145.1 ms/tok).**

End-to-end on (1,4) mesh fails: the recurrent state buffer does not evolve
across forward calls. Two consecutive warmups produce the same `next_id`
when manual produces different ones. The kernel is not writing back to the
state correctly on the mesh-sharded layout. Single-chip behaviour is fine,
so this is a mesh / shard interaction.

## Build state on qb1

- `~/tenstorrent/tt-metal` (commit `cf7232ab`, stock) has the qwen36 kernel
  source tree dropped in-place under
  `ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_*/`.
- Build files patched: `ttnn/CMakeLists.txt`, `ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt`, `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp`.
- Build log: `~/tt-xla/.cache/build_logs/qb1_gdn_build_retry1_*.log`.
- Original `_ttnn.so` snapshot: `~/tt-xla/.cache/ttnn_so_backups/_ttnn.so.pre_gdn_20260526_1300`.
- Fix applied at integration time: added `[[maybe_unused]]` to
  `qwen36_decay_gate_decode_owned/device/qwen36_decay_gate_decode_owned_program_factory.cpp:21`
  to satisfy `-Werror,-Wunused-const-variable` (warning policy difference
  vs qb2 where this file shipped).

## Validated

- Python import on qb1: `ttnn.experimental.qwen36_gdn_decode_owned` is exposed
  (20 qwen36 symbols visible in `nm -D _ttnn.so`).
- Single-chip kernel correctness at 35B shape via
  `experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py --slots 8 --key-dim 128 --value-dim 128`:
  pcc 0.99999918, max_abs_diff 0.000488. Clean pass over the 0.99999 gate.

## Not yet validated

- **Multi-step mesh state evolution.** The integration in
  `dn_forward_ttnn(... use_owned_gdn=True ...)` uses the 27B production
  pattern (`H_owned_in = ttnn.add(state, 0.0)` clone-then-commit, NOT the
  inplace variant). Per-chip state shape is `[1, NV_PER_CHIP=8, 128, 128]`
  bf16 (matches 27B `dn['ssm']` shape, sharded along dim 1).
- On `(1,4)` mesh, `trace_demo_full_step.py --owned-gdn` returns
  `next_id=2614` for BOTH warmup 1 and warmup 2, when manual gets
  `618`/`48106` (state-evolution visible in manual). End-to-end trace
  succeeds and eager==traced matches at 143.8 ms/tok, but the underlying
  state isn't mutating across iterations.

## Integration in tree (state.dn_owned_gdn = True path)

`server_35b_ttnn.py:dn_forward_ttnn` replaces the manual recurrence block
(15+ ops: state*g, sum(state*k), beta*(v-kv), state += k⊗δ, sum(state*q))
with one kernel call:

```
alpha = reshape(g_decay, [1, NV_PER_CHIP, 1, 1])
beta_r = reshape(beta,  [1, NV_PER_CHIP, 1, 1])
q_4d = reshape(q_rep, [1, NV_PER_CHIP, 1, 128])
k_4d = reshape(k_rep, [1, NV_PER_CHIP, 1, 128])
v_4d = reshape(v_h,   [1, NV_PER_CHIP, 1, 128])
H_owned_in = ttnn.add(recurrent_state_in, 0.0)        # CLONE
H_new, out_flat = ttnn.experimental.qwen36_gdn_decode_owned(
    H_owned_in, q_4d, k_4d, v_4d, alpha, beta_r,
    native_io=True)
core_attn_out = reshape(out_flat, [1, NV_PER_CHIP, HEAD_V_DIM])
# Shared post-block: ttnn.copy(state_new=H_new, recurrent_state_in)
```

State buffer was also moved from rank-5 `(NCHIPS, 1, NV, 128, 128)`
sharded dim 0 to rank-4 `(1, NUM_V_HEADS=32, 128, 128)` sharded dim 1 to
match the kernel's "state must be rank 4" requirement and 27B's upload
pattern. Manual path retested at 145.1 ms/tok with the new shape — no
regression.

## Hypotheses for the multi-step bug

1. **`ttnn.add(state, 0.0)` clone semantics on mesh** — the resulting tensor
   may be DRAM-interleaved while state is sharded, breaking the writer.
2. **Kernel writer is single-device-aware** — the device op declares
   `state` as the first output spec (`return {state, output}`) but the
   physical write may only land on chip 0's local slab.
3. **Padded vs logical shape mismatch** on mesh — single-device test
   uses `from_torch` directly; mesh-sharded tensors may carry padding
   metadata the kernel doesn't honor.

## Next-session debugging path

1. Add a probe that reads `_ttnn_to_numpy_perchip(recurrent_state_in)` L2
   norm before and after a single owned-GDN call to confirm mutation
   (or absence thereof) per chip.
2. Compare against single-chip eager outside the mesh wrapper using the
   same shape constants — if it works, isolate the mesh-sharding aspect.
3. Try `output_memory_config=ttnn.L1_MEMORY_CONFIG` matching 27B's exact
   call (`server_tp.py:806`).
4. If the kernel can't be made to write back on mesh, try the
   `owned_gdn_inplace` mode (pass `recurrent_state_in` directly without
   the `add(_, 0.0)` clone).

## Rollback

`state.dn_owned_gdn` defaults to False. Production is unaffected. To
fully roll back the build, run:

```
cp ~/tt-xla/.cache/ttnn_so_backups/_ttnn.so.pre_gdn_20260526_1300 \
   ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so
```
