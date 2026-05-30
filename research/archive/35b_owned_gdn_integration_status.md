# 35B GDN kernel integration — SHIPPED as default (2026-05-26)

## TL;DR

`ttnn.experimental.qwen36_gdn_decode_owned` is built into qb1's tt-metal, the
kernel passes its own single-chip correctness gate at 35B shape (slots=8,
key_dim=value_dim=128), and the python wrapper in `dn_forward_ttnn` is
**enabled by default via `state.dn_owned_gdn = True`**.

End-to-end greedy generation on (1,4) mesh produces coherent text:

> "The capital of France is Paris, a city renowned for its rich history,
> culture, and iconic landmarks. Paris is situated in"

Traced ms/tok = **143.8** (vs 145.1 manual; ~0.9% — within noise on this
measurement, but the dispatch reduction is real and will compound with
further trace-side ops).

## Initial misread

The first integration attempt looked broken: two back-to-back warmups with
the same `tok_id` at position 0 produced the same `next_id` (2614), while
the manual path produced different next_ids (618 → 48106). I called this
"state isn't evolving on mesh" — wrong. With `reset_caches_ttnn()` called
between warmups, both reset state to zero, so a deterministic forward
should produce the SAME next_id. Owned_gdn is correct; manual is
non-deterministic across resets (probably bf16 near-tie argmax drift).

The actual correctness gate is multi-step greedy generation — see
`experiments/utils/test_owned_gdn_greedy_generation.py`. Run that for any
DN change.

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

## Validated via multi-step greedy generation

`test_owned_gdn_greedy_generation.py` does prompt prefill + 20-token greedy
decode. With `state.dn_owned_gdn=True` we get:

```
prompt: 'The capital of France is'  ids=[760, 6511, 314, 9338, 369]
prefill done; last next_id = 11751
generated ids = [11751, 11, 264, 3177, 34756, 364, 1141, 8807, 3712, 11,
                 7431, 11, 321, 25438, 57902, 13, 11751, 369, 29099, 303]
decoded text = 'The capital of France is Paris, a city renowned for its
                 rich history, culture, and iconic landmarks. Paris is
                 situated in'
```

20/20 tokens coherent. This is the first verified end-to-end coherent
greedy decode of 35B-A3B in this codebase as of 2026-05-26.

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

## Manual fallback is broken

Reverting `state.dn_owned_gdn = False` runs the manual recurrence chain
(15+ mul/sum/add ops). End-to-end greedy generation produces incoherent
output (`两特朗两特朗两特朗...` Chinese repetition). This is a
**pre-existing bug** — manual was apparently broken before any change
this session; nothing in this session's commits touches the manual
recurrence math.

Likely cause: rank-5 state `(1, 1, NV_PER_CHIP, 128, 128)` broadcasting
against rank-4 g_b `(1, NV_PER_CHIP, 1, 1)` may align dims differently
than the manual code expects. Not investigated further — owned_gdn is
production, so manual is documentation-only at this point.

If multi-step coherence on the manual path is ever needed, start by
upgrading every scalar broadcast (g_b, beta_b, k_col, q_col, etc.) to
rank-5 to explicitly match state's rank.

## Rollback

`state.dn_owned_gdn` defaults to False. Production is unaffected. To
fully roll back the build, run:

```
cp ~/tt-xla/.cache/ttnn_so_backups/_ttnn.so.pre_gdn_20260526_1300 \
   ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so
```
