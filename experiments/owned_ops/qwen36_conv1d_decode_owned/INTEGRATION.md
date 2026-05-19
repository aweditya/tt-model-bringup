# Integration recipe — qwen36_conv1d_decode_owned

Mirrors the GDN op's integration flow. Run on qb2.

## Install

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python \
    experiments/owned_ops/qwen36_conv1d_decode_owned/integrate_into_ttmetal.py \
    --tt-metal ~/tenstorrent/tt-metal'
```

The installer is idempotent — re-running it after the patches are already
applied is a no-op. It copies the op tree into
`~/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned/`
and patches:
1. `ttnn/cpp/ttnn/operations/experimental/transformer/CMakeLists.txt` —
   kernel glob, api header, private sources, unity-build exclusion.
2. `ttnn/CMakeLists.txt` — nanobind source list.
3. `ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp` —
   include + bind registration.

All three patches anchor on the equivalent qwen36_gdn_decode_owned entries
that already exist after the GDN op was installed.

## Build

```bash
ssh qb2 'cd ~/tenstorrent/tt-metal && cmake --build build_tracy_gcc12_nodist --target ttnn -j 24'
```

Incremental build; only the new op + the nanobind binding rebuild
(~3-10 minutes). After the build completes, the refreshed
`build_tracy_gcc12_nodist/ttnn/_ttnn.so` (and `_ttnncpp.so` if also rebuilt)
already lives in the place serve_tp.sh points PYTHONPATH at.

## Smoke-test (debug-fill: scaffold sanity, no real math)

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python \
    experiments/owned_ops/qwen36_conv1d_decode_owned/test_qwen36_conv1d_decode_owned.py \
    --d 32 --debug-fill --device-id 0'
```

Expected: `out.pass: true` (out matches mixed), state shifts all pass.
This verifies the op registers, the runtime args wire up, the writer's
state-shift works, and the readback path is correct — all without
depending on the math.

NOTE: do NOT run this while the resident qb2 server is up — both processes
will try to claim the chips and SIGBUS each other (HANDOFF §3). Stop the
resident server first.

## Full math test (BF16-native gate)

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python \
    experiments/owned_ops/qwen36_conv1d_decode_owned/test_qwen36_conv1d_decode_owned.py \
    --d 32 --device-id 0 \
    --output-json ~/tt-xla/.cache/qb2_tp_deltanet/owned_conv1d_32_bf16_$(date +%Y%m%d).json'
```

Expected gate: `max_abs_diff ≤ 0.0005` on `out`, `state0/1/2_next` exact
(no math, just bf16 copy). If `out` fails the gate, expand to `--d 64`
and `--d 128` shape sweep to localise; if state-shift fails, the writer
kernel has a bug, not the compute kernel.

## Next steps after G0 passes

Per `research/conv1d_custom_kernel_plan_2026_05_18.md`:
- G1: add resident-server endpoint `handle_probe_deltanet_owned_conv1d_real_tensors_tp`
  that pulls live mixed/conv_st/w_conv tiles from layer 0 and validates.
- G2: add `state.deltanet_conv1d_mode` flag + guarded trace probe with
  20-token identity gate.
- G3: cosine_ladder_tp at 500 positions vs current production.
- G4: default flip + verify.

## Removal (if needed)

```bash
ssh qb2 'rm -rf ~/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_conv1d_decode_owned'
```

Then manually revert the three CMake/nanobind patches in tt-metal. (No
automated remove script today; the patches are stable strings so they're
straightforward to `git diff` out.)
