# Integration recipe — qwen36_decay_gate_decode_owned

Mirrors the conv1d op's flow. Run on qb2.

## Install

```bash
ssh qb2 'cd ~/tt-xla && .venv/bin/python \
    experiments/owned_ops/qwen36_decay_gate_decode_owned/integrate_into_ttmetal.py \
    --tt-metal ~/tenstorrent/tt-metal'
```

Idempotent. Anchors on the qwen36_conv1d_decode_owned entries the conv1d
installer already added.

## Build

```bash
ssh qb2 'cd ~/tenstorrent/tt-metal && cmake --build build_tracy_gcc12_nodist --target ttnn -j 24'
```

Incremental, ~3-5 minutes. The build only updates
`~/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnn.so` —
**you must manually sync the rebuilt .so into the source tree + venv**
(known gap, will be automated post-G0):

```bash
ssh qb2 'cp /home/aditya/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnn.so \
       /home/aditya/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so && \
   cp /home/aditya/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnncpp.so \
       /home/aditya/tenstorrent/tt-metal/ttnn/ttnn/_ttnncpp.so && \
   cp /home/aditya/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnn.so \
       /home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/_ttnn.cpython-310-x86_64-linux-gnu.so && \
   cp /home/aditya/tenstorrent/tt-metal/build_tracy_gcc12_nodist/ttnn/_ttnncpp.so \
       /home/aditya/tt-xla/.venv/lib/python3.10/site-packages/ttnn/_ttnncpp.so'
```

## Smoke-test (debug-fill — scaffold sanity)

Stop the resident server first (chip contention):

```bash
ssh qb2 'cd ~/tt-xla && bash experiments/serve/scripts/serve_cb.sh stop'
```

Then run the test (TT_METAL_HOME required for kernel search path):

```bash
ssh qb2 'cd ~/tt-xla && \
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \
    TT_BUILD_DIR=$TT_METAL_HOME/build_tracy_gcc12_nodist \
    ARCH_NAME=blackhole \
    PYTHONPATH=$TT_METAL_HOME/ttnn \
    LD_LIBRARY_PATH=$TT_METAL_HOME/ttnn/ttnn:$TT_BUILD_DIR/ttnn:$TT_BUILD_DIR/lib \
    .venv/bin/python \
    experiments/owned_ops/qwen36_decay_gate_decode_owned/test_qwen36_decay_gate_decode_owned.py \
    --nv 12 --debug-fill --device-id 0'
```

Expected: `decay.pass: true` (decay matches a), `beta.pass: true` (beta matches b).

## Full math test (BF16-native gate)

Same env, drop `--debug-fill`:

```bash
ssh qb2 'cd ~/tt-xla && \
    TT_METAL_HOME=$HOME/tenstorrent/tt-metal \
    ... \
    .venv/bin/python \
    experiments/owned_ops/qwen36_decay_gate_decode_owned/test_qwen36_decay_gate_decode_owned.py \
    --nv 12 --device-id 0 \
    --output-json ~/tt-xla/.cache/qb2_tp_deltanet/owned_decay_gate_g0_$(date +%Y%m%d).json'
```

Expected: PCC ≥ 0.99999 for both decay and beta; max_abs_diff ≤ 0.01.
If softplus drift is more than ~1 BF16 ULP, consider falling back from
SFPU `softplus_tile` to the manual `log(exp(x)+1)` chain inside the
kernel (per the plan's open question 1).

## Next gates after G0 passes

Per `research/decay_gate_custom_kernel_plan_2026_05_18.md`:
- G1: real-tensor probe through resident server (all 48 DeltaNet layers)
- G2: `state.deltanet_decay_gate_mode` flag + guarded trace 20-token gate
- G3: cosine_ladder_tp at 500 positions
- G4: default flip + verify

## Removal

```bash
ssh qb2 'rm -rf ~/tenstorrent/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/qwen36_decay_gate_decode_owned'
```

Then manually revert the three CMake/nanobind patches in tt-metal.
