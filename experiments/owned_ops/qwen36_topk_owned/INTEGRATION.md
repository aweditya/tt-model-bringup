# qwen36_topk_owned integration notes

## Install

On qb1:

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_topk_owned/integrate_into_ttmetal.py \
    --tt-metal ~/tenstorrent/tt-metal \
    --source-dir ~/tt-xla/experiments/owned_ops/qwen36_topk_owned
```

Dry-run (prints intended copies + patches, writes nothing):

```bash
python3 ~/tt-xla/experiments/owned_ops/qwen36_topk_owned/integrate_into_ttmetal.py \
    --tt-metal ~/tenstorrent/tt-metal --dry-run
```

Build and refresh source-package extensions:

```bash
cmake --build ~/tenstorrent/tt-metal/build_Release --target ttnn -j8
cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnn.so    ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnn.so
cp ~/tenstorrent/tt-metal/build_Release/ttnn/_ttnncpp.so ~/tenstorrent/tt-metal/ttnn/ttnn/_ttnncpp.so
```

When validating against the rebuilt source tree, set:

```bash
export PYTHONPATH=~/tenstorrent/tt-metal/ttnn
```

Without this, the local virtualenv wheel can shadow the rebuilt source-package
extension and the experimental symbol will not be visible.

## Validation gate (next session)

The validation probe at
`experiments/cb/isolate/qwen36_topk_owned_probe.py` compares
`ttnn.experimental.qwen36_topk_owned` against `numpy.argpartition` (lowest-idx
tie-break) on the Nemotron-3 MoE router score distribution (seed=99).
Pass criteria:
- per-row idx-set match: 8/8 rows
- weights cos vs numpy >= 0.9999
- determinism: 10/10 identical-bit indices across consecutive calls

Run:

```bash
PYTHONPATH=~/tenstorrent/tt-metal/ttnn \
  ~/tt-xla/.venv/bin/python ~/tt-xla/experiments/cb/isolate/qwen36_topk_owned_probe.py \
    --device-id 0 --seed 99 --summary-json ~/tt-xla/.cache/qb1/qwen36_topk_owned_probe.json
```

## Known build risks (resolve at first cmake run)

- `bind_function<"qwen36_topk_owned", "ttnn.experimental.">` — the
  `bind_function` template is the same one used by Mamba2 owned op; should
  Just Work, but if `_ttnn.so` won't link the binding symbol, double-check
  the function signature in `ttnn-nanobind/bind_function.hpp` between qb1
  HEAD and the GDN-owned commit.
- `ckernel::topk_local_sort</*stable_sort=*/true>` — both LLK header files
  (`tt_metal/hw/inc/api/compute/compute_kernel_api.h:513` and the LLK SFPU
  ckernel) already expose the template parameter, default false; the
  explicit `<true>` should compile cleanly. If a SFPU build error mentions
  a missing `STABLE_SORT` specialisation, fall back to checking the
  `tt-llk/{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_topk.h`
  templates for that arch.
