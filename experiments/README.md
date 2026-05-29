# experiments/

Device code for Qwen3.6 bringup. **Runs on a TT host only** (`ssh qb1`/`qb2`) —
local execution of device code is forbidden (`CLAUDE.md`). The dev loop +
full repo map are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Layout

| Dir | What |
|---|---|
| `serve/` | Production servers — `server_tp.py` (27B TP, qb2 prod), `server.py` (single-chip 27B), `server_35b_ttnn.py` (35B-A3B MoE), the shared on-device kernels + loaders (`ondevice_27b.py`, `generate_27b.py`), `cb_scheduler.py`, `protocol.py`, clients. |
| `cb/` | Continuous-batching suite — `validate/`, `bench/`, `profile/`, `isolate/`, `needle.py`. |
| `owned_ops/` | Custom TT-NN ops (rebuild ttnn) — see [`owned_ops/README.md`](owned_ops/README.md). |
| `kernel_patches/` | JIT device-kernel patches (no rebuild). |
| `utils/` | Re-usable diagnostics — see `utils/README.md`. |

Loose top-level scripts are one-off benches/tests (`bench_*`, `test_*`,
`demo_qwen36_27b.py`). Retired bringup probes live in `../archive/`; the
multi-model demos (Llama/Qwen/SmolLM/8B) are in [`../models/`](../models/README.md).

## Playbooks
- Porting a new model: `../wiki/bringup_checklist.md`
- Debugging methodology: `../wiki/debugging_methodology.md`

## Adding an experiment
Permanent files only (no inline scripts); name it for what it does, not a
number; put re-usable helpers in `utils/`; document non-trivial findings in
`../wiki/` or `../research/`. Validate correctness (cosine gate) before any perf
claim. See `CONTRIBUTING.md` for the canary gate on server changes.
