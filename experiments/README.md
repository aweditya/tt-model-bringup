# experiments/

Device code for Qwen3.6 bringup. **Runs on a TT host only** (`ssh qb1`/`qb2`) —
local execution of device code is forbidden (`CLAUDE.md`). The dev loop +
full repo map are in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Layout

| Dir | What |
|---|---|
| `serve/` | Production servers — `cb_api.py` + `cb_engine.py` + `cb_scheduler.py` (continuous-batching HTTP), `server_tp.py` / `server_35b_ttnn.py` / `server_gemma4_unified_ttnn.py` (per-backend forward graphs), the shared on-device kernels + loaders (`ondevice_27b.py`, `generate_27b.py`), and `protocol.py`. Pre-CB Unix-socket servers + clients live in `../archive/pre_cb_server_stack_2026-06-04/`. |
| `cb/` | Continuous-batching suite — `validate/`, `bench/`, `profile/`, `isolate/`, `needle.py`. |
| `owned_ops/` | Custom TT-NN ops (rebuild ttnn) — see [`owned_ops/README.md`](owned_ops/README.md). |
| `kernel_patches/` | JIT device-kernel patches (no rebuild). |
| `utils/` | Re-usable diagnostics — see `utils/README.md`. |

Retired bringup probes (top-level `bench_*` / `test_*` one-offs,
`demo_qwen36_27b.py`, pre-CB serve stack) live in `../archive/`. The
multi-model demos (Llama/Qwen/SmolLM/8B) are in [`../models/`](../models/README.md).

## Playbooks
- Porting a new model: `../wiki/bringup_checklist.md`
- Debugging methodology: `../wiki/debugging_methodology.md`

## Adding an experiment
Permanent files only (no inline scripts); name it for what it does, not a
number; put re-usable helpers in `utils/`; document non-trivial findings in
`../wiki/` or `../research/`. Validate correctness (cosine gate) before any perf
claim. See `CONTRIBUTING.md` for the canary gate on server changes.
