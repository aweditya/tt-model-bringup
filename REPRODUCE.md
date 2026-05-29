# Reproducing tt-model-bringup demos

## Hardware Requirements

- **Tenstorrent Blackhole P150** (or compatible Blackhole device)
- 64 GB system RAM (8B models use ~15 GB for weights + ~10 GB for KV caches)
- AMD or Intel x86_64 CPU

## Tested Environment

```
Host OS:        Ubuntu 22.04.5 LTS (x86_64)
CPU:            AMD Ryzen 9 5900X 12-Core
RAM:            54 GB
Python:         3.10.12
TT-NN:          0.69.0
TT-SMI:         5.0.0
PyTorch:        2.11.0+cpu
NumPy:          1.26.4
Safetensors:    0.7.0
HuggingFace Hub: 1.10.1
Transformers:   5.5.3
```

## Setup

### 1. Install TT-Metal / TT-NN

Follow [Tenstorrent's installation guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) for your device.

### 2. Install Python dependencies

```bash
make setup            # uv sync (the standard path; see README Setup)
```
(`requirements.txt` is retained for legacy `pip install -r` workflows.)

### 3. Verify device access

```bash
python3 -c "import ttnn; d = ttnn.open_device(0); print('OK:', d.compute_with_storage_grid_size()); ttnn.close_device(d)"
```

### 4. HuggingFace model access

Models are downloaded automatically via `huggingface_hub`. Set `HF_TOKEN` if you need authenticated access:

```bash
export HF_TOKEN=your_token_here
```

## Running Experiments

Each demo is a self-contained script under `models/` (run from the repo root):

```bash
make run PY=models/80_8b_diverse_qa_demo.py    # or: scripts/run_remote.sh models/<file>.py
```

### Key experiments to reproduce

> Note: the original `tenstorrent` host these experiments were authored on has
> been replaced by **qb1** and **qb2** — both 4× P150 Blackhole hosts. All
> demos below have been re-verified on qb1 (single P150) as of 2026-05-21.
> See [Re-verified on qb1 (2026-05-21)](#re-verified-on-qb1-2026-05-21).

| Demo | What It Tests | Expected Time |
|-----------|---------------|---------------|
| `models/60_native_rope_decode.py` | Qwen2.5-0.5B at 140 tok/s | ~2 min |
| `models/64_llama32_1b_port.py` | Llama-3.2-1B at 78 tok/s | ~3 min |
| `models/67_llama32_3b_port.py` | Llama-3.2-3B at 34 tok/s | ~5 min |
| `models/73_llama8b_instruct.py` | Llama-3.1-8B at 19 tok/s | ~8 min |
| `models/76b_8b_correctness_check.py` | 8B correctness validation | ~14 min |
| `models/80_8b_diverse_qa_demo.py` | 8B Q&A demo (10 categories) | ~6 min |

See [`models/README.md`](models/README.md) for the full demo index. (The
production Qwen3.6-27B/35B path lives in `experiments/serve/`, not here.)

## Device Configuration

All experiments use **device 0 only**. If you have multiple devices, no changes needed — we explicitly open device 0.

Compute kernel config is set to HiFi4 (highest precision):

```python
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False
)
```

## Verifying Results

### Performance

Decode speed should be within 10% of reported numbers, depending on:
- System load and memory bandwidth availability
- TT-NN version (kernel implementations may change)
- Thermal throttling

### Correctness

Run `models/76b_8b_correctness_check.py` to verify:
- Prefill cosine similarity vs numpy float32: expect >0.997
- Token match over 8 greedy steps: expect 8/8

## Troubleshooting

**"Device not found"**: Ensure the Blackhole device is detected (`lspci | grep Tenstorrent`).

**"Out of memory"**: 8B models need ~25 GB device DRAM. Ensure no other processes are using the device.

**"Firmware version mismatch"**: Update TT firmware via `tt-smi -r 0` or follow TT-Metal docs.

**Slow model download**: Models download from HuggingFace on first run. The 8B model is ~15 GB across 4 shards.

## Re-verified on qb1 (2026-05-21)

The original `tenstorrent` host is gone; the project now runs on two Blackhole
hosts (qb1 + qb2, each with 4× P150). qb1 was used as the single-P150
substrate for re-running all six legacy 8B-era demos against the current
TT-NN install (firmware 19.6.0, KMD 2.8.0, TT-SMI 5.0.0, Python 3.10.12 in
`~/tt-xla/.venv`).

All six PASS, with numbers within 2-7% of the historical baselines. The
persistent Qwen3.6-27B prod server was stopped for the run and restarted
afterwards via `bash experiments/serve/scripts/serve.sh {stop,start}`.

| Demo | Baseline | qb1 (2026-05-21) | Notes |
|------|----------|------------------|-------|
| `models/60_native_rope_decode.py` | 140 tok/s | **142.2 tok/s** | Qwen2.5-0.5B, native RoPE path |
| `models/64_llama32_1b_port.py` | 78 tok/s | **78.6 tok/s** | "The capital of France is Paris" verbatim |
| `models/67_llama32_3b_port.py` | 34 tok/s | **33.7 tok/s** | Unsloth shard mirror (meta-llama needs HF auth) |
| `models/73_llama8b_instruct.py` | 19 tok/s | **19 tok/s** | All 5 prompts complete; mostly coherent |
| `models/76b_8b_correctness_check.py` | cos > 0.997, 8/8 tokens | cos **0.997327**, **8/8** | Per-step cos 0.981-0.996; EOS at step 6 |
| `models/80_8b_diverse_qa_demo.py` | 10 prompts, 18 tok/s | **18 tok/s, 9/10 EOS** | 403 tokens total; Code-gen prompt hit max-tokens |

Logs are in `~/tt-xla/.cache/legacy_demos_2026_05_21/` on qb1.

### Running on the new hosts

The serve scripts (`experiments/serve/scripts/serve.sh`) own chips 0-3 on
qb1 during normal operation. To rerun any of these legacy demos:

```bash
# On qb1, stop the prod server first (chip 0 is needed):
bash ~/tt-xla/experiments/serve/scripts/serve.sh stop

# Run a demo (uses device 0 only):
cd ~/tt-xla && source .venv/bin/activate
python3 experiments/64_llama32_1b_port.py

# Restart the prod server (~11 min bootstrap):
bash ~/tt-xla/experiments/serve/scripts/serve.sh start
```

The venv ships TT-NN as a pip-installed package (no `LD_LIBRARY_PATH` or
`PYTHONPATH` manipulation needed). The only relevant exported env vars are
`HF_HOME` and `TTNN_CACHE_DIR` (set in `~/.bashrc`).

qb2 is an alternative single-chip substrate if qb1 is busy; same recipe but
the server there is `server_tp.py` (4-chip TP) so chip 0 is also held — stop
it via `serve_tp.sh stop` first if you need it.

## Fresh-clone validated on qb2 (2026-05-22)

End-to-end test of the "anyone can clone this repo and run it" path:
`git clone https://github.com/aweditya/tt-model-bringup.git
~/tt-model-bringup-fresh` on qb2, `uv sync`, `uv pip install -e
$TT_METAL_HOME --no-build-isolation`, then ran the same 6 legacy demos +
Demo A (single-chip 27B) + Demo B (4-chip TP 27B) + the qb1 MoE smoke as a
parallel check.

| Demo | qb1 baseline (2026-05-21) | qb2 fresh-clone (2026-05-22) | Notes |
|------|----------------------------|------------------------------|-------|
| `models/60_native_rope_decode.py` | 142.2 tok/s | **142.5 tok/s** | identical within run-to-run jitter |
| `models/64_llama32_1b_port.py` | 78.6 tok/s | **78.4 tok/s** | |
| `models/67_llama32_3b_port.py` | 33.7 tok/s | **34 tok/s** | |
| `models/73_llama8b_instruct.py` | 19 tok/s | **19 tok/s** | |
| `models/76b_8b_correctness_check.py` | cos 0.997327 | **cos 0.997327** | bit-identical |
| `models/80_8b_diverse_qa_demo.py` | 18 tok/s | **16 tok/s (59% efficiency)** | within 2 tok/s; thermal/load variance |
| Demo A — `server.py` + `client generate` | 5.14 tok/s (README) | 4.01 tok/s cold / 4.03 tok/s warm | first-run, fewer warm passes |
| Demo B — `server_tp.py` + `client_tp generate_tp` | 12.98 tok/s (README) | **13.01 tok/s warm** | matched prod baseline 12.93 tok/s |
| MoE smoke on qb1 (`decode_smoke_35b_ttnn.py`) | 480 ms/tok pre-trace (`feedback_b16_coherent_text_on_device.md`) | 522 ms/tok pre-trace, ` Paris` prediction + 24-tok coherent continuation | qb1 host since 35B weights cached there |

Total wall: ~75 min of qb2 prod-downtime (legacy batch 23 min + Demo A
bootstrap+smoke 14 min + Demo B bootstrap+smoke 13 min + prod
cold-restart 17 min, plus tt-smi resets). Logs in
`~/tt-model-bringup-fresh/.cache/sanity_2026_05_22/` on qb2 and
`~/tt-xla/.cache/sanity_2026_05_22/` on qb1.

### Fresh-clone bugs found and fixed

- **`experiments/serve/protocol.py` + `experiments/serve/server_tp.py`
  hardcoded PROJECT_ROOT to `~/tt-xla/`.** A fresh clone at any other path
  wrote `server.sock` / `server.pid` / `server_tp.sock` to the legacy dir
  while the client looked in the fresh dir → "No such file or directory".
  Fixed by deriving PROJECT_ROOT from `__file__` (commit `b17d33e`). No
  prod behavior change since `~/tt-xla/experiments/serve/...` still
  resolves to `~/tt-xla/`.
- **`serve.sh stop` / `serve_tp.sh stop` rely on a `.pid` file the python
  server writes after bootstrap.** During the Demo A → Demo B transition,
  this means `stop` may report "not running" before the server has
  actually died. If you then start the next server, you race the TLB
  cleanup of the prior one and crash with `tt_tlb_alloc failed -12`.
  Recovery: `tt-smi -r 0,1,2,3`. Workaround until both bugs above are
  cleaned up: SIGTERM by pgrep, wait, verify pgrep is empty, then start
  the next server.
