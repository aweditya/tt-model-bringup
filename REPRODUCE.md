# Reproducing tt-model-bringup

## Hardware

- Tenstorrent Blackhole P150 (or compatible Blackhole device).
- For 27B / 35B: 4× P150 in a (1, 4) mesh; both `qb1` and `qb2` have working
  `FABRIC_1D`.
- For the legacy 8B-era demos: single P150, ≥ 64 GB system RAM.

## Tested environment

Tenstorrent QuietBox (4× Blackhole P150), firmware 19.6.0.

| Component | Version |
|---|---|
| Python | 3.10.12 |
| TT-NN | 0.69.0 |
| TT-SMI | 5.0.0 |
| KMD | 2.8.0 |
| PyTorch | 2.11.0 (qb1 prod) / 2.12.0 (fresh `uv sync`) |
| NumPy | 1.26.4 |
| Safetensors | 0.7.0 |
| HuggingFace Hub | 1.10.1 |
| Transformers | 5.5.3 |
| tt-metal SHA | see [`tt-metal-sha.txt`](tt-metal-sha.txt) |

## Setup

```bash
git clone https://github.com/aweditya/tt-model-bringup.git ~/tt-xla && cd ~/tt-xla
make setup            # uv sync (legacy: pip install -r requirements.txt)
make install-ttnn     # editable ttnn from $TT_METAL_HOME (on the TT host)
make check            # sanity-check setup (no device open)
make kernels          # build the owned_ops custom kernels
```

Set `HF_TOKEN` (or `hf auth login`) for HuggingFace model access. See
README §Setup for the full env block.

Device-access smoke (one-liner; useful when debugging TT-NN install):

```bash
python3 -c "import ttnn; d = ttnn.open_device(0); print('OK:', d.compute_with_storage_grid_size()); ttnn.close_device(d)"
```

## Reproduce — chat server (Qwen3.6-27B CB, canonical chat path)

On the TT host:

```bash
bash experiments/serve/scripts/serve_cb.sh start   # ~6 min bootstrap; /health → 503 until ready
bash experiments/serve/scripts/serve_cb.sh status
```

Once `/health` returns 200, hit the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hi! What can you do?"}],"max_tokens":200}'
```

See README §"Chat server (production)" for streaming, sampling, `/metrics`,
and load-test (`experiments/cb/load/concurrent_chat.py`) examples.

P5 SLO (qb1, 8 clients × 60 s, 2026-05-30): 0 errors / 36 requests /
15 tok/s aggregate / TTFT p99 = 176 ms.

## Reproduce — legacy pre-CB single-stream servers (ARCHIVED)

The pre-CB single-stream Unix-socket servers (`server.py`, `server_tp.py`
wrapper, `server_35b.py`) and their launch scripts (`serve.sh`,
`serve_tp.sh`, `serve_35b.sh`) were moved to
`archive/pre_cb_server_stack_2026-06-04/` on 2026-06-04. They were
superseded by the continuous-batching HTTP server (`serve_cb.sh` above)
and are kept for historical reference only. Steady-state numbers at
retirement: single-seq TP 12.93 tok/s on qb2, single-chip 5.14 tok/s on qb1.

## Reproduce — legacy multi-model demos (single P150)

The prod server holds device 0 — stop it first.

```bash
bash experiments/serve/scripts/serve_cb.sh stop
make run PY=models/80_8b_diverse_qa_demo.py        # or scripts/run_remote.sh models/<file>.py
bash experiments/serve/scripts/serve_cb.sh start   # restart after
```

| Demo | What it tests | Expected | qb1 (2026-05-21) |
|---|---|---|---|
| `models/60_native_rope_decode.py` | Qwen2.5-0.5B native RoPE decode | 140 tok/s | **142.2 tok/s** |
| `models/64_llama32_1b_port.py` | Llama-3.2-1B greedy decode | 78 tok/s | **78.6 tok/s** |
| `models/67_llama32_3b_port.py` | Llama-3.2-3B (Unsloth shard mirror) | 34 tok/s | **33.7 tok/s** |
| `models/73_llama8b_instruct.py` | Llama-3.1-8B instruct decode | 19 tok/s | **19 tok/s** |
| `models/76b_8b_correctness_check.py` | 8B correctness vs numpy fp32 | cos > 0.997, 8/8 | **cos 0.997327, 8/8** |
| `models/80_8b_diverse_qa_demo.py` | 8B 10-category Q&A | 18 tok/s, 9/10 EOS | **18 tok/s, 9/10 EOS** |

Fresh-clone validation (qb2, 2026-05-22): all six PASS within 2-7 % of baseline;
Demo A `client generate` 4.01 tok/s cold, Demo B `client_tp generate_tp` 13.01 tok/s.

## Reproduce — owned_ops kernel gates

Each owned op ships an `INTEGRATION.md` with the validation gate and a
`test_*.py`. Two production ops (BF16 ladder vs CPU oracle):

| Kernel | Role | Gate |
|---|---|---|
| `qwen36_gdn_decode_owned` | Production fused GatedDeltaNet decode recurrence | state/out PCC > 0.9999 |
| `qwen36_decay_gate_decode_owned` | Production fused decay/gate (+2.5 % tok/s) | PCC > 0.9999 |

Run a gate (stop the prod server so device 0 is free):

```bash
scripts/run_remote.sh experiments/owned_ops/qwen36_gdn_decode_owned/test_qwen36_gdn_decode_owned.py \
  --device-id 0 --key-dim 128 --value-dim 128 --max-abs-diff-threshold 0.001
```

See [`experiments/owned_ops/README.md`](experiments/owned_ops/README.md) for the
full op index.

## Device configuration

All single-P150 experiments use device 0 only.
Compute kernel config:

```python
hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
)
```

This is HiFi4 + `fp32_dest_acc_en` on every matmul (the 91f recipe). Mixing
fidelities corrupts ops silently on Blackhole.

## Verifying results

- **Performance** — decode speed should be within 10 % of reported numbers
  (system load, TT-NN version drift, thermal throttling).
- **Correctness** — `models/76b_8b_correctness_check.py` expects prefill cosine
  similarity vs numpy fp32 > 0.997 and 8 / 8 token match over 8 greedy steps.

## Troubleshooting

- **Device not found** — confirm with `lspci | grep Tenstorrent`.
- **Out of memory** — 8B models need ~25 GB device DRAM; no other processes on
  the device.
- **Firmware mismatch** — `tt-smi -r 0` or follow TT-Metal docs.
- **Slow model download** — first run downloads weights from HuggingFace
  (8B ~15 GB across 4 shards; 27B cold ~6 min on qb2).
- **Server hangs / fabric wedged after a hard-kill** — `tt-smi -r 0,1,2,3` then
  restart. Always prefer the script's `stop` over `kill -9`.
