# Reproducing TT-XLA Experiments

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
TT-NN:          0.68.0
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
pip install -r requirements.txt
```

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

Each experiment is a self-contained Python script:

```bash
python3 experiments/80_8b_diverse_qa_demo.py
```

### Key experiments to reproduce

| Experiment | What It Tests | Expected Time |
|-----------|---------------|---------------|
| `60_traced_native_rope.py` | Qwen2.5-0.5B at 140 tok/s | ~2 min |
| `64_llama1b_port.py` | Llama-3.2-1B at 78 tok/s | ~3 min |
| `67_llama3b_port.py` | Llama-3.2-3B at 34 tok/s | ~5 min |
| `73_8b_port.py` | Llama-3.1-8B at 19 tok/s | ~8 min |
| `76b_8b_correctness_check.py` | 8B correctness validation | ~5 min |
| `80_8b_diverse_qa_demo.py` | 8B Q&A demo (10 categories) | ~8 min |

### Experiment numbering

- `01-05`: JAX/XLA fundamentals (no device needed)
- `06-20`: TT-NN basics, first models via Jaxpr interpreter
- `21-45`: Direct TT-NN path, Qwen2.5-0.5B optimization
- `46-62`: Quality validation, quantization, performance tuning
- `63-73`: Multi-model ports (1B, 3B, 8B)
- `74-80`: Quality investigation, sampling strategies
- Suffix `b`, `c`, `d`: Variants of the same experiment

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

Run `76b_8b_correctness_check.py` to verify:
- Prefill cosine similarity vs numpy float32: expect >0.997
- Token match over 8 greedy steps: expect 8/8

## Troubleshooting

**"Device not found"**: Ensure the Blackhole device is detected (`lspci | grep Tenstorrent`).

**"Out of memory"**: 8B models need ~25 GB device DRAM. Ensure no other processes are using the device.

**"Firmware version mismatch"**: Update TT firmware via `tt-smi -r 0` or follow TT-Metal docs.

**Slow model download**: Models download from HuggingFace on first run. The 8B model is ~15 GB across 4 shards.
