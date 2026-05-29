# models/ — multi-model bringup demos

End-to-end on-device decode for the smaller models we brought up on the way to
Qwen3.6. Each is a standalone script. **Run from the repo root on a TT host,
device 0 only, with the prod server stopped** (it holds the chips):

```bash
make run PY=models/80_8b_diverse_qa_demo.py        # or: scripts/run_remote.sh models/<file>.py
```

| Demo | Model | Notes |
|---|---|---|
| `60_native_rope_decode.py` | Qwen2.5-0.5B | native RoPE decode (≈142 tok/s) |
| `64_llama32_1b_port.py` | Llama-3.2-1B | port + greedy decode (≈78 tok/s) |
| `66_qwen3_06b_port.py` | Qwen3-0.6B | port |
| `67_llama32_3b_port.py` | Llama-3.2-3B | port (≈34 tok/s) |
| `68_smollm3_3b_port.py` | SmolLM3-3B | port |
| `70_llama_instruct_quality.py` | Llama instruct | output-quality probe |
| `71_llama3b_instruct_quality.py` | Llama-3.2-3B instruct | output-quality probe |
| `73_llama8b_instruct.py` | Llama-3.1-8B | instruct decode (≈19 tok/s) |
| `76b_8b_correctness_check.py` | Llama-3.1-8B | correctness gate (cos > 0.997, 8/8) |
| `80_8b_diverse_qa_demo.py` | Llama-3.1-8B | 10-category Q&A demo |

The production targets (Qwen3.6-27B TP, 35B-A3B MoE) live in
`experiments/serve/`, not here. tok/s figures are recorded single-P150 numbers;
see `REPRODUCE.md` for the verified runs.
