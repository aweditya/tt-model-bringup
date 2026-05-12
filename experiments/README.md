# Experiments

All experiments run on the remote Tenstorrent host (`ssh qb2` primary, `ssh qb1` fallback). Local execution of device code is forbidden — see `CLAUDE.md`.

For utility scripts (re-usable diagnostics), see `experiments/utils/README.md`.

For the playbook of HOW to port a new model, see `wiki/bringup_checklist.md`.
For the debugging methodology, see `wiki/debugging_methodology.md`.

## How to read this directory

Experiments are numbered roughly chronologically. The first digit clusters them into phases:

| Range | Phase | Topic |
|---|---|---|
| `01-29` | Foundations | JAX internals, first Tenstorrent device contact, trace capture, jaxpr interpreter |
| `30-45` | First model ports | GPT-2 + Qwen-small bringup, KV cache, decode loops |
| `46-65` | Quality + optimization | HiFi4, mixed precision, batch decode, multi-model validation |
| `66-89` | Scale + production | Llama 1B/3B/8B/SmolLM3 ports, instruct quality validation, MoE block, multichip primitives |
| `90-99` | MoE optimization | Fused ops, multi-CQ, DRAM sharding, profile passes |
| `91*` (with suffix) | **Branch III: Qwen3.6-27B port** | DeltaNet + Gated Attention + MLP; this is the current active line |
| `jax_qwen05b_*` | PJRT plugin investigation | StableHLO → ttnn bringup for the custom JAX backend |

When in doubt about which experiment is current, look at recent commits in `git log` for `experiments/`.

## Branch III (active development, 2026-05-12 onward)

The Qwen3.6-27B single-chip port. Experiments numbered `91X_*` where X is a-t:

| File | Phase | Purpose |
|---|---|---|
| `91_qwen36_27b_weight_skeleton.py` | B'1 | Memory budget + layer-type pattern verification |
| `91b_qwen36_27b_numpy_ref.py` | B'2 | Pure-numpy fp32 reference for first 2 layers (deprecated — see `wiki/seven_bugs_case_studies.md` for why) |
| `91c-91e_qwen36_27b_*.py` | B'3-B'5 | Layer-by-layer ttnn bringup; cosines against numpy ref (false-positive validation, see seven_bugs) |
| `91f_qwen36_27b_full_ondevice.py` | B'6 | **Production kernels.** `deltanet_step_ondevice`, `gated_attn_step_ondevice`, `mlp_step_ondevice`. All 7 bug fixes live here. |
| `91g_qwen36_27b_full_model.py` | B'7 | Full 64-layer forward (correctness check) |
| `91h_qwen36_27b_generate.py` | B'8 | First end-to-end greedy generation; revealed the FR fixed-point |
| `91i_shape_preflight.py` | B'8.5 | Validate shapes in 30s before paying 10-min weight load |
| `91j_decode_diagnostics.py` | B'8.5 | Hidden-state norm capture across decode steps |
| `91k_fp32_api_probe.py` | B'9 prep | Verify ttnn rms_norm/linear/add accept fp32 |
| `91l_fp32_residual_generate.py` | B'9 | **Production generation script.** fp32 residual stream + all 7 bug fixes. |
| `91n_lm_head_inspection.py` | B'9 | lm_head + embed stats audit |
| `91o_hf_reference_layer0.py` | B'9.5 | First HF oracle: layer 0 cosine vs HF |
| `91p_ttnn_layer0_vs_hf.py` | B'9.5 | Layer-0 ttnn validation with `--weight-dtype` CLI |
| `91q_ttnn_layer0_substep_dump.py` | B'9.5 | Minimal substep dump (outer boundaries only) |
| `91r_per_layer_diff.py` | B'9.5 | **Per-layer cosine vs HF.** The diagnostic that found bug #7. |
| `91s_layer2_full_substep_dump.py` | B'9.5 | Full intermediate capture for layer 2 |
| `91t_layer2_substep_compare.py` | B'9.5 | Substep-by-substep diff against HF |

To run the production generation:
```bash
cd ~/tt-xla
HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
    experiments/91l_fp32_residual_generate.py --tokens 60
```

## Older models (still useful for reference)

- GPT-2 path: `30-35_gpt2_*.py` (full bringup including KV cache)
- Llama 1B/3B/8B + SmolLM3: `64-80_*.py`
- Qwen2.5-0.5B: `36-49_qwen_*.py`
- MoE block + larger MoE: `82-99_moe_*.py`

## Guidelines for adding new experiments

- Each experiment tests a specific hypothesis. Name reveals the hypothesis.
- Run on remote, never locally for device code.
- Document the hypothesis, method, and result inline (docstring at top of file).
- If an experiment grows beyond ~200 lines, split it.
- If you ran the same diagnostic three times, promote it to `experiments/utils/`.
