# TT-XLA Learning Wiki

Practice-exam-style Q&A entries for learning Tenstorrent hardware and JAX/XLA backend development.

## Structure

Each wiki entry follows this format:
1. **Question** — a concrete, testable question
2. **Answer** — grounded in experiments or documentation, not hand-waving
3. **Experiment** — (when applicable) how to verify the answer on real hardware
4. **Sources** — where the information came from

## Index

### Foundations: JAX/XLA (Wiki 01-05)
- [01 — What is JAX?](01_what_is_jax.md)
- [02 — JAX Internals Explained](02_jax_internals_explained.md)
- [03 — Eager vs Compiled](03_eager_vs_compiled.md)
- [04 — Dialects, Backends, Interpretation](04_dialects_backends_interpretation.md)
- [05 — JAX Tracing Rules](05_jax_tracing_rules.md)

### Hardware Exploration (Wiki 06-10)
- [06 — Blackhole First Contact](06_blackhole_first_contact.md) — First computation on device, 29.9x vs CPU
- [07 — TT-XLA Installation Battle](07_tt_xla_installation_battle.md)
- [08 — Memory Hierarchy](08_memory_hierarchy.md) — L1 vs DRAM benchmarks
- [09 — Sharded Memory and Dispatch](09_sharded_memory_and_dispatch.md)
- [10 — Datatype Exploration](10_datatype_exploration.md) — BFP8 hits 221 TFLOPS

### Building Blocks (Wiki 11-18)
- [11 — MLP Inference](11_mlp_inference.md) — 1.69M samples/s, 5.5x vs CPU
- [12 — Trace Capture](12_trace_capture.md) — 3.23x speedup from eliminating dispatch
- [13 — PJRT Plugin Interface](13_pjrt_plugin_interface.md)
- [14 — Jaxpr Interpreter](14_jaxpr_interpreter.md) — Automated Jaxpr-to-TT-NN translation
- [15 — Reductions and Reshapes](15_reductions_and_reshapes.md)
- [16 — Attention](16_attention.md)
- [17 — Flash Attention](17_flash_attention.md) — TT-NN built-in, 4x faster than manual
- [17b — Command Queue and Cores](17_command_queue_and_cores.md)
- [18 — Extended Interpreter](18_extended_interpreter.md)

### Transformer & Compiler Infrastructure (Wiki 19-25)
- [19 — Transformer Block](19_transformer_block.md)
- [20 — tt_jax Module](20_tt_jax_module.md) — 19/19 tests passing
- [21 — Transformer Jaxpr Analysis](21_transformer_jaxpr_analysis.md)
- [22 — Broadcast Investigation](22_broadcast_investigation.md)
- [22b — PJRT Plugin Deep Dive](22_pjrt_plugin_deep_dive.md)
- [22c — Trace Capture Transformer](22_trace_capture_transformer.md)
- [23 — JAX MPS-Style Approach](23_jax_mps_style_approach.md)
- [24 — Scaling and Trace](24_scaling_and_trace.md)
- [25 — Work Partitioning](25_work_partitioning.md)

### First Real Models: GPT-2 (Wiki 26-29)
- [26 — GPT-2 on Blackhole](26_gpt2_on_blackhole.md) — First pretrained model, correct top-5
- [27 — GPT-2 Text Generation](27_gpt2_text_generation.md)
- [28 — Project Status](28_project_status.md)
- [29 — KV Cache Decode](29_kv_cache_decode.md)

### Qwen2.5-0.5B: The Optimization Journey (Wiki 30-38)
- [30 — Qwen Porting Plan](30_qwen_porting_plan.md)
- [31 — TT-Metal Open Issues](31_tt_metal_open_issues.md)
- [32 — Qwen Full Model](32_qwen_full_model.md) — 18.4 tok/s with 0.998 cosine
- [33 — Precision Analysis](33_precision_analysis.md) — HiFi4 + kernel config state leak
- [34 — Qwen KV Cache](34_qwen_kv_cache.md)
- [35 — Blackhole Status](35_blackhole_status.md)
- [36 — Optimization Journey](36_optimization_journey.md) — 1.7 to 135 tok/s, 80x speedup
- [37 — Trace Capture Deep Dive](37_trace_capture_deep_dive.md)
- [38 — Next Frontier](38_next_frontier.md)

### Hardware Deep Dives (Wiki 39-42)
- [39 — Blackhole Hardware Quirks](39_blackhole_hardware_quirks.md) — 4031x program cache, 39us dispatch
- [40 — Batch Decode Breakthrough](40_batch_decode_breakthrough.md) — 4,819 tok/s at batch=64
- [41 — Reflections](41_reflections.md) — From zero to 4,819 tok/s
- [42 — Optimization Ceiling](42_optimization_ceiling.md) — 7.1ms floor analysis

### Model Zoo Sprint (Wiki 43-45)
- [43 — Llama Port](43_llama_port.md) — Llama-3.2-1B at 78 tok/s, architecture generality
- [44 — Multi-Model Results](44_multi_model.md) — 4 models, near-linear parameter scaling
- [45 — Retrospective: Model Zoo](45_retrospective_model_zoo.md) — 5 models in one session

### Quality & Correctness Validation (Wiki 46-52)
- [46 — Quality Validation](46_quality_validation.md)
- [47 — Quality Retrospective](47_quality_retrospective.md)
- [48 — 8B Precision Analysis](48_8b_precision_analysis.md) — Cosine 0.9975, bf16 compounds over 32 layers
- [49 — Session Reflection](49_session_reflection.md) — Quality understood, correctness proven
- [50 — Sampling Investigation Complete](50_sampling_investigation_complete.md) — Model behavior, not a bug
- [51 — Quality Final Verdict](51_quality_final_verdict.md) — 8B correct at 18 tok/s
- [52 — Benchmark Audit](52_benchmark_audit.md) — from_dev readback is 33% of decode time

### Advanced Optimization & Bug Reports (Wiki 53-56)
- [53 — PJRT Plugin Test](53_pjrt_plugin_test.md) — Official plugin installs but segfaults
- [54 — Optimization Experiments](54_optimization_experiments.md) — BFP8 MLP = 1.20x on 8B, BFP4 catastrophic
- [55 — Flash Decode Blackhole Bug](55_flash_decode_blackhole_bug.md) — JIT build failure, all head configs
- [56 — Audit Report](56_audit_report.md) — All performance numbers verified clean

### Mixture of Experts (Wiki 57-59)
- [57 — MoE Feasibility](57_moe_feasibility.md) — Qwen1.5-MoE-A2.7B architecture analysis
- [58 — MoE First Light](58_moe_first_light.md) — First MoE on Blackhole, 14.3B params, exp 89-91
- [59 — MoE Session Reflection](59_moe_session_reflection.md) — 7th model, research sprint

### Journey Reflection (Wiki 60)
- [60 — Journey Reflection](60_journey_reflection.md) — Full project retrospective: 91 experiments, 7 models, 3 days

### Reference
- [Q&A: Correctness and Architecture](qa_correctness_and_architecture.md)
