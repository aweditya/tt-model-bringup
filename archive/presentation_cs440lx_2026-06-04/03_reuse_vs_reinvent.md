# Reuse vs reinvent: standing on tt-metal's shoulders

The bringup is dominated, by line count, by *forks* of Tenstorrent's open-source code — but the parts that were genuinely original are the ones that gated correctness. Here's the split.

## A. What we reused directly (forked TT-team code, almost as-is)

- **Paged SDPA decode kernel.** `ttnn.transformer.paged_scaled_dot_product_attention_decode` is called as-is in `experiments/serve/server_tp.py:1570`, `server_35b_ttnn.py:869`, and both Gemma 4 sliding/global attention paths in `server_gemma4_unified_ttnn.py`. The Gemma 4 global-attention `SDPAProgramConfig` (q_chunk_size=32, k_chunk_size=64 to fit head_dim=512 in 1.5 MB L1) is lifted from `experiments/.refs/tt-metal/models/demos/gemma4/tt/attention/decode.py:126-160` and cited inline at `server_gemma4_unified_ttnn.py:526-527`. This swap alone was a **+62%** win on 27B TP (`feedback_paged_sdpa_shipped_tp.md`, commit `4741253`).
- **All-reduce / all-gather CCLs.** `ttnn.all_reduce(cluster_axis=1)` and `ttnn.all_gather(dim=-1)` carry the residual stream and the vocab-gather. The `num_links=2 + Topology.Ring` config is forked from the Llama70B-Galaxy production path; commit `d739daf` ships it for +1.65% tok/s.
- **Llama70B-Galaxy MLP pattern.** `mlp_step_tp` (`server_tp.py:1432-1455`) cites it explicitly: *"Llama70B production pattern (llama3_70b_galaxy/tt/llama_mlp.py:141-193)"* — aggressive `ttnn.deallocate` after every intermediate's last use, learned the hard way (without these, the 3rd forward wedges).
- **Tile / mesh primitives.** `ShardTensorToMesh(mesh, dim=0)`, `ReplicateTensorToMesh`, `TILE_LAYOUT`, `bfloat8_b` dtype — straight TT-NN. The `np_stacked_to_sharded` / `np_to_replicated` helpers (`server_tp.py:110`, `server_35b_ttnn.py:96`) follow TT model-params conventions for shard axis (col-sharded matmul + all_reduce).
- **`paged_update_cache`, `paged_fused_update_cache`, RMSNorm primitives.** Direct calls; 20/24 fused-op candidates we audited already existed in TT-NN (`feedback_ttnn_fused_ops_gap_analysis.md`).
- **Owned-op packaging.** `experiments/owned_ops/<op>/device/` mirrors tt-metal's `ttnn/cpp/ttnn/operations/experimental/transformer/<op>/` skeleton (device-op + program-factory + reader/writer/compute dataflow kernels + nanobind). The directory layout, CMake wiring, and nanobind shape are stamped from `experiments/.refs/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/` siblings.
- **HF safetensors + `safe_open` loader pattern.** Standard HF, but the per-tensor naming conventions (which projections to col- vs row-shard, which to replicate) come from TT's `model_config.py` recipes in the demos directory.
- **DeepSeek-V3 MoE expert recipe.** `server_35b_ttnn.py:1264, 1322` cites `experiments/.refs/tt-metal/models/demos/deepseek_v3/tt/experts.py:255-267` (and `:185, 261`) for the batched expert FFN shape choices.

## B. What was original (built from scratch — no prior art for this model family / shape)

- **`qwen36_gdn_decode_owned`** (~1.3K LOC C++ + 700 LOC Python). The Qwen3.6-35B-A3B Gated DeltaNet recurrence (`state_next = α·state + k·delta; out = q·state_next`) has no equivalent in tt-metal. We wrote the compute kernel, both dataflow kernels, program factory, and device op from scratch following the user-mandated G0→G4 staged pattern ([`feedback_build_kernels_from_scratch.md`](../.. memory)). Validation log `INTEGRATION.md` records PCC ≥ 0.9999987 at the 4 production shapes. Ships in 27B prod (+stacked perf wins) and is also the production path for 35B's 48 DN layers.
- **`qwen36_decay_gate_decode_owned`.** Fuses 10 ttnn ops (`add → softplus → exp → neg → mul → exp → sigmoid → 2× reshape`) into one kernel. **+2.5%** tok/s ([`feedback_owned_decay_gate_shipped.md`](../.. memory), commit `08877d5`). The 10-op chain it replaces lived at `server_tp.py:681-690`; the kernel exists because no fused softplus+exp+sigmoid op existed in TT-NN.
- **`qwen36_moe_ffn_decode_owned`** (G0–G2 landed, G3/G4 in flight). Fuses 35B's batched MoE Pattern-A FFN: `gate_up = h@W1; mid = silu(gate)·up; eo = mid@W2; routed = Σ_e rw[e]·eo[e]` into one kernel keeping intermediates in L1. Sub-ops (`qwen36_gdn_delta`, `_prediction`, `_decay_state`, `_outer_update`, `_output`) were the G0–G3 decomposed bring-up that fused into the final op; their READMEs explicitly state *"not copied as correctness ground truth from `tt-qwen-36`."*
- **Unified Gemma 4 server** (`server_gemma4_unified_ttnn.py`, 1,544 LOC). Gemma 4's hybrid sliding/global attention with dual head_dim (256 / 512), `attention_k_eq_v=True`, per-attention `v_norm` with `with_scale=False`, per-layer `layer_scalar` buffer, and SDPA `scale=1.0` (HF's `self.scaling=1.0`) — none of these are in any TT demo. tt-metal has a Gemma 4 demo but it only covers the SDPA program-config; the model wiring is ours.
- **Continuous batching scheduler** (`experiments/serve/cb_scheduler.py`, 695 LOC for `server_tp_cb.py` + the scheduler). vLLM/Orca-inspired iteration-level admit/advance/evict, but a custom impl over our ttnn batched forward — TT-team's vLLM integration is different (`tenstorrent/vllm` proper). Backed by `cb_reset_slots()` masked-multiply for per-slot DN state reset (no prior art — we needed to clear recurrent state on admission without disturbing peers).
- **Dev harness pattern** (`experiments/cb/dev/gm4_dev_harness.py`, `cb35_dev_harness.py`). Long-lived tmux'd Python process, `importlib.reload`-on-touch, filesystem-trigger probes, ~30 sec/iter vs ~14 min full bootstrap. No TT-team equivalent.
- **Model bringup recipe + HF oracle pattern** (`research/model_bringup_recipe.md`, `experiments/utils/hf_reference_*.py`). Per-sub-op forward hooks dumped to `.npy`, then a cosine ladder gates every TT change against ground truth (not against "the previous working TT path"). The methodology — not the code — is what dropped Gemma 4 12B from spec → HTTP chat in ~36 hours.
- **Needle-haystack + cosine ladder probes** (`experiments/utils/needle_haystack_*.py`, `cosine_ladder_hf_ref.py`). Original.
- **36 isolation probes under `experiments/cb/isolate/`.** Every primitive (paged_sdpa, paged_update_cache, dn_recurrence, conv_reform, gm4_sliding_write_read, ...) has a standalone probe that runs in seconds. No TT analogue.

## C. What we adapted (TT-team pattern + non-trivial fork)

- **Vocab-sharded `lm_head` + on-device argmax.** Forked from 27B's P22 (`feedback_vocab_sharded_lm_head_result.md`, commit `ef3f336`) — itself patterned on DeepSeek-V3 — into Gemma 4's 262144-vocab case (`feedback_p22_gm4_vocab_shard_result.md`, +8% tok/s for Gemma 4, +5.1% for 27B).
- **Two-phase trace warmup.** The bug (multi-trace coexistence corrupts memory) was documented in `tenstorrent/vllm#352`, so partly community knowledge; the *ordering rule* — compile every path first with `enable_trace=False`, then capture all back-to-back — is ours, codified in `cb_scheduler.py:194`.
- **Owned-op installer scripts** (`integrate_into_ttmetal.py` per op). Mechanical — patches CMakeLists.txt and `experimental_nanobind.cpp`, anchored on a sibling op. Adapted from how the upstream demos are wired in.

## D. The split, quantified

- `experiments/owned_ops/`: **8,116 LOC C++/HPP + 4,264 LOC Python** = ~12.4K LOC of code we wrote, validated, and integrated.
- `experiments/serve/server_*.py`: **16,206 LOC Python** of model wiring, dispatcher, and HTTP serving.
- `experiments/cb/` (scheduler + probes + dev harness): **11,383 LOC Python**.
- `ttnn.*` call sites in the three main servers: ~1,570; **`ttnn.experimental.qwen36_*` (our custom kernels): 7** call sites. We call into TT-NN ~225× more than we call into our own kernels — but those 7 call sites are on the hot path of every recurrent layer.

**Per-layer subroutine breakdown for a Qwen3.6 hybrid block** (representative): of ~14 distinct sub-routines per decoder block (embed_scaled, in_norm, q/k/v_proj, q_norm/k_norm, RoPE, SDPA, o_proj, all_reduce, post_norm, MLP gate/up/down, post-residual, layer_scalar), roughly:
- **9 are direct TT-NN calls** (norms, projections, RoPE, SDPA, CCL, residual adds, MLP linears)
- **3 are forked-and-adapted** patterns (Llama70B MLP dealloc discipline, vocab-shard lm_head, RMSNorm `(1+γ)` zero-centered fix)
- **2 are entirely original kernels** (`owned_gdn`, `owned_decay_gate` on DN layers)

## Takeaway

Tenstorrent's openness — the entire tt-metal repo, the production demos for Llama70B-Galaxy / DeepSeek-V3 / Gemma 4, and a publicly-discussable issue tracker — is the *only* reason a one-person Stanford research project could bring up three modern LLM families in a quarter. We forked the dispatch/CCL/SDPA backbone, lifted the data-loading and sharding conventions, and stamped our owned-op directories from upstream skeletons. What we had to build ourselves was the model-family-specific math (Qwen3.6's GDN recurrence + Gemma 4's hybrid sliding/global with its triplet of unique quirks), the **research-grade scaffolding** the demos don't ship (HF oracle, cosine-ladder gate, isolation probes, dev harness, owned-op staging pattern), and a continuous-batching scheduler tuned to our forward. The kernel-count ratio is ~225:7 in TT-NN's favor; the *correctness-gating* ratio is closer to 50:50. That's the right split for a research project that exploits an open platform.
