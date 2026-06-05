# Nemotron-3 Nano 30B-A3B → CB chat server — bringup plan (revised 2026-06-04)

**Target model**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`

**Architecture brief**: [`research/nemotron3_nano_architecture_brief.md`](nemotron3_nano_architecture_brief.md)
(authoritative; all shape constants and code-path claims sourced there).

> **Bringup posture (revised after research)**: this is the **biggest
> single departure** from the 27B / 35B / Gemma 4 lineage. Nemotron-3
> Nano is a **Mamba2-Transformer hybrid MoE** (NOT a pure transformer),
> with 23 Mamba2 SSD layers, 23 DeepSeek-V3-style MoE layers, and 6
> RoPE-free GQA Attention layers, dispatched in an interleaved pattern.
> **tt-metal does not ship a Mamba2 SSD kernel** — that's the gating
> blocker for everything downstream. Realistic timeline is **3-4 weeks
> (manual ttnn Mamba2 for v0..v2)** → **6-8 weeks total (with an owned
> Mamba2 kernel at v3 perf pass)**, not days.

------------------------------------------------------------------------

## 0. Locked decisions

### 0.1 Path B — Owned kernel up-front (2026-06-04, user)

Build the owned Mamba2 SSD decode kernel from scratch (G0..G4 staging
per `[[build-kernels-from-scratch]]`) BEFORE the v0..v2 forward /
decode / CB / HTTP ladder.

Rationale: Path B frontloads the kernel work so by the time the ladder
starts, every Mamba2 forward call is already running at production-grade
perf. No intermediate "manual ttnn composite" detour that would have to
be ripped out at v3. Trades intermediate demo visibility for cleaner
integration and one less rewrite.

**Status (2026-06-05): Phase 0 DONE.** All G-stages PASS at cos ≥ 0.999;
multi-step replay PASS; G2 full Nemotron shapes (B=1, 64 heads × 8
groups) PASS at state cos=0.999999, y cos=0.999995; G4 Python wrapper
PASS at full Nemotron shapes.

### 0.2 Phase ordering — 27B path (2026-06-05, user)

**Order**: single-stream correctness (Phase 1) → single-stream
performance (Phase 2) → continuous batching (Phase 3) → HTTP server
(Phase 4). CB and HTTP are explicitly deprioritised until single-stream
is demo-ready (≥30 tok/s traced, full chat coherence). Same path 27B
took. Mirrors `[[feedback-correctness-first]]`: never optimise past
cos < 0.99, and never scale (batching) past a known-bad correctness
floor.

### 0.4 Everything on-device, no host-side SDPA (2026-06-05, user)

User direction after v0.1.1.b: "I'd like to do everything on-device
because it makes things harder to fix later." Research-first round
that followed confirmed:

- `ttnn.transformer.scaled_dot_product_attention` (non-paged prefill)
  takes Q[b,nqh,s,dh] + K/V[b,nkv,s,dh] with `nqh ≠ nkv` as a
  1st-class contract — no caller-side K/V repeat needed
  ([[reference-ttnn-sdpa-gqa-native]]).
- 27B already uses this at `server_tp.py:1832`. Single-call fork.
- The NKV_PER_CHIP>1 contract issue (tt-metal #12330 OPEN) only
  fires on the DECODE path. Gemma 4's two-call workaround
  ([[reference-gemma4-two-call-paged-decode]]) is the blessed fix
  there. We'll fork at v0.3 decode, NOT at v0.1.1 prefill.

**Implication**: every v0.1.x sub-step uses on-device ops only.
No numpy bridges. The v0.1.1.b host-SDPA implementation is
preserved as a regression baseline but v0.1.1.c rewrites it
on-device.

### 0.3 L5-before-L0 in v0.1.x (2026-06-05)

In Phase 1, bring up **L5 (Attention) BEFORE L0 (Mamba2)** as a
"warmup" — Attention is the simplest layer block we've ever shipped
(no RoPE, no q_norm/k_norm, standard `1/sqrt(128)` scale). Building
the bootstrap + paged SDPA + KV cache scaffold on a boring layer means
v0.1.2 (Mamba2) and v0.1.3 (MoE) integrate on top of a known-good
scaffold.

------------------------------------------------------------------------

------------------------------------------------------------------------

## 1. TL;DR (architecture, sourced from §1 of the brief)

- **Class**: `NemotronHForCausalLM` (`model_type=nemotron_h`); requires
  `trust_remote_code=True` for HF AutoModel oracle. Permissive
  Nvidia open license; no HF gating.
- **52 layers, `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`
  dispatch**: 23 Mamba2, 23 MoE, 6 Attention. Attention sits at
  indices `[5, 12, 19, 26, 33, 42]`. No 'mlp' (dense FFN) layers.
- **30-31.6B total, ~3.5B active per token**. 13 safetensors shards ≈
  63 GB on disk; 32 GB bf16 weights = 8 GB/chip on a (1,4) mesh.
- **Mamba2 mixer**: 64 heads × 64 head_dim = 4096 d_inner, ssm_state=128,
  n_groups=8, conv_kernel=4, expand=2, chunk_size=128, **fp32 SSM
  state**.
- **Attention**: 32 Q heads, 2 KV heads (16:1 GQA), head_dim=128,
  **NO RoPE applied** (config fields present but modeling does NOT
  consume them — positional info lives entirely in Mamba2 state).
  Standard SDPA scale = 1/sqrt(128).
- **MoE**: 128 routed + 1 shared, top-6 routed, sigmoid router with
  group-restricted top-k (n_group=8, topk_group=1), `routed_scaling_factor=2.5`,
  `relu²` activation, shared expert 2× wider (3712 vs 1856).
- **Norm**: Llama-style RMSNorm (`y = x/rms * w`, NO `+1.0`, NO bias);
  one pre-norm per layer; no post-norm; no layer_scalar.
- **Vocab 131072** (smallest of any model we've shipped — clean power
  of 2 for vocab-shard), `tie_word_embeddings=False`, **no logit softcap**.
- **Context** `max_position_embeddings=262144` (256K). Our current
  `MAX_POS=8192` cap will need a bump for the full claim, but v0 ships
  at 8K.
- **Tokenizer**: ChatML (`<|im_start|>{role}\n…<|im_end|>\n`), same
  family as 27B/35B. EOS frozenset `{2, 11}` = `{</s>, <|im_end|>}`.
  Active-prompt suffix is `<|im_start|>assistant\n<think>\n` (thinking
  on) or `<|im_start|>assistant\n<think></think>` (thinking off) —
  similar shape to Qwen3.6's. Our `_active_prompt_suffix` should cover.

**Source**: all the above quoted verbatim from §1-§5 of the brief.

------------------------------------------------------------------------

## REUSE MANDATE (`[[feedback-reuse-mandate]]`)

The REUSE map is now grounded in the brief's §6, not guesses. Confidence
levels are the brief's; restated here for completeness.

### Reused verbatim — zero changes (HIGH confidence)

| Existing | Role for Nemotron |
|---|---|
| `experiments/serve/cb_engine.py`, `cb_scheduler.py`, `cb_metrics.py`, `cb_api.py`, `openai_endpoint.py`, `live_slot_store.py`, `protocol.py` | Production CB stack — model-agnostic |
| `experiments/serve/openai_endpoint.py:_active_prompt_suffix` | Active-prompt-suffix detector (post `184753d`) — Nemotron's ChatML `<think>` suffix has the same shape Qwen had |
| `experiments/serve/scripts/{deploy.sh,serve_cb.sh}` | Deploy + lifecycle unchanged |
| `experiments/utils/ttnn_introspect.py`, `hf_download.py`, `npz_inspect.py`, `syntax_check.py` | Helpers — call as-is |
| `experiments/cb/isolate/paged_sdpa.py` | Attention is the **simplest** we've shipped (no Q-gate / q/k_norm / RoPE), just `scale=1/sqrt(128)` |
| `experiments/cb/isolate/paged_update_cache.py` | KV cache write/read — verify NKV=2 / 4 chips contract, may need replication trick (see §7.2 of brief) |
| `experiments/cb/isolate/chat_template_invariant.py`, `chat_template_inspect.py`, `chat_template_roundtrip.py` | Chat-template gates |
| `experiments/cb/validate/pc_token_match.py` | Prefix-cache regression gate — works on any HF tokenizer |
| `scripts/chat.py` (TUI) | Demo client — backend-agnostic |
| `scripts/stress_multiturn_http.py`, `stress_ttft_decode.py`, `stress_concurrent_chat.py` | Stress tests — backend-agnostic |
| `experiments/serve/server_tp.py:1680-1687` P22 vocab-sharded LM head + on-device argmax | Reuse verbatim with VOCAB=131072 (clean /4 split = 32768/chip) |
| Two-phase warmup pattern (`[[ttnn-multi-trace-two-phase-warmup]]`) | Universal trace discipline |

### Fork from 35B (HIGH structural confidence; MEDIUM math delta)

| Existing | Fork target for Nemotron | Delta |
|---|---|---|
| `experiments/serve/server_35b_ttnn.py` (2040 LOC) | `experiments/serve/server_nemotron3_nano_ttnn.py` | Bootstrap skeleton + per-layer dispatch carries over. New: Mamba2 layer math; sigmoid router + group restriction; relu² activation; shared expert 2× wider. 52 layers vs 40, vocab 131072 vs 248320, hidden 2688 vs 5120. |
| `experiments/serve/server_35b_cb.py` (967 LOC) | `experiments/serve/server_nemotron3_nano_cb.py` | Batched CB wrapper; same shape with new layer types |
| `experiments/utils/hf_reference_35b.py` | `hf_reference_nemotron3_nano.py` | Drop DN hooks; add **Mamba2 layer hooks** (in_proj, conv1d, ssm_state, out_proj per-step); keep attn + MoE hooks. Use `trust_remote_code=True`. Needs CPU RAM check (~62 GB needed for AutoModel bf16) |
| `experiments/utils/cosine_ladder_35b.py` | `cosine_ladder_nemotron3_nano.py` | Swap n_layers 40→52, oracle path |
| `experiments/cb/dev/cb35_dev_harness.py` | `cb_nemotron3_dev_harness.py` | Fork (already hardened #166); expect ~10-12 min bootstrap (between Gemma 4's 80s and 35B's 14 min) |
| `experiments/cb/isolate/cb35_per_layer_drift_pos1.py` | `cb_nemotron3_per_layer_drift_pos1.py` | Per-layer drift probe at pos>0 |
| `experiments/cb/isolate/cb35_needle_haystack.py` | `cb_nemotron3_needle_haystack.py` | Long-context retrieval gate (push to ≥8K) |
| 35B's `moe_forward_ttnn_pattern_a_batched` (`server_35b_ttnn.py:1225`) | New `moe_forward_ttnn_nemotron3` | ~150-200 LOC delta: sigmoid not softmax; group-restricted topk pre-mask; routed_scaling_factor=2.5 multiply; relu² in expert FFN; shared expert 2× wider (different program config) |

### NEW — Mamba2 is fresh kernel territory (LOW reuse from owned-GDN)

| Component | LOC est. | Notes |
|---|---|---|
| `experiments/cb/isolate/mamba2_decode_composite.py` | ~200 | Manual ttnn composite for SSD recursion. v0 path. |
| `experiments/cb/isolate/mamba2_conv1d_step.py` | ~100 | Per-step causal Conv1d-step with rolling `[B, conv_dim, 4]` state. ttnn has Conv1d, not the per-step caching form. |
| Mamba2 cache plumbing in `MeshServerState` / `State` | ~50 | Conv state + SSM state allocation per Mamba layer; fp32 SSM state |
| **35B's `qwen36_gdn_decode_owned` kernel** | — | **NOT directly reusable**. The plumbing pattern (recurrent state in/out, kernel-side fp32 acc) is transferable to a new `nemotron3_mamba2_decode_owned` kernel; the per-step math is NOT. Path B / v3 owned kernel only. |

### Reuse summary

- **Reused as-is**: ~70-80% of CB + tokenizer + chat-template + utility infra.
- **Forked from 35B with delta**: bootstrap, MoE Pattern A, dev harness, oracle, ladders.
- **Net new**: Mamba2 composite (v0), Mamba2 owned kernel (v3).

------------------------------------------------------------------------

## 2. Memory cross-references (read before any code)

The high-leverage memories for a Mamba2 + MoE bringup:

- `feedback_reuse_mandate.md` — fork-don't-write rule
- `feedback_two_phase_warmup.md` — multi-trace capture discipline
- `feedback_perf_workflow.md` — 6-step perf workflow
- `feedback_correctness_first.md` — never optimise past cosine < 0.99
- `feedback_validate_against_ground_truth.md` — HF oracle, not weaker TT path
- `feedback_gemma4_sdpa_scale_1.md` — model-specific SDPA scale. **Nemotron is `1/sqrt(128)` (standard)**, NOT Gemma's `1.0`.
- `feedback_qwen36_qnorm_knorm_zero_centered.md` — the `(1+w)` rule. **Nemotron uses Llama-style RMSNorm (no `+1`)**, same as Gemma 4.
- `feedback_ttnn_slice_view_decay.md`, `feedback_ttnn_list_rebinding_leaks.md`, `feedback_ttnn_rms_norm_shape_drift.md` — universal ttnn footguns
- `feedback_paged_update_cache_nkv_per_chip.md` — NKV=2 per chip-pair; see §7.2 of brief for the sharding decision
- `feedback_read_kernel_source_first.md` — read TT_FATAL assertions before forking into a new shape regime
- `feedback_use_existing_isolation_probes.md` — fork an isolation probe before iterating in a full forward
- `feedback_cb_backend_dispatch_holes.md` — grep every `27b`/`35b`/`gemma4_12b` literal in `experiments/serve/` when adding `nemotron3_nano`
- `feedback_deploy_serve_files_too.md` — `deploy.sh experiments/serve/*.py` before server restart
- `feedback_use_dev_harness_for_iteration.md` — never restart `serve_cb.sh` per fix; use the dev harness. **Mandatory from day 1 here** because bootstrap is 10-12 min.
- `feedback_harness_state_version_skew.md` — new State fields need `getattr(state, "X", None)` lazy-init
- `feedback_35b_dn_h_state_drift_lever.md` — **fp32 recurrent state inside a trace caused a 30+ min hang on 35B**. Nemotron's fp32 SSM state hits the same risk surface. See §7.3 of brief.
- `feedback_show_thinking_traces.md` — TUI default (Nemotron also emits `<think>` blocks; user wants them visible)

------------------------------------------------------------------------

## 3a. Phase 0 — Owned Mamba2 SSD kernel (G0..G4)

This phase runs FIRST per the Path B decision. Each G-stage gates the
next; the kernel ladder mirrors the 35B `qwen36_gdn_decode_owned` build
which took G0..G4 in ~10 days (`feedback_owned_decay_gate_shipped`).
Mamba2 SSD has a similar structure (recurrent state, per-head loop,
fp32 accumulator) but different math; reuse the build *pattern*, not
the math.

### Per-step Mamba2 SSD math (the kernel implements this)

Pseudocode for one decode step, per head h ∈ [0, 64), per token:

```
# Read state and inputs
ssm_state[h, :, :]  ∈ fp32, shape [head_dim=64, ssm_state=128]
x[h, :]             ∈ bf16, shape [head_dim=64]
dt[h]               ∈ bf16, scalar
A_log[h]            ∈ bf16, scalar  (learned)
B[h, :]             ∈ bf16, shape [ssm_state=128]  (per-group, broadcast)
C[h, :]             ∈ bf16, shape [ssm_state=128]
D[h]                ∈ bf16, scalar  (learned)

# Discretization (fp32)
dt_eff = softplus(dt + dt_bias[h]).clamp(time_step_floor, time_step_max)
A      = -exp(A_log[h])

# Update SSM state (fp32 accumulator)
# ssm_state_new[d, s] = exp(dt_eff * A) * ssm_state[d, s] + dt_eff * B[s] * x[d]
ssm_state_new[d, s] = exp(dt_eff * A) * ssm_state[d, s] + dt_eff * B[s] * x[d]

# Output projection (fp32 reduce, bf16 result)
y[d] = sum_s (C[s] * ssm_state_new[d, s]) + D[h] * x[d]

# y is then gated by z and norm'd outside the kernel (MambaRMSNormGated
# uses head_dim=64 groups; can be a separate ttnn call).
```

The kernel signature is:
```
mamba2_decode_owned(
    x:          [B, num_heads=64, head_dim=64]      bf16 input
    z:          [B, num_heads=64, head_dim=64]      bf16 gate (passed through to norm-gated outside)
    dt:         [B, num_heads=64]                    bf16 dt
    B_tensor:   [B, n_groups=8, ssm_state=128]      bf16 B matrix
    C_tensor:   [B, n_groups=8, ssm_state=128]      bf16 C matrix
    ssm_state:  [B, num_heads=64, head_dim=64, ssm_state=128]   fp32 in/out (mutated)
    A_log:      [num_heads=64]                       bf16 learned
    dt_bias:    [num_heads=64]                       bf16 learned
    D:          [num_heads=64]                       bf16 learned
)  →  y: [B, num_heads=64, head_dim=64]              bf16 output
```

Note: B and C are *grouped* (8 groups, 8 heads per group), so each
group's B/C broadcasts across its 8 heads. The kernel must implement
the per-group broadcast.

### G-stage ladder

| Stage | Adds | Gate | Reference |
|---|---|---|---|
| **G0** | Read existing tt-metal SSM-adjacent ops (`ttnn.experimental.*`, look for ssm/mamba/recurrence/scan). Read `state-spaces/mamba` mamba_ssm Triton kernel for math reference. Numpy oracle: pure-numpy single-token SSD step matching HF `modeling_nemotron_h.py:NemotronHMamba2Mixer.forward` at the `mamba_chunk_scan_combined` call site. | numpy oracle byte-match vs HF eager forward at L0 step 0, all 64 heads; doc what tt-metal exposes vs needs to be authored | Fork pattern from `[[dnk-g0-read-owned-gdn-source-plan-batching]]` |
| **G0a** | Isolation harness: `experiments/cb/isolate/mamba2_decode_oracle.py`. Generates random inputs, runs numpy oracle, returns expected outputs. Used as ground truth by all subsequent stages. | runs end-to-end on host; produces deterministic outputs |  fork from `test_pattern_a_moe_np.py` |
| **G1** | Single-core full-chain Mamba2 step in `tt-metal/ttnn/cpp/ttnn/operations/...`. Single-tile (B=1, single head) program-factory. Implements the per-head SSD recursion (discretization → state update → output reduce). Reads/writes fp32 ssm_state. | cos ≥ 0.999 + MAD bit-close to numpy oracle for B=1, single head | Fork pattern from owned_gdn G1; share circular-buffer + LLK conventions |
| **G2** | Multi-core: shard `num_heads=64` across Tensix cores. Each core processes its heads independently (no cross-head reduce; the reduce is per-head over `ssm_state` dim). | cos ≥ 0.999 at full B=1, num_heads=64; throughput ≥ 8× G1 | Fork pattern from owned_gdn G2 |
| **G3** | Batching: assert `state_logical[0] == B`; program-factory handles the leading batch dim. Mirror `qwen36_gdn_decode_owned`'s batched form. (35B's hard-assert batch=1 is exactly what we DON'T want to repeat.) | cos ≥ 0.999 at B=1..32, num_heads=64 | Fork pattern from `[[dnk-g1-batch-the-kernel-assert-plus-program-factory]]` |
| **G4** | Integrate into `experiments/serve/server_nemotron3_nano_ttnn.py:mamba2_forward_ttnn` (which exists by then, see Phase 1). Two-phase warmup + trace capture works. | end-to-end smoke + cos vs eager bit-close | Fork pattern from `[[dnk-g2-integrate-batched-owned-gdn-into-cb-dn-step-plus-measure]]` |

### Tasks (Phase 0) — tracked in the project task list

- **Task #183** — G0 numpy oracle + tt-metal SSM survey — **IN PROGRESS**:
  oracle written (`experiments/utils/mamba2_numpy_oracle.py`, commit
  `98fc43d`) + self-test ✓ on qb1; tt-metal SSM survey ✓ (no matches,
  build from scratch confirmed). HF byte-match gate still open
  (pending Nemotron weight download).
- **Task #184** — G0a isolation harness — **DONE** (commit `4352baf`):
  multi-step replay + per-head cos/MAD gate. Self-test PASSES; kernel-
  compare path stubbed via `--kernel-callable` for G1 to wire in.
- **Task #185** — G0b qb1 RAM + tt-metal SSM survey — **DONE**:
  qb1 has 503/468 GB RAM (plenty); ZERO ttnn SSM ops at module level.
- **Task #186** — G1 single-core (blocked by #184)
- **Task #187** — G2 multi-core (blocked by #186)
- **Task #188** — G3 batched (blocked by #187)
- **Task #189** — G4 server wrapper + Phase 1 unblock (blocked by #188)

### Phase 0 progress (2026-06-04 EOD)

- [x] **G0b** — qb1 RAM + ttnn SSM survey done. No tt-metal Mamba2
      primitive exists; building from scratch.
- [-] **G0** — numpy oracle written + internal self-test PASS. HF
      byte-match gate still open (pending Nemotron weight download
      to qb1).
- [x] **G0a** — isolation harness (multi-step replay + per-head
      cos/MAD + kernel-compare hook via `--kernel-callable`). Commit
      `4352baf`.
- [-] **G1** — in progress (task #186, latest commit `e96b9a9`):
  - ✅ Kernel design doc: `research/mm7_g1_mamba2_kernel_design.md`
  - ✅ Dataflow decisions log (10 D-numbered choices, each a future
    optimization lever): `research/mm7_g1_dataflow_decisions.md`
  - ✅ Conv1d reuse discovered: `qwen36_conv1d_decode_owned`
    parametrised at D=6144 IS Mamba2's `conv1d_step`. Zero new
    conv code.
  - ✅ Day-1 fork: `experiments/owned_ops/nemotron3_mamba2_decode_owned/`
  - ✅ Day-1.5 compute kernel skeleton (commit `e96b9a9`): full
    file header documenting the math + SPMD + fp32 acc + debug_mode
    pattern. CB layout pinned (15 CBs, cb_x..cb_y). debug_mode=1
    (fill_one smoke) implemented for first-build gate. Six TODO
    blocks for the math helpers with explicit LLK call sequences.
  - [x] Day-3: LLK API survey done — softplus/clamp/exp/negative ship
    as first-class SFPU primitives (D8 RESOLVED). compute_decay +
    mul_decay_state_to shipped; D11 introduced split-helper approach.
  - [x] Day-3.5 (commit pending push): finalize_decay_with_dt_eff body
    landed. Three-stage pipeline:
      1. Stage 1: dt_eff = clamp(softplus(dt + dt_bias), floor, max)
         → cb_dt_B (reused as scratch).
      2. Stage 2+3: decay = exp(A * dt_eff) → overwrites cb_decay via
         queue-pop-after-push semantics.
    compute_decay simplified to one tile_regs cycle (A only). D11
    marked RESOLVED with the scratch-CB pattern documented. debug_mode=2
    output now: state_out = decay * state_in (matches oracle's
    decay-only term; full math at debug_mode=5).
  - [x] Day-3.75: full Mamba2 plumbing landed in 6 files —
    device_operation_types.hpp (Mamba2 Params + Inputs structs),
    device_operation.hpp/.cpp (validate + compute_output_specs +
    create_output_tensors + ttnn::prim wrapper),
    program_factory.hpp/.cpp (15 CBs sized per per-block tile counts,
    compute compile-time args 0..14, reader/writer compile-time +
    runtime args), reader (loads x/z/dt/dt_bias/A_log/D/B/C/state),
    writer (drains state_out + y), outer wrapper hpp/cpp, nanobind
    binding registering ttnn.experimental.nemotron3_mamba2_decode_owned.
    Next: install via integrate_into_ttmetal.py → build on qb1 →
    debug_mode=1 smoke (output all 1.0).
  - [x] **Day-3.9: SMOKE PASSED bit-exact at debug_mode=1**.
    Build + install + dispatch + CB pipeline + readback all validated.
    4 fixes across 3 commits:
      (a) `ttnn::register_operation` → GDN-pattern `device_operation::launch`
      (b) unused `constexpr uint32_t TILE` → removed (Werror)
      (c) missing `#include "api/compute/tile_move_copy.h"` for copy_tile
      (d) `copy_tile_to_dst_init_short` → `copy_tile_init` (cleaner API)
      (e) smoke readback: ttnn.to_torch doesn't handle bf16 → typecast(fp32) first
    Smoke output: `y_out shape=(1,1,64) finite=True min=+1.000 max=+1.000 |y-1.0|=0.0e+00`
    `state_out shape=(1,1,64,128) finite=True min=+1.000 max=+1.000`
    JIT cache: 11/13 hits (84.6%). End-to-end runtime ~1 s after cache warm.
  - [~] **Day-4 PARTIAL (2026-06-04 EOD)**: mode=3 wired but partial.
    Mode=2 PASSES (decay × state — refactor of finalize_decay into
    compute_dt_eff + multiply_decay_by_dt_eff is sound). Mode=3 PARTIAL
    PASS: kernel runs and produces sensible state for d=0 (cos=0.21 with
    sentinel-fill of d=1 tiles), but HANGS at full 8-tile (d=0..1) loop.
    Key finding: **`mm_init` PRIME before the inner transpose+matmul loop
    is REQUIRED** — without it, the first matmul ever in the kernel
    hangs the TRISC pipeline. GDN avoids this implicitly via its
    `matmul_reduce` prelude; we have no such prelude until mode=5.
    Other findings:
    - `transpose_x_to_col` MUST keep `unary_op_init_common` (GDN pattern);
      removing it leaks sticky transpose-unpacker state into matmul
      (tt-metal #15930 — no `transpose_wh_uninit`).
    - `cb_outer` is bf16 (matches GDN), not fp32. Pack precision OK
      because matmul accumulates fp32 in dst before quantizing.
    - Single-phase loop works (transpose+matmul+mul_decay+add+pop per s)
      once mm_init prime is in place; two-phase split unnecessary.
    Bisect log + canonical patterns documented in commit `b5c93fa`.
  - [x] **Day-4 PASS (2026-06-05): mode=3 cos = 0.9997 vs oracle.**
    Full SSM state-update math working end-to-end:
    `state_out[d, s] = decay * state_in[d, s] + dt_eff * x[d] * B[s]`.
    Commit `d239875`. The 4-ingredient recipe (memory:
    [[feedback-mm-init-prime-required]]):
      1. matmul_reduce_C_state as GDN-structural prime (real matmul,
         transpose=1, produces y_partial for mode=4/5).
      2. Pre-transpose phase: transpose_x_to_col called ONCE per d
         in a separate pre-loop.
      3. Inner loop uses mm_init_short (NOT full mm_init) per iter.
         Full mm_init each iter triggers the ~4-iter Blackhole TRISC
         cap; light short variant bypasses it.
      4. Explicit pack_reconfig_data_format(cb_outer) after
         mm_init_short. Full mm_init implicitly does
         llk_pack_hw_configure; short doesn't.
    Each ingredient is necessary; any 3 alone don't unblock.
  - [x] **Day-4.5 PASS (2026-06-05): mode=4 — y = C·state_in^T + D·x**.
    y_out cos = **0.999998** (essentially bit-perfect), rel = 2.4e-3.
    Commit `978f23e`. Operand swap of matmul_reduce_C_state to
    (cb_C, cb_state_in) + mm_init transpose=1 puts the reduce in row 0
    (row-vec form), matching x's layout so the D·x add works with
    plain `add_tiles`. 2 new helpers: mul_D_x_to + add_y_partial_D_x.
    cb_outer reused as D·x scratch.
  - [x] **Day-4.6 PASS (2026-06-05): mode=5 — PRODUCTION y = C·state_out^T + D·x.**
    y_out cos = **0.999852** vs numpy oracle. Commit `b2c4ccc`.
    Implementation: add CB_STATE_POST_UPDATE (8 tiles fp32) +
    add_state_scaled_outer_two helper (dual-pack to writer cb_state_out
    AND compute-read cb_state_post_update). Phase 4 runs a SECOND
    matmul_reduce_C_state on cb_state_post_update for the corrected
    y_partial. Phase 2's prime matmul on cb_state_in stays for engine
    warm-up; its 2 tiles drained before Phase 4 pushes correct values.
    Forked from GDN's add_state_to_two two-output pattern.
  - [x] **G1 single-core kernel COMPLETE.** Modes 1-5 all PASS. The
    Nemotron-3 Mamba2 SSD decode kernel is end-to-end correct on
    single-core, B=1, single-head. Task #186 DONE.
  - [x] **Day-5 PASS (2026-06-05): 8-step multi-step replay**. Recurrence
    bit-correct at every step (cos ≥ 0.9999). Exposed and fixed a hidden
    precision bug: `mm_init_short` + bf16 cb_outer was silently dropping
    the outer-product contribution. The single-step smoke missed it
    because random state_in dominated; multi-step with state_in=ZEROS
    surfaced it. Fix: cb_outer now fp32, matmul_outer_x_dt_B uses full
    mm_init (the day-4.2 split-phase made the iter cap a non-issue).
    Commit `2386d97`. Probe: `experiments/cb/isolate/mamba2_multi_step_replay.py`.
    Regression sweep: `experiments/cb/isolate/mamba2_regression_sweep.sh`.
- [x] **G2 (2026-06-05): multi-core 64-head shard — PASS**. Full Nemotron
  shapes at NUM_HEADS=64, N_GROUPS=8 (1 head per core), state cos=0.9999,
  y cos=0.9999. The program_factory's existing split_work_to_cores
  worked out-of-the-box; the fix was host-side per-head input replication
  (B/C from per-group → per-head, x/z to per-head tile-rows). Commit
  `bcc2ce9`. Probe: `experiments/cb/isolate/mamba2_g2_multihead_smoke.py`.
  Task #187 done.
- [x] **G3 (2026-06-05): batched B>1 — PARTIAL**. B=2 single-head PASS,
  B=2 small-multi-head PASS, **B=2 full 64-head HANGS/NaN** when
  `blocks_per_core > 1` (some cores process 2 blocks). Static analysis
  of CB sizing didn't surface the bug; needs device-side DPrint/Tracy
  instrumentation. Parked: NOT a Phase 1 blocker because B=1 + full
  64-head (G2) is correct and the CB engine drives per-slot at the
  server layer (same pattern as 27B/35B). Commit `765e9c7`. Task #188 done.
- [x] **G4 step 1 PASS (2026-06-05): Mamba2 wrapper module**.
  `experiments/serve/nemotron3_mamba2_step.py` exposes
  `mamba2_decode_step_ttnn(...)` with signature matching the numpy
  oracle. Wrapper smoke at full Nemotron shapes: state cos=0.999999,
  y cos=0.999995 (commit `243cc0a`). Server scaffold can drop it in
  as the Mamba2-layer step function 1:1.
- [x] **v0.0 config probe PASS (2026-06-05)**:
  `experiments/utils/nemotron3_nano_config_probe.py` confirms model
  exists on HF, `trust_remote_code=True` loads `NemotronHForCausalLM`,
  hybrid_override_pattern present (52 layers: 23 M + 23 E + 6 *).
  Tokenizer is ChatML (TokenizersBackend). Kernel-relevant shape match:
  Mamba2 = 64 heads × 64 head_dim × 128 ssm_state, n_groups=8 — exactly
  matches our G2 multi-core kernel.
- [ ] **v0.0 HF oracle (running in background)**: downloading model
  weights now (~63 GB shards). Uses `use_mamba_kernels=False` to
  bypass the modeling code's hard CUDA-mamba_ssm dependency. Outputs
  to `.cache/hf_oracle_nemotron3_nano/`. Tmux session
  `nemotron_oracle` on the QuietBox.

Phase 0 timeline estimate: **3-5 weeks** depending on how many of the
G-stages hit unexpected snags. Each stage gates the next; do NOT
parallelise the G-ladder.

------------------------------------------------------------------------

## 3b. Phase 1 — Forward / decode / CB / HTTP ladder (REORDERED 2026-06-05)

This phase runs AFTER G4 lands. Each stage gates the next.

**ORDERING DECISION (user, 2026-06-05)**: follow the 27B path — finish
**single-stream correctness + performance FIRST**, then add batching,
then ship the HTTP server. CB and HTTP are explicitly deprioritised
until single-stream is demo-ready. Within Phase 1, **L5 Attention is
brought up BEFORE L0 Mamba2** as a "warmup" — it's the simplest layer
we've ever shipped (no RoPE, no q_norm/k_norm, standard scale), so we
land the bootstrap + paged-SDPA + KV-cache scaffold on a boring layer
before integrating the new Mamba2 wrapper on top of it.

The owned Mamba2 kernel (formerly v3) is no longer a separate stage —
Phase 0 already shipped it, so every Mamba2 layer call at v0.1.2+
already uses the production-grade kernel.

### Phase 1 — Single-stream correctness (v0.0 → v0.3)

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v0.0** | `hf_reference_nemotron3_nano.py` with `trust_remote_code=True`. 5 attr / closure bugs fixed (`model.backbone`, `layer.mixer`, CPU `torch.cuda.stream` patch, MoE tuple, closure late-binding). HARDENED 2026-06-05 commit `edea531` with: Mamba2 `mixer.norm` hook (MambaRMSNormGated — kernel post-cond gate), per-block mixer-output hooks for L0/L1/L5 (residual-input gate), MoE `shared_experts` hook, shape + layer-type assertions, `--gen N` multi-step path. | argmax=6993 → " Paris" ✓; hs_stack [53,5,2688] ✓; logits [5,131072] ✓; 19 sub-hook artifacts in `.cache/hf_oracle_nemotron3_nano/`; `--gen 4` 9-pos forward verified | ✅ **DONE** (commits `7fae453`, `ad677f6`, `8e6cea7`, `edea531`) |
| **v0.0.1** | `experiments/utils/nemotron3_tokenizer_probe.py` (~150 LOC). Active-prompt suffix detected on BOTH thinking branches (6 tokens each: `<|im_start|>assistant\n<think>\n` vs `<|im_start|>assistant\n<think></think>`). EOS list [2, 11] = [`</s>`, `<|im_end|>`]. Multi-turn truncate_history_thinking confirmed (matches Qwen 3.6 / Gemma 4 PC asymmetry). | TokenizersBackend, BOS=1/EOS=11, model_max=262144, vocab=131072, chat_template inlined, both suffix branches resolve correctly | ✅ DONE (commit `f70b6f2`) |
| **v0.0.2** | `experiments/utils/nemotron3_weights_introspect.py` (~200 LOC). Reads safetensors INDEX only (no tensor materialisation). Result: 6243 keys × shapes audited; **0 missing, 0 shape mismatches, 0 extras**. **Real finding for v0.1.3**: every MoE gate has `e_score_correction_bias` [128] (DeepSeek-V3 load-balance bias) — added to router scores before topk; brief did not flag this. 58.82 GB across 13 shards. Keys by kind: 30 attn + 207 mamba2 + 6003 moe + 3 non-layer. | 0 missing, 0 shape mismatches | ✅ DONE (commit `f70b6f2`) |
| **v0.1.0** | `experiments/serve/server_nemotron3_nano_ttnn.py` (~250 LOC, forks `server_35b_ttnn.py` helpers) — bootstrap-only scaffold (mesh + embed + final_norm + lm_head). Llama-style RMSNorm (no +1.0). Vocab=131072 (clean /4 split). Validator `experiments/cb/isolate/nemotron3_v010_bootstrap_smoke.py` (~170 LOC). Key finding: HF `logits.npy` is bf16-precision; use numpy fp32 as strict ground truth ([[feedback-hf-logits-npy-is-bf16-imprecise]]). | Bootstrap 5.5s; Gate A embed cos=1.000000; Gate B final_norm cos=0.999905; Gate C1 generation token TT==HF (' Paris'); Gate C2 logits cos vs numpy=0.999970; Gate C3 argmax 5/5 vs numpy | ✅ DONE (commit `010b98e`) |
| **v0.1.1.a/b** 🔄 reordered | **L5 (Attention) — eager — host-SDPA stages**. v0.1.1.a adds `upload_attn_layer` + `attn_projections_only` + sparse layer-upload env gate; v0.1.1.b adds `attn_block_eager` (TT pre-norm + qkv + numpy SDPA + TT o_proj + residual). GQA 16:1, NO RoPE, NO q/k_norm. | v0.1.1.a 4/4 (H/Q/K/V cos ≥ 0.9999); v0.1.1.b 3/3 (O 0.9998, M 0.9998, B 1.000000) | ✅ DONE (`95f6cf7` + `e7f3e59`) |
| **v0.1.1.c** | **L5 — fully on-device prefill SDPA** (replaces numpy bridge). Forked from 27B `server_tp.py:1832` per [[reference-ttnn-sdpa-gqa-native]]: single `ttnn.transformer.scaled_dot_product_attention` call handles GQA natively (Q[b,32,5,128] + K/V[b,2,5,128] with `is_causal=True`, scale=1/sqrt(128), B3 HiFi2). reshape+transpose for head reorder; transpose+reshape back for o_proj. Residual add on-device. | Gate O cos=0.999460; Gate M cos=0.999460; Gate B cos=1.000000 (bit-exact post-residual). Slightly lower O/M than v0.1.1.b's numpy fp32 ref (0.999799) — bf16 SDPA precision floor; well above 0.999 gate | ✅ DONE (commit `ee2b76e`) |
| **v0.1.2.a** | **L0 (Mamba2) — pre-norm + in_proj only**. `upload_mamba2_layer` ships {norm, in_proj, out_proj} replicated bf16 on the mesh; {conv1d_w/b, dt_bias, A_log, D, mixer_norm_w} held host-side (each tiny). `mamba2_in_proj_only` does TT pre-norm + matmul [HIDDEN=2688 → d_inner+conv_dim+num_heads=10304]. | H pre-norm cos=0.999949; I in_proj cos=0.999949 (vs HF L0_in_proj). Bootstrap (L0 only) 7.0s | ✅ DONE (`490f89f`) |
| **v0.1.2.b** | **L0 — in_proj split + conv1d**. `ttnn.conv1d` supports `groups=conv_dim` — fork verbatim. Pipeline: ttnn.slice splits in_proj output → reshape NHWC [B,1,S,6144] + to_layout ROW_MAJOR → `ttnn.conv1d(weight, bias_tensor=b, groups=6144, kernel=4, padding=3 sym)` → output [B,8,6144] matches HF's full pre-causal-slice tensor. Mesh open now passes `l1_small_size=65536` ([[reference-l1-small-for-conv1d]]); silu lives in v0.1.2.c. | H 0.999949; I 0.999949; **C conv1d_out cos = 0.999991** | ✅ DONE (`745a438`) |
| **v0.1.2.c** | **L0 — full forward (on-device chain).** Implements pre-norm → in_proj → split → conv1d → causal slice → silu → split x/B/C → host-orchestrated SSD loop (each call hits the on-device kernel via the G4 wrapper) → MambaRMSNormGated (group-RMSNorm → weight → silu(z)) → out_proj → residual. Wrapper patched for mesh-awareness (`isinstance(device, ttnn.MeshDevice)` branches the upload/readback). | Gate N norm cos=0.999904; Gate O o_proj=0.999937; Gate M mixer_out=0.999937; Gate B block_out=0.999930 — **4/4 PASS** after the v0.1.2.d clamp fix landed. Forward 2.4s, bootstrap 5.8s. | ✅ DONE (commits `587ae06` + `dd7b80d`) |
| **v0.1.2.d** | **dt clamp bug fix** — both the numpy oracle and the kernel hardcoded `(1e-4, 0.1)` for the softplus+clamp range, but HF Nemotron uses `self.time_step_limit = (0.0, inf)` (`time_step_min`/`max` config fields are not what HF clamps against). The wrong clamp gave dt_eff ~2.77x smaller than HF — oracle y cos=0.943 vs HF y_pre_norm; per-position drift growing pos 0=0.989 → pos 4=0.963. Hook on `mixer.norm` pre-forward + `nemotron3_v012c_debug_numpy_ref.py` localised the issue. Fixed in oracle defaults + `TIME_STEP_FLOOR/MAX_BITS` constants; kernel rebuilt; v0.1.2.c PASS. | numpy oracle vs HF y_pre_norm cos = 0.999999 (all positions ≥ 0.999999) | ✅ DONE (commit `dd7b80d`); memory: [[feedback-nemotron3-time-step-clamp-bug]] |
| **v0.1.3.a** | **L1 (MoE) — router only**. `upload_moe_layer_router_only` (norm + gate + bias). `moe_router_only` does TT matmul + sigmoid + host topk-6 with e_score_correction_bias. KEY finding: `n_group=topk_group=1` in actual Nemotron config (brief said 8) — group restriction degenerates to plain topk-6 over all 128 experts. norm_topk_prob=True + routed_scaling=2.5. | Gate T topk_indices per-token set match: 5/5 ✓; Gate W topk_weights cos=0.999991 | ✅ DONE (commit `7ad5681`) |
| **v0.1.3.b** | **L1 (MoE) — full block**. Upload 128 routed experts (~1.3 GB/chip) + 1 shared (2× wider). `moe_block_eager` does pre-norm + router + per-token dispatch (5 tokens × 6 experts = 30 routings grouped by expert) using relu² activation, weighted-add to routed accumulator; TT shared expert; combine + residual. Pattern A sharding deferred to v0.5 (23 layers × 1.3 GB > 8 GB/chip). | Gate S shared_out cos=0.999713 ✓; Gate M mixer_out=0.999826 ✓; Gate B block_out vs hs[2]=0.999805 ✓. Bootstrap 13.9s, forward 3.8s — all PASS on FIRST run. | ✅ DONE (commit `aadecbc`) |
| **v0.1.4.G0** | **All-to-all spike**. Toy (1,4) sanity test of `ttnn.all_to_all_dispatch` + `ttnn.all_to_all_combine` (8 experts × 4 chips × top_k=2). Forks calling pattern from `~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/moe.py:455 + :487`. Discovered combine's contract: input dim 0 = n_experts/n_devices (post-expert layout `[experts_per_device, B, S, H]`). cluster_axis=1 for our single-row (1,4) mesh. | dispatch_out [1,4,1,128] ✓; combine_out [2,1,1,128] ✓; readback ✓ | ✅ DONE (commit `138df8e`) — TRUE EP path open |
| **v0.1.4** | **MoE Expert-Parallel refactor (True EP, DeepSeek-V3-style)**. 128 experts sharded as 32/chip. Forward = pre-norm → router → `all_to_all_dispatch` → 32 local experts (32 × matmul + relu² + matmul, batched as Pattern A) → `all_to_all_combine` (local_reduce=True for the weighted sum) → shared expert (replicated) → residual. ~6 expert FFNs per token total (vs 128 in Pattern A — 21× less compute). Memory ≤ 7.8 GB/chip → unblocks v0.2+. | v0.1.3.b smoke re-runs with EP forward; 3 gates (S/M/B) all cos ≥ 0.999 | pending (task #215) |
| **v0.2** | All 52 layers (per-layer dispatch via `state.layer_types[L]`) + final_norm + lm_head + argmax. Now fits in memory thanks to v0.1.4. | per-layer cos ≥ 0.999 vs HF oracle; argmax matches HF at pos 0 | pending (blocked by #215) |
| **v0.3** | Multi-step eager decode: KV cache for 6 attn layers + Mamba2 state (conv + fp32 ssm) for 23 mamba layers. Long-context smoke at L=8192. | TT tokens 0..7 match HF token-for-token; needle-haystack at L=8192 ≥75% | pending |

### Phase 2 — Single-stream performance (v0.4 → v0.5)

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v0.4** | **Trace capture — fp32-in-trace risk check** ([[35b-dn-h-state-drift-lever]]). Two-phase warmup ([[ttnn-multi-trace-two-phase-warmup]]). Fallback: bf16 ssm + measure drift, ship if drift <3% at 64 tok. | 100 traced steps == 100 eager token-for-token; ≥3× speedup vs eager | pending |
| **v0.5** | **Single-stream PERF pass** (target ≥30 tok/s traced). Apply known wins: vocab-sharded LM head + on-device argmax (P22 — already proven on 27B/Gemma 4 at +5-8%); HiFi2 expert matmul (35B win); RMSNorm fusion ([[adoption-next-b]]); concatenated Mamba2 in_proj fusion; distributed RMSNorm. Profile-driven (Tracy A/B per win). | ≥30 tok/s single-stream traced. **This is the demo-ready state.** | pending |

### Phase 3 — Continuous batching (DEFERRED until v0.5 ships)

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v1.0-v1.6** | Forks 35B v1.0-v1.6 ladder. Deltas: Mamba2 state per-slot (fp32 ssm + bf16 conv); batched MoE Pattern A with sigmoid+group-topk; batched paged SDPA over per-slot KV. | 3a/3b/3c gates PASS at B=4 | deferred |

### Phase 4 — HTTP server (LAST)

| Stage | Adds | Gate | Status |
|---|---|---|---|
| **v2** | Register `nemotron3_nano` in `cb_api.BACKENDS` + `cb_scheduler._BACKEND_MODULES`. Tokenizer + chat template. | `curl /v1/chat/completions` returns sensible text; multi-turn coherence 3/3 | deferred |

------------------------------------------------------------------------

## 4. Backend wiring (exact edits at v2)

Per `[[feedback-cb-backend-dispatch-holes]]`, register the new backend
in BOTH dicts. Grep every `"27b"`, `"35b"`, `"gemma4_12b"` literal in
`experiments/serve/` before deploying. Deploy the WHOLE
`experiments/serve/` glob with `scripts/deploy.sh`.

```python
# experiments/serve/cb_api.py:50 — BACKENDS dict
BACKENDS = {
    "27b":            ("server_tp",                       "Qwen/Qwen3.6-27B"),
    "35b":            ("server_35b_ttnn",                 "Qwen/Qwen3.6-35B-A3B"),
    "gemma4_12b":     ("server_gemma4_unified_ttnn",      "google/gemma-4-12B"),
    "nemotron3_nano": ("server_nemotron3_nano_ttnn",      "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
}
```

```python
# experiments/serve/cb_scheduler.py:50 — _BACKEND_MODULES dict
_BACKEND_MODULES = {
    "27b":            ("server_tp",                   "server_tp_cb"),
    "35b":            ("server_35b_ttnn",             "server_35b_cb"),
    "gemma4_12b":     ("server_gemma4_unified_ttnn",  "server_gemma4_unified_cb"),
    "nemotron3_nano": ("server_nemotron3_nano_ttnn",  "server_nemotron3_nano_cb"),
}
```

------------------------------------------------------------------------

## 5. Risk register (from brief §7 — restated with project framing)

| Risk | Probability | Severity | Mitigation |
|---|---|---|---|
| Mamba2 SSD kernel does not exist on Blackhole | **CERTAIN** | **CRITICAL** | Path A — manual ttnn composite for v0. Plan §0 decision. |
| fp32 SSM state in trace → 30+ min Blackhole trace hang (35B fp32-H pattern) | medium | high | Validate fp32 path in eager (v0.3) before trace (v0.4). If hangs, fall back to bf16 + measure drift. |
| 2 KV heads on 4 chips — replication / sub-shard decision | medium | medium | Replicate for v0 (1.5 GB/chip budget for 256K context; fits). Revisit at v3 perf pass. |
| Squared ReLU (`relu²`) — fused vs split for precision | medium | low | Probe both compositions via `test_fused_*.py` pattern; prefer split for correctness, revisit fusion at perf pass (Gemma 4 lesson) |
| 30B model weights → ~10-12 min bootstrap | certain | low | Dev harness mandatory from day 1; never restart `serve_cb.sh` per fix |
| Tokenizer file end not captured in research | low | low | On-qb1 `tail -50` of `tokenizer_config.json` at v0.0.1 |
| Chat template active-prompt suffix has TWO branches (`<think>\n` vs `<think></think>`) | medium | medium | Our `_active_prompt_suffix` should cover; verify via `chat_template_invariant.py` test on both branches |
| Param count discrepancy (30B vs 31.6B) | resolved | n/a | Use 32 GB BF16 / 4 chips = 8 GB/chip for memory planning |
| NO RoPE — stale RoPE table application would silently degrade output | low (after audit) | high | Triple-check: q/k go directly q_proj→SDPA with NO intermediate transform. Memory entry predicted at `[[feedback-nemotron3-no-rope-silent-drift]]` |
| Custom modeling code requires `trust_remote_code=True` for HF oracle | certain | low | Set the flag in `hf_reference_nemotron3_nano.py`. Our TT path reads safetensors directly, doesn't need it. |

------------------------------------------------------------------------

## 6. Timeline (Path B — locked)

### Phase 0 — Owned Mamba2 SSD kernel (3-5 weeks)

- G0 read + numpy oracle: 3-4 days (math + research + numpy match against HF)
- G0a isolation harness: 0.5 day
- G1 single-core: 5-7 days (program factory + LLK + first cos gate)
- G2 multi-core: 3-5 days (sharding 64 heads across cores + cross-core reduce avoidance)
- G3 batched: 3-5 days (mirror owned_gdn batched form; B=1..32 gate)
- G4 server integration prep: 1-2 days (kernel ready to call from Python; can't fully integrate until §3b lands the server skeleton)

Subtotal: **~3-5 weeks**

### Phase 1 — Single-stream correctness (1-2 weeks)

Mamba2 is already a single-call kernel rather than a 200+ ms composite,
so bootstrap waits and layer-by-layer cosine ladders dominate.

- v0.0 + v0.0.1 + v0.0.2 (oracle + tokenizer + weights introspect): 1 day
- v0.1.0 (bootstrap on (1,4) + embed + final_norm + lm_head): 1 day
- v0.1.1 (Attention L5 — simplest, warmup): 0.5 day
- v0.1.2 (Mamba2 L0 via owned kernel call): 1-2 days
- v0.1.3 (MoE L1, fork from 35B Pattern A): 1-2 days
- v0.2 (full 52-layer forward via per-layer dispatch): 1-2 days
- v0.3 (multi-step decode + long-context smoke at 8K): 1-2 days

Subtotal: **~1-2 weeks**

### Phase 2 — Single-stream performance (3-5 days)

- v0.4 (trace capture, fp32-in-trace risk check + fallback): 1-3 days
- v0.5 (perf pass — vocab-shard, HiFi2, RMSNorm fusion, etc.): 2-3 days
- buffer / unknowns: 1 day

Subtotal: **~3-5 days**

### Phase 3 — Continuous batching (DEFERRED — 1 week when scheduled)

Per user direction (2026-06-05): finish single-stream correctness +
performance BEFORE touching CB. Same path 27B took.

- v1.0-1.6 (CB at B=4): ~1 week when scheduled

### Phase 4 — HTTP server + chat (LAST — 0.5 day)

### Grand total: 4-6 weeks to demo-ready single-stream (v0.5)
### Optional: +1.5 weeks to CB + HTTP shipped (v2)

Aligned with the brief's §9 estimate. The headline number to track
internally is "Phase 0 G4 lands" as the gating milestone; everything
after is much more predictable.

------------------------------------------------------------------------

## 6.5. Parallel adoption sidecar (NEW 2026-06-04)

While Phase 0 (kernel) runs in the foreground, the qb1 hardware has
idle windows between kernel iterations that we're using to land
**tt-metal adoption wins** from the 3 audits. Tracked separately in
[`research/tt_metal_adoption_plan_2026-06-04.md`](tt_metal_adoption_plan_2026-06-04.md).

- **NOW results (commit `058bedd`)**: task #191 BLOCKED on input-overlap
  shard contract (reverted; bumped to NEXT-F #198 — needs ~30 LOC
  disjoint K/V mem_cfg in `setup_state`); task #192 AUDIT-ONLY (our
  gm4 already clean — no redundant `to_memory_config` calls found).
  See `research/tt_metal_adoption_plan_2026-06-04.md` §2a for the
  detailed findings.
- **NEXT (post-Nemotron G4)**: tasks #193..#198 — GDN bake-off,
  RMSNorm fusion (+12-15 ms/tok), chunk-outer 2048-tok prefill,
  masked fixed-bucket prefill, wider deltanet_recurrence op,
  disjoint K/V mem_cfg for paged_fused (#198, unblocks +1.6 ms/tok).

Zero file overlap with Phase 0 (subagent touches `experiments/serve/
server_gemma4_unified_*` only; main agent touches
`experiments/owned_ops/nemotron3_mamba2_decode_owned/` only).

------------------------------------------------------------------------

## 7. Concrete next steps (Phase 1 — start here)

**Phase 0 DONE 2026-06-05.** Mamba2 SSD owned kernel ships G0..G4 with
the drop-in `mamba2_decode_step_ttnn` wrapper validated at cos=0.999998.
Phase 1 single-stream correctness is now the live workstream.

### Immediate (today)

1. **Task #199 — v0.0 HF oracle artifacts on qb1** [IN PROGRESS]
   - ✅ Weights downloaded (63 GB in 7m50s; 13 shards)
   - ✅ Model loaded to CPU RAM in 8 min (~6243 weight tensors)
   - ✅ Oracle script bug fixed (`model.backbone` not `model.model`)
   - ⏳ Oracle re-launched 2026-06-05 11:50 in tmux `nemotron_oracle`
   - Gate: `.cache/hf_oracle_nemotron3_nano/{hidden_states,logits,final_norm,argmax,prompt_ids,meta}.npy` populated

2. **Task #201 — v0.0.2 weights introspect** (can run in parallel with #199)
   - `experiments/utils/nemotron3_weights_introspect.py` (~50 LOC)
   - Reads 13 safetensors shard headers; audits all expected shapes match the brief
   - Gate: zero unexpected keys; all shapes match config

3. **Task #200 — v0.0.1 tokenizer verification** (blocks on #199 for sanity check tokens)
   - On-qb1 inspection + `chat_template_invariant.py` gate
   - Verify `<think>\n` AND `<think></think>` active-prompt branches

### Subsequent (gated)

4. **Task #202 — v0.1.0 bootstrap** (after #199, #200, #201).
5. **Task #203 — v0.1.1 L5 Attention** (warmup, simplest layer first).
6. **Task #204 — v0.1.2 L0 Mamba2** (drop in wrapper).
7. **Task #205 — v0.1.3 L1 MoE** (Pattern A fork).
8. **Task #206 — v0.2 full 52-layer forward.**
9. **Task #207 — v0.3 multi-step + long-context.**
10. **Task #208 — v0.4 trace capture.**
11. **Task #209 — v0.5 single-stream perf pass** (demo-ready milestone).
12. **Task #210 — v1.x continuous batching (DEFERRED).**
13. **Task #211 — v2 HTTP wire-up (LAST).**

Each task gates the next. **Do NOT touch CB or HTTP until v0.5 ships.**

------------------------------------------------------------------------

## 8. Open questions for the user

1. ~~Path A / B / C~~ **RESOLVED 2026-06-04**: Path B — owned kernel up-front.
2. ~~Phase ordering~~ **RESOLVED 2026-06-05**: 27B path — single-stream
   correctness → single-stream perf → batching → HTTP. CB and HTTP
   deprioritised until v0.5 demo-ready.
3. ~~L5-vs-L0 order in v0.1.x~~ **RESOLVED 2026-06-05**: L5 (Attention)
   first as warmup (simplest layer; builds bootstrap + paged SDPA + KV
   cache scaffold before adding Mamba2 complexity on top).
4. **Long-context priority** — open. Default plan: ship v0..v0.5 at 8K,
   push 256K to a separate workstream after v0.5 lands.
5. **fp32-in-trace fallback** — open. If Blackhole hangs on fp32 ssm
   state in trace (mirroring 35B fp32-H), the default fallback is bf16
   ssm + measure drift; ship if drift <3% at 64 tok.
6. **Server scaffold base** — open. Recommended: fork
   `server_35b_ttnn.py` (closest structural match: hybrid MoE + recurrent
   layer model). Alternative: `server_gemma4_unified_ttnn.py` (newer,
   hybrid-dispatch sliding+global).

------------------------------------------------------------------------

## 9. Memory entries predicted during bringup (will populate)

- `feedback_nemotron3_no_rope_silent_drift` — stray RoPE application
  degrading cosine without crashing
- `feedback_mamba2_ssm_fp32_state_in_trace` — Blackhole trace behaviour
  with fp32 recurrent state (mirror of 35B fp32-H)
- `feedback_nemotron3_moe_sigmoid_group_router` — diff from softmax
  routing; how it interacts with our Pattern A batched matmul
- `feedback_nemotron3_relu_squared` — fused vs split decision
- `feedback_nemotron3_thinking_template_double_branch` — `<think>\n`
  vs `<think></think>` active-prompt variants vs our `_active_prompt_suffix`

------------------------------------------------------------------------

## Sources

All architecture claims trace to
[`research/nemotron3_nano_architecture_brief.md`](nemotron3_nano_architecture_brief.md)
§2 (source table). Recipe + REUSE mandate from
[`research/model_bringup_recipe.md`](model_bringup_recipe.md). Reference
plans:
- [`research/35b_cb_bringup_plan.md`](35b_cb_bringup_plan.md) — close MoE precedent
- [`research/gemma4_12b_bringup_plan.md`](gemma4_12b_bringup_plan.md) — close hybrid-dispatch + Llama-RMSNorm precedent
