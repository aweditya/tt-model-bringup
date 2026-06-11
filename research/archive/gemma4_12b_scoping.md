# Gemma 4 12B — bringup scoping

Released 2026-06-03. Original capture from the HuggingFace model card
(`https://huggingface.co/google/gemma-4-12B`). **All architecture
numbers verified against `config.json` 2026-06-03** — see the deeper
plan-of-action at [`gemma4_12b_bringup_plan.md`](gemma4_12b_bringup_plan.md)
for `file:line` reuse mapping and per-sub-task validation gates.

## Architecture (verified from config.json)

| | |
|---|---|
| Params | 11.95B (dense, no MoE; `enable_moe_block=false`) |
| Layers | 48 |
| Hidden | 3840 |
| Heads (Q/KV sliding / KV global) | 16 / 8 / 1 |
| head_dim (sliding / global) | 256 / 512 (NEW: dual head dim) |
| Intermediate (MLP) | 15360 |
| Vocab | 262144 (vs 35B's 248320, 27B's 152064/248320) |
| `max_position_embeddings` | **131072 (128K)** — the model card's "256K" appears to anticipate RoPE scaling we do NOT need for v0 |
| Attention | Hybrid: 5 sliding (window=1024) + 1 full, repeating 8× = 48 layers. **L47 is full** |
| Multimodal | Encoder-free; raw image/audio projected via linear into the LLM embedding. **Text-only path supported as `Gemma4UnifiedTextModel`** |
| Position encoding (sliding) | default RoPE, theta=10000, full rotation |
| Position encoding (global) | "proportional" p-RoPE, theta=1000000, `partial_rotary_factor=0.25` over global head_dim=512 → 128 dims rotated |
| RMSNorm convention | **Llama-style `w`** (NOT Qwen's `(1+w)` — important distinction from 35B/27B) |
| Decoder norms | **FOUR per layer** (input/post-attn/pre-mlp/post-mlp) — Gemma 2 pattern |
| MLP activation | `gelu_pytorch_tanh` (gated SwiGLU shape, NOT SiLU) |
| `attention_k_eq_v` | true (GLOBAL layers only — V := K after norm+RoPE) |
| `tie_word_embeddings` | true — lm_head = embed.T |
| `final_logit_softcapping` | 30.0 — `logits = 30·tanh(logits/30)` before sampling |
| Embed scaling | `sqrt(hidden)` ≈ 61.97 at embedding lookup |
| Weights | bf16, Apache 2.0 |

## Bringup positioning vs. existing backends

| Backend | Pattern that transfers |
|---|---|
| `server_tp.py` (27B dense) | Dense forward, vocab-sharded lm_head, paged SDPA per-layer KV. Closest skeleton for Gemma 4's text-only path. |
| `server_35b_ttnn.py` (35B MoE+DN) | The `state.layer_types`-based per-layer dispatch is exactly what Gemma 4's sliding/global mix needs. Adapt that mechanism, swap DN-vs-attn for sliding-vs-global. |
| MoE infra (Pattern A batched experts) | Not used — Gemma 4 is dense. |

So Gemma 4 12B is **architecturally closer to 27B than 35B** for our
purposes, modulo the sliding-window + global hybrid which inherits
the *dispatch* shape from 35B's hybrid linear-attn + full-attn mix.

## Scoping (text-only first)

Strictly text. No image/audio paths — Gemma 4's multimodal is
encoder-free linear projections into the same embedding space, so a
text-only mode should be a simple matter of "only feed token
embeddings", but needs verification that no shared layer assumes the
multimodal input is present.

| Sub-task | Effort | Reuses from |
|---|---|---|
| `server_gemma4_ttnn.py` skeleton (config load, mesh boot, weight upload) | 1 day | 27B `server_tp.bootstrap` |
| Layer-type dispatch (sliding 1024 vs global p-RoPE) | 2-3 days | 35B `layer_types` pattern |
| Sliding window SDPA (chunked over the 1024 window) | 2-3 days | Existing chunked SDPA work for 27B (task #108-#118) |
| Global-layer p-RoPE | 1-2 days | Existing partial RoPE (`_apply_partial_rope`) — needs the "proportional" twist; read the Gemma paper before writing this |
| 262K-vocab embed + lm_head shard | 0.5 day | 27B vocab-sharded lm_head (248K → 262K is mechanical) |
| HF oracle (12B, 256K context cap) | 0.5 day | `hf_reference_35b.py` template |
| Cosine ladder vs HF | 0.5 day | `experiments/utils/cosine_ladder_*.py` template |
| `server_gemma4_cb.py` + cb_api `BACKENDS` registration | 1 day | 27B `server_tp_cb` + cb_api MM1 |
| End-to-end chat smoke through `/v1/chat/completions` | 0.5 day | Existing CB35 prod gate workflow |

**Total**: ~9-12 working days from a clean start, assuming no
novel-architecture surprises. Bootstrap time ~10-12 min (smaller
than 35B's 14 min).

## Risks / open questions

1. **Sliding window attention**: the existing paged SDPA assumes
   global attention. Sliding-window-per-layer needs either (a) a
   per-layer SDPA call with a windowed K/V slice, or (b) a custom
   sliding-window kernel. (a) is cheaper to start.
2. **p-RoPE**: "proportional" — read the technical report for the
   exact formula before writing it; do NOT extrapolate from "this
   sounds like our partial RoPE".
3. **Multimodal text-only mode**: verify the text-only path is a
   first-class supported mode (HF docs may say "use the text-only
   template + skip image preprocessing"). If it's not — we'd need
   to verify the LLM stack doesn't require the image projection
   tokens to be present.
4. **256K context vs our MAX_KV**: current `server_35b_ttnn` caps at
   4K (MAX_KV=4096). Gemma 4 advertises 256K. Initial bringup can
   cap at 8K or 16K to match our infrastructure; long-context push
   is a separate workstream (and runs into the open 35B drift cliff
   work — likely shared mechanism).
5. **Memory footprint**: 12B × bf16 = ~24 GB → ~6 GB/chip on our
   (1,4) mesh. Plenty of room.

## Scheduling

**This is a follow-on task — NOT to start until:**

1. Task #163 (35B long-context drift cliff) is resolved or
   triaged off the critical path. The drift investigation is
   actively running on the dev harness; do not interleave.
2. Task #164 (manual recurrence path repair) is decided (fix or
   defer).
3. Optionally: task #162 (35B B>1 batched forward empty-slot fix)
   for proper multi-client serving.

Bringup work would land as task #165 with its own sub-tasks
mirroring the 35B v0..v4 staging (single-slot → batched → traced →
prod wire-up).

## What I'm NOT scoping here

- The image/audio paths. Strictly text. Multimodal is a separate
  project-scope decision (mesh memory, host I/O, vision/audio
  tokenizers, etc.).
- Long context (>16K) for the initial bringup. 8-16K initial,
  push later.
- Performance optimization. Get correctness first.
