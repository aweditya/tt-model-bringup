#!/usr/bin/env python3
"""
Experiment 91 — Phase B′1 weight skeleton for Qwen3.6-27B.

Loads ONLY the model's config.json + safetensors.index.json. Verifies:
  1. Layer pattern matches [L L L F] × 16 (48 DeltaNet + 16 Gated Attention)
  2. Total parameter count (~27B)
  3. Per-component bf8 memory budget against actual safetensors metadata
  4. Confirms we fit one Blackhole P150 (~30 GB usable)

Architecture cross-check: Qwen3.6-27B differs from 35B-A3B in:
  - Dense MLP (intermediate_size 17408) instead of 256-expert MoE
  - hidden 5120 (vs 2048)
  - 64 layers in 16 repeats of [L L L F] (vs 10 repeats = 40 layers)
  - 48 V-heads, 16 K-heads in DeltaNet
  - 24 Q heads, 4 KV heads in Gated Attention, GQA 6:1

Run on qb1:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python experiments/91_qwen36_27b_weight_skeleton.py
"""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~"))

from huggingface_hub import hf_hub_download

MODEL_ID = "Qwen/Qwen3.6-27B"

# Expected architecture (from config.json read 2026-05-11)
N_LAYERS_EXPECTED = 64
PATTERN_REPEATS = 16
PATTERN = ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']


def categorize(param_name: str) -> str:
    n = param_name
    if 'vision' in n or 'visual' in n:
        return 'vision'
    if 'embed_tokens' in n or 'embed' in n.split('.')[-1]:
        return 'embed'
    if 'lm_head' in n:
        return 'lm_head'
    if 'norm' in n.split('.')[-1] or 'layernorm' in n:
        return 'norm'
    if 'gated_delta_net' in n or 'linear_attn' in n or 'A_log' in n or 'dt_bias' in n:
        return 'deltanet'
    if any(k in n for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        return 'attn'
    if 'mlp.' in n and 'experts' not in n:
        return 'mlp_dense'
    return 'other'


def main():
    print("=" * 64)
    print(f"Phase B′1 — Qwen3.6-27B weight skeleton")
    print("=" * 64)

    print(f"\n[1/4] Fetching config.json + index.json for {MODEL_ID}…")
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(idx_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']
    n_shards = len(set(weight_map.values()))
    text_cfg = cfg.get('text_config', cfg)

    print(f"  model_type:    {cfg.get('model_type')}")
    print(f"  total tensors: {len(weight_map)}")
    print(f"  total shards:  {n_shards}")

    HIDDEN = text_cfg.get('hidden_size')
    INTERMEDIATE = text_cfg.get('intermediate_size')
    NUM_LAYERS = text_cfg.get('num_hidden_layers')
    VOCAB = text_cfg.get('vocab_size')
    HEAD_DIM = text_cfg.get('head_dim')
    N_Q_HEADS = text_cfg.get('num_attention_heads')
    N_KV_HEADS = text_cfg.get('num_key_value_heads')
    LIN_K_HEADS = text_cfg.get('linear_num_key_heads')
    LIN_V_HEADS = text_cfg.get('linear_num_value_heads')
    LIN_K_DIM = text_cfg.get('linear_key_head_dim')
    LIN_V_DIM = text_cfg.get('linear_value_head_dim')

    print(f"\n  hidden_size:        {HIDDEN}")
    print(f"  num_hidden_layers:  {NUM_LAYERS}")
    print(f"  intermediate_size:  {INTERMEDIATE}  (dense MLP)")
    print(f"  vocab_size:         {VOCAB}")
    print(f"  Full attn: Q heads {N_Q_HEADS}, KV heads {N_KV_HEADS}, head_dim {HEAD_DIM}")
    print(f"  DeltaNet:  K heads {LIN_K_HEADS}, V heads {LIN_V_HEADS}, "
          f"k_dim {LIN_K_DIM}, v_dim {LIN_V_DIM}")

    # ── 2/4 Layer pattern verification ──────────────────────────
    print(f"\n[2/4] Verifying layer pattern matches [L L L F] × {PATTERN_REPEATS}…")
    layer_types = text_cfg.get('layer_types', [])
    assert len(layer_types) == N_LAYERS_EXPECTED, \
        f"Expected {N_LAYERS_EXPECTED} layers, got {len(layer_types)}"
    expected = PATTERN * PATTERN_REPEATS
    assert layer_types == expected, "Layer pattern mismatch!"
    n_deltanet = layer_types.count('linear_attention')
    n_fullattn = layer_types.count('full_attention')
    print(f"  ✓ {N_LAYERS_EXPECTED} layers in {PATTERN_REPEATS} repeats of [L L L F]")
    print(f"  ✓ {n_deltanet} DeltaNet (linear) + {n_fullattn} full attention")

    # ── 3/4 Categorize tensors ──────────────────────────────────
    print(f"\n[3/4] Categorizing {len(weight_map)} tensors…")
    cat_count = defaultdict(int)
    layers_seen = set()
    for name, shard in weight_map.items():
        c = categorize(name)
        cat_count[c] += 1
        if '.layers.' in name:
            try:
                layer_n = int(name.split('.layers.')[1].split('.')[0])
                layers_seen.add(layer_n)
            except (ValueError, IndexError):
                pass

    print(f"\n  Category         #tensors")
    for c in ['embed', 'lm_head', 'norm', 'deltanet', 'attn', 'mlp_dense', 'vision', 'other']:
        if cat_count[c]:
            print(f"  {c:15s}  {cat_count[c]:6d}")

    expected_layers = set(range(N_LAYERS_EXPECTED))
    missing = expected_layers - layers_seen
    if missing:
        print(f"  ⚠ missing layers: {sorted(missing)}")
    else:
        print(f"  ✓ all {N_LAYERS_EXPECTED} layers present")

    # ── 4/4 Memory budget (bf8) ─────────────────────────────────
    print(f"\n[4/4] Memory budget estimate (bf8 = 1 byte/param):")

    # DeltaNet: in_proj_qkv (hidden → 2*key_dim + value_dim), in_proj_z (→ value_dim),
    # in_proj_a (→ n_v_heads), in_proj_b (→ n_v_heads), out_proj (value_dim → hidden),
    # conv1d (linear_conv_kernel_dim=4), small A_log + dt_bias.
    key_dim = LIN_K_HEADS * LIN_K_DIM      # 16 * 128 = 2048
    value_dim = LIN_V_HEADS * LIN_V_DIM    # 48 * 128 = 6144
    deltanet_per_layer = (
        HIDDEN * (2 * key_dim + value_dim) +  # in_proj_qkv
        HIDDEN * value_dim +                   # in_proj_z
        HIDDEN * LIN_V_HEADS * 2 +             # in_proj_a + in_proj_b
        value_dim * HIDDEN +                   # out_proj
        (2 * key_dim + value_dim) * 4          # conv1d kernel=4
    )
    deltanet_total = n_deltanet * deltanet_per_layer

    # Gated Attention: q_proj is 2× width (Q + gate), k_proj, v_proj, o_proj
    q_dim = N_Q_HEADS * HEAD_DIM             # 24 * 256 = 6144
    kv_dim = N_KV_HEADS * HEAD_DIM           # 4 * 256 = 1024
    attn_per_layer = (
        HIDDEN * q_dim * 2 +                 # q_proj (Q + gate)
        HIDDEN * kv_dim +                    # k_proj
        HIDDEN * kv_dim +                    # v_proj
        q_dim * HIDDEN                       # o_proj
    )
    attn_total = n_fullattn * attn_per_layer

    # Dense MLP every layer: gate, up, down (SwiGLU)
    mlp_per_layer = 3 * HIDDEN * INTERMEDIATE
    mlp_total = NUM_LAYERS * mlp_per_layer

    embed_params = VOCAB * HIDDEN
    lm_head_params = VOCAB * HIDDEN if not text_cfg.get('tie_word_embeddings', False) else 0
    norm_params = NUM_LAYERS * 2 * HIDDEN + HIDDEN

    total_params = deltanet_total + attn_total + mlp_total + embed_params + lm_head_params + norm_params

    print(f"  Embeddings (in):          {embed_params/1e9:6.2f} B → {embed_params/1e9:6.2f} GB (bf8)")
    print(f"  Output head:              {lm_head_params/1e9:6.2f} B → {lm_head_params/1e9:6.2f} GB")
    print(f"  DeltaNet × {n_deltanet}:    {deltanet_total/1e9:6.2f} B → {deltanet_total/1e9:6.2f} GB")
    print(f"  Gated Attn × {n_fullattn}: {attn_total/1e9:6.2f} B → {attn_total/1e9:6.2f} GB")
    print(f"  Dense MLP × {NUM_LAYERS}:   {mlp_total/1e9:6.2f} B → {mlp_total/1e9:6.2f} GB")
    print(f"  RMSNorms etc:             {norm_params/1e9:6.2f} B → {norm_params/1e9:6.2f} GB")
    print(f"  {'-' * 56}")
    print(f"  TOTAL:                    {total_params/1e9:6.2f} B → {total_params/1e9:6.2f} GB (bf8)")
    print(f"  Expected (model card): ~27 B")

    # ── KV cache & runtime memory ───────────────────────────────
    print(f"\n  Runtime memory (one P150, ~30 GB usable):")
    for ctx in [1024, 4096, 8192, 16384]:
        kv_bf16 = n_fullattn * 2 * N_KV_HEADS * HEAD_DIM * ctx * 2
        kv_bf8 = n_fullattn * 2 * N_KV_HEADS * HEAD_DIM * ctx
        # DeltaNet state H: n_v_heads × k_dim × v_dim × 4 bytes (fp32) per layer
        H_bytes = n_deltanet * LIN_V_HEADS * LIN_K_DIM * LIN_V_DIM * 4
        total_bf16_kv = total_params + kv_bf16 + H_bytes + 2e9
        total_bf8_kv = total_params + kv_bf8 + H_bytes + 2e9
        print(f"    ctx={ctx:5d}: bf16 KV → {total_bf16_kv/1e9:5.1f} GB  |  bf8 KV → {total_bf8_kv/1e9:5.1f} GB"
              f"  {'✓' if total_bf16_kv < 30e9 else '✗ bf16'}")

    fits = total_params / 1e9 < 30
    if fits:
        headroom = 30 - total_params / 1e9
        print(f"\n  ✓ Weights fit one chip with ~{headroom:.1f} GB headroom")
    else:
        print(f"\n  ✗ Weights exceed one chip by {total_params/1e9 - 30:.1f} GB at bf8")

    print(f"\n=== B′1 complete ===")
    print(f"  Next: B′2 — write numpy fp32 reference for first 2 layers")


if __name__ == "__main__":
    main()
