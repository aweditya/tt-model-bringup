#!/usr/bin/env python3
"""
Experiment 91o — HF transformers oracle for Qwen3-Next layer 0.

After B'9 didn't fix the 'FR' fixed-point and 91n cleared lm_head/embed,
the suspect is the layer math itself. Our numpy ref might be co-buggy
with our ttnn impl. This script uses HF's official Qwen3NextDecoderLayer
as an external oracle and compares its layer-0 output to our numpy ref.

Phase 1: Probe HF transformers — confirm classes exist, print signatures
Phase 2: Load config + embed weights + layer 0 weights from safetensors
Phase 3: Forward "The capital of France is" → embed → layer 0 (CPU fp32)
Phase 4: Compare to our numpy reference (~/tt-xla/.cache/qwen36_27b_layer0_3_ref.npz)

Run on qb2:
    cd ~/tt-xla && HF_HOME=$HOME/tt-xla/.cache/hf .venv/bin/python \
        experiments/91o_hf_reference_layer0.py
"""
import os, sys, json, inspect, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
import transformers

MODEL_ID = "Qwen/Qwen3.6-27B"
NUMPY_REF_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_layer0_3_ref.npz")
OUT_PATH = os.path.expanduser("~/tt-xla/.cache/qwen36_27b_hf_layer0_ref.npz")
PROMPT = "The capital of France is"


def cosine(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    print("=" * 64)
    print("Experiment 91o — HF transformers oracle for Qwen3-Next layer 0")
    print("=" * 64)
    print(f"transformers version: {transformers.__version__}")

    # ----------------------------------------
    # Phase 1: Probe
    # ----------------------------------------
    print("\n[1/4] Probing HF Qwen3-Next availability…")
    try:
        from transformers.models.qwen3_next import modeling_qwen3_next as mqn
    except ImportError as e:
        print(f"  ImportError: {e}")
        print("  Available qwen3* in transformers.models:")
        import transformers.models as tm
        for n in sorted(dir(tm)):
            if 'qwen' in n.lower():
                print(f"    {n}")
        sys.exit(1)

    print(f"  module: {mqn.__file__}")
    classes_of_interest = sorted(c for c in dir(mqn)
                                  if 'Layer' in c or 'Model' in c or 'Rotary' in c
                                  or 'Attention' in c or 'RMSNorm' in c)
    print(f"  relevant classes:")
    for c in classes_of_interest:
        print(f"    {c}")

    DecoderLayer = getattr(mqn, 'Qwen3NextDecoderLayer', None)
    if DecoderLayer is None:
        print("  Qwen3NextDecoderLayer not found")
        sys.exit(1)
    print(f"\n  Qwen3NextDecoderLayer.__init__: {inspect.signature(DecoderLayer.__init__)}")
    print(f"  Qwen3NextDecoderLayer.forward:  {inspect.signature(DecoderLayer.forward)}")

    # Try to find a rotary embedding helper
    RotaryEmb = getattr(mqn, 'Qwen3NextRotaryEmbedding', None)
    if RotaryEmb is not None:
        print(f"  Qwen3NextRotaryEmbedding.__init__: {inspect.signature(RotaryEmb.__init__)}")
        print(f"  Qwen3NextRotaryEmbedding.forward:  {inspect.signature(RotaryEmb.forward)}")
    else:
        print("  Qwen3NextRotaryEmbedding NOT found — will need to construct cos/sin manually")

    # ----------------------------------------
    # Phase 2: Load config + weights for layer 0
    # ----------------------------------------
    print("\n[2/4] Loading config + weights…")
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_cfg = getattr(full_cfg, 'text_config', full_cfg)
    print(f"  text_config: hidden={text_cfg.hidden_size}, vocab={text_cfg.vocab_size}, "
          f"layers={text_cfg.num_hidden_layers}, head_dim={text_cfg.head_dim}")

    # Identify what 'layer 0' is — linear_attention vs full_attention.
    # Qwen3-Next config typically has 'layer_types' or computed via index modulo.
    layer_types = getattr(text_cfg, 'layer_types', None)
    if layer_types is not None:
        print(f"  layer_types[0:5]: {layer_types[:5]}")
        layer_0_type = layer_types[0]
    else:
        print(f"  no layer_types in config; will determine by attribute existence after load")
        layer_0_type = "linear_attention"  # our assumption

    # Read safetensors index
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)['weight_map']

    # Embed
    embed_key = "model.language_model.embed_tokens.weight"
    embed_shard = weight_map[embed_key]
    embed_path = hf_hub_download(MODEL_ID, embed_shard)
    with safe_open(embed_path, framework="pt") as f:
        embed_w = f.get_tensor(embed_key).float()
    print(f"  embed weight loaded: {tuple(embed_w.shape)}")

    embed = torch.nn.Embedding(text_cfg.vocab_size, text_cfg.hidden_size)
    embed.weight.data.copy_(embed_w)
    embed = embed.float().eval()

    # Layer 0
    layer = DecoderLayer(text_cfg, layer_idx=0).float().eval()
    layer_keys = sorted(k for k in weight_map.keys()
                        if k.startswith('model.language_model.layers.0.'))
    print(f"  layer 0 has {len(layer_keys)} weight tensors in safetensors:")
    for k in layer_keys:
        print(f"    {k}  →  shard={weight_map[k]}")

    layer_state = {}
    shards_needed = sorted(set(weight_map[k] for k in layer_keys))
    for shard in shards_needed:
        shard_path = hf_hub_download(MODEL_ID, shard)
        with safe_open(shard_path, framework="pt") as f:
            for k in layer_keys:
                if weight_map[k] == shard:
                    local_key = k.replace('model.language_model.layers.0.', '')
                    layer_state[local_key] = f.get_tensor(k).float()
    print(f"  loaded {len(layer_state)} tensors")

    # Inspect layer's expected state_dict keys
    expected = set(layer.state_dict().keys())
    print(f"  layer expects {len(expected)} keys")

    # Show diffs
    missing = expected - set(layer_state.keys())
    extra = set(layer_state.keys()) - expected
    print(f"  missing keys: {sorted(missing)}")
    print(f"  extra keys:   {sorted(extra)}")

    # Load (strict=False to handle any naming differences)
    info = layer.load_state_dict(layer_state, strict=False)
    print(f"  load_state_dict missing={info.missing_keys}")
    print(f"  load_state_dict unexpected={info.unexpected_keys}")

    # ----------------------------------------
    # Phase 3: Forward
    # ----------------------------------------
    print("\n[3/4] Forwarding prompt through embed → layer 0 (CPU fp32)…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = torch.tensor([tok.encode(PROMPT)])
    print(f"  input_ids: {input_ids.tolist()}")
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        hidden_states = embed(input_ids).float()  # [1, seq, hidden]
        print(f"  embed output: shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype}")

        # Construct attention_mask / position args based on forward signature
        sig = inspect.signature(layer.forward)
        params = list(sig.parameters.keys())
        print(f"  layer.forward params: {params}")

        # Try assembling the args HF needs
        forward_kwargs = {}

        # position_ids: [0..seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq]
        if 'position_ids' in params:
            forward_kwargs['position_ids'] = position_ids

        # cache_position: same as position_ids for first forward
        if 'cache_position' in params:
            forward_kwargs['cache_position'] = torch.arange(seq_len, dtype=torch.long)

        # attention_mask: causal — we let HF compute it from cache_position if None,
        # otherwise build a basic causal mask
        if 'attention_mask' in params:
            forward_kwargs['attention_mask'] = None

        # past_key_value / past_key_values: None for first call
        for k in ('past_key_value', 'past_key_values'):
            if k in params:
                forward_kwargs[k] = None

        # use_cache: False (we don't want HF to manage a cache here)
        if 'use_cache' in params:
            forward_kwargs['use_cache'] = False

        # output_attentions: False
        if 'output_attentions' in params:
            forward_kwargs['output_attentions'] = False

        # position_embeddings: HF often passes a (cos, sin) tuple
        if 'position_embeddings' in params:
            if RotaryEmb is not None:
                rot = RotaryEmb(config=text_cfg)
                rot = rot.float().eval()
                with torch.no_grad():
                    pe = rot(hidden_states, position_ids)
                # rot.forward typically returns (cos, sin)
                if isinstance(pe, tuple):
                    forward_kwargs['position_embeddings'] = (pe[0].float(), pe[1].float())
                else:
                    forward_kwargs['position_embeddings'] = pe
                print(f"  position_embeddings: cos shape={pe[0].shape}, sin shape={pe[1].shape}")
            else:
                print("  WARNING: position_embeddings expected but no RotaryEmbedding helper found")
                forward_kwargs['position_embeddings'] = None

        print(f"  calling layer.forward with kwargs: {list(forward_kwargs.keys())}")

        t0 = time.time()
        result = layer(hidden_states, **forward_kwargs)
        dt = time.time() - t0
        print(f"  forward took {dt:.2f}s")

        # result may be a tuple or a single tensor
        if isinstance(result, tuple):
            hf_layer0 = result[0]
            print(f"  result is tuple with {len(result)} elements; using result[0]")
        else:
            hf_layer0 = result
        print(f"  HF layer 0 output: shape={tuple(hf_layer0.shape)}")

        hf_layer0_np = hf_layer0.float().cpu().numpy()
        if hf_layer0_np.ndim == 3:
            hf_layer0_np = hf_layer0_np[0]  # drop batch
        print(f"  HF layer 0 np: shape={hf_layer0_np.shape}  ‖row_-1‖={np.linalg.norm(hf_layer0_np[-1]):.4f}")

    # ----------------------------------------
    # Phase 4: Compare to numpy reference
    # ----------------------------------------
    print("\n[4/4] Comparing to numpy reference…")
    if not os.path.exists(NUMPY_REF_PATH):
        print(f"  no numpy ref at {NUMPY_REF_PATH}, skipping comparison")
        np.savez(OUT_PATH, hf_layer0=hf_layer0_np, input_ids=input_ids.numpy())
        print(f"  HF reference saved to {OUT_PATH}")
        return

    ref = np.load(NUMPY_REF_PATH)
    print(f"  numpy ref keys: {list(ref.keys())}")

    # Find a layer-0 output key — try several candidates
    candidates = ['layer_0', 'layer0_out', 'x_after_layer_0', 'layer_0_out',
                  'h_after_layer_0', 'h_layer_0', 'l0_out']
    numpy_layer0 = None
    for c in candidates:
        if c in ref:
            numpy_layer0 = ref[c]
            print(f"  using key {c!r}: shape={numpy_layer0.shape}")
            break
    if numpy_layer0 is None:
        print(f"  no recognizable layer-0 key in numpy ref. Available: {list(ref.keys())}")
        print(f"  saving HF ref anyway for offline inspection")
        np.savez(OUT_PATH, hf_layer0=hf_layer0_np, input_ids=input_ids.numpy())
        return

    # Both should be [seq, hidden] after squeezing
    if numpy_layer0.ndim == 1:
        # The numpy ref might only have a single token's hidden state (if B'2 ran one token)
        numpy_vec = numpy_layer0
        hf_vec = hf_layer0_np[-1] if hf_layer0_np.ndim == 2 else hf_layer0_np
    elif numpy_layer0.ndim == 2:
        # [seq, hidden]
        numpy_vec = numpy_layer0[-1]
        hf_vec = hf_layer0_np[-1]
    else:
        numpy_vec = numpy_layer0.flatten()
        hf_vec = hf_layer0_np.flatten()

    print(f"  shapes: numpy={numpy_vec.shape}, HF={hf_vec.shape}")
    print(f"  norms:  numpy ‖·‖={np.linalg.norm(numpy_vec):.4f}, HF ‖·‖={np.linalg.norm(hf_vec):.4f}")
    if numpy_vec.shape == hf_vec.shape:
        cos = cosine(numpy_vec, hf_vec)
        maxabs = float(np.abs(numpy_vec.astype(np.float64) - hf_vec.astype(np.float64)).max())
        print(f"\n  ┌─────────────────────────────────────")
        print(f"  │ cosine(HF, numpy) = {cos:.6f}")
        print(f"  │ max|Δ|            = {maxabs:.6f}")
        print(f"  │")
        if cos >= 0.999:
            verdict = "numpy ref is CORRECT → bug is in ttnn impl"
        elif cos >= 0.9:
            verdict = "numpy ref has MINOR bug → localize per substep"
        elif cos >= 0.5:
            verdict = "numpy ref has MAJOR bug → re-derive from HF source"
        else:
            verdict = "numpy ref is essentially UNCORRELATED with HF → severe wiring error"
        print(f"  │ VERDICT: {verdict}")
        print(f"  └─────────────────────────────────────")

    np.savez(OUT_PATH, hf_layer0=hf_layer0_np, input_ids=input_ids.numpy())
    print(f"\n  HF reference saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
