#!/usr/bin/env python3
"""B16-profile — measure per-step / per-block cost of the numpy forward.

Imports server_35b's forward primitives and runs N tokens, accumulating
wall time per:
  - DN block (linear_attention)
  - attn block (full_attention)
  - MoE block
  - per-layer totals
  - prefill vs decode

Output: identifies which sub-blocks dominate the 291 ms/tok decode rate
so B16 ttnn-on-mesh refactor can target the biggest matmuls first.

Run (qb1 server should be running so the model is cached, but we re-import
+ run our own forward; doesn't touch the server's socket):
    ssh qb1 'cd ~/tt-xla && .venv/bin/python -u \\
        experiments/utils/profile_35b_forward.py'
"""
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_35b as srv  # noqa: E402


def main():
    print("[1] bootstrap (load 40-layer weights to RAM)…")
    state = srv.State()
    # mock log
    def log(msg): print(f"  {msg}")

    # Don't open mesh for pure-numpy profile
    import ttnn
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, srv.NCHIPS))

    # Config + tokenizer + rotary
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(srv.MODEL_ID, trust_remote_code=True)
    state.text_cfg = cfg.text_config
    state.text_cfg.dtype = torch.bfloat16
    state.layer_types = list(state.text_cfg.layer_types)
    state.tokenizer = AutoTokenizer.from_pretrained(srv.MODEL_ID)
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    state.rotary = Qwen3_5MoeTextRotaryEmbedding(state.text_cfg)
    state.rotary.eval()

    # Weights
    t0 = time.time()
    key_to_shard = srv.build_key_to_shard()
    state.embed_w = srv.load_t(key_to_shard, "model.language_model.embed_tokens.weight")
    state.final_norm_w = srv.load_t(key_to_shard, "model.language_model.norm.weight")
    state.lm_head_w = srv.load_t(key_to_shard, "lm_head.weight")
    state.per_layer = []
    for L in range(state.text_cfg.num_hidden_layers):
        state.per_layer.append(srv.load_layer_weights(key_to_shard, L))
    state.reset_caches()
    print(f"  weights loaded in {time.time()-t0:.1f}s")

    print("\n[2] timed forward — instrumented layer_forward…")
    # Replace srv.layer_forward with a timed version using local override.
    block_times = defaultdict(list)
    orig_dn = srv.dn_layer_forward
    orig_attn = srv.attn_layer_forward
    orig_moe = srv.moe_layer_forward
    orig_qwen35rms = srv.qwen35_rms_norm

    def timed_dn(h, sd, dn_state):
        t = time.time(); out = orig_dn(h, sd, dn_state); block_times["dn"].append(time.time() - t)
        return out
    def timed_attn(h, sd, kv, c, s):
        t = time.time(); out = orig_attn(h, sd, kv, c, s); block_times["attn"].append(time.time() - t)
        return out
    def timed_moe(h, sd):
        t = time.time(); out = orig_moe(h, sd); block_times["moe"].append(time.time() - t)
        return out
    def timed_rms(x, w, eps=srv.EPS):
        t = time.time(); out = orig_qwen35rms(x, w, eps); block_times["rmsnorm"].append(time.time() - t)
        return out

    srv.dn_layer_forward = timed_dn
    srv.attn_layer_forward = timed_attn
    srv.moe_layer_forward = timed_moe
    srv.qwen35_rms_norm = timed_rms

    prompt_ids = state.tokenizer.encode("The capital of France is")
    print(f"  prompt: {prompt_ids}")
    h = None
    t0 = time.time()
    for step, tid in enumerate(prompt_ids):
        h = state.embed_w[tid].reshape(1, srv.HIDDEN).astype(np.float32)
        h = srv.step_forward(state, h, step)
    prefill_wall = time.time() - t0
    print(f"  prefill {len(prompt_ids)} tok: {prefill_wall*1000:.1f} ms = {prefill_wall*1000/len(prompt_ids):.1f} ms/tok")

    # Snapshot prefill block times, then clear for decode
    prefill_block_times = {k: list(v) for k, v in block_times.items()}
    block_times.clear()

    # Decode 5 tokens
    pos = len(prompt_ids)
    t0 = time.time()
    for _ in range(5):
        logits = srv.logits_from_hidden(state, h)
        next_id = int(np.argmax(logits[0]))
        h = state.embed_w[next_id].reshape(1, srv.HIDDEN).astype(np.float32)
        h = srv.step_forward(state, h, pos)
        pos += 1
    decode_wall = time.time() - t0
    print(f"  decode 5 tok: {decode_wall*1000:.1f} ms = {decode_wall*1000/5:.1f} ms/tok")

    print("\n[3] breakdown per token (decode steady-state)…")
    n_layers = state.text_cfg.num_hidden_layers
    # 5 decode tokens × 40 layers
    print(f"  block totals over 5 decode tokens × 40 layers:")
    for k in ["dn", "attn", "moe", "rmsnorm"]:
        times = block_times[k]
        if times:
            total_ms = sum(times) * 1000
            per_call_ms = total_ms / len(times)
            calls_per_tok = len(times) / 5
            per_tok_ms = total_ms / 5
            print(f"    {k:8s}: total {total_ms:7.1f} ms  per-call {per_call_ms:6.2f} ms  "
                  f"calls/tok {calls_per_tok:5.1f}  per-tok {per_tok_ms:6.1f} ms")
    other_total = sum(sum(v) for v in block_times.values()) * 1000
    decode_total_ms = decode_wall * 1000
    print(f"    (sum tracked: {other_total:.1f} ms; "
          f"untracked: {decode_total_ms - other_total:.1f} ms = "
          f"~{(decode_total_ms - other_total)/decode_total_ms*100:.0f}% "
          f"— RoPE, embed lookup, scalar ops)")

    print(f"\n  decode tok/s: {5/decode_wall:.2f}")

    ttnn.close_mesh_device(state.mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
