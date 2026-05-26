#!/usr/bin/env python3
"""Pattern A MoE TT correctness test — topk vs pattern_a_batched on device.

Two bootstraps back-to-back, layer-0 MoE on the embed of
"The capital of France is". Cosine target > 0.9999.

Run (qb1): see HANDOFF.md for the env-var bootstrap; then
  .venv/bin/python -u experiments/utils/test_pattern_a_moe_tt.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402
import ttnn  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_with_mode(mode):
    """Bootstrap server in the given moe_mode, return one MoE call's output."""
    log(f"--- mode={mode!r} ---")
    state = srv.State()
    state.moe_mode = mode
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = state.tokenizer.encode("The capital of France is")
    tok_id = prompt_ids[0]
    h_np = state.embed_w_np[tok_id].reshape(1, srv.HIDDEN).astype(np.float32)
    h_tt = srv.np_to_replicated(h_np, state.mesh)
    # Apply post_attention_layernorm on layer 0's input (matches what would
    # feed into the MoE in a real forward).
    h_norm = ttnn.rms_norm(
        h_tt, weight=state.per_layer_tt[0]["post_attention_layernorm"],
        epsilon=srv.EPS,
    )
    ttnn.deallocate(h_tt)

    moe_fn = (srv.moe_forward_ttnn_pattern_a_batched if mode == "pattern_a_batched"
              else srv.moe_forward_ttnn)
    t0 = time.time()
    out = moe_fn(h_norm, state.per_layer_tt[0], state.mesh)
    ttnn.synchronize_device(state.mesh)
    elapsed = time.time() - t0
    log(f"  one MoE call took {elapsed * 1000:.1f} ms (eager, single layer)")

    out_np = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
    ).float().numpy()[0]
    ttnn.deallocate(out); ttnn.deallocate(h_norm)
    log(f"  out shape={out_np.shape}  |.|={np.linalg.norm(out_np):.4f}")

    ttnn.close_mesh_device(state.mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return out_np, elapsed


def main():
    out_topk, t_topk = run_with_mode("topk")
    log("")
    out_pa, t_pa = run_with_mode("pattern_a_batched")
    log("")
    log("=== COMPARISON ===")
    c = cos(out_topk, out_pa)
    abs_err = np.abs(out_topk - out_pa).max()
    rel_err = np.linalg.norm(out_topk - out_pa) / np.linalg.norm(out_topk)
    log(f"  cos(topk, batched):     {c:.8f}")
    log(f"  max |Δ|:                {abs_err:.6e}")
    log(f"  rel ||Δ|| / ||topk||:   {rel_err:.6e}")
    log(f"  one-MoE-call time:      topk={t_topk*1000:.1f} ms   batched={t_pa*1000:.1f} ms   ratio={t_pa/t_topk:.2f}x")

    assert c > 0.9999, f"COSINE TOO LOW: {c:.6f}"
    log("PASS  pattern_a_batched MoE matches topk MoE within bf16 noise.")


if __name__ == "__main__":
    main()
