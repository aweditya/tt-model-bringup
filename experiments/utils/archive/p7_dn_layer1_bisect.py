#!/usr/bin/env python3
"""
P7 — layer-1 DeltaNet TP bisection: per-op sync+print to locate the hang.

P6.5 showed layer 0 (DN+MLP) completes (~340 ms warm), then layer 1's
attn/dn block STARTed and never returned (process in S/sleeping state for
8+ minutes before SIGKILL).

Code paths for layer 0 vs layer 1 DN are IDENTICAL (same `deltanet_step_tp`
function, same per-chip shapes, same `K_DIM`, same `CONV_DIM_CHIP`). The
only differences are:
  (a) the tensor object identities for weights/state (different memory
      addresses across layers, but same dtype/layout/memory_config in
      principle)
  (b) what layer 0's MLP all_reduce left in `x_tt` (the residual stream)
  (c) accumulated allocator state (intermediates from layer 0)

This probe MANUALLY UNROLLS `deltanet_step_tp` for layer 1, syncing +
printing before AND after every ttnn op. When we run it on qb2, the LAST
"BEFORE …" line that prints without a matching "AFTER …" identifies the
hanging op. That single op is the bisection result.

Pattern: reuse `bootstrap` from server_tp.py with TP_MAX_LAYERS=4 — gets
us 3 DN layers + 1 attn, same as P6.5. Run layer 0 (DN + MLP) with the
high-level functions (proven to complete in P6.5). Then unroll layer 1.

NO device execution from this file's *author*. The main thread executes.

Usage (executed BY MAIN THREAD on qb2, NOT here):
    ssh qb2 'cd ~/tt-xla && .venv/bin/python experiments/utils/p7_dn_layer1_bisect.py'
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

os.environ['TP_MAX_LAYERS'] = '4'

PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))

from experiments.serve.server_tp import (
    bootstrap, MeshServerState,
    deltanet_step_tp, mlp_step_tp,
)


def _sync_print(mesh, msg, t0):
    """Sync the mesh, then print msg with elapsed-ms-since-t0. flush=True."""
    import ttnn
    ttnn.synchronize_device(mesh)
    print(f"  [{(time.time()-t0)*1000:7.1f} ms]  {msg}", flush=True)


def deltanet_unrolled(mesh, x_tt, dn, cfg, t0):
    """Manually unrolled deltanet_step_tp with sync+print between EVERY ttnn op.

    Mirrors server_tp.py:deltanet_step_tp exactly. If the hang is here, we'll
    see the last BEFORE without an AFTER → exact op identified.
    """
    import ttnn
    import numpy as np
    from full_layer_tp_probe import (
        N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )

    _sync_print(mesh, "BEFORE  rms_norm(input_norm)", t0)
    h_tt = ttnn.rms_norm(x_tt, weight=dn['input_norm'], epsilon=EPS)
    _sync_print(mesh, "AFTER   rms_norm(input_norm)", t0)

    _sync_print(mesh, "BEFORE  linear(w_in)", t0)
    all_tt = ttnn.linear(h_tt, dn['w_in'])
    _sync_print(mesh, "AFTER   linear(w_in)", t0)

    _sync_print(mesh, "BEFORE  slice(mixed_qkv)", t0)
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    _sync_print(mesh, "AFTER   slice(mixed_qkv)", t0)

    _sync_print(mesh, "BEFORE  slice(z)", t0)
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    _sync_print(mesh, "AFTER   slice(z)", t0)

    _sync_print(mesh, "BEFORE  slice(a)", t0)
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    _sync_print(mesh, "AFTER   slice(a)", t0)

    _sync_print(mesh, "BEFORE  slice(b)", t0)
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    _sync_print(mesh, "AFTER   slice(b)", t0)

    _sync_print(mesh, "BEFORE  reshape(mixed_col)", t0)
    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
    _sync_print(mesh, "AFTER   reshape(mixed_col)", t0)

    _sync_print(mesh, "BEFORE  concat([conv_st, mixed_col])", t0)
    conv_input = ttnn.concat([dn['conv_st'], mixed_col], dim=-1)
    _sync_print(mesh, "AFTER   concat([conv_st, mixed_col])", t0)

    _sync_print(mesh, "BEFORE  mul(conv_input, w_conv)", t0)
    conv_prod = ttnn.mul(conv_input, dn['w_conv'])
    _sync_print(mesh, "AFTER   mul(conv_input, w_conv)", t0)

    _sync_print(mesh, "BEFORE  sum(conv_prod, dim=-1) + silu", t0)
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
    _sync_print(mesh, "AFTER   sum(conv_prod, dim=-1) + silu", t0)

    _sync_print(mesh, "BEFORE  slice(conv_state_new)", t0)
    conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM_CHIP, cfg['conv_kernel']])
    _sync_print(mesh, "AFTER   slice(conv_state_new)", t0)

    _sync_print(mesh, "BEFORE  slice(q/k/v_flat)", t0)
    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])
    _sync_print(mesh, "AFTER   slice(q/k/v_flat)", t0)

    _sync_print(mesh, "BEFORE  gqa(q) [reshape+repeat+reshape]", t0)
    q2 = ttnn.reshape(q_flat, [NK_PER_CHIP, 1, K_DIM])
    q3 = ttnn.repeat(q2, ttnn.Shape([1, N_REP, 1]))
    q = ttnn.reshape(q3, [NK_PER_CHIP * N_REP, K_DIM])
    _sync_print(mesh, "AFTER   gqa(q)", t0)

    _sync_print(mesh, "BEFORE  gqa(k)", t0)
    k2 = ttnn.reshape(k_flat, [NK_PER_CHIP, 1, K_DIM])
    k3 = ttnn.repeat(k2, ttnn.Shape([1, N_REP, 1]))
    k = ttnn.reshape(k3, [NK_PER_CHIP * N_REP, K_DIM])
    _sync_print(mesh, "AFTER   gqa(k)", t0)

    _sync_print(mesh, "BEFORE  reshape(v_flat)", t0)
    v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])
    _sync_print(mesh, "AFTER   reshape(v_flat)", t0)

    EPS_RMS = EPS / K_DIM
    _sync_print(mesh, "BEFORE  rms_norm(q, q_l2_scale)", t0)
    q = ttnn.rms_norm(q, weight=dn['q_l2_scale'], epsilon=EPS_RMS)
    _sync_print(mesh, "AFTER   rms_norm(q, q_l2_scale)", t0)

    _sync_print(mesh, "BEFORE  rms_norm(k, k_l2_scale)", t0)
    k = ttnn.rms_norm(k, weight=dn['k_l2_scale'], epsilon=EPS_RMS)
    _sync_print(mesh, "AFTER   rms_norm(k, k_l2_scale)", t0)

    _sync_print(mesh, "BEFORE  softplus_a = log(exp(a+dt_bias)+1)", t0)
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, dn['dt_bias'])), 1.0))
    _sync_print(mesh, "AFTER   softplus_a", t0)

    _sync_print(mesh, "BEFORE  g = -exp(A_log) * softplus_a", t0)
    g = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
    _sync_print(mesh, "AFTER   g", t0)

    _sync_print(mesh, "BEFORE  beta = sigmoid(b)", t0)
    beta = ttnn.sigmoid(b_tt)
    _sync_print(mesh, "AFTER   beta", t0)

    _sync_print(mesh, "BEFORE  decay = reshape(exp(g))", t0)
    decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
    _sync_print(mesh, "AFTER   decay", t0)

    _sync_print(mesh, "BEFORE  reshape(ssm) → H_4d", t0)
    H_4d = ttnn.reshape(dn['ssm'], [1, NV_PER_CHIP, K_DIM, V_DIM])
    _sync_print(mesh, "AFTER   H_4d", t0)

    _sync_print(mesh, "BEFORE  H_decayed = H * decay", t0)
    H_decayed = ttnn.mul(H_4d, decay)
    _sync_print(mesh, "AFTER   H_decayed", t0)

    _sync_print(mesh, "BEFORE  k_col = reshape(k)", t0)
    k_col = ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])
    _sync_print(mesh, "AFTER   k_col", t0)

    _sync_print(mesh, "BEFORE  kv_mem = sum(H_decayed * k_col, -2)", t0)
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                          [1, NV_PER_CHIP, V_DIM])
    _sync_print(mesh, "AFTER   kv_mem", t0)

    _sync_print(mesh, "BEFORE  delta = (v-kv_mem)*beta", t0)
    v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
    _sync_print(mesh, "AFTER   delta", t0)

    _sync_print(mesh, "BEFORE  H_new = H_decayed + k_col*delta", t0)
    H_new = ttnn.add(H_decayed,
                     ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])))
    _sync_print(mesh, "AFTER   H_new", t0)

    _sync_print(mesh, "BEFORE  out = sum(H_new * q_col, -2)", t0)
    q_col = ttnn.reshape(q, [1, NV_PER_CHIP, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM_CHIP])
    _sync_print(mesh, "AFTER   out", t0)

    _sync_print(mesh, "BEFORE  per-head rms_norm(linear_attn_norm)", t0)
    out_per_head = ttnn.reshape(out, [NV_PER_CHIP, V_DIM])
    out_normed = ttnn.rms_norm(out_per_head, weight=dn['linear_attn_norm'], epsilon=EPS)
    _sync_print(mesh, "AFTER   per-head rms_norm", t0)

    _sync_print(mesh, "BEFORE  silu(z)*out + reshape", t0)
    z_per_head = ttnn.reshape(z_tt, [NV_PER_CHIP, V_DIM])
    silu_z = ttnn.silu(z_per_head)
    out_gated = ttnn.reshape(ttnn.mul(out_normed, silu_z), [1, VAL_DIM_CHIP])
    _sync_print(mesh, "AFTER   gated out", t0)

    _sync_print(mesh, "BEFORE  linear(out_gated, w_out) [out_proj]", t0)
    partial = ttnn.linear(out_gated, dn['w_out'])
    _sync_print(mesh, "AFTER   out_proj", t0)

    _sync_print(mesh, "BEFORE  all_reduce(partial) (or rs+ag fallback)", t0)
    try:
        reduced = ttnn.all_reduce(partial)
        coll = "all_reduce"
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
        coll = "reduce_scatter+all_gather"
    _sync_print(mesh, f"AFTER   collective={coll}", t0)

    _sync_print(mesh, "BEFORE  x_out = x_tt + reduced (residual)", t0)
    x_out = ttnn.add(x_tt, reduced)
    _sync_print(mesh, "AFTER   residual", t0)

    _sync_print(mesh, "BEFORE  copy(H_new_3d → ssm)", t0)
    H_new_3d = ttnn.reshape(H_new, [NV_PER_CHIP, K_DIM, V_DIM])
    ttnn.copy(H_new_3d, dn['ssm'])
    _sync_print(mesh, "AFTER   copy ssm", t0)

    _sync_print(mesh, "BEFORE  copy(conv_state_new → conv_st)", t0)
    ttnn.copy(conv_state_new, dn['conv_st'])
    _sync_print(mesh, "AFTER   copy conv_st", t0)

    return x_out


def main():
    print("=" * 78, flush=True)
    print("P7: layer-1 DeltaNet TP bisection (per-op sync+print)", flush=True)
    print("=" * 78, flush=True)

    state = MeshServerState()
    try:
        t_boot = time.time()
        bootstrap(state)
        print(f"[bootstrap] returned in {time.time() - t_boot:.1f}s", flush=True)

        import ttnn
        import torch
        import numpy as np

        cfg = state.cfg
        HIDDEN = cfg['hidden']

        # Match P6.5: token=128, cur_pos=0, replicated embed input
        token_id, cur_pos = 128, 0
        x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        ttnn.synchronize_device(state.mesh)
        print("\n[setup] x_tt ready", flush=True)

        # --- Layer 0: DN + MLP via the high-level functions (proven OK in P6.5) ---
        print("\n[layer 0] running DN + MLP via high-level fns (P6.5 proved this is OK)…", flush=True)
        t0 = time.time()
        assert state.layers[0]['type'] == 'linear_attention'
        x_tt = deltanet_step_tp(state, x_tt, state.layers[0]['dn'], cfg)
        ttnn.synchronize_device(state.mesh)
        print(f"  layer 0 DN  done in {(time.time()-t0)*1000:.0f} ms", flush=True)
        t0 = time.time()
        x_tt = mlp_step_tp(state, x_tt, state.layers[0]['mlp'])
        ttnn.synchronize_device(state.mesh)
        print(f"  layer 0 MLP done in {(time.time()-t0)*1000:.0f} ms", flush=True)

        # --- Layer 1 DN: UNROLLED, per-op sync+print ---
        print("\n[layer 1] DeltaNet UNROLLED — locating the hang", flush=True)
        assert state.layers[1]['type'] == 'linear_attention'
        t0 = time.time()
        x_tt = deltanet_unrolled(state.mesh, x_tt, state.layers[1]['dn'], cfg, t0)
        print(f"\n[layer 1] DN unrolled COMPLETED in {(time.time()-t0)*1000:.0f} ms (no hang)",
              flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P7 PASSED — no hang. layer-1 DN ran end-to-end with per-op visibility.", flush=True)
        print("=" * 78, flush=True)

    finally:
        try:
            import ttnn
            if state.mesh is not None:
                ttnn.close_mesh_device(state.mesh)
                print("\n  ✓ mesh closed cleanly", flush=True)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("  ✓ fabric reset to DISABLED", flush=True)
        except Exception as e:
            print(f"  ✗ cleanup error: {e}", flush=True)


if __name__ == "__main__":
    main()
