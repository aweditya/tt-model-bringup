#!/usr/bin/env python3
"""
P7.5 — H2 confirm/rule-out: layer-0 DN WITHOUT ttnn.copy state mutation.

P7 found layer 1 DN hangs at `ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])`.
The hypothesis (H2): layer 0's `ttnn.copy(H_new, dn['ssm'])` + `ttnn.copy(
conv_state_new, dn['conv_st'])` leave NOC traffic that fences layer 1's
first non-trivial mesh op.

Test: rerun layer 0 DN WITHOUT the two ttnn.copy calls, then walk layer 1
DN per-op (same as P7). If layer 1 reaches the same hang point → NOT H2.
If layer 1 progresses past `reshape(k)` → H2 CONFIRMED.

Reuses the unrolled walker from P7.
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
    bootstrap, MeshServerState, mlp_step_tp, MAX_POS,
)
from experiments.utils.p7_dn_layer1_bisect import deltanet_unrolled, _sync_print


def deltanet_no_state_copy(mesh, x_tt, dn, cfg, t0):
    """Same as deltanet_unrolled() but SKIPS the final ttnn.copy mutations.

    Returns x_out as usual; just doesn't update SSM/conv_state on device.
    This isolates H2 — if removing the copies makes layer-1 work, the
    copies were the cause.
    """
    import ttnn
    from full_layer_tp_probe import (
        N_V_HEADS, K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )
    HIDDEN = cfg['hidden']

    h_tt = ttnn.rms_norm(x_tt, weight=dn['input_norm'], epsilon=EPS)
    all_tt = ttnn.linear(h_tt, dn['w_in'])
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                       [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                       [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
    conv_input = ttnn.concat([dn['conv_st'], mixed_col], dim=-1)
    conv_prod = ttnn.mul(conv_input, dn['w_conv'])
    conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
    # conv_state_new computed but NOT copied:
    _conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM_CHIP, cfg['conv_kernel']])
    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])

    def gqa(t, n_kh, d):
        t2 = ttnn.reshape(t, [n_kh, 1, d])
        t3 = ttnn.repeat(t2, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t3, [n_kh * N_REP, d])
    q = gqa(q_flat, NK_PER_CHIP, K_DIM)
    k = gqa(k_flat, NK_PER_CHIP, K_DIM)
    v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])

    EPS_RMS = EPS / K_DIM
    q = ttnn.rms_norm(q, weight=dn['q_l2_scale'], epsilon=EPS_RMS)
    k = ttnn.rms_norm(k, weight=dn['k_l2_scale'], epsilon=EPS_RMS)
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, dn['dt_bias'])), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
    beta = ttnn.sigmoid(b_tt)
    decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
    H_4d = ttnn.reshape(dn['ssm'], [1, NV_PER_CHIP, K_DIM, V_DIM])
    H_decayed = ttnn.mul(H_4d, decay)
    k_col = ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])
    kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                           [1, NV_PER_CHIP, V_DIM])
    v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
    delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
    H_new = ttnn.add(H_decayed,
                      ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])))
    q_col = ttnn.reshape(q, [1, NV_PER_CHIP, K_DIM, 1])
    out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM_CHIP])
    out_per_head = ttnn.reshape(out, [NV_PER_CHIP, V_DIM])
    out_normed = ttnn.rms_norm(out_per_head, weight=dn['linear_attn_norm'], epsilon=EPS)
    z_per_head = ttnn.reshape(z_tt, [NV_PER_CHIP, V_DIM])
    silu_z = ttnn.silu(z_per_head)
    out_gated = ttnn.reshape(ttnn.mul(out_normed, silu_z), [1, VAL_DIM_CHIP])
    partial = ttnn.linear(out_gated, dn['w_out'])
    try:
        reduced = ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        reduced = ttnn.all_gather(scattered, dim=1)
    x_out = ttnn.add(x_tt, reduced)
    # *** NOTE: skipping ttnn.copy(H_new, dn['ssm']) + ttnn.copy(conv_state_new, dn['conv_st']) ***
    return x_out


def main():
    print("=" * 78, flush=True)
    print("P7.5: H2 test — layer 0 DN WITHOUT state-copy mutation", flush=True)
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

        token_id, cur_pos = 128, 0
        x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
        x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                 device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
        ttnn.synchronize_device(state.mesh)
        print("\n[setup] x_tt ready", flush=True)

        # --- Layer 0 DN WITHOUT state copy mutation ---
        print("\n[layer 0] DN WITHOUT state-copy mutation…", flush=True)
        t0 = time.time()
        assert state.layers[0]['type'] == 'linear_attention'
        x_tt = deltanet_no_state_copy(state.mesh, x_tt, state.layers[0]['dn'], cfg, t0)
        ttnn.synchronize_device(state.mesh)
        print(f"  layer 0 DN (no copy) done in {(time.time()-t0)*1000:.0f} ms", flush=True)
        t0 = time.time()
        x_tt = mlp_step_tp(state, x_tt, state.layers[0]['mlp'])
        ttnn.synchronize_device(state.mesh)
        print(f"  layer 0 MLP done in {(time.time()-t0)*1000:.0f} ms", flush=True)

        # --- Layer 1 DN: UNROLLED, same as P7 ---
        print("\n[layer 1] DN UNROLLED — testing if hang disappears", flush=True)
        assert state.layers[1]['type'] == 'linear_attention'
        t0 = time.time()
        x_tt = deltanet_unrolled(state.mesh, x_tt, state.layers[1]['dn'], cfg, t0)
        print(f"\n[layer 1] DN unrolled COMPLETED in {(time.time()-t0)*1000:.0f} ms",
              flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P7.5 PASSED — layer 1 DN works when layer 0 SKIPS state-copy", flush=True)
        print("    → H2 CONFIRMED: ttnn.copy state-mutation deadlocks the next DN on mesh.", flush=True)
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
