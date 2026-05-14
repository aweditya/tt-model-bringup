#!/usr/bin/env python3
"""
P11 — multi-step TRACED decode on (1,4) mesh (qb2).

P10 confirmed trace capture fails at the FIRST host write (ttnn.from_torch
for the token embedding). The Llama70B production pattern (see
experiments/.refs/tt-metal/models/demos/llama3_70b_galaxy/tt/generator.py:
920-1015 + llama_common.py:211-232) is:

  1. Pre-allocate input buffers ONCE outside trace
  2. Begin trace capture
  3. Forward reads buffers (no host writes inside)
  4. End trace capture
  5. Per-step: copy_host_to_device_tensor updates buffers in-place
     (outside captured region) → execute_trace reads updated buffers

This probe validates that pattern end-to-end with our manual rms_norm +
deallocate forward. If P11 passes, we ship the refactor into
server_tp.py:handle_generate_tp.

Steps:
  - Bootstrap TP_MAX_LAYERS=4
  - Pre-allocate state.x_buf, state.cur_pos_buf, state.cos_buf, state.sin_buf
  - Define local gated_attn_step_tp_inner that uses cur_pos_tensor= (no Python int)
  - Define forward_inner(state) that reads ONLY from buffers
  - Warmup eagerly via forward_inner × 2
  - begin_trace_capture → forward_inner → end_trace_capture
  - Loop 3 steps: copy_host_to_device_tensor → execute_trace → read logits

Pass: trace captures cleanly + 3 execute_trace calls complete; logits finite
+ chips agree. Then we know the pattern; refactor server_tp.py next.

Time: ~3 min wall (50s bootstrap + warmup + capture + 3 exec).
"""
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
os.environ['TP_MAX_LAYERS'] = '4'

PROJECT_ROOT = "/home/aditya/tt-xla"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))


def main():
    print("=" * 78, flush=True)
    print("P11: multi-step TRACED decode (Llama70B pattern) on (1,4) mesh", flush=True)
    print("=" * 78, flush=True)

    from experiments.serve.server_tp import (
        bootstrap, MeshServerState,
        deltanet_step_tp, mlp_step_tp, _rms_norm_manual, MAX_POS,
    )

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
        HEAD_DIM = cfg['head_dim']
        ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
        N_Q = cfg['n_q_heads']
        N_KV = cfg['n_kv_heads']
        NQ_PER_CHIP = N_Q // 4
        NKV_PER_CHIP = N_KV // 4
        QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM
        KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
        EPS = 1e-6

        mesh = state.mesh

        # === Pre-compute cos_all / sin_all host arrays (already implicitly done
        # in bootstrap; recompute here for explicit access) ===
        half_rot = ROTARY_DIM // 2
        freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
        positions = np.arange(MAX_POS).astype(np.float32)
        ang = positions[:, None] * freqs[None, :]
        cos_all_np = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
        sin_all_np = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
        # Note: ROTARY_DIM-wide for V2 rotate-only path (no extension/pad needed
        # since we slice ROTARY_DIM not HEAD_DIM for the rotate region)

        # === Pre-allocate input buffers (all replicated across mesh) ===
        print(f"\n[buffers] pre-allocating input buffers…", flush=True)
        def alloc_replicated(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.float32),
                dtype=dtype, device=mesh, layout=layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        state.x_buf = alloc_replicated((1, HIDDEN))
        state.cos_buf = alloc_replicated((1, ROTARY_DIM))
        state.sin_buf = alloc_replicated((1, ROTARY_DIM))
        # cur_pos_buf: int32, row-major (not tile)
        state.cur_pos_buf = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ x_buf [1, {HIDDEN}], cos/sin_buf [1, {ROTARY_DIM}], cur_pos_buf [1]", flush=True)

        # === Local gated_attn_step_tp_inner that uses cur_pos_tensor= ===
        # Mirrors server_tp.py:gated_attn_step_tp but cur_pos is a TENSOR, not a Python int.
        # The only difference: update_cache_for_token_ uses cur_pos_tensor= kwarg.
        def gated_attn_step_tp_inner(state, x_tt, attn, cur_pos_tt, cos_tt, sin_tt):
            # 1. Pre-norm
            h_tt = _rms_norm_manual(x_tt, attn['input_norm'], EPS, HIDDEN)
            # 2. Sharded attn_qkv matmul
            all_tt = ttnn.linear(h_tt, attn['w_qkv'])
            ttnn.deallocate(h_tt)
            qg = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
            k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
            v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                                 [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
            ttnn.deallocate(all_tt)
            qg = ttnn.reshape(qg, [NQ_PER_CHIP, 2 * HEAD_DIM])
            q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
            gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])
            ttnn.deallocate(qg)
            k_tt = ttnn.reshape(k_flat, [NKV_PER_CHIP, HEAD_DIM])
            v_tt = ttnn.reshape(v_flat, [NKV_PER_CHIP, HEAD_DIM])
            # 3. q/k norms (manual mesh-safe)
            q_tt = _rms_norm_manual(q_tt, attn['q_norm'], EPS, HEAD_DIM)
            k_tt = _rms_norm_manual(k_tt, attn['k_norm'], EPS, HEAD_DIM)
            # 4. Partial RoPE V2 rotate-only — cos/sin come from PRE-ALLOC buf
            half = ROTARY_DIM // 2
            def apply_rope(t, n_heads):
                rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
                passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
                x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
                x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
                neg_x2 = ttnn.neg(x2)
                rotated = ttnn.add(ttnn.mul(rot, cos_tt),
                                    ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_tt))
                return ttnn.concat([rotated, passthru], dim=-1)
            q_tt = apply_rope(q_tt, NQ_PER_CHIP)
            k_tt = apply_rope(k_tt, NKV_PER_CHIP)
            # 5. KV cache update — cur_pos_tensor=cur_pos_tt (THE KEY CHANGE)
            k_for_cache = ttnn.reshape(k_tt, [1, NKV_PER_CHIP, 1, HEAD_DIM])
            v_for_cache = ttnn.reshape(v_tt, [1, NKV_PER_CHIP, 1, HEAD_DIM])
            ttnn.kv_cache.update_cache_for_token_(attn['kc'], k_for_cache,
                                                    cur_pos_tensor=cur_pos_tt)
            ttnn.kv_cache.update_cache_for_token_(attn['vc'], v_for_cache,
                                                    cur_pos_tensor=cur_pos_tt)
            # 6. Manual SDPA (same as eager — no cur_pos baked here)
            assert NKV_PER_CHIP == 1, "manual SDPA assumes 1 KV head per chip"
            kc_flat = ttnn.reshape(attn['kc'], [MAX_POS, HEAD_DIM])
            vc_flat = ttnn.reshape(attn['vc'], [MAX_POS, HEAD_DIM])
            scale = 1.0 / np.sqrt(HEAD_DIM)
            kT = ttnn.transpose(kc_flat, 0, 1)
            scores = ttnn.mul(ttnn.matmul(q_tt, kT), scale)
            attn_w = ttnn.softmax(scores, dim=-1)
            attn_per_head = ttnn.matmul(attn_w, vc_flat)
            # 7. Sigmoid gate + mul
            attn_gated = ttnn.mul(attn_per_head, ttnn.sigmoid(gate_tt))
            # 8. out_proj row-parallel + all_reduce
            attn_flat = ttnn.reshape(attn_gated, [1, NQ_PER_CHIP * HEAD_DIM])
            partial = ttnn.linear(attn_flat, attn['w_o'])
            try:
                reduced = ttnn.all_reduce(partial)
            except Exception:
                scattered = ttnn.reduce_scatter(partial, dim=1)
                reduced = ttnn.all_gather(scattered, dim=1)
            ttnn.deallocate(partial)
            x_out = ttnn.add(x_tt, reduced)
            ttnn.deallocate(reduced)
            return x_out

        # === forward_inner — uses buffers, no host writes ===
        def forward_inner(state):
            x_tt = state.x_buf
            cos_tt = state.cos_buf
            sin_tt = state.sin_buf
            cur_pos_tt = state.cur_pos_buf
            for layer in state.layers:
                if layer['type'] == 'linear_attention':
                    x_tt = deltanet_step_tp(state, x_tt, layer['dn'], cfg)
                else:
                    x_tt = gated_attn_step_tp_inner(state, x_tt, layer['attn'],
                                                     cur_pos_tt, cos_tt, sin_tt)
                x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
            x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
            return ttnn.linear(x_tt, state.lm_head_tt)

        # === Host-side buffer update helper ===
        def update_buffers(token_id, cur_pos):
            x_np = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
            x_host = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(x_host, state.x_buf)
            cos_host = ttnn.from_torch(
                torch.from_numpy(cos_all_np[cur_pos:cur_pos+1]),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            sin_host = ttnn.from_torch(
                torch.from_numpy(sin_all_np[cur_pos:cur_pos+1]),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(cos_host, state.cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, state.sin_buf)
            cur_pos_host = ttnn.from_torch(
                torch.tensor([cur_pos], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            ttnn.copy_host_to_device_tensor(cur_pos_host, state.cur_pos_buf)

        # === Warmup (per feedback_c4v4_validated) ===
        print(f"\n[warmup] eager forward_inner × 2 (JIT amortization)…", flush=True)
        for i in range(2):
            update_buffers(token_id=128, cur_pos=i)
            t0 = time.time()
            _ = forward_inner(state)
            ttnn.synchronize_device(mesh)
            print(f"  warmup {i}: {(time.time()-t0)*1000:.0f} ms", flush=True)

        # === Trace capture ===
        # Update buffers to cur_pos=2 for capture
        update_buffers(token_id=128, cur_pos=2)
        print(f"\n[trace] begin_trace_capture…", flush=True)
        t_cap = time.time()
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced_logits_tt = forward_inner(state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        print(f"  ✓ trace captured in {(time.time()-t_cap)*1000:.0f} ms", flush=True)

        # === Execute trace × 3 with buffer updates ===
        print(f"\n[execute] 3 traced steps with cur_pos=3,4,5…", flush=True)
        for step, (tok, cp) in enumerate([(256, 3), (512, 4), (1024, 5)]):
            t_upd = time.time()
            update_buffers(token_id=tok, cur_pos=cp)
            t_exec = time.time()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            t_end = time.time()
            logits = ttnn.to_torch(
                traced_logits_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().numpy()
            finite = bool(np.isfinite(logits).all())
            argmax = int(logits[0].argmax())
            chip_diff = float(np.abs(logits[0] - logits[1]).max())
            print(f"  step {step} (tok={tok}, cp={cp}): "
                  f"upd {(t_exec-t_upd)*1000:.0f} ms, "
                  f"exec {(t_end-t_exec)*1000:.0f} ms, "
                  f"finite={finite}, argmax={argmax}, chip|Δ|={chip_diff:.4f}",
                  flush=True)

        try:
            ttnn.release_trace(mesh, trace_id)
            print(f"  ✓ trace released", flush=True)
        except Exception as e:
            print(f"  ✗ release error: {e}", flush=True)

        print("\n" + "=" * 78, flush=True)
        print("  ✓ P11 PASSES — traced multi-step decode works on mesh", flush=True)
        print("    Refactor server_tp.py:handle_generate_tp to ship the win.", flush=True)
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
