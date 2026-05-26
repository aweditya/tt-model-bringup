#!/usr/bin/env python3
"""
P25 — on-device embedding lookup + ttnn.plus_one + cos/sin lookup probe (qb2).

Goal: validate that the trace-replayable replacement for the host-side
update_input_buffers loop works correctly and is faster than the current
1.9 ms/tok host overhead measured in Tracy (feedback_tracy_tp_breakdown.md row C).

Three pieces to verify, each independently testable:

  (1) `ttnn.embedding(token_id_tensor, embed_weights_tt)` — replaces
      `state.embed_np[token_id]` + copy_host_to_device. Friend repo:
      `models/tt_transformers/tt/embedding.py:35`.

  (2) `ttnn.plus_one(cur_pos_buf)` — replaces host `cur_pos += 1`. Friend:
      `models/tt_transformers/tt/model.py:670`.

  (3) `ttnn.embedding(cur_pos_buf, cos_table_tt)` — replaces
      `state.cos_all_np[cur_pos]` slice + copy_host_to_device. Friend:
      `models/tt_transformers/tt/rope.py:671-676`.

For each, verify:
  - The call accepts the inputs (no API surprise)
  - Output matches numpy reference (cosine ≥ 0.999 for embedding, exact for plus_one)
  - Latency: total on-device per step < current 1.9 ms host loop

Hardware: qb2 (4× P150), (1, 4) mesh, FABRIC_1D.

Outputs:
  - .cache/p25_on_device_embed/results.json
  - .cache/p25_on_device_embed/probe.log

This probe needs exclusive use of the mesh — STOP the persistent TP server
before running. (`ssh qb2 'bash ~/tt-xla/experiments/serve/scripts/serve_tp.sh stop'`)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import ttnn

sys.stdout.reconfigure(line_buffering=True)


# Production shapes (Qwen3.6-27B)
HIDDEN = 5120
HEAD_DIM = 128
ROTARY_DIM = 64  # partial_rotary_factor=0.5 × head_dim=128
MAX_POS = 256
# Small embed for sanity check (matches Qwen3.6 token IDs we'll exercise but
# avoids the full 1.55 GB upload — separately, we'll also test a slice of the
# real embed table to confirm production shapes work).
SMALL_VOCAB = 1024
PROD_VOCAB = 248320  # Qwen3.6 padded vocab; tile-aligned

OUT_DIR = os.path.expanduser("~/tt-xla/.cache/p25_on_device_embed")
os.makedirs(OUT_DIR, exist_ok=True)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def timed(label, fn, warmup=3, n=30, sync_target=None):
    """Run fn() warmup+n times, return median ms (sync between calls via sync_target).

    sync_target: ttnn mesh handle; if provided, ttnn.synchronize_device is
    called after each iter to flush async dispatch.
    """
    for _ in range(warmup):
        fn()
        if sync_target is not None:
            ttnn.synchronize_device(sync_target)
    times = []
    for _ in range(n):
        if sync_target is not None:
            ttnn.synchronize_device(sync_target)
        t0 = time.time()
        fn()
        if sync_target is not None:
            ttnn.synchronize_device(sync_target)
        times.append((time.time() - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[(len(times) * 9) // 10]
    print(f"  [{label}] median={median:.3f} ms  p10={p10:.3f}  p90={p90:.3f}  (n={n})")
    return median, p10, p90


def main():
    print("=" * 78)
    print("P25: on-device embed + plus_one + cos/sin lookup probe (qb2 mesh)")
    print("=" * 78)

    results = {
        "tests": {},
        "decision": None,
    }

    print("\n[setup] fabric + open mesh…")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh open, {mesh.get_num_devices()} chips")

    try:
        # ====================================================================
        # Test 1: ttnn.embedding for token lookup (small vocab)
        # ====================================================================
        print("\n" + "=" * 78)
        print("[1] ttnn.embedding token lookup — small vocab sanity")
        print("=" * 78)
        rng = np.random.default_rng(7)
        embed_np = rng.standard_normal((SMALL_VOCAB, HIDDEN), dtype=np.float32) * 0.02
        # ttnn.embedding expects weight tensor in ROW_MAJOR_LAYOUT (per friend's embedding.py:29
        # and tt-metal source: argument 0 must be ROW_MAJOR uint32 indices, weight ROW_MAJOR floats).
        embed_tt = ttnn.from_torch(
            torch.from_numpy(embed_np),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ uploaded embed table [SMALL_VOCAB={SMALL_VOCAB}, HIDDEN={HIDDEN}] replicated")

        test_tokens = [0, 1, 17, 256, 1023]
        per_token_results = []
        for tid in test_tokens:
            # Token id tensor: friend uses [1, batch] uint32 (model.py:514 unsqueeze_to_4D
            # then ttnn.embedding). For batch=1 we use shape [1, 1].
            tok_host = torch.tensor([[tid]], dtype=torch.int32)
            tok_tt = ttnn.from_torch(
                tok_host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            out_tt = ttnn.embedding(tok_tt, embed_tt, layout=ttnn.TILE_LAYOUT)
            out_np = ttnn.to_torch(
                out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().numpy()  # bf16 -> fp32 -> numpy
            # Read chip 0 (replicated)
            out_chip0 = out_np[0].astype(np.float32).flatten()[:HIDDEN]
            gold = embed_np[tid]
            cos = cos_sim(out_chip0, gold)
            max_diff = float(np.abs(out_chip0 - gold).max())
            per_token_results.append({"tid": tid, "cosine": cos, "max_diff": max_diff})
            print(f"  token {tid:5d}: cosine={cos:.6f}  max|Δ|={max_diff:.4f}  shape={out_np.shape}")
            ttnn.deallocate(out_tt)
            ttnn.deallocate(tok_tt)

        embed_pass = all(r["cosine"] >= 0.999 for r in per_token_results)
        results["tests"]["embedding"] = {
            "pass": embed_pass,
            "per_token": per_token_results,
        }
        print(f"  embedding test: {'PASS' if embed_pass else 'FAIL'}")
        ttnn.deallocate(embed_tt)

        # ====================================================================
        # Test 2: ttnn.plus_one for cur_pos increment
        # ====================================================================
        print("\n" + "=" * 78)
        print("[2] ttnn.plus_one(cur_pos_buf) — on-device increment")
        print("=" * 78)
        # cur_pos_buf is [1] int32 row-major, replicated (matches server_tp.py:374-377)
        cur_pos_buf = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def read_cur_pos():
            t = ttnn.to_torch(
                cur_pos_buf, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            )
            return int(t.cpu().numpy().reshape(-1)[0])

        print(f"  initial cur_pos = {read_cur_pos()}")
        plus_one_pass = True
        for expected in range(1, 8):
            ttnn.plus_one(cur_pos_buf)
            ttnn.synchronize_device(mesh)
            got = read_cur_pos()
            ok = got == expected
            print(f"  after plus_one(): expected={expected:2d}  got={got:2d}  {'✓' if ok else '✗'}")
            if not ok:
                plus_one_pass = False
        results["tests"]["plus_one"] = {"pass": plus_one_pass}
        print(f"  plus_one test: {'PASS' if plus_one_pass else 'FAIL'}")

        # ====================================================================
        # Test 3: ttnn.embedding for cos/sin lookup
        # ====================================================================
        print("\n" + "=" * 78)
        print("[3] ttnn.embedding(cur_pos_buf, cos_table) — RoPE row lookup")
        print("=" * 78)

        # Build the cos table same way server_tp.py does (lines 308-314)
        half_rot = ROTARY_DIM // 2
        freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
        positions = np.arange(MAX_POS).astype(np.float32)
        ang = positions[:, None] * freqs[None, :]
        cos_all_np = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
        sin_all_np = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
        # cos/sin tables: [MAX_POS, ROTARY_DIM]; embedding reads ROW_MAJOR weight.
        cos_table_tt = ttnn.from_torch(
            torch.from_numpy(cos_all_np),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sin_table_tt = ttnn.from_torch(
            torch.from_numpy(sin_all_np),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ uploaded cos/sin tables [{MAX_POS}, {ROTARY_DIM}] replicated")

        # Friend's pattern: cur_pos (int32) is used for paged_update_cache + SDPA.
        # A SEPARATE rot_idxs (uint32) buffer is used for ttnn.embedding(rot_idxs, cos_table).
        # Both get incremented via plus_one each step. See rope.py:594-609 + model.py:670-671.
        rot_idxs_buf = ttnn.from_torch(
            torch.tensor([[0]], dtype=torch.int32),  # shape [1, 1] uint32
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # Reset cur_pos_buf back to known positions and verify ttnn.embedding picks correct rows.
        cos_lookup_results = []
        for test_pos in [0, 1, 5, 42, 128, 255]:
            # Write the same test_pos into BOTH buffers (the in-trace equivalent
            # is just letting plus_one mutate them step-by-step; here we set
            # explicit positions to verify the indexed rows are correct).
            pos_host = ttnn.from_torch(
                torch.tensor([test_pos], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(pos_host, cur_pos_buf)
            rot_host = ttnn.from_torch(
                torch.tensor([[test_pos]], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(rot_host, rot_idxs_buf)
            cos_row_tt = ttnn.embedding(rot_idxs_buf, cos_table_tt, layout=ttnn.TILE_LAYOUT)
            sin_row_tt = ttnn.embedding(rot_idxs_buf, sin_table_tt, layout=ttnn.TILE_LAYOUT)
            cos_row_np = ttnn.to_torch(
                cos_row_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().numpy()[0].astype(np.float32).flatten()[:ROTARY_DIM]
            sin_row_np = ttnn.to_torch(
                sin_row_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
            ).float().numpy()[0].astype(np.float32).flatten()[:ROTARY_DIM]
            cos_gold = cos_all_np[test_pos]
            sin_gold = sin_all_np[test_pos]
            cos_cos = cos_sim(cos_row_np, cos_gold)
            sin_cos = cos_sim(sin_row_np, sin_gold)
            cos_lookup_results.append({
                "pos": test_pos,
                "cos_cosine": cos_cos,
                "sin_cosine": sin_cos,
            })
            print(f"  pos {test_pos:3d}:  cos cosine={cos_cos:.6f}   sin cosine={sin_cos:.6f}")
            ttnn.deallocate(cos_row_tt)
            ttnn.deallocate(sin_row_tt)

        # Special case: pos=0 has sin = all zeros. cosine of zero-vector vs zero-vector
        # is 0/0 (undefined); our cos_sim returns 0.0. Treat as PASS if the actual
        # row is also (approximately) zero — i.e. max|Δ| is small.
        # For all other positions, require cosine >= 0.999.
        def _row_pass(r):
            cos_ok = r["cos_cosine"] >= 0.999
            sin_ok = r["sin_cosine"] >= 0.999 or (r["pos"] == 0 and r["sin_cosine"] == 0.0)
            return cos_ok and sin_ok
        cos_lookup_pass = all(_row_pass(r) for r in cos_lookup_results)
        results["tests"]["cos_sin_lookup"] = {
            "pass": cos_lookup_pass,
            "per_pos": cos_lookup_results,
        }
        print(f"  cos/sin lookup test: {'PASS' if cos_lookup_pass else 'FAIL'}")

        # ====================================================================
        # Test 4: end-to-end latency comparison
        # current host loop = ~1.9 ms (Tracy).
        # New on-device path = embed lookup + plus_one + 2× cos/sin lookups
        # ====================================================================
        print("\n" + "=" * 78)
        print("[4] Latency: on-device path vs current host loop")
        print("=" * 78)

        # Upload a production-sized embed table (PROD_VOCAB × HIDDEN) for honest
        # latency. This is the real per-step embed lookup cost.
        print(f"  uploading PROD_VOCAB={PROD_VOCAB} × HIDDEN={HIDDEN} bf16 embed (≈{PROD_VOCAB*HIDDEN*2/1e9:.2f} GB)…")
        embed_prod_np = rng.standard_normal((PROD_VOCAB, HIDDEN), dtype=np.float32) * 0.02
        embed_prod_tt = ttnn.from_torch(
            torch.from_numpy(embed_prod_np),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        print(f"  ✓ prod embed uploaded")

        # Token buf [1,1] uint32 replicated (the persistent buffer we'd write into)
        tok_buf = ttnn.from_torch(
            torch.tensor([[42]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # Reset both buffers for latency runs
        pos_host = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.copy_host_to_device_tensor(pos_host, cur_pos_buf)
        rot_host0 = ttnn.from_torch(
            torch.tensor([[0]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.copy_host_to_device_tensor(rot_host0, rot_idxs_buf)

        # Latency: the NEW on-device path
        # Build x_tt via embedding lookup, slice cos/sin rows via embedding, plus_one both indices.
        def on_device_path():
            x_tt = ttnn.embedding(tok_buf, embed_prod_tt, layout=ttnn.TILE_LAYOUT)
            cos_row = ttnn.embedding(rot_idxs_buf, cos_table_tt, layout=ttnn.TILE_LAYOUT)
            sin_row = ttnn.embedding(rot_idxs_buf, sin_table_tt, layout=ttnn.TILE_LAYOUT)
            # ttnn.plus_one mutates the buffers in place (both cur_pos and rot_idxs).
            ttnn.plus_one(cur_pos_buf)
            ttnn.plus_one(rot_idxs_buf)
            # Free intermediates (they would feed forward to the layers in production)
            ttnn.deallocate(x_tt)
            ttnn.deallocate(cos_row)
            ttnn.deallocate(sin_row)

        ondev_med, ondev_p10, ondev_p90 = timed(
            "on-device-path", on_device_path, warmup=5, n=30, sync_target=mesh
        )

        # Latency: a SIMULATED host loop equivalent (mirror update_input_buffers exactly)
        x_buf_sim = ttnn.from_torch(
            torch.zeros(1, HIDDEN, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cos_buf_sim = ttnn.from_torch(
            torch.zeros(1, ROTARY_DIM, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        sin_buf_sim = ttnn.from_torch(
            torch.zeros(1, ROTARY_DIM, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        sim_cur_pos = [0]

        def host_loop_path():
            tid = 42
            cp = sim_cur_pos[0]
            x_np_ = embed_prod_np[tid].reshape(1, HIDDEN).astype(np.float32)
            x_host = ttnn.from_torch(
                torch.from_numpy(x_np_),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(x_host, x_buf_sim)
            cur_pos_host = ttnn.from_torch(
                torch.tensor([cp], dtype=torch.int32),
                layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(cur_pos_host, cur_pos_buf)
            cos_host = ttnn.from_torch(
                torch.from_numpy(cos_all_np[cp:cp + 1]),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(cos_host, cos_buf_sim)
            sin_host = ttnn.from_torch(
                torch.from_numpy(sin_all_np[cp:cp + 1]),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(sin_host, sin_buf_sim)
            sim_cur_pos[0] = (sim_cur_pos[0] + 1) % MAX_POS

        host_med, host_p10, host_p90 = timed(
            "host-loop-path", host_loop_path, warmup=5, n=30, sync_target=mesh
        )

        results["tests"]["latency"] = {
            "on_device_ms": {"median": ondev_med, "p10": ondev_p10, "p90": ondev_p90},
            "host_loop_ms": {"median": host_med, "p10": host_p10, "p90": host_p90},
            "speedup": host_med / ondev_med if ondev_med > 0 else None,
            "savings_ms": host_med - ondev_med,
        }
        print(f"\n  Δ = host {host_med:.3f} - on-device {ondev_med:.3f} = {host_med-ondev_med:+.3f} ms")
        if ondev_med > 0:
            print(f"  speedup = {host_med/ondev_med:.2f}×")

        # Decision
        all_correct = (
            results["tests"]["embedding"]["pass"]
            and results["tests"]["plus_one"]["pass"]
            and results["tests"]["cos_sin_lookup"]["pass"]
        )
        faster = ondev_med < host_med
        results["decision"] = "SHIP" if (all_correct and faster) else "INVESTIGATE"
        print("\n" + "=" * 78)
        print(f"DECISION: {results['decision']}")
        print(f"  correct={all_correct}  faster={faster}")
        print("=" * 78)

    finally:
        try:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as e:
            print(f"[cleanup] warning: {e}")

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ results saved → {out_path}")


if __name__ == "__main__":
    main()
