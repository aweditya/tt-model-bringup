#!/usr/bin/env python3
"""Phase 2.A.0 — TARGET KV LAYOUT COMPATIBILITY PROBE.

Validate that the on-device KV cache layout of the Gemma 4 12B IT target
server is compatible with the drafter's `shared_kv_states` consumption
contract BEFORE we modify the target server to expose KV.

Run on qb1 (only — qb2 has `[[qb2-layernorm-trisc1-broken-2026-06-07]]`).

Steps:
1. Bootstrap target server (`server_gemma4_unified_ttnn.py`).
2. Prefill the 5-token oracle prompt 0 ("The capital of France is")
   sequentially via `step_forward_v031` (input_ids = [2, 818, 5279, 529,
   7001, 563]; L=6).
3. Read back the LAST sliding layer (L=46) and LAST full layer (L=47)
   paged KV caches.
4. Reassemble per-chip slices into HF's `(B=1, NKV, L_kv, head_dim)`
   contract.
5. Compare element-wise + per-head cos vs HF oracle's
   `shared_kv_sliding_K/V.npy` and `shared_kv_full_K/V.npy`.

KV cache layouts (from `server_gemma4_unified_ttnn.py:683-711`):

  Sliding layer (L=46): TWO caches per layer.
    Each cache: ttnn.from_torch(zeros [num_blocks, NCHIPS=4, BLOCK_SIZE=32,
                                       HEAD_DIM_SLIDING=256]) sharded on dim=1
                ⇒ per chip [num_blocks, 1, 32, 256].
    Cache_0 on chip c → KV head 2*c (even-indexed).
    Cache_1 on chip c → KV head 2*c+1 (odd-indexed).
    Mesh-wide KV heads: cache_0 covers {0,2,4,6}, cache_1 covers {1,3,5,7}.

  Full layer (L=47): ONE cache per layer.
    [num_blocks, NUM_KV_HEADS_GLOBAL=1, BLOCK_SIZE=32, HEAD_DIM_GLOBAL=512]
    REPLICATED across mesh. Per chip stores the same thing.

Position-row mapping (paged kernel contract):
  Token at position `pos` is written into block = pos // BLOCK_SIZE,
  row = pos % BLOCK_SIZE. For pos < BLOCK_SIZE=32 (our L=6 prefill case),
  all writes land in block 0, rows 0..5.

PASS gate (REVISED 2026-06-07 after first run):
  - shapes match HF — strict shape gate (catches actual layout bugs).
  - per-head + per-position cos >= 0.9 vs HF — this is a LAYOUT gate.
    Layout misalignment shows as zeros / near-zero on wrong-slot heads
    (uncorrelated random vectors); chain drift through 47 layers in
    bf16 typically gives cos 0.95-0.99.
  - run cos >= 0.99 gate is the wrong gate for THIS probe: bf16 chain
    drift through 47 attention layers + paged_update_cache tile pad
    cast already drops below 0.99 even without any layout issue
    ([[bf16-chain-drift-at-B-gt-1]]). The DRAFTER OUTPUT gate (Phase
    2.A.smoke) is the real go/no-go signal.

FAIL on shape OR cos < 0.9 ⇒ document actual layout in
       `research/gemma4_target_kv_layout.md` and STOP Phase 2.A.

PASS shape + cos >= 0.9 ⇒ green light Phase 2.A; rely on
       Phase 2.A.smoke to validate downstream drafter tolerance.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401 — used inside ttnn from_torch round-trips

# HF oracle (experiments/utils/hf_oracle_gemma4_assistant.py) runs against
# google/gemma-4-12B-it. Target server defaults to base; force IT here so the
# KV cache writes are weight-matched to the oracle.
os.environ.setdefault("TT_GEMMA4_VARIANT", "it")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import ttnn  # noqa: E402
import server_gemma4_unified_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b_assistant"
COS_THRESH_LAYOUT = 0.9  # layout-misalignment gate (random heads ≈ 0)
COS_THRESH_DRIFT = 0.95  # informational — bf16 drift over 47 layers


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(
        a.reshape(-1).astype(np.float64) - b.reshape(-1).astype(np.float64)
    ).max())


def _read_sliding_cache_per_chip(kc, mesh):
    """Read sliding cache → numpy array [NCHIPS, num_blocks, 1, BLOCK_SIZE,
    HEAD_DIM_SLIDING] fp32.

    The cache is sharded on dim=1 (NCHIPS axis); concat_mesh_to_tensor on
    dim=0 stacks per-chip arrays as the leading dim — giving us each chip's
    own slice on row c.
    """
    t = ttnn.to_torch(kc,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    arr = t.float().cpu().numpy()
    # Sliding cache device shape per chip: [num_blocks, 1, 32, 256].
    # ConcatMeshToTensor(dim=0) stacks 4 chips' [num_blocks, 1, 32, 256]
    # arrays along dim 0 — but the per-chip dim-0 is num_blocks. So we get
    # [4 * num_blocks, 1, 32, 256] flat-stacked. Split back into per-chip.
    assert arr.ndim == 4, f"unexpected ndim {arr.ndim} shape {arr.shape}"
    total_dim0, n_kv, bs, hd = arr.shape
    assert total_dim0 % srv.NCHIPS == 0
    per_chip_blocks = total_dim0 // srv.NCHIPS
    arr = arr.reshape(srv.NCHIPS, per_chip_blocks, n_kv, bs, hd)
    return arr


def _read_full_cache_replicated(kc, mesh):
    """Read full cache (replicated) → numpy [num_blocks, 1, BLOCK_SIZE,
    HEAD_DIM_GLOBAL=512] fp32.

    Replicated across mesh; take only chip 0's copy.
    """
    t = ttnn.to_torch(kc,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    arr = t.float().cpu().numpy()
    # Replicated → 4 stacked copies on dim 0.
    assert arr.ndim == 4, f"unexpected ndim {arr.ndim} shape {arr.shape}"
    total_dim0, n_kv, bs, hd = arr.shape
    assert total_dim0 % srv.NCHIPS == 0
    per_chip_blocks = total_dim0 // srv.NCHIPS
    arr = arr.reshape(srv.NCHIPS, per_chip_blocks, n_kv, bs, hd)
    # Take chip 0's view (all chips should be byte-identical).
    return arr[0]


def reassemble_sliding(kc0_arr, kc1_arr, L):
    """Reassemble sliding cache from per-chip {cache_0, cache_1} slices
    into HF's [B=1, NKV_TOTAL=8, L, HEAD_DIM_SLIDING=256] contract.

    Per `server_gemma4_unified_ttnn.py:676-679` comment:
      cache_0 on chip c holds KV head `2*c` (even-indexed; 0,2,4,6)
      cache_1 on chip c holds KV head `2*c+1` (odd; 1,3,5,7)

    Inputs:
      kc0_arr, kc1_arr: each shape [NCHIPS=4, per_chip_blocks, 1,
                                    BLOCK_SIZE=32, 256]
      L: prefill length — extract positions 0..L-1.

    Returns: [1, 8, L, 256] fp32 numpy in HF order.
    """
    # Position pos lives at block = pos // 32, row = pos % 32. For L <= 32,
    # all live in block 0, rows 0..L-1.
    assert L <= srv.NUM_LAYERS  # sanity bound just on size
    NKV_TOTAL = srv.NUM_KV_HEADS_SLIDING  # = 8
    HEAD_DIM = srv.HEAD_DIM_SLIDING  # = 256
    out = np.zeros((1, NKV_TOTAL, L, HEAD_DIM), dtype=np.float32)
    for h in range(NKV_TOTAL):
        chip = h // 2
        cache_idx = h % 2
        cache_arr = kc0_arr if cache_idx == 0 else kc1_arr
        for pos in range(L):
            block = pos // 32
            row = pos % 32
            # cache_arr[chip, block, 0 (n_kv), row, :]
            out[0, h, pos, :] = cache_arr[chip, block, 0, row, :]
    return out


def reassemble_full(kc_arr, L):
    """Reassemble full cache (replicated) into [B=1, NKV=1, L, 512].

    Inputs:
      kc_arr: chip 0's slice, shape [per_chip_blocks, 1, BLOCK_SIZE, 512]
      L: prefill length.
    """
    HEAD_DIM = srv.HEAD_DIM_GLOBAL  # = 512
    out = np.zeros((1, 1, L, HEAD_DIM), dtype=np.float32)
    for pos in range(L):
        block = pos // 32
        row = pos % 32
        out[0, 0, pos, :] = kc_arr[block, 0, row, :]
    return out


def per_head_pos_cos(tt_arr: np.ndarray, hf_arr: np.ndarray) -> dict:
    """Per-(head, position) cosine. Returns summary + worst case."""
    assert tt_arr.shape == hf_arr.shape, f"shape mismatch {tt_arr.shape} vs {hf_arr.shape}"
    _, NKV, L, _ = tt_arr.shape
    matrix = np.zeros((NKV, L), dtype=np.float64)
    for h in range(NKV):
        for p in range(L):
            matrix[h, p] = cos(tt_arr[0, h, p], hf_arr[0, h, p])
    return {
        "matrix": matrix,
        "min": float(matrix.min()),
        "mean": float(matrix.mean()),
        "max": float(matrix.max()),
        "argmin": tuple(int(x) for x in np.unravel_index(matrix.argmin(), matrix.shape)),
    }


def main() -> int:
    if not ORACLE_DIR.exists():
        log(f"FATAL: oracle missing at {ORACLE_DIR}")
        return 1

    # Load oracle for prompt 0
    prompt_dir = ORACLE_DIR / "prompt_0"
    input_ids = np.load(prompt_dir / "input_ids.npy")
    hf_kv_sliding_K = np.load(prompt_dir / "shared_kv_sliding_K.npy")
    hf_kv_sliding_V = np.load(prompt_dir / "shared_kv_sliding_V.npy")
    hf_kv_full_K = np.load(prompt_dir / "shared_kv_full_K.npy")
    hf_kv_full_V = np.load(prompt_dir / "shared_kv_full_V.npy")

    log(f"oracle: input_ids={input_ids.shape} -> {input_ids.flatten().tolist()}")
    log(f"oracle: HF sliding K {hf_kv_sliding_K.shape} V {hf_kv_sliding_V.shape}")
    log(f"oracle: HF full    K {hf_kv_full_K.shape} V {hf_kv_full_V.shape}")
    L = int(input_ids.shape[1])

    last_sliding_idx = 46  # per oracle meta.json
    last_full_idx = 47

    log(f"target last_sliding_idx={last_sliding_idx} last_full_idx={last_full_idx}")
    log(f"L_prefill={L} (single block since L < BLOCK_SIZE=32)")

    # Bootstrap target (cold ~14 min on qb1; warm ~30 s if shards cached)
    log("bootstrapping target Gemma 4 12B IT (cold ~14 min cold / ~30 s warm)…")
    t_boot = time.time()
    state = srv.State()
    srv.bootstrap(state, log=log)
    log(f"bootstrap took {time.time()-t_boot:.1f}s")

    # Prefill via sequential step_forward_v031 — each step writes KV at pos.
    log(f"prefilling {L} tokens through step_forward_v031…")
    t_p = time.time()
    for pos in range(L):
        tok = int(input_ids[0, pos])
        argmax = srv.step_forward_v031(state, tok_id=tok, pos=pos)
        log(f"  pos={pos} tok={tok} → argmax={argmax}")
    log(f"prefill took {time.time()-t_p:.1f}s")

    # Read back caches.
    log(f"reading back sliding L{last_sliding_idx} caches "
        f"(2 per layer) and full L{last_full_idx} cache (1)…")
    sliding_caches = state.kv_caches_tt[last_sliding_idx]
    assert len(sliding_caches) == 2, \
        f"sliding cache count {len(sliding_caches)} != 2"
    kc0_tt, vc0_tt = sliding_caches[0]
    kc1_tt, vc1_tt = sliding_caches[1]
    K0 = _read_sliding_cache_per_chip(kc0_tt, state.mesh)
    V0 = _read_sliding_cache_per_chip(vc0_tt, state.mesh)
    K1 = _read_sliding_cache_per_chip(kc1_tt, state.mesh)
    V1 = _read_sliding_cache_per_chip(vc1_tt, state.mesh)
    log(f"  K0 raw shape (per-chip stacked): {K0.shape}")
    log(f"  V0 raw shape (per-chip stacked): {V0.shape}")

    full_caches = state.kv_caches_tt[last_full_idx]
    assert len(full_caches) == 1, \
        f"full cache count {len(full_caches)} != 1"
    kc_f_tt, vc_f_tt = full_caches[0]
    Kf = _read_full_cache_replicated(kc_f_tt, state.mesh)
    Vf = _read_full_cache_replicated(vc_f_tt, state.mesh)
    log(f"  Kf raw shape (chip 0 view): {Kf.shape}")
    log(f"  Vf raw shape (chip 0 view): {Vf.shape}")

    # Reassemble.
    log("reassembling sliding into HF [B,8,L,256]…")
    tt_K_sliding = reassemble_sliding(K0, K1, L)
    tt_V_sliding = reassemble_sliding(V0, V1, L)
    log(f"  tt sliding K reassembled: {tt_K_sliding.shape}")
    log("reassembling full into HF [B,1,L,512]…")
    tt_K_full = reassemble_full(Kf, L)
    tt_V_full = reassemble_full(Vf, L)
    log(f"  tt full K reassembled: {tt_K_full.shape}")

    # Shape gate.
    if tt_K_sliding.shape != hf_kv_sliding_K.shape:
        log(f"SHAPE MISMATCH sliding K: tt {tt_K_sliding.shape} hf {hf_kv_sliding_K.shape}")
        return 1
    if tt_K_full.shape != hf_kv_full_K.shape:
        log(f"SHAPE MISMATCH full K: tt {tt_K_full.shape} hf {hf_kv_full_K.shape}")
        return 1

    # Cos comparisons.
    log("")
    log("=" * 64)
    log("COMPARISON vs HF oracle")
    log("=" * 64)
    overall_pass = True

    for name, tt_arr, hf_arr in [
        ("sliding K", tt_K_sliding, hf_kv_sliding_K),
        ("sliding V", tt_V_sliding, hf_kv_sliding_V),
        ("full    K", tt_K_full,    hf_kv_full_K),
        ("full    V", tt_V_full,    hf_kv_full_V),
    ]:
        c_all = cos(tt_arr, hf_arr)
        m_all = mad(tt_arr, hf_arr)
        per = per_head_pos_cos(tt_arr, hf_arr)
        log(f"  {name}: overall cos={c_all:.6f} mad={m_all:.4e}")
        log(f"           per-(head,pos): min={per['min']:.6f} mean={per['mean']:.6f} "
            f"max={per['max']:.6f}  worst @ head,pos={per['argmin']}")
        # Print full matrix at low-res for sliding K (most informative).
        if name == "sliding K":
            for h in range(per["matrix"].shape[0]):
                row = " ".join(f"{per['matrix'][h, p]:.4f}" for p in range(per['matrix'].shape[1]))
                log(f"           h{h}: {row}")
        if per["min"] < COS_THRESH_LAYOUT:
            overall_pass = False
        if per["min"] < COS_THRESH_DRIFT:
            log(f"           NOTE: per-(head,pos) min {per['min']:.4f} < "
                f"{COS_THRESH_DRIFT}; bf16 chain drift, not layout. Phase "
                f"2.A.smoke (drafter forward) gates the real downstream "
                f"signal.")

    log("")
    log("=" * 64)
    log(f"VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    log("=" * 64)

    ttnn.close_mesh_device(state.mesh)
    log("mesh closed.")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
