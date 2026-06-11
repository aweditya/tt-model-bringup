#!/usr/bin/env python3
"""Numpy sanity check: does HF V[pos=0] + repeat_kv + o_proj == HF mixer_out[pos=0]?

If YES, our understanding of the attention math at pos 0 is correct
and the bug in TT v0.1.2 is in the TT GQA/o_proj path (sharding,
all_reduce, or layout).

If NO, HF Gemma 4 attention is doing something we're not accounting
for (attn scaling, attn softcap, additional matmul, etc.) and we
need to read the HF code.

Run via main venv (no transformers needed; just safetensors + numpy):

    bash scripts/run_remote.sh experiments/cb/isolate/gm4_v012_oproj_sanity.py
"""
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORACLE = PROJECT_ROOT / ".cache" / "hf_oracle_gemma4_12b"


def cos(a, b):
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    snap = next((Path.home() / ".cache" / "huggingface" / "hub" /
                 "models--google--gemma-4-12B" / "snapshots").iterdir())
    # Gemma 4 12B ships a SINGLE model.safetensors (no index).
    sf = next(snap.glob("*.safetensors"))
    with safe_open(sf, framework="pt") as f:
        o_w = f.get_tensor("model.language_model.layers.0.self_attn.o_proj.weight").float().numpy()
    print(f"o_proj.weight shape: {o_w.shape}  (HF [out=HIDDEN, in=NQ*head_dim])")

    hf_v = np.load(ORACLE / "L0_attn_L0_v_proj.npy")[0]  # pos 0, shape [2048]
    print(f"HF v_proj[pos=0] shape: {hf_v.shape}  (= NKV * head_dim = 8 * 256)")

    # HF repeat_kv on V at pos 0: each KV head copied 2x to fill 16 Q heads.
    # V shape pre-repeat: [NKV=8, head_dim=256]
    # V shape post-repeat: [NQ=16, head_dim=256] with [v0, v0, v1, v1, ...].
    NKV, HEAD_DIM, NQ = 8, 256, 16
    GQA = NQ // NKV  # 2
    v_kv = hf_v.reshape(NKV, HEAD_DIM)
    v_repeated = np.repeat(v_kv, GQA, axis=0)  # [16, 256] interleaved
    flat = v_repeated.reshape(-1)  # [4096]
    print(f"v_repeated shape: {v_repeated.shape}, flat shape: {flat.shape}")

    # o_proj @ flat: HF Linear is out = x @ W^T so output = flat @ o_w.T = [HIDDEN]
    out_numpy = flat @ o_w.T  # [3840]
    print(f"computed o_proj output shape: {out_numpy.shape}")

    hf_mixer = np.load(ORACLE / "L0_mixer_out.npy")[0]  # pos 0
    hf_oproj = np.load(ORACLE / "L0_attn_L0_o_proj.npy")[0]
    print(f"HF mixer_out[pos=0] shape: {hf_mixer.shape}")
    print(f"HF o_proj_hook[pos=0] shape: {hf_oproj.shape}")

    print()
    print(f"sanity 1: HF mixer_out vs HF o_proj_hook        cos = {cos(hf_mixer, hf_oproj):.8f}")
    print(f"sanity 2: numpy(V*repeat_kv*o_proj) vs HF mixer cos = {cos(out_numpy, hf_mixer):.8f}")
    print(f"sanity 3: numpy(...)                vs HF oproj cos = {cos(out_numpy, hf_oproj):.8f}")
    print()
    print(f"rms HF mixer_out = {np.sqrt(np.mean(hf_mixer**2)):.4f}")
    print(f"rms numpy out    = {np.sqrt(np.mean(out_numpy**2)):.4f}")
    print(f"max abs diff numpy vs HF mixer: {np.abs(out_numpy - hf_mixer).max():.4f}")
    print(f"hf_mixer first 4 = {hf_mixer[:4]}")
    print(f"numpy out first 4= {out_numpy[:4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
