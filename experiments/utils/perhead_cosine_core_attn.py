#!/usr/bin/env python3
"""Per-head cosine diagnostic for dn_core_attn_out.

The on-device dn_core_attn_out has aggregate cosine 0.9999 vs HF at pos 0
but dn_norm output is only 0.95. Hypothesis: per-row (per-head) cosines
are highly non-uniform — some heads are perfect, some are very off,
averaging out to 0.9999 but breaking rms_norm.

This script:
  1. Bootstraps a small probe by running the cosine_probe_35b_ttnn machinery
     to capture dn_sub["dn_core_attn_out"] at pos 0
  2. Reshapes to [NV_HEADS=32, HEAD_V_DIM=128]
  3. Compares row-by-row to HF oracle's L0_dn_core_attn_out[0]

Run (qb1):
  cd ~/tt-xla && tt-smi -r && \
    export TT_METAL_HOME=... && \
    .venv/bin/python -u experiments/utils/perhead_cosine_core_attn.py
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_35b"
NV_HEADS = 32
HEAD_V_DIM = 128


def cosine(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("bootstrap…")
    state = srv.State()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    tok0 = int(prompt_ids[0])

    capt = {"sub_capture_layers": [0]}
    log(f"running step_forward_ttnn for pos 0 tok {tok0}…")
    next_id = srv.step_forward_ttnn(state, tok0, 0, capture=capt)
    log(f"  next_id={next_id}")

    my_core = capt["layer_0_sub"]["dn_sub"]["dn_core_attn_out"].reshape(NV_HEADS, HEAD_V_DIM)
    hf_core_full = np.load(ORACLE_DIR / "L0_dn_core_attn_out.npy")  # [5, 4096]
    hf_core = hf_core_full[0].reshape(NV_HEADS, HEAD_V_DIM)

    print("\nper-head cosines (server's dn_core_attn_out vs HF) at pos 0:")
    for h in range(NV_HEADS):
        c = cosine(my_core[h], hf_core[h])
        mag_my = np.linalg.norm(my_core[h])
        mag_hf = np.linalg.norm(hf_core[h])
        flag = "" if c > 0.99 else (" ⚠️" if c > 0.9 else " ❌")
        print(f"  head {h:2d}: cos={c:.4f}  |my|={mag_my:.4f}  |hf|={mag_hf:.4f}{flag}")

    # Also: print elementwise stats of the first divergent head
    bad_heads = [h for h in range(NV_HEADS) if cosine(my_core[h], hf_core[h]) < 0.99]
    if bad_heads:
        h0 = bad_heads[0]
        print(f"\nFirst divergent head {h0}: first 16 values:")
        print(f"  my: {my_core[h0, :16]}")
        print(f"  hf: {hf_core[h0, :16]}")


if __name__ == "__main__":
    main()
