#!/usr/bin/env python3
"""B16-cosine — Layer-by-layer cosine probe vs HF oracle.

Pre-req: run experiments/utils/hf_reference_35b.py first to populate
.cache/hf_oracle_35b/{prompt_ids,hidden_states,logits,argmax,final_norm}.npy.

What this does:
  - bootstrap the on-device 35B server (server_35b_ttnn.bootstrap)
  - reset_caches_ttnn()
  - for each prompt token position p in [0..seq-1]:
      step_forward_ttnn(state, prompt_ids[p], pos=p, capture=capt)
      compare capt["embed"|"layer_L"|"final_norm"|"logits"] vs
              hf_oracle.hidden_states[L+1][p] / logits[p]
      report cosine per layer, flag first layer cos < 0.95

  - at end, also report whether on-device argmax matches HF predicted
    next token at last position (' Paris' = id 11751)

Output: a markdown table per-position. Saves full numpy array of
cosines to .cache/cosine_probe_35b/cosines.npy for later analysis.

Run (qb1):
  rsync experiments/serve/server_35b_ttnn.py qb1:/home/aditya/tt-xla/.../
  rsync experiments/utils/cosine_probe_35b_ttnn.py qb1:/home/aditya/tt-xla/.../
  ssh qb1 'cd ~/tt-xla && tt-smi -r && [env exports] && \
    .venv/bin/python -u experiments/utils/cosine_probe_35b_ttnn.py'
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))
import server_35b_ttnn as srv  # noqa: E402

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_35b"
OUT_DIR = PROJECT_ROOT / ".cache" / "cosine_probe_35b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine(a, b):
    """Cosine similarity between two flat numpy vectors."""
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    if not (ORACLE_DIR / "meta.json").exists():
        log(f"FATAL: HF oracle not found at {ORACLE_DIR}/. "
            f"Run hf_reference_35b.py first.")
        sys.exit(1)

    meta = json.loads((ORACLE_DIR / "meta.json").read_text())
    prompt_ids = np.load(ORACLE_DIR / "prompt_ids.npy")
    hf_hidden = np.load(ORACLE_DIR / "hidden_states.npy")  # [41, seq, HIDDEN]
    hf_final_norm = np.load(ORACLE_DIR / "final_norm.npy")  # [seq, HIDDEN]
    hf_logits = np.load(ORACLE_DIR / "logits.npy")  # [seq, VOCAB]
    hf_argmax = np.load(ORACLE_DIR / "argmax.npy")  # [seq]

    seq = len(prompt_ids)
    n_layers = meta["n_layers"]
    log(f"oracle: prompt {meta['prompt']!r}, seq={seq}, n_layers={n_layers}")
    log(f"oracle predicted at pos {seq-1}: id={meta['predicted_token']} "
        f"text={meta['predicted_text']!r}")
    log(f"oracle argmax_per_pos: {[meta['argmax_text_per_position'][p] for p in range(seq)]}")

    log("bootstrap on-device server…")
    state = srv.State()
    srv.bootstrap(state, log)
    state.reset_caches_ttnn()

    # cosines[p, L+1] = cosine of layer L output vs HF; p=position
    # Plus 2 extras: [seq, n_layers + 1 (embed) + 1 (final_norm) + 1 (logits)]
    cosines = np.zeros((seq, n_layers + 3), dtype=np.float64)
    column_names = ["embed"] + [f"L{L}" for L in range(n_layers)] + ["final_norm", "logits"]
    on_device_argmax = np.zeros(seq, dtype=np.int32)

    # L0 sub-captures available?
    sub_keys = ["in_norm", "mixer_out", "post_attn_norm", "moe_out"]
    hf_L0_sub = {}
    for k in sub_keys:
        path = ORACLE_DIR / f"L0_{k}.npy"
        if path.exists():
            hf_L0_sub[k] = np.load(path)
    if hf_L0_sub:
        log(f"oracle L0 sub-captures available: {sorted(hf_L0_sub.keys())}")

    log("forwarding prompt token-by-token with per-layer capture…")
    for p in range(seq):
        tok_id = int(prompt_ids[p])
        capt = {"sub_capture_layers": [0]} if hf_L0_sub else {}
        t0 = time.time()
        next_id = srv.step_forward_ttnn(state, tok_id, p, capture=capt)
        dt = time.time() - t0
        on_device_argmax[p] = next_id

        # Compare embed
        cosines[p, 0] = cosine(capt["embed"], hf_hidden[0][p])
        # Compare each layer
        for L in range(n_layers):
            cosines[p, L + 1] = cosine(capt[f"layer_{L}"], hf_hidden[L + 1][p])
        # Compare final_norm + logits
        cosines[p, n_layers + 1] = cosine(capt["final_norm"], hf_final_norm[p])
        cosines[p, n_layers + 2] = cosine(capt["logits"], hf_logits[p])

        # Find first layer with cos < 0.95
        first_bad = None
        first_bad_cos = None
        for j, name in enumerate(column_names):
            if cosines[p, j] < 0.95:
                first_bad = name; first_bad_cos = cosines[p, j]
                break
        log(f"  pos {p} tok={tok_id} ({meta['argmax_text_per_position'][p]!r}): "
            f"argmax_on_dev={next_id} hf={int(hf_argmax[p])} "
            f"({'MATCH' if next_id == int(hf_argmax[p]) else 'MISS'})  "
            f"embed={cosines[p, 0]:.4f} L0={cosines[p, 1]:.4f} "
            f"L{n_layers-1}={cosines[p, n_layers]:.4f} "
            f"final={cosines[p, n_layers+1]:.4f} logits={cosines[p, n_layers+2]:.4f}  "
            f"({dt*1000:.0f} ms)"
            + (f"  first_bad={first_bad}@{first_bad_cos:.4f}" if first_bad else ""))

        if hf_L0_sub and "layer_0_sub" in capt:
            sub = capt["layer_0_sub"]
            sub_cos = {}
            for k in sub_keys:
                if k in sub and k in hf_L0_sub:
                    sub_cos[k] = cosine(sub[k], hf_L0_sub[k][p])
            log(f"    L0 sub: " + "  ".join(f"{k}={v:.4f}" for k, v in sub_cos.items()))

    # Save artifacts
    np.save(OUT_DIR / "cosines.npy", cosines)
    np.save(OUT_DIR / "on_device_argmax.npy", on_device_argmax)
    with open(OUT_DIR / "column_names.json", "w") as f:
        json.dump(column_names, f, indent=2)

    # Markdown table summary (compact: pos vs col, low-precision)
    log("\n=== cosine grid (rows=pos, cols=col_name, values=cos) ===")
    header = "pos | " + " ".join(f"{n:>7s}" for n in column_names[:4]) + " ... " + \
        " ".join(f"{n:>7s}" for n in column_names[-3:])
    log(header)
    for p in range(seq):
        row = f"{p:3d} | " + " ".join(f"{cosines[p, j]:7.4f}" for j in range(4)) + " ... " + \
            " ".join(f"{cosines[p, j]:7.4f}" for j in range(len(column_names) - 3, len(column_names)))
        log(row)

    log("\n=== first column (per pos) with cos < 0.95 ===")
    for p in range(seq):
        first_bad = None
        for j, name in enumerate(column_names):
            if cosines[p, j] < 0.95:
                first_bad = (name, cosines[p, j])
                break
        log(f"  pos {p}: " + (f"{first_bad[0]} cos={first_bad[1]:.4f}"
                              if first_bad else "all ≥ 0.95 ✓"))

    log("\nfull cosines saved to %s/cosines.npy" % OUT_DIR)


if __name__ == "__main__":
    main()
