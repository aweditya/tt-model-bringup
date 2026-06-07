#!/usr/bin/env python3
"""MM7 v0.5.bench — NIAH stability check at L=128/512/1024.

Goal: long-context stability number for Nemotron-3 at our current 203
ms/tok eager baseline (post-v0.5.P1). Uses a RULER-style NIAH-single
prompt template (NOT our IT-chat-template needle that bit us 3x per
[[needle-prompt-shape-not-precision]]).

L=4k+ infeasible right now (prefill ~770 ms/tok = ~53 min/sample).
L=128/512/1024 are the realistic window for this session.

Prompt format mirrors RULER's NIAH-single shape:
  <hint> <filler...> <needle line> <filler...> <question>
where filler is synthetic deterministic text (not Paul Graham essays —
no external corpus dep). The needle line and question use a BASE prompt
shape (no chat-role markers) so we don't trigger IT-template echo.

Score: substring match for the needle's answer in the first 32 decoded
tokens. Per-length: retrieval rate + average wall time + first decoded
text sample.

Run via the nm3 dev harness:
  ssh qb1 'touch ~/tt-xla/.cache/nm3_runtime/trig/v05bench_niah'
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

ORACLE_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"
OUT_DIR = PROJECT_ROOT / "research" / "nemotron3_niah_2026-06-07"
N_LAYERS = 52
MAX_NEW_TOKENS = 32  # plenty for "The magic number is XXXXX." style answer

# Test matrix: (length_tokens, n_samples)
TEST_MATRIX = [
    (128, 3),
    (512, 3),
    (1024, 2),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── RULER-style NIAH-single template ──────────────────────────────────
HAYSTACK_FILLER_SENTENCES = [
    "The grass grows tall during summer months in northern climates.",
    "Many books are stored on wooden shelves in the old library.",
    "Boats sail across calm lakes when the wind is gentle.",
    "Farmers harvest crops in autumn before the first frost arrives.",
    "Children learn to read through patient practice and encouragement.",
    "Mountains rise sharply above the valley floor near the river.",
    "Bakers prepare bread early each morning before sunrise comes.",
    "Trains carry passengers between cities along fixed steel rails.",
    "Stars appear bright on clear cold nights away from city lights.",
    "Painters mix colors carefully on wooden palettes near windows.",
    "Rivers flow steadily toward distant seas across many regions.",
    "Gardeners tend flowers and vegetables during the growing season.",
]


def build_niah_prompt(target_len_tokens: int, seed: int, tokenizer) -> tuple[str, str, str]:
    """Build a NIAH-single prompt at approximately target_len_tokens.

    Returns (prompt_text, needle_answer, needle_question).
    """
    rng = random.Random(seed)
    needle_id = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3))
    needle_value = "".join(rng.choice("0123456789") for _ in range(6))
    needle_line = (
        f" The special magic number for {needle_id} is {needle_value}. "
    )
    question = f"What is the special magic number for {needle_id}?"

    intro = (
        "There is an important piece of information hidden somewhere in "
        "the text below. Read carefully and remember it.\n\n"
    )
    tail = (
        f"\n\nBased on the content above, answer the question.\n"
        f"Question: {question}\nAnswer:"
    )
    fixed = intro + tail
    fixed_ids = tokenizer.encode(fixed, add_special_tokens=False)
    needle_ids = tokenizer.encode(needle_line, add_special_tokens=False)
    overhead = len(fixed_ids) + len(needle_ids)
    target_filler_tokens = max(0, target_len_tokens - overhead)

    filler_chunks = []
    filler_tokens = 0
    while filler_tokens < target_filler_tokens:
        sentence = rng.choice(HAYSTACK_FILLER_SENTENCES) + " "
        filler_chunks.append(sentence)
        filler_tokens += len(tokenizer.encode(sentence, add_special_tokens=False))

    full_filler = "".join(filler_chunks)
    # Insert needle at ~50% depth (varies a bit per seed via rng split).
    depth = rng.uniform(0.3, 0.7)
    split = int(len(full_filler) * depth)
    haystack = full_filler[:split] + needle_line + full_filler[split:]

    prompt = intro + haystack + tail
    return prompt, needle_value, question


def _forward_layers(state, h_tt, srv, ttnn, *, attn_fn_name: str):
    attn_fn = getattr(srv, attn_fn_name)
    for L in range(N_LAYERS):
        kind = state.layer_types[L]
        if kind == "attention":
            h_next_tt = attn_fn(state, h_tt, L)
        elif kind == "mamba2":
            h_next_tt = srv.mamba2_block_eager_tt(state, h_tt, L)
        elif kind == "moe":
            h_next_tt = srv.moe_block_eager_ep_tt(state, h_tt, L)
        ttnn.deallocate(h_tt)
        h_tt = h_next_tt
    return h_tt


def run_one_sample(state, srv, ttnn, tokenizer, prompt_text: str,
                   needle_value: str) -> dict:
    """Prefill + decode + score one NIAH sample. Returns result dict."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = np.asarray(ids, dtype=np.int64)
    L = len(prompt_ids)

    srv.reset_decode_state(state, B=1, log=lambda *_a, **_k: None)

    # ── PREFILL ──────────────────────────────────────────────────────
    log(f"  prefill L={L} tokens…")
    t_pre = time.time()
    h_np = srv.embed_lookup(state, prompt_ids[None, :])
    h_tt = ttnn.from_torch(
        torch.from_numpy(h_np.astype(np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    h_tt = _forward_layers(state, h_tt, srv, ttnn, attn_fn_name="attn_prefill_tt")
    h_np = ttnn.to_torch(
        h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )[:1].float().numpy()
    ttnn.deallocate(h_tt)
    h_final = srv.apply_final_norm(state, h_np)
    _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
    prev_token = int(argmax_np.flatten()[-1])
    prefill_s = time.time() - t_pre
    state.cur_pos = L

    # ── DECODE ───────────────────────────────────────────────────────
    log(f"  prefill done in {prefill_s:.1f}s  first_token={prev_token}; "
        f"decoding {MAX_NEW_TOKENS} tokens…")
    decoded_tokens = [prev_token]
    cur_token = prev_token
    t_dec = time.time()
    for s in range(MAX_NEW_TOKENS - 1):
        srv.update_cur_pos_buf(state, int(state.cur_pos))
        h_np_dec = srv.embed_lookup(
            state, np.asarray([[cur_token]], dtype=np.int64),
        )
        h_tt = ttnn.from_torch(
            torch.from_numpy(h_np_dec.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        h_tt = _forward_layers(state, h_tt, srv, ttnn,
                               attn_fn_name="attn_decode_step_tt")
        h_np = ttnn.to_torch(
            h_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )[:1].float().numpy()
        ttnn.deallocate(h_tt)
        h_final = srv.apply_final_norm(state, h_np)
        _, argmax_np = srv.apply_lm_head_and_argmax(state, h_final)
        cur_token = int(argmax_np.flatten()[-1])
        decoded_tokens.append(cur_token)
        state.cur_pos += 1
    decode_s = time.time() - t_dec

    decoded_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
    retrieved = needle_value in decoded_text

    log(f"  decoded ({decode_s:.1f}s, {decode_s / MAX_NEW_TOKENS * 1000:.0f} "
        f"ms/tok): {decoded_text!r}")
    log(f"  needle={needle_value!r}  → "
        f"{'HIT ✓' if retrieved else 'MISS ✗'}")

    return {
        "L": L,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "ms_per_decode": decode_s / MAX_NEW_TOKENS * 1000,
        "needle": needle_value,
        "decoded_text": decoded_text,
        "decoded_tokens": decoded_tokens,
        "retrieved": retrieved,
    }


def main(state=None) -> int:
    os.environ.setdefault("NEMOTRON3_UPLOAD_LAYERS", "all")
    os.environ.setdefault("NEMOTRON3_MOE_MODE", "ep")
    os.environ.pop("NM3_ROUTER_ON_DEVICE", None)  # host topk for greedy gate

    import server_nemotron3_nano_ttnn as srv

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"results_{int(time.time())}.json"

    import ttnn

    if state is None:
        log("bootstrap…")
        state = srv.State()
        srv.bootstrap(state, log)
    else:
        log("[harness] reusing live state ✓")

    # Tokenizer: use the model's tokenizer (HF cache populated at bootstrap).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        trust_remote_code=True,
    )

    all_results = []
    per_L = {}
    for target_L, n_samples in TEST_MATRIX:
        log("")
        log("=" * 60)
        log(f"L≈{target_L} tokens, n_samples={n_samples}")
        log("=" * 60)
        per_L_results = []
        for i in range(n_samples):
            log("")
            log(f"sample {i + 1}/{n_samples} (L≈{target_L}, seed={target_L * 17 + i})")
            prompt_text, needle, _q = build_niah_prompt(
                target_L, seed=target_L * 17 + i, tokenizer=tok,
            )
            result = run_one_sample(state, srv, ttnn, tok, prompt_text, needle)
            per_L_results.append(result)
            all_results.append(result)
        retrieval_rate = sum(r["retrieved"] for r in per_L_results) / len(per_L_results)
        mean_prefill = statistics.mean(r["prefill_s"] for r in per_L_results)
        mean_decode_ms = statistics.mean(r["ms_per_decode"] for r in per_L_results)
        per_L[target_L] = {
            "n": len(per_L_results),
            "retrieval": retrieval_rate,
            "mean_prefill_s": mean_prefill,
            "mean_decode_ms": mean_decode_ms,
        }
        log("")
        log(f"L≈{target_L} summary: retrieval {retrieval_rate * 100:.0f}%  "
            f"prefill mean {mean_prefill:.1f}s  decode mean {mean_decode_ms:.0f} ms/tok")

    # ── REPORT + DUMP ────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("OVERALL REPORT — v0.5.bench NIAH-single, Nemotron-3 eager")
    log("=" * 60)
    log(f"{'L':>6} {'n':>4} {'retr':>8} {'prefill':>10} {'decode/tok':>12}")
    for target_L, stats in per_L.items():
        log(f"{target_L:>6} {stats['n']:>4} {stats['retrieval'] * 100:>7.0f}% "
            f"{stats['mean_prefill_s']:>8.1f}s  {stats['mean_decode_ms']:>8.0f} ms")
    log("")
    log("Comparison anchors (upstream RULER NIAH-single @ L=4096):")
    log("  Llama-3.1-8B-Instr:    ~95%")
    log("  Mistral-Nemo-12B:      ~95%")
    log("  Qwen2.5-7B-Instr:      ~93%")
    log("  Caveat: we're at L=128/512/1024, not L=4096 (prefill bottleneck).")

    out_payload = {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "test_matrix": TEST_MATRIX,
        "per_length": per_L,
        "samples": all_results,
        "ts": int(time.time()),
    }
    out_file.write_text(json.dumps(out_payload, indent=2))
    log("")
    log(f"  saved → {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
