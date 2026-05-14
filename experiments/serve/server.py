"""Persistent weight-loaded inference server for Qwen3.6-27B on Blackhole.

Phase 1: status, reload_kernels, reset_state, run_91r, shutdown.
Weights live in this process for the full session; clients send JSON over
a Unix socket and get one JSON response per connection.
"""
import argparse
import gc
import importlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from typing import Any, Optional

from experiments.serve import protocol as P

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_ID = "Qwen/Qwen3.6-27B"
EPS = 1e-6
MAX_POS = 256
HF_HIDDEN_PATH = os.path.expanduser("~/tt-xla/.cache/hf_per_layer_hidden_states.npz")
_91F_PATH = os.path.expanduser("~/tt-xla/experiments/91f_qwen36_27b_full_ondevice.py")
_91L_PATH = os.path.expanduser("~/tt-xla/experiments/91l_fp32_residual_generate.py")


class ServerState:
    """Resident state: weights, device, embed/lm_head, tokenizer, kernel module.

    Transient state (ssm/conv/kv) is rebuilt by reset_state and is not held here;
    each run owns its own transient state to keep tensor lifetime function-local.
    """
    def __init__(self):
        self.mock = False
        self.device = None
        self.device_id = None
        self.cfg: dict = {}
        self.num_layers: int = 0
        self.layer_weights: list = []           # [(layer_type, w_tt), ...]
        self.embed_np = None                    # np.ndarray [vocab, hidden] fp32
        self.final_norm_tt = None
        self.lm_head_tt = None
        self.cos_table_tt = None
        self.sin_table_tt = None
        # Level 1 RoPE: extended cos/sin tables (passthrough region = 1/0).
        # Lets apply_partial_rope use a no-slice-of-q formula, saving ~4 ops
        # per attn step. Math-identical (validated bit-exact).
        self.cos_ext_table_tt = None
        self.sin_ext_table_tt = None
        # Number of positions baked into cos/sin (ext) tables. Set by bootstrap
        # to MAX_POS and grown by ensure_rope_tables when a long-context path
        # asks for more. Must be checked before any positional slice.
        self.rope_table_size = 0
        self.tok = None
        self._91f = None                        # kernel module (re-importable)
        self._91l = None
        self.boot_time = time.time()
        self.last_run: Optional[dict] = None
        self.loaded = False
        # C'4 v4: lazily allocated for bench_decode_traced. Holds the captured
        # trace + all persistent buffers it depends on. Buffers MUST NOT be
        # reallocated after capture (trace is bound to specific addresses).
        # See handle_bench_decode_traced for the populated shape.
        self.traced_decode: Optional[dict] = None

    def status_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "mock": self.mock,
            "num_layers": self.num_layers,
            "device_id": self.device_id,
            "uptime_sec": time.time() - self.boot_time,
            "last_run": self.last_run,
        }


def _load_kernel_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bootstrap(state: ServerState, mock: bool, device_id: int) -> None:
    state.mock = mock
    state.device_id = device_id
    if mock:
        state.num_layers = 0
        state.cfg = {"mock": True}
        state.loaded = True
        print("[bootstrap] mock mode — skipped device/weight load")
        return

    # Real bootstrap mirrors demo_qwen36_27b.py [1/4] [2/4]
    import numpy as np
    import torch
    import ttnn
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    state._91f = _load_kernel_module(_91F_PATH, "_91f")
    state._91l = _load_kernel_module(_91L_PATH, "_91l")
    upload = state._91f.upload
    load_layer_weights_all = state._91f.load_layer_weights_all
    load_embed_lm_head_weights = state._91l.load_embed_lm_head_weights

    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)["text_config"]
    cfg = {
        "hidden":      text_cfg["hidden_size"],
        "n_k_heads":   text_cfg["linear_num_key_heads"],
        "n_v_heads":   text_cfg["linear_num_value_heads"],
        "k_dim":       text_cfg["linear_key_head_dim"],
        "v_dim":       text_cfg["linear_value_head_dim"],
        "conv_kernel": text_cfg["linear_conv_kernel_dim"],
        "n_q_heads":   text_cfg["num_attention_heads"],
        "n_kv_heads":  text_cfg["num_key_value_heads"],
        "head_dim":    text_cfg["head_dim"],
        "partial_rotary_factor": text_cfg["partial_rotary_factor"],
    }
    NUM_LAYERS = text_cfg["num_hidden_layers"]
    state.cfg = cfg
    state.num_layers = NUM_LAYERS

    print(f"[bootstrap] tokenizer + embed + lm_head…")
    state.tok = AutoTokenizer.from_pretrained(MODEL_ID)
    eweights = load_embed_lm_head_weights()
    state.embed_np = eweights["embed"]
    final_norm_np = eweights["final_norm"]
    lm_head_np = eweights["lm_head"]

    print(f"[bootstrap] opening device {device_id} + loading {NUM_LAYERS} layers (bf8)…")
    state.device = ttnn.open_device(device_id=device_id)
    device = state.device
    state.final_norm_tt = upload(final_norm_np, device, dtype=ttnn.bfloat16)
    state.lm_head_tt = upload(lm_head_np, device, dtype=ttnn.bfloat8_b)

    t0 = time.time()
    for i in range(NUM_LAYERS):
        layer_type = "linear_attention" if i % 4 != 3 else "full_attention"
        w_np = load_layer_weights_all(i, layer_type)
        w_tt = {}
        for k, arr in w_np.items():
            if k == "conv1d_weight" and arr.ndim == 3:
                arr = arr.squeeze(1)
            if "proj" in k or k == "conv1d_weight":
                dt = ttnn.bfloat8_b
            elif k in ("A_log", "dt_bias"):
                dt = ttnn.float32
            else:
                dt = ttnn.bfloat16
            w_tt[k] = upload(arr, device, dtype=dt)
        state.layer_weights.append((layer_type, w_tt))
        del w_np
        gc.collect()
        if i % 16 == 0 or i == NUM_LAYERS - 1:
            print(f"    layer {i:2d}  ({time.time()-t0:.0f}s elapsed)")
    print(f"[bootstrap] all {NUM_LAYERS} layers loaded in {time.time()-t0:.0f}s")

    # Pre-compute RoPE tables for the default (short-context) path.
    # ensure_rope_tables grows them later for long-context (generate_long).
    _build_rope_tables(state, MAX_POS)

    state.loaded = True
    print("[bootstrap] ready")


def _build_rope_tables(state: ServerState, table_size: int) -> None:
    """(Re)build cos/sin and extended cos/sin tables at `table_size` rows.

    Idempotent: safe to call multiple times. Replaces existing tables in place
    on the device. Must be called before any per-token slice that may index
    positions >= state.rope_table_size.

    cos_ext / sin_ext are HEAD_DIM-wide (rotate part + passthrough pad of 1/0)
    so apply_partial_rope can use the no-slice-of-q formula (validated bit-
    exact vs V1 in attn_step_rope_swap_probe.py).
    """
    import numpy as np
    import ttnn

    cfg = state.cfg
    device = state.device
    upload = state._91f.upload
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(table_size).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    state.cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)
    state.sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)

    head_dim = cfg["head_dim"]
    pad_size = head_dim - rotary_dim
    cos_ext_pad = np.ones((table_size, pad_size), dtype=np.float32)
    sin_ext_pad = np.zeros((table_size, pad_size), dtype=np.float32)
    cos_ext_all = np.concatenate([cos_all, cos_ext_pad], axis=-1).astype(np.float32)
    sin_ext_all = np.concatenate([sin_all, sin_ext_pad], axis=-1).astype(np.float32)
    state.cos_ext_table_tt = upload(cos_ext_all, device, dtype=ttnn.float32)
    state.sin_ext_table_tt = upload(sin_ext_all, device, dtype=ttnn.float32)
    state.rope_table_size = table_size
    print(f"[rope] cos/sin tables built at table_size={table_size}")


def ensure_rope_tables(state: ServerState, required_size: int) -> None:
    """Ensure the RoPE tables cover at least `required_size` positions.

    Long-context handlers (generate_long, bench_decode_paged) must call this
    before any per-token slice. Rebuilds the tables if they are too small.
    No-op if tables already large enough.
    """
    if state.rope_table_size >= required_size:
        return
    _build_rope_tables(state, required_size)


# ---------- handlers ----------

def handle_status(state: ServerState, args: dict) -> dict:
    return state.status_dict()


def handle_reload_kernels(state: ServerState, args: dict) -> dict:
    """Re-exec the kernel modules from disk; never cache function refs.

    importlib.reload doesn't work with spec_from_file_location-loaded
    modules (no findable spec for the synthetic name). We re-run the
    loader instead — state._91f gets a fresh module object pointing at
    the same name in sys.modules, and per-call dereference
    (state._91f.deltanet_step_ondevice) picks up the new code.
    """
    reloaded = []
    if state._91f is not None:
        state._91f = _load_kernel_module(_91F_PATH, "_91f")
        reloaded.append("_91f")
    if state._91l is not None:
        state._91l = _load_kernel_module(_91L_PATH, "_91l")
        reloaded.append("_91l")
    return {"ok": True, "reloaded_modules": reloaded}


def _fresh_state(state: ServerState):
    """Rebuild ssm/conv/kv transient buffers — mirrors demo_qwen36_27b.py:168-183."""
    import numpy as np
    import torch
    import ttnn
    cfg = state.cfg
    NUM_LAYERS = state.num_layers
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    upload = state._91f.upload
    device = state.device

    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]), dtype=np.float32),
                  device, dtype=ttnn.float32) for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                  device, dtype=ttnn.float32) for _ in range(n_dn)]
    kvc = []
    kv_init = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kvc.append([kv_k, kv_v])
    return ssm, cvs, kvc


def handle_reset_state(state: ServerState, args: dict) -> dict:
    if state.mock:
        return {"ok": True, "dt_sec": 0.0, "mock": True}
    t0 = time.time()
    # Reset is fast and produces no resident state — _fresh_state allocates
    # and the result is discarded; tests build their own transient state.
    _ = _fresh_state(state)
    gc.collect()
    return {"ok": True, "dt_sec": time.time() - t0}


def handle_run_91r(state: ServerState, args: dict) -> dict:
    """Run the 91r per-layer cosine gate against the resident weights.

    args: {"layers": [0,3,7,...], "weight_dtype": "bf8"|"bf16"|"fp32"}
    weight_dtype is accepted for parity with 91r CLI but ignored in phase 1:
    weights are uploaded at boot with bf8 projections and re-uploading per
    request would defeat the point of the server. To ablate, restart.
    """
    if state.mock:
        layers = args.get("layers") or [0, 3]
        # Deterministic mock cosines so the client smoke test has something
        # to compare against.
        return {
            "mock": True,
            "layers": layers,
            "results": [
                {"layer": L, "type": "linear_attention" if L % 4 != 3 else "full_attention",
                 "cosines": [0.999, 0.998, 0.998, 0.997, 0.997], "dt_sec": 0.01}
                for L in layers
            ],
        }

    import numpy as np
    import torch
    import ttnn

    layers_to_test = args.get("layers") or [0, 1, 2, 3, 7, 11, 15, 31, 47, 63]
    cfg = state.cfg

    if not os.path.exists(HF_HIDDEN_PATH):
        raise RuntimeError(f"HF hidden states missing at {HF_HIDDEN_PATH}; "
                           f"run experiments/utils/hf_full_model_oracle.py --dump-hidden-states")
    hf_data = np.load(HF_HIDDEN_PATH)

    def cosine(a, b):
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    HIDDEN = cfg["hidden"]
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    upload = state._91f.upload
    device = state.device

    results = []
    t_total = time.time()
    for layer_idx in layers_to_test:
        layer_type = "linear_attention" if layer_idx % 4 != 3 else "full_attention"
        if layer_idx >= state.num_layers:
            raise RuntimeError(f"layer {layer_idx} out of range (num_layers={state.num_layers})")
        _, w_tt = state.layer_weights[layer_idx]
        hf_in = hf_data[f"hidden_{layer_idx}"][0]
        hf_out = hf_data[f"hidden_{layer_idx+1}"][0]
        seq = hf_in.shape[0]

        t_layer = time.time()
        outputs = []
        if layer_type == "linear_attention":
            ssm_state = upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                                        dtype=np.float32), device, dtype=ttnn.float32)
            conv_state = upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                                 device, dtype=ttnn.float32)
            for pos in range(seq):
                x_tt = upload(hf_in[pos].reshape(1, HIDDEN), device, dtype=ttnn.float32)
                x_tt, ssm_state, conv_state = state._91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm_state, conv_state, cfg)
                x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
                ttnn.synchronize_device(device)
                outputs.append(ttnn.to_torch(x_tt).float().cpu().numpy().flatten()[:HIDDEN])
        else:
            # PAGED path: only used when args["paged"] is true. Cache shape is
            # [max_num_blocks, N_KV, BLOCK_SIZE, HEAD_DIM] and the layer takes a
            # page_table_tt argument. Validates that the paged eager kernel
            # produces the same cosine-vs-HF as the non-paged path.
            use_paged = bool(args.get("paged", False))
            if use_paged:
                BLOCK_SIZE = int(args.get("block_size", 64))
                assert MAX_POS % BLOCK_SIZE == 0, (
                    f"MAX_POS {MAX_POS} must be multiple of block_size {BLOCK_SIZE}")
                max_num_blocks = MAX_POS // BLOCK_SIZE
                paged_zero = np.zeros((max_num_blocks, cfg["n_kv_heads"],
                                         BLOCK_SIZE, cfg["head_dim"]), dtype=np.float32)
                kv_k = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=ttnn.bfloat16,
                                        device=device, layout=ttnn.TILE_LAYOUT,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
                kv_v = ttnn.from_torch(torch.from_numpy(paged_zero), dtype=ttnn.bfloat16,
                                        device=device, layout=ttnn.TILE_LAYOUT,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
                page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
                page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                                  dtype=ttnn.int32, device=device,
                                                  layout=ttnn.ROW_MAJOR_LAYOUT)
            else:
                kv_init = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
                kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                        device=device, layout=ttnn.TILE_LAYOUT)
                kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                        device=device, layout=ttnn.TILE_LAYOUT)
            for pos in range(seq):
                x_tt = upload(hf_in[pos].reshape(1, HIDDEN), device, dtype=ttnn.float32)
                cos_tt = ttnn.slice(state.cos_ext_table_tt, [pos, 0], [pos + 1, cfg["head_dim"]])
                sin_tt = ttnn.slice(state.sin_ext_table_tt, [pos, 0], [pos + 1, cfg["head_dim"]])
                if use_paged:
                    cur_pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32),
                                                   device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
                    x_tt = state._91f.gated_attn_step_ondevice_paged(
                        x_tt, w_tt, kv_k, kv_v, page_table_tt, cur_pos_tt,
                        cos_tt, sin_tt, cfg)
                else:
                    cur_pos_tt = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
                    x_tt, kv_k, kv_v = state._91f.gated_attn_step_ondevice(
                        x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, pos, cos_tt, sin_tt, cfg, device)
                x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
                ttnn.synchronize_device(device)
                outputs.append(ttnn.to_torch(x_tt).float().cpu().numpy().flatten()[:HIDDEN])
        dt = time.time() - t_layer
        ttnn_out = np.stack(outputs)
        cos_per_pos = [cosine(hf_out[p], ttnn_out[p]) for p in range(seq)]
        norms_hf = [float(np.linalg.norm(hf_out[p])) for p in range(seq)]
        norms_ttnn = [float(np.linalg.norm(ttnn_out[p])) for p in range(seq)]
        results.append({
            "layer": layer_idx,
            "type": layer_type,
            "cosines": cos_per_pos,
            "norms_hf": norms_hf,
            "norms_ttnn": norms_ttnn,
            "dt_sec": dt,
        })
        gc.collect()

    summary = {
        "layers": layers_to_test,
        "results": results,
        "total_sec": time.time() - t_total,
    }
    state.last_run = {"cmd": "run_91r", "layers": layers_to_test,
                      "total_sec": summary["total_sec"], "ts": time.time()}
    return summary


def handle_bench_decode(state: ServerState, args: dict) -> dict:
    """Time N decode steps through all 64 layers + lm_head.

    args:
      prompt: str (default: "The capital of France is")
      n_steps: int (default: 20)
      warmup: int (default: 3)
    Returns per-step timings + median/mean/p95 ms.

    Uses sync-bounded host timing (synchronize_device before and after each
    step). Matches the perf_baseline.py prefill_plus_one_decode pattern but
    runs against resident weights — no 11-min reload between configs.
    """
    if state.mock:
        return {"mock": True, "median_ms": 100.0, "mean_ms": 100.0, "p95_ms": 100.0,
                "n_steps": args.get("n_steps", 20)}

    import numpy as np
    import torch
    import ttnn
    import statistics

    prompt = args.get("prompt", "The capital of France is")
    n_steps = int(args.get("n_steps", 20))
    warmup = int(args.get("warmup", 3))

    # ServerState doesn't carry a tokenizer (Phase 2 todo). For benchmarking
    # purposes the prompt content doesn't matter, only the token IDs that
    # drive the embed lookup. Default to the validated Paris prompt.
    DEFAULT_PROMPT_IDS = [760, 6511, 314, 9338, 369]   # "The capital of France is"
    explicit_ids = args.get("prompt_ids")
    if explicit_ids is not None:
        prompt_ids_override = list(explicit_ids)
    else:
        prompt_ids_override = DEFAULT_PROMPT_IDS

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    NUM_LAYERS = state.num_layers
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    upload = state._91f.upload
    device = state.device

    # Fresh state — analogous to 91l's `fresh_state` pattern
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]
    kvc = []
    kv_init = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kvc.append([kv_k, kv_v])

    prompt_ids = prompt_ids_override
    embed_np = state.embed_np

    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        # V2 rotate-only RoPE: slice ROTARY_DIM-wide ONCE per token (not HEAD_DIM).
        # 91f's apply_partial_rope detects ROTARY_DIM input and skips its own slice
        # → eliminates 32 slice dispatches/token (16 attn layers × 2 for cos+sin).
        cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = state.layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm[dn_idx], cvs[dn_idx] = state._91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt, kv_k, kv_v = state._91f.gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kvc[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
        logits_tt = ttnn.linear(x_tt, state.lm_head_tt, compute_kernel_config=state._91f.hifi4)
        return logits_tt

    # Prefill (untimed)
    for pos, tid in enumerate(prompt_ids):
        _ = forward_token(tid, pos)
    ttnn.synchronize_device(device)

    # The "decode step" we measure is one forward pass at pos=len(prompt_ids),
    # using the last prompt token as input. Reset state between repeats so
    # each measurement runs against IDENTICAL state (same as perf_baseline).
    times_ms = []
    for rep in range(warmup + n_steps):
        # rebuild fresh state
        for i in range(n_dn):
            ssm[i] = upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                                      dtype=np.float32), device, dtype=ttnn.float32)
            cvs[i] = upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                              device, dtype=ttnn.float32)
        for i in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kvc[i] = [kv_k, kv_v]
        # warm prefill (untimed)
        for pos, tid in enumerate(prompt_ids):
            _ = forward_token(tid, pos)
        ttnn.synchronize_device(device)
        # timed step
        ttnn.synchronize_device(device)
        t0 = time.time()
        _ = forward_token(prompt_ids[-1], len(prompt_ids))
        ttnn.synchronize_device(device)
        t1 = time.time()
        if rep >= warmup:
            times_ms.append((t1 - t0) * 1000)

    times_ms.sort()
    median = statistics.median(times_ms)
    mean = statistics.mean(times_ms)
    p95 = times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) >= 5 else max(times_ms)

    summary = {
        "prompt": prompt,
        "n_steps": n_steps,
        "warmup": warmup,
        "median_ms": median,
        "mean_ms": mean,
        "p95_ms": p95,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "tok_per_sec": 1000.0 / median,
        "all_ms": times_ms,
    }
    state.last_run = {"cmd": "bench_decode", "median_ms": median,
                      "tok_per_sec": 1000.0 / median, "ts": time.time()}
    return summary


def handle_bench_decode_paged(state: ServerState, args: dict) -> dict:
    """Bench eager decode using PAGED KV cache + paged_scaled_dot_product_-
    attention_decode. Same per-step methodology as bench_decode, but exercises
    the long-context path (unlocks MAX_POS > 256).

    args:
      n_steps:    int, default 20
      warmup:     int, default 3
      max_pos:    int, default 256 (cache max_seq_len; paged unlocks higher)
      block_size: int, default 64
    """
    if state.mock:
        return {"mock": True, "median_ms": 0.0, "paged": True}
    import numpy as np
    import torch
    import ttnn
    import statistics

    n_steps = int(args.get("n_steps", 20))
    warmup = int(args.get("warmup", 3))
    max_pos = int(args.get("max_pos", 256))
    block_size = int(args.get("block_size", 64))
    assert max_pos % block_size == 0, f"max_pos {max_pos} must be multiple of block_size {block_size}"
    ensure_rope_tables(state, max_pos)

    DEFAULT_PROMPT_IDS = [760, 6511, 314, 9338, 369]
    prompt_ids = list(args.get("prompt_ids") or DEFAULT_PROMPT_IDS)

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    ROTARY_DIM = int(HEAD_DIM * cfg["partial_rotary_factor"])
    NUM_LAYERS = state.num_layers
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    upload = state._91f.upload
    device = state.device

    max_num_blocks = max_pos // block_size  # single user
    print(f"[bench_decode_paged] max_pos={max_pos} block_size={block_size} "
          f"max_num_blocks={max_num_blocks}")

    # === Allocate page_table once (identity mapping for single user) ===
    page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

    # === Allocate DN state (unchanged from non-paged) ===
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]

    # === Allocate paged KV caches ===
    paged_kv_zero = np.zeros((max_num_blocks, N_KV, block_size, HEAD_DIM), dtype=np.float32)
    kvc = []
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_v = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kvc.append([kv_k, kv_v])

    embed_np = state.embed_np

    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        # V2 rotate-only RoPE: slice ROTARY_DIM-wide ONCE per token (paged path).
        cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                       device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = state.layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm[dn_idx], cvs[dn_idx] = state._91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt = state._91f.gated_attn_step_ondevice_paged(
                    x_tt, w_tt, kv_k, kv_v, page_table_tt, cur_pos_tt,
                    cos_tt, sin_tt, cfg)
                attn_idx += 1
            x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
        logits_tt = ttnn.linear(x_tt, state.lm_head_tt, compute_kernel_config=state._91f.hifi4)
        return logits_tt

    # === Prefill (untimed) ===
    for pos, tid in enumerate(prompt_ids):
        _ = forward_token(tid, pos)
    ttnn.synchronize_device(device)

    # === Timed step (reset state between reps for apples-to-apples) ===
    times_ms = []
    for rep in range(warmup + n_steps):
        for i in range(n_dn):
            ssm[i] = upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                                      dtype=np.float32), device, dtype=ttnn.float32)
            cvs[i] = upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                              device, dtype=ttnn.float32)
        for i in range(n_attn):
            kv_k = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kv_v = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
            kvc[i] = [kv_k, kv_v]
        for pos, tid in enumerate(prompt_ids):
            _ = forward_token(tid, pos)
        ttnn.synchronize_device(device)
        ttnn.synchronize_device(device)
        t0 = time.time()
        _ = forward_token(prompt_ids[-1], len(prompt_ids))
        ttnn.synchronize_device(device)
        t1 = time.time()
        if rep >= warmup:
            times_ms.append((t1 - t0) * 1000)

    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[int(len(times_ms) * 0.95) - 1] if len(times_ms) >= 5 else max(times_ms)
    summary = {
        "paged": True,
        "max_pos": max_pos,
        "block_size": block_size,
        "n_steps": n_steps,
        "median_ms": median,
        "p95_ms": p95,
        "min_ms": min(times_ms), "max_ms": max(times_ms),
        "tok_per_sec": 1000.0 / median,
        "all_ms": times_ms,
    }
    state.last_run = {"cmd": "bench_decode_paged", "median_ms": median,
                        "tok_per_sec": 1000.0 / median, "max_pos": max_pos,
                        "ts": time.time()}
    return summary


# ---------- C'4 v4: traced decode helpers ----------

def _setup_traced_decode(state: ServerState) -> None:
    """Allocate persistent state buffers + input buffers, then capture the
    decode trace using the current state._91f kernels (traced variants).

    All buffers stored on state.traced_decode are aliased by the captured
    trace. Reallocating any of them invalidates the trace; subsequent
    execute_trace calls will fault. The only safe in-place update is
    ttnn.copy_host_to_device_tensor with same shape/dtype.
    """
    import numpy as np
    import torch
    import ttnn

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    rotary_dim = int(HEAD_DIM * cfg["partial_rotary_factor"])
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    NUM_LAYERS = state.num_layers
    upload = state._91f.upload
    device = state.device

    # Persistent state buffers (in-place mutated by in-trace ttnn.copy)
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]
    kv_init = np.zeros((1, N_KV, MAX_POS, HEAD_DIM), dtype=np.float32)
    kvc = []
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kvc.append([kv_k, kv_v])

    # Per-step input buffers (written by update_input_buffers each step)
    embed_buf = upload(np.zeros((1, HIDDEN), dtype=np.float32),
                        device, dtype=ttnn.float32)
    cos_buf = upload(np.zeros((1, HEAD_DIM), dtype=np.float32),
                        device, dtype=ttnn.float32)
    sin_buf = upload(np.zeros((1, HEAD_DIM), dtype=np.float32),
                        device, dtype=ttnn.float32)
    cur_pos_buf = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), device=device)
    index_buf = ttnn.from_torch(
        torch.from_numpy(np.zeros((1, N_KV, 1, HEAD_DIM), dtype=np.int32)),
        dtype=ttnn.int32, device=device, layout=ttnn.TILE_LAYOUT)

    # Warmup pass: run forward once EAGERLY to JIT-compile + upload all kernel
    # binaries. Without this, trace capture would try to upload kernel binaries
    # MID-CAPTURE (via load_binaries -> enqueue_write_shard), which trips
    # "TT_FATAL: Writes are not supported during trace capture". The traced
    # kernels mutate state buffers in place via in-trace ttnn.copy, so this
    # warmup advances state by one decode step — we reset state below.
    print(f"[traced_decode] warmup pass (priming JIT)…")
    state.traced_decode = {  # placeholder so _update_input_buffers can find the bufs
        "embed_buf": embed_buf, "cos_buf": cos_buf, "sin_buf": sin_buf,
        "cur_pos_buf": cur_pos_buf, "index_buf": index_buf,
        "ssm": ssm, "cvs": cvs, "kvc": kvc,
    }
    _update_input_buffers(state, token_id=0, cur_pos=0)
    _x = embed_buf
    _dn = 0
    _attn = 0
    for i in range(NUM_LAYERS):
        layer_type, w_tt = state.layer_weights[i]
        if layer_type == "linear_attention":
            _x = state._91f.deltanet_step_ondevice_traced(
                _x, w_tt, ssm[_dn], cvs[_dn], cfg)
            _dn += 1
        else:
            kv_k, kv_v = kvc[_attn]
            _x = state._91f.gated_attn_step_ondevice_traced(
                _x, w_tt, kv_k, kv_v,
                cur_pos_buf, cos_buf, sin_buf, index_buf, cfg)
            _attn += 1
        _x = state._91f.mlp_step_ondevice(_x, w_tt)
    _x = ttnn.rms_norm(_x, weight=state.final_norm_tt, epsilon=EPS)
    _logits = ttnn.linear(_x, state.lm_head_tt,
                            compute_kernel_config=state._91f.hifi4)
    ttnn.synchronize_device(device)
    print(f"[traced_decode] warmup done; resetting state in place")

    # Reset state buffers to zero (in place — same buffer addresses) before capture
    # so the captured trace starts from a known state.
    _reset_traced_state_inplace(state)
    ttnn.synchronize_device(device)

    # Capture trace. All ops have warm JIT cache → no host writes during capture.
    print(f"[traced_decode] begin_trace_capture…")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    x_tt = embed_buf
    dn_idx = 0
    attn_idx = 0
    for i in range(NUM_LAYERS):
        layer_type, w_tt = state.layer_weights[i]
        if layer_type == "linear_attention":
            x_tt = state._91f.deltanet_step_ondevice_traced(
                x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
            dn_idx += 1
        else:
            kv_k, kv_v = kvc[attn_idx]
            x_tt = state._91f.gated_attn_step_ondevice_traced(
                x_tt, w_tt, kv_k, kv_v,
                cur_pos_buf, cos_buf, sin_buf, index_buf, cfg)
            attn_idx += 1
        x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
    x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=EPS)
    logits_tt = ttnn.linear(x_tt, state.lm_head_tt,
                              compute_kernel_config=state._91f.hifi4)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    state.traced_decode = {
        "trace_id": tid,
        "logits_tt": logits_tt,
        "ssm": ssm, "cvs": cvs, "kvc": kvc,
        "embed_buf": embed_buf, "cos_buf": cos_buf, "sin_buf": sin_buf,
        "cur_pos_buf": cur_pos_buf, "index_buf": index_buf,
        "n_dn": n_dn, "n_attn": n_attn,
    }


def _release_traced_decode(state: ServerState) -> None:
    """Release the captured trace and drop all buffer references so GC can
    free them before we recapture."""
    if state.traced_decode is None:
        return
    import ttnn
    try:
        ttnn.release_trace(state.device, state.traced_decode["trace_id"])
    except Exception as e:
        print(f"[traced_decode] release_trace error: {e}")
    state.traced_decode = None
    gc.collect()


def _reset_traced_state_inplace(state: ServerState) -> None:
    """Zero out all persistent state buffers WITHOUT reallocating (the trace
    is bound to these addresses). Uses copy_host_to_device_tensor."""
    import numpy as np
    import torch
    import ttnn
    cfg = state.cfg
    td = state.traced_decode
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM

    ssm_zero = ttnn.from_torch(
        torch.from_numpy(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                                   dtype=np.float32)),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    cvs_zero = ttnn.from_torch(
        torch.from_numpy(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32)),
        dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    kv_zero = ttnn.from_torch(
        torch.from_numpy(np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]),
                                   dtype=np.float32)),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for s in td["ssm"]:
        ttnn.copy_host_to_device_tensor(ssm_zero, s)
    for c in td["cvs"]:
        ttnn.copy_host_to_device_tensor(cvs_zero, c)
    for kv_k, kv_v in td["kvc"]:
        ttnn.copy_host_to_device_tensor(kv_zero, kv_k)
        ttnn.copy_host_to_device_tensor(kv_zero, kv_v)


def _update_input_buffers(state: ServerState, token_id: int, cur_pos: int) -> None:
    """Write per-step inputs into pre-allocated buffers (NO device alloc)."""
    import numpy as np
    import torch
    import ttnn
    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    td = state.traced_decode

    # embed row (host slice; copy into device buffer)
    embed_row = state.embed_np[token_id].reshape(1, HIDDEN).astype(np.float32)
    src_e = ttnn.from_torch(torch.from_numpy(embed_row),
                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(src_e, td["embed_buf"])
    # RoPE row (Level 1 extended, host-sliced from precomputed table)
    # cos/sin tables on device are upload(cos_ext_all). Recover host array by
    # reading them out... too slow per step. Recompute from cfg instead.
    if "_rope_cache_np" not in td:
        rotary_dim = int(HEAD_DIM * cfg["partial_rotary_factor"])
        half_rot = rotary_dim // 2
        freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
        positions = np.arange(MAX_POS).astype(np.float32)
        ang = positions[:, None] * freqs[None, :]
        cos_all = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
        sin_all = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
        pad = HEAD_DIM - rotary_dim
        cos_ext = np.concatenate([cos_all, np.ones((MAX_POS, pad), dtype=np.float32)], axis=-1)
        sin_ext = np.concatenate([sin_all, np.zeros((MAX_POS, pad), dtype=np.float32)], axis=-1)
        td["_rope_cache_np"] = (cos_ext, sin_ext)
    cos_ext, sin_ext = td["_rope_cache_np"]
    src_c = ttnn.from_torch(torch.from_numpy(cos_ext[cur_pos:cur_pos + 1]),
                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    src_s = ttnn.from_torch(torch.from_numpy(sin_ext[cur_pos:cur_pos + 1]),
                              dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(src_c, td["cos_buf"])
    ttnn.copy_host_to_device_tensor(src_s, td["sin_buf"])
    # cur_pos scalar
    src_p = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32))
    ttnn.copy_host_to_device_tensor(src_p, td["cur_pos_buf"])
    # scatter index [1, N_KV, 1, HEAD_DIM]
    idx = np.full((1, N_KV, 1, HEAD_DIM), cur_pos, dtype=np.int32)
    src_i = ttnn.from_torch(torch.from_numpy(idx),
                              dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_host_to_device_tensor(src_i, td["index_buf"])


def _eager_step_logits(state: ServerState, ssm, cvs, kvc,
                        token_id: int, cur_pos: int):
    """Run one eager forward, return logits np array. Mutates ssm/cvs/kvc by
    rebinding entries (eager kernels return new tensors)."""
    import numpy as np
    import torch
    import ttnn
    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    HEAD_DIM = cfg["head_dim"]
    device = state.device
    upload = state._91f.upload

    x_tt = upload(state.embed_np[token_id].reshape(1, HIDDEN),
                  device, dtype=ttnn.float32)
    cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, HEAD_DIM])
    sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, HEAD_DIM])
    cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
    dn_idx = 0
    attn_idx = 0
    for i in range(state.num_layers):
        layer_type, w_tt = state.layer_weights[i]
        if layer_type == "linear_attention":
            x_tt, ssm[dn_idx], cvs[dn_idx] = state._91f.deltanet_step_ondevice(
                x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
            dn_idx += 1
        else:
            kv_k, kv_v = kvc[attn_idx]
            x_tt, kv_k, kv_v = state._91f.gated_attn_step_ondevice(
                x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                cos_tt, sin_tt, cfg, device)
            kvc[attn_idx] = [kv_k, kv_v]
            attn_idx += 1
        x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
    x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=EPS)
    logits_tt = ttnn.linear(x_tt, state.lm_head_tt,
                              compute_kernel_config=state._91f.hifi4)
    ttnn.synchronize_device(device)
    return ttnn.to_torch(logits_tt).float().cpu().numpy().flatten()


def handle_bench_decode_traced(state: ServerState, args: dict) -> dict:
    """C'4 v4: capture + bench traced decode, validate vs eager per-step.

    args:
      n_steps:        timed traced steps (default 20)
      warmup:         untimed traced steps before timing (default 5)
      validate_steps: per-step cosine compare vs eager (default 5, 0 to skip)
      start_token_id: arbitrary starting token (default 760 = "The")
      recapture:      release + recapture trace (use after reload_kernels)

    Returns timing + min/per-step cosine + top1 match per step.
    """
    if state.mock:
        return {"mock": True, "median_ms": 0.0, "min_cosine": 1.0}
    import numpy as np
    import ttnn
    import statistics

    n_steps = int(args.get("n_steps", 20))
    n_warmup = int(args.get("warmup", 5))
    n_validate = int(args.get("validate_steps", 5))
    start_token = int(args.get("start_token_id", 760))
    recapture = bool(args.get("recapture", False))

    if state.traced_decode is None or recapture:
        if state.traced_decode is not None:
            _release_traced_decode(state)
        t0 = time.time()
        _setup_traced_decode(state)
        capture_sec = time.time() - t0
    else:
        capture_sec = 0.0

    td = state.traced_decode
    device = state.device

    # 1) Validation. Both paths start from zero state, run N steps, compare logits.
    cosines = []
    top1_match = []
    if n_validate > 0:
        ssm_e, cvs_e, kvc_e = _fresh_state(state)
        eager_logits_step = []
        next_id = start_token
        for i in range(n_validate):
            logits = _eager_step_logits(state, ssm_e, cvs_e, kvc_e, next_id, i)
            eager_logits_step.append(logits)
            next_id = int(np.argmax(logits))
        del ssm_e, cvs_e, kvc_e
        gc.collect()

        _reset_traced_state_inplace(state)
        next_id = start_token
        for i in range(n_validate):
            _update_input_buffers(state, next_id, i)
            ttnn.execute_trace(device, td["trace_id"], cq_id=0, blocking=True)
            logits = ttnn.to_torch(td["logits_tt"]).float().cpu().numpy().flatten()
            e = eager_logits_step[i]
            num = float(np.dot(e.astype(np.float64), logits.astype(np.float64)))
            den = float(np.linalg.norm(e) * np.linalg.norm(logits) + 1e-12)
            cosines.append(num / den)
            top1_match.append(int(np.argmax(e)) == int(np.argmax(logits)))
            next_id = int(np.argmax(logits))

    # 2) Perf bench. Reset state, time N traced steps. Each step is a full
    #    update_input_buffers + execute_trace + readback (matches eager bench
    #    methodology — full-step time, not just execute_trace).
    _reset_traced_state_inplace(state)
    times_ms = []
    times_exec_ms = []
    next_id = start_token
    cur_pos = 0
    for i in range(n_warmup + n_steps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        _update_input_buffers(state, next_id, cur_pos)
        t1 = time.perf_counter()
        ttnn.execute_trace(device, td["trace_id"], cq_id=0, blocking=True)
        t2 = time.perf_counter()
        logits = ttnn.to_torch(td["logits_tt"]).float().cpu().numpy().flatten()
        t3 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t3 - t0) * 1000)
            times_exec_ms.append((t2 - t1) * 1000)
        next_id = int(np.argmax(logits))
        cur_pos += 1

    median = statistics.median(times_ms)
    median_exec = statistics.median(times_exec_ms)
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95) - 1] if len(times_ms) >= 5 else max(times_ms)

    summary = {
        "n_steps": n_steps,
        "warmup": n_warmup,
        "validate_steps": n_validate,
        "capture_sec": capture_sec,
        "cosines": cosines,
        "min_cosine": min(cosines) if cosines else None,
        "top1_match": top1_match,
        "all_top1_match": all(top1_match) if top1_match else None,
        "median_ms": median,
        "median_exec_ms": median_exec,
        "p95_ms": p95,
        "tok_per_sec": 1000.0 / median,
        "all_ms": times_ms,
    }
    state.last_run = {"cmd": "bench_decode_traced", "median_ms": median,
                       "tok_per_sec": 1000.0 / median,
                       "min_cosine": summary["min_cosine"], "ts": time.time()}
    return summary


def handle_shutdown(state: ServerState, args: dict) -> dict:
    return {"ok": True, "shutting_down": True}


def _sample_next_id(
    logits_np,
    temperature: float,
    top_p: float,
    rng,
    repetition_penalty: float = 1.0,
    min_p: float = 0.0,
    recent_ids=None,
    no_repeat_ngram_size: int = 0,
    dry_multiplier: float = 0.0,
    dry_base: float = 1.75,
    dry_allowed_length: int = 2,
    full_history_ids=None,
):
    """Pick next token. Defaults reproduce greedy argmax.

    Pipeline (matches HuggingFace LogitsProcessor ordering, plus DRY between
    rep-penalty and temperature per llama.cpp convention):
      1. repetition_penalty — divide positive logits / multiply negative for ids
         in `recent_ids` (CTRL paper, Keskar et al. 2019, arXiv:1909.05858).
         Default 1.0 = no-op. Recommended 1.1-1.3 to break loops.
      1b. no_repeat_ngram_size — ban tokens that would complete a repeated
          n-gram against `full_history_ids` (HuggingFace generation flag).
          Default 0 = disabled. 3-4 is the standard setting; kills exact loops
          like the whitespace/newline drift on the paged path.
      1c. DRY sampler — Don't Repeat Yourself (llama.cpp PR #6839,
          l3utterfly 2024). Subtracts `multiplier * base^(match_len -
          allowed_length)` from logit of any token that would extend a
          repeat. Default multiplier=0 = disabled.
      2. temperature — scale logits. 0.0 (default) → skip softmax, argmax over
         the (possibly rep-penalized) logits → still deterministic.
      3. min-p — truncate tokens with prob < min_p * max(prob) (Nguyen et al.
         2024, arXiv:2407.01082). Default 0.0 = no-op. Recommended 0.05-0.1;
         the paper argues this beats top_p for coherence at higher temps.
      4. top-p — nucleus cutoff. Default 1.0 = no-op.
      5. discrete sample from the renormalized distribution.

    Implemented in numpy on host; cost is negligible vs the decode step.
    Same RNG seed yields deterministic sampling — caller passes a numpy Generator.
    `recent_ids` is the list of generated token ids so far (for rep-penalty).
    `full_history_ids` is prompt_ids + generated_ids (for n-gram / DRY). If
    `full_history_ids` is None, falls back to `recent_ids`.
    """
    import numpy as np

    # Step 1: repetition penalty
    if repetition_penalty != 1.0 and recent_ids:
        # Operate on a float64 copy so we don't mutate the caller's array.
        logits = logits_np.astype(np.float64).copy()
        unique_ids = np.fromiter(set(recent_ids), dtype=np.int64)
        if unique_ids.size > 0:
            scores = logits[unique_ids]
            # CTRL: penalize whether score>0 or <0 so |score| moves toward 0.
            penalized = np.where(
                scores < 0, scores * repetition_penalty, scores / repetition_penalty
            )
            logits[unique_ids] = penalized
    else:
        logits = logits_np.astype(np.float64)

    # Step 1b: no_repeat_ngram_size — bans tokens that would complete a
    # previously-seen n-gram. Uses the full history (prompt + generated) so
    # whitespace/newline loops in the paged drift mode actually get blocked.
    history = full_history_ids if full_history_ids is not None else recent_ids
    if no_repeat_ngram_size and no_repeat_ngram_size >= 2 and history and \
            len(history) >= no_repeat_ngram_size - 1:
        n = no_repeat_ngram_size
        # Build set of (n-1)-gram -> next-token tokens seen in history.
        # banned = { t : exists i s.t. history[i:i+n-1] == history[-(n-1):]
        #            and history[i+n-1] == t }
        if n == 1:
            # n=1 means "don't repeat any token ever" — degenerate but defined.
            banned = set(history)
        else:
            suffix = tuple(history[-(n - 1):])
            banned = set()
            # Vectorize with numpy: find all windows matching suffix.
            h = np.asarray(history, dtype=np.int64)
            if h.size >= n:
                # For small n, a slide-window comparison is fine: O(len(h)*n).
                # len(h) <= max_pos (≤4096 typical) so totally negligible.
                for i in range(h.size - (n - 1)):
                    match = True
                    for j in range(n - 1):
                        if h[i + j] != suffix[j]:
                            match = False
                            break
                    if match and i + (n - 1) < h.size:
                        banned.add(int(h[i + (n - 1)]))
        if banned:
            ban_arr = np.fromiter(banned, dtype=np.int64)
            logits[ban_arr] = -np.inf

    # Step 1c: DRY sampler — exponential penalty for any token that would
    # extend the longest suffix-of-history match beyond `allowed_length`.
    # Implementation follows llama.cpp PR #6839 (l3utterfly 2024):
    #   For each candidate token t, find the longest L such that
    #     history[-L:] + [t] appears as a substring of history (excluding the
    #     final suffix that overlaps the present). If L >= allowed_length,
    #     subtract multiplier * base^(L - allowed_length) from logits[t].
    if dry_multiplier > 0.0 and history and len(history) >= dry_allowed_length:
        L_max = len(history)
        # We compute the longest suffix-match-length ending at each historical
        # position i, against the current suffix history[-?:].
        # Standard trick: Z-function on (suffix + sep + history) — but for our
        # context lengths (≤ a few thousand) a plain pass suffices.
        h = list(history)
        suffix_len = min(L_max, 50)  # cap match window at 50 — past that the
                                      # base^L term blows past any finite logit
                                      # and we get -inf anyway. Keeps cost O(n).
        # For each position i where the suffix matches ending exactly at i:
        # find the maximum k s.t. h[i-k+1 : i+1] == h[L_max-k : L_max].
        # Then the token h[i+1] (if it exists) would extend the match to k+1.
        # We bucket: for token t = h[i+1], track max k.
        token_match_len = {}
        # Walk i from L_max-2 down to 0 (skip i = L_max-1, which IS the suffix).
        # We only care if h[i+1] exists (i.e. i < L_max-1).
        # Build longest matching prefix at each position by character-wise
        # extension. O(L_max * suffix_len) ≤ 50 * 4096 = 200k ops; trivial.
        for i in range(L_max - 1):
            # Extend match: how many chars of history[L_max-k : L_max] match
            # history[i-k+1 : i+1]?
            k = 0
            while k < suffix_len and (i - k) >= 0 and \
                    h[i - k] == h[L_max - 1 - k]:
                k += 1
            if k >= dry_allowed_length:
                # h[i+1] is the token that would extend the match to k+1.
                t = h[i + 1]
                # We score the *extension*: penalty exp uses (k+1 - allowed),
                # i.e. how far past the allowed length picking t would push.
                # llama.cpp uses (match_len - allowed_length) where match_len
                # is the length INCLUDING the would-be next token, so k+1.
                # If picking t would create a (k+1)-length repeat:
                ext = (k + 1) - dry_allowed_length  # >= 1 since k >= allowed
                prev = token_match_len.get(t, 0)
                if ext > prev:
                    token_match_len[t] = ext
        if token_match_len:
            for t, ext in token_match_len.items():
                # Subtract from logit (operate in log-space; equiv to
                # multiplying prob by exp(-penalty)).
                penalty = dry_multiplier * (dry_base ** ext)
                logits[t] -= penalty

    # Step 2: temperature
    if temperature <= 0.0:
        # Greedy argmax on (possibly rep-penalized) logits — still deterministic,
        # and the only path when no sampling sliders are touched.
        if min_p > 0.0 or (top_p < 1.0):
            # User asked for truncation but temperature=0 — apply truncation in
            # log-space by setting truncated logits to -inf, then argmax.
            # We need probs to compute min_p / top_p, so fall through to softmax
            # with temperature=1 internally; argmax of any monotone transform
            # of logits is the same anyway.
            scaled = logits - logits.max()
        else:
            return int(np.argmax(logits))
    else:
        scaled = logits / float(temperature)
        scaled = scaled - scaled.max()

    probs = np.exp(scaled)
    probs /= probs.sum()

    # Step 3: min-p
    if min_p > 0.0:
        threshold = min_p * probs.max()
        probs = np.where(probs >= threshold, probs, 0.0)
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            # Pathological: even the top token fell below threshold (can't
            # happen with min_p<=1, but guard anyway). Fall back to argmax.
            return int(np.argmax(logits_np))

    # Step 4: top-p (nucleus)
    if top_p < 1.0:
        order = np.argsort(-probs)
        sorted_p = probs[order]
        cum = np.cumsum(sorted_p)
        cutoff = int(np.searchsorted(cum, top_p)) + 1
        keep_idx = order[:cutoff]
        mask = np.zeros_like(probs)
        mask[keep_idx] = probs[keep_idx]
        probs = mask / mask.sum()

    # Step 5: sample (greedy if temperature was 0; sample otherwise)
    if temperature <= 0.0:
        return int(np.argmax(probs))
    return int(rng.choice(len(probs), p=probs))


def _encode_prompt(state: "ServerState", prompt: str, chat: bool, system: str = "") -> tuple[list[int], list[int]]:
    """Encode prompt with optional chat-template wrapping.

    Qwen3.6 is an instruct/thinking model. Without the <|im_start|>user...<|im_end|>
    <|im_start|>assistant\n wrapper, the model has no anchor for "this is a turn"
    and instead continues the document autoregressively — exactly the long-context
    drift we observe. Canonical reference:
      experiments/.refs/tt-metal/models/tt_transformers/tt/common.py:303
      def encode_prompt_hf(tokenizer, prompt_text, system_prompt_text=None):
          chat = []
          if system_prompt_text:
              chat.append({"role": "system", "content": system_prompt_text})
          chat.append({"role": "user", "content": prompt_text})
          return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)

    Returns (prompt_ids, stop_token_ids). stop_token_ids = [eos_id] for raw prompts,
    [eos_id, im_end_id, endoftext_id] for chat prompts. Qwen3.6 tokenizer_config.json:
      eos_token = <|im_end|> = 248046
      <|endoftext|> = 248044
    """
    tok = state.tok
    if chat:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        # Some tokenizers (Qwen3.6) return a dict {input_ids, attention_mask};
        # older / simpler ones return a flat list. Handle both.
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(out, dict):
            prompt_ids = out["input_ids"]
        else:
            prompt_ids = out
    else:
        prompt_ids = tok.encode(prompt)
    # Build stop-token set. For instruct models the model may emit <|im_end|>,
    # <|endoftext|>, or eos_token_id; any of these should terminate the turn.
    stop_ids = set()
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        stop_ids.add(int(eos))
    for s in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(s) if hasattr(tok, "convert_tokens_to_ids") else None
        if tid is not None and tid != getattr(tok, "unk_token_id", -1):
            stop_ids.add(int(tid))
    return list(prompt_ids), sorted(stop_ids)


def handle_generate(state: "ServerState", args: dict):
    """Sample N tokens from a prompt — STREAMS by default.

    Yields chunks of {token_text, token_id, tok_idx} every chunk_size tokens,
    then a final {_final: True, ...} summary.

    args:
      prompt:      str (required)
      max_tokens:  int (default: 40)
      chunk_size:  int (default: 1) — tokens per streaming chunk
      temperature: float (default: 0.0) — 0 means greedy argmax. >0 enables
                   sampling (e.g. 0.7). Greedy can collapse into repetition
                   on base models at long contexts; sampling avoids this.
      top_p:       float (default: 1.0) — nucleus cutoff when temperature>0.
      min_p:       float (default: 0.0) — min-p cutoff (arXiv:2407.01082).
                   Recommended 0.05-0.1. Composes with top_p; defaults to off.
      repetition_penalty: float (default: 1.0) — CTRL penalty on recently
                   generated ids (arXiv:1909.05858). 1.0 = no-op; 1.1-1.3
                   recommended to break repetition loops at long contexts.
                   Applies BEFORE temperature scaling. Active even at
                   temperature=0 (still deterministic, just rebiased argmax).
      seed:        int   (default: 0)   — RNG seed for reproducible sampling.
    """
    if state.mock:
        yield {"_final": True, "mock": True, "generated_text": "(mock)"}
        return

    import numpy as np
    import torch
    import ttnn

    prompt = args.get("prompt")
    if not prompt:
        yield {"_final": True, "error": "missing required arg: prompt"}
        return
    max_tokens = int(args.get("max_tokens", 40))
    chunk_size = max(1, int(args.get("chunk_size", 1)))
    temperature = float(args.get("temperature", 0.0))
    top_p = float(args.get("top_p", 1.0))
    min_p = float(args.get("min_p", 0.0))
    repetition_penalty = float(args.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(args.get("no_repeat_ngram_size", 0))
    dry_multiplier = float(args.get("dry_multiplier", 0.0))
    dry_base = float(args.get("dry_base", 1.75))
    dry_allowed_length = int(args.get("dry_allowed_length", 2))
    seed = int(args.get("seed", 0))
    chat = bool(args.get("chat", False))
    system = str(args.get("system", "") or "")
    rng = np.random.default_rng(seed)

    if not hasattr(state, "tok") or state.tok is None:
        yield {"_final": True, "error": "tokenizer not loaded on server"}
        return

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    NUM_LAYERS = state.num_layers
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    upload = state._91f.upload
    device = state.device

    # Tokenize (chat=True wraps prompt in Qwen3 <|im_start|>...<|im_end|> template)
    prompt_ids, stop_ids = _encode_prompt(state, prompt, chat=chat, system=system)
    if len(prompt_ids) + max_tokens > MAX_POS:
        yield {"_final": True,
               "error": f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} > MAX_POS {MAX_POS}; "
                        f"use generate_long for prompts/completions over 256 tokens"}
        return

    # Fresh state buffers (mirrors bench_decode)
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]
    kvc = []
    kv_init = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT)
        kvc.append([kv_k, kv_v])

    embed_np = state.embed_np

    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32), device=device)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = state.layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm[dn_idx], cvs[dn_idx] = state._91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt, kv_k, kv_v = state._91f.gated_attn_step_ondevice(
                    x_tt, w_tt, kv_k, kv_v, None, cur_pos_tt, cur_pos,
                    cos_tt, sin_tt, cfg, device)
                kvc[attn_idx] = [kv_k, kv_v]
                attn_idx += 1
            x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
        logits_tt = ttnn.linear(x_tt, state.lm_head_tt, compute_kernel_config=state._91f.hifi4)
        return logits_tt

    # Prefill — run forward on each prompt token. Final logits = last prompt token's output.
    t0 = time.time()
    last_logits = None
    for pos, tid in enumerate(prompt_ids):
        last_logits = forward_token(tid, pos)
    ttnn.synchronize_device(device)
    prefill_ms = (time.time() - t0) * 1000.0

    # Decode loop — argmax sampling, max_tokens steps
    generated_ids = []
    decode_times = []
    cur_pos = len(prompt_ids)
    stop_id_set = set(stop_ids)

    text_so_far = ""
    pending_chunk = []  # accumulates token dicts for batched yield
    stopped_on_eos = False

    def _flush_chunk():
        nonlocal pending_chunk
        if pending_chunk:
            yield_text = "".join(item["token_text"] for item in pending_chunk)
            yield_payload = {
                "token_text": yield_text,
                "token_ids": [item["token_id"] for item in pending_chunk],
                "tok_idx_start": pending_chunk[0]["tok_idx"],
                "tok_idx_end": pending_chunk[-1]["tok_idx"],
            }
            pending_chunk = []
            return yield_payload
        return None

    for step in range(max_tokens):
        logits_np = ttnn.to_torch(last_logits).float().cpu().numpy().flatten()
        next_id = _sample_next_id(
            logits_np, temperature, top_p, rng,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            recent_ids=generated_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            full_history_ids=list(prompt_ids) + generated_ids,
        )
        generated_ids.append(next_id)

        # Delta decode (handles multi-byte tokens)
        new_text = state.tok.decode(generated_ids, skip_special_tokens=True)
        delta = new_text[len(text_so_far):]
        text_so_far = new_text

        pending_chunk.append({
            "token_id": next_id, "token_text": delta, "tok_idx": step,
        })
        if len(pending_chunk) >= chunk_size:
            chunk = _flush_chunk()
            if chunk:
                yield chunk

        if next_id in stop_id_set:
            stopped_on_eos = True
            break
        td0 = time.time()
        last_logits = forward_token(next_id, cur_pos)
        ttnn.synchronize_device(device)
        decode_times.append((time.time() - td0) * 1000.0)
        cur_pos += 1

    # Flush any remaining pending tokens
    chunk = _flush_chunk()
    if chunk:
        yield chunk

    total_ms = (time.time() - t0) * 1000.0
    n_gen = len(generated_ids)
    ms_per_tok = (sum(decode_times) / len(decode_times)) if decode_times else float("nan")

    yield {
        "_final": True,
        "prompt": prompt,
        "generated_text": text_so_far,
        "full_text": prompt + text_so_far,
        "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids,
        "n_prompt_tokens": len(prompt_ids),
        "n_generated_tokens": n_gen,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "ms_per_tok": ms_per_tok,
        "tok_per_sec": 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0,
        "stopped_on_eos": stopped_on_eos,
    }



def handle_generate_paged(state: "ServerState", args: dict):
    """Sample from a prompt using paged KV cache — STREAMS by default.

    Same UX as handle_generate but uses gated_attn_step_ondevice_paged
    + paged_scaled_dot_product_attention_decode → unlocks prompts +
    completions up to max_pos tokens. Wire / RPC name: `generate_long`.

    KNOWN ISSUE — long-context quality on the paged forward (2026-05-13):
      Greedy decoding (temperature=0) diverges from the non-paged forward
      at step ~132 for the "JSON parser combinator in Rust" prompt; after
      that the model drifts into whitespace/newline repetition. Reproducer
      and analysis: experiments/serve/scripts/compare_paged_vs_nonpaged.py.
      The underlying kernels (paged_update_cache + paged SDPA decode) were
      verified bit-correct in isolation (cos≥0.99993 vs numpy oracle at
      all block boundaries) — the issue is accumulated bf16 noise pushing
      greedy argmax over a cliff. Cache-size doesn't help — divergence is
      identical at max_pos ∈ {256, 512, 1024}.
      Temperature sampling extends coherent output to ~150 tokens but
      doesn't fully fix it. Real fix needs fp32 KV cache or paged-kernel
      tuning — both deferred to a future session.

    args:
      prompt:      str (required)
      max_tokens:  int (default: 40)
      max_pos:     int (default: 1024) — KV cache size
      block_size:  int (default: 64)   — paged block size
      chunk_size:  int (default: 1)    — tokens per streaming chunk
      temperature: float (default: 0.0) — 0 means greedy. >0 enables
                   sampling. Sampling extends coherent output from
                   ~130 tok (greedy) to ~150 tok at temperature=0.7 +
                   top_p=0.9, but doesn't fully eliminate long-context drift.
      top_p:       float (default: 1.0) — nucleus cutoff when temperature>0.
      min_p:       float (default: 0.0) — min-p cutoff (arXiv:2407.01082).
                   Recommended 0.05-0.1. Composes with top_p.
      repetition_penalty: float (default: 1.0) — CTRL penalty on recently
                   generated ids (arXiv:1909.05858). 1.0 = no-op; try 1.1-1.3
                   to break the whitespace/newline loop the paged path falls
                   into past step ~130. Applies before temperature scaling
                   and works even at temperature=0 (deterministic rebiased
                   argmax).
      seed:        int   (default: 0)   — RNG seed for reproducible sampling.
    """
    if state.mock:
        yield {"_final": True, "mock": True, "generated_text": "(mock)"}
        return

    import numpy as np
    import torch
    import ttnn

    prompt = args.get("prompt")
    if not prompt:
        yield {"_final": True, "error": "missing required arg: prompt"}
        return
    max_tokens = int(args.get("max_tokens", 40))
    max_pos = int(args.get("max_pos", 1024))
    block_size = int(args.get("block_size", 64))
    chunk_size = max(1, int(args.get("chunk_size", 1)))
    # Default greedy; user can opt into sampling for slightly longer coherent output.
    temperature = float(args.get("temperature", 0.0))
    top_p = float(args.get("top_p", 1.0))
    min_p = float(args.get("min_p", 0.0))
    repetition_penalty = float(args.get("repetition_penalty", 1.0))
    no_repeat_ngram_size = int(args.get("no_repeat_ngram_size", 0))
    dry_multiplier = float(args.get("dry_multiplier", 0.0))
    dry_base = float(args.get("dry_base", 1.75))
    dry_allowed_length = int(args.get("dry_allowed_length", 2))
    seed = int(args.get("seed", 0))
    chat = bool(args.get("chat", False))
    system = str(args.get("system", "") or "")
    rng = np.random.default_rng(seed)
    assert max_pos % block_size == 0, f"max_pos {max_pos} must be multiple of block_size {block_size}"
    ensure_rope_tables(state, max_pos)

    if not hasattr(state, "tok") or state.tok is None:
        yield {"_final": True, "error": "tokenizer not loaded on server"}
        return

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    N_KV = cfg["n_kv_heads"]
    HEAD_DIM = cfg["head_dim"]
    ROTARY_DIM = int(HEAD_DIM * cfg["partial_rotary_factor"])
    NUM_LAYERS = state.num_layers
    KEY_DIM = cfg["n_k_heads"] * cfg["k_dim"]
    VAL_DIM = cfg["n_v_heads"] * cfg["v_dim"]
    CONV_DIM = 2 * KEY_DIM + VAL_DIM
    upload = state._91f.upload
    device = state.device

    prompt_ids, stop_ids = _encode_prompt(state, prompt, chat=chat, system=system)
    if len(prompt_ids) + max_tokens > max_pos:
        yield {"_final": True,
               "error": f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} > max_pos {max_pos}; "
                        f"increase --max-pos"}
        return

    max_num_blocks = max_pos // block_size
    page_table_np = np.arange(max_num_blocks, dtype=np.int32).reshape(1, max_num_blocks)
    page_table_tt = ttnn.from_torch(torch.from_numpy(page_table_np),
                                      dtype=ttnn.int32, device=device,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)

    # Fresh state
    n_dn = sum(1 for i in range(NUM_LAYERS) if i % 4 != 3)
    n_attn = NUM_LAYERS - n_dn
    ssm = [upload(np.zeros((cfg["n_v_heads"], cfg["k_dim"], cfg["v_dim"]),
                            dtype=np.float32), device, dtype=ttnn.float32)
           for _ in range(n_dn)]
    cvs = [upload(np.zeros((CONV_DIM, cfg["conv_kernel"] - 1), dtype=np.float32),
                    device, dtype=ttnn.float32) for _ in range(n_dn)]
    paged_kv_zero = np.zeros((max_num_blocks, N_KV, block_size, HEAD_DIM), dtype=np.float32)
    kvc = []
    for _ in range(n_attn):
        kv_k = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kv_v = ttnn.from_torch(torch.from_numpy(paged_kv_zero), dtype=ttnn.bfloat16,
                                device=device, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kvc.append([kv_k, kv_v])

    embed_np = state.embed_np

    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(state.cos_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        sin_tt = ttnn.slice(state.sin_ext_table_tt, [cur_pos, 0], [cur_pos + 1, ROTARY_DIM])
        cur_pos_tt = ttnn.from_torch(torch.tensor([cur_pos], dtype=torch.int32),
                                       device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        dn_idx = 0
        attn_idx = 0
        for i in range(NUM_LAYERS):
            layer_type, w_tt = state.layer_weights[i]
            if layer_type == "linear_attention":
                x_tt, ssm[dn_idx], cvs[dn_idx] = state._91f.deltanet_step_ondevice(
                    x_tt, w_tt, ssm[dn_idx], cvs[dn_idx], cfg)
                dn_idx += 1
            else:
                kv_k, kv_v = kvc[attn_idx]
                x_tt = state._91f.gated_attn_step_ondevice_paged(
                    x_tt, w_tt, kv_k, kv_v, page_table_tt, cur_pos_tt,
                    cos_tt, sin_tt, cfg)
                attn_idx += 1
            x_tt = state._91f.mlp_step_ondevice(x_tt, w_tt)
        x_tt = ttnn.rms_norm(x_tt, weight=state.final_norm_tt, epsilon=1e-6)
        logits_tt = ttnn.linear(x_tt, state.lm_head_tt, compute_kernel_config=state._91f.hifi4)
        return logits_tt

    # Prefill
    t0 = time.time()
    last_logits = None
    for pos, tid in enumerate(prompt_ids):
        last_logits = forward_token(tid, pos)
    ttnn.synchronize_device(device)
    prefill_ms = (time.time() - t0) * 1000.0

    # Decode loop
    generated_ids = []
    decode_times = []
    cur_pos = len(prompt_ids)
    stop_id_set = set(stop_ids)

    text_so_far = ""
    pending = []  # accumulates pre-chunk tokens
    stopped_on_eos = False

    for step in range(max_tokens):
        logits_np = ttnn.to_torch(last_logits).float().cpu().numpy().flatten()
        next_id = _sample_next_id(
            logits_np, temperature, top_p, rng,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            recent_ids=generated_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            full_history_ids=list(prompt_ids) + generated_ids,
        )
        generated_ids.append(next_id)
        new_text = state.tok.decode(generated_ids, skip_special_tokens=True)
        delta = new_text[len(text_so_far):]
        text_so_far = new_text
        pending.append({"token_id": next_id, "token_text": delta, "tok_idx": step})

        if len(pending) >= chunk_size:
            yield {
                "token_text": "".join(p["token_text"] for p in pending),
                "token_ids": [p["token_id"] for p in pending],
                "tok_idx_start": pending[0]["tok_idx"],
                "tok_idx_end": pending[-1]["tok_idx"],
            }
            pending = []

        if next_id in stop_id_set:
            stopped_on_eos = True
            break
        td0 = time.time()
        last_logits = forward_token(next_id, cur_pos)
        ttnn.synchronize_device(device)
        decode_times.append((time.time() - td0) * 1000.0)
        cur_pos += 1

    if pending:
        yield {
            "token_text": "".join(p["token_text"] for p in pending),
            "token_ids": [p["token_id"] for p in pending],
            "tok_idx_start": pending[0]["tok_idx"],
            "tok_idx_end": pending[-1]["tok_idx"],
        }

    total_ms = (time.time() - t0) * 1000.0
    n_gen = len(generated_ids)
    ms_per_tok = (sum(decode_times) / len(decode_times)) if decode_times else float("nan")

    yield {
        "_final": True,
        "prompt": prompt,
        "generated_text": text_so_far,
        "full_text": prompt + text_so_far,
        "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids,
        "n_prompt_tokens": len(prompt_ids),
        "n_generated_tokens": n_gen,
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "ms_per_tok": ms_per_tok,
        "tok_per_sec": 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0,
        "stopped_on_eos": stopped_on_eos,
        "max_pos": max_pos,
        "block_size": block_size,
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "dry_multiplier": dry_multiplier,
        "dry_base": dry_base,
        "dry_allowed_length": dry_allowed_length,
    }


HANDLERS = {
    "status":               handle_status,
    "reload_kernels":       handle_reload_kernels,
    "reset_state":          handle_reset_state,
    "run_91r":              handle_run_91r,
    "bench_decode":         handle_bench_decode,
    "bench_decode_paged":   handle_bench_decode_paged,
    "bench_decode_traced":  handle_bench_decode_traced,
    "generate":             handle_generate,         # short context (≤256), streams
    "generate_long":        handle_generate_paged,    # long context (paged), streams
    "shutdown":             handle_shutdown,
}


def _cleanup_socket(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _write_pidfile() -> None:
    P.ensure_cache_dir()
    with open(P.PID_PATH, "w") as f:
        f.write(str(os.getpid()) + "\n")


def _close_device(state: ServerState) -> None:
    if state.device is not None and not state.mock:
        try:
            import ttnn
            ttnn.close_device(state.device)
        except Exception as e:
            print(f"[shutdown] close_device error: {e}")


def serve(state: ServerState) -> None:
    P.ensure_cache_dir()
    _cleanup_socket(P.SOCKET_PATH)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(P.SOCKET_PATH)
    os.chmod(P.SOCKET_PATH, 0o600)
    sock.listen(4)
    print(f"[serve] listening on {P.SOCKET_PATH}")
    try:
        while True:
            conn, _ = sock.accept()
            try:
                raw = P.read_line(conn)
                if not raw:
                    conn.close()
                    continue
                try:
                    req = P.parse_request(raw)
                except Exception as e:
                    conn.sendall(P.pack_error(f"bad request: {e}"))
                    conn.close()
                    continue
                print(f"[serve] cmd={req.cmd} args={req.args}")
                handler = HANDLERS.get(req.cmd)
                if handler is None:
                    conn.sendall(P.pack_error(f"unknown cmd: {req.cmd}"))
                    conn.close()
                    continue
                try:
                    import types as _types
                    data = handler(state, req.args)
                    if isinstance(data, _types.GeneratorType):
                        # Streaming handler: iterate, send chunks until final.
                        # Convention: yield {"_final": True, ...} as the last item.
                        for item in data:
                            if isinstance(item, dict) and item.pop("_final", False):
                                conn.sendall(P.pack_result(item))
                            else:
                                conn.sendall(P.pack_chunk(item))
                    else:
                        conn.sendall(P.pack_result(data))
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[serve] handler error:\n{tb}")
                    conn.sendall(P.pack_error(f"{type(e).__name__}: {e}"))
                conn.close()
                if req.cmd == "shutdown":
                    print("[serve] shutdown requested; exiting")
                    break
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"[serve] accept-loop error: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        sock.close()
        _cleanup_socket(P.SOCKET_PATH)
        _close_device(state)
        try:
            os.unlink(P.PID_PATH)
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="skip weight load (for protocol smoke testing locally)")
    ap.add_argument("--device-id", type=int, default=0)
    args = ap.parse_args()

    _write_pidfile()
    state = ServerState()
    try:
        bootstrap(state, mock=args.mock, device_id=args.device_id)
    except Exception:
        traceback.print_exc()
        try:
            os.unlink(P.PID_PATH)
        except FileNotFoundError:
            pass
        sys.exit(1)
    serve(state)


if __name__ == "__main__":
    main()
