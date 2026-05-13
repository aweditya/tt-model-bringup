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
        self.tok = None
        self._91f = None                        # kernel module (re-importable)
        self._91l = None
        self.boot_time = time.time()
        self.last_run: Optional[dict] = None
        self.loaded = False

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

    # Pre-compute RoPE table once (demo_qwen36_27b.py:160-166)
    rotary_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(MAX_POS).astype(np.float32)
    all_angles = positions[:, None] * freqs[None, :]
    cos_all = np.concatenate([np.cos(all_angles), np.cos(all_angles)], axis=-1).astype(np.float32)
    sin_all = np.concatenate([np.sin(all_angles), np.sin(all_angles)], axis=-1).astype(np.float32)
    state.cos_table_tt = upload(cos_all, device, dtype=ttnn.float32)
    state.sin_table_tt = upload(sin_all, device, dtype=ttnn.float32)

    state.loaded = True
    print("[bootstrap] ready")


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
            kv_init = np.zeros((1, cfg["n_kv_heads"], MAX_POS, cfg["head_dim"]), dtype=np.float32)
            kv_k = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            kv_v = ttnn.from_torch(torch.from_numpy(kv_init), dtype=ttnn.bfloat16,
                                    device=device, layout=ttnn.TILE_LAYOUT)
            for pos in range(seq):
                x_tt = upload(hf_in[pos].reshape(1, HIDDEN), device, dtype=ttnn.float32)
                cos_tt = ttnn.slice(state.cos_table_tt, [pos, 0], [pos + 1, rotary_dim])
                sin_tt = ttnn.slice(state.sin_table_tt, [pos, 0], [pos + 1, rotary_dim])
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

    prompt_ids = state.tokenizer.encode(prompt)
    embed_np = state.embed_np

    def forward_token(token_id, cur_pos):
        x_np = embed_np[token_id]
        x_tt = upload(x_np.reshape(1, HIDDEN), device, dtype=ttnn.float32)
        cos_tt = ttnn.slice(state.cos_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
        sin_tt = ttnn.slice(state.sin_table_tt, [cur_pos, 0], [cur_pos + 1, rotary_dim])
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


def handle_shutdown(state: ServerState, args: dict) -> dict:
    return {"ok": True, "shutting_down": True}


HANDLERS = {
    "status":         handle_status,
    "reload_kernels": handle_reload_kernels,
    "reset_state":    handle_reset_state,
    "run_91r":        handle_run_91r,
    "bench_decode":   handle_bench_decode,
    "shutdown":       handle_shutdown,
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
                    data = handler(state, req.args)
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
