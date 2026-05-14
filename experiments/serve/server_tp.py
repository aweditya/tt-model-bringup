#!/usr/bin/env python3
"""
Multi-chip persistent inference server for Qwen3.6-27B on qb2 (4× P150).

C'7.8 implementation. Mirrors `experiments/serve/server.py` (single-chip) but:
  - Opens a (1, 4) mesh device + sets FABRIC_1D
  - Loads each layer's weights AS SHARDED tensors (per-chip slabs)
  - Forward uses validated TP probe machinery (deltanet_tp + attn_tp + mlp_tp)
  - Trace capture wraps the full 64-block forward (C'7.6.1 proved this works)
  - handle_generate runs execute_trace per token (vs Python eager loop)

Status of build (2026-05-13):
  Stage A: skeleton — open mesh, load sharded weights, status endpoint  ← THIS COMMIT
  Stage B: forward (per-layer TP cycle, eager-mode end-to-end correctness)
  Stage C: trace capture (build the persistent traced forward graph)
  Stage D: handle_generate_tp (tokenize → write inputs → execute_trace → argmax → decode)
  Stage E: bench_decode_tp for honest perf measurement

Reuses the validated probes:
  - experiments/utils/full_layer_tp_probe.py — relayout_in_proj/_conv, deltanet_tp, mlp_tp
  - experiments/utils/tp_attn_traced_probe.py — attn_tp_forward + relayout_attn_qkv/_o
  - experiments/91f_qwen36_27b_full_ondevice.py — load_layer_weights_all (real weights)

Protocol shared with single-chip server: experiments/serve/protocol.py
"""
import os
import sys
import time
import socket
import json
import signal
import importlib.util

# Stage A: device init only. Bigger imports gated to bootstrap to keep cold startup fast.

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = os.path.expanduser("~/tt-xla")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
SOCKET_PATH = os.path.join(CACHE_DIR, "server_tp.sock")
PID_FILE = os.path.join(CACHE_DIR, "server_tp.pid")
LOG_FILE = os.path.join(CACHE_DIR, "server_tp.log")

# Reuse single-chip protocol
sys.path.insert(0, PROJECT_ROOT)
from experiments.serve import protocol as P  # noqa: E402

# Model constants — sourced from config.json at bootstrap, mirrors 91f
MODEL_ID = "Qwen/Qwen3.6-27B"
MAX_POS = 256


# --- Mesh server state --------------------------------------------------------
class MeshServerState:
    """Resident state for the multi-chip server.

    Carries: mesh device, cfg, sharded layer weights, state buffers (SSM, conv,
    KV — all per-layer, all sharded), tokenizer, embed/lm_head, traced graph IDs.
    """
    def __init__(self):
        self.mesh = None
        self.cfg = None
        self.num_layers = 0
        self.tok = None
        self.embed_np = None
        self.lm_head_tt = None
        self.final_norm_tt = None
        self.cos_ext_table_tt = None
        self.sin_ext_table_tt = None
        # Per-layer sharded weights: list of {'type': 'linear_attention'|'full_attention',
        # 'w_dn': sharded DN weights (if dn), 'w_attn': sharded attn weights (if attn),
        # 'w_mlp': sharded MLP weights, 'state': sharded SSM/conv/KV buffers}
        self.layers = []
        # Persistent traced graph (Stage C)
        self.trace_id = None
        self.trace_x_buf = None
        self.trace_logits_buf = None
        self.last_run = None


# --- Bootstrap ----------------------------------------------------------------
def bootstrap(state: MeshServerState):
    """Stage A: open mesh + set fabric + load sharded weights + tokenizer."""
    print(f"[bootstrap] importing ttnn + torch + numpy…", flush=True)
    import numpy as np
    import torch
    import ttnn
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    print(f"[bootstrap] setting fabric_config = FABRIC_1D…", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    print(f"[bootstrap] opening (1, 4) mesh device…", flush=True)
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    print(f"  ✓ mesh {state.mesh.get_num_devices()} chips", flush=True)

    # Load HF config
    cfg_path = hf_hub_download(MODEL_ID, "config.json")
    with open(cfg_path) as f:
        text_cfg = json.load(f)['text_config']
    cfg = {
        'hidden':      text_cfg['hidden_size'],
        'n_k_heads':   text_cfg['linear_num_key_heads'],
        'n_v_heads':   text_cfg['linear_num_value_heads'],
        'k_dim':       text_cfg['linear_key_head_dim'],
        'v_dim':       text_cfg['linear_value_head_dim'],
        'conv_kernel': text_cfg['linear_conv_kernel_dim'],
        'n_q_heads':   text_cfg['num_attention_heads'],
        'n_kv_heads':  text_cfg['num_key_value_heads'],
        'head_dim':    text_cfg['head_dim'],
        'partial_rotary_factor': text_cfg['partial_rotary_factor'],
        'intermediate': text_cfg['intermediate_size'],
    }
    state.cfg = cfg
    state.num_layers = text_cfg['num_hidden_layers']
    print(f"  ✓ cfg: {cfg}", flush=True)
    print(f"  ✓ num_layers: {state.num_layers}", flush=True)

    print(f"[bootstrap] loading tokenizer…", flush=True)
    state.tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  ✓ tokenizer", flush=True)

    # TODO Stage B: load + shard layer weights
    # for i in range(state.num_layers):
    #     layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
    #     w_np = load_layer_weights_all(i, layer_type)
    #     state.layers.append(shard_and_upload(state.mesh, w_np, layer_type, cfg))

    # TODO Stage B: load + upload embed, lm_head, final_norm (replicated)
    # TODO Stage B: precompute cos/sin tables, upload (replicated)
    # TODO Stage C: trace capture
    # TODO Stage D: traced forward graph + persistent buffers

    print(f"[bootstrap] STAGE A COMPLETE — mesh + cfg + tokenizer ready.", flush=True)
    print(f"[bootstrap] TODO: weight loading (Stage B), trace (Stage C), generate (Stage D)", flush=True)


# --- Handlers -----------------------------------------------------------------
def handle_status(state: MeshServerState, args: dict) -> dict:
    return {
        "ok": True,
        "mesh_open": state.mesh is not None,
        "num_devices": state.mesh.get_num_devices() if state.mesh else 0,
        "num_layers_planned": state.num_layers,
        "num_layers_loaded": len(state.layers),
        "stage": "A_skeleton",
        "last_run": state.last_run,
    }


def handle_shutdown(state: MeshServerState, args: dict) -> dict:
    return {"ok": True, "shutting_down": True}


def handle_generate_tp(state: MeshServerState, args: dict) -> dict:
    """TODO Stage D — placeholder."""
    return {
        "error": "generate_tp not yet implemented",
        "stage_status": "A_skeleton (only status + shutdown work)",
        "next_stage": "B_weight_loading",
    }


HANDLERS = {
    "status":         handle_status,
    "generate_tp":    handle_generate_tp,
    "shutdown":       handle_shutdown,
}


# --- Socket main loop ---------------------------------------------------------
def _cleanup_socket(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def serve(state: MeshServerState):
    _cleanup_socket(SOCKET_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCKET_PATH)
    srv.listen(4)
    os.chmod(SOCKET_PATH, 0o600)
    print(f"[serve] listening on {SOCKET_PATH}", flush=True)

    shutdown_requested = False
    while not shutdown_requested:
        try:
            conn, _ = srv.accept()
        except OSError:
            continue
        try:
            req = P.read_msg(conn)
            cmd = req.get("cmd", "")
            args = req.get("args", {}) or {}
            handler = HANDLERS.get(cmd)
            if handler is None:
                resp = {"error": f"unknown cmd: {cmd}"}
            else:
                try:
                    resp = handler(state, args)
                except Exception as e:
                    import traceback
                    resp = {"error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[:2000]}
            if cmd == "shutdown":
                shutdown_requested = True
            P.write_msg(conn, resp)
        finally:
            conn.close()
    srv.close()
    _cleanup_socket(SOCKET_PATH)
    print("[serve] shutdown complete", flush=True)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    state = MeshServerState()
    try:
        bootstrap(state)
        print("[bootstrap] ready", flush=True)
        serve(state)
    finally:
        if state.mesh is not None:
            try:
                import ttnn
                ttnn.close_mesh_device(state.mesh)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                print("[shutdown] mesh closed, fabric disabled", flush=True)
            except Exception as e:
                print(f"[shutdown] cleanup error: {e}", flush=True)
        try:
            os.unlink(PID_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
