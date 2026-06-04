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
  - experiments/serve/ondevice_27b.py — load_layer_weights_all (real weights)

Protocol shared with single-chip server: experiments/serve/protocol.py
"""
import os
import sys
import time
import socket
import json
import contextlib

# Stage A: device init only. Bigger imports gated to bootstrap to keep cold startup fast.

# --- Paths --------------------------------------------------------------------
# Resolve repo root from this file's location so a fresh clone at any path
# works without hardcoding. Prod (~/tt-xla/) is unchanged. Override with
# TT_XLA_ROOT for tests that want a custom cache location.
PROJECT_ROOT = os.environ.get("TT_XLA_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
SOCKET_PATH = os.path.join(CACHE_DIR, "server_tp.sock")
PID_FILE = os.path.join(CACHE_DIR, "server_tp.pid")
LOG_FILE = os.path.join(CACHE_DIR, "server_tp.log")

# Reuse single-chip protocol
sys.path.insert(0, PROJECT_ROOT)
from experiments.serve import protocol as P  # noqa: E402

# Model constants — sourced from config.json at bootstrap, mirrors 91f
MODEL_ID = "Qwen/Qwen3.6-27B"
MAX_POS = 8192  # 2026-05-21: bumped from 2048 → 8192 to validate L=4000/8000.
                # KV cost is small thanks to GQA: n_kv_heads=4 × head_dim=256 ×
                # 2(K+V) × bf16 × 64 layers ÷ 4 chips = 64 KB/token/chip.
                # MAX_POS=8192 → 512 MB/chip extra over MAX_POS=2048, trivial
                # on P150 HBM. NUM_BLOCKS auto-scales to 256.
                # History: 256 → 512 (qb1 needle) → 2048 → 8192.
# Paged KV cache parameters — required for trace-compatible decode
# (paged_update_cache supports update_idxs_tensor=, non-paged doesn't).
# BLOCK_SIZE must be a multiple of TILE_HEIGHT=32. NUM_BLOCKS * BLOCK_SIZE = MAX_POS.
BLOCK_SIZE = 32
NUM_BLOCKS = MAX_POS // BLOCK_SIZE  # 8 at MAX_POS=256
TILE_HEIGHT = 32  # for height-sharded input padding


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
        self.embed_tt = None             # P25: on-device embed table for ttnn.embedding lookup
        self.cos_table_tt = None         # P25: full [MAX_POS, ROTARY_DIM] cos table on-device
        self.sin_table_tt = None         # P25: full [MAX_POS, ROTARY_DIM] sin table on-device
        self.lm_head_tt = None
        self.final_norm_tt = None
        self.cos_ext_table_tt = None
        self.sin_ext_table_tt = None
        # Per-layer sharded weights: list of {'type': 'linear_attention'|'full_attention',
        # 'w_dn': sharded DN weights (if dn), 'w_attn': sharded attn weights (if attn),
        # 'w_mlp': sharded MLP weights, 'state': sharded SSM/conv/KV buffers}
        self.layers = []
        # Persistent traced graph + on-device argmax output handle
        self.trace_id = None
        self.traced_argmax_tt = None
        # Vocab dims: real vocab from state.embed_np.shape[0] (152064 for
        # Qwen3.6); padded vocab in lm_head weight (248320).
        self.vocab_size = None
        self.vocab_padded = None
        # Per-step HtoD write targets (allocated in bootstrap, written via
        # ttnn.copy_host_to_device_tensor before each execute_trace).
        self.cur_pos_buf = None
        self.tok_buf = None
        self.rot_idxs_buf = None
        self.cos_all_np = None
        self.sin_all_np = None
        self.rotary_dim = None
        # Paged KV cache + SDPA shared state
        self.page_table_tt = None
        self.paged_write_mem_cfg = None
        self.fused_paged_write_mem_cfg_k = None
        self.fused_paged_write_mem_cfg_v = None
        self.paged_sdpa_progcfg = None
        self.sdpa_compute_kernel_config = None
        self.last_run = None
        # Experiment guard. Production default stays on the validated two-call
        # paged_update_cache path; probe endpoints may flip this temporarily.
        self.use_fused_paged_update = False
        # 2026-05-19: defaulted to explicit_all_reduce after P1 probe
        # (probe_ccl_components_tp) showed num_links=2 is 11.2% faster than
        # num_links=1 at production [1, 5120] bf16 shape; the bare
        # `ttnn.all_reduce(partial)` path uses unknown defaults.
        self.collective_mode = "explicit_all_reduce"
        self.rope_mode = "manual"
        self.deltanet_decay_mode = "manual"
        # 2026-05-18: defaulted to "owned_gdn" after Tier 3 long-context gate
        # passed at 500 positions (commit 040e2ac, research/owned_gdn_
        # diagnosis_2026_05_18.md). Probe endpoints still toggle this
        # explicitly. Set to "manual" to revert to the legacy TTNN
        # broadcast-reduce recurrence.
        self.deltanet_recurrence_mode = "owned_gdn"
        # G4 owned decay/gate fused kernel (qwen36_decay_gate_decode_owned).
        # Manual fallback path lives behind state.deltanet_decay_gate_mode=="manual".
        self.deltanet_decay_gate_mode = "owned_decay_gate"
        self.profile_records = None
        self.profile_context_stack = []


# --- Bootstrap ----------------------------------------------------------------
@contextlib.contextmanager
def _profile_scope(state, name: str):
    if getattr(state, "profile_records", None) is None:
        yield
        return
    state.profile_context_stack.append(name)
    try:
        yield
    finally:
        state.profile_context_stack.pop()


def bootstrap(state: MeshServerState, log=None):
    """Stage A: open mesh + set fabric + load sharded weights + tokenizer."""
    if log is None:
        def log(msg):
            print(msg, flush=True)
    log("[bootstrap] importing ttnn + torch + numpy…")
    import numpy as np
    import torch
    import ttnn
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    # Owned-op availability check (fail-soft) — if the user's mode wants an
    # owned kernel but `scripts/build_owned_ops.sh` was never run, fall back to
    # the manual path with a loud warning instead of raising mid-token deep in
    # the forward (which the user reads as "garbage output").
    _owned_op = {
        "owned_gdn":         "qwen36_gdn_decode_owned",
        "owned_gdn_inplace": "qwen36_gdn_decode_owned",
        "owned_decay_gate":  "qwen36_decay_gate_decode_owned",
    }
    _exp = getattr(ttnn, "experimental", object())
    for _attr in ("deltanet_recurrence_mode", "deltanet_decay_gate_mode"):
        _mode = getattr(state, _attr)
        if _mode in _owned_op and not hasattr(_exp, _owned_op[_mode]):
            print(f"\n[bootstrap] WARNING: ttnn.experimental.{_owned_op[_mode]} "
                  f"not found — state.{_attr} flipped {_mode!r} → 'manual'.\n"
                  f"            Run `bash scripts/build_owned_ops.sh` and rebuild "
                  f"ttnn to get the owned kernel.\n", flush=True)
            setattr(state, _attr, "manual")

    print("[bootstrap] setting fabric_config = FABRIC_1D…", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    print("[bootstrap] opening (1, 4) mesh device…", flush=True)
    # trace_region_size: chunked-prefill trace at L=32 ≈ 0.45 GB + decode ~0.2 GB
    # = ~0.65 GB. Set 800 MB. Tighter reserves leave room for the model — T5e
    # at 1.5 GB still OOM'd on 635 MB/bank lm_head load with 581 MB/bank free.
    state.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4),
                                        trace_region_size=800_000_000)
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
    _cap = os.environ.get('TP_MAX_LAYERS')
    if _cap:
        state.num_layers = min(state.num_layers, int(_cap))
        print(f"  TP_MAX_LAYERS={_cap} → capping num_layers to {state.num_layers}", flush=True)
    print(f"  ✓ cfg: {cfg}", flush=True)
    print(f"  ✓ num_layers: {state.num_layers}", flush=True)

    print("[bootstrap] loading tokenizer…", flush=True)
    state.tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print("  ✓ tokenizer", flush=True)

    # === Stage B: load + shard all layer weights ===
    print("[bootstrap] importing on-device 27B kernels + TP relayout helpers…", flush=True)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "utils"))
    from experiments.serve import ondevice_27b as _91f   # was the 91f importlib hack
    state._91f = _91f
    from full_layer_tp_probe import (
        relayout_in_proj, relayout_conv,
        N_V_HEADS, K_DIM, V_DIM, KERNEL, CONV_DIM,
    )
    from tp_attn_traced_probe import (
        relayout_attn_qkv, relayout_o,
        NQ_PER_CHIP, NKV_PER_CHIP,
    )

    def upload_replicated(arr, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))

    def upload_sharded(arr, dim, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.from_numpy(arr), dtype=dtype,
                                device=state.mesh, layout=ttnn.TILE_LAYOUT,
                                mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=dim))

    print(f"[bootstrap] loading + sharding {state.num_layers} layers (this is the slow part)…", flush=True)
    t_load_start = time.time()
    for i in range(state.num_layers):
        layer_type = 'linear_attention' if i % 4 != 3 else 'full_attention'
        w_np = _91f.load_layer_weights_all(i, layer_type)
        # MLP weights (shared between DN and attn layers)
        w_gate_tt = upload_sharded(w_np['gate_proj'], dim=1)
        w_up_tt = upload_sharded(w_np['up_proj'], dim=1)
        w_down_tt = upload_sharded(w_np['down_proj'], dim=0)
        post_norm_tt = upload_replicated(w_np['post_attention_layernorm'])
        input_norm_tt = upload_replicated(w_np['input_layernorm'])

        if layer_type == 'linear_attention':
            w_in_sh = relayout_in_proj(w_np['in_proj_all'])
            conv_w_np = w_np['conv1d_weight']
            if conv_w_np.ndim == 3:
                conv_w_np = conv_w_np.squeeze(1)
            w_conv_sh = relayout_conv(conv_w_np)
            conv_state_sh = relayout_conv(
                np.zeros((CONV_DIM, cfg['conv_kernel'] - 1), dtype=np.float32))

            w_in_tt = upload_sharded(w_in_sh, dim=1)
            w_conv_tt = upload_sharded(w_conv_sh, dim=0)
            conv_state_tt = upload_sharded(conv_state_sh, dim=0)
            # G4 conv1d pre-split: 3 single-column conv_st tensors + 4 single-column
            # w_conv tensors per layer. Used by the owned conv1d path to eliminate
            # the per-step slice/concat that caused the G3 wire-in bug (commit
            # df1cccc / e168c4d). Single-column source numpy arrays are created
            # at construction so the bf16 tile lands with data in column 0 (the
            # only position the owned kernel reads). The kernel mutates these
            # tensors in place via its writer — no concat-and-copy roundtrip.
            # Memory cost: 7 × [CONV_DIM/4, 1] bf16 × 48 DeltaNet layers ≈ 1.7 MB
            # per chip — negligible vs the 21 GB residency footprint.
            KERNEL = cfg['conv_kernel']  # 4
            conv_state_split_tt = [
                upload_sharded(
                    relayout_conv(np.zeros((CONV_DIM, 1), dtype=np.float32)),
                    dim=0)
                for _ in range(KERNEL - 1)]
            w_conv_split_tt = [
                upload_sharded(
                    relayout_conv(conv_w_np[:, k:k+1].astype(np.float32)),
                    dim=0)
                for k in range(KERNEL)]
            dt_bias_tt = upload_sharded(w_np['dt_bias'], dim=0)
            A_log_tt = upload_sharded(w_np['A_log'], dim=0)
            w_out_tt = upload_sharded(w_np['out_proj'], dim=0)
            ssm_tt = upload_sharded(
                np.zeros((1, N_V_HEADS, K_DIM, V_DIM), dtype=np.float32), dim=1)
            linear_attn_norm_tt = upload_replicated(w_np['linear_attn_norm'])
            # Q/K L2 scale constants for QK rms_norm fusion (mirrors 91f)
            K_DIM_LOCAL = 128
            q_l2_scale = np.full(K_DIM_LOCAL, 1.0 / K_DIM_LOCAL, dtype=np.float32)
            k_l2_scale = np.full(K_DIM_LOCAL, 1.0 / np.sqrt(K_DIM_LOCAL), dtype=np.float32)
            q_l2_tt = upload_replicated(q_l2_scale)
            k_l2_tt = upload_replicated(k_l2_scale)

            layer = {
                'type': 'linear_attention',
                'dn': {
                    'w_in': w_in_tt, 'w_conv': w_conv_tt, 'conv_st': conv_state_tt,
                    'conv_st_split': conv_state_split_tt,
                    'w_conv_split': w_conv_split_tt,
                    'dt_bias': dt_bias_tt, 'A_log': A_log_tt, 'w_out': w_out_tt,
                    'ssm': ssm_tt,
                    'input_norm': input_norm_tt,
                    'linear_attn_norm': linear_attn_norm_tt,
                    'q_l2_scale': q_l2_tt, 'k_l2_scale': k_l2_tt,
                },
                'mlp': {
                    'w_gate': w_gate_tt, 'w_up': w_up_tt, 'w_down': w_down_tt,
                    'post_norm': post_norm_tt,
                },
            }
        else:
            # full_attention layer
            w_qkv_sh = relayout_attn_qkv(w_np['attn_qkv'], NQ_PER_CHIP, NKV_PER_CHIP)
            w_o_sh = relayout_o(w_np['o_proj'])
            w_qkv_tt = upload_sharded(w_qkv_sh, dim=1)
            w_o_tt = upload_sharded(w_o_sh, dim=0)
            # PAGED KV cache: [NUM_BLOCKS, N_KV=4, BLOCK_SIZE, HEAD_DIM] sharded
            # along N_KV → per-chip [NUM_BLOCKS, 1, BLOCK_SIZE, HEAD_DIM].
            # Required for trace-compatible decode: paged_update_cache supports
            # update_idxs_tensor= (the non-paged update_cache_for_token_ in our
            # ttnn build takes only Python int — bakes into trace).
            # Validated on mesh via P12.1 (commit 74441a3).
            # SDPA is manual (paged SDPA also fails on mesh per P13/feedback_p1).
            kv_init_paged = np.zeros(
                (NUM_BLOCKS, cfg['n_kv_heads'], BLOCK_SIZE, cfg['head_dim']),
                dtype=np.float32)
            kv_k_tt = upload_sharded(kv_init_paged, dim=1)
            kv_v_tt = upload_sharded(kv_init_paged, dim=1)
            q_norm_tt = upload_replicated(w_np['q_norm'])
            k_norm_tt = upload_replicated(w_np['k_norm'])
            layer = {
                'type': 'full_attention',
                'attn': {
                    'w_qkv': w_qkv_tt, 'w_o': w_o_tt,
                    'kc': kv_k_tt, 'vc': kv_v_tt,
                    'q_norm': q_norm_tt, 'k_norm': k_norm_tt,
                    'input_norm': input_norm_tt,
                },
                'mlp': {
                    'w_gate': w_gate_tt, 'w_up': w_up_tt, 'w_down': w_down_tt,
                    'post_norm': post_norm_tt,
                },
            }
        state.layers.append(layer)
        if (i + 1) % 8 == 0 or i == 0:
            print(f"  layer {i + 1}/{state.num_layers} loaded ({time.time() - t_load_start:.0f}s elapsed)",
                  flush=True)

    print(f"[bootstrap] all {state.num_layers} layers loaded in {time.time() - t_load_start:.0f}s", flush=True)

    # === Stage B (cont): embed, lm_head, final_norm — replicated ===
    print("[bootstrap] loading embed + lm_head + final_norm + RoPE tables…", flush=True)
    # Reuse generate_27b's embed/lm_head loader (shared with the single-chip server).
    from experiments.serve import generate_27b as _91l
    embed_weights = _91l.load_embed_lm_head_weights()
    state.embed_np = embed_weights['embed']
    state.final_norm_tt = upload_replicated(embed_weights['final_norm'])
    # P25 (validated 2026-05-14, .cache/p25_on_device_embed/results.json):
    # Upload the full embed table on-device so per-step token embedding lookup
    # is `ttnn.embedding(tok_buf, embed_tt)` instead of host slice + HtoD copy.
    # Replicated across mesh. ROW_MAJOR layout required by ttnn.embedding (the
    # output is then tilized via layout=TILE_LAYOUT arg at lookup time).
    # Shape: [VOCAB_PADDED=248320, HIDDEN=5120] bf16 ≈ 2.54 GB per chip (fits).
    state.embed_tt = ttnn.from_torch(
        torch.from_numpy(state.embed_np),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    print(f"  ✓ embed_tt uploaded on-device [{state.embed_np.shape[0]}, {state.embed_np.shape[1]}] "
          f"bf16 replicated (~{state.embed_np.shape[0]*state.embed_np.shape[1]*2/1e9:.2f} GB/chip)",
          flush=True)
    # P22 (validated 2026-05-14, .cache/p22_vocab_sharded_lm_head/results.json):
    # shard lm_head along vocab axis. Weight is pre-transposed [HIDDEN, VOCAB_PADDED]
    # (generate_27b.load_embed_lm_head_weights) so dim=1 is the vocab dim. Per-chip slab
    # is [HIDDEN=5120, VOCAB_PADDED/NCHIPS=62080], which is tile-aligned.
    # Forward emits the argmax on device (saves ~35 ms/tok readback).
    NCHIPS = state.mesh.get_num_devices()
    state.vocab_padded = int(embed_weights['lm_head'].shape[1])  # 248320 for Qwen3.6
    # NOTE: prior to P22, prod argmaxed over `[: state.embed_np.shape[0]]` which equals
    # vocab_padded = 248320 (i.e. a no-op slice). The HF config vocab_size IS 248320 (the
    # tile-aligned padded layout); tokenizer.vocab_size is 248044. Keep behavior parity by
    # using the full padded vocab for argmax — model padding rows in lm_head are zero-or-tiny
    # by HF design, so argmax over them never wins in practice.
    state.vocab_size = state.vocab_padded
    assert state.vocab_padded % NCHIPS == 0, \
        f"vocab_padded {state.vocab_padded} not divisible by nchips {NCHIPS}"
    state.lm_head_tt = upload_sharded(embed_weights['lm_head'], dim=1)
    print(f"  ✓ embed uploaded; final_norm replicated; "
          f"lm_head sharded dim=1 (per-chip {state.vocab_padded // NCHIPS}; "
          f"vocab {state.vocab_size}/{state.vocab_padded})", flush=True)

    # RoPE cos/sin tables — ROTARY_DIM-wide (rotate-only path).
    HEAD_DIM = cfg['head_dim']
    rotary_dim = int(HEAD_DIM * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(MAX_POS).astype(np.float32)
    ang = positions[:, None] * freqs[None, :]
    state.cos_all_np = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
    state.sin_all_np = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
    state.rotary_dim = rotary_dim
    # Device-resident extended table for the eager (non-traced) path — sliced
    # at runtime by Python int.
    pad = HEAD_DIM - rotary_dim
    cos_ext = np.concatenate([state.cos_all_np, np.ones((MAX_POS, pad), dtype=np.float32)], axis=-1)
    sin_ext = np.concatenate([state.sin_all_np, np.zeros((MAX_POS, pad), dtype=np.float32)], axis=-1)
    state.cos_ext_table_tt = upload_replicated(cos_ext)
    state.sin_ext_table_tt = upload_replicated(sin_ext)
    # P25: ROW_MAJOR cos/sin tables for on-device per-step lookup via
    # ttnn.embedding(rot_idxs_buf, cos_table_tt, layout=TILE_LAYOUT).
    # Replaces host-side cos_all_np[cur_pos] slice + HtoD copy each step.
    state.cos_table_tt = ttnn.from_torch(
        torch.from_numpy(state.cos_all_np),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    state.sin_table_tt = ttnn.from_torch(
        torch.from_numpy(state.sin_all_np),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    print(f"  ✓ RoPE tables uploaded (MAX_POS={MAX_POS}; "
          f"+ on-device ROW_MAJOR cos/sin tables for ttnn.embedding lookup)",
          flush=True)

    # Paged KV cache page_table: identity mapping for B=1 (logical block i →
    # physical block i). Replicated across mesh. Fixed (doesn't change per step).
    page_table_np = np.arange(NUM_BLOCKS, dtype=np.int32).reshape(1, NUM_BLOCKS)
    state.page_table_tt = ttnn.from_torch(
        torch.from_numpy(page_table_np),
        device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    print(f"  ✓ page_table uploaded (NUM_BLOCKS={NUM_BLOCKS}, BLOCK_SIZE={BLOCK_SIZE})",
          flush=True)

    # Cache the L1-sharded memory_config for paged_update_cache inputs
    compute_grid = state.mesh.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(1, compute_grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [TILE_HEIGHT, cfg['head_dim']],
                                   ttnn.ShardOrientation.ROW_MAJOR)
    state.paged_write_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    print("  ✓ paged_write mem_cfg cached", flush=True)
    # paged_fused_update_cache requires K/V input tensors to occupy disjoint
    # L1 core ranges. Keep this separate from the production single-writer
    # config, which intentionally uses one core and remains the default path.
    if compute_grid.x >= 8 and compute_grid.y >= 8:
        k_cores = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))
        })
        v_cores = ttnn.CoreRangeSet({
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(7, 7))
        })
        shard_spec_k = ttnn.ShardSpec(k_cores, [TILE_HEIGHT, cfg['head_dim']],
                                      ttnn.ShardOrientation.ROW_MAJOR)
        shard_spec_v = ttnn.ShardSpec(v_cores, [TILE_HEIGHT, cfg['head_dim']],
                                      ttnn.ShardOrientation.ROW_MAJOR)
        state.fused_paged_write_mem_cfg_k = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec_k)
        state.fused_paged_write_mem_cfg_v = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec_v)
        print("  ✓ fused paged_write disjoint K/V mem_cfg cached", flush=True)
    else:
        print(f"  ! fused paged_write disjoint mem_cfg unavailable for grid "
              f"{compute_grid.x}x{compute_grid.y}", flush=True)

    # Paged SDPA program + compute kernel config (P18 winner, see
    # feedback_mesh_paged_sdpa_works.md + feedback_p19_chained_paged_sdpa.md).
    # Required because the default kernel grabs ~110 cores/head, triggering a
    # tree-reduction error on (1,4) mesh. CoreCoord(4,4) fits the per-chip slab.
    state.paged_sdpa_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4),
        q_chunk_size=0,
        k_chunk_size=0,
        exp_approx_mode=False,
    )
    state.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    print("  ✓ paged SDPA program_config + compute_kernel_config cached", flush=True)

    # Pre-allocated input buffers for trace-compatible decode.
    # These are READ by forward_token_tp_inner (which is the trace target).
    # They are UPDATED before each execute_trace via copy_host_to_device_tensor
    # — outside the captured region, so the writes don't violate trace semantics.
    state.cur_pos_buf = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    # Tiny index buffers — the ONLY per-step HtoD writes. Shape [1,1] (not [1])
    # because ttnn.embedding requires idx ndim >= 2.
    state.tok_buf = ttnn.from_torch(
        torch.tensor([[0]], dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.rot_idxs_buf = ttnn.from_torch(
        torch.tensor([[0]], dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    print("  ✓ input buffers pre-allocated (cur_pos_buf, tok_buf, rot_idxs_buf)",
          flush=True)

    # Prefill trace input buffers: tok IDs + positions for a fixed chunk_size.
    # forward_prefill_chunked_traced reads from these; host updates before replay.
    # chunk_size=32 keeps prefill trace memory ~0.45 GB so both traces + model
    # weights fit. Trade: traced replay only helps L<=32 (single chunk); T3
    # multi-chunk would cover longer prompts but is deferred.
    PREFILL_CHUNK_SIZE = 32
    state.prefill_chunk_size = PREFILL_CHUNK_SIZE
    state.prefill_tok_buf = ttnn.from_torch(
        torch.zeros((1, PREFILL_CHUNK_SIZE), dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.prefill_pos_buf = ttnn.from_torch(
        torch.zeros((1, PREFILL_CHUNK_SIZE), dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    print(f"  ✓ prefill trace input buffers pre-allocated (chunk_size={PREFILL_CHUNK_SIZE})",
          flush=True)

    # Pre-allocate Neumann DN accumulators at fixed C=32 (the inner block size
    # used by _prefill_dn_chunked_blocks). Lets _chunked_dn_with_chunked_recurrence_tp
    # work inside a captured trace (which forbids host->device allocations).
    # All positions are slice-written per iter → safe to reuse across calls.
    from full_layer_tp_probe import (
        NV_PER_CHIP, K_DIM, V_DIM, VAL_DIM_CHIP, NCHIPS,
    )
    PREFILL_INNER_C = 32
    state.prefill_inner_c = PREFILL_INNER_C
    total_NV = NCHIPS * NV_PER_CHIP
    state.dn_chunked_q = ttnn.from_torch(
        torch.zeros((total_NV, PREFILL_INNER_C, K_DIM), dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
    state.dn_chunked_k = ttnn.from_torch(
        torch.zeros((total_NV, PREFILL_INNER_C, K_DIM), dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
    state.dn_chunked_v = ttnn.from_torch(
        torch.zeros((total_NV, PREFILL_INNER_C, V_DIM), dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
    state.dn_chunked_z = ttnn.from_torch(
        torch.zeros((PREFILL_INNER_C, NCHIPS * VAL_DIM_CHIP), dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
        device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1))
    print(f"  ✓ DN Neumann buffers pre-allocated (C={PREFILL_INNER_C})", flush=True)

    # Pre-compute the constant masks used by _chunked_recurrence_tp +
    # _neumann_inverse_via_mesh_tp. They depend only on C and NV_PER_CHIP,
    # so building them once at bootstrap lets the prefill path run inside
    # a captured trace.
    import numpy as _np
    _tril_np = _np.tril(_np.ones((PREFILL_INNER_C, PREFILL_INNER_C), dtype=_np.float32))
    _strict_np = _tril_np - _np.eye(PREFILL_INNER_C, dtype=_np.float32)
    _tril_per_chip = _np.broadcast_to(_tril_np, (NV_PER_CHIP, PREFILL_INNER_C, PREFILL_INNER_C)).copy()
    _strict_per_chip = _np.broadcast_to(_strict_np, (NV_PER_CHIP, PREFILL_INNER_C, PREFILL_INNER_C)).copy()
    state.dn_tril_mask = ttnn.from_torch(
        torch.from_numpy(_tril_per_chip), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.dn_strict_lower_mask = ttnn.from_torch(
        torch.from_numpy(_strict_per_chip), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    _I_per_chip = _np.zeros((NV_PER_CHIP, PREFILL_INNER_C, PREFILL_INNER_C), dtype=_np.float32)
    for i in range(NV_PER_CHIP):
        _np.fill_diagonal(_I_per_chip[i], 1.0)
    state.dn_I_tt = ttnn.from_torch(
        torch.from_numpy(_I_per_chip), dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT, device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    print("  ✓ DN tril/strict_lower/I masks pre-allocated", flush=True)

    print("[bootstrap] STAGE B COMPLETE — all weights + state buffers on mesh.", flush=True)


# ============================================================================
# Stage C: TP forward functions (eager mode first; trace in Stage D)
# ============================================================================
# These mirror 91f's deltanet_step_ondevice / gated_attn_step_ondevice /
# mlp_step_ondevice but operate on mesh-sharded weights and add ttnn.all_reduce
# at the out_proj exits (single-chip 91f doesn't need that step).
#
# Validated correctness via the per-layer-type TP probes (C'7.2/C'7.3/C'7.4).
# Reuses validated layout helpers from full_layer_tp_probe + tp_attn_traced_probe.


def _rms_norm_manual(x_tt, weight_tt, eps, dim_size):
    """Mesh RMS norm — thin wrapper around ttnn.rms_norm."""
    import ttnn
    return ttnn.rms_norm(x_tt, weight=weight_tt, epsilon=eps)


def _tp_all_reduce(state: MeshServerState, partial):
    """All-reduce a row-parallel partial across the (1, 4) mesh.

    `explicit_all_reduce` mode uses num_links=2 + Topology.Ring on the qb1/qb2
    P150x4 mesh — 11.2% faster than num_links=1 at [1, 5120] bf16."""
    import ttnn
    if state.collective_mode == "explicit_all_reduce":
        return ttnn.all_reduce(
            partial,
            cluster_axis=1,
            memory_config=partial.memory_config(),
            num_links=2,
            topology=ttnn.Topology.Ring,
        )
    try:
        return ttnn.all_reduce(partial)
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        return ttnn.all_gather(scattered, dim=1)


def deltanet_step_tp(state, x_tt, dn, cfg):
    """One DeltaNet TP step on the mesh. Returns the residual-added output.

    `dn` = per-layer sharded weights dict (see Stage B): w_in, w_conv, conv_st,
    dt_bias, A_log, w_out, ssm, input_norm, linear_attn_norm, q_l2_scale, k_l2_scale.

    REFACTOR (2026-05-20, task #77 prep): inner body moved to
    `_deltanet_step_tp_from_inproj` so the v4 chunked stub can call the
    inner body with pre-computed batched in_proj output. Behaviorally
    identical to the pre-refactor function.
    """
    import ttnn
    from full_layer_tp_probe import EPS

    HIDDEN = cfg['hidden']
    # 1. Pre-norm (manual: see _rms_norm_manual doc)
    h_tt = _rms_norm_manual(x_tt, dn['input_norm'], EPS, HIDDEN)
    # 2. in_proj (replicated x × sharded weight → per-chip slab)
    all_tt = ttnn.linear(h_tt, dn['w_in'])
    ttnn.deallocate(h_tt)
    return _deltanet_step_tp_from_inproj(state, x_tt, all_tt, dn, cfg)


def _deltanet_step_tp_from_inproj(state, x_residual_tt, all_tt, dn, cfg,
                                   precomputed_decay=None, precomputed_beta=None):
    """Inner DN body, starting AFTER rms_norm + in_proj.

    Used by both `deltanet_step_tp` (which runs rms+linear inline) and the
    v4 chunked stub `deltanet_chunked_neumann_tp` (which runs rms+linear
    batched once per chunk, then loops per-position calling this helper
    with sliced `all_tt`).

    Args:
      x_residual_tt:  [1, HIDDEN] — the original input, used for the final
                      residual add at the end. NOT the rms_norm'd or in_proj'd
                      version; this is what gets added to `reduced`.
      all_tt:         [1, IN_PROJ_OUT_CHIP] — the output of `linear(h_tt, w_in)`,
                      i.e., already past stages 1+2 of the DN forward.
      precomputed_decay:  optional [1, NV_PER_CHIP, 1, 1] — if provided, skip
                          the a-slice + decay-gate computation; use this directly.
                          Pair with precomputed_beta. (v4 Stage 3 path.)
      precomputed_beta:   optional [1, NV_PER_CHIP] — if provided, skip the
                          b-slice + decay-gate computation. Pair with decay.
    """
    import ttnn
    from full_layer_tp_probe import (
        K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )

    _have_pre_decay = (precomputed_decay is not None and precomputed_beta is not None)

    # 3. slice per-chip [Q | K | V | Z | A | B]
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    if not _have_pre_decay:
        a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                          [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
        b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                          [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    ttnn.deallocate(all_tt)
    # 4. conv1d on per-chip slab (manual: 3-tap mul+sum recurrence).
    with _profile_scope(state, "deltanet_conv"):
        mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
        ttnn.deallocate(mixed_qkv)
        conv_input = ttnn.concat([dn['conv_st'], mixed_col], dim=-1)
        ttnn.deallocate(mixed_col)
        conv_prod = ttnn.mul(conv_input, dn['w_conv'])
        conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
        ttnn.deallocate(conv_prod)
        conv_state_new = ttnn.slice(conv_input, [0, 1], [CONV_DIM_CHIP, cfg['conv_kernel']])
        ttnn.deallocate(conv_input)
    # 5. Q/K/V per-chip head-sliced
    q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
    k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
    v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])
    ttnn.deallocate(conv_out)

    def gqa(t, n_kh, d):
        t2 = ttnn.reshape(t, [n_kh, 1, d])
        t3 = ttnn.repeat(t2, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t3, [n_kh * N_REP, d])

    def gqa4(t, n_kh, d):
        t2 = ttnn.reshape(t, [1, n_kh, 1, d])
        t3 = ttnn.repeat(t2, ttnn.Shape([1, 1, N_REP, 1]))
        return ttnn.reshape(t3, [1, n_kh * N_REP, 1, d])

    with _profile_scope(state, "deltanet_qkv_repeat"):
        if state.deltanet_recurrence_mode in ("owned_gdn", "owned_gdn_inplace"):
            q = gqa4(q_flat, NK_PER_CHIP, K_DIM)
            k = gqa4(k_flat, NK_PER_CHIP, K_DIM)
            v = ttnn.reshape(v_flat, [1, NV_PER_CHIP, 1, V_DIM])
        else:
            q = gqa(q_flat, NK_PER_CHIP, K_DIM)
            k = gqa(k_flat, NK_PER_CHIP, K_DIM)
            v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])
    # 6. QK l2-norm (manual mesh-safe form — q_l2_scale=1/K_DIM, k_l2_scale=1/sqrt(K_DIM)
    # baked into the weights; eps=EPS/K_DIM matches the rms_norm semantics exactly).
    EPS_RMS = EPS / K_DIM
    q = _rms_norm_manual(q, dn['q_l2_scale'], EPS_RMS, K_DIM)
    k = _rms_norm_manual(k, dn['k_l2_scale'], EPS_RMS, K_DIM)
    # 7. gate/decay/beta on per-chip head subset
    with _profile_scope(state, "deltanet_decay_gate"):
        if _have_pre_decay:
            # v4 Stage 3: chunked function pre-computed decay + beta batched.
            # Skip the a/b slice + decay-gate computation; use what was passed.
            decay = precomputed_decay  # [1, NV_PER_CHIP, 1, 1]
            beta = precomputed_beta    # [1, NV_PER_CHIP]
        elif state.deltanet_decay_gate_mode == "owned_decay_gate":
            # Owned decay/gate kernel (G2 wire-in). Kernel expects rank-2
            # [1, NV_PER_CHIP] for all 4 inputs; production dn['dt_bias'] +
            # dn['A_log'] are rank-1 [NV_PER_CHIP] per chip (uploaded as
            # dim=0 sharded [NV] numpy). Reshape per-step here; G4 default
            # flip will pre-allocate rank-2 versions at bootstrap.
            # reshape returns a VIEW of dn['dt_bias']/dn['A_log']; do NOT
            # deallocate the reshaped handles or we free the underlying
            # weight tensor and the next decode step crashes with
            # "Tensor is not allocated".
            dt_bias_r2 = ttnn.reshape(dn['dt_bias'], [1, NV_PER_CHIP])
            A_log_r2 = ttnn.reshape(dn['A_log'], [1, NV_PER_CHIP])
            decay_compact, beta = ttnn.experimental.qwen36_decay_gate_decode_owned(
                a_tt, b_tt, dt_bias_r2, A_log_r2)
            # Downstream (manual + owned_gdn paths) expects decay shaped
            # [1, NV_PER_CHIP, 1, 1] for broadcast against H_4d.
            decay = ttnn.reshape(decay_compact, [1, NV_PER_CHIP, 1, 1])
            ttnn.deallocate(decay_compact)
        else:
            a_biased = ttnn.add(a_tt, dn['dt_bias'])
            if state.deltanet_decay_mode == "native_softplus":
                softplus_a = ttnn.softplus(a_biased)
            else:
                softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0))
            g = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
            beta = ttnn.sigmoid(b_tt)
            decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
    # 8. Recurrence
    with _profile_scope(state, "deltanet_recurrence"):
        H_4d = dn['ssm']
        if state.deltanet_recurrence_mode in ("owned_gdn", "owned_gdn_inplace"):
            beta4 = ttnn.reshape(beta, [1, NV_PER_CHIP, 1, 1])
            # Default owned_gdn keeps the old copy/commit discipline.
            # owned_gdn_inplace uses the op's state-input writer directly.
            H_owned_in = (
                H_4d
                if state.deltanet_recurrence_mode == "owned_gdn_inplace"
                else ttnn.add(H_4d, 0.0)
            )
            H_new, out = ttnn.experimental.qwen36_gdn_decode_owned(
                H_owned_in,
                q,
                k,
                v,
                decay,
                beta4,
                native_io=True,
                output_memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(beta4)
        else:
            H_decayed = ttnn.mul(H_4d, decay)
            k_col = ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])
            kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                                  [1, NV_PER_CHIP, V_DIM])
            v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
            delta = ttnn.mul(ttnn.sub(v_3d, kv_mem), ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
            H_new = ttnn.add(H_decayed,
                             ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])))
            q_col = ttnn.reshape(q, [1, NV_PER_CHIP, K_DIM, 1])
            out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2), [1, VAL_DIM_CHIP])
    # 9. Per-head rms_norm + silu(z) gate
    with _profile_scope(state, "deltanet_output_gate"):
        out_per_head = ttnn.reshape(out, [NV_PER_CHIP, V_DIM])
        out_normed = _rms_norm_manual(out_per_head, dn['linear_attn_norm'], EPS, V_DIM)
        z_per_head = ttnn.reshape(z_tt, [NV_PER_CHIP, V_DIM])
        silu_z = ttnn.silu(z_per_head)
        out_gated = ttnn.reshape(ttnn.mul(out_normed, silu_z), [1, VAL_DIM_CHIP])
    # 10. out_proj row-parallel + all_reduce
    partial = ttnn.linear(out_gated, dn['w_out'])
    ttnn.deallocate(out_gated)
    reduced = _tp_all_reduce(state, partial)
    ttnn.deallocate(partial)
    # 11. residual add + update SSM/conv state in place
    x_out = ttnn.add(x_residual_tt, reduced)
    ttnn.deallocate(reduced)
    with _profile_scope(state, "deltanet_state_update"):
        if state.deltanet_recurrence_mode == "owned_gdn_inplace":
            # qwen36_gdn_decode_owned writes the next state into its state input
            # buffer, which aliases dn['ssm'] through H_4d.
            pass
        else:
            ttnn.copy(H_new, dn['ssm'])
            ttnn.deallocate(H_new)
        # G4: owned conv1d mutates dn['conv_st_split'] tensors in place via
        # its writer kernel — no concat-and-copy back. conv_state_new is None.
        if conv_state_new is not None:
            ttnn.copy(conv_state_new, dn['conv_st'])
            ttnn.deallocate(conv_state_new)
    return x_out


def _chunked_recurrence_tp(state, q_seq, k_seq, v_seq, g_seq, beta_seq, S_prev, C):
    """v4 Stage 4b: chunked Neumann recurrence on mesh.

    Mirrors the validated numpy reference impl in
    experiments/utils/chunked_recurrence_numpy_probe.py:chunked_recurrence.

    Each chip processes its NV_PER_CHIP heads INDEPENDENTLY (no CCL).
    All ops are per-chip batched matmuls / elementwise.

    Inputs (per-chip shapes, mesh tensors):
      q_seq, k_seq: [NV_PER_CHIP, C, K_DIM] bf16
      v_seq:        [NV_PER_CHIP, C, V_DIM]
      g_seq:        [NV_PER_CHIP, C]  (decay values g_t, NOT exp(g_t))
      beta_seq:     [NV_PER_CHIP, C]
      S_prev:       [NV_PER_CHIP, K_DIM, V_DIM]
      C: chunk size (power of 2, default 8)

    Returns:
      O_seq:  [NV_PER_CHIP, C, V_DIM]
      S_new:  [NV_PER_CHIP, K_DIM, V_DIM]
    """
    import ttnn
    import torch
    import numpy as np
    from full_layer_tp_probe import NV_PER_CHIP

    # --- Build the lower-triangular & strict-lower-tri masks (precomputed) ---
    # tril (inclusive of diag): used in D, A
    # strict_lower (diag zeroed): used to mask attn = -(K_β @ K^T) * D
    tril_np = np.tril(np.ones((C, C), dtype=np.float32))
    strict_lower_np = tril_np - np.eye(C, dtype=np.float32)
    # Broadcast-friendly per-head shapes: [NV_PER_CHIP, C, C]
    tril_per_chip = np.broadcast_to(tril_np, (NV_PER_CHIP, C, C)).copy()
    strict_lower_per_chip = np.broadcast_to(strict_lower_np, (NV_PER_CHIP, C, C)).copy()
    # Trace-safe: reuse bootstrap-pre-allocated masks when C matches.
    if hasattr(state, 'dn_tril_mask') and C == state.prefill_inner_c:
        tril_mask = state.dn_tril_mask
        strict_lower_mask = state.dn_strict_lower_mask
    else:
        tril_mask = ttnn.from_torch(
            torch.from_numpy(tril_per_chip),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        strict_lower_mask = ttnn.from_torch(
            torch.from_numpy(strict_lower_per_chip),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    # --- G = cumsum(g) along last dim ---
    G = ttnn.cumsum(g_seq, dim=-1)  # [NV_PER_CHIP, C]

    # --- D = exp(G_a - G_b) * tril_mask ---  shape [NV_PER_CHIP, C, C]
    G_a = ttnn.reshape(G, [NV_PER_CHIP, C, 1])
    G_b = ttnn.reshape(G, [NV_PER_CHIP, 1, C])
    diff = ttnn.sub(G_a, G_b)            # broadcast → [NV_PER_CHIP, C, C]
    D_raw = ttnn.exp(diff)
    D = ttnn.mul(D_raw, tril_mask)
    ttnn.deallocate(diff)
    ttnn.deallocate(D_raw)

    # --- k_beta = beta * k,  v_beta = beta * v ---
    beta_3d = ttnn.reshape(beta_seq, [NV_PER_CHIP, C, 1])
    k_beta = ttnn.mul(k_seq, beta_3d)    # [NV_PER_CHIP, C, K_DIM]
    v_beta = ttnn.mul(v_seq, beta_3d)    # [NV_PER_CHIP, C, V_DIM]

    # --- attn = -(k_beta @ k^T) * D, diagonal zeroed via strict_lower ---
    k_T = ttnn.transpose(k_seq, -2, -1)  # [NV_PER_CHIP, K_DIM, C]
    kkT = ttnn.matmul(k_beta, k_T)       # [NV_PER_CHIP, C, C]
    attn = ttnn.mul(ttnn.neg(kkT), D)    # neg + mul D
    attn = ttnn.mul(attn, strict_lower_mask)  # zero diagonal & upper
    ttnn.deallocate(kkT)

    # --- T = (I - attn)^{-1} via Neumann factorization ---
    # PRECISION FIX (2026-05-20): cast attn to fp32 BEFORE inverse (the
    # Neumann series accumulates log2(C) matmuls; bf16 loses precision
    # rapidly for C≥64). Cast T back to bf16 for downstream matmuls.
    # Validated single-device: fp32 gives cos>0.99999 vs np.linalg.inv;
    # bf16 only ~0.99 at C=64 (degrades to cos ~0.8 in full chunked path).
    attn_fp32 = ttnn.typecast(attn, ttnn.float32)
    ttnn.deallocate(attn)
    T_fp32 = _neumann_inverse_via_mesh_tp(state, attn_fp32, C)
    ttnn.deallocate(attn_fp32)
    T = ttnn.typecast(T_fp32, ttnn.bfloat16)
    ttnn.deallocate(T_fp32)

    # 2026-05-21: production caps the chunked dispatcher at seq_len ≤ 32 (see
    # deltanet_chunked_neumann_tp). At C=32 the chained bf16 matmuls are
    # tolerable (chunked top1 within 1 of v3); at C=64/128 they accumulate
    # too much error to ship. fp32_dest_acc compute_kernel_config experiment
    # was inconclusive (test was confounded by synthetic prompts); if revisited,
    # use real-text token IDs (see scripts/v4_precision_sweep.py).

    # --- V_prime = T @ v_beta ---
    V_prime = ttnn.matmul(T, v_beta)     # [NV_PER_CHIP, C, V_DIM]

    # --- K_prime = T @ (k_beta * exp(G)) ---
    expG = ttnn.exp(G)                    # [NV_PER_CHIP, C]
    expG_3d = ttnn.reshape(expG, [NV_PER_CHIP, C, 1])
    k_beta_scaled = ttnn.mul(k_beta, expG_3d)
    K_prime = ttnn.matmul(T, k_beta_scaled)  # [NV_PER_CHIP, C, K_DIM]
    ttnn.deallocate(k_beta_scaled)
    ttnn.deallocate(k_beta)
    ttnn.deallocate(T)
    ttnn.deallocate(v_beta)

    # --- v_prime = K_prime @ S_prev ---
    v_prime = ttnn.matmul(K_prime, S_prev)  # [NV_PER_CHIP, C, V_DIM]
    ttnn.deallocate(K_prime)

    # --- v_new = V_prime - v_prime ---
    v_new = ttnn.sub(V_prime, v_prime)
    ttnn.deallocate(V_prime)
    ttnn.deallocate(v_prime)

    # --- attn_int = (q * exp(G)) @ S_prev ---
    q_scaled = ttnn.mul(q_seq, expG_3d)
    attn_int = ttnn.matmul(q_scaled, S_prev)  # [NV_PER_CHIP, C, V_DIM]
    ttnn.deallocate(q_scaled)

    # --- A = (q @ k^T) * D ---
    A_raw = ttnn.matmul(q_seq, k_T)
    A = ttnn.mul(A_raw, D)
    ttnn.deallocate(A_raw)
    ttnn.deallocate(k_T)
    ttnn.deallocate(D)

    # --- O = attn_int + A @ v_new ---
    Av_new = ttnn.matmul(A, v_new)
    O = ttnn.add(attn_int, Av_new)        # [NV_PER_CHIP, C, V_DIM]
    ttnn.deallocate(Av_new)
    ttnn.deallocate(A)
    ttnn.deallocate(attn_int)

    # --- S_new = exp(G[-1]) * S_prev + (k * exp(G[-1] - G))^T @ v_new ---
    # G_last = G[:, -1] of shape [NV_PER_CHIP] — reshape to [NV_PER_CHIP, 1] for broadcast
    G_last_2d = ttnn.slice(G, [0, C - 1], [NV_PER_CHIP, C])  # [NV_PER_CHIP, 1]
    expG_last = ttnn.exp(G_last_2d)  # [NV_PER_CHIP, 1]
    # decay_factor = exp(G_last - G) of shape [NV_PER_CHIP, C]
    decay_factor = ttnn.exp(ttnn.sub(G_last_2d, G))  # broadcast → [NV_PER_CHIP, C]
    decay_factor_3d = ttnn.reshape(decay_factor, [NV_PER_CHIP, C, 1])
    k_decayed = ttnn.mul(k_seq, decay_factor_3d)  # [NV_PER_CHIP, C, K_DIM]
    k_decayed_T = ttnn.transpose(k_decayed, -2, -1)  # [NV_PER_CHIP, K_DIM, C]
    rank1_update = ttnn.matmul(k_decayed_T, v_new)   # [NV_PER_CHIP, K_DIM, V_DIM]
    # S_prev_scaled = expG_last (scalar per head) * S_prev
    expG_last_3d = ttnn.reshape(expG_last, [NV_PER_CHIP, 1, 1])
    S_prev_scaled = ttnn.mul(S_prev, expG_last_3d)
    S_new = ttnn.add(S_prev_scaled, rank1_update)
    ttnn.deallocate(v_new)
    ttnn.deallocate(rank1_update)
    ttnn.deallocate(S_prev_scaled)
    ttnn.deallocate(k_decayed)
    ttnn.deallocate(k_decayed_T)
    ttnn.deallocate(decay_factor)
    ttnn.deallocate(decay_factor_3d)
    ttnn.deallocate(G_last_2d)
    ttnn.deallocate(expG_last)
    ttnn.deallocate(expG_last_3d)
    ttnn.deallocate(G)
    ttnn.deallocate(expG)
    ttnn.deallocate(expG_3d)
    ttnn.deallocate(G_a)
    ttnn.deallocate(G_b)
    ttnn.deallocate(beta_3d)
    if not (hasattr(state, 'dn_tril_mask') and C == state.prefill_inner_c):
        ttnn.deallocate(tril_mask)
        ttnn.deallocate(strict_lower_mask)

    return O, S_new


def _neumann_inverse_via_mesh_tp(state, L_tt, C):
    """Compute (I - L)^{-1} for strict-lower-triangular L via Neumann
    factorization on the mesh. Each chip processes its slice independently.

    L_tt: [NV_PER_CHIP, C, C] sharded on NV head dim (each chip has its heads)
    C:    must be a power of 2

    Returns: [NV_PER_CHIP, C, C] with the SAME mesh placement.

    Single-device validation: experiments/utils/neumann_inverse_probe.py
    (cos ≥ 0.99999 at fp32, [32, 64, 64] batched shape).
    """
    import ttnn
    import torch
    import numpy as np

    n_levels = int(np.log2(C))
    assert 2 ** n_levels == C, f"Neumann requires C=power of 2, got {C}"

    # Build I tensor matching L's shape, sharded the same way (NV dim).
    # Each chip will have I = eye(C) for each of its NV_PER_CHIP heads.
    # NV_PER_CHIP per-chip dim is shape[0] (since L is sharded on dim 0).
    nv_per_chip = L_tt.shape[0]
    I_per_chip = np.zeros((nv_per_chip, C, C), dtype=np.float32)
    for i in range(nv_per_chip):
        np.fill_diagonal(I_per_chip[i], 1.0)
    # Use ReplicateTensorToMesh — same I on every chip's slice (chips have
    # different heads but the I is just identity for each).
    if hasattr(state, 'dn_I_tt') and C == state.prefill_inner_c:
        I_tt = state.dn_I_tt
    else:
        I_tt = ttnn.from_torch(
            torch.from_numpy(I_per_chip),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    # Step 1: compute powers L², L⁴, L⁸, ..., L^(C/2)
    powers = [L_tt]
    cur = L_tt
    for _ in range(n_levels - 1):
        cur = ttnn.matmul(cur, cur)
        powers.append(cur)

    # Step 2: compose product (I + L)(I + L²)(I + L⁴)...
    T = ttnn.add(I_tt, powers[0])
    for p in powers[1:]:
        factor = ttnn.add(I_tt, p)
        T = ttnn.matmul(T, factor)
        ttnn.deallocate(factor)

    # Deallocate intermediates we no longer need
    for p in powers[1:]:  # powers[0] is L_tt (caller may own)
        ttnn.deallocate(p)
    if not (hasattr(state, 'dn_I_tt') and C == state.prefill_inner_c):
        ttnn.deallocate(I_tt)

    return T


def _chunked_dn_with_chunked_recurrence_tp(state, x_seq_tt, dn, cfg, seq_len):
    """v4 Stage 4b-iii: FULLY chunked DeltaNet using _chunked_recurrence_tp.

    Replaces the per-position recurrence loop with a single chunked Neumann
    call. Per-position work is reduced to:
      - PRE-recurrence: conv1d + QKV split + GQA + L2-norm  (collected into batched)
      - POST-recurrence: output gate + w_out + all_reduce + residual add (per pos)

    Requires seq_len to be a power of 2 (Neumann factorization constraint).

    All math/primitives validated standalone:
    - _chunked_recurrence_tp: probe_chunked_recurrence_tp cos 0.99996 @ C=8
    - _neumann_inverse_via_mesh_tp: probe_neumann_inverse_mesh_tp cos 0.99999967 @ C=64
    """
    import ttnn
    import torch
    from full_layer_tp_probe import (
        IN_PROJ_OUT_CHIP, EPS, CONV_DIM_CHIP, VAL_DIM_CHIP, NV_PER_CHIP,
        NK_PER_CHIP, K_DIM, V_DIM, KEY_DIM_CHIP, N_REP, NCHIPS,
    )

    HIDDEN = cfg['hidden']
    C = seq_len

    # Per-phase timing — gated on state.profile_chunked_dn. Only fires once
    # per probe (counter capped). Writes accumulated phase ms to
    # state._phase_times dict so probe can sum across layers.
    _profile = getattr(state, 'profile_chunked_dn', False)
    if _profile:
        import time as _ptime
        if not hasattr(state, '_phase_times'):
            state._phase_times = {}
        def _tic():
            ttnn.synchronize_device(state.mesh)
            return _ptime.perf_counter()
        def _toc(t0, label):
            ttnn.synchronize_device(state.mesh)
            ms = (_ptime.perf_counter() - t0) * 1000.0
            state._phase_times[label] = state._phase_times.get(label, 0.0) + ms
    else:
        _tic = lambda: 0
        _toc = lambda t0, label: None

    _t = _tic()
    # === Stage 1: batched pre-norm + in_proj ===
    h_seq = _rms_norm_manual(x_seq_tt, dn['input_norm'], EPS, HIDDEN)
    all_seq = ttnn.linear(h_seq, dn['w_in'])  # [C, IN_PROJ_OUT_CHIP]
    ttnn.deallocate(h_seq)
    _toc(_t, "1_stage_1_prenorm_inproj")

    _t = _tic()
    # === Stage 3: batched decay/gate (keep g_seq + beta_seq for chunked recurrence) ===
    a_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                       [C, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                       [C, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    a_biased = ttnn.add(a_seq, dn['dt_bias'])
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0))
    g_seq = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)  # [C, NV_PER_CHIP]
    beta_seq = ttnn.sigmoid(b_seq)                                   # [C, NV_PER_CHIP]
    ttnn.deallocate(a_seq); ttnn.deallocate(b_seq)
    ttnn.deallocate(a_biased); ttnn.deallocate(softplus_a)
    _toc(_t, "2_stage_3_decay_gate")

    _t = _tic()
    # === PRE-RECURRENCE: per-pos conv1d + QKV split + L2-norm, collect into batched ===
    total_NV = NCHIPS * NV_PER_CHIP
    # Trace-safe: reuse pre-allocated buffers when shape matches (C=PREFILL_INNER_C).
    # Per-position slice_write covers ALL positions, so previous values are
    # fully overwritten — safe to share across calls.
    if hasattr(state, 'dn_chunked_q') and C == state.prefill_inner_c:
        q_collected = state.dn_chunked_q
        k_collected = state.dn_chunked_k
        v_collected = state.dn_chunked_v
        z_collected = state.dn_chunked_z
    else:
        q_collected = ttnn.from_torch(
            torch.zeros((total_NV, C, K_DIM), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
        k_collected = ttnn.from_torch(
            torch.zeros((total_NV, C, K_DIM), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
        v_collected = ttnn.from_torch(
            torch.zeros((total_NV, C, V_DIM), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0))
        z_collected = ttnn.from_torch(
            torch.zeros((C, NCHIPS * VAL_DIM_CHIP), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh, mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1))

    def _gqa(t, n_kh, d):
        t2 = ttnn.reshape(t, [n_kh, 1, d])
        t3 = ttnn.repeat(t2, ttnn.Shape([1, N_REP, 1]))
        return ttnn.reshape(t3, [n_kh * N_REP, d])

    EPS_RMS = EPS / K_DIM

    for pos in range(C):
        all_pos = ttnn.slice(all_seq, [pos, 0], [pos + 1, IN_PROJ_OUT_CHIP])
        mixed_qkv = ttnn.slice(all_pos, [0, 0], [1, CONV_DIM_CHIP])
        z_tt_pos = ttnn.slice(all_pos, [0, CONV_DIM_CHIP],
                               [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
        ttnn.deallocate(all_pos)

        # Conv1d (manual 3-tap mul+sum recurrence).
        mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
        ttnn.deallocate(mixed_qkv)
        conv_input = ttnn.concat([dn['conv_st'], mixed_col], dim=-1)
        ttnn.deallocate(mixed_col)
        conv_prod = ttnn.mul(conv_input, dn['w_conv'])
        conv_out = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
        ttnn.deallocate(conv_prod)
        conv_state_new = ttnn.slice(conv_input, [0, 1],
                                     [CONV_DIM_CHIP, cfg['conv_kernel']])
        ttnn.deallocate(conv_input)
        ttnn.copy(conv_state_new, dn['conv_st'])
        ttnn.deallocate(conv_state_new)

        q_flat = ttnn.slice(conv_out, [0], [KEY_DIM_CHIP])
        k_flat = ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP])
        v_flat = ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP])
        ttnn.deallocate(conv_out)

        q = _gqa(q_flat, NK_PER_CHIP, K_DIM)
        k = _gqa(k_flat, NK_PER_CHIP, K_DIM)
        v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])
        ttnn.deallocate(q_flat); ttnn.deallocate(k_flat); ttnn.deallocate(v_flat)

        q = _rms_norm_manual(q, dn['q_l2_scale'], EPS_RMS, K_DIM)
        k = _rms_norm_manual(k, dn['k_l2_scale'], EPS_RMS, K_DIM)

        # Collect into batched buffers via slice_write
        q_3d = ttnn.reshape(q, [NV_PER_CHIP, 1, K_DIM])
        k_3d = ttnn.reshape(k, [NV_PER_CHIP, 1, K_DIM])
        v_3d = ttnn.reshape(v, [NV_PER_CHIP, 1, V_DIM])
        q_rm = ttnn.to_layout(q_3d, ttnn.ROW_MAJOR_LAYOUT)
        k_rm = ttnn.to_layout(k_3d, ttnn.ROW_MAJOR_LAYOUT)
        v_rm = ttnn.to_layout(v_3d, ttnn.ROW_MAJOR_LAYOUT)
        z_rm = ttnn.to_layout(z_tt_pos, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(q_rm, q_collected,
            [0, pos, 0], [NV_PER_CHIP, pos + 1, K_DIM], [1, 1, 1])
        ttnn.experimental.slice_write(k_rm, k_collected,
            [0, pos, 0], [NV_PER_CHIP, pos + 1, K_DIM], [1, 1, 1])
        ttnn.experimental.slice_write(v_rm, v_collected,
            [0, pos, 0], [NV_PER_CHIP, pos + 1, V_DIM], [1, 1, 1])
        ttnn.experimental.slice_write(z_rm, z_collected,
            [pos, 0], [pos + 1, VAL_DIM_CHIP], [1, 1])

        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v); ttnn.deallocate(z_tt_pos)
        ttnn.deallocate(q_3d); ttnn.deallocate(k_3d); ttnn.deallocate(v_3d)
        ttnn.deallocate(q_rm); ttnn.deallocate(k_rm); ttnn.deallocate(v_rm); ttnn.deallocate(z_rm)

    ttnn.deallocate(all_seq)

    # Convert collected to TILE for matmul
    q_seq_tile = ttnn.to_layout(q_collected, ttnn.TILE_LAYOUT)
    k_seq_tile = ttnn.to_layout(k_collected, ttnn.TILE_LAYOUT)
    v_seq_tile = ttnn.to_layout(v_collected, ttnn.TILE_LAYOUT)
    z_seq_tile = ttnn.to_layout(z_collected, ttnn.TILE_LAYOUT)
    # Only deallocate if WE allocated them (eager fallback). Pre-allocated
    # state buffers are owned by state — leave them alive for the next call.
    if not (hasattr(state, 'dn_chunked_q') and C == state.prefill_inner_c):
        ttnn.deallocate(q_collected); ttnn.deallocate(k_collected)
        ttnn.deallocate(v_collected); ttnn.deallocate(z_collected)

    _toc(_t, "3_pre_recurrence_collect")

    _t = _tic()
    # Transpose g_seq, beta_seq from [C, NV_PER_CHIP] to [NV_PER_CHIP, C]
    g_seq_NV = ttnn.transpose(g_seq, -2, -1)
    beta_seq_NV = ttnn.transpose(beta_seq, -2, -1)
    ttnn.deallocate(g_seq); ttnn.deallocate(beta_seq)

    # === CHUNKED NEUMANN RECURRENCE ===
    S_prev = ttnn.reshape(dn['ssm'], [NV_PER_CHIP, K_DIM, V_DIM])
    O_seq, S_new = _chunked_recurrence_tp(
        state, q_seq_tile, k_seq_tile, v_seq_tile, g_seq_NV, beta_seq_NV, S_prev, C)

    # Update dn['ssm'] from S_new
    S_new_4d = ttnn.reshape(S_new, [1, NV_PER_CHIP, K_DIM, V_DIM])
    ttnn.copy(S_new_4d, dn['ssm'])

    ttnn.deallocate(q_seq_tile); ttnn.deallocate(k_seq_tile); ttnn.deallocate(v_seq_tile)
    ttnn.deallocate(g_seq_NV); ttnn.deallocate(beta_seq_NV)
    ttnn.deallocate(S_new); ttnn.deallocate(S_new_4d)
    _toc(_t, "4_chunked_recurrence")

    _t = _tic()
    # === STAGE 6: batched post-recurrence ===
    # O_seq is [NV_PER_CHIP, C, V_DIM]; transpose to [C, NV_PER_CHIP, V_DIM]
    O_C_first = ttnn.transpose(O_seq, 0, 1)
    ttnn.deallocate(O_seq)

    # Batched rms_norm: reshape to 2D [C*NV_PER_CHIP, V_DIM], normalize, reshape back.
    # rms_norm operates over last dim (V_DIM) → per-head per-position independently.
    O_2d = ttnn.reshape(O_C_first, [C * NV_PER_CHIP, V_DIM])
    ttnn.deallocate(O_C_first)
    out_normed_2d = _rms_norm_manual(O_2d, dn['linear_attn_norm'], EPS, V_DIM)
    ttnn.deallocate(O_2d)
    out_normed_3d = ttnn.reshape(out_normed_2d, [C, NV_PER_CHIP, V_DIM])
    ttnn.deallocate(out_normed_2d)

    # Batched silu(z) * out_normed
    z_3d = ttnn.reshape(z_seq_tile, [C, NV_PER_CHIP, V_DIM])
    ttnn.deallocate(z_seq_tile)
    silu_z_3d = ttnn.silu(z_3d)
    ttnn.deallocate(z_3d)
    out_gated_3d = ttnn.mul(out_normed_3d, silu_z_3d)
    ttnn.deallocate(out_normed_3d); ttnn.deallocate(silu_z_3d)
    out_gated_2d = ttnn.reshape(out_gated_3d, [C, VAL_DIM_CHIP])
    ttnn.deallocate(out_gated_3d)

    # Batched w_out matmul (row-parallel) + all_reduce + residual add
    partial = ttnn.linear(out_gated_2d, dn['w_out'])  # [C, HIDDEN] per-chip partial
    ttnn.deallocate(out_gated_2d)
    reduced = _tp_all_reduce(state, partial)  # [C, HIDDEN] replicated
    ttnn.deallocate(partial)
    x_out_seq_tt = ttnn.add(x_seq_tt, reduced)  # [C, HIDDEN] fresh allocation
    ttnn.deallocate(reduced)
    _toc(_t, "5_post_recurrence_batched")
    return x_out_seq_tt


def deltanet_chunked_neumann_tp(state, x_seq_tt, dn, cfg, seq_len):
    """v4 (task #75): chunked-parallel DeltaNet across `seq_len` positions.

    Replaces the per-position sequential DN loop in v3 prefill with a single
    chunked call that exploits the Neumann factorization of (I - attn)^{-1}
    to compute all positions' recurrence in parallel.

    Inputs:
      x_seq_tt: [seq_len, HIDDEN] bf16 TILE, replicated across mesh
      dn:       per-layer DN weights dict (same as deltanet_step_tp)
      cfg:      model config
      seq_len:  number of positions to process

    Returns:
      x_out_seq_tt: [seq_len, HIDDEN] bf16 TILE, residual-added DN output
                   (replicated, mirrors deltanet_step_tp's output contract per position)

    STATUS (2026-05-20, v4 Stage 1, task #77): batched pre-norm + in_proj
    run ONCE per chunk on [seq_len, HIDDEN] / [seq_len, IN_PROJ_OUT_CHIP]
    instead of per-position. Per-position loop slices the in_proj output
    and calls `_deltanet_step_tp_from_inproj` (the refactored helper that
    holds stages 3-11 of DN).

    Saving: (seq_len - 1) × (rms_norm + in_proj) ops per DN layer.
    Same math as v3 — cos should be bit-identical.

    Design doc: research/v4_chunked_dn_design_2026_05_20.md
    Reference impl plan: research/c5_chunked_prefill_plan.md
    Validated primitives: feedback_c5_primitives_green (Neumann factorization
    + ttnn.cumsum both confirmed at production shape on single-chip).

    NEXT STAGES (per design doc §Stages):
      Stage 2: batched conv1d
      Stage 3: batched decay/gate + cumsum
      Stage 4: Neumann (I - attn)^{-1} chunked recurrence  ← THE BIG ONE
      Stage 5: multi-chunk loop + state thread
      Stage 6: batched output gate
    """
    import ttnn
    import torch
    from full_layer_tp_probe import (
        IN_PROJ_OUT_CHIP, EPS, CONV_DIM_CHIP, VAL_DIM_CHIP, NV_PER_CHIP,
    )

    # === Stage 4b-iii dispatch: chunked DN only for seq_len in {4, 8, 16, 32}.
    # Precision sweep 2026-05-21 (feedback_v4_chunked_dn_seq32_shipped.md):
    # at C=32 chunked is 1.58× faster with -1 top1 (within noise); at C=64
    # top1 drops -22%, at C=128 top1 drops -59% — unshippable. The 7 chained
    # bf16 matmuls in _chunked_recurrence_tp accumulate too much error at C≥64.
    # Fall through to v3 per-position path for longer prefill.
    if seq_len in (4, 8, 16, 32):
        return _chunked_dn_with_chunked_recurrence_tp(state, x_seq_tt, dn, cfg, seq_len)

    HIDDEN = cfg['hidden']

    # === STAGE 1: batched pre-norm + in_proj (run ONCE for the whole chunk) ===
    h_seq = _rms_norm_manual(x_seq_tt, dn['input_norm'], EPS, HIDDEN)
    all_seq = ttnn.linear(h_seq, dn['w_in'])  # [seq_len, IN_PROJ_OUT_CHIP]
    ttnn.deallocate(h_seq)

    # === STAGE 3: batched decay/gate computation (position-independent) ===
    # Slice a_seq, b_seq from all_seq batched, compute decay_seq + beta_seq
    # batched via the manual softplus path (owned_decay_gate kernel is
    # single-pos only). Per-pos loop slices decay_pos + beta_pos and passes
    # to helper as precomputed (skipping a/b slice + decay-gate stage in
    # helper). Note: forces "manual" decay-gate computation regardless of
    # state.deltanet_decay_gate_mode default — manual is mathematically
    # equivalent to owned within bf16 noise.
    a_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                       [seq_len, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_seq = ttnn.slice(all_seq, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                       [seq_len, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    a_biased_seq = ttnn.add(a_seq, dn['dt_bias'])
    if state.deltanet_decay_mode == "native_softplus":
        softplus_a_seq = ttnn.softplus(a_biased_seq)
    else:
        softplus_a_seq = ttnn.log(ttnn.add(ttnn.exp(a_biased_seq), 1.0))
    g_seq = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a_seq)
    beta_seq = ttnn.sigmoid(b_seq)   # [seq_len, NV_PER_CHIP]
    decay_seq = ttnn.exp(g_seq)       # [seq_len, NV_PER_CHIP]
    ttnn.deallocate(a_seq)
    ttnn.deallocate(b_seq)
    ttnn.deallocate(a_biased_seq)
    ttnn.deallocate(softplus_a_seq)
    ttnn.deallocate(g_seq)

    # Per-position loop: slice all_seq[pos] + decay/beta and pass to helper.
    dn_buf_init = torch.zeros((1, 1, seq_len, HIDDEN), dtype=torch.bfloat16)
    dn_out_buf = ttnn.from_torch(
        dn_buf_init,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for pos in range(seq_len):
        x_pos = ttnn.slice(x_seq_tt, [pos, 0], [pos + 1, HIDDEN])      # residual
        all_pos = ttnn.slice(all_seq, [pos, 0], [pos + 1, IN_PROJ_OUT_CHIP])
        decay_pos_2d = ttnn.slice(decay_seq, [pos, 0], [pos + 1, NV_PER_CHIP])
        beta_pos = ttnn.slice(beta_seq, [pos, 0], [pos + 1, NV_PER_CHIP])
        # Reshape decay to [1, NV_PER_CHIP, 1, 1] (what helper's downstream expects)
        decay_pos = ttnn.reshape(decay_pos_2d, [1, NV_PER_CHIP, 1, 1])
        ttnn.deallocate(decay_pos_2d)
        # x_pos is a VIEW of x_seq_tt — DO NOT deallocate (per B.2.2 lesson)
        # all_pos, decay_pos, beta_pos are fresh slices — helper deallocs internally.
        x_pos_out = _deltanet_step_tp_from_inproj(
            state, x_pos, all_pos, dn, cfg,
            precomputed_decay=decay_pos, precomputed_beta=beta_pos)
        x_pos_4d = ttnn.reshape(x_pos_out, [1, 1, 1, HIDDEN])
        x_pos_rm = ttnn.to_layout(x_pos_4d, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            x_pos_rm, dn_out_buf,
            [0, 0, pos, 0],
            [1, 1, pos + 1, HIDDEN],
            [1, 1, 1, 1],
        )
        ttnn.deallocate(x_pos_out)
        ttnn.deallocate(x_pos_4d)
        ttnn.deallocate(x_pos_rm)

    ttnn.deallocate(all_seq)
    ttnn.deallocate(decay_seq)
    ttnn.deallocate(beta_seq)

    # Reassemble into [seq_len, HIDDEN] TILE via clone (per B.2.2 fix).
    dn_out_4d_tile = ttnn.to_layout(dn_out_buf, ttnn.TILE_LAYOUT)
    ttnn.deallocate(dn_out_buf)
    _view = ttnn.reshape(dn_out_4d_tile, [seq_len, HIDDEN])
    x_out_seq_tt = ttnn.clone(_view)
    ttnn.deallocate(_view)
    ttnn.deallocate(dn_out_4d_tile)
    return x_out_seq_tt


def mlp_step_tp(state, x_tt, mlp):
    """One SwiGLU MLP TP step on the mesh.

    Aggressive ttnn.deallocate after each intermediate's last use — the
    Llama70B production pattern (llama3_70b_galaxy/tt/llama_mlp.py:141-193).
    Without these, intermediates accumulate across forward_token_tp calls
    and the 3rd call wedges (see feedback_p6_step2_hangs.md).
    """
    import ttnn
    from full_layer_tp_probe import EPS
    HIDDEN = state.cfg['hidden']
    h_tt = _rms_norm_manual(x_tt, mlp['post_norm'], EPS, HIDDEN)
    g = ttnn.linear(h_tt, mlp['w_gate'], activation="silu")
    u = ttnn.linear(h_tt, mlp['w_up'])
    ttnn.deallocate(h_tt)
    h = ttnn.mul(g, u)
    ttnn.deallocate(g)
    ttnn.deallocate(u)
    partial = ttnn.linear(h, mlp['w_down'])
    ttnn.deallocate(h)
    reduced = _tp_all_reduce(state, partial)
    ttnn.deallocate(partial)
    out = ttnn.add(x_tt, reduced)
    ttnn.deallocate(reduced)
    return out


def gated_attn_step_tp(state, x_tt, attn, cur_pos_tt, cur_pos, cos_tt, sin_tt, cfg):
    """One Gated Attention TP step on the mesh. Heads sharded across chips.

    Per-chip: N_Q/4 = 6 Q heads + N_KV/4 = 1 KV head. KV stays local-per-chip
    (no comm during SDPA). Only out_proj + residual all_reduce.
    """
    import ttnn
    HIDDEN = cfg['hidden']
    HEAD_DIM = cfg['head_dim']
    N_Q = cfg['n_q_heads']
    N_KV = cfg['n_kv_heads']
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
    NQ_PER_CHIP = N_Q // 4
    NKV_PER_CHIP = N_KV // 4
    QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM
    KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
    EPS = 1e-6
    # 1. Pre-norm (manual: see _rms_norm_manual doc)
    h_tt = _rms_norm_manual(x_tt, attn['input_norm'], EPS, HIDDEN)
    # 2. Sharded attn_qkv matmul → per-chip slab [Q+gate | K | V] for this chip's heads
    all_tt = ttnn.linear(h_tt, attn['w_qkv'])
    qg = ttnn.slice(all_tt, [0, 0], [1, QG_DIM_CHIP])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP], [1, QG_DIM_CHIP + KV_DIM_CHIP])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                       [1, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
    qg = ttnn.reshape(qg, [NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0], [NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, HEAD_DIM], [NQ_PER_CHIP, 2 * HEAD_DIM])
    k_tt = ttnn.reshape(k_flat, [NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [NKV_PER_CHIP, HEAD_DIM])
    # 3. q_norm, k_norm per-head (replicated weights — applied per chip)
    q_tt = _rms_norm_manual(q_tt, attn['q_norm'], EPS, HEAD_DIM)
    k_tt = _rms_norm_manual(k_tt, attn['k_norm'], EPS, HEAD_DIM)
    # 4. Partial RoPE V2 rotate-only
    half = ROTARY_DIM // 2
    def apply_rope_manual(t, n_heads):
        with _profile_scope(state, "rope"):
            rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
            passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
            x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
            x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
            neg_x2 = ttnn.neg(x2)
            rotated = ttnn.add(ttnn.mul(rot, cos_tt),
                                ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_tt))
            return ttnn.concat([rotated, passthru], dim=-1)
    def apply_rope_native_partial(t, n_heads):
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        rot4d = ttnn.reshape(rot, [1, 1, n_heads, ROTARY_DIM])
        passthru4d = ttnn.reshape(passthru, [1, 1, n_heads, HEAD_DIM - ROTARY_DIM])
        cos4d = ttnn.reshape(cos_tt, [1, 1, 1, ROTARY_DIM])
        sin4d = ttnn.reshape(sin_tt, [1, 1, 1, ROTARY_DIM])
        native_rot = ttnn.experimental.rotary_embedding(
            rot4d, cos4d, sin4d,
            compute_kernel_config=state.sdpa_compute_kernel_config)
        # rotary_embedding pads the head axis to tile height; trim it before
        # concatenating with the logical pass-through head rows.
        native_rot = ttnn.slice(native_rot, [0, 0, 0, 0], [1, 1, n_heads, ROTARY_DIM])
        out4d = ttnn.concat([native_rot, passthru4d], dim=-1)
        return ttnn.reshape(out4d, [n_heads, HEAD_DIM])
    if state.rope_mode == "native_partial":
        q_tt = apply_rope_native_partial(q_tt, NQ_PER_CHIP)
        k_tt = apply_rope_native_partial(k_tt, NKV_PER_CHIP)
    else:
        q_tt = apply_rope_manual(q_tt, NQ_PER_CHIP)
        k_tt = apply_rope_manual(k_tt, NKV_PER_CHIP)
    # 5. PAGED KV cache update via paged_update_cache.
    # Cache: [NUM_BLOCKS, N_KV, BLOCK_SIZE, HEAD_DIM] per-chip sharded along N_KV.
    # Input must be HEIGHT_SHARDED in L1, padded to TILE_HEIGHT=32 on dim -2.
    # cur_pos_tt is a [1] int32 replicated tensor — supports trace replay
    # (the non-paged update_cache_for_token_ takes Python int, bakes into trace).
    # See feedback_paged_refactor_constraints.md + P12.1 validated recipe.
    def _shard_for_paged_write(t_per_head, mem_cfg):
        # t_per_head: [NKV_PER_CHIP, HEAD_DIM] → [1, 1, NKV_PER_CHIP, HEAD_DIM]
        # → pad dim -2 to TILE_HEIGHT → HEIGHT_SHARDED L1 per supplied mem_cfg.
        t4d = ttnn.reshape(t_per_head, [1, 1, NKV_PER_CHIP, HEAD_DIM])
        t_padded = ttnn.pad(t4d, [[0, 0], [0, 0], [0, TILE_HEIGHT - NKV_PER_CHIP], [0, 0]],
                              value=0.0)
        return ttnn.to_memory_config(t_padded, mem_cfg)
    if state.use_fused_paged_update:
        if (state.fused_paged_write_mem_cfg_k is None or
                state.fused_paged_write_mem_cfg_v is None):
            raise RuntimeError("fused paged write mem_cfg is unavailable on this grid")
        k_sharded = _shard_for_paged_write(k_tt, state.fused_paged_write_mem_cfg_k)
        v_sharded = _shard_for_paged_write(v_tt, state.fused_paged_write_mem_cfg_v)
        ttnn.experimental.paged_fused_update_cache(
            attn['kc'], k_sharded, attn['vc'], v_sharded,
            update_idxs_tensor=cur_pos_tt,
            page_table=state.page_table_tt)
    else:
        k_sharded = _shard_for_paged_write(k_tt, state.paged_write_mem_cfg)
        v_sharded = _shard_for_paged_write(v_tt, state.paged_write_mem_cfg)
        ttnn.experimental.paged_update_cache(attn['kc'], k_sharded,
                                               update_idxs_tensor=cur_pos_tt,
                                               page_table=state.page_table_tt)
        ttnn.experimental.paged_update_cache(attn['vc'], v_sharded,
                                               update_idxs_tensor=cur_pos_tt,
                                               page_table=state.page_table_tt)
    ttnn.deallocate(k_sharded)
    ttnn.deallocate(v_sharded)
    # 6. PAGED SDPA decode (P18/P19 validated).
    # Replaces the manual Q@K^T softmax V path. Key wins:
    #   - Correctness: paged SDPA uses cur_pos_tensor to mask positions > cur_pos.
    #     The manual path did NOT mask, silently degrading at cur_pos > MAX_POS/2.
    #   - Long-context: paged SDPA is O(1) in MAX_POS; manual was O(MAX_POS).
    # NOTE (P19): at MAX_POS=256 under trace, the eager win (0.37 ms/layer)
    # collapses to wash (0.0075 ms/layer). Production uses trace, so this swap
    # is for CORRECTNESS + LONG-CONTEXT scaling, not a tok/s claim.
    # See feedback_p19_chained_paged_sdpa.md.
    assert NKV_PER_CHIP == 1, "paged SDPA per-chip assumes 1 KV head"
    q_for_sdpa = ttnn.reshape(q_tt, [1, 1, NQ_PER_CHIP, HEAD_DIM])
    attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_for_sdpa, attn['kc'], attn['vc'],
        cur_pos_tensor=cur_pos_tt,
        page_table_tensor=state.page_table_tt,
        program_config=state.paged_sdpa_progcfg,
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    attn_per_head = ttnn.reshape(attn_out, [NQ_PER_CHIP, HEAD_DIM])
    # 7. Sigmoid gate + multiply
    attn_gated = ttnn.mul(attn_per_head, ttnn.sigmoid(gate_tt))
    # 8. out_proj row-parallel + all_reduce
    attn_flat = ttnn.reshape(attn_gated, [1, NQ_PER_CHIP * HEAD_DIM])
    partial = ttnn.linear(attn_flat, attn['w_o'])
    reduced = _tp_all_reduce(state, partial)
    return ttnn.add(x_tt, reduced)


def update_input_buffers(state, token_id, cur_pos):
    """Write new token id + cur_pos + rotary index into the pre-allocated index
    buffers. Called OUTSIDE any captured trace, between execute_trace calls.
    Uses ttnn.copy_host_to_device_tensor for in-place buffer updates.

    P25 (2026-05-14): collapsed from 4 HtoD calls (x_buf, cur_pos, cos, sin)
    to 3 tiny index writes (tok_buf, cur_pos_buf, rot_idxs_buf). The embed
    lookup + cos/sin row lookup now happen on-device inside the trace via
    ttnn.embedding. Payload per step dropped from ~10.3 KB to ~12 bytes; more
    importantly, eliminated 4 host-side from_torch calls per step.
    """
    import ttnn
    import torch
    mesh = state.mesh

    # tok_buf [1, 1] uint32 — for ttnn.embedding(tok_buf, embed_tt) inside the trace.
    tok_host = ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(tok_host, state.tok_buf)
    # cur_pos_buf [1] int32 — used by paged_update_cache + paged SDPA (gated_attn).
    cur_pos_host = ttnn.from_torch(
        torch.tensor([cur_pos], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(cur_pos_host, state.cur_pos_buf)
    # rot_idxs_buf [1, 1] uint32 — for ttnn.embedding(rot_idxs_buf, cos/sin_table_tt).
    rot_host = ttnn.from_torch(
        torch.tensor([[cur_pos]], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    ttnn.copy_host_to_device_tensor(rot_host, state.rot_idxs_buf)


def forward_token_tp_inner(state, return_logits: bool = False):
    """The trace-captureable forward function. Reads ONLY from pre-allocated
    state buffers (state.tok_buf, state.cur_pos_buf, state.rot_idxs_buf) and
    does the embed + cos/sin row lookups on-device via ttnn.embedding (P25,
    2026-05-14). No host writes inside, no Python-int-baked args. Returns
    on-device argmax tensor (P22, 2026-05-14): vocab-sharded matmul +
    all_gather + slice + untilize + argmax. Returns UINT32 row-major tensor
    of shape (1, 1), replicated across all chips. Saves ~35 ms/tok vs reading
    back 152064 fp32 logits (P22) and ~0.7 ms/tok eliminating the 4-call
    host loop in update_input_buffers (P25).

    Callers MUST call update_input_buffers(state, token_id, cur_pos) FIRST
    to populate the tok/cur_pos/rot index buffers (outside any captured trace).
    """
    import ttnn
    cfg = state.cfg
    HIDDEN = cfg['hidden']
    # P25: on-device token embedding lookup. tok_buf is [1, 1] uint32; the
    # output is [1, 1, HIDDEN] bf16 — reshape to [1, HIDDEN] for the layer loop.
    # Force DRAM memory_config (friend's pattern, model.py:560) so downstream
    # rms_norm + linear get the same layout they got from the legacy x_buf
    # (which was created via from_torch into default DRAM).
    embed_out = ttnn.embedding(
        state.tok_buf, state.embed_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_tt = ttnn.reshape(embed_out, [1, HIDDEN])
    # P25: on-device cos/sin row lookup. Outputs [1, 1, ROTARY_DIM] bf16 —
    # reshape to [1, ROTARY_DIM] for gated_attn_step_tp (which expects 2-D).
    cos_row_raw = ttnn.embedding(
        state.rot_idxs_buf, state.cos_table_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_row_raw = ttnn.embedding(
        state.rot_idxs_buf, state.sin_table_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_row_tt = ttnn.reshape(cos_row_raw, [1, state.rotary_dim])
    sin_row_tt = ttnn.reshape(sin_row_raw, [1, state.rotary_dim])
    for _li, layer in enumerate(state.layers):
        if layer['type'] == 'linear_attention':
            x_tt = deltanet_step_tp(state, x_tt, layer['dn'], cfg)
        else:
            x_tt = gated_attn_step_tp(state, x_tt, layer['attn'],
                                       state.cur_pos_buf,
                                       0,  # vestigial cur_pos int (paged path ignores it)
                                       cos_row_tt, sin_row_tt, cfg)
        x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
    x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
    # Vocab-sharded LM head + on-device argmax. Per-chip linear produces
    # [1, VOCAB_PADDED/NCHIPS] then all_gather replicates to [1, VOCAB_PADDED]
    # on every chip. Slice to real vocab, untilize for argmax compatibility,
    # then argmax. Result is small UINT32 tensor — tiny readback.
    # Use keepdim=True + use_multicore=True (the only combo that returns
    # correct indices on [1, 152064] in our ttnn build — see p22_argmax_sanity6).
    sharded_logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, state.vocab_size])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    if return_logits:
        return rm_logits_tt
    argmax_tt = ttnn.argmax(rm_logits_tt, dim=-1, keepdim=True, use_multicore=True)
    return argmax_tt


def forward_token_tp(state, token_id, cur_pos):
    """Eager wrapper: update buffers then run inner forward.
    For traced decode, use update_input_buffers + execute_trace directly.
    """
    update_input_buffers(state, token_id, cur_pos)
    return forward_token_tp_inner(state)


_PREFILL_DEBUG_LAYER_IDX = 0  # incremented per attn call; only first prints

def gated_attn_step_prefill_tp(state, x_seq_tt, attn, cos_seq_tt, sin_seq_tt, cfg, seq_len):
    """B.2.2: prefill version of gated_attn_step_tp — parallel SDPA over all
    positions at once. Sister function to the existing single-position
    gated_attn_step_tp.

    Inputs:
      x_seq_tt: [seq_len, HIDDEN] bf16 TILE_LAYOUT replicated on mesh
      cos_seq_tt, sin_seq_tt: [seq_len, ROTARY_DIM] per-position RoPE tables
      cfg, seq_len: as usual

    Returns: [seq_len, HIDDEN] post-residual (replicated).

    Side effect: writes K, V to paged cache at positions [0, seq_len) via
    paged_fill_cache.

    Compared to gated_attn_step_tp (decode):
      - Q/K/V matmul broadcasts on seq_len leading dim
      - RoPE applied per-position (manual rotate-half with batched cos/sin)
      - KV cache write via paged_fill_cache (multi-position) instead of
        paged_update_cache (single-position)
      - SDPA via ttnn.transformer.scaled_dot_product_attention with
        is_causal=True (instead of paged_scaled_dot_product_attention_decode)
      - out_proj + all_reduce + residual all broadcast on seq_len
    """
    import ttnn
    import math
    import time as _time
    HIDDEN = cfg['hidden']
    HEAD_DIM = cfg['head_dim']
    NQ_PER_CHIP = cfg['n_q_heads'] // 4
    NKV_PER_CHIP = cfg['n_kv_heads'] // 4
    QG_DIM_CHIP = 2 * NQ_PER_CHIP * HEAD_DIM
    KV_DIM_CHIP = NKV_PER_CHIP * HEAD_DIM
    ROTARY_DIM = int(HEAD_DIM * cfg['partial_rotary_factor'])
    EPS = 1e-6

    # First-call diagnostic prints to localize any wedge
    global _PREFILL_DEBUG_LAYER_IDX
    DBG = _PREFILL_DEBUG_LAYER_IDX == 0
    _PREFILL_DEBUG_LAYER_IDX += 1
    def _t():
        ttnn.synchronize_device(state.mesh)
        return _time.time()
    if DBG:
        t0 = _t()
        print(f"  [prefill_attn dbg] enter, seq_len={seq_len}, NQ_PER_CHIP={NQ_PER_CHIP}, "
              f"NKV_PER_CHIP={NKV_PER_CHIP}", flush=True)

    # 1. Pre-norm (broadcasts on seq_len leading dim)
    h_tt = _rms_norm_manual(x_seq_tt, attn['input_norm'], EPS, HIDDEN)
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s after pre_norm  shape={list(h_tt.shape)}", flush=True)

    # 2. QKV matmul (batched on seq_len)
    all_tt = ttnn.linear(h_tt, attn['w_qkv'])  # [seq_len, QG_DIM_CHIP + 2*KV_DIM_CHIP]
    ttnn.deallocate(h_tt)
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s after QKV matmul  shape={list(all_tt.shape)}", flush=True)

    # 3. Slice QG / K / V and reshape per-head
    qg = ttnn.slice(all_tt, [0, 0], [seq_len, QG_DIM_CHIP])
    k_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP],
                          [seq_len, QG_DIM_CHIP + KV_DIM_CHIP])
    v_flat = ttnn.slice(all_tt, [0, QG_DIM_CHIP + KV_DIM_CHIP],
                          [seq_len, QG_DIM_CHIP + 2 * KV_DIM_CHIP])
    ttnn.deallocate(all_tt)

    # qg → [seq_len, NQ_PER_CHIP, 2*HEAD_DIM]
    qg = ttnn.reshape(qg, [seq_len, NQ_PER_CHIP, 2 * HEAD_DIM])
    q_tt = ttnn.slice(qg, [0, 0, 0], [seq_len, NQ_PER_CHIP, HEAD_DIM])
    gate_tt = ttnn.slice(qg, [0, 0, HEAD_DIM], [seq_len, NQ_PER_CHIP, 2 * HEAD_DIM])
    ttnn.deallocate(qg)
    k_tt = ttnn.reshape(k_flat, [seq_len, NKV_PER_CHIP, HEAD_DIM])
    v_tt = ttnn.reshape(v_flat, [seq_len, NKV_PER_CHIP, HEAD_DIM])
    ttnn.deallocate(k_flat)
    ttnn.deallocate(v_flat)

    # 4. QK normalize (broadcasts on seq_len, n_heads — applied per HEAD_DIM)
    q_tt = _rms_norm_manual(q_tt, attn['q_norm'], EPS, HEAD_DIM)
    k_tt = _rms_norm_manual(k_tt, attn['k_norm'], EPS, HEAD_DIM)

    # 5. Batched Manual RoPE (rotate-half on first ROTARY_DIM cols).
    # Per-position cos/sin via broadcasting [seq_len, 1, ROTARY_DIM] over heads dim.
    half = ROTARY_DIM // 2
    cos_b = ttnn.reshape(cos_seq_tt, [seq_len, 1, ROTARY_DIM])
    sin_b = ttnn.reshape(sin_seq_tt, [seq_len, 1, ROTARY_DIM])

    def apply_rope_seq(t, n_heads):
        rot = ttnn.slice(t, [0, 0, 0], [seq_len, n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, 0, ROTARY_DIM], [seq_len, n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0, 0], [seq_len, n_heads, half])
        x2 = ttnn.slice(rot, [0, 0, half], [seq_len, n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.add(
            ttnn.mul(rot, cos_b),
            ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_b),
        )
        return ttnn.concat([rotated, passthru], dim=-1)

    q_tt = apply_rope_seq(q_tt, NQ_PER_CHIP)
    k_tt = apply_rope_seq(k_tt, NKV_PER_CHIP)
    ttnn.deallocate(cos_b)
    ttnn.deallocate(sin_b)
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s after batched RoPE  q.shape={list(q_tt.shape)} k.shape={list(k_tt.shape)}", flush=True)

    # 6. Write K, V to paged cache for future decode tokens.
    # paged_fill_cache signature: (cache, input, page_table, batch_idx=0)
    # input shape: [1, N_KV, input_seq_len, HEAD_DIM]
    k_for_cache = ttnn.reshape(k_tt, [1, NKV_PER_CHIP, seq_len, HEAD_DIM])
    v_for_cache = ttnn.reshape(v_tt, [1, NKV_PER_CHIP, seq_len, HEAD_DIM])
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s before paged_fill_cache  k_for_cache.shape={list(k_for_cache.shape)}", flush=True)
    try:
        ttnn.experimental.paged_fill_cache(
            attn['kc'], k_for_cache, state.page_table_tt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(
            attn['vc'], v_for_cache, state.page_table_tt, batch_idx=0)
        if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s after paged_fill_cache", flush=True)
    except Exception as e:
        # Non-fatal for prefill validation: SDPA below uses the fresh K/V
        # directly, not the cache. Cache write only matters for subsequent
        # decode. Log + continue.
        print(f"  [warn] paged_fill_cache failed (cache will be stale for decode): "
              f"{type(e).__name__}: {e}", flush=True)

    # 7. Parallel SDPA with causal mask.
    # Q: [1, NQ_PER_CHIP, seq_len, HEAD_DIM] (1 batch, n_q heads, seq_len Q, head_dim)
    # K, V: [1, NKV_PER_CHIP, seq_len, HEAD_DIM]
    # SDPA handles GQA when N_KV < N_Q automatically.
    q_for_sdpa = ttnn.reshape(q_tt, [1, NQ_PER_CHIP, seq_len, HEAD_DIM])
    k_for_sdpa = ttnn.reshape(k_tt, [1, NKV_PER_CHIP, seq_len, HEAD_DIM])
    v_for_sdpa = ttnn.reshape(v_tt, [1, NKV_PER_CHIP, seq_len, HEAD_DIM])
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s before SDPA  q.shape={list(q_for_sdpa.shape)} k.shape={list(k_for_sdpa.shape)}", flush=True)
    attn_out = ttnn.transformer.scaled_dot_product_attention(
        q_for_sdpa, k_for_sdpa, v_for_sdpa,
        is_causal=True,
        scale=1.0 / math.sqrt(HEAD_DIM),
        compute_kernel_config=state.sdpa_compute_kernel_config,
    )
    if DBG: print(f"  [prefill_attn dbg]  +{_t()-t0:.2f}s after SDPA  shape={list(attn_out.shape)}", flush=True)
    # attn_out: [1, NQ_PER_CHIP, seq_len, HEAD_DIM]
    attn_per_head = ttnn.reshape(attn_out, [seq_len, NQ_PER_CHIP, HEAD_DIM])
    ttnn.deallocate(q_for_sdpa)
    ttnn.deallocate(k_for_sdpa)
    ttnn.deallocate(v_for_sdpa)
    ttnn.deallocate(k_for_cache)
    ttnn.deallocate(v_for_cache)
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_tt)
    ttnn.deallocate(v_tt)

    # 8. Sigmoid gate + multiply (broadcasts on seq_len)
    attn_gated = ttnn.mul(attn_per_head, ttnn.sigmoid(gate_tt))
    ttnn.deallocate(attn_per_head)
    ttnn.deallocate(gate_tt)

    # 9. out_proj row-parallel + all_reduce on [seq_len, HIDDEN]
    attn_flat = ttnn.reshape(attn_gated, [seq_len, NQ_PER_CHIP * HEAD_DIM])
    ttnn.deallocate(attn_gated)
    partial = ttnn.linear(attn_flat, attn['w_o'])  # [seq_len, HIDDEN] partial
    ttnn.deallocate(attn_flat)
    reduced = _tp_all_reduce(state, partial)
    ttnn.deallocate(partial)
    out = ttnn.add(x_seq_tt, reduced)
    ttnn.deallocate(reduced)
    return out


def forward_prefill_tp_inner(state, prompt_ids, capture_logits=False):
    """Process a multi-token prompt, populating KV cache + DeltaNet state.

    Production usage: returns last-position logits tensor (handle_generate_tp
    uses the argmax to sample the first generated token).

    Validation usage (capture_logits=True): returns a [seq_len, VOCAB] numpy
    array — per-position logits at fp32, host-side. Slow (sync per position)
    but exact for the cosine comparison gate.

    INITIAL STUB (Phase B.1): loops the existing decode `forward_token_tp_inner`
    once per prompt token. Functionally identical to the current
    `handle_generate_tp` prompt loop. This stub establishes the validation
    harness; cos against the loop-reference path is trivially 1.0.

    Phase B.2 will replace this body with parallel SDPA + paged_fill_cache +
    parallel MLP while retaining sequential DeltaNet (still mirrors decode for
    the recurrence). Phase B.3 upgrades DeltaNet to chunked-parallel via the
    Neumann factorization.
    """
    import ttnn
    import numpy as np

    seq_len = len(prompt_ids)
    if seq_len < 1:
        raise ValueError(f"prompt_ids must have len >= 1, got {seq_len}")
    if seq_len > MAX_POS:
        raise ValueError(f"prompt_ids len {seq_len} > MAX_POS {MAX_POS}")

    # S1a: opt-in whole-prompt chunked prefill (default OFF → prod path unchanged).
    # Returns production-equivalent last-position logits; functionally validated
    # (coherent generation). See research/27b_chunked_prefill_plan.md.
    if getattr(state, "prefill_chunked", False) and not capture_logits:
        return forward_prefill_chunked_tp(state, prompt_ids)

    VOCAB = state.vocab_size
    if capture_logits:
        logits_arr = np.empty((seq_len, VOCAB), dtype=np.float32)

    last_logits_tt = None
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        last_logits_tt = forward_token_tp_inner(state, return_logits=True)
        if capture_logits:
            ttnn.synchronize_device(state.mesh)
            arr = ttnn.to_torch(
                last_logits_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            )[0].float().cpu().numpy().reshape(VOCAB)
            logits_arr[pos] = arr

    if capture_logits:
        return logits_arr
    ttnn.synchronize_device(state.mesh)
    return last_logits_tt


def _prefill_dn_chunked_blocks(state, x_seq_tt, dn, cfg, L, C=32):
    """S1b: block-chunked DeltaNet prefill. Process L positions in C-token blocks
    via deltanet_chunked_neumann_tp (Neumann recurrence) instead of its
    per-position fallback for L>C. Correct because the chunked DN threads BOTH its
    recurrence state (dn['ssm']) and conv1d state (dn['conv_st'], updated in-place
    per position, server_tp.py:1176) across calls — so sequential 32-blocks see the
    prior block's tail. The last ragged block (<C) uses the op's own per-position
    fallback. Block slices are 32-aligned (tile-aligned views); we never dealloc a
    view or the source mid-loop (view-decay safety)."""
    import ttnn
    if L <= C:
        return deltanet_chunked_neumann_tp(state, x_seq_tt, dn, cfg, L)
    HIDDEN = cfg['hidden']
    outs = []
    for start in range(0, L, C):
        blen = min(C, L - start)
        x_block = ttnn.slice(x_seq_tt, [start, 0], [start + blen, HIDDEN])
        outs.append(deltanet_chunked_neumann_tp(state, x_block, dn, cfg, blen))
    return ttnn.concat(outs, dim=0)


def forward_prefill_chunked_tp(state, prompt_ids, capture_logits=False):
    """S1a chunked prefill (Phase B.2): process the whole prompt in ONE parallel
    pass instead of looping single-token decode per position. Additive sister to
    forward_prefill_tp_inner (the stub), which is left untouched.

    Per layer: deltanet_chunked_neumann_tp (chunked Neumann for L<=32, per-position
    fallback above) for DeltaNet; gated_attn_step_prefill_tp (one causal SDPA +
    paged_fill_cache over the whole prompt) for attention; mlp_step_tp (leading-dim
    agnostic) for MLP. LM head runs over all L rows — avoids a sub-tile last-row
    slice in TILE layout (view-decay); the last row is taken in row-major.

    capture_logits=True -> [seq_len, VOCAB] fp32 numpy (per-position, for the cosine
    gate vs the stub). False -> last-position logits (row-major [1, VOCAB]).
    Assumes a fresh sequence at positions 0..L-1 (reset state buffers first).
    """
    import ttnn
    import torch
    cfg = state.cfg
    HIDDEN = cfg['hidden']
    seq_len = len(prompt_ids)
    if not (1 <= seq_len <= MAX_POS):
        raise ValueError(f"prompt_ids len {seq_len} out of range [1, {MAX_POS}]")
    mesh = state.mesh

    # Multi-position embed + RoPE rows (mirror forward_token_tp_inner's single-pos
    # path, but for L positions; fresh sequence => positions 0..L-1).
    tok_tt = ttnn.from_torch(
        torch.tensor([list(prompt_ids)], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    x_tt = ttnn.reshape(
        ttnn.embedding(tok_tt, state.embed_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [seq_len, HIDDEN])
    pos_tt = ttnn.from_torch(
        torch.tensor([list(range(seq_len))], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
    cos_seq_tt = ttnn.reshape(
        ttnn.embedding(pos_tt, state.cos_table_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [seq_len, state.rotary_dim])
    sin_seq_tt = ttnn.reshape(
        ttnn.embedding(pos_tt, state.sin_table_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [seq_len, state.rotary_dim])

    for layer in state.layers:
        if layer['type'] == 'linear_attention':
            x_tt = _prefill_dn_chunked_blocks(state, x_tt, layer['dn'], cfg, seq_len)
        else:
            x_tt = gated_attn_step_prefill_tp(state, x_tt, layer['attn'],
                                              cos_seq_tt, sin_seq_tt, cfg, seq_len)
        x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
    x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)

    # Vocab-sharded LM head over all L rows (mirror forward_token_tp_inner).
    sharded = ttnn.linear(x_tt, state.lm_head_tt)
    gathered = ttnn.all_gather(sharded, dim=-1)
    sliced = ttnn.slice(gathered, [0, 0], [seq_len, state.vocab_size])
    rm_logits = ttnn.untilize(sliced, use_multicore=True)
    ttnn.synchronize_device(mesh)
    if capture_logits:
        arr = ttnn.to_torch(rm_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        return arr[:seq_len].float().cpu().numpy().reshape(seq_len, state.vocab_size)
    return ttnn.slice(rm_logits, [seq_len - 1, 0], [seq_len, state.vocab_size])


def forward_prefill_chunked_traced_inner(state):
    """Trace-friendly twin of forward_prefill_chunked_tp at fixed L=chunk_size.

    Reads tokens + positions from pre-allocated state.prefill_tok_buf /
    state.prefill_pos_buf instead of allocating inline. No host-side allocations
    happen in this function — safe to capture as a trace once the input buffers
    are populated. Host updates the buffers between replays via
    copy_host_to_device_tensor.

    Returns a [chunk_size, VOCAB] row-major device logits tensor. Caller slices
    [actual_L - 1] for the real last-position.
    """
    import ttnn
    cfg = state.cfg
    HIDDEN = cfg['hidden']
    L = state.prefill_chunk_size

    x_tt = ttnn.reshape(
        ttnn.embedding(state.prefill_tok_buf, state.embed_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [L, HIDDEN])
    cos_seq_tt = ttnn.reshape(
        ttnn.embedding(state.prefill_pos_buf, state.cos_table_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [L, state.rotary_dim])
    sin_seq_tt = ttnn.reshape(
        ttnn.embedding(state.prefill_pos_buf, state.sin_table_tt, layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG),
        [L, state.rotary_dim])

    for layer in state.layers:
        if layer['type'] == 'linear_attention':
            x_tt = _prefill_dn_chunked_blocks(state, x_tt, layer['dn'], cfg, L)
        else:
            x_tt = gated_attn_step_prefill_tp(state, x_tt, layer['attn'],
                                              cos_seq_tt, sin_seq_tt, cfg, L)
        x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
    x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)

    sharded = ttnn.linear(x_tt, state.lm_head_tt)
    gathered = ttnn.all_gather(sharded, dim=-1)
    sliced = ttnn.slice(gathered, [0, 0], [L, state.vocab_size])
    return ttnn.untilize(sliced, use_multicore=True)


def update_prefill_input_buffers(state, prompt_ids, chunk_start_idx=0):
    """Host-side: write padded prompt + position indices into the pre-allocated
    trace input buffers. Call BEFORE execute_trace (outside the captured region).
    For L < chunk_size: prompt_ids gets padded with 0. For multi-chunk (later),
    chunk_start_idx > 0 shifts the position indices."""
    import ttnn
    import torch
    L = state.prefill_chunk_size
    L_actual = len(prompt_ids)
    if L_actual > L:
        raise ValueError(f"prompt L={L_actual} > chunk_size={L}; needs chunking")
    padded = list(prompt_ids) + [0] * (L - L_actual)
    tok_host = ttnn.from_torch(
        torch.tensor(padded, dtype=torch.int32).reshape(1, L),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(tok_host, state.prefill_tok_buf)
    pos_host = ttnn.from_torch(
        torch.arange(chunk_start_idx, chunk_start_idx + L, dtype=torch.int32).reshape(1, L),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    ttnn.copy_host_to_device_tensor(pos_host, state.prefill_pos_buf)


# --- Handlers -----------------------------------------------------------------
def handle_status(state: MeshServerState, args: dict) -> dict:
    return {
        "ok": True,
        "mesh_open": state.mesh is not None,
        "num_devices": state.mesh.get_num_devices() if state.mesh else 0,
        "num_layers_planned": state.num_layers,
        "num_layers_loaded": len(state.layers),
        "stage": "production_p25_traced_tp",
        "last_run": state.last_run,
    }


def handle_shutdown(state: MeshServerState, args: dict) -> dict:
    return {"ok": True, "shutting_down": True}


def _reset_state_buffers(state):
    """Zero out per-layer SSM, conv_state, and paged KV cache buffers.

    Required between queries because the warmup in _ensure_decode_trace
    advances state, and prior queries' state must not leak into new ones.
    Uses ttnn.copy_host_to_device_tensor for in-place updates so the device
    buffers stay at their allocated addresses (the trace was captured
    against those addresses).
    """
    import ttnn, torch
    from full_layer_tp_probe import N_V_HEADS, K_DIM, V_DIM, CONV_DIM
    cfg = state.cfg
    mesh = state.mesh

    ssm_host = ttnn.from_torch(
        torch.zeros(1, N_V_HEADS, K_DIM, V_DIM, dtype=torch.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
    conv_host = ttnn.from_torch(
        torch.zeros(CONV_DIM, cfg['conv_kernel'] - 1, dtype=torch.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    # G4 conv1d pre-split: zero buffer for each split conv_st tap ([CONV_DIM, 1]).
    conv_split_host = ttnn.from_torch(
        torch.zeros(CONV_DIM, 1, dtype=torch.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    kv_host = ttnn.from_torch(
        torch.zeros(NUM_BLOCKS, cfg['n_kv_heads'], BLOCK_SIZE, cfg['head_dim'],
                       dtype=torch.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1))
    for layer in state.layers:
        if layer['type'] == 'linear_attention':
            ttnn.copy_host_to_device_tensor(ssm_host, layer['dn']['ssm'])
            ttnn.copy_host_to_device_tensor(conv_host, layer['dn']['conv_st'])
            # G4: also reset split tensors used by owned conv1d path.
            for split_tt in layer['dn'].get('conv_st_split', []):
                ttnn.copy_host_to_device_tensor(conv_split_host, split_tt)
        else:
            ttnn.copy_host_to_device_tensor(kv_host, layer['attn']['kc'])
            ttnn.copy_host_to_device_tensor(kv_host, layer['attn']['vc'])


def _ensure_decode_trace(state):
    """Capture the decode forward trace once. Subsequent calls reuse it.

    Per P14 (commit 2d30af7): captures forward_token_tp_inner — reads only
    from pre-allocated state buffers, no host writes inside captured region.
    Per-step execute_trace then runs the captured graph after update_input_buffers
    fills new token/cur_pos values into the buffers.

    Warmup: run 2 eager forwards FIRST so all JIT kernels are compiled.
    Per feedback_c4v4_validated, JIT during capture hangs on Blackhole.
    """
    import ttnn
    if state.trace_id is not None:
        return
    print("[trace] warmup + capture decode trace…", flush=True)
    import time as _time
    t0 = _time.time()
    # Warmup eager — JIT all kernels for the inner forward
    update_input_buffers(state, token_id=0, cur_pos=0)
    _ = forward_token_tp_inner(state)
    ttnn.synchronize_device(state.mesh)
    update_input_buffers(state, token_id=0, cur_pos=1)
    _ = forward_token_tp_inner(state)
    ttnn.synchronize_device(state.mesh)
    # Capture
    update_input_buffers(state, token_id=0, cur_pos=2)
    state.trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
    # P22: forward returns on-device argmax tensor (UINT32 [1,1]) not full logits.
    state.traced_argmax_tt = forward_token_tp_inner(state)
    ttnn.end_trace_capture(state.mesh, state.trace_id, cq_id=0)
    print(f"  ✓ decode trace captured in {(_time.time()-t0)*1000:.0f} ms "
          f"(id={state.trace_id})", flush=True)


def _traced_forward(state, token_id, cur_pos):
    """Equivalent to forward_token_tp but uses the captured trace.

    P22 (2026-05-14): returns on-device argmax tensor (UINT32 [1,1]) — read
    via to_torch(..., ConcatMeshToTensor(dim=0))[0] for the next-token id.
    """
    import ttnn
    update_input_buffers(state, token_id, cur_pos)
    ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
    return state.traced_argmax_tt


def _summary_ms(samples):
    import numpy as np
    if not samples:
        return {}
    arr = np.array(samples, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def handle_bench_decode_tp_components(state: MeshServerState, args: dict) -> dict:
    """Server-resident TP decomposition for the current production trace.

    This endpoint intentionally runs inside the persistent server process so it
    does not open a second mesh device. It measures the current P25 production
    trace components directly:
      - update_input_buffers only (sync-bounded)
      - execute_trace only (sync-bounded)
      - update + execute_trace combined (production timed region)
      - on-device argmax readback (tiny tensor)

    It does not claim a speedup; it sizes opportunity before we choose a patch.
    """
    import ttnn
    import time as _time

    prompt = args.get("prompt", "The capital of France is")
    iters = int(args.get("iters", 20))
    warmup = int(args.get("warmup", 3))
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if state.tok is None or not state.layers:
        return {"error": "server not fully loaded"}

    _ensure_decode_trace(state)
    _reset_state_buffers(state)
    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}

    def sync():
        ttnn.synchronize_device(state.mesh)

    def timed(fn):
        sync()
        t0 = _time.perf_counter()
        out = fn()
        sync()
        return (_time.perf_counter() - t0) * 1000.0, out

    # Seed trace output for readback timing and put buffers in a valid state.
    update_input_buffers(state, prompt_ids[0], 0)
    ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
    sync()

    # Warm up the exact measurement surfaces.
    for i in range(warmup):
        tid = prompt_ids[i % len(prompt_ids)]
        pos = i % MAX_POS
        update_input_buffers(state, tid, pos)
        ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
    sync()

    update_ms = []
    execute_ms = []
    combined_ms = []
    readback_ms = []
    next_ids = []

    for i in range(iters):
        tid = prompt_ids[i % len(prompt_ids)]
        pos = i % MAX_POS

        dt, _ = timed(lambda tid=tid, pos=pos: update_input_buffers(state, tid, pos))
        update_ms.append(dt)

        dt, _ = timed(lambda: ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False))
        execute_ms.append(dt)

        def read_argmax():
            return ttnn.to_torch(
                state.traced_argmax_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )
        t0 = _time.perf_counter()
        idx_concat = read_argmax()
        readback_ms.append((_time.perf_counter() - t0) * 1000.0)
        next_ids.append(int(idx_concat.cpu().numpy().reshape(-1)[0]))

        tid2 = next_ids[-1]
        pos2 = (pos + 1) % MAX_POS
        def update_execute():
            update_input_buffers(state, tid2, pos2)
            ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)
        dt, _ = timed(update_execute)
        combined_ms.append(dt)

    # Leave the server in a clean state for the next request.
    _reset_state_buffers(state)

    result = {
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "iters": iters,
        "warmup": warmup,
        "summary_ms": {
            "update_input_buffers": _summary_ms(update_ms),
            "execute_trace": _summary_ms(execute_ms),
            "update_plus_execute": _summary_ms(combined_ms),
            "argmax_readback": _summary_ms(readback_ms),
        },
        "samples_ms": {
            "update_input_buffers": update_ms,
            "execute_trace": execute_ms,
            "update_plus_execute": combined_ms,
            "argmax_readback": readback_ms,
        },
        "next_ids_sample": next_ids[: min(8, len(next_ids))],
        "note": (
            "Component timings are sync-bounded and may not sum exactly to the "
            "production generate_tp ms/tok because update and execute are "
            "also measured in isolated loops."
        ),
    }
    state.last_run = {
        "cmd": "bench_decode_tp_components",
        "median_update_ms": result["summary_ms"]["update_input_buffers"].get("median"),
        "median_execute_ms": result["summary_ms"]["execute_trace"].get("median"),
        "median_combined_ms": result["summary_ms"]["update_plus_execute"].get("median"),
    }
    return result
















def _read_argmax_id(state: MeshServerState, argmax_tt) -> int:
    import ttnn
    idx_concat = ttnn.to_torch(
        argmax_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )
    return int(idx_concat.cpu().numpy().reshape(-1)[0])


























def _profile_category(op_name: str, context: str) -> str:
    if "rope" in context:
        return "RoPE"
    if "rms_norm" in context:
        return "RMSNorm"
    if "paged_scaled_dot_product_attention" in op_name:
        return "SDPA"
    if "paged_update_cache" in op_name or "paged_fused_update_cache" in op_name:
        return "cache_update"
    if op_name in ("ttnn.all_reduce", "ttnn.reduce_scatter", "ttnn.all_gather"):
        return "collectives"
    if op_name == "ttnn.linear":
        return "matmul"
    if op_name in ("ttnn.embedding", "ttnn.argmax", "ttnn.untilize"):
        return "lm_head_or_io"
    if "deltanet_conv" in context:
        return "DeltaNet_conv"
    if "deltanet_qkv_repeat" in context:
        return "DeltaNet_qkv_repeat"
    if "deltanet_decay_gate" in context:
        return "DeltaNet_decay_gate"
    if "deltanet_recurrence" in context:
        return "DeltaNet_recurrence"
    if "deltanet_output_gate" in context:
        return "DeltaNet_output_gate"
    if "deltanet_state_update" in context:
        return "DeltaNet_state_update"
    if "deltanet" in context:
        return "DeltaNet_other"
    if "attention" in context:
        return "attention_other"
    if "mlp" in context:
        return "MLP_other"
    return "other"


def _summarize_profile_records(records: list[dict]) -> dict:
    import numpy as np
    from collections import Counter, defaultdict

    category_counts = Counter(r["category"] for r in records)
    op_counts = Counter(r["op"] for r in records)
    category_ms = defaultdict(float)
    op_ms = defaultdict(float)
    for record in records:
        ms = record.get("ms")
        if ms is not None:
            category_ms[record["category"]] += float(ms)
            op_ms[record["op"]] += float(ms)

    total_ms = sum(category_ms.values()) if category_ms else None
    categories = []
    for category, count in category_counts.most_common():
        ms = category_ms.get(category)
        categories.append({
            "category": category,
            "count": int(count),
            "sync_bounded_ms": ms if category_ms else None,
            "pct_of_profiled_ms": (
                (100.0 * ms / total_ms) if total_ms and ms is not None else None
            ),
        })
    top_ops = []
    for op, count in op_counts.most_common(30):
        ms = op_ms.get(op)
        top_ops.append({
            "op": op,
            "count": int(count),
            "sync_bounded_ms": ms if op_ms else None,
            "pct_of_profiled_ms": (
                (100.0 * ms / total_ms) if total_ms and ms is not None else None
            ),
        })

    per_record_ms = [r["ms"] for r in records if r.get("ms") is not None]
    return {
        "total_ops": len(records),
        "total_profiled_ms": float(total_ms) if total_ms is not None else None,
        "per_op_ms_summary": {
            "median": float(np.median(per_record_ms)) if per_record_ms else None,
            "mean": float(np.mean(per_record_ms)) if per_record_ms else None,
            "max": float(np.max(per_record_ms)) if per_record_ms else None,
        },
        "categories": categories,
        "top_ops": top_ops,
    }


def handle_profile_decode_tp_ops(state: MeshServerState, args: dict) -> dict:
    """In-server op-count and optional sync-bounded eager timing profile.

    This is a safe fallback for qb2, where true Tracy/device-profiler timing is
    not currently available in the resident server build. It profiles the same
    production forward function used to capture the decode trace, but runs it
    eagerly with TTNN calls monkey-patched inside this process.
    """
    import time as _time
    import ttnn

    prompt = args.get("prompt", "The capital of France is")
    timed = bool(args.get("timed", False))
    include_records = bool(args.get("include_records", False))
    deltanet_decay_mode = args.get("deltanet_decay_mode", "manual")
    if deltanet_decay_mode not in ("manual", "native_softplus"):
        return {"error": "deltanet_decay_mode must be manual or native_softplus"}
    deltanet_recurrence_mode = args.get("deltanet_recurrence_mode", "manual")
    if deltanet_recurrence_mode not in ("manual", "owned_gdn", "owned_gdn_inplace"):
        return {"error": "deltanet_recurrence_mode must be manual, owned_gdn, or owned_gdn_inplace"}
    if state.tok is None or not state.layers:
        return {"error": "server not fully loaded"}
    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}

    # Ensure kernels are already compiled and the production trace exists.
    _ensure_decode_trace(state)
    _reset_state_buffers(state)

    records = []
    patched = []

    def sync():
        ttnn.synchronize_device(state.mesh)

    def patch_attr(owner, attr, op_name):
        if not hasattr(owner, attr):
            return
        orig = getattr(owner, attr)
        if not callable(orig):
            return

        def wrapped(*f_args, **f_kwargs):
            context = "/".join(state.profile_context_stack)
            t0 = None
            if timed:
                sync()
                t0 = _time.perf_counter()
            out = orig(*f_args, **f_kwargs)
            ms = None
            if timed:
                sync()
                ms = (_time.perf_counter() - t0) * 1000.0
            records.append({
                "op": op_name,
                "context": context,
                "category": _profile_category(op_name, context),
                "ms": ms,
            })
            return out

        setattr(owner, attr, wrapped)
        patched.append((owner, attr, orig))

    for name in (
        "embedding", "reshape", "linear", "slice", "concat", "neg", "add",
        "sub", "mul", "div", "sqrt", "sum", "exp", "log", "sigmoid", "silu",
        "rms_norm", "repeat", "copy", "pad", "to_memory_config", "all_reduce",
        "reduce_scatter", "all_gather", "argmax", "untilize",
    ):
        patch_attr(ttnn, name, f"ttnn.{name}")
    for name in (
        "paged_update_cache", "paged_fused_update_cache", "rotary_embedding",
        "rotary_embedding_llama_fused_qk", "qwen36_gdn_decode_owned",
    ):
        patch_attr(ttnn.experimental, name, f"ttnn.experimental.{name}")
    patch_attr(
        ttnn.transformer,
        "paged_scaled_dot_product_attention_decode",
        "ttnn.transformer.paged_scaled_dot_product_attention_decode",
    )

    # Add high-level context without changing production code paths.
    orig_dn = globals()["deltanet_step_tp"]
    orig_attn = globals()["gated_attn_step_tp"]
    orig_mlp = globals()["mlp_step_tp"]
    orig_norm = globals()["_rms_norm_manual"]
    orig_all_reduce = globals()["_tp_all_reduce"]

    def dn_wrapped(*f_args, **f_kwargs):
        with _profile_scope(state, "deltanet"):
            return orig_dn(*f_args, **f_kwargs)

    def attn_wrapped(*f_args, **f_kwargs):
        with _profile_scope(state, "attention"):
            return orig_attn(*f_args, **f_kwargs)

    def mlp_wrapped(*f_args, **f_kwargs):
        with _profile_scope(state, "mlp"):
            return orig_mlp(*f_args, **f_kwargs)

    def norm_wrapped(*f_args, **f_kwargs):
        with _profile_scope(state, "rms_norm"):
            return orig_norm(*f_args, **f_kwargs)

    def all_reduce_wrapped(*f_args, **f_kwargs):
        with _profile_scope(state, "collective"):
            return orig_all_reduce(*f_args, **f_kwargs)

    globals()["deltanet_step_tp"] = dn_wrapped
    globals()["gated_attn_step_tp"] = attn_wrapped
    globals()["mlp_step_tp"] = mlp_wrapped
    globals()["_rms_norm_manual"] = norm_wrapped
    globals()["_tp_all_reduce"] = all_reduce_wrapped

    old_decay = state.deltanet_decay_mode
    old_recurrence = state.deltanet_recurrence_mode
    state.deltanet_decay_mode = deltanet_decay_mode
    state.deltanet_recurrence_mode = deltanet_recurrence_mode
    state.profile_records = records
    state.profile_context_stack = []
    try:
        update_input_buffers(state, int(prompt_ids[0]), 0)
        sync()
        t0 = _time.perf_counter()
        out = forward_token_tp_inner(state)
        sync()
        eager_forward_ms = (_time.perf_counter() - t0) * 1000.0
        argmax_id = _read_argmax_id(state, out)
    finally:
        state.deltanet_decay_mode = old_decay
        state.deltanet_recurrence_mode = old_recurrence
        state.profile_records = None
        state.profile_context_stack = []
        globals()["deltanet_step_tp"] = orig_dn
        globals()["gated_attn_step_tp"] = orig_attn
        globals()["mlp_step_tp"] = orig_mlp
        globals()["_rms_norm_manual"] = orig_norm
        globals()["_tp_all_reduce"] = orig_all_reduce
        for owner, attr, orig in reversed(patched):
            setattr(owner, attr, orig)
        _reset_state_buffers(state)

    result = {
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "token_id": int(prompt_ids[0]),
        "timed": timed,
        "deltanet_decay_mode": deltanet_decay_mode,
        "deltanet_recurrence_mode": deltanet_recurrence_mode,
        "eager_forward_ms": eager_forward_ms,
        "argmax_id": argmax_id,
        "summary": _summarize_profile_records(records),
        "limitations": [
            "This profiles eager execution of the production trace body inside the resident server.",
            "When timed=true, every recorded op is sync-bounded; totals are not equal to execute_trace replay time.",
            "Use categories and counts to choose candidates; require full trace/full decode measurement before speedup claims.",
        ],
    }
    if include_records:
        result["records"] = records
    state.last_run = {
        "cmd": "profile_decode_tp_ops",
        "timed": timed,
        "total_ops": result["summary"]["total_ops"],
        "eager_forward_ms": eager_forward_ms,
    }
    return result




















def _sample_from_logits(logits, temperature, top_p, top_k, rng):
    """Host-side temperature + top-k + top-p (nucleus) sampling from 1-D logits.
    Returns a token id. temperature<=0 never reaches here (greedy = traced argmax)."""
    import numpy as np
    lg = logits.astype(np.float64) / max(temperature, 1e-6)
    if top_k and 0 < top_k < lg.size:
        lg[lg < np.partition(lg, -top_k)[-top_k]] = -np.inf
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]
        keep = order[:np.searchsorted(np.cumsum(p[order]), top_p) + 1]
        masked = np.zeros_like(p); masked[keep] = p[keep]
        p = masked / masked.sum()
    return int(rng.choice(p.size, p=p))


def _generate_sampled_tp(state, prompt, prompt_ids, max_tokens, chunk_size,
                         temperature, top_p, top_k, seed):
    """Sampling generate (temperature>0): non-traced forward returning logits +
    host-side temp/top-p/top-k sampling. Additive sister to the greedy traced path
    (left untouched). Same streamed chunk format. Slower (per-step logits readback,
    no trace) — sampling is a quality feature, not the perf path."""
    import ttnn
    import numpy as np
    import time as _time
    rng = np.random.default_rng(seed)

    def _read(rm):
        t = ttnn.to_torch(rm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
        return t.float().numpy().reshape(-1)[:state.vocab_size]

    _reset_state_buffers(state)
    t0 = _time.time()
    last_rm = None
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        last_rm = forward_token_tp_inner(state, return_logits=True)
    ttnn.synchronize_device(state.mesh)
    prefill_ms = (_time.time() - t0) * 1000.0
    logits = _read(last_rm); ttnn.deallocate(last_rm)

    generated_ids, decode_times = [], []
    cur_pos = len(prompt_ids)
    eos_id = getattr(state.tok, "eos_token_id", None)
    text_so_far, pending, stopped_on_eos = "", [], False

    def _flush():
        return {
            "token_text": "".join(p["token_text"] for p in pending),
            "token_ids": [p["token_id"] for p in pending],
            "tok_idx_start": pending[0]["tok_idx"],
            "tok_idx_end": pending[-1]["tok_idx"],
        }

    for step in range(max_tokens):
        next_id = _sample_from_logits(logits, temperature, top_p, top_k, rng)
        generated_ids.append(next_id)
        new_text = state.tok.decode(generated_ids, skip_special_tokens=True)
        delta = new_text[len(text_so_far):]; text_so_far = new_text
        pending.append({"token_id": next_id, "token_text": delta, "tok_idx": step})
        if len(pending) >= chunk_size:
            yield _flush(); pending = []
        if eos_id is not None and next_id == eos_id:
            stopped_on_eos = True; break
        td0 = _time.time()
        update_input_buffers(state, next_id, cur_pos)
        rm = forward_token_tp_inner(state, return_logits=True)
        ttnn.synchronize_device(state.mesh)
        logits = _read(rm); ttnn.deallocate(rm)
        decode_times.append((_time.time() - td0) * 1000.0); cur_pos += 1

    if pending:
        yield _flush()
    total_ms = (_time.time() - t0) * 1000.0
    ms_per_tok = (sum(decode_times) / len(decode_times)) if decode_times else float("nan")
    yield {
        "_final": True, "prompt": prompt, "generated_text": text_so_far,
        "full_text": prompt + text_so_far, "prompt_ids": list(prompt_ids),
        "generated_ids": generated_ids, "n_prompt_tokens": len(prompt_ids),
        "n_generated_tokens": len(generated_ids), "prefill_ms": prefill_ms,
        "total_ms": total_ms, "ms_per_tok": ms_per_tok,
        "tok_per_sec": 1000.0 / ms_per_tok if ms_per_tok > 0 else 0.0,
        "stopped_on_eos": stopped_on_eos,
        "sampling": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed},
        "multi_chip": True,
    }


def handle_generate_tp(state: MeshServerState, args: dict):
    """Multi-chip TP generate — streams by default (mirrors server.py UX).

    Greedy by default (TRACED forward, P14; warmup+capture on first call, then
    execute_trace). With temperature>0 it delegates to the non-traced sampling
    path (_generate_sampled_tp); the greedy traced path below is unchanged.
    """
    import ttnn
    import time as _time

    prompt = args.get("prompt")
    if not prompt:
        yield {"_final": True, "error": "missing required arg: prompt"}
        return
    max_tokens = int(args.get("max_tokens", 40))
    chunk_size = max(1, int(args.get("chunk_size", 1)))

    if state.tok is None:
        yield {"_final": True, "error": "tokenizer not loaded on mesh server"}
        return
    if not state.layers:
        yield {"_final": True, "error": "weights not loaded (server still bootstrapping?)"}
        return

    prompt_ids = state.tok.encode(prompt)
    cap = MAX_POS
    if len(prompt_ids) + max_tokens > cap:
        yield {"_final": True,
               "error": f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} > MAX_POS {cap}"}
        return

    # Sampling path (temperature>0): non-traced logits + host sampling. Greedy
    # (default temperature=0) falls through to the traced argmax path below.
    temperature = float(args.get("temperature", 0.0))
    if temperature > 0.0:
        yield from _generate_sampled_tp(
            state, prompt, prompt_ids, max_tokens, chunk_size, temperature,
            float(args.get("top_p", 1.0)), int(args.get("top_k", 0)), args.get("seed"))
        return

    # Ensure trace is captured (one-time, ~85ms + 2 warmup forwards)
    _ensure_decode_trace(state)

    # Reset per-layer state (SSM, conv_state, paged KV) — must run BEFORE
    # prefill because warmup or prior queries left state non-zero.
    _reset_state_buffers(state)

    # Prefill (use traced forward; the trace doesn't care about position values).
    # Variable name reflects P22 change: trace now emits on-device argmax tensor.
    t0 = _time.time()
    last_argmax = None
    for pos, tid in enumerate(prompt_ids):
        last_argmax = _traced_forward(state, tid, pos)
    ttnn.synchronize_device(state.mesh)
    prefill_ms = (_time.time() - t0) * 1000.0

    # Decode loop with chunked streaming
    generated_ids = []
    decode_times = []
    cur_pos = len(prompt_ids)
    eos_id = getattr(state.tok, "eos_token_id", None)
    text_so_far = ""
    pending = []
    stopped_on_eos = False

    for step in range(max_tokens):
        # P22: on-device argmax. last_argmax is UINT32 [1,1] replicated on
        # mesh post-AG, so ConcatMeshToTensor(dim=0) yields [NCHIPS, 1, 1]
        # with all chips agreeing. Read chip 0's value. Tiny readback (~8 bytes
        # of payload) vs prior 152064 fp32 readback (~600 KB, ~35 ms).
        idx_concat = ttnn.to_torch(
            last_argmax, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0))
        next_id = int(idx_concat.cpu().numpy().reshape(-1)[0])
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
        if eos_id is not None and next_id == eos_id:
            stopped_on_eos = True
            break
        td0 = _time.time()
        last_argmax = _traced_forward(state, next_id, cur_pos)
        ttnn.synchronize_device(state.mesh)
        decode_times.append((_time.time() - td0) * 1000.0)
        cur_pos += 1

    if pending:
        yield {
            "token_text": "".join(p["token_text"] for p in pending),
            "token_ids": [p["token_id"] for p in pending],
            "tok_idx_start": pending[0]["tok_idx"],
            "tok_idx_end": pending[-1]["tok_idx"],
        }

    total_ms = (_time.time() - t0) * 1000.0
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
        "multi_chip": True,
    }


def handle_cosine_ladder_tp(state: MeshServerState, args: dict) -> dict:
    """Teacher-forced per-position logits dump on qb2 multi-chip TP. Eager path
    (mirrors qb1 server.handle_cosine_ladder). Toggles
    state.deltanet_recurrence_mode for the run and restores it on exit, so the
    captured production trace is unaffected.

    Used to gate the custom GDN kernel's long-context coherence vs the manual
    TTNN broadcast-reduce recurrence (Tier 1 of the owned_gdn promotion gate;
    see research/owned_gdn_diagnosis_2026_05_18.md).

    args:
      prompt_ids:    list[int]   (required) — initial prompt token ids
      generated_ids: list[int]   (required) — teacher-forced continuation
                                  (typically from a prior generate_tp run)
      deltanet_recurrence_mode: str (default "manual") — "manual" /
                                  "owned_gdn" / "owned_gdn_inplace"
      out_path:      str         (default .cache/qb2_tp_deltanet/
                                  cosine_ladder_tp_logits.npz)

    Saves NPZ {logits[M, vocab] fp32, prompt_ids[P], generated_ids[M]}.
    Returns: {ok, path, deltanet_recurrence_mode, n_prompt, n_steps, vocab,
              prefill_ms, decode_ms, ms_per_step}.
    """
    import os as _os
    import time as _time
    import numpy as np
    import ttnn

    if state.mesh is None or not state.layers:
        return {"error": "mesh/weights not loaded"}

    prompt_ids = list(args.get("prompt_ids") or [])
    generated_ids = list(args.get("generated_ids") or [])
    if not prompt_ids:
        return {"error": "missing prompt_ids"}
    if not generated_ids:
        return {"error": "missing generated_ids"}

    mode = str(args.get("deltanet_recurrence_mode", "manual"))
    if mode not in ("manual", "owned_gdn", "owned_gdn_inplace"):
        return {"error": f"deltanet_recurrence_mode must be one of manual/owned_gdn/owned_gdn_inplace, got {mode}"}

    decay_gate_mode = str(args.get("deltanet_decay_gate_mode", "manual"))
    if decay_gate_mode not in ("manual", "owned_decay_gate"):
        return {"error": f"deltanet_decay_gate_mode must be one of manual/owned_decay_gate, got {decay_gate_mode}"}

    rope_mode = str(args.get("rope_mode", state.rope_mode))
    if rope_mode not in ("manual", "native_partial"):
        return {"error": f"rope_mode must be one of manual/native_partial, got {rope_mode}"}

    P = len(prompt_ids)
    M = len(generated_ids)
    if P + M > MAX_POS:
        return {"error": f"P {P} + M {M} > MAX_POS {MAX_POS}"}

    out_path = str(args.get("out_path") or
                   _os.path.join(CACHE_DIR, "qb2_tp_deltanet",
                                  "cosine_ladder_tp_logits.npz"))
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)

    VOCAB = state.vocab_size

    _reset_state_buffers(state)
    old_mode = state.deltanet_recurrence_mode
    old_decay_gate_mode = state.deltanet_decay_gate_mode
    old_rope_mode = state.rope_mode
    state.deltanet_recurrence_mode = mode
    state.deltanet_decay_gate_mode = decay_gate_mode
    state.rope_mode = rope_mode

    logits_arr = np.empty((M, VOCAB), dtype=np.float32)

    def _readback(t):
        # forward_token_tp_inner(return_logits=True) returns [1, VOCAB] replicated
        # across chips post-all_gather. Collect with ConcatMeshToTensor(dim=0) →
        # [NCHIPS, 1, VOCAB]; take chip 0.
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[0].float().cpu().numpy().reshape(VOCAB)

    try:
        t0 = _time.time()
        last_logits_tt = None
        for pos, tid in enumerate(prompt_ids):
            update_input_buffers(state, tid, pos)
            last_logits_tt = forward_token_tp_inner(state, return_logits=True)
        ttnn.synchronize_device(state.mesh)
        prefill_ms = (_time.time() - t0) * 1000.0

        # Step 0: prefill's last logits = prediction for generated_ids[0].
        logits_arr[0] = _readback(last_logits_tt)

        t_decode = _time.time()
        cur_pos = P
        for i in range(1, M):
            update_input_buffers(state, generated_ids[i - 1], cur_pos)
            last_logits_tt = forward_token_tp_inner(state, return_logits=True)
            ttnn.synchronize_device(state.mesh)
            logits_arr[i] = _readback(last_logits_tt)
            cur_pos += 1
        decode_ms = (_time.time() - t_decode) * 1000.0
    finally:
        state.deltanet_recurrence_mode = old_mode
        state.deltanet_decay_gate_mode = old_decay_gate_mode
        state.rope_mode = old_rope_mode

    np.savez(out_path,
             logits=logits_arr,
             prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
             generated_ids=np.asarray(generated_ids, dtype=np.int32))

    state.last_run = {
        "cmd": "cosine_ladder_tp",
        "deltanet_recurrence_mode": mode,
        "deltanet_decay_gate_mode": decay_gate_mode,
        "rope_mode": rope_mode,
        "n_prompt": P,
        "n_steps": M,
    }

    return {
        "ok": True,
        "path": out_path,
        "deltanet_recurrence_mode": mode,
        "deltanet_decay_gate_mode": decay_gate_mode,
        "rope_mode": rope_mode,
        "n_prompt": P,
        "n_steps": M,
        "vocab": VOCAB,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "ms_per_step": decode_ms / max(M - 1, 1),
    }





HANDLERS = {
    "status":                     handle_status,
    "generate_tp":                handle_generate_tp,
    "bench_decode_tp_components": handle_bench_decode_tp_components,
    "profile_decode_tp_ops":      handle_profile_decode_tp_ops,
    "cosine_ladder_tp":           handle_cosine_ladder_tp,
    "shutdown":                   handle_shutdown,
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
    # Stable readiness marker — `serve_tp.sh status` (or any wait-for-ready
    # loop) can grep $LOG_FILE for this exact line after the long bootstrap.
    print("[serve] READY", flush=True)

    shutdown_requested = False
    import types as _types
    import traceback
    while not shutdown_requested:
        try:
            conn, _ = srv.accept()
        except OSError:
            continue
        try:
            raw = P.read_line(conn)
            if not raw:
                conn.close()
                continue
            req = P.parse_request(raw)
            handler = HANDLERS.get(req.cmd)
            if handler is None:
                conn.sendall(P.pack_error(f"unknown cmd: {req.cmd}"))
                conn.close()
                continue
            try:
                result = handler(state, req.args)
                if isinstance(result, _types.GeneratorType):
                    for item in result:
                        if isinstance(item, dict) and item.pop("_final", False):
                            conn.sendall(P.pack_result(item))
                        else:
                            conn.sendall(P.pack_chunk(item))
                else:
                    conn.sendall(P.pack_result(result))
            except Exception as e:
                print(f"[serve_tp] handler error:\n{traceback.format_exc()}", flush=True)
                conn.sendall(P.pack_error(f"{type(e).__name__}: {e}"))
            if req.cmd == "shutdown":
                shutdown_requested = True
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
