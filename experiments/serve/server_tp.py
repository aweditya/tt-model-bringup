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
        # Persistent traced graph (Stage D — P14 commit 2d30af7)
        self.trace_id = None
        self.traced_logits_tt = None    # legacy field, retained for safety; unused after vocab-sharded LM head ship
        self.traced_argmax_tt = None    # P22: on-device argmax output of forward_token_tp_inner
        # P22: vocab-sharded LM head dimensions
        self.vocab_size = None          # real vocab from state.embed_np.shape[0] (152064 for Qwen3.6)
        self.vocab_padded = None        # padded vocab in lm_head weight (248320 for Qwen3.6)
        # Pre-allocated input buffers (populated at bootstrap; updated per step
        # via update_input_buffers using ttnn.copy_host_to_device_tensor)
        self.x_buf = None                # legacy: kept allocated but no longer written per step (P25)
        self.cur_pos_buf = None
        self.cos_buf = None              # legacy
        self.sin_buf = None              # legacy
        self.tok_buf = None              # P25: [1,1] uint32 — current token id, sole per-step host write
        self.rot_idxs_buf = None         # P25: [1,1] uint32 — current rotary index for cos/sin embedding lookup
        self.cos_all_np = None
        self.sin_all_np = None
        self.rotary_dim = None
        # Paged KV cache shared state
        self.page_table_tt = None
        self.paged_write_mem_cfg = None
        self.fused_paged_write_mem_cfg_k = None
        self.fused_paged_write_mem_cfg_v = None
        # Paged SDPA configs (P18/P19 — see feedback_p19_chained_paged_sdpa.md)
        self.paged_sdpa_progcfg = None
        self.sdpa_compute_kernel_config = None
        self.trace_x_buf = None
        self.trace_logits_buf = None
        self.last_run = None
        # Experiment guard. Production default stays on the validated two-call
        # paged_update_cache path; probe endpoints may flip this temporarily.
        self.use_fused_paged_update = False
        self.collective_mode = "baseline"


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
    _cap = os.environ.get('TP_MAX_LAYERS')
    if _cap:
        state.num_layers = min(state.num_layers, int(_cap))
        print(f"  TP_MAX_LAYERS={_cap} → capping num_layers to {state.num_layers}", flush=True)
    print(f"  ✓ cfg: {cfg}", flush=True)
    print(f"  ✓ num_layers: {state.num_layers}", flush=True)

    print(f"[bootstrap] loading tokenizer…", flush=True)
    state.tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  ✓ tokenizer", flush=True)

    # === Stage B: load + shard all layer weights ===
    print(f"[bootstrap] importing 91f kernels + TP relayout helpers…", flush=True)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "experiments", "utils"))
    spec = importlib.util.spec_from_file_location(
        "_91f", os.path.join(PROJECT_ROOT, "experiments", "91f_qwen36_27b_full_ondevice.py"))
    _91f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_91f)
    state._91f = _91f
    from full_layer_tp_probe import (
        relayout_in_proj, relayout_conv,
        N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, KERNEL, KEY_DIM, VAL_DIM, CONV_DIM,
        NCHIPS,
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
            dt_bias_tt = upload_sharded(w_np['dt_bias'], dim=0)
            A_log_tt = upload_sharded(w_np['A_log'], dim=0)
            w_out_tt = upload_sharded(w_np['out_proj'], dim=0)
            ssm_tt = upload_sharded(
                np.zeros((N_V_HEADS, K_DIM, V_DIM), dtype=np.float32), dim=0)
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
    print(f"[bootstrap] loading embed + lm_head + final_norm + RoPE tables…", flush=True)
    # Reuse 91l's loader (used by single-chip server too)
    spec2 = importlib.util.spec_from_file_location(
        "_91l", os.path.join(PROJECT_ROOT, "experiments", "91l_fp32_residual_generate.py"))
    _91l = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(_91l)
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
    # (91l_fp32_residual_generate.py:82) so dim=1 is the vocab dim. Per-chip slab
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

    # RoPE cos/sin tables — ROTARY_DIM-wide (V2 rotate-only path).
    # Keep host arrays in state for per-step slicing (trace-compatible: we
    # write the current row into state.cos_buf/sin_buf via copy_host_to_device).
    HEAD_DIM = cfg['head_dim']
    rotary_dim = int(HEAD_DIM * cfg['partial_rotary_factor'])
    half_rot = rotary_dim // 2
    freqs = 1.0 / (10_000_000.0 ** (np.arange(half_rot).astype(np.float32) / half_rot))
    positions = np.arange(MAX_POS).astype(np.float32)
    ang = positions[:, None] * freqs[None, :]
    state.cos_all_np = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1).astype(np.float32)
    state.sin_all_np = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1).astype(np.float32)
    state.rotary_dim = rotary_dim
    # Keep the device-resident extended table for the legacy eager path (slice
    # at runtime by Python int). The traced path uses state.cos_buf/sin_buf
    # populated outside the trace via copy_host_to_device.
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
    print(f"  ✓ paged_write mem_cfg cached", flush=True)
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
        print(f"  ✓ fused paged_write disjoint K/V mem_cfg cached", flush=True)
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
    print(f"  ✓ paged SDPA program_config + compute_kernel_config cached", flush=True)

    # Pre-allocated input buffers for trace-compatible decode.
    # These are READ by forward_token_tp_inner (which is the trace target).
    # They are UPDATED before each execute_trace via copy_host_to_device_tensor
    # — outside the captured region, so the writes don't violate trace semantics.
    HIDDEN_DIM = cfg['hidden']
    ROTARY_DIM = state.rotary_dim
    state.x_buf = ttnn.from_torch(
        torch.zeros(1, HIDDEN_DIM, dtype=torch.float32),
        dtype=ttnn.bfloat16, device=state.mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.cur_pos_buf = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.cos_buf = ttnn.from_torch(
        torch.zeros(1, ROTARY_DIM, dtype=torch.float32),
        dtype=ttnn.bfloat16, device=state.mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.sin_buf = ttnn.from_torch(
        torch.zeros(1, ROTARY_DIM, dtype=torch.float32),
        dtype=ttnn.bfloat16, device=state.mesh, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    # P25: tiny index buffers — the ONLY per-step HtoD writes after the swap.
    # tok_buf: [1,1] uint32, current token id (for ttnn.embedding token lookup).
    # rot_idxs_buf: [1,1] uint32, current rotary index (for cos/sin lookup).
    # Both shape [1,1] (not [1]) because ttnn.embedding wants idx ndim >= 2.
    state.tok_buf = ttnn.from_torch(
        torch.tensor([[0]], dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    state.rot_idxs_buf = ttnn.from_torch(
        torch.tensor([[0]], dtype=torch.int32),
        dtype=ttnn.uint32, device=state.mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh))
    print(f"  ✓ input buffers pre-allocated (x_buf, cur_pos_buf, cos_buf, sin_buf, "
          f"tok_buf, rot_idxs_buf)",
          flush=True)

    print(f"[bootstrap] STAGE B COMPLETE — all weights + state buffers on mesh.", flush=True)


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
    """Mesh RMS norm.

    P15 (2026-05-14): re-testing plain ttnn.rms_norm after the deallocate
    discipline (faff42a) + paged refactor (4cd0ce1) + buffer pre-allocation
    (1fabf07) all landed. The original wedge in feedback_p7_mlp_wedges_next_dn.md
    may have been caused by intermediate buffer accumulation (the H2 we ruled
    out at that time but the deallocate fix later confirmed was the real issue).
    If plain rms_norm now works, drop in: 7 manual ops → 1 fused op × 305 calls/
    token = ~18.3 ms/tok savings per Agent N's gap analysis (5a9808d).

    Falls back to manual form if plain rms_norm wedges multi-step P6 — see
    feedback_ttnn_fused_ops_gap_analysis.md.
    """
    import ttnn
    return ttnn.rms_norm(x_tt, weight=weight_tt, epsilon=eps)


def _tp_all_reduce(state: MeshServerState, partial):
    import ttnn
    if state.collective_mode == "explicit_all_reduce":
        return ttnn.all_reduce(
            partial,
            cluster_axis=1,
            memory_config=partial.memory_config(),
            num_links=1,
            topology=ttnn.Topology.Linear,
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
    """
    import ttnn
    import numpy as np
    from full_layer_tp_probe import (
        N_K_HEADS, N_V_HEADS, K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )

    HIDDEN = cfg['hidden']
    # 1. Pre-norm (manual: see _rms_norm_manual doc)
    h_tt = _rms_norm_manual(x_tt, dn['input_norm'], EPS, HIDDEN)
    # 2. in_proj (replicated x × sharded weight → per-chip slab)
    all_tt = ttnn.linear(h_tt, dn['w_in'])
    ttnn.deallocate(h_tt)
    # 3. slice per-chip [Q | K | V | Z | A | B]
    mixed_qkv = ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP])
    z_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP], [1, CONV_DIM_CHIP + VAL_DIM_CHIP])
    a_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP])
    b_tt = ttnn.slice(all_tt, [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
                      [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP])
    ttnn.deallocate(all_tt)
    # 4. conv1d on per-chip slab
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

    q = gqa(q_flat, NK_PER_CHIP, K_DIM)
    k = gqa(k_flat, NK_PER_CHIP, K_DIM)
    v = ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM])
    # 6. QK l2-norm (manual mesh-safe form — q_l2_scale=1/K_DIM, k_l2_scale=1/sqrt(K_DIM)
    # baked into the weights; eps=EPS/K_DIM matches the rms_norm semantics exactly).
    EPS_RMS = EPS / K_DIM
    q = _rms_norm_manual(q, dn['q_l2_scale'], EPS_RMS, K_DIM)
    k = _rms_norm_manual(k, dn['k_l2_scale'], EPS_RMS, K_DIM)
    # 7. gate/decay/beta on per-chip head subset
    softplus_a = ttnn.log(ttnn.add(ttnn.exp(ttnn.add(a_tt, dn['dt_bias'])), 1.0))
    g = ttnn.mul(ttnn.neg(ttnn.exp(dn['A_log'])), softplus_a)
    beta = ttnn.sigmoid(b_tt)
    decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
    # 8. Recurrence
    H_4d = ttnn.reshape(dn['ssm'], [1, NV_PER_CHIP, K_DIM, V_DIM])
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
    x_out = ttnn.add(x_tt, reduced)
    ttnn.deallocate(reduced)
    H_new_3d = ttnn.reshape(H_new, [NV_PER_CHIP, K_DIM, V_DIM])
    ttnn.copy(H_new_3d, dn['ssm'])
    ttnn.deallocate(H_new)
    ttnn.copy(conv_state_new, dn['conv_st'])
    ttnn.deallocate(conv_state_new)
    return x_out


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
    import torch
    import numpy as np
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
    def apply_rope(t, n_heads):
        rot = ttnn.slice(t, [0, 0], [n_heads, ROTARY_DIM])
        passthru = ttnn.slice(t, [0, ROTARY_DIM], [n_heads, HEAD_DIM])
        x1 = ttnn.slice(rot, [0, 0], [n_heads, half])
        x2 = ttnn.slice(rot, [0, half], [n_heads, ROTARY_DIM])
        neg_x2 = ttnn.neg(x2)
        rotated = ttnn.add(ttnn.mul(rot, cos_tt),
                            ttnn.mul(ttnn.concat([neg_x2, x1], dim=-1), sin_tt))
        return ttnn.concat([rotated, passthru], dim=-1)
    q_tt = apply_rope(q_tt, NQ_PER_CHIP)
    k_tt = apply_rope(k_tt, NKV_PER_CHIP)
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


def forward_token_tp_inner(state):
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
    for layer in state.layers:
        if layer['type'] == 'linear_attention':
            x_tt = deltanet_step_tp(state, x_tt, layer['dn'], cfg)
        else:
            x_tt = gated_attn_step_tp(state, x_tt, layer['attn'],
                                       state.cur_pos_buf,
                                       0,  # vestigial cur_pos int (paged path ignores it)
                                       cos_row_tt, sin_row_tt, cfg)
        x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
    x_tt = _rms_norm_manual(x_tt, state.final_norm_tt, 1e-6, HIDDEN)
    # P22 vocab-sharded LM head + on-device argmax (see Agent X's resolution at
    # feedback_lm_head_argmax_unknown.md). Per-chip linear produces
    # [1, VOCAB_PADDED/NCHIPS] then all_gather replicates to [1, VOCAB_PADDED]
    # on every chip. Slice to real vocab, untilize for argmax compatibility,
    # then argmax. Result is small UINT32 tensor — tiny readback.
    # Use keepdim=True + use_multicore=True (the only combo that returns
    # correct indices on [1, 152064] in our ttnn build — see p22_argmax_sanity6).
    sharded_logits_tt = ttnn.linear(x_tt, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, state.vocab_size])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    argmax_tt = ttnn.argmax(rm_logits_tt, dim=-1, keepdim=True, use_multicore=True)
    return argmax_tt


def forward_token_tp(state, token_id, cur_pos):
    """Eager wrapper: update buffers then run inner forward.
    For traced decode, use update_input_buffers + execute_trace directly.
    """
    update_input_buffers(state, token_id, cur_pos)
    return forward_token_tp_inner(state)


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
    import ttnn, torch, numpy as np
    from full_layer_tp_probe import N_V_HEADS, K_DIM, V_DIM, CONV_DIM
    cfg = state.cfg
    mesh = state.mesh

    ssm_host = ttnn.from_torch(
        torch.zeros(N_V_HEADS, K_DIM, V_DIM, dtype=torch.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    conv_host = ttnn.from_torch(
        torch.zeros(CONV_DIM, cfg['conv_kernel'] - 1, dtype=torch.float32),
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
    print(f"[trace] warmup + capture decode trace…", flush=True)
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


def _forward_argmax_id_with_cache_writer(state: MeshServerState, token_id: int,
                                         cur_pos: int, use_fused: bool,
                                         collective_mode: str | None = None) -> int:
    import ttnn
    old_fused = state.use_fused_paged_update
    old_collective = state.collective_mode
    state.use_fused_paged_update = use_fused
    if collective_mode is not None:
        state.collective_mode = collective_mode
    try:
        _reset_state_buffers(state)
        update_input_buffers(state, token_id, cur_pos)
        out = forward_token_tp_inner(state)
        ttnn.synchronize_device(state.mesh)
        return _read_argmax_id(state, out)
    finally:
        state.use_fused_paged_update = old_fused
        state.collective_mode = old_collective
        _reset_state_buffers(state)


def _capture_temp_decode_trace(state: MeshServerState, use_fused: bool,
                               collective_mode: str | None = None):
    """Capture an alternate decode trace for a probe, without replacing prod."""
    import ttnn
    old_fused = state.use_fused_paged_update
    old_collective = state.collective_mode
    state.use_fused_paged_update = use_fused
    if collective_mode is not None:
        state.collective_mode = collective_mode
    trace_id = None
    argmax_tt = None
    try:
        _reset_state_buffers(state)
        update_input_buffers(state, token_id=0, cur_pos=0)
        _ = forward_token_tp_inner(state)
        ttnn.synchronize_device(state.mesh)
        update_input_buffers(state, token_id=0, cur_pos=1)
        _ = forward_token_tp_inner(state)
        ttnn.synchronize_device(state.mesh)
        update_input_buffers(state, token_id=0, cur_pos=2)
        trace_id = ttnn.begin_trace_capture(state.mesh, cq_id=0)
        argmax_tt = forward_token_tp_inner(state)
        ttnn.end_trace_capture(state.mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(state.mesh)
        return trace_id, argmax_tt
    except Exception:
        if trace_id is not None:
            try:
                ttnn.release_trace(state.mesh, trace_id)
            except Exception:
                pass
        raise
    finally:
        state.use_fused_paged_update = old_fused
        state.collective_mode = old_collective
        _reset_state_buffers(state)


def handle_probe_fused_paged_update_cache_tp(state: MeshServerState, args: dict) -> dict:
    """Validate the fused K/V paged cache writer in the resident qb2 server.

    Hypothesis: the two K/V paged_update_cache dispatches in each full-attention
    layer can be replaced by one paged_fused_update_cache dispatch. This endpoint
    only reports measured compatibility/correctness/timing. Production remains
    on the validated two-call path unless a later patch changes the default.
    """
    import ttnn
    import time as _time

    prompt = args.get("prompt", "The capital of France is")
    iters = int(args.get("iters", 10))
    warmup = int(args.get("warmup", 2))
    bench_trace = bool(args.get("bench_trace", True))
    allow_wedge_prone_disjoint = bool(args.get("allow_wedge_prone_disjoint", False))
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if state.tok is None or not state.layers:
        return {"error": "server not fully loaded"}

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    token_id = int(prompt_ids[0])

    if not allow_wedge_prone_disjoint:
        return {
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "token_id": token_id,
            "compatibility": {
                "accepted": False,
                "phase": "disjoint_variant_disabled",
                "error": (
                    "The original same-core fused writer was rejected because "
                    "K/V input tensors overlapped. The disjoint-core variant "
                    "then wedged qb2 for >10 minutes and required SIGTERM. "
                    "Pass allow_wedge_prone_disjoint=true only for a "
                    "coordinated isolation run."
                ),
            },
            "trace_bench": None,
        }

    # First validate the exact production forward with the writer swapped.
    try:
        baseline_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False)
    except Exception as e:
        return {
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "token_id": token_id,
            "compatibility": {
                "accepted": False,
                "phase": "baseline_forward",
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_bench": None,
        }
    try:
        fused_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=True)
    except Exception as e:
        state.use_fused_paged_update = False
        _reset_state_buffers(state)
        result = {
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "token_id": token_id,
            "compatibility": {
                "accepted": False,
                "phase": "fused_forward",
                "baseline_argmax_id": baseline_id,
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_bench": None,
        }
        state.last_run = {
            "cmd": "probe_fused_paged_update_cache_tp",
            "accepted": False,
            "phase": "fused_forward",
        }
        return result
    argmax_match = (baseline_id == fused_id)

    result = {
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "token_id": token_id,
        "compatibility": {
            "accepted": True,
            "baseline_argmax_id": baseline_id,
            "fused_argmax_id": fused_id,
            "argmax_match": argmax_match,
            "note": (
                "Argmax equality is a narrow correctness check for the same "
                "token/position after resetting mutable state; it is not a "
                "full quality validation."
            ),
        },
        "trace_bench": None,
    }

    if not bench_trace:
        state.last_run = {
            "cmd": "probe_fused_paged_update_cache_tp",
            "accepted": True,
            "argmax_match": argmax_match,
            "bench_trace": False,
        }
        return result

    trace_id = None
    try:
        t_capture0 = _time.perf_counter()
        trace_id, fused_argmax_tt = _capture_temp_decode_trace(state, use_fused=True)
        capture_ms = (_time.perf_counter() - t_capture0) * 1000.0

        def sync():
            ttnn.synchronize_device(state.mesh)

        def timed(fn):
            sync()
            t0 = _time.perf_counter()
            fn()
            sync()
            return (_time.perf_counter() - t0) * 1000.0

        for i in range(warmup):
            tid = prompt_ids[i % len(prompt_ids)]
            pos = i % MAX_POS
            update_input_buffers(state, tid, pos)
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
        sync()

        execute_ms = []
        update_execute_ms = []
        for i in range(iters):
            tid = prompt_ids[i % len(prompt_ids)]
            pos = i % MAX_POS
            execute_ms.append(timed(
                lambda: ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)))

            tid2 = prompt_ids[(i + 1) % len(prompt_ids)]
            pos2 = (pos + 1) % MAX_POS
            def update_execute(tid2=tid2, pos2=pos2):
                update_input_buffers(state, tid2, pos2)
                ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
            update_execute_ms.append(timed(update_execute))

        traced_id = _read_argmax_id(state, fused_argmax_tt)
        result["trace_bench"] = {
            "iters": iters,
            "warmup": warmup,
            "capture_ms": capture_ms,
            "summary_ms": {
                "fused_execute_trace": _summary_ms(execute_ms),
                "fused_update_plus_execute": _summary_ms(update_execute_ms),
            },
            "samples_ms": {
                "fused_execute_trace": execute_ms,
                "fused_update_plus_execute": update_execute_ms,
            },
            "traced_argmax_id_sample": traced_id,
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(state.mesh, trace_id)
        state.use_fused_paged_update = False
        _reset_state_buffers(state)

    state.last_run = {
        "cmd": "probe_fused_paged_update_cache_tp",
        "accepted": True,
        "argmax_match": argmax_match,
        "median_fused_execute_ms": (
            result["trace_bench"]["summary_ms"]["fused_execute_trace"].get("median")
            if result["trace_bench"] else None
        ),
        "median_fused_combined_ms": (
            result["trace_bench"]["summary_ms"]["fused_update_plus_execute"].get("median")
            if result["trace_bench"] else None
        ),
    }
    return result


def handle_probe_explicit_all_reduce_tp(state: MeshServerState, args: dict) -> dict:
    """Validate explicit axis/topology kwargs for row-parallel TP exits."""
    import ttnn
    import time as _time

    prompt = args.get("prompt", "The capital of France is")
    iters = int(args.get("iters", 10))
    warmup = int(args.get("warmup", 2))
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if state.tok is None or not state.layers:
        return {"error": "server not fully loaded"}

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    token_id = int(prompt_ids[0])

    baseline_id = _forward_argmax_id_with_cache_writer(
        state, token_id=token_id, cur_pos=0, use_fused=False,
        collective_mode="baseline")
    explicit_id = _forward_argmax_id_with_cache_writer(
        state, token_id=token_id, cur_pos=0, use_fused=False,
        collective_mode="explicit_all_reduce")
    argmax_match = (baseline_id == explicit_id)

    trace_id = None
    try:
        t_capture0 = _time.perf_counter()
        trace_id, explicit_argmax_tt = _capture_temp_decode_trace(
            state, use_fused=False, collective_mode="explicit_all_reduce")
        capture_ms = (_time.perf_counter() - t_capture0) * 1000.0

        def sync():
            ttnn.synchronize_device(state.mesh)

        def timed(fn):
            sync()
            t0 = _time.perf_counter()
            fn()
            sync()
            return (_time.perf_counter() - t0) * 1000.0

        for i in range(warmup):
            tid = prompt_ids[i % len(prompt_ids)]
            pos = i % MAX_POS
            update_input_buffers(state, tid, pos)
            ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
        sync()

        execute_ms = []
        update_execute_ms = []
        for i in range(iters):
            tid = prompt_ids[i % len(prompt_ids)]
            pos = i % MAX_POS
            execute_ms.append(timed(
                lambda: ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)))

            tid2 = prompt_ids[(i + 1) % len(prompt_ids)]
            pos2 = (pos + 1) % MAX_POS
            def update_execute(tid2=tid2, pos2=pos2):
                update_input_buffers(state, tid2, pos2)
                ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)
            update_execute_ms.append(timed(update_execute))

        result = {
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "token_id": token_id,
            "compatibility": {
                "accepted": True,
                "baseline_argmax_id": baseline_id,
                "explicit_argmax_id": explicit_id,
                "argmax_match": argmax_match,
                "mode": "ttnn.all_reduce(cluster_axis=1, topology=Linear, num_links=1)",
            },
            "trace_bench": {
                "iters": iters,
                "warmup": warmup,
                "capture_ms": capture_ms,
                "summary_ms": {
                    "explicit_execute_trace": _summary_ms(execute_ms),
                    "explicit_update_plus_execute": _summary_ms(update_execute_ms),
                },
                "samples_ms": {
                    "explicit_execute_trace": execute_ms,
                    "explicit_update_plus_execute": update_execute_ms,
                },
                "traced_argmax_id_sample": _read_argmax_id(state, explicit_argmax_tt),
            },
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(state.mesh, trace_id)
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False
        _reset_state_buffers(state)

    state.last_run = {
        "cmd": "probe_explicit_all_reduce_tp",
        "argmax_match": argmax_match,
        "median_explicit_execute_ms": (
            result["trace_bench"]["summary_ms"]["explicit_execute_trace"].get("median")
        ),
        "median_explicit_combined_ms": (
            result["trace_bench"]["summary_ms"]["explicit_update_plus_execute"].get("median")
        ),
    }
    return result


def handle_generate_tp(state: MeshServerState, args: dict):
    """Multi-chip TP generate — streams by default (mirrors server.py UX).

    Now uses TRACED forward (P14 unblocked). On first call: warmup + capture.
    Subsequent calls reuse the trace via execute_trace.
    """
    import numpy as np
    import torch
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


HANDLERS = {
    "status":         handle_status,
    "generate_tp":    handle_generate_tp,
    "bench_decode_tp_components": handle_bench_decode_tp_components,
    "probe_fused_paged_update_cache_tp": handle_probe_fused_paged_update_cache_tp,
    "probe_explicit_all_reduce_tp": handle_probe_explicit_all_reduce_tp,
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
