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
import contextlib

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
MAX_POS = 512  # 2026-05-18: bumped from 256 to clear the qb1 single-chip
               # 500-position long-context bar (cf. feedback_fp32_sdpa_cliff_probe.md,
               # feedback_needle_haystack_qb1.md) for the owned_gdn promotion gate.
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
        # 2026-05-19: defaulted to explicit_all_reduce after P1 probe
        # (probe_ccl_components_tp) showed num_links=2 is 11.2% faster than
        # num_links=1 at production [1, 5120] bf16 shape; the bare
        # `ttnn.all_reduce(partial)` path uses unknown defaults.
        self.collective_mode = "explicit_all_reduce"
        # B.2.2 workaround: when True, _tp_all_reduce uses composite
        # reduce_scatter + all_gather instead of all_reduce. Different code
        # path → different output tensor lineage → may avoid the wedge in
        # downstream slice + DN. v3 prefill toggles this on per-call.
        # (CONFIRMED to also wedge — same underlying kernels as all_reduce.)
        self.force_composite_ccl = False
        # B.2.2 workaround #2: when True, _tp_all_reduce uses all_gather +
        # ttnn.sum instead of all_reduce. Different kernel set (no
        # reduce_scatter), different semaphore lifecycle. Mirrors the
        # ttnn.all_reduce fallback path for edge cases.
        self.force_custom_allreduce = False
        self.rope_mode = "manual"
        self.deltanet_decay_mode = "manual"
        # 2026-05-18: defaulted to "owned_gdn" after Tier 3 long-context gate
        # passed at 500 positions (commit 040e2ac, research/owned_gdn_
        # diagnosis_2026_05_18.md). Probe endpoints still toggle this
        # explicitly. Set to "manual" to revert to the legacy TTNN
        # broadcast-reduce recurrence.
        self.deltanet_recurrence_mode = "owned_gdn"
        # 2026-05-18 (later): owned conv1d kernel gate work.
        # "manual" = production manual concat/mul/sum/silu/slice chain.
        # "owned_conv1d" = ttnn.experimental.qwen36_conv1d_decode_owned
        #   (per-step slices conv_st/w_conv into single-column views,
        #    restitches state_next; slower than manual but correctness-
        #    isolated for G3 cosine_ladder_tp validation).
        # G4 production flip will pre-split weights/state at bootstrap to
        # eliminate the per-step slice/restitch overhead.
        self.deltanet_conv1d_mode = "manual"
        # 2026-05-19: owned decay/gate kernel gate work.
        # "manual" = production manual log(exp+1) + neg-exp + sigmoid chain.
        # "owned_decay_gate" = ttnn.experimental.qwen36_decay_gate_decode_owned
        #   (reshapes dn['dt_bias']/dn['A_log'] rank-1 → rank-2 per step,
        #    calls kernel, reshapes decay output back to [1, NV, 1, 1] for
        #    downstream recurrence). G2 correctness gate; G4 default flip
        #    will move the reshape to bootstrap for zero per-step overhead.
        # G4 default flip (decay/gate G3 PASS: 6/500 = 1.2% top-1 disag at
        # 500-token cosine ladder; med_cos 0.9988). Fused kernel ships as
        # production default.
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
    """All-reduce a row-parallel partial across the (1, 4) mesh.

    DIAG (2026-05-20): when state.ccl_debug is True, prints partial shape +
    memory_config + per-chip mean and chip0[0,0]. Use to compare prefill-path
    vs decode-path partial values when CCL equivalence is known correct but
    downstream cos != 0.999.

    NOTE (B.2.2 wedge fix 2026-05-19): ttnn.all_reduce's output tensor can
    enter a `DeallocatedTombStone` storage state because composite all_reduce
    deallocates intermediates during execution, leaving shared ownership in
    a zombie state. Decode paths never slice from all_reduce output so it
    doesn't trigger; but prefill's per-position SLICE → deltanet_step_tp
    chain hits Tensor::buffer()→DeviceStorage::get_mesh_buffer() validation
    that silently hangs (99% CPU, no error) on the tombstone state.

    Probe `probe_dn_source_isolation_tp` (commit 3822293) confirmed: linear /
    embed / from_torch / slice_write outputs all work as DN input; only
    all_reduce output wedges DN. Fix: ttnn.clone the result to force a fresh
    Allocated storage, severing the tombstone link. Cost: one memory copy
    per all_reduce call (~10KB at [1, HIDDEN], negligible; ~50KB at
    [5, HIDDEN]).
    """
    import ttnn

    if getattr(state, 'ccl_debug', False):
        _limit = getattr(state, 'ccl_debug_limit', 8)
        _count = getattr(state, '_ccl_debug_count', 0)
        if _count < _limit:
            try:
                _full = ttnn.to_torch(
                    partial, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                ).float()
                _H = _full.shape[-1] // 4
                _per_chip_means = [round(float(_full[..., c*_H:(c+1)*_H].mean()), 6) for c in range(4)]
                _per_chip_v0 = [round(float(_full[..., 0, c*_H]), 6) for c in range(4)]
                _tag = getattr(state, 'ccl_debug_tag', 'ar')
                print(f"[{_tag} call={_count}] shape={list(partial.shape)} "
                      f"mem={partial.memory_config().memory_layout} "
                      f"chip_means={_per_chip_means} "
                      f"chip_v0={_per_chip_v0}", flush=True)
            except Exception as _e:
                print(f"[ar diag error] {_e!r}", flush=True)
            state._ccl_debug_count = _count + 1

    # Local helper to print OUTPUT after each path's all_reduce.
    def _diag_output(_result, _path_name: str):
        if getattr(state, 'ccl_debug', False):
            _olim = getattr(state, 'ccl_debug_limit', 8)
            _oc = getattr(state, '_ccl_out_count', 0)
            if _oc < _olim:
                try:
                    _of = ttnn.to_torch(
                        _result, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                    ).float()
                    _OH = _of.shape[-1] // 4
                    _ocv = [round(float(_of[..., 0, c*_OH]), 6) for c in range(4)]
                    _ocm = [round(float(_of[..., 0, c*_OH:(c+1)*_OH].mean()), 6) for c in range(4)]
                    _ocn = [round(float(_of[..., 0, c*_OH:(c+1)*_OH].norm()), 4) for c in range(4)]
                    _otag = getattr(state, 'ccl_debug_tag', 'ar')
                    print(f"[{_otag} OUT call={_oc} via={_path_name}] shape={list(_result.shape)} "
                          f"chip_v0={_ocv} chip_means={_ocm} chip_norms={_ocn}", flush=True)
                except Exception as _oe:
                    print(f"[ar out diag err] {_oe!r}", flush=True)
                state._ccl_out_count = _oc + 1
        return _result

    # B.2.2 workaround: composite reduce_scatter + all_gather (instead of
    # all_reduce) — different op chain at API surface, but per the
    # all_reduce kernel audit it uses the SAME underlying kernels as
    # all_reduce's internal implementation. Confirmed to also wedge.
    if getattr(state, 'force_composite_ccl', False):
        scattered = ttnn.reduce_scatter(
            partial, dim=1, cluster_axis=1,
            num_links=2, topology=ttnn.Topology.Linear,
        )
        gathered = ttnn.all_gather(
            scattered, dim=1, cluster_axis=1,
            num_links=2, topology=ttnn.Topology.Linear,
        )
        ttnn.deallocate(scattered)
        return _diag_output(gathered, "composite")

    # B.2.2 workaround #2: CUSTOM all_reduce via all_gather + local sum.
    # This is what ttnn.all_reduce's fallback path uses for edge cases
    # (per audit: all_reduce_async.cpp:203-238) but avoiding reduce_scatter
    # entirely. all_gather is a single collective; sum is pure compute (no
    # fabric). Different kernel set, different semaphore lifecycle.
    if getattr(state, 'force_custom_allreduce', False):
        seq_len_local = partial.shape[-2]
        hidden_local = partial.shape[-1]
        # all_gather along dim=1 with cluster_axis=1 → [seq_len, NCHIPS*HIDDEN]
        gathered = ttnn.all_gather(
            partial, dim=1, cluster_axis=1,
            num_links=2, topology=ttnn.Topology.Linear,
        )
        # Reshape [seq_len, 4*HIDDEN] → [seq_len, 4, HIDDEN]
        reshaped = ttnn.reshape(gathered, [seq_len_local, 4, hidden_local])
        # Sum across the chip axis → [seq_len, HIDDEN]
        summed = ttnn.sum(reshaped, dim=1)
        ttnn.deallocate(gathered)
        ttnn.deallocate(reshaped)
        return _diag_output(summed, "custom_AG+sum")

    if state.collective_mode == "explicit_all_reduce":
        return _diag_output(ttnn.all_reduce(
            partial,
            cluster_axis=1,
            memory_config=partial.memory_config(),
            # P1 SHIPPED 2026-05-19: num_links=2 measured 11.2% faster than
            # num_links=1 at [1, 5120] bf16 (probe_ccl_components_tp). BH
            # P150x4 supports 2 eth links per axis.
            num_links=2,
            topology=ttnn.Topology.Linear,
        ), "explicit_all_reduce")
    try:
        return _diag_output(ttnn.all_reduce(partial), "default_all_reduce")
    except Exception:
        scattered = ttnn.reduce_scatter(partial, dim=1)
        return _diag_output(ttnn.all_gather(scattered, dim=1), "fallback_RS+AG")


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
    with _profile_scope(state, "deltanet_conv"):
        if state.deltanet_conv1d_mode == "owned_conv1d":
            # Owned conv1d kernel — G4 PRE-SPLIT design (after the G3 wire-in
            # bug investigation in df1cccc / e168c4d showed per-step slice+
            # concat+copy_back was broken at multi-step state persistence).
            #
            # dn['conv_st_split'] holds 3 single-column state tensors (one per
            # historical tap). dn['w_conv_split'] holds 4 single-column weight
            # tensors. Both pre-allocated at Stage B bootstrap with data
            # in tile-column 0 (the only position the kernel reads).
            #
            # The kernel mutates state0/1/2 IN PLACE via its writer (state0 ←
            # state1, state1 ← state2, state2 ← mixed). No concat. No copy
            # back to dn['conv_st']. Next forward call reads the already-
            # shifted split tensors directly.
            state0, state1, state2 = dn['conv_st_split']
            w0, w1, w2, w3 = dn['w_conv_split']
            mixed_col = ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1])
            ttnn.deallocate(mixed_qkv)
            _, _, _, conv_out_2d = ttnn.experimental.qwen36_conv1d_decode_owned(
                mixed_col, state0, state1, state2, w0, w1, w2, w3)
            ttnn.deallocate(mixed_col)
            conv_state_new = None  # owned path mutates split tensors in place
            conv_out = ttnn.reshape(conv_out_2d, [CONV_DIM_CHIP])
            ttnn.deallocate(conv_out_2d)
        else:
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
        if state.deltanet_decay_gate_mode == "owned_decay_gate":
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
    x_out = ttnn.add(x_tt, reduced)
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
    # B.2.2 Test 10: targeted residual-add diag — fires once per probe
    _resid_dbg = getattr(state, 'debug_mlp_resid', False)
    _resid_count = getattr(state, '_mlp_resid_count', 0)
    if _resid_dbg and _resid_count < 2:
        try:
            _xt = ttnn.to_torch(
                x_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
            ).float()
            _rd = ttnn.to_torch(
                reduced, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
            ).float()
            _tg = getattr(state, '_debug_state_tag', 'dec')
            print(f"  [{_tg} MLP RESID #{_resid_count}] x_tt.shape={list(x_tt.shape)} x_tt[0,0]={float(_xt[..., 0, 0]):.6f} x_tt.mc={x_tt.memory_config().memory_layout} "
                  f"| reduced.shape={list(reduced.shape)} reduced[0,0]={float(_rd[..., 0, 0]):.6f} reduced.mc={reduced.memory_config().memory_layout}",
                  flush=True)
        except Exception as _re:
            print(f"  [MLP resid diag pre err] {_re!r}", flush=True)
    out = ttnn.add(x_tt, reduced)
    if _resid_dbg and _resid_count < 2:
        try:
            _ot = ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
            ).float()
            _tg = getattr(state, '_debug_state_tag', 'dec')
            print(f"  [{_tg} MLP RESID #{_resid_count}] out.shape={list(out.shape)} out[0,0]={float(_ot[..., 0, 0]):.6f} out.mc={out.memory_config().memory_layout} "
                  f"(expected x_tt[0,0]+reduced[0,0])",
                  flush=True)
            state._mlp_resid_count = _resid_count + 1
        except Exception as _re2:
            print(f"  [MLP resid diag post err] {_re2!r}", flush=True)
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
        # B.2.2 Test 8: print x BEFORE MLP at L0 (single-shot, pos 0)
        if _li == 0 and getattr(state, 'debug_layer_boundary', False):
            _bm = getattr(state, '_pre_mlp_count', 0)
            if _bm < 1:
                try:
                    _xpf = ttnn.to_torch(
                        x_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                    ).float()
                    _Hpm = _xpf.shape[-1] // 4
                    _pv = [round(float(_xpf[..., 0, c*_Hpm]), 6) for c in range(4)]
                    _pm = [round(float(_xpf[..., 0, c*_Hpm:(c+1)*_Hpm].mean()), 6) for c in range(4)]
                    _pn = [round(float(_xpf[..., 0, c*_Hpm:(c+1)*_Hpm].norm()), 4) for c in range(4)]
                    _ptg = getattr(state, '_debug_state_tag', 'dec')
                    print(f"  [{_ptg} L0 pre-MLP x[0,:]] chip_v0={_pv} chip_means={_pm} chip_norms={_pn}",
                          flush=True)
                    state._pre_mlp_count = _bm + 1
                except Exception as _e:
                    print(f"  [decode L0 pre-MLP diag err] {_e!r}", flush=True)
        x_tt = mlp_step_tp(state, x_tt, layer['mlp'])
        # B.2.2 Test 7: print x at layer-1 entry on first call only
        if _li == 0 and getattr(state, 'debug_layer_boundary', False):
            _lcount = getattr(state, '_layer_bd_count', 0)
            if _lcount < 1:
                try:
                    _xf = ttnn.to_torch(
                        x_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                    ).float()
                    _Hcc = _xf.shape[-1] // 4
                    _cv = [round(float(_xf[..., 0, c*_Hcc]), 6) for c in range(4)]
                    _cm = [round(float(_xf[..., 0, c*_Hcc:(c+1)*_Hcc].mean()), 6) for c in range(4)]
                    _cn = [round(float(_xf[..., 0, c*_Hcc:(c+1)*_Hcc].norm()), 4) for c in range(4)]
                    _tg = getattr(state, '_debug_state_tag', 'dec')
                    print(f"  [{_tg} L1 entry x_after_L0[0,:]] chip_v0={_cv} chip_means={_cm} chip_norms={_cn}",
                          flush=True)
                    state._layer_bd_count = _lcount + 1
                except Exception as _e:
                    print(f"  [decode L1 entry diag err] {_e!r}", flush=True)
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


def forward_prefill_tp_inner_v3_parallel_attn(state, prompt_ids, capture_logits=False):
    """B.2.2: parallel attention + batched MLP + sequential DeltaNet with slice_write.

    Builds on validated primitives:
      - Batched ttnn.embedding for input + RoPE table lookup (B.2.1.5a)
      - Slice from directly-constructed multi-row TILE_LAYOUT works (B.2.1.5a)
      - slice_write + to_layout(TILE) round-trip (B.2.1.5b)

    Structure:
      Step 1: Batched embed → x_seq_tt [seq_len, HIDDEN] TILE_LAYOUT
      Step 2: Batched RoPE cos/sin lookup → cos_seq_tt, sin_seq_tt
      Step 3: For each layer:
        - DeltaNet (linear_attention): sequential per-position; slice from
          x_seq, run decode-step, slice_write per-position output into a
          ROW_MAJOR working buffer; convert to TILE.
        - Gated Attention: gated_attn_step_prefill_tp (parallel SDPA)
        - MLP: existing mlp_step_tp (broadcasts on leading dim)
      Step 4: Final norm + LM head (slice last position for production;
        batched LM head for capture_logits validation).
    """
    import ttnn
    import torch
    import numpy as np

    cfg = state.cfg
    HIDDEN = cfg['hidden']
    ROTARY_DIM = state.rotary_dim
    seq_len = len(prompt_ids)
    VOCAB = state.vocab_size

    if seq_len < 1:
        raise ValueError(f"prompt_ids must have len >= 1, got {seq_len}")
    if seq_len > MAX_POS:
        raise ValueError(f"prompt_ids len {seq_len} > MAX_POS {MAX_POS}")

    # ====== Step 1: Batched embed → x_seq_tt [seq_len, HIDDEN] ======
    prompt_ids_idx = ttnn.from_torch(
        torch.tensor([[int(t)] for t in prompt_ids], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    embed_raw = ttnn.embedding(
        prompt_ids_idx, state.embed_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_seq = ttnn.reshape(embed_raw, [seq_len, HIDDEN])
    ttnn.deallocate(prompt_ids_idx)
    ttnn.deallocate(embed_raw)

    # ====== Step 2: Batched RoPE cos/sin lookup ======
    positions_idx = ttnn.from_torch(
        torch.tensor([[t] for t in range(seq_len)], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    cos_seq_raw = ttnn.embedding(
        positions_idx, state.cos_table_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_seq_raw = ttnn.embedding(
        positions_idx, state.sin_table_tt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_seq_tt = ttnn.reshape(cos_seq_raw, [seq_len, ROTARY_DIM])
    sin_seq_tt = ttnn.reshape(sin_seq_raw, [seq_len, ROTARY_DIM])
    ttnn.deallocate(positions_idx)
    ttnn.deallocate(cos_seq_raw)
    ttnn.deallocate(sin_seq_raw)

    # Reset the per-call attention debug counter so first call this forward
    # gets the prints (helps debugging fresh calls).
    global _PREFILL_DEBUG_LAYER_IDX
    _PREFILL_DEBUG_LAYER_IDX = 0

    import time as _time
    _last_t = _time.time()
    def _layer_dbg(idx, layer_type, stage="end"):
        nonlocal _last_t
        ttnn.synchronize_device(state.mesh)
        now = _time.time()
        dt = now - _last_t
        _last_t = now
        # Print every layer for B.2.2 debug
        print(f"  [v3 prefill] layer {idx:2d} ({layer_type[:14]:14s}) {stage} dt={dt*1000:5.0f}ms", flush=True)

    # B.2.2 fix attempt 10: CUSTOM all_reduce via all_gather + ttnn.sum.
    # Composite path (fix 9, ttnn.reduce_scatter + ttnn.all_gather) ALSO
    # wedged — per audit, it uses same kernels as all_reduce internally.
    # This time use all_gather (no reduce_scatter) + compute-only sum.
    # Different kernel set entirely, different semaphore lifecycle.
    state.force_custom_allreduce = True

    # ====== Step 3: Layer loop ======
    for layer_idx, layer in enumerate(state.layers):
        # Layer-entry x_seq metadata (B.2.2 debug)
        try:
            print(f"  [v3 layer {layer_idx:2d} entry] x_seq.shape={list(x_seq.shape)} "
                  f"layout={x_seq.layout} mem={x_seq.memory_config()}", flush=True)
        except Exception as e:
            print(f"  [v3 layer {layer_idx:2d} entry] metadata read failed: {e}", flush=True)

        # B.2.2 SLICE-CORRECTNESS DIAG (2026-05-20): at layer 1 entry (and 0 for
        # control), dump x_seq row 0 contents per chip. Should be replicated.
        # Compare layer 0 entry (embedding lineage, slice known to work) vs
        # layer 1 entry (MLP lineage, slice suspected wrong).
        if getattr(state, 'ccl_debug', False) and layer_idx <= 1:
            try:
                _full = ttnn.to_torch(
                    x_seq, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                ).float()
                _Hc = _full.shape[-1] // 4
                _chip_v = [round(float(_full[0, c*_Hc]), 6) for c in range(4)]
                _chip_m = [round(float(_full[0, c*_Hc:(c+1)*_Hc].mean()), 6) for c in range(4)]
                _chip_n = [round(float(_full[0, c*_Hc:(c+1)*_Hc].norm()), 4) for c in range(4)]
                print(f"  [v3 L{layer_idx} x_seq[0,:]] chip_v0={_chip_v} chip_means={_chip_m} chip_norms={_chip_n}",
                      flush=True)
            except Exception as e:
                print(f"  [v3 L{layer_idx} x_seq diag err] {e!r}", flush=True)
        _layer_dbg(layer_idx, layer['type'], stage="start")
        if layer['type'] == 'linear_attention':
            # DeltaNet: sequential per-position with slice_write assembly
            # Pre-allocate ROW_MAJOR working buffer [1, 1, seq_len, HIDDEN]
            dn_buf_init = torch.zeros((1, 1, seq_len, HIDDEN), dtype=torch.bfloat16)
            dn_out_buf = ttnn.from_torch(
                dn_buf_init,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            for pos, tid in enumerate(prompt_ids):
                # Set state.cur_pos_buf etc (DeltaNet itself doesn't use them,
                # but harmless and keeps invariants consistent across modes).
                update_input_buffers(state, tid, pos)
                _dbg_l = layer_idx <= 1  # DN debug only for layers 0 and 1
                if _dbg_l: print(f"    [DN inner] layer={layer_idx} pos={pos} BEFORE slice", flush=True)
                x_pos = ttnn.slice(x_seq, [pos, 0], [pos + 1, HIDDEN])
                if _dbg_l:
                    ttnn.synchronize_device(state.mesh)
                    print(f"    [DN inner] layer={layer_idx} pos={pos} AFTER slice shape={list(x_pos.shape)}", flush=True)
                # B.2.2 SLICE-CORRECTNESS DIAG (2026-05-20): print x_pos[0,:]
                # values per chip. Should match x_seq[pos,:] exactly. If not,
                # slice is corrupting data on MLP-output TILE tensors.
                if getattr(state, 'ccl_debug', False) and layer_idx <= 1 and pos == 0:
                    try:
                        _full = ttnn.to_torch(
                            x_pos, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                        ).float()
                        _Hc = _full.shape[-1] // 4
                        _chip_v = [round(float(_full[0, c*_Hc]), 6) for c in range(4)]
                        _chip_m = [round(float(_full[0, c*_Hc:(c+1)*_Hc].mean()), 6) for c in range(4)]
                        print(f"    [v3 L{layer_idx} x_pos[0,:] after slice] chip_v0={_chip_v} "
                              f"chip_means={_chip_m}", flush=True)
                    except Exception as e:
                        print(f"    [v3 L{layer_idx} x_pos diag err] {e!r}", flush=True)
                # Run DN step (existing per-position decode step)
                x_pos_out = deltanet_step_tp(state, x_pos, layer['dn'], cfg)
                if _dbg_l:
                    ttnn.synchronize_device(state.mesh)
                    print(f"    [DN inner] layer={layer_idx} pos={pos} AFTER decode-step shape={list(x_pos_out.shape)}", flush=True)
                # x_pos is a VIEW of x_seq — DO NOT deallocate (would free
                # x_seq's underlying storage and break next iteration's slice).
                # Reshape to rank-4 + convert to ROW_MAJOR for slice_write
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
                # B.2.2 STATE INSPECTION (2026-05-20): if debug_state set, print
                # Layer N AND Layer 1's state — so we can see if Layer 1 gets
                # contaminated during Layer 0's processing.
                if getattr(state, 'debug_state', False) and layer_idx <= 1:
                    def _dump_state(li: int, label: str):
                        try:
                            if li >= len(state.layers):
                                return
                            lyr = state.layers[li]
                            if lyr['type'] != 'linear_attention':
                                return
                            _ssm_t = ttnn.to_torch(
                                lyr['dn']['ssm'],
                                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                            ).float()
                            _ssm_mean = round(float(_ssm_t.mean()), 8)
                            _ssm_norm = round(float(_ssm_t.norm()), 6)
                            _ssm_v0 = round(float(_ssm_t.flatten()[0]), 8)
                            if 'conv_st' in lyr['dn']:
                                _cs_c = ttnn.to_torch(
                                    lyr['dn']['conv_st'],
                                    mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                                ).float()
                                _cs_sum = (f"conv_st(mean={float(_cs_c.mean()):.6g} "
                                           f"norm={float(_cs_c.norm()):.4g})")
                            else:
                                _cs_sum = "conv_st(missing)"
                            _tag = getattr(state, '_debug_state_tag', 'pre')
                            print(f"  [{_tag} L{li} state {label}] "
                                  f"ssm(mean={_ssm_mean} v0={_ssm_v0} norm={_ssm_norm}) "
                                  f"{_cs_sum}", flush=True)
                        except Exception as _e:
                            print(f"  [v3 state err L{li}] {_e!r}", flush=True)
                    _dump_state(layer_idx, f"after pos{pos}")
                    # Also peek at Layer 1's state — if non-zero while we're
                    # still in Layer 0, it's been contaminated.
                    if layer_idx == 0:
                        _dump_state(1, f"L1view_during_L0_pos{pos}")
                # B.2.2 Test 2: optional sync between per-position DN calls
                # to test the async-ordering hypothesis. Adds ~5 syncs per
                # DN layer × 32 DN layers = 160 syncs per forward.
                if getattr(state, 'force_sync_per_position', False):
                    ttnn.synchronize_device(state.mesh)

            # Convert dn_out_buf to TILE_LAYOUT [seq_len, HIDDEN]
            ttnn.deallocate(x_seq)
            dn_out_4d_tile = ttnn.to_layout(dn_out_buf, ttnn.TILE_LAYOUT)
            ttnn.deallocate(dn_out_buf)
            x_seq = ttnn.reshape(dn_out_4d_tile, [seq_len, HIDDEN])
            ttnn.deallocate(dn_out_4d_tile)
        else:
            # Gated Attention: parallel SDPA across all seq_len positions
            new_x_seq = gated_attn_step_prefill_tp(
                state, x_seq, layer['attn'], cos_seq_tt, sin_seq_tt, cfg, seq_len)
            ttnn.deallocate(x_seq)
            x_seq = new_x_seq

        _layer_dbg(layer_idx, layer['type'], stage="pre_mlp")
        # B.2.2 Test 8: print x_seq[0,:] BEFORE MLP at L0 in v3 (single-shot)
        if layer_idx == 0 and getattr(state, 'debug_layer_boundary', False):
            _v3bm = getattr(state, '_pre_mlp_count', 0)
            if _v3bm < 2:  # allow both dec (in fwd_token) and pre (here) one print each
                try:
                    _v3xf = ttnn.to_torch(
                        x_seq, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
                    ).float()
                    _v3Hpm = _v3xf.shape[-1] // 4
                    _v3pv = [round(float(_v3xf[0, c*_v3Hpm]), 6) for c in range(4)]
                    _v3pm = [round(float(_v3xf[0, c*_v3Hpm:(c+1)*_v3Hpm].mean()), 6) for c in range(4)]
                    _v3pn = [round(float(_v3xf[0, c*_v3Hpm:(c+1)*_v3Hpm].norm()), 4) for c in range(4)]
                    _v3tg = getattr(state, '_debug_state_tag', 'pre')
                    print(f"  [{_v3tg} L0 pre-MLP x_seq[0,:]] chip_v0={_v3pv} chip_means={_v3pm} chip_norms={_v3pn}",
                          flush=True)
                    state._pre_mlp_count = _v3bm + 1
                except Exception as _v3e:
                    print(f"  [v3 L0 pre-MLP diag err] {_v3e!r}", flush=True)
        # MLP: batched on [seq_len, HIDDEN] (broadcasts on leading dim)
        # Note: the B.2.2 wedge fix is now in _tp_all_reduce (clones the
        # result to escape DeallocatedTombStone state at the source), so we
        # don't need to materialize again here.
        new_x_seq = mlp_step_tp(state, x_seq, layer['mlp'])
        ttnn.deallocate(x_seq)
        x_seq = new_x_seq
        _layer_dbg(layer_idx, layer['type'], stage="end")

    ttnn.deallocate(cos_seq_tt)
    ttnn.deallocate(sin_seq_tt)

    # Restore default CCL mode
    state.force_custom_allreduce = False

    # ====== Step 4: Final norm + LM head ======
    x_seq = _rms_norm_manual(x_seq, state.final_norm_tt, 1e-6, HIDDEN)

    if capture_logits:
        # Batched LM head over all positions
        sharded_logits_tt = ttnn.linear(x_seq, state.lm_head_tt)
        gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
        sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [seq_len, VOCAB])
        rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
        ttnn.deallocate(sharded_logits_tt)
        ttnn.deallocate(gathered_logits_tt)
        ttnn.deallocate(sliced_logits_tt)
        ttnn.synchronize_device(state.mesh)
        full_arr = ttnn.to_torch(
            rm_logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[:seq_len].float().cpu().numpy()
        ttnn.deallocate(rm_logits_tt)
        ttnn.deallocate(x_seq)
        return full_arr

    # Production: slice last position
    x_last = ttnn.slice(x_seq, [seq_len - 1, 0], [seq_len, HIDDEN])
    ttnn.deallocate(x_seq)
    sharded_logits_tt = ttnn.linear(x_last, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, VOCAB])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    ttnn.deallocate(sharded_logits_tt)
    ttnn.deallocate(gathered_logits_tt)
    ttnn.deallocate(sliced_logits_tt)
    ttnn.deallocate(x_last)
    ttnn.synchronize_device(state.mesh)
    return rm_logits_tt


def forward_prefill_tp_inner_v2_per_position_list(state, prompt_ids, capture_logits=False):
    """B.2.1 ISOLATION #2: layer-outer iter, keep per-position [1, HIDDEN]
    tensors in a Python list — NO slice, NO concat of multi-position tensors.

    If THIS mode gives cos=1.0, the slice/concat round-trip on TILE_LAYOUT
    [seq_len, HIDDEN] tensors was the bug. If still <1.0, the layer-outer
    iteration ordering itself is the problem.
    """
    import ttnn
    import numpy as np

    cfg = state.cfg
    HIDDEN = cfg['hidden']
    seq_len = len(prompt_ids)
    VOCAB = state.vocab_size

    # Step 1: per-position embed (each chip gets its own [1, HIDDEN])
    x_per_pos = []  # list of [1, HIDDEN] tensors, one per position
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        embed_out = ttnn.embedding(
            state.tok_buf, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_pos = ttnn.reshape(embed_out, [1, HIDDEN])
        x_per_pos.append(x_pos)

    # Step 2: layer-outer, position-inner — operating on Python list
    for layer in state.layers:
        layer_outs = []
        for pos, tid in enumerate(prompt_ids):
            update_input_buffers(state, tid, pos)
            cos_row_raw = ttnn.embedding(state.rot_idxs_buf, state.cos_table_tt,
                                          layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sin_row_raw = ttnn.embedding(state.rot_idxs_buf, state.sin_table_tt,
                                          layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cos_row_tt = ttnn.reshape(cos_row_raw, [1, state.rotary_dim])
            sin_row_tt = ttnn.reshape(sin_row_raw, [1, state.rotary_dim])

            x_pos = x_per_pos[pos]  # direct list access — no slice

            if layer['type'] == 'linear_attention':
                x_pos_dn = deltanet_step_tp(state, x_pos, layer['dn'], cfg)
            else:
                x_pos_dn = gated_attn_step_tp(
                    state, x_pos, layer['attn'],
                    state.cur_pos_buf, 0,
                    cos_row_tt, sin_row_tt, cfg)
            ttnn.deallocate(cos_row_tt)
            ttnn.deallocate(sin_row_tt)

            x_pos_out = mlp_step_tp(state, x_pos_dn, layer['mlp'])
            ttnn.deallocate(x_pos_dn)
            layer_outs.append(x_pos_out)

        # Deallocate previous layer's tensors
        for t in x_per_pos:
            ttnn.deallocate(t)
        x_per_pos = layer_outs  # next layer reads from this list

    # Per-position final norm + LM head (no batching, no slice/concat)
    if capture_logits:
        logits_arr = np.empty((seq_len, VOCAB), dtype=np.float32)
        for pos in range(seq_len):
            x_normed = _rms_norm_manual(x_per_pos[pos], state.final_norm_tt, 1e-6, HIDDEN)
            sharded_logits_tt = ttnn.linear(x_normed, state.lm_head_tt)
            gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
            sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, VOCAB])
            rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
            ttnn.synchronize_device(state.mesh)
            logits_arr[pos] = ttnn.to_torch(
                rm_logits_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            )[0].float().cpu().numpy().reshape(VOCAB)
            ttnn.deallocate(x_normed)
            ttnn.deallocate(sharded_logits_tt)
            ttnn.deallocate(gathered_logits_tt)
            ttnn.deallocate(sliced_logits_tt)
            ttnn.deallocate(rm_logits_tt)
        for t in x_per_pos:
            ttnn.deallocate(t)
        return logits_arr

    # Production: only need last position's logits
    x_last_normed = _rms_norm_manual(x_per_pos[-1], state.final_norm_tt, 1e-6, HIDDEN)
    for t in x_per_pos:
        ttnn.deallocate(t)
    sharded_logits_tt = ttnn.linear(x_last_normed, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, VOCAB])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    ttnn.deallocate(sharded_logits_tt)
    ttnn.deallocate(gathered_logits_tt)
    ttnn.deallocate(sliced_logits_tt)
    ttnn.deallocate(x_last_normed)
    ttnn.synchronize_device(state.mesh)
    return rm_logits_tt


def forward_prefill_tp_inner_v2_sequential_via_slices(state, prompt_ids, capture_logits=False):
    """B.2.1 ISOLATION: layer-outer iter with slice/concat plumbing AND
    sequential MLP per position. Used to localize whether B.2.1's batched_mlp
    failure is from slice/concat round-trip (TILE_LAYOUT padding artifacts)
    or from batched MLP itself.

    If THIS path gives cos=1.0 vs decode-loop reference: slice/concat is fine
    → bug is in batched MLP (`mlp_step_tp` on [seq_len, HIDDEN] differs from
    per-row MLP).

    If THIS path also gives cos<1.0: slice/concat introduces noise → fix the
    tensor-handling plumbing before any batching.
    """
    import ttnn
    import numpy as np

    cfg = state.cfg
    HIDDEN = cfg['hidden']
    seq_len = len(prompt_ids)
    VOCAB = state.vocab_size

    # Step 1: per-position embed
    embed_per_pos = []
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        embed_out = ttnn.embedding(
            state.tok_buf, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_pos = ttnn.reshape(embed_out, [1, HIDDEN])
        embed_per_pos.append(x_pos)
    x_seq = ttnn.concat(embed_per_pos, dim=0)
    for t in embed_per_pos:
        ttnn.deallocate(t)

    # Step 2: layer-outer; per-position DN/Attn AND per-position MLP
    for layer in state.layers:
        layer_outs = []
        for pos, tid in enumerate(prompt_ids):
            update_input_buffers(state, tid, pos)
            cos_row_raw = ttnn.embedding(state.rot_idxs_buf, state.cos_table_tt,
                                          layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sin_row_raw = ttnn.embedding(state.rot_idxs_buf, state.sin_table_tt,
                                          layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cos_row_tt = ttnn.reshape(cos_row_raw, [1, state.rotary_dim])
            sin_row_tt = ttnn.reshape(sin_row_raw, [1, state.rotary_dim])

            x_pos = ttnn.slice(x_seq, [pos, 0], [pos + 1, HIDDEN])

            if layer['type'] == 'linear_attention':
                x_pos_dn = deltanet_step_tp(state, x_pos, layer['dn'], cfg)
            else:
                x_pos_dn = gated_attn_step_tp(
                    state, x_pos, layer['attn'],
                    state.cur_pos_buf, 0,
                    cos_row_tt, sin_row_tt, cfg)
            ttnn.deallocate(x_pos)
            ttnn.deallocate(cos_row_tt)
            ttnn.deallocate(sin_row_tt)

            # SEQUENTIAL MLP per position (vs batched_mlp's batched MLP)
            x_pos_out = mlp_step_tp(state, x_pos_dn, layer['mlp'])
            ttnn.deallocate(x_pos_dn)
            layer_outs.append(x_pos_out)

        ttnn.deallocate(x_seq)
        x_seq = ttnn.concat(layer_outs, dim=0)
        for t in layer_outs:
            ttnn.deallocate(t)

    x_seq = _rms_norm_manual(x_seq, state.final_norm_tt, 1e-6, HIDDEN)

    if capture_logits:
        sharded_logits_tt = ttnn.linear(x_seq, state.lm_head_tt)
        gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
        sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [seq_len, VOCAB])
        rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
        ttnn.deallocate(sharded_logits_tt)
        ttnn.deallocate(gathered_logits_tt)
        ttnn.deallocate(sliced_logits_tt)
        ttnn.synchronize_device(state.mesh)
        full_arr = ttnn.to_torch(
            rm_logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[0].float().cpu().numpy()
        ttnn.deallocate(rm_logits_tt)
        ttnn.deallocate(x_seq)
        return full_arr

    x_last = ttnn.slice(x_seq, [seq_len - 1, 0], [seq_len, HIDDEN])
    ttnn.deallocate(x_seq)
    sharded_logits_tt = ttnn.linear(x_last, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, VOCAB])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    ttnn.deallocate(sharded_logits_tt)
    ttnn.deallocate(gathered_logits_tt)
    ttnn.deallocate(sliced_logits_tt)
    ttnn.deallocate(x_last)
    ttnn.synchronize_device(state.mesh)
    return rm_logits_tt


def forward_prefill_tp_inner_v2_batched_mlp(state, prompt_ids, capture_logits=False):
    """Phase B.2.1 prefill: layer-outer iteration with batched MLP per layer.

    First non-trivial restructure away from the B.1 stub. Structure:

      1. Per-position embedding (sequential, populates x_seq [seq_len, HIDDEN])
      2. For each layer:
         a. Per-position DN/Attn (sequential, slice from x_seq, run step,
            accumulate outputs)
         b. Concat per-position outputs into [seq_len, HIDDEN]
         c. Batched MLP on the [seq_len, HIDDEN] tensor (mlp_step_tp
            broadcasts on leading dim — no inter-position deps in MLP math)
      3. Final norm on [seq_len, HIDDEN]
      4. LM head: per-position via batched matmul (capture_logits=True) or
         slice last position (production mode)

    Gate: per-position cos >= 0.999 vs decode-loop reference. Any failure
    here is a restructure plumbing bug (MLP math is trivially correct on
    leading-dim batch), NOT a math change.

    Sequential DN/Attn for now; B.2.2 batches Attn; B.3 batches DN via
    Neumann chunked-parallel.
    """
    import ttnn
    import numpy as np

    cfg = state.cfg
    HIDDEN = cfg['hidden']
    seq_len = len(prompt_ids)
    VOCAB = state.vocab_size

    if seq_len < 1:
        raise ValueError(f"prompt_ids must have len >= 1, got {seq_len}")
    if seq_len > MAX_POS:
        raise ValueError(f"prompt_ids len {seq_len} > MAX_POS {MAX_POS}")

    # Step 1: per-position embed lookup, accumulate into x_seq [seq_len, HIDDEN].
    # update_input_buffers writes tok_buf for ttnn.embedding to read.
    embed_per_pos = []
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        embed_out = ttnn.embedding(
            state.tok_buf, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_pos = ttnn.reshape(embed_out, [1, HIDDEN])
        embed_per_pos.append(x_pos)
    x_seq = ttnn.concat(embed_per_pos, dim=0)  # [seq_len, HIDDEN]
    for t in embed_per_pos:
        ttnn.deallocate(t)

    # Step 2: layer-outer, position-inner
    for layer in state.layers:
        # 2a. Per-position DN/Attn
        layer_outs = []
        for pos, tid in enumerate(prompt_ids):
            # Update buffers so the per-position step has correct cur_pos
            # (for KV cache writes in attention) and rot_idxs (for cos/sin lookup).
            # tid value is irrelevant here — we use x_pos sliced from x_seq.
            update_input_buffers(state, tid, pos)

            # Look up cos_row, sin_row for this position
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

            # Slice x_pos from x_seq for this position
            x_pos = ttnn.slice(x_seq, [pos, 0], [pos + 1, HIDDEN])

            # Run DN or Attn for this position
            if layer['type'] == 'linear_attention':
                x_pos_out = deltanet_step_tp(state, x_pos, layer['dn'], cfg)
            else:
                x_pos_out = gated_attn_step_tp(
                    state, x_pos, layer['attn'],
                    state.cur_pos_buf, 0,
                    cos_row_tt, sin_row_tt, cfg)
            ttnn.deallocate(x_pos)
            ttnn.deallocate(cos_row_tt)
            ttnn.deallocate(sin_row_tt)
            layer_outs.append(x_pos_out)

        # 2b. Concat per-position outputs
        ttnn.deallocate(x_seq)
        x_dn_seq = ttnn.concat(layer_outs, dim=0)  # [seq_len, HIDDEN]
        for t in layer_outs:
            ttnn.deallocate(t)

        # 2c. Batched MLP on [seq_len, HIDDEN]
        x_seq = mlp_step_tp(state, x_dn_seq, layer['mlp'])
        ttnn.deallocate(x_dn_seq)

    # Step 3: final norm on [seq_len, HIDDEN]
    x_seq = _rms_norm_manual(x_seq, state.final_norm_tt, 1e-6, HIDDEN)

    # Step 4: LM head
    if capture_logits:
        # Batched LM head — per-position via batched matmul
        sharded_logits_tt = ttnn.linear(x_seq, state.lm_head_tt)
        gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
        sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [seq_len, VOCAB])
        rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
        ttnn.deallocate(sharded_logits_tt)
        ttnn.deallocate(gathered_logits_tt)
        ttnn.deallocate(sliced_logits_tt)

        # Read back per-position
        ttnn.synchronize_device(state.mesh)
        # ConcatMeshToTensor(dim=0) on a REPLICATED [seq_len, VOCAB] tensor
        # produces [NCHIPS*seq_len, VOCAB] (each chip's seq_len rows
        # concatenated). Take chip 0's view = first seq_len rows.
        # (Bugfix: previous [0] indexed only the FIRST ROW, not chip 0's
        # full [seq_len, VOCAB] view — caused per-position cosine = 1.0
        # only at pos 0 because numpy broadcast that one vector against
        # the reference's seq_len rows.)
        full_arr = ttnn.to_torch(
            rm_logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[:seq_len].float().cpu().numpy()  # [seq_len, VOCAB]
        ttnn.deallocate(rm_logits_tt)
        ttnn.deallocate(x_seq)
        return full_arr

    # Production: slice last position then standard LM head
    x_last = ttnn.slice(x_seq, [seq_len - 1, 0], [seq_len, HIDDEN])
    ttnn.deallocate(x_seq)
    sharded_logits_tt = ttnn.linear(x_last, state.lm_head_tt)
    gathered_logits_tt = ttnn.all_gather(sharded_logits_tt, dim=-1)
    sliced_logits_tt = ttnn.slice(gathered_logits_tt, [0, 0], [1, VOCAB])
    rm_logits_tt = ttnn.untilize(sliced_logits_tt, use_multicore=True)
    ttnn.deallocate(sharded_logits_tt)
    ttnn.deallocate(gathered_logits_tt)
    ttnn.deallocate(sliced_logits_tt)
    ttnn.deallocate(x_last)
    ttnn.synchronize_device(state.mesh)
    return rm_logits_tt


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


def handle_probe_ccl_components_tp(state: MeshServerState, args: dict) -> dict:
    """Micro-bench CCL primitives at production TP shape.

    Measures all_reduce / reduce_scatter / all_gather × num_links ∈ {1, 2}
    at shape [1, HIDDEN=5120] bf16 on the (1, 4) mesh (cluster_axis=1,
    topology=Linear).

    Answers two questions from one bootstrap:
      P1 (free win?): all_reduce(num_links=1) vs all_reduce(num_links=2).
      P2 (composite?): single all_reduce vs reduce_scatter + all_gather
        (the synchronous composite path); per-link variants compared.

    All measurements are sync-bounded around a single ttnn call; warmup is
    discarded. The composite latency is derived (rs + ag, same num_links)
    rather than measured fused — kept simple for the gate decision.
    """
    import ttnn
    import torch
    import time as _time
    import numpy as np

    iters = int(args.get("iters", 30))
    warmup = int(args.get("warmup", 5))
    shape = args.get("shape", [1, 5120])
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if len(shape) != 2 or shape[0] != 1:
        return {"error": f"shape must be [1, H], got {shape}"}
    H = int(shape[1])
    if H % 4 != 0:
        return {"error": f"H={H} must be divisible by 4 (NCHIPS)"}

    rng = np.random.default_rng(42)

    def upload_replicated_2d(H_):
        x = rng.standard_normal((1, H_)).astype(np.float32) * 0.05
        return ttnn.from_torch(
            torch.from_numpy(x),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def upload_sharded_2d(H_):
        x = rng.standard_normal((1, H_)).astype(np.float32) * 0.05
        return ttnn.from_torch(
            torch.from_numpy(x),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def sync():
        ttnn.synchronize_device(state.mesh)

    def time_op(fn, x_in):
        sync()
        t0 = _time.perf_counter()
        out = fn(x_in)
        sync()
        return (_time.perf_counter() - t0) * 1000.0, out

    # ---- variant table ------------------------------------------------
    # all_reduce + reduce_scatter take the per-chip [1, H] partial.
    # all_gather takes a per-chip [1, H/4] slice that gets gathered to [1, H].
    H_SHARD = H // 4
    variants = []  # list of (name, x_factory, op_fn)
    for L in (1, 2):
        variants.append((
            f"all_reduce_L{L}_linear",
            lambda: upload_replicated_2d(H),
            lambda x, L=L: ttnn.all_reduce(
                x, cluster_axis=1,
                memory_config=x.memory_config(),
                num_links=L, topology=ttnn.Topology.Linear,
            ),
        ))
        variants.append((
            f"reduce_scatter_L{L}_linear",
            lambda: upload_replicated_2d(H),
            lambda x, L=L: ttnn.reduce_scatter(
                x, dim=1, cluster_axis=1,
                num_links=L, topology=ttnn.Topology.Linear,
            ),
        ))
        variants.append((
            f"all_gather_L{L}_linear",
            lambda H_=H_SHARD: upload_sharded_2d(H),  # global [1,H], sharded → per-chip [1, H/4]
            lambda x, L=L: ttnn.all_gather(
                x, dim=1, cluster_axis=1,
                num_links=L, topology=ttnn.Topology.Linear,
            ),
        ))

    results = {}
    errors = {}

    for name, x_factory, op_fn in variants:
        try:
            x_in = x_factory()
            # Warmup (discarded).
            for _ in range(warmup):
                _, out = time_op(op_fn, x_in)
                if out is not None:
                    ttnn.deallocate(out)
            samples = []
            for _ in range(iters):
                ms, out = time_op(op_fn, x_in)
                samples.append(ms)
                if out is not None:
                    ttnn.deallocate(out)
            ttnn.deallocate(x_in)
            results[name] = {
                "samples_ms": samples,
                "summary_ms": _summary_ms(samples),
            }
        except Exception as e:
            errors[name] = repr(e)

    # ---- derived composite (rs + ag at matching num_links) ------------
    composites = {}
    for L in (1, 2):
        rs_key = f"reduce_scatter_L{L}_linear"
        ag_key = f"all_gather_L{L}_linear"
        ar_key = f"all_reduce_L{L}_linear"
        if rs_key in results and ag_key in results:
            rs_med = results[rs_key]["summary_ms"].get("median", float("nan"))
            ag_med = results[ag_key]["summary_ms"].get("median", float("nan"))
            ar_med = results.get(ar_key, {}).get("summary_ms", {}).get("median", float("nan"))
            composites[f"composite_L{L}_linear"] = {
                "rs_median_ms": rs_med,
                "ag_median_ms": ag_med,
                "sum_median_ms": rs_med + ag_med,
                "vs_all_reduce_median_ms": ar_med,
                "composite_minus_all_reduce_ms": (rs_med + ag_med) - ar_med,
            }

    state.last_run = {
        "cmd": "probe_ccl_components_tp",
        "shape": shape,
        "iters": iters,
        "n_variants": len(results),
        "n_errors": len(errors),
    }
    return {
        "ok": True,
        "shape": shape,
        "iters": iters,
        "warmup": warmup,
        "variants": results,
        "composites": composites,
        "errors": errors,
        "note": (
            "Sync-bounded per-op timing. all_reduce + reduce_scatter input is "
            "replicated [1, H] on each chip; all_gather input is sharded along "
            "dim=1 (per-chip [1, H/4]). Composite latency is derived (rs + ag, "
            "same num_links). Compare composite_minus_all_reduce_ms against zero "
            "to decide composite-vs-fused; compare L2 vs L1 within all_reduce "
            "to decide the free-bandwidth probe."
        ),
    }


def handle_probe_async_ccl_components_tp(state: MeshServerState, args: dict) -> dict:
    """G0: async-CCL component bench at production [1, HIDDEN] shape.

    Compares the SYNC all_reduce (production path, num_links=2) against
    `ttnn.experimental.all_reduce_async` in three regimes:

      v1 sync_baseline        — production today
      v2 async_immediate_sync — async launch + immediate barrier; isolates
                                pure async-vs-sync overhead
      v3 async_double         — two async ARs back-to-back + single barrier;
                                tests whether the eth fabric supports
                                multiple in-flight collectives (parallelism)
      v4 async_with_matmul    — async AR + an independent DRAM-bandwidth-
                                bound matmul + single barrier; tests
                                compute-comm overlap
      v5 matmul_alone         — matmul without CCL (overlap math baseline)
      v6 sync_ar_plus_matmul  — sync AR then matmul, both serialized
                                (overlap math baseline)

    Derived composites:
      async_overhead_ms        = v2 - v1   (>0 → async slower than sync)
      async_double_minus_2x    = v3 - 2*v1 (<0 → fabric parallelizes)
      overlap_savings_ms       = v6 - v4   (>0 → comm overlapped with mm)
      overlap_capacity_ms      = v5        (max possible save)

    Gate for G1 (single-layer prototype):
      either async_double_minus_2x < 0 OR overlap_savings_ms > 0.5×v1
    """
    import ttnn
    import torch
    import time as _time
    import numpy as np

    iters = int(args.get("iters", 30))
    warmup = int(args.get("warmup", 5))
    H = int(args.get("hidden", 5120))
    K = int(args.get("matmul_k", 5120))
    N = int(args.get("matmul_n", 32768))

    if iters <= 0:
        return {"error": "iters must be > 0"}
    if H % 4 != 0:
        return {"error": f"H={H} must be divisible by 4 (NCHIPS)"}
    if N % 4 != 0:
        return {"error": f"matmul_n={N} must be divisible by 4"}

    grid = state.mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(grid.x - 1, grid.y - 1),
        )
    })

    # Pre-allocate semaphore POOLS (4 sets — enough headroom for double-buffered
    # back-to-back launches in v3). Each AR-async call needs 2 barrier + 3 RS +
    # 2 AG = 7 sems. We allocate 4 * 7 = 28 sems and pick set [0] / set [1] in v3.
    def make_sem():
        return ttnn.create_global_semaphore(state.mesh, cores, 0)

    sem_sets = []
    for _ in range(4):
        sem_sets.append({
            "barrier": [make_sem() for _ in range(2)],
            "rs": [make_sem() for _ in range(3)],
            "ag": [make_sem() for _ in range(2)],
        })

    rng = np.random.default_rng(42)

    def fresh_input():
        x = rng.standard_normal((1, H)).astype(np.float32) * 0.05
        return ttnn.from_torch(
            torch.from_numpy(x),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Persistent matmul tensors for v4/v5/v6 (~DRAM-bandwidth-bound matmul:
    # [1, K] @ [K, N] sharded along N=dim1 → per-chip [K, N/4] weight ≈ 84 MB
    # at default bf16 → ~0.16 ms at 512 GB/s peak. Realistic stand-in for the
    # next-layer's MLP gate matmul we'd want to overlap.)
    w_np = (rng.standard_normal((K, N)).astype(np.float32) * 0.01)
    w_tt = ttnn.from_torch(
        torch.from_numpy(w_np),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=state.mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    h_mm_np = rng.standard_normal((1, K)).astype(np.float32) * 0.05
    h_mm_tt = ttnn.from_torch(
        torch.from_numpy(h_mm_np),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def sync():
        ttnn.synchronize_device(state.mesh)

    def measure(op_fn):
        # Fresh input per iter to dodge any caching / state leaks.
        x = fresh_input()
        sync()
        t0 = _time.perf_counter()
        outs = op_fn(x)
        sync()
        dt = (_time.perf_counter() - t0) * 1000.0
        if isinstance(outs, (list, tuple)):
            for o in outs:
                if o is not None:
                    ttnn.deallocate(o)
        elif outs is not None:
            ttnn.deallocate(outs)
        ttnn.deallocate(x)
        return dt

    def ar_sync(x, L=2):
        return ttnn.all_reduce(
            x, cluster_axis=1, memory_config=x.memory_config(),
            num_links=L, topology=ttnn.Topology.Linear,
        )

    def ar_async(x, sem_idx=0, L=2):
        s = sem_sets[sem_idx]
        return ttnn.experimental.all_reduce_async(
            x,
            cluster_axis=1, mesh_device=state.mesh,
            barrier_semaphores=s["barrier"],
            rs_global_semaphores=s["rs"],
            ag_global_semaphores=s["ag"],
            math_op=ttnn.ReduceType.Sum,
            num_links=L,
            memory_config=x.memory_config(),
            topology=ttnn.Topology.Linear,
        )

    def mm_op():
        return ttnn.linear(h_mm_tt, w_tt)

    variants = [
        ("v1_sync_baseline_L2",         lambda x: ar_sync(x, L=2)),
        ("v2_async_immediate_sync_L2",  lambda x: ar_async(x, sem_idx=0, L=2)),
        ("v3_async_double_L1",          lambda x: [
            ar_async(x, sem_idx=0, L=1),
            ar_async(x, sem_idx=1, L=1),
        ]),
        ("v4_async_with_matmul_L2",     lambda x: [
            ar_async(x, sem_idx=0, L=2),
            mm_op(),
        ]),
        ("v5_matmul_alone",             lambda x: mm_op()),
        ("v6_sync_ar_plus_matmul",      lambda x: [ar_sync(x, L=2), mm_op()]),
    ]

    results = {}
    errors = {}

    for name, op_fn in variants:
        try:
            for _ in range(warmup):
                measure(op_fn)
            samples = [measure(op_fn) for _ in range(iters)]
            results[name] = {"samples_ms": samples, "summary_ms": _summary_ms(samples)}
        except Exception as e:
            errors[name] = repr(e)

    # Cleanup persistent tensors.
    ttnn.deallocate(w_tt)
    ttnn.deallocate(h_mm_tt)

    composites = {}

    def med(name):
        return results.get(name, {}).get("summary_ms", {}).get("median", float("nan"))

    v1 = med("v1_sync_baseline_L2")
    v2 = med("v2_async_immediate_sync_L2")
    v3 = med("v3_async_double_L1")
    v4 = med("v4_async_with_matmul_L2")
    v5 = med("v5_matmul_alone")
    v6 = med("v6_sync_ar_plus_matmul")
    composites["async_overhead_ms"] = v2 - v1
    composites["async_double_minus_2x_v1"] = v3 - 2 * v1
    composites["overlap_savings_ms"] = v6 - v4
    composites["overlap_capacity_ms"] = v5
    composites["v6_serial_baseline_ms"] = v6

    state.last_run = {
        "cmd": "probe_async_ccl_components_tp",
        "n_variants": len(results),
        "n_errors": len(errors),
    }
    return {
        "ok": True,
        "iters": iters,
        "warmup": warmup,
        "shape": [1, H],
        "matmul_shape": [K, N],
        "variants": results,
        "composites": composites,
        "errors": errors,
        "note": (
            "G0 async-CCL component bench. Decision rules: "
            "(1) ship async (cheaper than sync) iff async_overhead_ms < 0; "
            "(2) explore fabric parallelism iff async_double_minus_2x_v1 < 0; "
            "(3) build single-layer overlap iff overlap_savings_ms > 0.5 * v1 "
            "(equivalently, async truly overlaps a meaningful comm window with compute)."
        ),
    }


def handle_probe_prefill_vs_decode_loop_tp(state: MeshServerState, args: dict) -> dict:
    """B.1 prefill validation harness.

    Compares per-position logits from:
      A. Sequential decode-loop reference — established HF-validated path
         via cosine_ladder_tp (cos ≥0.999 per memory feedback_long_context_
         cosine_ladder.md).
      B. forward_prefill_tp_inner (the function we're building). Starts as
         a decode-loop wrapper (cos = 1.0 trivially); subsequent phases
         (B.2, B.3) replace internals with parallel SDPA, paged_fill_cache,
         and Neumann chunked-parallel DeltaNet, validated each step by this
         same harness.

    Gate per position: cos ≥ 0.999.

    Strategy rationale (vs the numpy-ref approach we tried + abandoned at
    research/b1_numpy_ref_vs_hf_failure_2026_05_19.md): 91b's pure-numpy
    deltanet has drifted from HF without being caught, so a numpy ref is
    not a trustworthy reference. owned_gdn (the shipped decode kernel) IS
    HF-validated, so the decode-loop IS our authoritative reference for
    prefill parity.

    args:
      prompt_ids: list[int] (optional — defaults to "The capital of France is")
      prompt:     str       (optional — used if prompt_ids absent)
    """
    import time as _time
    import numpy as np
    import ttnn

    if state.mesh is None or not state.layers:
        return {"error": "mesh/weights not loaded"}

    prompt_ids = list(args.get("prompt_ids") or [])
    if not prompt_ids:
        if state.tok is None:
            return {"error": "missing prompt_ids and tokenizer unavailable"}
        prompt = str(args.get("prompt", "The capital of France is"))
        prompt_ids = state.tok.encode(prompt)

    mode = str(args.get("mode", "stub"))
    valid_modes = ("stub", "batched_mlp", "sequential_via_slices",
                   "per_position_list", "parallel_attn")
    if mode not in valid_modes:
        return {"error": f"mode must be one of {valid_modes}, got {mode}"}

    seq_len = len(prompt_ids)
    if seq_len < 2:
        return {"error": f"prompt_ids must have len >= 2, got {seq_len}"}
    if seq_len > MAX_POS:
        return {"error": f"prompt_ids len {seq_len} > MAX_POS {MAX_POS}"}

    VOCAB = state.vocab_size

    debug_ccl = bool(args.get("debug_ccl", False))
    debug_state = bool(args.get("debug_state", False))

    def _print_dn_state_one(tag: str, layer_idx: int, pos_label: str):
        """Read back ONE layer's dn['ssm'], conv_st, conv_st_split. Print summary."""
        if not debug_state:
            return
        if layer_idx >= len(state.layers):
            return
        layer = state.layers[layer_idx]
        if layer['type'] != 'linear_attention':
            return
        try:
            import ttnn as _t
            ssm = _t.to_torch(
                layer['dn']['ssm'],
                mesh_composer=_t.ConcatMeshToTensor(state.mesh, dim=1)
            ).float()
            ssm_mean = round(float(ssm.mean()), 8)
            ssm_norm = round(float(ssm.norm()), 6)
            ssm_v0 = round(float(ssm.flatten()[0]), 8)
            # conv_st: COMBINED tensor used by manual conv1d (default mode)
            if 'conv_st' in layer['dn']:
                cs_combined = _t.to_torch(
                    layer['dn']['conv_st'],
                    mesh_composer=_t.ConcatMeshToTensor(state.mesh, dim=0)
                ).float()
                cs_combined_summary = (
                    f"conv_st(mean={float(cs_combined.mean()):.6g} "
                    f"norm={float(cs_combined.norm()):.4g})"
                )
            else:
                cs_combined_summary = "conv_st(missing)"
            print(f"  [{tag} L{layer_idx} state {pos_label}] "
                  f"ssm(mean={ssm_mean} v0={ssm_v0} norm={ssm_norm}) "
                  f"{cs_combined_summary}",
                  flush=True)
        except Exception as e:
            print(f"  [{tag} L{layer_idx} state err] {e!r}", flush=True)

    def _print_dn_state(tag: str, layer_idx: int, pos_label: str):
        """Print state for the specified layer AND layer 1 (always — to see
        if Layer 1's state changes during Layer 0's processing)."""
        _print_dn_state_one(tag, layer_idx, pos_label)
        if layer_idx == 0:
            _print_dn_state_one(tag, 1, f"{pos_label}_L1view")
    # B.2.2 TEST: optionally override the recurrence mode for the test path.
    # Used to test the hypothesis that owned_gdn(_inplace) kernels have hidden
    # state that contaminates layer-1+ DN output in v3 prefill context.
    # When None, uses the current production default. Modes: 'manual',
    # 'owned_gdn', 'owned_gdn_inplace'.
    test_dn_mode = args.get("test_dn_mode")
    if test_dn_mode is not None and test_dn_mode not in (
        "manual", "owned_gdn", "owned_gdn_inplace"
    ):
        return {"error": f"test_dn_mode must be manual/owned_gdn/owned_gdn_inplace, got {test_dn_mode!r}"}
    test_decay_gate_mode = args.get("test_decay_gate_mode")
    if test_decay_gate_mode is not None and test_decay_gate_mode not in (
        "manual", "owned_decay_gate"
    ):
        return {"error": f"test_decay_gate_mode must be manual/owned_decay_gate, got {test_decay_gate_mode!r}"}

    # Path A — reference: explicit sequential decode-loop, captured per-position.
    _reset_state_buffers(state)
    if debug_ccl:
        state.ccl_debug = True
        state.ccl_debug_tag = "dec"
        state._ccl_debug_count = 0
        state._ccl_out_count = 0
    if debug_state:
        # Layer-boundary print is inside forward_token_tp_inner — called by the
        # REFERENCE path. So set the flag BEFORE the reference loop.
        state.debug_layer_boundary = True
        state._layer_bd_count = 0
        state._pre_mlp_count = 0
        state.debug_mlp_resid = True
        state._mlp_resid_count = 0
        state._debug_state_tag = "dec"
    t0 = _time.time()
    ref_logits = np.empty((seq_len, VOCAB), dtype=np.float32)
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        logits_tt = forward_token_tp_inner(state, return_logits=True)
        ttnn.synchronize_device(state.mesh)
        ref_logits[pos] = ttnn.to_torch(
            logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[0].float().cpu().numpy().reshape(VOCAB)
        _print_dn_state("dec", 0, f"pos{pos}")
    ref_ms = (_time.time() - t0) * 1000.0

    # Path B — test: forward_prefill_tp_inner.
    # mode="stub": calls decode-loop internally → cos = 1.0 trivially
    # mode="batched_mlp": B.2.1 layer-outer with batched MLP per layer → expect cos ≥ 0.999
    _reset_state_buffers(state)
    if debug_ccl:
        state.ccl_debug_tag = "pre"
        state._ccl_debug_count = 0
        state._ccl_out_count = 0
    _orig_dn_mode = None
    if test_dn_mode is not None:
        _orig_dn_mode = state.deltanet_recurrence_mode
        state.deltanet_recurrence_mode = test_dn_mode
        print(f"[probe] override deltanet_recurrence_mode={test_dn_mode} for test path "
              f"(was {_orig_dn_mode})", flush=True)
    _orig_dg_mode = None
    if test_decay_gate_mode is not None:
        _orig_dg_mode = state.deltanet_decay_gate_mode
        state.deltanet_decay_gate_mode = test_decay_gate_mode
        print(f"[probe] override deltanet_decay_gate_mode={test_decay_gate_mode} for test path "
              f"(was {_orig_dg_mode})", flush=True)
    if debug_state:
        state.debug_state = True
        state._debug_state_tag = "pre"
        # _layer_bd_count reset so v3 also gets one print (it doesn't call
        # forward_token_tp_inner, but reset for safety/consistency).
        state._layer_bd_count = 0
        # NOTE: do NOT reset _pre_mlp_count here — the limit is 2 (one for dec
        # path which already happened, one for pre path which is about to).
    force_sync = bool(args.get("force_sync_per_position", False))
    if force_sync:
        state.force_sync_per_position = True
        print(f"[probe] force_sync_per_position=True for test path", flush=True)
    t0 = _time.time()
    if mode == "stub":
        test_logits = forward_prefill_tp_inner(state, prompt_ids, capture_logits=True)
    elif mode == "batched_mlp":
        test_logits = forward_prefill_tp_inner_v2_batched_mlp(
            state, prompt_ids, capture_logits=True)
    elif mode == "sequential_via_slices":
        test_logits = forward_prefill_tp_inner_v2_sequential_via_slices(
            state, prompt_ids, capture_logits=True)
    elif mode == "per_position_list":
        test_logits = forward_prefill_tp_inner_v2_per_position_list(
            state, prompt_ids, capture_logits=True)
    elif mode == "parallel_attn":
        test_logits = forward_prefill_tp_inner_v3_parallel_attn(
            state, prompt_ids, capture_logits=True)
    test_ms = (_time.time() - t0) * 1000.0
    if debug_ccl:
        state.ccl_debug = False
    if _orig_dn_mode is not None:
        state.deltanet_recurrence_mode = _orig_dn_mode
    if _orig_dg_mode is not None:
        state.deltanet_decay_gate_mode = _orig_dg_mode
    if force_sync:
        state.force_sync_per_position = False
    if debug_state:
        state.debug_state = False
        state.debug_layer_boundary = False
        state.debug_mlp_resid = False

    # Per-position comparison
    a64 = ref_logits.astype(np.float64)
    b64 = test_logits.astype(np.float64)
    dot = (a64 * b64).sum(axis=-1)
    na = np.linalg.norm(a64, axis=-1)
    nb = np.linalg.norm(b64, axis=-1)
    pos_cos = dot / (na * nb + 1e-12)
    max_abs_diff = float(np.abs(a64 - b64).max())

    ref_top1 = np.argmax(ref_logits, axis=-1)
    test_top1 = np.argmax(test_logits, axis=-1)
    top1_agree = int((ref_top1 == test_top1).sum())

    state.last_run = {
        "cmd": "probe_prefill_vs_decode_loop_tp",
        "mode": mode,
        "seq_len": seq_len,
        "min_cosine": float(pos_cos.min()),
    }
    return {
        "ok": True,
        "mode": mode,
        "seq_len": seq_len,
        "vocab": VOCAB,
        "reference_ms": ref_ms,
        "test_ms": test_ms,
        "per_position_cosine": {
            "min": float(pos_cos.min()),
            "median": float(np.median(pos_cos)),
            "mean": float(pos_cos.mean()),
            "max": float(pos_cos.max()),
        },
        "per_position_cosine_arr": pos_cos.tolist(),
        "max_abs_diff": max_abs_diff,
        "top1_agreement": f"{top1_agree}/{seq_len}",
        "pass_gate_0p999": bool(pos_cos.min() >= 0.999),
        "note": (
            "Gate: per-position cos >= 0.999 vs decode-loop reference. "
            "Initial B.1 stub == reference so cos == 1.0 expected. "
            "B.2/B.3 will progressively replace forward_prefill_tp_inner "
            "internals with real parallel components."
        ),
    }


def handle_probe_multirow_construct_vs_per_position(state: MeshServerState, args: dict) -> dict:
    """B.2.1.5: validate that directly-constructed multi-row [seq_len, HIDDEN]
    tensors support correct row-slicing.

    Background: B.2.1 found that concat of TILE_LAYOUT [1, HIDDEN] tensors
    + subsequent slice gives wrong row data (see
    research/b2_1_findings_2026_05_19.md). Hypothesis: the bug is specific
    to CONCAT — a multi-row tensor constructed DIRECTLY via a single
    ttnn.embedding lookup might not have the same row-padding issue.

    Approach:
      A. Reference: per-position embed loop, save list of [1, HIDDEN] readbacks
      B. Test: single ttnn.embedding call with [seq_len, 1] index tensor →
         [seq_len, HIDDEN] directly. Slice each row and read back.
      Compare per-row cosine (test vs reference).

    Gate: per-row cos = 1.0 (math is identical lookup, just different
    construction). If pass, B.2.2 can use batched construction throughout
    without slice/concat plumbing. If fail, fall back to pre-allocated +
    slice_write (option 2 in research note).
    """
    import ttnn
    import torch
    import numpy as np

    if state.mesh is None:
        return {"error": "mesh not loaded"}

    prompt_ids = list(args.get("prompt_ids") or [])
    if not prompt_ids:
        if state.tok is None:
            return {"error": "missing prompt_ids and tokenizer unavailable"}
        prompt = str(args.get("prompt", "The capital of France is"))
        prompt_ids = state.tok.encode(prompt)

    seq_len = len(prompt_ids)
    if seq_len < 2:
        return {"error": f"seq_len must be >= 2, got {seq_len}"}
    HIDDEN = state.cfg['hidden']

    # Reference: per-position embed via existing path (tok_buf single-index lookup)
    per_pos_refs = []
    for pos, tid in enumerate(prompt_ids):
        update_input_buffers(state, tid, pos)
        embed_out = ttnn.embedding(
            state.tok_buf, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_pos = ttnn.reshape(embed_out, [1, HIDDEN])
        ttnn.synchronize_device(state.mesh)
        ref_arr = ttnn.to_torch(
            x_pos, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[0].float().cpu().numpy().reshape(HIDDEN)
        per_pos_refs.append(ref_arr)
        ttnn.deallocate(embed_out)
        ttnn.deallocate(x_pos)

    # Test: direct batched embed via single ttnn.embedding call with
    # [seq_len, 1] index tensor.
    prompt_ids_idx = ttnn.from_torch(
        torch.tensor([[int(t)] for t in prompt_ids], dtype=torch.int32),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
    )
    try:
        batched_embed_raw = ttnn.embedding(
            prompt_ids_idx, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    except Exception as e:
        ttnn.deallocate(prompt_ids_idx)
        return {
            "error": f"batched ttnn.embedding failed: {type(e).__name__}: {e}",
            "verdict": "option 1 (direct batched embed) NOT VIABLE — fall back to slice_write",
        }
    # batched_embed_raw expected shape: [seq_len, 1, HIDDEN] per chip
    # (per-chip replication via ReplicateTensorToMesh).
    # Reshape to [seq_len, HIDDEN].
    raw_shape = list(batched_embed_raw.shape)
    try:
        batched = ttnn.reshape(batched_embed_raw, [seq_len, HIDDEN])
    except Exception as e:
        ttnn.deallocate(batched_embed_raw)
        ttnn.deallocate(prompt_ids_idx)
        return {
            "error": f"reshape to [seq_len, HIDDEN] failed: {type(e).__name__}: {e}",
            "batched_embed_raw_shape": raw_shape,
        }

    # For each row, slice and compare to reference
    per_row = []
    for pos in range(seq_len):
        x_pos_test = ttnn.slice(batched, [pos, 0], [pos + 1, HIDDEN])
        ttnn.synchronize_device(state.mesh)
        test_arr = ttnn.to_torch(
            x_pos_test, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        )[0].float().cpu().numpy().reshape(HIDDEN)
        ttnn.deallocate(x_pos_test)

        ref = per_pos_refs[pos].astype(np.float64)
        test = test_arr.astype(np.float64)
        dot = float((ref * test).sum())
        nref = float(np.linalg.norm(ref))
        ntest = float(np.linalg.norm(test))
        cos = dot / (nref * ntest + 1e-12)
        max_diff = float(np.abs(ref - test).max())
        per_row.append({"pos": pos, "cos": cos, "max_abs_diff": max_diff})

    ttnn.deallocate(batched)
    ttnn.deallocate(batched_embed_raw)
    ttnn.deallocate(prompt_ids_idx)

    all_cos = [r["cos"] for r in per_row]
    all_diff = [r["max_abs_diff"] for r in per_row]
    pass_gate = all(c >= 0.999 for c in all_cos)

    state.last_run = {
        "cmd": "probe_multirow_construct_vs_per_position",
        "seq_len": seq_len,
        "min_cos": min(all_cos),
        "pass": pass_gate,
    }
    return {
        "ok": True,
        "seq_len": seq_len,
        "batched_embed_raw_shape": raw_shape,
        "batched_reshaped_to": [seq_len, HIDDEN],
        "per_row": per_row,
        "min_cos": min(all_cos),
        "max_cos": max(all_cos),
        "max_abs_diff": max(all_diff),
        "pass_gate_0p999": pass_gate,
        "verdict": (
            "PASS — batched ttnn.embedding + slice works; B.2.2 can use direct "
            "batched construction throughout"
            if pass_gate else
            "FAIL — even direct construction has TILE_LAYOUT row issues; "
            "need pre-allocated tensor + slice_write (option 2) for B.2.2"
        ),
    }


def handle_probe_slice_write_round_trip(state: MeshServerState, args: dict) -> dict:
    """B.2.1.5b: probe ttnn.experimental.slice_write for pre-allocate + per-row write.

    `slice_write` constraints (per ttnn help):
      - rank == 4
      - dtype bfloat16
      - ROW_MAJOR layout (NOT TILE_LAYOUT)
      - output interleaved
      - last-dim slicing unsupported (we slice dim 2)

    Test:
      1. Pre-allocate dst = zeros([1, 1, seq_len, HIDDEN]) ROW_MAJOR bf16 on mesh
      2. For each pos: src = full([1, 1, 1, HIDDEN], value=pos+1.0) ROW_MAJOR bf16
      3. slice_write src into dst at [0,0,pos,0]→[1,1,pos+1,HIDDEN]
      4. Read back each row via ttnn.slice, verify value
      5. ALSO test: convert dst to TILE_LAYOUT, slice from there, verify

    Gate: every row reads back its written value (allclose to expected scalar).

    Verdict:
      PASS → B.2.2 uses pre-alloc + slice_write (with ROW_MAJOR working buffer,
             convert to TILE_LAYOUT for batched matmul)
      FAIL → fall back to per-position list everywhere (no batched ops between
             sequential layers), or build a custom ttnn op
    """
    import ttnn
    import torch
    import numpy as np

    if state.mesh is None:
        return {"error": "mesh not loaded"}

    seq_len = int(args.get("seq_len", 5))
    HIDDEN = int(args.get("hidden", state.cfg['hidden']))
    if seq_len < 2:
        return {"error": f"seq_len must be >= 2, got {seq_len}"}

    # Step 1: pre-allocate dst [1, 1, seq_len, HIDDEN] ROW_MAJOR bf16 interleaved
    dst_init = torch.zeros((1, 1, seq_len, HIDDEN), dtype=torch.bfloat16)
    dst = ttnn.from_torch(
        dst_init,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    write_results = []
    write_error = None

    for pos in range(seq_len):
        # Step 2: src [1, 1, 1, HIDDEN] filled with (pos + 1.0)
        src_value = float(pos + 1)
        src_init = torch.full((1, 1, 1, HIDDEN), src_value, dtype=torch.bfloat16)
        src = ttnn.from_torch(
            src_init,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Step 3: slice_write src into dst at row `pos`
        # signature: slice_write(input, output, start, end, step) — step required
        try:
            ttnn.experimental.slice_write(
                src, dst,
                [0, 0, pos, 0],
                [1, 1, pos + 1, HIDDEN],
                [1, 1, 1, 1],
            )
            write_results.append({"pos": pos, "wrote_value": src_value, "ok": True})
        except Exception as e:
            write_error = f"slice_write failed at pos={pos}: {type(e).__name__}: {e}"
            write_results.append({"pos": pos, "wrote_value": src_value,
                                   "ok": False, "error": write_error})
            ttnn.deallocate(src)
            break
        ttnn.deallocate(src)

    if write_error:
        ttnn.deallocate(dst)
        return {
            "error": write_error,
            "write_results": write_results,
            "verdict": "FAIL — slice_write doesn't work as expected; fall back",
        }

    # Step 4: read back each row via slice (ROW_MAJOR slice)
    rowmajor_readback = []
    for pos in range(seq_len):
        try:
            row_tt = ttnn.slice(dst, [0, 0, pos, 0], [1, 1, pos + 1, HIDDEN])
            ttnn.synchronize_device(state.mesh)
            arr = ttnn.to_torch(
                row_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            )[0].float().cpu().numpy().reshape(HIDDEN)
            ttnn.deallocate(row_tt)
            expected = float(pos + 1)
            mean = float(arr.mean())
            min_v = float(arr.min())
            max_v = float(arr.max())
            ok = bool(np.allclose(arr, expected, atol=0.1))
            rowmajor_readback.append({
                "pos": pos, "expected": expected,
                "mean": mean, "min": min_v, "max": max_v, "ok": ok,
            })
        except Exception as e:
            rowmajor_readback.append({"pos": pos, "expected": float(pos+1),
                                       "error": f"{type(e).__name__}: {e}", "ok": False})

    # Step 5: convert dst to TILE_LAYOUT, slice from there, verify
    # (this is what we'd actually use in B.2.2 since matmul wants TILE)
    tile_readback = []
    tile_convert_error = None
    try:
        dst_tile = ttnn.to_layout(dst, ttnn.TILE_LAYOUT)
        for pos in range(seq_len):
            try:
                row_tt = ttnn.slice(dst_tile, [0, 0, pos, 0], [1, 1, pos + 1, HIDDEN])
                ttnn.synchronize_device(state.mesh)
                arr = ttnn.to_torch(
                    row_tt, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                )[0].float().cpu().numpy().reshape(HIDDEN)
                ttnn.deallocate(row_tt)
                expected = float(pos + 1)
                ok = bool(np.allclose(arr, expected, atol=0.1))
                tile_readback.append({
                    "pos": pos, "expected": expected,
                    "mean": float(arr.mean()),
                    "min": float(arr.min()), "max": float(arr.max()),
                    "ok": ok,
                })
            except Exception as e:
                tile_readback.append({"pos": pos, "expected": float(pos+1),
                                       "error": f"{type(e).__name__}: {e}", "ok": False})
        ttnn.deallocate(dst_tile)
    except Exception as e:
        tile_convert_error = f"to_layout TILE failed: {type(e).__name__}: {e}"

    ttnn.deallocate(dst)

    rowmajor_pass = all(r.get("ok", False) for r in rowmajor_readback)
    tile_pass = (tile_convert_error is None) and all(r.get("ok", False) for r in tile_readback)

    return {
        "ok": True,
        "seq_len": seq_len,
        "hidden": HIDDEN,
        "write_results": write_results,
        "rowmajor_readback": rowmajor_readback,
        "tile_readback": tile_readback,
        "tile_convert_error": tile_convert_error,
        "rowmajor_pass": rowmajor_pass,
        "tile_pass": tile_pass,
        "verdict": (
            "PASS — slice_write + ROW_MAJOR + to_layout(TILE) round-trip works. "
            "B.2.2 path: use pre-alloc ROW_MAJOR working buffer; slice_write per "
            "position; convert to TILE_LAYOUT for batched matmuls."
            if (rowmajor_pass and tile_pass) else
            f"PARTIAL: rowmajor_pass={rowmajor_pass} tile_pass={tile_pass}. "
            "Investigate further."
        ),
    }


def handle_probe_dn_source_isolation_tp(state: MeshServerState, args: dict) -> dict:
    """B.2.2 wedge isolation: which source tensor type wedges deltanet_step_tp?

    Tests deltanet_step_tp(state.layers[1]['dn'], x_pos) where x_pos is sliced
    from various source tensor types. Each test runs sequentially. If one
    wedges, the server dies — but the log will show which test was last to
    print "BEFORE DN". That's the wedger.

    Test order (most-likely-to-work → most-dangerous):
      1. from_torch upload (zeros) — fresh tensor, no compute lineage
      2. batched embed (already validated B.2.1.5a) — sanity check
      3. slice_write-assembled (B.2.1.5b primitive) — validated for readback
      4. linear (matmul) output — first computed tensor, no all_reduce
      5. all_reduce output — agent's hypothesized culprit
      6. ttnn.add of two fresh tensors — confirms add isn't the issue alone
      7. mlp_step_tp output — known wedger (skipped if we wedged earlier)

    Layer 1's DN state is reset between tests via _reset_state_buffers.
    """
    import ttnn
    import torch
    import numpy as np
    import time as _time

    if state.mesh is None or not state.layers:
        return {"error": "mesh/weights not loaded"}

    cfg = state.cfg
    HIDDEN = cfg['hidden']
    seq_len = int(args.get("seq_len", 5))

    # Find layer 1 (should be linear_attention per the i % 4 != 3 pattern)
    layer1 = None
    for layer in state.layers:
        if layer['type'] == 'linear_attention':
            layer1 = layer
            break
    if layer1 is None:
        return {"error": "no linear_attention layer found"}

    results = []

    def _sync():
        ttnn.synchronize_device(state.mesh)

    def _log(msg):
        print(f"  [DN-iso] {msg}", flush=True)

    def run_one_test(name, source_factory):
        _reset_state_buffers(state)
        update_input_buffers(state, 0, 0)  # cur_pos = 0
        _log(f"=== TEST '{name}' ===")
        try:
            x_seq = source_factory()
            _sync()
            _log(f"  {name}: source built, shape={list(x_seq.shape)}")
            x_pos = ttnn.slice(x_seq, [0, 0], [1, HIDDEN])
            _sync()
            _log(f"  {name}: sliced, shape={list(x_pos.shape)}")
            _log(f"  {name}: BEFORE deltanet_step_tp")
            t0 = _time.time()
            x_pos_out = deltanet_step_tp(state, x_pos, layer1['dn'], cfg)
            _sync()
            dt_ms = (_time.time() - t0) * 1000.0
            _log(f"  {name}: AFTER deltanet_step_tp dt={dt_ms:.0f}ms shape={list(x_pos_out.shape)}")
            ttnn.deallocate(x_pos_out)
            ttnn.deallocate(x_seq)
            return {"name": name, "result": "OK", "ms": dt_ms}
        except Exception as e:
            _log(f"  {name}: EXCEPTION: {type(e).__name__}: {e}")
            return {"name": name, "result": "ERROR", "error": f"{type(e).__name__}: {e}"}

    # ── Test 1: from_torch upload (fresh tensor) ──
    rng = np.random.default_rng(42)
    def src_from_torch():
        x_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
        return ttnn.from_torch(
            torch.from_numpy(x_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    results.append(run_one_test("from_torch", src_from_torch))

    # ── Test 2: batched embed output ──
    def src_embed():
        idx = ttnn.from_torch(
            torch.tensor([[t] for t in range(seq_len)], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        embed_raw = ttnn.embedding(
            idx, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_seq = ttnn.reshape(embed_raw, [seq_len, HIDDEN])
        ttnn.deallocate(idx)
        ttnn.deallocate(embed_raw)
        return x_seq
    results.append(run_one_test("embed", src_embed))

    # ── Test 3: slice_write-assembled buffer ──
    def src_slice_write():
        # Build [1, 1, seq_len, HIDDEN] via slice_write from per-position fresh tensors
        dst = ttnn.from_torch(
            torch.zeros((1, 1, seq_len, HIDDEN), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for pos in range(seq_len):
            x_np = rng.standard_normal((1, 1, 1, HIDDEN)).astype(np.float32) * 0.05
            src = ttnn.from_torch(
                torch.from_numpy(x_np).to(torch.bfloat16),
                layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                device=state.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.experimental.slice_write(
                src, dst,
                [0, 0, pos, 0], [1, 1, pos + 1, HIDDEN], [1, 1, 1, 1],
            )
            ttnn.deallocate(src)
        dst_tile_4d = ttnn.to_layout(dst, ttnn.TILE_LAYOUT)
        ttnn.deallocate(dst)
        x_seq = ttnn.reshape(dst_tile_4d, [seq_len, HIDDEN])
        ttnn.deallocate(dst_tile_4d)
        return x_seq
    results.append(run_one_test("slice_write", src_slice_write))

    # ── Test 4: ttnn.linear output (no all_reduce) ──
    def src_linear():
        # Fresh input × fresh weight = fresh matmul result
        x_in_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
        x_in = ttnn.from_torch(
            torch.from_numpy(x_in_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Use layer 1's input_norm weight as a "linear" — it's actually a vector
        # but ttnn.linear with a vector works? Probably not — use w_in instead.
        # Actually just create a square fresh weight for the test
        w_np = rng.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) * 0.02
        w_tt = ttnn.from_torch(
            torch.from_numpy(w_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_seq = ttnn.linear(x_in, w_tt)  # [seq_len, HIDDEN]
        ttnn.deallocate(x_in)
        ttnn.deallocate(w_tt)
        return x_seq
    results.append(run_one_test("linear", src_linear))

    # ── Test 5: all_reduce output directly ──
    def src_all_reduce():
        x_in_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
        x_in = ttnn.from_torch(
            torch.from_numpy(x_in_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_seq = _tp_all_reduce(state, x_in)
        ttnn.deallocate(x_in)
        return x_seq
    results.append(run_one_test("all_reduce", src_all_reduce))

    # ── Test 6: ttnn.add of two fresh tensors ──
    def src_add():
        a_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
        b_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
        a = ttnn.from_torch(
            torch.from_numpy(a_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            torch.from_numpy(b_np),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_seq = ttnn.add(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return x_seq
    results.append(run_one_test("add", src_add))

    # ── Test 7: full mlp_step_tp output (known wedger) ──
    # Find an MLP layer dict
    mlp_layer = state.layers[0]['mlp']
    def src_mlp():
        idx = ttnn.from_torch(
            torch.tensor([[t] for t in range(seq_len)], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )
        embed_raw = ttnn.embedding(
            idx, state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_seq = ttnn.reshape(embed_raw, [seq_len, HIDDEN])
        ttnn.deallocate(idx)
        ttnn.deallocate(embed_raw)
        x_seq_out = mlp_step_tp(state, x_seq, mlp_layer)
        ttnn.deallocate(x_seq)
        return x_seq_out
    results.append(run_one_test("mlp_step_tp", src_mlp))

    state.last_run = {
        "cmd": "probe_dn_source_isolation_tp",
        "seq_len": seq_len,
        "n_tests_completed": len(results),
        "results": results,
    }
    return {
        "ok": True,
        "seq_len": seq_len,
        "results": results,
        "note": (
            "If any test wedged, the server is now dead. Check the server log "
            "for the LAST 'BEFORE deltanet_step_tp' print without a matching "
            "'AFTER deltanet_step_tp' — that's the wedger."
        ),
    }


def handle_probe_dn_op_isolation_tp(state: MeshServerState, args: dict) -> dict:
    """B.2.2 deep isolation: which specific ttnn op wedges on all_reduce-slice input?

    Builds the "bad" source: slice of all_reduce output → x_pos [1, HIDDEN].
    Then tests each ttnn op individually on it. Logs BEFORE/AFTER per op.
    The last "BEFORE op X" without a matching "AFTER op X" identifies the
    wedger.

    Ordered LEAST-likely-to-wedge → MOST-likely so we get max info before
    server dies:
      1. ttnn.reshape (pure metadata op)
      2. ttnn.add (eltwise binary)
      3. ttnn.mul (eltwise binary)
      4. ttnn.exp (eltwise unary)
      5. ttnn.softplus (eltwise unary, used in DN)
      6. ttnn.linear (matmul — DN's 2nd op)
      7. ttnn.rms_norm — DN's FIRST op, suspected wedger
    """
    import ttnn
    import torch
    import numpy as np

    if state.mesh is None or not state.layers:
        return {"error": "mesh/weights not loaded"}

    HIDDEN = state.cfg['hidden']
    seq_len = int(args.get("seq_len", 5))

    # Build "bad" source: all_reduce output, sliced to [1, HIDDEN]
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((seq_len, HIDDEN)).astype(np.float32) * 0.05
    x_in = ttnn.from_torch(
        torch.from_numpy(x_np),
        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        device=state.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_seq = _tp_all_reduce(state, x_in)
    ttnn.deallocate(x_in)
    ttnn.synchronize_device(state.mesh)
    print(f"  [DN-op-iso] all_reduce output built, shape={list(x_seq.shape)}", flush=True)
    x_pos = ttnn.slice(x_seq, [0, 0], [1, HIDDEN])
    ttnn.synchronize_device(state.mesh)
    print(f"  [DN-op-iso] sliced, shape={list(x_pos.shape)}", flush=True)

    # B.2.2 FIX: previous probe used "first linear_attention layer" which is
    # layer index 0. But v3 wedges at LAYER 1 specifically. Test rms_norm +
    # linear with EXPLICIT layer indices (0, 1, 2 — all DN layers; 4, 5
    # also DN; layer 3 is full_attention).
    dn_indices_to_test = [0, 1, 2, 4]
    layer_weights = {}
    for idx in dn_indices_to_test:
        if idx >= len(state.layers):
            continue
        lyr = state.layers[idx]
        if lyr['type'] != 'linear_attention':
            continue
        layer_weights[idx] = lyr['dn']

    results = []

    def try_op(name, op_fn):
        print(f"  [DN-op-iso] BEFORE {name}", flush=True)
        ttnn.synchronize_device(state.mesh)
        try:
            out = op_fn()
            ttnn.synchronize_device(state.mesh)
            try:
                shape = list(out.shape) if hasattr(out, 'shape') else "no-shape"
            except Exception:
                shape = "shape-read-failed"
            print(f"  [DN-op-iso] AFTER {name} shape={shape}", flush=True)
            results.append({"op": name, "result": "OK"})
            if hasattr(out, 'shape'):
                try:
                    ttnn.deallocate(out)
                except Exception:
                    pass
        except Exception as e:
            print(f"  [DN-op-iso] EXCEPTION {name}: {type(e).__name__}: {e}", flush=True)
            results.append({"op": name, "result": "ERROR", "error": f"{type(e).__name__}: {e}"})

    # ── Generic ops (input only, no per-layer weight) ──
    try_op("reshape", lambda: ttnn.reshape(x_pos, [HIDDEN]))
    try_op("add", lambda: ttnn.add(x_pos, x_pos))
    try_op("mul", lambda: ttnn.mul(x_pos, x_pos))
    try_op("exp", lambda: ttnn.exp(x_pos))
    try_op("softplus", lambda: ttnn.softplus(x_pos))

    # ── Per-layer ops (test EACH DN layer's weights individually) ──
    for idx, dn in sorted(layer_weights.items()):
        try_op(f"linear_layer{idx}",
               lambda dn=dn: ttnn.linear(x_pos, dn['w_in']))
        try_op(f"rms_norm_layer{idx}",
               lambda dn=dn: ttnn.rms_norm(x_pos, weight=dn['input_norm'], epsilon=1e-6))

    ttnn.deallocate(x_pos)
    ttnn.deallocate(x_seq)

    state.last_run = {
        "cmd": "probe_dn_op_isolation_tp",
        "results": results,
    }
    return {"ok": True, "results": results, "seq_len": seq_len}


def _read_argmax_id(state: MeshServerState, argmax_tt) -> int:
    import ttnn
    idx_concat = ttnn.to_torch(
        argmax_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    )
    return int(idx_concat.cpu().numpy().reshape(-1)[0])


def _forward_argmax_id_with_cache_writer(state: MeshServerState, token_id: int,
                                         cur_pos: int, use_fused: bool,
                                         collective_mode: str | None = None,
                                         rope_mode: str | None = None,
                                         deltanet_decay_mode: str | None = None,
                                         deltanet_recurrence_mode: str | None = None) -> int:
    import ttnn
    old_fused = state.use_fused_paged_update
    old_collective = state.collective_mode
    old_rope = state.rope_mode
    old_decay = state.deltanet_decay_mode
    old_recurrence = state.deltanet_recurrence_mode
    state.use_fused_paged_update = use_fused
    if collective_mode is not None:
        state.collective_mode = collective_mode
    if rope_mode is not None:
        state.rope_mode = rope_mode
    if deltanet_decay_mode is not None:
        state.deltanet_decay_mode = deltanet_decay_mode
    if deltanet_recurrence_mode is not None:
        state.deltanet_recurrence_mode = deltanet_recurrence_mode
    try:
        _reset_state_buffers(state)
        update_input_buffers(state, token_id, cur_pos)
        out = forward_token_tp_inner(state)
        ttnn.synchronize_device(state.mesh)
        return _read_argmax_id(state, out)
    finally:
        state.use_fused_paged_update = old_fused
        state.collective_mode = old_collective
        state.rope_mode = old_rope
        state.deltanet_decay_mode = old_decay
        state.deltanet_recurrence_mode = old_recurrence
        _reset_state_buffers(state)


def _capture_temp_decode_trace(state: MeshServerState, use_fused: bool,
                               collective_mode: str | None = None,
                               rope_mode: str | None = None,
                               deltanet_decay_mode: str | None = None,
                               deltanet_recurrence_mode: str | None = None):
    """Capture an alternate decode trace for a probe, without replacing prod."""
    import ttnn
    old_fused = state.use_fused_paged_update
    old_collective = state.collective_mode
    old_rope = state.rope_mode
    old_decay = state.deltanet_decay_mode
    old_recurrence = state.deltanet_recurrence_mode
    state.use_fused_paged_update = use_fused
    if collective_mode is not None:
        state.collective_mode = collective_mode
    if rope_mode is not None:
        state.rope_mode = rope_mode
    if deltanet_decay_mode is not None:
        state.deltanet_decay_mode = deltanet_decay_mode
    if deltanet_recurrence_mode is not None:
        state.deltanet_recurrence_mode = deltanet_recurrence_mode
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
        state.rope_mode = old_rope
        state.deltanet_decay_mode = old_decay
        state.deltanet_recurrence_mode = old_recurrence
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


def handle_probe_ccl_equivalence_tp(state: MeshServerState, args: dict) -> dict:
    """B.2.2 CCL semantics probe: verify all_reduce ≟ composite ≟ custom on
    a known per-chip-different constant input.

    Builds per-chip constants via ShardTensorToMesh: chip i is filled with
    value (i+1.0). Ground-truth all-reduced output: 10.0 (1+2+3+4) on every
    element of every chip.

    Tests 3 CCL paths × N shapes (1-row control + multi-row prefill-like):
      R0: ttnn.all_reduce(partial, cluster_axis=1, num_links=2, Linear)
      R1: composite — reduce_scatter(dim=1) → all_gather(dim=1)
      R2: custom    — all_gather(dim=1) → reshape → sum(dim=1)

    Reports per (shape, path):
      - per-chip mean of result (should be 10.0 on every chip)
      - cosine vs ground truth
      - max_abs_diff vs ground truth (10.0)
      - sample value at [chip 0, pos 0, dim 0]
      - latency

    Answers definitively:
      - Does standalone all_reduce wedge on multi-row? (no downstream op)
      - Does composite RS+AG produce the correct sum?
      - Does custom AG+reshape+sum produce the correct sum?
      - Does the math/wedge behavior depend on seq dim being > 1?

    Order matters: composite + custom run BEFORE all_reduce per shape, so
    that if all_reduce wedges we still get the other paths' results from
    server-side prints. The whole request hangs on a wedge (no Python-level
    op timeout), but server log preserves what we learned.
    """
    import ttnn
    import torch
    import numpy as np
    import time as _time

    if state.mesh is None:
        return {"error": "mesh not loaded"}

    NCHIPS = 4
    H = int(args.get("hidden", 5120))
    if H % NCHIPS != 0:
        return {"error": f"H={H} must be divisible by NCHIPS={NCHIPS}"}
    shapes = args.get("shapes", [[1, H], [5, H]])
    topology_name = str(args.get("topology", "Linear"))
    if topology_name not in ("Linear", "Ring"):
        return {"error": f"topology must be 'Linear' or 'Ring', got {topology_name!r}"}
    topo = ttnn.Topology.Ring if topology_name == "Ring" else ttnn.Topology.Linear

    def upload_per_chip_constant(seq, H_):
        """Each chip i ends with [seq, H_] tensor filled with (i+1.0)."""
        arr = np.zeros((seq, NCHIPS * H_), dtype=np.float32)
        for i in range(NCHIPS):
            arr[:, i * H_:(i + 1) * H_] = float(i + 1)
        return ttnn.from_torch(
            torch.from_numpy(arr),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def readback_per_chip(tensor):
        """Returns torch [NCHIPS, seq, H_] = each chip's local data."""
        full = ttnn.to_torch(
            tensor, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
        ).float()
        seq, total = full.shape[-2], full.shape[-1]
        H_local = total // NCHIPS
        return full.view(seq, NCHIPS, H_local).permute(1, 0, 2).contiguous()

    def run_path(partial, path: str, seq: int, H_: int):
        if path == "all_reduce":
            return ttnn.all_reduce(
                partial, cluster_axis=1,
                memory_config=partial.memory_config(),
                num_links=2, topology=topo,
            )
        if path == "composite":
            scattered = ttnn.reduce_scatter(
                partial, dim=1, cluster_axis=1,
                num_links=2, topology=topo,
            )
            gathered = ttnn.all_gather(
                scattered, dim=1, cluster_axis=1,
                num_links=2, topology=topo,
            )
            ttnn.deallocate(scattered)
            return gathered
        if path == "custom":
            gathered = ttnn.all_gather(
                partial, dim=1, cluster_axis=1,
                num_links=2, topology=topo,
            )
            reshaped = ttnn.reshape(gathered, [seq, NCHIPS, H_])
            summed = ttnn.sum(reshaped, dim=1)
            ttnn.deallocate(gathered)
            ttnn.deallocate(reshaped)
            return summed
        raise ValueError(path)

    expected_value = float(sum(range(1, NCHIPS + 1)))  # 1+2+3+4 = 10
    results = {}
    for shape in shapes:
        seq, H_ = int(shape[0]), int(shape[1])
        shape_key = f"{seq}x{H_}"
        results[shape_key] = {}
        for path in ("composite", "custom", "all_reduce"):
            print(f"[ccl_eq] shape={shape_key} path={path} START", flush=True)
            try:
                partial = upload_per_chip_constant(seq, H_)
                t0 = _time.perf_counter()
                out = run_path(partial, path, seq, H_)
                ttnn.synchronize_device(state.mesh)
                ms = (_time.perf_counter() - t0) * 1000.0
                per_chip = readback_per_chip(out)
                expected = torch.full_like(per_chip, expected_value)
                max_abs_diff = float((per_chip - expected).abs().max())
                per_chip_means = [float(per_chip[c].mean()) for c in range(NCHIPS)]
                sample_val = float(per_chip[0, 0, 0])
                flat = per_chip.flatten()
                exp_flat = expected.flatten()
                cos = float(
                    (flat * exp_flat).sum()
                    / (flat.norm() * exp_flat.norm() + 1e-12)
                )
                ttnn.deallocate(partial)
                ttnn.deallocate(out)
                passed = max_abs_diff < 0.5
                results[shape_key][path] = {
                    "ok": True,
                    "ms": ms,
                    "expected_value": expected_value,
                    "per_chip_means": per_chip_means,
                    "sample_value_c0p0d0": sample_val,
                    "max_abs_diff": max_abs_diff,
                    "cosine_vs_expected": cos,
                    "passed": passed,
                }
                print(f"[ccl_eq] shape={shape_key} path={path} DONE "
                      f"means={per_chip_means} sample={sample_val:.4f} "
                      f"cos={cos:.6f} maxabs={max_abs_diff:.4f} "
                      f"{'PASS' if passed else 'FAIL'}", flush=True)
            except Exception as e:
                results[shape_key][path] = {
                    "ok": False,
                    "error": repr(e),
                }
                print(f"[ccl_eq] shape={shape_key} path={path} ERROR {e!r}",
                      flush=True)

    return {
        "ok": True,
        "expected_value": expected_value,
        "nchips": NCHIPS,
        "topology": topology_name,
        "results": results,
        "note": (
            "Per-chip-constant probe: chip i = (i+1.0). Expected all-reduced "
            "output is 10.0 everywhere on every chip. If per_chip_means == "
            "[10,10,10,10], math is correct. If means == [2.5,2.5,2.5,2.5], "
            "the op is averaging. If means differ across chips, op did not "
            "fully reduce. Compare composite/custom against all_reduce as "
            "ground truth (or against expected_value)."
        ),
    }


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


def _pcc_and_maxdiff(a, b):
    import numpy as np
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    flat_diff = diff.reshape(-1)
    max_idx = int(np.argmax(flat_diff)) if flat_diff.size else 0
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv) + 1e-12
    return {
        "pcc": float(np.dot(av, bv) / denom),
        "max_abs_diff": float(flat_diff[max_idx]) if flat_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(flat_diff)) if flat_diff.size else 0.0,
        "p99_abs_diff": float(np.quantile(flat_diff, 0.99)) if flat_diff.size else 0.0,
        "p999_abs_diff": float(np.quantile(flat_diff, 0.999)) if flat_diff.size else 0.0,
        "num_gt_0_001": int(np.count_nonzero(flat_diff > 0.001)),
        "num_gt_0_002": int(np.count_nonzero(flat_diff > 0.002)),
        "num_gt_0_004": int(np.count_nonzero(flat_diff > 0.004)),
        "numel": int(flat_diff.size),
        "max_index": max_idx,
        "max_values": {
            "a": float(a_flat[max_idx]) if flat_diff.size else 0.0,
            "b": float(b_flat[max_idx]) if flat_diff.size else 0.0,
        },
    }


def _ms_summary(samples):
    import numpy as np
    arr = np.array(samples, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def handle_probe_rope_fused_qk_tp(state: MeshServerState, args: dict) -> dict:
    """Check fused Q/K RoPE against the current Qwen partial-RoPE semantics.

    This is intentionally a compatibility/equivalence probe only. It uses
    production per-chip shapes on the resident qb2 mesh, but does not modify
    model state or capture a trace.
    """
    import numpy as np
    import torch
    import ttnn

    if state.mesh is None or state.cfg is None:
        return {"error": "server not fully loaded"}

    if not bool(args.get("allow_wedge_prone_fused_qk", False)):
        return {
            "candidate": "rotary_embedding_llama_fused_qk",
            "summary": {
                "all_positions_accepted": False,
                "pass_gate": False,
                "phase": "fused_qk_variant_disabled",
                "error": (
                    "The resident-server fused QK RoPE semantics probe wedged "
                    "qb2 for several minutes and required SIGTERM plus "
                    "tt-smi reset. Pass allow_wedge_prone_fused_qk=true only "
                    "for a coordinated isolation run."
                ),
            },
            "rows": [],
        }

    positions = [int(p) for p in args.get("positions", [0, 1, 7, 31, 32, 127, 255])]
    positions = [p for p in positions if 0 <= p < MAX_POS]
    if not positions:
        return {"error": "no valid positions"}

    cfg = state.cfg
    head_dim = cfg['head_dim']
    rotary_dim = state.rotary_dim
    half = rotary_dim // 2
    nq_per_chip = cfg['n_q_heads'] // state.mesh.get_num_devices()
    nkv_per_chip = cfg['n_kv_heads'] // state.mesh.get_num_devices()
    rng = np.random.default_rng(440)
    q_np = (rng.standard_normal((1, 1, nq_per_chip, head_dim)).astype(np.float32) * 0.1)
    k_np = (rng.standard_normal((1, 1, nkv_per_chip, head_dim)).astype(np.float32) * 0.1)

    def qwen_partial_rope_np(x, pos):
        rot = x[..., :rotary_dim]
        tail = x[..., rotary_dim:]
        cos = state.cos_all_np[pos].reshape((1, 1, 1, rotary_dim))
        sin = state.sin_all_np[pos].reshape((1, 1, 1, rotary_dim))
        rotated = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
        return np.concatenate([rot * cos + rotated * sin, tail], axis=-1).astype(np.float32)

    q_ref_by_pos = {p: qwen_partial_rope_np(q_np, p) for p in positions}
    k_ref_by_pos = {p: qwen_partial_rope_np(k_np, p) for p in positions}

    q_cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    })
    k_cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))
    })
    qk_cores = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))
    })
    q_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(q_cores, [TILE_HEIGHT, head_dim], ttnn.ShardOrientation.ROW_MAJOR))
    k_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(k_cores, [TILE_HEIGHT, head_dim], ttnn.ShardOrientation.ROW_MAJOR))
    cos_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(qk_cores, [TILE_HEIGHT, head_dim], ttnn.ShardOrientation.ROW_MAJOR))
    trans_mem_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(qk_cores, [TILE_HEIGHT, TILE_HEIGHT], ttnn.ShardOrientation.ROW_MAJOR))

    def upload(arr, mem_cfg):
        return ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
            dtype=ttnn.bfloat16,
            device=state.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    trans = np.zeros((1, 2, TILE_HEIGHT, TILE_HEIGHT), dtype=np.float32)
    trans[..., np.arange(0, TILE_HEIGHT, 2), np.arange(1, TILE_HEIGHT, 2)] = 1.0
    trans[..., np.arange(1, TILE_HEIGHT, 2), np.arange(0, TILE_HEIGHT, 2)] = -1.0

    q_tt = None
    k_tt = None
    trans_tt = None
    try:
        q_tt = upload(q_np, q_mem_cfg)
        k_tt = upload(k_np, k_mem_cfg)
        trans_tt = upload(trans, trans_mem_cfg)
        rows = []
        for pos in positions:
            cos_ext = np.concatenate([
                state.cos_all_np[pos],
                np.ones(head_dim - rotary_dim, dtype=np.float32),
            ]).astype(np.float32)
            sin_ext = np.concatenate([
                state.sin_all_np[pos],
                np.zeros(head_dim - rotary_dim, dtype=np.float32),
            ]).astype(np.float32)
            cos_np = np.tile(cos_ext.reshape(1, 1, 1, head_dim), (1, 2, TILE_HEIGHT, 1))
            sin_np = np.tile(sin_ext.reshape(1, 1, 1, head_dim), (1, 2, TILE_HEIGHT, 1))
            cos_tt = upload(cos_np, cos_mem_cfg)
            sin_tt = upload(sin_np, cos_mem_cfg)
            try:
                q_out, k_out = ttnn.experimental.rotary_embedding_llama_fused_qk(
                    q_tt, k_tt, cos_tt, sin_tt, trans_tt,
                    compute_kernel_config=state.sdpa_compute_kernel_config)
                ttnn.synchronize_device(state.mesh)
                q_host = ttnn.to_torch(
                    q_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                ).float().cpu().numpy()[:1].reshape(q_np.shape)
                k_host = ttnn.to_torch(
                    k_out, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                ).float().cpu().numpy()[:1].reshape(k_np.shape)
                q_tail = _pcc_and_maxdiff(q_host[..., rotary_dim:], q_np[..., rotary_dim:])
                k_tail = _pcc_and_maxdiff(k_host[..., rotary_dim:], k_np[..., rotary_dim:])
                rows.append({
                    "position": pos,
                    "accepted": True,
                    "q_vs_manual": _pcc_and_maxdiff(q_host, q_ref_by_pos[pos]),
                    "k_vs_manual": _pcc_and_maxdiff(k_host, k_ref_by_pos[pos]),
                    "q_tail_vs_input": q_tail,
                    "k_tail_vs_input": k_tail,
                })
                ttnn.deallocate(q_out)
                ttnn.deallocate(k_out)
            except Exception as e:
                rows.append({
                    "position": pos,
                    "accepted": False,
                    "error": f"{type(e).__name__}: {e}",
                })
            finally:
                ttnn.deallocate(cos_tt)
                ttnn.deallocate(sin_tt)

        accepted = [r for r in rows if r.get("accepted")]
        all_positions_accepted = len(accepted) == len(rows)
        min_q_pcc = min((r["q_vs_manual"]["pcc"] for r in accepted), default=None)
        min_k_pcc = min((r["k_vs_manual"]["pcc"] for r in accepted), default=None)
        max_tail_diff = max(
            ([r["q_tail_vs_input"]["max_abs_diff"] for r in accepted] +
             [r["k_tail_vs_input"]["max_abs_diff"] for r in accepted]),
            default=None,
        )
        pass_gate = (
            all_positions_accepted and
            min_q_pcc is not None and min_q_pcc >= 0.9999 and
            min_k_pcc is not None and min_k_pcc >= 0.9999 and
            max_tail_diff is not None and max_tail_diff <= 1e-2
        )
        result = {
            "candidate": "rotary_embedding_llama_fused_qk",
            "production_shape": {
                "q": list(q_np.shape),
                "k": list(k_np.shape),
                "head_dim": head_dim,
                "rotary_dim": rotary_dim,
                "nq_per_chip": nq_per_chip,
                "nkv_per_chip": nkv_per_chip,
            },
            "positions": positions,
            "op_count_removed_estimate": {
                "manual_rope_ops_per_full_attention_layer": 20,
                "full_attention_layers": sum(
                    1 for layer in state.layers if layer.get("type") == "full_attention"),
                "manual_rope_ops_per_token": 20 * sum(
                    1 for layer in state.layers if layer.get("type") == "full_attention"),
                "note": "Operation count only; not a speedup claim.",
            },
            "summary": {
                "all_positions_accepted": all_positions_accepted,
                "min_q_pcc": min_q_pcc,
                "min_k_pcc": min_k_pcc,
                "max_tail_diff": max_tail_diff,
                "pass_gate": pass_gate,
            },
            "rows": rows,
        }
        state.last_run = {
            "cmd": "probe_rope_fused_qk_tp",
            "pass_gate": pass_gate,
            "min_q_pcc": min_q_pcc,
            "min_k_pcc": min_k_pcc,
        }
        return result
    finally:
        for tensor in (q_tt, k_tt, trans_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)


def handle_probe_rope_native_partial_tp(state: MeshServerState, args: dict) -> dict:
    """Check slice-first native rotary_embedding against manual partial RoPE."""
    import numpy as np
    import torch
    import ttnn

    if state.mesh is None or state.cfg is None:
        return {"error": "server not fully loaded"}

    positions = [int(p) for p in args.get("positions", [0, 1, 7, 31, 32, 127, 255])]
    positions = [p for p in positions if 0 <= p < MAX_POS]
    if not positions:
        return {"error": "no valid positions"}

    cfg = state.cfg
    head_dim = cfg['head_dim']
    rotary_dim = state.rotary_dim
    half = rotary_dim // 2
    nq_per_chip = cfg['n_q_heads'] // state.mesh.get_num_devices()
    nkv_per_chip = cfg['n_kv_heads'] // state.mesh.get_num_devices()
    rng = np.random.default_rng(441)
    q_np = (rng.standard_normal((1, 1, nq_per_chip, head_dim)).astype(np.float32) * 0.1)
    k_np = (rng.standard_normal((1, 1, nkv_per_chip, head_dim)).astype(np.float32) * 0.1)

    def qwen_partial_rope_np(x, pos):
        rot = x[..., :rotary_dim]
        tail = x[..., rotary_dim:]
        cos = state.cos_all_np[pos].reshape((1, 1, 1, rotary_dim))
        sin = state.sin_all_np[pos].reshape((1, 1, 1, rotary_dim))
        rotated = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
        return np.concatenate([rot * cos + rotated * sin, tail], axis=-1).astype(np.float32)

    q_ref_by_pos = {p: qwen_partial_rope_np(q_np, p) for p in positions}
    k_ref_by_pos = {p: qwen_partial_rope_np(k_np, p) for p in positions}

    def upload(arr):
        return ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
            dtype=ttnn.bfloat16,
            device=state.mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    q_rot_tt = None
    q_tail_tt = None
    k_rot_tt = None
    k_tail_tt = None
    cos_tt = None
    sin_tt = None
    try:
        q_rot_tt = upload(q_np[..., :rotary_dim])
        q_tail_tt = upload(q_np[..., rotary_dim:])
        k_rot_tt = upload(k_np[..., :rotary_dim])
        k_tail_tt = upload(k_np[..., rotary_dim:])
        cos_cache = state.cos_all_np.reshape(1, 1, MAX_POS, rotary_dim)
        sin_cache = state.sin_all_np.reshape(1, 1, MAX_POS, rotary_dim)
        cos_tt = upload(cos_cache)
        sin_tt = upload(sin_cache)

        rows = []
        for pos in positions:
            try:
                q_native_rot = ttnn.experimental.rotary_embedding(
                    q_rot_tt, cos_tt, sin_tt,
                    token_index=pos,
                    compute_kernel_config=state.sdpa_compute_kernel_config)
                k_native_rot = ttnn.experimental.rotary_embedding(
                    k_rot_tt, cos_tt, sin_tt,
                    token_index=pos,
                    compute_kernel_config=state.sdpa_compute_kernel_config)
                # rotary_embedding pads the head axis to tile height. Trim back
                # to the production logical head count before reattaching tail.
                q_native_rot_trim = ttnn.slice(
                    q_native_rot, [0, 0, 0, 0], [1, 1, nq_per_chip, rotary_dim])
                k_native_rot_trim = ttnn.slice(
                    k_native_rot, [0, 0, 0, 0], [1, 1, nkv_per_chip, rotary_dim])
                q_native = ttnn.concat([q_native_rot_trim, q_tail_tt], dim=-1)
                k_native = ttnn.concat([k_native_rot_trim, k_tail_tt], dim=-1)
                ttnn.synchronize_device(state.mesh)
                q_host = ttnn.to_torch(
                    q_native, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                ).float().cpu().numpy()[:1].reshape(q_np.shape)
                k_host = ttnn.to_torch(
                    k_native, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
                ).float().cpu().numpy()[:1].reshape(k_np.shape)
                rows.append({
                    "position": pos,
                    "accepted": True,
                    "q_vs_manual": _pcc_and_maxdiff(q_host, q_ref_by_pos[pos]),
                    "k_vs_manual": _pcc_and_maxdiff(k_host, k_ref_by_pos[pos]),
                    "q_tail_vs_input": _pcc_and_maxdiff(q_host[..., rotary_dim:], q_np[..., rotary_dim:]),
                    "k_tail_vs_input": _pcc_and_maxdiff(k_host[..., rotary_dim:], k_np[..., rotary_dim:]),
                })
                for tensor in (q_native_rot, k_native_rot, q_native_rot_trim,
                               k_native_rot_trim, q_native, k_native):
                    ttnn.deallocate(tensor)
            except Exception as e:
                rows.append({
                    "position": pos,
                    "accepted": False,
                    "error": f"{type(e).__name__}: {e}",
                })

        accepted = [r for r in rows if r.get("accepted")]
        all_positions_accepted = len(accepted) == len(rows)
        min_q_pcc = min((r["q_vs_manual"]["pcc"] for r in accepted), default=None)
        min_k_pcc = min((r["k_vs_manual"]["pcc"] for r in accepted), default=None)
        max_tail_diff = max(
            ([r["q_tail_vs_input"]["max_abs_diff"] for r in accepted] +
             [r["k_tail_vs_input"]["max_abs_diff"] for r in accepted]),
            default=None,
        )
        pass_gate = (
            all_positions_accepted and
            min_q_pcc is not None and min_q_pcc >= 0.9999 and
            min_k_pcc is not None and min_k_pcc >= 0.9999 and
            max_tail_diff is not None and max_tail_diff <= 1e-2
        )
        result = {
            "candidate": "rotary_embedding_slice_first_partial",
            "production_shape": {
                "q": list(q_np.shape),
                "k": list(k_np.shape),
                "rotary_dim": rotary_dim,
                "tail_dim": head_dim - rotary_dim,
            },
            "positions": positions,
            "op_count_removed_estimate": {
                "manual_rope_ops_per_full_attention_layer": 20,
                "native_partial_ops_per_full_attention_layer": 6,
                "full_attention_layers": sum(
                    1 for layer in state.layers if layer.get("type") == "full_attention"),
                "note": "Operation count only; not a speedup claim.",
            },
            "summary": {
                "all_positions_accepted": all_positions_accepted,
                "min_q_pcc": min_q_pcc,
                "min_k_pcc": min_k_pcc,
                "max_tail_diff": max_tail_diff,
                "pass_gate": pass_gate,
            },
            "rows": rows,
        }
        state.last_run = {
            "cmd": "probe_rope_native_partial_tp",
            "pass_gate": pass_gate,
            "min_q_pcc": min_q_pcc,
            "min_k_pcc": min_k_pcc,
        }
        return result
    finally:
        for tensor in (q_rot_tt, q_tail_tt, k_rot_tt, k_tail_tt, cos_tt, sin_tt):
            if tensor is not None:
                ttnn.deallocate(tensor)


def handle_probe_rope_native_partial_trace_tp(state: MeshServerState, args: dict) -> dict:
    """Compare and time the trace-safe native partial RoPE production variant."""
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

    try:
        baseline_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="manual")
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
        native_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="native_partial")
    except Exception as e:
        state.rope_mode = "manual"
        _reset_state_buffers(state)
        return {
            "prompt": prompt,
            "prompt_ids": list(prompt_ids),
            "token_id": token_id,
            "compatibility": {
                "accepted": False,
                "phase": "native_forward",
                "baseline_argmax_id": baseline_id,
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_bench": None,
        }
    argmax_match = (baseline_id == native_id)

    result = {
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "token_id": token_id,
        "compatibility": {
            "accepted": True,
            "baseline_argmax_id": baseline_id,
            "native_argmax_id": native_id,
            "argmax_match": argmax_match,
            "mode": "native_partial_rope_row_no_token_index",
            "note": (
                "Argmax equality is a narrow production-forward check after "
                "resetting mutable state. The earlier tensor PCC probe is the "
                "stronger RoPE-local semantic gate."
            ),
        },
        "trace_bench": None,
    }

    trace_id = None
    try:
        t_capture0 = _time.perf_counter()
        trace_id, native_argmax_tt = _capture_temp_decode_trace(
            state, use_fused=False, collective_mode="baseline",
            rope_mode="native_partial")
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

        result["trace_bench"] = {
            "iters": iters,
            "warmup": warmup,
            "capture_ms": capture_ms,
            "summary_ms": {
                "native_execute_trace": _summary_ms(execute_ms),
                "native_update_plus_execute": _summary_ms(update_execute_ms),
            },
            "samples_ms": {
                "native_execute_trace": execute_ms,
                "native_update_plus_execute": update_execute_ms,
            },
            "traced_argmax_id_sample": _read_argmax_id(state, native_argmax_tt),
        }
    except Exception as e:
        result["trace_bench"] = {
            "accepted": False,
            "phase": "native_trace_or_bench",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(state.mesh, trace_id)
        state.rope_mode = "manual"
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False
        _reset_state_buffers(state)

    state.last_run = {
        "cmd": "probe_rope_native_partial_trace_tp",
        "argmax_match": argmax_match,
        "median_native_execute_ms": (
            result["trace_bench"]["summary_ms"]["native_execute_trace"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
        "median_native_combined_ms": (
            result["trace_bench"]["summary_ms"]["native_update_plus_execute"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
    }
    return result


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
    import types
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


def handle_probe_deltanet_recurrence_matmul_tp(state: MeshServerState, args: dict) -> dict:
    """Validate a no-rebuild matmul formulation of the DeltaNet recurrence."""
    import time as _time
    import numpy as np
    import torch
    import ttnn
    from full_layer_tp_probe import K_DIM, V_DIM, NV_PER_CHIP, VAL_DIM_CHIP

    if state.mesh is None or state.cfg is None:
        return {"error": "server not fully loaded"}

    iters = int(args.get("iters", 20))
    warmup = int(args.get("warmup", 3))
    if iters <= 0:
        return {"error": "iters must be > 0"}

    rng = np.random.default_rng(442)
    H_np = (rng.standard_normal((NV_PER_CHIP, K_DIM, V_DIM)).astype(np.float32) * 0.03)
    q_np = (rng.standard_normal((NV_PER_CHIP, K_DIM)).astype(np.float32) * 0.03)
    k_np = (rng.standard_normal((NV_PER_CHIP, K_DIM)).astype(np.float32) * 0.03)
    v_np = (rng.standard_normal((NV_PER_CHIP, V_DIM)).astype(np.float32) * 0.03)
    decay_np = np.exp(-np.abs(rng.standard_normal((NV_PER_CHIP,)).astype(np.float32)) * 0.05)
    beta_np = 1.0 / (1.0 + np.exp(-rng.standard_normal((NV_PER_CHIP,)).astype(np.float32)))

    def upload(arr):
        return ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
            dtype=ttnn.float32,
            device=state.mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    def recurrence_manual(H, q, k, v, decay, beta):
        H_4d = ttnn.reshape(H, [1, NV_PER_CHIP, K_DIM, V_DIM])
        H_decayed = ttnn.mul(H_4d, ttnn.reshape(decay, [1, NV_PER_CHIP, 1, 1]))
        k_col = ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])
        kv_mem = ttnn.reshape(ttnn.sum(ttnn.mul(H_decayed, k_col), dim=-2),
                              [1, NV_PER_CHIP, V_DIM])
        v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
        delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                         ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
        H_new = ttnn.add(
            H_decayed,
            ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])),
        )
        q_col = ttnn.reshape(q, [1, NV_PER_CHIP, K_DIM, 1])
        out = ttnn.reshape(ttnn.sum(ttnn.mul(H_new, q_col), dim=-2),
                           [1, VAL_DIM_CHIP])
        return H_new, out

    def recurrence_matmul(H, q, k, v, decay, beta):
        H_4d = ttnn.reshape(H, [1, NV_PER_CHIP, K_DIM, V_DIM])
        H_decayed = ttnn.mul(H_4d, ttnn.reshape(decay, [1, NV_PER_CHIP, 1, 1]))
        k_row = ttnn.reshape(k, [1, NV_PER_CHIP, 1, K_DIM])
        kv_mem = ttnn.reshape(
            ttnn.matmul(k_row, H_decayed,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.float32,
                        compute_kernel_config=state.sdpa_compute_kernel_config),
            [1, NV_PER_CHIP, V_DIM],
        )
        v_3d = ttnn.reshape(v, [1, NV_PER_CHIP, V_DIM])
        delta = ttnn.mul(ttnn.sub(v_3d, kv_mem),
                         ttnn.reshape(beta, [1, NV_PER_CHIP, 1]))
        k_col = ttnn.reshape(k, [1, NV_PER_CHIP, K_DIM, 1])
        outer = ttnn.matmul(
            k_col,
            ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM]),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.float32,
            compute_kernel_config=state.sdpa_compute_kernel_config,
        )
        H_new = ttnn.add(H_decayed, outer)
        q_row = ttnn.reshape(q, [1, NV_PER_CHIP, 1, K_DIM])
        out = ttnn.reshape(
            ttnn.matmul(q_row, H_new,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.float32,
                        compute_kernel_config=state.sdpa_compute_kernel_config),
            [1, VAL_DIM_CHIP],
        )
        return H_new, out

    H_tt = q_tt = k_tt = v_tt = decay_tt = beta_tt = None
    Hm = out_m = Hv = out_v = None
    try:
        H_tt = upload(H_np)
        q_tt = upload(q_np)
        k_tt = upload(k_np)
        v_tt = upload(v_np)
        decay_tt = upload(decay_np)
        beta_tt = upload(beta_np)

        Hm, out_m = recurrence_manual(H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt)
        ttnn.synchronize_device(state.mesh)
        try:
            Hv, out_v = recurrence_matmul(H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt)
            ttnn.synchronize_device(state.mesh)
            accepted = True
            error = None
        except Exception as e:
            accepted = False
            error = f"{type(e).__name__}: {e}"

        result = {
            "candidate": "deltanet_recurrence_matmul_form",
            "shape": {
                "state": [NV_PER_CHIP, K_DIM, V_DIM],
                "q": [NV_PER_CHIP, K_DIM],
                "k": [NV_PER_CHIP, K_DIM],
                "value": [NV_PER_CHIP, V_DIM],
            },
            "compatibility": {
                "accepted": accepted,
                "error": error,
            },
            "timing": None,
        }
        if accepted:
            Hm_host = ttnn.to_torch(
                Hm, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().cpu().numpy()[:1].reshape(1, NV_PER_CHIP, K_DIM, V_DIM)
            Hv_host = ttnn.to_torch(
                Hv, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().cpu().numpy()[:1].reshape(1, NV_PER_CHIP, K_DIM, V_DIM)
            out_m_host = ttnn.to_torch(
                out_m, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().cpu().numpy()[:1].reshape(1, VAL_DIM_CHIP)
            out_v_host = ttnn.to_torch(
                out_v, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
            ).float().cpu().numpy()[:1].reshape(1, VAL_DIM_CHIP)
            state_cmp = _pcc_and_maxdiff(Hv_host, Hm_host)
            out_cmp = _pcc_and_maxdiff(out_v_host, out_m_host)
            pass_gate = (
                state_cmp["pcc"] >= 0.9999 and out_cmp["pcc"] >= 0.9999 and
                state_cmp["max_abs_diff"] <= 1e-2 and out_cmp["max_abs_diff"] <= 1e-2
            )

            def sync():
                ttnn.synchronize_device(state.mesh)

            def timed_call(fn):
                sync()
                t0 = _time.perf_counter()
                tmp_H, tmp_out = fn(H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt)
                sync()
                dt = (_time.perf_counter() - t0) * 1000.0
                ttnn.deallocate(tmp_H)
                ttnn.deallocate(tmp_out)
                return dt

            for _ in range(warmup):
                timed_call(recurrence_manual)
                timed_call(recurrence_matmul)
            manual_ms = [timed_call(recurrence_manual) for _ in range(iters)]
            matmul_ms = [timed_call(recurrence_matmul) for _ in range(iters)]
            result["correctness"] = {
                "state_vs_manual": state_cmp,
                "out_vs_manual": out_cmp,
                "pass_gate": pass_gate,
            }
            result["timing"] = {
                "iters": iters,
                "warmup": warmup,
                "manual_ms": _summary_ms(manual_ms),
                "matmul_ms": _summary_ms(matmul_ms),
                "samples_ms": {
                    "manual": manual_ms,
                    "matmul": matmul_ms,
                },
                "note": "Synthetic recurrence body only; not trace or full-decode timing.",
            }
            state.last_run = {
                "cmd": "probe_deltanet_recurrence_matmul_tp",
                "accepted": True,
                "pass_gate": pass_gate,
                "manual_median_ms": result["timing"]["manual_ms"].get("median"),
                "matmul_median_ms": result["timing"]["matmul_ms"].get("median"),
            }
        return result
    finally:
        for tensor in (Hm, out_m, Hv, out_v, H_tt, q_tt, k_tt, v_tt, decay_tt, beta_tt):
            if tensor is not None:
                try:
                    ttnn.deallocate(tensor)
                except Exception:
                    pass


def handle_probe_deltanet_native_gdn_real_tensors_tp(state: MeshServerState, args: dict) -> dict:
    """Validate native Qwen36 GDN recurrence on real resident server tensors."""
    import numpy as np
    import torch
    import ttnn
    from full_layer_tp_probe import (
        K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )

    if state.mesh is None or state.cfg is None or not state.layers:
        return {"error": "server not fully loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode is not exposed"}

    prompt = args.get("prompt", "The capital of France is")
    reset_state = bool(args.get("reset_state", True))
    layer_idx = int(args.get("layer_idx", 0))
    mode = args.get("mode", "fp32_cast")
    if mode not in ("fp32_cast", "current_dtype"):
        return {"error": f"unsupported mode {mode!r}"}
    if layer_idx < 0 or layer_idx >= len(state.layers):
        return {"error": f"layer_idx out of range: {layer_idx}"}
    if state.layers[layer_idx]["type"] != "linear_attention":
        return {"error": f"layer_idx {layer_idx} is not a linear_attention layer"}

    if reset_state:
        _reset_state_buffers(state)

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    token_ids = state.tok.encode(prompt, add_special_tokens=False)
    if not token_ids:
        return {"error": "prompt produced no token ids"}
    token_id = int(token_ids[0])
    dn = state.layers[layer_idx]["dn"]

    tensors = []

    def remember(t):
        tensors.append(t)
        return t

    def host(tensor):
        return ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        ).float().cpu().numpy()

    try:
        tok_tt = remember(ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        ))
        embed_out = remember(ttnn.embedding(
            tok_tt,
            state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ))
        x_tt = remember(ttnn.reshape(embed_out, [1, HIDDEN]))

        h_tt = remember(_rms_norm_manual(x_tt, dn["input_norm"], EPS, HIDDEN))
        all_tt = remember(ttnn.linear(h_tt, dn["w_in"]))
        mixed_qkv = remember(ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP]))
        a_tt = remember(ttnn.slice(
            all_tt,
            [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
            [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
        ))
        b_tt = remember(ttnn.slice(
            all_tt,
            [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
            [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP],
        ))

        mixed_col = remember(ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1]))
        conv_input = remember(ttnn.concat([dn["conv_st"], mixed_col], dim=-1))
        conv_prod = remember(ttnn.mul(conv_input, dn["w_conv"]))
        conv_out = remember(ttnn.silu(ttnn.sum(conv_prod, dim=-1)))

        q_flat = remember(ttnn.slice(conv_out, [0], [KEY_DIM_CHIP]))
        k_flat = remember(ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP]))
        v_flat = remember(ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP]))

        def gqa(t, n_kh, d):
            t2 = remember(ttnn.reshape(t, [n_kh, 1, d]))
            t3 = remember(ttnn.repeat(t2, ttnn.Shape([1, N_REP, 1])))
            return remember(ttnn.reshape(t3, [n_kh * N_REP, d]))

        q = gqa(q_flat, NK_PER_CHIP, K_DIM)
        k = gqa(k_flat, NK_PER_CHIP, K_DIM)
        v = remember(ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM]))

        EPS_RMS = EPS / K_DIM
        q = remember(_rms_norm_manual(q, dn["q_l2_scale"], EPS_RMS, K_DIM))
        k = remember(_rms_norm_manual(k, dn["k_l2_scale"], EPS_RMS, K_DIM))

        a_biased = remember(ttnn.add(a_tt, dn["dt_bias"]))
        if state.deltanet_decay_mode == "native_softplus":
            softplus_a = remember(ttnn.softplus(a_biased))
        else:
            softplus_a = remember(ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0)))
        g = remember(ttnn.mul(ttnn.neg(ttnn.exp(dn["A_log"])), softplus_a))
        beta = remember(ttnn.sigmoid(b_tt))
        decay = remember(ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1]))

        # Do not register this for cleanup: it is a view of persistent dn["ssm"].
        H_4d = ttnn.reshape(dn["ssm"], [1, NV_PER_CHIP, K_DIM, V_DIM])
        q4 = remember(ttnn.reshape(q, [1, NV_PER_CHIP, 1, K_DIM]))
        k4 = remember(ttnn.reshape(k, [1, NV_PER_CHIP, 1, K_DIM]))
        v4 = remember(ttnn.reshape(v, [1, NV_PER_CHIP, 1, V_DIM]))
        beta4 = remember(ttnn.reshape(beta, [1, NV_PER_CHIP, 1, 1]))

        if mode == "fp32_cast":
            H_in = remember(ttnn.typecast(H_4d, ttnn.float32))
            q_in = remember(ttnn.typecast(q4, ttnn.float32))
            k_in = remember(ttnn.typecast(k4, ttnn.float32))
            v_in = remember(ttnn.typecast(v4, ttnn.float32))
            decay_in = remember(ttnn.typecast(decay, ttnn.float32))
            beta_in = remember(ttnn.typecast(beta4, ttnn.float32))
        else:
            H_in = remember(ttnn.add(H_4d, 0.0))
            q_in = q4
            k_in = k4
            v_in = v4
            decay_in = decay
            beta_in = beta4

        state_scaled = remember(ttnn.mul(H_in, decay_in))
        k_col = remember(ttnn.reshape(k_in, [1, NV_PER_CHIP, K_DIM, 1]))
        prediction = remember(ttnn.reshape(
            ttnn.sum(ttnn.mul(state_scaled, k_col), dim=-2),
            [1, NV_PER_CHIP, 1, V_DIM],
        ))
        delta = remember(ttnn.mul(ttnn.sub(v_in, prediction), beta_in))
        H_manual = remember(ttnn.add(
            state_scaled,
            ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])),
        ))
        q_col = remember(ttnn.reshape(q_in, [1, NV_PER_CHIP, K_DIM, 1]))
        out_manual = remember(ttnn.reshape(
            ttnn.sum(ttnn.mul(H_manual, q_col), dim=-2),
            [1, NV_PER_CHIP, 1, V_DIM],
        ))

        # qwen36_gdn_decode returns the input state tensor after updating it.
        # Use a copied state so this probe cannot mutate dn["ssm"].
        H_native_in = remember(ttnn.add(H_in, 0.0))
        try:
            H_native, out_native = ttnn.experimental.qwen36_gdn_decode(
                H_native_in,
                q_in,
                k_in,
                v_in,
                decay_in,
                beta_in,
                normalize_qk_l2=False,
                output_memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            native_accepted = True
            native_error = None
            tensors.append(out_native)
        except Exception as e:
            native_accepted = False
            native_error = f"{type(e).__name__}: {e}"
            H_native = None
            out_native = None

        result = {
            "candidate": "qwen36_gdn_decode_real_tensors",
            "mode": mode,
            "prompt": prompt,
            "token_id": token_id,
            "layer_idx": layer_idx,
            "reset_state": reset_state,
            "symbols": {
                "qwen36_gdn_prepare_decode": hasattr(ttnn.experimental, "qwen36_gdn_prepare_decode"),
                "qwen36_gdn_decode": hasattr(ttnn.experimental, "qwen36_gdn_decode"),
            },
            "shape": {
                "state": [1, NV_PER_CHIP, K_DIM, V_DIM],
                "qkv": [1, NV_PER_CHIP, 1, K_DIM],
                "alpha_beta": [1, NV_PER_CHIP, 1, 1],
            },
            "compatibility": {
                "accepted": native_accepted,
                "error": native_error,
            },
        }
        if native_accepted:
            ttnn.synchronize_device(state.mesh)
            Hm_host = host(H_manual)
            Hn_host = host(H_native)
            out_m_raw = host(out_manual)
            out_n_raw = host(out_native)
            out_leading = out_m_raw.size // (NV_PER_CHIP * V_DIM)
            out_native_tile_rows = out_n_raw.size // (out_leading * NV_PER_CHIP * V_DIM)
            out_m_host = out_m_raw.reshape(out_leading, NV_PER_CHIP, 1, V_DIM)
            out_n_host = out_n_raw.reshape(
                out_leading, NV_PER_CHIP, out_native_tile_rows, V_DIM
            )[:, :, :1, :]
            state_cmp = _pcc_and_maxdiff(Hn_host, Hm_host)
            out_cmp = _pcc_and_maxdiff(out_n_host, out_m_host)
            pass_gate = (
                state_cmp["pcc"] >= 0.9999 and
                out_cmp["pcc"] >= 0.9999 and
                state_cmp["max_abs_diff"] <= 1e-2 and
                out_cmp["max_abs_diff"] <= 1e-2
            )
            result["correctness"] = {
                "state_vs_manual": state_cmp,
                "output_vs_manual": out_cmp,
                "pass_gate": pass_gate,
                "host_shapes": {
                    "state_manual": list(Hm_host.shape),
                    "state_native": list(Hn_host.shape),
                    "output_manual": list(out_m_host.shape),
                    "output_native": list(out_n_host.shape),
                },
            }
            state.last_run = {
                "cmd": "probe_deltanet_native_gdn_real_tensors_tp",
                "accepted": native_accepted,
                "pass_gate": pass_gate,
                "mode": mode,
            }
        return result
    finally:
        seen = set()
        for tensor in reversed(tensors):
            if tensor is None:
                continue
            ident = id(tensor)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass


def handle_probe_deltanet_owned_gdn_real_tensors_tp(state: MeshServerState, args: dict) -> dict:
    """Validate the owned GDN decode op on real resident DeltaNet tensors.

    This intentionally runs inside the persistent server process. It does not
    open devices directly and it does not mutate the resident SSM state.
    """
    import torch
    import ttnn
    from full_layer_tp_probe import (
        K_DIM, V_DIM, CONV_DIM_CHIP, KEY_DIM_CHIP, VAL_DIM_CHIP,
        NK_PER_CHIP, NV_PER_CHIP, N_REP, EPS,
    )

    if state.mesh is None or state.cfg is None or not state.layers:
        return {"error": "server not fully loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode_owned is not exposed"}

    prompt = args.get("prompt", "The capital of France is")
    reset_state = bool(args.get("reset_state", True))
    layer_idx = int(args.get("layer_idx", 0))
    use_pretransposed_k = bool(args.get("use_pretransposed_k", False))
    compact_vectors = bool(args.get("compact_vectors", False))
    native_io = bool(args.get("native_io", False))
    stepwise = bool(args.get("stepwise", False))
    seed_state = args.get("seed_state", "resident")
    direct_state_input = bool(args.get("direct_state_input", False))
    component_debug_modes = [int(x) for x in (args.get("component_debug_modes") or [])]
    allowed_component_debug_modes = {2, 10, 11, 12}
    bad_component_debug_modes = sorted(set(component_debug_modes) - allowed_component_debug_modes)
    if bad_component_debug_modes:
        return {"error": f"unsupported component_debug_modes {bad_component_debug_modes}"}
    if seed_state not in ("resident", "manual_once"):
        return {"error": f"unsupported seed_state {seed_state!r}"}
    if direct_state_input and seed_state != "manual_once":
        return {"error": "direct_state_input is only safe with seed_state='manual_once'"}
    if layer_idx < 0 or layer_idx >= len(state.layers):
        return {"error": f"layer_idx out of range: {layer_idx}"}
    if state.layers[layer_idx]["type"] != "linear_attention":
        return {"error": f"layer_idx {layer_idx} is not a linear_attention layer"}

    if (compact_vectors or native_io) and use_pretransposed_k:
        return {"error": "compact_vectors/native_io currently do not support use_pretransposed_k"}

    if reset_state:
        _reset_state_buffers(state)

    cfg = state.cfg
    HIDDEN = cfg["hidden"]
    token_ids = state.tok.encode(prompt, add_special_tokens=False)
    if not token_ids:
        return {"error": "prompt produced no token ids"}
    token_id = int(token_ids[0])
    dn = state.layers[layer_idx]["dn"]

    tensors = []

    def remember(t):
        tensors.append(t)
        return t

    def host(tensor):
        return ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        ).float().cpu().numpy()

    try:
        tok_tt = remember(ttnn.from_torch(
            torch.tensor([[token_id]], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        ))
        embed_out = remember(ttnn.embedding(
            tok_tt,
            state.embed_tt,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ))
        x_tt = remember(ttnn.reshape(embed_out, [1, HIDDEN]))

        h_tt = remember(_rms_norm_manual(x_tt, dn["input_norm"], EPS, HIDDEN))
        all_tt = remember(ttnn.linear(h_tt, dn["w_in"]))
        mixed_qkv = remember(ttnn.slice(all_tt, [0, 0], [1, CONV_DIM_CHIP]))
        a_tt = remember(ttnn.slice(
            all_tt,
            [0, CONV_DIM_CHIP + VAL_DIM_CHIP],
            [1, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
        ))
        b_tt = remember(ttnn.slice(
            all_tt,
            [0, CONV_DIM_CHIP + VAL_DIM_CHIP + NV_PER_CHIP],
            [1, CONV_DIM_CHIP + VAL_DIM_CHIP + 2 * NV_PER_CHIP],
        ))

        mixed_col = remember(ttnn.reshape(mixed_qkv, [CONV_DIM_CHIP, 1]))
        conv_input = remember(ttnn.concat([dn["conv_st"], mixed_col], dim=-1))
        conv_prod = remember(ttnn.mul(conv_input, dn["w_conv"]))
        conv_out = remember(ttnn.silu(ttnn.sum(conv_prod, dim=-1)))

        q_flat = remember(ttnn.slice(conv_out, [0], [KEY_DIM_CHIP]))
        k_flat = remember(ttnn.slice(conv_out, [KEY_DIM_CHIP], [2 * KEY_DIM_CHIP]))
        v_flat = remember(ttnn.slice(conv_out, [2 * KEY_DIM_CHIP], [CONV_DIM_CHIP]))

        def gqa(t, n_kh, d):
            t2 = remember(ttnn.reshape(t, [n_kh, 1, d]))
            t3 = remember(ttnn.repeat(t2, ttnn.Shape([1, N_REP, 1])))
            return remember(ttnn.reshape(t3, [n_kh * N_REP, d]))

        q = gqa(q_flat, NK_PER_CHIP, K_DIM)
        k = gqa(k_flat, NK_PER_CHIP, K_DIM)
        v = remember(ttnn.reshape(v_flat, [NV_PER_CHIP, V_DIM]))

        EPS_RMS = EPS / K_DIM
        q = remember(_rms_norm_manual(q, dn["q_l2_scale"], EPS_RMS, K_DIM))
        k = remember(_rms_norm_manual(k, dn["k_l2_scale"], EPS_RMS, K_DIM))

        a_biased = remember(ttnn.add(a_tt, dn["dt_bias"]))
        if state.deltanet_decay_mode == "native_softplus":
            softplus_a = remember(ttnn.softplus(a_biased))
        else:
            softplus_a = remember(ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0)))
        g = remember(ttnn.mul(ttnn.neg(ttnn.exp(dn["A_log"])), softplus_a))
        beta = remember(ttnn.sigmoid(b_tt))
        decay = remember(ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1]))

        # Do not register this for cleanup: this is persistent resident state.
        H_4d = dn["ssm"]
        q4 = remember(ttnn.reshape(q, [1, NV_PER_CHIP, 1, K_DIM]))
        k4 = remember(ttnn.reshape(k, [1, NV_PER_CHIP, 1, K_DIM]))
        v4 = remember(ttnn.reshape(v, [1, NV_PER_CHIP, 1, V_DIM]))
        beta4 = remember(ttnn.reshape(beta, [1, NV_PER_CHIP, 1, 1]))

        H_input = H_4d
        if seed_state == "manual_once":
            seed_state_scaled = remember(ttnn.mul(H_4d, decay))
            seed_k_col = remember(ttnn.reshape(k4, [1, NV_PER_CHIP, K_DIM, 1]))
            seed_prediction = remember(ttnn.reshape(
                ttnn.sum(ttnn.mul(seed_state_scaled, seed_k_col), dim=-2),
                [1, NV_PER_CHIP, 1, V_DIM],
            ))
            seed_delta = remember(ttnn.mul(ttnn.sub(v4, seed_prediction), beta4))
            H_input = remember(ttnn.add(
                seed_state_scaled,
                ttnn.mul(seed_k_col, ttnn.reshape(seed_delta, [1, NV_PER_CHIP, 1, V_DIM])),
            ))

        # Manual reference in the same dtype/layout path as production.
        state_scaled = remember(ttnn.mul(H_input, decay))
        k_col = remember(ttnn.reshape(k4, [1, NV_PER_CHIP, K_DIM, 1]))
        prediction = remember(ttnn.reshape(
            ttnn.sum(ttnn.mul(state_scaled, k_col), dim=-2),
            [1, NV_PER_CHIP, 1, V_DIM],
        ))
        delta = remember(ttnn.mul(ttnn.sub(v4, prediction), beta4))
        H_manual = remember(ttnn.add(
            state_scaled,
            ttnn.mul(k_col, ttnn.reshape(delta, [1, NV_PER_CHIP, 1, V_DIM])),
        ))
        q_col = remember(ttnn.reshape(q4, [1, NV_PER_CHIP, K_DIM, 1]))
        out_manual = remember(ttnn.reshape(
            ttnn.sum(ttnn.mul(H_manual, q_col), dim=-2),
            [1, NV_PER_CHIP, 1, V_DIM],
        ))
        matmul_reference_error = None
        prediction_matmul = None
        H_matmul = None
        out_matmul = None
        if stepwise:
            try:
                prediction_matmul = remember(ttnn.reshape(
                    ttnn.matmul(
                        k4,
                        state_scaled,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.float32,
                        compute_kernel_config=state.sdpa_compute_kernel_config,
                    ),
                    [1, NV_PER_CHIP, 1, V_DIM],
                ))
                delta_matmul = remember(ttnn.mul(ttnn.sub(v4, prediction_matmul), beta4))
                outer_matmul = remember(ttnn.matmul(
                    k_col,
                    delta_matmul,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.float32,
                    compute_kernel_config=state.sdpa_compute_kernel_config,
                ))
                H_matmul = remember(ttnn.add(state_scaled, outer_matmul))
                out_matmul = remember(ttnn.reshape(
                    ttnn.matmul(
                        q4,
                        H_matmul,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.float32,
                        compute_kernel_config=state.sdpa_compute_kernel_config,
                    ),
                    [1, NV_PER_CHIP, 1, V_DIM],
                ))
            except Exception as e:
                matmul_reference_error = f"{type(e).__name__}: {e}"

        # Owned op contract variants. The default copy path preserves resident
        # state. direct_state_input is a diagnostic for the temporary seeded
        # state path only, used to separate copy drift from kernel drift.
        H_owned_copy_for_compare = None
        if stepwise:
            H_owned_copy_for_compare = remember(ttnn.add(H_input, 0.0))
        H_owned_in = H_input if direct_state_input else remember(ttnn.add(H_input, 0.0))
        if native_io:
            q_rows = q4
            k_rows = k4
            v_rows = v4
            decay_tiles = decay
            beta_tiles = beta4
        elif compact_vectors:
            q_rows = remember(ttnn.pad(q4, [[0, 0], [0, 0], [0, 31], [0, 0]], value=0.0))
            k_rows = remember(ttnn.pad(k4, [[0, 0], [0, 0], [0, 31], [0, 0]], value=0.0))
            v_rows = remember(ttnn.pad(v4, [[0, 0], [0, 0], [0, 31], [0, 0]], value=0.0))
            decay_tiles = remember(ttnn.repeat(decay, ttnn.Shape([1, 1, 32, 32])))
            beta_tiles = remember(ttnn.repeat(beta4, ttnn.Shape([1, 1, 32, 32])))
        else:
            q_rows = remember(ttnn.repeat(q4, ttnn.Shape([1, 1, 32, 1])))
            k_rows = remember(ttnn.repeat(k4, ttnn.Shape([1, 1, 32, 1])))
            v_rows = remember(ttnn.repeat(v4, ttnn.Shape([1, 1, 32, 1])))
            decay_tiles = remember(ttnn.repeat(decay, ttnn.Shape([1, 1, 32, 32])))
            beta_tiles = remember(ttnn.repeat(beta4, ttnn.Shape([1, 1, 32, 32])))
        k_col_tiles = None
        if use_pretransposed_k:
            k_col_tiles = remember(ttnn.repeat(k_col, ttnn.Shape([1, 1, 1, 32])))

        try:
            if use_pretransposed_k:
                H_owned, out_owned = ttnn.experimental.qwen36_gdn_decode_owned(
                    H_owned_in,
                    q_rows,
                    k_rows,
                    v_rows,
                    decay_tiles,
                    beta_tiles,
                    k_col=k_col_tiles,
                    compact_vectors=compact_vectors,
                    native_io=native_io,
                    output_memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                H_owned, out_owned = ttnn.experimental.qwen36_gdn_decode_owned(
                    H_owned_in,
                    q_rows,
                    k_rows,
                    v_rows,
                    decay_tiles,
                    beta_tiles,
                    compact_vectors=compact_vectors,
                    native_io=native_io,
                    output_memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            owned_accepted = True
            owned_error = None
            tensors.append(out_owned)
        except Exception as e:
            owned_accepted = False
            owned_error = f"{type(e).__name__}: {e}"
            H_owned = None
            out_owned = None

        result = {
            "candidate": "qwen36_gdn_decode_owned_real_tensors",
            "prompt": prompt,
            "token_id": token_id,
            "layer_idx": layer_idx,
            "reset_state": reset_state,
            "use_pretransposed_k": use_pretransposed_k,
            "compact_vectors": compact_vectors,
                "native_io": native_io,
                "seed_state": seed_state,
                "direct_state_input": direct_state_input,
                "symbols": {
                "qwen36_gdn_decode_owned": hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"),
            },
            "shape": {
                "state": [1, NV_PER_CHIP, K_DIM, V_DIM],
                "q_k_rows": [1, NV_PER_CHIP, 1 if native_io else 32, K_DIM],
                "value_rows": [1, NV_PER_CHIP, 1 if native_io else 32, V_DIM],
                "alpha_beta": [1, NV_PER_CHIP, 1 if native_io else 32, 1 if native_io else 32],
                "out_manual": [1, NV_PER_CHIP, 1, V_DIM],
                "out_owned": [1, VAL_DIM_CHIP] if native_io else [1, NV_PER_CHIP, 32, V_DIM],
            },
            "compatibility": {
                "accepted": owned_accepted,
                "error": owned_error,
            },
        }
        if owned_accepted:
            ttnn.synchronize_device(state.mesh)
            Hm_host = host(H_manual)
            Ho_host = host(H_owned)
            out_m_raw = host(out_manual)
            out_o_raw = host(out_owned)
            out_leading = out_m_raw.size // (NV_PER_CHIP * V_DIM)
            out_m_host = out_m_raw.reshape(out_leading, NV_PER_CHIP, 1, V_DIM)
            out_owned_tile_rows = out_o_raw.size // (out_leading * NV_PER_CHIP * V_DIM)
            out_o_host = out_o_raw.reshape(
                out_leading, NV_PER_CHIP, out_owned_tile_rows, V_DIM
            )[:, :, :1, :]
            state_cmp = _pcc_and_maxdiff(Ho_host, Hm_host)
            out_cmp = _pcc_and_maxdiff(out_o_host, out_m_host)
            pass_gate = (
                state_cmp["pcc"] >= 0.9999 and
                out_cmp["pcc"] >= 0.9999 and
                state_cmp["max_abs_diff"] <= 0.0015 and
                out_cmp["max_abs_diff"] <= 0.0015
            )
            result["correctness"] = {
                "state_vs_manual": state_cmp,
                "output_vs_manual": out_cmp,
                "pass_gate": pass_gate,
                "host_shapes": {
                    "state_manual": list(Hm_host.shape),
                    "state_owned": list(Ho_host.shape),
                    "output_manual": list(out_m_host.shape),
                    "output_owned": list(out_o_host.shape),
                },
            }
            if stepwise:
                import numpy as np

                def compare(name, actual, expected):
                    cmp = _pcc_and_maxdiff(actual, expected)
                    exact_or_tiny = (
                        cmp["max_abs_diff"] <= 0.0015 and
                        cmp["mean_abs_diff"] <= 0.0015
                    )
                    cmp["pass_gate"] = (
                        exact_or_tiny and (cmp["pcc"] >= 0.9999 or cmp["max_abs_diff"] == 0.0)
                    )
                    cmp["name"] = name
                    return cmp

                def owned_debug(debug_mode):
                    H_dbg_in = remember(ttnn.add(H_input, 0.0))
                    kwargs = {
                        "compact_vectors": compact_vectors,
                        "native_io": native_io,
                        "output_memory_config": ttnn.L1_MEMORY_CONFIG,
                        "debug_mode": debug_mode,
                    }
                    if use_pretransposed_k:
                        kwargs["k_col"] = k_col_tiles
                    H_dbg, out_dbg = ttnn.experimental.qwen36_gdn_decode_owned(
                        H_dbg_in,
                        q_rows,
                        k_rows,
                        v_rows,
                        decay_tiles,
                        beta_tiles,
                        **kwargs,
                    )
                    tensors.append(H_dbg)
                    tensors.append(out_dbg)
                    return H_dbg, out_dbg

                def owned_debug_with_inputs(debug_mode, H_arg, alpha_arg, beta_arg):
                    H_dbg_in = remember(ttnn.add(H_arg, 0.0))
                    kwargs = {
                        "compact_vectors": compact_vectors,
                        "native_io": native_io,
                        "output_memory_config": ttnn.L1_MEMORY_CONFIG,
                        "debug_mode": debug_mode,
                    }
                    if use_pretransposed_k:
                        kwargs["k_col"] = k_col_tiles
                    H_dbg, out_dbg = ttnn.experimental.qwen36_gdn_decode_owned(
                        H_dbg_in,
                        q_rows,
                        k_rows,
                        v_rows,
                        alpha_arg,
                        beta_arg,
                        **kwargs,
                    )
                    tensors.append(H_dbg)
                    tensors.append(out_dbg)
                    return H_dbg, out_dbg

                def owned_out_first_row(tensor):
                    raw = host(tensor)
                    leading = out_m_raw.size // (NV_PER_CHIP * V_DIM)
                    tile_rows = raw.size // (leading * NV_PER_CHIP * V_DIM)
                    return raw.reshape(leading, NV_PER_CHIP, tile_rows, V_DIM)[:, :, :1, :]

                def component_out_first_row(tensor):
                    raw = host(tensor)
                    if raw.ndim == 4:
                        return raw[:, :, :1, :]
                    return owned_out_first_row(tensor)

                H_in_host = host(H_input)
                H_copy_host = host(H_owned_copy_for_compare) if H_owned_copy_for_compare is not None else None
                q_host = host(q4)
                k_host = host(k4)
                v_host = host(v4)
                decay_host = host(decay)
                beta_host = host(beta4)

                k_col_host = np.reshape(k_host, k_host.shape[:-2] + (K_DIM, 1))
                q_col_host = np.reshape(q_host, q_host.shape[:-2] + (K_DIM, 1))
                cpu_state_scaled = H_in_host * decay_host
                cpu_prediction = np.sum(cpu_state_scaled * k_col_host, axis=-2, keepdims=True)
                cpu_delta = (v_host - cpu_prediction) * beta_host
                cpu_state_next = cpu_state_scaled + k_col_host * cpu_delta
                cpu_out = np.sum(cpu_state_next * q_col_host, axis=-2, keepdims=True)

                state_scaled_host = host(state_scaled)
                prediction_host = host(prediction)
                delta_host = host(delta)
                H_manual_host = host(H_manual)
                out_manual_host = out_m_host
                prediction_matmul_host = host(prediction_matmul) if prediction_matmul is not None else None
                H_matmul_host = host(H_matmul) if H_matmul is not None else None
                out_matmul_host = host(out_matmul) if out_matmul is not None else None

                H_dbg2, out_dbg2 = owned_debug(2)
                H_dbg3, out_dbg3 = owned_debug(3)
                H_dbg4, out_dbg4 = owned_debug(4)
                H_dbg5, out_dbg5 = owned_debug(5)
                H_dbg9, out_dbg9 = owned_debug(9)

                one_decay = remember(ttnn.add(ttnn.mul(decay, 0.0), 1.0))
                if native_io:
                    one_decay_tiles = one_decay
                    k_prediction_rows = remember(ttnn.repeat(k4, ttnn.Shape([1, 1, 32, 1])))
                else:
                    one_decay_tiles = remember(ttnn.repeat(one_decay, ttnn.Shape([1, 1, 32, 32])))
                    k_prediction_rows = k_rows
                H_iso_pred, out_iso_pred = owned_debug_with_inputs(3, state_scaled, one_decay_tiles, beta_tiles)
                out_component_pred = None
                component_debug_tensors = {}
                component_product0_expected_tt = None
                component_reduce0_expected_tt = None
                component_pred_error = None
                if hasattr(ttnn.experimental, "qwen36_gdn_prediction"):
                    try:
                        if any(mode in component_debug_modes for mode in (11, 12)):
                            # Build the mode 11/12 expected intermediates with TTNN
                            # itself. CPU products/sums are still useful sanity
                            # checks, but they can hide bf16 materialization details.
                            key_tile_rows = 32
                            # Match the manual recurrence order: materialize
                            # the full k_col, multiply the full state, then
                            # slice the first key tile for debug comparison.
                            k_col_full = remember(ttnn.reshape(k4, [1, NV_PER_CHIP, K_DIM, 1]))
                            k_col_full = remember(ttnn.repeat(
                                k_col_full,
                                ttnn.Shape([1, 1, 1, V_DIM]),
                            ))
                            product_full = remember(ttnn.mul(state_scaled, k_col_full))
                            component_product0_expected_tt = remember(ttnn.slice(
                                product_full,
                                [0, 0, 0, 0],
                                [1, NV_PER_CHIP, key_tile_rows, V_DIM],
                            ))
                            if 12 in component_debug_modes:
                                component_reduce0_expected_tt = remember(ttnn.reshape(
                                    ttnn.sum(component_product0_expected_tt, dim=-2),
                                    [1, NV_PER_CHIP, 1, V_DIM],
                                ))
                        out_component_pred = ttnn.experimental.qwen36_gdn_prediction(
                            state_scaled,
                            k_prediction_rows,
                            output_memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                        tensors.append(out_component_pred)
                        for component_mode in component_debug_modes:
                            out_component_debug = ttnn.experimental.qwen36_gdn_prediction(
                                state_scaled,
                                k_prediction_rows,
                                debug_mode=component_mode,
                                output_memory_config=ttnn.L1_MEMORY_CONFIG,
                            )
                            component_debug_tensors[component_mode] = out_component_debug
                            tensors.append(out_component_debug)
                    except Exception as e:
                        component_pred_error = f"{type(e).__name__}: {e}"

                ttnn.synchronize_device(state.mesh)
                H_dbg2_host = host(H_dbg2)
                H_dbg3_host = host(H_dbg3)
                H_dbg4_host = host(H_dbg4)
                H_dbg5_host = host(H_dbg5)
                H_dbg9_host = host(H_dbg9)
                H_iso_pred_host = host(H_iso_pred)
                pred_dbg_host = owned_out_first_row(out_dbg3)
                delta_dbg_host = owned_out_first_row(out_dbg4)
                pred_iso_host = owned_out_first_row(out_iso_pred)
                pred_component_host = (
                    component_out_first_row(out_component_pred)
                    if out_component_pred is not None
                    else None
                )
                component_debug_hosts = {}
                for component_mode, component_tensor in component_debug_tensors.items():
                    if component_mode in (2, 12):
                        component_debug_hosts[component_mode] = component_out_first_row(component_tensor)
                    else:
                        component_debug_hosts[component_mode] = host(component_tensor)
                component_product0_expected_ttnn_host = (
                    host(component_product0_expected_tt)
                    if component_product0_expected_tt is not None
                    else None
                )
                component_reduce0_expected_ttnn_host = (
                    component_out_first_row(component_reduce0_expected_tt)
                    if component_reduce0_expected_tt is not None
                    else None
                )
                k0_col_expected = np.repeat(
                    np.reshape(k_host[:, :, :1, :32], k_host.shape[:2] + (32, 1)),
                    repeats=V_DIM,
                    axis=-1,
                )
                product0_expected = state_scaled_host[:, :, :32, :] * k0_col_expected
                reduce0_expected = np.sum(product0_expected, axis=-2, keepdims=True)
                component_debug_results = {}
                if 2 in component_debug_hosts:
                    component_debug_results["component_prediction_strict"] = compare(
                        "component_prediction_strict", component_debug_hosts[2], prediction_host
                    )
                if 10 in component_debug_hosts:
                    component_debug_results["component_kcol0"] = compare(
                        "component_kcol0", component_debug_hosts[10], k0_col_expected
                    )
                if 11 in component_debug_hosts:
                    component_debug_results["component_product0"] = compare(
                        "component_product0", component_debug_hosts[11], product0_expected
                    )
                    if component_product0_expected_ttnn_host is not None:
                        component_debug_results["component_product0_vs_ttnn"] = compare(
                            "component_product0_vs_ttnn",
                            component_debug_hosts[11],
                            component_product0_expected_ttnn_host,
                        )
                if 12 in component_debug_hosts:
                    component_debug_results["component_reduce0"] = compare(
                        "component_reduce0", component_debug_hosts[12], reduce0_expected
                    )
                    if component_reduce0_expected_ttnn_host is not None:
                        component_debug_results["component_reduce0_vs_ttnn"] = compare(
                            "component_reduce0_vs_ttnn",
                            component_debug_hosts[12],
                            component_reduce0_expected_ttnn_host,
                        )

                result["stepwise"] = {
                    "cpu_vs_ttnn_manual": {
                        "owned_copy_vs_input": (
                            compare("owned_copy_vs_input", H_copy_host, H_in_host)
                            if H_copy_host is not None
                            else None
                        ),
                        "state_scaled": compare("state_scaled", state_scaled_host, cpu_state_scaled),
                        "prediction": compare("prediction", prediction_host, cpu_prediction),
                        "delta": compare("delta", delta_host, cpu_delta),
                        "state_next": compare("state_next", H_manual_host, cpu_state_next),
                        "out": compare("out", out_manual_host, cpu_out),
                    },
                    "owned_debug_vs_ttnn_manual": {
                        "debug2_state_scaled": compare(
                            "debug2_state_scaled", H_dbg2_host, state_scaled_host
                        ),
                        "debug3_state_scaled": compare(
                            "debug3_state_scaled", H_dbg3_host, state_scaled_host
                        ),
                        "debug3_prediction": compare(
                            "debug3_prediction", pred_dbg_host, prediction_host
                        ),
                        "isolated_prediction_state_scaled": compare(
                            "isolated_prediction_state_scaled", H_iso_pred_host, state_scaled_host
                        ),
                        "isolated_prediction": compare(
                            "isolated_prediction", pred_iso_host, prediction_host
                        ),
                        "component_prediction": (
                            compare("component_prediction", pred_component_host, prediction_host)
                            if pred_component_host is not None
                            else None
                        ),
                        **component_debug_results,
                        "component_prediction_error": component_pred_error,
                        "debug4_state_scaled": compare(
                            "debug4_state_scaled", H_dbg4_host, state_scaled_host
                        ),
                        "debug4_delta": compare(
                            "debug4_delta", delta_dbg_host, delta_host
                        ),
                        "debug5_state_next": compare(
                            "debug5_state_next", H_dbg5_host, H_manual_host
                        ),
                        "debug9_state_next": compare(
                            "debug9_state_next", H_dbg9_host, H_manual_host
                        ),
                        "full_state_next": compare(
                            "full_state_next", Ho_host, H_manual_host
                        ),
                        "full_out": compare("full_out", out_o_host, out_manual_host),
                    },
                    "ttnn_matmul_contract": {
                        "error": matmul_reference_error,
                        "prediction_matmul_vs_broadcast": (
                            compare("prediction_matmul_vs_broadcast", prediction_matmul_host, prediction_host)
                            if prediction_matmul_host is not None
                            else None
                        ),
                        "state_next_matmul_vs_broadcast": (
                            compare("state_next_matmul_vs_broadcast", H_matmul_host, H_manual_host)
                            if H_matmul_host is not None
                            else None
                        ),
                        "out_matmul_vs_broadcast": (
                            compare("out_matmul_vs_broadcast", out_matmul_host, out_manual_host)
                            if out_matmul_host is not None
                            else None
                        ),
                        "component_prediction_vs_matmul": (
                            compare("component_prediction_vs_matmul", pred_component_host, prediction_matmul_host)
                            if pred_component_host is not None and prediction_matmul_host is not None
                            else None
                        ),
                        "full_state_next_vs_matmul": (
                            compare("full_state_next_vs_matmul", Ho_host, H_matmul_host)
                            if H_matmul_host is not None
                            else None
                        ),
                        "full_out_vs_matmul": (
                            compare("full_out_vs_matmul", out_o_host, out_matmul_host)
                            if out_matmul_host is not None
                            else None
                        ),
                    },
                }
            state.last_run = {
                "cmd": "probe_deltanet_owned_gdn_real_tensors_tp",
                "accepted": owned_accepted,
                "pass_gate": pass_gate,
                "stepwise": stepwise,
            }
        return result
    finally:
        seen = set()
        for tensor in reversed(tensors):
            if tensor is None:
                continue
            ident = id(tensor)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass


def handle_probe_deltanet_native_gdn_synthetic_mesh_tp(state: MeshServerState, args: dict) -> dict:
    """Validate native Qwen36 GDN recurrence on synthetic tensors on the resident mesh."""
    import numpy as np
    import torch
    import ttnn

    if state.mesh is None:
        return {"error": "server mesh is not loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode is not exposed"}

    slots = int(args.get("slots", 12))
    key_dim = int(args.get("key_dim", 128))
    value_dim = int(args.get("value_dim", 128))
    seed = int(args.get("seed", 20260515))
    scale = float(args.get("scale", 0.03125))
    iters = int(args.get("iters", 0))
    warmup = int(args.get("warmup", 0))
    distribution = args.get("distribution", "replicated")
    if distribution not in ("replicated", "sharded_dim0"):
        return {"error": f"unsupported distribution {distribution!r}"}

    rng = np.random.default_rng(seed)
    if distribution == "replicated":
        state_np = rng.normal(0.0, scale, size=(1, slots, key_dim, value_dim)).astype(np.float32)
        q_np = rng.normal(0.0, scale, size=(1, slots, 1, key_dim)).astype(np.float32)
        k_np = rng.normal(0.0, scale, size=(1, slots, 1, key_dim)).astype(np.float32)
        value_np = rng.normal(0.0, scale, size=(1, slots, 1, value_dim)).astype(np.float32)
        alpha_np = rng.uniform(0.75, 0.995, size=(1, slots, 1, 1)).astype(np.float32)
        beta_np = rng.uniform(0.05, 0.95, size=(1, slots, 1, 1)).astype(np.float32)
    else:
        mesh_devices = int(state.mesh.get_num_devices())
        global_slots = slots * mesh_devices
        state_np = rng.normal(0.0, scale, size=(global_slots, key_dim, value_dim)).astype(np.float32)
        q_np = rng.normal(0.0, scale, size=(global_slots, key_dim)).astype(np.float32)
        k_np = rng.normal(0.0, scale, size=(global_slots, key_dim)).astype(np.float32)
        value_np = rng.normal(0.0, scale, size=(global_slots, value_dim)).astype(np.float32)
        alpha_np = rng.uniform(0.75, 0.995, size=(global_slots,)).astype(np.float32)
        beta_np = rng.uniform(0.05, 0.95, size=(global_slots,)).astype(np.float32)

    tensors = []

    def remember(t):
        tensors.append(t)
        return t

    def upload(arr):
        return remember(ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        ))

    def upload_sharded_dim0(arr):
        return remember(ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr)),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=0),
        ))

    def host(tensor):
        return ttnn.to_torch(
            tensor,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        ).float().cpu().numpy()

    try:
        if distribution == "replicated":
            state_manual = upload(state_np)
            state_native_seed = upload(state_np)
            q = upload(q_np)
            k = upload(k_np)
            value = upload(value_np)
            alpha = upload(alpha_np)
            beta = upload(beta_np)
        else:
            state_manual = remember(ttnn.reshape(upload_sharded_dim0(state_np), [1, slots, key_dim, value_dim]))
            state_native_seed = remember(ttnn.reshape(upload_sharded_dim0(state_np), [1, slots, key_dim, value_dim]))
            q = remember(ttnn.reshape(upload_sharded_dim0(q_np), [1, slots, 1, key_dim]))
            k = remember(ttnn.reshape(upload_sharded_dim0(k_np), [1, slots, 1, key_dim]))
            value = remember(ttnn.reshape(upload_sharded_dim0(value_np), [1, slots, 1, value_dim]))
            alpha = remember(ttnn.reshape(upload_sharded_dim0(alpha_np), [1, slots, 1, 1]))
            beta = remember(ttnn.reshape(upload_sharded_dim0(beta_np), [1, slots, 1, 1]))

        def manual(state_in):
            state_scaled = remember(ttnn.mul(state_in, alpha))
            k_col = remember(ttnn.reshape(k, [1, slots, key_dim, 1]))
            prediction = remember(ttnn.reshape(
                ttnn.sum(ttnn.mul(state_scaled, k_col), dim=-2),
                [1, slots, 1, value_dim],
            ))
            delta = remember(ttnn.mul(ttnn.sub(value, prediction), beta))
            state_next = remember(ttnn.add(
                state_scaled,
                ttnn.mul(k_col, ttnn.reshape(delta, [1, slots, 1, value_dim])),
            ))
            q_col = remember(ttnn.reshape(q, [1, slots, key_dim, 1]))
            out = remember(ttnn.reshape(
                ttnn.sum(ttnn.mul(state_next, q_col), dim=-2),
                [1, slots, 1, value_dim],
            ))
            return state_next, out

        def native(state_in):
            state_next, out = ttnn.experimental.qwen36_gdn_decode(
                state_in,
                q,
                k,
                value,
                alpha,
                beta,
                normalize_qk_l2=False,
                output_memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            tensors.append(out)
            return state_next, out

        manual_state, manual_out = manual(state_manual)
        native_state_in = remember(ttnn.add(state_native_seed, 0.0))
        native_state, native_out = native(native_state_in)
        ttnn.synchronize_device(state.mesh)

        manual_state_host = host(manual_state)
        native_state_host = host(native_state)
        manual_out_raw = host(manual_out)
        native_out_raw = host(native_out)
        out_leading = manual_out_raw.size // (slots * value_dim)
        native_tile_rows = native_out_raw.size // (out_leading * slots * value_dim)
        manual_out_host = manual_out_raw.reshape(out_leading, slots, 1, value_dim)
        native_out_host = native_out_raw.reshape(out_leading, slots, native_tile_rows, value_dim)[:, :, :1, :]

        state_cmp = _pcc_and_maxdiff(native_state_host, manual_state_host)
        out_cmp = _pcc_and_maxdiff(native_out_host, manual_out_host)
        pass_gate = (
            state_cmp["pcc"] >= 0.9999 and
            out_cmp["pcc"] >= 0.9999 and
            state_cmp["max_abs_diff"] <= 1e-2 and
            out_cmp["max_abs_diff"] <= 1e-2
        )

        timing = {
            "iters": iters,
            "warmup": warmup,
            "note": (
                "Synthetic resident-mesh correctness control only. Timing is "
                "disabled here because qwen36_gdn_decode mutates state and a "
                "naive repeated loop can fragment or exhaust resident L1."
            ),
        }
        if iters > 0:
            timing["skipped"] = True
            timing["skipped_reason"] = (
                "Use a dedicated raw-device or trace-safe timing harness after "
                "correctness isolation. This endpoint intentionally avoids "
                "running repeated mutating native-GDN calls inside the resident "
                "server."
            )

        result = {
            "candidate": "qwen36_gdn_decode_synthetic_mesh",
            "distribution": distribution,
            "seed": seed,
            "scale": scale,
            "symbols": {
                "qwen36_gdn_prepare_decode": hasattr(ttnn.experimental, "qwen36_gdn_prepare_decode"),
                "qwen36_gdn_decode": hasattr(ttnn.experimental, "qwen36_gdn_decode"),
            },
            "shape": {
                "state": [1, slots, key_dim, value_dim],
                "qkv": [1, slots, 1, key_dim],
                "alpha_beta": [1, slots, 1, 1],
            },
            "correctness": {
                "state_vs_manual": state_cmp,
                "output_vs_manual": out_cmp,
                "pass_gate": pass_gate,
                "host_shapes": {
                    "state_manual": list(manual_state_host.shape),
                    "state_native": list(native_state_host.shape),
                    "output_manual": list(manual_out_host.shape),
                    "output_native": list(native_out_host.shape),
                },
            },
            "timing": timing,
        }
        state.last_run = {
            "cmd": "probe_deltanet_native_gdn_synthetic_mesh_tp",
            "pass_gate": pass_gate,
            "state_pcc": state_cmp["pcc"],
            "output_pcc": out_cmp["pcc"],
        }
        return result
    finally:
        seen = set()
        for tensor in reversed(tensors):
            if tensor is None:
                continue
            ident = id(tensor)
            if ident in seen:
                continue
            seen.add(ident)
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass


def handle_probe_deltanet_softplus_decay_tp(state: MeshServerState, args: dict) -> dict:
    """Validate native softplus in the DeltaNet decay/gate path."""
    import time as _time
    import numpy as np
    import torch
    import ttnn
    from full_layer_tp_probe import NV_PER_CHIP

    prompt = args.get("prompt", "The capital of France is")
    iters = int(args.get("iters", 10))
    warmup = int(args.get("warmup", 2))
    max_tokens = int(args.get("max_tokens", 20))
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if max_tokens <= 0:
        return {"error": "max_tokens must be > 0"}
    if state.tok is None or not state.layers:
        return {"error": "server not fully loaded"}

    rng = np.random.default_rng(443)
    a_np = (rng.standard_normal((1, NV_PER_CHIP)).astype(np.float32) * 0.2)
    b_np = (rng.standard_normal((1, NV_PER_CHIP)).astype(np.float32) * 0.2)
    dt_bias_np = (rng.standard_normal((NV_PER_CHIP,)).astype(np.float32) * 0.1)
    A_log_np = (rng.standard_normal((NV_PER_CHIP,)).astype(np.float32) * 0.1)

    def upload(arr):
        return ttnn.from_torch(
            torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)),
            dtype=ttnn.bfloat16,
            device=state.mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(state.mesh),
        )

    def decay_manual(a, b, dt_bias, A_log):
        a_biased = ttnn.add(a, dt_bias)
        softplus_a = ttnn.log(ttnn.add(ttnn.exp(a_biased), 1.0))
        g = ttnn.mul(ttnn.neg(ttnn.exp(A_log)), softplus_a)
        beta = ttnn.sigmoid(b)
        decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
        return softplus_a, beta, decay

    def decay_native(a, b, dt_bias, A_log):
        a_biased = ttnn.add(a, dt_bias)
        softplus_a = ttnn.softplus(a_biased)
        g = ttnn.mul(ttnn.neg(ttnn.exp(A_log)), softplus_a)
        beta = ttnn.sigmoid(b)
        decay = ttnn.reshape(ttnn.exp(g), [1, NV_PER_CHIP, 1, 1])
        return softplus_a, beta, decay

    tensors = []
    tensor_gate = None
    try:
        a_tt = upload(a_np)
        b_tt = upload(b_np)
        dt_bias_tt = upload(dt_bias_np)
        A_log_tt = upload(A_log_np)
        tensors.extend([a_tt, b_tt, dt_bias_tt, A_log_tt])
        manual = decay_manual(a_tt, b_tt, dt_bias_tt, A_log_tt)
        native = decay_native(a_tt, b_tt, dt_bias_tt, A_log_tt)
        tensors.extend(manual)
        tensors.extend(native)
        ttnn.synchronize_device(state.mesh)

        def host(tensor, shape):
            return ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            ).float().cpu().numpy()[:1].reshape(shape)

        softplus_cmp = _pcc_and_maxdiff(
            host(native[0], (1, NV_PER_CHIP)),
            host(manual[0], (1, NV_PER_CHIP)),
        )
        beta_cmp = _pcc_and_maxdiff(
            host(native[1], (1, NV_PER_CHIP)),
            host(manual[1], (1, NV_PER_CHIP)),
        )
        decay_cmp = _pcc_and_maxdiff(
            host(native[2], (1, NV_PER_CHIP, 1, 1)),
            host(manual[2], (1, NV_PER_CHIP, 1, 1)),
        )
        tensor_pass_gate = (
            softplus_cmp["pcc"] >= 0.9999 and
            beta_cmp["pcc"] >= 0.9999 and
            decay_cmp["pcc"] >= 0.9999 and
            softplus_cmp["max_abs_diff"] <= 1e-2 and
            beta_cmp["max_abs_diff"] <= 1e-2 and
            decay_cmp["max_abs_diff"] <= 1e-2
        )
        tensor_gate = {
            "softplus_vs_manual": softplus_cmp,
            "beta_vs_manual": beta_cmp,
            "decay_vs_manual": decay_cmp,
            "pass_gate": tensor_pass_gate,
            "shape": {
                "a": [1, NV_PER_CHIP],
                "b": [1, NV_PER_CHIP],
                "dt_bias": [NV_PER_CHIP],
                "A_log": [NV_PER_CHIP],
            },
        }
    except Exception as e:
        return {
            "candidate": "deltanet_native_softplus_decay_gate",
            "tensor_gate": {
                "pass_gate": False,
                "error": f"{type(e).__name__}: {e}",
            },
            "compatibility": None,
            "trace_bench": None,
        }
    finally:
        for tensor in reversed(tensors):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass

    if not tensor_gate["pass_gate"]:
        state.last_run = {
            "cmd": "probe_deltanet_softplus_decay_tp",
            "tensor_pass_gate": False,
        }
        return {
            "candidate": "deltanet_native_softplus_decay_gate",
            "tensor_gate": tensor_gate,
            "compatibility": None,
            "trace_bench": None,
        }

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    token_id = int(prompt_ids[0])

    try:
        baseline_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="manual",
            deltanet_decay_mode="manual")
        native_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="manual",
            deltanet_decay_mode="native_softplus")
    except Exception as e:
        state.deltanet_decay_mode = "manual"
        _reset_state_buffers(state)
        return {
            "candidate": "deltanet_native_softplus_decay_gate",
            "tensor_gate": tensor_gate,
            "compatibility": {
                "accepted": False,
                "phase": "production_forward",
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_bench": None,
        }

    argmax_match = (baseline_id == native_id)
    result = {
        "candidate": "deltanet_native_softplus_decay_gate",
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "token_id": token_id,
        "tensor_gate": tensor_gate,
        "compatibility": {
            "accepted": True,
            "baseline_argmax_id": baseline_id,
            "native_argmax_id": native_id,
            "argmax_match": argmax_match,
            "mode": "native_softplus",
        },
        "trace_bench": None,
        "decode_bench": None,
    }
    if not argmax_match:
        state.last_run = {
            "cmd": "probe_deltanet_softplus_decay_tp",
            "tensor_pass_gate": True,
            "argmax_match": False,
        }
        return result

    trace_id = None
    try:
        t_capture0 = _time.perf_counter()
        trace_id, native_argmax_tt = _capture_temp_decode_trace(
            state, use_fused=False, collective_mode="baseline",
            rope_mode="manual", deltanet_decay_mode="native_softplus")
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

        result["trace_bench"] = {
            "iters": iters,
            "warmup": warmup,
            "capture_ms": capture_ms,
            "summary_ms": {
                "native_execute_trace": _summary_ms(execute_ms),
                "native_update_plus_execute": _summary_ms(update_execute_ms),
            },
            "samples_ms": {
                "native_execute_trace": execute_ms,
                "native_update_plus_execute": update_execute_ms,
            },
            "traced_argmax_id_sample": _read_argmax_id(state, native_argmax_tt),
            "note": (
                "Temporary trace component timing only; compare against a "
                "same-session manual baseline before any end-to-end claim."
            ),
        }

        if len(prompt_ids) + max_tokens > MAX_POS:
            result["decode_bench"] = {
                "accepted": False,
                "error": (
                    f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} "
                    f"> MAX_POS {MAX_POS}"
                ),
            }
        else:
            _ensure_decode_trace(state)

            def run_decode(trace_to_run, argmax_tensor):
                _reset_state_buffers(state)
                t_prefill0 = _time.perf_counter()
                last_argmax = None
                for pos, tid in enumerate(prompt_ids):
                    update_input_buffers(state, int(tid), pos)
                    ttnn.execute_trace(state.mesh, trace_to_run, cq_id=0, blocking=False)
                    last_argmax = argmax_tensor
                ttnn.synchronize_device(state.mesh)
                prefill_ms = (_time.perf_counter() - t_prefill0) * 1000.0

                generated_ids = []
                decode_times = []
                cur_pos = len(prompt_ids)
                eos_id = getattr(state.tok, "eos_token_id", None)
                stopped_on_eos = False
                for _step in range(max_tokens):
                    next_id = _read_argmax_id(state, last_argmax)
                    generated_ids.append(next_id)
                    if eos_id is not None and next_id == eos_id:
                        stopped_on_eos = True
                        break
                    td0 = _time.perf_counter()
                    update_input_buffers(state, next_id, cur_pos)
                    ttnn.execute_trace(state.mesh, trace_to_run, cq_id=0, blocking=False)
                    ttnn.synchronize_device(state.mesh)
                    decode_times.append((_time.perf_counter() - td0) * 1000.0)
                    last_argmax = argmax_tensor
                    cur_pos += 1
                return {
                    "generated_ids": generated_ids,
                    "n_generated_tokens": len(generated_ids),
                    "prefill_ms": prefill_ms,
                    "decode_ms": _summary_ms(decode_times),
                    "ms_per_tok": (
                        float(np.mean(decode_times)) if decode_times else float("nan")
                    ),
                    "tok_per_sec": (
                        1000.0 / float(np.mean(decode_times))
                        if decode_times and float(np.mean(decode_times)) > 0 else 0.0
                    ),
                    "stopped_on_eos": stopped_on_eos,
                }

            manual_decode = run_decode(state.trace_id, state.traced_argmax_tt)
            native_decode = run_decode(trace_id, native_argmax_tt)
            result["decode_bench"] = {
                "max_tokens": max_tokens,
                "manual": manual_decode,
                "native_softplus": native_decode,
                "generated_ids_match": (
                    manual_decode["generated_ids"] == native_decode["generated_ids"]
                ),
                "note": (
                    "Full decode loop timing includes token readback and buffer "
                    "updates, using same-session manual and temporary native traces."
                ),
            }
    except Exception as e:
        result["trace_bench"] = {
            "accepted": False,
            "phase": "native_trace_or_bench",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(state.mesh, trace_id)
        state.deltanet_decay_mode = "manual"
        state.rope_mode = "manual"
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False
        _reset_state_buffers(state)

    state.last_run = {
        "cmd": "probe_deltanet_softplus_decay_tp",
        "tensor_pass_gate": tensor_gate["pass_gate"],
        "argmax_match": argmax_match,
        "median_native_execute_ms": (
            result["trace_bench"]["summary_ms"]["native_execute_trace"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
        "median_native_combined_ms": (
            result["trace_bench"]["summary_ms"]["native_update_plus_execute"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
    }
    return result


def handle_probe_deltanet_owned_gdn_trace_tp(state: MeshServerState, args: dict) -> dict:
    """Guarded production-trace probe for the owned GDN recurrence path."""
    import time as _time
    import numpy as np
    import ttnn

    if state.mesh is None or state.cfg is None or not state.layers:
        return {"error": "server not fully loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode_owned is not exposed"}

    prompt = args.get("prompt", "The capital of France is")
    iters = int(args.get("iters", 10))
    warmup = int(args.get("warmup", 2))
    max_tokens = int(args.get("max_tokens", 20))
    decay_mode = args.get("deltanet_decay_mode", "manual")
    if decay_mode not in ("manual", "native_softplus"):
        return {"error": "deltanet_decay_mode must be manual or native_softplus"}
    recurrence_mode = args.get("deltanet_recurrence_mode", "owned_gdn")
    if recurrence_mode not in ("owned_gdn", "owned_gdn_inplace"):
        return {"error": "deltanet_recurrence_mode must be owned_gdn or owned_gdn_inplace"}
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if max_tokens < 0:
        return {"error": "max_tokens must be >= 0"}

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    token_id = int(prompt_ids[0])

    try:
        baseline_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="manual",
            deltanet_decay_mode="manual", deltanet_recurrence_mode="manual")
        owned_id = _forward_argmax_id_with_cache_writer(
            state, token_id=token_id, cur_pos=0, use_fused=False,
            collective_mode="baseline", rope_mode="manual",
            deltanet_decay_mode=decay_mode, deltanet_recurrence_mode=recurrence_mode)
    except Exception as e:
        state.deltanet_recurrence_mode = "manual"
        _reset_state_buffers(state)
        return {
            "candidate": "deltanet_owned_gdn_recurrence",
            "compatibility": {
                "accepted": False,
                "phase": "production_forward",
                "error": f"{type(e).__name__}: {e}",
            },
            "trace_bench": None,
            "decode_bench": None,
        }

    argmax_match = baseline_id == owned_id
    result = {
        "candidate": "deltanet_owned_gdn_recurrence",
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "token_id": token_id,
        "compatibility": {
            "accepted": True,
            "baseline_argmax_id": baseline_id,
            "owned_argmax_id": owned_id,
            "argmax_match": argmax_match,
            "mode": recurrence_mode,
            "decay_mode": decay_mode,
        },
        "trace_bench": None,
        "decode_bench": None,
    }
    if not argmax_match:
        state.last_run = {
            "cmd": "probe_deltanet_owned_gdn_trace_tp",
            "argmax_match": False,
        }
        return result

    trace_id = None
    try:
        _ensure_decode_trace(state)
        t_capture0 = _time.perf_counter()
        trace_id, owned_argmax_tt = _capture_temp_decode_trace(
            state, use_fused=False, collective_mode="baseline",
            rope_mode="manual", deltanet_decay_mode=decay_mode,
            deltanet_recurrence_mode=recurrence_mode)
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

        manual_execute_ms = []
        owned_execute_ms = []
        manual_update_execute_ms = []
        owned_update_execute_ms = []
        for i in range(iters):
            tid = prompt_ids[i % len(prompt_ids)]
            pos = i % MAX_POS
            manual_execute_ms.append(timed(
                lambda: ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)))
            owned_execute_ms.append(timed(
                lambda: ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)))

            tid2 = prompt_ids[(i + 1) % len(prompt_ids)]
            pos2 = (pos + 1) % MAX_POS

            def manual_update_execute(tid2=tid2, pos2=pos2):
                update_input_buffers(state, tid2, pos2)
                ttnn.execute_trace(state.mesh, state.trace_id, cq_id=0, blocking=False)

            def owned_update_execute(tid2=tid2, pos2=pos2):
                update_input_buffers(state, tid2, pos2)
                ttnn.execute_trace(state.mesh, trace_id, cq_id=0, blocking=False)

            manual_update_execute_ms.append(timed(manual_update_execute))
            owned_update_execute_ms.append(timed(owned_update_execute))

        result["trace_bench"] = {
            "iters": iters,
            "warmup": warmup,
            "capture_ms": capture_ms,
            "summary_ms": {
                "manual_execute_trace": _summary_ms(manual_execute_ms),
                "owned_execute_trace": _summary_ms(owned_execute_ms),
                "manual_update_plus_execute": _summary_ms(manual_update_execute_ms),
                "owned_update_plus_execute": _summary_ms(owned_update_execute_ms),
            },
            "samples_ms": {
                "manual_execute_trace": manual_execute_ms,
                "owned_execute_trace": owned_execute_ms,
                "manual_update_plus_execute": manual_update_execute_ms,
                "owned_update_plus_execute": owned_update_execute_ms,
            },
            "traced_argmax_id_sample": _read_argmax_id(state, owned_argmax_tt),
            "note": "Same-session temporary trace timing; still gated by decode correctness below.",
        }

        if len(prompt_ids) + max_tokens > MAX_POS:
            result["decode_bench"] = {
                "accepted": False,
                "error": (
                    f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} "
                    f"> MAX_POS {MAX_POS}"
                ),
            }
        else:
            def run_decode(trace_to_run, argmax_tensor):
                _reset_state_buffers(state)
                t_prefill0 = _time.perf_counter()
                last_argmax = None
                for pos, tid in enumerate(prompt_ids):
                    update_input_buffers(state, int(tid), pos)
                    ttnn.execute_trace(state.mesh, trace_to_run, cq_id=0, blocking=False)
                    last_argmax = argmax_tensor
                ttnn.synchronize_device(state.mesh)
                prefill_ms = (_time.perf_counter() - t_prefill0) * 1000.0

                generated_ids = []
                decode_times = []
                cur_pos = len(prompt_ids)
                eos_id = getattr(state.tok, "eos_token_id", None)
                stopped_on_eos = False
                for _step in range(max_tokens):
                    next_id = _read_argmax_id(state, last_argmax)
                    generated_ids.append(next_id)
                    if eos_id is not None and next_id == eos_id:
                        stopped_on_eos = True
                        break
                    td0 = _time.perf_counter()
                    update_input_buffers(state, next_id, cur_pos)
                    ttnn.execute_trace(state.mesh, trace_to_run, cq_id=0, blocking=False)
                    ttnn.synchronize_device(state.mesh)
                    decode_times.append((_time.perf_counter() - td0) * 1000.0)
                    last_argmax = argmax_tensor
                    cur_pos += 1
                return {
                    "generated_ids": generated_ids,
                    "n_generated_tokens": len(generated_ids),
                    "prefill_ms": prefill_ms,
                    "decode_ms": _summary_ms(decode_times),
                    "ms_per_tok": (
                        float(np.mean(decode_times)) if decode_times else float("nan")
                    ),
                    "tok_per_sec": (
                        1000.0 / float(np.mean(decode_times))
                        if decode_times and float(np.mean(decode_times)) > 0 else 0.0
                    ),
                    "stopped_on_eos": stopped_on_eos,
                }

            manual_decode = run_decode(state.trace_id, state.traced_argmax_tt)
            owned_decode = run_decode(trace_id, owned_argmax_tt)
            result["decode_bench"] = {
                "max_tokens": max_tokens,
                "manual": manual_decode,
                "owned_gdn": owned_decode,
                "owned_mode": recurrence_mode,
                "decay_mode": decay_mode,
                "generated_ids_match": (
                    manual_decode["generated_ids"] == owned_decode["generated_ids"]
                ),
                "note": (
                    "Full decode loop timing includes token readback and buffer "
                    "updates, using same-session manual and temporary owned traces."
                ),
            }
    except Exception as e:
        result["trace_bench"] = {
            "accepted": False,
            "phase": "owned_trace_or_bench",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        if trace_id is not None:
            ttnn.release_trace(state.mesh, trace_id)
        state.deltanet_recurrence_mode = "manual"
        state.deltanet_decay_mode = "manual"
        state.rope_mode = "manual"
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False
        _reset_state_buffers(state)

    state.last_run = {
        "cmd": "probe_deltanet_owned_gdn_trace_tp",
        "owned_mode": recurrence_mode,
        "decay_mode": decay_mode,
        "argmax_match": argmax_match,
        "generated_ids_match": (
            result.get("decode_bench", {}).get("generated_ids_match")
            if result.get("decode_bench") else None
        ),
        "median_manual_execute_ms": (
            result["trace_bench"]["summary_ms"]["manual_execute_trace"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
        "median_owned_execute_ms": (
            result["trace_bench"]["summary_ms"]["owned_execute_trace"].get("median")
            if result.get("trace_bench") and result["trace_bench"].get("summary_ms") else None
        ),
    }
    return result


def handle_probe_deltanet_owned_gdn_divergence_tp(state: MeshServerState, args: dict) -> dict:
    """Eager top-k diagnostic for manual-vs-owned GDN decode divergence."""
    import torch
    import ttnn

    if state.mesh is None or state.cfg is None or not state.layers:
        return {"error": "server not fully loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode_owned is not exposed"}

    prompt = args.get("prompt", "In Python, a simple function to add two numbers is")
    max_tokens = int(args.get("max_tokens", 24))
    top_k = int(args.get("top_k", 8))
    top_k = max(1, min(top_k, 32))
    if max_tokens < 0:
        return {"error": "max_tokens must be >= 0"}

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    if len(prompt_ids) + max_tokens > MAX_POS:
        return {
            "error": (
                f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} "
                f"> MAX_POS {MAX_POS}"
            )
        }

    def logits_summary(logits_tt):
        logits_host = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        row = logits_host.reshape(-1, state.vocab_size)[0].float().cpu()
        k = min(top_k, row.numel())
        vals, idx = torch.topk(row, k=k)
        entries = [
            {"token_id": int(i), "logit": float(v)}
            for i, v in zip(idx.tolist(), vals.tolist())
        ]
        return {
            "top_k": entries,
            "argmax_id": int(idx[0]),
            "top2_margin": (
                float(vals[0] - vals[1]) if k > 1 else None
            ),
        }

    def run_mode(mode: str):
        old_recurrence = state.deltanet_recurrence_mode
        state.deltanet_recurrence_mode = mode
        records = []
        generated_ids = []
        logits_tt = None
        try:
            _reset_state_buffers(state)
            for pos, tid in enumerate(prompt_ids):
                if logits_tt is not None:
                    ttnn.deallocate(logits_tt)
                update_input_buffers(state, int(tid), pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)

            cur_pos = len(prompt_ids)
            eos_id = getattr(state.tok, "eos_token_id", None)
            for step in range(max_tokens):
                summary = logits_summary(logits_tt)
                next_id = int(summary["argmax_id"])
                generated_ids.append(next_id)
                records.append({
                    "step": step,
                    "token_id": next_id,
                    "top2_margin": summary["top2_margin"],
                    "top_k": summary["top_k"],
                })
                if eos_id is not None and next_id == eos_id:
                    break
                ttnn.deallocate(logits_tt)
                logits_tt = None
                update_input_buffers(state, next_id, cur_pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)
                cur_pos += 1
        finally:
            if logits_tt is not None:
                try:
                    ttnn.deallocate(logits_tt)
                except Exception:
                    pass
            state.deltanet_recurrence_mode = old_recurrence
            _reset_state_buffers(state)
        return {
            "generated_ids": generated_ids,
            "records": records,
        }

    old_recurrence = state.deltanet_recurrence_mode
    old_decay = state.deltanet_decay_mode
    old_rope = state.rope_mode
    old_collective = state.collective_mode
    old_fused = state.use_fused_paged_update
    try:
        state.deltanet_decay_mode = "manual"
        state.rope_mode = "manual"
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False
        manual = run_mode("manual")
        owned = run_mode("owned_gdn")
    finally:
        state.deltanet_recurrence_mode = old_recurrence
        state.deltanet_decay_mode = old_decay
        state.rope_mode = old_rope
        state.collective_mode = old_collective
        state.use_fused_paged_update = old_fused
        _reset_state_buffers(state)

    first_diff = None
    for i, (manual_id, owned_id) in enumerate(zip(
            manual["generated_ids"], owned["generated_ids"])):
        if int(manual_id) != int(owned_id):
            first_diff = i
            break

    first_diff_detail = None
    if first_diff is not None:
        first_diff_detail = {
            "step": first_diff,
            "manual": manual["records"][first_diff],
            "owned_gdn": owned["records"][first_diff],
        }

    result = {
        "candidate": "deltanet_owned_gdn_recurrence",
        "diagnostic": "eager_topk_decode_divergence",
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "max_tokens": max_tokens,
        "top_k": top_k,
        "generated_ids_match": manual["generated_ids"] == owned["generated_ids"],
        "first_diff_step": first_diff,
        "first_diff": first_diff_detail,
        "manual": manual,
        "owned_gdn": owned,
        "note": (
            "This endpoint is diagnostic only. It uses eager forwards and reads "
            "top-k logits to explain divergence; it is not a performance run."
        ),
    }
    state.last_run = {
        "cmd": "probe_deltanet_owned_gdn_divergence_tp",
        "generated_ids_match": result["generated_ids_match"],
        "first_diff_step": first_diff,
    }
    return result


def handle_probe_deltanet_owned_gdn_teacher_forced_tp(state: MeshServerState, args: dict) -> dict:
    """Teacher-forced manual-vs-owned GDN comparison on the same token stream."""
    import math
    import torch
    import ttnn

    if state.mesh is None or state.cfg is None or not state.layers:
        return {"error": "server not fully loaded"}
    if not hasattr(ttnn.experimental, "qwen36_gdn_decode_owned"):
        return {"error": "ttnn.experimental.qwen36_gdn_decode_owned is not exposed"}

    prompt = args.get("prompt", "In Python, a simple function to add two numbers is")
    max_tokens = int(args.get("max_tokens", 24))
    top_k = int(args.get("top_k", 8))
    state_layers = args.get("state_layers")
    if state_layers is None:
        state_layers = [
            i for i, layer in enumerate(state.layers)
            if layer.get("type") == "linear_attention"
        ][:8]
    else:
        state_layers = [int(i) for i in state_layers]
    top_k = max(1, min(top_k, 32))
    if max_tokens < 0:
        return {"error": "max_tokens must be >= 0"}

    prompt_ids = state.tok.encode(prompt)
    if not prompt_ids:
        return {"error": "prompt encoded to zero tokens"}
    if len(prompt_ids) + max_tokens > MAX_POS:
        return {
            "error": (
                f"prompt_len {len(prompt_ids)} + max_tokens {max_tokens} "
                f"> MAX_POS {MAX_POS}"
            )
        }

    def logits_row(logits_tt):
        logits_host = ttnn.to_torch(
            logits_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
        )
        return logits_host.reshape(-1, state.vocab_size)[0].float().cpu()

    def topk_summary(row):
        k = min(top_k, row.numel())
        vals, idx = torch.topk(row, k=k)
        return {
            "argmax_id": int(idx[0]),
            "top2_margin": (
                float(vals[0] - vals[1]) if k > 1 else None
            ),
            "top_k": [
                {"token_id": int(i), "logit": float(v)}
                for i, v in zip(idx.tolist(), vals.tolist())
            ],
        }

    def row_compare(manual_row, owned_row):
        diff = (manual_row - owned_row).float()
        manual64 = manual_row.double()
        owned64 = owned_row.double()
        manual_centered = manual64 - manual64.mean()
        owned_centered = owned64 - owned64.mean()
        denom = torch.sqrt(
            torch.sum(manual_centered * manual_centered)
            * torch.sum(owned_centered * owned_centered)
        )
        pcc = float(torch.sum(manual_centered * owned_centered) / denom) if float(denom) > 0 else math.nan
        return {
            "max_abs": float(torch.max(torch.abs(diff))),
            "mean_abs": float(torch.mean(torch.abs(diff))),
            "rms": float(torch.sqrt(torch.mean(diff * diff))),
            "pcc": pcc,
        }

    def collect_state_snapshots():
        snapshots = {}
        for layer_idx in state_layers:
            if layer_idx < 0 or layer_idx >= len(state.layers):
                continue
            layer = state.layers[layer_idx]
            if layer.get("type") != "linear_attention":
                continue
            ssm = layer["dn"]["ssm"]
            host = ttnn.to_torch(
                ssm,
                mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
            )
            snapshots[str(layer_idx)] = host.float().cpu().reshape(-1)
        return snapshots

    def compare_states(manual_states, owned_states):
        out = {}
        for key, manual_state in manual_states.items():
            owned_state = owned_states.get(key)
            if owned_state is None:
                continue
            out[key] = row_compare(manual_state, owned_state)
        return out

    def set_modes(mode: str):
        state.deltanet_recurrence_mode = mode
        state.deltanet_decay_mode = "manual"
        state.rope_mode = "manual"
        state.collective_mode = "baseline"
        state.use_fused_paged_update = False

    def make_teacher_tokens():
        set_modes("manual")
        generated = []
        logits_tt = None
        try:
            _reset_state_buffers(state)
            for pos, tid in enumerate(prompt_ids):
                if logits_tt is not None:
                    ttnn.deallocate(logits_tt)
                update_input_buffers(state, int(tid), pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)
            cur_pos = len(prompt_ids)
            for _step in range(max_tokens):
                row = logits_row(logits_tt)
                next_id = int(torch.argmax(row).item())
                generated.append(next_id)
                ttnn.deallocate(logits_tt)
                logits_tt = None
                update_input_buffers(state, next_id, cur_pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)
                cur_pos += 1
        finally:
            if logits_tt is not None:
                try:
                    ttnn.deallocate(logits_tt)
                except Exception:
                    pass
            _reset_state_buffers(state)
        return generated

    def run_forced(mode: str, teacher_ids):
        set_modes(mode)
        rows = []
        records = []
        logits_tt = None
        try:
            _reset_state_buffers(state)
            for pos, tid in enumerate(prompt_ids):
                if logits_tt is not None:
                    ttnn.deallocate(logits_tt)
                update_input_buffers(state, int(tid), pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)

            cur_pos = len(prompt_ids)
            for step, forced_id in enumerate(teacher_ids):
                row = logits_row(logits_tt)
                rows.append(row)
                summary = topk_summary(row)
                summary.update({
                    "step": step,
                    "forced_token_id": int(forced_id),
                    "forced_token_logit": float(row[int(forced_id)]),
                })
                records.append(summary)
                ttnn.deallocate(logits_tt)
                logits_tt = None
                update_input_buffers(state, int(forced_id), cur_pos)
                logits_tt = forward_token_tp_inner(state, return_logits=True)
                ttnn.synchronize_device(state.mesh)
                cur_pos += 1
            state_snapshots = collect_state_snapshots()
        finally:
            if logits_tt is not None:
                try:
                    ttnn.deallocate(logits_tt)
                except Exception:
                    pass
        return {
            "rows": rows,
            "records": records,
            "states": state_snapshots,
        }

    old_recurrence = state.deltanet_recurrence_mode
    old_decay = state.deltanet_decay_mode
    old_rope = state.rope_mode
    old_collective = state.collective_mode
    old_fused = state.use_fused_paged_update
    try:
        teacher_ids = make_teacher_tokens()
        manual = run_forced("manual", teacher_ids)
        owned = run_forced("owned_gdn", teacher_ids)
    finally:
        state.deltanet_recurrence_mode = old_recurrence
        state.deltanet_decay_mode = old_decay
        state.rope_mode = old_rope
        state.collective_mode = old_collective
        state.use_fused_paged_update = old_fused
        _reset_state_buffers(state)

    step_comparisons = []
    first_argmax_diff = None
    first_forced_token_rank_change = None
    for step, (manual_row, owned_row) in enumerate(zip(manual["rows"], owned["rows"])):
        manual_rec = manual["records"][step]
        owned_rec = owned["records"][step]
        comp = row_compare(manual_row, owned_row)
        forced_id = int(teacher_ids[step])
        manual_better = int(manual_rec["argmax_id"]) == forced_id
        owned_better = int(owned_rec["argmax_id"]) == forced_id
        if first_argmax_diff is None and manual_rec["argmax_id"] != owned_rec["argmax_id"]:
            first_argmax_diff = step
        if first_forced_token_rank_change is None and manual_better != owned_better:
            first_forced_token_rank_change = step
        step_comparisons.append({
            "step": step,
            "forced_token_id": forced_id,
            "manual_argmax_id": manual_rec["argmax_id"],
            "owned_argmax_id": owned_rec["argmax_id"],
            "manual_forced_token_logit": manual_rec["forced_token_logit"],
            "owned_forced_token_logit": owned_rec["forced_token_logit"],
            "forced_token_logit_delta_manual_minus_owned": (
                manual_rec["forced_token_logit"] - owned_rec["forced_token_logit"]
            ),
            "manual_top2_margin": manual_rec["top2_margin"],
            "owned_top2_margin": owned_rec["top2_margin"],
            "logit_compare": comp,
        })

    result = {
        "candidate": "deltanet_owned_gdn_recurrence",
        "diagnostic": "teacher_forced_manual_vs_owned_gdn",
        "prompt": prompt,
        "prompt_ids": list(prompt_ids),
        "teacher_generated_ids": teacher_ids,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "state_layers": state_layers,
        "first_argmax_diff_step": first_argmax_diff,
        "first_forced_token_rank_change_step": first_forced_token_rank_change,
        "step_comparisons": step_comparisons,
        "manual_records": manual["records"],
        "owned_gdn_records": owned["records"],
        "final_state_compare": compare_states(manual["states"], owned["states"]),
        "note": (
            "Teacher-forced diagnostic: both modes receive the same manual-greedy "
            "teacher token stream. This separates numeric drift from autoregressive "
            "trajectory drift and is not a performance benchmark."
        ),
    }
    state.last_run = {
        "cmd": "probe_deltanet_owned_gdn_teacher_forced_tp",
        "first_argmax_diff_step": first_argmax_diff,
        "first_forced_token_rank_change_step": first_forced_token_rank_change,
    }
    return result


def handle_probe_deltanet_owned_gdn_benchmark_tp(state: MeshServerState, args: dict) -> dict:
    """Run the guarded owned-GDN trace probe over a fixed prompt set."""
    import math
    import statistics

    default_prompts = [
        "The capital of France is",
        "Write a short explanation of tensor parallelism.",
        "In Python, a simple function to add two numbers is",
        "The main bottleneck in batch-one language model decode is",
        "Summarize why recurrent state updates matter in hybrid attention models.",
    ]
    prompts = args.get("prompts")
    if prompts is None:
        prompt = args.get("prompt")
        prompts = [prompt] if prompt else default_prompts
    elif isinstance(prompts, str):
        prompts = [prompts]
    else:
        prompts = list(prompts)
    prompts = [str(p) for p in prompts if str(p)]
    if not prompts:
        return {"error": "no prompts supplied"}

    iters = int(args.get("iters", 6))
    warmup = int(args.get("warmup", 2))
    max_tokens = int(args.get("max_tokens", 64))
    decay_mode = args.get("deltanet_decay_mode", "manual")
    if decay_mode not in ("manual", "native_softplus"):
        return {"error": "deltanet_decay_mode must be manual or native_softplus"}
    recurrence_mode = args.get("deltanet_recurrence_mode", "owned_gdn")
    if recurrence_mode not in ("owned_gdn", "owned_gdn_inplace"):
        return {"error": "deltanet_recurrence_mode must be owned_gdn or owned_gdn_inplace"}
    if iters <= 0:
        return {"error": "iters must be > 0"}
    if max_tokens < 0:
        return {"error": "max_tokens must be >= 0"}

    prompt_results = []
    for prompt in prompts:
        prompt_results.append(handle_probe_deltanet_owned_gdn_trace_tp(state, {
            "prompt": prompt,
            "iters": iters,
            "warmup": warmup,
            "max_tokens": max_tokens,
            "deltanet_decay_mode": decay_mode,
            "deltanet_recurrence_mode": recurrence_mode,
        }))

    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    def summarize(values):
        xs = [float(x) for x in values if finite(x)]
        if not xs:
            return {"count": 0}
        return {
            "count": len(xs),
            "min": min(xs),
            "max": max(xs),
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
        }

    def trace_median(result, key):
        trace = result.get("trace_bench") or {}
        summary = trace.get("summary_ms") or {}
        item = summary.get(key) or {}
        return item.get("median")

    def decode_ms_per_tok(result, key):
        decode = result.get("decode_bench") or {}
        item = decode.get(key) or {}
        return item.get("ms_per_tok")

    accepted = []
    manual_trace = []
    owned_trace = []
    manual_update_trace = []
    owned_update_trace = []
    manual_decode = []
    owned_decode = []
    trace_delta = []
    trace_delta_pct = []
    decode_delta = []
    decode_delta_pct = []

    per_prompt_summary = []
    for result in prompt_results:
        compat = result.get("compatibility") or {}
        decode = result.get("decode_bench") or {}
        ok = (
            compat.get("accepted") is True
            and compat.get("argmax_match") is True
            and decode.get("generated_ids_match") is True
        )
        accepted.append(ok)

        mt = trace_median(result, "manual_execute_trace")
        ot = trace_median(result, "owned_execute_trace")
        mut = trace_median(result, "manual_update_plus_execute")
        out = trace_median(result, "owned_update_plus_execute")
        md = decode_ms_per_tok(result, "manual")
        od = decode_ms_per_tok(result, "owned_gdn")

        if finite(mt):
            manual_trace.append(mt)
        if finite(ot):
            owned_trace.append(ot)
        if finite(mut):
            manual_update_trace.append(mut)
        if finite(out):
            owned_update_trace.append(out)
        if finite(md):
            manual_decode.append(md)
        if finite(od):
            owned_decode.append(od)
        if finite(mt) and finite(ot):
            delta = float(mt) - float(ot)
            trace_delta.append(delta)
            if float(mt) > 0.0:
                trace_delta_pct.append(100.0 * delta / float(mt))
        if finite(md) and finite(od):
            delta = float(md) - float(od)
            decode_delta.append(delta)
            if float(md) > 0.0:
                decode_delta_pct.append(100.0 * delta / float(md))

        per_prompt_summary.append({
            "prompt": result.get("prompt"),
            "accepted": ok,
            "argmax_match": compat.get("argmax_match"),
            "generated_ids_match": decode.get("generated_ids_match"),
            "manual_execute_trace_median_ms": mt,
            "owned_execute_trace_median_ms": ot,
            "manual_update_plus_execute_median_ms": mut,
            "owned_update_plus_execute_median_ms": out,
            "manual_decode_ms_per_tok": md,
            "owned_decode_ms_per_tok": od,
            "n_generated_tokens_manual": (decode.get("manual") or {}).get("n_generated_tokens"),
            "n_generated_tokens_owned": (decode.get("owned_gdn") or {}).get("n_generated_tokens"),
            "error": result.get("error") or (result.get("trace_bench") or {}).get("error"),
        })

    result = {
        "candidate": "deltanet_owned_gdn_recurrence",
        "benchmark": "multi_prompt_guarded_trace_decode",
        "num_prompts": len(prompts),
        "iters": iters,
        "warmup": warmup,
        "max_tokens": max_tokens,
        "owned_mode": recurrence_mode,
        "decay_mode": decay_mode,
        "all_prompts_passed": all(accepted),
        "per_prompt_summary": per_prompt_summary,
        "aggregate": {
            "manual_execute_trace_median_ms": summarize(manual_trace),
            "owned_execute_trace_median_ms": summarize(owned_trace),
            "execute_trace_delta_ms_manual_minus_owned": summarize(trace_delta),
            "execute_trace_delta_pct_manual_minus_owned": summarize(trace_delta_pct),
            "manual_update_plus_execute_median_ms": summarize(manual_update_trace),
            "owned_update_plus_execute_median_ms": summarize(owned_update_trace),
            "manual_decode_ms_per_tok": summarize(manual_decode),
            "owned_decode_ms_per_tok": summarize(owned_decode),
            "decode_delta_ms_per_tok_manual_minus_owned": summarize(decode_delta),
            "decode_delta_pct_manual_minus_owned": summarize(decode_delta_pct),
        },
        "prompt_results": prompt_results,
        "note": (
            "Each prompt reuses the guarded single-prompt manual-vs-owned flow. "
            "Timing claims are valid only for prompts whose argmax and generated "
            "token streams match."
        ),
    }
    state.last_run = {
        "cmd": "probe_deltanet_owned_gdn_benchmark_tp",
        "owned_mode": recurrence_mode,
        "decay_mode": decay_mode,
        "all_prompts_passed": result["all_prompts_passed"],
        "num_prompts": len(prompts),
        "median_manual_decode_ms_per_tok": (
            result["aggregate"]["manual_decode_ms_per_tok"].get("median")
        ),
        "median_owned_decode_ms_per_tok": (
            result["aggregate"]["owned_decode_ms_per_tok"].get("median")
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

    conv1d_mode = str(args.get("deltanet_conv1d_mode", "manual"))
    if conv1d_mode not in ("manual", "owned_conv1d"):
        return {"error": f"deltanet_conv1d_mode must be one of manual/owned_conv1d, got {conv1d_mode}"}

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
    old_conv1d_mode = state.deltanet_conv1d_mode
    old_decay_gate_mode = state.deltanet_decay_gate_mode
    old_rope_mode = state.rope_mode
    state.deltanet_recurrence_mode = mode
    state.deltanet_conv1d_mode = conv1d_mode
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
        state.deltanet_conv1d_mode = old_conv1d_mode
        state.deltanet_decay_gate_mode = old_decay_gate_mode
        state.rope_mode = old_rope_mode

    np.savez(out_path,
             logits=logits_arr,
             prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
             generated_ids=np.asarray(generated_ids, dtype=np.int32))

    state.last_run = {
        "cmd": "cosine_ladder_tp",
        "deltanet_recurrence_mode": mode,
        "deltanet_conv1d_mode": conv1d_mode,
        "deltanet_decay_gate_mode": decay_gate_mode,
        "rope_mode": rope_mode,
        "n_prompt": P,
        "n_steps": M,
    }

    return {
        "ok": True,
        "path": out_path,
        "deltanet_recurrence_mode": mode,
        "deltanet_conv1d_mode": conv1d_mode,
        "deltanet_decay_gate_mode": decay_gate_mode,
        "rope_mode": rope_mode,
        "n_prompt": P,
        "n_steps": M,
        "vocab": VOCAB,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "ms_per_step": decode_ms / max(M - 1, 1),
    }


def handle_probe_deltanet_conv1d_split_check_tp(state: MeshServerState, args: dict) -> dict:
    """Mesh-aware probe for the owned conv1d wire-in bug investigation
    (commit 64a31b1 documents the G3/G4 failures).

    Reads back both the combined dn['conv_st']/dn['w_conv'] (rank-2,
    [CONV_DIM_CHIP, K] padded [chip_rows, 32]) and the split tensors
    dn['conv_st_split'][k]/dn['w_conv_split'][k] (rank-2, [CONV_DIM_CHIP, 1]
    padded [chip_rows, 32]) from a chosen DeltaNet layer, via the mesh
    composer. Compares each split[k] to the column-k slice of combined.

    If split[k] != combined[:, k:k+1] at BF16-tolerance, the G4 bootstrap
    pre-split upload is buggy on mesh (likely a relayout_conv interaction
    with single-column input + ShardTensorToMesh dim=0). If they match,
    the bug is in mesh kernel dispatch or multi-step state evolution.

    args:
      layer_idx:    int  (default 0) — DeltaNet layer to inspect
      max_abs_diff: float (default 0.05) — BF16-tolerance threshold

    Returns: {ok, layer_idx, comparisons, all_match, diagnosis}.
    """
    import numpy as np
    import ttnn

    if state.mesh is None or not state.layers:
        return {"error": "server not fully loaded"}

    layer_idx = int(args.get("layer_idx", 0))
    threshold = float(args.get("max_abs_diff", 0.05))

    dn_layers = [(i, L) for i, L in enumerate(state.layers) if L["type"] == "linear_attention"]
    if not dn_layers:
        return {"error": "no DeltaNet layers in state.layers"}
    if layer_idx < 0 or layer_idx >= len(dn_layers):
        return {"error": f"layer_idx out of range; only {len(dn_layers)} DeltaNet layers"}
    dn = dn_layers[layer_idx][1]["dn"]
    if "conv_st_split" not in dn or "w_conv_split" not in dn:
        return {"error": "this server lacks pre-split conv_st/w_conv tensors (rebuild against G4 server_tp)"}

    def readback(t):
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().cpu().numpy()

    conv_st_combined = readback(dn["conv_st"])
    w_conv_combined = readback(dn["w_conv"])
    conv_st_split_np = [readback(t) for t in dn["conv_st_split"]]
    w_conv_split_np = [readback(t) for t in dn["w_conv_split"]]

    comparisons = {}
    all_match = True

    def compare(name, combined, split_list, K):
        nonlocal all_match
        for k in range(K):
            if combined.ndim != 2:
                comparisons[f"{name}_{k}"] = {
                    "shape_match": False,
                    "combined_shape": list(combined.shape),
                    "reason": f"combined is not rank-2: shape {combined.shape}",
                }
                all_match = False
                continue
            expected = combined[:, k:k + 1]
            actual = split_list[k]
            if expected.shape != actual.shape:
                comparisons[f"{name}_{k}"] = {
                    "shape_match": False,
                    "expected_shape": list(expected.shape),
                    "actual_shape": list(actual.shape),
                }
                all_match = False
                continue
            diff = np.abs(expected - actual)
            max_diff = float(diff.max())
            pass_gate = max_diff <= threshold
            comparisons[f"{name}_{k}"] = {
                "shape_match": True,
                "expected_shape": list(expected.shape),
                "max_abs_diff": max_diff,
                "pass_gate": pass_gate,
                "expected_first_3": expected[:3].flatten().tolist(),
                "actual_first_3": actual[:3].flatten().tolist(),
            }
            if not pass_gate:
                all_match = False

    compare("conv_st_split", conv_st_combined, conv_st_split_np, 3)
    compare("w_conv_split", w_conv_combined, w_conv_split_np, 4)

    # =========================================================================
    # CHECK B — memory_config / layout asymmetry between split tensors and
    # column slices of the combined tensors. Even when data is bit-equivalent,
    # different memory_config (interleaved/sharded, L1/DRAM, alignment) can
    # cause the kernel to mis-interpret a tensor.
    # =========================================================================
    def tensor_meta(t):
        try:
            return {
                "memory_config": str(t.memory_config()),
                "layout": str(t.layout),
                "dtype": str(t.dtype),
                "shape": str(t.shape),
            }
        except Exception as e:
            return {"error": f"meta probe failed: {type(e).__name__}: {e}"}

    layout_check = {
        "combined_conv_st": tensor_meta(dn["conv_st"]),
        "combined_w_conv": tensor_meta(dn["w_conv"]),
        "split_conv_st_0": tensor_meta(dn["conv_st_split"][0]),
        "split_w_conv_0":  tensor_meta(dn["w_conv_split"][0]),
        "live_slice_conv_st_0": tensor_meta(
            ttnn.slice(dn["conv_st"], [0, 0], [dn["conv_st"].shape[0], 1])),
        "live_slice_w_conv_0": tensor_meta(
            ttnn.slice(dn["w_conv"], [0, 0], [dn["w_conv"].shape[0], 1])),
    }
    # If split's metadata differs from a live slice's metadata at any field, flag it.
    layout_meta_match = (
        layout_check["split_conv_st_0"].get("memory_config")
            == layout_check["live_slice_conv_st_0"].get("memory_config")
        and layout_check["split_w_conv_0"].get("memory_config")
            == layout_check["live_slice_w_conv_0"].get("memory_config")
    )

    # =========================================================================
    # CHECK A — single-forward conv_out comparison. State is freshly zeroed
    # before each run. Synthesize a fixed mixed_qkv (mesh-sharded dim=1 to
    # match production layout). Run both manual and owned conv1d blocks;
    # compare conv_out element-wise.
    # =========================================================================
    import numpy as np
    import torch
    from full_layer_tp_probe import CONV_DIM, CONV_DIM_CHIP
    cfg = state.cfg
    KERNEL = cfg["conv_kernel"]
    rng = np.random.default_rng(int(args.get("seed", 0)))
    mixed_full = rng.uniform(-0.1, 0.1, (1, CONV_DIM)).astype(np.float32)

    def fresh_mixed():
        return ttnn.from_torch(
            torch.from_numpy(mixed_full),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1),
        )

    # Reset state to zeros (also zeros split tensors).
    _reset_state_buffers(state)

    # --- MANUAL conv1d block (matches deltanet_step_tp else-branch) ---
    mixed_tt = fresh_mixed()
    mixed_col_m = ttnn.reshape(mixed_tt, [CONV_DIM_CHIP, 1])
    conv_input = ttnn.concat([dn["conv_st"], mixed_col_m], dim=-1)
    conv_prod = ttnn.mul(conv_input, dn["w_conv"])
    conv_out_manual = ttnn.silu(ttnn.sum(conv_prod, dim=-1))
    manual_back = ttnn.to_torch(
        conv_out_manual,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    ).float().cpu().numpy()
    if manual_back.ndim == 2 and manual_back.shape[1] == 1:
        manual_back = manual_back[:, 0]
    elif manual_back.ndim == 1:
        pass
    else:
        manual_back = manual_back.reshape(-1)

    # Reset state again to zero conv_st_split (manual run didn't touch them,
    # but be defensive).
    _reset_state_buffers(state)

    # --- OWNED conv1d block (matches deltanet_step_tp if-branch) ---
    mixed_tt2 = fresh_mixed()
    state0, state1, state2 = dn["conv_st_split"]
    w0, w1, w2, w3 = dn["w_conv_split"]
    mixed_col_o = ttnn.reshape(mixed_tt2, [CONV_DIM_CHIP, 1])
    _, _, _, conv_out_owned_2d = ttnn.experimental.qwen36_conv1d_decode_owned(
        mixed_col_o, state0, state1, state2, w0, w1, w2, w3)
    conv_out_owned = ttnn.reshape(conv_out_owned_2d, [CONV_DIM_CHIP])
    owned_back = ttnn.to_torch(
        conv_out_owned,
        mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0),
    ).float().cpu().numpy().reshape(-1)

    # Reset state one more time so we don't leave dn['conv_st_split'] polluted.
    _reset_state_buffers(state)

    forward_diff = np.abs(manual_back - owned_back)
    forward_check = {
        "manual_first_5": manual_back[:5].tolist(),
        "owned_first_5": owned_back[:5].tolist(),
        "max_abs_diff": float(forward_diff.max()),
        "mean_abs_diff": float(forward_diff.mean()),
        "p99_abs_diff": float(np.percentile(forward_diff, 99)),
        "num_above_1e_3": int((forward_diff > 1e-3).sum()),
        "shape_match": manual_back.shape == owned_back.shape,
        "shape": list(manual_back.shape),
    }
    FORWARD_THRESHOLD = 0.05  # ~half a BF16 ULP at magnitude 1
    forward_clean = forward_check["max_abs_diff"] <= FORWARD_THRESHOLD

    # =========================================================================
    # DIAGNOSIS — chained reasoning over all 3 checks
    # =========================================================================
    diagnosis_lines = []
    if all_match:
        diagnosis_lines.append(
            "CHECK split-data: PASS (bootstrap pre-split is bit-exact vs slicing)")
    else:
        diagnosis_lines.append(
            "CHECK split-data: FAIL (bootstrap pre-split doesn't match slice; "
            "fix relayout_conv or pre-split-at-slice-level)")
    if layout_meta_match:
        diagnosis_lines.append(
            "CHECK layout-meta: PASS (split and live-slice memory_config match)")
    else:
        diagnosis_lines.append(
            "CHECK layout-meta: FAIL (split memory_config differs from live-slice's; "
            "fix to upload split with same memory_config as ttnn.slice produces)")
    if forward_clean:
        diagnosis_lines.append(
            "CHECK single-forward: PASS (owned kernel produces same conv_out as "
            "manual chain on mesh at zero state). Bug is in MULTI-STEP state "
            "evolution — kernel's per-call output is right, but state mutation "
            "across forwards goes wrong.")
    else:
        diagnosis_lines.append(
            "CHECK single-forward: FAIL (owned conv_out differs from manual at "
            "step 0 with zero state, max_diff={:.4f}). Mesh kernel dispatch is "
            "buggy — kernel produces wrong output per-call on mesh despite "
            "passing G0 standalone single-device.".format(
                forward_check["max_abs_diff"]))

    state.last_run = {
        "cmd": "probe_deltanet_conv1d_split_check_tp",
        "layer_idx": layer_idx,
        "all_match": all_match,
        "layout_meta_match": layout_meta_match,
        "forward_clean": forward_clean,
    }

    return {
        "ok": True,
        "layer_idx": layer_idx,
        "threshold": threshold,
        "comparisons": comparisons,
        "all_match": all_match,
        "layout_check": layout_check,
        "layout_meta_match": layout_meta_match,
        "forward_check": forward_check,
        "forward_clean": forward_clean,
        "diagnosis": " | ".join(diagnosis_lines),
    }


def handle_probe_deltanet_owned_decay_gate_real_tensors_tp(state: MeshServerState, args: dict) -> dict:
    """G1 real-tensor probe for owned decay/gate kernel.

    For each DeltaNet layer (or a single layer if --layer-idx is set),
    reads the production dn['dt_bias'] + dn['A_log'] weights, synthesizes
    random a/b inputs of the same shape, runs:
      - numpy oracle (manual log(exp+1) softplus chain)
      - owned kernel (ttnn.experimental.qwen36_decay_gate_decode_owned)
    and compares decay + beta outputs element-wise. Sweeps all 48 DeltaNet
    layers by default. Gate: PCC ≥ 0.99999, max_abs_diff ≤ threshold.

    args:
      layer_idx: int or None (default None — sweep all DeltaNet layers)
      seed:      int (default 0)
      max_abs_diff: float (default 0.01)
    """
    import numpy as np
    import torch
    import ttnn
    from full_layer_tp_probe import N_V_HEADS  # 48

    if state.mesh is None or not state.layers:
        return {"error": "server not loaded"}
    if not hasattr(ttnn.experimental, "qwen36_decay_gate_decode_owned"):
        return {"error": "ttnn.experimental.qwen36_decay_gate_decode_owned not exposed"}

    seed = int(args.get("seed", 0))
    threshold = float(args.get("max_abs_diff", 0.01))
    layer_idx_filter = args.get("layer_idx")
    NV = N_V_HEADS  # 48

    dn_indices = [i for i, L in enumerate(state.layers) if L["type"] == "linear_attention"]
    if layer_idx_filter is not None:
        idx = int(layer_idx_filter)
        if idx < 0 or idx >= len(dn_indices):
            return {"error": f"layer_idx out of DeltaNet range; {len(dn_indices)} layers"}
        dn_indices = [dn_indices[idx]]

    rng = np.random.default_rng(seed)

    def numpy_oracle(a, b, dt_bias, A_log):
        a_biased = a + dt_bias
        softplus_a = np.log(np.exp(a_biased) + 1.0)
        g = -np.exp(A_log) * softplus_a
        decay = np.exp(g)
        beta = 1.0 / (1.0 + np.exp(-b))
        return decay, beta

    def mesh_upload_row(np_2d):
        # np_2d shape [1, NV]; sharded dim=1 across mesh → per-chip [1, NV_PER_CHIP].
        return ttnn.from_torch(
            torch.from_numpy(np_2d.astype(np.float32)),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=state.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(state.mesh, dim=1))

    def mesh_readback_row(tensor):
        return ttnn.to_torch(
            tensor, mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=1)
        ).float().cpu().numpy()

    def pcc(a, b):
        am = a.flatten() - a.mean()
        bm = b.flatten() - b.mean()
        return float(np.dot(am, bm) / (np.linalg.norm(am) * np.linalg.norm(bm) + 1e-30))

    per_layer = {}
    all_pass = True

    for layer_idx in dn_indices:
        dn = state.layers[layer_idx]["dn"]

        # Read real dt_bias and A_log; production shards them dim=0 so per-chip
        # is [NV_PER_CHIP]. Concat across mesh dim=0 gives full [NV] (rank-1).
        dt_bias_full = ttnn.to_torch(
            dn["dt_bias"], mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().cpu().numpy()
        A_log_full = ttnn.to_torch(
            dn["A_log"], mesh_composer=ttnn.ConcatMeshToTensor(state.mesh, dim=0)
        ).float().cpu().numpy()
        # Flatten + reshape to [1, NV] for kernel input.
        dt_bias_full = dt_bias_full.flatten()[:NV].reshape(1, NV)
        A_log_full = A_log_full.flatten()[:NV].reshape(1, NV)

        # Synthesize a/b matching production activation magnitudes.
        a_full = rng.uniform(-1.0, 1.0, (1, NV)).astype(np.float32)
        b_full = rng.uniform(-1.0, 1.0, (1, NV)).astype(np.float32)

        oracle_decay, oracle_beta = numpy_oracle(a_full, b_full, dt_bias_full, A_log_full)

        a_tt = mesh_upload_row(a_full)
        b_tt = mesh_upload_row(b_full)
        dt_bias_tt = mesh_upload_row(dt_bias_full)
        A_log_tt = mesh_upload_row(A_log_full)

        decay_tt, beta_tt = ttnn.experimental.qwen36_decay_gate_decode_owned(
            a_tt, b_tt, dt_bias_tt, A_log_tt)

        decay_kernel = mesh_readback_row(decay_tt)
        beta_kernel = mesh_readback_row(beta_tt)

        decay_diff = float(np.abs(decay_kernel - oracle_decay).max())
        beta_diff = float(np.abs(beta_kernel - oracle_beta).max())
        layer_pass = decay_diff <= threshold and beta_diff <= threshold
        if not layer_pass:
            all_pass = False

        per_layer[str(layer_idx)] = {
            "decay_max_diff": decay_diff,
            "decay_pcc": pcc(decay_kernel, oracle_decay),
            "beta_max_diff": beta_diff,
            "beta_pcc": pcc(beta_kernel, oracle_beta),
            "pass": layer_pass,
        }

    state.last_run = {
        "cmd": "probe_deltanet_owned_decay_gate_real_tensors_tp",
        "n_layers_swept": len(dn_indices),
        "all_pass": all_pass,
    }

    return {
        "ok": True,
        "n_layers_swept": len(dn_indices),
        "all_pass": all_pass,
        "threshold": threshold,
        "per_layer": per_layer,
    }


HANDLERS = {
    "status":         handle_status,
    "generate_tp":    handle_generate_tp,
    "bench_decode_tp_components": handle_bench_decode_tp_components,
    "probe_ccl_components_tp": handle_probe_ccl_components_tp,
    "probe_async_ccl_components_tp": handle_probe_async_ccl_components_tp,
    "probe_prefill_vs_decode_loop_tp": handle_probe_prefill_vs_decode_loop_tp,
    "probe_multirow_construct_vs_per_position": handle_probe_multirow_construct_vs_per_position,
    "probe_slice_write_round_trip": handle_probe_slice_write_round_trip,
    "probe_dn_source_isolation_tp": handle_probe_dn_source_isolation_tp,
    "probe_dn_op_isolation_tp": handle_probe_dn_op_isolation_tp,
    "probe_fused_paged_update_cache_tp": handle_probe_fused_paged_update_cache_tp,
    "probe_explicit_all_reduce_tp": handle_probe_explicit_all_reduce_tp,
    "probe_rope_fused_qk_tp": handle_probe_rope_fused_qk_tp,
    "probe_rope_native_partial_tp": handle_probe_rope_native_partial_tp,
    "probe_rope_native_partial_trace_tp": handle_probe_rope_native_partial_trace_tp,
    "profile_decode_tp_ops": handle_profile_decode_tp_ops,
    "probe_deltanet_recurrence_matmul_tp": handle_probe_deltanet_recurrence_matmul_tp,
    "probe_deltanet_native_gdn_synthetic_mesh_tp": handle_probe_deltanet_native_gdn_synthetic_mesh_tp,
    "probe_deltanet_native_gdn_real_tensors_tp": handle_probe_deltanet_native_gdn_real_tensors_tp,
    "probe_deltanet_owned_gdn_real_tensors_tp": handle_probe_deltanet_owned_gdn_real_tensors_tp,
    "probe_deltanet_owned_gdn_trace_tp": handle_probe_deltanet_owned_gdn_trace_tp,
    "probe_deltanet_owned_gdn_divergence_tp": handle_probe_deltanet_owned_gdn_divergence_tp,
    "probe_deltanet_owned_gdn_teacher_forced_tp": handle_probe_deltanet_owned_gdn_teacher_forced_tp,
    "probe_deltanet_owned_gdn_benchmark_tp": handle_probe_deltanet_owned_gdn_benchmark_tp,
    "probe_deltanet_softplus_decay_tp": handle_probe_deltanet_softplus_decay_tp,
    "cosine_ladder_tp": handle_cosine_ladder_tp,
    "probe_deltanet_conv1d_split_check_tp": handle_probe_deltanet_conv1d_split_check_tp,
    "probe_deltanet_owned_decay_gate_real_tensors_tp": handle_probe_deltanet_owned_decay_gate_real_tensors_tp,
    "probe_ccl_equivalence_tp": handle_probe_ccl_equivalence_tp,
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
