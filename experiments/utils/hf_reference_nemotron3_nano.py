#!/usr/bin/env python3
"""MM7 v0.0 — HuggingFace reference for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

Generates a per-layer activation snapshot that the upcoming
`server_nemotron3_ttnn.py` integration will use as ground truth for
layer-by-layer cosine comparison (v0.1.1 onwards).

Architecture (per research/nemotron3_nano_architecture_brief.md §1):
  - `NemotronHForCausalLM` (model_type="nemotron_h"); requires
    `trust_remote_code=True` for AutoModel.
  - 52 hybrid-pattern layers: 23 Mamba2 + 23 MoE + 6 GQA Attention.
    Per-layer kind lives in `cfg.hybrid_override_pattern` (string of
    M/E/* characters — see brief §1 for the exact pattern).
  - Tokenizer is ChatML (`<|im_start|>{role}\\n…<|im_end|>\\n`).
  - hidden_dim = 2688 (per brief; verify at load time).

What this script saves (under `.cache/hf_oracle_nemotron3_nano/`):
  - meta.json: prompt, prompt_ids, predicted_token, predicted_text,
    layer_kinds (decoded from hybrid_override_pattern)
  - prompt_ids.npy: [seq] int32 tokens
  - hidden_states.npy: [n_layers+1, seq, hidden_dim] fp32 — index 0 is
    the embed output, indices 1..52 are after each decoder layer.
  - logits.npy: [seq, vocab=131072] fp32 final logits (post lm_head)
  - final_norm.npy: [seq, hidden_dim] fp32 (post final norm, pre lm_head)
  - argmax.npy: [seq] int32 — greedy argmax at every position.

Memory: HF NemotronHForCausalLM in bf16 on CPU expects ~62 GB
(architecture-brief §2). QuietBox host has ~478 GB free — comfortable.
Loading takes ~5 min for the first time (downloads ~63 GB of shards).

Run on the QuietBox:
  cd ~/tt-xla
  .venv/bin/python -u experiments/utils/hf_reference_nemotron3_nano.py

Optional sub-hooks (saves L<N>_<sub>.npy alongside hidden_states):
  --hook-mamba2-layer N  : hook in_proj, conv1d, ssm_chunk, out_proj at L<N>
  --hook-moe-layer N     : hook the router + per-expert output on L<N>
  --hook-attn-layer N    : hook q/k/v/o projections + GQA repeat-kv on L<N>

REUSE: forks the structure of `experiments/utils/hf_reference_35b.py`
(Qwen3.6 35B) and `hf_reference_gemma4_12b.py`. Same on-disk artefact
shape so the existing per-layer-ladder probes work unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# CPU-only oracle: NemotronHBlock.forward (modeling_nemotron_h.py:769)
# wraps the layer body in `torch.cuda.stream(torch.cuda.default_stream(...))`
# unconditionally, which calls _lazy_init() and crashes on a host with
# no NVIDIA driver. Patch them to no-ops *before* the model is loaded.
if not torch.cuda.is_available():
    import contextlib

    @contextlib.contextmanager
    def _noop_stream(_stream=None):
        yield

    torch.cuda.stream = _noop_stream

    def _noop_default_stream(_device=None):
        return None

    torch.cuda.default_stream = _noop_default_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".cache" / "hf_oracle_nemotron3_nano"

# Same single-line prompt the 27B / 35B oracles use, so per-layer
# cosine ladders are directly comparable across models.
DEFAULT_PROMPT = "The capital of France is"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_layer_kinds(pattern: str) -> list[str]:
    """Convert hybrid_override_pattern (e.g. "MEMEM*EMEMEM*...") into a
    list of layer kinds: 'mamba2' | 'moe' | 'attention'.
    """
    mapping = {"M": "mamba2", "E": "moe", "*": "attention"}
    out = []
    for ch in pattern:
        if ch not in mapping:
            raise ValueError(f"unknown hybrid_override_pattern char {ch!r}")
        out.append(mapping[ch])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None,
                    help="text to tokenize (overrides --prompt-file)")
    ap.add_argument("--prompt-file", default=None,
                    help="path to a UTF-8 file whose contents are the prompt")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                    help="directory to save oracle artefacts")
    ap.add_argument("--no-special-tokens", action="store_true",
                    help="pass add_special_tokens=False to the tokenizer (use when prompt is "
                         "already chat-template-rendered).")
    ap.add_argument("--hook-mamba2-layer", type=int, default=None,
                    help="ALSO hook Mamba2 sub-modules (in_proj, conv1d, norm, out_proj, "
                         "and the mixer output itself) on this layer index "
                         "(must be a 'mamba2' kind per hybrid_override_pattern).")
    ap.add_argument("--hook-moe-layer", type=int, default=None,
                    help="ALSO hook MoE router (idx + weights), shared_experts output, "
                         "and the mixer output on this layer (must be 'moe').")
    ap.add_argument("--hook-attn-layer", type=int, default=None,
                    help="ALSO hook q/k/v/o projection outputs AND the mixer output on this "
                         "layer (must be 'attention').")
    ap.add_argument("--gen", type=int, default=0,
                    help="If >0, run greedy generation for this many tokens after the "
                         "prompt and save per-position hidden_states / logits for ALL "
                         "(prompt + gen) positions. Used for v0.3 multi-step validation.")
    args = ap.parse_args()

    if args.prompt is not None:
        prompt_text = args.prompt
    elif args.prompt_file is not None:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt_text = DEFAULT_PROMPT

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output dir: {out_dir}")

    log(f"loading config + tokenizer ({MODEL_ID})…")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Some Nemotron configs wrap the text-model config under .text_config;
    # newer ones flatten everything at top level. Handle both.
    text_cfg = getattr(cfg, "text_config", cfg)
    pattern = getattr(text_cfg, "hybrid_override_pattern", None)
    if pattern is None:
        raise RuntimeError(
            "cfg.hybrid_override_pattern missing — confirm NemotronH config "
            "structure (see architecture brief §1 for expected shape).")
    layer_kinds = decode_layer_kinds(pattern)
    n_layers = len(layer_kinds)
    n_mamba = sum(1 for k in layer_kinds if k == "mamba2")
    n_moe = sum(1 for k in layer_kinds if k == "moe")
    n_attn = sum(1 for k in layer_kinds if k == "attention")
    log(f"  {n_layers} layers: {n_mamba} Mamba2 + {n_moe} MoE + {n_attn} attention")
    log(f"  hidden_dim = {text_cfg.hidden_size}, vocab = {text_cfg.vocab_size}")

    log("loading model (bf16, CPU)…")
    t0 = time.time()
    # use_mamba_kernels=False avoids the modeling code's hard import of
    # `mamba-ssm` (which is CUDA-only). On CPU we route through a slower
    # but pure-PyTorch Mamba2 path — fine for one-shot oracle generation.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_mamba_kernels=False,
    )
    model.eval()
    log(f"  model loaded in {time.time() - t0:.1f}s")

    log("tokenising prompt …")
    add_special = not args.no_special_tokens
    inputs = tok(prompt_text, return_tensors="pt", add_special_tokens=add_special)
    prompt_ids = inputs["input_ids"][0].tolist()
    log(f"  prompt: {prompt_text!r}")
    log(f"  prompt_ids: {prompt_ids}")

    sub_hooks: dict[str, torch.Tensor] = {}

    def register_sub_hooks() -> list:
        """Optionally install sub-module hooks on the requested layer indices."""
        handles = []
        # Nemotron-H stores the decoder at `model.backbone` (set in
        # `NemotronHForCausalLM.__init__`, modeling_nemotron_h.py:1585).
        # The inherited `get_decoder()` returns `self.model` which is only
        # set if `set_decoder()` was called first — so go direct.
        if hasattr(model, "backbone"):
            decoder_layers = model.backbone.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            decoder_layers = model.model.layers
        else:
            raise RuntimeError(
                "could not find decoder layers on model "
                f"(type {type(model).__name__}, attrs: "
                f"{[a for a in dir(model) if not a.startswith('_')][:20]})"
            )
        # Block-level mixer output hook factory — captures what comes OUT
        # of the entire layer.mixer call (= the residual input added to the
        # layer's residual stream). Closes the per-layer cosine gate for
        # v0.1.x. The mixer's output is either a Tensor or a tuple
        # (e.g. attention returns (output, attn_weights[, …])); take [0].
        def make_mixer_out_hook(layer_idx: int, label: str):
            def hook(_m, _inp, output):
                out_t = output[0] if isinstance(output, tuple) else output
                sub_hooks[f"L{layer_idx}_{label}_mixer_out"] = \
                    out_t.detach().float().cpu().numpy()
            return hook

        if args.hook_mamba2_layer is not None:
            Nm = args.hook_mamba2_layer
            if layer_kinds[Nm] != "mamba2":
                raise ValueError(
                    f"--hook-mamba2-layer {Nm} is {layer_kinds[Nm]!r}, not mamba2")
            layer = decoder_layers[Nm]
            # Cross-check the modeling code's own classification — catches
            # silent drift if hybrid_override_pattern ever stops matching
            # the per-layer block_type.
            bt = getattr(layer, "block_type", None)
            if bt != "mamba":
                raise RuntimeError(
                    f"layer {Nm} block_type={bt!r}, expected 'mamba' — "
                    f"hybrid_override_pattern may have drifted from model state")
            mixer = getattr(layer, "mixer", None) or getattr(layer, "mamba", None)
            if mixer is None:
                raise RuntimeError(f"could not find Mamba2 mixer on layer {Nm}")

            # NB: capture Nm and submodule_path as default args to dodge
            # the Python closure-late-binding trap — `N` was being rebound
            # by the moe/attn blocks below, so all hooks fired with the
            # final value (L5_*) regardless of intended layer.
            def hook_named(submodule_path: str, layer_idx: int = Nm):
                def hook(_m, _inp, output):
                    out_t = output[0] if isinstance(output, tuple) else output
                    sub_hooks[f"L{layer_idx}_{submodule_path}"] = \
                        out_t.detach().float().cpu().numpy()
                return hook

            # `norm` is MambaRMSNormGated — the gated norm AFTER the SSM
            # step, BEFORE out_proj. This is the kernel post-condition we
            # validate at v0.1.2: our owned kernel emits `y` (pre-norm),
            # and ttnn rms_norm_gated emits this `norm` output.
            # Pre-forward hook on `mixer.norm` captures its INPUT — that's
            # the y output of the SSD step (= what our wrapper should
            # match). Critical for v0.1.2.d oracle-vs-HF diagnostic.
            def make_y_pre_norm_hook(layer_idx: int = Nm):
                def hook(_m, inputs):
                    # inputs is a tuple; the first positional arg is y,
                    # the second (kwarg `gate`) is z. Save both.
                    y = inputs[0] if len(inputs) > 0 else None
                    if y is not None:
                        sub_hooks[f"L{layer_idx}_y_pre_norm"] = \
                            y.detach().float().cpu().numpy()
                return hook
            if hasattr(mixer, "norm"):
                handles.append(
                    mixer.norm.register_forward_pre_hook(make_y_pre_norm_hook())
                )
            for name in ["in_proj", "conv1d", "norm", "out_proj"]:
                sub = getattr(mixer, name, None)
                if sub is not None:
                    handles.append(sub.register_forward_hook(hook_named(name)))
                else:
                    log(f"  warn: {name} not found on layer {Nm}'s mixer (sub-hook skipped)")
            # Whole-mixer output for the per-layer cosine ladder gate.
            handles.append(mixer.register_forward_hook(make_mixer_out_hook(Nm, "mamba2")))

        if args.hook_moe_layer is not None:
            Ne = args.hook_moe_layer
            if layer_kinds[Ne] != "moe":
                raise ValueError(f"--hook-moe-layer {Ne} is {layer_kinds[Ne]!r}, not moe")
            layer = decoder_layers[Ne]
            bt = getattr(layer, "block_type", None)
            if bt != "moe":
                raise RuntimeError(
                    f"layer {Ne} block_type={bt!r}, expected 'moe' — "
                    f"hybrid_override_pattern may have drifted from model state")
            # NemotronHBlock uses a uniform `mixer` attr for all layer types
            # (modeling_nemotron_h.py:752-758) — including MoE.
            mlp = getattr(layer, "mixer", None) or getattr(layer, "mlp", None)
            if mlp is None:
                raise RuntimeError(f"could not find MoE mixer on layer {Ne}")
            if hasattr(mlp, "gate"):
                # NemotronHTopkRouter.forward returns a tuple
                # (topk_indices, topk_weights) — save both for v0.1.3
                # group-restricted topk validation.
                def moe_router_hook(_m, _inp, output, layer_idx: int = Ne):
                    if isinstance(output, tuple):
                        idx, w = output[0], output[1]
                        sub_hooks[f"L{layer_idx}_moe_router_idx"] = \
                            idx.detach().float().cpu().numpy()
                        sub_hooks[f"L{layer_idx}_moe_router_w"] = \
                            w.detach().float().cpu().numpy()
                    else:
                        sub_hooks[f"L{layer_idx}_moe_router"] = \
                            output.detach().float().cpu().numpy()
                handles.append(mlp.gate.register_forward_hook(moe_router_hook))
            # Shared expert (single NemotronHMLP, 2× wider than routed)
            # — its output is added straight to the routed combine. We
            # need it isolated for v0.1.3 sub-step validation.
            if hasattr(mlp, "shared_experts"):
                def shared_expert_hook(_m, _inp, output, layer_idx: int = Ne):
                    out_t = output[0] if isinstance(output, tuple) else output
                    sub_hooks[f"L{layer_idx}_moe_shared_out"] = \
                        out_t.detach().float().cpu().numpy()
                handles.append(mlp.shared_experts.register_forward_hook(shared_expert_hook))
            handles.append(mlp.register_forward_hook(make_mixer_out_hook(Ne, "moe")))

        if args.hook_attn_layer is not None:
            Na = args.hook_attn_layer
            if layer_kinds[Na] != "attention":
                raise ValueError(f"--hook-attn-layer {Na} is {layer_kinds[Na]!r}, not attention")
            layer = decoder_layers[Na]
            bt = getattr(layer, "block_type", None)
            if bt != "attention":
                raise RuntimeError(
                    f"layer {Na} block_type={bt!r}, expected 'attention' — "
                    f"hybrid_override_pattern may have drifted from model state")
            attn = getattr(layer, "self_attn", None) or getattr(layer, "mixer", None)
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                sub = getattr(attn, name, None)
                if sub is not None:
                    def make_hook(n: str, layer_idx: int = Na):
                        def hook(_m, _inp, output):
                            sub_hooks[f"L{layer_idx}_attn_{n}"] = \
                                output.detach().float().cpu().numpy()
                        return hook
                    handles.append(sub.register_forward_hook(make_hook(name)))
            handles.append(attn.register_forward_hook(make_mixer_out_hook(Na, "attn")))

        return handles

    handles = register_sub_hooks()
    if sub_hooks or handles:
        log(f"  registered {len(handles)} sub-hook(s) on requested layer(s)")

    # Optional multi-step greedy decode (v0.3 validation gate).
    # If --gen > 0, we extend the prompt by N greedy-decoded tokens
    # FIRST, then run the captured forward over the full extended
    # sequence — so hidden_states / logits / argmax cover all
    # (prompt_len + gen) positions. This mirrors how a teacher-forced
    # TT trace would be cosine-compared at each position.
    input_ids = inputs["input_ids"]
    if args.gen > 0:
        log(f"greedy generate {args.gen} tokens (extends sequence)…")
        gen_t0 = time.time()
        with torch.no_grad():
            for _ in range(args.gen):
                tmp = model(input_ids=input_ids, use_cache=False)
                next_id = tmp.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=1)
        log(f"  generated {args.gen} tokens in {time.time() - gen_t0:.1f}s "
            f"(seq now {input_ids.shape[1]})")

    log("HF forward pass with output_hidden_states=True …")
    t0 = time.time()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    log(f"  forward in {time.time() - t0:.1f}s")
    for h in handles:
        h.remove()

    hidden_states = out.hidden_states  # tuple of (n_layers+1) tensors [1, seq, hidden]
    logits = out.logits[0].float().numpy()  # [seq, vocab]
    log(f"  hidden_states: {len(hidden_states)} tensors, "
        f"first shape: {list(hidden_states[0].shape)}")
    log(f"  logits shape: {logits.shape}")

    # Shape gates — catch silent modeling-code drift before the artifacts
    # ship to a downstream cosine probe.
    expected_n_hs = n_layers + 1
    if len(hidden_states) != expected_n_hs:
        raise RuntimeError(
            f"hidden_states length {len(hidden_states)} != expected {expected_n_hs} "
            f"(n_layers={n_layers} + embed)")
    if hidden_states[0].shape[-1] != text_cfg.hidden_size:
        raise RuntimeError(
            f"hidden_states last dim {hidden_states[0].shape[-1]} != "
            f"config.hidden_size {text_cfg.hidden_size}")
    if logits.shape[-1] != text_cfg.vocab_size:
        raise RuntimeError(
            f"logits last dim {logits.shape[-1]} != "
            f"config.vocab_size {text_cfg.vocab_size}")

    hs_stack = torch.stack([h[0] for h in hidden_states]).float().numpy()
    log(f"  hs_stack: {hs_stack.shape}  dtype={hs_stack.dtype}")

    # Pull final-norm output (everything before lm_head). Same trick the 35B
    # oracle uses — we re-run final_norm on the LAST hidden_states tensor.
    # Same Nemotron-H attr-naming quirk as register_sub_hooks above: decoder
    # is `model.backbone`, NOT `model.model`.
    if hasattr(model, "backbone"):
        decoder = model.backbone
    elif hasattr(model, "model"):
        decoder = model.model
    else:
        raise RuntimeError("could not find decoder module")
    final_norm = getattr(decoder, "norm", None) or getattr(decoder, "final_layernorm", None) \
        or getattr(decoder, "norm_f", None)
    if final_norm is None:
        raise RuntimeError("could not find final norm module")
    with torch.no_grad():
        fn_out = final_norm(hidden_states[-1]).float().numpy()[0]  # [seq, hidden]
    log(f"  final_norm: shape={fn_out.shape}")

    argmax = logits.argmax(axis=-1).astype(np.int32)
    pred_token = int(argmax[-1])
    pred_text = tok.decode([pred_token])
    log(f"  argmax[-1] = {pred_token}  -> {pred_text!r}")

    # ── Save artefacts ─────────────────────────────────────────────────
    # If --gen ran, input_ids now has prompt_len + gen tokens. Save the
    # FULL extended sequence as the canonical prompt_ids (matches what
    # the forward saw).
    full_ids = input_ids[0].tolist()
    np.save(out_dir / "prompt_ids.npy", np.array(full_ids, dtype=np.int32))
    np.save(out_dir / "hidden_states.npy", hs_stack)
    np.save(out_dir / "logits.npy", logits)
    np.save(out_dir / "final_norm.npy", fn_out)
    np.save(out_dir / "argmax.npy", argmax)
    for key, arr in sub_hooks.items():
        np.save(out_dir / f"{key}.npy", arr)

    meta = {
        "model_id": MODEL_ID,
        "prompt": prompt_text,
        "prompt_ids": prompt_ids,
        "prompt_len": len(prompt_ids),
        "gen": args.gen,
        "full_ids": full_ids,
        "predicted_token": pred_token,
        "predicted_text": pred_text,
        "n_layers": n_layers,
        "n_mamba2": n_mamba,
        "n_moe": n_moe,
        "n_attention": n_attn,
        "hidden_size": text_cfg.hidden_size,
        "vocab_size": text_cfg.vocab_size,
        "hybrid_override_pattern": pattern,
        "layer_kinds": layer_kinds,
        "sub_hooks": sorted(sub_hooks.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"saved meta.json + {5 + len(sub_hooks)} npys to {out_dir}")

    log("\nv0.0 oracle PASS ✓ — ready for v0.1.x per-layer cosine ladders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
