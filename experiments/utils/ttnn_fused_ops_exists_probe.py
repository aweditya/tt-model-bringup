#!/usr/bin/env python3
"""Verify which fused ttnn ops exist in our build (no device opened).

For each candidate fused op identified by the tt-metal source survey, check:
  1. The Python binding exists (getattr resolves)
  2. The signature (callable, has docstring or accepts expected kwargs)

Does NOT open the mesh. Pure introspection, safe to run alongside the qb1 server.

Usage:
  ssh qb1 'cd ~/tt-xla && python experiments/utils/ttnn_fused_ops_exists_probe.py'
"""
import sys


def _check(name, fn=None):
    try:
        obj = fn() if fn else None
        if obj is None:
            print(f"  MISSING  {name}")
            return False
        # Truncate docstring/signature noise
        doc = (getattr(obj, "__doc__", "") or "")[:140].replace("\n", " ")
        print(f"  EXISTS   {name}    [{doc[:120]}]")
        return True
    except AttributeError as e:
        print(f"  MISSING  {name}    [AttributeError: {e}]")
        return False
    except Exception as e:
        print(f"  ERROR    {name}    [{type(e).__name__}: {e}]")
        return False


def main():
    print("=== ttnn fused-op existence probe (no device) ===", flush=True)
    import ttnn

    ok = {}
    print("\n[1] Normalization / RMS norm family")
    ok["rms_norm"] = _check("ttnn.rms_norm", lambda: ttnn.rms_norm)
    ok["rms_norm_pre_all_gather"] = _check(
        "ttnn.rms_norm_pre_all_gather", lambda: ttnn.rms_norm_pre_all_gather)
    ok["rms_norm_post_all_gather"] = _check(
        "ttnn.rms_norm_post_all_gather", lambda: ttnn.rms_norm_post_all_gather)
    ok["layer_norm"] = _check("ttnn.layer_norm", lambda: ttnn.layer_norm)

    print("\n[2] Rotary embedding family")
    ok["rotary_embedding"] = _check(
        "ttnn.experimental.rotary_embedding",
        lambda: ttnn.experimental.rotary_embedding)
    ok["rotary_embedding_hf"] = _check(
        "ttnn.experimental.rotary_embedding_hf",
        lambda: ttnn.experimental.rotary_embedding_hf)
    ok["rotary_embedding_llama"] = _check(
        "ttnn.experimental.rotary_embedding_llama",
        lambda: ttnn.experimental.rotary_embedding_llama)
    ok["rotary_embedding_llama_fused_qk"] = _check(
        "ttnn.experimental.rotary_embedding_llama_fused_qk",
        lambda: ttnn.experimental.rotary_embedding_llama_fused_qk)
    ok["rotate_half"] = _check(
        "ttnn.experimental.rotate_half",
        lambda: ttnn.experimental.rotate_half)

    print("\n[3] SDPA variants")
    ok["sdpa_decode"] = _check(
        "ttnn.transformer.scaled_dot_product_attention_decode",
        lambda: ttnn.transformer.scaled_dot_product_attention_decode)
    ok["paged_sdpa_decode"] = _check(
        "ttnn.transformer.paged_scaled_dot_product_attention_decode",
        lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode)
    ok["flash_mla_decode"] = _check(
        "ttnn.transformer.flash_multi_latent_attention_decode",
        lambda: ttnn.transformer.flash_multi_latent_attention_decode)

    print("\n[4] Paged cache fused updates")
    ok["paged_update_cache"] = _check(
        "ttnn.experimental.paged_update_cache",
        lambda: ttnn.experimental.paged_update_cache)
    ok["paged_fused_update_cache"] = _check(
        "ttnn.experimental.paged_fused_update_cache",
        lambda: ttnn.experimental.paged_fused_update_cache)

    print("\n[5] SSM-family fused ops (DeltaNet candidates)")
    try:
        ok["prefix_scan"] = _check(
            "ttnn.experimental.prefix_scan",
            lambda: ttnn.experimental.prefix_scan)
    except Exception:
        ok["prefix_scan"] = False
    try:
        ok["hc_sum_reduce"] = _check(
            "ttnn.experimental.hc_sum_reduce",
            lambda: ttnn.experimental.hc_sum_reduce)
    except Exception:
        ok["hc_sum_reduce"] = False
    try:
        ok["repeat_and_interleave_eltwise_mul"] = _check(
            "ttnn.experimental.repeat_and_interleave_eltwise_mul",
            lambda: ttnn.experimental.repeat_and_interleave_eltwise_mul)
    except Exception:
        ok["repeat_and_interleave_eltwise_mul"] = False

    print("\n[6] Fused matmul/binary activation paths")
    # `ttnn.linear(... activation='silu')` already used in server_tp mlp_step_tp.
    # Verify the eltwise mul fused-activation flag (Llama70B SwiGLU pattern).
    ok["UnaryOpType_SILU"] = _check(
        "ttnn.UnaryOpType.SILU", lambda: ttnn.UnaryOpType.SILU)

    print("\n[7] QKV head splitting / concat (alternative for our slice+reshape blocks)")
    ok["nlp_create_qkv_heads_decode"] = _check(
        "ttnn.experimental.nlp_create_qkv_heads_decode",
        lambda: ttnn.experimental.nlp_create_qkv_heads_decode)
    ok["nlp_concat_heads_decode"] = _check(
        "ttnn.experimental.nlp_concat_heads_decode",
        lambda: ttnn.experimental.nlp_concat_heads_decode)
    ok["create_qkv_heads"] = _check(
        "ttnn.experimental.create_qkv_heads",
        lambda: ttnn.experimental.create_qkv_heads)

    print("\n[8] Distributed RMSNorm pre/post all-gather (mesh-safe)")
    ok["rms_norm_pre_all_gather_v2"] = _check(
        "ttnn.experimental.rmsnorm_pre_all_gather",
        lambda: ttnn.experimental.rmsnorm_pre_all_gather)
    ok["rms_norm_post_all_gather_v2"] = _check(
        "ttnn.experimental.rmsnorm_post_all_gather",
        lambda: ttnn.experimental.rmsnorm_post_all_gather)
    # Try fused_distributed_rmsnorm (newer single-call variant)
    ok["fused_distributed_rmsnorm"] = _check(
        "ttnn.experimental.fused_distributed_rmsnorm",
        lambda: ttnn.experimental.fused_distributed_rmsnorm)

    # Summary
    have = sorted(k for k, v in ok.items() if v)
    miss = sorted(k for k, v in ok.items() if not v)
    print("\n=== summary ===")
    print(f"  EXISTS ({len(have)}):    {', '.join(have)}")
    print(f"  MISSING ({len(miss)}):   {', '.join(miss)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
