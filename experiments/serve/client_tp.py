"""Minimal CLI client for server_tp.py (multi-chip persistent server on qb2).

Usage:
    python -m experiments.serve.client_tp status
    python -m experiments.serve.client_tp generate_tp --prompt "..." --max-tokens 60
    python -m experiments.serve.client_tp shutdown

Points at $TT_CACHE/server_tp.sock (different from the single-chip server.sock).
"""
import argparse
import json
import os
import socket
import sys

from experiments.serve import protocol as P


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOCKET_PATH = os.path.join(PROJECT_ROOT, ".cache", "server_tp.sock")


def _send(cmd: str, args: dict, timeout: float = 7200.0) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client_tp: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        sock.sendall(P.pack_request(cmd, args))
        raw = P.read_line(sock, max_bytes=64 << 20)
    finally:
        sock.close()
    if not raw:
        print("client_tp: server returned no data (process likely died)", file=sys.stderr)
        sys.exit(3)
    resp = P.parse_response(raw)
    if resp.type == "error":
        print(f"server error: {resp.msg}", file=sys.stderr)
        sys.exit(4)
    return resp.data or {}


def cmd_status(_):
    print(json.dumps(_send("status", {}), indent=2, default=str))


def cmd_shutdown(_):
    print(json.dumps(_send("shutdown", {}), indent=2))


def cmd_bench_decode_tp_components(args):
    data = _send("bench_decode_tp_components", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_fused_paged_update_cache_tp(args):
    data = _send("probe_fused_paged_update_cache_tp", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
        "bench_trace": not args.skip_trace_bench,
        "allow_wedge_prone_disjoint": args.allow_wedge_prone_disjoint,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_explicit_all_reduce_tp(args):
    data = _send("probe_explicit_all_reduce_tp", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_rope_fused_qk_tp(args):
    data = _send("probe_rope_fused_qk_tp", {
        "positions": args.positions,
        "allow_wedge_prone_fused_qk": args.allow_wedge_prone_fused_qk,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_rope_native_partial_tp(args):
    data = _send("probe_rope_native_partial_tp", {
        "positions": args.positions,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_rope_native_partial_trace_tp(args):
    data = _send("probe_rope_native_partial_trace_tp", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_profile_decode_tp_ops(args):
    data = _send("profile_decode_tp_ops", {
        "prompt": args.prompt,
        "timed": args.timed,
        "include_records": args.include_records,
        "deltanet_decay_mode": args.deltanet_decay_mode,
        "deltanet_recurrence_mode": args.deltanet_recurrence_mode,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_recurrence_matmul_tp(args):
    data = _send("probe_deltanet_recurrence_matmul_tp", {
        "iters": args.iters,
        "warmup": args.warmup,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_native_gdn_real_tensors_tp(args):
    data = _send("probe_deltanet_native_gdn_real_tensors_tp", {
        "prompt": args.prompt,
        "layer_idx": args.layer_idx,
        "mode": args.mode,
        "reset_state": not args.no_reset_state,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_gdn_real_tensors_tp(args):
    data = _send("probe_deltanet_owned_gdn_real_tensors_tp", {
        "prompt": args.prompt,
        "layer_idx": args.layer_idx,
        "reset_state": not args.no_reset_state,
        "use_pretransposed_k": args.use_pretransposed_k,
        "compact_vectors": args.compact_vectors,
        "native_io": args.native_io,
        "stepwise": args.stepwise,
        "seed_state": args.seed_state,
        "direct_state_input": args.direct_state_input,
        "component_debug_modes": args.component_debug_mode,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_gdn_trace_tp(args):
    data = _send("probe_deltanet_owned_gdn_trace_tp", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "deltanet_decay_mode": args.deltanet_decay_mode,
        "deltanet_recurrence_mode": args.deltanet_recurrence_mode,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_gdn_benchmark_tp(args):
    data = _send("probe_deltanet_owned_gdn_benchmark_tp", {
        "prompts": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
        "deltanet_decay_mode": args.deltanet_decay_mode,
        "deltanet_recurrence_mode": args.deltanet_recurrence_mode,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_gdn_divergence_tp(args):
    data = _send("probe_deltanet_owned_gdn_divergence_tp", {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_gdn_teacher_forced_tp(args):
    data = _send("probe_deltanet_owned_gdn_teacher_forced_tp", {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "top_k": args.top_k,
        "state_layers": args.state_layer,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_native_gdn_synthetic_mesh_tp(args):
    data = _send("probe_deltanet_native_gdn_synthetic_mesh_tp", {
        "slots": args.slots,
        "key_dim": args.key_dim,
        "value_dim": args.value_dim,
        "seed": args.seed,
        "scale": args.scale,
        "distribution": args.distribution,
        "iters": args.iters,
        "warmup": args.warmup,
    })
    if args.output_json:
        from pathlib import Path
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_softplus_decay_tp(args):
    data = _send("probe_deltanet_softplus_decay_tp", {
        "prompt": args.prompt,
        "iters": args.iters,
        "warmup": args.warmup,
        "max_tokens": args.max_tokens,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_conv1d_split_check_tp(args):
    data = _send("probe_deltanet_conv1d_split_check_tp", {
        "layer_idx": args.layer_idx,
        "max_abs_diff": args.max_abs_diff,
    })
    print(json.dumps(data, indent=2, default=str))


def cmd_probe_deltanet_owned_decay_gate_real_tensors_tp(args):
    payload = {
        "seed": args.seed,
        "max_abs_diff": args.max_abs_diff,
    }
    if args.layer_idx is not None:
        payload["layer_idx"] = args.layer_idx
    data = _send("probe_deltanet_owned_decay_gate_real_tensors_tp", payload)
    print(json.dumps(data, indent=2, default=str))


def cmd_cosine_ladder_tp(args):
    """Run cosine ladder for one or more recurrence modes and, if ≥2 modes,
    print + save a per-position comparison vs the first mode. Used to gate
    long-context coherence of the owned GDN kernel vs the manual TTNN
    recurrence (Tier 1 of the owned_gdn promotion gate).

    Workflow:
      1. Call generate_tp once to obtain the baseline greedy token stream
         (uses whatever mode the server's traced forward was captured with —
         currently 'manual').
      2. For each mode in --modes: call cosine_ladder_tp on the SAME
         prompt_ids + generated_ids; server saves NPZ.
      3. Load NPZs; compute per-position cosine + top-1 disagreement of every
         non-base mode vs the first mode listed. Save comparison JSON.
    """
    import json as _json, os as _os, time as _t
    import numpy as np

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("client_tp: --modes is empty", file=sys.stderr); sys.exit(2)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(7200.0)
    try:
        sock.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client_tp: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr); sys.exit(2)
    sock.sendall(P.pack_request("generate_tp", {
        "prompt": args.prompt, "max_tokens": args.max_tokens,
        "chunk_size": args.max_tokens, "seed": 0,
    }))
    baseline = None
    while True:
        raw = P.read_line(sock, max_bytes=64 << 20)
        if not raw: break
        obj = _json.loads(raw.decode("utf-8"))
        t = obj.get("type", "")
        if t == "error":
            print(f"\nERROR (generate_tp baseline): {obj.get('msg')}", file=sys.stderr); sys.exit(4)
        if t == "result":
            baseline = obj.get("data", {}); break
    sock.close()
    if baseline is None:
        print("client_tp: no final response from generate_tp", file=sys.stderr); sys.exit(4)
    prompt_ids = baseline["prompt_ids"]
    generated_ids = baseline["generated_ids"]
    print(f"[baseline] {len(prompt_ids)} prompt + {len(generated_ids)} generated tokens "
          f"({baseline.get('ms_per_tok', 0):.2f} ms/tok)")

    timestamp = _t.strftime("%Y%m%d_%H%M")
    artifacts = {}
    for mode in modes:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(7200.0)
        sock.connect(SOCKET_PATH)
        out_path = _os.path.expanduser(
            f"~/tt-xla/.cache/qb2_tp_deltanet/_cosine_ladder_tp_{mode}_{timestamp}.npz")
        sock.sendall(P.pack_request("cosine_ladder_tp", {
            "prompt_ids": prompt_ids,
            "generated_ids": generated_ids,
            "deltanet_recurrence_mode": mode,
            "deltanet_conv1d_mode": args.deltanet_conv1d_mode,
            "deltanet_decay_gate_mode": args.deltanet_decay_gate_mode,
            "out_path": out_path,
        }))
        raw = P.read_line(sock, max_bytes=64 << 20)
        sock.close()
        obj = _json.loads(raw.decode("utf-8"))
        if obj.get("type") == "error":
            print(f"\nERROR (cosine_ladder_tp mode={mode}): {obj.get('msg')}", file=sys.stderr); sys.exit(4)
        data = obj.get("data", {})
        artifacts[mode] = data
        print(f"[{mode:<18}] {data.get('n_steps', 0)} steps, "
              f"prefill {data.get('prefill_ms', 0):.0f} ms, "
              f"decode {data.get('ms_per_step', 0):.1f} ms/step → {data.get('path')}")

    if len(modes) < 2:
        return
    base_mode = modes[0]
    base_logits = np.load(artifacts[base_mode]["path"])["logits"].astype(np.float32)
    M_steps, V = base_logits.shape
    argmax_base = np.argmax(base_logits, axis=1)

    comparison = {
        "prompt": args.prompt,
        "base_mode": base_mode,
        "n_prompt": int(len(prompt_ids)),
        "n_steps": int(M_steps),
        "vocab": int(V),
        "prompt_ids": list(prompt_ids),
        "generated_ids": list(generated_ids),
        "comparisons": {},
    }
    print(f"\n[compare] base={base_mode} ({M_steps} steps)")
    print(f"{'mode':<22} {'min_cos':>10} {'med_cos':>10} {'mean_cos':>10} {'top1_disagree':>15} {'first_disag':>12}")
    for mode in modes[1:]:
        other_logits = np.load(artifacts[mode]["path"])["logits"].astype(np.float32)
        dots = np.sum(base_logits * other_logits, axis=1)
        cosines = dots / (np.linalg.norm(base_logits, axis=1) *
                           np.linalg.norm(other_logits, axis=1) + 1e-30)
        argmax_other = np.argmax(other_logits, axis=1)
        disagree_mask = (argmax_base != argmax_other)
        disagree = int(disagree_mask.sum())
        first_disagree = int(np.argmax(disagree_mask)) if disagree else -1
        comparison["comparisons"][mode] = {
            "min_cos": float(cosines.min()),
            "med_cos": float(np.median(cosines)),
            "mean_cos": float(cosines.mean()),
            "max_cos": float(cosines.max()),
            "top1_disagree_count": disagree,
            "top1_disagree_rate": disagree / M_steps,
            "first_disagree_step": first_disagree,
            "cosines": cosines.tolist(),
        }
        print(f"{mode:<22} {cosines.min():>10.6f} {np.median(cosines):>10.6f} "
              f"{cosines.mean():>10.6f} {f'{disagree}/{M_steps}':>15} {first_disagree:>12}")

    out_json = _os.path.expanduser(
        f"~/tt-xla/.cache/qb2_tp_deltanet/cosine_ladder_tp_compare_{timestamp}.json")
    _os.makedirs(_os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        _json.dump(comparison, f, indent=2)
    print(f"\n[save] comparison → {out_json}")


def cmd_generate_tp(args):
    """Streaming generate_tp — same chunk/result protocol as client.cmd_generate."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(7200.0)
    try:
        sock.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client_tp: cannot connect to {SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    payload = {"prompt": args.prompt, "max_tokens": args.max_tokens,
               "chunk_size": args.chunk_size, "seed": args.seed}
    final = None
    try:
        sock.sendall(P.pack_request("generate_tp", payload))
        print("=" * 72)
        print(f"prompt: {args.prompt}")
        print("-" * 72)
        while True:
            raw = P.read_line(sock, max_bytes=64 << 20)
            if not raw:
                print("\nclient_tp: server closed connection before final", file=sys.stderr)
                break
            obj = json.loads(raw.decode("utf-8"))
            t = obj.get("type", "")
            if t == "error":
                print(f"\nERROR: {obj.get('msg')}", file=sys.stderr)
                sys.exit(4)
            if t == "chunk":
                print(obj.get("data", {}).get("token_text", ""), end="", flush=True)
            elif t == "result":
                final = obj.get("data", {})
                break
    finally:
        sock.close()
    print("\n" + "=" * 72)
    if final is None:
        return
    print(f"  prompt: {final.get('n_prompt_tokens', 0)} tokens, "
          f"prefill {final.get('prefill_ms', 0):.1f} ms")
    print(f"  generated: {final.get('n_generated_tokens', 0)} tokens, "
          f"decode {final.get('ms_per_tok', 0):.2f} ms/tok = "
          f"{final.get('tok_per_sec', 0):.2f} tok/s")
    print(f"  total wall: {final.get('total_ms', 0):.1f} ms")
    if final.get('stopped_on_eos'):
        print(f"  (stopped on EOS)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("shutdown").set_defaults(fn=cmd_shutdown)
    b = sub.add_parser("bench_decode_tp_components",
                       help="server-resident TP component timing")
    b.add_argument("--prompt", default="The capital of France is")
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--warmup", type=int, default=3)
    b.set_defaults(fn=cmd_bench_decode_tp_components)
    f = sub.add_parser("probe_fused_paged_update_cache_tp",
                       help="validate fused K/V paged-cache writer in resident TP server")
    f.add_argument("--prompt", default="The capital of France is")
    f.add_argument("--iters", type=int, default=10)
    f.add_argument("--warmup", type=int, default=2)
    f.add_argument("--skip-trace-bench", action="store_true")
    f.add_argument("--allow-wedge-prone-disjoint", action="store_true",
                   help="actually run the disjoint fused writer path that previously wedged qb2")
    f.set_defaults(fn=cmd_probe_fused_paged_update_cache_tp)
    ar = sub.add_parser("probe_explicit_all_reduce_tp",
                        help="validate explicit axis/topology all-reduce exits")
    ar.add_argument("--prompt", default="The capital of France is")
    ar.add_argument("--iters", type=int, default=10)
    ar.add_argument("--warmup", type=int, default=2)
    ar.set_defaults(fn=cmd_probe_explicit_all_reduce_tp)
    rope = sub.add_parser("probe_rope_fused_qk_tp",
                          help="validate fused Q/K RoPE semantics on production shapes")
    rope.add_argument("--positions", type=int, nargs="+",
                      default=[0, 1, 7, 31, 32, 127, 255])
    rope.add_argument("--allow-wedge-prone-fused-qk", action="store_true",
                      help="actually run fused QK RoPE path that previously wedged qb2")
    rope.set_defaults(fn=cmd_probe_rope_fused_qk_tp)
    rope_native = sub.add_parser("probe_rope_native_partial_tp",
                                 help="validate native slice-first partial RoPE semantics")
    rope_native.add_argument("--positions", type=int, nargs="+",
                             default=[0, 1, 7, 31, 32, 127, 255])
    rope_native.set_defaults(fn=cmd_probe_rope_native_partial_tp)
    rope_native_trace = sub.add_parser("probe_rope_native_partial_trace_tp",
                                       help="bench guarded native partial RoPE trace variant")
    rope_native_trace.add_argument("--prompt", default="The capital of France is")
    rope_native_trace.add_argument("--iters", type=int, default=10)
    rope_native_trace.add_argument("--warmup", type=int, default=2)
    rope_native_trace.set_defaults(fn=cmd_probe_rope_native_partial_trace_tp)
    prof = sub.add_parser("profile_decode_tp_ops",
                          help="resident-server op-count/timing profile of decode trace body")
    prof.add_argument("--prompt", default="The capital of France is")
    prof.add_argument("--timed", action="store_true",
                      help="sync-bound every profiled op; slower but includes category timing")
    prof.add_argument("--include-records", action="store_true")
    prof.add_argument("--deltanet-decay-mode", choices=["manual", "native_softplus"],
                      default="manual")
    prof.add_argument("--deltanet-recurrence-mode", choices=["manual", "owned_gdn", "owned_gdn_inplace"],
                      default="manual")
    prof.set_defaults(fn=cmd_profile_decode_tp_ops)
    dn = sub.add_parser("probe_deltanet_recurrence_matmul_tp",
                        help="validate matmul-form DeltaNet recurrence body")
    dn.add_argument("--iters", type=int, default=20)
    dn.add_argument("--warmup", type=int, default=3)
    dn.set_defaults(fn=cmd_probe_deltanet_recurrence_matmul_tp)
    dn_gdn_mesh = sub.add_parser("probe_deltanet_native_gdn_synthetic_mesh_tp",
                                 help="validate native Qwen36 GDN op on synthetic resident mesh tensors")
    dn_gdn_mesh.add_argument("--slots", type=int, default=12)
    dn_gdn_mesh.add_argument("--key-dim", type=int, default=128)
    dn_gdn_mesh.add_argument("--value-dim", type=int, default=128)
    dn_gdn_mesh.add_argument("--seed", type=int, default=20260515)
    dn_gdn_mesh.add_argument("--scale", type=float, default=0.03125)
    dn_gdn_mesh.add_argument("--distribution", choices=["replicated", "sharded_dim0"],
                             default="replicated")
    dn_gdn_mesh.add_argument("--iters", type=int, default=0)
    dn_gdn_mesh.add_argument("--warmup", type=int, default=0)
    dn_gdn_mesh.add_argument("--output-json")
    dn_gdn_mesh.set_defaults(fn=cmd_probe_deltanet_native_gdn_synthetic_mesh_tp)
    dn_gdn = sub.add_parser("probe_deltanet_native_gdn_real_tensors_tp",
                            help="validate native Qwen36 GDN op on real resident DeltaNet tensors")
    dn_gdn.add_argument("--prompt", default="The capital of France is")
    dn_gdn.add_argument("--layer-idx", type=int, default=0)
    dn_gdn.add_argument("--mode", choices=["fp32_cast", "current_dtype"], default="fp32_cast")
    dn_gdn.add_argument("--no-reset-state", action="store_true")
    dn_gdn.set_defaults(fn=cmd_probe_deltanet_native_gdn_real_tensors_tp)
    dn_gdn_owned = sub.add_parser("probe_deltanet_owned_gdn_real_tensors_tp",
                                  help="validate owned Qwen36 GDN op on real resident DeltaNet tensors")
    dn_gdn_owned.add_argument("--prompt", default="The capital of France is")
    dn_gdn_owned.add_argument("--layer-idx", type=int, default=0)
    dn_gdn_owned.add_argument("--use-pretransposed-k", action="store_true")
    dn_gdn_owned.add_argument("--compact-vectors", action="store_true")
    dn_gdn_owned.add_argument("--native-io", action="store_true")
    dn_gdn_owned.add_argument("--stepwise", action="store_true")
    dn_gdn_owned.add_argument("--seed-state", choices=["resident", "manual_once"],
                              default="resident")
    dn_gdn_owned.add_argument("--direct-state-input", action="store_true",
                              help="pass H_input directly to owned op; only safe with --seed-state manual_once")
    dn_gdn_owned.add_argument("--component-debug-mode", type=int, action="append",
                              help="optional qwen36_gdn_prediction debug mode to run inside stepwise probe; may repeat")
    dn_gdn_owned.add_argument("--no-reset-state", action="store_true")
    dn_gdn_owned.add_argument("--output-json")
    dn_gdn_owned.set_defaults(fn=cmd_probe_deltanet_owned_gdn_real_tensors_tp)
    dn_gdn_owned_trace = sub.add_parser("probe_deltanet_owned_gdn_trace_tp",
                                        help="validate and bench guarded owned GDN recurrence trace")
    dn_gdn_owned_trace.add_argument("--prompt", default="The capital of France is")
    dn_gdn_owned_trace.add_argument("--iters", type=int, default=10)
    dn_gdn_owned_trace.add_argument("--warmup", type=int, default=2)
    dn_gdn_owned_trace.add_argument("--max-tokens", type=int, default=20)
    dn_gdn_owned_trace.add_argument("--deltanet-recurrence-mode",
                                    choices=["owned_gdn", "owned_gdn_inplace"],
                                    default="owned_gdn")
    dn_gdn_owned_trace.add_argument("--deltanet-decay-mode",
                                    choices=["manual", "native_softplus"],
                                    default="manual")
    dn_gdn_owned_trace.add_argument("--output-json")
    dn_gdn_owned_trace.set_defaults(fn=cmd_probe_deltanet_owned_gdn_trace_tp)
    dn_gdn_owned_bench = sub.add_parser("probe_deltanet_owned_gdn_benchmark_tp",
                                        help="run guarded owned GDN recurrence trace benchmark over a prompt set")
    dn_gdn_owned_bench.add_argument("--prompt", action="append",
                                    help="prompt to include; may be repeated; server defaults are used if omitted")
    dn_gdn_owned_bench.add_argument("--iters", type=int, default=6)
    dn_gdn_owned_bench.add_argument("--warmup", type=int, default=2)
    dn_gdn_owned_bench.add_argument("--max-tokens", type=int, default=64)
    dn_gdn_owned_bench.add_argument("--deltanet-recurrence-mode",
                                    choices=["owned_gdn", "owned_gdn_inplace"],
                                    default="owned_gdn")
    dn_gdn_owned_bench.add_argument("--deltanet-decay-mode",
                                    choices=["manual", "native_softplus"],
                                    default="manual")
    dn_gdn_owned_bench.add_argument("--output-json")
    dn_gdn_owned_bench.set_defaults(fn=cmd_probe_deltanet_owned_gdn_benchmark_tp)
    dn_gdn_owned_div = sub.add_parser("probe_deltanet_owned_gdn_divergence_tp",
                                      help="diagnose owned GDN decode divergence with eager top-k logits")
    dn_gdn_owned_div.add_argument("--prompt", default="In Python, a simple function to add two numbers is")
    dn_gdn_owned_div.add_argument("--max-tokens", type=int, default=24)
    dn_gdn_owned_div.add_argument("--top-k", type=int, default=8)
    dn_gdn_owned_div.add_argument("--output-json")
    dn_gdn_owned_div.set_defaults(fn=cmd_probe_deltanet_owned_gdn_divergence_tp)
    dn_gdn_owned_tf = sub.add_parser("probe_deltanet_owned_gdn_teacher_forced_tp",
                                     help="compare manual vs owned GDN under the same teacher-forced token stream")
    dn_gdn_owned_tf.add_argument("--prompt", default="In Python, a simple function to add two numbers is")
    dn_gdn_owned_tf.add_argument("--max-tokens", type=int, default=24)
    dn_gdn_owned_tf.add_argument("--top-k", type=int, default=8)
    dn_gdn_owned_tf.add_argument("--state-layer", type=int, action="append",
                                 help="linear-attention layer index to compare; may be repeated")
    dn_gdn_owned_tf.add_argument("--output-json")
    dn_gdn_owned_tf.set_defaults(fn=cmd_probe_deltanet_owned_gdn_teacher_forced_tp)
    dn_sp = sub.add_parser("probe_deltanet_softplus_decay_tp",
                           help="validate native softplus in DeltaNet decay/gate")
    dn_sp.add_argument("--prompt", default="The capital of France is")
    dn_sp.add_argument("--iters", type=int, default=10)
    dn_sp.add_argument("--warmup", type=int, default=2)
    dn_sp.add_argument("--max-tokens", type=int, default=20)
    dn_sp.set_defaults(fn=cmd_probe_deltanet_softplus_decay_tp)
    sp_check = sub.add_parser("probe_deltanet_conv1d_split_check_tp",
                                help="mesh-aware split-vs-combined comparison for owned conv1d wire-in bug investigation")
    sp_check.add_argument("--layer-idx", type=int, default=0)
    sp_check.add_argument("--max-abs-diff", type=float, default=0.05)
    sp_check.set_defaults(fn=cmd_probe_deltanet_conv1d_split_check_tp)
    dg_g1 = sub.add_parser("probe_deltanet_owned_decay_gate_real_tensors_tp",
                            help="G1 real-tensor sweep for owned decay/gate kernel")
    dg_g1.add_argument("--layer-idx", type=int, default=None,
                        help="DeltaNet layer index (None = sweep all)")
    dg_g1.add_argument("--seed", type=int, default=0)
    dg_g1.add_argument("--max-abs-diff", type=float, default=0.01)
    dg_g1.set_defaults(fn=cmd_probe_deltanet_owned_decay_gate_real_tensors_tp)
    g = sub.add_parser("generate_tp", help="multi-chip generate (streams)")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=60)
    g.add_argument("--chunk-size", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate_tp)
    cl = sub.add_parser("cosine_ladder_tp",
                          help="teacher-forced cosine ladder; compare deltanet recurrence modes")
    cl.add_argument("--prompt", required=True)
    cl.add_argument("--max-tokens", type=int, default=200,
                     help="positions to teacher-force (P + M ≤ MAX_POS=256)")
    cl.add_argument("--modes", default="manual,owned_gdn",
                     help="comma-separated recurrence modes (e.g. manual,owned_gdn)")
    cl.add_argument("--deltanet-conv1d-mode", choices=["manual", "owned_conv1d"],
                     default="manual",
                     help="DeltaNet conv1d kernel toggle. owned_conv1d routes through "
                          "ttnn.experimental.qwen36_conv1d_decode_owned for the G3 long-"
                          "context correctness gate (slower than manual per-step due to "
                          "per-step weight/state slicing; G4 will pre-split at bootstrap).")
    cl.add_argument("--deltanet-decay-gate-mode", choices=["manual", "owned_decay_gate"],
                     default="manual",
                     help="DeltaNet decay/gate kernel toggle. owned_decay_gate routes "
                          "through ttnn.experimental.qwen36_decay_gate_decode_owned (G2 "
                          "wire-in: reshapes dt_bias/A_log per-step; G4 will pre-allocate "
                          "rank-2 versions at bootstrap).")
    cl.set_defaults(fn=cmd_cosine_ladder_tp)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
