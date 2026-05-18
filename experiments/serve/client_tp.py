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
    g = sub.add_parser("generate_tp", help="multi-chip generate (streams)")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=60)
    g.add_argument("--chunk-size", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=cmd_generate_tp)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
