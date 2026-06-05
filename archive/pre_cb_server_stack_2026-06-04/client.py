"""CLI client for the persistent weight server.

Usage:
    python -m experiments.serve.client status
    python -m experiments.serve.client reset_state
    python -m experiments.serve.client reload_kernels
    python -m experiments.serve.client run_91r --layers 0,3,7
    python -m experiments.serve.client shutdown
"""
import argparse
import json
import socket
import sys

from experiments.serve import protocol as P


def send(cmd: str, args: dict, timeout: float = 7200.0) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(P.SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client: cannot connect to {P.SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        sock.sendall(P.pack_request(cmd, args))
        raw = P.read_line(sock, max_bytes=64 << 20)
    finally:
        sock.close()
    if not raw:
        print("client: server returned no data (process likely died)", file=sys.stderr)
        sys.exit(3)
    resp = P.parse_response(raw)
    if resp.type == "error":
        print(f"server error: {resp.msg}", file=sys.stderr)
        sys.exit(4)
    return resp.data or {}


def cmd_status(_):
    data = send("status", {})
    print(json.dumps(data, indent=2, default=str))


def cmd_reset(_):
    print(json.dumps(send("reset_state", {}), indent=2))


def cmd_reload(_):
    print(json.dumps(send("reload_kernels", {}), indent=2))


def cmd_shutdown(_):
    print(json.dumps(send("shutdown", {}), indent=2))


def cmd_bench_decode(args):
    payload = {"n_steps": args.n_steps, "warmup": args.warmup}
    if args.prompt:
        payload["prompt"] = args.prompt
    data = send("bench_decode", payload)
    print("=" * 72)
    print(f"bench_decode prompt='{data.get('prompt')}' n_steps={data.get('n_steps')} "
          f"warmup={data.get('warmup')}")
    print(f"  median: {data.get('median_ms', 0):.2f} ms/tok  ({data.get('tok_per_sec', 0):.2f} tok/s)")
    print(f"  mean:   {data.get('mean_ms', 0):.2f} ms/tok")
    print(f"  p95:    {data.get('p95_ms', 0):.2f} ms/tok")
    print(f"  min:    {data.get('min_ms', 0):.2f} ms/tok")
    print(f"  max:    {data.get('max_ms', 0):.2f} ms/tok")
    print("=" * 72)


def cmd_bench_decode_paged(args):
    payload = {"n_steps": args.n_steps, "warmup": args.warmup,
               "max_pos": args.max_pos, "block_size": args.block_size}
    data = send("bench_decode_paged", payload)
    print("=" * 72)
    print(f"bench_decode_paged  max_pos={data.get('max_pos')}  "
          f"block_size={data.get('block_size')}  n_steps={data.get('n_steps')}")
    print(f"  median: {data.get('median_ms', 0):.2f} ms/tok  "
          f"({data.get('tok_per_sec', 0):.2f} tok/s)")
    print(f"  p95:    {data.get('p95_ms', 0):.2f} ms/tok")
    print(f"  min:    {data.get('min_ms', 0):.2f} ms/tok")
    print(f"  max:    {data.get('max_ms', 0):.2f} ms/tok")
    print("=" * 72)


def cmd_bench_decode_traced(args):
    payload = {
        "n_steps": args.n_steps,
        "warmup": args.warmup,
        "validate_steps": args.validate_steps,
        "start_token_id": args.start_token_id,
        "recapture": args.recapture,
    }
    data = send("bench_decode_traced", payload)
    print("=" * 72)
    print(f"bench_decode_traced  n_steps={data.get('n_steps')}  warmup={data.get('warmup')}  "
          f"validate_steps={data.get('validate_steps')}")
    if data.get("capture_sec", 0) > 0:
        print(f"  trace captured in {data['capture_sec']:.1f}s")
    cos = data.get("cosines") or []
    if cos:
        print(f"  validation cosines: {[f'{c:.6f}' for c in cos]}")
        print(f"  min cosine:         {data.get('min_cosine'):.6f}  "
              f"(top1 match all: {data.get('all_top1_match')})")
    print(f"  median:             {data.get('median_ms', 0):.2f} ms/tok  "
          f"({data.get('tok_per_sec', 0):.2f} tok/s)")
    print(f"  median (exec only): {data.get('median_exec_ms', 0):.2f} ms/tok")
    print(f"  p95:                {data.get('p95_ms', 0):.2f} ms/tok")
    print("=" * 72)


def cmd_run_91r(args):
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    payload = {}
    if layers is not None:
        payload["layers"] = layers
    if args.weight_dtype:
        payload["weight_dtype"] = args.weight_dtype
    data = send("run_91r", payload)
    # Pretty summary first, then full JSON.
    print("=" * 72)
    print(f"run_91r layers={data.get('layers')} total_sec={data.get('total_sec', 0):.1f}")
    print("-" * 72)
    print(f"{'layer':>6s} {'type':>20s}  cosines (per pos)  -> worst")
    for r in data.get("results", []):
        cs = r["cosines"]
        worst = min(cs) if cs else float("nan")
        cs_str = " ".join(f"{c:.5f}" for c in cs)
        print(f"{r['layer']:6d} {r['type']:>20s}  {cs_str}  -> {worst:.5f}")
    print("=" * 72)
    print(json.dumps(data, indent=2))


def _stream_generate(server_cmd: str, payload: dict, *, prompt: str) -> None:
    """Streaming RPC client for generate / generate_long. Prints tokens live."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(7200.0)
    try:
        sock.connect(P.SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        print(f"client: cannot connect to {P.SOCKET_PATH}: {e}", file=sys.stderr)
        sys.exit(2)
    final = None
    try:
        sock.sendall(P.pack_request(server_cmd, payload))
        print("=" * 72)
        print(f"prompt: {prompt}")
        print("-" * 72)
        while True:
            raw = P.read_line(sock, max_bytes=64 << 20)
            if not raw:
                print("\nclient: server closed connection before final", file=sys.stderr)
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
            else:
                print(f"\nclient: unknown response type: {t}", file=sys.stderr)
                break
    finally:
        sock.close()
    print("\n" + "=" * 72)
    if final is None:
        return
    if "error" in final:
        print(f"ERROR: {final['error']}")
        return
    print(f"  prompt: {final.get('n_prompt_tokens', 0)} tokens, "
          f"prefill {final.get('prefill_ms', 0):.1f} ms")
    print(f"  generated: {final.get('n_generated_tokens', 0)} tokens, "
          f"decode {final.get('ms_per_tok', 0):.2f} ms/tok = "
          f"{final.get('tok_per_sec', 0):.2f} tok/s")
    if "max_pos" in final:
        print(f"  paged: max_pos={final['max_pos']}, block_size={final.get('block_size')}")
    print(f"  total wall: {final.get('total_ms', 0):.1f} ms")
    if final.get('stopped_on_eos'):
        print("  (stopped on EOS)")


def cmd_generate(args):
    _stream_generate("generate",
                      {"prompt": args.prompt, "max_tokens": args.max_tokens,
                       "chunk_size": args.chunk_size,
                       "temperature": args.temperature,
                       "top_p": args.top_p,
                       "min_p": args.min_p,
                       "repetition_penalty": args.repetition_penalty,
                       "no_repeat_ngram_size": args.no_repeat_ngram_size,
                       "dry_multiplier": args.dry_multiplier,
                       "dry_base": args.dry_base,
                       "dry_allowed_length": args.dry_allowed_length,
                       "seed": args.seed,
                       "chat": args.chat,
                       "system": args.system},
                      prompt=args.prompt)


def cmd_generate_long(args):
    _stream_generate("generate_long",
                      {"prompt": args.prompt, "max_tokens": args.max_tokens,
                       "max_pos": args.max_pos, "block_size": args.block_size,
                       "chunk_size": args.chunk_size,
                       "temperature": args.temperature,
                       "top_p": args.top_p,
                       "min_p": args.min_p,
                       "repetition_penalty": args.repetition_penalty,
                       "no_repeat_ngram_size": args.no_repeat_ngram_size,
                       "dry_multiplier": args.dry_multiplier,
                       "dry_base": args.dry_base,
                       "dry_allowed_length": args.dry_allowed_length,
                       "seed": args.seed,
                       "chat": args.chat,
                       "system": args.system},
                      prompt=args.prompt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("reset_state").set_defaults(fn=cmd_reset)
    sub.add_parser("reload_kernels").set_defaults(fn=cmd_reload)
    sub.add_parser("shutdown").set_defaults(fn=cmd_shutdown)
    g = sub.add_parser("generate", help="generate text (streams; ≤256 tokens)")
    g.add_argument("--prompt", type=str, required=True)
    g.add_argument("--max-tokens", type=int, default=40)
    g.add_argument("--chunk-size", type=int, default=1, help="tokens per stream chunk")
    g.add_argument("--temperature", type=float, default=0.0,
                    help="0=greedy (default). >0 enables sampling — recommended for long output")
    g.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus cutoff (default 1.0 = no cutoff)")
    g.add_argument("--min-p", type=float, default=0.0,
                    help="min-p cutoff: drop tokens with prob < min_p * max_prob "
                         "(arXiv:2407.01082; default 0.0 = no cutoff; try 0.05-0.1)")
    g.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="CTRL repetition penalty on generated ids "
                         "(arXiv:1909.05858; default 1.0 = no-op; try 1.1-1.3 to break loops; "
                         "active even at temperature=0)")
    g.add_argument("--no-repeat-ngram-size", type=int, default=0,
                    help="ban tokens that would form a repeated n-gram against the "
                         "full history (HF generate flag; default 0 = off; try 3-4 to kill "
                         "exact loops like the whitespace drift on generate_long; "
                         "active at temperature=0)")
    g.add_argument("--dry-multiplier", type=float, default=0.0,
                    help="DRY sampler multiplier (llama.cpp PR #6839; default 0.0 = off; "
                         "try 0.5-1.5 — exponential penalty for any token extending a recent "
                         "match beyond --dry-allowed-length; active at temperature=0)")
    g.add_argument("--dry-base", type=float, default=1.75,
                    help="DRY base (default 1.75 per llama.cpp); penalty = mult * base^(L-allowed)")
    g.add_argument("--dry-allowed-length", type=int, default=2,
                    help="DRY allowed repeat length before penalty kicks in (default 2)")
    g.add_argument("--seed", type=int, default=0,
                    help="RNG seed for deterministic sampling")
    g.add_argument("--chat", action="store_true",
                    help="wrap prompt in Qwen3 chat template "
                         "(<|im_start|>user...<|im_end|><|im_start|>assistant\\n). "
                         "Qwen3.6 is an instruct/thinking model; without this flag we "
                         "send raw text and the model continues the doc rather than "
                         "responding to a turn. Default off for back-compat.")
    g.add_argument("--system", type=str, default="",
                    help="optional system prompt (only used with --chat)")
    g.set_defaults(fn=cmd_generate)
    gl = sub.add_parser("generate_long",
                          help="generate text (streams; long context via paged KV)")
    gl.add_argument("--prompt", type=str, required=True)
    gl.add_argument("--max-tokens", type=int, default=40)
    gl.add_argument("--max-pos", type=int, default=1024,
                     help="KV cache size (1024 default; use 4096+ for long context)")
    gl.add_argument("--block-size", type=int, default=64)
    gl.add_argument("--chunk-size", type=int, default=1, help="tokens per stream chunk")
    gl.add_argument("--temperature", type=float, default=0.0,
                     help="0=greedy (default). Paged greedy degenerates past ~130 tok; "
                          "try --temperature 0.7 --top-p 0.9 for ~150 tok before drift, or "
                          "--repetition-penalty 1.15 (works at temperature=0).")
    gl.add_argument("--top-p", type=float, default=1.0,
                     help="nucleus cutoff for temperature>0 (default 1.0 = no cutoff)")
    gl.add_argument("--min-p", type=float, default=0.0,
                     help="min-p cutoff: drop tokens with prob < min_p * max_prob "
                          "(arXiv:2407.01082; default 0.0 = no cutoff; try 0.05-0.1)")
    gl.add_argument("--repetition-penalty", type=float, default=1.0,
                     help="CTRL repetition penalty on generated ids "
                          "(arXiv:1909.05858; default 1.0 = no-op; try 1.1-1.3 to break the "
                          "whitespace/newline loop the paged path falls into past step ~130; "
                          "active even at temperature=0)")
    gl.add_argument("--no-repeat-ngram-size", type=int, default=0,
                     help="ban tokens that would form a repeated n-gram against the "
                          "full history (HF generate flag; default 0 = off; try 3-4 to kill "
                          "the exact whitespace/newline loops that show up past step ~130 "
                          "on the paged path; active at temperature=0)")
    gl.add_argument("--dry-multiplier", type=float, default=0.0,
                     help="DRY sampler multiplier (llama.cpp PR #6839; default 0.0 = off; "
                          "try 0.5-1.5 — exponential penalty for any token extending a recent "
                          "match beyond --dry-allowed-length; active at temperature=0. "
                          "BEST drift-resistant combo measured on the Rust-parser prompt: "
                          "--dry-multiplier 0.8 --repetition-penalty 1.1 (greedy, "
                          "+57% coherent chars vs temp+top-p)")
    gl.add_argument("--dry-base", type=float, default=1.75,
                     help="DRY base (default 1.75 per llama.cpp); penalty = mult * base^(L-allowed)")
    gl.add_argument("--dry-allowed-length", type=int, default=2,
                     help="DRY allowed repeat length before penalty kicks in (default 2)")
    gl.add_argument("--seed", type=int, default=0,
                     help="RNG seed for deterministic sampling")
    gl.add_argument("--chat", action="store_true",
                     help="wrap prompt in Qwen3 chat template "
                          "(<|im_start|>user...<|im_end|><|im_start|>assistant\\n). "
                          "Qwen3.6 is an instruct/thinking model; without this flag "
                          "we send raw text and the model continues the doc rather "
                          "than responding to a turn. Default off for back-compat.")
    gl.add_argument("--system", type=str, default="",
                     help="optional system prompt (only used with --chat)")
    gl.set_defaults(fn=cmd_generate_long)
    r = sub.add_parser("run_91r")
    r.add_argument("--layers", type=str, default=None,
                   help="comma-separated layer indices (default: server default)")
    r.add_argument("--weight-dtype", type=str, default=None,
                   choices=["bf8", "bf16", "fp32"])
    r.set_defaults(fn=cmd_run_91r)
    b = sub.add_parser("bench_decode")
    b.add_argument("--prompt", type=str, default=None)
    b.add_argument("--n-steps", type=int, default=20)
    b.add_argument("--warmup", type=int, default=3)
    b.set_defaults(fn=cmd_bench_decode)
    bp = sub.add_parser("bench_decode_paged",
                          help="bench eager decode with paged KV cache + paged SDPA (unlocks long context)")
    bp.add_argument("--n-steps", type=int, default=20)
    bp.add_argument("--warmup", type=int, default=3)
    bp.add_argument("--max-pos", type=int, default=256)
    bp.add_argument("--block-size", type=int, default=64)
    bp.set_defaults(fn=cmd_bench_decode_paged)
    bt = sub.add_parser("bench_decode_traced",
                         help="capture trace + multi-step validate vs eager + perf bench")
    bt.add_argument("--n-steps", type=int, default=20)
    bt.add_argument("--warmup", type=int, default=5)
    bt.add_argument("--validate-steps", type=int, default=5,
                     help="per-step cosine compare vs eager (0 to skip)")
    bt.add_argument("--start-token-id", type=int, default=760)
    bt.add_argument("--recapture", action="store_true",
                     help="release old trace and capture fresh (after reload_kernels)")
    bt.set_defaults(fn=cmd_bench_decode_traced)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
