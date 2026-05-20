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
from pathlib import Path as _Path

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


def cmd_probe_ccl_components_tp(args):
    data = _send("probe_ccl_components_tp", {
        "iters": args.iters,
        "warmup": args.warmup,
        "shape": [1, args.hidden],
    }, timeout=600.0)
    # Pretty headline first; full JSON saved to artifact if requested.
    variants = data.get("variants", {})
    composites = data.get("composites", {})
    print(f"shape={data.get('shape')} iters={data.get('iters')} warmup={data.get('warmup')}")
    print("\nper-variant median (min, p99) ms:")
    for name, v in variants.items():
        s = v.get("summary_ms", {})
        samples = v.get("samples_ms", [])
        p99 = sorted(samples)[int(len(samples) * 0.99) - 1] if samples else float("nan")
        print(f"  {name:<32} {s.get('median', float('nan')):.4f}  "
              f"(min {s.get('min', float('nan')):.4f}, p99 {p99:.4f})")
    print("\ncomposite vs single all_reduce:")
    for name, c in composites.items():
        delta = c.get("composite_minus_all_reduce_ms", float("nan"))
        sign = "+" if delta >= 0 else ""
        print(f"  {name}: RS {c.get('rs_median_ms', float('nan')):.4f} + AG "
              f"{c.get('ag_median_ms', float('nan')):.4f} = "
              f"{c.get('sum_median_ms', float('nan')):.4f} ms  "
              f"vs AR {c.get('vs_all_reduce_median_ms', float('nan')):.4f} ms  "
              f"(delta {sign}{delta:.4f} ms)")
    errors = data.get("errors", {})
    if errors:
        print("\nerrors:")
        for name, msg in errors.items():
            print(f"  {name}: {msg}")
    if args.out:
        _Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _Path(args.out).write_text(json.dumps(data, indent=2, default=str))
        print(f"\n[save] {args.out}")


def cmd_probe_slice_write_round_trip(args):
    payload = {}
    if args.seq_len:
        payload["seq_len"] = args.seq_len
    if args.hidden:
        payload["hidden"] = args.hidden
    data = _send("probe_slice_write_round_trip", payload, timeout=120.0)
    if data.get("error"):
        print(f"ERROR: {data['error']}", file=sys.stderr)
        if data.get("verdict"):
            print(f"VERDICT: {data['verdict']}", file=sys.stderr)
        sys.exit(4)
    print(f"seq_len={data['seq_len']} hidden={data['hidden']}")
    print(f"\nslice_write per-pos:")
    for r in data["write_results"]:
        status = "OK" if r["ok"] else f"FAIL ({r.get('error', '?')})"
        print(f"  pos {r['pos']}: wrote {r['wrote_value']}  → {status}")
    print(f"\nROW_MAJOR readback:")
    for r in data["rowmajor_readback"]:
        status = "OK" if r.get("ok") else f"FAIL"
        print(f"  pos {r['pos']}: expected={r['expected']:.1f}  mean={r.get('mean', float('nan')):.4f}  "
              f"min={r.get('min', float('nan')):.4f}  max={r.get('max', float('nan')):.4f}  {status}")
    print(f"\nTILE_LAYOUT readback (after to_layout convert):")
    if data.get("tile_convert_error"):
        print(f"  to_layout failed: {data['tile_convert_error']}")
    else:
        for r in data["tile_readback"]:
            status = "OK" if r.get("ok") else f"FAIL"
            print(f"  pos {r['pos']}: expected={r['expected']:.1f}  mean={r.get('mean', float('nan')):.4f}  "
                  f"min={r.get('min', float('nan')):.4f}  max={r.get('max', float('nan')):.4f}  {status}")
    print(f"\nrowmajor_pass={data['rowmajor_pass']}  tile_pass={data['tile_pass']}")
    print(f"\n{data['verdict']}")


def cmd_probe_multirow_construct_vs_per_position(args):
    payload = {}
    if args.prompt:
        payload["prompt"] = args.prompt
    data = _send("probe_multirow_construct_vs_per_position", payload, timeout=300.0)
    if data.get("error"):
        print(f"ERROR: {data['error']}", file=sys.stderr)
        if data.get("verdict"):
            print(f"VERDICT: {data['verdict']}", file=sys.stderr)
        sys.exit(4)
    print(f"seq_len={data['seq_len']}  raw_embed_shape={data.get('batched_embed_raw_shape')}  "
          f"reshaped_to={data.get('batched_reshaped_to')}")
    print(f"\nper-row cosine (batched_construct vs per_position_embed):")
    for r in data["per_row"]:
        print(f"  pos {r['pos']}: cos={r['cos']:.6f}  max_abs_diff={r['max_abs_diff']:.6e}")
    print(f"\nmin_cos={data['min_cos']:.6f}  max_cos={data['max_cos']:.6f}  "
          f"max_abs_diff={data['max_abs_diff']:.6e}")
    print(f"\n{data['verdict']}")


def cmd_probe_prefill_vs_decode_loop_tp(args):
    payload = {"mode": args.mode}
    if args.prompt:
        payload["prompt"] = args.prompt
    data = _send("probe_prefill_vs_decode_loop_tp", payload, timeout=600.0)
    if data.get("error"):
        print(f"ERROR: {data['error']}", file=sys.stderr)
        sys.exit(4)
    pc = data["per_position_cosine"]
    print(f"mode={data['mode']}  seq_len={data['seq_len']}  vocab={data['vocab']}")
    print(f"reference (decode-loop) wall: {data['reference_ms']:.0f} ms")
    print(f"test ({data['mode']}) wall: {data['test_ms']:.0f} ms")
    print(f"\nper-position cosine: min={pc['min']:.6f}  median={pc['median']:.6f}  "
          f"mean={pc['mean']:.6f}  max={pc['max']:.6f}")
    print(f"max_abs_diff: {data['max_abs_diff']:.6e}")
    print(f"top1 agreement: {data['top1_agreement']}")
    verdict = "PASS" if data["pass_gate_0p999"] else "FAIL"
    print(f"\nGate (per-pos cos >= 0.999): {verdict}")


def cmd_probe_async_ccl_components_tp(args):
    data = _send("probe_async_ccl_components_tp", {
        "iters": args.iters,
        "warmup": args.warmup,
        "hidden": args.hidden,
        "matmul_k": args.matmul_k,
        "matmul_n": args.matmul_n,
    }, timeout=900.0)
    variants = data.get("variants", {})
    composites = data.get("composites", {})
    print(f"shape={data.get('shape')} matmul={data.get('matmul_shape')} "
          f"iters={data.get('iters')} warmup={data.get('warmup')}")
    print("\nper-variant median (min, p99) ms:")
    for name, v in variants.items():
        s = v.get("summary_ms", {})
        samples = v.get("samples_ms", [])
        p99 = sorted(samples)[int(len(samples) * 0.99) - 1] if samples else float("nan")
        print(f"  {name:<34} {s.get('median', float('nan')):.4f}  "
              f"(min {s.get('min', float('nan')):.4f}, p99 {p99:.4f})")
    print("\ncomposites:")
    for k, v in composites.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:<28} {sign}{v:.4f} ms")
    errors = data.get("errors", {})
    if errors:
        print("\nerrors:")
        for name, msg in errors.items():
            print(f"  {name}: {msg}")
    if args.out:
        _Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _Path(args.out).write_text(json.dumps(data, indent=2, default=str))
        print(f"\n[save] {args.out}")


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
            "rope_mode": args.rope_mode,
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
    swr = sub.add_parser("probe_slice_write_round_trip",
                          help="B.2.1.5b: probe pre-alloc + slice_write per-row write + readback. "
                               "Tests ROW_MAJOR write and TILE_LAYOUT conversion. Decides whether "
                               "B.2.2 can use this pattern.")
    swr.add_argument("--seq-len", type=int, default=5)
    swr.add_argument("--hidden", type=int, default=None,
                      help="default: state.cfg['hidden'] = 5120")
    swr.set_defaults(fn=cmd_probe_slice_write_round_trip)
    mrc = sub.add_parser("probe_multirow_construct_vs_per_position",
                          help="B.2.1.5: validate direct batched ttnn.embedding "
                               "→ [seq_len, HIDDEN] tensor supports correct row "
                               "slicing. If pass, B.2.2 can avoid slice/concat "
                               "plumbing entirely.")
    mrc.add_argument("--prompt", default=None)
    mrc.set_defaults(fn=cmd_probe_multirow_construct_vs_per_position)
    pvd = sub.add_parser("probe_prefill_vs_decode_loop_tp",
                          help="B.1 prefill validation harness: compare per-position logits "
                               "from sequential decode-loop reference vs forward_prefill_tp_inner. "
                               "Gate: cos >= 0.999 per position. Initial B.1 stub is decode-loop "
                               "wrapped → cos = 1.0 trivially (validates harness itself).")
    pvd.add_argument("--prompt", default=None,
                      help="prompt to tokenize and validate (default: 'The capital of France is')")
    pvd.add_argument("--mode", default="stub",
                      choices=["stub", "batched_mlp", "sequential_via_slices",
                               "per_position_list"],
                      help="prefill implementation under test: 'stub' (B.1 decode-loop "
                           "wrapper; cos=1.0 expected) or 'batched_mlp' (B.2.1 layer-outer "
                           "iteration with batched MLP per layer; gate cos>=0.999)")
    pvd.set_defaults(fn=cmd_probe_prefill_vs_decode_loop_tp)
    ccl = sub.add_parser("probe_ccl_components_tp",
                          help="micro-bench CCL primitives (all_reduce, reduce_scatter, all_gather) "
                               "at production [1, HIDDEN] bf16 shape on (1,4) mesh; answers num_links "
                               "free-bandwidth probe (P1) and composite-vs-fused probe (P2)")
    ccl.add_argument("--iters", type=int, default=30)
    ccl.add_argument("--warmup", type=int, default=5)
    ccl.add_argument("--hidden", type=int, default=5120,
                      help="HIDDEN dim of the bench tensor [1, HIDDEN]")
    ccl.add_argument("--out", default=None,
                      help="path to save full JSON artifact (default: print headline only)")
    ccl.set_defaults(fn=cmd_probe_ccl_components_tp)
    accl = sub.add_parser("probe_async_ccl_components_tp",
                            help="G0 async-CCL component bench: sync_baseline vs "
                                 "async_immediate_sync vs async_double vs async_with_matmul "
                                 "at production [1,HIDDEN] bf16 shape; gates async-CCL G1 "
                                 "single-layer overlap prototype")
    accl.add_argument("--iters", type=int, default=30)
    accl.add_argument("--warmup", type=int, default=5)
    accl.add_argument("--hidden", type=int, default=5120,
                       help="HIDDEN dim of the all_reduce bench tensor")
    accl.add_argument("--matmul-k", type=int, default=5120,
                       help="K dim of the overlap-test matmul (input dim)")
    accl.add_argument("--matmul-n", type=int, default=32768,
                       help="N dim of the overlap-test matmul (sharded output dim)")
    accl.add_argument("--out", default=None,
                       help="path to save full JSON artifact")
    accl.set_defaults(fn=cmd_probe_async_ccl_components_tp)
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
    cl.add_argument("--rope-mode", choices=["manual", "native_partial"],
                     default="manual",
                     help="Gated-attention RoPE path toggle. native_partial routes through "
                          "slice-first ttnn.experimental.rotary_embedding on rotary_dim=64 "
                          "with passthrough concat (G3 long-context correctness gate). "
                          "Prior isolation + guarded-trace probes PASSED May 14-15.")
    cl.set_defaults(fn=cmd_cosine_ladder_tp)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
