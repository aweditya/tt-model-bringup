"""Vanilla tt-nn vs PJRT-traced comparison.

This is the "is PJRT worth it?" benchmark. Six programs, three
implementations each:

  A. Vanilla tt-nn eager  — hand-written, no trace
  B. Vanilla tt-nn traced  — hand-written, begin/end_trace_capture
  C. PJRT traced  — engine.execute_stablehlo with trace cache warm

Each implementation runs the same program with the same input shapes,
matched bf16/TILE_LAYOUT/device-0 environment.

Run on qb1:
    cd ~/tt-xla
    TT_PJRT_USE_DEVICE=1 .venv/bin/python pjrt_plugin/tests/bench_vanilla_vs_pjrt.py

Appends a table to research/pjrt_phase5_benchmarks.md.

Method:
- 5 warmup iters, 100 measurement iters
- time.perf_counter_ns(), median + p90 (us)
- Input numpy → device upload AND output device → numpy download are
  inside the timed loop (matches PJRT execute_stablehlo's signature)
- Numerical equivalence check before timing: cosine > 0.99 across A/B/C
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PJRT_DIR = REPO_ROOT / "pjrt_plugin"
sys.path.insert(0, str(PJRT_DIR))

os.environ.setdefault("TT_PJRT_USE_DEVICE", "1")

from jax_plugins.tt import engine

if not engine._USE_DEVICE:
    raise SystemExit(
        "TT_PJRT_USE_DEVICE=1 required; engine.py says ttnn isn't loaded.")

import ttnn
import torch
import jax
import jax.numpy as jnp
from jaxlib.mlir._mlir_libs._stablehlo import (
    serialize_portable_artifact, get_current_version,
)


WARMUP = 5
ITERS = 100

# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _sync():
    ttnn.synchronize_device(engine._get_device())


def time_loop(fn, warmup=WARMUP, iters=ITERS):
    """Run fn() warmup+iters times. Return median and p90 in microseconds.

    fn must be a complete unit of work that includes its own
    synchronization (i.e. it returns a numpy array or otherwise drains
    the queue). We don't add a sync between iters — each fn is
    self-contained.
    """
    for _ in range(warmup):
        fn()
    _sync()

    samples_ns = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        samples_ns.append(time.perf_counter_ns() - t0)

    arr_us = np.array(samples_ns) / 1000.0
    return float(np.median(arr_us)), float(np.percentile(arr_us, 90))


# ---------------------------------------------------------------------------
# Helpers: vanilla manual trace harness
# ---------------------------------------------------------------------------


def _np_to_torch_match_placeholder(arr, placeholder):
    """Build a host torch tensor whose logical shape matches the placeholder.

    Mirrors the engine's _replay_trace path: from_numpy → unsqueeze to 2D →
    right-pad with zeros to match placeholder.shape (logical shape).
    """
    if isinstance(arr, (int, float, np.integer, np.floating)):
        arr = np.array([[float(arr)]], dtype=np.float32)
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)
    t = torch.from_numpy(arr.copy()).float()
    while t.dim() < 2:
        t = t.unsqueeze(0)
    ph_shape = tuple(placeholder.shape)
    if tuple(t.shape) != ph_shape:
        pads = []
        for src, dst in zip(t.shape[::-1], ph_shape[::-1]):
            pads.extend([0, max(0, dst - src)])
        if any(p > 0 for p in pads):
            t = torch.nn.functional.pad(t, pads)
    return t


def _read_output(out_tensor, shape):
    """Bring a device tensor to a numpy array with the logical shape."""
    return engine._from_device(out_tensor, shape)


class VanillaTrace:
    """Hand-written tt-nn trace: begin_trace_capture, run user lambda, end.

    Mirrors the pattern in experiments/tt_jax/trace.py without any of the
    Jaxpr-driven infrastructure. The user provides:

        - sample_inputs: list of numpy arrays
        - build_fn(inputs_dev) -> output_tensor (ttnn ops, no host transfers)
        - output_shape: logical shape of the output

    Use:
        vt = VanillaTrace()
        vt.compile(sample_inputs, build_fn, output_shape)
        out = vt.run(new_inputs)
        ...
        vt.release()
    """

    def __init__(self):
        self.trace_id = None
        self.input_placeholders = []
        self.output_tensor = None
        self.output_shape = None

    def compile(self, sample_inputs, build_fn, output_shape):
        device = engine._get_device()
        self.output_shape = output_shape

        # Stage 1: regular eager run to allocate placeholder tensors and
        # populate ttnn's program cache. Trace capture forbids any
        # cache-miss kernel compilation.
        self.input_placeholders = [engine._to_device(a) for a in sample_inputs]
        _ = build_fn(self.input_placeholders)
        _sync()

        # Second warm-up to make sure compile cache is hot before trace
        _ = build_fn(self.input_placeholders)
        _sync()

        # Stage 2: trace capture
        self.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self.output_tensor = build_fn(self.input_placeholders)
        ttnn.end_trace_capture(device, self.trace_id, cq_id=0)

    def run(self, inputs):
        device = engine._get_device()
        # Copy inputs into placeholders (matches PJRT replay path)
        for i, arr in enumerate(inputs):
            ph = self.input_placeholders[i]
            t = _np_to_torch_match_placeholder(arr, ph)
            new_t = ttnn.from_torch(t, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(new_t, ph)

        ttnn.execute_trace(device, self.trace_id, cq_id=0, blocking=True)
        return _read_output(self.output_tensor, self.output_shape)

    def release(self):
        if self.trace_id is not None:
            try:
                ttnn.release_trace(engine._get_device(), self.trace_id)
            except Exception:
                pass
            self.trace_id = None


# ---------------------------------------------------------------------------
# Program definitions
# ---------------------------------------------------------------------------
# Each entry: (id, label, jax_fn, vanilla_build_fn, sample_inputs_factory,
#              output_shape).
#
# sample_inputs_factory returns a list of numpy arrays — deterministic
# via np.random.seed so all paths use identical data.


def _seed_inputs(seed):
    rng = np.random.RandomState(seed)
    return rng


def make_programs():
    """Build the six programs with deterministic inputs."""

    progs = []

    # ---- P1: x + 1, (1, 32) ----
    def p1_inputs():
        rng = _seed_inputs(101)
        return [rng.randn(1, 32).astype(np.float32)]

    def p1_jax(x):
        return x + 1.0

    def p1_vanilla(inputs):
        return ttnn.add(inputs[0], 1.0)

    progs.append(("P1", "x + 1 (1x32)", p1_jax, p1_vanilla, p1_inputs, (1, 32)))

    # ---- P2: exp(x), (1, 32) ----
    def p2_inputs():
        rng = _seed_inputs(102)
        return [rng.randn(1, 32).astype(np.float32)]

    def p2_jax(x):
        return jnp.exp(x)

    def p2_vanilla(inputs):
        return ttnn.exp(inputs[0])

    progs.append(("P2", "exp(x) (1x32)", p2_jax, p2_vanilla, p2_inputs, (1, 32)))

    # ---- P3: a @ b, (64, 64) ----
    def p3_inputs():
        rng = _seed_inputs(103)
        a = (rng.randn(64, 64) * 0.1).astype(np.float32)
        b = (rng.randn(64, 64) * 0.1).astype(np.float32)
        return [a, b]

    def p3_jax(a, b):
        return a @ b

    def p3_vanilla(inputs):
        return ttnn.matmul(inputs[0], inputs[1])

    progs.append(("P3", "a @ b (64x64)", p3_jax, p3_vanilla, p3_inputs, (64, 64)))

    # ---- P4: softmax(x), (1, 64) ----
    def p4_inputs():
        rng = _seed_inputs(104)
        return [rng.randn(1, 64).astype(np.float32)]

    def p4_jax(x):
        m = jnp.max(x, axis=-1, keepdims=True)
        e = jnp.exp(x - m)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    def p4_vanilla(inputs):
        # Hand-written softmax: subtract max, exp, divide by sum.
        # We use ttnn.softmax to be "what an expert would write" — that
        # is the fairer baseline since they would never write the
        # decomposition by hand.
        return ttnn.softmax(inputs[0], dim=-1)

    progs.append(("P4", "softmax (1x64)", p4_jax, p4_vanilla, p4_inputs, (1, 64)))

    # ---- P5: linear(a, w, b) = a @ w + b, a:(2,64), w:(64,32), b:(32,) ----
    def p5_inputs():
        rng = _seed_inputs(105)
        a = (rng.randn(2, 64) * 0.1).astype(np.float32)
        w = (rng.randn(64, 32) * 0.1).astype(np.float32)
        b = (rng.randn(32) * 0.1).astype(np.float32)
        return [a, w, b]

    def p5_jax(a, w, b):
        return a @ w + b

    def p5_vanilla(inputs):
        # ttnn.linear has a bias arg
        return ttnn.linear(inputs[0], inputs[1], bias=inputs[2])

    progs.append(("P5", "linear (a@w+b)", p5_jax, p5_vanilla, p5_inputs, (2, 32)))

    # ---- P6: attention(x, wq, wk, wv, wo), 8x32 single head ----
    # softmax(q @ k.T) @ v  with q=x@wq, k=x@wk, v=x@wv, out=attn@wo
    def p6_inputs():
        rng = _seed_inputs(106)
        x = (rng.randn(8, 32) * 0.1).astype(np.float32)
        wq = (rng.randn(32, 32) * 0.1).astype(np.float32)
        wk = (rng.randn(32, 32) * 0.1).astype(np.float32)
        wv = (rng.randn(32, 32) * 0.1).astype(np.float32)
        wo = (rng.randn(32, 32) * 0.1).astype(np.float32)
        return [x, wq, wk, wv, wo]

    def p6_jax(x, wq, wk, wv, wo):
        q = x @ wq
        k = x @ wk
        v = x @ wv
        # Single head, no scale (keep things simple and traceable)
        attn = jnp.exp(q @ k.T - jnp.max(q @ k.T, axis=-1, keepdims=True))
        attn = attn / jnp.sum(attn, axis=-1, keepdims=True)
        out = attn @ v
        return out @ wo

    def p6_vanilla(inputs):
        x, wq, wk, wv, wo = inputs
        q = ttnn.matmul(x, wq)
        k = ttnn.matmul(x, wk)
        v = ttnn.matmul(x, wv)
        # k.T via ttnn.transpose on last two dims
        kT = ttnn.transpose(k, -2, -1)
        scores = ttnn.matmul(q, kT)
        attn = ttnn.softmax(scores, dim=-1)
        ctx = ttnn.matmul(attn, v)
        return ttnn.matmul(ctx, wo)

    progs.append(("P6", "attention (8x32)", p6_jax, p6_vanilla, p6_inputs, (8, 32)))

    return progs


# ---------------------------------------------------------------------------
# Implementation runners
# ---------------------------------------------------------------------------


def run_vanilla_eager(program, n_iters=ITERS):
    """Implementation A: vanilla ttnn, no trace.

    Each iteration: numpy → device → ops → numpy.
    """
    pid, label, _, vanilla_fn, inputs_factory, out_shape = program
    inputs_np = inputs_factory()

    def one_iter():
        # Upload
        inputs_dev = [engine._to_device(a) for a in inputs_np]
        out = vanilla_fn(inputs_dev)
        return _read_output(out, out_shape)

    # Sanity check output
    sample = one_iter()
    median, p90 = time_loop(one_iter, iters=n_iters)
    return median, p90, sample


def run_vanilla_trace(program, n_iters=ITERS):
    """Implementation B: vanilla ttnn with manual begin/end_trace_capture."""
    pid, label, _, vanilla_fn, inputs_factory, out_shape = program
    inputs_np = inputs_factory()

    vt = VanillaTrace()
    try:
        vt.compile(inputs_np, vanilla_fn, out_shape)
    except Exception as e:
        # Trace capture failed — return None so caller can mark it.
        return None, None, None, f"trace-capture-failed: {e}"

    def one_iter():
        return vt.run(inputs_np)

    sample = one_iter()
    median, p90 = time_loop(one_iter, iters=n_iters)
    vt.release()
    return median, p90, sample, None


def _serialize_bytecode(fn, *example_args):
    """Lower a jax fn to a stablehlo VHLO portable artifact."""
    cpu_dev = jax.devices("cpu")[0]
    with jax.default_device(cpu_dev):
        lowered = jax.jit(fn).lower(*example_args)
        module = lowered.compiler_ir(dialect="stablehlo")
    return serialize_portable_artifact(module, get_current_version())


def run_pjrt_traced(program, n_iters=ITERS):
    """Implementation C: engine.execute_stablehlo with trace cache warm.

    Lowers the JAX function, hands the bytecode to the engine, runs
    once to warm the trace cache, then times cache-hit replays.
    """
    pid, label, jax_fn, _, inputs_factory, out_shape = program
    inputs_np = inputs_factory()

    bc = _serialize_bytecode(jax_fn, *inputs_np)

    # Make sure trace cache is enabled for this run
    saved = engine._NO_TRACE
    engine._NO_TRACE = False

    # Warm: first call captures (or attempts to capture) the trace
    try:
        first_out = engine.execute_stablehlo(bc, inputs_np)
    except Exception as e:
        engine._NO_TRACE = saved
        return None, None, None, f"execute_stablehlo-failed: {e}"

    # Inspect cache
    cache_entry = engine._trace_cache.get(hash(bc), {})
    failed = cache_entry.get('failed', True)
    note = None
    if failed:
        note = f"no-trace (cache failed: {cache_entry.get('error', '?')[:80]})"

    def one_iter():
        return engine.execute_stablehlo(bc, inputs_np)

    sample = one_iter()
    median, p90 = time_loop(one_iter, iters=n_iters)
    engine._NO_TRACE = saved
    return median, p90, sample, note


# ---------------------------------------------------------------------------
# Equivalence check
# ---------------------------------------------------------------------------


def _to_numpy_1d(x):
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.flatten().astype(np.float32)
    return np.array(x, dtype=np.float32).flatten()


def cosine(a, b):
    a = _to_numpy_1d(a)
    b = _to_numpy_1d(b)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    num = float(np.dot(a, b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return num / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "research" / "pjrt_phase5_benchmarks.md")
    ap.add_argument("--label", default="vanilla-vs-pjrt",
                    help="Run label.")
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()

    print(f"[bench] Opening device 0 and warming kernel cache...")
    _ = engine._get_device()
    _ = engine._to_device(np.zeros((32, 32), dtype=np.float32))
    _sync()

    programs = make_programs()
    rows = []  # (id, label, vanilla_eager_med, vanilla_eager_p90,
               #  vanilla_traced_med, vanilla_traced_p90,
               #  pjrt_traced_med, pjrt_traced_p90, ratio, note)

    for prog in programs:
        pid, label, _, _, _, _ = prog
        print(f"\n[{pid}] {label}")
        notes = []

        # Vanilla eager
        a_med, a_p90, a_out = run_vanilla_eager(prog, n_iters=args.iters)
        print(f"  vanilla eager   : med={a_med:8.1f}us  p90={a_p90:8.1f}us")

        # Vanilla traced
        try:
            b_med, b_p90, b_out, b_note = run_vanilla_trace(prog,
                                                            n_iters=args.iters)
        except Exception as e:
            b_med = b_p90 = None
            b_out = None
            b_note = f"crashed: {e}"
        if b_med is None:
            print(f"  vanilla traced  : SKIP ({b_note})")
            notes.append(f"vanilla-traced: {b_note}")
        else:
            print(f"  vanilla traced  : med={b_med:8.1f}us  p90={b_p90:8.1f}us")
            if b_note:
                notes.append(f"vanilla-traced: {b_note}")

        # PJRT traced
        try:
            c_med, c_p90, c_out, c_note = run_pjrt_traced(prog,
                                                          n_iters=args.iters)
        except Exception as e:
            c_med = c_p90 = None
            c_out = None
            c_note = f"crashed: {e}"
        if c_med is None:
            print(f"  PJRT traced     : SKIP ({c_note})")
            notes.append(f"pjrt-traced: {c_note}")
        else:
            print(f"  PJRT traced     : med={c_med:8.1f}us  p90={c_p90:8.1f}us")
            if c_note:
                notes.append(f"pjrt-traced: {c_note}")

        # Equivalence
        cos_ab = cosine(a_out, b_out) if b_out is not None else float("nan")
        cos_ac = cosine(a_out, c_out) if c_out is not None else float("nan")
        print(f"  cosine(A vs B)  : {cos_ab:.4f}")
        print(f"  cosine(A vs C)  : {cos_ac:.4f}")
        if cos_ab == cos_ab and cos_ab < 0.99:
            notes.append(f"cos(A,B)={cos_ab:.3f}")
        if cos_ac == cos_ac and cos_ac < 0.99:
            notes.append(f"cos(A,C)={cos_ac:.3f}")

        # Ratio: PJRT-traced / vanilla-traced (the headline)
        if c_med is not None and b_med is not None and b_med > 0:
            ratio = c_med / b_med
        else:
            ratio = None

        rows.append({
            "id": pid,
            "label": label,
            "a_med": a_med, "a_p90": a_p90,
            "b_med": b_med, "b_p90": b_p90,
            "c_med": c_med, "c_p90": c_p90,
            "ratio": ratio,
            "notes": "; ".join(notes) if notes else "",
        })

    # Render markdown
    git_sha = ""
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        pass

    header = (f"\n## Vanilla tt-nn vs PJRT comparison "
              f"({time.strftime('%Y-%m-%d %H:%M:%S')}, sha={git_sha}"
              f", label={args.label})\n\n")

    table = [
        "| Program | Vanilla eager med/p90 (us) | Vanilla traced med/p90 (us) | PJRT traced med/p90 (us) | PJRT / vanilla-traced | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    def fmt(med, p90):
        if med is None:
            return "—"
        return f"{med:.1f} / {p90:.1f}"

    def fmt_ratio(r):
        if r is None:
            return "—"
        return f"{r:.2f}"

    for r in rows:
        table.append(
            f"| {r['id']} {r['label']} "
            f"| {fmt(r['a_med'], r['a_p90'])} "
            f"| {fmt(r['b_med'], r['b_p90'])} "
            f"| {fmt(r['c_med'], r['c_p90'])} "
            f"| {fmt_ratio(r['ratio'])} "
            f"| {r['notes']} |"
        )

    # Prose summary
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    if ratios:
        mean_r = float(np.mean(ratios))
        if abs(mean_r - 1.0) <= 0.10:
            verdict = (f"PJRT-traced is at parity with vanilla tt-nn-traced "
                       f"(mean ratio {mean_r:.2f} across "
                       f"{len(ratios)} comparable programs; within ±10%).")
        elif mean_r > 1.0:
            pct = (mean_r - 1.0) * 100
            verdict = (f"PJRT-traced is **{pct:.0f}% slower** than vanilla "
                       f"tt-nn-traced (mean ratio {mean_r:.2f} across "
                       f"{len(ratios)} comparable programs).")
        else:
            pct = (1.0 - mean_r) * 100
            verdict = (f"PJRT-traced is **{pct:.0f}% faster** than vanilla "
                       f"tt-nn-traced (mean ratio {mean_r:.2f} across "
                       f"{len(ratios)} comparable programs).")
    else:
        verdict = "No comparable program pairs — every PJRT or vanilla trace failed."

    body = header + "\n".join(table) + "\n\n" + verdict + "\n"
    print("\n" + body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        args.out.write_text(
            "# Phase 5 Benchmark Results\n\n"
            "Run with the bench scripts in pjrt_plugin/tests/.\n")
    with args.out.open("a") as f:
        f.write(body)
    print(f"[bench] Appended to {args.out}")


if __name__ == "__main__":
    main()
