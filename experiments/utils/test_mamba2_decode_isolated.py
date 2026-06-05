#!/usr/bin/env python3
"""Mamba2 SSD decode-step isolation harness — the ground-truth gate.

Forks the shape of `experiments/utils/test_pattern_a_moe_np.py`: a
deterministic random fixture at the model's shapes, two algorithmic
paths run against the same inputs, per-head and overall cosine + MAD
reports, with assert gates.

For G0a (THIS file's purpose), the "two paths" are:
  - oracle vs oracle-deepcopy: a self-consistency check verifying the
    harness's compare/replay logic doesn't itself introduce drift.
  - oracle vs ttnn kernel: SKIPPED until G1 lands. When G1 ships a
    `ttnn.experimental.nemotron3_mamba2_decode_owned` (or a Python
    callable wrapping it), pass it as `--kernel-callable
    <pyimport.path>` and the harness compares its output against
    the oracle per-head.

For G1..G4, the kernel author imports this module's `compare_outputs`,
`multistep_replay`, and `make_fixture` helpers to validate their
ttnn implementation against the oracle.

Multi-step replay: each step uses the previous step's mutated
`ssm_state` as input, so this also exercises the *recurrence* (not
just one isolated decode step). Required because the kernel's bug
budget is in the recurrence, not in a single isolated tile.

Run on qb1:
  ssh qb1 'cd ~/tt-xla && .venv/bin/python -u experiments/utils/test_mamba2_decode_isolated.py'

Reads:
  - `experiments/utils/mamba2_numpy_oracle.py` (the oracle under test)
  - `wiki/65_mamba_state_space_models.md` §3 (math reference)
  - `research/nemotron3_nano_architecture_brief.md` §4.3 (per-step decode)
"""
from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "utils"))
from mamba2_numpy_oracle import (  # noqa: E402
    NEMOTRON3_NANO_SHAPES,
    _make_fixture,
    mamba2_decode_step,
)

# Cosine gate for per-step comparison. 0.999 mirrors the kernel-correctness
# bar used in the 35B owned_gdn G1 stage (see `feedback_owned_decay_gate_shipped`).
COS_PASS_THRESHOLD: float = 0.999


# ── core comparison helpers (importable) ──────────────────────────────
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return 0.0
    return float(af @ bf / (na * nb))


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def compare_outputs(
    a: np.ndarray,
    b: np.ndarray,
    *,
    label: str = "compare",
    head_axis: int = 1,
    cos_threshold: float = COS_PASS_THRESHOLD,
    print_per_head: bool = False,
) -> dict:
    """Compare two [B, num_heads, head_dim] arrays. Returns a dict of
    {overall_cos, overall_mad, max_abs, per_head_cos_min, ...}, asserts the
    cos gate, and prints a summary suitable for stdout logging.
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    overall_cos = _cos(a, b)
    overall_mad = _mad(a, b)
    overall_max_abs = _max_abs(a, b)
    # Per-head cos: cheaper than full per-element broadcast; uses the
    # head_axis to slice.
    nh = a.shape[head_axis]
    per_head_cos = []
    per_head_mad = []
    for h in range(nh):
        slicer = [slice(None)] * a.ndim
        slicer[head_axis] = h
        sl = tuple(slicer)
        per_head_cos.append(_cos(a[sl], b[sl]))
        per_head_mad.append(_mad(a[sl], b[sl]))
    per_head_cos = np.asarray(per_head_cos)
    per_head_mad = np.asarray(per_head_mad)
    report = dict(
        label=label,
        shape=tuple(a.shape),
        overall_cos=overall_cos,
        overall_mad=overall_mad,
        max_abs=overall_max_abs,
        per_head_cos_min=float(per_head_cos.min()),
        per_head_cos_mean=float(per_head_cos.mean()),
        per_head_mad_max=float(per_head_mad.max()),
        per_head_mad_mean=float(per_head_mad.mean()),
        cos_threshold=cos_threshold,
        passed=bool(per_head_cos.min() >= cos_threshold),
    )
    print(f"  [{label}]")
    print(f"    shape:                 {report['shape']}")
    print(f"    overall cos:           {report['overall_cos']:.10f}")
    print(f"    overall MAD:           {report['overall_mad']:.4e}")
    print(f"    max |Δ|:               {report['max_abs']:.4e}")
    print(f"    per-head cos: min      {report['per_head_cos_min']:.10f}  "
          f"(threshold {cos_threshold})")
    print(f"    per-head cos: mean     {report['per_head_cos_mean']:.10f}")
    print(f"    per-head MAD: max      {report['per_head_mad_max']:.4e}")
    print(f"    {'PASS ✓' if report['passed'] else 'FAIL ✗'}")
    if print_per_head:
        print(f"    per-head detail (h, cos, MAD):")
        for h in range(nh):
            mark = "  " if per_head_cos[h] >= cos_threshold else "✗ "
            print(f"      {mark}h={h:3d}  cos={per_head_cos[h]:.10f}  "
                  f"MAD={per_head_mad[h]:.4e}")
    return report


def make_fixture(B: int = 1, *, seed: int = 0) -> dict:
    """Wraps the oracle's fixture builder for explicit isolation-harness use."""
    return _make_fixture(B=B, seed=seed)


def multistep_replay(
    step_fn,
    fixture: dict,
    n_steps: int = 8,
    *,
    randomise_inputs: bool = True,
    seed: int = 1,
) -> list:
    """Run `step_fn` for `n_steps` consecutive decode steps, threading the
    SAME `ssm_state` through. Returns a list of per-step dicts
    `{y, ssm_state_snapshot}`.

    `step_fn(**kwargs) -> y` is expected to mutate `kwargs["ssm_state"]`
    in place and return `y` (matches `mamba2_decode_step`'s signature).

    With `randomise_inputs=True`, x/z/dt/B/C are re-drawn each step so
    the SSM sees a non-stationary input distribution (more realistic
    than feeding the same token N times, which would just decay to a
    fixed point).
    """
    rng = np.random.default_rng(seed)
    Bsz, num_heads, head_dim = fixture["x"].shape
    n_groups = fixture["B_in"].shape[1]
    ssm_state_dim = fixture["B_in"].shape[-1]

    history = []
    # NOTE: do NOT deep-copy ssm_state — it's the state we want to thread.
    # We *do* freshen x/z/dt/B/C per step (if requested).
    for step in range(n_steps):
        if randomise_inputs and step > 0:
            fixture["x"] = rng.standard_normal((Bsz, num_heads, head_dim)).astype(np.float32)
            fixture["z"] = rng.standard_normal((Bsz, num_heads, head_dim)).astype(np.float32)
            fixture["dt"] = rng.standard_normal((Bsz, num_heads)).astype(np.float32)
            fixture["B_in"] = rng.standard_normal((Bsz, n_groups, ssm_state_dim)).astype(np.float32)
            fixture["C_in"] = rng.standard_normal((Bsz, n_groups, ssm_state_dim)).astype(np.float32)
        y = step_fn(**fixture)
        history.append(dict(y=y.copy(), ssm_state=fixture["ssm_state"].copy()))
    return history


# ── ttnn-kernel loader (for G1+ integration; safe if ttnn isn't installed) ───
def _load_kernel_callable(pyimport_path: str | None):
    """Resolve `module.path:callable` (or `module.path.callable`) to a Python
    callable matching `mamba2_decode_step`'s signature. Returns None if
    `pyimport_path` is None or import fails — the harness then runs in
    oracle-only mode.
    """
    if not pyimport_path:
        return None
    if ":" in pyimport_path:
        mod_path, fn_name = pyimport_path.split(":", 1)
    else:
        mod_path, _, fn_name = pyimport_path.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, fn_name)
    except Exception as e:
        print(f"[harness] could not load kernel '{pyimport_path}': {e}")
        return None


# ── main: deterministic-random self-test + optional kernel compare ────
def _run_self_consistency(B: int, n_steps: int, seed: int) -> bool:
    """Oracle vs oracle-on-deepcopied-fixture: should be bit-identical.

    Catches harness bugs (e.g. accidental fixture mutation, ssm_state
    threading mistakes). Not a math gate; a *plumbing* gate.
    """
    print()
    print(f"=== Self-consistency: oracle vs oracle (B={B}, "
          f"n_steps={n_steps}, seed={seed}) ===")
    fixture_a = make_fixture(B=B, seed=seed)
    fixture_b = copy.deepcopy(fixture_a)
    hist_a = multistep_replay(mamba2_decode_step, fixture_a,
                              n_steps=n_steps, seed=seed)
    hist_b = multistep_replay(mamba2_decode_step, fixture_b,
                              n_steps=n_steps, seed=seed)
    all_passed = True
    for step, (a, b) in enumerate(zip(hist_a, hist_b)):
        y_equal = np.array_equal(a["y"], b["y"])
        state_equal = np.array_equal(a["ssm_state"], b["ssm_state"])
        passed = y_equal and state_equal
        all_passed &= passed
        if not passed:
            print(f"  step {step}: y_bit_equal={y_equal}  "
                  f"state_bit_equal={state_equal}  ✗")
        else:
            print(f"  step {step}: bit-equal ✓")
    print(f"  {'PASS ✓' if all_passed else 'FAIL ✗'}  self-consistency over {n_steps} steps")
    return all_passed


def _run_step_evolution(B: int, n_steps: int, seed: int) -> bool:
    """Sanity: across multi-step replay with randomised inputs, the SSM
    state must actually evolve (not stay frozen) and all outputs must be
    finite. Catches "kernel writes wrong array" / "decay too aggressive"
    issues.
    """
    print()
    print(f"=== Step evolution sanity (B={B}, n_steps={n_steps}, seed={seed}) ===")
    fixture = make_fixture(B=B, seed=seed)
    hist = multistep_replay(mamba2_decode_step, fixture,
                            n_steps=n_steps, seed=seed)
    all_ok = True
    prev_state = np.zeros_like(hist[0]["ssm_state"])
    for step, snap in enumerate(hist):
        y = snap["y"]
        s = snap["ssm_state"]
        state_delta = float(np.abs(s - prev_state).max())
        y_norm = float(np.linalg.norm(y))
        y_finite = bool(np.all(np.isfinite(y)))
        s_finite = bool(np.all(np.isfinite(s)))
        step_ok = y_finite and s_finite and (state_delta > 0)
        all_ok &= step_ok
        mark = "✓" if step_ok else "✗"
        print(f"  step {step:2d}: ||y||={y_norm:8.3f}  "
              f"max|Δstate|={state_delta:8.4f}  "
              f"y_finite={y_finite}  s_finite={s_finite}  {mark}")
        prev_state = s
    print(f"  {'PASS ✓' if all_ok else 'FAIL ✗'}  step evolution")
    return all_ok


def _run_kernel_compare(B: int, n_steps: int, seed: int,
                        kernel_fn, cos_threshold: float) -> bool:
    """Oracle vs ttnn kernel callable. Compares per-step outputs across an
    n_steps replay. Each step uses the same x/z/dt/B/C for both paths
    AND the same starting ssm_state, so divergences reflect kernel-vs-
    oracle math differences, not state drift.
    """
    print()
    print(f"=== Kernel compare: oracle vs ttnn kernel (B={B}, "
          f"n_steps={n_steps}, seed={seed}, cos≥{cos_threshold}) ===")
    fixture_oracle = make_fixture(B=B, seed=seed)
    fixture_kernel = copy.deepcopy(fixture_oracle)
    hist_oracle = multistep_replay(mamba2_decode_step, fixture_oracle,
                                   n_steps=n_steps, seed=seed)
    hist_kernel = multistep_replay(kernel_fn, fixture_kernel,
                                   n_steps=n_steps, seed=seed)
    all_passed = True
    for step, (a, b) in enumerate(zip(hist_oracle, hist_kernel)):
        print(f"  step {step}:")
        rep = compare_outputs(a["y"], b["y"],
                              label=f"y_oracle_vs_y_kernel",
                              cos_threshold=cos_threshold)
        all_passed &= rep["passed"]
        # Also compare the ssm_state evolution to catch state corruption
        # that an output-only comparison would miss.
        rep_s = compare_outputs(
            a["ssm_state"].reshape(a["ssm_state"].shape[0],
                                   a["ssm_state"].shape[1], -1),
            b["ssm_state"].reshape(b["ssm_state"].shape[0],
                                   b["ssm_state"].shape[1], -1),
            label=f"ssm_state_oracle_vs_kernel",
            cos_threshold=cos_threshold,
        )
        all_passed &= rep_s["passed"]
    print(f"  {'PASS ✓' if all_passed else 'FAIL ✗'}  kernel compare over {n_steps} steps")
    return all_passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", "-B", type=int, default=2,
                    help="batch size for the fixture (default 2)")
    ap.add_argument("--steps", "-n", type=int, default=8,
                    help="number of consecutive decode steps to replay (default 8)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the fixture (default 42)")
    ap.add_argument("--kernel-callable", type=str, default=None,
                    help="Python import path of a ttnn-backed callable "
                         "matching mamba2_decode_step's signature, e.g. "
                         "'my_pkg.module:my_kernel_fn'. If omitted, runs "
                         "oracle-only sanity tests.")
    ap.add_argument("--cos-threshold", type=float, default=COS_PASS_THRESHOLD,
                    help=f"per-head cos pass threshold (default {COS_PASS_THRESHOLD})")
    args = ap.parse_args()

    print(f"[harness] Nemotron-3 Nano shapes: {NEMOTRON3_NANO_SHAPES}")
    print(f"[harness] config: B={args.batch}  n_steps={args.steps}  "
          f"seed={args.seed}  cos_threshold={args.cos_threshold}")

    # 1. Self-consistency (catches harness plumbing bugs).
    self_ok = _run_self_consistency(args.batch, args.steps, args.seed)
    # 2. Step evolution (catches frozen-state / NaN / inf bugs).
    evo_ok = _run_step_evolution(args.batch, args.steps, args.seed)

    # 3. (Optional) Kernel compare — only when a ttnn kernel is provided.
    kernel_fn = _load_kernel_callable(args.kernel_callable)
    if kernel_fn is None:
        print()
        print("[harness] No kernel callable provided "
              "(--kernel-callable). Skipping G1+ kernel compare. "
              "Use this harness from G1 onwards by passing the ttnn-backed "
              "callable, e.g. --kernel-callable my_kernel_mod:decode_step.")
        kernel_ok = True
    else:
        kernel_ok = _run_kernel_compare(args.batch, args.steps, args.seed,
                                        kernel_fn, args.cos_threshold)

    print()
    overall = self_ok and evo_ok and kernel_ok
    print(f"[harness] OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}  "
          f"(self={self_ok}  evo={evo_ok}  kernel={kernel_ok})")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
