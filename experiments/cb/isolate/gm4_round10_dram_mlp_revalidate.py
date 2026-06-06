#!/usr/bin/env python3
"""Round 10 Phase 4 — production integration validator.

Forks the dev-harness state-version-skew pattern from
`gm4_invalidate_trace.py` and the workflow from `gm4_v04_trace_validate.py`.

Workflow:
  1. Re-upload every layer's MLP weights under `TT_GM4_DRAM_PREFETCH=1`
     (replaces in-place; deallocates the prior bf16 INTERLEAVED-DRAM
     tensors before rebinding to WIDTH_SHARDED DRAM).
  2. Invalidate the captured decode trace (state.trace_id → None).
  3. Re-capture trace + run v04 validator (100 eager + 100 traced,
     token-for-token gate).
  4. Print per-tok ms and any divergence.

Why this exists: the resident gm4 dev harness has State.layer_tt[L]
bootstrapped under the default bf16 INTERLEAVED-DRAM path. The Round 10
integration in `server_gemma4_unified_ttnn.py` gates upload-side
behaviour on TT_GM4_DRAM_PREFETCH; a `_reload` of the server module
does NOT re-upload existing weights. This trigger does the re-upload
explicitly (only the MLP, not Q/K/V/O — Round 10 scope) so we can A/B
without a 14-min cold restart.

Reuse precedent: `gm4_invalidate_trace.py:1` (state.trace_id release
pattern). Re-upload loop forks
`server_gemma4_unified_ttnn.py:upload_mlp_layer` (called once per
layer in the bootstrap loop at `:694`).

Trigger via the dev harness on qb2:
  ssh qb2 'touch tt-xla/.cache/gm4_runtime/trig/round10_dram_mlp_revalidate'
  ssh qb2 'cat tt-xla/.cache/gm4_runtime/trig/last.log'
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "serve"))

import server_gemma4_unified_ttnn as srv  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _reupload_all_mlp(state, srv_mod, log=log):
    """For every layer index, re-read MLP weights from the safetensors
    shards, deallocate the existing layer_tt[L]['gate/up/down_proj']
    tensors, and upload fresh ones under the current env-gate.

    Forks `bootstrap()`'s per-layer loop at server_gemma4_unified_ttnn.py:
    694-697 (which calls `upload_mlp_layer(layer_sd, mesh)` per layer).
    """
    import ttnn

    # Build key→shard index once (same call bootstrap uses).
    variant = os.environ.get("TT_GEMMA4_VARIANT", "it")
    key_to_shard = srv_mod.build_key_to_shard(variant=variant)

    # Load all layer state dicts. Each layer's three MLP tensors live in
    # the same shard file (HF Gemma 4 12B sharded by layer ranges).
    from safetensors import safe_open

    NUM_LAYERS = srv_mod.NUM_LAYERS
    t0 = time.time()
    for L in range(NUM_LAYERS):
        # Gather just the three MLP keys we need. HF key prefix in Gemma 4
        # 12B is "model.language_model.layers.{L}." (server_gemma4...py:817).
        # `upload_mlp_layer` consumes the stripped form "mlp.<name>.weight".
        layer_sd = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.language_model.layers.{L}.mlp.{name}.weight"
            path = key_to_shard[key]
            with safe_open(path, framework="pt", device="cpu") as f:
                layer_sd[f"mlp.{name}.weight"] = f.get_tensor(key).float().numpy()
        # Deallocate the existing MLP weights before rebinding (avoids
        # the [[ttnn-list-rebinding-leaks]] L1 fragmentation pitfall).
        # Defensive: a prior failed re-upload may have left dangling
        # references — `is_allocated()` would be the right check but
        # isn't exposed on every ttnn build, so we swallow the exception.
        layer_tt = state.per_layer_tt[L]
        for name in ("gate_proj", "up_proj", "down_proj"):
            old = layer_tt.get(name)
            if old is not None:
                try:
                    ttnn.deallocate(old)
                except Exception:
                    pass
        new_mlp = srv_mod.upload_mlp_layer(layer_sd, state.mesh)
        layer_tt.update(new_mlp)
        if L % 8 == 0:
            log(f"  L{L:2d} re-uploaded ({(time.time()-t0):.1f}s)")
    log(f"all {NUM_LAYERS} MLP layers re-uploaded in {time.time()-t0:.1f}s")


def _invalidate_trace(state, log=log):
    import ttnn
    old_id = getattr(state, "trace_id", None)
    if old_id is None:
        log("state.trace_id already None")
        return
    try:
        ttnn.release_trace(state.mesh, old_id)
    except Exception as e:
        log(f"release_trace warning: {e}")
    state.trace_id = None
    log(f"released trace id {old_id}")


def _run_validator(state, srv_mod, label, log=log):
    """Forks the per-run loop from gm4_v04_trace_validate.py."""
    PROMPT_IDS = [2, 818, 5279, 529, 7001, 563]
    N_STEPS = 100
    log(f"[{label}] running {N_STEPS} EAGER steps…")
    t0 = time.time()
    eager_argmax = []
    prev = None
    for pos in range(N_STEPS):
        tok = PROMPT_IDS[pos] if pos < len(PROMPT_IDS) else prev
        a = srv_mod.step_forward_v031(state, int(tok), pos)
        eager_argmax.append(int(a))
        prev = int(a)
    eager_ms = (time.time() - t0) * 1000 / N_STEPS
    log(f"  eager {eager_ms:.1f} ms/tok; first 10: {eager_argmax[:10]}")

    log(f"[{label}] capturing decode trace…")
    t0 = time.time()
    srv_mod.ensure_decode_trace(state, log=log)
    log(f"  trace capture took {time.time()-t0:.1f}s")

    log(f"[{label}] running {N_STEPS} TRACED steps…")
    t0 = time.time()
    traced_argmax = []
    prev = None
    for pos in range(N_STEPS):
        tok = PROMPT_IDS[pos] if pos < len(PROMPT_IDS) else prev
        a = srv_mod.step_forward_traced(state, int(tok), pos)
        traced_argmax.append(int(a))
        prev = int(a)
    traced_ms = (time.time() - t0) * 1000 / N_STEPS
    log(f"  traced {traced_ms:.1f} ms/tok; first 10: {traced_argmax[:10]}")

    n_match = sum(1 for a, b in zip(eager_argmax, traced_argmax) if a == b)
    first_div = next((i for i, (a, b) in enumerate(zip(eager_argmax, traced_argmax)) if a != b),
                     None)
    log(f"[{label}] eager-vs-traced match: {n_match}/{N_STEPS}")
    if first_div is not None:
        log(f"  first divergence at pos={first_div}: "
            f"eager={eager_argmax[first_div]} traced={traced_argmax[first_div]}")
    return {
        "match": n_match,
        "total": N_STEPS,
        "first_div": first_div,
        "eager_ms": eager_ms,
        "traced_ms": traced_ms,
        "eager_first10": eager_argmax[:10],
        "traced_first10": traced_argmax[:10],
    }


def main(state=None):
    if state is None:
        log("ERROR: needs dev harness (resident State)")
        return 1

    # The dev harness pre-sets TT_GM4_DRAM_PREFETCH=0 (default) at session
    # start. To run this probe we explicitly enable for the duration of
    # this trigger; the env var is process-wide so subsequent triggers
    # without explicit re-set will pick it up too. That's fine here — the
    # whole point of this probe is to land the DRAM-sharded MLP path and
    # measure both correctness + perf.
    os.environ["TT_GM4_DRAM_PREFETCH"] = "1"
    log(f"set TT_GM4_DRAM_PREFETCH=1 (was: previously unset/empty)")

    # Defensive: reload server module so the new upload_mlp_layer is in
    # use. (Harness `_reload` happens before triggers anyway, but explicit
    # is better.)
    importlib.reload(srv)
    srv_mod = sys.modules["server_gemma4_unified_ttnn"]
    log(f"server module reloaded: id={id(srv_mod)}")
    log(f"_dram_sharded_enabled() = {srv_mod._dram_sharded_enabled()}")

    log("re-uploading all 48 MLP layers under current env-gate…")
    _reupload_all_mlp(state, srv_mod, log=log)

    log("invalidating cached trace…")
    _invalidate_trace(state, log=log)

    log("running v04 validator under new MLP path…")
    result = _run_validator(state, srv_mod, label="round10-dram", log=log)

    log("=" * 78)
    verdict = "PASS" if result["match"] == result["total"] else "FAIL"
    log(f"VERDICT: {verdict}  (match {result['match']}/{result['total']}, "
        f"eager {result['eager_ms']:.1f}, traced {result['traced_ms']:.1f} ms/tok)")
    log("=" * 78)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
