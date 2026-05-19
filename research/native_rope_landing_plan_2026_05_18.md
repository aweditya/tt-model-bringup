# Native Partial RoPE Landing Plan (2026-05-18 evening)

Lands the slice-first `ttnn.experimental.rotary_embedding` recipe as the
production default for the gated-attention RoPE path. Targets **RoPE
cluster (#5 in the post-owned-gdn profile menu)**: 320 ops / 54.68 ms /
7.44% per `research/post_owned_gdn_profile_2026_05_18.md`.

**This is not a kernel build.** It's a small Python integration + G3
validation + default flip. Estimated <1 day. The hard infrastructure
work (`state.rope_mode` flag, branch in `gated_attn_step_tp`, isolated
correctness probe, guarded trace probe) was all done in earlier sessions
(May 14-15) and the artifacts are in `.cache/qb2_tp_rope/`. Today's
session would only do G3 + G4.

## Why this is the right next move (after conv1d/decay_gate ship)

Per `research/post_owned_gdn_profile_2026_05_18.md` ranked menu:
- Native RoPE has the **best win-per-day-of-work ratio** of any item on
  the list. No new C++ kernel. No new tt-metal build cycle. No new owned
  op tree to maintain.
- Existing TTNN primitive `ttnn.experimental.rotary_embedding` is
  already well-tested; the only Tenstorrent-side risk
  (`rotary_embedding_llama_fused_qk` wedge — `feedback_p1` and ACTIVE_
  CONTEXT) is sidestepped by using the unfused variant on only the
  rotary subset of dims (slice-first recipe).
- Projected savings: ~1-2 ms/tok in trace at the same eager→trace
  compression ratio observed for GDN/conv1d. Smaller than GDN's 2.6 ms
  but real and additive.

## Math contract

Production RoPE on qb2 currently uses two paths, toggled by
`state.rope_mode` (`server_tp.py:118`, `:812`):

```python
def apply_rope_manual(q_or_k, n_heads):
    # Pure Python TTNN: rotate-half on the first ROTARY_DIM=64 columns;
    # concatenate the passthrough HEAD_DIM-ROTARY_DIM=192 columns.
    # Many small ttnn ops; lots of slices + concats.
    ...

def apply_rope_native_partial(q_or_k, n_heads):
    # Slice-first ttnn.experimental.rotary_embedding on rotary_dim=64
    # only, then concat with the passthrough 192-dim tail.
    # Single experimental op for the rotation, two slice/concat at the
    # boundaries.
    ...
```

Default is `"manual"`. Setting to `"native_partial"` routes to the
fused-internally TTNN op.

## Prior-art audit (already paid for)

Done in earlier sessions; do **NOT** re-attempt these:

| approach | result | citation |
|---|---|---|
| `ttnn.experimental.rotary_embedding_llama_fused_qk` (full Q+K combined) | wedged qb2 for >10 min; required `pkill -9` + `tt-smi -r` | `feedback_p1`, ACTIVE_CONTEXT "Invalidated Or Safe-But-No-Win Paths" |
| Slice-first `ttnn.experimental.rotary_embedding` on rotary dims only | **PASSED isolation gate** — 7/7 positions, min Q PCC 0.9999975, min K PCC 0.9999975, max tail diff 0.00091 | `.cache/qb2_tp_rope/results_native_partial_pass_20260515_0030.json` |
| Same recipe under a guarded production trace | **PASSED 20-token guarded trace** (argmax match), measured trace deltas: 0.42 ms/step execute, 0.27 ms/step update+execute vs manual baseline same-session | `.cache/qb2_tp_rope/results_native_partial_trace_20260515_0041.json` + `..._manual_baseline_after_native_20260515_0042.json` |

The slice-first recipe is already wired into `gated_attn_step_tp` at
`server_tp.py:812-817` behind the `rope_mode` flag.

## Outstanding work (G3 + G4 — this plan covers these)

### G3 — `cosine_ladder_tp` 500-position long-context

Extend `handle_cosine_ladder_tp` + the client wrapper to accept a
`--rope-mode {manual,native_partial}` arg (same pattern as the
recently-added `--deltanet-conv1d-mode`). Then on qb2:

```bash
# Server lifetime 1
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    cosine_ladder_tp --prompt "Implement a JSON parser combinator in Rust" \
    --max-tokens 500 --modes owned_gdn --rope-mode manual'
# Restart server (owned_gdn 2nd-invocation slowdown still applies)
ssh qb2 'cd ~/tt-xla && bash experiments/serve/scripts/serve_tp.sh stop'
ssh qb2 'tt-smi -r 0,1,2,3'
ssh qb2 'cd ~/tt-xla && bash experiments/serve/scripts/serve_tp.sh start'
# Wait for ready
# Server lifetime 2
ssh qb2 'cd ~/tt-xla && .venv/bin/python -m experiments.serve.client_tp \
    cosine_ladder_tp --prompt "Implement a JSON parser combinator in Rust" \
    --max-tokens 500 --modes owned_gdn --rope-mode native_partial'
# Compare
python experiments/utils/cosine_ladder_compare_two_npzs.py \
    --base <manual_npz> --other <native_partial_npz> \
    --base-mode manual --other-mode native_partial \
    --prompt "..." \
    --out .cache/qb2_tp_deltanet/cosine_ladder_tp_compare_rope_<timestamp>.json
```

Gate: 10/500 disagreement rate or better (matches conv1d/GDN ships),
median cosine ≥ 0.999, NO cliff in rolling 50-step bucket medians.

### G4 — Default flip

If G3 passes, edit `MeshServerState.__init__` to set
`self.rope_mode = "native_partial"` (currently `"manual"`).
Cold-bootstrap qb2; verify-after-flip on canonical prompts.

Expected per-tok delta vs current production: the May-15 component
benchmark measured **0.42 ms execute, 0.27 ms update+execute** native
vs manual on the same captured trace. At ~80 ms/tok production decode
that's roughly **0.5% ship perf delta** if all of the component delta
survives to full decode (it usually doesn't — pipelining hides ~50%
per the `feedback_v2_rope_perf_wash.md` lesson). Net: maybe 0.1-0.3
ms/tok = 0.1-0.4% perf, plus 320 fewer dispatched ops per token (cleaner
trace).

Rollback: `state.rope_mode = "manual"` in `MeshServerState.__init__`
and re-bootstrap.

## Effort

| phase | wall time |
|---|---|
| Extend `cosine_ladder_tp` endpoint + client with `--rope-mode` | <30 min (mirrors `--deltanet-conv1d-mode`) |
| G3 run (2 server lifetimes × ~3.5 min + 2 bootstraps × 17 min cold) | ~45 min |
| Compare + analyze | 10 min |
| G4 commit + verify | 30 min |

Total: ~2 hours of focused work, mostly bootstraps. Same shape as today's
conv1d G3 session.

## Why this comes BEFORE the next custom kernel (output_gate or QKV repeat)

1. **It's already 90% done** — only G3 + G4 remain, both small.
2. **It's not a kernel build** — no risk of tt-metal compile loops or
   `.so`-sync gotchas during the wait.
3. **It clears the queue** — landing native RoPE means the next custom
   kernel (when we pick one) lands on a cleaner trace with fewer
   dispatched RoPE ops, making the next profile cleaner to read.
4. **Risk-adjusted return is highest** — the prior-art audit shows
   correctness AND perf wins are already measured at isolation;
   production landing is mechanical.

## What we will NOT do in this plan

- Will not revive `rotary_embedding_llama_fused_qk` (Q+K combined). The
  prior wedge probe is decisive. The unfused variant is what we ship.
- Will not also fuse cos/sin lookup into the RoPE op — P25 already
  ships on-device cos/sin via `ttnn.embedding(rot_idxs_buf,
  cos_table_tt)`. No further fusion runway there.
- Will not also touch DeltaNet's QK-norm path (which has its own
  L2-norm logic, separate from RoPE).
