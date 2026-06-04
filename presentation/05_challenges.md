# Challenges: what bit us, and what we now do about it

Three bringups (Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE, Gemma 4 12B hybrid) produced a long bug list. The talk-worthy ones cluster into seven categories; each rule below was paid for in chip-hours and is now load-bearing in `research/model_bringup_recipe.md`.

---

## Slide 1: the silent-failure classes

### 1. `ttnn.slice` and `ttnn.reshape` return *views*

The single most common Tenstorrent footgun. Both ops alias the source buffer; `ttnn.deallocate(source)` while any view is still read returns freed memory. Caught in the 27B CB batched attention port: a "tidy" `ttnn.deallocate(all_tt)` after slicing Q/K/V/gate from a fused projection caused per-attn-layer cosine erosion of ~0.998 that compounded to ~0.49 by L63. Invisible at decode pos 0 — attention output equals V with one KV slot (softmax of one score = 1.0), so only the V slice was correct and argmax on easy prompts still matched HF. **Rule:** when porting a proven ttnn function to a batched/owned variant, do NOT add `deallocate` calls the original didn't have. A view is only safe to free after the last read of every sibling view of the same buffer.

Sibling class: **rebinding a Python list of ttnn tensors does NOT deallocate them**. `State.reset_caches_ttnn` assigned `self.dn_caches_tt = new_list` and leaked 40 layers × 2 tensors per reset. Over a long-lived harness this fragmented the allocator — logits returned different random top-1 tokens (82530, 198294, 219673, …) for the same input. **Rule:** before rebinding any container of ttnn tensors, explicitly `ttnn.deallocate` every tensor in it.

### 2. Model-family semantic surprises that hide at pos 0

Three stand out, all sharing a fingerprint:

- **Gemma 4 SDPA uses `scale=1.0`, NOT `1/sqrt(d_k)`.** HF `Gemma4TextAttention.__init__` sets `self.scaling = 1.0`; we copy-pasted the Qwen scale. INVISIBLE at pos 0 (single K slot, softmax = 1.0 regardless). At pos 1: L0 cos 0.99998 → 0.66380, final_norm cos 0.9995 → 0.26. One-line fix per call site, commit `c97bf15`.
- **Qwen3.6 RMSNorm is zero-centered: `y = x/rms(x) * (1+γ)`** for `q_norm`/`k_norm` and input/post/final norms (but NOT `dn_norm`). A two-character fix (`+ 1.0`) at `server_35b_ttnn.py:225-226` flipped top-1 from 82/97 → **95/97**, median cos 0.95 → 0.997, and made needle-haystack retrieve `N4Y2BWLS` verbatim.
- **Gemma 4 attention has THREE per-head RMSNorms** (`q_norm`, `k_norm`, **`v_norm`** with `with_scale=False` — pure `x/rms(x)`, no weight). Missing it: mixer_out cos=0.95, RMS 3.7× too high. **Plus a per-layer `h *= layer_scalar`** (L0=0.054, L24=0.82, L47=0.053). Missing it: L0 cos=0.999975 PASS, MAD=3.64e+2; L1 collapsed to 0.49, cascade to noise.

**Rule:** validate teacher-forced multi-step (positions 0..N), not pos 0 alone. Anything inside softmax direction (Q/K rotations, attention scale) is invisible at pos 0. Read the HF source for the family — every Gemma 4 novelty surfaced by carefully reading `Gemma4TextAttention.forward` and `Gemma4UnifiedTextDecoderLayer.forward`.

### 3. Cosine alone is not enough — always check MAD

Cosine is invariant under scalar multiplication: `cos(a, k·a) = 1` for any nonzero k. The missing `layer_scalar` passed cosine at 0.999975 but had MAD 3.64e+2 — an 18× magnitude error. **Rule:** every validator prints `cos=X.XXXXXX mad=Y.YYYY` per sub-step. A sudden order-of-magnitude jump in MAD on one sub-step is a magnitude-bug fingerprint even if cos stays high. Also check `rms(my)/rms(hf)`; it should be ≈ 1.0.

---

## Slide 2: methodology, integration, and chip-state failures

### 4. Validate against ground truth, not a weaker TT path

The chunked-prefill drift hunt burned a device cycle comparing `forward_prefill_chunked_tp` against the single-token decode stub: worst cos 0.52 at L=32, "obvious bug." It wasn't — the chunked DN is independently-validated and *more accurate* than the bf16-noisy stub. Cosine-vs-stub conflated "bug" with "different and better." **Rule:** if both paths are TT approximations, the cosine gate is meaningless. Use HF/numpy oracles (`AutoModelForCausalLM` crashes on qb1's torchvision install — we build pure-numpy fp32 oracles by hand via `safe_open`) or functional gates.

### 5. bf16 chain drift at B>1 is the floor, not a bug

Every primitive in the CB35-v1.5 batched path verified bit-identical between B=1 and B=2 slot 0. Chained through 40 layers, argmax tokens diverged: B=1 ref `[8, 198, 12, 220]` vs B=2 slot 0 `[134768, 96177, 105126, 96177]`. Per-op drift is ~1 ULP from rank-3 `[B,N,D]` accumulating differently at B=1 vs B=2; 40 layers compounds past the argmax decision boundary. The MoE per-slot path showed the same — two sequential calls with identical inputs drifted ~13% rel (allocator-state-dependent kernel scheduling; bf16 `(a+b)+c ≠ a+(b+c)`). **Rule:** at B>1 in bf16, don't gate on `slot-0 == B-1`. Gate on per-slot independence, determinism, and sensible top-K.

### 6. Dev-harness PASS ≠ HTTP PASS

CB35-prod shipped after `cb35_prod_topk.py` was 4/4 PASS. The first real `/v1/chat/completions` after a 14-minute bootstrap crashed at step 1: `TypeError: only length-1 arrays can be converted to Python scalars`. The validator drove `_step_sampled_topk` directly; `cb_engine` wraps it through `cb_scheduler.advance` → `update_input_buffers` → `int(token_ids[0])`, which fails when given per-slot tensors. Two more from the same class: `cb_reset_slots` in `server_35b_cb.py` was hardcoded `# else: no-op` for `cb_B > 1` (an unlanded TODO) — DN warmup pollution caused prompt-independent Chinese-character loops at TT_CB_SLOTS=2; and when `TT_BACKEND` was added, `cb_scheduler.py` still had `import server_tp as base` hardcoded — the mesh bootstrapped 35B weights but the scheduler ran 27B's forward over them, producing coherent Qwen-style English until `/v1/models` exposed the lie. **Rule:** any HTTP-facing backend swap requires grepping `experiments/serve/` for every `import server_*`, a one-shot HTTP smoke before ship, and `/v1/models` in the smoke checklist.

### 7. The chip can change your test results overnight

The 35B teacher-forced drift cliff (cos_L32 pos5 = 0.32, FAIL) was reproducible across multiple sessions, then disappeared the next morning with NO code change. Rerun with identical code, oracle, env: 7/8 PASS, all 40 layers cos ≥ 0.987. Suspects: TT chip state across power cycles, ttnn/firmware drift, latent harness state we never identified. P150 firmware v19.5.0 also *silently disabled 20 Tensix cores per chip* (140 → 120) in Jan 2026 — roofline numbers from before that release are wrong. **Rule:** bake in periodic regression runs after every ttnn bump; check `tt-smi -s` firmware before citing peak numbers.

### Bonus: long-running processes need tmux

Both `nohup … & disown` and `setsid -f` were verified to die within minutes of SSH disconnect on qb1, mid-bootstrap. tmux survives because the tmux *server* is its own daemon. Canonical launcher: `bash scripts/run_harness_tmux.sh`.

---

## Meta-lesson

Six of seven categories failed **silently** or **only at pos 1+**: view-decay and SDPA scale=1.0 both masked by single-K-slot softmax; Qwen3.6 zero-centered RMSNorm hid behind Q/K direction errors that only matter when Q dots a prior K; missing `layer_scalar` passed cosine; bf16 chain drift was bit-identical per-op; the cb_engine call-shape gap passed the dev harness; backend-dispatch holes produced coherent text in the wrong model's voice. The single methodology that catches all of them is the **teacher-forced multi-step cosine + MAD ladder against an HF oracle** — every position, magnitudes not just directions, ground truth not a sibling TT path. Free-run argmax on an easy prompt is the worst possible correctness gate; it is exactly where every silent bug above passes.
