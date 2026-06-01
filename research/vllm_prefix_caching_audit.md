# vLLM Prefix Caching — Audit for tt-model-bringup CB Server

**Date:** 2026-06-01
**Goal:** Self-contained reference of vLLM's automatic prefix caching (APC), so we can decide what (if anything) to port into our Qwen3.6 continuous-batching server. Does **not** propose an implementation — this is recon.

All citations are file paths in `vllm-project/vllm@main` (commit-floating; line numbers may drift) or external URLs.

---

## 1. High-level mechanism

### 1.1 What problem APC solves
APC caches the KV-cache blocks of already-processed requests and reuses them when a new request shares a prefix. The new request **skips prefill compute for the matched prefix**; only the unmatched suffix is prefilled.

> "Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part."
> — `docs/features/automatic_prefix_caching.md` (vllm-project/vllm)

Two canonical wins called out in the docs:
- **Long document Q&A**: same long document, many queries → document is prefilled exactly once.
- **Multi-turn chat**: conversation history is prefilled exactly once, regardless of round count.

Caveat (from the same doc): **APC does not reduce decode time**, so it is irrelevant when the answer is much longer than the prompt, or when no prefix is shared.

User-facing knob: `enable_prefix_caching=True` on the engine (`LLM(..., enable_prefix_caching=True)` / `vllm serve --enable-prefix-caching`).

### 1.2 Block-hash matching
vLLM uses **block-level hashing**, not a trie. Each full KV block is hashed by:

```
block_hash = hash((parent_block_hash, tuple(token_ids), extra_keys))
```

Where:
- **Parent hash** = the hash of the previous block in the same sequence (so each hash transitively identifies the whole prefix).
- **Token IDs** = the exact tokens in this block (lowers collision risk).
- **Extra keys** = LoRA ID, multi-modality input hash, optional `cache_salt` for tenant isolation.

This is illustrated literally in the design doc:

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```
— `docs/design/prefix_caching.md`

> "We only cache full blocks." Partial blocks are not hashed.

Hash function as of v0.11 defaults to **`sha256`**, with `sha256_cbor` for reproducibility across Python versions, and `xxhash` / `xxhash_cbor` for speed (non-crypto). Selectable via `--prefix-caching-hash-algo`.

### 1.3 APC vs RadixAttention (SGLang)
| Aspect | vLLM APC | SGLang RadixAttention |
|---|---|---|
| Granularity | Block-level (block_size tokens) | Token-level |
| Data structure | Hash map: `block_hash → KVCacheBlock` | Radix tree (compressed trie) of token sequences |
| Match resolution | Hash chain lookup, block-by-block | Tree traversal (longest common prefix) |
| Handles partial blocks? | No — only full blocks are caching candidates | Yes — splits tree nodes on a fork |
| Eviction | LRU on doubly-linked free queue | LRU on tree nodes |
| Best at | Many requests sharing a fixed system prompt | Branchy / forking workloads (agents, tree search) |

Sources:
- [vllm/docs/design/prefix_caching.md](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [SGLang RadixAttention blog](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [Don Moon — Prefix Caching SGLang vs vLLM](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1)

vLLM chose the hash-map approach explicitly because **it doesn't need to maintain a tree** — blocks are allocated and freed independently, with O(1) operations on the free queue.

### 1.4 "Evict-only" policy
This phrase is a slight misnomer. vLLM does **not** have a separate eviction policy that "kicks in" — eviction is the natural consequence of the LRU free queue:

1. All blocks live in a fixed-size **block pool** (`vllm/v1/core/block_pool.py`), allocated up front.
2. Free blocks are tracked in a doubly-linked **`FreeKVCacheBlockQueue`** in LRU order (head = oldest, tail = newest).
3. Cached blocks stay in `cached_block_hash_to_block` (the global hash table) **and** also live in the free queue when `ref_cnt == 0`, marked but reclaimable.
4. When the scheduler asks for a new block:
   - Pop from the head of the free queue.
   - If the popped block was a cached block (`block.block_hash is not None`), **drop its hash table entry** before reuse — that's the "eviction" step.
5. On request completion, blocks are freed back to the **tail** of the queue in **reverse order**, so the last (tail) blocks of a sequence are evicted first and prefix blocks live longer.

Pseudocode quote from `docs/design/prefix_caching.md`:
> "Pop the block from the head of the free queue. This is the LRU block to be evicted. Remove the block ID from the cache block. Remove the block hash."

There is no high-/low-watermark, no async sweeper, no TTL — purely demand-driven LRU.

---

## 2. Concrete API + admit-time flow

### 2.1 What the scheduler does at admit time

From `vllm/v1/core/kv_cache_manager.py` (`KVCacheManager.get_computed_blocks`, line ~194):

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    """Get the computed (cached) blocks for the request.
    Note that the computed blocks must be full.
    """
```

Flow (paraphrased from `docs/design/prefix_caching.md` §"Block Allocation"):

1. `Request.update_block_hashes()` — compute the chained block hashes for all full blocks in the prompt. Called at request creation.
2. `kv_cache_manager.get_computed_blocks(request)` — look up each hash in `cached_block_hash_to_block`. Returns the **longest prefix of hash hits** (stops at first miss). Returns `(KVCacheBlocks, num_computed_tokens)`.
3. `kv_cache_manager.allocate_slots(request, num_new_tokens, num_computed_tokens, ...)`:
   - Compute how many *additional* blocks are needed (prompt + decode slots beyond cached prefix).
   - "**Touch**" the computed blocks: `ref_cnt += 1`, and if `ref_cnt` was 0, splice them out of the middle of the free queue so they can't be evicted while in use.
   - Pop new blocks from the head of the free queue (this is where eviction happens for any cached-but-unreferenced LRU blocks).
   - For any newly-full block, immediately compute its hash and insert into `cached_block_hash_to_block` so siblings in the same batch can reuse it.

This is exposed to the scheduler via `KVCacheBlocks` (defined at the top of `kv_cache_manager.py`, line 25):

```python
@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """
    blocks: tuple[Sequence[KVCacheBlock], ...]
```

It carries one list of blocks per KV cache **group** (there can be multiple groups for hybrid models — see §4).

### 2.2 Data structures

From `vllm/v1/core/block_pool.py` and `kv_cache_utils.py`:

```python
class KVCacheBlock:
    block_id: int            # immutable
    block_hash: BlockHash    # set when block is full; cleared on evict
    ref_cnt: int             # # requests using this block
    prev_free_block: "KVCacheBlock | None"
    next_free_block: "KVCacheBlock | None"
```

The `BlockPool` owns:
- `blocks: list[KVCacheBlock]` — the pre-allocated pool (no GC overhead during serving).
- `free_block_queue: FreeKVCacheBlockQueue` — doubly-linked LRU list with sentinel head/tail; O(1) remove-from-middle.
- `cached_block_hash_to_block: dict[BlockHashWithGroupId, KVCacheBlock]` — the global cache index.

**Why a flat hash map, not a trie:** the design explicitly notes that the hash-chain property (each block's hash transitively identifies its full prefix) means tree structure is unnecessary; the map is sufficient and lets blocks be freed independently.

### 2.3 Does it literally skip prefill?

**Yes.** Cached blocks are inserted into the request's per-request block table; the prefill kernel is launched on `num_prompt_tokens - num_computed_tokens` tokens starting at offset `num_computed_tokens`. Attention reads the cached KV through the page table — no recompute.

In vLLM v1's tt fork (`vllm/v1/worker/tt_model_runner.py`, see PR #272), this is exactly what changed:

```python
# Before:
input_positions = 0
# After:
# num_computed_tokens for each request is the input position
# (=computed previously and cached)
input_positions = input_batch.num_computed_tokens_cpu[:num_reqs]
```
— [tenstorrent/vllm#272](https://github.com/tenstorrent/vllm/pull/272), `vllm/v1/worker/tt_model_runner.py:399`

So at the kernel boundary, "prefix cache hit" reduces to **"prefill starts at offset `num_computed_tokens` and the block table already points at the cached pages."** The attention op itself is unchanged.

---

## 3. KV cache + block table mechanics under APC

- **Immutable cached blocks.** Once hashed and inserted, the contents are frozen. `ref_cnt` controls lifetime; a block with `ref_cnt > 0` is unevictable.
- **Refcounting** is per-block, not per-request. Multiple in-flight requests sharing a prefix all bump the same block's `ref_cnt`.
- **Block tables are append-only in v1.** The doc explicitly calls this out: in v0, a block found to be a duplicate of an existing cached block could be swapped in-place; in v1 the block table cannot be rewritten, so **duplicate blocks are tolerated and reclaimed only on request free** (`docs/design/prefix_caching.md` §"Duplicated blocks").
- **On eviction**: pop from free queue head → `del cached_block_hash_to_block[block.block_hash]` → `block.block_hash = None` → return for reuse. The physical page is overwritten by the next prefill that lands on it.
- **Free on request done**: blocks freed to the **tail** of the queue, in **reverse order** of the request's block list. Tail (suffix) blocks are evicted first; root prefix blocks linger.
- **Attention kernel is transparent.** PagedAttention already operates against a per-request block table; APC just populates the early entries of that table with cached block IDs instead of newly-allocated ones. No SDPA changes needed.

---

## 4. Non-attention state — Mamba / SSM / Gated DeltaNet (CRITICAL FOR US)

### 4.1 Does vLLM's APC support recurrent-state models?

**Yes, for Mamba-2-based hybrid models, but only as of late 2025 and still experimental.** Pure Mamba-1 is not yet supported.

- Tracking issue: [vllm-project/vllm#26201 "[Tracking Issue]: Prefix Caching for Hybrid Models"](https://github.com/vllm-project/vllm/issues/26201) — open follow-ups include "implementing policy for freeing mamba blocks", "relaxing the constraint that mamba block size must be multiple of chunk size", "enabling prefix caching for Mamba1".
- RFC: [vllm-project/vllm#17140 "Native support for Mamba, SSM, and hybrid transformer models in vLLM V1"](https://github.com/vllm-project/vllm/issues/17140).
- Design doc: [vllm Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/).
- Per-PyTorch blog: "In vLLM V1, hybrid model support was rebuilt around a unified allocator that manages both KV cache and Mamba state, which enables advanced features like prefix caching, KV cache transfer, and prefill/decode disaggregation." ([pytorch.org](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/))

### 4.2 How is recurrent state checkpointed at block boundaries?

The unified allocator manages **multiple KV cache groups** (cf. `KVCacheBlocks.blocks: tuple[Sequence[KVCacheBlock], ...]` — one inner sequence per group). For a hybrid model there is a "transformer KV" group and a "mamba state" group.

The `mamba_block_size` config and `mamba_cache_mode` flag (see [vllm.config.cache](https://docs.vllm.ai/en/latest/api/vllm/config/cache/)) control checkpoint policy:

- `none` — set when prefix caching is disabled.
- `all` (default with APC on) — cache the Mamba state at positions `i * block_size` for all `i`.
- `align` — cache only the Mamba state of the **last token of each scheduler step** when that token sits at a block boundary. Much less memory, but less reusable.

`mamba_block_size` must be a **multiple of the SSM chunk size** (= multiple of 8 for `causal_conv1d`). The chunked prefill kernel naturally produces intra-chunk states, and Marconi's trick (paper [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)) is to **materialize and persist the second-to-last chunk state** — that gives you the state at the prefix boundary without re-running prefill. For models without chunked prefill, the paper proposes a **two-pass prefill** to recover the exact prefix state.

> "SSM model states are updated in place, so states at the end of a sequence cannot be rolled back to represent a prefix of the sequence. Maximizing reuse opportunities mandates caching fine-grained state checkpoints at regular intervals, but increased checkpointing frequency inflates the number of cache entries generated per sequence, each of which is large but most present limited reuse opportunities."
> — Marconi paper, arXiv:2411.19379

### 4.3 Known caveats specific to hybrid prefix caching

- **Block-granularity floor.** Prompts shorter than `mamba_block_size` get 0% hit rate. Reported as bug: [vllm-project/vllm#40696 "Prefix caching completely ineffective for Mamba-hybrid models (Qwen3.5) when prompt < block_size (528 tokens)"](https://github.com/vllm-project/vllm/issues/40696).
- **Off-by-one with MTP**: [vllm-project/vllm#39809 "Mamba prefix caching + MTP speculative decoding crashes on startup for NemotronH"](https://github.com/vllm-project/vllm/issues/39809); also a Twitter report from Yifei Hu noting "Mamba/GDN state cache reused one block too many while full attention correctly recomputed the boundary block" in Qwen3.5.
- **MM + Mamba broken**: [vllm-project/vllm#43587 "Prefix caching fails for incremental multimodal requests on Mamba-Attention hybrid models (Qwen3.5)"](https://github.com/vllm-project/vllm/issues/43587).
- **Ascend backend mirror**: [vllm-project/vllm-ascend#7103 "support prefix cache for Qwen3.5/Next with --mamba-cache-mode align"](https://github.com/vllm-project/vllm-ascend/pull/7103) — note the explicit `align` flag in the title; Ascend disables `all` mode and only ships `align`.

**Implication for us (Qwen3.6-35B-A3B has GatedDeltaNet, which is the same family as Qwen3.5-Next):** the upstream design pattern is real and proven, but the implementation is fresh and has open known-bad cases for hybrid models. If we port APC, we are walking into an actively-bug-shaken area.

---

## 5. Tenstorrent vLLM fork specifics

### 5.1 Current state — APC is **on** for V1, gated by model capability

The Tenstorrent fork at [tenstorrent/vllm](https://github.com/tenstorrent/vllm) **landed initial APC support** in late 2025:

- Tracking: [tenstorrent/vllm#268 "[Feature]: Add support for automatic prefix caching (V1)"](https://github.com/tenstorrent/vllm/issues/268) (still open as of 2026-03-31).
- Landing PR: [tenstorrent/vllm#272 "Automatic Prefix Caching support"](https://github.com/tenstorrent/vllm/pull/272) — closed (merged).
- Companion model-side PR in tt-metal: [tenstorrent/tt-metal#33883](https://github.com/tenstorrent/tt-metal/pull/33883) (TT-Transformers non-traced) + [tenstorrent/tt-metal#35904](https://github.com/tenstorrent/tt-metal/pull/35904) (Llama70B-Galaxy).

Status checklist from the tracking issue:
- [x] vLLM-side support (excluding sliding window)
- [x] TT-Transformers text models (non-traced)
- [x] Llama70B-Galaxy
- [ ] **TT-Transformers models with traced prefill (TODO)** ← *this is us*
- [ ] Multi-modal models (TODO)

### 5.2 What the TT plugin actually changed

PR #272 is *tiny* — 5 files, ~110 LOC of substantive change. Key edits:

**a) `vllm/platforms/tt.py`** — drop the blanket "TT can't do APC" assert, replace with a capability check on the model class:

```python
# Get model capabilities from the class
model_capabilities: Optional[dict] = getattr(model_class, "model_capabilities", None)

if vllm_config.cache_config.enable_prefix_caching:
    supports_prefix_caching = (model_capabilities.get(
        "supports_prefix_caching", False)
        if model_capabilities else False)

    if not supports_prefix_caching:
        vllm_config.cache_config.enable_prefix_caching = False
        logger.warning(
            "Prefix caching is not supported in TT backend for %s, "
            "disabling it", model_class.__module__)
    else:
        uses_sliding_window = (
            vllm_config.model_config.get_sliding_window() is not None)
        if uses_sliding_window:
            vllm_config.cache_config.enable_prefix_caching = False
            logger.warning(
                "Prefix caching is not supported in TT backend for "
                "models with sliding window, disabling it")
```

Models opt in by declaring `model_capabilities = {"supports_prefix_caching": True}`. The TT-side bringup (tt-metal PR #33883) adds that flag plus the per-model code to honour the prefix-start offset.

**b) `vllm/v1/worker/tt_model_runner.py`** — the substantive runtime change:

```python
# Was:
input_positions = 0
# Is now:
# num_computed_tokens for each request is the input position
# (=computed previously and cached)
input_positions = input_batch.num_computed_tokens_cpu[:num_reqs]
```

And in `concat_dp_model_inputs` / `execute_with_model_input`, `input_positions` gets propagated all the way through to `kwargs["start_pos"]` for both prefill and decode paths.

**c) `vllm/engine/arg_utils.py`** — delete the V1-disables-APC fallback for TT.

### 5.3 Measured TTFT improvement (from PR #272 test results)

Llama-3.1-8B-Instruct on N300, 32 prompts × block_size 64, repeating each prompt N times:

| repeat_count | prefill (s) |
|---|---|
| 1 | 0.243 |
| 2 | 0.152 |
| 4 | 0.106 |
| 8 | 0.091 |

≈ 2.7× TTFT win at high repeat counts, consistent with "fully cached prompt → only the decode is real work".

### 5.4 What is **not** done in the TT fork

- **Traced prefill + APC** — not yet (and that's our path). The non-traced TT-Transformers code can take a `start_pos` argument; the traced path bakes `start_pos = 0` into the captured graph, so they need a tensor-borne `start_pos` (analogous to the `update_idxs_tensor=` work we already did for `paged_update_cache`).
- **Hybrid / Mamba models** — no mention in PR #272 or issue #268. The capability check is per-model, so any DeltaNet bringup will be `model_capabilities["supports_prefix_caching"] = False` by default.
- **Multi-modal** — also TODO.

So in practice, on TT, APC today means "dense-attention text models, eager (non-traced) prefill, no sliding window."

---

## 6. Minimum-viable APC design for our server

### 6.1 Our constraints (recap)
- Single device or `(1, 4)` mesh; trace-captured decode is non-negotiable for perf.
- KV cache already block-paged at `page_size = 32` tokens (per `paged_update_cache`).
- Recurrent state per layer: `H_t ∈ [B, NV, K, V]` (DeltaNet) + `conv_cols` 3 columns (per slot).
- 27B is dense attention. 35B-A3B is hybrid (attention + GatedDeltaNet) — our hard case.
- Single-user chat is the dominant use case, but the server is multi-tenant CB (B up to 32).

### 6.2 The three honest options

**Option A — "session passthrough" (no APC, but the chat case still wins).**
Key the *whole CB slot* by `client_id` + `session_id`. When a returning client posts to `/v1/chat/completions`, look up their slot if it's still alive (not yet evicted by the scheduler), verify their new prompt's prefix matches the slot's `tokens_so_far`, and resume from `cur_pos`. No hashing, no block-level reuse, no DN state checkpointing — the state is *already in the slot*.

- LOC budget: ~150-300 LOC in `cb_scheduler.py` + a session→slot map.
- Wins: 100% TTFT elimination for the "user is still chatting in the same tab" case.
- Loses: no cross-tenant prefix sharing (no shared system prompt benefit), no benefit after a session is evicted.
- DN state: zero new work — it's already correct in the slot.
- This is what we should do **first**. It mirrors what most "GPT chat" toy servers do under the hood.

**Option B — "dense-only APC, 27B path."**
Port the TT fork's PR #272 pattern: capability flag on the model, prefill from `start_pos = num_computed_tokens`, expose `cached_block_hash → block_id` from the scheduler.
- For 27B (no DN), this is mechanically the same as the upstream TT plugin work.
- Critical adaptation for us: **our prefill is traced**, so `start_pos` must be a device tensor (the analog of `update_idxs_tensor`). The TT fork explicitly punts this — we'd be on point.
- DN state: not in scope (27B has none).
- This is the right v2 if Option A's hit rate is too low across distinct users.

**Option C — "hybrid APC for 35B with DN."**
The Marconi pattern: checkpoint DN's `(H_t, conv_cols)` at every `page_size`-aligned token of prefill, store as a second cache group keyed by the same block hash, and re-load on hit.
- DN state per layer is `bf16 [NV, K, V] = bf16 [32, 128, 128] = 1 MiB/layer` plus `conv_cols [3, hidden] ≈ 30 KiB/layer`. For 36 layers ≈ 36 MiB per checkpoint. At one checkpoint every 32 tokens, that's **1.1 MiB/token** of DRAM, which dwarfs the KV pages themselves. We'd want `mamba_cache_mode="align"` semantics: checkpoint only at the *last* boundary of the prefix.
- Trace implications: the DN state copy-out has to happen *inside* the prefill trace at the right offset, and the copy-in has to land before the first decode step in the next request's trace.
- This is **not** a v1. It's research-grade and tracks the open work in vllm#26201.

### 6.3 Recommendation (research view, not a commitment)
Build **Option A** as the MVP. It captures the dominant chat use case, requires no kernel changes, no hash bookkeeping, and no DN checkpointing. Defer B until we have evidence (server logs) that cross-user prefix overlap is material. Defer C until Marconi-style hybrid APC stabilizes upstream — the cost/benefit for 35B-A3B is unclear today.

---

## 7. Open questions / known unknowns

1. **TT traced-prefill + APC API.** The TT fork's checklist (tenstorrent/vllm#268) explicitly leaves "TT-Transformers models with traced prefill" unchecked. We need to understand *how* tt-metal plans to thread `start_pos` into a captured prefill graph — is it the same `update_idxs_tensor=` pattern we used for `paged_update_cache`, or something else? Look at how tt-metal PR #33883 handles `start_pos` in non-traced mode first.

2. **DeltaNet state checkpoint cost on Blackhole.** What does it actually cost (kernel time + DRAM BW) to copy `H_t [32, 128, 128] bf16` out of L1 to a DRAM "checkpoint" buffer per layer, mid-prefill? If it's cheap relative to a prefill step, Option C is plausible; if it's a 20% prefill regression, it's dead.

3. **Hash algorithm choice.** vLLM defaults to `sha256` (Python `pickle`). Our CB scheduler is Python-side single-threaded; do we care about hash compute time, or is it noise next to the prefill? Probably noise — but worth checking with the `xxhash` option before committing.

4. **Eviction policy for hybrid groups.** vllm#26201 explicitly lists "implementing policy for freeing mamba blocks" as open. The naive "evict KV block → invalidate any DN checkpoint pointing to or past it" is correct but might be wasteful (DN checkpoints are 30× larger than KV blocks per token). What's the right eviction unit for the DN group?

5. **One-block-too-many bug.** Yifei Hu's report on Qwen3.5 (Mamba/GDN reusing one block too many vs. attention) is the exact failure mode we'd hit on 35B-A3B. We need to **understand** that bug before we trust any hybrid-APC port — find the issue/PR, read the fix, decide whether our DN state lives at the same "after the boundary token" position or one before.

6. **Tenant isolation in our setting.** vLLM's `cache_salt` model assumes the *client* opts in to sharing by sending a matching salt. For a single-tenant research server, we can ignore this; for any multi-user deployment, decide if cross-user prefix reuse is desirable (it usually is — the system prompt is shared) or a privacy risk (it sometimes is — the user content is not).

7. **Interaction with our trace cache.** Our traced decode is bit-identical only when `cur_pos` is consistent with the cached KV layout. If APC hands us a request that starts at `cur_pos = 1024` instead of `cur_pos = 0`, does the existing trace handle that, or do we need a new trace per starting position? Suspect the latter is fine because `cur_pos` is already a tensor input — but verify.

---

## Sources (full list)

vLLM upstream:
- [`docs/features/automatic_prefix_caching.md`](https://github.com/vllm-project/vllm/blob/main/docs/features/automatic_prefix_caching.md)
- [`docs/design/prefix_caching.md`](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
- [`docs/design/hybrid_kv_cache_manager`](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py` (BlockHash, KVCacheBlock)
- `vllm/v1/core/kv_cache_coordinator.py` (multi-group dispatch for hybrid)
- `vllm/v1/core/single_type_kv_cache_manager.py` (full attn / sliding window / Mamba)

vLLM issues / PRs:
- [#2614 — original APC RFC](https://github.com/vllm-project/vllm/issues/2614)
- [#26201 — Tracking Issue: Prefix Caching for Hybrid Models](https://github.com/vllm-project/vllm/issues/26201)
- [#17140 — RFC: Native Mamba/SSM/hybrid support in V1](https://github.com/vllm-project/vllm/issues/17140)
- [#40696 — Mamba-hybrid APC ineffective < block_size](https://github.com/vllm-project/vllm/issues/40696)
- [#43587 — APC + multimodal on Mamba hybrid (Qwen3.5)](https://github.com/vllm-project/vllm/issues/43587)
- [#39809 — Mamba APC + MTP crash on NemotronH](https://github.com/vllm-project/vllm/issues/39809)
- [vllm-ascend#7103 — `--mamba-cache-mode align`](https://github.com/vllm-project/vllm-ascend/pull/7103)

Tenstorrent fork:
- [tenstorrent/vllm repo](https://github.com/tenstorrent/vllm)
- [tenstorrent/vllm#268 — Feature: APC (V1)](https://github.com/tenstorrent/vllm/issues/268)
- [tenstorrent/vllm#272 — APC support PR](https://github.com/tenstorrent/vllm/pull/272)
- [tenstorrent/tt-metal#33883 — TT-Transformers non-traced APC](https://github.com/tenstorrent/tt-metal/pull/33883)
- [tenstorrent/tt-metal#35904 — Llama70B-Galaxy APC](https://github.com/tenstorrent/tt-metal/pull/35904)

External:
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [Marconi paper (arXiv:2411.19379)](https://arxiv.org/abs/2411.19379) — hybrid LLM prefix caching with SSM checkpoints
- [SGLang RadixAttention blog (LMSYS)](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [PyTorch blog: Hybrid Models as First-Class Citizens in vLLM](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)
- [Inside vLLM — Anatomy of a High-Throughput Inference System (vLLM blog)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Don Moon — Prefix Caching: SGLang vs vLLM](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1)
- [DeepWiki: KV Cache Management and Prefix Caching (vllm-project/vllm)](https://deepwiki.com/vllm-project/vllm/3.4-kv-cache-management-and-prefix-caching)
