# DeepSeek-V3 `_build_verify_alias_page_table_host` reference

Lifted verbatim from `~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/
generator.py:43-90` on qb1 (2026-06-07 read). To be forked into our
`experiments/serve/spec_dec_scheduler.py` at Phase 2.B.

## Purpose

For spec-dec verify at `B=K+1`: builds a page-table such that K+1 logical
batch rows all read from the SAME physical KV slot. The B=K+1 forward
then re-evaluates the target on K+1 hypothetical next-tokens in
parallel (the K drafter candidates + 1 bonus), each row seeing the
identical KV history.

## Reference code (verbatim from tt-metal)

```python
def _build_verify_alias_page_table_host(
    base_page_table: torch.Tensor,
    num_prompts: int,
    verify_offset: int,
    prompt_indices: List[int] | None = None,
    interleaved: bool = False,
) -> torch.Tensor:
    """Build a host-side aliased page table for verify batching."""
    if num_prompts <= 0:
        return base_page_table.clone()
    if base_page_table.dim() != 2:
        raise RuntimeError(
            f"Unexpected page table rank for MTP verify aliasing: "
            f"{tuple(base_page_table.shape)}")

    alias_page_table = base_page_table.clone().to(torch.int32)
    num_rows = int(alias_page_table.shape[0])
    if num_rows <= 0:
        raise RuntimeError(
            "Page table has zero rows; cannot build MTP verify aliasing.")

    prompt_indices_for_alias = prompt_indices
    if not interleaved and prompt_indices_for_alias is None:
        prompt_indices_for_alias = list(range(num_prompts))

    if interleaved:
        if prompt_indices_for_alias is None:
            for row in range(1, num_rows, 2):
                alias_page_table[row] = alias_page_table[row - 1]
        else:
            for i in prompt_indices_for_alias:
                if i < 0:
                    continue
                src_row = (2 * i) % num_rows
                dst_row = (src_row + 1) % num_rows
                alias_page_table[dst_row] = alias_page_table[src_row]
    else:
        for i in prompt_indices_for_alias:
            if i < 0 or i >= num_prompts:
                continue
            src_row = i % num_rows
            dst_row = (verify_offset + i) % num_rows
            alias_page_table[dst_row] = alias_page_table[src_row]

    return alias_page_table
```

## Argument semantics

- `base_page_table`: shape `[num_rows, num_pages_per_seq]` int32. The
  target's current page-table where row `i` indexes the KV blocks for
  prompt `i`.
- `num_prompts`: how many prompts are in the active CB pool. For
  single-stream spec-dec, this is `1`.
- `verify_offset`: where in the page-table to write the alias rows. For
  K=5 verify-batching against 1 active prompt at slot 0, this is
  typically the first free slot after the prompt's own row.
- `prompt_indices`: which active prompt rows get aliased into the
  verify rows. For single-stream B=K+1: `[0]` aliases prompt 0's KV
  into K+1 verify rows starting at `verify_offset`.
- `interleaved`: if True, alias rows are placed every other row
  (`src=2i, dst=2i+1`). For our parallel-drafter pattern, use
  `interleaved=False`.

## Our usage (single-stream spec-dec, K=5)

```python
# Caller: spec_dec_scheduler.SpecDecScheduler._target_verify_kp1
# Single active prompt at row 0 of the target's page-table.
# Want K+1 = 6 verify rows (one per candidate + 1 bonus) all aliased to row 0's KV.

alias = _build_verify_alias_page_table_host(
    base_page_table=target_state.page_table_tt_cpu,
    num_prompts=1,
    verify_offset=1,          # rows 1..6 = aliased verify rows
    prompt_indices=[0],       # but actually we want K+1 not 1; see below
    interleaved=False,
)
```

NOTE: The DeepSeek-V3 reference is designed for *multi-prompt* MTP
verify (alias each active prompt once into a verify slot). For our
*single-prompt, multi-candidate* spec-dec we need to make
**K+1 alias rows** all pointing at prompt 0:

```python
# Our adaptation for single-stream B=K+1 spec-dec
alias = base_page_table.clone().to(torch.int32)
for i in range(K + 1):
    alias[verify_offset + i] = alias[0]  # all K+1 verify rows alias to prompt 0's KV
```

## Two-call paged SDPA caveat

Gemma 4 12B uses the two-call paged-SDPA workaround
([[reference-gemma4-two-call-paged-decode]]) because `NKV_PER_CHIP=2`.
This was authored for B=1; spec-dec at B=K+1 may need the workaround
generalized. Test at Phase 2.A — likely fine since the workaround
addresses NKV-per-chip, not B, but verify.

## Verify-trace shape contract

The B=K+1 verify forward will need:
- `x` input: shape `[B=K+1, S=1, HIDDEN=3840]` — each row is the
  hypothetical next-token's embedding (i.e., the K drafter candidates +
  the bonus prev-token's continuation)
- `page_table_tt`: the aliased K+1-row table from above
- Output logits: `[B=K+1, S=1, VOCAB=262144]` after lm_head — host
  takes argmax per row and runs the accept walk

## Source

`~/tenstorrent/tt-metal/models/demos/deepseek_v3/tt/generator.py` lines 43-90.
SHA at qb1 read time: unverified (qb1 tracks main branch of tt-metal vendored subtree).
