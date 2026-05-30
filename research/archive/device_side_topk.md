# Device-Side Top-K Routing for Traced MoE Decode

## 1. Does ttnn.topk Exist?

**Yes.** Confirmed available and tested on our Blackhole P150.

```python
vals, idxs = ttnn.topk(input_tensor, k, dim=-1, largest=True, sorted=True)
# Returns: (values_tensor, indices_tensor)
# Indices dtype: UINT16 (if dim <= 65535) or UINT32
```

**API from help(ttnn.topk):**
- `input_tensor`: must be on device, BFLOAT16 or BFLOAT8 in TILE layout
- `k`: number of top elements (must be <= 64 for multicore)
- `dim`: dimension to reduce (default: last dim, -1)
- `largest`: True for largest elements (default)
- `sorted`: True to return sorted (default)
- Returns tuple: `(values_tensor, indices_tensor)`

**Constraints:**
- Input is internally treated as 4D `[N, C, H, W]` with `dim=-1` (operates on W)
- `N*C*H` must be a multiple of 32
- W ideally >= 64 (padded automatically if smaller)
- Multicore: requires W >= 8192 AND W < 65536, with k <= 64
- Single-core fallback for W < 8192 or W >= 65536

**For our MoE case (60 experts):** W=60 (padded to 64 by the op). This falls into single-core mode since 64 < 8192, but on such a tiny dimension it should be sub-millisecond. This is **not** the same situation as exp 82 where topk on vocab=151936 took 1890ms -- that was 2400x wider.

Source: `ttnn.topk` is a first-class C++ op (`is_experimental=False`).

## 2. How tt-metal's DeepSeek V3 Handles MoE Routing

tt-metal has a full DeepSeek V3 implementation at:
`models/demos/deepseek_v3/tt/` with files: `moe.py`, `moe_gate.py`, `experts.py`

### Routing Architecture (MoEGate)

DeepSeek V3 uses a **hierarchical 3-stage topk** approach, all on device:

1. **Gate projection**: `logits = ttnn.linear(x, W_gate)` then `scores = ttnn.sigmoid(logits)`
2. **Score correction**: adds a learned bias to scores (DeepSeek's auxiliary-loss-free balancing)
3. **Top-2 within expert groups**: reshape scores into groups, `ttnn.topk(grouped_scores, k=2)`
4. **Top-K expert groups**: sum top-2 scores per group, `ttnn.topk(group_sums, k=K)`
5. **Create active mask**: `ttnn.scatter` to build a binary mask of active groups
6. **Expand + mask**: replicate group mask to all experts, multiply with original scores
7. **Final top-K experts**: `ttnn.topk(masked_scores, k=K)` to get final expert indices
8. **Gather original scores**: `ttnn.gather(original_scores, dim=3, index=topk_indices)`
9. **Normalize**: divide each score by sum of selected scores

Key detail: they have a **topk_fallback** path that reads data to CPU and uses `torch.topk` when the on-device topk has issues. This suggests `ttnn.topk` may still have edge cases.

### Expert Execution (Experts)

DeepSeek V3 uses **dense execution with stacked weights**:
```python
# ALL experts run on every input (no conditional execution)
w1_out = ttnn.linear(x, experts_stacked_w1)  # stacked [n_experts, hidden, intermediate]
w3_out = ttnn.linear(x, experts_stacked_w3)
activated = ttnn.mul(w1_out, w3_out)
output = ttnn.linear(activated, experts_stacked_w2)
```

Expert weights are stacked into single 3D tensors. The routing mask is applied AFTER all experts compute, not before. This wastes compute on non-selected experts but enables:
- Static computation graph (traceable)
- Single large matmul instead of many small ones
- No data-dependent control flow

### Token Dispatch (MoE)

For multi-device, they use `ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` -- custom collective ops that route tokens to devices holding the relevant experts. Not applicable to our single-device case.

The weighted combination is:
```python
# post_combine shaped [num_experts_per_tok, 1, tokens, hidden]
# topk_weights shaped to match expert dim
weighted = ttnn.mul(expert_outputs, topk_weights)
result = ttnn.sum(weighted, dim=0)  # sum across expert dimension
```

## 3. The 60-Expert Problem: Why We Can't Use DeepSeek's Approach

DeepSeek V3 runs ALL experts densely. For their multi-device setup with 8+ devices, each device only holds a subset of experts, so the "run all experts on this device" approach makes sense.

**Our situation is different:**
- Single device, 60 experts, top-4 routing
- Running all 60 experts = 15x the compute of running 4
- Each expert MLP: 3 matmuls of [1, 2048] x [2048, 1408] and [1, 1408] x [1408, 2048]
- 60 experts x 3 matmuls = 180 matmuls per layer vs 12 matmuls for top-4
- At ~25us dispatch per op: 180 x 25us = 4.5ms dispatch overhead per layer
- 24 layers = 108ms just in dispatch -- already worse than current 91ms/tok

**Running all 60 experts is not viable for single-device batch=1.**

## 4. Recommended Approach: Device-Side Top-4 Masking + Selective Expert Execution

### Strategy: On-device topk for routing, loop over top-4 experts in Python

```python
def moe_forward_traced(h2, dl, layer_idx):
    """MoE with on-device routing -- only CPU intervention is loop control."""

    # ── Step 1: Router (fully on device) ──────────────────
    # h2: [1, 1, 1, 2048] on device
    rl = ttnn.matmul(h2, dl["router_w"], compute_kernel_config=hifi4)  # [1,1,1,64] (60 padded to 64)
    probs = ttnn.softmax(rl, dim=-1)  # [1,1,1,64]

    # ── Step 2: On-device top-4 ───────────────────────────
    # 60 experts padded to 64 -- need to zero out positions 60-63 before topk
    # Pre-create a mask tensor: [1,1,1,64] with 1s at [0:60] and -inf at [60:64]
    # Apply: probs_masked = ttnn.add(probs, padding_mask)  (mask has -inf at padding)
    # OR: softmax already makes padding positions near-zero if router_w has zeros in those cols
    #     (our router_w is [2048, 60] padded to [2048, 64] with zeros -- softmax gives ~equal prob to padding)
    # SAFER: multiply by a binary mask [1,1,1,64] with 1s at [0:60], 0s at [60:64]
    probs_masked = ttnn.mul(probs, dl["expert_mask"])  # zero out padding positions

    top4_vals, top4_idxs = ttnn.topk(probs_masked, k=4)  # [1,1,1,4] values and indices

    # ── Step 3: Read only top-4 indices + weights (8 floats) ──
    # This is the ONLY CPU sync -- 32 bytes instead of 240 bytes
    top4_vals_np = from_dev(top4_vals, (4,))
    top4_idxs_np = from_dev(top4_idxs, (4,)).astype(int)

    # ── Step 4: Execute top-4 experts on device ───────────
    moe_acc = None
    for i in range(4):
        e = top4_idxs_np[i]
        prob = float(top4_vals_np[i])
        ew = dl["experts"][e]
        g = ttnn.matmul(h2, ew["g"], compute_kernel_config=hifi4)
        u = ttnn.matmul(h2, ew["u"], compute_kernel_config=hifi4)
        d = ttnn.matmul(ttnn.mul(ttnn.silu(g), u), ew["d"], compute_kernel_config=hifi4)
        weighted = ttnn.multiply(d, prob)
        moe_acc = weighted if moe_acc is None else ttnn.add(moe_acc, weighted)

    # ── Step 5: Shared expert (fully on device) ───────────
    sg = ttnn.matmul(h2, dl["s_gate_w"], compute_kernel_config=hifi4)
    su = ttnn.matmul(h2, dl["s_up_w"], compute_kernel_config=hifi4)
    sd = ttnn.matmul(ttnn.mul(ttnn.silu(sg), su), dl["s_down_w"], compute_kernel_config=hifi4)
    # Shared expert gate: on-device matmul + sigmoid
    if dl["seg_w"] is not None:
        seg_logit = ttnn.matmul(h2, dl["seg_w"], compute_kernel_config=hifi4)  # [1,1,1,1]
        seg_val = ttnn.sigmoid(seg_logit)  # on-device sigmoid!
        sd = ttnn.mul(sd, seg_val)
    moe_acc = ttnn.add(moe_acc, sd)

    return moe_acc
```

### What This Changes vs Exp 91

| Aspect | Exp 91 (current) | Proposed |
|--------|-----------------|----------|
| Router matmul | On device | On device |
| Softmax | CPU (numpy) | On device (ttnn.softmax) |
| Top-4 selection | CPU (np.argsort) | On device (ttnn.topk) |
| Data read to CPU | 60 floats (240 bytes) | 8 floats (32 bytes) -- indices + weights |
| h2 readback for seg | Yes (8KB) | No -- sigmoid on device |
| Expert execution | On device | On device (same) |
| Shared expert gate | CPU sigmoid | On device (ttnn.sigmoid) |
| **Total CPU syncs** | **2 per layer** (router + h2) | **1 per layer** (top4 indices only) |

### Key Implementation Details

**Padding mask for positions 60-63:**
The router weight is [2048, 60] but gets padded to [2048, 64] by TILE_LAYOUT. After matmul + softmax, positions 60-63 will have non-zero probabilities (softmax distributes over all positions). We MUST mask these before topk:

```python
# Pre-create once during weight upload:
mask_np = np.ones((1, 1, 1, 64), dtype=np.float32)
mask_np[0, 0, 0, 60:] = 0.0  # zero out padding positions
dl["expert_mask"] = ttnn.from_torch(torch.from_numpy(mask_np),
    dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
```

**On-device sigmoid for shared expert gate:**
Exp 91 reads h2 back to CPU to compute `h2 @ seg_w` then sigmoid. Instead:
```python
seg_logit = ttnn.matmul(h2, dl["seg_w"])  # [1,1,1,1] on device
seg_val = ttnn.sigmoid(seg_logit)          # on device
sd_gated = ttnn.mul(sd, seg_val)           # on device
```
This eliminates the second CPU sync entirely.

**topk output format:**
- `top4_vals`: shape [1,1,1,4], dtype bfloat16, TILE layout -- the probability weights
- `top4_idxs`: shape [1,1,1,4], dtype uint16, TILE layout -- expert indices 0-59
- Both need `ttnn.to_torch()` to read back. The indices tensor contains the actual expert numbers.

## 5. Fully Traced Approach (No CPU Sync At All)

The approach above still requires 1 CPU sync per layer (reading top-4 indices). To eliminate ALL CPU intervention, we'd need to run all experts unconditionally. Two options:

### Option A: Run All 60 Experts (Not Viable)

As analyzed above, 15x compute overhead makes this slower than the CPU-sync approach.

### Option B: Fixed Expert Subsets via Trace Variants

Pre-capture traces for common routing patterns. For batch=1 with 60 experts and top-4, there are C(60,4) = 487,635 possible combinations -- not practical.

### Option C: Masking Approach (Traceable but Wasteful)

Run all 60 experts but mask outputs by probability:
```python
# Stack all expert outputs: [60, 2048]
# probs after topk masking: [1, 60] with only 4 non-zero entries
# result = probs @ stacked_outputs  # [1, 2048]
```

This runs all 60 experts but only does the weighted sum from 60 pre-computed outputs. The compute cost is 60 expert MLPs instead of 4 -- still 15x overhead.

### Option D: The Practical Middle Ground

**Keep the 1-sync-per-layer approach.** The sync reads 32 bytes (8 floats). At PCIe gen4 speeds, this is ~1-2us of transfer time. The actual overhead is the sync latency (~50-100us for `ttnn.synchronize_device` + `to_torch`). Over 24 layers: 1.2-2.4ms total.

Compare to the expert compute savings: NOT running 56 extra experts saves ~56 x 3 x 0.025ms = 4.2ms per layer = 100ms per token. The 2.4ms sync cost is trivially worth it.

## 6. Quality: Running All 60 vs Top-4

**Not recommended.** Qwen1.5-MoE-A2.7B was trained with top-4 routing. The softmax normalization means the top-4 probabilities sum to a significant fraction of total probability mass. Including all 60 experts would:
- Dilute strong expert signals with many near-zero contributions
- Change the effective model behavior from what was trained
- Add noise from experts that were never meant to fire for this input

The original Qwen1.5-MoE paper and DeepSeek-MoE research both show that sparse routing (activating only k experts) is essential to model quality, not just an efficiency hack. The experts specialize during training under the assumption that only top-k will fire.

## 7. Summary and Recommended Next Steps

### What We Learned

1. **ttnn.topk exists and works.** For 60 experts (width=64), it will run in single-core mode but should be fast (sub-ms) since the dimension is tiny.

2. **tt-metal's DeepSeek V3 uses on-device topk** with a hierarchical routing scheme. They also have a CPU fallback path, suggesting topk may have edge cases.

3. **DeepSeek V3 runs all experts densely** with stacked weights. This is viable for their multi-device setup but not for our single-device 60-expert case.

4. **Key ops confirmed available:** `ttnn.topk`, `ttnn.softmax`, `ttnn.sigmoid`, `ttnn.scatter`, `ttnn.gather`. Our approach only needs topk + softmax + sigmoid.

### Recommended Implementation Plan

**Experiment 92: Device-Side MoE Routing**

1. Test `ttnn.topk(k=4)` on a [1,1,1,64] tensor on Blackhole -- verify correctness and speed
2. Test `ttnn.sigmoid` on a scalar tensor [1,1,1,1] -- verify it works for shared expert gate
3. Implement the approach from Section 4: on-device router + topk, CPU reads only 32 bytes per layer
4. Benchmark vs exp 91 (current: 1 large sync per layer vs proposed: 1 tiny sync per layer)
5. Verify text quality matches exp 91 exactly (same routing decisions, just computed on device)

**Expected improvement:** Eliminating the h2 readback (8KB) and router logit readback (240 bytes) in favor of a single 32-byte readback should save ~50-100us per layer. Over 24 layers, maybe 1-2ms savings. The bigger win is that this architecture is closer to fully traceable -- the only CPU-dependent part is the expert loop control, which could eventually be replaced with a fixed dispatch pattern.

---

*Research conducted 2026-04-22 for Stanford CS440LX tt-xla project.*
*Sources: ttnn.topk API (help output on device), tt-metal DeepSeek V3 implementation (models/demos/deepseek_v3/tt/), experiment 82 (topk benchmarks), experiment 91 (current MoE decode).*
