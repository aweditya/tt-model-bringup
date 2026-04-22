# Wiki 30: Qwen2.5-0.5B Porting Plan

## Q: What model are we targeting?

**A:** Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B-Instruct`). 490M params, 24 layers, hidden=896, 14 query heads / 2 KV heads (GQA 7:1), head_dim=64. ~980MB in bfloat16 — fits easily in Blackhole's 12GB.

## Q: How does Qwen differ from GPT-2?

| Component | GPT-2 | Qwen2.5 | New Op? |
|-----------|-------|---------|---------|
| Position encoding | Learned embeddings | RoPE (rotary) | **YES** |
| Normalization | LayerNorm | RMSNorm | No — `ttnn.rms_norm` |
| Activation | GELU | SwiGLU | No — `ttnn.swiglu` |
| MLP structure | Linear→GELU→Linear | Gate+Up→SwiGLU→Down | Wiring change |
| Attention | MHA (12 heads) | GQA (14Q / 2KV) | SDPA supports GQA |
| QKV bias | No | Yes | Just add bias |
| Norm placement | Post-attention | Pre-attention | Wiring change |
| Tied embeddings | No | Yes | No new op |

## Q: What new ops do we need?

**A:** Only **RoPE** (rotary position embeddings). Two implementation paths:

1. **Decompose into existing ops:** reshape, slice, mul, add, neg, concatenate. We have all of these.
   ```
   q_rotated = [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos]
   ```

2. **Native TT-NN:** `ttnn.transformer.rotary_embedding` may exist — test on device.

## Q: What's the Qwen layer structure?

```
Input x (1, T, 896)
  ├─ RMSNorm
  ├─ Q = W_q @ x + b_q → (1, T, 896) → reshape to (1, 14, T, 64)
  ├─ K = W_k @ x + b_k → (1, T, 128) → reshape to (1, 2, T, 64)
  ├─ V = W_v @ x + b_v → (1, T, 128) → reshape to (1, 2, T, 64)
  ├─ Apply RoPE to Q, K
  ├─ GQA attention (14 Q heads, 2 KV heads — each KV head serves 7 Q heads)
  ├─ Output projection + residual
  ├─ RMSNorm
  ├─ gate = W_gate @ x → (1, T, 4864)
  ├─ up = W_up @ x → (1, T, 4864)
  ├─ SwiGLU(gate, up) → (1, T, 4864)
  ├─ down = W_down @ SwiGLU → (1, T, 896)
  └─ Residual
```

## Q: What's the porting plan?

**Phase 1:** Validate ops �� test `ttnn.rms_norm`, `ttnn.swiglu`, `ttnn.silu`, RoPE on Qwen-shaped tensors.

**Phase 2:** Single Qwen layer — load one block's weights, wire everything up, verify vs HuggingFace reference.

**Phase 3:** Full 24-layer model — stack layers, check logits match (cosine > 0.999).

**Phase 4:** KV-cached decode — adapt experiment 35. GQA makes this cheaper (only 2 KV heads per layer → tiny cache).

**Phase 5:** Text generation — tokenizer via `transformers.AutoTokenizer`, autoregressive loop.

## Q: Memory budget?

| Component | Size |
|-----------|------|
| Parameters (bf16) | 980 MB |
| KV cache (2048 context) | 25 MB |
| Activations (batch=1) | ~10 MB |
| **Total** | **~1 GB / 12 GB** |

---

*Based on Qwen2 technical report, HuggingFace model card, and TT-Metal existing implementations.*
