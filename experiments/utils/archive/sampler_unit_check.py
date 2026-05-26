#!/usr/bin/env python3
"""Unit checks for _sample_next_id (server.py).

Runs OFF the device (pure numpy) — verifies:
  1. Default args reproduce greedy argmax bit-for-bit.
  2. temperature=0 with repetition_penalty>1 still deterministic.
  3. Repetition penalty actually re-ranks recently-seen ids downward.
  4. min-p truncates low-prob tokens.
  5. top-p still works as before.
  6. Combined penalties compose without NaNs.

No device, no ttnn, no server bring-up. Run locally to gate the server restart.
"""
import importlib.util
import numpy as np
import os
import sys


def _load_server_module():
    path = os.path.join(os.path.dirname(__file__), "..", "serve", "server.py")
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_server_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    # server.py imports a LOT of device code at import time. We can't actually
    # exec the whole module. Instead, copy out _sample_next_id by hand-parsing
    # is brittle — better: use ast to extract the function, exec into a local ns.
    import ast
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_sample_next_id":
            # Wrap into a module body with just the function and `import numpy as np`
            new_mod = ast.Module(
                body=[
                    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                    node,
                ],
                type_ignores=[],
            )
            new_mod = ast.fix_missing_locations(new_mod)
            code = compile(new_mod, path, "exec")
            ns = {}
            exec(code, ns)
            return ns["_sample_next_id"]
    raise RuntimeError("could not find _sample_next_id in server.py")


def main():
    _sample_next_id = _load_server_module()
    rng = np.random.default_rng(7)

    # Construct a small logits vector with a clear winner.
    V = 100
    logits = np.zeros(V, dtype=np.float32)
    logits[42] = 5.0
    logits[7]  = 4.5
    logits[99] = 4.0
    logits[1]  = 3.0

    # 1. Default args == argmax
    out = _sample_next_id(logits, 0.0, 1.0, rng)
    assert out == 42, f"default greedy expected 42 got {out}"
    print("PASS  default greedy → 42")

    # 2. Repetition penalty downweights recent ids
    # If 42 is "recently generated", penalty 1.5 → logit[42] becomes 5.0/1.5 = 3.33
    # New argmax should be 7 (which still has 4.5).
    out = _sample_next_id(logits, 0.0, 1.0, rng,
                          repetition_penalty=1.5, recent_ids=[42])
    assert out == 7, f"rep penalty 1.5 on id 42 expected 7 got {out}"
    print("PASS  repetition_penalty 1.5 on [42] → 7")

    # 3. Repetition penalty on multiple ids
    out = _sample_next_id(logits, 0.0, 1.0, rng,
                          repetition_penalty=1.5, recent_ids=[42, 7])
    # 42→3.33, 7→3.0, 99 stays at 4.0 → argmax should be 99
    assert out == 99, f"rep penalty on [42,7] expected 99 got {out}"
    print("PASS  repetition_penalty 1.5 on [42,7] → 99")

    # 4. Negative logits get multiplied (CTRL convention)
    logits_neg = np.zeros(V, dtype=np.float32)
    logits_neg[5] = 1.0
    logits_neg[6] = -2.0
    # No penalty: argmax = 5
    out = _sample_next_id(logits_neg, 0.0, 1.0, rng)
    assert out == 5, f"unpenalized negative case got {out}"
    # Penalty 2.0 on id 5: 5→0.5, 6 untouched. Argmax = 5 still (0.5 > -2.0).
    out = _sample_next_id(logits_neg, 0.0, 1.0, rng,
                          repetition_penalty=2.0, recent_ids=[5])
    assert out == 5
    # Penalty 2.0 on id 6 too: 6 → -4.0 (more negative). Argmax should be 5.
    out = _sample_next_id(logits_neg, 0.0, 1.0, rng,
                          repetition_penalty=2.0, recent_ids=[5, 6])
    assert out == 5
    print("PASS  CTRL negative-logit multiply convention works")

    # 5. min-p truncation at temperature>0 — sample many times, verify only top tokens hit
    # With min_p=0.9, only tokens with prob >= 0.9*max_prob survive. At temp=1, after
    # softmax on our logits the runners-up are well below 0.9*max, so only 42 survives.
    sampled = set()
    for _ in range(200):
        sampled.add(_sample_next_id(logits, 1.0, 1.0, rng, min_p=0.9))
    assert sampled == {42}, f"min_p=0.9 should give only 42, got {sampled}"
    print(f"PASS  min_p=0.9 truncates to top token (got {sampled})")

    # min_p=0.5 should let through 42 AND 7 (probs ratio ~0.6 vs max), but not 99 or 1.
    sampled_5 = set()
    for _ in range(500):
        sampled_5.add(_sample_next_id(logits, 1.0, 1.0, rng, min_p=0.5))
    assert sampled_5.issubset({42, 7}), f"min_p=0.5 should restrict to {{42,7}}, got {sampled_5}"
    assert 42 in sampled_5 and 7 in sampled_5, f"both 42 and 7 should appear, got {sampled_5}"
    print(f"PASS  min_p=0.5 → {{42,7}} only (got {sampled_5})")

    # 6. top-p still works
    # Make logits a bit more spread so a few tokens carry the mass.
    spread = np.linspace(0, 5, V)  # 0..5 increasing; argmax=V-1
    rng2 = np.random.default_rng(13)
    sampled2 = set()
    for _ in range(300):
        sampled2.add(_sample_next_id(spread, 1.0, 0.5, rng2))
    # With top_p=0.5 we should see only the upper tail
    assert min(sampled2) >= V - 20, f"top_p=0.5 should restrict to top tokens, got min={min(sampled2)}"
    print(f"PASS  top_p=0.5 restricts to upper tail (min seen={min(sampled2)})")

    # 7. Combined: rep penalty + min-p + top-p + temperature — no crashes
    out = _sample_next_id(logits, 0.7, 0.9, np.random.default_rng(0),
                          repetition_penalty=1.15, min_p=0.05,
                          recent_ids=[42, 42, 42, 7])
    assert 0 <= out < V
    print(f"PASS  combined sampler produces valid id ({out})")

    # 8. min_p+top_p at temperature=0 (deterministic truncated argmax)
    out = _sample_next_id(logits, 0.0, 0.9, rng,
                          min_p=0.05, recent_ids=[42], repetition_penalty=1.5)
    # 42 is penalized to 3.33, 7 wins at 4.5
    assert out == 7
    print(f"PASS  rep+min_p+top_p at temp=0 → deterministic ({out})")

    # 9. no_repeat_ngram_size — bans tokens completing a recent n-gram
    # History: ... 7, 42, 99 — the (n-1)=2 suffix is [42, 99].
    # If history contains an earlier "42 99 X", X should be banned.
    # Construct: history = [42, 99, 7, ...stuff..., 42, 99]  (suffix at end is [42, 99])
    # Earlier occurrence of [42, 99] is followed by token 7. So 7 should be banned.
    history9 = [42, 99, 7, 1, 2, 3, 42, 99]
    # With suffix [42, 99], earlier match is at idx 0; next token is history9[2] = 7.
    # So 7 should be banned. argmax over logits should NOT be 7.
    # Default logits: 42=5, 7=4.5, 99=4, 1=3, others 0. argmax = 42 (not in banned).
    out = _sample_next_id(logits, 0.0, 1.0, rng,
                          no_repeat_ngram_size=3,
                          full_history_ids=history9)
    assert out == 42, f"n-gram should leave 42 best, got {out}"

    # Now make 7 the unique winner: zero out 42, 99, 1.
    logits9 = np.zeros(V, dtype=np.float32)
    logits9[7] = 10.0  # would-be greedy winner
    logits9[3] = 1.0   # safe runner-up
    out = _sample_next_id(logits9, 0.0, 1.0, rng,
                          no_repeat_ngram_size=3,
                          full_history_ids=history9)
    assert out == 3, f"n-gram of 3 should ban 7 → fall through to 3, got {out}"
    print(f"PASS  no_repeat_ngram_size=3 bans completion-of-recent-2-gram (got {out})")

    # n_repeat=2: the 1-gram suffix is just [99]. Earlier 99 is at idx 1, followed
    # by 7. So 7 banned. (Note: with n=2, banning is aggressive — basically banning
    # the immediate-next-token after seeing X earlier in history.)
    out = _sample_next_id(logits9, 0.0, 1.0, rng,
                          no_repeat_ngram_size=2,
                          full_history_ids=history9)
    assert out == 3, f"n=2 should also ban 7, got {out}"
    print(f"PASS  no_repeat_ngram_size=2 bans completion of suffix [99] → {out}")

    # 10. DRY sampler — penalize tokens extending a long repeat.
    # History: 1 2 3 4 1 2 3, suffix = [1,2,3]. Token 4 would extend to [1,2,3,4]
    # which matches the early prefix. So token 4 should get a penalty.
    history10 = [1, 2, 3, 4, 1, 2, 3]
    # Without DRY: greedy of these logits is whichever has highest.
    logits10 = np.zeros(V, dtype=np.float32)
    logits10[4] = 5.0   # current greedy
    logits10[5] = 4.0   # runner-up
    out_no_dry = _sample_next_id(logits10, 0.0, 1.0, rng,
                                  full_history_ids=history10)
    assert out_no_dry == 4
    # With heavy DRY: token 4 gets penalty ~ mult * base^(L-allowed) where L is
    # match length (we matched [1,2,3]→len=3 against earlier [1,2,3], so picking
    # 4 makes match L=4; allowed=2; extension=2 → penalty = mult * 1.75^2 ≈ 3.06*mult).
    # With mult=5 → penalty ~15 → logits[4] drops to 5-15 = -10, runner-up 5 wins.
    out_dry = _sample_next_id(logits10, 0.0, 1.0, rng,
                               dry_multiplier=5.0,
                               dry_base=1.75,
                               dry_allowed_length=2,
                               full_history_ids=history10)
    assert out_dry == 5, f"DRY should ban id 4 → pick 5, got {out_dry}"
    print(f"PASS  DRY mult=5 penalizes 1,2,3→4 extension (picked {out_dry} not 4)")

    # 11. DRY does nothing if multiplier=0
    out_default = _sample_next_id(logits10, 0.0, 1.0, rng,
                                   dry_multiplier=0.0,
                                   full_history_ids=history10)
    assert out_default == 4
    print(f"PASS  DRY mult=0 is no-op (got {out_default})")

    # 12. full_history_ids defaults to recent_ids when not given
    out_fallback = _sample_next_id(logits9, 0.0, 1.0, rng,
                                    no_repeat_ngram_size=3,
                                    recent_ids=history9)
    # Should match the explicit-full-history case.
    assert out_fallback == 3
    print(f"PASS  full_history_ids defaults to recent_ids ({out_fallback})")

    # 13. Defaults unchanged: no new arg passed → identical to before
    out_default2 = _sample_next_id(logits, 0.0, 1.0, rng)
    assert out_default2 == 42
    print(f"PASS  no new flags → still greedy 42")

    print("\nALL OK — _sample_next_id semantics verified.")


if __name__ == "__main__":
    main()
