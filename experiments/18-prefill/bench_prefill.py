"""Step 11: parallel prefill — does it match the old step-by-step prefill,
and how much faster is it on long prompts?"""

import time

import mlx.core as mx
from transformers import AutoTokenizer

from mamba_metal import generate, generate_fast, load_mamba_hf


REPO = "state-spaces/mamba-130m-hf"


def main() -> None:
    print(f"Loading {REPO} ...")
    model, _ = load_mamba_hf(REPO)
    tokenizer = AutoTokenizer.from_pretrained(REPO)

    # Correctness: greedy output should still match O(L^2) baseline
    print("\n=== Correctness ===")
    for prompt in ["Mamba is a", "Once upon a time", "The capital of Japan is"]:
        slow = generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.0)
        fast = generate_fast(model, tokenizer, prompt, max_new_tokens=40, temperature=0.0)
        marker = "✓" if slow == fast else "✗"
        print(f"  {marker} {prompt!r}: identical={slow == fast}")

    # Prefill speedup at growing prompt lengths
    base = "The cat sat on the mat. " * 10  # ~70 tokens per copy
    print("\n=== generate_fast: prefill+decode wall time vs prompt length ===")
    print(f"{'prompt tok':>11} {'wall (s)':>10} {'decode 50 ms/tok':>20}")
    print("-" * 44)
    for n_repeat in [1, 5, 20, 80]:
        p = base * n_repeat
        n_prompt = len(tokenizer(p).input_ids)
        # Warm up
        _ = generate_fast(model, tokenizer, p, max_new_tokens=2)
        t0 = time.perf_counter()
        _ = generate_fast(model, tokenizer, p, max_new_tokens=50, temperature=0.0)
        sec = time.perf_counter() - t0
        # We bake prefill into total; ms/decoded includes amortised prefill cost.
        # Per-decode-token cost AFTER prefill should be ~7 ms regardless of prompt length.
        print(f"{n_prompt:>11} {sec:>10.2f}   {sec*1000/50:>18.2f}")


if __name__ == "__main__":
    main()
