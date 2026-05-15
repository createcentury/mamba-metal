"""Step 10: verify O(L) incremental decoding matches O(L²) full forward.

For a short prompt + a few greedy generated tokens, the two paths should produce
identical token sequences (greedy is deterministic; only numerical noise in
softmax argmax can cause divergence late on).
"""

import time

import mlx.core as mx
from transformers import AutoTokenizer

from mamba_metal import generate, generate_fast, load_mamba_hf


REPO = "state-spaces/mamba-130m-hf"
PROMPTS = [
    "Mamba is a",
    "Once upon a time",
    "The capital of Japan is",
]


def main() -> None:
    print(f"Loading {REPO} ...")
    model, _ = load_mamba_hf(REPO)
    tokenizer = AutoTokenizer.from_pretrained(REPO)

    n = 40

    # Correctness — greedy must match
    print("\n=== Correctness: O(L^2) vs O(L) greedy ===")
    for prompt in PROMPTS:
        slow = generate(model, tokenizer, prompt, max_new_tokens=n, temperature=0.0)
        fast = generate_fast(model, tokenizer, prompt, max_new_tokens=n, temperature=0.0)
        match = slow == fast
        marker = "✓" if match else "✗"
        print(f"  {marker} {prompt!r}: greedy output identical = {match}")
        if not match:
            print(f"    slow: {slow!r}")
            print(f"    fast: {fast!r}")

    # Speed comparison at increasing decode lengths
    print("\n=== Speed at varying decode length (prompt fixed) ===")
    prompt = "The capital of Japan is"
    print(f"{'n_new':>6} {'O(L^2) (s)':>12} {'O(L) (s)':>12} {'speedup':>10}")
    print("-" * 44)
    for n_new in [10, 40, 100, 200]:
        # Warm up so kernel-cache effects don't bias the first run
        _ = generate(model, tokenizer, prompt, max_new_tokens=2)
        _ = generate_fast(model, tokenizer, prompt, max_new_tokens=2)

        t0 = time.perf_counter()
        _ = generate(model, tokenizer, prompt, max_new_tokens=n_new, temperature=0.0)
        t_slow = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = generate_fast(model, tokenizer, prompt, max_new_tokens=n_new, temperature=0.0)
        t_fast = time.perf_counter() - t0

        print(f"{n_new:>6} {t_slow:>12.2f} {t_fast:>12.2f} {t_slow/t_fast:>9.2f}x")


if __name__ == "__main__":
    main()
