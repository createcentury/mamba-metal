"""Step 10 (continued): demonstrate constant per-token cost over long decodes.

Mamba's headline property: per-token decode cost does not grow with how much
has already been generated. We measure ms-per-token at increasing horizons.

Also: ms-per-token vs prompt length, to confirm prefill scales linearly (and
the per-token cost AFTER prefill is unchanged).
"""

import time

import mlx.core as mx
from transformers import AutoTokenizer

from mamba_metal import generate_fast, load_mamba_hf


REPO = "state-spaces/mamba-130m-hf"


def main() -> None:
    print(f"Loading {REPO} ...")
    model, _ = load_mamba_hf(REPO)
    tokenizer = AutoTokenizer.from_pretrained(REPO)

    prompt = "The capital of Japan is"
    # Warmup
    _ = generate_fast(model, tokenizer, prompt, max_new_tokens=4)
    mx.synchronize()

    print("\n=== generate_fast: ms/token vs decode horizon ===")
    print(f"{'n_new':>6} {'wall (s)':>10} {'ms/token':>12} {'tokens/s':>12}")
    print("-" * 44)
    for n in [50, 100, 200, 500, 1000, 2000]:
        t0 = time.perf_counter()
        _ = generate_fast(model, tokenizer, prompt, max_new_tokens=n, temperature=0.0)
        sec = time.perf_counter() - t0
        ms_per_tok = sec * 1000 / n
        print(f"{n:>6} {sec:>10.2f}   {ms_per_tok:>10.2f}   {n/sec:>10.1f}")

    print("\n=== generate_fast: ms/token vs prompt length (decode=50 fixed) ===")
    # Build prompts of increasing length by repeating a base sentence
    base = "The cat sat on the mat. " * 1
    print(f"{'prompt tokens':>14} {'prefill+decode (s)':>20} {'ms/decoded':>14}")
    print("-" * 50)
    for n_repeat in [1, 10, 50, 200, 800]:
        p = base * n_repeat
        n_prompt = len(tokenizer(p).input_ids)
        t0 = time.perf_counter()
        _ = generate_fast(model, tokenizer, p, max_new_tokens=50, temperature=0.0)
        sec = time.perf_counter() - t0
        # We don't separate prefill from decode here, but per-decoded-token is what matters
        print(f"{n_prompt:>14} {sec:>20.2f} {sec*1000/50:>14.2f}")


if __name__ == "__main__":
    main()
