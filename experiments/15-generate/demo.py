"""Step 9.3: text generation with mamba-130m on Metal."""

import sys
import time

from transformers import AutoTokenizer

from mamba_metal import generate, load_mamba_hf


REPO = "state-spaces/mamba-130m-hf"


def main() -> None:
    print(f"Loading model ({REPO}) ...")
    t0 = time.perf_counter()
    model, cfg = load_mamba_hf(REPO)
    print(f"  model loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  params: d_model={cfg.d_model} n_layer={cfg.n_layer} vocab={cfg.vocab_size}")

    print("\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    print(f"  vocab_size={tokenizer.vocab_size}  eos={tokenizer.eos_token_id}")

    prompts = [
        "Mamba is a",
        "Once upon a time",
        "The capital of Japan is",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: {prompt!r} ---")
        print(prompt, end="", flush=True)

        def stream(s: str):
            print(s, end="", flush=True)

        t0 = time.perf_counter()
        _ = generate(
            model, tokenizer, prompt,
            max_new_tokens=40, temperature=0.0,
            on_token=stream,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  [{elapsed:.2f}s total, {40/elapsed:.1f} tokens/s greedy]")


if __name__ == "__main__":
    sys.exit(main())
