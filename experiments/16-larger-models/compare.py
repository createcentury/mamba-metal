"""Step 10: compare generation across Mamba model sizes.

Loads each model in turn (releasing the previous) and generates from the
same prompt. Measures wall time per token and prints the output.
"""

import gc
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from transformers import AutoTokenizer

from mamba_metal import generate_fast, load_mamba_hf


MODELS = [
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
]

PROMPT = "The capital of Japan is"
MAX_NEW = 40


def run_one(repo: str):
    t0 = time.perf_counter()
    model, cfg = load_mamba_hf(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    load_t = time.perf_counter() - t0

    # Warm up
    _ = generate_fast(model, tokenizer, "warm up", max_new_tokens=2)

    t0 = time.perf_counter()
    out = generate_fast(model, tokenizer, PROMPT, max_new_tokens=MAX_NEW, temperature=0.0)
    gen_t = time.perf_counter() - t0

    params_m = sum(a.size for _, a in tree_flatten(model.parameters())) / 1e6

    del model, tokenizer
    gc.collect()
    if hasattr(mx, "metal"):
        mx.metal.clear_cache()
    return load_t, gen_t, out, params_m


def main() -> None:
    print(f"Prompt: {PROMPT!r}")
    print(f"max_new_tokens: {MAX_NEW}\n")

    results = []
    for repo in MODELS:
        print(f"=== {repo} ===")
        try:
            load_t, gen_t, out, params_m = run_one(repo)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            continue
        tps = MAX_NEW / gen_t
        ms_per_tok = gen_t * 1000 / MAX_NEW
        print(f"  params:    {params_m:.0f} M")
        print(f"  load:      {load_t:.2f} s")
        print(f"  generate:  {gen_t:.2f} s  ({tps:.1f} tok/s, {ms_per_tok:.1f} ms/tok)")
        print(f"  output:    {out}\n")
        results.append((repo.split("/")[-1], params_m, gen_t, tps, out))

    if results:
        print("\nSummary")
        print(f"{'model':<22} {'params':>8} {'gen (s)':>10} {'tok/s':>8}")
        print("-" * 52)
        for name, params_m, gen_t, tps, _ in results:
            print(f"{name:<22} {params_m:>6.0f} M {gen_t:>10.2f} {tps:>8.1f}")


if __name__ == "__main__":
    main()
