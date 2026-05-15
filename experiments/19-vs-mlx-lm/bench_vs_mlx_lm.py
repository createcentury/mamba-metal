"""Step 12: head-to-head — mamba-metal vs mlx-lm Mamba on the same model.

mlx-lm's Mamba uses an explicit Python loop over time
(`for t in range(T): ssm_step(...)`) — no parallel prefix scan kernel.
Our path uses the Metal selective_scan kernel for prefill.

We expect:
  - Decode (T=1): similar or comparable (both are one-step elementwise)
  - Prefill (T=L): we should win on long prompts by ≥ O(L) factor
"""

import time

import mlx.core as mx
from transformers import AutoTokenizer

# ours
from mamba_metal import generate_fast, load_mamba_hf as load_ours

# mlx-lm
from mlx_lm import load as load_mlx_lm
from mlx_lm import generate as generate_mlx_lm


REPO = "state-spaces/mamba-130m-hf"
PROMPT_BASE = "The cat sat on the mat. " * 10  # ~70 tokens per copy


def time_ours(model, tokenizer, prompt, n_new):
    _ = generate_fast(model, tokenizer, prompt, max_new_tokens=2)
    t0 = time.perf_counter()
    out = generate_fast(model, tokenizer, prompt, max_new_tokens=n_new, temperature=0.0)
    return time.perf_counter() - t0, out


def time_mlx_lm(model, tokenizer, prompt, n_new):
    _ = generate_mlx_lm(model, tokenizer, prompt=prompt, max_tokens=2, verbose=False)
    t0 = time.perf_counter()
    out = generate_mlx_lm(model, tokenizer, prompt=prompt, max_tokens=n_new, verbose=False)
    return time.perf_counter() - t0, out


def main() -> None:
    print(f"Loading {REPO} in both stacks ...")
    t0 = time.perf_counter()
    ours_model, _ = load_ours(REPO)
    ours_tok = AutoTokenizer.from_pretrained(REPO)
    print(f"  ours    loaded in {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    mlx_model, mlx_tok = load_mlx_lm(REPO)
    print(f"  mlx-lm  loaded in {time.perf_counter()-t0:.2f}s")

    n_new = 50

    print(f"\nDecode {n_new} tokens at varying prompt length\n")
    print(f"{'prompt tok':>11} {'ours (s)':>10} {'mlx-lm (s)':>12} {'speedup':>10}")
    print("-" * 48)

    for n_repeat in [1, 5, 20, 80]:
        p = PROMPT_BASE * n_repeat
        n_prompt = len(ours_tok(p).input_ids)
        t_ours, _ = time_ours(ours_model, ours_tok, p, n_new)
        t_mlx, _ = time_mlx_lm(mlx_model, mlx_tok, p, n_new)
        speedup = t_mlx / t_ours
        print(f"{n_prompt:>11} {t_ours:>10.2f} {t_mlx:>12.2f}   {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
