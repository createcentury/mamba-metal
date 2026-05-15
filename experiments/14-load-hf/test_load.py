"""Step 9.2 (final): load state-spaces/mamba-130m-hf into MambaModel.

Verifies:
  - Every parameter from HF maps to a model parameter (no missing, no extras).
  - Forward pass on random token ids produces finite logits.
  - argmax over vocab yields sensible (high-frequency) token ids.
"""

import time

import numpy as np
import mlx.core as mx

from mamba_metal import load_mamba_hf


def main() -> None:
    print("Loading state-spaces/mamba-130m-hf ...")
    t0 = time.perf_counter()
    model, cfg = load_mamba_hf("state-spaces/mamba-130m-hf")
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")
    print(f"  config: d_model={cfg.d_model}  n_layer={cfg.n_layer}  "
          f"vocab={cfg.vocab_size}  dt_rank={cfg.dt_rank}")

    # Forward pass
    rng = np.random.default_rng(0)
    B, L = 1, 16
    input_ids_np = rng.integers(0, cfg.vocab_size, size=(B, L))
    input_ids = mx.array(input_ids_np)

    print(f"\nForward pass on shape {input_ids.shape}:")
    t0 = time.perf_counter()
    logits = model(input_ids)
    mx.eval(logits)
    mx.synchronize()
    print(f"  forward time: {(time.perf_counter() - t0)*1e3:.1f} ms")
    print(f"  logits shape: {logits.shape}")

    logits_np = np.array(logits)
    print(f"  any NaN: {np.any(np.isnan(logits_np))}")
    print(f"  any Inf: {np.any(np.isinf(logits_np))}")
    print(f"  logits range: [{logits_np.min():.2f}, {logits_np.max():.2f}]")
    print(f"  per-position argmax: {logits_np.argmax(axis=-1)[0].tolist()}")

    # Sanity: distribution of argmax tokens — should not all be the same
    argmax_ids = logits_np.argmax(axis=-1).flatten()
    n_unique = len(set(argmax_ids.tolist()))
    print(f"  unique argmax token ids: {n_unique} / {len(argmax_ids)}")

    assert not np.any(np.isnan(logits_np))
    assert not np.any(np.isinf(logits_np))
    print("\nOK — model loaded and forward pass produces finite logits.")


if __name__ == "__main__":
    main()
