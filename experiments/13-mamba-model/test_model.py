"""Step 9.1: smoke test of MambaModel.

  - Build with mamba-130m config (small n_layer for speed)
  - Random init, forward pass on token IDs
  - Verify parameter naming matches HF expectations (post backbone-prefix strip)
"""

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mamba_metal import MambaConfig, MambaModel


HF_EXPECTED_LAYER_NAMES = {
    "norm.weight",
    "mixer.A_log",
    "mixer.D",
    "mixer.conv1d.bias",
    "mixer.conv1d.weight",
    "mixer.dt_proj.bias",
    "mixer.dt_proj.weight",
    "mixer.in_proj.weight",
    "mixer.out_proj.weight",
    "mixer.x_proj.weight",
}

HF_EXPECTED_TOP_NAMES = {
    "embeddings.weight",
    "norm_f.weight",
}


def collect_params(module) -> list[tuple[str, mx.array]]:
    """Flat (name, array) list for the module's parameter tree."""
    return tree_flatten(module.parameters())


def smoke_test():
    cfg = MambaConfig(
        d_model=128,
        n_layer=2,
        vocab_size=512,
        d_state=16,
        d_conv=4,
        expand=2,
    )
    model = MambaModel(cfg)

    B, L = 1, 32
    rng = np.random.default_rng(0)
    input_ids = mx.array(rng.integers(0, cfg.vocab_size, size=(B, L)))
    logits = model(input_ids)
    mx.eval(logits)
    mx.synchronize()

    print(f"input_ids shape: {input_ids.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"expected: ({B}, {L}, {cfg.vocab_size})")
    assert logits.shape == (B, L, cfg.vocab_size)

    logits_np = np.array(logits)
    print(f"logits range: [{logits_np.min():.3f}, {logits_np.max():.3f}]")
    print(f"any NaN: {np.any(np.isnan(logits_np))}")
    assert not np.any(np.isnan(logits_np))


def check_param_names():
    """Verify our parameter names match HF (after stripping 'backbone.')."""
    cfg = MambaConfig(d_model=64, n_layer=1, vocab_size=128)
    model = MambaModel(cfg)
    flat = collect_params(model)
    names = {n for n, _ in flat}

    print(f"\nTotal parameter tensors: {len(names)}")
    print("\nParam names:")
    for n in sorted(names):
        print(f"  {n}")

    layer_names = {n.removeprefix("layers.0.") for n in names if n.startswith("layers.0.")}
    missing = HF_EXPECTED_LAYER_NAMES - layer_names
    extra = layer_names - HF_EXPECTED_LAYER_NAMES
    print(f"\n  layer-level names match HF: {layer_names == HF_EXPECTED_LAYER_NAMES}")
    if missing: print(f"  MISSING (in HF, not ours): {missing}")
    if extra:   print(f"  EXTRA   (in ours, not HF): {extra}")

    top_names = {n for n in names if "layers." not in n}
    print(f"\n  top-level names: {top_names}")
    print(f"  matches expected: {top_names == HF_EXPECTED_TOP_NAMES}")


def parameter_count_check():
    """Estimate parameter count for mamba-130m config — should be ~130M."""
    cfg = MambaConfig(
        d_model=768,
        n_layer=24,
        vocab_size=50280,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=48,
    )
    model = MambaModel(cfg)
    flat = collect_params(model)
    total = sum(a.size for _, a in flat)
    print(f"\nmamba-130m equivalent param count: {total/1e6:.1f} M")


if __name__ == "__main__":
    print("=== Smoke test ===")
    smoke_test()
    print("\n=== Parameter name verification ===")
    check_param_names()
    print("\n=== Parameter count (130m config) ===")
    parameter_count_check()
