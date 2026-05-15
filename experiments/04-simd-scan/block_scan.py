"""Step 1D-2: block-level prefix sum (1024 elements per threadgroup).

Uses block_scan.metal.
"""

import numpy as np
import mlx.core as mx

from mamba_metal import load_kernel


TG_SIZE = 1024


k_block_scan = load_kernel(
    name="block_scan",
    input_names=["in_", "n"],
    output_names=["out"],
)


def run_block_scan(a: mx.array) -> mx.array:
    n = a.size
    assert n <= TG_SIZE
    (out,) = k_block_scan(
        inputs=[a, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(TG_SIZE, 1, 1),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return out


def main() -> None:
    for n in [32, 64, 256, 1000, 1024]:
        rng = np.random.default_rng(seed=n)
        a_np = rng.standard_normal(n).astype(np.float32)
        a = mx.array(a_np)
        out = run_block_scan(a)
        mx.eval(out)
        ref = a_np.cumsum()
        err = np.max(np.abs(np.array(out) - ref))
        rel_err = np.max(np.abs(np.array(out) - ref) / (np.abs(ref) + 1e-6))
        print(f"n = {n:>5}: max abs err = {err:.3e}   rel err = {rel_err:.3e}")


if __name__ == "__main__":
    main()
