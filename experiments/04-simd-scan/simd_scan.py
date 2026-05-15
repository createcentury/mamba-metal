"""Step 1D-1: SIMD-group level prefix sum.

Uses simd_scan_builtin.metal (calls simd_prefix_inclusive_sum) and
simd_scan_handrolled.metal (Hillis-Steele).
"""

import numpy as np
import mlx.core as mx

from mamba_metal import load_kernel


SIMD_W = 32


k_builtin = load_kernel(
    name="simd_scan_builtin",
    input_names=["in_", "n"],
    output_names=["out"],
)

k_handrolled = load_kernel(
    name="simd_scan_handrolled",
    input_names=["in_", "n"],
    output_names=["out"],
)


def run(kernel, a: mx.array) -> mx.array:
    n = a.size
    (out,) = kernel(
        inputs=[a, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(SIMD_W, 1, 1),
    )
    return out


def expected_simd_scan(a: np.ndarray, simd_w: int = SIMD_W) -> np.ndarray:
    return a.reshape(-1, simd_w).cumsum(axis=1).reshape(a.size)


def main() -> None:
    n = 4 * SIMD_W
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal(n).astype(np.float32)
    a = mx.array(a_np)

    out_b = run(k_builtin, a)
    out_h = run(k_handrolled, a)
    mx.eval(out_b, out_h)

    ref = expected_simd_scan(a_np)
    err_b = np.max(np.abs(np.array(out_b) - ref))
    err_h = np.max(np.abs(np.array(out_h) - ref))

    print(f"n = {n}, SIMD width = {SIMD_W}")
    print(f"builtin    max abs err: {err_b:.3e}")
    print(f"handrolled max abs err: {err_h:.3e}")
    assert err_b < 1e-4 and err_h < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
