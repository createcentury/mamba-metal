"""Step 1D-1: SIMD-group level prefix sum.

Verify the basic primitive that we'll build block-level scan on top of.

Two variants:
- builtin: simd_prefix_inclusive_sum
- handrolled: Hillis-Steele with simd_shuffle_up

Both should produce the same result as numpy cumsum within each 32-lane SIMD group.
"""

import numpy as np
import mlx.core as mx


SIMD_W = 32


# Builtin: each thread holds one element, calls simd_prefix_inclusive_sum.
BUILTIN_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n) return;
    float v = in_[i];
    v = simd_prefix_inclusive_sum(v);
    out[i] = v;
"""

# Handrolled Hillis-Steele scan over a SIMD group using simd_shuffle_up.
# After log2(32) = 5 iterations, each lane holds the inclusive prefix sum.
HANDROLLED_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= n) return;
    uint lane = thread_index_in_simdgroup;
    float v = in_[i];

    // Hillis-Steele step
    float u;
    u = simd_shuffle_up(v, 1u);  if (lane >= 1u)  v += u;
    u = simd_shuffle_up(v, 2u);  if (lane >= 2u)  v += u;
    u = simd_shuffle_up(v, 4u);  if (lane >= 4u)  v += u;
    u = simd_shuffle_up(v, 8u);  if (lane >= 8u)  v += u;
    u = simd_shuffle_up(v, 16u); if (lane >= 16u) v += u;

    out[i] = v;
"""


k_builtin = mx.fast.metal_kernel(
    name="simd_scan_builtin",
    input_names=["in_", "n"],
    output_names=["out"],
    source=BUILTIN_SRC,
)

k_handrolled = mx.fast.metal_kernel(
    name="simd_scan_handrolled",
    input_names=["in_", "n"],
    output_names=["out"],
    source=HANDROLLED_SRC,
)


def run(kernel, a: mx.array) -> mx.array:
    n = a.size
    (out,) = kernel(
        inputs=[a, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(SIMD_W, 1, 1),  # one SIMD group per threadgroup
    )
    return out


def expected_simd_scan(a: np.ndarray, simd_w: int = SIMD_W) -> np.ndarray:
    """Inclusive prefix sum *within each contiguous block of `simd_w` lanes*."""
    n = a.size
    assert n % simd_w == 0
    return a.reshape(-1, simd_w).cumsum(axis=1).reshape(n)


def main() -> None:
    n_groups = 4
    n = n_groups * SIMD_W
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
    print(f"builtin   max abs err vs numpy cumsum/SIMD: {err_b:.3e}")
    print(f"handrolled max abs err vs numpy cumsum/SIMD: {err_h:.3e}")

    print()
    print("Sample (first SIMD group):")
    print(f"  input: {a_np[:8].round(3)} ...")
    print(f"  builtin    output: {np.array(out_b)[:8].round(3)} ...")
    print(f"  handrolled output: {np.array(out_h)[:8].round(3)} ...")
    print(f"  numpy ref:         {ref[:8].round(3)} ...")

    assert err_b < 1e-4 and err_h < 1e-4, "scan mismatch"
    print("\nOK — both kernels match numpy reference within float32 noise.")


if __name__ == "__main__":
    main()
