"""Step 1A: vector add — toolchain smoke test.

Uses ``mamba_metal.kernels.vector_add.metal`` (loaded at import time).
"""

import mlx.core as mx

from mamba_metal import load_kernel


_kernel = load_kernel(
    name="vector_add",
    input_names=["a", "b", "n"],
    output_names=["out"],
)


def vector_add(a: mx.array, b: mx.array) -> mx.array:
    assert a.shape == b.shape and a.dtype == b.dtype
    n = a.size
    (out,) = _kernel(
        inputs=[a, b, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(min(n, 256), 1, 1),
    )
    return out


def main() -> None:
    n = 1 << 20
    a = mx.random.uniform(shape=(n,))
    b = mx.random.uniform(shape=(n,))

    out = vector_add(a, b)
    ref = a + b

    mx.eval(out, ref)
    max_err = mx.max(mx.abs(out - ref)).item()
    print(f"n = {n}")
    print(f"max abs error = {max_err:.3e}")
    assert max_err < 1e-6
    print("OK")


if __name__ == "__main__":
    main()
