"""Step 1A: vector add via mx.fast.metal_kernel — toolchain smoke test."""

import mlx.core as mx


vector_add_source = """
    uint i = thread_position_in_grid.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
"""


kernel = mx.fast.metal_kernel(
    name="vector_add",
    input_names=["a", "b", "n"],
    output_names=["out"],
    source=vector_add_source,
)


def vector_add(a: mx.array, b: mx.array) -> mx.array:
    assert a.shape == b.shape and a.dtype == b.dtype
    n = a.size
    (out,) = kernel(
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
    assert max_err < 1e-6, "vector_add diverged from a + b"
    print("OK")


if __name__ == "__main__":
    main()
