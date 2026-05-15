"""Step 1B: measure Unified Memory bandwidth.

Uses ``copy_scalar.metal`` and ``copy_vec4.metal``.
"""

import time

import mlx.core as mx

from mamba_metal import load_kernel


copy_scalar = load_kernel(
    name="copy_scalar",
    input_names=["in_", "n"],
    output_names=["out"],
)
copy_vec4 = load_kernel(
    name="copy_vec4",
    input_names=["in_", "n4"],
    output_names=["out"],
)


def run_scalar(a: mx.array) -> mx.array:
    n = a.size
    (out,) = copy_scalar(
        inputs=[a, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def run_vec4(a: mx.array) -> mx.array:
    assert a.size % 4 == 0
    n4 = a.size // 4
    (out,) = copy_vec4(
        inputs=[a, mx.array(n4, dtype=mx.uint32)],
        output_shapes=[a.shape],
        output_dtypes=[a.dtype],
        grid=(n4, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def time_kernel(fn, a: mx.array, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        out = fn(a)
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(a)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    sizes_mb = [1, 4, 16, 64, 256, 512]
    bytes_per_float = 4

    def mlx_add0(x: mx.array) -> mx.array:
        return x + mx.array(0.0, dtype=x.dtype)

    print(f"{'size (MB)':>10} {'kernel':>14} {'time (ms)':>12} {'GB/s (R+W)':>12}")
    print("-" * 52)

    for size_mb in sizes_mb:
        n = (size_mb * 1024 * 1024) // bytes_per_float
        a = mx.random.uniform(shape=(n,), dtype=mx.float32)
        mx.eval(a)
        mx.synchronize()

        iters = max(5, int(500 / size_mb))
        total_bytes = 2 * n * bytes_per_float

        for label, fn in [
            ("scalar", run_scalar),
            ("vec4", run_vec4),
            ("mlx_add0", mlx_add0),
        ]:
            sec = time_kernel(fn, a, iters)
            gbs = total_bytes / sec / 1e9
            print(f"{size_mb:>10} {label:>14} {sec*1e3:>10.3f}   {gbs:>10.1f}")


if __name__ == "__main__":
    main()
