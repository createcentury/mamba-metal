"""Step 1C: threadgroup-memory tiling vs naive global-memory access.

Uses parametric kernels conv1d_global.metal and conv1d_tg.metal with K
substituted at load time.
"""

import time

import mlx.core as mx

from mamba_metal import load_kernel


def make_kernels(K: int):
    kg = load_kernel(
        name="conv1d_global",
        input_names=["in_", "n_out"],
        output_names=["out"],
        params={"K": K},
    )
    kt = load_kernel(
        name="conv1d_tg",
        input_names=["in_", "n_in", "n_out"],
        output_names=["out"],
        params={"K": K, "K_minus_1": K - 1, "tile_size": 256 + K - 1},
    )
    return kg, kt


def run_global(kernel, a: mx.array, K: int) -> mx.array:
    n_out = a.size - K + 1
    (out,) = kernel(
        inputs=[a, mx.array(n_out, dtype=mx.uint32)],
        output_shapes=[(n_out,)],
        output_dtypes=[a.dtype],
        grid=(((n_out + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def run_tg(kernel, a: mx.array, K: int) -> mx.array:
    n_in = a.size
    n_out = a.size - K + 1
    (out,) = kernel(
        inputs=[
            a,
            mx.array(n_in, dtype=mx.uint32),
            mx.array(n_out, dtype=mx.uint32),
        ],
        output_shapes=[(n_out,)],
        output_dtypes=[a.dtype],
        grid=(((n_out + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def time_call(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    size_mb = 128
    bytes_per_float = 4
    n = (size_mb * 1024 * 1024) // bytes_per_float
    a = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(a)
    mx.synchronize()

    print(f"input size: {size_mb} MB ({n} elements)\n")
    print(f"{'K':>4} {'kernel':>8} {'time (ms)':>11} {'effective GB/s':>17} {'speedup':>10}")
    print("-" * 56)

    for K in [1, 4, 16, 64]:
        kg, kt = make_kernels(K)
        n_out = n - K + 1
        logical_bytes = (n + n_out) * bytes_per_float

        t_global = time_call(lambda: run_global(kg, a, K), iters=20)
        t_tg = time_call(lambda: run_tg(kt, a, K), iters=20)

        gbs_global = logical_bytes / t_global / 1e9
        gbs_tg = logical_bytes / t_tg / 1e9
        speedup = t_global / t_tg

        print(f"{K:>4} {'global':>8} {t_global*1e3:>9.3f}   {gbs_global:>15.1f}   {'-':>8}")
        print(f"{K:>4} {'tg':>8} {t_tg*1e3:>9.3f}   {gbs_tg:>15.1f}   {speedup:>8.2f}x")

    # Sanity: outputs match
    print()
    kg, kt = make_kernels(16)
    og, ot = run_global(kg, a, 16), run_tg(kt, a, 16)
    mx.eval(og, ot)
    max_diff = mx.max(mx.abs(og - ot)).item()
    print(f"sanity (K=16): max abs diff between global and tg = {max_diff:.3e}")


if __name__ == "__main__":
    main()
