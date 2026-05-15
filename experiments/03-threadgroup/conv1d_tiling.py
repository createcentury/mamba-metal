"""Step 1C: threadgroup memory benefit via 1D convolution (stencil).

Workload: out[i] = sum over k in [0, K) of in[i + k]

Without threadgroup memory: each in[j] is loaded K times from global memory.
With threadgroup memory: each tile is loaded once, then reused K times from SRAM.

Expected: K-fold reduction in global memory traffic → higher GB/s seen at the
out array level when K is large.

This is the same reuse pattern as Mamba's selective scan: for each state_idx
(0..N-1), the kernel re-reads delta_u_vals etc. from the chunk. Using
threadgroup memory means those reads hit SRAM, not DRAM.
"""

import time
from dataclasses import dataclass

import mlx.core as mx


# Both kernels parameterize K via a constant we substitute into the source.
# (Function constants would be cleaner, but template-style substitution is
# sufficient for this experiment.)


def make_global_kernel(K: int):
    src = f"""
        uint i = thread_position_in_grid.x;
        if (i >= n_out) return;
        float acc = 0.0;
        for (uint k = 0; k < {K}u; ++k) {{
            acc += in_[i + k];
        }}
        out[i] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"conv1d_global_K{K}",
        input_names=["in_", "n_out"],
        output_names=["out"],
        source=src,
    )


def make_tg_kernel(K: int, tg_size: int = 256):
    tile_size = tg_size + K - 1
    src = f"""
        uint local_i = thread_position_in_threadgroup.x;
        uint tg_id = threadgroup_position_in_grid.x;
        uint base = tg_id * {tg_size}u;
        uint i = base + local_i;

        threadgroup float tile[{tile_size}];

        // Each thread loads one element of the main tile.
        if (base + local_i < n_in) {{
            tile[local_i] = in_[base + local_i];
        }} else {{
            tile[local_i] = 0.0;
        }}
        // The first K-1 threads also load the halo at the end.
        if (local_i < {K - 1}u) {{
            uint halo_idx = base + {tg_size}u + local_i;
            tile[{tg_size}u + local_i] = (halo_idx < n_in) ? in_[halo_idx] : 0.0;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i >= n_out) return;
        float acc = 0.0;
        for (uint k = 0; k < {K}u; ++k) {{
            acc += tile[local_i + k];
        }}
        out[i] = acc;
    """
    return mx.fast.metal_kernel(
        name=f"conv1d_tg_K{K}",
        input_names=["in_", "n_in", "n_out"],
        output_names=["out"],
        source=src,
    )


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


@dataclass
class Row:
    K: int
    label: str
    time_ms: float
    out_gbs: float  # output-volume bandwidth (real work) — useful comparison
    in_gbs_naive: float  # what global-read traffic implies


def main() -> None:
    size_mb = 128
    bytes_per_float = 4
    n = (size_mb * 1024 * 1024) // bytes_per_float
    a = mx.random.uniform(shape=(n,), dtype=mx.float32)
    mx.eval(a)
    mx.synchronize()

    print(f"input size: {size_mb} MB ({n} elements)")
    print()
    print(f"{'K':>4} {'kernel':>8} {'time (ms)':>11} {'effective GB/s':>17} {'speedup':>10}")
    print("-" * 56)

    for K in [1, 4, 16, 64]:
        kg = make_global_kernel(K)
        kt = make_tg_kernel(K)

        n_out = n - K + 1
        # Effective bandwidth: bytes the algorithm logically touches per call.
        # For conv1d: read n_in + write n_out ≈ 2n bytes_per_float
        logical_bytes = (n + n_out) * bytes_per_float

        t_global = time_call(lambda: run_global(kg, a, K), iters=20)
        t_tg = time_call(lambda: run_tg(kt, a, K), iters=20)

        gbs_global = logical_bytes / t_global / 1e9
        gbs_tg = logical_bytes / t_tg / 1e9
        speedup = t_global / t_tg

        print(f"{K:>4} {'global':>8} {t_global*1e3:>9.3f}   {gbs_global:>15.1f}   {'-':>8}")
        print(f"{K:>4} {'tg':>8} {t_tg*1e3:>9.3f}   {gbs_tg:>15.1f}   {speedup:>8.2f}x")

    # Sanity check: outputs should match
    print()
    print("Sanity check (K=16):")
    kg = make_global_kernel(16)
    kt = make_tg_kernel(16)
    og = run_global(kg, a, 16)
    ot = run_tg(kt, a, 16)
    mx.eval(og, ot)
    max_diff = mx.max(mx.abs(og - ot)).item()
    print(f"  max abs diff = {max_diff:.3e}")


if __name__ == "__main__":
    main()
