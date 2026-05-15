"""Step 7: benchmark the Metal selective scan.

Goals:
  - Measure tokens/sec throughput at various seqlen.
  - Verify near-linear scaling in seqlen (Mamba's headline property).
  - Compare with the numpy reference at a small size (sanity / scale-of-magnitude).
"""

import time

import numpy as np
import mlx.core as mx

from mamba_metal import selective_scan


def make_inputs(batch: int, dim: int, dstate: int, seqlen: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    delta = rng.uniform(0.01, 0.1, size=(batch, dim, seqlen)).astype(np.float32)
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    return (
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C),
        u, delta, A, B, C,
    )


def time_metal(u, delta, A, B, C, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        y = selective_scan(u, delta, A, B, C)
        mx.eval(y)
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = selective_scan(u, delta, A, B, C)
        mx.eval(y)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


def numpy_reference(u, delta, A, B, C):
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    y = np.zeros_like(u)
    for bi in range(batch):
        for di in range(dim):
            h = np.zeros(dstate, dtype=np.float32)
            for t in range(seqlen):
                a_t = np.exp(delta[bi, di, t] * A[di])
                b_t = delta[bi, di, t] * u[bi, di, t] * B[bi, :, t]
                h = a_t * h + b_t
                y[bi, di, t] = np.dot(h, C[bi, :, t])
    return y


def estimate_flops(batch, dim, dstate, seqlen) -> int:
    # Per (batch, dim, state, token):
    #   exp(delta * A)        ~10 ops (approx)
    #   pair combine in scan: ~4 ops (Hillis-Steele 5 iters * 2 muladd, amortized to ~4)
    #   y += h * C           : 2 ops
    # ~16 ops per (state, token), times batch * dim
    return batch * dim * dstate * seqlen * 16


def fmt_throughput(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    return f"{n / 1e3:.2f} k"


def main() -> None:
    # Fixed model dims, vary seqlen
    batch, dim, dstate = 1, 16, 16

    print(f"Model: batch={batch}  dim={dim}  dstate={dstate}\n")
    print(f"{'seqlen':>8} {'time (ms)':>11} {'tokens/s':>12} {'GFLOPS':>10} {'ms/4k tok':>11}")
    print("-" * 56)

    seqlens = [256, 1024, 4096, 8192, 16384, 32768, 65536, 131072]
    first_time_per_token = None
    for T in seqlens:
        u, delta, A, B, C, *_ = make_inputs(batch, dim, dstate, T)
        # Pick iters based on size so we get steady timing
        iters = max(3, int(1e8 / (batch * dim * dstate * T)))
        sec = time_metal(u, delta, A, B, C, iters=iters)

        tokens = batch * T  # number of sequence tokens processed
        tps = tokens / sec
        flops = estimate_flops(batch, dim, dstate, T) / sec
        time_per_4k = sec / T * 4096

        if first_time_per_token is None:
            first_time_per_token = sec / T

        print(f"{T:>8} {sec*1e3:>9.3f}   {fmt_throughput(tps):>10}  {flops/1e9:>8.2f}   {time_per_4k*1e3:>9.4f}")

    # Linearity check: time-per-token at 65536 vs 1024
    print("\nLinearity check (ms per 4k tokens — should be flat if scaling is linear)")

    # Compare with numpy reference at one tractable size
    print("\nSanity vs numpy reference (small case)")
    bs, d, n, T = 1, 4, 8, 512
    *mx_inp, u_np, delta_np, A_np, B_np, C_np = make_inputs(bs, d, n, T)
    u_mx, delta_mx, A_mx, B_mx, C_mx = mx_inp

    t0 = time.perf_counter()
    y_ref = numpy_reference(u_np, delta_np, A_np, B_np, C_np)
    t_ref = time.perf_counter() - t0

    sec_metal = time_metal(u_mx, delta_mx, A_mx, B_mx, C_mx, iters=10)
    y_metal = np.array(selective_scan(u_mx, delta_mx, A_mx, B_mx, C_mx))
    mx.synchronize()

    err = np.max(np.abs(y_metal - y_ref))
    print(f"  numpy ref: {t_ref*1e3:.2f} ms")
    print(f"  Metal:     {sec_metal*1e3:.3f} ms")
    print(f"  speedup:   {t_ref / sec_metal:.1f}x")
    print(f"  max abs error: {err:.3e}")


if __name__ == "__main__":
    main()
