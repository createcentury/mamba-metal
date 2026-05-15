"""Step 5: chunked selective scan, arbitrary seqlen.

This script just exercises the package's top-level ``selective_scan`` against
a numpy reference. The kernel lives in
``mamba_metal/kernels/selective_scan_chunked.metal``.
"""

import numpy as np
import mlx.core as mx

from mamba_metal import selective_scan


def selective_scan_ref(u, delta, A, B, C):
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


def run_test(batch, dim, dstate, seqlen, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    delta = rng.uniform(0.01, 0.1, size=(batch, dim, seqlen)).astype(np.float32)
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)

    y_ref = selective_scan_ref(u, delta, A, B, C)
    y_metal = np.array(selective_scan(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C)
    ))
    mx.synchronize()

    abs_err = np.max(np.abs(y_metal - y_ref))
    mask = np.abs(y_ref) > 1e-3
    rel_err = (np.max(np.abs(y_metal[mask] - y_ref[mask]) / np.abs(y_ref[mask]))
               if mask.any() else 0.0)
    n_chunks = (seqlen + 1023) // 1024
    print(f"  B={batch} D={dim} N={dstate} T={seqlen:>5} (chunks={n_chunks}):  "
          f"abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    print("Chunked selective scan vs numpy reference\n")
    run_test(1, 2, 8, 512)
    run_test(1, 2, 8, 1024)
    run_test(1, 2, 8, 1025)
    run_test(1, 4, 16, 2048)
    run_test(2, 4, 16, 4096)
    run_test(1, 2, 16, 8192)
    run_test(1, 2, 16, 16384)


if __name__ == "__main__":
    main()
