"""Step 3: minimal selective scan (single chunk, seqlen <= 1024).

Uses selective_scan.metal directly (not the package's top-level
``selective_scan`` which is the chunked version).
"""

import numpy as np
import mlx.core as mx

from mamba_metal import load_kernel


TG_SIZE = 1024


k_selective_scan = load_kernel(
    name="selective_scan",
    input_names=["u", "delta", "A", "B", "C", "batch", "dim", "dstate", "seqlen"],
    output_names=["y"],
)


def selective_scan_metal(
    u: mx.array, delta: mx.array, A: mx.array, B: mx.array, C: mx.array
) -> mx.array:
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    assert seqlen <= TG_SIZE
    (y,) = k_selective_scan(
        inputs=[
            u, delta, A, B, C,
            mx.array(batch, dtype=mx.uint32),
            mx.array(dim, dtype=mx.uint32),
            mx.array(dstate, dtype=mx.uint32),
            mx.array(seqlen, dtype=mx.uint32),
        ],
        output_shapes=[u.shape],
        output_dtypes=[u.dtype],
        grid=(TG_SIZE, batch, dim),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return y


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
    y_metal = np.array(selective_scan_metal(
        mx.array(u), mx.array(delta), mx.array(A), mx.array(B), mx.array(C)
    ))
    mx.synchronize()
    abs_err = np.max(np.abs(y_metal - y_ref))
    print(f"  B={batch} D={dim} N={dstate} T={seqlen}: abs={abs_err:.3e}")


def main() -> None:
    print("Minimal selective scan (single chunk) vs numpy reference\n")
    run_test(1, 1, 4, 32)
    run_test(1, 1, 16, 128)
    run_test(2, 4, 16, 256)
    run_test(2, 8, 16, 1024)


if __name__ == "__main__":
    main()
