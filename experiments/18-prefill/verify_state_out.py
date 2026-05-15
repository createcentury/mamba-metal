"""Step 11.1: verify ssm_state_out (kernel output) matches the true h at seqlen-1.

This is the new output from selective_scan_chunked. We compute the reference
h state by running the recurrence in numpy and compare.
"""

import numpy as np
import mlx.core as mx

from mamba_metal import selective_scan


def reference_h_final(u, delta, A, B):
    """Return h at the last position, shape (batch, dim, dstate)."""
    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    h_final = np.zeros((batch, dim, dstate), dtype=np.float32)
    for bi in range(batch):
        for di in range(dim):
            h = np.zeros(dstate, dtype=np.float32)
            for t in range(seqlen):
                a_t = np.exp(delta[bi, di, t] * A[di])
                b_t = delta[bi, di, t] * u[bi, di, t] * B[bi, :, t]
                h = a_t * h + b_t
            h_final[bi, di] = h
    return h_final


def main() -> None:
    rng = np.random.default_rng(0)
    for B, D, N, T in [(1, 2, 8, 256), (2, 4, 16, 1024), (1, 4, 16, 4096)]:
        u = rng.standard_normal((B, D, T)).astype(np.float32)
        delta = rng.uniform(0.01, 0.1, size=(B, D, T)).astype(np.float32)
        A = -rng.uniform(0.1, 2.0, size=(D, N)).astype(np.float32)
        Bw = rng.standard_normal((B, N, T)).astype(np.float32)
        Cw = rng.standard_normal((B, N, T)).astype(np.float32)

        h_ref = reference_h_final(u, delta, A, Bw)

        _, h_metal = selective_scan(
            mx.array(u), mx.array(delta), mx.array(A), mx.array(Bw), mx.array(Cw),
            return_state=True,
        )
        mx.eval(h_metal)
        h_metal_np = np.array(h_metal)

        abs_err = np.max(np.abs(h_metal_np - h_ref))
        rel_err = np.max(np.abs(h_metal_np - h_ref) / (np.abs(h_ref) + 1e-6))
        print(f"  B={B} D={D} N={N} T={T:>5}:  abs={abs_err:.3e}  rel={rel_err:.3e}")


if __name__ == "__main__":
    main()
