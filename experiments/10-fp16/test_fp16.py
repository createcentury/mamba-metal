"""Step 6 (fp16): verify mixed-precision selective scan.

Mamba precision conventions:
  - fp16: u, delta, B, C, z, output y
  - fp32: A, D (weights), internal scan accumulation

MLX generates the Metal kernel signature from each input's dtype, so a single
kernel body handles both fp32-only and mixed-precision calls. fp16 reads/writes
are auto-promoted to float for the scan math.

Acceptance: fp16 output matches fp32 reference within fp16 noise (~1e-3 typical).
"""

import numpy as np
import mlx.core as mx

from mamba_metal import selective_scan


def softplus_safe(x):
    return np.where(x <= 20.0, np.log1p(np.exp(np.clip(x, None, 20.0))), x).astype(x.dtype)


def silu(x):
    return x / (1.0 + np.exp(-x))


def reference_fp32(u, delta, A, B, C, D=None, z=None, delta_softplus=False):
    """Full-precision reference."""
    u = u.astype(np.float32)
    delta = delta.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = C.astype(np.float32)
    if D is not None: D = D.astype(np.float32)
    if z is not None: z = z.astype(np.float32)

    batch, dim, seqlen = u.shape
    _, dstate = A.shape
    if delta_softplus:
        delta = softplus_safe(delta)
    y = np.zeros_like(u)
    for bi in range(batch):
        for di in range(dim):
            h = np.zeros(dstate, dtype=np.float32)
            for t in range(seqlen):
                a_t = np.exp(delta[bi, di, t] * A[di])
                b_t = delta[bi, di, t] * u[bi, di, t] * B[bi, :, t]
                h = a_t * h + b_t
                y[bi, di, t] = np.dot(h, C[bi, :, t])
    if D is not None:
        y = y + D[None, :, None] * u
    if z is not None:
        y = y * silu(z)
    return y


def make_inputs(batch, dim, dstate, seqlen, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32) * 0.5
    delta = rng.uniform(0.01, 0.1, size=(batch, dim, seqlen)).astype(np.float32)
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32) * 0.5
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32) * 0.5
    D = rng.standard_normal((dim,)).astype(np.float32)
    z = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    return u, delta, A, B, C, D, z


def run_case(name, batch, dim, dstate, seqlen, use_D, use_z, softplus, seed=0):
    u, delta, A, B, C, D, z = make_inputs(batch, dim, dstate, seqlen, seed)

    # Full fp32 reference
    y_ref = reference_fp32(
        u, delta, A, B, C,
        D=D if use_D else None,
        z=z if use_z else None,
        delta_softplus=softplus,
    )

    # Metal mixed precision: fp16 for u/delta/B/C/z; fp32 for A/D
    y_mx = selective_scan(
        mx.array(u.astype(np.float16)),
        mx.array(delta.astype(np.float16)),
        mx.array(A),
        mx.array(B.astype(np.float16)),
        mx.array(C.astype(np.float16)),
        D=mx.array(D) if use_D else None,
        z=mx.array(z.astype(np.float16)) if use_z else None,
        delta_softplus=softplus,
    )
    mx.eval(y_mx)
    mx.synchronize()
    y_metal = np.array(y_mx).astype(np.float32)

    assert y_mx.dtype == mx.float16, f"output dtype is {y_mx.dtype}, expected float16"

    abs_err = np.max(np.abs(y_metal - y_ref))
    mask = np.abs(y_ref) > 1e-2  # higher threshold for fp16
    rel_err = (np.max(np.abs(y_metal[mask] - y_ref[mask]) / np.abs(y_ref[mask]))
               if mask.any() else 0.0)
    flags = []
    if use_D: flags.append("D")
    if use_z: flags.append("z")
    if softplus: flags.append("softplus")
    flags_str = "+".join(flags) if flags else "(none)"
    print(f"  {name:>14}  {flags_str:<20}  abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    print("fp16 selective scan vs fp32 numpy reference\n")
    B, D_dim, N, T = 2, 4, 16, 1024
    print(f"  shape: B={B} D={D_dim} N={N} T={T}\n")

    run_case("baseline", B, D_dim, N, T, use_D=False, use_z=False, softplus=False)
    run_case("D only", B, D_dim, N, T, use_D=True, use_z=False, softplus=False)
    run_case("z only", B, D_dim, N, T, use_D=False, use_z=True, softplus=False)
    run_case("softplus only", B, D_dim, N, T, use_D=False, use_z=False, softplus=True)
    run_case("all", B, D_dim, N, T, use_D=True, use_z=True, softplus=True)

    print("\n  Chunked (T=4096):")
    run_case("all-chunked", B, D_dim, N, 4096, use_D=True, use_z=True, softplus=True)
    print("\n  Long chunked (T=16384):")
    run_case("long-chunked", 1, 2, 16, 16384, use_D=True, use_z=True, softplus=True)


if __name__ == "__main__":
    main()
