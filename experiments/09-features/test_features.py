"""Step 6: verify D / z / delta_softplus extensions against numpy reference.

Each feature is independently toggled and the kernel output is compared with
an extended numpy reference that mirrors Mamba's official selective_scan_ref.
"""

import numpy as np
import mlx.core as mx

from mamba_metal import selective_scan


def softplus_safe(x: np.ndarray) -> np.ndarray:
    # Matches the kernel's branch: use log1p(exp(x)) for x <= 20, else x.
    out = np.where(x <= 20.0, np.log1p(np.exp(np.clip(x, None, 20.0))), x)
    return out.astype(x.dtype)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def reference(u, delta, A, B, C, D=None, z=None, delta_softplus=False):
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
    return y.astype(np.float32)


def make_inputs(batch, dim, dstate, seqlen, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    # delta in a range where softplus is informative (some negative, some positive)
    delta = rng.uniform(-1.0, 2.0, size=(batch, dim, seqlen)).astype(np.float32)
    A = -rng.uniform(0.1, 2.0, size=(dim, dstate)).astype(np.float32)
    B = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    C = rng.standard_normal((batch, dstate, seqlen)).astype(np.float32)
    D = rng.standard_normal((dim,)).astype(np.float32)
    z = rng.standard_normal((batch, dim, seqlen)).astype(np.float32)
    return u, delta, A, B, C, D, z


def run_case(name, batch, dim, dstate, seqlen, use_D, use_z, softplus):
    u, delta, A, B, C, D, z = make_inputs(batch, dim, dstate, seqlen, seed=hash(name) % (2**32))

    # For non-softplus cases, ensure delta is positive (matches typical use)
    delta_eff = softplus_safe(delta).copy() if not softplus else delta
    if not softplus:
        # Use already-softplus'd delta directly (no kernel softplus); but the reference
        # also won't apply softplus, so they match.
        pass

    delta_input = delta if softplus else delta_eff

    # Reference
    y_ref = reference(
        u, delta_input, A, B, C,
        D=D if use_D else None,
        z=z if use_z else None,
        delta_softplus=softplus,
    )

    # Metal
    y_metal = np.array(selective_scan(
        mx.array(u), mx.array(delta_input), mx.array(A), mx.array(B), mx.array(C),
        D=mx.array(D) if use_D else None,
        z=mx.array(z) if use_z else None,
        delta_softplus=softplus,
    ))
    mx.synchronize()

    abs_err = np.max(np.abs(y_metal - y_ref))
    mask = np.abs(y_ref) > 1e-3
    rel_err = (np.max(np.abs(y_metal[mask] - y_ref[mask]) / np.abs(y_ref[mask]))
               if mask.any() else 0.0)
    flags = []
    if use_D: flags.append("D")
    if use_z: flags.append("z")
    if softplus: flags.append("softplus")
    flags_str = "+".join(flags) if flags else "(none)"
    print(f"  {name:>14}  {flags_str:<20}  abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    print("Feature flag matrix — Metal vs numpy reference\n")
    B, D_dim, N, T = 2, 4, 16, 1024
    print(f"  shape: B={B} D={D_dim} N={N} T={T}\n")

    run_case("baseline", B, D_dim, N, T, use_D=False, use_z=False, softplus=False)
    run_case("D only", B, D_dim, N, T, use_D=True, use_z=False, softplus=False)
    run_case("softplus only", B, D_dim, N, T, use_D=False, use_z=False, softplus=True)
    run_case("z only", B, D_dim, N, T, use_D=False, use_z=True, softplus=False)
    run_case("D+z", B, D_dim, N, T, use_D=True, use_z=True, softplus=False)
    run_case("D+softplus", B, D_dim, N, T, use_D=True, use_z=False, softplus=True)
    run_case("all", B, D_dim, N, T, use_D=True, use_z=True, softplus=True)

    # Verify chunked path with features
    print("\n  Chunked (T=4096):")
    run_case("all-chunked", B, D_dim, N, 4096, use_D=True, use_z=True, softplus=True)


if __name__ == "__main__":
    main()
