"""Step 2: block-level pair scan.

Uses pair_scan.metal — solves h_i = a_i * h_{i-1} + b_i via associative
pair composition.
"""

import numpy as np
import mlx.core as mx

from mamba_metal import load_kernel


TG_SIZE = 1024


k_pair_scan = load_kernel(
    name="pair_scan",
    input_names=["a_in", "b_in", "n"],
    output_names=["a_out", "h_out"],
)


def pair_scan(a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
    assert a.shape == b.shape and a.size <= TG_SIZE
    n = a.size
    a_out, h_out = k_pair_scan(
        inputs=[a, b, mx.array(n, dtype=mx.uint32)],
        output_shapes=[a.shape, a.shape],
        output_dtypes=[a.dtype, a.dtype],
        grid=(TG_SIZE, 1, 1),
        threadgroup=(TG_SIZE, 1, 1),
    )
    return a_out, h_out


def reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = np.zeros_like(a)
    s = 0.0
    for i in range(a.size):
        s = a[i] * s + b[i]
        h[i] = s
    return h


def test_case(name: str, a_np: np.ndarray, b_np: np.ndarray) -> None:
    a, b = mx.array(a_np), mx.array(b_np)
    _, h_out = pair_scan(a, b)
    mx.eval(h_out)
    h_metal = np.array(h_out)
    h_ref = reference(a_np, b_np)
    abs_err = np.max(np.abs(h_metal - h_ref))
    rel_err = np.max(np.abs(h_metal - h_ref) / (np.abs(h_ref) + 1e-6))
    print(f"  {name:>30}: abs={abs_err:.3e}  rel={rel_err:.3e}")


def main() -> None:
    n = 1024
    a_np = np.full(n, 0.5, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    test_case("constant a=0.5, b=1 (-> 2)", a_np, b_np)

    rng = np.random.default_rng(0)
    a_np = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    b_np = rng.standard_normal(n).astype(np.float32)
    test_case("random a in (0,1), b ~ N(0,1)", a_np, b_np)

    A = -1.0
    delta = rng.uniform(0.01, 0.1, size=n).astype(np.float32)
    u = rng.standard_normal(n).astype(np.float32)
    a_np = np.exp(delta * A)
    b_np = delta * u
    test_case("Mamba-like (A=-1)", a_np, b_np)

    for n in [33, 100, 513]:
        a_np = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        b_np = rng.standard_normal(n).astype(np.float32)
        _, h_out = pair_scan(mx.array(a_np), mx.array(b_np))
        mx.eval(h_out)
        h_ref = reference(a_np, b_np)
        err = np.max(np.abs(np.array(h_out) - h_ref))
        print(f"  n={n:>4} (non-multiple of 32): abs={err:.3e}")


if __name__ == "__main__":
    main()
