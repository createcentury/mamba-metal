"""Step 2: block-level prefix scan over (a, b) pairs.

The associative operator for solving h_i = a_i * h_{i-1} + b_i (with h_{-1} = 0):

    (a_2, b_2) ∘ (a_1, b_1) = (a_2 * a_1, a_2 * b_1 + b_2)

After an inclusive scan of {(a_i, b_i)}_i, the .b component at position i equals h_i.

Since `simd_prefix_inclusive_sum` is scalar-only, we hand-roll a Hillis-Steele
scan over the pair within a SIMD group, then bridge SIMD groups via threadgroup
memory — same shape as block_scan.py but applied to pair composition.

This is the heart of selective scan: replace `+` with pair-composition.
"""

import numpy as np
import mlx.core as mx


SIMD_W = 32
TG_SIZE = 1024


PAIR_SCAN_SRC = """
    uint i = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint n_sg = simdgroups_per_threadgroup;

    threadgroup float warp_a[32];
    threadgroup float warp_b[32];

    float a = (i < n) ? a_in[i] : 1.0;  // identity for combine: (1, 0)
    float b = (i < n) ? b_in[i] : 0.0;

    // (1) SIMD-level inclusive scan via hand-rolled Hillis-Steele on pairs
    // Order: update b first (uses old a), then a.
    for (uint d = 1u; d < 32u; d <<= 1) {
        float a_prev = simd_shuffle_up(a, d);
        float b_prev = simd_shuffle_up(b, d);
        if (lane >= d) {
            b = a * b_prev + b;
            a = a * a_prev;
        }
    }

    // (2) Last lane writes the group total to threadgroup memory
    if (lane == 31u) {
        warp_a[sg] = a;
        warp_b[sg] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (3) First SIMD group scans the group totals (also pair-composition)
    if (sg == 0u) {
        float ta = (lane < n_sg) ? warp_a[lane] : 1.0;
        float tb = (lane < n_sg) ? warp_b[lane] : 0.0;
        for (uint d = 1u; d < 32u; d <<= 1) {
            float ta_prev = simd_shuffle_up(ta, d);
            float tb_prev = simd_shuffle_up(tb, d);
            if (lane >= d) {
                tb = ta * tb_prev + tb;
                ta = ta * ta_prev;
            }
        }
        if (lane < n_sg) {
            warp_a[lane] = ta;
            warp_b[lane] = tb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (4) Combine the carry from previous groups into this thread's pair
    if (sg > 0u) {
        float ca = warp_a[sg - 1u];
        float cb = warp_b[sg - 1u];
        b = a * cb + b;
        a = a * ca;
    }

    if (i < n) {
        a_out[i] = a;
        h_out[i] = b;  // b component == h_i when initial state is 0
    }
"""


k_pair_scan = mx.fast.metal_kernel(
    name="pair_scan",
    input_names=["a_in", "b_in", "n"],
    output_names=["a_out", "h_out"],
    source=PAIR_SCAN_SRC,
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
    """Naive sequential recurrence h_i = a_i * h_{i-1} + b_i, h_{-1} = 0."""
    h = np.zeros_like(a)
    s = 0.0
    for i in range(a.size):
        s = a[i] * s + b[i]
        h[i] = s
    return h


def test_case(name: str, a_np: np.ndarray, b_np: np.ndarray) -> None:
    a = mx.array(a_np)
    b = mx.array(b_np)
    _, h_out = pair_scan(a, b)
    mx.eval(h_out)
    h_metal = np.array(h_out)
    h_ref = reference(a_np, b_np)
    abs_err = np.max(np.abs(h_metal - h_ref))
    rel_err = np.max(np.abs(h_metal - h_ref) / (np.abs(h_ref) + 1e-6))
    print(f"  {name:>30}: max abs err = {abs_err:.3e}   rel err = {rel_err:.3e}")
    print(f"  {'h_metal[:5]':>30}: {h_metal[:5].round(4)}")
    print(f"  {'h_ref[:5]':>30}: {h_ref[:5].round(4)}")
    print(f"  {'h_metal[-1]':>30}: {h_metal[-1]:.6f}")
    print(f"  {'h_ref[-1]':>30}: {h_ref[-1]:.6f}")
    print()


def main() -> None:
    print(f"SIMD width = {SIMD_W}, threadgroup = {TG_SIZE}\n")

    # Case 1: leaky integrator with constant a, b — converges to b/(1-a) = 2
    n = 1024
    a_np = np.full(n, 0.5, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    test_case("constant a=0.5, b=1 (→ 2)", a_np, b_np)

    # Case 2: random a in (0, 1), random b
    rng = np.random.default_rng(0)
    a_np = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    b_np = rng.standard_normal(n).astype(np.float32)
    test_case("random a∈(0,1), b∼N(0,1)", a_np, b_np)

    # Case 3: Mamba-like — a_i = exp(delta * A) with negative A, b_i = delta * u
    A = -1.0
    delta = rng.uniform(0.01, 0.1, size=n).astype(np.float32)
    u = rng.standard_normal(n).astype(np.float32)
    a_np = np.exp(delta * A)
    b_np = delta * u
    test_case("Mamba-like (A=-1)", a_np, b_np)

    # Case 4: smaller sizes (not multiple of 32)
    for n in [33, 100, 513]:
        a_np = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        b_np = rng.standard_normal(n).astype(np.float32)
        a = mx.array(a_np)
        b = mx.array(b_np)
        _, h_out = pair_scan(a, b)
        mx.eval(h_out)
        h_ref = reference(a_np, b_np)
        err = np.max(np.abs(np.array(h_out) - h_ref))
        print(f"  n={n:>4} (non-multiple of 32): max abs err = {err:.3e}")


if __name__ == "__main__":
    main()
